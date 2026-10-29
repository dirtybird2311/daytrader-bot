"""Opening-range breakout on stocks in play (Zarattini, Barbon & Aziz, 2024).

Each day at 9:35, using only what was known then:
  * screen: open > $5, 14-day avg volume >= 1M, 14-day ATR > $0.50, relative volume >= 100%
  * take the top N by relative volume
  * first 5-minute candle up -> buy-stop at its high; down -> sell-stop at its low; doji -> skip
  * once filled, a protective stop sits `stop_atr` x ATR from the fill
  * anything still open is closed at the 4:00 close; unfilled entry orders are cancelled
Sizing: each of the N slots gets equity/N; a stop-out loses `risk_per_trade` of that slot,
capped at `max_leverage` x the slot.

Fill rules (orders rest at the broker, so no decision delay):
  * entry stop triggers on the first bar (from 9:35) whose high (low for shorts) reaches the
    level; fill = the worse of the level and that bar's open, plus slippage
  * protective stop: if a bar opens through it, fill at the open; else if the bar's range
    touches it, fill at the stop; slippage on top.
  * the entry bar itself is ambiguous when the fill happened mid-bar: config fills.same_bar_stop
    picks pessimistic / close_confirmed / optimistic / ticks (see config/orb.yaml).

Research variants (all off by default, see RESEARCH.md):
  * params.or_minutes: opening range of 1, 5, 15 or 30 minutes. Stocks are still picked at
    9:35, so with a 1-minute range orders can only go in from 9:35.
  * params.exit_mode: "close" (paper) or "vwap": also exit at the next open once a 1-minute
    bar closes on the wrong side of the session VWAP.
  * params.ta_filter: none / sma / rsi / macd / bollinger, from prior daily bars (data/ta.py).
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from ..data.model import minute_to_clock
from ..data.orb_data import OrbData
from ..data.ta import passes
from ..data.universe import screen
from .engine import BacktestResult, Decision

OR_BARS = 5  # 9:30-9:35: stocks in play are picked when this range closes
FILL_MODES = ("pessimistic", "close_confirmed", "optimistic", "ticks")


class OrbEngine:
    def __init__(self, cfg: dict):
        p, s, c = cfg["params"], cfg["sizing"], cfg["costs"]
        self.cfg = cfg
        self.screen_cfg = {"min_price": float(p["min_price"]), "min_avg_volume": float(p["min_avg_volume"]),
                           "min_atr": float(p["min_atr"]), "min_rel_volume": float(p["min_rel_volume"])}
        self.top_n = int(p["top_n"])
        self.stop_atr = float(p["stop_atr"])
        self.allow_short = bool(p["allow_short"])
        self.risk = float(s["risk_per_trade"])
        self.max_lev = float(s["max_leverage"])
        self.commission = float(c["commission_per_share"])
        self.slip = float(c["slippage_per_share"])
        self.same_bar = cfg.get("fills", {}).get("same_bar_stop", "pessimistic")
        if self.same_bar not in FILL_MODES:
            raise ValueError(f"fills.same_bar_stop must be one of {FILL_MODES}")
        self.or_minutes = int(p.get("or_minutes", OR_BARS))
        if self.or_minutes not in (1, 5, 15, 30):
            raise ValueError("params.or_minutes must be 1, 5, 15 or 30")
        self.start_bar = max(self.or_minutes, OR_BARS)  # first bar an entry order can trigger on
        self.exit_mode = p.get("exit_mode", "close")
        if self.exit_mode not in ("close", "vwap"):
            raise ValueError("params.exit_mode must be 'close' or 'vwap'")
        self.ta_filter = p.get("ta_filter", "none")
        self.ta_args = {"rsi_hi": float(p.get("rsi_hi", 80)), "rsi_lo": float(p.get("rsi_lo", 20)),
                        "bb_k": float(p.get("bb_k", 2.0))}
        self.sma_days = int(p.get("sma_days", 20))
        self.ambiguous: list[tuple] = []  # (date, symbol, bar): entry minutes minute bars can't resolve
        self.fill_sources = {"bar": 0, "ticks": 0, "ticks_missing": 0}
        self.flat_band = float(cfg.get("benchmark", {}).get("flat_band", 0.003))

    def run(self, data: OrbData, start: dt.date | None = None, end: dt.date | None = None) -> BacktestResult:
        equity = float(self.cfg["account"]["starting_equity"])
        rows, trades, decisions = [], [], []
        for date in data.dates:
            if (start and date < start) or (end and date > end):
                continue
            eq0 = equity
            feats = data.features(date)
            picks = screen(feats, **self.screen_cfg) if not feats.empty else feats
            picks = picks[picks["rank"] <= self.top_n] if not picks.empty else picks
            day_trades, missing, notional = [], 0, 0.0
            alloc = eq0 / self.top_n
            ta = data.ta_features(date, self.sma_days) if self.ta_filter != "none" and not picks.empty else None
            for _, pk in picks.iterrows():
                day = None
                if self.or_minutes == OR_BARS:
                    o5, h5, l5, c5 = pk["open5"], pk["high5"], pk["low5"], pk["close5"]
                else:
                    day = data.minute_day(date, pk["symbol"])
                    if day is None or day.n <= self.start_bar + 1:
                        missing += 1
                        continue
                    k = self.or_minutes
                    o5, h5, l5, c5 = day.open[0], day.high[:k].max(), day.low[:k].min(), day.close[k - 1]
                direction = int(np.sign(c5 - o5))
                if direction == 0 or (direction < 0 and not self.allow_short):
                    continue
                if ta is not None:
                    row = ta.loc[pk["symbol"]] if pk["symbol"] in ta.index else pd.Series(dtype=float)
                    if not passes(row, direction, self.ta_filter, **self.ta_args):
                        continue
                level = float(h5 if direction > 0 else l5)
                atr = float(pk["atr"])
                decisions.append(Decision(date, self.start_bar - 1, minute_to_clock(self.start_bar - 1), direction,
                                          f"order:{pk['symbol']}", level, {"rank": int(pk["rank"])}))
                if day is None:
                    day = data.minute_day(date, pk["symbol"])
                if day is None or day.n <= self.start_bar + 1:
                    missing += 1
                    continue
                risk_ps = self.stop_atr * atr
                shares = math.floor(min(alloc * self.risk / risk_ps, alloc * self.max_lev / level)) if risk_ps > 0 else 0
                if shares < 1:
                    continue
                t = self._simulate(day, pk, direction, level, risk_ps, shares, data)
                if t is None:
                    continue
                decisions.append(Decision(date, t["entry_bar"], t["entry_time"], direction,
                                          f"entry:{pk['symbol']}", t["entry_px"]))
                decisions.append(Decision(date, t["exit_bar"], t["exit_time"], 0,
                                          f"exit:{pk['symbol']}:{t['exit_reason']}", t["exit_px"]))
                notional += t["entry_px"] * shares
                day_trades.append(t)
            pnl = sum(t["pnl"] for t in day_trades)
            equity = eq0 + pnl
            trades.extend(day_trades)
            mret, moc = data.market(date)
            rows.append({
                "date": pd.Timestamp(date), "equity_start": eq0, "equity": equity, "ret": pnl / eq0 if eq0 else 0.0,
                "n_trades": len(day_trades), "n_candidates": int(len(picks)), "n_missing": missing,
                "market_ret": mret, "market_oc_ret": moc,
                "market_day": "up" if mret > self.flat_band else ("down" if mret < -self.flat_band else "flat"),
                "leverage": notional / eq0 if eq0 else 0.0, "weekday": date.strftime("%a"),
            })
        daily = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
        tdf = pd.DataFrame(trades)
        if not tdf.empty:
            tdf["date"] = pd.to_datetime(tdf["date"])
        return BacktestResult(daily=daily, trades=tdf, decisions=decisions, config=self.cfg)

    def _simulate(self, day, pk, direction: int, level: float, risk_ps: float, shares: int,
                  data: OrbData | None = None) -> dict | None:
        o, h, l, c = day.open, day.high, day.low, day.close
        last = day.n - 1
        s0 = self.start_bar
        # entry: first bar after the opening range whose range reaches the level (cancelled at the close)
        reach = (h[s0:last] >= level) if direction > 0 else (l[s0:last] <= level)
        idx = np.flatnonzero(reach)
        if idx.size == 0:
            return None
        i = s0 + int(idx[0])
        raw = max(level, o[i]) if direction > 0 else min(level, o[i])
        entry = raw + direction * self.slip
        stop = entry - direction * risk_ps
        exit_i, exit_px, reason = last, c[last] - direction * self.slip, "close"
        filled_at_open = (o[i] >= level) if direction > 0 else (o[i] <= level)
        vwap = None
        if self.exit_mode == "vwap":
            v = day.volume
            cum_v = np.cumsum(v)
            vwap = np.where(cum_v > 0, np.cumsum((h + l + c) / 3.0 * v) / np.where(cum_v > 0, cum_v, 1), c)

        source, first_j, done = "bar", i, False
        ambiguous = not filled_at_open and ((l[i] <= stop) if direction > 0 else (h[i] >= stop))
        if ambiguous:
            self.ambiguous.append((day.date, str(pk["symbol"]), i))
        if ambiguous and self.same_bar == "ticks":
            prints = data.tick_prices(day.date, str(pk["symbol"]), i) if data is not None else None
            hit = None if prints is None else np.flatnonzero(prints >= level if direction > 0 else prints <= level)
            if hit is None or hit.size == 0:
                source = "ticks_missing"  # fall back to the pessimistic reading of the minute bar
            else:
                source = "ticks"
                k = int(hit[0])
                raw = max(level, float(prints[k])) if direction > 0 else min(level, float(prints[k]))
                entry = raw + direction * self.slip
                stop = entry - direction * risk_ps
                after = prints[k + 1:]
                through = np.flatnonzero(after <= stop if direction > 0 else after >= stop)
                if through.size:
                    px = min(stop, float(after[through[0]])) if direction > 0 else max(stop, float(after[through[0]]))
                    exit_i, exit_px, reason, done = i, px - direction * self.slip, "stop", True
                else:
                    first_j = i + 1  # the entry minute is resolved: no stop inside it
                    if vwap is not None and ((c[i] < vwap[i]) if direction > 0 else (c[i] > vwap[i])) and i < last:
                        exit_i, exit_px, reason, done = i + 1, o[i + 1] - direction * self.slip, "vwap", True
        self.fill_sources[source] += 1

        pending_vwap = False
        for j in range(first_j, day.n) if not done else ():
            if pending_vwap:
                exit_i, exit_px, reason = j, o[j] - direction * self.slip, "vwap"
                break
            if j > i and ((o[j] <= stop) if direction > 0 else (o[j] >= stop)):
                exit_i, exit_px, reason = j, o[j] - direction * self.slip, "stop_gap"
                break
            touched = (l[j] <= stop) if direction > 0 else (h[j] >= stop)
            if j == i and not filled_at_open:
                # Entry happened mid-bar: minute data can't say whether the bar's low (high for
                # shorts) came before or after the fill. How to treat it is a model choice.
                closed_through = (c[j] <= stop) if direction > 0 else (c[j] >= stop)
                if self.same_bar == "optimistic":
                    touched = False
                elif self.same_bar == "close_confirmed":
                    touched = closed_through  # closing beyond the stop proves it was crossed after entry
            if touched:
                exit_i, exit_px, reason = j, stop - direction * self.slip, "stop"
                break
            if vwap is not None and j < last and ((c[j] < vwap[j]) if direction > 0 else (c[j] > vwap[j])):
                pending_vwap = True
        hi, lo = h[i:exit_i + 1].max(), l[i:exit_i + 1].min()
        gross = direction * (exit_px - entry) * shares
        costs = 2 * self.commission * shares
        adverse = (entry - lo) if direction > 0 else (hi - entry)
        favorable = (hi - entry) if direction > 0 else (entry - lo)
        return {
            "date": day.date, "symbol": pk["symbol"], "rank": int(pk["rank"]), "rel_volume": float(pk["rel_volume"]),
            "side": direction, "shares": shares, "entry_bar": i, "entry_time": minute_to_clock(i - 1),
            "entry_px": float(entry), "exit_bar": exit_i,
            "exit_time": minute_to_clock(exit_i) if reason == "close" else minute_to_clock(exit_i - 1),
            "exit_px": float(exit_px), "entry_reason": "breakout_long" if direction > 0 else "breakout_short",
            "exit_reason": reason, "stop_at_entry": float(stop), "gross": float(gross), "costs": float(costs),
            "pnl": float(gross - costs), "ret": float((gross - costs) / (entry * shares)),
            "r_multiple": float((gross - costs) / (risk_ps * shares)),
            "mae_r": float(max(adverse, 0) / risk_ps), "mfe_r": float(max(favorable, 0) / risk_ps),
            "fill_source": source,
        }
