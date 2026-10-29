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
    picks pessimistic / close_confirmed / optimistic (see config/orb.yaml).
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from ..data.model import minute_to_clock
from ..data.orb_data import OrbData
from ..data.universe import screen
from .engine import BacktestResult, Decision

OR_BARS = 5  # 9:30-9:35


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
        if self.same_bar not in ("pessimistic", "close_confirmed", "optimistic"):
            raise ValueError("fills.same_bar_stop must be pessimistic, close_confirmed or optimistic")
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
            for _, pk in picks.iterrows():
                direction = int(np.sign(pk["close5"] - pk["open5"]))
                if direction == 0 or (direction < 0 and not self.allow_short):
                    continue
                level = float(pk["high5"] if direction > 0 else pk["low5"])
                atr = float(pk["atr"])
                decisions.append(Decision(date, OR_BARS - 1, minute_to_clock(OR_BARS - 1), direction,
                                          f"order:{pk['symbol']}", level, {"rank": int(pk["rank"])}))
                day = data.minute_day(date, pk["symbol"])
                if day is None or day.n <= OR_BARS + 1:
                    missing += 1
                    continue
                risk_ps = self.stop_atr * atr
                shares = math.floor(min(alloc * self.risk / risk_ps, alloc * self.max_lev / level)) if risk_ps > 0 else 0
                if shares < 1:
                    continue
                t = self._simulate(day, pk, direction, level, risk_ps, shares)
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

    def _simulate(self, day, pk, direction: int, level: float, risk_ps: float, shares: int) -> dict | None:
        o, h, l, c = day.open, day.high, day.low, day.close
        last = day.n - 1
        # entry: first bar after the opening range whose range reaches the level (cancelled at the close)
        reach = (h[OR_BARS:last] >= level) if direction > 0 else (l[OR_BARS:last] <= level)
        idx = np.flatnonzero(reach)
        if idx.size == 0:
            return None
        i = OR_BARS + int(idx[0])
        raw = max(level, o[i]) if direction > 0 else min(level, o[i])
        entry = raw + direction * self.slip
        stop = entry - direction * risk_ps
        exit_i, exit_px, reason = last, c[last] - direction * self.slip, "close"
        filled_at_open = (o[i] >= level) if direction > 0 else (o[i] <= level)
        for j in range(i, day.n):
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
        }
