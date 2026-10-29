"""Buy the dip on parabolic runners (research queue #8, pre-registered in RESEARCH.md).

Before the open (daily bars only):
  * setup: at least `run_days` consecutive up days, each with a wider range than the day
    before, and yesterday's close above the highest high of the `resistance_lookback` days
    before the run began (the old resistance)
  * liquid: open > $5, 14-day average volume >= 1M, 14-day ATR > $0.50
Intraday, on 5-minute bars built from 1-minute bars:
  * the pullback: a 5-minute low within `zone_atr` x ATR of the resistance
  * then a reversal bar: higher low than the prior 5-minute bar and a close above its high
  * buy at the next 1-minute open, no entries after `last_entry`
  * stop: resistance - `stop_atr` x ATR; if price breaks it before an entry, the setup is void
  * half sells at `target_r` x risk (limit order); the stop on the rest moves to the entry price
  * anything left exits at the close; long only
Sizing: risk `risk_per_trade` of equity per trade, notional capped at equity / max_positions;
only the first `max_positions` setups to trigger each day are taken.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from ..data.model import clock_to_bar_index, minute_to_clock
from ..data.orb_data import OrbData
from ..data.setups import dip_setups, dip_table
from .engine import BacktestResult, Decision


class DipEngine:
    def __init__(self, cfg: dict):
        p, s, c = cfg["params"], cfg["sizing"], cfg["costs"]
        self.cfg = cfg
        self.run_days = int(p["run_days"])
        self.lookback = int(p["resistance_lookback"])
        self.zone_atr = float(p["zone_atr"])
        self.stop_atr = float(p["stop_atr"])
        self.target_r = float(p["target_r"])
        self.last_entry_bar = clock_to_bar_index(p["last_entry"]) + 1  # entry fills at this bar's open at the latest
        self.min_price, self.min_vol, self.min_atr = float(p["min_price"]), float(p["min_avg_volume"]), float(p["min_atr"])
        self.max_pos = int(s["max_positions"])
        self.risk = float(s["risk_per_trade"])
        self.commission = float(c["commission_per_share"])
        self.slip = float(c["slippage_per_share"])
        self.flat_band = float(cfg.get("benchmark", {}).get("flat_band", 0.003))
        self.missing_total = 0

    def setups(self, data: OrbData) -> dict:
        cache = data.__dict__.setdefault("_dip_cache", {})
        key = self.lookback
        if key not in cache:
            cache[key] = dip_table(data.daily_frame(), self.lookback)
        s = dip_setups(cache[key], self.run_days)
        return {d: g for d, g in s.groupby("date")}

    def run(self, data: OrbData, start: dt.date | None = None, end: dt.date | None = None) -> BacktestResult:
        equity = float(self.cfg["account"]["starting_equity"])
        by_date = self.setups(data)
        rows, trades, decisions = [], [], []
        for date in data.dates:
            if (start and date < start) or (end and date > end):
                continue
            eq0 = equity
            cands = by_date.get(date)
            plans, missing, n_cand = [], 0, 0
            if cands is not None and not cands.empty:
                feats = data.features(date)
                if not feats.empty:
                    f = feats[["symbol", "open5", "avg_volume", "atr"]].copy()
                    f["symbol"] = f["symbol"].astype(str)
                    m = cands.merge(f, on="symbol")
                    m = m[(m["open5"] > self.min_price) & (m["avg_volume"] >= self.min_vol) & (m["atr"] > self.min_atr)]
                    n_cand = len(m)
                    for _, sp in m.sort_values("symbol").iterrows():
                        decisions.append(Decision(date, -1, "09:30", 1, f"setup:{sp['symbol']}", float(sp["resistance"])))
                        day = data.minute_day(date, sp["symbol"])
                        if day is None or day.n < 30:
                            missing += 1
                            continue
                        plan = self._plan(day, float(sp["resistance"]), float(sp["atr"]))
                        if plan is not None:
                            plans.append((plan["entry_bar"], sp["symbol"], plan, day, float(sp["atr"])))
            plans.sort(key=lambda x: (x[0], x[1]))
            day_trades, notional = [], 0.0
            for entry_bar, sym, plan, day, atr in plans[: self.max_pos]:
                risk_ps = plan["entry"] - plan["stop"]
                shares = math.floor(min(eq0 * self.risk / risk_ps, (eq0 / self.max_pos) / plan["entry"]))
                if shares < 2:
                    continue
                t = self._manage(day, sym, plan, shares)
                decisions.append(Decision(date, entry_bar - 1, minute_to_clock(entry_bar - 1), 1, f"entry:{sym}",
                                          t["entry_px"]))
                decisions.append(Decision(date, t["exit_bar"], t["exit_time"], 0, f"exit:{sym}:{t['exit_reason']}",
                                          t["exit_px"]))
                notional += t["entry_px"] * shares
                day_trades.append(t)
            self.missing_total += missing
            pnl = sum(t["pnl"] for t in day_trades)
            equity = eq0 + pnl
            trades.extend(day_trades)
            mret, moc = data.market(date)
            rows.append({
                "date": pd.Timestamp(date), "equity_start": eq0, "equity": equity, "ret": pnl / eq0 if eq0 else 0.0,
                "n_trades": len(day_trades), "n_candidates": n_cand, "n_missing": missing,
                "market_ret": mret, "market_oc_ret": moc,
                "market_day": "up" if mret > self.flat_band else ("down" if mret < -self.flat_band else "flat"),
                "leverage": notional / eq0 if eq0 else 0.0, "weekday": date.strftime("%a"),
            })
        daily = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
        tdf = pd.DataFrame(trades)
        if not tdf.empty:
            tdf["date"] = pd.to_datetime(tdf["date"])
        return BacktestResult(daily=daily, trades=tdf, decisions=decisions, config=self.cfg)

    def _plan(self, day, resistance: float, atr: float) -> dict | None:
        """Find the entry: zone touch, then a reversal 5-minute bar; buy at the next minute's open."""
        zone = resistance + self.zone_atr * atr
        stop = resistance - self.stop_atr * atr
        n5 = day.n // 5
        lo5 = day.low[: n5 * 5].reshape(n5, 5).min(axis=1)
        hi5 = day.high[: n5 * 5].reshape(n5, 5).max(axis=1)
        cl5 = day.close[: n5 * 5].reshape(n5, 5)[:, -1]
        touched = False
        for b in range(n5):
            entry_bar = 5 * (b + 1)
            if entry_bar > self.last_entry_bar or entry_bar >= day.n:
                return None
            if lo5[b] <= stop:
                return None  # broke the stop level before any entry: setup void
            if b >= 1 and touched and lo5[b] > lo5[b - 1] and cl5[b] > hi5[b - 1]:
                entry = float(day.open[entry_bar]) + self.slip
                if entry <= stop:
                    return None
                return {"entry_bar": entry_bar, "entry": entry, "stop": stop, "resistance": resistance}
            if lo5[b] <= zone:
                touched = True
        return None

    def _manage(self, day, sym: str, plan: dict, shares: int) -> dict:
        o, h, l, c = day.open, day.high, day.low, day.close
        i, entry, stop0 = plan["entry_bar"], plan["entry"], plan["stop"]
        risk = entry - stop0
        target = entry + self.target_r * risk
        stop, left, half = stop0, shares, shares // 2
        proceeds, exits, reason, last_exit = 0.0, [], "close", day.n - 1
        hit_target = False
        for j in range(i, day.n):
            if j > i and o[j] <= stop:
                px = o[j] - self.slip
                proceeds += px * left
                exits.append((j, px, left)); left = 0
                reason, last_exit = ("stop_gap" if not hit_target else "breakeven_gap"), j
                break
            if l[j] <= stop:  # if a bar reaches both the stop and the target, the stop comes first
                px = stop - self.slip
                proceeds += px * left
                exits.append((j, px, left)); left = 0
                reason, last_exit = ("stop" if not hit_target else "breakeven"), j
                break
            if not hit_target and h[j] >= target:
                proceeds += target * half
                exits.append((j, target, half)); left -= half
                hit_target, stop = True, entry  # the rest rides with a breakeven stop
        if left > 0:
            px = c[-1] - self.slip
            proceeds += px * left
            exits.append((day.n - 1, px, left))
            last_exit = day.n - 1
            reason = "target_then_close" if hit_target else "close"
        exit_px = proceeds / shares
        gross = proceeds - entry * shares
        costs = self.commission * shares * 2
        hi, lo = h[i:last_exit + 1].max(), l[i:last_exit + 1].min()
        return {
            "date": day.date, "symbol": sym, "side": 1, "shares": shares, "entry_bar": i,
            "entry_time": minute_to_clock(i - 1), "entry_px": float(entry), "exit_bar": int(last_exit),
            "exit_time": minute_to_clock(last_exit) if reason.endswith("close") else minute_to_clock(last_exit - 1),
            "exit_px": float(exit_px), "entry_reason": "dip_reversal", "exit_reason": reason,
            "stop_at_entry": float(stop0), "resistance": float(plan["resistance"]), "gross": float(gross),
            "costs": float(costs), "pnl": float(gross - costs), "ret": float((gross - costs) / (entry * shares)),
            "r_multiple": float((gross - costs) / (risk * shares)),
            "mae_r": float(max(entry - lo, 0) / risk), "mfe_r": float(max(hi - entry, 0) / risk),
        }
