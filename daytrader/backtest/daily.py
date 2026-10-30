"""Overnight-to-few-days strategies (research batch D, pre-registered in RESEARCH.md).

Each trading day:
  1. at the open: any position marked "exit at next open" is sold at the official open
  2. at 3:50 pm ("snapshot", 10 minutes before the close): the strategy decides, seeing only
     completed days' daily bars, today's open, and today's 3:50 pm price
  3. at the close: entries and close-exits fill at the official close
Positions are marked to market at every close, so daily returns are real daily returns.
Half-days have no 3:50 pm price: no decisions that day (open exits still happen).

Prices are raw (unadjusted). A held stock whose overnight gap is over 40% is treated as a split
when the gap is within 15% of a common split ratio; shares and entry price are adjusted.

Strategies (cfg["strategy"]):
  spy_overnight   buy SPY at the close, sell at the next open
  spy_rsi2        RSI(2) pullback above the 200-day SMA, exit above the 5-day SMA
  reversal        buy the n biggest large-cap losers (prior close -> 3:50 pm) at the close
  spy_rules       batch G: classic daily rules from QuantifiedStrategies' list (params.rule):
                    ibs            buy when IBS < ibs_entry; sell when price > yesterday's high
                    turnaround_tue on Monday, if below Friday's close, buy; sell > yesterday's high
                    turn_of_month  buy on the 5th-last trading day; sell on the 3rd trading day
                    down3_overnight after 3 down closes, buy at the close, sell at the next open
                    five_day_low   buy when price < the lowest low of the prior 5 days; sell > yesterday's high
                    triple_rsi     Connors: RSI(5) < 30, falling 3 days, below 60 three days ago,
                                   above the 200-day SMA; sell when RSI(5) > 50
                    rsi2_high      RSI(2) < rsi_entry; sell when price > yesterday's high
                  "price" and "today's high/low" are the 3:50 pm price and the 9:30-3:50 range.
                  The trading calendar (which days are trading days) is known in advance.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.model import Day
from .engine import BacktestResult, Decision

SNAP_BAR = 379          # index of the 3:49-3:50 bar in a full 390-minute session
SPLIT_GAP = 0.4
SPLIT_RATIOS = [1 / 20, 1 / 10, 1 / 8, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 2, 2, 3, 4, 5, 8, 10, 20]


def split_ratio(open_px: float, prev_close: float) -> float | None:
    """open / prev_close snapped to a split ratio, or None if the gap looks like a real move."""
    if not (prev_close > 0 and open_px > 0):
        return None
    g = open_px / prev_close
    if abs(g - 1) <= SPLIT_GAP:
        return None
    best = min(SPLIT_RATIOS, key=lambda r: abs(math.log(g / r)))
    return best if abs(math.log(g / best)) < math.log(1.15) else None


# ---- data ------------------------------------------------------------------------------------
class DailyData:
    dates: list[dt.date]

    def bar(self, date, symbol) -> tuple[float, float | None, float] | None:
        """(open, 3:50 pm price or None on half-days, close) for one symbol-day."""
        raise NotImplementedError

    def closes_before(self, symbol: str, date: dt.date, n: int) -> np.ndarray:
        raise NotImplementedError

    def candidates(self, date) -> pd.DataFrame:
        """Reversal universe: symbol, prev_close, open, snap, dv_rank (all known at 3:50 pm)."""
        raise NotImplementedError

    def market(self, date) -> tuple[float, float]:
        raise NotImplementedError


class SpyDaily(DailyData):
    """From SPY 1-minute Days."""

    def __init__(self, days: list[Day], symbol: str = "SPY"):
        self.symbol = symbol
        self.dates = [d.date for d in days]
        self._o = np.array([d.open[0] for d in days], float)
        self._c = np.array([d.close[-1] for d in days], float)
        self._s = np.array([d.close[SNAP_BAR] if d.n == 390 else np.nan for d in days], float)
        self._h = np.array([d.high.max() for d in days], float)
        self._l = np.array([d.low.min() for d in days], float)
        self._hs = np.array([d.high[:SNAP_BAR + 1].max() if d.n == 390 else np.nan for d in days], float)
        self._ls = np.array([d.low[:SNAP_BAR + 1].min() if d.n == 390 else np.nan for d in days], float)
        self._idx = {d: i for i, d in enumerate(self.dates)}

    def bar(self, date, symbol):
        i = self._idx.get(date)
        if i is None:
            return None
        s = self._s[i]
        return float(self._o[i]), (None if np.isnan(s) else float(s)), float(self._c[i])

    def closes_before(self, symbol, date, n):
        i = self._idx[date]
        return self._c[max(0, i - n):i]

    def highs_lows_before(self, date, n):
        i = self._idx[date]
        return self._h[max(0, i - n):i], self._l[max(0, i - n):i]

    def range_to_snap(self, date):
        i = self._idx[date]
        return float(self._hs[i]), float(self._ls[i])

    def market(self, date):
        i = self._idx[date]
        if i == 0:
            return 0.0, float(self._c[0] / self._o[0] - 1)
        return float(self._c[i] / self._c[i - 1] - 1), float(self._c[i] / self._o[i] - 1)


class UniverseDaily(DailyData):
    """Stocks-in-play cache (daily bars, features, SPY) plus 3:50 pm snapshots."""

    def __init__(self, orb_data, snaps: pd.DataFrame, large_cap_rank: int = 500, min_price: float = 5.0,
                 exclude: set | None = None):
        self.base = orb_data
        self.exclude = set(exclude or ())  # ETFs/ETNs: not stocks, and leveraged ones decay overnight
        self.dates = list(orb_data.dates)
        d = orb_data.daily_frame()[["symbol", "date", "o", "c"]].copy()
        d["symbol"] = d["symbol"].astype(str)
        self._oc = {(s, dd): (o, c) for s, dd, o, c in zip(d["symbol"], d["date"], d["o"], d["c"])}
        snaps = snaps.copy()
        snaps["symbol"] = snaps["symbol"].astype(str)
        self._snap = {(s, dd): p for s, dd, p in zip(snaps["symbol"], snaps["date"], snaps["snap"])}
        self.rank, self.min_price = large_cap_rank, min_price

    def bar(self, date, symbol):
        oc = self._oc.get((symbol, date))
        if oc is None:
            return None
        return float(oc[0]), self._snap.get((symbol, date)), float(oc[1])

    def candidates(self, date):
        f = self.base.features(date)
        if f.empty:
            return f
        f = f[["symbol", "prev_close", "avg_volume"]].copy()
        f["symbol"] = f["symbol"].astype(str)
        if self.exclude:
            f = f[~f["symbol"].isin(self.exclude)]
        f["dv_rank"] = (f["avg_volume"] * f["prev_close"]).rank(ascending=False, method="first")
        f = f[f["dv_rank"] <= self.rank]
        f["open"] = [self._oc.get((s, date), (np.nan, np.nan))[0] for s in f["symbol"]]
        f["snap"] = [self._snap.get((s, date), np.nan) for s in f["symbol"]]
        return f.dropna(subset=["open", "snap"])

    def market(self, date):
        return self.base.market(date)


# ---- engine ----------------------------------------------------------------------------------
@dataclass
class Pos:
    symbol: str
    shares: float
    entry_px: float
    entry_date: dt.date
    exit_rule: str           # next_open | next_close | rule
    last_close: float
    cost: float
    hi: float
    lo: float


def rsi_wilder(closes: np.ndarray, n: int = 2) -> float:
    if len(closes) < n + 1:
        return float("nan")
    d = np.diff(closes)
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au, ad = up[:n].mean(), dn[:n].mean()
    for u, w in zip(up[n:], dn[n:]):
        au = (au * (n - 1) + u) / n
        ad = (ad * (n - 1) + w) / n
    if ad == 0:
        return 100.0
    return float(100 - 100 / (1 + au / ad))


class DailyEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.kind = cfg["strategy"]
        p, c = cfg["params"], cfg["costs"]
        self.p = p
        self.slip = float(c["slippage_per_share"])
        self.comm = float(c["commission_per_share"])
        self.max_lev = float(cfg["sizing"].get("max_leverage", 1.0))
        self.flat_band = float(cfg.get("benchmark", {}).get("flat_band", 0.003))
        self.split_events = 0

    def run(self, data: DailyData, start=None, end=None) -> BacktestResult:
        self.__dict__.pop("_v", None); self.__dict__.pop("_exit_mode", None); self.__dict__.pop("_cal", None)
        cash = float(self.cfg["account"]["starting_equity"])
        positions: dict[str, Pos] = {}
        rows, trades, decisions = [], [], []
        prev_equity = cash
        for date in data.dates:
            if (start and date < start) or (end and date > end):
                continue
            eq0 = prev_equity
            # 0) splits on held names, then 1) open exits
            for sym, ps in list(positions.items()):
                b = data.bar(date, sym)
                if b is None:
                    continue
                r = split_ratio(b[0], ps.last_close)
                if r is not None:
                    self.split_events += 1
                    ps.shares, ps.entry_px, ps.last_close = ps.shares / r, ps.entry_px * r, ps.last_close * r
                    ps.hi, ps.lo = ps.hi * r, ps.lo * r
                if ps.exit_rule == "next_open":
                    px = b[0] - self.slip
                    cash += ps.shares * px - self.comm * ps.shares
                    trades.append(self._trade(ps, date, 0, px, "open"))
                    decisions.append(Decision(date, 0, "09:30", 0, f"exit:{sym}:open", px))
                    del positions[sym]
            # 2) decide at 3:50 pm
            orders = self._decide(data, date, positions, eq0)
            # 3) fills at the close
            for sym, action, shares in orders:
                b = data.bar(date, sym)
                if b is None:
                    continue
                close = b[2]
                if action == "sell" and sym in positions:
                    ps = positions.pop(sym)
                    px = close - self.slip
                    cash += ps.shares * px - self.comm * ps.shares
                    trades.append(self._trade(ps, date, 389, px, "close"))
                elif action == "buy" and sym not in positions and shares >= 1:
                    px = close + self.slip
                    cash -= shares * px + self.comm * shares
                    positions[sym] = Pos(sym, shares, px, date, self._exit_rule(), close, self.comm * shares,
                                         close, close)
            if self.kind == "spy_rules":  # a combined account may switch between overnight and multi-day holds
                for ps in positions.values():
                    ps.exit_rule = self._exit_rule()
            # 4) mark to market at the close
            value = 0.0
            for sym, ps in positions.items():
                b = data.bar(date, sym)
                px = b[2] if b is not None else ps.last_close
                ps.last_close = px
                ps.hi, ps.lo = max(ps.hi, px), min(ps.lo, px)
                value += ps.shares * px
            equity = cash + value
            mret, moc = data.market(date)
            day_trades = sum(1 for t in trades if t["date"] == date)
            rows.append({
                "date": pd.Timestamp(date), "equity_start": eq0, "equity": equity,
                "ret": equity / eq0 - 1.0 if eq0 else 0.0, "n_trades": day_trades,
                "market_ret": mret, "market_oc_ret": moc,
                "market_day": "up" if mret > self.flat_band else ("down" if mret < -self.flat_band else "flat"),
                "leverage": value / equity if equity else 0.0, "weekday": date.strftime("%a"),
            })
            for sym, action, shares in orders:
                decisions.append(Decision(date, SNAP_BAR, "15:50", 1 if action == "buy" else 0,
                                          f"{action}:{sym}:{shares}", float("nan")))  # size is part of the decision
            prev_equity = equity
        daily = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
        tdf = pd.DataFrame(trades)
        if not tdf.empty:
            tdf["date"] = pd.to_datetime(tdf["date"])
        return BacktestResult(daily=daily, trades=tdf, decisions=decisions, config=self.cfg)

    # ---- strategy rules (3:50 pm information only) ------------------------------------------
    def _exit_rule(self) -> str:
        if self.kind == "spy_overnight":
            return "next_open"
        if self.kind == "reversal":
            return self.p.get("exit", "next_open")
        if self.kind == "spy_rules":
            return getattr(self, "_exit_mode", "rule")
        return "rule"

    def _decide(self, data, date, positions, equity) -> list[tuple[str, str, int]]:
        orders = []
        if self.kind == "reversal":
            for sym, ps in positions.items():
                if ps.exit_rule == "next_close" and ps.entry_date < date:
                    orders.append((sym, "sell", 0))
        if self.kind in ("spy_overnight", "spy_rsi2"):
            sym = getattr(data, "symbol", "SPY")
            b = data.bar(date, sym)
            if b is None or b[1] is None:
                return orders
            snap = b[1]
            if self.kind == "spy_overnight":
                if sym not in positions:
                    orders.append((sym, "buy", math.floor(equity * self.max_lev / snap)))
                return orders
            closes = np.append(data.closes_before(sym, date, 260), snap)
            if len(closes) < int(self.p["sma_trend"]) + 1:
                return orders
            if sym in positions:
                if snap > closes[-int(self.p["sma_exit"]):].mean():
                    orders.append((sym, "sell", 0))
            else:
                rsi = rsi_wilder(closes[-60:], 2)
                if rsi < float(self.p["rsi_entry"]) and snap > closes[-int(self.p["sma_trend"]):].mean():
                    orders.append((sym, "buy", math.floor(equity * self.max_lev / snap)))
            return orders
        if self.kind == "spy_rules":
            return self._rules(data, date, positions, equity)
        if self.kind == "reversal":
            c = data.candidates(date)
            if c is None or c.empty:
                return orders
            c = c[(c["snap"] > float(self.p["min_price"])) & ((c["open"] / c["prev_close"] - 1).abs() <= SPLIT_GAP)]
            c = c.assign(move=c["snap"] / c["prev_close"] - 1.0)
            c = c[c["move"] < 0].sort_values(["move", "symbol"]).head(int(self.p["n_picks"]))
            budget = equity * self.max_lev / int(self.p["n_picks"])
            for sym, snap in zip(c["symbol"], c["snap"]):
                if sym not in positions:
                    orders.append((sym, "buy", math.floor(budget / snap)))
            return orders
        raise ValueError(f"Unknown daily strategy {self.kind}")

    def _month_position(self, data, date) -> tuple[int, int]:
        """(trading day of month, 1-based; trading days left after today) from the known calendar."""
        if not hasattr(self, "_cal"):
            ds = pd.Series(pd.to_datetime(data.dates))
            key = ds.dt.year * 12 + ds.dt.month
            first = ds.groupby(key).cumcount() + 1
            left = ds.groupby(key).transform("size") - first
            self._cal = {d: (int(f), int(l)) for d, f, l in zip(data.dates, first, left)}
        return self._cal[date]

    def _rules(self, data, date, positions, equity) -> list[tuple[str, str, int]]:
        """One rule, or several sharing one account (rule: any, rules: [...]): each rule keeps its
        own virtual in/out state; the account holds the symbol at 1x while any rule is in."""
        sym = getattr(data, "symbol", "SPY")
        b = data.bar(date, sym)
        if b is None or b[1] is None:
            return []
        snap, p = b[1], self.p
        rules = list(p["rules"]) if p["rule"] == "any" else [p["rule"]]
        if not hasattr(self, "_v"):
            self._v = {r: False for r in rules}
        self._v["down3_overnight"] = False  # always out after the next open
        closes = np.append(data.closes_before(sym, date, 260), snap)
        highs, lows = data.highs_lows_before(date, 6)
        if len(closes) < 30 or len(highs) < 6:
            return []
        ctx = {"snap": snap, "closes": closes, "highs": highs, "lows": lows}
        for r in rules:
            self._v[r] = (not self._rule_exit(r, data, date, ctx)) if self._v[r] else self._rule_entry(r, data, date, ctx)
        active = [r for r in rules if self._v[r]]
        self._exit_mode = "next_open" if active == ["down3_overnight"] else "rule"
        held = sym in positions
        if active and not held:
            return [(sym, "buy", math.floor(equity * self.max_lev / snap))]
        if held and not active:
            return [(sym, "sell", 0)]
        return []

    def _rule_entry(self, rule, data, date, c) -> bool:
        snap, closes, lows, p = c["snap"], c["closes"], c["lows"], self.p
        if rule == "ibs":
            h, l = data.range_to_snap(date)
            return ((snap - l) / (h - l) if h > l else 0.5) < float(p.get("ibs_entry", 0.2))
        if rule == "turnaround_tue":
            return date.weekday() == 0 and snap < closes[-2]
        if rule == "turn_of_month":
            return self._month_position(data, date)[1] == int(p.get("entry_days_left", 4))  # 5th-last day
        if rule == "down3_overnight":
            return bool((np.diff(closes[-4:]) < 0).all())
        if rule == "five_day_low":
            return snap < lows[-5:].min()
        if rule == "triple_rsi":
            if len(closes) < 201:
                return False
            r = self._rsi5(closes)
            return r[0] < 30 and r[0] < r[1] < r[2] < r[3] and r[3] < 60 and snap > closes[-200:].mean()
        if rule == "rsi2_high":
            return rsi_wilder(closes[-60:], 2) < float(p.get("rsi_entry", 10))
        raise ValueError(f"Unknown spy_rules rule {rule}")

    def _rule_exit(self, rule, data, date, c) -> bool:
        if rule == "turn_of_month":
            return self._month_position(data, date)[0] == int(self.p.get("exit_day", 3))
        if rule == "triple_rsi":
            return self._rsi5(c["closes"])[0] > 50
        if rule == "down3_overnight":
            return True
        return c["snap"] > c["highs"][-1]  # ibs, turnaround_tue, five_day_low, rsi2_high

    @staticmethod
    def _rsi5(closes):  # RSI(5) today, 1, 2 and 3 days ago
        n = len(closes)
        return [rsi_wilder(closes[max(0, n - 100 - k):n - k], 5) for k in range(4)]

    def _trade(self, ps: Pos, date, exit_bar, px, how) -> dict:
        gross = (px - ps.entry_px) * ps.shares
        costs = ps.cost + self.comm * ps.shares
        risk = 0.01 * ps.entry_px  # no stop: R-multiples are per 1% of price
        return {
            "date": date, "symbol": ps.symbol, "side": 1, "shares": int(round(ps.shares)),
            "entry_date": ps.entry_date, "entry_bar": SNAP_BAR + 10, "entry_time": "16:00",
            "entry_px": float(ps.entry_px), "exit_bar": exit_bar, "exit_time": "09:30" if how == "open" else "16:00",
            "exit_px": float(px), "entry_reason": self.kind, "exit_reason": how,
            "stop_at_entry": float("nan"), "gross": float(gross), "costs": float(costs),
            "pnl": float(gross - costs), "ret": float((gross - costs) / (ps.entry_px * ps.shares)),
            "r_multiple": float((gross - costs) / (risk * ps.shares)),
            "mae_r": float(max(ps.entry_px - ps.lo, 0) / risk), "mfe_r": float(max(ps.hi - ps.entry_px, 0) / risk),
            "days_held": (date - ps.entry_date).days,
        }
