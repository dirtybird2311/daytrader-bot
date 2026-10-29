"""Minute-by-minute replay of each historical day as if it were today.

For every day:
  1. The strategy is set up from completed days only (History).
  2. Bars arrive one at a time; the strategy sees only bars up to the cursor.
  3. A decision made on a bar's close fills `fill_delay_bars` later at that bar's
     open, worsened by slippage. Never at the price used to decide.
  4. Anything still open is closed at the final bar's close (market-on-close).
Commissions and slippage are charged on every share, both sides.

Optional `execution` config (research batch C2): decide on one symbol's bars (e.g. SPY) but
trade another (e.g. UPRO, 3x SPY) at that symbol's own minute prices. The position's notional
is the signal's notional / multiplier, capped at max_leverage x equity. Days where the traded
symbol's session doesn't line up with the signal's are sat out.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..data.model import Day, minute_to_clock
from ..data.view import History, TodayView
from ..strategies.base import Intent, Strategy


@dataclass
class Trade:
    date: dt.date
    side: int
    shares: int
    entry_bar: int
    entry_time: str
    entry_px: float
    exit_bar: int = -1
    exit_time: str = ""
    exit_px: float = float("nan")
    exit_reason: str = ""
    entry_reason: str = ""
    stop_at_entry: float = float("nan")
    costs: float = 0.0
    hi: float = float("-inf")
    lo: float = float("inf")

    @property
    def gross(self) -> float:
        return self.side * (self.exit_px - self.entry_px) * self.shares

    @property
    def pnl(self) -> float:
        return self.gross - self.costs


@dataclass
class Decision:
    date: dt.date
    bar: int
    time: str
    target: int
    reason: str
    price: float
    info: dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    decisions: list[Decision]
    config: dict

    def decision_keys(self) -> list[tuple]:
        return [(d.date, d.bar, d.target, d.reason) for d in self.decisions]


class Engine:
    def __init__(self, cfg: dict, strategy: Strategy):
        c = cfg["costs"]
        self.cfg = cfg
        self.strategy = strategy
        self.commission = float(c["commission_per_share"])
        self.slippage = float(c["slippage_per_share"])
        self.delay = int(c["fill_delay_bars"])
        self.flat_band = float(cfg.get("benchmark", {}).get("flat_band", 0.003))
        ex = cfg.get("execution") or {}
        self.exec_symbol = ex.get("symbol")
        self.exec_mult = float(ex.get("multiplier", 1.0))
        self.exec_max_lev = float(ex.get("max_leverage", 1.0))
        self.exec_skipped = 0

    def run(self, days: list[Day], start: dt.date | None = None, end: dt.date | None = None,
            exec_days: dict | None = None) -> BacktestResult:
        equity = float(self.cfg["account"]["starting_equity"])
        daily_rows, trades, decisions = [], [], []

        for idx, day in enumerate(days):
            if (start and day.date < start) or (end and day.date > end):
                continue
            history = History(days, idx)
            equity_start = equity
            today = TodayView(day)
            active = self.strategy.on_day_start(history, equity)
            self._ex = None
            if self.exec_symbol:
                self._ex = (exec_days or {}).get(day.date)
                if self._ex is None or self._ex.n != day.n:
                    if active:
                        self.exec_skipped += 1
                    active, self._ex = False, None
            self._equity_start = equity_start

            position, shares, open_trade = 0, 0, None
            pending: list[tuple[int, Intent, int]] = []  # (fill_bar, intent, decided_bar)
            day_trades: list[Trade] = []

            for i in range(day.n):
                today._advance(i)
                # 1) fills scheduled for this bar happen at its open
                for fill_bar, intent, decided in [p for p in pending if p[0] == i]:
                    pending.remove((fill_bar, intent, decided))
                    position, shares, open_trade = self._fill(day, i, intent, position, shares, open_trade, day_trades)
                # 2) track excursion of the open trade (in the traded symbol's prices)
                if open_trade is not None:
                    px_day = self._ex if self._ex is not None else day
                    open_trade.hi = max(open_trade.hi, px_day.high[i])
                    open_trade.lo = min(open_trade.lo, px_day.low[i])
                # 3) strategy decides on this bar's close
                if not active:
                    continue
                planned = pending[-1][1].target if pending else position
                intent = self.strategy.on_bar(today, planned)
                if intent is None or intent.target == planned:
                    continue
                decisions.append(Decision(day.date, i, minute_to_clock(i), intent.target, intent.reason,
                                          float(day.close[i]), dict(intent.info)))
                fill_bar = i + self.delay
                if fill_bar < day.n:
                    pending.append((fill_bar, intent, i))

            # 4) market-on-close for anything still open
            if position != 0 and open_trade is not None:
                last = day.n - 1
                px_day = self._ex if self._ex is not None else day
                px = px_day.close[last] - np.sign(position) * self.slippage
                self._close(open_trade, last, px, "close", shares)
                position, shares, open_trade = 0, 0, None

            pnl = sum(t.pnl for t in day_trades)
            equity = equity_start + pnl
            trades.extend(day_trades)
            daily_rows.append(self._daily_row(days, idx, equity_start, equity, day_trades))

        daily = pd.DataFrame(daily_rows).set_index("date") if daily_rows else pd.DataFrame()
        trade_df = self._trades_frame(trades)
        return BacktestResult(daily=daily, trades=trade_df, decisions=decisions, config=self.cfg)

    # ---- fills ------------------------------------------------------------------------------
    def _fill(self, day: Day, i: int, intent: Intent, position: int, shares: int,
              open_trade: Trade | None, day_trades: list[Trade]):
        px_day = self._ex if self._ex is not None else day
        raw = float(px_day.open[i])
        if self._ex is not None and intent.target != 0 and intent.shares > 0:
            signal_notional = intent.shares * float(day.open[0])  # what the strategy sized for
            notional = min(signal_notional / self.exec_mult, self._equity_start * self.exec_max_lev)
            intent = Intent(intent.target, int(notional / raw), intent.reason, None, intent.info)
        if position != 0 and open_trade is not None:
            px = raw - np.sign(position) * self.slippage  # selling a long fills lower, covering a short higher
            self._close(open_trade, i, px, intent.reason, shares)
            position, shares, open_trade = 0, 0, None
        if intent.target != 0 and intent.shares > 0:
            px = raw + intent.target * self.slippage
            open_trade = Trade(date=day.date, side=intent.target, shares=int(intent.shares), entry_bar=i,
                               entry_time=minute_to_clock(i - 1), entry_px=px, entry_reason=intent.reason,
                               stop_at_entry=float(intent.stop) if intent.stop is not None else float("nan"),
                               costs=self.commission * intent.shares, hi=float(px_day.high[i]), lo=float(px_day.low[i]))
            day_trades.append(open_trade)
            position, shares = intent.target, int(intent.shares)
        return position, shares, open_trade

    def _close(self, trade: Trade, i: int, px: float, reason: str, shares: int) -> None:
        trade.exit_bar = i
        trade.exit_time = minute_to_clock(i) if reason == "close" else minute_to_clock(i - 1)
        trade.exit_px = float(px)
        trade.exit_reason = reason
        trade.costs += self.commission * shares
        # slippage is already inside the fill prices

    # ---- records ----------------------------------------------------------------------------
    def _daily_row(self, days: list[Day], idx: int, eq0: float, eq1: float, day_trades: list[Trade]) -> dict:
        day = days[idx]
        prev_close = days[idx - 1].session_close if idx > 0 else day.session_open
        spy_ret = day.session_close / prev_close - 1.0
        if spy_ret > self.flat_band:
            mkt = "up"
        elif spy_ret < -self.flat_band:
            mkt = "down"
        else:
            mkt = "flat"
        lev = getattr(self.strategy, "leverage", np.nan)
        return {
            "date": pd.Timestamp(day.date), "equity_start": eq0, "equity": eq1,
            "ret": (eq1 / eq0 - 1.0) if eq0 else 0.0, "n_trades": len(day_trades),
            "market_ret": spy_ret, "market_oc_ret": day.session_close / day.session_open - 1.0,
            "market_day": mkt, "leverage": lev, "weekday": day.date.strftime("%a"),
        }

    @staticmethod
    def _trades_frame(trades: list[Trade]) -> pd.DataFrame:
        if not trades:
            return pd.DataFrame(columns=["date", "side", "shares", "entry_time", "entry_px", "exit_time", "exit_px",
                                         "entry_reason", "exit_reason", "gross", "costs", "pnl", "ret", "r_multiple",
                                         "mae_r", "mfe_r", "stop_at_entry"])
        rows = []
        for t in trades:
            notional = t.entry_px * t.shares
            risk_ps = abs(t.entry_px - t.stop_at_entry) if np.isfinite(t.stop_at_entry) else np.nan
            risk_ps = max(risk_ps, 0.001 * t.entry_px) if np.isfinite(risk_ps) else 0.001 * t.entry_px
            adverse = (t.entry_px - t.lo) if t.side > 0 else (t.hi - t.entry_px)
            favorable = (t.hi - t.entry_px) if t.side > 0 else (t.entry_px - t.lo)
            row = asdict(t)
            row.pop("hi"), row.pop("lo")
            row.update(gross=t.gross, pnl=t.pnl, ret=t.pnl / notional if notional else 0.0,
                       r_multiple=t.pnl / (risk_ps * t.shares), mae_r=max(adverse, 0) / risk_ps,
                       mfe_r=max(favorable, 0) / risk_ps)
            rows.append(row)
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df
