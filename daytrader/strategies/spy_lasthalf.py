"""First half hour predicts the last half hour (Gao, Han, Li & Zhou, JFE 2018).

  * signal = SPY return from the prior close to 10:00 ET
  * at 15:30: long if the signal is positive, short if negative (shorts optional)
  * exit at the 16:00 close (the engine closes anything still open)
Sizing: the same volatility target as the SPY momentum strategy.
"""

from __future__ import annotations

from ..data.model import clock_to_bar_index
from ..data.view import History, TodayView
from .base import Intent, Strategy, vol_target_leverage


class SpyLastHalfHour(Strategy):
    name = "spy_lasthalf"

    def __init__(self, cfg: dict):
        p = cfg["params"]
        self.sizing = cfg["sizing"]
        self.signal_bar = clock_to_bar_index(p.get("signal_time", "10:00"))
        self.entry_bar = clock_to_bar_index(p.get("entry_time", "15:30"))
        self.allow_short = bool(p["allow_short"])
        self.min_history = int(self.sizing.get("vol_lookback_days", 14)) + 1

    def on_day_start(self, history: History, equity: float) -> bool:
        self.leverage = float("nan")
        if len(history) < self.min_history:
            return False
        self._prev_close = history.day(1).session_close
        self._lev = vol_target_leverage(history, self.sizing)
        self._equity = equity
        self._signal = None
        self._shares = 0
        return True

    def on_bar(self, today: TodayView, position: int) -> Intent | None:
        i = today.cursor
        if i == 0:
            self.leverage = self._lev
            o = today.session_open
            self._shares = int(self._equity * self._lev / o) if o > 0 and self._lev == self._lev else 0
        if i == self.signal_bar:
            self._signal = today.price() / self._prev_close - 1.0
        if i != self.entry_bar or position != 0 or self._signal is None or self._shares <= 0:
            return None
        if i >= today.session_length - 1:
            return None  # half-day: no last half hour to trade
        info = {"signal": self._signal}
        if self._signal > 0:
            return Intent(1, self._shares, "entry_long", info=info)
        if self._signal < 0 and self.allow_short:
            return Intent(-1, self._shares, "entry_short", info=info)
        return None
