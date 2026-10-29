"""Mean reversion to VWAP on SPY.

Bands = session VWAP +/- k x the volume-weighted standard deviation of the typical price
around VWAP, both from 9:30 through the current minute.
  * checks every `check_every_min` minutes from `first_check` to `last_entry`:
    long below VWAP - entry_sd x sd, short above VWAP + entry_sd x sd
  * exits checked every minute: back to VWAP (target) or beyond VWAP -/+ stop_sd x sd (stop)
  * at most `max_entries` entries a day; anything open is closed at 16:00 by the engine
"""

from __future__ import annotations

import numpy as np

from ..data.model import clock_to_bar_index
from ..data.view import History, TodayView
from .base import Intent, Strategy, vol_target_leverage


class SpyVwapRevert(Strategy):
    name = "spy_vwap_revert"

    def __init__(self, cfg: dict):
        p = cfg["params"]
        self.sizing = cfg["sizing"]
        self.entry_sd = float(p["entry_sd"])
        self.stop_sd = float(p["stop_sd"])
        self.first = clock_to_bar_index(p["first_check"])
        self.last = clock_to_bar_index(p["last_entry"])
        self.step = int(p["check_every_min"])
        self.max_entries = int(p["max_entries"])
        self.allow_short = bool(p["allow_short"])
        self.min_history = int(self.sizing.get("vol_lookback_days", 14)) + 1

    def on_day_start(self, history: History, equity: float) -> bool:
        self.leverage = float("nan")
        if len(history) < self.min_history:
            return False
        self._lev = vol_target_leverage(history, self.sizing)
        self._equity = equity
        self._entries = 0
        self._shares = 0
        return True

    @staticmethod
    def bands(today: TodayView) -> tuple[float, float]:
        tp = (today.high + today.low + today.close) / 3.0
        v = today.volume
        tot = v.sum()
        if tot <= 0:
            return float(today.price()), 0.0
        vwap = float((tp * v).sum() / tot)
        sd = float(np.sqrt((v * (tp - vwap) ** 2).sum() / tot))
        return vwap, sd

    def on_bar(self, today: TodayView, position: int) -> Intent | None:
        i = today.cursor
        if i == 0:
            self.leverage = self._lev
            o = today.session_open
            self._shares = int(self._equity * self._lev / o) if o > 0 and self._lev == self._lev else 0
        if self._shares <= 0 or i >= today.session_length - 1:
            return None
        price = today.price()
        vwap, sd = self.bands(today)
        info = {"vwap": vwap, "sd": sd}
        if position == 0:
            check = self.first <= i <= self.last and (i - self.first) % self.step == 0
            if not check or self._entries >= self.max_entries or sd <= 0:
                return None
            if price < vwap - self.entry_sd * sd:
                self._entries += 1
                return Intent(1, self._shares, "entry_long", stop=vwap - self.stop_sd * sd, info=info)
            if price > vwap + self.entry_sd * sd and self.allow_short:
                self._entries += 1
                return Intent(-1, self._shares, "entry_short", stop=vwap + self.stop_sd * sd, info=info)
            return None
        if position > 0:
            if price >= vwap:
                return Intent(0, 0, "target_vwap", info=info)
            if price < vwap - self.stop_sd * sd:
                return Intent(0, 0, "stop", info=info)
        else:
            if price <= vwap:
                return Intent(0, 0, "target_vwap", info=info)
            if price > vwap + self.stop_sd * sd:
                return Intent(0, 0, "stop", info=info)
        return None
