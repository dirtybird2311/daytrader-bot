"""SPY intraday momentum with a time-of-day "noise area".

Rules (Zarattini, Aziz & Barbon, 2024):
  * sigma(t, HH:MM) = mean over the prior N days of |Close(HH:MM) / Open(9:30) - 1|
  * Upper = max(Open today, Close yesterday) * (1 + VM * sigma)
    Lower = min(Open today, Close yesterday) * (1 - VM * sigma)
  * Entries only at HH:00 / HH:30 from 10:00: long above Upper, short below Lower.
  * Trailing stop: long max(Upper, VWAP); short min(Lower, VWAP).
  * Crossing to the opposite band at a check reverses the position.
  * Flat at the close. Size targets 2% daily vol, capped at 4x leverage.
"""

from __future__ import annotations

import math

import numpy as np

from ..data.model import clock_to_bar_index
from ..data.view import History, TodayView
from .base import Intent, Strategy


class SpyNoiseArea(Strategy):
    name = "spy_noise"

    def __init__(self, cfg: dict):
        p, s = cfg["params"], cfg["sizing"]
        self.lookback = int(p["lookback_days"])
        self.vm = float(p["vol_multiplier"])
        self.first_check = clock_to_bar_index(p["first_check"])
        self.step = int(p["check_every_min"])
        self.stop_every_minute = p["stop_check"] == "every_minute"
        self.allow_short = bool(p["allow_short"])
        self.sizing_mode = s["mode"]
        self.target_vol = float(s["target_daily_vol"])
        self.max_lev = float(s["max_leverage"])
        self.vol_lookback = int(s["vol_lookback_days"])
        self._ready = False

    # ---- set-up from completed days only -------------------------------------------------
    def on_day_start(self, history: History, equity: float) -> bool:
        self._ready = False
        self._pending_setup = False
        self.leverage = float("nan")
        need = max(self.lookback, self.vol_lookback) + 1
        if len(history) < need:
            return False
        past = history.last(self.lookback)
        width = max(d.n for d in past)
        moves = np.full((len(past), width), np.nan)
        for k, d in enumerate(past):
            moves[k, : d.n] = np.abs(d.close / d.session_open - 1.0)
        self._sigma = self.vm * np.nanmean(moves, axis=0)
        self._prev_close = history.day(1).session_close

        closes = np.array([d.session_close for d in history.last(self.vol_lookback + 1)])
        rets = np.diff(closes) / closes[:-1]
        self._daily_vol = float(np.std(rets, ddof=1)) if len(rets) > 1 else float("nan")
        self._equity = equity
        self._pending_setup = True
        return True

    def _setup_open(self, today: TodayView) -> None:
        o = today.session_open
        self._ub_base = max(o, self._prev_close)
        self._lb_base = min(o, self._prev_close)
        if self.sizing_mode == "vol_target" and self._daily_vol > 0 and math.isfinite(self._daily_vol):
            lev = min(self.max_lev, self.target_vol / self._daily_vol)
        else:
            lev = 1.0
        self.leverage = lev
        self._shares = int(self._equity * lev / o) if o > 0 else 0
        self._ready = self._shares > 0
        self._pending_setup = False

    def bands(self, i: int) -> tuple[float, float]:
        sig = self._sigma[min(i, len(self._sigma) - 1)]
        if not np.isfinite(sig):
            sig = float(np.nanmax(self._sigma))
        return self._ub_base * (1 + sig), self._lb_base * (1 - sig)

    def is_check(self, i: int, session_length: int) -> bool:
        return i >= self.first_check and (i - self.first_check) % self.step == 0 and i < session_length - 1

    # ---- per bar ----------------------------------------------------------------------------
    def on_bar(self, today: TodayView, position: int) -> Intent | None:
        i = today.cursor
        if getattr(self, "_pending_setup", False) and i == 0:
            self._setup_open(today)
        if not self._ready:
            return None
        price = today.price()
        ub, lb = self.bands(i)
        check = self.is_check(i, today.session_length)
        info = {"ub": ub, "lb": lb}

        if position == 0:
            if not check:
                return None
            if price > ub:
                vwap = today.vwap()
                return Intent(1, self._shares, "entry_long", stop=max(ub, vwap), info={**info, "vwap": vwap})
            if price < lb and self.allow_short:
                vwap = today.vwap()
                return Intent(-1, self._shares, "entry_short", stop=min(lb, vwap), info={**info, "vwap": vwap})
            return None

        if not (check or self.stop_every_minute):
            return None
        vwap = today.vwap()
        info["vwap"] = vwap
        if position > 0:
            stop = max(ub, vwap)
            if price < stop:
                if check and price < lb and self.allow_short:
                    return Intent(-1, self._shares, "reversal_short", stop=min(lb, vwap), info=info)
                return Intent(0, 0, "trailing_stop", info=info)
        else:
            stop = min(lb, vwap)
            if price > stop:
                if check and price > ub:
                    return Intent(1, self._shares, "reversal_long", stop=max(ub, vwap), info=info)
                return Intent(0, 0, "trailing_stop", info=info)
        return None
