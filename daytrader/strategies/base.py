"""Strategy interface shared by the backtester and (later) the live engine.

A strategy only ever sees History and TodayView. It returns Intents; it never
places orders, sizes against the broker or reads the clock itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..data.view import History, TodayView


@dataclass
class Intent:
    target: int               # desired position direction: +1 long, -1 short, 0 flat
    shares: int               # share count for the new position (ignored when target == 0)
    reason: str               # e.g. "entry_long", "stop", "reversal"
    stop: float | None = None  # protective level at decision time, for R-multiples
    info: dict = field(default_factory=dict)


class Strategy:
    name = "base"

    def on_day_start(self, history: History, equity: float) -> bool:
        """Called before the open. Return False to sit the day out (e.g. not enough history)."""
        raise NotImplementedError

    def on_bar(self, today: TodayView, position: int) -> Intent | None:
        """Called at each bar close. position is the current (or already-ordered) direction."""
        raise NotImplementedError


def vol_target_leverage(history: History, sizing: dict) -> float:
    """Leverage that targets `target_daily_vol`, from completed days' close-to-close returns only."""
    import math

    import numpy as np

    if sizing.get("mode", "vol_target") != "vol_target":
        return 1.0
    n = int(sizing.get("vol_lookback_days", 14))
    closes = np.array([d.session_close for d in history.last(n + 1)])
    if len(closes) < 3:
        return float("nan")
    rets = np.diff(closes) / closes[:-1]
    vol = float(np.std(rets, ddof=1))
    if not (vol > 0 and math.isfinite(vol)):
        return 1.0
    return min(float(sizing["max_leverage"]), float(sizing["target_daily_vol"]) / vol)
