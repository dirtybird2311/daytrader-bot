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
