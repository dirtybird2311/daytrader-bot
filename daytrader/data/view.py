"""Point-in-time views: the only way a strategy can see market data.

A strategy never receives the full dataset. At each step it gets:
  * History: completed days strictly before today, and
  * TodayView: today's bars up to and including the current bar.
Reading past the cursor raises an error, so lookahead is structurally impossible
rather than a matter of discipline.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from .model import Day


class LookaheadError(RuntimeError):
    """Raised when code asks for data that would not exist yet."""


class History:
    """Completed days before today, oldest first. Read-only."""

    def __init__(self, days: list[Day], today_index: int):
        self._days = days
        self._end = today_index  # exclusive

    def __len__(self) -> int:
        return self._end

    def last(self, n: int) -> list[Day]:
        start = max(0, self._end - n)
        return self._days[start:self._end]

    def day(self, back: int) -> Day:
        """back=1 is yesterday."""
        if back < 1 or back > self._end:
            raise LookaheadError(f"History.day({back}) outside the {self._end} completed days")
        return self._days[self._end - back]


class TodayView:
    """Today's bars up to the cursor (inclusive). The engine advances the cursor."""

    def __init__(self, day: Day):
        self._day = day
        self._cursor = -1

    @property
    def date(self) -> dt.date:
        return self._day.date

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def session_length(self) -> int:
        """Number of bars in today's session. Known in advance from the exchange calendar."""
        return self._day.n

    def _advance(self, i: int) -> None:  # engine only
        if i != self._cursor + 1:
            raise RuntimeError("cursor must advance one bar at a time")
        self._cursor = i

    def _slice(self, arr: np.ndarray) -> np.ndarray:
        if self._cursor < 0:
            raise LookaheadError("no bars available before the open")
        return arr[: self._cursor + 1]

    @property
    def open(self) -> np.ndarray:
        return self._slice(self._day.open)

    @property
    def high(self) -> np.ndarray:
        return self._slice(self._day.high)

    @property
    def low(self) -> np.ndarray:
        return self._slice(self._day.low)

    @property
    def close(self) -> np.ndarray:
        return self._slice(self._day.close)

    @property
    def volume(self) -> np.ndarray:
        return self._slice(self._day.volume)

    @property
    def session_open(self) -> float:
        return float(self._slice(self._day.open)[0])

    def price(self) -> float:
        return float(self._slice(self._day.close)[-1])

    def vwap(self) -> float:
        """Regular-session VWAP through the current bar, using typical price."""
        h, l, c, v = self.high, self.low, self.close, self.volume
        vol = v.sum()
        if vol <= 0:
            return float(c[-1])
        return float(((h + l + c) / 3.0 * v).sum() / vol)
