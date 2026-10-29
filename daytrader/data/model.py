"""In-memory market data: one regular-session day of 1-minute bars per Day.

Minute index 0 is the 9:30-9:31 bar (Alpaca stamps bars with their start time),
so a bar's close at index i is the price at 9:31 + i minutes.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

ET = "America/New_York"
SESSION_MINUTES = 390  # 9:30 to 16:00


@dataclass(frozen=True)
class Day:
    date: dt.date
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __post_init__(self) -> None:
        for arr in (self.open, self.high, self.low, self.close, self.volume):
            arr.setflags(write=False)

    @property
    def n(self) -> int:
        return len(self.close)

    @property
    def session_open(self) -> float:
        return float(self.open[0])

    @property
    def session_close(self) -> float:
        return float(self.close[-1])


def minute_to_clock(i: int) -> str:
    """Wall-clock time at the END of bar i, e.g. 0 -> '09:31', 29 -> '10:00'."""
    total = 9 * 60 + 30 + i + 1
    return f"{total // 60:02d}:{total % 60:02d}"


def clock_to_bar_index(hhmm: str) -> int:
    """Index of the bar whose close is the price at wall-clock hhmm (e.g. '10:00' -> 29)."""
    h, m = (int(x) for x in hhmm.split(":"))
    return h * 60 + m - (9 * 60 + 30) - 1


def days_from_frame(df: pd.DataFrame) -> list[Day]:
    """Build regular-session Days from a frame with columns t (UTC), o, h, l, c, v.

    Missing minutes inside the session are filled flat at the previous close with
    zero volume. Days that end early (half-days) keep their shorter length.
    """
    if df.empty:
        return []
    ts = pd.to_datetime(df["t"], utc=True).dt.tz_convert(ET)
    frame = df.assign(ts=ts)
    minute = (ts.dt.hour * 60 + ts.dt.minute) - (9 * 60 + 30)
    frame = frame.assign(minute=minute, date=ts.dt.date)
    frame = frame[(frame["minute"] >= 0) & (frame["minute"] < SESSION_MINUTES)]

    days: list[Day] = []
    for date, g in frame.groupby("date", sort=True):
        g = g.sort_values("minute").drop_duplicates("minute", keep="last")
        if g.empty or g["minute"].iloc[0] > 5:
            continue  # no usable opening bars: skip the day rather than invent an open
        last = int(g["minute"].iloc[-1])
        length = SESSION_MINUTES if last >= SESSION_MINUTES - 10 else last + 1
        idx = np.arange(length)
        g = g.set_index("minute").reindex(idx)
        close = np.array(g["c"].ffill().to_numpy(dtype=float), copy=True)  # pandas may hand back read-only views
        if np.isnan(close[0]):
            close[0] = float(g["o"].dropna().iloc[0])
            close = np.array(pd.Series(close).ffill().to_numpy(), copy=True)
        open_ = np.array(g["o"].to_numpy(dtype=float), copy=True)
        open_ = np.where(np.isnan(open_), close, open_)
        high = np.where(np.isnan(g["h"].to_numpy(dtype=float)), close, g["h"].to_numpy(dtype=float))
        low = np.where(np.isnan(g["l"].to_numpy(dtype=float)), close, g["l"].to_numpy(dtype=float))
        vol = np.nan_to_num(np.array(g["v"].to_numpy(dtype=float), copy=True), nan=0.0)
        days.append(Day(date=date, open=open_, high=high, low=low, close=close, volume=vol))
    return days
