"""Read cached bars into Days."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from .model import Day, days_from_frame


def load_days(symbol: str, start: dt.date, end: dt.date | None, cache_dir: str | Path) -> list[Day]:
    folder = Path(cache_dir) / "bars" / symbol
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No cached bars for {symbol} in {folder}. Run: python -m daytrader download --config <file>")
    frames = []
    for f in files:
        y, m = (int(x) for x in f.stem.split("-"))
        month = dt.date(y, m, 1)
        if month < dt.date(start.year, start.month, 1):
            continue
        if end is not None and month > end:
            continue
        frames.append(pd.read_parquet(f))
    if not frames:
        return []
    days = days_from_frame(pd.concat(frames, ignore_index=True))
    return [d for d in days if d.date >= start and (end is None or d.date <= end)]
