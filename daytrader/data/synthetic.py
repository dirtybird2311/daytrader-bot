"""Synthetic SPY-like minute data for tests and dry runs.

Clearly fake: reports built from it carry a SYNTHETIC banner. Optional intraday
momentum lets tests check that the strategy reacts to trend days.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from .model import SESSION_MINUTES, Day


def business_days(start: dt.date, n: int) -> list[dt.date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def make_days(n_days: int = 300, start: dt.date = dt.date(2018, 1, 2), seed: int = 0,
              momentum: float = 0.0, price: float = 250.0) -> list[Day]:
    """Random-walk minute bars with a U-shaped volume profile.

    momentum > 0 adds a per-day drift in the direction of the first half hour's move,
    which is the effect the noise-area strategy is designed to capture.
    """
    rng = np.random.default_rng(seed)
    days: list[Day] = []
    minutes = np.arange(SESSION_MINUTES)
    u_shape = 1.0 + 2.5 * (np.exp(-minutes / 25) + np.exp(-(SESSION_MINUTES - 1 - minutes) / 25))
    for date in business_days(start, n_days):
        gap = rng.normal(0, 0.004)
        open_px = price * (1 + gap)
        day_vol = abs(rng.normal(0.00045, 0.00015)) + 0.0002  # roughly 1% daily moves
        rets = rng.normal(0, day_vol, SESSION_MINUTES)
        if momentum:
            early = rets[:30].sum()
            rets[30:] += np.sign(early) * momentum * day_vol
        path = open_px * np.exp(np.cumsum(rets))
        opens = np.concatenate([[open_px], path[:-1]])
        wick = np.abs(rng.normal(0, day_vol / 2, SESSION_MINUTES)) * path
        highs = np.maximum(opens, path) + wick
        lows = np.minimum(opens, path) - wick
        vol = (u_shape * rng.lognormal(10, 0.3, SESSION_MINUTES)).round()
        days.append(Day(date=date, open=opens.copy(), high=highs, low=lows, close=path.copy(), volume=vol))
        price = float(path[-1])
    return days


def leveraged_days(days: list[Day], mult: float = 3.0, start_price: float = 50.0) -> dict:
    """A made-up daily-reset leveraged ETF of `days` (minute returns x mult), keyed by date. Tests only."""
    out, px = {}, start_price
    prev_close = None
    for d in days:
        base = d.close[-1] if prev_close is None else prev_close
        ref = days[0].open[0] if prev_close is None else base

        def lev(a):
            return px * (1 + mult * (a / ref - 1))

        o, h, l, c = lev(d.open), lev(d.high), lev(d.low), lev(d.close)
        hi, lo = np.maximum.reduce([o, h, l, c]), np.minimum.reduce([o, h, l, c])
        out[d.date] = Day(date=d.date, open=o, high=hi, low=lo, close=c, volume=d.volume.copy())
        px, prev_close = float(c[-1]), float(d.close[-1])
    return out
