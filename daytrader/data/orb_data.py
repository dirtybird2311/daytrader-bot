"""Data sources for the stocks-in-play backtest.

The engine needs three things, all point-in-time:
  * features(date): one row per candidate with what was known at 9:35 (see universe.build_features)
  * minute_day(date, symbol): that stock's 1-minute bars for the day (the engine only reads
    bars up to its cursor through TodayView)
  * market(date): SPY's close-to-close and open-to-close return, for benchmarking only
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from .model import SESSION_MINUTES, Day, days_from_frame
from .ta import ta_table
from .universe import build_features


class OrbData:
    dates: list[dt.date]

    def features(self, date: dt.date) -> pd.DataFrame:
        raise NotImplementedError

    def minute_day(self, date: dt.date, symbol: str) -> Day | None:
        raise NotImplementedError

    def market(self, date: dt.date) -> tuple[float, float]:
        raise NotImplementedError

    def daily_frame(self) -> pd.DataFrame:
        """symbol, date, o, c for every candidate symbol (only for the technical filters)."""
        raise NotImplementedError

    def ta_features(self, date: dt.date, sma_days: int = 20) -> pd.DataFrame:
        cache = self.__dict__.setdefault("_ta_cache", {})
        if sma_days not in cache:
            t = ta_table(self.daily_frame(), sma_days=sma_days)
            cache[sma_days] = {d: g.set_index("symbol") for d, g in t.groupby("date")}
        return cache[sma_days].get(date, pd.DataFrame())

    def tick_prices(self, date: dt.date, symbol: str, bar: int) -> np.ndarray | None:
        """Round-lot trade prices, in time order, inside one minute (see data/ticks.py)."""
        return None


class CachedOrbData(OrbData):
    def __init__(self, cache_dir: str | Path):
        self.root = Path(cache_dir) / "universe"
        fpath = self.root / "features.parquet"
        if not fpath.exists():
            raise FileNotFoundError("No stocks-in-play data yet. Run: python -m daytrader download --config config/orb.yaml")
        f = pd.read_parquet(fpath)
        self._features = {d: g for d, g in f.groupby("date")}
        self._minute_months = {p.stem for p in (self.root / "minute").glob("*.parquet")}
        spy = self._load_spy()
        self._market = spy
        self.dates = sorted(d for d in spy.index if d in self._features)
        self._month_key = None
        self._month_days: dict[tuple, Day] = {}

    def _load_spy(self) -> pd.DataFrame:
        frames = []
        for f in sorted((self.root / "daily").glob("*.parquet")):
            df = pd.read_parquet(f, filters=[("symbol", "==", "SPY")])
            frames.append(df)
        spy = pd.concat(frames, ignore_index=True)
        spy["date"] = pd.to_datetime(spy["t"], utc=True).dt.tz_convert("America/New_York").dt.date
        spy = spy.drop_duplicates("date").set_index("date").sort_index()
        spy["ret"] = spy["c"] / spy["c"].shift(1) - 1
        spy["oc"] = spy["c"] / spy["o"] - 1
        return spy

    def daily_frame(self) -> pd.DataFrame:
        syms = sorted({str(x) for x in pd.concat(self._features.values())["symbol"].unique()} | {"SPY"})
        frames = []
        for f in sorted((self.root / "daily").glob("*.parquet")):
            df = pd.read_parquet(f, columns=["symbol", "t", "o", "c"], filters=[("symbol", "in", syms)])
            frames.append(df)
        d = pd.concat(frames, ignore_index=True)
        d["date"] = pd.to_datetime(d["t"], utc=True).dt.tz_convert("America/New_York").dt.date
        return d.drop(columns="t").drop_duplicates(["symbol", "date"], keep="last")

    def tick_prices(self, date: dt.date, symbol: str, bar: int) -> np.ndarray | None:
        from .ticks import TickStore
        if not hasattr(self, "_ticks"):
            self._ticks = TickStore(self.root / "ticks")
        return self._ticks.get(date, symbol, bar)

    def has_minutes(self, date: dt.date) -> bool:
        return f"{date:%Y-%m}" in self._minute_months

    def features(self, date: dt.date) -> pd.DataFrame:
        return self._features.get(date, pd.DataFrame())

    def minute_day(self, date: dt.date, symbol: str) -> Day | None:
        key = f"{date:%Y-%m}"
        if key != self._month_key:
            self._month_days = {}
            path = self.root / "minute" / f"{key}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                for sym, g in df.groupby("symbol"):
                    for d in days_from_frame(g.drop(columns="symbol")):
                        self._month_days[(d.date, sym)] = d
            self._month_key = key
        return self._month_days.get((date, symbol))

    def market(self, date: dt.date) -> tuple[float, float]:
        row = self._market.loc[date]
        return float(row["ret"]), float(row["oc"])


class SyntheticOrbData(OrbData):
    """A made-up market of n_symbols stocks. Some days a stock gets 'news': heavy opening
    volume and (if momentum > 0) a drift in the direction of its first five minutes, which is
    the effect the stocks-in-play strategy is built to capture. Clearly fake; for tests only.
    """

    def __init__(self, n_days: int = 300, n_symbols: int = 60, seed: int = 0, momentum: float = 0.0,
                 start: dt.date = dt.date(2018, 1, 2), news_prob: float = 0.04):
        self.n_symbols, self.seed, self.momentum, self.news_prob = n_symbols, seed, momentum, news_prob
        rng = np.random.default_rng(seed)
        self.symbols = [f"S{i:03d}" for i in range(n_symbols)]
        self.dates = []
        d = start
        while len(self.dates) < n_days:
            if d.weekday() < 5:
                self.dates.append(d)
            d += dt.timedelta(days=1)
        self.base_price = rng.uniform(8, 120, n_symbols)
        self.base_vol = rng.lognormal(14.2, 0.6, n_symbols)  # ~1.5M shares/day typical
        self.daily_sigma = rng.uniform(0.012, 0.04, n_symbols)
        self._cache_idx = None
        self._build()

    def _day_matrix(self, k: int, prev_close: np.ndarray):
        """All symbols' minute bars for day k, regenerated deterministically on demand."""
        rng = np.random.default_rng((self.seed, k))
        n, m = self.n_symbols, SESSION_MINUTES
        news = rng.random(n) < self.news_prob
        gap = rng.normal(0, 0.006, n) + news * rng.normal(0, 0.03, n)
        open_px = prev_close * (1 + gap)
        sig = self.daily_sigma / np.sqrt(m) * (1 + news * 1.5)
        rets = rng.normal(0, 1, (n, m)) * sig[:, None]
        if self.momentum:
            early = rets[:, :5].sum(axis=1)
            rets[:, 5:] += (news * np.sign(early) * self.momentum * sig)[:, None]
        close = open_px[:, None] * np.exp(np.cumsum(rets, axis=1))
        opens = np.concatenate([open_px[:, None], close[:, :-1]], axis=1)
        wick = np.abs(rng.normal(0, 1, (n, m))) * sig[:, None] * close * 0.5
        high = np.maximum(opens, close) + wick
        low = np.minimum(opens, close) - wick
        minutes = np.arange(m)
        u = 1.0 + 3.0 * (np.exp(-minutes / 20) + np.exp(-(m - 1 - minutes) / 25))
        vol = (self.base_vol[:, None] / u.sum() * u[None, :] * rng.lognormal(0, 0.3, (n, m))
               * (1 + news * 4)[:, None]).round()
        return opens, high, low, close, vol

    def _build(self) -> None:
        daily_rows, opening_rows, mkt = [], [], []
        prev = self.base_price.copy()
        self._prev_close = []
        for k, date in enumerate(self.dates):
            self._prev_close.append(prev.copy())
            o, h, l, c, v = self._day_matrix(k, prev)
            for i, sym in enumerate(self.symbols):
                daily_rows.append((sym, date, o[i, 0], h[i].max(), l[i].min(), c[i, -1], v[i].sum()))
                opening_rows.append((sym, date, o[i, 0], h[i, :5].max(), l[i, :5].min(), c[i, 4], v[i, :5].sum()))
            mkt.append((c[:, -1] / prev - 1).mean())
            prev = c[:, -1].copy()
        cols = ["symbol", "date", "o", "h", "l", "c", "v"]
        self.daily = pd.DataFrame(daily_rows, columns=cols)
        self.opening = pd.DataFrame(opening_rows, columns=cols)
        f = build_features(self.daily, self.opening)
        self._features = {d: g for d, g in f.groupby("date")}
        self._mkt = dict(zip(self.dates, mkt))

    def features(self, date: dt.date) -> pd.DataFrame:
        return self._features.get(date, pd.DataFrame())

    def minute_day(self, date: dt.date, symbol: str) -> Day | None:
        k = self.dates.index(date)
        if self._cache_idx != k:
            self._cache = self._day_matrix(k, self._prev_close[k])
            self._cache_idx = k
        i = self.symbols.index(symbol)
        o, h, l, c, v = (a[i].copy() for a in self._cache)
        return Day(date=date, open=o, high=h, low=l, close=c, volume=v)

    def daily_frame(self) -> pd.DataFrame:
        return self.daily[["symbol", "date", "o", "c"]]

    def market(self, date: dt.date) -> tuple[float, float]:
        r = self._mkt[date]
        return float(r), float(r)
