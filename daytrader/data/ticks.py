"""Trade-by-trade prices for the entry minutes that 1-minute bars can't resolve.

When a stocks-in-play entry triggers mid-minute and that minute's range also reaches the
stop, bars can't say whether the stop was hit after the entry. For just those minutes we
download every trade from Alpaca (SIP feed) and keep the round-lot prints (100+ shares)
in time order. The engine then replays the minute print by print (fills.same_bar_stop: ticks).

Cache: data/universe/ticks/<YYYY-MM>.parquet with columns date, symbol, bar, prices (list of
float32). A minute that was fetched but had no prints is stored with an empty list, so it is
never fetched again.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .universe import _et

TRADES_URL = "https://data.alpaca.markets/v2/stocks/trades"
MIN_SIZE = 100


class TickStore:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self._months: dict[str, dict[tuple, np.ndarray]] = {}

    def _month(self, key: str) -> dict[tuple, np.ndarray]:
        if key not in self._months:
            path = self.folder / f"{key}.parquet"
            m: dict[tuple, np.ndarray] = {}
            if path.exists():
                df = pd.read_parquet(path)
                for d, s, b, p in zip(df["date"], df["symbol"], df["bar"], df["prices"]):
                    m[(pd.Timestamp(d).date(), str(s), int(b))] = np.asarray(p, dtype=float)
            self._months[key] = m
        return self._months[key]

    def get(self, date: dt.date, symbol: str, bar: int) -> np.ndarray | None:
        p = self._month(f"{date:%Y-%m}").get((date, symbol, int(bar)))
        return p if p is not None and p.size else None

    def has(self, date: dt.date, symbol: str, bar: int) -> bool:
        return (date, symbol, int(bar)) in self._month(f"{date:%Y-%m}")


def fetch_minute_trades(client, symbols: list[str], date: dt.date, bar: int) -> dict[str, np.ndarray]:
    """Round-lot trade prices, in time order, for each symbol during minute `bar` of `date`."""
    start = _et(date, 9, 30) + dt.timedelta(minutes=bar)
    params = {"symbols": ",".join(symbols), "start": start.isoformat(),
              "end": (start + dt.timedelta(minutes=1)).isoformat(), "limit": 10000, "feed": client.feed,
              "sort": "asc"}
    prices: dict[str, list] = defaultdict(list)
    token = None
    while True:
        if token:
            params["page_token"] = token
        data = client._get(params, url=TRADES_URL)
        for sym, rows in (data.get("trades") or {}).items():
            for r in rows or []:
                if r.get("s", 0) >= MIN_SIZE:
                    prices[sym].append(r["p"])  # the API returns trades in time order (sort=asc)
        token = data.get("next_page_token")
        if not token:
            break
    out = {}
    for sym in symbols:
        out[sym] = np.array(prices.get(sym, []), dtype=np.float32)
    return out


def sync_ticks(client, needed: list[tuple], folder: Path, budget, log=print) -> dict:
    """Fetch every (date, symbol, bar) in `needed` that isn't cached yet. Resumable."""
    folder.mkdir(parents=True, exist_ok=True)
    store = TickStore(folder)
    todo = sorted({(d, s, int(b)) for d, s, b in needed if not store.has(d, s, b)})
    log(f"Tick minutes needed: {len(set(needed)):,}; already cached: {len(set(needed)) - len(todo):,}; "
        f"to fetch: {len(todo):,}")
    groups: dict[tuple, list[str]] = defaultdict(list)
    for d, s, b in todo:
        groups[(d, b)].append(s)
    new_rows: dict[str, list] = defaultdict(list)
    done = 0
    complete = True

    def flush():
        for key, rows in new_rows.items():
            path = folder / f"{key}.parquet"
            df = pd.DataFrame(rows, columns=["date", "symbol", "bar", "prices"])
            if path.exists():
                df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
            df.to_parquet(path, index=False)
        new_rows.clear()

    for n, ((d, b), syms) in enumerate(sorted(groups.items()), 1):
        if budget.out():
            complete = False
            break
        for i in range(0, len(syms), 100):
            got = fetch_minute_trades(client, syms[i:i + 100], d, b)
            for s in syms[i:i + 100]:
                new_rows[f"{d:%Y-%m}"].append((d, s, b, got.get(s, np.array([], np.float32)).tolist()))
                done += 1
        if n % 500 == 0:
            flush()
            log(f"  ticks: {done:,}/{len(todo):,} minutes")
    flush()
    return {"needed": len(set(needed)), "fetched": done, "remaining": len(todo) - done, "complete": complete}
