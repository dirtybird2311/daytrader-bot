"""Download 1-minute bars from Alpaca into a local parquet cache.

* Uses the consolidated SIP feed. On Alpaca's free plan, SIP history older than
  15 minutes is available, which is all a backtest needs.
* Raw (unadjusted) prices: adjusted history quietly encodes future splits.
* Self-throttles below the free plan's 200 requests/minute. The limit is shared
  by every key on the same Alpaca login, including the portfolio project's.
* Cached by symbol and month; completed months are never fetched twice.
"""

from __future__ import annotations

import datetime as dt
import os
import time
from collections import deque
from pathlib import Path

import pandas as pd
import requests

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.calls: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self.calls and now - self.calls[0] > 60:
            self.calls.popleft()
        if len(self.calls) >= self.per_minute:
            time.sleep(60 - (now - self.calls[0]) + 0.05)
        self.calls.append(time.monotonic())


class AlpacaBars:
    def __init__(self, key_id: str | None = None, secret: str | None = None,
                 requests_per_minute: int = 120, feed: str = "sip"):
        self.key_id = key_id or os.environ.get("ALPACA_API_KEY_ID")
        self.secret = secret or os.environ.get("ALPACA_API_SECRET_KEY")
        if not self.key_id or not self.secret:
            raise RuntimeError("Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY (repo secrets in GitHub Actions).")
        self.limiter = RateLimiter(requests_per_minute)
        self.feed = feed
        self.session = requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": self.key_id, "APCA-API-SECRET-KEY": self.secret})

    def fetch(self, symbol: str, start: dt.datetime, end: dt.datetime, timeframe: str = "1Min") -> pd.DataFrame:
        params = {
            "symbols": symbol, "timeframe": timeframe, "start": start.isoformat(), "end": end.isoformat(),
            "limit": 10000, "adjustment": "raw", "feed": self.feed, "sort": "asc",
        }
        rows: list[dict] = []
        token = None
        while True:
            if token:
                params["page_token"] = token
            data = self._get(params)
            rows.extend(data.get("bars", {}).get(symbol, []) or [])
            token = data.get("next_page_token")
            if not token:
                break
        if not rows:
            return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"])
        df = pd.DataFrame(rows)
        return df[["t", "o", "h", "l", "c", "v"]]

    def _get(self, params: dict) -> dict:
        for attempt in range(6):
            self.limiter.wait()
            resp = self.session.get(BARS_URL, params=params, timeout=60)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(min(60, 2 ** attempt * 2))
                continue
            if resp.status_code == 403:
                raise RuntimeError("Alpaca refused the request (403). Check the API keys and data subscription.")
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("Alpaca kept rate-limiting or failing after 6 attempts.")


def month_starts(start: dt.date, end: dt.date) -> list[dt.date]:
    out, cur = [], dt.date(start.year, start.month, 1)
    while cur <= end:
        out.append(cur)
        cur = dt.date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
    return out


def sync_symbol(client: AlpacaBars, symbol: str, start: dt.date, end: dt.date, cache_dir: str | Path,
                log=print) -> int:
    """Fill the cache for symbol between start and end. Returns months downloaded."""
    folder = Path(cache_dir) / "bars" / symbol
    folder.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()
    fetched = 0
    for m in month_starts(start, end):
        path = folder / f"{m:%Y-%m}.parquet"
        month_end = dt.date(m.year + (m.month == 12), m.month % 12 + 1, 1)
        complete = month_end <= today
        if path.exists() and complete:
            continue
        df = client.fetch(symbol, dt.datetime(m.year, m.month, 1, tzinfo=dt.timezone.utc),
                          dt.datetime(month_end.year, month_end.month, 1, tzinfo=dt.timezone.utc))
        df.to_parquet(path, index=False)
        fetched += 1
        log(f"  {symbol} {m:%Y-%m}: {len(df):,} bars")
    return fetched
