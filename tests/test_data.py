import datetime as dt

import numpy as np
import pandas as pd

from daytrader.data.model import days_from_frame
from daytrader.data.store import load_days


def alpaca_frame(date: str, minutes: int = 390, start_et: str = "09:30", skip=(), extended=True):
    """Bars shaped like Alpaca's response (UTC timestamps, bar start time)."""
    start = pd.Timestamp(f"{date} {start_et}", tz="America/New_York")
    rows = []
    if extended:  # premarket bar that must be ignored
        rows.append({"t": (start - pd.Timedelta(minutes=30)).tz_convert("UTC").isoformat(),
                     "o": 1, "h": 1, "l": 1, "c": 1, "v": 5})
    for i in range(minutes):
        if i in skip:
            continue
        px = 100 + i * 0.01
        rows.append({"t": (start + pd.Timedelta(minutes=i)).tz_convert("UTC").isoformat(),
                     "o": px, "h": px + 0.05, "l": px - 0.05, "c": px + 0.01, "v": 1000})
    return pd.DataFrame(rows)


def test_regular_session_and_timezones():
    # One winter (EST) and one summer (EDT) day: both must start at 9:30 local.
    df = pd.concat([alpaca_frame("2023-01-17"), alpaca_frame("2023-07-17")])
    days = days_from_frame(df)
    assert [d.date for d in days] == [dt.date(2023, 1, 17), dt.date(2023, 7, 17)]
    for d in days:
        assert d.n == 390
        assert d.open[0] == 100  # premarket bar excluded


def test_missing_minutes_filled_flat_with_zero_volume():
    d = days_from_frame(alpaca_frame("2023-03-01", skip={100, 101}))[0]
    assert d.n == 390
    assert d.volume[100] == 0 and d.close[100] == d.close[99] and d.open[101] == d.close[99]


def test_half_day_keeps_short_session():
    d = days_from_frame(alpaca_frame("2023-11-24", minutes=210))[0]  # closes 13:00
    assert d.n == 210


def test_store_round_trip(tmp_path):
    folder = tmp_path / "bars" / "SPY"
    folder.mkdir(parents=True)
    alpaca_frame("2023-03-01").to_parquet(folder / "2023-03.parquet", index=False)
    alpaca_frame("2023-04-03").to_parquet(folder / "2023-04.parquet", index=False)
    days = load_days("SPY", dt.date(2023, 3, 1), dt.date(2023, 3, 31), tmp_path)
    assert [d.date for d in days] == [dt.date(2023, 3, 1)]
    assert np.isclose(days[0].session_close, days[0].close[-1])


def test_summary_markdown():
    from daytrader.notify import summary_markdown
    md = summary_markdown({"period": "tune", "passed": False, "sharpe": 0.8, "annual_return": 0.05,
                           "max_drawdown": -0.12, "trades": 1234, "checks": {"Sharpe": False, "Plateau": None}})
    assert "FAIL" in md and "Not run" in md


def test_asset_cleaning_and_tolerant_fetch():
    import requests
    from daytrader.data.universe import clean_assets, fetch_tolerant
    a = pd.DataFrame({"symbol": ["AAPL", "BRK.B", "003CVR016", "26885b100", "SPY", "OTCX"],
                      "exchange": ["NASDAQ", "NYSE", "NASDAQ", "NYSE", "ARCA", "OTC"]})
    assert clean_assets(a)["symbol"].tolist() == ["AAPL", "BRK.B", "SPY"]

    class Fake:
        calls = 0

        def fetch_multi(self, syms, *a):
            self.calls += 1
            if "BAD" in syms:
                raise requests.HTTPError("400 Client Error: Bad Request")
            return pd.DataFrame({"symbol": syms, "t": "x", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1})

    out = fetch_tolerant(Fake(), ["A", "B", "BAD", "C"], None, None, "1Day", log=lambda *_: None)
    assert sorted(out["symbol"]) == ["A", "B", "C"]
