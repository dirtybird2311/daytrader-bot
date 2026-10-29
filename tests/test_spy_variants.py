"""Rule checks for the extra SPY strategies (leak hunters for them live in test_lookahead.py)."""

import numpy as np

from daytrader.backtest.engine import Engine
from daytrader.config import load_config, override
from daytrader.data.model import Day, clock_to_bar_index
from daytrader.data.synthetic import make_days
from daytrader.strategies import build_strategy
from daytrader.strategies.spy_vwap_revert import SpyVwapRevert
from daytrader.data.view import TodayView
from tests.conftest import CONFIG_DIR


def run(cfg, days):
    return Engine(cfg, build_strategy(cfg)).run(days)


def test_last_half_hour_trades_in_direction_of_morning_move():
    cfg = load_config(CONFIG_DIR / "spy_lasthalf.yaml")
    days = make_days(120, seed=3, momentum=0.03)
    res = run(cfg, days)
    assert len(res.trades) > 50
    entry = clock_to_bar_index("15:30") + 1
    by_date = {d.date: d for d in days}
    prev = {days[k].date: days[k - 1] for k in range(1, len(days))}
    for _, t in res.trades.iterrows():
        d = by_date[t["date"].date()]
        sig = d.close[clock_to_bar_index("10:00")] / prev[d.date].session_close - 1
        assert t["side"] == np.sign(sig)
        assert t["entry_bar"] == entry and t["exit_reason"] == "close"


def test_last_half_hour_long_only():
    cfg = override(load_config(CONFIG_DIR / "spy_lasthalf.yaml"), {"params.allow_short": False})
    res = run(cfg, make_days(120, seed=3))
    assert len(res.trades) > 20 and (res.trades["side"] > 0).all()


def test_vwap_bands_formula():
    n = 50
    rng = np.random.default_rng(0)
    c = 100 + np.cumsum(rng.normal(0, 0.05, n))
    o = np.r_[100, c[:-1]]
    h, l = np.maximum(o, c) + 0.02, np.minimum(o, c) - 0.02
    v = rng.integers(100, 1000, n).astype(float)
    tv = TodayView(Day(date=None, open=o, high=h, low=l, close=c, volume=v))
    for i in range(n):
        tv._advance(i)
    tp = (h + l + c) / 3
    vwap = (tp * v).sum() / v.sum()
    sd = np.sqrt((v * (tp - vwap) ** 2).sum() / v.sum())
    got = SpyVwapRevert.bands(tv)
    assert np.isclose(got[0], vwap) and np.isclose(got[1], sd)


def test_vwap_revert_entries_capped_and_in_window():
    cfg = load_config(CONFIG_DIR / "spy_vwap_revert.yaml")
    cfg = override(cfg, {"params.entry_sd": 1.0})
    res = run(cfg, make_days(80, seed=4))
    assert len(res.trades) > 20
    assert res.trades.groupby("date").size().max() <= 3
    first, last = clock_to_bar_index("10:00") + 1, clock_to_bar_index("15:30") + 1
    assert res.trades["entry_bar"].between(first, last).all()
    assert set(res.trades["exit_reason"]) <= {"target_vwap", "stop", "close"}


def test_regime_filter_only_removes_days():
    base = load_config(CONFIG_DIR / "spy_noise.yaml")
    days = make_days(300, seed=9, momentum=0.03)
    plain = run(base, days)
    filt = run(override(base, {"params.regime_pct": 40, "params.regime_min_days": 20}), days)
    traded_plain = set(plain.trades["date"])
    traded_filt = set(filt.trades["date"])
    assert traded_filt < traded_plain  # strictly fewer days, never new ones
    keep = plain.trades[plain.trades["date"].isin(traded_filt)].reset_index(drop=True)
    cols = ["date", "side", "entry_px", "exit_px"]  # share counts differ: skipped days change equity
    assert keep[cols].equals(filt.trades[cols].reset_index(drop=True))
    off = run(override(base, {"params.regime_pct": 0}), days)
    assert off.trades[cols].equals(plain.trades[cols])
