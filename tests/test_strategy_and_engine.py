import numpy as np
import pytest

from daytrader.analysis import metrics as M
from daytrader.analysis import stats as S
from daytrader.backtest.engine import Engine
from daytrader.config import override
from daytrader.data.model import clock_to_bar_index, minute_to_clock
from daytrader.data.synthetic import make_days
from daytrader.data.view import History, TodayView
from daytrader.strategies import build_strategy
from daytrader.strategies.spy_noise import SpyNoiseArea


def test_clock_mapping():
    assert minute_to_clock(0) == "09:31"
    assert clock_to_bar_index("10:00") == 29
    assert minute_to_clock(clock_to_bar_index("15:30")) == "15:30"


def test_noise_bands_match_paper_formula(cfg):
    days = make_days(40, seed=2)
    idx = 30
    strat = SpyNoiseArea(cfg)
    assert strat.on_day_start(History(days, idx), 25_000)
    today = TodayView(days[idx])
    today._advance(0)
    strat.on_bar(today, 0)
    i = 59  # 10:30
    past = days[idx - 14: idx]
    sigma = np.mean([abs(d.close[i] / d.open[0] - 1) for d in past])
    o, pc = days[idx].open[0], days[idx - 1].close[-1]
    ub, lb = strat.bands(i)
    assert ub == pytest.approx(max(o, pc) * (1 + sigma))
    assert lb == pytest.approx(min(o, pc) * (1 - sigma))


def test_entries_only_at_half_hour_checks(cfg):
    res = Engine(cfg, build_strategy(cfg)).run(make_days(150, seed=4, momentum=0.03))
    entries = [d for d in res.decisions if d.reason.startswith("entry")]
    assert entries
    assert all(d.time[-2:] in ("00", "30") and d.time >= "10:00" for d in entries)


def test_long_only_never_shorts(cfg):
    lo = override(cfg, {"params.allow_short": False})
    res = Engine(lo, build_strategy(lo)).run(make_days(150, seed=4, momentum=0.03))
    assert len(res.trades) > 0 and (res.trades["side"] > 0).all()


def test_always_flat_overnight(cfg):
    res = Engine(cfg, build_strategy(cfg)).run(make_days(120, seed=9, momentum=0.03))
    assert res.trades["exit_px"].notna().all()


def test_no_edge_without_momentum(cfg):
    """On pure random walks the strategy should not find an edge (costs make it negative)."""
    sharpes = []
    for seed in range(4):
        res = Engine(cfg, build_strategy(cfg)).run(make_days(400, seed=100 + seed, momentum=0.0))
        sharpes.append(M.sharpe(res.daily["ret"]))
    assert np.mean(sharpes) < 0.6


def test_costs_reduce_returns(cfg):
    days = make_days(200, seed=11, momentum=0.03)
    cheap = override(cfg, {"costs.commission_per_share": 0.0, "costs.slippage_per_share": 0.0})
    a = Engine(cheap, build_strategy(cheap)).run(days)
    b = Engine(cfg, build_strategy(cfg)).run(days)
    assert a.daily["equity"].iloc[-1] > b.daily["equity"].iloc[-1]


def test_metrics_basics():
    import pandas as pd
    r = pd.Series([0.01, -0.02, 0.03, 0.0, -0.01], index=pd.date_range("2020-01-01", periods=5))
    perf = M.performance(r)
    assert perf["max_drawdown"] == pytest.approx(-0.02)
    assert perf["total_return"] == pytest.approx(np.prod(1 + r) - 1)
    dd = S.deflated_sharpe(pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 500)), [0.05, 0.02, -0.01, 0.08])
    assert 0 <= dd["dsr"] <= 1 and dd["trials"] == 4
