"""Leak hunters. If any of these fail, the backtest is peeking at the future.

* Poison test: corrupt every price after a moment in time. Every decision made
  before that moment must be identical.
* Truncation test: a run that stops in June must make exactly the same decisions
  through June as a run that continues to December.
"""

import numpy as np
import pytest

from daytrader.backtest.engine import Engine
from daytrader.data.model import Day
from daytrader.data.synthetic import make_days
from daytrader.data.view import History, LookaheadError, TodayView
from daytrader.strategies import build_strategy


def run(cfg, days):
    return Engine(cfg, build_strategy(cfg)).run(days)


def poison(days, day_idx, minute):
    """Replace all prices from (day_idx, minute) onward with wild garbage."""
    rng = np.random.default_rng(99)
    out = list(days[:day_idx])
    for k, d in enumerate(days[day_idx:]):
        start = minute if k == 0 else 0
        arrs = {}
        for name in ("open", "high", "low", "close", "volume"):
            a = getattr(d, name).copy()
            noise = rng.uniform(0.5, 1.5, len(a) - start)
            a[start:] = a[start:] * noise if name != "volume" else a[start:] * noise * 10
            arrs[name] = a
        arrs["high"] = np.maximum.reduce([arrs["high"], arrs["open"], arrs["close"]])
        arrs["low"] = np.minimum.reduce([arrs["low"], arrs["open"], arrs["close"]])
        out.append(Day(date=d.date, **arrs))
    return out


def _before(res, cut_date, minute):
    return [k for k in res.decision_keys() if k[0] < cut_date or (k[0] == cut_date and k[1] < minute)]


@pytest.mark.parametrize("day_idx,minute", [(60, 0), (90, 45), (120, 200), (150, 389)])
def test_poisoned_future_does_not_change_past_decisions(cfg, day_idx, minute):
    days = make_days(200, seed=5, momentum=0.03)
    clean = run(cfg, days)
    dirty = run(cfg, poison(days, day_idx, minute))
    cut_date = days[day_idx].date
    assert _before(clean, cut_date, minute) == _before(dirty, cut_date, minute)
    assert len(_before(clean, cut_date, minute)) > 0  # the test must actually compare something


def test_poison_right_after_each_check(cfg):
    """Cut one bar after a decision bar on many days: catches even a peek of a few minutes."""
    days = make_days(140, seed=5, momentum=0.03)
    clean = run(cfg, days)
    compared = 0
    for day_idx in range(40, 140, 7):
        for minute in (30, 60, 180):  # one bar after the 10:00, 10:30 and 12:30 decisions
            dirty = run(cfg, poison(days, day_idx, minute))
            cut = days[day_idx].date
            a, b = _before(clean, cut, minute), _before(dirty, cut, minute)
            assert a == b, f"decision changed before poison at {cut} bar {minute}"
            compared += sum(1 for k in a if k[0] == cut)
    assert compared > 0  # at least some cuts landed right after a same-day decision


def test_truncated_run_matches_full_run(cfg):
    days = make_days(220, seed=6, momentum=0.03)
    full = run(cfg, days)
    short = run(cfg, days[:140])
    cut = days[140].date
    assert [k for k in full.decision_keys() if k[0] < cut] == short.decision_keys()
    ft = full.trades[full.trades["date"] < str(cut)].reset_index(drop=True)
    assert ft[["date", "side", "shares", "entry_px", "exit_px"]].equals(
        short.trades[["date", "side", "shares", "entry_px", "exit_px"]].reset_index(drop=True))


def test_fills_happen_after_the_decision_bar(cfg):
    res = run(cfg, make_days(150, seed=8, momentum=0.03))
    decided = {(d.date, d.bar + cfg["costs"]["fill_delay_bars"]) for d in res.decisions}
    for _, t in res.trades.iterrows():
        assert (t["date"].date(), t["entry_bar"]) in decided


def test_views_refuse_future_data():
    days = make_days(20, seed=1)
    hist = History(days, 10)
    assert len(hist) == 10
    with pytest.raises(LookaheadError):
        hist.day(0)  # "today" is not history yet
    assert hist.last(100)[-1].date == days[9].date
    today = TodayView(days[10])
    with pytest.raises(LookaheadError):
        _ = today.close  # nothing before the first bar completes
    today._advance(0)
    today._advance(1)
    assert len(today.close) == 2
    with pytest.raises(RuntimeError):
        today._advance(5)  # the cursor cannot jump ahead
    with pytest.raises(ValueError):
        days[0].close[0] = 1.0  # market data is read-only
