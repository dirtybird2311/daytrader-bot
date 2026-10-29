"""Executing on a second symbol, and combining sleeves into one account."""

import numpy as np
import pandas as pd

from daytrader.backtest.engine import Engine
from daytrader.config import load_config, override
from daytrader.data.synthetic import leveraged_days, make_days
from daytrader.research import combine
from daytrader.strategies import build_strategy
from tests.conftest import CONFIG_DIR
from tests.test_lookahead import poison


def base():
    return override(load_config(CONFIG_DIR / "spy_noise.yaml"), {"params.allow_short": False})


def run(cfg, days, ex=None):
    return Engine(cfg, build_strategy(cfg)).run(days, exec_days=ex)


def test_execution_on_identical_symbol_matches_plain_run():
    days = make_days(150, seed=5, momentum=0.03)
    plain = run(base(), days)
    cfg = override(base(), {"execution": {"symbol": "SAME", "multiplier": 1, "max_leverage": 100}})
    same = run(cfg, days, {d.date: d for d in days})
    cols = ["date", "side", "entry_px", "exit_px", "exit_reason"]
    pd.testing.assert_frame_equal(plain.trades[cols], same.trades[cols])
    # execution mode converts notional to shares at the fill price, not the 9:30 open
    by_date = {d.date: d for d in days}
    for (_, p), (_, q) in zip(plain.trades.iterrows(), same.trades.iterrows()):
        d = by_date[p["date"].date()]
        expected = int(p["shares"] * d.open[0] / d.open[int(p["entry_bar"])])
        assert abs(q["shares"] - expected) <= 1


def test_leveraged_execution_is_capped_and_priced_on_the_traded_symbol():
    days = make_days(150, seed=5, momentum=0.03)
    ex = leveraged_days(days, 3.0)
    cfg = override(base(), {"execution": {"symbol": "UPRO", "multiplier": 3, "max_leverage": 1.0}})
    res = run(cfg, days, ex)
    t = res.trades.merge(res.daily["equity_start"].reset_index(), on="date")
    assert len(t) > 20
    assert (t["entry_px"] * t["shares"] <= t["equity_start"] * 1.0 + t["entry_px"]).all()
    for _, r in t.head(20).iterrows():
        d = ex[r["date"].date()]
        assert d.low.min() - 0.01 <= r["entry_px"] <= d.high.max() + 0.01


def test_execution_symbol_has_no_lookahead():
    days = make_days(160, seed=5, momentum=0.03)
    cfg = override(base(), {"execution": {"symbol": "UPRO", "multiplier": 3, "max_leverage": 1.0}})
    clean = run(cfg, days, leveraged_days(days, 3.0))
    cut_idx, minute = 100, 60
    dirty_days = poison(days, cut_idx, minute)
    dirty = run(cfg, dirty_days, leveraged_days(dirty_days, 3.0))
    cut = days[cut_idx].date
    before = lambda r: [k for k in r.decision_keys() if k[0] < cut or (k[0] == cut and k[1] < minute)]
    assert before(clean) == before(dirty) and len(before(clean)) > 0


def test_missing_execution_days_are_sat_out():
    days = make_days(120, seed=5, momentum=0.03)
    ex = leveraged_days(days, 3.0)
    drop = {d.date for d in days[60:70]}
    ex = {k: v for k, v in ex.items() if k not in drop}
    cfg = override(base(), {"execution": {"symbol": "UPRO", "multiplier": 3, "max_leverage": 1.0}})
    eng = Engine(cfg, build_strategy(cfg))
    res = eng.run(days, exec_days=ex)
    assert not set(res.trades["date"].dt.date) & drop
    assert eng.exec_skipped > 0


def test_combine_sums_sleeves():
    a_days = make_days(100, seed=5, momentum=0.03)
    b_days = make_days(100, seed=6, momentum=0.03)
    half = override(base(), {"account.starting_equity": 12500})
    ra, rb = run(half, a_days), run(half, b_days)
    c = combine([ra, rb], ["A", "B"], base())
    np.testing.assert_allclose(c.daily["equity"], ra.daily["equity"] + rb.daily["equity"])
    expect = (ra.daily["equity"] + rb.daily["equity"]) / (ra.daily["equity_start"] + rb.daily["equity_start"]) - 1
    np.testing.assert_allclose(c.daily["ret"], expect)
    assert set(c.trades["symbol"]) == {"A", "B"} and len(c.trades) == len(ra.trades) + len(rb.trades)
