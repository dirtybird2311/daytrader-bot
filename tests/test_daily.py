"""Batch D: decide at 3:50 pm, fill at the close or next open."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from daytrader.backtest.daily import SNAP_BAR, DailyEngine, SpyDaily, UniverseDaily, rsi_wilder, split_ratio
from daytrader.config import load_config, override
from daytrader.data.orb_data import SyntheticOrbData
from daytrader.data.synthetic import make_days
from tests.conftest import CONFIG_DIR
from tests.test_lookahead import poison


def cfg(name, **kw):
    c = load_config(CONFIG_DIR / f"{name}.yaml")
    return override(c, {f"params.{k}": v for k, v in kw.items()}) if kw else c


def test_split_ratio():
    assert split_ratio(50.0, 100.0) == pytest.approx(0.5)
    assert split_ratio(101.0, 10.0) == pytest.approx(10)
    assert split_ratio(95.0, 100.0) is None      # a normal gap
    assert split_ratio(41.0, 100.0) is None      # a 59% crash isn't near a split ratio... 
    assert split_ratio(30.0, 100.0) == pytest.approx(1 / 3)


def test_rsi_wilder():
    assert rsi_wilder(np.array([1.0, 2, 3, 4])) == 100.0
    x = np.array([10, 11, 10.5, 10.2, 10.8, 10.4])
    d = np.diff(x); up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au, ad = up[:2].mean(), dn[:2].mean()
    for u, w in zip(up[2:], dn[2:]):
        au, ad = (au + u) / 2, (ad + w) / 2
    assert rsi_wilder(x) == pytest.approx(100 - 100 / (1 + au / ad))


def test_overnight_fills_and_pnl():
    days = make_days(60, seed=2)
    c = cfg("spy_overnight")
    res = DailyEngine(c).run(SpyDaily(days))
    t = res.trades.iloc[0]
    d0, d1 = days[0], days[1]
    shares = int(25000 / d0.close[SNAP_BAR])
    assert t["shares"] == shares
    assert t["entry_px"] == pytest.approx(d0.close[-1] + 0.01) and t["exit_px"] == pytest.approx(d1.open[0] - 0.01)
    assert t["pnl"] == pytest.approx((d1.open[0] - d0.close[-1] - 0.02) * shares - 2 * 0.0035 * shares)
    # equity is marked to market at each close: the day after entry carries the overnight move
    eq = res.daily["equity"]
    assert eq.iloc[0] == pytest.approx(25000 - (0.01 + 0.0035) * shares)
    assert len(res.trades) == len(days) - 1


class _Full:
    """make_days gives full 390-minute sessions."""


@pytest.mark.parametrize("name,kw", [("spy_overnight", {}), ("spy_rsi2", {"sma_trend": 20, "rsi_entry": 30})])
def test_spy_daily_no_lookahead(name, kw):
    days = make_days(160, seed=5, momentum=0.03)
    c = cfg(name, **kw)
    clean = DailyEngine(c).run(SpyDaily(days))
    same_day = 0
    for day_idx in range(30, 150, 6):
        dirty = DailyEngine(c).run(SpyDaily(poison(days, day_idx, SNAP_BAR + 1)))
        cut = days[day_idx].date
        before = lambda r: [k for k in r.decision_keys() if k[0] < cut or (k[0] == cut and k[1] <= SNAP_BAR)]
        assert before(clean) == before(dirty), f"decision changed at {cut}"
        same_day += sum(1 for k in before(clean) if k[0] == cut)
    assert same_day >= 3  # cuts landed on days with a 3:50 pm decision


def test_rsi2_rules():
    days = make_days(300, seed=7, momentum=0.02)
    c = cfg("spy_rsi2", sma_trend=50, rsi_entry=20)
    res = DailyEngine(c).run(SpyDaily(days))
    assert len(res.trades) > 5
    assert (res.trades["exit_reason"] == "close").all()
    # sized at 1x on the 3:50 pm price; the close can drift a little either way
    assert res.daily["leverage"].max() <= 1.02


# ---- reversal on a made-up universe ----------------------------------------------------------
@pytest.fixture(scope="module")
def market():
    return SyntheticOrbData(n_days=120, n_symbols=40, seed=3, momentum=0.0)


class Poisoned:
    """Scramble daily closes from the cut day on, and every later day's open and 3:50 pm price."""

    def __init__(self, base, cut):
        self.base, self.dates = base, base.dates
        rng = np.random.default_rng(4)
        d = base.daily.copy()
        later, today = d["date"] > cut, d["date"] == cut
        d.loc[later, ["o", "c"]] *= rng.uniform(0.7, 1.3, (later.sum(), 2))
        d.loc[today, "c"] *= rng.uniform(0.7, 1.3, today.sum())
        self._daily = d
        s = base.snapshots()
        lat = s["date"] > cut
        s.loc[lat, "snap"] *= rng.uniform(0.7, 1.3, lat.sum())
        self._snaps = s

    def daily_frame(self):
        return self._daily

    def features(self, date):
        return self.base.features(date)

    def market(self, date):
        return self.base.market(date)


@pytest.mark.parametrize("exit_rule", ["next_open", "next_close"])
def test_reversal_no_lookahead_and_picks_losers(market, exit_rule):
    c = cfg("reversal", large_cap_rank=30, exit=exit_rule, min_price=1.0)
    clean = DailyEngine(c).run(UniverseDaily(market, market.snapshots(), 30, 1.0))
    cut = market.dates[70]
    p = Poisoned(market, cut)
    dirty = DailyEngine(c).run(UniverseDaily(p, p._snaps, 30, 1.0))
    before = lambda r: [k for k in r.decision_keys() if k[0] < cut or (k[0] == cut and k[1] <= SNAP_BAR)]
    assert before(clean) == before(dirty) and len(before(clean)) > 50
    t = clean.trades
    assert len(t) > 100 and t.groupby("entry_date").size().max() <= 10
    if exit_rule == "next_open":
        assert (t["exit_reason"] == "open").all() and (t["days_held"] >= 1).all()
    else:
        assert (t["exit_reason"] == "close").all()
    if exit_rule != "next_open":
        return  # with next-close exits, names still held can't be re-bought; checked below for next_open
    # the picks really are the day's biggest decliners at 3:50 pm
    snaps = market.snapshots()
    d = market.dates[50]
    ud = UniverseDaily(market, snaps, 30, 1.0)
    cand = ud.candidates(d)
    cand = cand.assign(move=cand["snap"] / cand["prev_close"] - 1).sort_values("move")
    picked = set(t.loc[pd.to_datetime(t["entry_date"]) == pd.Timestamp(d), "symbol"])
    expect = set(cand[cand["move"] < 0].head(10)["symbol"])
    assert picked == expect


def test_split_while_held_keeps_value():
    from daytrader.backtest.daily import DailyData

    class Two(DailyData):
        def __init__(self):
            self.dates = [dt.date(2021, 1, 4), dt.date(2021, 1, 5), dt.date(2021, 1, 6)]
            self.px = {self.dates[0]: (100, 99, 98), self.dates[1]: (49.5, 50, 50), self.dates[2]: (50, 50, 51)}

        def bar(self, date, symbol):
            return self.px[date]

        def candidates(self, date):
            if date != self.dates[0]:
                return pd.DataFrame()
            return pd.DataFrame({"symbol": ["X"], "prev_close": [101.0], "open": [100.0], "snap": [99.0], "dv_rank": [1]})

        def market(self, date):
            return 0.0, 0.0

    c = override(cfg("reversal", exit="next_close", n_picks=1), {"costs.slippage_per_share": 0.0,
                                                                  "costs.commission_per_share": 0.0})
    res = DailyEngine(c).run(Two())
    t = res.trades.iloc[0]
    # bought at 98, 2-for-1 split overnight, sold at 50 on day 2 -> 2 x 50 = 100 per original share
    assert t["exit_px"] == pytest.approx(50) and t["entry_px"] == pytest.approx(49)
    assert res.daily["equity"].iloc[1] / res.daily["equity"].iloc[0] == pytest.approx(100 / 98, rel=1e-3)
