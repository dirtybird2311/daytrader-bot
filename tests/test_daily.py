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
        # the day's full high and low aren't known at 3:50 pm either
        d.loc[later | today, "h"] *= rng.uniform(1.0, 1.3, (later | today).sum())
        d.loc[later | today, "l"] *= rng.uniform(0.7, 1.0, (later | today).sum())
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


def test_fund_names_are_excluded_from_reversal_universe(market):
    from daytrader.data.universe import is_fund
    funds = ["ProShares Ultra VIX Short-Term Futures ETF", "Direxion Daily Semiconductor Bull 3X Shares",
             "iShares Silver Trust", "VelocityShares Daily Inverse VIX Short-Term ETN", "SPDR S&P 500 ETF Trust",
             "Invesco QQQ Trust, Series 1", "United States Oil Fund"]
    stocks = ["Apple Inc. Common Stock", "Invesco Ltd", "The Charles Schwab Corporation Common Stock",
              "Ultra Clean Holdings, Inc. Common Stock", "Bull Run Corp", "WisdomTree, Inc. Common Stock"]
    assert all(is_fund(n) for n in funds) and not any(is_fund(n) for n in stocks)
    d = market.dates[50]
    full = UniverseDaily(market, market.snapshots(), 30, 1.0).candidates(d)
    drop = set(full["symbol"][:5])
    cut = UniverseDaily(market, market.snapshots(), 30, 1.0, exclude=drop).candidates(d)
    assert drop and not (set(cut["symbol"]) & drop) and 0 < len(cut) <= 30


RULES = ["ibs", "turnaround_tue", "turn_of_month", "down3_overnight", "five_day_low", "triple_rsi", "rsi2_high", "any"]


@pytest.mark.parametrize("rule", RULES)
def test_spy_rules_no_lookahead(rule):
    days = make_days(900 if rule == "triple_rsi" else 320, seed=11, momentum=0.0)
    c = cfg("spy_rules", rule=rule, rules=["ibs", "turn_of_month", "down3_overnight", "turnaround_tue"])
    clean = DailyEngine(c).run(SpyDaily(days))
    assert len(clean.trades) >= 3, rule
    for day_idx in range(60, len(days) - 10, 7 if len(days) < 400 else 19):
        dirty = DailyEngine(c).run(SpyDaily(poison(days, day_idx, SNAP_BAR + 1)))
        cut = days[day_idx].date
        before = lambda r: [k for k in r.decision_keys() if k[0] < cut or (k[0] == cut and k[1] <= SNAP_BAR)]
        assert before(clean) == before(dirty), f"{rule}: decision changed at {cut}"


def test_spy_rules_specifics():
    days = make_days(320, seed=11, momentum=0.0)
    t = DailyEngine(cfg("spy_rules", rule="turn_of_month")).run(SpyDaily(days)).trades
    ent = pd.to_datetime(t["entry_date"])
    for e, x in zip(ent, pd.to_datetime(t["date"])):
        same = [d for d in (dd.date for dd in days) if (d.year, d.month) == (e.year, e.month)]
        assert same.index(e.date()) == len(same) - 5                 # the 5th-last trading day
        nxt = [d for d in (dd.date for dd in days) if (d.year, d.month) == (x.year, x.month)]
        assert nxt.index(x.date()) == 2                               # the 3rd trading day
    t = DailyEngine(cfg("spy_rules", rule="turnaround_tue")).run(SpyDaily(days)).trades
    assert (pd.to_datetime(t["entry_date"]).dt.weekday == 0).all()
    t = DailyEngine(cfg("spy_rules", rule="down3_overnight")).run(SpyDaily(days)).trades
    assert (t["exit_reason"] == "open").all() and (t["days_held"] >= 1).all()


def test_combined_account_is_in_whenever_any_rule_is():
    days = make_days(400, seed=13, momentum=0.0)
    parts = ["ibs", "turn_of_month", "down3_overnight"]
    held = lambda r: r.daily["leverage"] > 0.5
    single = [held(DailyEngine(cfg("spy_rules", rule=x)).run(SpyDaily(days))) for x in parts]
    combo = DailyEngine(cfg("spy_rules", rule="any", rules=parts)).run(SpyDaily(days))
    either = single[0] | single[1] | single[2]
    # same days in the market as the union of the three (a day or two can differ when one rule's
    # exit and another's entry fall on the same close)
    assert (held(combo) == either).mean() > 0.97
    assert combo.daily["leverage"].max() <= 1.02


STOCK_RULES = ["down3_overnight", "five_day_low", "rsi2_high", "turnaround_tue"]


@pytest.mark.parametrize("sma,rules", [(0, STOCK_RULES), (50, STOCK_RULES)] + [(0, [r]) for r in STOCK_RULES])
def test_stock_rules_no_lookahead_and_limits(sma, rules):
    m = SyntheticOrbData(n_days=320, n_symbols=40, seed=5, momentum=0.0)
    c = cfg("stock_rules", large_cap_rank=30, min_price=1.0, sma_trend=sma, max_positions=5, rules=rules)
    clean = DailyEngine(c).run(UniverseDaily(m, m.snapshots(), 30, 1.0))
    t = clean.trades
    assert len(t) > 20
    assert clean.daily["leverage"].max() <= 1.05
    for cut_idx in range(215, 315, 6):
        cut = m.dates[cut_idx]
        p = Poisoned(m, cut)
        dirty = DailyEngine(c).run(UniverseDaily(p, p._snaps, 30, 1.0))
        before = lambda r: [k for k in r.decision_keys() if k[0] < cut or (k[0] == cut and k[1] <= SNAP_BAR)]
        assert before(clean) == before(dirty) and len(before(clean)) > 20, cut


def test_stock_rules_position_cap():
    m = SyntheticOrbData(n_days=320, n_symbols=40, seed=5, momentum=0.0)
    c = cfg("stock_rules", large_cap_rank=30, min_price=1.0, sma_trend=0, max_positions=3,
            rules=["down3_overnight", "five_day_low", "rsi2_high"])
    res = DailyEngine(c).run(UniverseDaily(m, m.snapshots(), 30, 1.0))
    t = res.trades
    for d in m.dates[20:300:5]:
        ent, ex = pd.to_datetime(t["entry_date"]), pd.to_datetime(t["date"])
        open_n = ((ent <= pd.Timestamp(d)) & (ex > pd.Timestamp(d))).sum()
        assert open_n <= 3


def _pool_data(days_by_fund):
    from daytrader.backtest.daily import MultiDaily
    return MultiDaily({f: SpyDaily(d, f) for f, d in days_by_fund.items()}, "SPY")


def test_etf_pool_no_lookahead_and_full_use_of_cash():
    funds = {f: make_days(320, seed=20 + k, momentum=0.0) for k, f in enumerate(["SPY", "QQQ", "IWM", "DIA"])}
    c = cfg("etf_pool")
    clean = DailyEngine(c).run(_pool_data(funds))
    assert len(clean.trades) > 30
    lev = clean.daily["leverage"]
    assert lev.max() <= 1.02
    # whenever anything is held, (nearly) the whole account is invested: no idle slices
    assert (lev[lev > 0.05] > 0.9).mean() > 0.9
    for day_idx in range(40, 310, 9):
        dirty = {f: (poison(d, day_idx, SNAP_BAR + 1) if f == "QQQ" else d) for f, d in funds.items()}
        res = DailyEngine(c).run(_pool_data(dirty))
        cut = funds["SPY"][day_idx].date
        before = lambda r: [k for k in r.decision_keys() if k[0] < cut or (k[0] == cut and k[1] <= SNAP_BAR)]
        assert before(clean) == before(res), f"decision changed at {cut}"


@pytest.mark.parametrize("msma", [0, 50])
def test_stock_momentum_no_lookahead_and_monthly(msma):
    m = SyntheticOrbData(n_days=420, n_symbols=40, seed=8, momentum=0.0)
    c = cfg("stock_momentum", large_cap_rank=30, min_price=1.0, n_hold=5, lookback=120, skip=10, market_sma=msma,
            market_symbol="S000")
    clean = DailyEngine(c).run(UniverseDaily(m, m.snapshots(), 30, 1.0))
    t = clean.trades
    assert len(t) > 10 and clean.daily["leverage"].max() <= 1.05
    # trades happen only on a month's last trading day
    ds = pd.Series(pd.to_datetime(m.dates))
    last = set(ds.groupby(ds.dt.year * 12 + ds.dt.month).max().dt.date)
    assert set(pd.to_datetime(t["date"]).dt.date) <= last
    for cut_idx in range(150, 410, 13):
        cut = m.dates[cut_idx]
        p = Poisoned(m, cut)
        dirty = DailyEngine(c).run(UniverseDaily(p, p._snaps, 30, 1.0))
        before = lambda r: [k for k in r.decision_keys() if k[0] < cut or (k[0] == cut and k[1] <= SNAP_BAR)]
        assert before(clean) == before(dirty), cut


def test_stock_rules_market_filter_only_removes_entries():
    m = SyntheticOrbData(n_days=320, n_symbols=40, seed=5, momentum=0.0)
    base = dict(large_cap_rank=30, min_price=1.0, sma_trend=0, max_positions=5, rules=["down3_overnight", "rsi2_high"])
    a = DailyEngine(cfg("stock_rules", **base)).run(UniverseDaily(m, m.snapshots(), 30, 1.0)).trades
    b = DailyEngine(cfg("stock_rules", market_sma=50, market_symbol="S000", **base)).run(UniverseDaily(m, m.snapshots(), 30, 1.0)).trades
    assert 0 < len(b) < len(a)
