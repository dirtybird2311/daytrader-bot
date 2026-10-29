"""Gap-and-go and buy-the-dip: morning lists, rules, and leak hunters."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from daytrader.backtest.dip import DipEngine
from daytrader.backtest.orb import OrbEngine
from daytrader.config import load_config, override
from daytrader.data.model import Day
from daytrader.data.orb_data import OrbData, SyntheticOrbData
from daytrader.data.setups import dip_setups, dip_table, gap_candidates
from tests.conftest import CONFIG_DIR
from tests.test_orb import PoisonedMinutes as _Poisoned


class PoisonedMinutes(_Poisoned):
    def daily_frame(self):
        return self.base.daily_frame()


@pytest.fixture(scope="module")
def market():
    return SyntheticOrbData(n_days=260, n_symbols=60, seed=3, momentum=0.12)


def gcfg():
    return load_config(CONFIG_DIR / "gapgo.yaml")


def dcfg(**kw):
    c = override(load_config(CONFIG_DIR / "dip.yaml"), {"params.zone_atr": 1.0, "params.stop_atr": 0.5})
    return override(c, {f"params.{k}": v for k, v in kw.items()}) if kw else c


# ---- morning lists ---------------------------------------------------------------------------
def test_gap_candidates_rules():
    d = dt.date(2021, 1, 4)
    f = pd.DataFrame({
        "symbol": list("ABCDE"), "date": d, "prev_close": [100, 100, 100, 100, 5.0],
        "open5": [103, 101, 115, 104, 5.3], "avg_volume": [5e6, 5e6, 5e6, 1e3, 9e9],
        "rel_volume": [2.0, 9.0, 9.0, 3.0, 9.0],
    })
    c = gap_candidates(f, min_gap=0.02, max_gap=0.10, large_cap_rank=3, min_price=10)
    # B gaps 1% (too small), C 15% (too big), D is not a top-3 dollar-volume name, E is under $10
    assert list(c["symbol"]) == ["A"] and list(c["rank"]) == [1]


def _daily_rows(closes, highs, lows, opens=None, sym="X", start="2021-01-04"):
    days = pd.bdate_range(start, periods=len(closes)).date
    return pd.DataFrame({"symbol": sym, "date": days, "o": opens if opens is not None else closes,
                         "h": highs, "l": lows, "c": closes, "v": 2e6})


def test_dip_setup_definition():
    flat_c = [10.0] * 22
    c = flat_c + [10.3, 10.8, 11.6]          # three up days...
    h = [10.5] * 22 + [10.6, 11.2, 12.2]     # ...with wider ranges each day (0.5, 0.6, 0.8, 1.3)
    l = [10.0] * 22 + [10.0, 10.5, 10.9]
    c, h, l = c + [11.0], h + [11.5], l + [10.8]  # the day being traded
    t = dip_table(_daily_rows(c, h, l))
    s = dip_setups(t, 3)
    assert len(s) == 1 and s.iloc[0]["date"] == t["date"].iloc[-1]
    assert s.iloc[0]["resistance"] == pytest.approx(10.5)
    assert dip_setups(t, 4).empty  # only a 3-day run
    # the same run without expanding ranges is not "parabolic"
    h2 = h.copy(); h2[24] = 11.9; l2 = l.copy(); l2[24] = 11.4
    assert dip_setups(dip_table(_daily_rows(c, h2, l2)), 3).empty


def test_dip_table_ignores_today_and_future(market):
    d = market.daily
    cut = market.dates[150]
    clean = dip_table(d)
    dirty = d.copy()
    rng = np.random.default_rng(2)
    later = dirty["date"] >= cut  # today's whole daily bar, and every later one
    for col in ("o", "h", "l", "c", "v"):
        dirty.loc[later, col] *= rng.uniform(0.5, 2.0, later.sum())
    a = clean[clean["date"] <= cut].reset_index(drop=True)
    b = dip_table(dirty)[lambda x: x["date"] <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_reverse_split_is_not_a_breakout():
    c = [2.0] * 22 + [2.1, 2.2, 23.5]  # a 1-for-10 reverse split on the last run day
    h = [2.05] * 22 + [2.2, 2.35, 24.5]
    l = [1.95] * 22 + [2.0, 2.1, 22.0]
    o = list(c); o[24] = 23.0
    t = dip_table(_daily_rows(c + [23.6], h + [24.0], l + [23.0], opens=o + [23.5]))
    assert dip_setups(t, 3).empty


# ---- dip engine rules ------------------------------------------------------------------------
class OneDip(OrbData):
    """One stock, one setup day: daily history + that day's minutes."""

    def __init__(self, daily, day, feat):
        self.daily, self.day, self.feat = daily, day, feat
        self.dates = [day.date]

    def daily_frame(self):
        return self.daily

    def features(self, date):
        return pd.DataFrame([self.feat])

    def minute_day(self, date, symbol):
        return self.day

    def market(self, date):
        return 0.0, 0.0


def _setup_daily():
    c = [10.0] * 22 + [10.3, 10.8, 11.6]
    h = [10.5] * 22 + [10.6, 11.2, 12.2]
    l = [10.0] * 22 + [10.0, 10.5, 10.9]
    return _daily_rows(c + [11.0], h + [11.5], l + [10.8])


def _minutes(path5):
    """1-minute bars from a list of 5-minute (open, high, low, close) tuples."""
    o, h, l, c = [], [], [], []
    for (bo, bh, bl, bc) in path5:
        for k in range(5):
            o.append(bo if k == 0 else c[-1]); c.append(bc if k == 4 else (bo + bc) / 2)
            h.append(bh if k == 2 else max(o[-1], c[-1])); l.append(bl if k == 1 else min(o[-1], c[-1]))
    n = len(o)
    date = _setup_daily()["date"].iloc[-1]
    return Day(date=date, open=np.array(o), high=np.array(h), low=np.array(l), close=np.array(c),
               volume=np.full(n, 1000.0))


FEAT = {"symbol": "X", "open5": 11.4, "avg_volume": 2e6, "atr": 1.0}


def _dip(path5, **kw):
    cfg = override(load_config(CONFIG_DIR / "dip.yaml"), {"costs.slippage_per_share": 0.0,
                                                          "costs.commission_per_share": 0.0,
                                                          **{f"params.{k}": v for k, v in kw.items()}})
    daily = _setup_daily()
    feat = {**FEAT, "date": daily["date"].iloc[-1]}
    return DipEngine(cfg).run(OneDip(daily, _minutes(path5), feat)).trades


def test_dip_entry_target_half_and_breakeven():
    # resistance 10.5, ATR 1 -> zone up to 10.75, stop 10.35; risk from a 10.9 entry = 0.55
    path = [(11.4, 11.5, 11.0, 11.1), (11.1, 11.15, 10.7, 10.8),   # pullback into the zone
            (10.8, 11.2, 10.75, 11.18),                             # higher low, close above prior high
            (10.9, 11.3, 10.85, 11.2),                              # entry at this bar's open, 10.9
            (11.2, 12.1, 11.1, 12.0),                               # 2R target 12.0 hit: half sold
            (12.0, 12.0, 10.8, 10.85)] + [(10.85, 10.9, 10.8, 10.85)] * 70  # falls back to breakeven
    t = _dip(path).iloc[0]
    assert t["entry_bar"] == 15 and t["entry_px"] == pytest.approx(10.9)
    assert t["stop_at_entry"] == pytest.approx(10.35)
    assert t["exit_reason"] == "breakeven"
    assert t["exit_px"] == pytest.approx((12.0 + 10.9) / 2, abs=0.06)  # half at 2R, half at entry


def test_dip_void_if_stop_level_breaks_first():
    path = [(11.4, 11.5, 11.0, 11.1), (11.1, 11.15, 10.2, 10.8),   # straight through the 10.35 stop level
            (10.8, 11.2, 10.75, 11.18), (10.9, 11.3, 10.85, 11.2)] + [(11.2, 11.3, 11.1, 11.2)] * 70
    assert _dip(path).empty


def test_dip_no_entry_after_last_entry_time():
    path = [(11.4, 11.5, 11.2, 11.3)] * 30 + [(11.3, 11.3, 10.7, 10.8), (10.8, 11.4, 10.75, 11.35)] \
        + [(11.2, 11.3, 11.1, 11.2)] * 40  # the reversal completes at 12:10, after 11:30
    assert _dip(path).empty
    assert len(_dip(path, last_entry="12:30")) == 1


# ---- leak hunters on the full engines --------------------------------------------------------
ENGINES = {"gapgo": lambda: OrbEngine(gcfg()), "dip": lambda: DipEngine(dcfg())}


@pytest.mark.parametrize("name", sorted(ENGINES))
@pytest.mark.parametrize("day_idx,bar", [(120, 7), (200, 60)])
def test_poisoned_minutes(market, name, day_idx, bar):
    cut = market.dates[day_idx]
    clean = ENGINES[name]().run(market)
    dirty = ENGINES[name]().run(PoisonedMinutes(market, cut, bar))

    def before(res):
        return [(d.date, d.bar, d.target, d.reason) for d in res.decisions
                if d.date < cut or (d.date == cut and d.bar < bar)]

    assert before(clean) == before(dirty)
    assert len([k for k in before(clean) if not k[3].startswith("setup")]) > 0


@pytest.mark.parametrize("name", sorted(ENGINES))
def test_truncation(market, name):
    full = ENGINES[name]().run(market)
    cut = market.dates[180]
    short = ENGINES[name]().run(market, end=market.dates[179])
    a = full.trades[full.trades["date"] < pd.Timestamp(cut)].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, short.trades.reset_index(drop=True))
    assert len(a) > 0


def test_gapgo_long_only_and_capped(market):
    t = OrbEngine(gcfg()).run(market).trades
    assert len(t) > 20 and (t["side"] > 0).all()
    assert t.groupby("date").size().max() <= 5
    assert (t["entry_bar"] < 60).all()  # orders cancelled at 10:30
    res = OrbEngine(gcfg()).run(market)
    eq = res.daily["equity_start"].rename("eq").reset_index()
    m = res.trades.merge(eq, on="date")
    level = m["entry_px"] - 0.01  # slippage on the entry fill
    assert (level * m["shares"] <= m["eq"] / 5 + 1e-6).all()  # 1x of a slot, sized at the order level
