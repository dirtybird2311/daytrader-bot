"""Stocks-in-play: leak hunters, fill rules and sizing."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from daytrader.backtest.orb import OrbEngine
from daytrader.config import load_config, override
from daytrader.data.model import Day
from daytrader.data.orb_data import OrbData, SyntheticOrbData
from daytrader.data.universe import build_features, daily_features, opening_features, screen
from tests.conftest import CONFIG

ORB_CONFIG = CONFIG.parent / "orb.yaml"


@pytest.fixture(scope="module")
def ocfg():
    return load_config(ORB_CONFIG)


@pytest.fixture(scope="module")
def market():
    return SyntheticOrbData(n_days=120, n_symbols=40, seed=3, momentum=0.12)


# ---- features: only what was known at 9:35 -------------------------------------------------
def _poison_rows(df, cut, include_today_cols=None, seed=0):
    """Scramble every row after `cut`; optionally also scramble some columns on `cut` itself."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    future = out["date"] > cut
    for col in ("o", "h", "l", "c", "v"):
        out.loc[future, col] = out.loc[future, col] * rng.uniform(0.3, 3.0, future.sum())
    if include_today_cols:
        today = out["date"] == cut
        for col in include_today_cols:
            out.loc[today, col] = out.loc[today, col] * rng.uniform(0.3, 3.0, today.sum())
    return out


def test_today_daily_bar_never_used(market):
    cut = market.dates[60]
    clean = build_features(market.daily, market.opening)
    # Today's full-day high, low, close and volume are unknown at 9:35. So is everything after today.
    dirty = build_features(_poison_rows(market.daily, cut, ["h", "l", "c", "v"]), market.opening)
    a = clean[clean["date"] <= cut].reset_index(drop=True)
    b = dirty[dirty["date"] <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_future_opening_bars_never_used(market):
    cut = market.dates[70]
    clean = build_features(market.daily, market.opening)
    dirty = build_features(market.daily, _poison_rows(market.opening, cut))
    pd.testing.assert_frame_equal(clean[clean["date"] <= cut].reset_index(drop=True),
                                  dirty[dirty["date"] <= cut].reset_index(drop=True))


def test_relative_volume_definition():
    days = [dt.date(2020, 1, d) for d in range(1, 21)]
    opening = pd.DataFrame({"symbol": "X", "date": days, "o": 10.0, "h": 10.5, "l": 9.5, "c": 10.2,
                            "v": [100.0] * 19 + [300.0]})
    f = opening_features(opening)
    assert f["rel_volume"].iloc[-1] == pytest.approx(3.0)  # today vs the mean of the prior 14
    assert np.isnan(f["rel_volume"].iloc[0])


def test_atr_ignores_split_gaps():
    days = [dt.date(2020, 2, d) for d in range(1, 21)]
    px = np.array([100.0] * 10 + [50.0] * 10)  # 2-for-1 split on day 11
    daily = pd.DataFrame({"symbol": "X", "date": days, "o": px, "h": px + 1, "l": px - 1, "c": px, "v": 1e6})
    f = daily_features(daily)
    assert f["atr"].dropna().max() == pytest.approx(2.0)  # the $50 "gap" never enters the ATR


# ---- engine: decisions can't depend on later minutes ----------------------------------------
class PoisonedMinutes(OrbData):
    def __init__(self, base, cut_date, cut_bar):
        self.base, self.cut_date, self.cut_bar = base, cut_date, cut_bar
        self.dates = base.dates

    def features(self, date):
        return self.base.features(date)

    def market(self, date):
        return self.base.market(date)

    def minute_day(self, date, symbol):
        d = self.base.minute_day(date, symbol)
        if d is None or date < self.cut_date:
            return d
        start = 0 if date > self.cut_date else self.cut_bar
        rng = np.random.default_rng(hash((date, symbol)) % 2**32)
        arrs = {k: getattr(d, k).copy() for k in ("open", "high", "low", "close", "volume")}
        for k in arrs:
            arrs[k][start:] = arrs[k][start:] * rng.uniform(0.5, 1.5, d.n - start)
        arrs["high"] = np.maximum.reduce([arrs["high"], arrs["open"], arrs["close"]])
        arrs["low"] = np.minimum.reduce([arrs["low"], arrs["open"], arrs["close"]])
        return Day(date=date, **arrs)


@pytest.mark.parametrize("day_idx,bar", [(40, 6), (60, 30), (80, 200)])
def test_poisoned_minutes_do_not_change_earlier_decisions(ocfg, market, day_idx, bar):
    cut = market.dates[day_idx]
    clean = OrbEngine(ocfg).run(market)
    dirty = OrbEngine(ocfg).run(PoisonedMinutes(market, cut, bar))

    def before(res):
        return [(d.date, d.bar, d.target, d.reason) for d in res.decisions
                if d.date < cut or (d.date == cut and d.bar < bar)]

    assert before(clean) == before(dirty)
    assert len(before(clean)) > 0
    # Which orders go in on the cut day depends only on 9:35 information.
    orders = lambda r: [(d.reason, d.price) for d in r.decisions if d.date == cut and d.reason.startswith("order")]
    assert orders(clean) == orders(dirty)


def test_truncated_run_matches(ocfg, market):
    full = OrbEngine(ocfg).run(market)
    cut = market.dates[70]
    short = OrbEngine(ocfg).run(market, end=market.dates[69])
    a = full.trades[full.trades["date"] < pd.Timestamp(cut)].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, short.trades.reset_index(drop=True))


# ---- fill rules and sizing ------------------------------------------------------------------
class OneStock(OrbData):
    def __init__(self, day: Day, feat: dict):
        self.dates, self.day, self.feat = [day.date], day, feat

    def features(self, date):
        return pd.DataFrame([self.feat])

    def minute_day(self, date, symbol):
        return self.day

    def market(self, date):
        return 0.0, 0.0


def _day(opens, highs, lows, closes):
    n = len(opens)
    return Day(date=dt.date(2021, 3, 1), open=np.array(opens, float), high=np.array(highs, float),
               low=np.array(lows, float), close=np.array(closes, float), volume=np.full(n, 1000.0))


FEAT = {"symbol": "X", "date": dt.date(2021, 3, 1), "open5": 10.0, "high5": 10.5, "low5": 9.8, "close5": 10.4,
        "volume5": 5e5, "rel_volume": 3.0, "atr": 2.0, "avg_volume": 2e6, "prev_close": 9.9}


def _cfg(ocfg, **kw):
    return override(ocfg, {"costs.slippage_per_share": 0.0, "costs.commission_per_share": 0.0,
                           **{f"params.{k}": v for k, v in kw.items()}})


def test_gap_through_entry_fills_at_open_and_stop_same_bar_assumed_hit(ocfg):
    # bars 0-4 opening range; bar 5 gaps above the 10.5 level to 10.8; bar 5's low touches the stop
    o = [10, 10.1, 10.2, 10.3, 10.35, 10.8, 10.9, 11, 11]
    h = [10.2, 10.3, 10.4, 10.5, 10.45, 10.9, 11, 11.1, 11.1]
    l = [9.8, 10.0, 10.1, 10.2, 10.3, 10.55, 10.85, 10.9, 10.9]
    c = [10.1, 10.2, 10.3, 10.4, 10.4, 10.85, 10.95, 11, 11]
    res = OrbEngine(_cfg(ocfg)).run(OneStock(_day(o, h, l, c), FEAT))
    t = res.trades.iloc[0]
    assert t["entry_px"] == pytest.approx(10.8)            # worse of level (10.5) and the gap open
    assert t["stop_at_entry"] == pytest.approx(10.8 - 0.2)  # 10% of ATR 2.0
    assert t["exit_reason"] == "stop" and t["exit_px"] == pytest.approx(10.6)
    assert t["exit_bar"] == t["entry_bar"] == 5


def test_untouched_trade_exits_at_close_and_sizing(ocfg):
    o = [10, 10.1, 10.2, 10.3, 10.35, 10.45, 10.6, 10.7, 10.8]
    h = [10.2, 10.3, 10.4, 10.5, 10.45, 10.55, 10.7, 10.8, 10.9]
    l = [9.8, 10.0, 10.1, 10.2, 10.3, 10.44, 10.55, 10.65, 10.75]
    c = [10.1, 10.2, 10.3, 10.4, 10.4, 10.5, 10.65, 10.75, 10.85]
    cfg = _cfg(ocfg)
    res = OrbEngine(cfg).run(OneStock(_day(o, h, l, c), FEAT))
    t = res.trades.iloc[0]
    assert t["entry_px"] == pytest.approx(10.5) and t["exit_reason"] == "close"
    assert t["exit_px"] == pytest.approx(10.85)
    alloc = cfg["account"]["starting_equity"] / cfg["params"]["top_n"]
    expected = int(min(alloc * 0.01 / 0.2, alloc * 4 / 10.5))
    assert t["shares"] == expected


def test_doji_and_long_only_skip(ocfg):
    o = [10] * 9
    day = _day(o, [10.6] * 9, [9.7] * 9, [10.0] * 9)
    assert OrbEngine(_cfg(ocfg)).run(OneStock(day, {**FEAT, "close5": 10.0})).trades.empty  # doji
    short_feat = {**FEAT, "close5": 9.9}
    assert len(OrbEngine(_cfg(ocfg)).run(OneStock(day, short_feat)).trades) == 1
    assert OrbEngine(_cfg(ocfg, allow_short=False)).run(OneStock(day, short_feat)).trades.empty


def test_top_n_and_screen(ocfg, market):
    res = OrbEngine(_cfg(ocfg, top_n=5)).run(market)
    assert res.trades.groupby("date").size().max() <= 5
    assert (res.trades["rank"] <= 5).all()
    f = market.features(market.dates[50])
    s = screen(f, 5.0, 1e6, 0.5, 1.0)
    assert (s["rel_volume"] >= 1.0).all() and (s["open5"] > 5).all()


def test_no_edge_without_momentum(ocfg):
    flat = SyntheticOrbData(n_days=150, n_symbols=40, seed=11, momentum=0.0)
    res = OrbEngine(ocfg).run(flat)
    r = res.daily["ret"]
    assert r.mean() < 0  # with no real breakout effect, costs make it lose
