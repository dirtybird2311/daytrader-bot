"""Stocks-in-play research variants: leak hunters and rule checks."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from daytrader.backtest.orb import OrbEngine
from daytrader.config import load_config, override
from daytrader.data.ta import passes, ta_table
from tests.conftest import CONFIG_DIR
from tests.test_orb import FEAT, OneStock, _cfg, _day
from tests.test_orb import PoisonedMinutes as _Poisoned


class PoisonedMinutes(_Poisoned):
    def daily_frame(self):
        return self.base.daily_frame()

VARIANTS = {
    "or1": {"params.or_minutes": 1},
    "or15": {"params.or_minutes": 15},
    "or30": {"params.or_minutes": 30},
    "vwap": {"params.exit_mode": "vwap"},
    "sma": {"params.ta_filter": "sma"},
    "rsi": {"params.ta_filter": "rsi"},
    "macd": {"params.ta_filter": "macd"},
    "bollinger": {"params.ta_filter": "bollinger"},
}


@pytest.fixture(scope="module")
def ocfg():
    return load_config(CONFIG_DIR / "orb.yaml")


@pytest.fixture(scope="module")
def market():
    from daytrader.data.orb_data import SyntheticOrbData
    return SyntheticOrbData(n_days=120, n_symbols=40, seed=3, momentum=0.12)


@pytest.mark.parametrize("name", sorted(VARIANTS))
@pytest.mark.parametrize("day_idx,bar", [(50, 31), (80, 200)])
def test_variant_poisoned_minutes(ocfg, market, name, day_idx, bar):
    cfg = override(ocfg, VARIANTS[name])
    cut = market.dates[day_idx]
    clean = OrbEngine(cfg).run(market)
    dirty = OrbEngine(cfg).run(PoisonedMinutes(market, cut, bar))

    def before(res):
        return [(d.date, d.bar, d.target, d.reason) for d in res.decisions
                if d.date < cut or (d.date == cut and d.bar < bar)]

    assert before(clean) == before(dirty)
    assert len(before(clean)) > 0
    orders = lambda r: [(d.reason, d.price) for d in r.decisions if d.date == cut and d.reason.startswith("order")]
    assert orders(clean) == orders(dirty)


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_variant_truncation(ocfg, market, name):
    cfg = override(ocfg, VARIANTS[name])
    full = OrbEngine(cfg).run(market)
    cut = market.dates[70]
    short = OrbEngine(cfg).run(market, end=market.dates[69])
    a = full.trades[full.trades["date"] < pd.Timestamp(cut)].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, short.trades.reset_index(drop=True))
    assert len(a) > 0


def test_filters_only_remove_trades(ocfg, market):
    base = OrbEngine(ocfg).run(market).trades
    key = ["date", "symbol", "side", "entry_px", "exit_px"]
    for f in ("sma", "rsi", "macd", "bollinger"):
        t = OrbEngine(override(ocfg, {"params.ta_filter": f})).run(market).trades
        assert 0 < len(t) < len(base)
        merged = t[key].merge(base[key], how="left", indicator=True)
        assert (merged["_merge"] == "both").all()


# ---- indicators -----------------------------------------------------------------------------
def _daily(px, opens=None, sym="X"):
    days = pd.bdate_range("2020-01-01", periods=len(px)).date
    o = np.asarray(opens if opens is not None else px, float)
    return pd.DataFrame({"symbol": sym, "date": days, "o": o, "c": np.asarray(px, float)})


def test_ta_uses_only_prior_days_and_todays_open(market):
    d = market.daily[["symbol", "date", "o", "c"]]
    cut = market.dates[60]
    clean = ta_table(d)
    dirty = d.copy()
    rng = np.random.default_rng(1)
    fut = dirty["date"] > cut
    dirty.loc[fut, ["o", "c"]] *= rng.uniform(0.3, 3, (fut.sum(), 2))
    today = dirty["date"] == cut
    dirty.loc[today, "c"] *= rng.uniform(0.3, 3, today.sum())  # today's close is unknown at 9:35
    pd.testing.assert_frame_equal(clean[clean["date"] <= cut].reset_index(drop=True),
                                  ta_table(dirty)[lambda x: x["date"] <= cut].reset_index(drop=True))


def test_sma_trend_and_rsi_values():
    up = np.linspace(10, 20, 40)
    t = ta_table(_daily(up)).iloc[-1]
    assert t["ta_trend"] == 1 and t["ta_rsi"] == pytest.approx(100.0) and t["ta_macd"] in (1, -1)
    down = np.linspace(20, 10, 40)
    t = ta_table(_daily(down)).iloc[-1]
    assert t["ta_trend"] == -1 and t["ta_rsi"] == pytest.approx(0.0)


def test_ta_is_split_safe():
    rng = np.random.default_rng(0)
    px = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, 60)))
    px[30:] *= px[29] / px[30]  # flat across the split day: the one return a split hides
    split = px.copy()
    split[30:] = split[30:] / 2  # 2-for-1 split on day 30 in raw prices
    a = ta_table(_daily(px)).iloc[35:].reset_index(drop=True)
    b = ta_table(_daily(split)).iloc[35:].reset_index(drop=True)
    cols = ["ta_trend", "ta_macd"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    np.testing.assert_allclose(a["ta_rsi"], b["ta_rsi"], rtol=0.05)
    # Bollinger: today's open far above the 20-day band
    opens = np.r_[np.full(30, 10.0), [13.0]]
    z = ta_table(_daily(np.r_[np.full(30, 10.0) + rng.normal(0, 0.05, 30), [13.0]], opens)).iloc[-1]["ta_bb_z"]
    assert z > 2


def test_filter_rules():
    r = pd.Series({"ta_trend": 1, "ta_rsi": 85.0, "ta_macd": -1, "ta_bb_z": 2.5})
    assert passes(r, 1, "sma") and not passes(r, -1, "sma")
    assert not passes(r, 1, "rsi") and passes(r, -1, "rsi")
    assert not passes(r, 1, "macd") and passes(r, -1, "macd")
    assert not passes(r, 1, "bollinger")
    assert not passes(pd.Series(dtype=float), 1, "rsi")  # no data: no trade
    assert passes(r, 1, "none")


# ---- opening-range length, VWAP exit, tick resolution ----------------------------------------
def test_fifteen_minute_range_levels(ocfg):
    n = 40
    o = np.full(n, 10.0); h = np.full(n, 10.1); l = np.full(n, 9.9); c = np.full(n, 10.05)
    h[12] = 10.6  # range high inside minutes 0-14 but after the first five
    h[20], o[21], c[20] = 10.7, 10.65, 10.65
    res = OrbEngine(_cfg(ocfg, or_minutes=15)).run(OneStock(_day(o, h, l, c), FEAT))
    t = res.trades.iloc[0]
    assert t["entry_px"] == pytest.approx(10.6) and t["entry_bar"] == 20


def test_one_minute_range_orders_wait_for_935(ocfg):
    n = 20
    o = np.full(n, 10.0); h = np.full(n, 10.08); l = np.full(n, 9.95); c = np.full(n, 10.05)
    h[0] = 10.1  # the 1-minute range high
    h[2] = 10.5  # breaks the 1-minute high at 9:32, before the stock is known to be in play
    res = OrbEngine(_cfg(ocfg, or_minutes=1)).run(OneStock(_day(o, h, l, c), FEAT))
    assert res.trades.empty
    h[7] = 10.3
    t = OrbEngine(_cfg(ocfg, or_minutes=1)).run(OneStock(_day(o, h, l, c), FEAT)).trades.iloc[0]
    assert t["entry_bar"] == 7 and t["entry_px"] == pytest.approx(10.1)


def test_vwap_exit_next_open(ocfg):
    o = [10, 10.1, 10.2, 10.3, 10.35, 10.45, 10.6, 10.5, 10.1, 10.0]
    h = [10.2, 10.3, 10.4, 10.5, 10.45, 10.55, 10.7, 10.6, 10.15, 10.05]
    l = [9.8, 10.0, 10.1, 10.2, 10.3, 10.44, 10.55, 10.12, 10.0, 9.95]
    c = [10.1, 10.2, 10.3, 10.4, 10.4, 10.5, 10.62, 10.15, 10.05, 10.0]
    cfg = _cfg(ocfg, exit_mode="vwap", stop_atr=0.5)
    t = OrbEngine(cfg).run(OneStock(_day(o, h, l, c), FEAT)).trades.iloc[0]
    # bar 7 closes at 10.15, below the session VWAP (~10.3): out at bar 8's open
    assert t["exit_reason"] == "vwap" and t["exit_bar"] == 8 and t["exit_px"] == pytest.approx(10.1)
    plain = OrbEngine(_cfg(ocfg, stop_atr=0.5)).run(OneStock(_day(o, h, l, c), FEAT)).trades.iloc[0]
    assert plain["exit_reason"] == "close"


class WithTicks(OneStock):
    def __init__(self, day, feat, prints):
        super().__init__(day, feat)
        self.prints = prints

    def tick_prices(self, date, symbol, bar):
        return None if self.prints is None else np.asarray(self.prints, float)


# Ambiguous entry bar (same as test_orb.test_same_bar_stop_modes_mid_bar_entry): level 10.5, stop 10.3
AMB = dict(o=[10, 10.1, 10.2, 10.3, 10.35, 10.4, 10.55, 10.6, 10.7],
           h=[10.2, 10.3, 10.4, 10.5, 10.45, 10.6, 10.65, 10.7, 10.8],
           l=[9.8, 10.0, 10.1, 10.2, 10.3, 10.25, 10.5, 10.55, 10.65],
           c=[10.1, 10.2, 10.3, 10.4, 10.4, 10.55, 10.6, 10.7, 10.75])


@pytest.mark.parametrize("prints,reason,source", [
    ([10.4, 10.25, 10.45, 10.5, 10.6, 10.55], "close", "ticks"),   # dip came before the breakout
    ([10.4, 10.5, 10.6, 10.28, 10.55], "stop", "ticks"),           # breakout, then down through the stop
    (None, "stop", "ticks_missing"),                               # no tick data: pessimistic
])
def test_tick_resolution(ocfg, prints, reason, source):
    cfg = override(_cfg(ocfg), {"fills.same_bar_stop": "ticks"})
    eng = OrbEngine(cfg)
    t = eng.run(WithTicks(_day(AMB["o"], AMB["h"], AMB["l"], AMB["c"]), FEAT, prints)).trades.iloc[0]
    assert t["exit_reason"] == reason and t["fill_source"] == source
    assert eng.ambiguous == [(dt.date(2021, 3, 1), "X", 5)]
    if reason == "stop" and source == "ticks":
        assert t["exit_px"] == pytest.approx(10.28) and t["exit_bar"] == 5  # worse of stop and the print


def test_tick_entry_price_from_first_print_through_level(ocfg):
    cfg = override(_cfg(ocfg), {"fills.same_bar_stop": "ticks"})
    t = OrbEngine(cfg).run(WithTicks(_day(AMB["o"], AMB["h"], AMB["l"], AMB["c"]), FEAT, [10.4, 10.52, 10.6])).trades.iloc[0]
    assert t["entry_px"] == pytest.approx(10.52) and t["stop_at_entry"] == pytest.approx(10.32)


def test_tick_sync_fetches_once_and_keeps_round_lots(tmp_path):
    from daytrader.data.ticks import TickStore, sync_ticks
    from daytrader.data.universe import Budget

    class FakeClient:
        feed = "sip"
        calls = []

        def _get(self, params, url=None):
            self.calls.append(params["symbols"])
            syms = params["symbols"].split(",")
            if "page_token" not in params:
                return {"trades": {s: [{"t": "a", "p": 10.0, "s": 100}, {"t": "b", "p": 10.5, "s": 5}]
                                   for s in syms}, "next_page_token": "x"}
            return {"trades": {syms[0]: [{"t": "c", "p": 10.2, "s": 300}]}, "next_page_token": None}

    d = dt.date(2021, 3, 1)
    need = [(d, "AAA", 5), (d, "BBB", 5), (d, "AAA", 7)]
    client = FakeClient()
    st = sync_ticks(client, need, tmp_path, Budget(5), log=lambda *_: None)
    assert st["fetched"] == 3 and st["complete"]
    assert len(client.calls) == 4  # (date, bar 5) shared by two symbols: one paged request; bar 7: another
    store = TickStore(tmp_path)
    np.testing.assert_allclose(store.get(d, "AAA", 5), [10.0, 10.2])  # odd lot (5 shares) dropped
    np.testing.assert_allclose(store.get(d, "BBB", 5), [10.0])
    again = sync_ticks(client, need, tmp_path, Budget(5), log=lambda *_: None)
    assert again["fetched"] == 0 and len(client.calls) == 4
