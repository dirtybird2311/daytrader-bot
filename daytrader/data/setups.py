"""Morning candidate lists for gap-and-go and buy-the-dip (research queue #7 and #8).

Both use only what is known before (or, for the gap, at) the 9:30 open:
  * gap_candidates: large caps (top `large_cap_rank` by 14-day average dollar volume) gapping
    up between `min_gap` and `max_gap`, ranked by 5-minute relative volume
  * dip_table: stocks finishing a "parabolic" run that closed above old resistance

Prices are raw (unadjusted). Any day in a window that looks like a split (overnight gap over
40%) disqualifies that window, so a reverse split can't pose as a breakout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPLIT_GAP = 0.4


def gap_candidates(features: pd.DataFrame, min_gap: float, max_gap: float, large_cap_rank: int,
                   min_price: float) -> pd.DataFrame:
    """features: rows for one or many dates (from universe.build_features). Adds gap, dv_rank, rank."""
    f = features.copy()
    f["gap"] = f["open5"] / f["prev_close"] - 1.0
    f["dollar_volume"] = f["avg_volume"] * f["prev_close"]
    f["dv_rank"] = f.groupby("date")["dollar_volume"].rank(ascending=False, method="first")
    s = f[(f["dv_rank"] <= large_cap_rank) & (f["open5"] > min_price) & (f["gap"] >= min_gap)
          & (f["gap"] <= max_gap) & f["rel_volume"].notna()]
    s = s.sort_values(["date", "rel_volume"], ascending=[True, False]).copy()
    s["rank"] = s.groupby("date").cumcount() + 1
    return s


def dip_table(daily: pd.DataFrame, resistance_lookback: int = 20, max_run: int = 6) -> pd.DataFrame:
    """One row per (symbol, date) with the run and resistance known before that date's open.

    run_len:  consecutive days, ending yesterday, with close > prior close AND a wider high-low
              range than the prior day (capped at max_run)
    res_k:    highest high of the `resistance_lookback` days before a k-day run began (k = 3..5)
    prev_close, split_free_k: whether the run plus its resistance window is free of split gaps
    """
    d = daily[["symbol", "date", "o", "h", "l", "c"]].sort_values(["symbol", "date"]).reset_index(drop=True)
    d["symbol"] = d["symbol"].astype(str)
    out = []
    for sym, s in d.groupby("symbol", sort=False):
        c, h, l, o = (s[k].to_numpy(float) for k in ("c", "h", "l", "o"))
        n = len(s)
        pc = np.r_[np.nan, c[:-1]]
        rng_ = h - l
        prng = np.r_[np.nan, rng_[:-1]]
        good = (c > pc) & (rng_ > prng)
        split = np.abs(o / pc - 1) > SPLIT_GAP
        # length of the streak of `good` days ending at each day, capped at max_run
        grp = np.cumsum(~good)
        run = np.minimum(max_run, pd.Series(good.astype(int)).groupby(grp).cumsum().to_numpy())
        splits_cum = np.cumsum(np.nan_to_num(split.astype(float)))
        row = {"symbol": sym, "date": s["date"].to_numpy()}
        # values for date index t use days <= t-1
        row["run_len"] = np.r_[0, run[:-1]]
        row["prev_close"] = np.r_[np.nan, c[:-1]]
        hs = pd.Series(h)
        for k in (3, 4, 5):
            # run occupies days t-k .. t-1; resistance window is t-k-L .. t-k-1
            res = hs.rolling(resistance_lookback, min_periods=resistance_lookback).max().shift(k + 1).to_numpy()
            span = resistance_lookback + k
            sc = np.r_[0.0, splits_cum[:-1]]
            before = np.r_[np.zeros(span), sc[:-span]] if n > span else np.zeros(n)
            row[f"res_{k}"] = res
            row[f"split_free_{k}"] = (sc - before) == 0
        out.append(pd.DataFrame(row))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def dip_setups(table: pd.DataFrame, run_days: int) -> pd.DataFrame:
    k = int(run_days)
    if k not in (3, 4, 5):
        raise ValueError("run_days must be 3, 4 or 5")
    t = table
    ok = (t["run_len"] >= k) & (t["prev_close"] > t[f"res_{k}"]) & t[f"split_free_{k}"]
    return t.loc[ok, ["symbol", "date", "prev_close", "run_len", f"res_{k}"]].rename(columns={f"res_{k}": "resistance"})


def extra_minute_pairs(features: pd.DataFrame, daily: pd.DataFrame, cfg_extra: dict,
                       already: dict) -> dict:
    """(date -> symbols) needing minute bars for gap-and-go and buy-the-dip, minus `already`
    (date -> symbols already in the main minute download)."""
    want: dict = {}
    g = cfg_extra["gap"]
    for thr in g["min_gaps"]:
        c = gap_candidates(features, float(thr), float(g["max_gap"]), int(g["large_cap_rank"]),
                           float(g["min_price"]))
        c = c[c["rank"] <= int(g["top"])]
        for d, s in zip(c["date"], c["symbol"]):
            want.setdefault(d, set()).add(str(s))
    dcfg = cfg_extra["dip"]
    table = dip_table(daily, int(dcfg["resistance_lookback"]))
    liq = features[(features["open5"] > float(dcfg["min_price"])) & (features["avg_volume"] >= float(dcfg["min_avg_volume"]))
                   & (features["atr"] > float(dcfg["min_atr"]))][["symbol", "date", "avg_volume"]].copy()
    liq["symbol"] = liq["symbol"].astype(str)
    frames = [dip_setups(table, k) for k in dcfg["run_days"]]
    setups = pd.concat(frames).drop_duplicates(["symbol", "date"]).merge(liq, on=["symbol", "date"])
    setups["dv"] = setups["avg_volume"] * setups["prev_close"]
    setups = setups.sort_values(["date", "dv"], ascending=[True, False]).groupby("date").head(int(dcfg["top"]))
    for d, s in zip(setups["date"], setups["symbol"]):
        want.setdefault(d, set()).add(str(s))
    return {d: sorted(s - set(already.get(d, ()))) for d, s in want.items()}
