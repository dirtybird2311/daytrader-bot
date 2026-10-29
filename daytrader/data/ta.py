"""Daily technical indicators for the stocks-in-play filters (research queue #2).

Every value on row (symbol, date) uses daily bars from BEFORE `date`, plus today's 9:30
open (known when the filter is applied at 9:35). Prices in the cache are raw (unadjusted),
so indicators are computed on a split-safe price index: on a day that looks like a split
(an overnight gap over 40%), the overnight move is dropped and only open-to-close counts.

Columns produced:
  ta_trend      +1 if the prior close is above its `sma_days` simple moving average, -1 below
  ta_rsi        Wilder RSI(14) of the prior closes
  ta_macd       +1 if MACD(12, 26) is above its 9-day signal line on the prior close, -1 below
  ta_bb_z       today's open, in standard deviations from the 20-day mean of prior closes
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPLIT_GAP = 0.4


def _rsi(x: pd.Series, n: int) -> pd.Series:
    d = x.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn > 0, 100.0).where(up.notna())


def ta_table(daily: pd.DataFrame, sma_days: int = 20, rsi_days: int = 14, bb_days: int = 20) -> pd.DataFrame:
    """daily: symbol, date, o, c (one row per symbol per trading day)."""
    d = daily[["symbol", "date", "o", "c"]].sort_values(["symbol", "date"]).reset_index(drop=True)
    d["symbol"] = d["symbol"].astype(str)
    g = d.groupby("symbol", sort=False)
    pc = g["c"].shift(1)
    split = ((d["o"] / pc - 1).abs() > SPLIT_GAP) | pc.isna()
    step = np.where(split, d["c"] / d["o"], d["c"] / pc)
    d["px"] = np.exp(pd.Series(np.log(step)).groupby(d["symbol"]).cumsum())
    d["gap"] = np.where(split, 1.0, d["o"] / pc)  # today's open relative to the prior close

    def per_symbol(s: pd.DataFrame) -> pd.DataFrame:
        px = s["px"]
        sma = px.rolling(sma_days, min_periods=sma_days).mean()
        macd = px.ewm(span=12, adjust=False, min_periods=26).mean() - px.ewm(span=26, adjust=False, min_periods=26).mean()
        signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        mean = px.rolling(bb_days, min_periods=bb_days).mean()
        sd = px.rolling(bb_days, min_periods=bb_days).std(ddof=0)
        out = pd.DataFrame({
            "trend": np.sign(px - sma).where(sma.notna()),
            "rsi": _rsi(px, rsi_days),
            "macd": np.sign(macd - signal).where(signal.notna()),
            "mean": mean, "sd": sd, "px": px,
        }, index=s.index)
        out = out.shift(1)  # everything from completed days only
        open_px = out["px"] * s["gap"]  # today's open on the same index
        return pd.DataFrame({
            "ta_trend": out["trend"], "ta_rsi": out["rsi"], "ta_macd": out["macd"],
            "ta_bb_z": (open_px - out["mean"]) / out["sd"].replace(0, np.nan),
        }, index=s.index)

    parts = [per_symbol(s) for _, s in d.groupby("symbol", sort=False)]
    res = pd.concat(parts).sort_index()
    return pd.concat([d[["symbol", "date"]], res], axis=1)


def passes(row: pd.Series, direction: int, rule: str, rsi_hi: float = 80, rsi_lo: float = 20,
           bb_k: float = 2.0) -> bool:
    """Whether a breakout in `direction` is allowed by filter `rule`. Missing data = not allowed."""
    if rule == "none":
        return True
    if rule == "sma":
        v = row.get("ta_trend")
        return bool(v == direction)
    if rule == "rsi":
        v = row.get("ta_rsi")
        if v is None or not np.isfinite(v):
            return False
        return bool(v <= rsi_hi) if direction > 0 else bool(v >= rsi_lo)
    if rule == "macd":
        v = row.get("ta_macd")
        return bool(v == direction)
    if rule == "bollinger":
        v = row.get("ta_bb_z")
        return bool(v is not None and np.isfinite(v) and abs(v) <= bb_k)
    raise ValueError(f"Unknown ta_filter '{rule}' (none, sma, rsi, macd, bollinger)")
