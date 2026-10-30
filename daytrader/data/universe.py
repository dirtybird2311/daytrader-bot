"""Market-wide data for the stocks-in-play strategy.

Four stages, each cached and resumable (a run that hits its time budget stops
cleanly and the next run picks up where it left off):

  1. assets   every US equity Alpaca knows, active and delisted
  2. daily    raw daily bars for all of them
  3. opening  the 9:30-9:35 bar for every stock that could plausibly be in play
              (prior close >= $3 and 14-day average volume >= 500k)
  4. minute   full 1-minute days, but only for the top-K by relative volume

`build_features` turns stages 2 and 3 into one row per (symbol, date) holding only
what was knowable at 9:35 that morning. It is pure and separately leak-tested.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ET = "America/New_York"
EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"}

# Loose pre-screen for which stocks get an opening bar downloaded. Must be looser
# than any screen the strategy will use.
PRE_MIN_PRICE = 3.0
PRE_MIN_AVG_VOLUME = 500_000


# ---------------------------------------------------------------------------------------------
# Pure, point-in-time feature building
# ---------------------------------------------------------------------------------------------
def daily_features(daily: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
    """For each (symbol, date): prior close, ATR and average volume from days BEFORE `date`.

    daily: columns symbol, date, o, h, l, c, v (one row per symbol per trading day).
    Today's own daily bar (its high, low, close, volume) is never used for today's row.
    """
    d = daily.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = d.groupby("symbol", sort=False)
    pc = g["c"].shift(1)
    hl = d["h"] - d["l"]
    tr = np.maximum.reduce([hl.to_numpy(), (d["h"] - pc).abs().to_numpy(), (d["l"] - pc).abs().to_numpy()])
    split_like = ((d["o"] / pc - 1).abs() > 0.4) | pc.isna()  # raw prices: a split looks like a huge gap
    d["tr"] = np.where(split_like, hl, tr)
    min_p = max(2, int(lookback * 0.7))
    atr = g["tr"].rolling(lookback, min_periods=min_p).mean().reset_index(level=0, drop=True)
    vol = g["v"].rolling(lookback, min_periods=min_p).mean().reset_index(level=0, drop=True)
    d["atr_incl"], d["vol_incl"] = atr, vol
    g = d.groupby("symbol", sort=False)
    out = pd.DataFrame({
        "symbol": d["symbol"], "date": d["date"], "prev_close": pc,
        "atr": g["atr_incl"].shift(1), "avg_volume": g["vol_incl"].shift(1),
    })
    return out


def opening_features(opening: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
    """Relative volume: today's 9:30-9:35 volume over the mean of the prior `lookback` opening bars."""
    o = opening.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = o.groupby("symbol", sort=False)
    min_p = max(2, int(lookback * 0.7))
    mean_prior = g["v"].rolling(lookback, min_periods=min_p).mean().reset_index(level=0, drop=True)
    o["avg_open_volume"] = mean_prior.groupby(o["symbol"]).shift(1)
    o["rel_volume"] = o["v"] / o["avg_open_volume"]
    return o.rename(columns={"o": "open5", "h": "high5", "l": "low5", "c": "close5", "v": "volume5"})


def build_features(daily: pd.DataFrame, opening: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
    """One row per (symbol, date) with only what was known at 9:35 that morning."""
    f = opening_features(opening, lookback).merge(daily_features(daily, lookback), on=["symbol", "date"],
                                                   how="inner")
    return f.sort_values(["date", "rel_volume"], ascending=[True, False]).reset_index(drop=True)


def screen(features: pd.DataFrame, min_price: float, min_avg_volume: float, min_atr: float,
           min_rel_volume: float) -> pd.DataFrame:
    """Stocks in play, ranked by relative volume within each day."""
    s = features[(features["open5"] > min_price) & (features["avg_volume"] >= min_avg_volume)
                 & (features["atr"] > min_atr) & (features["rel_volume"] >= min_rel_volume)].copy()
    s = s.sort_values(["date", "rel_volume"], ascending=[True, False])
    s["rank"] = s.groupby("date").cumcount() + 1
    return s


def pre_screen_next_day(daily: pd.DataFrame, calendar: list[dt.date]) -> dict[dt.date, list[str]]:
    """Which symbols to fetch an opening bar for on each day, from the PREVIOUS day's data."""
    d = daily.sort_values(["symbol", "date"]).reset_index(drop=True)
    vol = d.groupby("symbol", sort=False)["v"].rolling(14, min_periods=5).mean().reset_index(level=0, drop=True)
    ok = (d["c"] >= PRE_MIN_PRICE) & (vol >= PRE_MIN_AVG_VOLUME)
    nxt = {a: b for a, b in zip(calendar, calendar[1:])}
    d = d[ok]
    d = d.assign(next_day=d["date"].map(nxt)).dropna(subset=["next_day"])
    return {day: sorted(g["symbol"].tolist()) for day, g in d.groupby("next_day")}


# ---------------------------------------------------------------------------------------------
# Download stages
# ---------------------------------------------------------------------------------------------
class Budget:
    def __init__(self, minutes: float):
        self.deadline = time.monotonic() + minutes * 60

    def out(self) -> bool:
        return time.monotonic() > self.deadline


def _et(day: dt.date, hh: int, mm: int) -> dt.datetime:
    return pd.Timestamp(dt.datetime(day.year, day.month, day.day, hh, mm), tz=ET).to_pydatetime()


def _to_date(t: pd.Series) -> pd.Series:
    return pd.to_datetime(t, utc=True).dt.tz_convert(ET).dt.date


def _months(start: dt.date, end: dt.date) -> list[dt.date]:
    out, cur = [], dt.date(start.year, start.month, 1)
    while cur <= end:
        out.append(cur)
        cur = dt.date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
    return out


def _month_end(m: dt.date) -> dt.date:
    return dt.date(m.year + (m.month == 12), m.month % 12 + 1, 1) - dt.timedelta(days=1)


def last_complete_day() -> dt.date:
    return (pd.Timestamp.now(tz=ET) - pd.Timedelta(days=1)).date()


def fetch_tolerant(client, symbols: list[str], start, end, timeframe: str, log=print) -> pd.DataFrame:
    """fetch_multi, but if Alpaca rejects the request (400), split it and drop the bad ticker(s)."""
    try:
        return client.fetch_multi(symbols, start, end, timeframe)
    except Exception as exc:  # requests.HTTPError 400 for an invalid symbol
        if "400" not in str(exc):
            raise
        if len(symbols) == 1:
            log(f"    skipping rejected symbol {symbols[0]}")
            return pd.DataFrame(columns=["symbol", "t", "o", "h", "l", "c", "v"])
        mid = len(symbols) // 2
        return pd.concat([fetch_tolerant(client, symbols[:mid], start, end, timeframe, log),
                          fetch_tolerant(client, symbols[mid:], start, end, timeframe, log)], ignore_index=True)


def clean_assets(a: pd.DataFrame) -> pd.DataFrame:
    """Real tickers only (e.g. AAPL, BRK.B); drops escrow/CVR/rights placeholders like 003CVR016."""
    keep = a["exchange"].isin(EXCHANGES) & a["symbol"].astype(str).str.fullmatch(r"[A-Z]{1,5}(\.[A-Z]{1,2})?")
    return a[keep].drop_duplicates("symbol").reset_index(drop=True)


def sync_assets(client, root: Path, log=print) -> pd.DataFrame:
    path = root / "assets.parquet"
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < 7 * 86400
    a = pd.read_parquet(path) if fresh else client.list_assets()
    a = clean_assets(a)
    a.to_parquet(path, index=False)
    log(f"Assets: {len(a):,} US equities ({(a['status'] == 'inactive').sum():,} delisted/inactive)")
    return a


def sync_daily(client, symbols: list[str], start: dt.date, end: dt.date, root: Path, budget: Budget,
               log=print, batch: int = 200) -> bool:
    """Daily bars by year (past years) or month (current year). Returns True when complete."""
    folder = root / "daily"
    folder.mkdir(parents=True, exist_ok=True)
    this_year = end.year
    chunks = [(dt.date(y, 1, 1), dt.date(y, 12, 31), f"{y}", True) for y in range(start.year, this_year)]
    chunks += [(m, min(_month_end(m), end), f"{m:%Y-%m}", _month_end(m) < end)
               for m in _months(dt.date(this_year, 1, 1), end)]
    for s, e, name, final in chunks:
        path = folder / f"{name}.parquet"
        if path.exists() and final:
            continue
        if budget.out():
            return False
        frames = []
        for i in range(0, len(symbols), batch):
            frames.append(fetch_tolerant(client, symbols[i:i + batch], _et(s, 0, 0), _et(e, 23, 59), "1Day", log))
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(path, index=False)
        log(f"  daily {name}: {len(df):,} bars, {df['symbol'].nunique():,} symbols")
    return True


def load_daily(root: Path) -> pd.DataFrame:
    files = sorted((root / "daily").glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["symbol", "date", "o", "h", "l", "c", "v"])
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = _to_date(df["t"])
    df = df.drop(columns="t").drop_duplicates(["symbol", "date"], keep="last")
    for c in ("o", "h", "l", "c"):
        df[c] = df[c].astype("float64")
    df["v"] = df["v"].astype("float64")
    df["symbol"] = df["symbol"].astype("category")
    return df


def trading_calendar(daily: pd.DataFrame, anchor: str = "SPY") -> list[dt.date]:
    days = daily.loc[daily["symbol"] == anchor, "date"]
    if days.empty:
        days = daily["date"]
    return sorted(set(days))


def sync_opening(client, daily: pd.DataFrame, start: dt.date, end: dt.date, root: Path, budget: Budget,
                 log=print, batch: int = 400) -> bool:
    folder = root / "opening"
    folder.mkdir(parents=True, exist_ok=True)
    cal = trading_calendar(daily)
    wanted = pre_screen_next_day(daily, cal)
    for m in _months(start, end):
        path = folder / f"{m:%Y-%m}.parquet"
        final = _month_end(m) < end
        if path.exists() and final:
            continue
        if budget.out():
            return False
        frames = []
        for day in [d for d in cal if d.year == m.year and d.month == m.month and start <= d <= end]:
            syms = wanted.get(day, [])
            for i in range(0, len(syms), batch):
                frames.append(fetch_tolerant(client, syms[i:i + batch], _et(day, 9, 30), _et(day, 9, 34), "5Min", log))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["symbol", "t", "o", "h", "l", "c", "v"])
        df.to_parquet(path, index=False)
        log(f"  opening {m:%Y-%m}: {len(df):,} bars")
    return True


def load_opening(root: Path) -> pd.DataFrame:
    files = sorted((root / "opening").glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["symbol", "date", "o", "h", "l", "c", "v"])
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ts = pd.to_datetime(df["t"], utc=True).dt.tz_convert(ET)
    df = df[(ts.dt.hour == 9) & (ts.dt.minute == 30)].copy()  # the 9:30-9:35 bar only
    df["date"] = ts[df.index].dt.date
    df = df.drop(columns="t").drop_duplicates(["symbol", "date"], keep="last")
    df["symbol"] = df["symbol"].astype(str)
    return df


def sync_minutes(client, features: pd.DataFrame, screen_cfg: dict, top_k: int, start: dt.date, end: dt.date,
                 root: Path, budget: Budget, log=print) -> bool:
    folder = root / "minute"
    folder.mkdir(parents=True, exist_ok=True)
    picks = screen(features, **screen_cfg)
    picks = picks[picks["rank"] <= top_k]
    by_day = {d: g["symbol"].tolist() for d, g in picks.groupby("date")}
    for m in _months(start, end):
        path = folder / f"{m:%Y-%m}.parquet"
        final = _month_end(m) < end
        if path.exists() and final:
            continue
        if budget.out():
            return False
        frames = []
        for day in sorted(d for d in by_day if d.year == m.year and d.month == m.month and start <= d <= end):
            frames.append(fetch_tolerant(client, by_day[day], _et(day, 9, 30), _et(day, 15, 59), "1Min", log))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["symbol", "t", "o", "h", "l", "c", "v"])
        df.to_parquet(path, index=False)
        n = sum(len(v) for d, v in by_day.items() if d.year == m.year and d.month == m.month)
        log(f"  minute {m:%Y-%m}: {len(df):,} bars for {n} stock-days")
    return True


def sync_extra_minutes(client, features: pd.DataFrame, daily: pd.DataFrame, cfg: dict, start: dt.date,
                       end: dt.date, root: Path, budget: Budget, log=print) -> bool:
    """Minute bars for gap-and-go and buy-the-dip candidates not in the main minute download."""
    from .setups import extra_minute_pairs

    folder = root / "minute_extra"
    folder.mkdir(parents=True, exist_ok=True)
    s = cfg["download"]
    picks = screen(features, **s["screen"])
    picks = picks[picks["rank"] <= int(s["top_k"])]
    already = {d: set(g["symbol"].astype(str)) for d, g in picks.groupby("date")}
    by_day = extra_minute_pairs(features, daily, s["extra"], already)
    for m in _months(start, end):
        path = folder / f"{m:%Y-%m}.parquet"
        final = _month_end(m) < end
        if path.exists() and final:
            continue
        if budget.out():
            return False
        frames = []
        days = sorted(d for d in by_day if d.year == m.year and d.month == m.month and start <= d <= end and by_day[d])
        for day in days:
            frames.append(fetch_tolerant(client, by_day[day], _et(day, 9, 30), _et(day, 15, 59), "1Min", log))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["symbol", "t", "o", "h", "l", "c", "v"])
        df.to_parquet(path, index=False)
        n = sum(len(by_day[d]) for d in days)
        log(f"  extra minute {m:%Y-%m}: {len(df):,} bars for {n} stock-days")
    return True


def sync_snapshots(client, features: pd.DataFrame, snap_cfg: dict, start: dt.date, end: dt.date, root: Path,
                   budget: Budget, log=print) -> bool:
    """3:50 pm prices (close of the 3:49 bar) for the largest stocks by prior dollar volume."""
    folder = root / "snap"
    folder.mkdir(parents=True, exist_ok=True)
    f = features[features["prev_close"] > float(snap_cfg["min_price"])][["symbol", "date", "prev_close", "avg_volume"]].copy()
    f["dv"] = f["avg_volume"] * f["prev_close"]
    f["r"] = f.groupby("date")["dv"].rank(ascending=False, method="first")
    f = f[f["r"] <= int(snap_cfg["top"])]
    by_day = {d: sorted(g["symbol"].astype(str)) for d, g in f.groupby("date")}
    for m in _months(start, end):
        path = folder / f"{m:%Y-%m}.parquet"
        if path.exists() and _month_end(m) < end:
            continue
        if budget.out():
            return False
        rows = []
        days = sorted(d for d in by_day if d.year == m.year and d.month == m.month and start <= d <= end)
        for day in days:
            t0 = _et(day, 15, 49)
            got = fetch_tolerant(client, by_day[day], t0, t0 + dt.timedelta(seconds=59), "1Min", log)
            if not got.empty:
                rows.append(pd.DataFrame({"symbol": got["symbol"].astype(str), "date": day, "snap": got["c"].astype(float)}))
        df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["symbol", "date", "snap"])
        df.to_parquet(path, index=False)
        log(f"  snap {m:%Y-%m}: {len(df):,} prices over {len(days)} days")
    return True


def load_snapshots(root: Path) -> pd.DataFrame:
    files = sorted((root / "snap").glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["symbol", "date", "snap"])
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.drop_duplicates(["symbol", "date"], keep="last")


def sync_all(client, cfg: dict, minutes: float, log=print) -> dict:
    """Run every stage that still needs work within the time budget. Returns a status dict."""
    root = Path(cfg["data"]["cache_dir"]) / "universe"
    root.mkdir(parents=True, exist_ok=True)
    budget = Budget(minutes)
    start = dt.date.fromisoformat(str(cfg["data"]["start"]))
    pre_start = (pd.Timestamp(start) - pd.Timedelta(days=60)).date()  # warm-up for 14-day lookbacks
    end = last_complete_day()
    status = {"assets": False, "daily": False, "opening": False, "features": False, "minute": False}
    if cfg["download"].get("extra"):
        status["extra"] = False
    if cfg["download"].get("snap"):
        status["snap"] = False

    assets = sync_assets(client, root, log)
    status["assets"] = True
    symbols = sorted(set(assets["symbol"].astype(str)) | {"SPY"})
    log(f"Stage daily: {len(symbols):,} symbols")
    status["daily"] = sync_daily(client, symbols, pre_start, end, root, budget, log)
    if status["daily"]:
        daily = load_daily(root)
        log("Stage opening")
        status["opening"] = sync_opening(client, daily, pre_start, end, root, budget, log)
        if status["opening"]:
            feats = build_features(daily, load_opening(root), int(cfg["params"]["lookback_days"]))
            feats.to_parquet(root / "features.parquet", index=False)
            status["features"] = True
            log(f"Features: {len(feats):,} stock-days")
            s = cfg["download"]
            status["minute"] = sync_minutes(client, feats, s["screen"], int(s["top_k"]), start, end, root,
                                            budget, log)
            if status["minute"] and "extra" in status:
                log("Stage extra minutes (gap-and-go, buy-the-dip)")
                status["extra"] = sync_extra_minutes(client, feats, daily, cfg, start, end, root, budget, log)
            if status["minute"] and status.get("extra", True) and "snap" in status:
                log("Stage 3:50 pm snapshots (batch D)")
                status["snap"] = sync_snapshots(client, feats, s["snap"], start, end, root, budget, log)
    status["complete"] = all(status.values())
    status["through"] = str(end)
    (root / "status.json").write_text(json.dumps(status, indent=2))
    return status
