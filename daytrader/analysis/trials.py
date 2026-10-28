"""Log of every variant ever run, so luck can be discounted.

Each backtest appends one line to runs/trials.jsonl and saves its daily returns.
The deflated Sharpe ratio and overfitting probability read this log. The folder is
committed to the repo (it is small), so the count survives across GitHub Actions runs.
Holdout views are counted too: the holdout is meant to be scored once.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS_DIR = Path("runs")


def config_hash(cfg: dict) -> str:
    keys = {k: cfg.get(k) for k in ("strategy", "symbol", "params", "sizing", "costs")}
    return hashlib.sha1(json.dumps(keys, sort_keys=True, default=str).encode()).hexdigest()[:10]


def record_trial(run_id: str, cfg: dict, period: str, daily: pd.DataFrame, label: str = "",
                 synthetic: bool = False, runs_dir: Path = RUNS_DIR) -> None:
    if synthetic:
        return  # synthetic runs never count toward the trial log
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "returns").mkdir(exist_ok=True)
    r = daily["ret"]
    entry = {
        "run_id": run_id, "time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config_hash": config_hash(cfg), "strategy": cfg["strategy"], "period": period, "label": label,
        "days": int(len(r)), "daily_sharpe": float(r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else 0.0,
    }
    with open(runs_dir / "trials.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    r.rename(entry["config_hash"]).to_frame().to_parquet(runs_dir / "returns" / f"{run_id}.parquet")


def load_trials(runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    path = runs_dir / "trials.jsonl"
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line.strip()])


def trial_sharpes(strategy: str, period: str, runs_dir: Path = RUNS_DIR) -> list[float]:
    """Latest daily Sharpe per distinct variant tried on this period."""
    t = load_trials(runs_dir)
    if t.empty:
        return []
    t = t[(t["strategy"] == strategy) & (t["period"] == period)]
    return t.groupby("config_hash")["daily_sharpe"].last().tolist()


def trial_matrix(strategy: str, period: str, runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    t = load_trials(runs_dir)
    if t.empty:
        return pd.DataFrame()
    t = t[(t["strategy"] == strategy) & (t["period"] == period)]
    latest = t.groupby("config_hash")["run_id"].last()
    cols = []
    for h, run_id in latest.items():
        f = runs_dir / "returns" / f"{run_id}.parquet"
        if f.exists():
            cols.append(pd.read_parquet(f).iloc[:, 0].rename(h))
    return pd.concat(cols, axis=1) if cols else pd.DataFrame()


def holdout_views(strategy: str, runs_dir: Path = RUNS_DIR) -> int:
    t = load_trials(runs_dir)
    if t.empty:
        return 0
    return int(((t["strategy"] == strategy) & (t["period"] == "holdout")).sum())


def neighbours_positive(sweep: pd.DataFrame, chosen: dict) -> float:
    """Share of one-step neighbours of the chosen setting with positive return (the plateau check)."""
    params = [c for c in sweep.columns if c not in ("sharpe", "annual_return", "total_return")]
    if not params:
        return float("nan")
    grids = {p: sorted(sweep[p].unique()) for p in params}
    try:
        pos = {p: grids[p].index(chosen[p]) for p in params}
    except (KeyError, ValueError):
        return float("nan")
    hits = total = 0
    for p in params:
        for step in (-1, 1):
            j = pos[p] + step
            if 0 <= j < len(grids[p]):
                mask = np.ones(len(sweep), dtype=bool)
                for q in params:
                    val = grids[q][j] if q == p else grids[q][pos[q]]
                    mask &= np.isclose(sweep[q].astype(float), float(val))
                row = sweep[mask]
                if not row.empty:
                    total += 1
                    hits += int(row["total_return"].iloc[0] > 0)
    return hits / total if total else float("nan")
