"""Batch research: run a list of pre-registered experiments in one go, and fetch tick data.

  python -m daytrader research --file config/experiments/phase_a.yaml
  python -m daytrader ticks --config config/orb.yaml

An experiment file looks like:

  name: phase_a
  experiments:
    - name: gao
      config: config/spy_lasthalf.yaml
      period: tune                      # tune (default) or validate
      set: {params.allow_short: false}  # optional overrides
      grid:                             # optional: one run per combination (tune only)
        params.entry_sd: [1.5, 2.0]

Every run goes into the trial log and gets a full HTML report. A one-line summary per run is
written to runs/research/<file name>.csv, which also lets an interrupted batch resume: runs
already in that table with the same settings are skipped.

Stocks-in-play runs also save the entry minutes that 1-minute bars can't resolve to
runs/ambiguous/; the `ticks` command downloads trade data for exactly those minutes.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from .analysis.analyze import analyze
from .analysis.trials import config_hash, record_trial
from .config import load_config, override, period_bounds
from .report.html import build_report

RESEARCH_DIR = Path("runs") / "research"
AMBIGUOUS_DIR = Path("runs") / "ambiguous"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")[:120]


def expand(spec: dict) -> list[dict]:
    """Experiments with any grid expanded into one entry per combination."""
    out = []
    for e in spec["experiments"]:
        grid = e.get("grid") or {}
        if not grid:
            out.append({**e, "set": dict(e.get("set") or {})})
            continue
        if e.get("period", "tune") != "tune":
            raise ValueError(f"{e['name']}: grids run on the tune period only")
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            tag = ",".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, combo))
            out.append({**e, "name": f"{e['name']}[{tag}]", "set": {**(e.get("set") or {}), **dict(zip(keys, combo))},
                        "_group": e["name"], "_grid_keys": keys})
    names = [e["name"] for e in out]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"Duplicate experiment names: {sorted(dupes)}")
    return out


def _summary_row(name, cfg, period, a, res, report, engine) -> dict:
    row = {
        "name": name, "strategy": cfg["strategy"], "period": period, "config_hash": config_hash(cfg),
        "sharpe": a.perf.get("sharpe"), "annual_return": a.perf.get("annual_return"),
        "max_drawdown": a.perf.get("max_drawdown"), "trades": a.trades.get("trades", 0),
        "profit_factor": a.trades.get("profit_factor"), "win_rate": a.trades.get("win_rate"),
        "expectancy_r": a.trades.get("expectancy_r"), "sharpe_2x_costs": a.sharpe_2x,
        "alpha_annual": a.alpha.get("alpha_annual"), "alpha_t": a.alpha.get("alpha_t"), "beta": a.alpha.get("beta"),
        "dsr": a.dsr.get("dsr"), "passed": a.passed,
        "failed_checks": "; ".join(c.name for c in a.checks if c.passed is False),
        "report": str(report),
    }
    fills = getattr(engine, "fill_sources", None)
    if fills:
        row.update({f"fills_{k}": v for k, v in fills.items()})
    if engine is not None and hasattr(engine, "ambiguous"):
        row["ambiguous_minutes"] = len(engine.ambiguous)
    if engine is not None and hasattr(engine, "missing_total"):
        row["missing_minute_days"] = engine.missing_total  # candidates skipped for lack of minute data
    return row


def cmd_research(args) -> None:
    from .cli import Runner, _code_version, _on_github

    spec = yaml.safe_load(Path(args.file).read_text())
    exps = expand(spec)
    stem = _slug(spec.get("name") or Path(args.file).stem)
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    table_path = RESEARCH_DIR / f"{stem}.csv"
    done = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame(columns=["name", "config_hash", "period"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.max_minutes * 60
    runners: dict[tuple, Runner] = {}
    rows, skipped, stopped = [], 0, False

    for k, e in enumerate(exps, 1):
        cfg = load_config(e["config"])
        cfg = override(cfg, {key: val for key, val in e["set"].items()}) if e["set"] else cfg
        period = e.get("period", "tune")
        if period not in ("tune", "validate"):
            sys.exit(f"{e['name']}: research batches run tune or validate only; the holdout is scored separately.")
        h = config_hash(cfg)
        if not args.force and ((done["name"] == e["name"]) & (done["config_hash"] == h)
                               & (done["period"] == period)).any():
            skipped += 1
            continue
        if time.monotonic() > deadline:
            stopped = True
            break
        key = ("universe",) if cfg["strategy"] in ("orb", "gapgo", "dip") else ("single", cfg["symbol"])
        if key not in runners:
            runners[key] = Runner(cfg, synthetic=args.synthetic)
        runner = runners[key]
        start, end = period_bounds(cfg, period)
        t0 = time.monotonic()
        res = runner.run(cfg, start, end)
        engine = getattr(runner, "last_engine", None)
        run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + h[:6]
        record_trial(run_id, cfg, period, res.daily, label=f"research:{stem}:{e['name']}", synthetic=args.synthetic)

        a = analyze(res, period, sweep=None)  # the plateau check is read off the batch table instead
        report_stem = f"{run_id}_{_slug(e['name'])}_{period}" + ("_SYNTHETIC" if args.synthetic else "")
        report = build_report(res, a, cfg, period, run_id, out_dir / f"{report_stem}.html",
                              day_lookup=runner.day_lookup, synthetic=args.synthetic, code_version=_code_version())
        if engine is not None and getattr(engine, "ambiguous", None) and not args.synthetic:
            AMBIGUOUS_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(engine.ambiguous, columns=["date", "symbol", "bar"]).to_parquet(
                AMBIGUOUS_DIR / f"{stem}__{_slug(e['name'])}__{period}.parquet", index=False)
        row = _summary_row(e["name"], cfg, period, a, res, report.name, engine)
        row["minutes"] = round((time.monotonic() - t0) / 60, 1)
        rows.append(row)
        line = (f"[{k}/{len(exps)}] {e['name']} ({period}): Sharpe {row['sharpe']:.2f}, "
                f"{row['annual_return']:.1%}/yr, DD {row['max_drawdown']:.1%}, {row['trades']:,} trades"
                + (" PASS" if row["passed"] else ""))
        print(line, flush=True)
        if not args.synthetic:
            table = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            table.to_csv(table_path, index=False)

    table = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    if args.synthetic:
        table_path = out_dir / f"research_{stem}_SYNTHETIC.csv"
    table.to_csv(table_path, index=False)
    table.to_csv(out_dir / f"research_{stem}.csv", index=False)
    msg = f"{len(rows)} run(s) this time, {skipped} already done" + (", stopped at the time budget" if stopped else "")
    print(msg)
    _step_summary(stem, table, msg)
    if stopped and _on_github():
        print(f"::warning title=Research batch incomplete::{msg}. Run it again to continue.")


def _step_summary(stem: str, table: pd.DataFrame, msg: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path or table.empty:
        return
    lines = [f"## Research batch: {stem}", "", msg, "",
             "| Experiment | Period | Sharpe | Annual | Max DD | Trades | Pass |", "|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r['name']} | {r['period']} | {r['sharpe']:.2f} | {r['annual_return']:.1%} | "
                     f"{r['max_drawdown']:.1%} | {int(r['trades']):,} | {'✓' if r['passed'] else '✕'} |")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def cmd_ticks(args) -> None:
    from .cli import _on_github
    from .data.alpaca import AlpacaBars
    from .data.ticks import sync_ticks
    from .data.universe import Budget

    cfg = load_config(args.config)
    files = sorted(AMBIGUOUS_DIR.glob(f"*__{args.period}.parquet")) if args.period else sorted(AMBIGUOUS_DIR.glob("*.parquet"))
    if not files:
        sys.exit("No ambiguous-minute lists in runs/ambiguous/. Run a stocks-in-play research batch first.")
    needed = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).drop_duplicates()
    needed["date"] = pd.to_datetime(needed["date"]).dt.date
    triples = list(zip(needed["date"], needed["symbol"].astype(str), needed["bar"].astype(int)))
    client = AlpacaBars(requests_per_minute=args.rpm)
    folder = Path(cfg["data"]["cache_dir"]) / "universe" / "ticks"
    status = sync_ticks(client, triples, folder, Budget(args.max_minutes))
    print(json.dumps(status))
    line = (f"Tick minutes: {status['needed']:,} needed, {status['fetched']:,} fetched this run, "
            f"{status['remaining']:,} remaining")
    print(f"::notice title=Tick download::{line}" if _on_github() else line)
    if not status["complete"]:
        msg = "Tick download stopped at its time budget; run it again to continue."
        print(f"::warning title=Tick download incomplete::{msg}" if _on_github() else msg)
