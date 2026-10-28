"""Command line: download data, run a backtest, sweep parameters.

  python -m daytrader download --config config/spy_noise.yaml
  python -m daytrader backtest --config config/spy_noise.yaml --period tune
  python -m daytrader sweep    --config config/spy_noise.yaml --period tune \
         --grid lookback_days=10,12,14,16,18 --grid vol_multiplier=0.8,0.9,1.0,1.1,1.2
  python -m daytrader backtest --config config/spy_noise.yaml --synthetic   # no data needed
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .analysis.analyze import analyze
from .analysis.metrics import performance
from .analysis.trials import config_hash, holdout_views, record_trial
from .backtest.engine import Engine
from .config import load_config, override, period_bounds
from .data.synthetic import make_days
from .report.html import build_report
from .strategies import build_strategy


def _days(cfg: dict, synthetic: bool):
    if synthetic:
        return make_days(n_days=2100, start=dt.date(2016, 1, 4), seed=7, momentum=0.03)
    from .data.store import load_days
    start = dt.date.fromisoformat(str(cfg["data"]["start"]))
    end = cfg["data"].get("end")
    return load_days(cfg["symbol"], start, dt.date.fromisoformat(str(end)) if end else None, cfg["data"]["cache_dir"])


def _code_version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def _apply_flags(cfg: dict, args) -> dict:
    if getattr(args, "long_only", False):
        cfg = override(cfg, {"params.allow_short": False})
    for item in getattr(args, "set", None) or []:
        key, value = item.split("=", 1)
        cfg = override(cfg, {key: json.loads(value) if value[:1] in "0123456789-[{tf" else value})
    return cfg


def cmd_download(args) -> None:
    from .data.alpaca import AlpacaBars, sync_symbol
    cfg = load_config(args.config)
    start = dt.date.fromisoformat(str(cfg["data"]["start"]))
    # Pull a month of extra history so the first test days have their 14-day lookback.
    start = (start.replace(day=1) - dt.timedelta(days=45)).replace(day=1)
    end = dt.date.today()
    client = AlpacaBars(requests_per_minute=args.rpm)
    print(f"Syncing {cfg['symbol']} {start} to {end} into {cfg['data']['cache_dir']}/")
    n = sync_symbol(client, cfg["symbol"], start, end, cfg["data"]["cache_dir"])
    print(f"Done: {n} month file(s) downloaded or refreshed.")


def cmd_backtest(args) -> None:
    cfg = _apply_flags(load_config(args.config), args)
    period = args.period
    if period == "holdout" and not args.unlock_holdout and not args.synthetic:
        sys.exit("The holdout is scored once, at the very end. Re-run with --unlock-holdout if that time has come.")
    days = _days(cfg, args.synthetic)
    start, end = period_bounds(cfg, period)
    res = Engine(cfg, build_strategy(cfg)).run(days, start, end)
    if res.daily.empty:
        sys.exit(f"No trading days in {period} ({start} to {end or 'now'}). Is the data downloaded?")

    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + config_hash(cfg)[:6]
    record_trial(run_id, cfg, period, res.daily, label=args.label or "", synthetic=args.synthetic)
    sweep_path = Path("runs") / f"sweep_{cfg['strategy']}_tune.csv"
    sweep = pd.read_csv(sweep_path) if sweep_path.exists() and not args.synthetic else None
    a = analyze(res, period, sweep=sweep)

    out_dir = Path(args.out)
    stem = f"{run_id}_{cfg['strategy']}_{period}" + ("_SYNTHETIC" if args.synthetic else "")
    report = build_report(res, a, cfg, period, run_id, out_dir / f"{stem}.html",
                          days_by_date={d.date: d for d in days}, sweep=sweep, synthetic=args.synthetic,
                          holdout_views=holdout_views(cfg["strategy"]) if period == "holdout" else 0,
                          code_version=_code_version())
    res.trades.to_csv(out_dir / f"{stem}_trades.csv", index=False)
    res.daily.to_csv(out_dir / f"{stem}_daily.csv")
    summary = {
        "run_id": run_id, "period": period, "passed": a.passed, "synthetic": args.synthetic,
        "sharpe": round(a.perf.get("sharpe", float("nan")), 3),
        "annual_return": round(a.perf.get("annual_return", float("nan")), 4),
        "max_drawdown": round(a.perf.get("max_drawdown", float("nan")), 4),
        "trades": a.trades.get("trades", 0),
        "checks": {c.name: c.passed for c in a.checks}, "report": str(report),
    }
    print(json.dumps(summary, indent=2))
    with open(out_dir / "latest_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)


def cmd_sweep(args) -> None:
    cfg0 = _apply_flags(load_config(args.config), args)
    if args.period != "tune":
        sys.exit("Sweeps run on the tune period only. Tuning on later periods leaks them into the choice.")
    grid = {}
    for g in args.grid:
        key, values = g.split("=", 1)
        grid[key] = [json.loads(v) for v in values.split(",")]
    days = _days(cfg0, args.synthetic)
    start, end = period_bounds(cfg0, "tune")
    rows = []
    combos = list(itertools.product(*grid.values()))
    for k, combo in enumerate(combos, 1):
        params = dict(zip(grid.keys(), combo))
        cfg = override(cfg0, {f"params.{p}": v for p, v in params.items()})
        res = Engine(cfg, build_strategy(cfg)).run(days, start, end)
        perf = performance(res.daily["ret"])
        run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-s{k:03d}-" + config_hash(cfg)[:6]
        record_trial(run_id, cfg, "tune", res.daily, label="sweep", synthetic=args.synthetic)
        rows.append({**params, "sharpe": perf.get("sharpe"), "annual_return": perf.get("annual_return"),
                     "total_return": perf.get("total_return")})
        print(f"[{k}/{len(combos)}] {params} sharpe={perf.get('sharpe'):.2f}")
    out = pd.DataFrame(rows)
    folder = Path(args.out) if args.synthetic else Path("runs")  # synthetic sweeps never enter the trial log
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"sweep_{cfg0['strategy']}_tune{'_SYNTHETIC' if args.synthetic else ''}.csv"
    out.to_csv(path, index=False)
    print(f"Saved {path}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="daytrader")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="sync 1-minute bars from Alpaca into the cache")
    d.add_argument("--config", required=True)
    d.add_argument("--rpm", type=int, default=120, help="requests per minute (Alpaca free plan allows 200, shared per login)")
    d.set_defaults(func=cmd_download)

    for name, fn in (("backtest", cmd_backtest), ("sweep", cmd_sweep)):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--period", default="tune", choices=["tune", "validate", "holdout", "all"])
        p.add_argument("--synthetic", action="store_true", help="use made-up data (testing the code only)")
        p.add_argument("--long-only", action="store_true", help="disable shorting")
        p.add_argument("--set", action="append", help="override a config value, e.g. --set costs.slippage_per_share=0.002")
        p.add_argument("--out", default="reports")
        p.add_argument("--label", default="")
        if name == "backtest":
            p.add_argument("--unlock-holdout", action="store_true")
        else:
            p.add_argument("--grid", action="append", required=True, help="param=v1,v2,...")
        p.set_defaults(func=fn)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
