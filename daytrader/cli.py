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


UNIVERSE_STRATEGIES = ("orb", "gapgo", "dip")


class Runner:
    """Loads a strategy's data once, then runs any config variant over any date range."""

    def __init__(self, cfg: dict, synthetic: bool):
        self.kind = cfg["strategy"]
        if self.kind in UNIVERSE_STRATEGIES:
            from .data.orb_data import CachedOrbData, SyntheticOrbData
            self.data = (SyntheticOrbData(n_days=1600, n_symbols=60, seed=7, momentum=0.12, start=dt.date(2016, 1, 4))
                         if synthetic else CachedOrbData(cfg["data"]["cache_dir"]))
        else:
            if synthetic:
                self.days = make_days(n_days=2100, start=dt.date(2016, 1, 4), seed=7, momentum=0.03)
            else:
                from .data.store import load_days
                start = dt.date.fromisoformat(str(cfg["data"]["start"]))
                end = cfg["data"].get("end")
                self.days = load_days(cfg["symbol"], start, dt.date.fromisoformat(str(end)) if end else None,
                                      cfg["data"]["cache_dir"])
            self._by_date = {d.date: d for d in self.days}
        self.synthetic = synthetic

    def run(self, cfg: dict, start, end):
        if cfg["strategy"] in UNIVERSE_STRATEGIES:
            from .backtest.dip import DipEngine
            from .backtest.orb import OrbEngine
            if not self.synthetic:
                extra = cfg["strategy"] != "orb"
                missing = [d for d in self.data.dates if (not start or d >= start) and (not end or d <= end)
                           and not self.data.has_minutes(d, extra=extra)]
                if missing:
                    sys.exit(f"Minute data is missing for {len(missing)} days ({missing[0]} to {missing[-1]}). "
                             "Run the stocks-in-play download again; it resumes where it stopped.")
            self.last_engine = DipEngine(cfg) if cfg["strategy"] == "dip" else OrbEngine(cfg)
            return self.last_engine.run(self.data, start, end)
        self.last_engine = Engine(cfg, build_strategy(cfg))
        return self.last_engine.run(self.days, start, end)

    def day_lookup(self, date, symbol=None):
        if self.kind in UNIVERSE_STRATEGIES:
            return self.data.minute_day(date, symbol)
        return self._by_date.get(date)


def _on_github() -> bool:
    import os
    return bool(os.environ.get("GITHUB_ACTIONS"))


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
    if cfg["strategy"] == "orb":
        from .data.universe import sync_all
        client = AlpacaBars(requests_per_minute=args.rpm)
        status = sync_all(client, cfg, args.max_minutes)
        print(json.dumps(status, indent=2))
        root = Path(cfg["data"]["cache_dir"]) / "universe"
        counts = {k: len(list((root / k).glob("*.parquet"))) for k in ("daily", "opening", "minute", "minute_extra")}
        stages = ", ".join(k for k in ("assets", "daily", "opening", "features", "minute", "extra") if status.get(k))
        line = (f"Stages done: {stages or 'none'} | files: daily {counts['daily']}, opening {counts['opening']} months, "
                f"minute {counts['minute']} months, extra {counts['minute_extra']} months | through {status.get('through')}")
        print(f"::notice title=Download progress::{line}" if _on_github() else line)
        if not status["complete"]:
            msg = "Download stopped at its time budget; run it again to continue (cached progress is kept)."
            print(f"::warning title=Download incomplete::{msg}" if _on_github() else msg)
        return
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
    runner = Runner(cfg, args.synthetic)
    start, end = period_bounds(cfg, period)
    res = runner.run(cfg, start, end)
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
                          day_lookup=runner.day_lookup, sweep=sweep, synthetic=args.synthetic,
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
    runner = Runner(cfg0, args.synthetic)
    start, end = period_bounds(cfg0, "tune")
    rows = []
    combos = list(itertools.product(*grid.values()))
    for k, combo in enumerate(combos, 1):
        params = dict(zip(grid.keys(), combo))
        cfg = override(cfg0, {f"params.{p}": v for p, v in params.items()})
        res = runner.run(cfg, start, end)
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
    d.add_argument("--max-minutes", type=float, default=300, help="stop cleanly after this long (resumable)")
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

    from .research import cmd_research, cmd_ticks
    r = sub.add_parser("research", help="run a batch of pre-registered experiments")
    r.add_argument("--file", required=True)
    r.add_argument("--synthetic", action="store_true")
    r.add_argument("--force", action="store_true", help="re-run experiments already in the results table")
    r.add_argument("--max-minutes", type=float, default=320)
    r.add_argument("--out", default="reports")
    r.set_defaults(func=cmd_research)
    t = sub.add_parser("ticks", help="download trades for the ambiguous stocks-in-play entry minutes")
    t.add_argument("--config", required=True)
    t.add_argument("--period", default="", help="only minutes saved from runs on this period (e.g. tune)")
    t.add_argument("--rpm", type=int, default=120)
    t.add_argument("--max-minutes", type=float, default=300)
    t.set_defaults(func=cmd_ticks)

    args = ap.parse_args(argv)
    try:
        args.func(args)
    except (Exception, SystemExit) as exc:
        if isinstance(exc, SystemExit) and exc.code in (None, 0):
            raise
        if _on_github():
            # Shows on the run page (and is readable through the API) even when logs aren't.
            text = str(exc.code) if isinstance(exc, SystemExit) else f"{type(exc).__name__}: {exc}"
            msg = text.replace("%", "%25").replace("\n", "%0A")
            print(f"::error title=daytrader {args.cmd} failed::{msg}")
        raise


if __name__ == "__main__":
    main()
