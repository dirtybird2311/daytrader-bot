"""Turn a backtest result into everything the report shows, including the scorecard."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult
from . import metrics as M
from . import stats as S
from .trials import neighbours_positive, trial_matrix, trial_sharpes


@dataclass
class Check:
    name: str
    value: float | None
    display: str
    needs: str
    passed: bool | None  # None = not run yet


@dataclass
class Analysis:
    perf: dict
    market_perf: dict
    trades: dict
    alpha: dict
    mean_test: dict
    boot: dict
    boot_ret: dict
    dsr: dict
    pbo: dict
    mc: dict
    yearly: pd.DataFrame
    rolling: pd.Series
    conditional: dict
    cost_curve: pd.DataFrame
    break_even: float
    sharpe_2x: float
    checks: list[Check]
    worst_days: pd.DataFrame
    extra: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        ran = [c for c in self.checks if c.passed is not None]
        return bool(ran) and all(c.passed for c in ran)


def analyze(res: BacktestResult, period: str, sweep: pd.DataFrame | None = None,
            runs_dir: Path = Path("runs")) -> Analysis:
    cfg = res.config
    daily, trades = res.daily, res.trades
    r = daily["ret"]
    mkt = daily["market_ret"]
    perf = M.performance(r)
    market_perf = M.performance(mkt)
    tstats = M.trade_stats(trades, len(daily))
    comm = float(cfg["costs"]["commission_per_share"])
    slip = float(cfg["costs"]["slippage_per_share"])
    curve = M.cost_curve(daily, trades)
    sharpe_2x = M.sharpe(M.cost_adjusted_returns(daily, trades, comm + slip))
    trailing = float((1 + r.iloc[-252:]).prod() - 1) if len(r) >= 60 else float("nan")

    plateau = float("nan")
    if sweep is not None and not sweep.empty:
        chosen = {k: cfg["params"].get(k) for k in sweep.columns if k in cfg["params"]}
        plateau = neighbours_positive(sweep, chosen)

    pf = tstats.get("profit_factor", float("nan"))
    checks = [
        Check("Closed trades", tstats.get("trades", 0), f"{tstats.get('trades', 0):,}", "≥ 100",
              tstats.get("trades", 0) >= 100),
        Check("Sharpe ratio, net of costs", perf.get("sharpe"), _f(perf.get("sharpe")), "> 1.0",
              _ok(perf.get("sharpe"), lambda v: v > 1.0)),
        Check("Profit factor", pf, _f(pf), "> 1.3", _ok(pf, lambda v: v > 1.3)),
        Check("Max drawdown", perf.get("max_drawdown"), _pct(perf.get("max_drawdown")), "≤ 25%",
              _ok(perf.get("max_drawdown"), lambda v: v >= -0.25)),
        Check("Beats SPY, risk-adjusted (Sharpe gap)", _sub(perf.get("sharpe"), market_perf.get("sharpe")),
              _f(_sub(perf.get("sharpe"), market_perf.get("sharpe")), signed=True), "> 0",
              _ok(_sub(perf.get("sharpe"), market_perf.get("sharpe")), lambda v: v > 0)),
        Check("Sharpe with costs doubled", sharpe_2x, _f(sharpe_2x), "> 0.5", _ok(sharpe_2x, lambda v: v > 0.5)),
        Check("Positive trailing year", trailing, _pct(trailing), "> 0%", _ok(trailing, lambda v: v > 0)),
        Check("Neighbouring settings also profitable", plateau,
              "not run yet" if np.isnan(plateau) else _pct(plateau, signed=False), "≥ 50%",
              None if np.isnan(plateau) else plateau >= 0.5),
    ]

    worst = daily.nsmallest(5, "ret")[["ret", "market_ret", "n_trades"]]
    return Analysis(
        perf=perf, market_perf=market_perf, trades=tstats, alpha=M.alpha_beta(r, mkt), mean_test=S.mean_test(r),
        boot=S.bootstrap_sharpe(r), boot_ret=S.bootstrap_annual_return(r),
        dsr=S.deflated_sharpe(r, trial_sharpes(cfg["strategy"], period, runs_dir)),
        pbo=S.pbo_cscv(trial_matrix(cfg["strategy"], period, runs_dir)), mc=S.monte_carlo(r),
        yearly=pd.DataFrame({"strategy": M.yearly_returns(r), "market": M.yearly_returns(mkt)}),
        rolling=M.rolling_sharpe(r), conditional=M.conditional(daily), cost_curve=curve,
        break_even=M.break_even_slippage(curve, slip), sharpe_2x=sharpe_2x, checks=checks, worst_days=worst,
        extra={"trailing_year": trailing, "plateau": plateau},
    )


def _ok(v, fn) -> bool | None:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return False
    return bool(fn(v))


def _sub(a, b):
    if a is None or b is None:
        return float("nan")
    return a - b


def _f(v, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a" if not (isinstance(v, float) and v == float("inf")) else "∞"
    return f"{v:+.2f}" if signed else f"{v:.2f}"


def _pct(v, signed: bool = True) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:+.1%}" if signed else f"{v:.0%}"
