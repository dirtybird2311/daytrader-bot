"""Is the edge real? Significance, resampling and overfitting checks.

* Block bootstrap of daily returns keeps streaks intact when estimating ranges.
* Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014) discounts the best
  result for how many variants were tried.
* Probability of backtest overfitting via combinatorially symmetric
  cross-validation (Bailey, Borwein, Lopez de Prado & Zhu, 2017).
* Monte Carlo reshuffles of daily returns for drawdown and streak ranges.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats as st

from .metrics import TRADING_DAYS, newey_west_se

EULER_GAMMA = 0.5772156649


def mean_test(r: pd.Series) -> dict:
    """t-test that the mean daily return is above zero, with autocorrelation-robust errors."""
    y = r.dropna().to_numpy()
    if len(y) < 30:
        return {}
    x = np.ones((len(y), 1))
    se = newey_west_se(x, y - y.mean())[0]
    t = y.mean() / se if se > 0 else float("nan")
    return {"t_stat": float(t), "p_value": float(1 - st.norm.cdf(t))}


def block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Stationary bootstrap (Politis & Romano): random-length blocks, mean length `block`."""
    idx = np.empty(n, dtype=int)
    i = 0
    while i < n:
        start = rng.integers(0, n)
        length = rng.geometric(1 / block)
        take = min(length, n - i)
        idx[i:i + take] = (start + np.arange(take)) % n
        i += take
    return idx


def bootstrap_sharpe(r: pd.Series, n_boot: int = 2000, block: int = 10, seed: int = 0) -> dict:
    y = r.dropna().to_numpy()
    if len(y) < 30:
        return {}
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        s = y[block_bootstrap_indices(len(y), block, rng)]
        sd = s.std(ddof=1)
        out[b] = s.mean() / sd * math.sqrt(TRADING_DAYS) if sd > 0 else 0.0
    return {"samples": out, "lo": float(np.percentile(out, 2.5)), "hi": float(np.percentile(out, 97.5)),
            "p_below_1": float((out < 1.0).mean()), "p_below_0": float((out < 0.0).mean())}


def bootstrap_annual_return(r: pd.Series, n_boot: int = 2000, block: int = 10, seed: int = 1) -> dict:
    y = r.dropna().to_numpy()
    if len(y) < 30:
        return {}
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        s = y[block_bootstrap_indices(len(y), block, rng)]
        out[b] = np.prod(1 + s) ** (TRADING_DAYS / len(s)) - 1
    return {"lo": float(np.percentile(out, 2.5)), "hi": float(np.percentile(out, 97.5))}


def probabilistic_sharpe(r: pd.Series, benchmark_daily_sr: float = 0.0) -> float:
    """Probability the true (daily) Sharpe exceeds the benchmark, given skew and fat tails."""
    y = r.dropna().to_numpy()
    n = len(y)
    if n < 30 or y.std(ddof=1) == 0:
        return float("nan")
    sr = y.mean() / y.std(ddof=1)
    g3 = st.skew(y)
    g4 = st.kurtosis(y, fisher=False)
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(st.norm.cdf((sr - benchmark_daily_sr) * math.sqrt(n - 1) / denom))


def deflated_sharpe(r: pd.Series, trial_daily_srs: list[float]) -> dict:
    """Deflated Sharpe: PSR against the Sharpe you'd expect from the best of N luck-only trials."""
    n_trials = max(1, len(trial_daily_srs))
    if n_trials == 1:
        sr0 = 0.0
    else:
        var = float(np.var(trial_daily_srs, ddof=1))
        z1 = st.norm.ppf(1 - 1 / n_trials)
        z2 = st.norm.ppf(1 - 1 / (n_trials * math.e))
        sr0 = math.sqrt(var) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    return {"dsr": probabilistic_sharpe(r, sr0), "trials": n_trials,
            "expected_max_sr_annual": sr0 * math.sqrt(TRADING_DAYS)}


def pbo_cscv(matrix: pd.DataFrame, n_splits: int = 8) -> dict:
    """Probability of backtest overfitting.

    matrix: daily returns, one column per variant tried, aligned on dates.
    Splits time into n_splits blocks; for each half/half combination, picks the
    best variant in-sample and records its rank out-of-sample.
    """
    m = matrix.dropna()
    if m.shape[1] < 2 or len(m) < n_splits * 20:
        return {}
    blocks = np.array_split(np.arange(len(m)), n_splits)
    logits = []
    for combo in itertools.combinations(range(n_splits), n_splits // 2):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_splits) if i not in combo])
        is_sr = m.iloc[is_idx].mean() / m.iloc[is_idx].std(ddof=1)
        oos_sr = m.iloc[oos_idx].mean() / m.iloc[oos_idx].std(ddof=1)
        best = is_sr.idxmax()
        rank = (oos_sr.rank(pct=True))[best]
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(rank / (1 - rank)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "combinations": len(logits), "variants": int(m.shape[1])}


def monte_carlo(r: pd.Series, n_sims: int = 2000, seed: int = 2) -> dict:
    """Reshuffle the order of daily returns to see drawdowns and losing streaks history didn't produce."""
    y = r.dropna().to_numpy()
    if len(y) < 30:
        return {}
    rng = np.random.default_rng(seed)
    dds, streaks = np.empty(n_sims), np.empty(n_sims)
    for k in range(n_sims):
        s = rng.permutation(y)
        eq = np.cumprod(1 + s)
        dds[k] = (eq / np.maximum.accumulate(eq) - 1).min()
        streak = best = 0
        for v in s:
            streak = streak + 1 if v < 0 else 0
            best = max(best, streak)
        streaks[k] = best
    return {"max_dd_samples": dds, "dd_median": float(np.median(dds)), "dd_p95": float(np.percentile(dds, 5)),
            "dd_p99": float(np.percentile(dds, 1)), "streak_median": float(np.median(streaks)),
            "streak_p95": float(np.percentile(streaks, 95)),
            "worst_day_p99": float(np.percentile(y, 1))}
