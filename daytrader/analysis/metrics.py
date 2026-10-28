"""Performance, trade and market-relative metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe(r: pd.Series | np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def drawdown(r: pd.Series) -> pd.Series:
    eq = (1 + r).cumprod()
    return eq / eq.cummax() - 1.0


def longest_underwater(r: pd.Series) -> int:
    dd = drawdown(r).to_numpy()
    longest = cur = 0
    for x in dd:
        cur = cur + 1 if x < 0 else 0
        longest = max(longest, cur)
    return int(longest)


def performance(r: pd.Series) -> dict:
    r = r.dropna()
    n = len(r)
    if n == 0:
        return {}
    growth = float((1 + r).prod())
    years = n / TRADING_DAYS
    ann = growth ** (1 / years) - 1 if years > 0 and growth > 0 else float("nan")
    downside = r[r < 0].std(ddof=1)
    mdd = float(drawdown(r).min())
    return {
        "days": n, "total_return": growth - 1, "annual_return": ann,
        "annual_vol": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)), "sharpe": sharpe(r),
        "sortino": float(r.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and downside > 0 else float("nan"),
        "max_drawdown": mdd, "calmar": ann / abs(mdd) if mdd < 0 else float("nan"),
        "longest_underwater_days": longest_underwater(r),
        "pct_up_days": float((r > 0).mean()), "pct_flat_days": float((r == 0).mean()),
        "best_day": float(r.max()), "worst_day": float(r.min()),
    }


def trade_stats(trades: pd.DataFrame, n_days: int) -> dict:
    if trades is None or trades.empty:
        return {"trades": 0, "trades_per_day": 0.0}
    pnl = trades["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_loss = abs(losses.sum())
    r = trades["r_multiple"]
    return {
        "trades": int(len(trades)), "trades_per_day": len(trades) / max(n_days, 1),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf"),
        "avg_win_ret": float(trades.loc[pnl > 0, "ret"].mean()) if len(wins) else float("nan"),
        "avg_loss_ret": float(trades.loc[pnl <= 0, "ret"].mean()) if len(losses) else float("nan"),
        "avg_trade_ret": float(trades["ret"].mean()),
        "expectancy_r": float(r.mean()), "median_r": float(r.median()),
        "long_share": float((trades["side"] > 0).mean()),
    }


def yearly_returns(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.year).prod() - 1


def rolling_sharpe(r: pd.Series, window: int = 126) -> pd.Series:
    m = r.rolling(window).mean()
    s = r.rolling(window).std(ddof=1)
    return (m / s * np.sqrt(TRADING_DAYS)).dropna()


def newey_west_se(x: np.ndarray, resid: np.ndarray, lags: int = 5) -> np.ndarray:
    """HAC standard errors for OLS coefficients (robust to autocorrelation)."""
    n, k = x.shape
    xtx_inv = np.linalg.inv(x.T @ x)
    u = x * resid[:, None]
    s = u.T @ u
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        g = u[lag:].T @ u[:-lag]
        s += w * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv
    return np.sqrt(np.diag(cov))


def alpha_beta(r: pd.Series, mkt: pd.Series) -> dict:
    df = pd.concat([r, mkt], axis=1).dropna()
    if len(df) < 30:
        return {}
    y = df.iloc[:, 0].to_numpy()
    x = np.column_stack([np.ones(len(df)), df.iloc[:, 1].to_numpy()])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    se = newey_west_se(x, resid)
    return {
        "alpha_daily": float(coef[0]), "alpha_annual": float(coef[0] * TRADING_DAYS),
        "alpha_t": float(coef[0] / se[0]) if se[0] > 0 else float("nan"),
        "beta": float(coef[1]), "beta_t": float(coef[1] / se[1]) if se[1] > 0 else float("nan"),
        "correlation": float(np.corrcoef(y, x[:, 1])[0, 1]),
    }


def conditional(daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Average strategy vs market day, grouped by market condition."""
    d = daily.copy()
    out = {}
    agg = {"strategy": ("ret", "mean"), "market": ("market_ret", "mean"), "days": ("ret", "size"),
           "strategy_hit": ("ret", lambda x: float((x > 0).mean()))}
    out["market_day"] = d.groupby("market_day").agg(**agg).reindex(["up", "flat", "down"])
    # Volatility regime from the PRIOR 14 days of market returns (known before the open)
    vol = d["market_ret"].rolling(14).std().shift(1)
    if vol.notna().sum() > 30:
        d["vol_regime"] = pd.qcut(vol, 3, labels=["low vol", "mid vol", "high vol"])
        out["vol_regime"] = d.dropna(subset=["vol_regime"]).groupby("vol_regime", observed=True).agg(**agg)
    out["weekday"] = d.groupby("weekday").agg(**agg).reindex(["Mon", "Tue", "Wed", "Thu", "Fri"])
    return out


def cost_adjusted_returns(daily: pd.DataFrame, trades: pd.DataFrame, extra_per_share: float) -> pd.Series:
    """Daily returns if every fill cost `extra_per_share` more (both sides of every trade)."""
    if trades.empty or extra_per_share == 0:
        return daily["ret"]
    extra = (2 * extra_per_share * trades["shares"]).groupby(trades["date"]).sum()
    extra = extra.reindex(daily.index, fill_value=0.0)
    pnl = daily["equity"] - daily["equity_start"] - extra
    return pnl / daily["equity_start"]


def cost_curve(daily: pd.DataFrame, trades: pd.DataFrame, max_cents: float = 5.0, steps: int = 51) -> pd.DataFrame:
    rows = []
    for extra in np.linspace(0, max_cents / 100, steps):
        rows.append({"extra_slippage": extra, "sharpe": sharpe(cost_adjusted_returns(daily, trades, extra))})
    return pd.DataFrame(rows)


def break_even_slippage(curve: pd.DataFrame, base_slippage: float) -> float:
    below = curve[curve["sharpe"] <= 0]
    if below.empty:
        return float("inf")
    return float(below["extra_slippage"].iloc[0] + base_slippage)
