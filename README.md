# daytrader-bot

Backtest-first intraday trading research. Phase 1 is the backtester: replay every
historical day minute by minute, exactly as the bot would have lived it, and grade
the result against a fixed scorecard before any real money is involved.

Research tool, not investment advice.

## What is here (phase 1)

| Strategy | Source | Status |
|---|---|---|
| SPY "noise area" intraday momentum | Zarattini, Aziz & Barbon (2024) | built |
| Opening-range breakout on stocks in play | Zarattini, Barbon & Aziz (2024) | next |

* **No lookahead by construction.** Strategies only ever receive `History` (completed
  days) and `TodayView` (today's bars up to the current minute). Market data is
  read-only. A decision made on a bar's close fills at a later bar's open.
* **Leak hunters run before every result.** `tests/test_lookahead.py` corrupts all
  data after a moment in time and checks nothing before it changed, and checks that
  a run cut short matches a longer run day for day.
* **Every trading day counts.** The history is split by time into *tune* (2016–2021),
  *validate* (2022–2023) and *holdout* (2024–now). Sweeps only run on tune; the
  holdout needs `--unlock-holdout` and every look at it is counted.
* **Every variant is logged** in `runs/trials.jsonl` (committed), so the deflated
  Sharpe ratio and overfitting probability can discount luck.
* **One HTML report per run**: verdict and scorecard, equity vs SPY, bootstrap and
  deflated Sharpe, Monte Carlo drawdowns, performance by market day and volatility,
  year by year, parameter heatmap, slippage break-even, trade-level charts.

## Running it from your phone (GitHub Actions)

One-time setup, in the repo's **Settings → Secrets and variables → Actions**:

| Secret | Value |
|---|---|
| `ALPACA_API_KEY_ID` | Alpaca API key (paper or live keys both work for market data) |
| `ALPACA_API_SECRET_KEY` | Alpaca secret |
| `SMTP_USER` *(optional)* | Gmail address that sends the report |
| `SMTP_PASSWORD` *(optional)* | Gmail app password for that address |
| `REPORT_EMAIL_TO` *(optional)* | Where to send reports |

Then in the GitHub app: **Actions → Backtest → Run workflow**, pick a command and period.

* `synthetic-check`: runs everything on fake data. Use this first to confirm the pipeline works.
* `backtest`: downloads any new data, runs the backtest, emails the report (if set up),
  shows a pass/fail summary on the run page, and attaches the report as an artifact.
* `sweep`: tries a grid of settings on the tune period (fills in the heatmap and plateau check).
* `download-only`: refresh the data cache.

The first real run downloads about ten years of SPY minute bars (a few minutes);
after that the data is cached and only new months are fetched. Downloads throttle
themselves to 120 requests a minute because Alpaca's free limit (200) is shared by
every key on the same login.

## Running it on a computer

```bash
pip install -e ".[dev]"
pytest -q
python -m daytrader backtest --config config/spy_noise.yaml --synthetic          # no data needed
export ALPACA_API_KEY_ID=... ALPACA_API_SECRET_KEY=...
python -m daytrader download --config config/spy_noise.yaml
python -m daytrader backtest --config config/spy_noise.yaml --period tune
python -m daytrader backtest --config config/spy_noise.yaml --period tune --long-only
python -m daytrader sweep --config config/spy_noise.yaml --period tune \
  --grid lookback_days=10,12,14,16,18 --grid vol_multiplier=0.8,0.9,1.0,1.1,1.2
python -m daytrader backtest --config config/spy_noise.yaml --period validate
```

Reports land in `reports/`.

## Workflow

1. `synthetic-check` once, to prove the pipeline.
2. `backtest` on **tune**. Compare with the paper (Sharpe 1.33 over 2007–2024; ours starts in 2016).
3. Run it again with **long-only**, in case Robinhood can't short.
4. `sweep` on tune to check the defaults sit on a plateau, not a spike.
5. `backtest` on **validate**. If it fails, going back to tuning is allowed but every trip is logged.
6. Only at the very end: `backtest` on **holdout** with *unlock holdout* ticked. Once.

## Layout

```
config/spy_noise.yaml        strategy, periods, costs, sizing
daytrader/data/              Alpaca download + cache, point-in-time views, synthetic data
daytrader/strategies/        signal code shared by backtest and (later) live trading
daytrader/backtest/engine.py minute-by-minute replay, fills, costs
daytrader/analysis/          metrics, statistics, scorecard, trial log
daytrader/report/html.py     the HTML report
tests/                       lookahead leak hunters and unit tests
runs/                        trial log (committed)
```
