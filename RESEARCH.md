# Research queue

Ranked list of strategy ideas to test, and the rules for testing them. Every idea is
written down here **before** it is run, so the trial log can't be gamed after the fact.

## Rules

1. **Pre-register.** Fill in the template below (rule, why it should work, who loses,
   default parameters, what would kill it) before writing any code.
2. **Tune only (2016–2021)** for building, sweeping and debugging. Kill fast if it can't
   clear the scorecard.
3. **Validate (2022–2023)** once, after the rules are frozen. Going back to tuning is
   allowed but every trip is logged.
4. **Batch the holdout.** Finalists are scored on 2024–now together, once, at the end.
5. **After the holdout is spent**, new ideas are judged on new data only: live trading
   at small size or months that have accumulated since.
6. **Every run is logged** in `runs/trials.jsonl`, including failures. The deflated Sharpe
   ratio counts them all.
7. **No LLM in backtests.** Models have memorized past news and prices; anything an LLM
   decides is judged on live data only.

## Status

| Strategy | Tune | Validate | Holdout |
|---|---|---|---|
| SPY noise-area momentum, long-only | pass (Sharpe 1.08) | pass (Sharpe 1.67) | waiting (batch) |
| SPY noise-area momentum, long + short | fail (Sharpe 0.93, DD 28%) | — | — |
| Stocks-in-play ORB | data downloading | — | — |

## Ranked queue

Ranked by expected value per unit of effort and statistical risk. Rank 1 goes next.

| # | Idea | Source / evidence | Effort | Overfitting risk | Notes |
|---|---|---|---|---|---|
| 1 | **Combined portfolio:** SPY momentum + stocks in play, capital split between them | Diversification; both strategies already built | Low | Low (no new parameters beyond the split, fix it at 50/50) | Expect a smoother equity curve if the two are loosely correlated. Test once both have passed validate. |
| 2 | **ORB opening-range length:** 1, 15, 30 minutes instead of 5 | Same paper tested 5/15/30/60; a QuantConnect user reported 1-minute did better | Low | Medium (sweep) | Needs minute data from 9:30, which we already download. Paper: 5-min best. |
| 3 | **ORB VWAP trailing exit** instead of fixed stop + close | Zarattini's VWAP paper; Humbled Trader's VWAP rules | Low | Medium | Could hold winners longer. One new rule, no new parameters. |
| 4 | **Volatility regime filter:** skip or halve size when prior 14-day volatility is low | Gao et al.: intraday momentum stronger on volatile days; SPY results weak in quiet 2016–17 | Low | **High**, easy to fit to the past | Decide the threshold from the tune period only, as a fixed percentile. |
| 5 | **First half hour predicts last half hour** (Gao, Han, Li & Zhou, JFE 2018) | Peer-reviewed, replicated across ETFs and 16 countries | Low | Low (almost no parameters) | Small edge alone (~6% a year gross). Possibly more useful as a filter for #1. |
| 6 | **Gap-and-go on S&P 500 gappers** (Humbled Trader's "Trend Join Long") | Practitioner setup, TradingView backtest only | Medium | Medium | A simpler, large-cap cousin of stocks in play. Long only, Robinhood-friendly. |
| 7 | **VWAP-band mean reversion**: fade moves past ±2 standard deviations back to VWAP | Practitioner technique; weak published evidence | Medium | High | Different kind of strategy that may earn on choppy days when momentum loses. |
| 8 | **ORB on a wider universe** (top 50 / 100 by relative volume) | QuantConnect variations | Medium (more data) | Medium | Paper finds the edge concentrates in the top names; expect it to fade. |
| — | **LLM news filter** on stocks in play | Lopez-Lira & Tang; memorization research | Medium | Can't backtest honestly | **Live only.** Log verdicts first; turn on only if vetoed trades measurably underperform. |

## Pre-registration template

```
### <name>
Rule:              exact entry, exit, sizing
Why it should work: the behaviour it exploits
Who loses:         the other side of the trade
Defaults:          parameter values, fixed before any test
Sweep (tune only): which parameters, which grid
Kill criteria:     what result ends it (e.g. scorecard fail on tune, Sharpe < 0.5 at 2x costs)
Registered:        date
```

## Registered

*(none yet beyond the two built strategies)*
