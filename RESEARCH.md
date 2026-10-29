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
| Stocks-in-play ORB | tune: unresolved | Paper stop (0.1× ATR): Sharpe −9.46 if same-minute stops count, 1.24 if only close-confirmed ones do | Wide stops (1–2× ATR): Sharpe ~0.1 either way, no edge. Edge depends on intrabar order; needs tick data to settle |

## Ranked queue

Ranked by expected value per unit of effort and statistical risk. Rank 1 goes next.

| # | Idea | Source / evidence | Effort | Overfitting risk | Notes |
|---|---|---|---|---|---|
| 1 | **Combined portfolio:** SPY momentum + stocks in play, capital split between them | Diversification; both strategies already built | Low | Low (no new parameters beyond the split, fix it at 50/50) | Expect a smoother equity curve if the two are loosely correlated. Test once both have passed validate. |
| 2 | **Technical-analysis filters on stocks in play:** 20-day SMA trend, 14-day RSI, MACD 12/26/9, Bollinger 20/2 | Humbled Trader's indicator set; textbook settings; Brock et al. (1992) for MA rules | Low | Medium (four trials, standard settings, no tuning) | Pre-registered below. Each tested alone against the baseline, right after baseline ORB results. |
| 3 | **ORB opening-range length:** 1, 15, 30 minutes instead of 5 | Same paper tested 5/15/30/60; a QuantConnect user reported 1-minute did better | Low | Medium (sweep) | Needs minute data from 9:30, which we already download. Paper: 5-min best. |
| 4 | **ORB VWAP trailing exit** instead of fixed stop + close | Zarattini's VWAP paper; Humbled Trader's VWAP rules | Low | Medium | Could hold winners longer. One new rule, no new parameters. |
| 5 | **Volatility regime filter:** skip or halve size when prior 14-day volatility is low | Gao et al.: intraday momentum stronger on volatile days; SPY results weak in quiet 2016–17 | Low | **High**, easy to fit to the past | Decide the threshold from the tune period only, as a fixed percentile. |
| 6 | **First half hour predicts last half hour** (Gao, Han, Li & Zhou, JFE 2018) | Peer-reviewed, replicated across ETFs and 16 countries | Low | Low (almost no parameters) | Small edge alone (~6% a year gross). Possibly more useful as a filter for #1. |
| 7 | **Gap-and-go on S&P 500 gappers** (Humbled Trader's "Trend Join Long") | Practitioner setup, TradingView backtest only | Medium | Medium | A simpler, large-cap cousin of stocks in play. Long only, Robinhood-friendly. |
| 8 | **Buy the dip on parabolic runners**: after a multi-day run and a breakout, buy the pullback to old resistance | Humbled Trader, "How to Grow a Small Account" (practitioner, no published results) | Medium | **High** (discretionary terms need numbers) | Contrarian to our breakout entry. Long only, Robinhood-friendly. Pre-registered below. |
| 9 | **VWAP-band mean reversion**: fade moves past ±2 standard deviations back to VWAP | Practitioner technique; weak published evidence | Medium | High | Different kind of strategy that may earn on choppy days when momentum loses. |
| 10 | **ORB on a wider universe** (top 50 / 100 by relative volume) | QuantConnect variations | Medium (more data) | Medium | Paper finds the edge concentrates in the top names; expect it to fade. |
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

### Technical-analysis filters on stocks in play (queue #2)
```
Rule:               Baseline: the stocks-in-play ORB exactly as configured (config/orb.yaml).
                    Add ONE filter at a time; a filtered-out stock's slot stays empty
                    (sizing unchanged, so filters can only remove trades).
                    All inputs are from prior days' daily bars, known before the open.
                    F1 SMA trend:  long only if prior close > 20-day SMA; short only if below.
                    F2 RSI:        long only if prior-day RSI(14) <= 80; short only if >= 20.
                    F3 MACD:       long only if MACD(12,26) > signal(9) on prior close;
                                   short only if below.
                    F4 Bollinger:  skip if today's 9:30 open is outside the 20-day, 2-sigma
                                   bands computed from prior closes.
Why it should work: F1/F3 trade with the stock's own trend; F2/F4 avoid chasing moves that
                    are already stretched and prone to reverse.
Who loses:          Counter-trend breakout traders who get stopped out on fading moves.
Defaults:           Standard textbook settings above, not tuned.
Sweep (tune only):  None for the first pass. If a filter helps, one robustness check:
                    F1 with 50-day SMA; F2 with 70/30 thresholds.
Kill criteria:      A filter is dropped unless, on tune, it raises Sharpe by >= 0.2 without
                    cutting trades below 100 or worsening max drawdown. Also reported:
                    long-only results, since Robinhood may not allow shorting.
Caveats:            Four filters = four more trials in the log; the deflated Sharpe counts them.
                    If one wins, it must still pass validate before use.
Registered:         2026-10-28
```

### Buy the dip on parabolic runners (queue #8)
```
Rule:               Universe: stocks passing the stocks-in-play liquidity screen
                    (price > $5, 14-day avg volume >= 1M, ATR > $0.50).
                    Setup (all known before the open):
                      * "parabolic": at least 3 consecutive up days (close > prior close),
                        each with a wider high-low range than the day before
                      * "breakout": the latest close is above the highest high of the
                        20 days before the run started (the old resistance)
                    Entry (intraday): price pulls back to within 0.25 x ATR of the old
                      resistance, then a 5-minute bar makes a higher low and closes above
                      the prior 5-minute bar's high -> buy at the next bar's open.
                      No entries after 11:30 ET.
                    Stop: old resistance minus 0.15 x ATR ("10-15 cents of cushion",
                      scaled by ATR as the source suggests for higher-priced names).
                    Target: 2R; sell half there, move the stop on the rest to breakeven.
                    Exit anything left at the close. Long only.
                    Size: risk 1% of equity per trade, max 5 positions, max 1x leverage each.
Why it should work: Late buyers who missed the breakout step in when price returns to
                    the breakout level; former resistance becomes support.
Who loses:          Traders who bought the breakout high and sell the pullback in fear,
                    and shorts betting the run has topped.
Defaults:           run_days=3, resistance_lookback=20, zone_atr=0.25, stop_atr=0.15,
                    target_r=2.0, last_entry=11:30
Sweep (tune only):  run_days in {3,4,5}; zone_atr in {0.15,0.25,0.35}; stop_atr in {0.1,0.15,0.25}
Kill criteria:      Fails the scorecard on tune; or fewer than 100 trades on tune
                    (setup too rare to judge); or Sharpe < 0.5 with costs doubled.
Caveats:            Every number above is our translation of qualitative advice; the source
                    gives no thresholds or results. Data: needs minute bars for setups, which
                    the stocks-in-play download only has for top relative-volume names, so a
                    separate download pass is required.
Registered:         2026-10-28
```

### First half hour predicts last half hour on SPY (queue #6)
```
Rule:               Signal: SPY return from the prior close to 10:00 ET.
                    At 15:30, if the signal is positive go long, if negative go short
                    (long-only run: skip the shorts). Exit at the 16:00 close.
                    Decision on the 15:30 price, fill at the next minute's open.
                    Size: same vol-target sizing as the SPY momentum strategy (2% daily vol, 4x cap).
Why it should work: Gao, Han, Li & Zhou (JFE 2018): informed traders and late-day hedging
                    (index rebalancers, levered ETFs) trade in the direction of the morning move.
Who loses:          Liquidity providers absorbing the closing imbalance.
Defaults:           signal 16:00 prior close -> 10:00; trade 15:30 -> 16:00.
Sweep (tune only):  None.
Kill criteria:      Fails the scorecard on tune, or Sharpe < 0.5 with costs doubled.
Registered:         2026-10-29
```

### VWAP-band mean reversion on SPY (queue #9)
```
Rule:               Bands = session VWAP +/- k x the volume-weighted standard deviation of the
                    typical price around VWAP, both from 9:30 through the current minute.
                    Checks every 5 minutes from 10:00 to 15:30:
                      long if price < VWAP - 2 sd; short if price > VWAP + 2 sd.
                    Exit (checked every minute, filled at the next open): price back to VWAP,
                    or stop when price is beyond VWAP -/+ 3 sd. Anything open exits at the close.
                    At most 3 entries a day. Same vol-target sizing as SPY momentum.
Why it should work: On days with no news, large deviations from VWAP are liquidity shocks
                    that fade as institutions benchmarked to VWAP trade against them.
Who loses:          Impatient traders who push price away from fair value.
Defaults:           entry_sd=2.0, stop_sd=3.0, check every 5 min, 10:00-15:30, max 3 entries.
Sweep (tune only):  entry_sd in {1.5, 2.0, 2.5}; stop_sd in {3.0, 4.0}.
Kill criteria:      Fails the scorecard on tune, or Sharpe < 0.5 with costs doubled.
Caveat:             The opposite bet to SPY momentum; may only be useful as a diversifier.
Registered:         2026-10-29
```

### Volatility regime filter on SPY momentum, long-only (queue #5)
```
Rule:               The frozen long-only SPY momentum strategy, plus: sit the day out when the
                    prior 14-day realized volatility (std of daily close-to-close returns, known
                    before the open) is below the p-th percentile of its own trailing 252-day
                    history. A rolling percentile, so no threshold is fit to the whole tune period.
Why it should work: Gao et al. and Zarattini et al. find intraday momentum stronger when
                    volatility is high; quiet days are mostly whipsaws plus costs.
Who loses:          Same as the base strategy.
Defaults:           p = 25.
Sweep (tune only):  p in {10, 25, 40}.
Kill criteria:      Dropped unless tune Sharpe rises by >= 0.2 over the unfiltered 1.08 without
                    a worse max drawdown. High overfitting risk: counts as 3 more trials.
Registered:         2026-10-29
```

### Stocks-in-play: opening-range length (queue #3)
```
Rule:               Baseline ORB with the opening range set to the first 1, 5, 15 or 30 minutes.
                    Selection always happens at 9:35 by 5-minute relative volume (the only
                    opening volume we have for every stock; the paper measured it over the range).
                    For the 1-minute range the levels come from the 9:30 bar but orders can only
                    be placed from 9:35, when the stock is first known to be in play.
                    Direction comes from the range's own open-to-close; doji skipped.
Defaults:           or_minutes in {1, 5, 15, 30}; everything else as config/orb.yaml.
Kill criteria:      Kept only if a length beats 5 minutes by Sharpe >= 0.2 under every
                    fill rule we can trust (tick-resolved if available, else both bounds).
Registered:         2026-10-29
```

### Stocks-in-play: VWAP exit (queue #4)
```
Rule:               Baseline ORB, but after entry, exit at the next minute's open once a 1-minute
                    bar closes on the wrong side of the session VWAP (below for longs, above for
                    shorts). The ATR stop stays as protection; the close exit stays as a backstop.
Defaults:           stop_atr in {0.1, 0.5, 1.0} (the VWAP exit may make a wider stop workable).
Kill criteria:      Fails the scorecard on tune under both fill bounds, or no better than the
                    same stop without the VWAP exit by Sharpe >= 0.2.
Registered:         2026-10-29
```

### Tick-data check for ambiguous entry minutes
```
Rule:               For each trade where minute bars can't tell whether the stop was hit after
                    the entry (entry mid-bar and the bar's range reaches the stop), download that
                    symbol's trades for that one minute from Alpaca (SIP feed).
                    Using round-lot prints (100+ shares) in time order: entry = first print at or
                    through the level (fill at that print, plus slippage); stopped in that minute
                    if any later print is at or through the stop (fill at the stop or that print,
                    whichever is worse). Minutes without tick data fall back to "pessimistic".
Why:                Replaces the assumption with what happened. Not a new strategy; the
                    tick-resolved run becomes the ORB baseline that every ORB variant is judged on.
Registered:         2026-10-29
```
