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

All tune results are 2016–2021, net of costs. "Best" = best setting in its pre-registered sweep.

| Strategy | Tune | Validate | Holdout |
|---|---|---|---|
| SPY noise-area momentum, long-only | pass (Sharpe 1.08) | pass (Sharpe 1.67) | **fail: Sharpe 0.39**, 2.9%/yr (2024 +11.6%, 2025 +0.5%, 2026 −3.5%) |
| SPY momentum, long-only, stop checked every minute (C4) | pass (Sharpe 1.38, DD −7.9%) | pass (Sharpe 2.17; 2.23 at 1x) | **fail: Sharpe 0.82**, 5.3%/yr, DD −7.9% (2024 +13.0%, 2025 +3.4%, 2026 −1.0%); at 1x: 0.83, 2.5%/yr, DD −2.9%. SPY itself: Sharpe 1.21 |
| SPY momentum long-only at 1x, no margin (C0) | pass (Sharpe 1.06, 3.5%/yr, DD −3.0%) | pass (Sharpe 1.62) | reference |
| Same rules on QQQ / IWM / DIA (C1) | fail: 0.90 / 0.49 / 0.28; 4-ETF portfolio 0.90 | — | — |
| SPY signals traded in UPRO, no margin (C2) | fail: Sharpe 0.68 (wider spreads, 3x costs per $) | — | — |
| SPY momentum with Robinhood costs (C3) | Sharpe 1.07 (costs ≈ the paper's assumption) | — | — |
| SPY noise-area momentum, long + short | fail (Sharpe 0.93, DD 28%) | — | — |
| SPY momentum + volatility regime filter (#5) | dropped: Sharpe 1.05 / 0.87 / 0.75 at p = 10 / 25 / 40, below the unfiltered 1.08 | — | — |
| SPY first half hour → last half hour (#6) | fail: Sharpe −1.14 (long-only −1.33) | — | — |
| SPY VWAP-band reversion (#9) | fail: best −0.28 (2.5 / 4.0 sd); long-only −0.02 | — | — |
| Stocks-in-play ORB, tick-resolved | **fail: Sharpe −2.77**, −28%/yr (long-only −1.84). Bar-only bounds were −9.46 / +1.24 | — | — |
| ORB opening range 1 / 15 / 30 min (#3), ticks | fail: −1.11 / −2.60 / −3.15 | — | — |
| ORB VWAP exit (#4), ticks | fail: −3.23 / −0.89 / −0.80 at stop 0.1 / 0.5 / 1.0 ATR | — | — |
| ORB wider stops (E1), ticks | fail: −0.30 / −0.16 / +0.08 at stop 0.25 / 0.5 / 1.0 ATR; costs doubled −2.83 / −2.15 / −1.68. Implied before costs ≈ 2.2 / 1.8 / 1.8 | — | — |
| SPY overnight hold (D1) | fail: Sharpe 0.66, 7.6%/yr, DD −30%, beta 0.50; buy-and-hold SPY was better | — | — |
| SPY RSI(2) pullback (D3) | fail: 23 / 45 / 65 trades at RSI < 5 / 10 / 15 (needs 100); Sharpe 0.33 / 0.46 / 0.54; win rate 73–83% | — | — |
| Large-cap one-day reversal (D2), funds excluded | fail: next open −0.51 / −0.76 / −0.78, next close −0.59 / −0.78 / −0.72 at 5 / 10 / 20 picks; −19% to −35%/yr. Costs on daily full turnover ≈ 0.7 Sharpe; roughly flat before costs. (First run with ETFs in the universe void.) | — | — |
| ORB without ETFs/ETNs (F1), ticks | fail: wider stops −0.38 / −0.21 / −0.09 at 0.25 / 0.5 / 1.0 ATR (with funds: −0.30 / −0.16 / +0.08). Paper stop −3.35, long-only −2.33, but ~1,500 refilled entry minutes had no ticks (fell back to worst case), so those two read low | — | — |
| Gap-and-go without ETFs/ETNs (F2) | fail: 0.27–0.69 across the 9 settings (with funds 0.28–0.62); best 3% gap / 11:30, costs doubled 0.27 | — | — |
| QuantifiedStrategies daily rules (G), SPY | fail, all below Sharpe 1.0: turn of month 0.92 (70 trades, DD −8.7%), IBS 0.86 (11.1%/yr, DD −17%), 3-down overnight 0.73, turnaround Tue 0.61, RSI(2) 0.58, 5-day low 0.45, triple RSI −0.16 (15 trades). Costs barely matter (few trades) | — | — |
| Same, QQQ (G) | fail: IBS 0.86 (12.8%/yr, DD −20%), RSI(2) 0.46 | — | — |
| Classic rules combined (H): H3 turn of month + IBS + 3-down, one SPY account | **pass**: Sharpe 1.03, 15.5%/yr (SPY 15.4%), DD −17.7%, 152 trades, beta 0.69, costs doubled 1.01. But 2020 alone +53% (SPY +16%); 2021 +6.6% vs +27%. PBO 71% across 12 variants | fail on the letter (59 trades < 100; Sharpe 0.90), but 14.5%/yr, DD −11%, alpha t 2.5: 2022 +2.1% (SPY −19.5%), 2023 +28.2% (SPY +24.3%) | **fail: Sharpe 0.95**, 11.9%/yr vs SPY 17.9%, DD −19.8%, 72 trades; 2024 +10.4% (SPY +23.3%), 2025 +11.3% (+16.4%), 2026 +11.7% (+10.8%). Behind SPY risk-adjusted (gap −0.22) |
| Combined H1 (9 sleeves) / H2 (7 SPY rules) / H4 (SPY + QQQ halves) | fail: 0.79 / 0.77 / 0.85 | — | — |
| Swing rules on large-cap stocks (I) | fail. I2 (3-down + 5-day low + RSI(2), no trend filter, 10 stocks) Sharpe 1.09, 20.5%/yr (SPY 16.4%), but DD −33% (needs ≤ 25%); beta 0.83, corr 0.78; 2016 +45%, 2020 +40%, 2017 +6% vs +19%. With the 200-day filter worse: I1 0.66, I3 (RSI(2)) 0.72, I4 (4 rules, 20 stocks) 0.55 | — | — |
| H3 rules on more funds (J), equal slices | fail, just: J1 SPY/QQQ/IWM/DIA Sharpe 0.96, 14.1%/yr, DD −18%, 584 trades; J2 + 9 sectors 0.93, 11.9%/yr, DD −18%, 1,970 trades. Per fund (not trials): XLK 1.46, QQQ 1.10, DIA 0.96, XLV 0.94, XLY 0.92, XLU 0.90, XLP 0.89, XLI 0.46, XLB 0.41, IWM 0.39, XLE 0.14, XLF 0.09. Picking the winners after the fact would be overfitting | — | — |
| ORB + TA filters (#2), ticks | fail: SMA −1.45, RSI −2.80, MACD −1.85, Bollinger −1.88 | — | — |
| Gap-and-go on large caps (#7) | fail: all 9 settings positive (0.28–0.62) but none clears Sharpe 1.0; ~0 with costs doubled | — | — |
| Buy the dip on parabolic runners (#8) | fail: 3-day runs −0.42 to −0.63; 4–5-day runs too rare (4–106 trades) | — | — |

Batches: `runs/research/phase_a.csv` (bar-based bounds) and `runs/research/phase_b.csv` (tick-resolved
and the new strategies). Ticks: 24,529 ambiguous entry minutes downloaded; ~94% resolved from prints.

### Holdout verdict (2026-10-30)

Scored once each, 2024-01-02 to 2026-10-29. Both finalists stayed profitable but fail the scorecard
(Sharpe < 1.0, profit factor < 1.3, below SPY's own Sharpe, negative trailing year). The edge is
fading year by year: Sharpe 1.67 → 0.61 → −0.17 (4x version). The paper was published in 2024,
at the start of this window; published anomalies typically weaken afterwards (McLean & Pontiff,
2016). The every-minute stop still beat the 30-minute stop on unseen data (0.82 vs 0.39).

Per rule 5, the holdout is now spent: any further claim about these strategies needs new data,
which means live trading at small size or months that accumulate from here.

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

### Gap-and-go on large-cap gappers (queue #7)
```
Rule:               Universe each morning: the 500 largest stocks by 14-day average dollar volume
                    (prior days only; delisted names included), opening above $10.
                    Setup: gap up of 2% to 10% (9:30 open vs prior close).
                    Rank gappers by 5-minute relative volume; trade the top 5.
                    Entry: buy-stop at the high of the first 5-minute bar, live 9:35 to 10:30.
                    Stop: the low of the first 5-minute bar. Exit at the close.
                    Size: each of 5 slots gets equity/5; a stop-out loses 1% of the slot;
                    notional capped at 1x the slot (no margin, Robinhood-friendly). Long only.
                    Same fill rules as stocks in play (fills.same_bar_stop, ticks where available).
Why it should work: Large caps that gap on news and push through the opening high are seeing
                    institutional demand that takes more than one morning to fill.
Who loses:          Shorts fading the gap and holders selling into strength too early.
Defaults:           min_gap 0.02, max_gap 0.10, large_cap_rank 500, top_n 5, last_entry 10:30.
Sweep (tune only):  min_gap in {0.01, 0.02, 0.03}; last_entry in {10:00, 10:30, 11:30}.
Kill criteria:      Fails the scorecard on tune, fewer than 100 trades, or Sharpe < 0.5 at 2x costs.
Data:               Needs minute bars for gappers outside the top-30 relative-volume list:
                    an extra download of up to 15 large-cap gap-ups per day.
Registered:         2026-10-29
```

Buy-the-dip implementation notes (fixed before coding, filling gaps in the registration
above): "max 5 positions" = the first 5 setups to trigger each day, each with notional capped
at equity/5 (1x overall); the 5-minute bars are built from 1-minute bars starting 9:30; the
pullback must touch the zone before the reversal bar; half the position sells at 2R (limit,
no slippage), the stop on the rest moves to the entry price; if a bar reaches both the stop
and the target, the stop is assumed first. Minute data: up to 30 setups per day, largest by
dollar volume first.

### Improving SPY momentum, long-only (batch C)
```
Base:               config/spy_noise.yaml with allow_short false (tune Sharpe 1.08). Rules frozen;
                    only the four changes below, each judged against that base on tune.
C1 More indexes:    the same rules, same settings, on QQQ, IWM and DIA. Portfolio: capital split
                    equally across SPY, QQQ, IWM, DIA sleeves at the start (no rebalancing).
                    Kept if the portfolio's tune Sharpe >= 1.28 (base + 0.2) without a worse max
                    drawdown. Each ETF is also reported alone.
C2 Leveraged ETF:   SPY signals, trades executed in UPRO (3x S&P 500) at UPRO's own minute prices.
                    UPRO notional = SPY target notional / 3, capped at 1x equity (cash account,
                    no margin). Slippage $0.01/share (wider spreads). A realism change, not an
                    edge: compared with SPY capped at 1x (C0) and with the 4x-margin base.
C0 Cash account:    SPY base with max_leverage 1.0, the honest reference for a small account.
C3 Robinhood costs: commission $0, slippage $0.005/share each side (half a 1-cent spread).
                    Realism change; reported, not kept or killed.
C4 Stop check:      trailing stop checked every minute instead of every 30 (the paper's text is
                    ambiguous). One trial. Kept if tune Sharpe >= 1.28.
Validate:           anything kept from C1/C4, once, on 2022-2023.
Registered:         2026-10-29
```

## Batch D: holding overnight to a few days (registered 2026-10-30)

Common to all three: decide at 3:50 pm ET (10 minutes before the close, the deadline for
market-on-close orders) using prices up to 3:50 pm; fill entries and close-exits at the official
close, open-exits at the next official open; slippage $0.01/share each side plus $0.0035/share
commission; long only, no margin (at most 1x equity); half-days skipped (no 3:50 pm price).
Prices are raw: an overnight gap over 40% counts as a split; its ratio is snapped to the nearest
common split ratio for P&L, and such stocks are never picked on the gap day.

Holdout note: rule 5 says ideas after the holdout is spent are judged on new data. These three
have never been run on 2024+, but we already know how markets behaved then. A 2024+ score for
them is therefore weaker evidence than a clean holdout; the real judge is live trading.

### D1 SPY overnight hold
```
Rule:               Every day buy SPY at the close, sell at the next open. Size 1x equity at the
                    3:50 pm price.
Why it should work: Most of the US equity premium has historically been earned overnight
                    (Cliff, Cooper & Gulen 2008; Lou, Polk & Skouras 2019); intraday returns
                    have been near zero or negative.
Who loses:          Intraday traders and liquidity providers who unwind into the close.
Defaults:           none.     Sweep: none.
Kill criteria:      Fails the scorecard on tune, or Sharpe below buy-and-hold SPY's on tune.
```

### D2 Large-cap one-day reversal
```
Rule:               Universe each day: top 500 by 14-day average dollar volume (prior days),
                    price > $5. At 3:50 pm rank by return from the prior close to 3:50 pm;
                    buy the 10 biggest losers (must be down) equally at the close.
                    D2a: sell at the next open. D2b: sell at the next close.
Why it should work: Short-term reversal (Lehmann 1990; Jegadeesh 1990): big one-day moves in
                    liquid stocks partly revert as liquidity providers are paid to absorb them
                    (Nagel 2012). Measuring to 3:50 pm and filling at the close avoids the
                    bid-ask bounce that inflates close-to-close tests.
Who loses:          Forced and impatient sellers.
Defaults:           top 500, 10 names.   Sweep (tune only): n_picks in {5, 10, 20}.
Kill criteria:      Fails the scorecard on tune, or Sharpe < 0.5 with costs doubled.
```

### D3 SPY RSI(2) pullback (Connors)
```
Rule:               At 3:50 pm, using daily closes with today's 3:50 pm price as today's close:
                    if flat, RSI(2) < 10 and price > 200-day SMA -> buy at the close.
                    If long and price > 5-day SMA -> sell at the close. No stop.
                    Size 1x equity.
Why it should work: Short pullbacks inside long-term uptrends in index ETFs tend to revert
                    (Connors & Alvarez 2009). Practitioner evidence; known to have weakened.
Who loses:          Short-term sellers in a rising market.
Defaults:           rsi_entry 10, sma_trend 200, sma_exit 5.
Sweep (tune only):  rsi_entry in {5, 10, 15}.
Kill criteria:      Fails the scorecard on tune, or fewer than 100 trades.
```

## Batch E: stocks-in-play ORB with wider stops (registered 2026-10-30)

### E1 ORB stop distance
```
Rule:               Baseline stocks-in-play ORB (config/orb.yaml), tick-resolved fills, with the
                    stop at 0.25, 0.5 or 1.0 x 14-day ATR instead of the paper's 0.10. Risk per
                    trade stays 1% of the slot, so wider stops mean fewer shares; leverage cap
                    unchanged. Exit at the stop or the close, as in the paper.
Why:                At 0.10 ATR most of the result was decided inside the entry minute (bounds
                    -9.46 / +1.24, ticks -2.77) and costs were a large share of the risk. A wider
                    stop tests whether the breakout itself has an edge once noise stops it out
                    less often. Batch B's VWAP-exit runs at 0.5 / 1.0 ATR scored -0.89 / -0.80,
                    so the prior is low.
Kill criteria:      Fails the scorecard on tune (2016-2021). Three trials, counted in the log.
                    A survivor gets one validate run (2022-2023); the holdout is not used.
Registered:         2026-10-30
```

### Batch D, first run (2026-10-30, run 41094382206)
10 trials ran (counted). The trial-log commit failed on a push race, so they are recorded here.
D2's universe ranked every us_equity asset by dollar volume, which includes ETFs and ETNs; the
biggest one-day losers were often leveraged or volatility products that kept falling overnight
(worst days Mar 17 2020 −38%, Feb 6 2018 −19%). That breaks the registered rule ("large caps"),
so D2 is rerun with `params.exclude_funds: true` (fund names: ETF/ETN/Fund/leveraged/VIX and fund
issuers; tested). The snapshot download ranked funds too, so after exclusion some days have fewer
than 500 stocks. D1 and D3 are unaffected (SPY only). ORB and gap-and-go universes also include
funds; not changed here.

## Batch F: ORB and gap-and-go without ETFs/ETNs (registered 2026-10-30)
```
Rule:               Same as the tick-resolved ORB baseline, long-only, and wider-stop runs, and the
                    nine gap-and-go settings, but ETFs/ETNs (by name; see daytrader/data/universe.py
                    is_fund) are dropped from each day's list before ranking to top_n. Their slots go
                    to the next-ranked stocks (the download keeps 10 spare ORB names and 5 spare
                    gappers per day; any day short of minute data is counted as missing).
Why:                Batch D showed fund products polluting a "stocks" universe. The ORB paper and the
                    gap-and-go rule are about stocks.
Kill criteria:      Same scorecard on tune. 14 trials, counted. A survivor gets one validate run.
Registered:         2026-10-30
```

## Batch G: classic daily rules from QuantifiedStrategies (registered 2026-10-30)
Source: Brandon's printout of "200 Trading Strategies (Free) 2026" (quantifiedstrategies.com), a
list of article titles. The rules below are the standard published versions of the SPY/QQQ daily
ones on that list, written down before any run; they may differ in detail from each article.
```
Rules (decide at 3:50 pm, fill at the close; "price" = 3:50 pm price; long-only, 1x, no margin):
  ibs             IBS = (price - today's low) / (today's high - low) < 0.2 -> buy;
                  sell when price > yesterday's high                          (SPY, QQQ)
  turnaround_tue  Monday and price < Friday's close -> buy; sell when price > yesterday's high
  turn_of_month   buy on the 5th-last trading day of the month; sell on the 3rd trading day
  down3_overnight three lower closes in a row (today's at 3:50 pm) -> buy at the close, sell at
                  the next open
  five_day_low    price < lowest low of the prior 5 days -> buy; sell when price > yesterday's high
  triple_rsi      RSI(5) < 30, RSI(5) lower 3 days running, RSI(5) 3 days ago < 60, price above the
                  200-day SMA -> buy; sell when RSI(5) > 50
  rsi2_high       RSI(2) < 10 -> buy; sell when price > yesterday's high          (SPY, QQQ)
Costs:              $0.0035 commission + 1 cent slippage per share each side.
Why:                Brandon asked to test the list. These are the clearly specified SPY/QQQ daily
                    rules on it; the rest are titles without fixed rules, other markets, or
                    near-duplicates. Close cousins of D3 (RSI(2) above the 200-day SMA).
Kill criteria:      Scorecard on tune 2016-2021 (Sharpe > 1, >= 100 trades, beats SPY risk-adjusted,
                    costs doubled > 0.5, ...). 9 trials, counted with the ~90 before (deflated
                    Sharpe). A survivor gets one validate run 2022-2023. Their 2024+ data has been
                    seen (SPY momentum holdout), so a holdout score would be weak evidence.
Registered:         2026-10-30
```

## Batch H: the batch G rules run together (registered 2026-10-30)
```
Rule:               H1 nine sleeves, 1/9 of the account each. H2 one account, fully in SPY (1x)
                    whenever any of the seven SPY rules is in (each rule keeps its own in/out
                    state; the overnight rule's exit at the open applies only when it is the
                    only rule in). H3 the same with turn of month + IBS + 3-down overnight.
                    H4 half H2 on SPY, half IBS + RSI(2) on QQQ.
Why:                Brandon asked to run them all at once. Each rule alone is in cash most days;
                    together they are invested more often.
Kill criteria:      Scorecard on tune 2016-2021, including beating SPY risk-adjusted. 4 trials,
                    counted. Survivors get one validate run 2022-2023.
Registered:         2026-10-30
```

## Batch I: the swing rules on individual large caps (registered 2026-10-30)
```
Rule:               Top 500 stocks by prior dollar volume (ETFs/ETNs excluded), price > $5.
                    At 3:50 pm each (stock, rule) is checked on its own; a stock is held while any
                    of its rules is in, at most max_positions stocks in equal slices (1x, no
                    margin); new entries go to the most oversold by RSI(2). Rules as in batch G:
                    3 lower closes (exit next open, unless another rule is also in), price below
                    the prior 5-day low, RSI(2) < 10, down Monday; exits at price > yesterday's
                    high. IBS and turn of month aren't used: IBS needs each stock's 9:30-3:50 range
                    (not downloaded), and turn of month on 500 stocks is just the index.
                    Stocks with a split-like gap in the prior 200 days are skipped; a held stock
                    whose history breaks is sold.
  I1 three rules (3-down, 5-day low, RSI(2)), above 200-day SMA, 10 positions
  I2 same, no trend filter
  I3 RSI(2) only, above 200-day SMA, 10 positions (the classic Connors stock version)
  I4 all four rules, above 200-day SMA, 20 positions
Why:                Brandon asked to run H3's idea on the stocks in the index.
Kill criteria:      Scorecard on tune 2016-2021 (the 3:50 pm snapshots cover it). 4 trials, counted.
                    Survivors get one validate run after the snapshot download finishes 2022-2023.
                    No earnings-date filter (no earnings calendar in the data).
Registered:         2026-10-30
```

## Batch J: H3 on more funds (registered 2026-10-30)
```
Rule:               H3 unchanged (turn of month + IBS + 3-down overnight; a fund is held at its
                    slice while any of the three is in), run on each fund in its own equal slice:
  J1 SPY, QQQ, IWM, DIA (1/4 each)
  J2 those plus the nine SPDR sectors XLK XLF XLE XLV XLI XLY XLP XLU XLB (1/13 each)
                    Single-fund runs are reported for the table, not scored as trials.
Why:                Brandon wants more trades. Same rules on more funds adds signals on different
                    days without loosening any threshold.
Kill criteria:      Scorecard on tune 2016-2021. 2 trials, counted. A survivor gets one validate
                    run 2022-2023. Idle cash in each slice earns nothing (no shared pool yet).
Registered:         2026-10-30
```

## Batch K (registered 2026-10-30)
```
H3 holdout:         H3 unchanged, scored once on 2024-01-02 to the latest day. This uses up the
                    holdout for H3. The years have been seen by everyone (and SPY's 2024+ was used for
                    the momentum holdout), so it is weaker evidence than the 2022-2023 validate run.
                    Pass = the same scorecard, judged with the 2-year trade count caveat noted.
K1 shared pool:     H3's rules on SPY, QQQ, IWM, DIA; each (fund, rule) keeps its own in/out state;
                    the whole account splits equally across the funds with any rule in, re-split
                    only on days that set changes (partial buys/sells at the close, costs charged).
                    Fixes J1's idle cash. 1 trial, counted. Scorecard on tune; a survivor gets one
                    validate run.
Registered:         2026-10-30
```

