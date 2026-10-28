"""Self-contained HTML backtest report (one file, opens on a phone, no server)."""

from __future__ import annotations

import datetime as dt
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..analysis.analyze import Analysis
from ..backtest.engine import BacktestResult
from ..config import period_bounds
from ..data.model import Day

# Reference palette (light mode): see the dataviz skill's palette.md
SERIES = "#2a78d6"     # strategy
BENCH = "#898781"      # SPY benchmark (neutral)
NEG = "#e34948"        # losses / negative pole
POS = "#2a78d6"        # gains / positive pole
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
GOOD, CRIT, WARN = "#006300", "#d03b3b", "#fab219"
PERIOD_FILL = {"tune": "rgba(42,120,214,0.06)", "validate": "rgba(98,80,214,0.08)", "holdout": "rgba(235,104,52,0.09)"}

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def _layout(fig: go.Figure, height: int = 320, **kw) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=48, r=16, t=16, b=40), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK2), hovermode=kw.pop("hovermode", "closest"),
        showlegend=kw.pop("showlegend", False),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=12)), **kw)
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED))
    return fig


def _div(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})


def _pct(v, d=1, signed=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:+.{d}%}" if signed else f"{v:.{d}%}"


def _num(v, d=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "∞" if v == float("inf") else "n/a"
    return f"{v:.{d}f}"


# ---- charts ---------------------------------------------------------------------------------
def equity_chart(res: BacktestResult, cfg: dict) -> str:
    d = res.daily
    eq = (1 + d["ret"]).cumprod()
    mk = (1 + d["market_ret"]).cumprod()
    dd = eq / eq.cummax() - 1
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.05)
    for name in ("tune", "validate", "holdout"):
        s, e = period_bounds(cfg, name)
        s, e = pd.Timestamp(s), pd.Timestamp(e) if e else d.index.max()
        if e < d.index.min() or s > d.index.max():
            continue
        fig.add_vrect(x0=max(s, d.index.min()), x1=min(e, d.index.max()), fillcolor=PERIOD_FILL[name],
                      line_width=0, layer="below", row="all", col=1)
    fig.add_trace(go.Scatter(x=mk.index, y=mk, name="SPY buy-and-hold", line=dict(color=BENCH, width=2, dash="dash"),
                             hovertemplate="%{x|%b %d, %Y}<br>SPY $%{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq.index, y=eq, name="Strategy", line=dict(color=SERIES, width=2),
                             hovertemplate="%{x|%b %d, %Y}<br>Strategy $%{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd, name="Drawdown", fill="tozeroy", line=dict(color=NEG, width=1),
                             fillcolor="rgba(227,73,72,0.18)", showlegend=False,
                             hovertemplate="%{x|%b %d, %Y}<br>Drawdown %{y:.1%}<extra></extra>"), row=2, col=1)
    fig.update_yaxes(title_text="Growth of $1 (log scale)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    return _div(_layout(fig, 460, showlegend=True, hovermode="x unified"))


def hist_chart(samples: np.ndarray, color: str, lines: list[tuple[float, str, str]], xfmt: str = "",
               xtitle: str = "", bins: int = 40, tail_below: float | None = None) -> str:
    fig = go.Figure()
    if tail_below is not None:
        fig.add_trace(go.Histogram(x=samples[samples >= tail_below], nbinsx=bins, marker=dict(color=color),
                                   hovertemplate="%{x}<br>%{y} runs<extra></extra>"))
        fig.add_trace(go.Histogram(x=samples[samples < tail_below], nbinsx=max(5, bins // 3),
                                   marker=dict(color=NEG), hovertemplate="%{x}<br>%{y} runs<extra></extra>"))
        fig.update_layout(barmode="overlay")
    else:
        fig.add_trace(go.Histogram(x=samples, nbinsx=bins, marker=dict(color=color),
                                   hovertemplate="%{x}<br>%{y} samples<extra></extra>"))
    for x, label, c in lines:
        fig.add_vline(x=x, line=dict(color=c, width=2, dash="dash"),
                      annotation=dict(text=label, font=dict(color=c, size=12), yanchor="top"))
    fig.update_traces(marker_line=dict(color=SURFACE, width=1))
    fig.update_xaxes(tickformat=xfmt, title_text=xtitle)
    fig.update_yaxes(title_text="Count")
    return _div(_layout(fig, 300, bargap=0.04))


def scatter_market(res: BacktestResult, a: Analysis) -> str:
    d = res.daily
    fig = go.Figure(go.Scatter(x=d["market_ret"], y=d["ret"], mode="markers",
                               marker=dict(color=SERIES, size=8, opacity=0.45, line=dict(color=SURFACE, width=1)),
                               customdata=d.index.strftime("%b %d, %Y"),
                               hovertemplate="%{customdata}<br>SPY %{x:.2%}<br>Strategy %{y:.2%}<extra></extra>"))
    if a.alpha:
        xs = np.linspace(d["market_ret"].min(), d["market_ret"].max(), 2)
        fig.add_trace(go.Scatter(x=xs, y=a.alpha["alpha_daily"] + a.alpha["beta"] * xs, mode="lines",
                                 line=dict(color=INK, width=2), hoverinfo="skip"))
    fig.update_xaxes(title_text="SPY daily return", tickformat=".1%")
    fig.update_yaxes(title_text="Strategy daily return", tickformat=".1%")
    return _div(_layout(fig, 360))


def grouped_bars(table: pd.DataFrame, title_x: str = "") -> str:
    t = table.dropna(how="all")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=t.index.astype(str), y=t["strategy"], name="Strategy", marker=dict(color=SERIES),
                         customdata=t["days"], hovertemplate="%{x}<br>Strategy avg %{y:.3%}<br>%{customdata} days<extra></extra>"))
    fig.add_trace(go.Bar(x=t.index.astype(str), y=t["market"], name="SPY", marker=dict(color=BENCH),
                         hovertemplate="%{x}<br>SPY avg %{y:.3%}<extra></extra>"))
    fig.update_traces(marker_line=dict(color=SURFACE, width=2))
    fig.update_yaxes(tickformat=".2%", title_text="Average day")
    fig.update_xaxes(title_text=title_x)
    return _div(_layout(fig, 300, showlegend=True, barmode="group", bargap=0.3))


def yearly_chart(a: Analysis) -> str:
    y = a.yearly
    fig = go.Figure()
    fig.add_trace(go.Bar(x=y.index.astype(str), y=y["strategy"], name="Strategy",
                         marker=dict(color=[SERIES if v >= 0 else NEG for v in y["strategy"]]),
                         hovertemplate="%{x}<br>Strategy %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Bar(x=y.index.astype(str), y=y["market"], name="SPY", marker=dict(color=BENCH),
                         hovertemplate="%{x}<br>SPY %{y:.1%}<extra></extra>"))
    fig.update_traces(marker_line=dict(color=SURFACE, width=2))
    fig.update_yaxes(tickformat=".0%", title_text="Return")
    return _div(_layout(fig, 300, showlegend=True, barmode="group", bargap=0.25))


def rolling_chart(a: Analysis) -> str:
    rs = a.rolling
    fig = go.Figure(go.Scatter(x=rs.index, y=rs, line=dict(color=SERIES, width=2),
                               hovertemplate="%{x|%b %Y}<br>6-month Sharpe %{y:.2f}<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color=CRIT, width=1.5, dash="dash"),
                  annotation=dict(text="1.0 bar", font=dict(color=CRIT, size=12)))
    fig.add_hline(y=0.0, line=dict(color=AXIS, width=1))
    fig.update_yaxes(title_text="Rolling 6-month Sharpe")
    return _div(_layout(fig, 300))


def heatmap_chart(sweep: pd.DataFrame, cfg: dict) -> str:
    px, py = [c for c in sweep.columns if c not in ("sharpe", "annual_return", "total_return")][:2]
    grid = sweep.pivot(index=py, columns=px, values="sharpe")
    fig = go.Figure(go.Heatmap(z=grid.values, x=[str(v) for v in grid.columns], y=[str(v) for v in grid.index],
                               colorscale=[[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]], zmid=0,
                               hovertemplate=f"{px} %{{x}}<br>{py} %{{y}}<br>Sharpe %{{z:.2f}}<extra></extra>",
                               colorbar=dict(title="Sharpe", thickness=12)))
    cx, cy = str(cfg["params"].get(px)), str(cfg["params"].get(py))
    fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers", marker=dict(symbol="square-open", size=26,
                             color=INK, line=dict(width=3)), hoverinfo="skip"))
    fig.update_xaxes(title_text=px, type="category")
    fig.update_yaxes(title_text=py, type="category")
    return _div(_layout(fig, 340))


def cost_chart(a: Analysis, cfg: dict) -> str:
    c = a.cost_curve
    base = float(cfg["costs"]["slippage_per_share"])
    x = (c["extra_slippage"] + base) * 100
    fig = go.Figure(go.Scatter(x=x, y=c["sharpe"], line=dict(color=SERIES, width=2),
                               hovertemplate="Slippage %{x:.2f}¢/share<br>Sharpe %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    fig.add_vline(x=base * 100, line=dict(color=MUTED, width=1.5, dash="dot"),
                  annotation=dict(text="assumed", font=dict(color=MUTED, size=12)))
    if np.isfinite(a.break_even):
        fig.add_trace(go.Scatter(x=[a.break_even * 100], y=[0], mode="markers",
                                 marker=dict(size=12, color=SURFACE, line=dict(color=CRIT, width=3)),
                                 hovertemplate="Break-even %{x:.2f}¢/share<extra></extra>"))
    fig.update_xaxes(title_text="Slippage per share (cents, each side)")
    fig.update_yaxes(title_text="Sharpe")
    return _div(_layout(fig, 300))


def excursion_chart(trades: pd.DataFrame) -> str:
    win = trades["pnl"] > 0
    fig = go.Figure()
    for mask, name, color in ((win, "Winners", SERIES), (~win, "Losers", NEG)):
        t = trades[mask]
        fig.add_trace(go.Scatter(x=t["mae_r"], y=t["mfe_r"], mode="markers", name=name,
                                 marker=dict(color=color, size=8, opacity=0.5, line=dict(color=SURFACE, width=1)),
                                 customdata=np.stack([t["date"].dt.strftime("%b %d, %Y"), t["r_multiple"]], axis=-1),
                                 hovertemplate="%{customdata[0]}<br>Against %{x:.2f}R · For %{y:.2f}R<br>Result %{customdata[1]:.2f}R<extra></extra>"))
    fig.add_vline(x=1.0, line=dict(color=CRIT, width=1.5, dash="dash"),
                  annotation=dict(text="1R (initial stop)", font=dict(color=CRIT, size=12)))
    fig.update_xaxes(title_text="Went against (R)")
    fig.update_yaxes(title_text="Went for (R)")
    return _div(_layout(fig, 340, showlegend=True))


def trade_chart(day: Day, trade: pd.Series) -> str:
    n = day.n
    t0 = pd.Timestamp.combine(day.date, dt.time(9, 31))
    times = [t0 + pd.Timedelta(minutes=i) for i in range(n)]
    tp = (day.high + day.low + day.close) / 3
    vwap = np.cumsum(tp * day.volume) / np.maximum(np.cumsum(day.volume), 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=day.close, name="Price", line=dict(color=INK, width=1.5),
                             hovertemplate="%{x|%H:%M}<br>$%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=times, y=vwap, name="VWAP", line=dict(color="#6250d6", width=1.5, dash="dash"),
                             hovertemplate="%{x|%H:%M}<br>VWAP $%{y:.2f}<extra></extra>"))
    eb, xb = int(trade["entry_bar"]), int(trade["exit_bar"])
    fig.add_trace(go.Scatter(x=[times[max(eb - 1, 0)]], y=[trade["entry_px"]], mode="markers", name="Entry",
                             marker=dict(symbol="triangle-up" if trade["side"] > 0 else "triangle-down", size=14,
                                         color=SERIES, line=dict(color=SURFACE, width=2)),
                             hovertemplate="Entry $%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[times[min(max(xb - 1, 0), n - 1)]], y=[trade["exit_px"]], mode="markers", name="Exit",
                             marker=dict(symbol="x", size=13, color=INK), hovertemplate="Exit $%{y:.2f}<extra></extra>"))
    if np.isfinite(trade.get("stop_at_entry", np.nan)):
        fig.add_hline(y=trade["stop_at_entry"], line=dict(color=CRIT, width=1, dash="dot"),
                      annotation=dict(text="stop at entry", font=dict(color=CRIT, size=11)))
    fig.update_xaxes(tickformat="%H:%M")
    return _div(_layout(fig, 280, showlegend=True))


# ---- page ---------------------------------------------------------------------------------
def _tile(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="tile"><span class="tl">{html.escape(label)}</span>'
            f'<span class="tv">{html.escape(value)}</span><span class="ts">{html.escape(sub)}</span></div>')


def _cond_table(t: pd.DataFrame) -> pd.DataFrame:
    t = t.reset_index()
    return pd.DataFrame({
        "SPY day": t.iloc[:, 0].astype(str), "Days": t["days"].fillna(0).astype(int),
        "Strategy avg day": t["strategy"].map(lambda v: f"{v:+.3%}" if pd.notna(v) else "n/a"),
        "SPY avg day": t["market"].map(lambda v: f"{v:+.3%}" if pd.notna(v) else "n/a"),
        "Strategy up-day share": t["strategy_hit"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "n/a")})


def _table(df: pd.DataFrame, cls: str = "") -> str:
    return df.to_html(index=False, border=0, classes=f"tbl {cls}", escape=True)


def build_report(res: BacktestResult, a: Analysis, cfg: dict, period: str, run_id: str, out_path: Path,
                 days_by_date: dict | None = None, sweep: pd.DataFrame | None = None,
                 synthetic: bool = False, holdout_views: int = 0, code_version: str = "") -> Path:
    p, t, perf, mp = cfg["params"], a.trades, a.perf, a.market_perf
    d = res.daily
    start, end = d.index.min().date(), d.index.max().date()
    verdict_ok = a.passed
    n_fail = sum(1 for c in a.checks if c.passed is False)
    n_run = sum(1 for c in a.checks if c.passed is not None)

    banner = ""
    if synthetic:
        banner = '<div class="banner warn"><b>Synthetic data.</b> These results come from made-up prices used to test the code. They say nothing about the strategy.</div>'
    elif period == "holdout":
        msg = ("This is the first look at the holdout." if holdout_views <= 1 else
               f"The holdout has now been viewed {holdout_views} times. Results after the first look are no longer clean out-of-sample evidence.")
        banner = f'<div class="banner {"warn" if holdout_views > 1 else "info"}"><b>Holdout.</b> {msg}</div>'

    verdict = (f'<div class="verdict ok"><span aria-hidden="true">✓</span> Passes {n_run} of {n_run} checks</div>' if verdict_ok
               else f'<div class="verdict bad"><span aria-hidden="true">✕</span> Fails {n_fail} of {n_run} checks</div>')

    boot = a.boot or {}
    kpis = "".join([
        _tile("Sharpe ratio", _num(perf.get("sharpe")), f"95% range {_num(boot.get('lo'))} – {_num(boot.get('hi'))}"),
        _tile("Annual return", _pct(perf.get("annual_return")), f"SPY same days {_pct(mp.get('annual_return'))}"),
        _tile("Max drawdown", _pct(perf.get("max_drawdown")), f"Monte Carlo 95th pct {_pct(a.mc.get('dd_p95'))}"),
        _tile("Trades", f"{t.get('trades', 0):,}", f"{_num(t.get('trades_per_day'), 2)} per day"),
        _tile("Deflated Sharpe", _pct(a.dsr.get("dsr"), 0), f"after {a.dsr.get('trials', 1)} variant(s) tested"),
        _tile("Alpha vs SPY", _pct(a.alpha.get("alpha_annual")), f"t-stat {_num(a.alpha.get('alpha_t'), 1)} · beta {_num(a.alpha.get('beta'))}"),
    ])

    rows = []
    for c in a.checks:
        if c.passed is None:
            badge = '<span class="badge na">— Not run</span>'
        elif c.passed:
            badge = '<span class="badge ok">✓ Pass</span>'
        else:
            badge = '<span class="badge bad">✕ Fail</span>'
        rows.append(f"<tr><td>{html.escape(c.name)}</td><td class='mono'>{html.escape(c.display)}</td>"
                    f"<td class='mono'>{html.escape(c.needs)}</td><td>{badge}</td></tr>")
    scorecard = ("<div class='scroll'><table class='tbl'><thead><tr><th>Check (period: "
                 f"{html.escape(period)})</th><th>Value</th><th>Needs</th><th>Result</th></tr></thead><tbody>"
                 + "".join(rows) + "</tbody></table></div>")

    ref = cfg.get("reference", {})
    real_tiles = "".join([
        _tile("t-statistic", _num(a.mean_test.get("t_stat")), "average day above zero"),
        _tile("p-value", _num(a.mean_test.get("p_value"), 3), "chance a zero-edge strategy looks this good"),
        _tile("Deflated Sharpe", _pct(a.dsr.get("dsr"), 0), f"chance the edge is real after {a.dsr.get('trials', 1)} tries"),
        _tile("Overfit probability", _pct(a.pbo.get("pbo"), 0) if a.pbo else "n/a",
              f"{a.pbo.get('variants')} variants" if a.pbo else "needs 2+ variants on this period"),
        _tile("Resamples with Sharpe below 1", _pct(boot.get("p_below_1"), 0), "block bootstrap"),
        _tile("Published Sharpe", _num(ref.get("sharpe")) if ref else "n/a", ref.get("source", "") if ref else ""),
    ])

    mc = a.mc or {}
    loss_limit = abs(mc.get("worst_day_p99", float("nan"))) * 1.25
    worst = a.worst_days.reset_index()
    worst_tbl = pd.DataFrame({
        "Date": worst["date"].dt.strftime("%b %d, %Y"), "Strategy": worst["ret"].map(lambda v: f"{v:+.2%}"),
        "SPY": worst["market_ret"].map(lambda v: f"{v:+.2%}"), "Trades": worst["n_trades"]})

    cond = a.conditional
    cond_html = grouped_bars(cond["market_day"], "Market day (SPY close-to-close)")
    vol_html = grouped_bars(cond["vol_regime"], "Volatility over the prior 14 days") if "vol_regime" in cond else ""

    heat_html = heatmap_chart(sweep, cfg) if sweep is not None and not sweep.empty else (
        "<p class='note'>Not run yet. Run <code>sweep</code> on the tune period to fill this in.</p>")

    trades = res.trades
    trade_section = "<p class='note'>No trades.</p>"
    if not trades.empty:
        r_hist = hist_chart(trades["r_multiple"].clip(-5, 10).to_numpy(), SERIES, [(0.0, "0", AXIS)],
                            xtitle="Result per trade (R multiples, clipped at -5 and +10)", bins=50)
        exc = excursion_chart(trades)
        show = trades.sort_values(["date", "entry_bar"], ascending=False).head(25)
        tt = pd.DataFrame({
            "Date": show["date"].dt.strftime("%Y-%m-%d"), "Side": show["side"].map({1: "Long", -1: "Short"}),
            "Shares": show["shares"], "Entry": show["entry_time"] + " @ " + show["entry_px"].map("{:.2f}".format),
            "Exit": show["exit_time"] + " @ " + show["exit_px"].map("{:.2f}".format),
            "Exit reason": show["exit_reason"].str.replace("_", " "), "P&L": show["pnl"].map("{:+,.0f}".format),
            "R": show["r_multiple"].map("{:+.2f}".format)})
        examples = ""
        if days_by_date:
            picks = pd.concat([trades.nlargest(2, "pnl"), trades.nsmallest(2, "pnl")])
            for _, tr in picks.iterrows():
                day = days_by_date.get(tr["date"].date())
                if day is None:
                    continue
                label = "Best" if tr["pnl"] > 0 else "Worst"
                side = "long" if tr["side"] > 0 else "short"
                examples += (f"<h4>{label}: {tr['date']:%b %d, %Y}, {side}, {tr['r_multiple']:+.2f}R "
                             f"(${tr['pnl']:+,.0f}), exit: {html.escape(tr['exit_reason'].replace('_', ' '))}</h4>"
                             + trade_chart(day, tr))
        trade_section = f"""
<div class="grid2">
  <div><h3>Result per trade</h3>{r_hist}
  <p class="note">Win rate {_pct(t.get('win_rate'), 0)} · profit factor {_num(t.get('profit_factor'))} ·
  expectancy {_num(t.get('expectancy_r'))}R · avg win {_pct(t.get('avg_win_ret'), 2)} · avg loss {_pct(t.get('avg_loss_ret'), 2)}</p></div>
  <div><h3>How far trades went against vs. for us</h3>{exc}
  <p class="note">Losers bunched near 1R mean stops work as designed. Winners that first went far against us suggest the stop could be tighter.</p></div>
</div>
<h3>Most recent trades</h3><div class="scroll">{_table(tt)}</div>
<p class="note">Full list in the trades CSV saved next to this report.</p>
{"<h3>Example days</h3>" + examples if examples else ""}"""

    details = pd.DataFrame([
        ("Run ID", run_id), ("Code version", code_version or "n/a"), ("Strategy", cfg["strategy"]),
        ("Data", "Synthetic" if synthetic else "Alpaca SIP, 1-minute, raw prices"),
        ("Scored period", f"{period}: {start} to {end}"),
        ("Tune / validate / holdout", " | ".join(f"{k} {v[0]}–{v[1] or 'now'}" for k, v in cfg["periods"].items())),
        ("Commission", f"${cfg['costs']['commission_per_share']}/share"),
        ("Slippage", f"${cfg['costs']['slippage_per_share']}/share each side"),
        ("Fill delay", f"{cfg['costs']['fill_delay_bars']} bar(s) after the decision"),
        ("Shorting", "on" if p.get("allow_short") else "off (long-only)"),
        ("Stop checks", p.get("stop_check")), ("Sizing", f"{cfg['sizing']['mode']}, max {cfg['sizing']['max_leverage']}x"),
        ("Parameters", ", ".join(f"{k}={v}" for k, v in p.items())),
        ("Starting equity", f"${cfg['account']['starting_equity']:,}"),
    ], columns=["Item", "Value"])

    css = """
:root{color-scheme:light}
*{box-sizing:border-box}
body{margin:0;background:#f9f9f7;color:#0b0b0b;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;line-height:1.55}
main{max-width:1180px;margin:0 auto;padding:28px 16px 64px;display:flex;flex-direction:column;gap:22px}
h1{font-size:30px;line-height:1.2;margin:0;font-weight:600}
h2{font-size:21px;margin:0;font-weight:600}
h3{font-size:15px;margin:0 0 6px;font-weight:600}
h4{font-size:14px;margin:18px 0 4px;font-weight:500;color:#52514e}
.eyebrow{font-size:12px;letter-spacing:.06em;color:#52514e;text-transform:uppercase}
section{background:#fcfcfb;border:1px solid rgba(11,11,11,.10);border-radius:12px;padding:22px;display:flex;flex-direction:column;gap:14px}
.sub{color:#52514e;margin:0;font-size:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.tile{border:1px solid rgba(11,11,11,.10);border-radius:10px;padding:12px 14px;display:flex;flex-direction:column;gap:4px;background:#fff}
.tl{font-size:13px;color:#52514e}.tv{font-size:22px;font-weight:600}.ts{font-size:12px;color:#52514e}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}
.scroll{overflow-x:auto;border:1px solid rgba(11,11,11,.10);border-radius:10px}
.tbl{width:100%;border-collapse:collapse;font-size:13px;min-width:520px}
.tbl th{background:#f0efec;text-align:left;font-weight:500;padding:8px 12px;color:#52514e}
.tbl td{padding:8px 12px;border-top:1px solid #ecebe6;font-variant-numeric:tabular-nums}
.mono{font-variant-numeric:tabular-nums}
.badge{font-size:12px;font-weight:600;border-radius:999px;padding:3px 10px;white-space:nowrap}
.badge.ok{color:#006300;background:#e5f3e5}.badge.bad{color:#9b1c1c;background:#fbe9e9}.badge.na{color:#52514e;background:#f0efec}
.verdict{font-size:18px;font-weight:600;border-radius:10px;padding:10px 16px;display:inline-flex;gap:8px;align-items:center}
.verdict.ok{color:#006300;background:#e5f3e5}.verdict.bad{color:#9b1c1c;background:#fbe9e9}
.banner{border-radius:10px;padding:12px 16px;font-size:14px}
.banner.warn{background:#fdf3dc;border:1px solid #f0cf83;color:#5a3d00}.banner.info{background:#e8f0fb;border:1px solid #b7d3f6;color:#104281}
.note{font-size:12.5px;color:#52514e;margin:4px 0 0}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:#52514e}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
code{background:#f0efec;border-radius:4px;padding:1px 5px}
footer{font-size:12px;color:#52514e}
"""

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backtest {html.escape(cfg['strategy'])} {html.escape(period)}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js" charset="utf-8"></script>
<style>{css}</style></head>
<body><main>
<header style="display:flex;flex-direction:column;gap:8px">
  <span class="eyebrow">Run {html.escape(run_id)} · {dt.datetime.now():%Y-%m-%d %H:%M}</span>
  <h1>Backtest report: {html.escape(cfg['strategy'].replace('_', ' '))} ({html.escape(period)})</h1>
  <p class="sub">{html.escape(cfg['symbol'])} · {start} to {end} · {'long and short' if p.get('allow_short') else 'long-only'} ·
  costs ${cfg['costs']['commission_per_share']} + ${cfg['costs']['slippage_per_share']} per share each side</p>
</header>
{banner}

<section><span class="eyebrow">1 · Verdict</span>
<div style="display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between">
<h2>Does it pass the scorecard?</h2>{verdict}</div>
<div class="tiles">{kpis}</div>{scorecard}
<p class="note">Checks follow the agreed scorecard. "Not run" checks need a parameter sweep first and do not count toward the verdict.</p>
</section>

<section><span class="eyebrow">2 · Equity and drawdown</span><h2>Strategy vs. holding SPY</h2>
<div class="legend"><span><span class="sw" style="background:{PERIOD_FILL['tune']};border:1px solid #b7d3f6"></span>Tune</span>
<span><span class="sw" style="background:{PERIOD_FILL['validate']};border:1px solid #c9c2ee"></span>Validate</span>
<span><span class="sw" style="background:{PERIOD_FILL['holdout']};border:1px solid #f3c3ad"></span>Holdout</span></div>
{equity_chart(res, cfg)}
<p class="note">Worst drawdown {_pct(perf.get('max_drawdown'))} · longest time underwater {perf.get('longest_underwater_days')} trading days ·
up days {_pct(perf.get('pct_up_days'), 0)} · no-trade days {_pct(perf.get('pct_flat_days'), 0)}</p>
</section>

<section><span class="eyebrow">3 · Is it real?</span><h2>Edge vs. luck</h2>
<div class="grid2"><div><h3>Bootstrapped Sharpe ratio (block resampling)</h3>
{hist_chart(boot['samples'], SERIES, [(1.0, '1.0 bar', CRIT), (boot['lo'], '2.5%', MUTED), (boot['hi'], '97.5%', MUTED)], xtitle='Sharpe ratio') if boot else '<p class="note">Too few days.</p>'}
<p class="note">Each resample rebuilds the history from random stretches of real days. The spread shows how much the Sharpe could be luck.</p></div>
<div class="tiles" style="align-content:start">{real_tiles}</div></div>
</section>

<section><span class="eyebrow">4 · Risk</span><h2>How bad can it get?</h2>
<div class="grid2"><div><h3>Worst drawdown across 2,000 reshuffled histories</h3>
{hist_chart(mc['max_dd_samples'], BENCH, [(-0.25, '25% limit', CRIT)], xfmt='.0%', xtitle='Worst drawdown', tail_below=-0.25) if mc else ''}
<p class="note">Median {_pct(mc.get('dd_median'))} · 95th pct {_pct(mc.get('dd_p95'))} · 99th pct {_pct(mc.get('dd_p99'))} ·
longest losing streak (95th pct) {_num(mc.get('streak_p95'), 0)} days</p></div>
<div style="display:flex;flex-direction:column;gap:12px">
<div class="tile" style="background:#fdf3dc;border-color:#f0cf83"><span class="tl">Suggested daily loss limit (rule of thumb)</span>
<span class="tv">{_pct(loss_limit)} of account</span><span class="ts">1.25 × the 1-in-100 worst day. Final limit is your call.</span></div>
<h3>Worst days</h3><div class="scroll">{_table(worst_tbl)}</div></div></div>
</section>

<section><span class="eyebrow">5 · Where the edge comes from</span><h2>Market days vs. our days</h2>
<div class="grid2"><div><h3>Our day vs. SPY's day</h3>{scatter_market(res, a)}
<p class="note">Alpha {_pct(a.alpha.get('alpha_annual'))}/yr (t = {_num(a.alpha.get('alpha_t'), 1)}) · beta {_num(a.alpha.get('beta'))} ·
correlation {_num(a.alpha.get('correlation'))}. A flat line means returns don't depend on market direction.</p></div>
<div><h3>Average day by market direction</h3>{cond_html}{('<h3>Average day by volatility regime</h3>' + vol_html) if vol_html else ''}</div></div>
<div class="scroll">{_table(_cond_table(cond['market_day']))}</div>
</section>

<section><span class="eyebrow">6 · Stability</span><h2>Does it hold up across time, settings and costs?</h2>
<div class="grid2"><div><h3>Return by year</h3>{yearly_chart(a)}</div><div><h3>Rolling 6-month Sharpe</h3>{rolling_chart(a)}</div>
<div><h3>Parameter heatmap</h3>{heat_html}
<p class="note">Outlined cell = chosen setting. A broad plateau is good; a lone bright cell is a warning.</p></div>
<div><h3>Sharpe vs. slippage</h3>{cost_chart(a, cfg)}
<p class="note">Break-even slippage: {('none within 5¢' if not np.isfinite(a.break_even) else f'{a.break_even * 100:.2f}¢ per share each side')} ·
Sharpe with commission and slippage doubled: {_num(a.sharpe_2x)}</p></div></div>
<div class="scroll">{_table(a.yearly.reset_index().rename(columns={'date': 'Year', 'index': 'Year', 'strategy': 'Strategy', 'market': 'SPY'}).assign(Strategy=lambda x: x['Strategy'].map('{:+.1%}'.format), SPY=lambda x: x['SPY'].map('{:+.1%}'.format)))}</div>
</section>

<section><span class="eyebrow">7 · Trades</span><h2>Trade-level detail</h2>{trade_section}</section>

<section><span class="eyebrow">8 · Run details</span><h2>Reproduce this result</h2><div class="scroll">{_table(details)}</div></section>

<footer>Research tool, not investment advice. Backtests overstate live results; past performance does not guarantee future returns.</footer>
</main></body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return out_path
