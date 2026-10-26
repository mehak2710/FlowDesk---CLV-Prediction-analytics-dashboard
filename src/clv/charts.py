"""Reusable Plotly figure builders (pure functions: DataFrame in, Figure out)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import config as C

SEGMENT_COLORS = {"High Value": "#1f6feb", "Medium Value": "#8bb4f7", "Low Value": "#c7ccd4"}
TIER_COLORS = {"Low": "#2da44e", "Medium": "#d4a72c", "High": "#cf222e"}
PALETTE = ["#1f6feb", "#2da44e", "#d4a72c", "#8250df", "#cf222e", "#57606a"]


def _style(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
                      showlegend=legend, legend=dict(orientation="h", y=-0.2),
                      font=dict(family="Inter, Segoe UI, sans-serif", size=12), title_font_size=14)
    return fig


def empty_figure(message: str = "No data for the current filters") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color="#57606a"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, 260, False)


def revenue_trend(monthly: pd.DataFrame) -> go.Figure:
    """monthly: month, recognized_revenue, mrr."""
    if monthly.empty:
        return empty_figure()
    fig = go.Figure()
    fig.add_bar(x=monthly["month"], y=monthly["recognized_revenue"], name="Recognised revenue",
                marker_color="#8bb4f7", hovertemplate="%{x|%b %Y}<br>Revenue: $%{y:,.0f}<extra></extra>")
    fig.add_scatter(x=monthly["month"], y=monthly["mrr"], name="MRR", line=dict(color="#1f6feb", width=3),
                    hovertemplate="%{x|%b %Y}<br>MRR: $%{y:,.0f}<extra></extra>")
    fig.update_layout(title="Monthly revenue and MRR", yaxis_tickprefix="$", yaxis_tickformat="~s")
    return _style(fig)


def segment_donut(summary: pd.DataFrame) -> go.Figure:
    if summary.empty:
        return empty_figure()
    fig = go.Figure(go.Pie(labels=summary.index, values=summary["customers"], hole=0.6,
                           marker_colors=[SEGMENT_COLORS[s] for s in summary.index],
                           hovertemplate="%{label}<br>%{value:,} customers (%{percent})<extra></extra>"))
    fig.update_layout(title="Customers by value segment")
    return _style(fig)


def clv_distribution(df: pd.DataFrame, thresholds: dict) -> go.Figure:
    if df.empty:
        return empty_figure()
    fig = px.histogram(df, x="predicted_12m", color="segment", nbins=50, log_y=False,
                       color_discrete_map=SEGMENT_COLORS, category_orders={"segment": C.SEGMENTS})
    fig.update_traces(hovertemplate="Predicted 12m revenue: $%{x:,.0f}<br>Customers: %{y}<extra></extra>")
    for k, label in [("low_upper", "Low | Medium"), ("high_lower", "Medium | High")]:
        fig.add_vline(x=thresholds[k], line_dash="dot", line_color="#57606a",
                      annotation_text=f"{label}: ${thresholds[k]:,.0f}", annotation_position="top")
    fig.update_layout(title="Distribution of predicted 12-month revenue", xaxis_title="Predicted 12-month revenue (USD)",
                      yaxis_title="Customers", barmode="stack", xaxis_tickprefix="$")
    return _style(fig)


def predicted_vs_actual(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["actual_12m"])
    if d.empty:
        return empty_figure("Actual 12-month revenue is not yet observed for this snapshot")
    fig = px.scatter(d, x="predicted_12m", y="actual_12m", color="segment", opacity=0.55, log_x=True, log_y=True,
                     color_discrete_map=SEGMENT_COLORS, category_orders={"segment": C.SEGMENTS},
                     hover_data={"customer_id": True, "predicted_12m": ":$,.0f", "actual_12m": ":$,.0f"})
    lo = max(min(d["predicted_12m"].min(), d["actual_12m"].clip(lower=1).min()), 1)
    hi = max(d["predicted_12m"].max(), d["actual_12m"].max())
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#57606a", dash="dash"), name="Perfect prediction")
    fig.update_layout(title="Predicted vs actual 12-month revenue (log scale)",
                      xaxis_title="Predicted (USD)", yaxis_title="Actual (USD)")
    return _style(fig, 400)


def segment_pred_vs_actual(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["actual_12m"])
    if d.empty:
        return empty_figure("Actual 12-month revenue is not yet observed for this snapshot")
    g = d.groupby("segment", observed=True)[["predicted_12m", "actual_12m"]].sum().reindex(C.SEGMENTS).dropna(how="all")
    fig = go.Figure()
    fig.add_bar(x=g.index, y=g["predicted_12m"], name="Predicted", marker_color="#1f6feb",
                hovertemplate="%{x}<br>Predicted: $%{y:,.0f}<extra></extra>")
    fig.add_bar(x=g.index, y=g["actual_12m"], name="Actual", marker_color="#8bb4f7",
                hovertemplate="%{x}<br>Actual: $%{y:,.0f}<extra></extra>")
    fig.update_layout(title="Segment totals: predicted vs actual", barmode="group", yaxis_tickprefix="$", yaxis_tickformat="~s")
    return _style(fig)


def churn_trend(churn: pd.DataFrame) -> go.Figure:
    if churn.empty:
        return empty_figure()
    d = churn.dropna(subset=["churn_rate"])
    fig = go.Figure()
    fig.add_scatter(x=d["month"], y=d["churn_rate"], mode="lines+markers", line=dict(color="#cf222e"), name="Monthly logo churn",
                    customdata=np.stack([d["cancellations"], d["active_at_start"]], axis=-1),
                    hovertemplate="%{x|%b %Y}<br>Churn: %{y:.2%}<br>%{customdata[0]:.0f} of %{customdata[1]:.0f} customers<extra></extra>")
    fig.update_layout(title="Monthly logo churn rate", yaxis_tickformat=".1%", yaxis_rangemode="tozero")
    return _style(fig, legend=False)


def tier_bar(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure()
    g = df.groupby("risk_tier", observed=True).size().reindex(["Low", "Medium", "High"]).fillna(0)
    fig = go.Figure(go.Bar(x=g.index, y=g.values, marker_color=[TIER_COLORS[t] for t in g.index],
                           hovertemplate="%{x} risk<br>%{y:,.0f} customers<extra></extra>"))
    fig.update_layout(title="Customers by churn-risk tier", yaxis_title="Customers")
    return _style(fig, legend=False)


def value_vs_risk(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure()
    fig = px.scatter(df, x="churn_probability", y="predicted_12m", color="segment", log_y=True, opacity=0.6,
                     color_discrete_map=SEGMENT_COLORS, category_orders={"segment": C.SEGMENTS},
                     hover_data={"customer_id": True, "churn_probability": ":.0%", "predicted_12m": ":$,.0f"})
    fig.add_vline(x=0.15, line_dash="dot", line_color="#57606a")
    fig.add_vline(x=0.35, line_dash="dot", line_color="#57606a")
    fig.update_layout(title="Predicted value vs estimated churn probability",
                      xaxis_title="Estimated 12-month churn probability", yaxis_title="Predicted 12-month revenue (USD, log)",
                      xaxis_tickformat=".0%")
    return _style(fig, 400)


def flag_frequency(df: pd.DataFrame, labels: dict[str, str]) -> go.Figure:
    if df.empty:
        return empty_figure()
    counts = df["risk_flags"].str.split("|").explode().replace("", np.nan).dropna().value_counts()
    if counts.empty:
        return empty_figure("No risk signals in the current selection")
    fig = go.Figure(go.Bar(y=[labels[i] for i in counts.index][::-1], x=counts.values[::-1], orientation="h",
                           marker_color="#d4a72c", hovertemplate="%{y}<br>%{x:,} customers<extra></extra>"))
    fig.update_layout(title="Risk signals present (customers)")
    return _style(fig, 340, False)


def channel_bar(perf: pd.DataFrame, column: str, title: str, fmt: str = "$,.0f", suffix: str = "") -> go.Figure:
    if perf.empty or perf[column].dropna().empty:
        return empty_figure()
    d = perf[column].dropna()
    fig = go.Figure(go.Bar(x=d.index, y=d.values, marker_color=PALETTE[: len(d)],
                           hovertemplate="%{x}<br>%{y:" + fmt + "}" + suffix + "<extra></extra>"))
    fig.update_layout(title=title)
    return _style(fig, legend=False)


def revenue_share_pie(perf: pd.DataFrame) -> go.Figure:
    if perf.empty:
        return empty_figure()
    fig = go.Figure(go.Pie(labels=perf.index, values=perf["actual_revenue_to_date"], hole=0.55, marker_colors=PALETTE,
                           hovertemplate="%{label}<br>$%{value:,.0f} (%{percent})<extra></extra>"))
    fig.update_layout(title="Revenue contribution by channel (to date)")
    return _style(fig)


def driver_bar(drivers: pd.DataFrame) -> go.Figure:
    d = drivers.iloc[::-1]
    fig = go.Figure(go.Bar(y=d["driver"], x=d["contribution"], orientation="h",
                           marker_color=np.where(d["contribution"] >= 0, "#2da44e", "#cf222e"),
                           hovertemplate="%{y}<br>%{x:+$,.0f} vs a typical customer<extra></extra>"))
    fig.update_layout(title="What moves this prediction (vs a typical customer)", xaxis_tickprefix="$")
    return _style(fig, 360, False)


def customer_series(df: pd.DataFrame, x: str, ys: dict[str, str], title: str, money: bool = False) -> go.Figure:
    if df.empty:
        return empty_figure("No history for this customer")
    fig = go.Figure()
    for i, (col, name) in enumerate(ys.items()):
        fig.add_scatter(x=df[x], y=df[col], name=name, mode="lines+markers", line=dict(color=PALETTE[i]))
    fig.update_layout(title=title, yaxis_tickprefix="$" if money else "")
    return _style(fig, 300)