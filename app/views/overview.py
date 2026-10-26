"""Page 1 - Executive overview."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui import Ctx, empty_state, kpi, note
from clv import charts
from clv.marketing import channel_performance
from clv.metrics import clv_to_cac, format_pct, format_usd, monthly_churn, mrr_series
from clv.segmentation import segment_summary


def render(ctx: Ctx) -> None:
    st.title("Executive overview")
    st.caption("Flowdesk (fictional B2B SaaS) - all figures computed from synthetic data and model outputs.")
    if ctx.empty:
        return empty_state()

    snap = ctx.snap
    total_rev = ctx.revenue["recognized_revenue"].sum()
    mrr = mrr_series(ctx.revenue)
    churn = monthly_churn(ctx.revenue, ctx.events)
    avg_churn = (churn["cancellations"].sum() / churn["active_at_start"].sum()) if churn["active_at_start"].sum() else float("nan")
    avg_cac = snap["acquisition_cost"].mean()
    avg_clv = snap["predicted_12m"].mean()

    row1, row2 = st.columns(3), st.columns(3)
    c = [*row1, *row2]  # two rows of three keep labels and values readable on narrow screens
    kpi(c[0], "Historical revenue", format_usd(total_rev), "Recognised revenue (net of discounts and refunds) in the selected period.")
    kpi(c[1], "Avg predicted 12-month CLV", format_usd(avg_clv), "Mean predicted revenue over the next 12 months for scored customers. Not lifetime value or profit.")
    kpi(c[2], "Active customers", f"{len(snap):,}", "Customers with an active subscription at the snapshot date.")
    kpi(c[3], "MRR (last month in period)", format_usd(mrr.iloc[-1]) if len(mrr) else "n/a")
    kpi(c[4], "Avg monthly logo churn", format_pct(avg_churn, 2), "Cancellations / customers subscribed at the start of each month, pooled over the selected period.")
    kpi(c[5], "CLV-to-CAC (revenue-based)", f"{clv_to_cac(avg_clv, avg_cac):,.1f}x" if pd.notna(avg_cac) else "n/a",
        "Avg predicted 12-month revenue / avg acquisition cost. Revenue-based, not profit-based.")

    monthly = (ctx.revenue.groupby("month")["recognized_revenue"].sum().to_frame().join(mrr.rename("mrr")).reset_index())
    left, right = st.columns([3, 2])
    left.plotly_chart(charts.revenue_trend(monthly), width="stretch")
    seg = segment_summary(snap)
    right.plotly_chart(charts.segment_donut(seg), width="stretch")

    l2, r2 = st.columns(2)
    both = ctx.snap_both.groupby("snapshot_date")["predicted_12m"].mean().reset_index()
    if len(both):
        fig = charts.channel_bar(both.assign(label=both["snapshot_date"].dt.strftime("%Y-%m-%d")).set_index("label"),
                                 "predicted_12m", "Avg predicted 12-month CLV by snapshot")
        l2.plotly_chart(fig, width="stretch")
    r2.plotly_chart(charts.churn_trend(churn), width="stretch")

    st.subheader("Key business insights")
    for line in insights(ctx, seg):
        st.markdown(f"- {line}")
    note("Insights are generated from the filtered data and describe associations, not causes.")


def insights(ctx: Ctx, seg: pd.DataFrame) -> list[str]:
    s, out = ctx.snap, []
    if "High Value" in seg.index:
        hv = seg.loc["High Value"]
        share = hv["predicted_12m_revenue"] / seg["predicted_12m_revenue"].sum()
        out.append(f"High Value customers are {hv['pct_of_customers']:.0%} of scored accounts but {share:.0%} of predicted 12-month revenue.")
    hv_risk = s[(s["segment"] == "High Value") & (s["risk_tier"].isin(["High", "Medium"]))]
    if len(hv_risk):
        out.append(f"{len(hv_risk):,} High Value accounts show Medium/High churn risk, carrying {format_usd(hv_risk['predicted_12m'].sum())} of predicted revenue. See Churn & Retention.")
    perf = channel_performance(ctx.customers, ctx.revenue, s)
    p = perf["clv_to_cac"].dropna() if not perf.empty else pd.Series(dtype=float)
    if len(p) >= 2:
        out.append(f"Revenue-based CLV-to-CAC is highest for {p.idxmax()} ({p.max():.1f}x) and lowest for {p.idxmin()} ({p.min():.1f}x). Ratios are correlational and ignore margin.")
    decl = s[s["risk_flags"].str.contains("usage_decline")]
    if len(decl):
        out.append(f"{len(decl):,} active accounts ({len(decl) / len(s):.0%}) show a 25%+ drop in logins vs the prior three months.")
    pf = s[s["payment_failures_6m"] > 0]
    if len(pf):
        out.append(f"{len(pf):,} accounts had a payment failure in the last six months, a billing-recovery opportunity.")
    return out