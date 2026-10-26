"""Page 4 - Acquisition and marketing ROI."""
from __future__ import annotations

import streamlit as st

from app.ui import Ctx, empty_state, note
from clv import charts
from clv import config as C
from clv.marketing import allocation_signals, campaign_performance, channel_performance


def render(ctx: Ctx) -> None:
    st.title("Acquisition & marketing ROI")
    st.caption("Revenue-based channel economics. Nothing here is profit ROI or incremental impact.")
    if ctx.empty or ctx.customers.empty:
        return empty_state()
    gm = st.slider("Assumed gross margin (illustrative, not in the data)", 0.5, 0.95, C.DEFAULT_GROSS_MARGIN, 0.05,
                   help="Used only for the margin-adjusted ratio and payback months.")
    perf = channel_performance(ctx.customers, ctx.revenue, ctx.snap, gm)
    if perf.empty:
        return empty_state()

    a, b, c = st.columns(3)
    a.plotly_chart(charts.channel_bar(perf, "avg_cac", "Average CAC by channel"), width="stretch")
    b.plotly_chart(charts.channel_bar(perf, "avg_predicted_clv", "Avg predicted 12-month CLV (active customers)"), width="stretch")
    c.plotly_chart(charts.channel_bar(perf, "clv_to_cac", "CLV-to-CAC (revenue-based)", ",.1f", "x"), width="stretch")

    d, e = st.columns(2)
    d.plotly_chart(charts.revenue_share_pie(perf), width="stretch")
    e.plotly_chart(charts.channel_bar(perf, "realised_revenue_to_cac", "Realised revenue per acquired customer / CAC", ",.1f", "x"),
                   width="stretch")
    note("Two lenses: predicted CLV covers currently active customers only (survivors). Realised revenue / CAC covers every acquired "
         "customer including those who churned, so it is the more conservative view.")

    st.subheader("Channel scorecard")
    cols = {"customers_acquired": "Customers acquired", "active_customers": "Active", "avg_cac": "Avg CAC",
            "avg_predicted_clv": "Avg predicted CLV", "clv_to_cac": "CLV-to-CAC", "gm_adj_clv_to_cac": "Margin-adj. CLV-to-CAC",
            "realised_revenue_to_cac": "Realised rev / CAC", "actual_revenue_to_date": "Actual revenue to date",
            "payback_months": "CAC payback (months, at assumed margin)"}
    st.dataframe(perf[list(cols)].rename(columns=cols).style.format({
        "Customers acquired": "{:,.0f}", "Active": "{:,.0f}", "Avg CAC": "${:,.0f}", "Avg predicted CLV": "${:,.0f}",
        "CLV-to-CAC": "{:.1f}x", "Margin-adj. CLV-to-CAC": "{:.1f}x", "Realised rev / CAC": "{:.1f}x",
        "Actual revenue to date": "${:,.0f}", "CAC payback (months, at assumed margin)": "{:.1f}"}, na_rep="n/a"),
        width="stretch")

    st.subheader("Campaign performance (platform-attributed)")
    camp = campaign_performance(ctx.marketing)
    if camp.empty:
        empty_state("No campaign rows in the selected period/channels.")
    else:
        st.dataframe(camp.rename(columns={
            "campaign_cost": "Campaign cost", "attributed_new_customers": "Attributed customers",
            "attributed_revenue": "Attributed revenue (first 3 months)", "cost_per_attributed_customer": "Cost / attributed customer",
            "attributed_revenue_per_cost": "Attributed revenue per $ cost"}).style.format({
            "Campaign cost": "${:,.0f}", "Attributed customers": "{:,.0f}", "Attributed revenue (first 3 months)": "${:,.0f}",
            "Cost / attributed customer": "${:,.0f}", "Attributed revenue per $ cost": "{:.2f}"}, na_rep="n/a"),
            width="stretch")
        note("Attributed revenue per dollar is a revenue ratio over the first three months, not profit and not incremental.")

    st.subheader("Allocation opportunities")
    sig = allocation_signals(perf)
    st.dataframe(sig[["spend_share", "clv_to_cac", "realised_revenue_to_cac", "signal"]].rename(columns={
        "spend_share": "Share of CAC spend", "clv_to_cac": "CLV-to-CAC", "realised_revenue_to_cac": "Realised rev / CAC",
        "signal": "Signal"}).style.format({"Share of CAC spend": "{:.0%}", "CLV-to-CAC": "{:.1f}x", "Realised rev / CAC": "{:.1f}x"}, na_rep="n/a"),
        width="stretch")
    st.warning("These signals are observational. Channels attract different customer mixes (e.g. Direct Sales targets larger accounts), so a high "
               "ratio does not prove that extra spend would produce proportional new revenue. Validate with a holdout or geo/budget experiment "
               "before reallocating budget.", icon="⚠️")