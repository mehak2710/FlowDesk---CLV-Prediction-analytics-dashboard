"""Page 2 - CLV prediction and segmentation."""
from __future__ import annotations

import streamlit as st

from app.ui import Ctx, empty_state, kpi, note
from clv import charts
from clv.churn import flag_labels
from clv.metrics import format_pct, format_usd
from clv.modeling import regression_metrics
from clv.segmentation import segment_summary

TABLE_COLS = {"customer_id": "Customer", "segment": "Segment", "predicted_12m": "Predicted 12m revenue",
              "actual_12m": "Actual 12m revenue", "mrr_now": "MRR", "plan": "Plan", "company_size": "Size",
              "industry": "Industry", "acquisition_channel": "Channel", "churn_probability": "Churn probability",
              "risk_tier": "Risk tier"}


def render(ctx: Ctx) -> None:
    st.title("CLV prediction & segmentation")
    st.caption("Predicted CLV here = expected revenue over the 12 months after the snapshot date. It is not full lifetime value and not profit.")
    if ctx.empty:
        return empty_state()
    s, thr = ctx.snap, ctx.thresholds
    note(f"Segments use fixed cut-offs from training-period quantiles: Low < ${thr['low_upper']:,.0f} <= Medium < ${thr['high_lower']:,.0f} <= High.")

    seg = segment_summary(s)
    a, b = st.columns([2, 3])
    a.plotly_chart(charts.segment_donut(seg), width="stretch")
    b.plotly_chart(charts.clv_distribution(s, thr), width="stretch")

    st.subheader("Segment-level performance")
    show = seg.rename(columns={
        "customers": "Customers", "pct_of_customers": "% of customers", "predicted_12m_revenue": "Predicted 12m revenue",
        "avg_predicted_clv": "Avg predicted CLV", "actual_revenue_to_date": "Actual revenue to date", "avg_mrr": "Avg MRR",
        "avg_logins_3m": "Avg logins (3m)", "avg_adoption": "Feature adoption", "avg_cac": "Avg CAC",
        "observed_12m_churn_rate": "Observed 12m churn", "pct_elevated_risk": "% Medium/High risk", "clv_to_cac": "CLV-to-CAC"})
    st.dataframe(show.style.format({
        "% of customers": "{:.1%}", "Predicted 12m revenue": "${:,.0f}", "Avg predicted CLV": "${:,.0f}",
        "Actual revenue to date": "${:,.0f}", "Avg MRR": "${:,.0f}", "Avg logins (3m)": "{:,.0f}", "Feature adoption": "{:.0%}",
        "Avg CAC": "${:,.0f}", "Observed 12m churn": "{:.1%}", "% Medium/High risk": "{:.0%}", "CLV-to-CAC": "{:.1f}x"}, na_rep="n/a"),
        width="stretch")
    note("Observed 12m churn exists only for the backtest snapshot. 'Actual revenue to date' is cumulative recognised revenue up to the snapshot (all history, not just 12 months).")

    st.subheader("Predicted vs actual")
    if ctx.snapshot_key == "current":
        st.info("The current snapshot's 12-month outcomes have not happened yet. Switch to the backtest snapshot in the sidebar to compare predictions with actuals.", icon="ℹ️")
    else:
        m = regression_metrics(s["actual_12m"], s["predicted_12m"])
        c = st.columns(4)
        kpi(c[0], "MAE", format_usd(m["MAE"]))
        kpi(c[1], "RMSE", format_usd(m["RMSE"]))
        kpi(c[2], "R-squared", f"{m['R2']:.3f}")
        kpi(c[3], "Portfolio bias", f"{format_usd(m['Bias'])} / customer", "Mean predicted minus mean actual.")
        l, r = st.columns(2)
        l.plotly_chart(charts.predicted_vs_actual(s), width="stretch")
        r.plotly_chart(charts.segment_pred_vs_actual(s), width="stretch")

    st.subheader("Customer table")
    q = st.text_input("Search customer ID, industry, plan or channel", "").strip().lower()
    t = s
    if q:
        hay = (s["customer_id"] + " " + s["industry"] + " " + s["plan"] + " " + s["acquisition_channel"]).str.lower()
        t = s[hay.str.contains(q, regex=False)]
    if t.empty:
        empty_state("No customers match that search.")
    else:
        view = t.sort_values("predicted_12m", ascending=False)[list(TABLE_COLS)].rename(columns=TABLE_COLS)
        view["Churn probability"] = view["Churn probability"] * 100
        st.dataframe(view, width="stretch", hide_index=True, height=360, column_config={
            "Predicted 12m revenue": st.column_config.NumberColumn(format="$%.0f"),
            "Actual 12m revenue": st.column_config.NumberColumn(format="$%.0f"),
            "MRR": st.column_config.NumberColumn(format="$%.0f"),
            "Churn probability": st.column_config.NumberColumn(format="%.0f%%"),
        })
        st.caption(f"{len(t):,} customers shown.")

    st.subheader("Individual customer lookup")
    cid = st.selectbox("Customer", s.sort_values("predicted_12m", ascending=False)["customer_id"], key="clv_lookup")
    r = s[s["customer_id"] == cid].iloc[0]
    c = st.columns(5)
    kpi(c[0], "Predicted 12m revenue", format_usd(r["predicted_12m"]))
    kpi(c[1], "Segment", str(r["segment"]))
    kpi(c[2], "Current MRR", format_usd(r["mrr_now"]))
    kpi(c[3], "Churn probability", format_pct(r["churn_probability"], 0), "Estimated chance of cancelling within 12 months. A probability, not a verdict.")
    kpi(c[4], "Risk tier", str(r["risk_tier"]))
    if r["actual_12m"] == r["actual_12m"]:
        st.write(f"Actual 12-month revenue: **{format_usd(r['actual_12m'])}**")
    if r["risk_flags"]:
        st.write("Risk signals: " + "; ".join(flag_labels(r["risk_flags"])))
    note("Open the Customer Explorer page for history, drivers and recommended actions.")