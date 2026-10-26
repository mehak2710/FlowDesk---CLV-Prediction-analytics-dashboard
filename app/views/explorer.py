"""Page 5 - Customer explorer."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.data import load_model
from app.ui import Ctx, empty_state, kpi, note
from clv import charts
from clv.churn import flag_labels, recommend_actions
from clv.features import FEATURE_COLUMNS
from clv.metrics import format_pct, format_usd
from clv.modeling import explain_customer


def render(ctx: Ctx) -> None:
    st.title("Customer explorer")
    if ctx.customers.empty:
        return empty_state()
    d = ctx.data
    q = st.text_input("Search by customer ID", "").strip().upper()
    ids = ctx.customers["customer_id"]
    if q:
        ids = ids[ids.str.contains(q, regex=False)]
    if ids.empty:
        return empty_state("No customer ID matches that search within the current filters.")
    scored = ctx.snap.sort_values("predicted_12m", ascending=False)["customer_id"]
    ordered = pd.concat([scored[scored.isin(ids)], ids[~ids.isin(scored)]]).head(500)
    cid = st.selectbox("Customer", ordered, help="Scored (active) customers first, then churned/ineligible ones. Up to 500 shown.")

    cu = d["customers"].set_index("customer_id").loc[cid]
    st.subheader(f"{cid} - {cu['industry']}, {cu['company_size']} employees")
    st.caption(f"Signed up {cu['signup_date']:%b %d, %Y} via {cu['acquisition_channel']} ({cu['sales_region']}). "
               f"Acquisition cost: {format_usd(cu['acquisition_cost'])}.")

    row = ctx.snap[ctx.snap["customer_id"] == cid]
    if row.empty:
        st.info("This customer is not scored at the selected snapshot (no active subscription or under one month of history). "
                "Their history is shown below.", icon="ℹ️")
    else:
        r = row.iloc[0]
        c = st.columns(5)
        kpi(c[0], "Predicted 12m revenue", format_usd(r["predicted_12m"]))
        kpi(c[1], "Segment", str(r["segment"]))
        kpi(c[2], "Churn probability", format_pct(r["churn_probability"], 0), "Estimated chance of cancelling within 12 months.")
        kpi(c[3], "Risk tier", str(r["risk_tier"]))
        kpi(c[4], "Plan / billing", f"{r['plan']} / {r['billing_cycle']}")
        if r["actual_12m"] == r["actual_12m"]:
            st.write(f"Actual revenue in the following 12 months: **{format_usd(r['actual_12m'])}**")

    rev = d["revenue_monthly"].query("customer_id == @cid").sort_values("month")
    rev = rev.assign(mrr=rev["recurring_revenue"] + rev["expansion_revenue"] - rev["contraction_revenue"])
    use = d["usage_monthly"].query("customer_id == @cid").sort_values("observation_month")
    tabs = st.tabs(["Revenue", "Engagement", "Support & billing", "Prediction drivers", "Recommended actions"])
    with tabs[0]:
        st.plotly_chart(charts.customer_series(rev, "month", {"recognized_revenue": "Recognised revenue", "mrr": "MRR"},
                                               "Monthly revenue and MRR", money=True), width="stretch")
    with tabs[1]:
        a, b = st.columns(2)
        a.plotly_chart(charts.customer_series(use, "observation_month", {"monthly_logins": "Logins", "active_days": "Active days"},
                                              "Logins and active days"), width="stretch")
        b.plotly_chart(charts.customer_series(use, "observation_month", {"feature_adoption_rate": "Feature adoption"},
                                              "Feature adoption rate"), width="stretch")
    with tabs[2]:
        tk = d["support_tickets"].query("customer_id == @cid").sort_values("created_date", ascending=False)
        ev = d["billing_events"].query("customer_id == @cid").sort_values("event_date", ascending=False)
        st.markdown("**Support tickets** (blank resolution time = unresolved)")
        st.dataframe(tk.drop(columns="customer_id"), hide_index=True, width="stretch") if len(tk) else st.caption("No tickets.")
        st.markdown("**Billing and plan events**")
        st.dataframe(ev.drop(columns=["customer_id"]), hide_index=True, width="stretch") if len(ev) else st.caption("No events.")
    with tabs[3]:
        if row.empty:
            empty_state("Drivers are available for scored customers only.")
        else:
            bundle = load_model(ctx.snapshot_key)
            drivers = explain_customer(bundle["pipeline"], row[FEATURE_COLUMNS].reset_index(drop=True), bundle["reference"])
            st.plotly_chart(charts.driver_bar(drivers), width="stretch")
            note("Each bar is the change in this customer's predicted revenue when that group of features is replaced by typical "
                 "(median/mode) values. It describes the model's reasoning, not causal effects.")
    with tabs[4]:
        if row.empty:
            empty_state("Recommendations are available for scored customers only.")
        else:
            r = row.iloc[0]
            if r["risk_flags"]:
                st.write("Signals: " + "; ".join(flag_labels(r["risk_flags"])))
            for a_ in recommend_actions(r["risk_flags"], str(r["segment"]), str(r["risk_tier"])):
                st.markdown(f"- {a_}")