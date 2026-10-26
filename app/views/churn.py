"""Page 3 - Churn and retention."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui import Ctx, empty_state, kpi, note
from clv import charts
from clv.churn import ACTIONS, LABELS, flag_labels, recommend_actions
from clv.metrics import format_pct, format_usd, monthly_churn, period_churn_rate


def render(ctx: Ctx) -> None:
    st.title("Churn & retention")
    st.caption("Risk tiers come from a validated probability model; the rule-based signals explain why an account looks risky. "
               "Neither says a customer *will* churn.")
    if ctx.empty:
        return empty_state()
    s = ctx.snap

    churn = monthly_churn(ctx.revenue, ctx.events)
    t0, t1 = ctx.month_range
    start = t0 if t0 > pd.Timestamp(ctx.data["revenue_monthly"]["month"].min()) else None
    c = st.columns(4)
    pooled = churn["cancellations"].sum() / churn["active_at_start"].sum() if churn["active_at_start"].sum() else float("nan")
    kpi(c[0], "Avg monthly logo churn", format_pct(pooled, 2), "Cancellations / customers subscribed at start of month, pooled over the period.")
    if start is not None:
        coh = period_churn_rate(ctx.revenue, ctx.events, start, t1)
        kpi(c[1], "Period cohort churn", format_pct(coh, 1), "Of customers subscribed at the start of the period, share that cancelled by its end.")
    else:
        kpi(c[1], "Period cohort churn", "n/a", "Choose a period start after the first month to see cohort churn.")
    kpi(c[2], "High-risk accounts", f"{(s['risk_tier'] == 'High').sum():,}", "Estimated 12-month churn probability >= 35%.")
    hv = s[(s["segment"] == "High Value") & s["risk_tier"].isin(["Medium", "High"])]
    kpi(c[3], "High-value at risk", f"{len(hv):,}", f"{format_usd(hv['predicted_12m'].sum())} predicted 12-month revenue.")

    l, r = st.columns(2)
    l.plotly_chart(charts.churn_trend(churn), width="stretch")
    r.plotly_chart(charts.tier_bar(s), width="stretch")

    val = ctx.report.get("risk_validation", {})
    with st.expander("How reliable is the risk tier? (out-of-time backtest)"):
        if val:
            st.write(f"On {val['n']:,} customers scored on {ctx.report['splits']['test_snapshots'][0]} (base 12-month churn rate "
                     f"{val['base_rate']:.1%}), the churn model's AUC was **{val['auc_churn_probability']:.2f}** vs "
                     f"**{val['auc_rule_signal_score']:.2f}** for the rule-signal score.")
            st.dataframe(pd.DataFrame(val["by_tier"]).T.rename(columns={"customers": "Customers", "churn_rate_12m": "Observed 12m churn"})
                         .style.format({"Observed 12m churn": "{:.1%}"}), width="content")
            note("Tiers are monotonic but imperfect: many Low-tier accounts still churn and many High-tier accounts stay.")

    st.subheader("High-value customers with elevated churn risk")
    if hv.empty:
        empty_state("No High Value accounts with Medium/High risk in this selection.")
    else:
        st.dataframe(_table(hv).head(50), width="stretch", hide_index=True, column_config=_cfg())

    st.subheader("Usage decline and payment-failure indicators")
    a, b = st.columns(2)
    a.plotly_chart(charts.flag_frequency(s, LABELS), width="stretch")
    b.plotly_chart(charts.value_vs_risk(s), width="stretch")

    st.subheader("At-risk customer list")
    tier = st.multiselect("Risk tier", ["High", "Medium"], default=["High"])
    risky = s[s["risk_tier"].isin(tier)]
    if risky.empty:
        empty_state("No customers in the selected risk tiers.")
    else:
        st.dataframe(_table(risky).head(200), width="stretch", hide_index=True, height=340, column_config=_cfg())
        st.caption(f"{len(risky):,} accounts; top 200 by predicted revenue shown.")

    st.subheader("Recommended retention actions")
    pool = hv if len(hv) else risky
    if pool.empty:
        empty_state("No accounts to recommend actions for.")
    else:
        pick = st.selectbox("Account", pool.sort_values("predicted_12m", ascending=False)["customer_id"], key="churn_pick")
        row = pool[pool["customer_id"] == pick].iloc[0]
        for a_ in recommend_actions(row["risk_flags"], str(row["segment"]), str(row["risk_tier"])):
            st.markdown(f"- {a_}")
    with st.expander("Signal-to-action playbook"):
        st.dataframe(pd.DataFrame({"Signal": [LABELS[k] for k in ACTIONS], "Recommended action": list(ACTIONS.values())}),
                     hide_index=True, width="stretch")


def _table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("predicted_12m", ascending=False).copy()
    d["Signals"] = d["risk_flags"].apply(lambda f: "; ".join(flag_labels(f)))
    d["Churn probability"] = d["churn_probability"] * 100
    return d[["customer_id", "segment", "predicted_12m", "Churn probability", "risk_tier", "logins_trend_ratio",
              "payment_failures_6m", "unresolved_180d", "Signals"]].rename(columns={
        "customer_id": "Customer", "segment": "Segment", "predicted_12m": "Predicted 12m revenue", "risk_tier": "Risk tier",
        "logins_trend_ratio": "Login trend (3m / prior 3m)", "payment_failures_6m": "Payment failures (6m)",
        "unresolved_180d": "Unresolved tickets"})


def _cfg() -> dict:
    return {"Predicted 12m revenue": st.column_config.NumberColumn(format="$%.0f"),
            "Churn probability": st.column_config.NumberColumn(format="%.0f%%"),
            "Login trend (3m / prior 3m)": st.column_config.NumberColumn(format="%.2f")}