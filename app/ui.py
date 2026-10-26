"""Shared UI: styling, KPI cards, sidebar filters and the filtered-data context."""
from __future__ import annotations

import html
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from clv import config as C
from clv.metrics import format_usd

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1280px;}
.kpi {background: #f6f8fa; border: 1px solid #d8dee4; border-radius: 10px; padding: 12px 14px; margin-bottom: 0.6rem; min-height: 92px;}
.kpi-label {font-size: 0.8rem; color: #57606a; line-height: 1.25; white-space: normal; word-break: break-word;}
.kpi-value {font-size: 1.6rem; font-weight: 600; color: #1f2328; line-height: 1.25; margin-top: 4px; white-space: normal; word-break: break-word;}
h1, h2, h3 {letter-spacing: -0.01em;}
.caption-note {color: #57606a; font-size: 0.85rem;}
</style>
"""

SNAPSHOT_LABELS = {
    "current": f"{C.SCORING_SNAPSHOT} - current scoring (outcomes not yet observed)",
    "backtest": f"{C.TEST_SNAPSHOTS[0]} - out-of-time backtest (outcomes observed)",
}


@dataclass
class Ctx:
    data: dict
    report: dict
    snapshot_key: str
    snap: pd.DataFrame           # scored customers for the chosen snapshot, after filters
    snap_both: pd.DataFrame      # both snapshots after filters (for comparisons)
    customers: pd.DataFrame      # filtered customer universe (includes churned)
    revenue: pd.DataFrame        # revenue rows for universe within the month range
    events: pd.DataFrame
    marketing: pd.DataFrame
    month_range: tuple
    thresholds: dict
    filters_active: bool

    @property
    def empty(self) -> bool:
        return self.snap.empty


def kpi(col, label: str, value: str, help_: str | None = None) -> None:
    """KPI card that always wraps (never truncates with '...'). Hover shows the help text."""
    def esc(s: str) -> str:
        return html.escape(str(s)).replace("$", "&#36;")  # '$' would otherwise trigger LaTeX rendering
    tip = f' title="{esc(help_)}"' if help_ else ""
    col.markdown(f'<div class="kpi"{tip}><div class="kpi-label">{esc(label)}</div>'
                 f'<div class="kpi-value">{esc(value)}</div></div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f"<div class='caption-note'>{text}</div>", unsafe_allow_html=True)


def empty_state(msg: str = "No customers match the current filters. Widen the filters in the sidebar.") -> None:
    st.info(msg, icon="ℹ️")


def build_context(data: dict, report: dict) -> Ctx:
    cu, snaps = data["customers"], data["customer_snapshots"]
    st.sidebar.markdown("### Filters")
    snapshot_key = st.sidebar.radio("Prediction snapshot", list(SNAPSHOT_LABELS), format_func=SNAPSHOT_LABELS.get,
                                    help="Backtest = predictions made on 2024-12-31 and compared with what actually happened in 2025.")
    plans = st.sidebar.multiselect("Subscription plan", C.PLANS, default=C.PLANS)
    inds = st.sidebar.multiselect("Industry", C.INDUSTRIES, default=C.INDUSTRIES)
    sizes = st.sidebar.multiselect("Company size (employees)", C.SIZE_BANDS, default=C.SIZE_BANDS)
    chans = st.sidebar.multiselect("Acquisition channel", C.CHANNELS, default=C.CHANNELS)
    s_min, s_max = cu["signup_date"].min().date(), cu["signup_date"].max().date()
    signup = st.sidebar.date_input("Signup date range", value=(s_min, s_max), min_value=s_min, max_value=s_max)
    months = sorted(data["revenue_monthly"]["month"].unique())
    mr = st.sidebar.select_slider("Revenue / churn period", options=months, value=(months[0], months[-1]),
                                  format_func=lambda d: pd.Timestamp(d).strftime("%b %Y"))
    st.sidebar.caption("Plan filters the plan held at the snapshot date for scored customers, and the latest plan for "
                       "historical (revenue/churn) views.")

    d0, d1 = (signup if isinstance(signup, tuple) and len(signup) == 2 else (s_min, s_max))
    static = (cu["industry"].isin(inds) & cu["company_size"].isin(sizes) & cu["acquisition_channel"].isin(chans)
              & cu["signup_date"].between(pd.Timestamp(d0), pd.Timestamp(d1) + pd.Timedelta(days=1)))
    static_ids = set(cu.loc[static, "customer_id"])
    universe = cu[static & cu["final_plan"].isin(plans)]
    ids = set(universe["customer_id"])

    sn = snaps[snaps["customer_id"].isin(static_ids) & snaps["plan"].isin(plans)]
    sn_both = sn
    sn = sn[sn["split"] == snapshot_key]
    t0, t1 = pd.Timestamp(mr[0]), pd.Timestamp(mr[1])
    rev = data["revenue_monthly"]
    rev = rev[rev["customer_id"].isin(ids) & rev["month"].between(t0, t1)]
    ev = data["billing_events"]
    ev = ev[ev["customer_id"].isin(ids) & ev["event_date"].between(t0.to_period("M").start_time, t1)]
    mk = data["marketing_channels"]
    mk = mk[mk["acquisition_channel"].isin(chans) & mk["campaign_month"].between(t0, t1)]
    active = (len(plans) < 3 or len(inds) < 8 or len(sizes) < 4 or len(chans) < 5 or (d0, d1) != (s_min, s_max)
              or (t0, t1) != (pd.Timestamp(months[0]), pd.Timestamp(months[-1])))
    return Ctx(data, report, snapshot_key, sn, sn_both, universe, rev, ev, mk, (t0, t1), report["segment_thresholds"], active)