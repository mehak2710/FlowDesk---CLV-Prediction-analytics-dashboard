"""Data validation: keys, referential integrity, ranges, types, and lifecycle logic.

`validate_tables` returns a list of `Check` results; `assert_valid` raises if any
check with severity "error" fails. Warnings (e.g. missing-value rates) are reported
but do not fail the run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

PRIMARY_KEYS = {
    "customers": ["customer_id"],
    "subscriptions": ["subscription_id"],
    "support_tickets": ["ticket_id"],
    "billing_events": ["event_id"],
    "usage_monthly": ["customer_id", "observation_month"],
    "revenue_monthly": ["customer_id", "month"],
    "marketing_channels": ["acquisition_channel", "campaign_month"],
}
FOREIGN_KEYS = [  # (child, column, parent, column)
    ("subscriptions", "customer_id", "customers", "customer_id"),
    ("usage_monthly", "customer_id", "customers", "customer_id"),
    ("support_tickets", "customer_id", "customers", "customer_id"),
    ("revenue_monthly", "customer_id", "customers", "customer_id"),
    ("billing_events", "customer_id", "customers", "customer_id"),
    ("billing_events", "subscription_id", "subscriptions", "subscription_id"),
]
NON_NEGATIVE = {
    "customers": ["employee_count", "acquisition_cost"],
    "subscriptions": ["monthly_recurring_revenue", "upgrade_count", "downgrade_count", "payment_failure_count"],
    "usage_monthly": ["monthly_logins", "active_days", "features_used", "session_duration_minutes"],
    "revenue_monthly": ["recurring_revenue", "expansion_revenue", "contraction_revenue", "refunds", "discounts"],
    "marketing_channels": ["campaign_cost", "attributed_new_customers", "attributed_revenue"],
    "support_tickets": ["resolution_time_hours"],
}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    severity: str = "error"  # "error" | "warning"


def _month(s: pd.Series) -> pd.Series:
    return s.dt.to_period("M")


def validate_tables(t: dict[str, pd.DataFrame]) -> list[Check]:
    out: list[Check] = []

    def add(name, ok, detail="", severity="error"):
        out.append(Check(name, bool(ok), detail, severity))

    # 1. primary keys / duplicates
    for tbl, cols in PRIMARY_KEYS.items():
        dups = int(t[tbl].duplicated(cols).sum())
        add(f"PK unique: {tbl}({', '.join(cols)})", dups == 0, f"{dups} duplicates")
        nulls = int(t[tbl][cols].isna().any(axis=1).sum())
        add(f"PK not null: {tbl}", nulls == 0, f"{nulls} nulls")

    # 2. referential integrity
    for child, ccol, parent, pcol in FOREIGN_KEYS:
        orphans = int((~t[child][ccol].isin(t[parent][pcol])).sum())
        add(f"FK: {child}.{ccol} -> {parent}.{pcol}", orphans == 0, f"{orphans} orphans")

    # 3. date ranges
    cu = t["customers"]
    add("signup_date within simulation window",
        cu["signup_date"].between(C.FIRST_MONTH.start_time, C.LAST_SIGNUP_MONTH.end_time).all())
    for tbl, col in [("revenue_monthly", "month"), ("usage_monthly", "observation_month")]:
        ok = t[tbl][col].between(C.FIRST_MONTH.start_time, C.DATA_END).all()
        add(f"{tbl}.{col} within window", ok)
        add(f"{tbl}.{col} is month-end", (t[tbl][col] == t[tbl][col] + pd.offsets.MonthEnd(0)).all())
    sub = t["subscriptions"]
    ended = sub.dropna(subset=["subscription_end_date"])
    add("subscription start <= end", (ended["subscription_start_date"] <= ended["subscription_end_date"]).all())
    add("status matches end date",
        ((sub["subscription_status"] == "Active") == sub["subscription_end_date"].isna()).all())
    first_start = sub.groupby("customer_id")["subscription_start_date"].min()
    add("first subscription starts on signup date",
        (first_start.reindex(cu["customer_id"]).values == cu["signup_date"].values).all())
    tk = t["support_tickets"].merge(cu[["customer_id", "signup_date"]], on="customer_id")
    add("tickets not created before signup", (tk["created_date"] >= tk["signup_date"].dt.normalize()).all())

    # 4. domain checks
    for tbl, cols in NON_NEGATIVE.items():
        for col in cols:
            bad = int((t[tbl][col] < 0).sum())
            add(f"non-negative: {tbl}.{col}", bad == 0, f"{bad} negative")
    add("plans in allowed set", sub["subscription_plan"].isin(C.PLANS).all())
    add("channels in allowed set", cu["acquisition_channel"].isin(C.CHANNELS).all())
    add("size bands in allowed set", cu["company_size"].isin(C.SIZE_BANDS).all())
    add("discount within 0-30%", sub["discount_percentage"].between(0, 30).all())
    csat = t["support_tickets"]["customer_satisfaction_score"].dropna()
    add("CSAT within 1-5", csat.between(1, 5).all())
    ad = t["usage_monthly"]["active_days"]
    add("active_days within 0-31", ad.between(0, 31).all())
    adopt = t["usage_monthly"]["feature_adoption_rate"].dropna()
    add("adoption rate within 0-1", adopt.between(0, 1).all())

    # 5. revenue identity and lifecycle logic
    rv = t["revenue_monthly"]
    ident = rv["recurring_revenue"] + rv["expansion_revenue"] - rv["contraction_revenue"] - rv["discounts"] - rv["refunds"]
    add("recognized = recurring + expansion - contraction - discounts - refunds",
        np.allclose(ident, rv["recognized_revenue"], atol=0.03))

    # no subscription revenue / usage after cancellation unless covered by a later (reactivated) subscription
    subs = sub.copy()
    subs["start_m"] = _month(subs["subscription_start_date"])
    subs["end_m"] = np.where(subs["subscription_end_date"].isna(), C.LAST_MONTH.ordinal,
                             _month(subs["subscription_end_date"].fillna(C.DATA_END)).map(lambda p: p.ordinal))
    subs["start_o"] = subs["start_m"].map(lambda p: p.ordinal)
    billed = rv[rv["recurring_revenue"] > 0][["customer_id", "month"]].copy()
    billed["o"] = _month(billed["month"]).map(lambda p: p.ordinal)
    merged = billed.merge(subs[["customer_id", "start_o", "end_m"]], on="customer_id")
    covered = merged[(merged["o"] >= merged["start_o"]) & (merged["o"] <= merged["end_m"])]
    n_uncovered = len(billed) - len(covered[["customer_id", "o"]].drop_duplicates())
    add("no recurring revenue outside a subscription period", n_uncovered == 0, f"{n_uncovered} rows")
    adj = rv[(rv["recurring_revenue"] == 0)]
    add("zero-recurring rows are explicit refund adjustments only",
        ((adj["refunds"] > 0) & (adj["recognized_revenue"] < 0)).all(), f"{len(adj)} adjustment rows")
    us = t["usage_monthly"].copy()
    us["o"] = _month(us["observation_month"]).map(lambda p: p.ordinal)
    mu = us.merge(subs[["customer_id", "start_o", "end_m"]], on="customer_id")
    cov_u = mu[(mu["o"] >= mu["start_o"]) & (mu["o"] <= mu["end_m"])][["customer_id", "o"]].drop_duplicates()
    add("no usage outside a subscription period", len(us) - len(cov_u) == 0)
    ev = t["billing_events"]
    canc = ev[ev["event_type"] == "cancellation"]
    add("every cancelled subscription has one cancellation event",
        len(canc) == int((sub["subscription_status"] == "Cancelled").sum()))

    # 6. missing values (reported, not failed)
    for tbl, df in t.items():
        miss = df.isna().mean()
        miss = miss[miss > 0]
        if len(miss):
            add(f"missing values: {tbl}", True, ", ".join(f"{c}={v:.1%}" for c, v in miss.items()), "warning")
    return out


def assert_valid(tables: dict[str, pd.DataFrame]) -> list[Check]:
    results = validate_tables(tables)
    failed = [c for c in results if c.severity == "error" and not c.passed]
    if failed:
        raise ValueError("Data validation failed:\n" + "\n".join(f" - {c.name}: {c.detail}" for c in failed))
    return results


def format_report(results: list[Check]) -> str:
    lines = []
    for c in results:
        mark = "PASS" if c.passed else ("WARN" if c.severity == "warning" else "FAIL")
        lines.append(f"[{mark}] {c.name}" + (f"  ({c.detail})" if c.detail else ""))
    return "\n".join(lines)