"""Point-in-time feature engineering and 12-month revenue target.

Prediction setup
----------------
For an as-of ("snapshot") date T (a month-end):

* Features use ONLY rows dated <= T.
* Target = sum of `recognized_revenue` in the 12 calendar months after T
  (T+1 ... T+12), including zeros after cancellation and explicit refund adjustments.
* Eligibility rule (documented in README):
    1. customer signed up on or before T and has >= MIN_TENURE_MONTHS full months of history,
    2. the customer holds an active subscription at T (a cancellation effective on or before T excludes them),
    3. for labelled snapshots, the full 12-month window must be observed (T + 12 months <= data end).
  The scoring snapshot (data end) has no observable target; its rows are scored, not evaluated.

`tests/test_features.py` verifies leakage-freedom by truncating all tables at T and
checking that features are unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

NUMERIC_FEATURES = [
    "tenure_months", "log_employees", "discount_pct",
    "mrr_now", "mrr_change_3m", "rev_3m_avg", "rev_6m_avg", "cum_revenue", "expansion_6m", "contraction_6m",
    "logins_1m", "logins_3m", "logins_trend_ratio", "active_days_3m", "adoption_3m", "session_min_3m",
    "usage_change_3m", "usage_change_min_6m",
    "tickets_90d", "tickets_180d", "unresolved_180d", "high_priority_90d", "csat_180d", "resolution_hours_180d",
    "payment_failures_6m", "payment_failures_total", "upgrades_total", "downgrades_total", "downgrades_6m",
    "reactivation_flag", "months_to_renewal", "usage_intensity",
]
CATEGORICAL_FEATURES = ["plan", "billing_cycle", "industry", "company_size", "acquisition_channel", "sales_region"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ID_COLUMNS = ["customer_id", "snapshot_date"]
TARGET = "target_12m"

# Human-readable labels used by the dashboard's explanation panel
FEATURE_LABELS = {
    "tenure_months": "Tenure (months)", "log_employees": "Company size (log employees)", "discount_pct": "Discount %",
    "mrr_now": "Current MRR", "mrr_change_3m": "MRR change (3m)", "rev_3m_avg": "Avg monthly revenue (3m)",
    "rev_6m_avg": "Avg monthly revenue (6m)", "cum_revenue": "Cumulative revenue", "expansion_6m": "Expansion revenue (6m)",
    "contraction_6m": "Contraction revenue (6m)", "logins_1m": "Logins (last month)", "logins_3m": "Avg logins (3m)",
    "logins_trend_ratio": "Login trend (3m vs prior 3m)", "active_days_3m": "Active days (3m avg)",
    "adoption_3m": "Feature adoption (3m avg)", "session_min_3m": "Session minutes (3m avg)",
    "usage_change_3m": "Usage change % (3m avg)", "usage_change_min_6m": "Worst monthly usage change (6m)",
    "tickets_90d": "Support tickets (90d)", "tickets_180d": "Support tickets (180d)",
    "unresolved_180d": "Unresolved tickets (180d)", "high_priority_90d": "High/critical tickets (90d)",
    "csat_180d": "Avg CSAT (180d)", "resolution_hours_180d": "Avg resolution hours (180d)",
    "payment_failures_6m": "Payment failures (6m)", "payment_failures_total": "Payment failures (all time)",
    "upgrades_total": "Upgrades (all time)", "downgrades_total": "Downgrades (all time)",
    "downgrades_6m": "Downgrades (6m)", "reactivation_flag": "Previously reactivated",
    "months_to_renewal": "Months until annual renewal", "usage_intensity": "Usage intensity (logins per sqrt employee)",
    "plan": "Subscription plan", "billing_cycle": "Billing cycle", "industry": "Industry",
    "company_size": "Company size band", "acquisition_channel": "Acquisition channel", "sales_region": "Sales region",
}


def _mo(s: pd.Series) -> pd.Series:
    """Month ordinal (year*12 + month) for a datetime series."""
    return s.dt.year * 12 + s.dt.month


def _window_mean(df: pd.DataFrame, col: str, lo: int, hi: int, name: str) -> pd.Series:
    sel = df[(df["_mo"] >= lo) & (df["_mo"] <= hi)]
    return sel.groupby("customer_id")[col].mean().rename(name)


def build_snapshot(tables: dict[str, pd.DataFrame], snapshot_date, with_target: bool = True) -> pd.DataFrame:
    """Build the modelling frame (one row per eligible customer) as of `snapshot_date`."""
    T = pd.Timestamp(snapshot_date).normalize()
    t_o = T.year * 12 + T.month
    cu = tables["customers"]
    sub = tables["subscriptions"]

    # ---- eligibility ------------------------------------------------------------
    c = cu[cu["signup_date"] <= T].copy()
    c["tenure_months"] = t_o - _mo(c["signup_date"])
    c = c[c["tenure_months"] >= C.MIN_TENURE_MONTHS]
    started = sub[sub["subscription_start_date"] <= T]
    # subscription still running after T (end date is only compared to T, so future values cannot leak)
    live = started[started["subscription_end_date"].isna() | (started["subscription_end_date"] > T)]
    live = live.sort_values("subscription_start_date").drop_duplicates("customer_id", keep="last")
    c = c.merge(live[["customer_id", "billing_cycle", "discount_percentage", "subscription_start_date"]],
                on="customer_id", how="inner")
    c = c.rename(columns={"discount_percentage": "discount_pct"})
    c["log_employees"] = np.log(c["employee_count"])  # NaN stays NaN -> imputed in the pipeline
    # Annual contracts can only be cancelled at renewal (every 12th billed month of the subscription).
    # Months until the next renewal end-of-month after T, in 1..12; NaN for monthly billing.
    start_o = _mo(c["subscription_start_date"])
    r = (start_o + 11 - t_o) % 12
    c["months_to_renewal"] = np.where(c["billing_cycle"] == "Annual", np.where(r == 0, 12, r), np.nan)
    X = c.set_index("customer_id")[["tenure_months", "log_employees", "discount_pct", "billing_cycle",
                                    "industry", "company_size", "acquisition_channel", "sales_region",
                                    "months_to_renewal", "employee_count"]].copy()

    # ---- revenue features -----------------------------------------------------------
    rv = tables["revenue_monthly"]
    rv = rv[rv["month"] <= T].copy()
    rv["_mo"] = _mo(rv["month"])
    rv["gross"] = rv["recurring_revenue"] + rv["expansion_revenue"] - rv["contraction_revenue"]
    now = rv[rv["_mo"] == t_o].set_index("customer_id")["gross"].rename("mrr_now")
    ago = rv[rv["_mo"] == t_o - 3].set_index("customer_id")["gross"].rename("mrr_3m_ago")
    X = X.join(now).join(ago)
    X["mrr_change_3m"] = (X["mrr_now"] / X["mrr_3m_ago"] - 1.0)
    X = X.drop(columns="mrr_3m_ago")
    X = X.join(_window_mean(rv, "recognized_revenue", t_o - 2, t_o, "rev_3m_avg"))
    X = X.join(_window_mean(rv, "recognized_revenue", t_o - 5, t_o, "rev_6m_avg"))
    X = X.join(rv.groupby("customer_id")["recognized_revenue"].sum().rename("cum_revenue"))
    w6 = rv[rv["_mo"] >= t_o - 5].groupby("customer_id")
    X = X.join(w6["expansion_revenue"].sum().rename("expansion_6m")).join(w6["contraction_revenue"].sum().rename("contraction_6m"))

    # ---- usage features ----------------------------------------------------------------
    us = tables["usage_monthly"]
    us = us[us["observation_month"] <= T].copy()
    us["_mo"] = _mo(us["observation_month"])
    X = X.join(us[us["_mo"] == t_o].set_index("customer_id")["monthly_logins"].rename("logins_1m"))
    X = X.join(_window_mean(us, "monthly_logins", t_o - 2, t_o, "logins_3m"))
    prior = _window_mean(us, "monthly_logins", t_o - 5, t_o - 3, "logins_prior3m")
    X = X.join(prior)
    X["logins_trend_ratio"] = X["logins_3m"] / X["logins_prior3m"].where(X["logins_prior3m"] > 0)
    X = X.drop(columns="logins_prior3m")
    X["usage_intensity"] = X["logins_3m"] / np.sqrt(X["employee_count"])  # usage normalised by account size
    X = X.drop(columns="employee_count")
    X = X.join(_window_mean(us, "active_days", t_o - 2, t_o, "active_days_3m"))
    X = X.join(_window_mean(us, "feature_adoption_rate", t_o - 2, t_o, "adoption_3m"))
    X = X.join(_window_mean(us, "session_duration_minutes", t_o - 2, t_o, "session_min_3m"))
    X = X.join(_window_mean(us, "usage_change_percentage", t_o - 2, t_o, "usage_change_3m"))
    X = X.join(us[us["_mo"] >= t_o - 5].groupby("customer_id")["usage_change_percentage"].min().rename("usage_change_min_6m"))

    # ---- support features ------------------------------------------------------------------
    tk = tables["support_tickets"]
    tk = tk[tk["created_date"] <= T].copy()
    d = (T - tk["created_date"]).dt.days
    tk["unresolved"] = tk["resolution_time_hours"].isna().astype(int)
    tk["high"] = tk["priority"].isin(["High", "Critical"]).astype(int)
    t90, t180 = tk[d < 90], tk[d < 180]
    X = X.join(t90.groupby("customer_id").size().rename("tickets_90d"))
    X = X.join(t180.groupby("customer_id").size().rename("tickets_180d"))
    X = X.join(t180.groupby("customer_id")["unresolved"].sum().rename("unresolved_180d"))
    X = X.join(t90.groupby("customer_id")["high"].sum().rename("high_priority_90d"))
    X = X.join(t180.groupby("customer_id")["customer_satisfaction_score"].mean().rename("csat_180d"))
    X = X.join(t180.groupby("customer_id")["resolution_time_hours"].mean().rename("resolution_hours_180d"))
    for col in ["tickets_90d", "tickets_180d", "unresolved_180d", "high_priority_90d"]:
        X[col] = X[col].fillna(0)  # no tickets is a real zero, not a missing value

    # ---- billing events --------------------------------------------------------------------------
    ev = tables["billing_events"]
    ev = ev[ev["event_date"] <= T].copy()
    ev["_mo"] = _mo(ev["event_date"])
    plan = (ev[ev["event_type"].isin(["new_subscription", "upgrade", "downgrade", "reactivation"])]
            .sort_values(["event_date", "event_id"]).drop_duplicates("customer_id", keep="last")
            .set_index("customer_id")["plan_after"].rename("plan"))
    X = X.join(plan)
    recent = ev[ev["_mo"] >= t_o - 5]
    X = X.join(recent[recent["event_type"] == "payment_failure"].groupby("customer_id").size().rename("payment_failures_6m"))
    X = X.join(ev[ev["event_type"] == "payment_failure"].groupby("customer_id").size().rename("payment_failures_total"))
    X = X.join(ev[ev["event_type"] == "upgrade"].groupby("customer_id").size().rename("upgrades_total"))
    X = X.join(ev[ev["event_type"] == "downgrade"].groupby("customer_id").size().rename("downgrades_total"))
    X = X.join(recent[recent["event_type"] == "downgrade"].groupby("customer_id").size().rename("downgrades_6m"))
    X = X.join(ev[ev["event_type"] == "reactivation"].groupby("customer_id").size().rename("reactivation_flag"))
    for col in ["payment_failures_6m", "payment_failures_total", "upgrades_total", "downgrades_total",
                "downgrades_6m", "reactivation_flag"]:
        X[col] = X[col].fillna(0)
    X["reactivation_flag"] = (X["reactivation_flag"] > 0).astype(int)

    X = X.reset_index()
    X.insert(1, "snapshot_date", T)

    # ---- target & outcome labels (use FUTURE data; never fed to the model as features) ---------
    end_o = C.DATA_END.year * 12 + C.DATA_END.month
    complete = (t_o + C.HORIZON_MONTHS) <= end_o
    X["window_complete"] = complete
    if with_target:
        full = tables["revenue_monthly"].copy()
        full["_mo"] = _mo(full["month"])
        fut = full[(full["_mo"] > t_o) & (full["_mo"] <= t_o + C.HORIZON_MONTHS)]
        y = fut.groupby("customer_id")["recognized_revenue"].sum().rename(TARGET)
        X = X.merge(y, on="customer_id", how="left")
        evf = tables["billing_events"]
        end_f = T + pd.DateOffset(months=C.HORIZON_MONTHS)
        ch = evf[(evf["event_type"] == "cancellation") & (evf["event_date"] > T) & (evf["event_date"] <= end_f)]
        X["churned_12m"] = X["customer_id"].isin(ch["customer_id"]).astype(int)
        if complete:
            X[TARGET] = X[TARGET].fillna(0.0)  # eligible customers with no future rows earned 0
        else:  # incomplete window: target is not observable -> leave NaN
            X[TARGET] = np.nan
            X["churned_12m"] = np.nan
    return X


def build_snapshots(tables, dates) -> pd.DataFrame:
    return pd.concat([build_snapshot(tables, d) for d in dates], ignore_index=True)


def build_all(tables) -> dict[str, pd.DataFrame]:
    """Train / validation / test / scoring frames according to the config."""
    return {
        "train": build_snapshots(tables, C.TRAIN_SNAPSHOTS),
        "valid": build_snapshots(tables, C.VALID_SNAPSHOTS),
        "test": build_snapshots(tables, C.TEST_SNAPSHOTS),
        "score": build_snapshot(tables, C.SCORING_SNAPSHOT, with_target=True),
    }