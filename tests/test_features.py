import numpy as np
import pandas as pd
import pytest

from clv import config as C
from clv.features import FEATURE_COLUMNS, TARGET, build_snapshot

T = pd.Timestamp("2024-06-30")


def truncate(tables, t):
    """What an analyst would have on day t: nothing dated after t, and no knowledge of future cancellations."""
    out = {k: v.copy() for k, v in tables.items()}
    out["revenue_monthly"] = out["revenue_monthly"][out["revenue_monthly"]["month"] <= t]
    out["usage_monthly"] = out["usage_monthly"][out["usage_monthly"]["observation_month"] <= t]
    out["support_tickets"] = out["support_tickets"][out["support_tickets"]["created_date"] <= t]
    out["billing_events"] = out["billing_events"][out["billing_events"]["event_date"] <= t]
    sub = out["subscriptions"][out["subscriptions"]["subscription_start_date"] <= t].copy()
    future_end = sub["subscription_end_date"] > t
    sub.loc[future_end, "subscription_end_date"] = pd.NaT
    sub.loc[future_end, "subscription_status"] = "Active"
    out["subscriptions"] = sub
    out["customers"] = out["customers"][out["customers"]["signup_date"] <= t]
    return out


def test_features_do_not_use_future_information(small_tables):
    full = build_snapshot(small_tables, T, with_target=False).set_index("customer_id")
    trunc = build_snapshot(truncate(small_tables, T), T, with_target=False).set_index("customer_id")
    assert len(full) > 50
    pd.testing.assert_frame_equal(full[FEATURE_COLUMNS].sort_index(), trunc[FEATURE_COLUMNS].sort_index(), check_dtype=False)


def test_target_only_uses_revenue_after_snapshot(small_tables):
    base = build_snapshot(small_tables, T).set_index("customer_id")[TARGET]
    t2 = {k: v.copy() for k, v in small_tables.items()}
    rv = t2["revenue_monthly"]
    rv.loc[rv["month"] <= T, "recognized_revenue"] *= 10  # change only the past
    same = build_snapshot(t2, T).set_index("customer_id")[TARGET]
    pd.testing.assert_series_equal(base.sort_index(), same.sort_index())


def test_target_equals_manual_12_month_sum(small_tables):
    f = build_snapshot(small_tables, T)
    cid = f.sort_values(TARGET).iloc[len(f) // 2]["customer_id"]
    rv = small_tables["revenue_monthly"]
    manual = rv[(rv["customer_id"] == cid) & (rv["month"] > T) & (rv["month"] <= T + pd.DateOffset(months=12))]["recognized_revenue"].sum()
    assert f.loc[f["customer_id"] == cid, TARGET].iloc[0] == pytest.approx(manual)


def test_eligibility_rules(small_tables):
    f = build_snapshot(small_tables, T)
    assert (f["tenure_months"] >= C.MIN_TENURE_MONTHS).all()
    sub = small_tables["subscriptions"]
    ended_before = sub[sub["subscription_end_date"] <= T]["customer_id"]
    last = sub.sort_values("subscription_start_date").groupby("customer_id").tail(1)
    gone = last[(last["subscription_end_date"] <= T)]["customer_id"]
    assert not f["customer_id"].isin(gone).any()
    assert f["customer_id"].is_unique


def test_incomplete_future_window_has_no_target(small_tables):
    f = build_snapshot(small_tables, C.SCORING_SNAPSHOT)
    assert f[TARGET].isna().all()
    assert (~f["window_complete"]).all()


def test_churned_customers_have_lower_future_revenue(small_tables):
    f = build_snapshot(small_tables, T)
    assert f.loc[f["churned_12m"] == 1, TARGET].mean() < f.loc[f["churned_12m"] == 0, TARGET].mean()


def test_months_to_renewal_range(small_tables):
    f = build_snapshot(small_tables, T)
    ann = f[f["billing_cycle"] == "Annual"]["months_to_renewal"]
    assert ann.between(1, 12).all()
    assert f[f["billing_cycle"] == "Monthly"]["months_to_renewal"].isna().all()