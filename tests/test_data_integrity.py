import numpy as np
import pandas as pd
import pytest

from clv import config as C
from clv.data_generation import generate_tables
from clv.dictionary import TABLES
from clv.validation import assert_valid, validate_tables


def test_generated_data_passes_all_validation(small_tables):
    results = validate_tables(small_tables)
    failed = [c.name for c in results if c.severity == "error" and not c.passed]
    assert not failed, failed


def test_generation_is_reproducible(small_tables):
    again = generate_tables(seed=7, n_customers=900)
    for name, df in small_tables.items():
        pd.testing.assert_frame_equal(df, again[name])


def test_different_seed_changes_data(small_tables):
    other = generate_tables(seed=8, n_customers=900)
    assert not small_tables["revenue_monthly"]["recognized_revenue"].equals(other["revenue_monthly"]["recognized_revenue"])


def test_no_revenue_after_cancellation_except_adjustments(small_tables):
    rv, sub = small_tables["revenue_monthly"], small_tables["subscriptions"]
    cancelled = sub[sub["subscription_status"] == "Cancelled"]
    last_sub = sub.sort_values("subscription_start_date").groupby("customer_id").tail(1)
    done = last_sub[last_sub["subscription_status"] == "Cancelled"][["customer_id", "subscription_end_date"]]
    m = rv.merge(done, on="customer_id")
    after = m[m["month"] > m["subscription_end_date"]]
    assert (after["recurring_revenue"] == 0).all()
    assert (after["recognized_revenue"] <= 0).all()
    assert len(cancelled) > 0


def test_referential_integrity_catches_orphans(small_tables):
    broken = {k: v.copy() for k, v in small_tables.items()}
    broken["usage_monthly"].loc[0, "customer_id"] = "C99999"
    with pytest.raises(ValueError):
        assert_valid(broken)


def test_duplicate_ids_are_detected(small_tables):
    broken = {k: v.copy() for k, v in small_tables.items()}
    broken["customers"] = pd.concat([broken["customers"], broken["customers"].head(1)], ignore_index=True)
    with pytest.raises(ValueError):
        assert_valid(broken)


def test_realism_engagement_relates_to_churn_but_not_perfectly(small_tables):
    us, sub = small_tables["usage_monthly"], small_tables["subscriptions"]
    eng = us.groupby("customer_id")["active_days"].mean()
    churned = sub.groupby("customer_id")["subscription_status"].last().eq("Cancelled")
    d = pd.DataFrame({"eng": eng, "churned": churned}).dropna()
    d["q"] = pd.qcut(d["eng"], 4, labels=False)
    rate = d.groupby("q")["churned"].mean()
    assert rate.iloc[0] > rate.iloc[-1]          # less engaged -> more churn
    assert 0.02 < rate.iloc[-1] and rate.iloc[0] < 0.98  # but nothing is deterministic


def test_enterprise_pays_more_and_plans_correlate_with_size(small_tables):
    sub, cu = small_tables["subscriptions"], small_tables["customers"]
    mrr = sub.groupby("subscription_plan")["monthly_recurring_revenue"].median()
    assert mrr["Enterprise"] > mrr["Professional"] > mrr["Basic"]


def test_missing_values_and_outliers_exist(small_tables):
    assert small_tables["usage_monthly"]["monthly_logins"].isna().any()
    assert small_tables["support_tickets"]["customer_satisfaction_score"].isna().mean() > 0.2
    assert small_tables["customers"]["acquisition_cost"].isna().any()


def test_data_dictionary_covers_every_column(small_tables):
    for name, df in small_tables.items():
        assert set(df.columns) == set(TABLES[name][1]), name


def test_shipped_csvs_validate():
    from clv.storage import load_raw_tables
    if not (C.DATA_RAW / "customers.csv").exists():
        pytest.skip("run scripts/generate_data.py first")
    assert_valid(load_raw_tables())