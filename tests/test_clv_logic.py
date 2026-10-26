import sqlite3

import numpy as np
import pandas as pd
import pytest

from clv import config as C
from clv.churn import RISK_RULES, add_signals, assign_risk_tier, recommend_actions
from clv.marketing import allocation_signals, channel_performance
from clv.metrics import (clv_to_cac, format_pct, format_usd, monthly_churn, payback_months,
                         period_churn_rate, profit_clv, safe_divide)
from clv.modeling import RunRateRegressor, regression_metrics, run_rate, select_model
from clv.segmentation import assign_segments, fit_thresholds, segment_summary
from clv.storage import build_database, load_queries


# ---- segmentation ------------------------------------------------------------------
def test_thresholds_are_training_quantiles():
    y = np.arange(1, 101, dtype=float)
    thr = fit_thresholds(y, {"low_upper": 0.4, "high_lower": 0.8})
    assert thr["low_upper"] == pytest.approx(np.quantile(y, 0.4))
    assert thr["high_lower"] == pytest.approx(np.quantile(y, 0.8))


def test_segment_assignment_boundaries():
    thr = {"low_upper": 100.0, "high_lower": 1000.0}
    seg = assign_segments([0, 99.99, 100, 999.99, 1000, 5000, np.nan], thr)
    assert list(seg[:6]) == ["Low Value", "Low Value", "Medium Value", "Medium Value", "High Value", "High Value"]
    assert pd.isna(seg.iloc[6])


def test_segment_summary_math():
    df = pd.DataFrame({"segment": pd.Categorical(["High Value", "High Value", "Low Value"], categories=C.SEGMENTS, ordered=True),
                       "predicted_12m": [100.0, 300.0, 10.0], "cum_revenue": [1, 2, 3.0], "mrr_now": [10, 20, 5.0],
                       "logins_3m": [1, 2, 3.0], "adoption_3m": [.5, .5, .5], "acquisition_cost": [50.0, 50.0, 5.0],
                       "churned_12m": [0, 1, 1.0], "risk_tier": pd.Categorical(["Low", "High", "Medium"])})
    s = segment_summary(df)
    assert s.loc["High Value", "customers"] == 2
    assert s.loc["High Value", "avg_predicted_clv"] == 200
    assert s.loc["High Value", "clv_to_cac"] == 4
    assert s.loc["High Value", "observed_12m_churn_rate"] == 0.5
    assert s["pct_of_customers"].sum() == pytest.approx(1)
    assert segment_summary(df.iloc[0:0]).empty


# ---- metrics --------------------------------------------------------------------------
def test_safe_divide_and_ratios():
    assert np.isnan(safe_divide(1, 0)) and np.isnan(safe_divide(1, np.nan)) and np.isnan(safe_divide(1, None))
    assert clv_to_cac(900, 300) == 3
    assert payback_months(800, 100, 0.8) == pytest.approx(10)
    assert profit_clv(1000, 0.8) == 800


def test_formatting():
    assert format_usd(1_234_567) == "$1.23M" and format_usd(12_345) == "$12.3K" and format_usd(999) == "$999"
    assert format_usd(np.nan) == "n/a" and format_pct(0.1234) == "12.3%"


def _toy_tables():
    ms = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    rev = pd.DataFrame([(c, m, 100.0) for c in ["A", "B", "C", "D"] for m in ms[:3] if not (c == "D" and m == ms[0])],
                       columns=["customer_id", "month", "recurring_revenue"])
    rev["expansion_revenue"] = 0.0; rev["contraction_revenue"] = 0.0; rev["recognized_revenue"] = 100.0
    # B cancels end of Feb (so billed Jan, Feb only) ; D signs up in Feb
    rev = rev[~((rev.customer_id == "B") & (rev.month == ms[2]))]
    ev = pd.DataFrame([("A", "2024-01-05", "new_subscription"), ("B", "2024-01-06", "new_subscription"),
                       ("C", "2024-01-07", "new_subscription"), ("D", "2024-02-10", "new_subscription"),
                       ("B", "2024-02-29", "cancellation")], columns=["customer_id", "event_date", "event_type"])
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    ev["event_id"] = [f"E{i}" for i in range(len(ev))]; ev["subscription_id"] = "S"; ev["plan_before"] = None; ev["plan_after"] = None; ev["mrr_after"] = 0.0
    return rev, ev


def test_monthly_churn_known_answer():
    rev, ev = _toy_tables()
    ch = monthly_churn(rev, ev).set_index("month")
    # Feb: billed A,B,C,D = 4; D started in Feb -> 3 at start; B cancelled -> 1/3
    assert ch.loc["2024-02-29", "churn_rate"] == pytest.approx(1 / 3)
    assert ch.loc["2024-03-31", "cancellations"] == 0
    assert ch.loc["2024-01-31", "active_at_start"] == 0  # all three are new in the first month


def test_period_churn_rate_known_answer():
    rev, ev = _toy_tables()
    # active at end of Jan: A, B, C. B cancels by Mar -> 1/3
    assert period_churn_rate(rev, ev, "2024-01-31", "2024-03-31") == pytest.approx(1 / 3)


def test_sql_churn_matches_pandas(tmp_path):
    rev, ev = _toy_tables()
    tables = {"revenue_monthly": rev, "billing_events": ev}
    db = build_database(tables, db_path=tmp_path / "t.db")
    with sqlite3.connect(db) as con:
        sql = pd.read_sql_query(load_queries()["monthly_churn"], con)
    py = monthly_churn(rev, ev)
    sql = sql.dropna(subset=["churn_rate"]).reset_index(drop=True)
    py = py.dropna(subset=["churn_rate"]).reset_index(drop=True)
    assert np.allclose(sql["churn_rate"], py["churn_rate"])


def test_sql_matches_pandas_on_real_data(small_tables, tmp_path):
    db = build_database(small_tables, db_path=tmp_path / "s.db")
    with sqlite3.connect(db) as con:
        sql = pd.read_sql_query(load_queries()["monthly_churn"], con).dropna(subset=["churn_rate"])
    py = monthly_churn(small_tables["revenue_monthly"], small_tables["billing_events"]).dropna(subset=["churn_rate"])
    assert np.allclose(sql["churn_rate"].to_numpy(), py["churn_rate"].to_numpy())


# ---- churn rules -------------------------------------------------------------------------
def _row(**kw):
    base = dict(logins_trend_ratio=1.0, usage_change_min_6m=0.0, payment_failures_6m=0, unresolved_180d=0, tickets_90d=0,
                csat_180d=4.5, downgrades_6m=0, adoption_3m=0.6)
    base.update(kw)
    return pd.DataFrame([base])


def test_signals_score_and_flags():
    quiet = add_signals(_row()).iloc[0]
    assert quiet["signal_score"] == 0 and quiet["risk_flags"] == ""
    bad = add_signals(_row(logins_trend_ratio=0.3, payment_failures_6m=2, downgrades_6m=1)).iloc[0]
    assert bad["signal_score"] == 30 + 25 + 15
    assert set(bad["risk_flags"].split("|")) == {"usage_decline_severe", "payment_failures_multi", "recent_downgrade"}


def test_missing_signals_earn_no_points():
    assert add_signals(_row(logins_trend_ratio=np.nan, csat_180d=np.nan)).iloc[0]["signal_score"] == 0


def test_score_is_capped_and_rules_are_unique():
    worst = add_signals(_row(logins_trend_ratio=0.1, usage_change_min_6m=-90, payment_failures_6m=3, unresolved_180d=2,
                             tickets_90d=5, csat_180d=1, downgrades_6m=2, adoption_3m=0.05)).iloc[0]
    assert worst["signal_score"] <= 100
    assert len({r[0] for r in RISK_RULES}) == len(RISK_RULES)


def test_risk_tiers_and_actions():
    t = assign_risk_tier(pd.Series([0.05, 0.15, 0.34, 0.35, 0.9]))
    assert list(t) == ["Low", "Medium", "Medium", "High", "High"]
    acts = recommend_actions("payment_failure|low_adoption", "High Value", "High")
    assert "executive sponsor" in acts[0].lower() and len(acts) == 3
    assert "No elevated" in recommend_actions("")[0]


# ---- models ---------------------------------------------------------------------------------
def test_regression_metrics_known_values():
    m = regression_metrics([100, 200, 300], [110, 190, 330])
    assert m["MAE"] == pytest.approx(50 / 3) and m["Bias"] == pytest.approx(10.0)
    assert m["RMSE"] == pytest.approx(np.sqrt((100 + 100 + 900) / 3))


def test_model_selection_prefers_simpler_within_tolerance():
    m = {"Linear Regression": {"RMSE": 101.0}, "Random Forest": {"RMSE": 100.0}, "Gradient Boosting": {"RMSE": 99.0}}
    assert select_model(m, 0.03)[0] == "Linear Regression"
    assert select_model(m, 0.005)[0] == "Gradient Boosting"


def test_run_rate_regressor_scales_with_run_rate():
    from sklearn.dummy import DummyRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    X = pd.DataFrame({"mrr_now": [100.0, 1000.0, 100.0], "rev_3m_avg": [100.0, 1000.0, np.nan]})
    y = np.array([960.0, 9600.0, 960.0])
    inner = Pipeline([("prep", FunctionTransformer(lambda d: d[["mrr_now"]].to_numpy())), ("model", DummyRegressor())])
    m = RunRateRegressor(inner).fit(X, y)
    # ratio is 0.8 for every row, so prediction = 0.8 x run-rate and 10x accounts get 10x predictions
    assert m.predict(X) == pytest.approx(0.8 * run_rate(X).to_numpy())
    assert run_rate(X).iloc[2] == 1200.0  # falls back to MRR when trailing average is missing


def test_saved_model_beats_baseline_out_of_time():
    import json
    p = C.MODELS_DIR / "model_report.json"
    if not p.exists():
        pytest.skip("run scripts/run_pipeline.py first")
    r = json.loads(p.read_text())
    sel = r["selected_model"]
    best_base = min(v["RMSE"] for v in r["test_baselines"].values())
    assert r["test_metrics"][sel]["RMSE"] < best_base
    assert r["churn_model"]["test_auc"] > 0.65
    assert r["splits"]["test_snapshots"][0] > max(r["splits"]["train_snapshots"] + r["splits"]["valid_snapshots"])


def test_saved_bundle_is_reusable():
    import joblib
    p = C.MODELS_DIR / "clv_model.joblib"
    if not p.exists():
        pytest.skip("run scripts/run_pipeline.py first")
    b = joblib.load(p)
    snaps = pd.read_csv(C.DATA_PROCESSED / "customer_snapshots.csv").head(20)
    from clv.modeling import predict_revenue
    pred = predict_revenue(b["pipeline"], snaps)
    assert len(pred) == 20 and (pred >= 0).all()


# ---- marketing -----------------------------------------------------------------------------------
def test_channel_performance_and_signals():
    cu = pd.DataFrame({"customer_id": ["a", "b", "c"], "acquisition_channel": ["Referral", "Referral", "Paid Advertising"],
                       "acquisition_cost": [100.0, 300.0, 1000.0]})
    rev = pd.DataFrame({"customer_id": ["a", "b", "c"], "recognized_revenue": [1000.0, 200.0, 500.0]})
    snap = pd.DataFrame({"customer_id": ["a", "c"], "predicted_12m": [2000.0, 1000.0], "mrr_now": [100.0, 100.0]})
    p = channel_performance(cu, rev, snap)
    assert p.loc["Referral", "avg_cac"] == 200 and p.loc["Referral", "clv_to_cac"] == 10
    assert p.loc["Referral", "realised_revenue_to_cac"] == pytest.approx(600 / 200)
    assert p.loc["Paid Advertising", "clv_to_cac"] == 1
    sig = allocation_signals(p)
    assert sig.loc["Referral", "signal"].startswith("Strong") and sig.loc["Paid Advertising", "signal"].startswith("Weak")
    assert channel_performance(cu.iloc[0:0], rev, snap).empty