"""Model training, evaluation, selection, persistence and explanation.

Protocol
--------
1. Candidates are fitted on the TRAIN snapshots and compared on the later VALIDATION snapshot
   (hyper-parameters chosen on validation only).
2. Model selection uses validation RMSE with a parsimony rule: the simplest model whose validation
   RMSE is within `PARSIMONY_TOLERANCE` of the best wins (interpretability vs accuracy trade-off).
3. Every candidate is refitted on TRAIN+VALID and evaluated ONCE on the out-of-time TEST snapshot.
4. The production artifact is the selected model refitted on all labelled snapshots (train+valid+test)
   and is used to score the current (unlabelled) snapshot.
"""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from .features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, FEATURE_LABELS, NUMERIC_FEATURES, TARGET

PARSIMONY_TOLERANCE = 0.03
SIMPLICITY_ORDER = ["Linear Regression", "Random Forest", "Gradient Boosting"]

# Related features are explained together: occluding one of several collinear columns understates its effect.
DRIVER_GROUPS = {
    "Current revenue level": ["mrr_now", "rev_3m_avg", "rev_6m_avg", "cum_revenue"],
    "Revenue momentum": ["mrr_change_3m", "expansion_6m", "contraction_6m", "upgrades_total", "downgrades_total", "downgrades_6m"],
    "Product usage level": ["logins_1m", "logins_3m", "active_days_3m", "adoption_3m", "session_min_3m", "usage_intensity"],
    "Usage trend": ["logins_trend_ratio", "usage_change_3m", "usage_change_min_6m"],
    "Support experience": ["tickets_90d", "tickets_180d", "unresolved_180d", "high_priority_90d", "csat_180d", "resolution_hours_180d"],
    "Billing health": ["payment_failures_6m", "payment_failures_total", "discount_pct", "billing_cycle"],
    "Plan & firmographics": ["plan", "log_employees", "company_size", "industry", "sales_region"],
    "Tenure, renewal & acquisition": ["tenure_months", "months_to_renewal", "acquisition_channel", "reactivation_flag"],
}


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "WAPE": float(np.abs(y_true - y_pred).sum() / max(y_true.sum(), 1e-9)),
        "Bias": float(y_pred.mean() - y_true.mean()),
    }


def baseline_predictions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Rule-based benchmarks that need no training."""
    return {
        "Baseline: MRR x 12": (frame["mrr_now"] * 12).to_numpy(),
        "Baseline: 3-mo avg revenue x 12": (frame["rev_3m_avg"].fillna(frame["mrr_now"]) * 12).to_numpy(),
    }


# ----------------------------------------------------------------------------
# Pipelines
# ----------------------------------------------------------------------------
def build_preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), CATEGORICAL_FEATURES),
    ])


def run_rate(X: pd.DataFrame) -> pd.Series:
    """Annualised current run-rate: trailing 3-month average recognised revenue x 12 (floored at $50)."""
    return (X["rev_3m_avg"].fillna(X["mrr_now"]) * 12.0).clip(lower=50.0)


class RunRateRegressor(BaseEstimator, RegressorMixin):
    """Predict 12-month revenue as run-rate x realisation ratio.

    The wrapped estimator learns the *ratio* target / run_rate (the share of today's annualised revenue the
    customer is expected to realise over the next 12 months, which captures churn, contraction and expansion),
    weighted by run-rate so that large accounts matter in proportion to their dollars. Predictions are
    ratio_hat x run_rate. This lets tree models extrapolate to accounts larger than any in training and makes
    the linear model interpretable as "what moves realised revenue relative to run-rate".
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        scale = run_rate(X)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, np.asarray(y, float) / scale.to_numpy(), model__sample_weight=scale.to_numpy())
        return self

    def predict(self, X):
        ratio = np.clip(self.estimator_.predict(X), 0.0, None)
        return ratio * run_rate(X).to_numpy()


def make_candidates(seed: int = C.SEED) -> dict[str, list[RunRateRegressor]]:
    """Candidate models per family (a few hyper-parameter settings for the ensembles)."""
    lin = [Pipeline([("prep", build_preprocessor(True)), ("model", LinearRegression())])]
    rf = [Pipeline([("prep", build_preprocessor(False)),
                    ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=leaf, max_features=mf,
                                                    n_jobs=-1, random_state=seed))])
          for leaf, mf in [(5, 0.5), (10, 0.5), (3, 0.33)]]
    gb = [Pipeline([("prep", build_preprocessor(False)),
                    ("model", GradientBoostingRegressor(n_estimators=n, learning_rate=lr, max_depth=d, subsample=0.8,
                                                        min_samples_leaf=10, random_state=seed))])
          for lr, d, n in [(0.05, 3, 300), (0.05, 2, 400), (0.03, 4, 400)]]
    wrap = lambda ps: [RunRateRegressor(p) for p in ps]  # noqa: E731
    return {"Linear Regression": wrap(lin), "Random Forest": wrap(rf), "Gradient Boosting": wrap(gb)}


def predict_revenue(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Predicted 12-month revenue, floored at zero (linear models can extrapolate below 0)."""
    return np.clip(pipeline.predict(X[FEATURE_COLUMNS]), 0.0, None)


def select_model(valid_metrics: dict[str, dict[str, float]], tolerance: float = PARSIMONY_TOLERANCE) -> tuple[str, str]:
    """Pick the simplest model within `tolerance` of the best validation RMSE. Returns (name, rationale)."""
    best = min(valid_metrics, key=lambda k: valid_metrics[k]["RMSE"])
    best_rmse = valid_metrics[best]["RMSE"]
    for name in SIMPLICITY_ORDER:
        if name in valid_metrics and valid_metrics[name]["RMSE"] <= best_rmse * (1 + tolerance):
            if name == best:
                return name, f"{name} had the lowest validation RMSE ({best_rmse:,.0f})."
            return name, (f"{name} is within {tolerance:.0%} of the best validation RMSE ({best} = {best_rmse:,.0f}) "
                          f"and is simpler, so it was preferred.")
    return best, f"{best} had the lowest validation RMSE."


# ----------------------------------------------------------------------------
# Training orchestration
# ----------------------------------------------------------------------------
@dataclass
class TrainingResult:
    bundle: dict          # production artifact (pipeline, reference values, thresholds, metadata)
    eval_pipeline: Pipeline
    churn_eval_pipeline: Pipeline
    report: dict          # metrics, selection and settings (JSON-serialisable)
    test_predictions: pd.DataFrame
    permutation_importance: pd.DataFrame


def _fit(p, frame: pd.DataFrame):
    return clone(p).fit(frame[FEATURE_COLUMNS], frame[TARGET])


def reference_values(frame: pd.DataFrame) -> pd.Series:
    ref = {c: frame[c].median() for c in NUMERIC_FEATURES}
    ref.update({c: frame[c].mode().iloc[0] for c in CATEGORICAL_FEATURES})
    return pd.Series(ref)


def train_and_evaluate(frames: dict[str, pd.DataFrame], seed: int = C.SEED) -> TrainingResult:
    from .segmentation import fit_thresholds  # local import avoids a cycle

    train, valid, test = frames["train"], frames["valid"], frames["test"]
    for name, f in [("train", train), ("valid", valid), ("test", test)]:
        assert f[TARGET].notna().all(), f"{name} frame has unobserved targets"

    # -- 1. validation: tune and compare ----------------------------------------------------------
    cands = make_candidates(seed)
    valid_metrics, chosen_settings = {}, {}
    for fam, plist in cands.items():
        scored = []
        for p in plist:
            fitted = _fit(p, train)
            scored.append((regression_metrics(valid[TARGET], predict_revenue(fitted, valid)), p))
        best_m, best_p = min(scored, key=lambda x: x[0]["RMSE"])
        valid_metrics[fam] = best_m
        chosen_settings[fam] = best_p
    baseline_valid = {k: regression_metrics(valid[TARGET], v) for k, v in baseline_predictions(valid).items()}

    selected, rationale = select_model(valid_metrics)

    # -- 2. test: refit everything on train+valid, evaluate once ----------------------------------------
    tv = pd.concat([train, valid], ignore_index=True)
    test_metrics, test_pred, eval_models = {}, {}, {}
    for fam, p in chosen_settings.items():
        fitted = _fit(p, tv)
        eval_models[fam] = fitted
        test_pred[fam] = predict_revenue(fitted, test)
        test_metrics[fam] = regression_metrics(test[TARGET], test_pred[fam])
    baseline_test = {k: regression_metrics(test[TARGET], v) for k, v in baseline_predictions(test).items()}

    # -- 3. permutation importance of the selected model on the test snapshot -------------------------
    eval_pipe = eval_models[selected]
    pi = permutation_importance(eval_pipe, test[FEATURE_COLUMNS], test[TARGET], n_repeats=5, random_state=seed,
                                scoring="neg_mean_absolute_error")
    pi_df = (pd.DataFrame({"feature": FEATURE_COLUMNS, "importance_mae_increase": pi.importances_mean,
                           "std": pi.importances_std})
             .assign(label=lambda d: d["feature"].map(FEATURE_LABELS))
             .sort_values("importance_mae_increase", ascending=False).reset_index(drop=True))

    # -- 4. segmentation thresholds from TRAINING-data quantiles (train+valid labels) -------------------------
    thresholds = fit_thresholds(tv[TARGET], C.SEGMENT_QUANTILES)

    # -- 5. production refit on every labelled snapshot ------------------------------------------------------
    allfit = pd.concat([train, valid, test], ignore_index=True)
    prod = _fit(chosen_settings[selected], allfit)

    def _desc(p):
        m = p.estimator.named_steps["model"]
        keep = ("n_estimators", "learning_rate", "max_depth", "min_samples_leaf", "max_features", "subsample")
        return {k: v for k, v in m.get_params().items() if k in keep}

    report = {
        "selected_model": selected, "selection_rationale": rationale,
        "parsimony_tolerance": PARSIMONY_TOLERANCE,
        "validation_metrics": valid_metrics, "validation_baselines": baseline_valid,
        "test_metrics": test_metrics, "test_baselines": baseline_test,
        "hyperparameters": {fam: _desc(p) for fam, p in chosen_settings.items()},
        "splits": {
            "train_snapshots": C.TRAIN_SNAPSHOTS, "valid_snapshots": C.VALID_SNAPSHOTS,
            "test_snapshots": C.TEST_SNAPSHOTS, "scoring_snapshot": C.SCORING_SNAPSHOT,
            "n_train": len(train), "n_valid": len(valid), "n_test": len(test),
        },
        "segment_thresholds": thresholds,
        "target_definition": "Sum of recognized_revenue over the 12 months after the snapshot date (revenue, not profit).",
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    churn = train_churn_model(frames, seed)
    report["churn_model"] = churn["report"]
    bundle = {
        "pipeline": prod, "model_name": selected, "feature_columns": FEATURE_COLUMNS,
        "churn_pipeline": churn["prod_pipeline"],
        "reference": reference_values(tv), "thresholds": thresholds, "report": report,
    }
    test_out = test[["customer_id", "snapshot_date", TARGET]].copy()
    for fam, pred in test_pred.items():
        test_out[f"pred_{fam}"] = pred
    return TrainingResult(bundle, eval_pipe, churn["eval_pipeline"], report, test_out, pi_df)


# ----------------------------------------------------------------------------
# Churn classifier (separately validated; drives risk tiers)
# ----------------------------------------------------------------------------
CHURN_TARGET = "churned_12m"
CHURN_AUC_TOLERANCE = 0.01


def make_churn_candidates(seed: int = C.SEED) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([("prep", build_preprocessor(True)),
                                         ("model", LogisticRegression(C=0.3, max_iter=3000))]),
        "Gradient Boosting": Pipeline([("prep", build_preprocessor(False)),
                                       ("model", GradientBoostingClassifier(
                                           n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                           min_samples_leaf=20, random_state=seed))]),
    }


def predict_churn(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(X[FEATURE_COLUMNS])[:, 1]


def train_churn_model(frames: dict[str, pd.DataFrame], seed: int = C.SEED) -> dict:
    """Probability that a customer cancels within 12 months of the snapshot (same split protocol as revenue)."""
    from .churn import TIER_PROB_CUTOFFS, assign_risk_tier

    train, valid, test = frames["train"], frames["valid"], frames["test"]
    cands = make_churn_candidates(seed)
    val_auc = {}
    for name, p in cands.items():
        m = clone(p).fit(train[FEATURE_COLUMNS], train[CHURN_TARGET])
        val_auc[name] = float(roc_auc_score(valid[CHURN_TARGET], predict_churn(m, valid)))
    best = max(val_auc, key=val_auc.get)
    selected = "Logistic Regression" if val_auc["Logistic Regression"] >= val_auc[best] - CHURN_AUC_TOLERANCE else best
    tv = pd.concat([train, valid], ignore_index=True)
    eval_pipe = clone(cands[selected]).fit(tv[FEATURE_COLUMNS], tv[CHURN_TARGET])
    p_test = predict_churn(eval_pipe, test)
    tiers = assign_risk_tier(pd.Series(p_test))
    tier_tbl = (pd.DataFrame({"tier": tiers, "y": test[CHURN_TARGET].to_numpy(), "p": p_test})
                .groupby("tier", observed=True).agg(customers=("y", "size"), observed_churn_rate=("y", "mean"),
                                                   mean_predicted=("p", "mean")))
    deciles = pd.qcut(pd.Series(p_test).rank(method="first"), 10, labels=False)
    cal = (pd.DataFrame({"decile": deciles, "y": test[CHURN_TARGET].to_numpy(), "p": p_test})
           .groupby("decile").agg(observed=("y", "mean"), predicted=("p", "mean")))
    allfit = pd.concat([train, valid, test], ignore_index=True)
    prod_pipe = clone(cands[selected]).fit(allfit[FEATURE_COLUMNS], allfit[CHURN_TARGET])
    report = {
        "selected_model": selected, "validation_auc": val_auc,
        "test_auc": float(roc_auc_score(test[CHURN_TARGET], p_test)),
        "test_brier": float(brier_score_loss(test[CHURN_TARGET], p_test)),
        "test_base_rate": float(test[CHURN_TARGET].mean()),
        "tier_cutoffs": TIER_PROB_CUTOFFS,
        "tier_backtest": {str(k): {c: float(v) for c, v in r.items()} for k, r in tier_tbl.iterrows()},
        "calibration_deciles": {int(k): {c: float(v) for c, v in r.items()} for k, r in cal.iterrows()},
    }
    return {"eval_pipeline": eval_pipe, "prod_pipeline": prod_pipe, "report": report}


# ----------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------
def save_artifacts(result: TrainingResult, models_dir=C.MODELS_DIR) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.bundle, models_dir / "clv_model.joblib")
    joblib.dump({**result.bundle, "pipeline": result.eval_pipeline, "churn_pipeline": result.churn_eval_pipeline},
                models_dir / "clv_model_eval.joblib")
    (models_dir / "model_report.json").write_text(json.dumps(result.report, indent=2, default=float))
    result.permutation_importance.to_csv(models_dir / "permutation_importance.csv", index=False)


def load_bundle(path=None) -> dict:
    return joblib.load(path or (C.MODELS_DIR / "clv_model.joblib"))


# ----------------------------------------------------------------------------
# Explanation (model-agnostic, group occlusion against a reference customer)
# ----------------------------------------------------------------------------
def explain_customer(pipeline: Pipeline, row: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    """Effect of each driver group on one customer's prediction.

    For every group, the group's columns are replaced with the training-population reference values
    (median / mode) and the change in predicted revenue is measured:
    contribution = prediction(actual row) - prediction(row with group set to typical values).
    This is an interpretation aid, not a causal estimate.
    """
    base = predict_revenue(pipeline, row)[0]
    out = []
    for group, cols in DRIVER_GROUPS.items():
        alt = row[FEATURE_COLUMNS].copy()
        for c in cols:
            alt[c] = reference[c]
        out.append({"driver": group, "contribution": float(base - predict_revenue(pipeline, alt)[0])})
    df = pd.DataFrame(out)
    df["abs"] = df["contribution"].abs()
    return df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)