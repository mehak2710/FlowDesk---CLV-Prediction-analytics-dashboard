"""End-to-end orchestration: CSVs -> validation -> features -> models -> scored snapshots -> SQLite."""
from __future__ import annotations

import json

import pandas as pd

from . import config as C
from .churn import add_signals, assign_risk_tier, validate_risk
from .features import FEATURE_COLUMNS, TARGET, build_all
from .modeling import TrainingResult, predict_churn, predict_revenue, save_artifacts, train_and_evaluate
from .segmentation import assign_segments
from .storage import build_database, load_raw_tables
from .validation import assert_valid


def score_snapshot(frame: pd.DataFrame, pipeline, churn_pipeline, thresholds: dict, split: str,
                   customers: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["split"] = split
    out["predicted_12m"] = predict_revenue(pipeline, out)
    out["actual_12m"] = out[TARGET]
    out["segment"] = assign_segments(out["predicted_12m"], thresholds)
    out["churn_probability"] = predict_churn(churn_pipeline, out)
    out["risk_tier"] = assign_risk_tier(out["churn_probability"])
    out = add_signals(out)
    out = out.merge(customers[["customer_id", "signup_date", "acquisition_cost"]], on="customer_id", how="left")
    return out.drop(columns=[TARGET])


def run(verbose: bool = True) -> TrainingResult:
    tables = load_raw_tables()
    assert_valid(tables)
    frames = build_all(tables)
    result = train_and_evaluate(frames)
    save_artifacts(result)

    thr = result.bundle["thresholds"]
    cust = tables["customers"]
    snaps = pd.concat([
        score_snapshot(frames["test"], result.eval_pipeline, result.churn_eval_pipeline, thr, "backtest", cust),
        score_snapshot(frames["score"], result.bundle["pipeline"], result.bundle["churn_pipeline"], thr, "current", cust),
    ], ignore_index=True)

    # empirical 12-month churn rate per risk tier from the out-of-time backtest (documented in the report)
    bt = snaps[snaps["split"] == "backtest"]
    result.report["risk_validation"] = validate_risk(bt)
    (C.MODELS_DIR / "model_report.json").write_text(json.dumps(result.report, indent=2, default=float))

    snaps.to_csv(C.DATA_PROCESSED / "customer_snapshots.csv", index=False)
    result.permutation_importance.to_csv(C.DATA_PROCESSED / "permutation_importance.csv", index=False)
    build_database(tables, {"customer_snapshots": snaps})
    if verbose:
        print(json.dumps({k: result.report[k] for k in ("selected_model", "selection_rationale")}, indent=2))
    return result