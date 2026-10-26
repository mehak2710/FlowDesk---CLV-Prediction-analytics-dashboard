"""Cached data/model loading for the dashboard (reads SQLite + joblib artifacts)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clv import config as C  # noqa: E402
from clv.modeling import load_bundle  # noqa: E402
from clv.storage import read_table  # noqa: E402

TABLES = ["customers", "subscriptions", "revenue_monthly", "usage_monthly", "support_tickets",
          "billing_events", "marketing_channels", "customer_snapshots"]


def artifacts_ready() -> bool:
    return C.DB_PATH.exists() and (C.MODELS_DIR / "clv_model.joblib").exists()


@st.cache_data(show_spinner="Loading data...")
def load_data() -> dict[str, pd.DataFrame]:
    data = {name: read_table(name) for name in TABLES}
    final_plan = (data["subscriptions"].sort_values("subscription_start_date")
                  .groupby("customer_id")["subscription_plan"].last().rename("final_plan"))
    data["customers"] = data["customers"].merge(final_plan, on="customer_id", how="left")
    return data


@st.cache_data(show_spinner=False)
def load_report() -> dict:
    return json.loads((C.MODELS_DIR / "model_report.json").read_text())


@st.cache_resource(show_spinner="Loading model...")
def load_model(kind: str):
    """kind: 'backtest' -> evaluation model (out-of-sample for 2024-12-31); 'current' -> production model."""
    name = "clv_model_eval.joblib" if kind == "backtest" else "clv_model.joblib"
    return load_bundle(C.MODELS_DIR / name)