"""Central configuration: seeds, dates, paths, business constants.

Everything that defines the fictional company ("Flowdesk") or a modelling rule
lives here so that the generator, features, models, tests and dashboard share
one source of truth.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
DB_PATH = ROOT / "data" / "clv.db"

COMPANY_NAME = "Flowdesk"  # fictional B2B workflow & productivity SaaS
SEED = 42
N_CUSTOMERS = 5_000

# --- Calendar ---------------------------------------------------------------
# 36 monthly periods are simulated so that a 12-month forward target can be
# evaluated on a genuinely out-of-time test snapshot (see README, "Why 36 months").
FIRST_MONTH = pd.Period("2023-01", freq="M")
LAST_MONTH = pd.Period("2025-12", freq="M")
LAST_SIGNUP_MONTH = pd.Period("2025-06", freq="M")
DATA_END = LAST_MONTH.end_time.normalize()  # 2025-12-31
HORIZON_MONTHS = 12

# --- Prediction snapshots (as-of dates) -------------------------------------
TRAIN_SNAPSHOTS = ["2023-03-31", "2023-06-30", "2023-09-30"]
VALID_SNAPSHOTS = ["2023-12-31"]
TEST_SNAPSHOTS = ["2024-12-31"]
SCORING_SNAPSHOT = "2025-12-31"  # "today": 12-month target is not yet observed
MIN_TENURE_MONTHS = 1  # eligibility: at least one full billed month of history

# --- Business vocab ----------------------------------------------------------
PLANS = ["Basic", "Professional", "Enterprise"]
PLAN_BASE_PRICE = {"Basic": 79.0, "Professional": 299.0, "Enterprise": 899.0}
PLAN_FEATURES_AVAILABLE = {"Basic": 12, "Professional": 20, "Enterprise": 30}
CHANNELS = ["Organic Search", "Paid Advertising", "Referral", "Social Media", "Direct Sales"]
SIZE_BANDS = ["1-49", "50-249", "250-999", "1000+"]
INDUSTRIES = [
    "Technology", "Financial Services", "Healthcare", "Retail & E-commerce",
    "Manufacturing", "Professional Services", "Education", "Media & Marketing",
]
SEGMENTS = ["High Value", "Medium Value", "Low Value"]

# --- Segmentation rule (quantiles of *training* 12-month revenue) ------------
SEGMENT_QUANTILES = {"low_upper": 0.40, "high_lower": 0.80}  # bottom 40% / mid 40% / top 20%

# --- Illustrative profit assumption (NOT derived from data) -------------------
DEFAULT_GROSS_MARGIN = 0.80