"""CSV loading, SQLite database build and SQL helpers."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from . import config as C

DATE_COLUMNS = {
    "customers": ["signup_date"],
    "subscriptions": ["subscription_start_date", "subscription_end_date"],
    "usage_monthly": ["observation_month"],
    "support_tickets": ["created_date"],
    "revenue_monthly": ["month"],
    "marketing_channels": ["campaign_month"],
    "billing_events": ["event_date"],
}
TABLE_NAMES = list(DATE_COLUMNS)
SQL_FILE = C.ROOT / "sql" / "analytics.sql"


def load_raw_tables(raw_dir: Path = C.DATA_RAW) -> dict[str, pd.DataFrame]:
    out = {}
    for name, dates in DATE_COLUMNS.items():
        out[name] = pd.read_csv(Path(raw_dir) / f"{name}.csv", parse_dates=dates)
    return out


def load_queries(path: Path = SQL_FILE) -> dict[str, str]:
    text = Path(path).read_text()
    parts = re.split(r"^-- name:\s*(\w+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def build_database(tables: dict[str, pd.DataFrame], derived: dict[str, pd.DataFrame] | None = None,
                   db_path: Path = C.DB_PATH) -> Path:
    """(Re)build the SQLite database from raw CSV tables plus derived model-output tables."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as con:
        for name, df in {**tables, **(derived or {})}.items():
            d = df.copy()
            for col in d.columns:
                if pd.api.types.is_datetime64_any_dtype(d[col]):
                    d[col] = d[col].dt.strftime("%Y-%m-%d")
            d.to_sql(name, con, index=False, if_exists="replace")
        present = {**tables, **(derived or {})}
        for idx, tbl, cols in [("idx_rev_cust", "revenue_monthly", "customer_id, month"),
                               ("idx_usage_cust", "usage_monthly", "customer_id, observation_month"),
                               ("idx_tix_cust", "support_tickets", "customer_id"),
                               ("idx_evt_cust", "billing_events", "customer_id")]:
            if tbl in present:
                con.execute(f"CREATE INDEX {idx} ON {tbl}({cols})")
    return db_path


def query(sql: str, params=None, db_path: Path = C.DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(sql, con, params=params)


def read_table(name: str, db_path: Path = C.DB_PATH) -> pd.DataFrame:
    df = query(f"SELECT * FROM {name}", db_path=db_path)
    for col in df.columns:
        if col in sum(DATE_COLUMNS.values(), []) or col.endswith("_date") or col in ("snapshot_date", "month", "campaign_month"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df