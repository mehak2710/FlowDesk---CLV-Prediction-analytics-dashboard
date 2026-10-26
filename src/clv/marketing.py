"""Acquisition-channel economics. Revenue-based measures only (no margin data in the dataset)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .metrics import clv_to_cac, payback_months, safe_divide


def channel_performance(customers: pd.DataFrame, revenue: pd.DataFrame, snapshot: pd.DataFrame,
                        gross_margin: float = C.DEFAULT_GROSS_MARGIN) -> pd.DataFrame:
    """Per-channel table.

    customers : acquired customers in scope (all, including churned) with acquisition_cost
    revenue   : revenue_monthly rows
    snapshot  : scored snapshot rows (active customers) with predicted_12m, mrr_now
    Two complementary views:
      * realised revenue per acquired customer / CAC   -> includes churned customers (no survivorship bias)
      * avg predicted 12-month revenue (active customers only) / CAC -> forward-looking, survivors only
    """
    if customers.empty:
        return pd.DataFrame()
    realised = revenue.groupby("customer_id")["recognized_revenue"].sum().rename("realised")
    c = customers.merge(realised, on="customer_id", how="left").fillna({"realised": 0.0})
    g = c.groupby("acquisition_channel")
    out = pd.DataFrame({
        "customers_acquired": g.size(),
        "avg_cac": g["acquisition_cost"].mean(),
        "total_cac_spend": g["acquisition_cost"].sum(),
        "actual_revenue_to_date": g["realised"].sum(),
    })
    out["revenue_per_acquired_customer"] = out["actual_revenue_to_date"] / out["customers_acquired"]
    out["realised_revenue_to_cac"] = out["revenue_per_acquired_customer"] / out["avg_cac"]
    if not snapshot.empty:
        s = snapshot.merge(customers[["customer_id", "acquisition_channel"]].drop_duplicates(), on="customer_id",
                           how="inner", suffixes=("", "_c"))
        sg = s.groupby("acquisition_channel")
        out["active_customers"] = sg.size()
        out["avg_predicted_clv"] = sg["predicted_12m"].mean()
        out["total_predicted_12m"] = sg["predicted_12m"].sum()
        out["avg_mrr_active"] = sg["mrr_now"].mean()
    else:
        for col in ["active_customers", "avg_predicted_clv", "total_predicted_12m", "avg_mrr_active"]:
            out[col] = np.nan
    out["clv_to_cac"] = [clv_to_cac(a, b) for a, b in zip(out["avg_predicted_clv"], out["avg_cac"])]
    out["payback_months"] = [payback_months(a, b, gross_margin) for a, b in zip(out["avg_cac"], out["avg_mrr_active"])]
    out["gm_adj_clv_to_cac"] = out["clv_to_cac"] * gross_margin
    return out.reindex([ch for ch in C.CHANNELS if ch in out.index])


def campaign_performance(marketing: pd.DataFrame) -> pd.DataFrame:
    """Platform-attributed campaign results per channel (revenue-based return, NOT incremental, NOT profit)."""
    if marketing.empty:
        return pd.DataFrame()
    g = marketing.groupby("acquisition_channel")[["campaign_cost", "attributed_new_customers", "attributed_revenue"]].sum()
    g["cost_per_attributed_customer"] = [safe_divide(a, b) for a, b in zip(g["campaign_cost"], g["attributed_new_customers"])]
    g["attributed_revenue_per_cost"] = [safe_divide(a, b) for a, b in zip(g["attributed_revenue"], g["campaign_cost"])]
    return g.reindex([ch for ch in C.CHANNELS if ch in g.index])


def allocation_signals(perf: pd.DataFrame) -> pd.DataFrame:
    """Rule-based, descriptive signals for budget discussions. They flag where to RUN EXPERIMENTS; they do not
    estimate incremental impact (observational data cannot separate channel effect from customer mix)."""
    if perf.empty:
        return perf
    d = perf[["customers_acquired", "total_cac_spend", "avg_cac", "avg_predicted_clv", "clv_to_cac",
              "realised_revenue_to_cac"]].copy()
    d["spend_share"] = d["total_cac_spend"] / d["total_cac_spend"].sum()
    def sig(r):
        if pd.isna(r["clv_to_cac"]):
            return "Insufficient data"
        if r["clv_to_cac"] >= 3.0:
            return "Strong ratio: candidate for an incremental-budget test"
        if r["clv_to_cac"] >= 1.5:
            return "Adequate: optimise before scaling"
        return "Weak ratio: review targeting and cost efficiency"
    d["signal"] = d.apply(sig, axis=1)
    return d