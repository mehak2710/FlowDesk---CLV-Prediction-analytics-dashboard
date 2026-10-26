"""Transparent value segmentation from predicted 12-month revenue."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def fit_thresholds(y_train, quantiles: dict[str, float] = C.SEGMENT_QUANTILES) -> dict[str, float]:
    """USD cut-offs from quantiles of *training-period* realised 12-month revenue.

    Low Value    : revenue <  `low_upper`   (default: bottom 40% of training customers)
    Medium Value : low_upper <= revenue < high_lower
    High Value   : revenue >= `high_lower`  (default: top 20%)
    Cut-offs are fixed in dollars so segments stay comparable across snapshots.
    """
    y = pd.Series(y_train).dropna()
    return {"low_upper": float(y.quantile(quantiles["low_upper"])),
            "high_lower": float(y.quantile(quantiles["high_lower"])),
            "quantiles": dict(quantiles)}


def assign_segments(values, thresholds: dict[str, float]) -> pd.Series:
    v = pd.Series(values)
    seg = np.where(v >= thresholds["high_lower"], "High Value",
                   np.where(v >= thresholds["low_upper"], "Medium Value", "Low Value"))
    out = pd.Series(pd.Categorical(seg, categories=C.SEGMENTS, ordered=True), index=v.index)
    return out.where(v.notna())


def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Segment-level table. Expects columns: segment, predicted_12m, cum_revenue, mrr_now, logins_3m,
    adoption_3m, acquisition_cost, churned_12m (NaN when unobserved), risk_tier."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("segment", observed=True)
    out = pd.DataFrame({
        "customers": g.size(),
        "predicted_12m_revenue": g["predicted_12m"].sum(),
        "avg_predicted_clv": g["predicted_12m"].mean(),
        "actual_revenue_to_date": g["cum_revenue"].sum(),
        "avg_mrr": g["mrr_now"].mean(),
        "avg_logins_3m": g["logins_3m"].mean(),
        "avg_adoption": g["adoption_3m"].mean(),
        "avg_cac": g["acquisition_cost"].mean(),
        "observed_12m_churn_rate": g["churned_12m"].mean(),
        "pct_elevated_risk": g["risk_tier"].apply(lambda s: (s.isin(["High", "Medium"])).mean()),
    })
    out["pct_of_customers"] = out["customers"] / out["customers"].sum()
    out["clv_to_cac"] = out["avg_predicted_clv"] / out["avg_cac"]
    return out.reindex([s for s in C.SEGMENTS if s in out.index])