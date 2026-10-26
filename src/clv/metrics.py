"""Reusable business metric functions (pure, unit-tested)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def safe_divide(a, b):
    """a / b, returning NaN for zero or missing denominators instead of inf."""
    try:
        if b is None or pd.isna(b) or b == 0:
            return np.nan
        return a / b
    except TypeError:
        return np.nan


def clv_to_cac(avg_clv: float, avg_cac: float) -> float:
    """Revenue-based ratio: predicted 12-month revenue per customer / average acquisition cost."""
    return safe_divide(avg_clv, avg_cac)


def payback_months(cac: float, monthly_revenue: float, gross_margin: float = C.DEFAULT_GROSS_MARGIN) -> float:
    """Months of gross-margin-adjusted revenue needed to recover CAC (illustrative; margin is an assumption)."""
    return safe_divide(cac, monthly_revenue * gross_margin)


def profit_clv(revenue_clv: float, gross_margin: float = C.DEFAULT_GROSS_MARGIN) -> float:
    """Illustrative gross-profit CLV: revenue x assumed gross margin (not derived from data)."""
    return revenue_clv * gross_margin


def monthly_churn(revenue: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Monthly logo churn.

    churn(m) = cancellations effective in month m / customers already subscribed at the start of m,
    where customers subscribed at the start of m = customers billed in m minus subscriptions that began in m.
    Mirrors the `monthly_churn` SQL query exactly.
    """
    billed = (revenue[revenue["recurring_revenue"] > 0].groupby(revenue["month"].dt.to_period("M"))["customer_id"]
              .nunique().rename("billed_customers"))
    ev = events.assign(ym=events["event_date"].dt.to_period("M"))
    starts = ev[ev["event_type"].isin(["new_subscription", "reactivation"])].groupby("ym").size().rename("starts")
    cancels = ev[ev["event_type"] == "cancellation"].groupby("ym").size().rename("cancellations")
    out = pd.concat([billed, starts, cancels], axis=1).dropna(subset=["billed_customers"]).fillna(0)
    out["active_at_start"] = out["billed_customers"] - out["starts"]
    out["churn_rate"] = out["cancellations"] / out["active_at_start"].replace(0, np.nan)
    out = out.reset_index().rename(columns={"index": "ym"})
    out["month"] = out["ym"].dt.to_timestamp(how="end").dt.normalize()
    return out[["month", "active_at_start", "cancellations", "churn_rate"]]


def period_churn_rate(revenue: pd.DataFrame, events: pd.DataFrame, start, end) -> float:
    """Cohort churn over (start, end]: of customers billed in the month of `start` and still subscribed after it,
    the share that cancelled by `end`. Denominator = customers active at the start of the period."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    billed_start = revenue[(revenue["month"] == start) & (revenue["recurring_revenue"] > 0)]["customer_id"].unique()
    ev = events[events["event_type"] == "cancellation"]
    cancelled_at_start = ev[ev["event_date"] <= start]["customer_id"].unique()
    cohort = np.setdiff1d(billed_start, cancelled_at_start)
    churned = ev[(ev["event_date"] > start) & (ev["event_date"] <= end) & ev["customer_id"].isin(cohort)]["customer_id"].nunique()
    return safe_divide(churned, len(cohort))


def mrr_series(revenue: pd.DataFrame) -> pd.Series:
    g = revenue.assign(mrr=revenue["recurring_revenue"] + revenue["expansion_revenue"] - revenue["contraction_revenue"])
    return g.groupby("month")["mrr"].sum()


def format_usd(x, compact: bool = True) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    sign = "-" if x < 0 else ""
    a = abs(x)
    if compact and a >= 1e6:
        return f"{sign}${a / 1e6:,.2f}M"
    if compact and a >= 1e4:
        return f"{sign}${a / 1e3:,.1f}K"
    return f"{sign}${a:,.0f}"


def format_pct(x, digits: int = 1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.{digits}f}%"