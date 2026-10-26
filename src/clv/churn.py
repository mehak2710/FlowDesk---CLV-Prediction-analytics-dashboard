"""Churn risk: a validated probability model for prioritisation + transparent rule signals for explanation.

* `churn_probability` (from `modeling.train_churn_model`) is the estimated chance of cancelling within 12 months.
  Risk tiers (Low / Medium / High) are fixed probability bands, validated on the out-of-time backtest.
* `signal_score` / `risk_flags` are a documented additive rule set (0-100) that explains WHY an account looks
  risky and drives the recommended actions. It is a weaker ranker than the model and is reported as such.
* No customer is "certain to churn": a tier is a probabilistic prioritisation aid, not a prediction of fate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# (flag id, label, points, condition)  -- single source of truth for score, flags and docs
RISK_RULES = [
    ("usage_decline_severe", "Logins down >50% vs prior 3 months", 30, lambda d: d["logins_trend_ratio"] < 0.50),
    ("usage_decline", "Logins down 25-50% vs prior 3 months", 20, lambda d: (d["logins_trend_ratio"] >= 0.50) & (d["logins_trend_ratio"] < 0.75)),
    ("usage_drop_spike", "A monthly usage drop of 40%+ in last 6 months", 10, lambda d: d["usage_change_min_6m"] <= -40),
    ("payment_failures_multi", "2+ payment failures in last 6 months", 25, lambda d: d["payment_failures_6m"] >= 2),
    ("payment_failure", "1 payment failure in last 6 months", 15, lambda d: d["payment_failures_6m"] == 1),
    ("unresolved_tickets", "Unresolved support ticket(s) in last 180 days", 10, lambda d: d["unresolved_180d"] >= 1),
    ("repeat_tickets", "3+ support tickets in last 90 days", 10, lambda d: d["tickets_90d"] >= 3),
    ("low_csat", "Average CSAT below 3 (last 180 days)", 10, lambda d: d["csat_180d"] < 3),
    ("recent_downgrade", "Downgrade in last 6 months", 15, lambda d: d["downgrades_6m"] >= 1),
    ("low_adoption", "Feature adoption below 30%", 10, lambda d: d["adoption_3m"] < 0.30),
]
TIER_PROB_CUTOFFS = {"High": 0.35, "Medium": 0.15}  # P(churn within 12m): >=0.35 High, >=0.15 Medium, else Low

ACTIONS = {
    "usage_decline_severe": "CSM outreach within a week: usage review call and re-onboarding of key workflows.",
    "usage_decline": "Send adoption nudges and schedule a value-realisation check-in.",
    "usage_drop_spike": "Check for an owner change or integration outage; confirm the account champion is still active.",
    "payment_failures_multi": "Billing team to contact the account owner; offer a payment-method update and annual invoicing.",
    "payment_failure": "Trigger dunning follow-up and a payment-method update reminder.",
    "unresolved_tickets": "Escalate open tickets to support leadership and give the customer a dated resolution plan.",
    "repeat_tickets": "Review ticket themes for a product or training root cause; offer a technical review session.",
    "low_csat": "Follow up personally on low-CSAT interactions before the next renewal conversation.",
    "recent_downgrade": "Run a downgrade review: understand what value gap triggered it and propose a right-sized plan.",
    "low_adoption": "Offer targeted training on unused features that match the customer's use case.",
}
LABELS = {r[0]: r[1] for r in RISK_RULES}


def assign_risk_tier(prob: pd.Series) -> pd.Series:
    p = pd.Series(prob)
    tier = np.where(p >= TIER_PROB_CUTOFFS["High"], "High", np.where(p >= TIER_PROB_CUTOFFS["Medium"], "Medium", "Low"))
    return pd.Series(pd.Categorical(tier, categories=["Low", "Medium", "High"], ordered=True), index=p.index).where(p.notna())


def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add signal_score (0-100) and risk_flags (pipe-separated rule ids). Missing signals earn no points."""
    d = df.copy()
    score = pd.Series(0, index=d.index, dtype=float)
    flags = pd.Series("", index=d.index)
    for fid, _, pts, cond in RISK_RULES:
        hit = cond(d).fillna(False)
        score = score + hit * pts
        flags = flags + np.where(hit, fid + "|", "")
    d["signal_score"] = score.clip(upper=100)
    d["risk_flags"] = flags.str.rstrip("|")
    return d


def flag_labels(flags: str) -> list[str]:
    return [LABELS[f] for f in flags.split("|") if f]


def recommend_actions(flags: str, segment: str | None = None, risk_tier: str | None = None) -> list[str]:
    acts = [ACTIONS[f] for f in flags.split("|") if f]
    if segment == "High Value" and risk_tier in ("Medium", "High"):
        acts.insert(0, "Assign an executive sponsor and agree a joint success plan; this account carries high predicted revenue.")
    if not acts:
        acts = ["No elevated risk signals. Keep standard cadence; consider an expansion conversation."]
    return acts


def validate_risk(df: pd.DataFrame) -> dict:
    """Backtest on a frame with observed `churned_12m`: AUC of the model probability and of the rule score,
    plus observed churn by tier."""
    d = df.dropna(subset=["churned_12m"])
    out = {"n": int(len(d)), "base_rate": float(d["churned_12m"].mean())}
    if d["churned_12m"].nunique() > 1:
        out["auc_churn_probability"] = float(roc_auc_score(d["churned_12m"], d["churn_probability"]))
        out["auc_rule_signal_score"] = float(roc_auc_score(d["churned_12m"], d["signal_score"]))
    tiers = d.groupby("risk_tier", observed=True)["churned_12m"].agg(["size", "mean"])
    out["by_tier"] = {t: {"customers": int(r["size"]), "churn_rate_12m": float(r["mean"])} for t, r in tiers.iterrows()}
    return out