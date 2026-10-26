"""Data dictionary and simulation assumptions (rendered to docs/data_dictionary.md)."""
from __future__ import annotations

from . import config as C

TABLES = {
    "customers": ("One row per customer (account). Static attributes known at signup.", {
        "customer_id": "Primary key (C00001...).",
        "signup_date": "Date the customer first subscribed. Range 2023-01-01 to 2025-06-30.",
        "industry": "One of 8 industries.",
        "company_size": "Employee band derived from employee_count: 1-49, 50-249, 250-999, 1000+.",
        "employee_count": "Employees at the customer company. ~1.2% missing (not provided at signup).",
        "acquisition_channel": "Organic Search, Paid Advertising, Referral, Social Media, Direct Sales.",
        "acquisition_cost": "Customer acquisition cost (CAC) in USD attributed to this customer. ~1.5% missing; includes rare outliers.",
        "country": "Country of the customer (11 countries).",
        "sales_region": "North America, EMEA, APAC, LATAM (derived from country)."}),
    "subscriptions": ("One row per subscription episode. A reactivated customer has a second row. "
                      "Count columns and MRR are lifetime/final values of the episode - NEVER used as model features.", {
        "subscription_id": "Primary key (S000001...).",
        "customer_id": "Foreign key to customers.",
        "subscription_plan": "Plan at the end of the episode (Basic, Professional, Enterprise).",
        "billing_cycle": "Monthly or Annual. Annual contracts can only be cancelled at renewal.",
        "monthly_recurring_revenue": "List MRR (before discount) at the end of the episode, USD.",
        "subscription_start_date": "Start date of the episode.",
        "subscription_end_date": "Cancellation effective date (end of month). Empty while active.",
        "subscription_status": "Active or Cancelled.",
        "upgrade_count": "Plan upgrades during the episode.",
        "downgrade_count": "Plan downgrades during the episode.",
        "discount_percentage": "Contract discount (annual prepay 10% + promotional/negotiated), 0-30.",
        "payment_failure_count": "Payment failures during the episode."}),
    "usage_monthly": ("Customer-month product usage while subscribed (month-end dated).", {
        "customer_id": "Foreign key to customers (with observation_month forms the key).",
        "observation_month": "Month-end date.",
        "monthly_logins": "Total logins in the month. ~1.5% missing; rare spikes from integrations/bots.",
        "active_days": "Days with any activity (1-28).",
        "features_used": "Distinct features used. ~2.5% missing.",
        "feature_adoption_rate": "features_used / features available on the plan (0-1). ~2.5% missing.",
        "session_duration_minutes": "Average session length. ~3% missing.",
        "usage_change_percentage": "Month-over-month change in logins (%). Empty in a subscription's first month or when logins are missing."}),
    "support_tickets": ("One row per support ticket.", {
        "ticket_id": "Primary key (T000001...).",
        "customer_id": "Foreign key to customers.",
        "created_date": "Creation date.",
        "issue_category": "Billing, Bug Report, How-To / Training, Integration, Performance, Feature Request, Account Access.",
        "priority": "Low, Medium, High, Critical.",
        "resolution_time_hours": "Hours to resolve. EMPTY = unresolved ticket (~4%).",
        "customer_satisfaction_score": "CSAT 1-5; ~38% of tickets have no survey response."}),
    "revenue_monthly": ("Customer-month revenue ledger (month-end dated). Identity: recognized = recurring + expansion - contraction - discounts - refunds.", {
        "customer_id": "Foreign key to customers (with month forms the key).",
        "month": "Month-end date.",
        "recurring_revenue": "MRR carried into the month (list price, before changes). 0 on post-cancellation adjustment rows.",
        "expansion_revenue": "Increase in MRR effective this month (upgrades, seat growth; rare large jumps).",
        "contraction_revenue": "Decrease in MRR effective this month (downgrades, seat reductions), positive number.",
        "refunds": "Refunds issued (~0.35% of customer-months; plus explicit post-cancellation adjustments).",
        "discounts": "Contract discount applied to the month's gross MRR.",
        "recognized_revenue": "Net revenue recognised. Negative only on post-cancellation refund adjustment rows."}),
    "marketing_channels": ("Campaign spend and platform-attributed results per channel-month.", {
        "acquisition_channel": "Channel (with campaign_month forms the key).",
        "campaign_cost": "Cohort CAC total x 1.00-1.20 overhead (non-attributable programme cost).",
        "attributed_new_customers": "Platform-reported new customers; differs slightly from true cohort size for paid/social.",
        "attributed_revenue": "Recognised revenue in the cohort's first three months.",
        "campaign_month": "Month-end of the signup cohort."}),
    "billing_events": ("SUPPLEMENTARY table (not in the original brief): dated lifecycle events that make point-in-time "
                       "features possible without leaking lifetime totals.", {
        "event_id": "Primary key (E0000001...).",
        "customer_id": "Foreign key to customers.",
        "subscription_id": "Foreign key to subscriptions.",
        "event_date": "Event date.",
        "event_type": "new_subscription, upgrade, downgrade, payment_failure, cancellation, reactivation.",
        "plan_before": "Plan before the event (empty for new subscriptions; equals plan for payment failures).",
        "plan_after": "Plan after the event (empty for cancellations).",
        "mrr_after": "List MRR after the event (0 on cancellation)."}),
}

ASSUMPTIONS = [
    "All data is SYNTHETIC and produced by `src/clv/data_generation.py` with seed %d. It does not describe any real company." % C.SEED,
    "36 monthly periods (Jan 2023 - Dec 2025) are simulated, not 24, so a 12-month target can be evaluated out-of-time (see README). Signups run Jan 2023 - Jun 2025.",
    "Billing is whole-month: a customer is billed for the signup month and for the cancellation month; there is no proration.",
    "Churn is a monthly hazard (annual contracts: only at renewal) driven by plan, company size, industry, channel, latent engagement, usage decline, payment failures in the last 3 months, unresolved tickets, low CSAT, downgrades in the last 6 months, tenure, seasonality (higher in Jan/Feb/Dec), heavy discounts and a customer-level random effect (larger for Enterprise, so Enterprise retention is more varied). Minimum lifetime is 3 billed months.",
    "Usage follows a mean-reverting health state with random 2-5 month decline episodes; episodes raise churn hazard but many do not end in churn, so churn is not separable from usage alone.",
    "Seasonality: signups peak in Q4/Jan, usage dips in Jul/Aug/Dec, churn rises around Jan.",
    "7%% of cancellations reactivate 2-8 months later at 95%% of prior MRR; 6%% of cancellations incur a one-off refund adjustment in the following month (the only revenue allowed after cancellation).",
    "Missing values and outliers are injected deliberately (see per-column notes). Rows are not otherwise cleaned.",
    "Gross margin and any profit measure are NOT in the data. The dashboard's margin slider is an illustrative assumption.",
    "Pricing: plan base price x seat factor ((employees/40)^0.45, clipped 0.8-6) x lognormal noise. Base prices: Basic $79, Professional $299, Enterprise $899.",
]


def render_markdown() -> str:
    out = ["# Data dictionary", "", "> Synthetic data for a fictional company (Flowdesk). Generated by `scripts/generate_data.py`.", "",
           "## Simulation assumptions", ""]
    out += [f"- {a}" for a in ASSUMPTIONS]
    for name, (desc, cols) in TABLES.items():
        out += ["", f"## `{name}.csv`", "", desc, "", "| Column | Description |", "|---|---|"]
        out += [f"| `{c}` | {d} |" for c, d in cols.items()]
    out += ["", "## Model-output tables (`data/processed/customer_snapshots.csv`, SQLite `customer_snapshots`)", "",
            "One row per scored customer per snapshot: all model features (see `src/clv/features.py`), plus `snapshot_date`, `split` "
            "(`backtest` = 2024-12-31 with observed outcomes, `current` = 2025-12-31), `predicted_12m`, `actual_12m` (empty for current), "
            "`segment`, `churn_probability`, `risk_tier`, `signal_score`, `risk_flags`, `churned_12m`, `acquisition_cost`.", ""]
    return "\n".join(out)