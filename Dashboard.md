# Dashboard guide

A walkthrough of every section of the Streamlit dashboard: what it shows, how each number is calculated, and how to read it.
Start it with `streamlit run app/streamlit_app.py` (after `generate_data.py` and `run_pipeline.py`).

> All data is synthetic (fictional company "Flowdesk"). "Predicted CLV" means predicted **revenue over the next 12 months**, not lifetime value and not profit.

---

## Sidebar filters (apply to every page)

| Filter | What it does |
|---|---|
| **Prediction snapshot** | *Current scoring* (2025-12-31): predictions for today, outcomes not yet known. *Out-of-time backtest* (2024-12-31): predictions made a year ago, compared with what actually happened in 2025. |
| **Subscription plan** | Basic, Professional, Enterprise. For scored customers it uses the plan held at the snapshot date; for revenue and churn history it uses the customer's latest plan. |
| **Industry** | 8 industries. |
| **Company size (employees)** | 1-49, 50-249, 250-999, 1000+. |
| **Acquisition channel** | Organic Search, Paid Advertising, Referral, Social Media, Direct Sales. |
| **Signup date range** | Limits customers by when they first subscribed. |
| **Revenue / churn period** | Limits the months used for revenue, MRR and churn figures. |

If the filters leave no customers, each page shows a short message instead of charts. A blue note appears in the sidebar when any filter is active.

---

## 1. Executive Overview

**Purpose:** a one-screen health check of the business and the model's outlook.

**KPI cards**

| Card | How it is calculated |
|---|---|
| Historical revenue | Sum of recognised revenue (after discounts and refunds) in the selected period |
| Avg predicted 12-month CLV | Mean predicted revenue for the next 12 months across scored customers |
| Active customers | Customers with an active subscription at the snapshot date |
| MRR (last month in period) | Monthly recurring revenue in the last month of the selected period |
| Avg monthly logo churn | Cancellations ÷ customers already subscribed at the start of each month, pooled over the period |
| CLV-to-CAC (revenue-based) | Average predicted 12-month revenue ÷ average acquisition cost. Based on revenue, not profit. |

**Charts**
- **Monthly revenue and MRR:** bars for recognised revenue, line for MRR. Hover for exact dollar values.
- **Customers by value segment:** share of customers in High, Medium and Low Value.
- **Avg predicted 12-month CLV by snapshot:** compares the two snapshots under the current filters.
- **Monthly logo churn rate:** hover shows cancellations and the customers they are measured against.

**Key business insights:** sentences generated from the filtered data: how much of predicted revenue the High Value segment holds, how many high-value accounts have elevated churn risk and their predicted revenue, the best and worst channels by CLV-to-CAC, accounts with a 25%+ drop in logins, and accounts with recent payment failures. They describe patterns, not causes.

---

## 2. CLV & Segmentation

**Purpose:** see who is worth the most, how reliable the predictions are, and look up individual customers.

**Segments**
- Thresholds are fixed dollar cut-offs from quantiles of *training-period* realised revenue: **Low < $2,180**, **Medium $2,180 to $10,100**, **High ≥ $10,100** (bottom 40%, middle 40%, top 20%).
- The donut shows customer mix; the histogram shows the distribution of predicted revenue with the two cut-offs marked.

**Segment-level performance table:** customers and share, predicted 12-month revenue, average predicted CLV, actual revenue to date (cumulative, all history), average MRR, average logins, feature adoption, average CAC, observed 12-month churn, share of customers at Medium/High risk, and CLV-to-CAC.
Observed churn is only available on the backtest snapshot.

**Predicted vs actual (backtest snapshot only)**
- Cards: MAE, RMSE, R² and portfolio bias (average predicted minus average actual per customer).
- Scatter on a log scale: points near the dashed line are accurate. A second chart compares predicted and actual totals per segment.
- On the current snapshot this section explains that outcomes have not happened yet.

**Customer table:** sorted by predicted revenue. The search box matches customer ID, industry, plan or channel. Columns include predicted and actual revenue, MRR, churn probability and risk tier.

**Individual customer lookup:** pick a customer to see predicted revenue, segment, MRR, churn probability, risk tier, actual revenue (backtest) and the warning signals that apply.

---

## 3. Churn & Retention

**Purpose:** find customers who may leave, especially the valuable ones, and decide what to do.

**Cards:** average monthly logo churn; *period cohort churn* (of customers subscribed at the start of the period, the share that cancelled by its end; shows n/a if the period starts at the first month); number of High-risk accounts; and High Value accounts at Medium/High risk with their predicted revenue.

**Risk tiers** come from a churn-probability model (chance of cancelling within 12 months):

| Tier | Probability | Observed 12-month churn in the backtest |
|---|---|---:|
| Low | below 15% | 11% |
| Medium | 15% to 35% | 24% |
| High | 35% or more | 45% |

Tiers rank customers well but are not certain: many High-tier customers stay and some Low-tier customers leave. The expander *How reliable is the risk tier?* shows the backtest AUC (0.73) and the table above. No customer is labelled "will churn".

**Sections**
- **Monthly logo churn** line and **customers by risk tier** bars.
- **High-value customers with elevated churn risk:** top 50 by predicted revenue, with login trend, payment failures, unresolved tickets and the signals that apply.
- **Usage decline and payment-failure indicators:** how many customers show each warning signal, and a scatter of predicted revenue against churn probability (dotted lines at 15% and 35%). Customers high on the chart and to the right are worth protecting first.
- **At-risk customer list:** choose High and/or Medium; shows the top 200 by predicted revenue.
- **Recommended retention actions:** pick an account to see actions tied to its signals. High Value accounts at elevated risk also get an executive-sponsor action.

**Warning signals** (transparent rules that explain *why* an account looks risky):

| Signal | Condition | Points |
|---|---|---:|
| Severe usage decline | Logins down more than 50% vs the prior 3 months | 30 |
| Usage decline | Logins down 25-50% | 20 |
| Sharp usage drop | A monthly usage drop of 40%+ in the last 6 months | 10 |
| Repeated payment failures | 2+ in the last 6 months | 25 |
| Payment failure | 1 in the last 6 months | 15 |
| Unresolved tickets | Any in the last 180 days | 10 |
| Repeat tickets | 3+ in the last 90 days | 10 |
| Low CSAT | Average below 3 over 180 days | 10 |
| Recent downgrade | In the last 6 months | 15 |
| Low adoption | Feature adoption below 30% | 10 |

The rule score on its own ranks churn weakly (AUC 0.58), so it is used for explanations and actions, while the model sets the tier. The *Signal-to-action playbook* expander lists every action.

---

## 4. Acquisition & Marketing ROI

**Purpose:** compare channels on cost and return, without overstating what the data proves.

**Gross margin slider** (default 80%): an illustrative assumption used only for the margin-adjusted ratio and payback months. Margin is not in the data.

**Charts:** average CAC; average predicted 12-month CLV; CLV-to-CAC; revenue contribution by channel; realised revenue per acquired customer ÷ CAC.

**Two ways to read return**
- **Predicted CLV ÷ CAC** looks forward but covers currently active customers only (survivors).
- **Realised revenue ÷ CAC** covers every acquired customer, including those who churned, so it is the more conservative view.

**Channel scorecard:** customers acquired, active customers, average CAC, predicted CLV, both ratios, margin-adjusted ratio, actual revenue to date and CAC payback months at the assumed margin.

**Campaign performance:** campaign cost, platform-attributed new customers and revenue (first three months), cost per attributed customer, and attributed revenue per dollar. This is a revenue ratio, not profit, and not incremental impact.

**Allocation opportunities:** a rule-based label per channel from its CLV-to-CAC: **3x or more** "candidate for an incremental-budget test", **1.5x to 3x** "optimise before scaling", **below 1.5x** "review targeting and cost efficiency". The page warns that this is observational: channels attract different kinds of customers, so a strong ratio does not prove extra spend would give the same return. Test with a holdout or budget experiment before moving money.

---

## 5. Customer Explorer

**Purpose:** a full profile of one customer.

- **Find a customer:** search by ID, then choose from the list (scored customers first, then churned or ineligible ones; up to 500 shown).
- **Header:** industry, size, signup date, channel, region and acquisition cost.
- **Cards:** predicted 12-month revenue, segment, churn probability, risk tier, plan and billing cycle. On the backtest snapshot the actual revenue for the following 12 months is shown too.
- **Tabs:**
  - *Revenue:* monthly recognised revenue and MRR.
  - *Engagement:* logins, active days and feature adoption over time.
  - *Support & billing:* tickets (a blank resolution time means unresolved) and billing events such as upgrades, downgrades and payment failures.
  - *Prediction drivers:* how much each group of features (revenue level, revenue momentum, usage level, usage trend, support, billing health, plan and firmographics, tenure and renewal) pushes this customer's prediction up or down compared with a typical customer. It is calculated by replacing each group with typical values and measuring the change. It explains the model, not cause and effect.
  - *Recommended actions:* the warning signals and the actions linked to them.
- Customers that are not scored at the chosen snapshot (cancelled before it, or under one month of history) still show their history, and an info message explains why there is no prediction.

---

## Reading the numbers correctly

- **12-month revenue is not lifetime value and not profit.**
- Predicted CLV and CLV-to-CAC include active customers only.
- A churn probability is an estimate of risk, never a certainty.
- Channel and segment comparisons show association, not causation.
- The data is synthetic, so the results show that the pipeline works, not how a real company would perform.
