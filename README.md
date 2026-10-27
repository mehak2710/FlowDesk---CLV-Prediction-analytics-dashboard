# SaaS CLV Analytics

![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-models-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red) ![Tests](https://img.shields.io/badge/tests-59%20passing-brightgreen)

## Description

An end-to-end analytics project for a fictional B2B SaaS company ("Flowdesk") that sells Basic, Professional and Enterprise subscriptions. It predicts how much revenue each customer will generate over the **next 12 months**, groups customers into High / Medium / Low value segments, estimates churn risk, compares acquisition channels, and presents everything in an interactive Streamlit dashboard.

The business questions it answers:
- Which customers will generate the most revenue, and which segments are most and least valuable?
- Which customers show early signs of churn, and which of them are high value?
- Which acquisition channels bring customers with the best long-term return?
- Where should retention and marketing effort go?

> **All data is synthetic.** It comes from a simulator with a fixed random seed. The project shows a working pipeline on data with known structure; it is not evidence about real businesses. Predicted CLV here means 12-month **revenue**, not lifetime value and not profit.

## Features

- **Realistic synthetic data:** 5,000 customers and 7 linked tables (about 194K rows) covering usage, support tickets, payment failures, upgrades and downgrades, annual billing, discounts, refunds, reactivations, missing values and outliers. Churn is probabilistic, never perfectly predictable.
- **Automated data validation:** unique keys, foreign keys, date ranges, the revenue identity, and no revenue or usage after cancellation.
- **Leakage-safe features and target:** 38 point-in-time features; the target is revenue in the 12 months after a snapshot date; tested by truncating the data at the snapshot date.
- **Model comparison:** two baselines, Linear Regression, Random Forest and Gradient Boosting, evaluated on a time-separated holdout with MAE, RMSE and R².
- **Value segmentation:** High / Medium / Low from predicted revenue, with fixed dollar thresholds from training-data quantiles.
- **Churn analytics:** historical churn rate, a validated churn-probability model, rule-based warning signals, and recommended retention actions.
- **Marketing ROI:** CAC, predicted CLV, CLV-to-CAC, realised revenue per acquired customer and campaign results by channel, with clear revenue-vs-profit caveats.
- **Prediction explanations** for individual customers.
- **SQLite and SQL** for analytics, saved models with joblib, and **59 automated tests**.

## Results at a glance

Out-of-time test on 2,719 customers (predictions made 2024-12-31, outcomes observed through 2025):

| Model | MAE | RMSE | R² | Bias / customer |
|---|---:|---:|---:|---:|
| Baseline: MRR × 12 | $1,778 | $4,256 | 0.859 | +$1,197 |
| Baseline: trailing 3-month revenue × 12 | **$1,212** | $3,709 | 0.893 | +$327 |
| Linear Regression | $1,684 | $3,941 | 0.879 | +$128 |
| **Random Forest (selected)** | $1,487 | **$3,600** | **0.899** | -$101 |
| Gradient Boosting | $1,544 | $3,634 | 0.897 | -$19 |

- The selected model beats both baselines on RMSE, R² and bias. Its portfolio total is within **1.2%** of actual revenue; the trailing-revenue baseline is off by +3.9% and MRR × 12 by +14.3%.
- It does **not** beat the trailing-revenue baseline on MAE. Most customers don't cancel in a year, so "same as the last 3 months" is close to the median outcome, while squared-error models target the mean that portfolio forecasts need.
- **Churn model:** out-of-time AUC **0.73**. Observed 12-month churn by risk tier: **Low 11%, Medium 24%, High 45%**.

| Value segment | Customers | Avg predicted 12-month revenue | Observed 12-month churn |
|---|---:|---:|---:|
| High | 635 (23%) | $24,005 | 11% |
| Medium | 1,229 (45%) | $5,171 | 18% |
| Low | 855 (31%) | $1,000 | 36% |

## Dashboard

| Page | What it shows |
|---|---|
| **Executive Overview** | Revenue, MRR, active customers, churn, average predicted CLV, CLV-to-CAC, trends, automatic insights |
| **CLV & Segmentation** | Segment table, CLV distribution, predicted vs actual, searchable customer table, single-customer lookup |
| **Churn & Retention** | Monthly churn trend, risk tiers, high-value customers at risk, usage-decline and payment-failure signals, recommended actions |
| **Acquisition & Marketing ROI** | CAC, predicted CLV and CLV-to-CAC by channel, campaign results, allocation signals |
| **Customer Explorer** | One customer's revenue, engagement, tickets, billing events, prediction drivers and actions |

Sidebar filters (snapshot, plan, industry, company size, acquisition channel, signup dates, period) apply on every page, and each is covered by a test.

<!-- Add screenshots to docs/screenshots/ and remove these comment markers:
![Executive overview](docs/screenshots/01_overview.png)
![Churn and retention](docs/screenshots/03_churn_retention.png)
![Customer explorer](docs/screenshots/05_customer_explorer.png)
-->

## How to run

```bash
git clone https://github.com/mehak2710/saas-clv-analytics.git
cd saas-clv-analytics
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/generate_data.py    # simulate, validate, write data/raw/*.csv   (~5 s)
python scripts/run_pipeline.py     # features, models, scoring, SQLite database  (~1 min)
python -m pytest -q                # 59 tests
streamlit run app/streamlit_app.py
```

## How it works

1. **Simulate** (`data_generation.py`): each customer has a latent engagement level and a usage-health state that changes monthly. Usage, tickets, payment failures, plan changes and revenue are drawn from that state. Churn comes from a hazard driven by past signals (usage decline, payment failures, unresolved tickets, downgrades) plus plan, size, industry, channel, tenure and seasonality. Annual contracts can only cancel at renewal.
2. **Validate** (`validation.py`): checks keys, relationships, dates and the revenue identity on every run.
3. **Build features** (`features.py`): revenue level and momentum, usage level and trend, support experience, billing health, plan, firmographics, tenure and months to renewal, all computed as of a snapshot date.
4. **Model** (`modeling.py`): each model predicts a realisation ratio (12-month revenue ÷ current run-rate), then multiplies back. The model family is chosen on validation, refit on train + validation, and scored once on test.
5. **Segment, score risk, analyse channels** (`segmentation.py`, `churn.py`, `marketing.py`), then load everything into SQLite for the dashboard.

| Split | Snapshot dates |
|---|---|
| Train | 2023-03-31, 2023-06-30, 2023-09-30 |
| Validation | 2023-12-31 |
| Test | 2024-12-31 (outcomes through 2025) |
| Current scoring | 2025-12-31 (outcomes not yet observed) |

Key design choices: 36 months are simulated, not 24, so a 12-month target can be tested on genuinely unseen months. A seventh table, `billing_events`, lets features use trailing windows such as "payment failures in the last 6 months" instead of lifetime totals that would leak the future.

## Tech stack

| Tool | How it is used in this project |
|---|---|
| **Python 3.11+** | Everything: simulation, features, modelling, dashboard and tests |
| **NumPy** | Vectorised monthly lifecycle simulation (usage, churn hazard, plan changes, revenue) |
| **pandas** | Table joins, point-in-time feature engineering, the 12-month target, segment and channel analytics |
| **scikit-learn** | Pipelines with imputation, scaling and one-hot encoding; Linear Regression, Random Forest, Gradient Boosting; churn classifier; permutation importance; MAE, RMSE, R² and AUC |
| **SQL and SQLite** | Local database for the dashboard; named queries for monthly revenue, MRR and churn, tested to match the pandas results |
| **joblib** | Saves the fitted preprocessing + model pipelines so the dashboard reuses them without retraining |
| **Streamlit** | Five-page interactive dashboard with sidebar filters and empty states |
| **Plotly** | Interactive charts with hover details and currency formatting |
| **pytest** | 59 tests: data integrity, leakage, business logic, SQL-vs-pandas parity and dashboard smoke tests |
| **Jupyter, Matplotlib, Seaborn** | Exploratory data analysis notebook |

## Project structure

```
saas-clv-analytics/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── generate_data.py        
│   └── run_pipeline.py         
├── src/clv/
│   ├── config.py               
│   ├── data_generation.py      
│   ├── dictionary.py           
│   ├── validation.py           
│   ├── features.py             
│   ├── modeling.py             
│   ├── segmentation.py         
│   ├── churn.py                
│   ├── marketing.py            
│   ├── metrics.py              
│   ├── charts.py               
│   ├── storage.py              
│   └── pipeline.py             
├── sql/analytics.sql           
├── app/
│   ├── streamlit_app.py        
│   ├── ui.py, data.py          
│   └── views/                  
├── notebooks/01_eda.ipynb      
├── tests/                      
├── docs/                       
├── data/raw/                   
├── data/processed/             
└── models/                     
```

Full column definitions and simulation assumptions: [`docs/data_dictionary.md`](docs/data_dictionary.md).

## Future enhancements

- **Profit-based CLV:** add cost and margin inputs so CLV-to-CAC and payback are measured in gross profit, not revenue.
- **Lifetime value:** extend beyond 12 months with survival models for time-to-churn and discounting.
- **Better churn calibration:** recalibrate the top risk tier, which is over-confident (about 55% predicted vs 45% observed).
- **Richer data and models:** more training snapshots, hyper-parameter search, gradient-boosting libraries such as XGBoost or LightGBM, and drift monitoring.
- **Channel experiments:** design holdout or budget-split tests to measure the incremental impact of each channel before moving spend.
- **Real data:** adapt the loaders to a real customer, billing and product-usage warehouse.
