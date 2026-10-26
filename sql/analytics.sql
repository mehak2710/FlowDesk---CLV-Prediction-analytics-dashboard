-- Named analytical queries used by the Streamlit app and tests.
-- Loaded by clv.storage.load_queries(); each block starts with "-- name: <key>".

-- name: monthly_revenue
SELECT month,
       SUM(recognized_revenue)                                   AS recognized_revenue,
       SUM(recurring_revenue + expansion_revenue - contraction_revenue) AS mrr,
       COUNT(DISTINCT CASE WHEN recurring_revenue > 0 THEN customer_id END) AS billed_customers
FROM revenue_monthly
GROUP BY month
ORDER BY month;

-- name: monthly_churn
-- Logo churn(m) = cancellations effective in month m / customers already subscribed at the start of m.
-- Customers subscribed at start of m = customers billed in m minus subscriptions that started in m.
WITH billed AS (
    SELECT month, COUNT(DISTINCT customer_id) AS billed_customers
    FROM revenue_monthly WHERE recurring_revenue > 0 GROUP BY month
), starts AS (
    SELECT substr(event_date, 1, 7) AS ym, COUNT(*) AS starts
    FROM billing_events WHERE event_type IN ('new_subscription', 'reactivation') GROUP BY ym
), cancels AS (
    SELECT substr(event_date, 1, 7) AS ym, COUNT(*) AS cancellations
    FROM billing_events WHERE event_type = 'cancellation' GROUP BY ym
)
SELECT b.month,
       b.billed_customers - COALESCE(s.starts, 0)            AS active_at_start,
       COALESCE(c.cancellations, 0)                          AS cancellations,
       1.0 * COALESCE(c.cancellations, 0) / NULLIF(b.billed_customers - COALESCE(s.starts, 0), 0) AS churn_rate
FROM billed b
LEFT JOIN starts  s ON s.ym = substr(b.month, 1, 7)
LEFT JOIN cancels c ON c.ym = substr(b.month, 1, 7)
ORDER BY b.month;

-- name: channel_cohort_revenue
-- Realised revenue to date per acquisition channel (includes churned customers: no survivorship bias).
SELECT c.acquisition_channel,
       COUNT(DISTINCT c.customer_id)           AS customers,
       SUM(c.acquisition_cost)                 AS total_cac_spend,
       AVG(c.acquisition_cost)                 AS avg_cac,
       SUM(r.recognized_revenue)               AS realised_revenue
FROM customers c
JOIN revenue_monthly r USING (customer_id)
GROUP BY c.acquisition_channel;