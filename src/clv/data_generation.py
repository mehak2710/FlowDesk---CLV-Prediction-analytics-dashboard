"""Synthetic data generator for Flowdesk, a fictional B2B workflow SaaS company.

The simulator is a *monthly customer-lifecycle engine*, not a table of independent
random columns. Each customer carries latent traits (engagement propensity, a
"frailty" term) and a time-varying usage-health state. Every month the engine
draws usage, support tickets, payment failures, plan changes and revenue from
that state, and then draws churn from a hazard that depends on the *past*
(usage decline, payment failures, unresolved tickets, recent downgrades, ...).
Churned customers stop generating revenue, except for explicitly modelled
post-cancellation billing adjustments, and a few reactivate later.

Because behaviour, revenue and churn are generated from the same state,
the tables are internally consistent and churn is predictable only in a
noisy, probabilistic way.

All data is SYNTHETIC. Nothing here reflects a real company.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

# ----------------------------------------------------------------------------
# Tunable behavioural parameters (documented in docs/data_dictionary.md)
# ----------------------------------------------------------------------------
INDUSTRY_LOG_EMP_MEAN = {
    "Technology": 4.2, "Financial Services": 4.8, "Healthcare": 5.0,
    "Retail & E-commerce": 4.3, "Manufacturing": 5.0, "Professional Services": 3.6,
    "Education": 4.7, "Media & Marketing": 3.8,
}
INDUSTRY_CHURN_EFFECT = {
    "Technology": 0.0, "Financial Services": -0.25, "Healthcare": -0.15,
    "Retail & E-commerce": 0.25, "Manufacturing": -0.05, "Professional Services": 0.0,
    "Education": 0.35, "Media & Marketing": 0.20,
}
COUNTRIES = {  # country -> (probability, sales_region)
    "United States": (0.34, "North America"), "Canada": (0.06, "North America"),
    "United Kingdom": (0.10, "EMEA"), "Germany": (0.08, "EMEA"),
    "France": (0.05, "EMEA"), "Netherlands": (0.04, "EMEA"),
    "India": (0.12, "APAC"), "Singapore": (0.05, "APAC"), "Australia": (0.07, "APAC"),
    "Brazil": (0.06, "LATAM"), "Mexico": (0.03, "LATAM"),
}
# P(plan | size band) rows ordered like C.PLANS
PLAN_PROBS = np.array([
    [0.65, 0.32, 0.03],
    [0.30, 0.55, 0.15],
    [0.08, 0.50, 0.42],
    [0.02, 0.28, 0.70],
])
# P(channel | size band) rows ordered like C.CHANNELS
CHANNEL_PROBS = np.array([
    [0.30, 0.28, 0.14, 0.24, 0.04],
    [0.26, 0.26, 0.17, 0.13, 0.18],
    [0.16, 0.18, 0.17, 0.04, 0.45],
    [0.10, 0.12, 0.14, 0.02, 0.62],
])
CHANNEL_ENGAGEMENT = np.array([0.20, -0.20, 0.50, -0.35, 0.10])  # latent engagement shift
CHANNEL_CHURN = np.array([0.0, 0.25, -0.35, 0.30, -0.15])
CHANNEL_CAC_BASE = np.array([180.0, 780.0, 320.0, 520.0, 2600.0])
SIZE_CAC_MULT = np.array([0.7, 1.0, 1.8, 3.2])
SIZE_CHURN = np.array([0.35, 0.10, -0.15, -0.25])
PLAN_CHURN = np.array([0.55, 0.0, -0.20])
PLAN_FRAILTY_SD = np.array([0.45, 0.45, 0.90])  # Enterprise: more *varied* retention
USAGE_SEASON = np.array([1.00, 1.02, 1.03, 1.02, 1.00, 0.97, 0.90, 0.88, 1.00, 1.04, 1.03, 0.92])
CHURN_SEASON = np.array([0.30, 0.10, 0.0, 0.0, 0.0, 0.0, -0.10, 0.0, 0.0, 0.0, 0.05, 0.15])
SIGNUP_SEASON = np.array([1.15, 1.0, 1.05, 1.0, 0.95, 0.90, 0.85, 0.85, 1.05, 1.10, 1.10, 0.90])
HAZARD_INTERCEPT = -4.55  # calibrated so portfolio monthly logo churn is ~2%

TICKET_CATEGORIES = ["Billing", "Bug Report", "How-To / Training", "Integration", "Performance", "Feature Request", "Account Access"]
TICKET_CAT_P = [0.14, 0.24, 0.22, 0.16, 0.08, 0.08, 0.08]
PRIORITIES = ["Low", "Medium", "High", "Critical"]
PRIORITY_P = [0.38, 0.40, 0.17, 0.05]
PRIORITY_RES_HOURS = np.array([48.0, 26.0, 10.0, 4.5])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _rand_day_in_month(rng, month_idx: np.ndarray) -> pd.DatetimeIndex:
    """Random calendar day inside month `month_idx` (0 = first simulated month)."""
    periods = C.FIRST_MONTH + month_idx
    starts = pd.PeriodIndex(periods, freq="M").to_timestamp()
    dim = pd.PeriodIndex(periods, freq="M").days_in_month
    return starts + pd.to_timedelta(rng.integers(0, dim), unit="D")


# ----------------------------------------------------------------------------
# Customers
# ----------------------------------------------------------------------------
def _generate_customers(rng: np.random.Generator, n: int) -> dict:
    n_signup_months = (C.LAST_SIGNUP_MONTH - C.FIRST_MONTH).n + 1
    m = np.arange(n_signup_months)
    cal = (m % 12)
    w = (1.0 + 1.8 * m / (n_signup_months - 1)) * SIGNUP_SEASON[cal]  # growing company
    signup_idx = rng.choice(m, size=n, p=w / w.sum())
    signup_idx.sort()

    industry = rng.choice(C.INDUSTRIES, size=n)
    log_mean = np.array([INDUSTRY_LOG_EMP_MEAN[i] for i in industry])
    emp = np.clip(np.round(np.exp(rng.normal(log_mean, 1.25))), 5, 25_000).astype(int)
    size_idx = np.digitize(emp, [50, 250, 1000])

    plan_idx = np.array([rng.choice(3, p=PLAN_PROBS[s]) for s in size_idx])
    chan_idx = np.array([rng.choice(5, p=CHANNEL_PROBS[s]) for s in size_idx])

    countries = list(COUNTRIES)
    cp = np.array([COUNTRIES[c][0] for c in countries])
    country = rng.choice(countries, size=n, p=cp / cp.sum())
    region = np.array([COUNTRIES[c][1] for c in country])

    annual_p = np.array([0.18, 0.30, 0.55])[plan_idx] + np.where(chan_idx == 2, 0.08, 0.0)
    annual = rng.random(n) < annual_p
    disc = np.where(annual, 10.0, 0.0)
    promo_p = np.where(chan_idx == 4, 0.55, np.where(np.isin(chan_idx, [1, 3]), 0.32, 0.15))
    promo = rng.random(n) < promo_p
    disc = disc + np.where(promo, rng.choice([5.0, 10.0, 15.0], size=n, p=[0.5, 0.35, 0.15]), 0.0)
    disc = disc + np.where((plan_idx == 2) & (rng.random(n) < 0.35), 5.0, 0.0)
    disc = np.minimum(disc, 30.0)

    # list price: plan base x seat factor x noise
    seat_factor = np.clip((emp / 40.0) ** 0.45, 0.8, 6.0)
    base = np.array([C.PLAN_BASE_PRICE[C.PLANS[p]] for p in plan_idx])
    mrr0 = np.round(base * seat_factor * np.exp(rng.normal(0, 0.12, n)), 0)

    # latent engagement propensity (standardised); size/plan/channel shift it, noise dominates
    e_raw = 0.35 * plan_idx + 0.15 * size_idx + CHANNEL_ENGAGEMENT[chan_idx] + rng.normal(0, 1, n)
    e = (e_raw - e_raw.mean()) / e_raw.std()
    frailty = rng.normal(0, PLAN_FRAILTY_SD[plan_idx])
    ind_effect = np.array([INDUSTRY_CHURN_EFFECT[i] for i in industry])

    cac = CHANNEL_CAC_BASE[chan_idx] * SIZE_CAC_MULT[size_idx] * np.exp(rng.normal(0, 0.35, n))
    cac = np.where(rng.random(n) < 0.005, cac * 3.0, cac)  # outliers
    cac = np.round(cac, 2)

    return dict(
        signup_idx=signup_idx, industry=industry, emp=emp, size_idx=size_idx,
        plan_idx=plan_idx, chan_idx=chan_idx, country=country, region=region,
        annual=annual, disc=disc, mrr0=mrr0, e=e, frailty=frailty, ind_effect=ind_effect,
        cac=cac, usage_scale=7.0 * np.sqrt(np.minimum(emp, 4000)),
    )


# ----------------------------------------------------------------------------
# Monthly lifecycle simulation
# ----------------------------------------------------------------------------
def generate_tables(seed: int = C.SEED, n_customers: int = C.N_CUSTOMERS) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cu = _generate_customers(rng, n_customers)
    n = n_customers
    M = (C.LAST_MONTH - C.FIRST_MONTH).n + 1
    avail = np.array([C.PLAN_FEATURES_AVAILABLE[p] for p in C.PLANS], dtype=float)
    base_price = np.array([C.PLAN_BASE_PRICE[p] for p in C.PLANS])

    # state
    active = np.zeros(n, bool)
    sub_tenure = np.zeros(n, int)
    plan = cu["plan_idx"].copy()
    mrr = np.zeros(n)
    s = np.zeros(n)  # log usage-health state
    ep_rem = np.zeros(n, int)
    ep_rate = np.zeros(n)
    prev_logins = np.full(n, np.nan)
    react_idx = np.full(n, -1)
    adj_month = np.full(n, -1)
    adj_amount = np.zeros(n)
    last_mrr = np.zeros(n)
    sub_counter = 0
    cur_sub = np.full(n, -1)
    sub_start_m = np.zeros(n, int)
    sub_up = np.zeros(n, int)
    sub_dn = np.zeros(n, int)
    sub_pf = np.zeros(n, int)
    # per-month history matrices for hazard
    pf_hist = np.zeros((n, M))
    unres_hist = np.zeros((n, M))
    lowcsat_hist = np.zeros((n, M))
    dn_hist = np.zeros((n, M))

    usage_rows, rev_rows, ticket_rows, event_rows, sub_rows = [], [], [], [], []

    def close_sub(idx, end_m, status):
        nonlocal sub_rows
        for i in idx:
            end_date = (C.FIRST_MONTH + int(end_m)).end_time.normalize() if status == "Cancelled" else pd.NaT
            sub_rows.append(dict(
                _sid=cur_sub[i], customer_idx=i, subscription_plan=C.PLANS[plan[i]],
                billing_cycle="Annual" if cu["annual"][i] else "Monthly",
                monthly_recurring_revenue=round(float(mrr[i]), 2),
                _start_m=sub_start_m[i], subscription_end_date=end_date,
                subscription_status=status, upgrade_count=int(sub_up[i]),
                downgrade_count=int(sub_dn[i]), discount_percentage=float(cu["disc"][i]),
                payment_failure_count=int(sub_pf[i]),
            ))

    for m in range(M):
        cal = m % 12
        # ---- subscription starts (new signups + reactivations) --------------
        new = cu["signup_idx"] == m
        react = (react_idx == m) & (~active)
        starts = new | react
        for i in np.where(starts)[0]:
            sub_counter += 1
            cur_sub[i] = sub_counter
            sub_start_m[i] = m
            sub_up[i] = sub_dn[i] = sub_pf[i] = 0
        if new.any():
            mrr[new] = cu["mrr0"][new]
        if react.any():
            mrr[react] = np.round(last_mrr[react] * 0.95, 0)
            s[react] = -0.25
            ep_rem[react] = 0
        sub_tenure[starts] = 0
        prev_logins[starts] = np.nan
        active[starts] = True
        # event rows for starts
        for kind, mask in (("new_subscription", new), ("reactivation", react)):
            idx = np.where(mask)[0]
            if len(idx):
                d = _rand_day_in_month(rng, np.full(len(idx), m))
                for j, i in enumerate(idx):
                    event_rows.append((i, cur_sub[i], d[j], kind, None, C.PLANS[plan[i]], mrr[i]))

        # ---- post-cancellation billing adjustments (explicit, modelled) -----
        adj = np.where(adj_month == m)[0]
        if len(adj):
            rev_rows.append(pd.DataFrame(dict(
                customer_idx=adj, month_idx=m, recurring_revenue=0.0, expansion_revenue=0.0,
                contraction_revenue=0.0, refunds=np.round(adj_amount[adj], 2), discounts=0.0,
                recognized_revenue=-np.round(adj_amount[adj], 2),
            )))

        act = np.where(active)[0]
        if len(act) == 0:
            continue
        na = len(act)
        e = cu["e"][act]
        tenure_c = (m - cu["signup_idx"][act])  # customer tenure in months
        st = sub_tenure[act]

        # ---- 1. usage-health state ------------------------------------------
        in_ep = ep_rem[act] > 0
        s_new = s[act].copy()
        s_new = np.where(
            in_ep, s_new - ep_rate[act] + rng.normal(0, 0.04, na),
            0.88 * s_new + rng.normal(0, 0.07, na),
        )
        ep_rem[act] = np.where(in_ep, ep_rem[act] - 1, ep_rem[act])
        p_ep = 0.022 * np.exp(-0.55 * e) * np.where(plan[act] == 0, 1.3, 1.0)
        start_ep = (~in_ep) & (st > 0) & (rng.random(na) < p_ep)
        ep_rem[act[start_ep]] = rng.integers(2, 6, start_ep.sum())
        ep_rate[act[start_ep]] = rng.uniform(0.10, 0.28, start_ep.sum())
        s_new = np.where(st > 0, s_new, s[act])
        s[act] = s_new
        sc = s_new

        # ---- 2. usage metrics -------------------------------------------------
        ramp = np.where(st == 0, 0.65, np.where(st == 1, 0.85, 1.0))
        logins_mean = cu["usage_scale"][act] * np.exp(0.40 * e + sc) * USAGE_SEASON[cal] * ramp
        logins = np.maximum(np.round(logins_mean * np.exp(rng.normal(0, 0.18, na))), 0)
        logins = np.where(rng.random(na) < 0.004, logins * 4.0, logins)  # outlier spikes (bots/integrations)
        active_days = np.clip(np.round(3 + 21 * _sigmoid(0.9 * e + 1.2 * sc - 0.2 + rng.normal(0, 0.35, na))), 1, 28)
        adoption = np.clip(_sigmoid(-0.4 + 0.75 * e + 0.8 * sc + 0.25 * np.minimum(tenure_c, 8) / 8
                                    + rng.normal(0, 0.15, na)), 0.03, 0.98)
        av = avail[plan[act]]
        feats = np.maximum(np.round(adoption * av), 1)
        session = 18 * np.exp(0.2 * e + 0.3 * sc) * np.exp(rng.normal(0, 0.25, na))
        usage_change = (logins / prev_logins[act] - 1.0) * 100.0
        prev_logins[act] = logins
        miss_l = rng.random(na) < 0.015
        miss_s = rng.random(na) < 0.03
        miss_f = rng.random(na) < 0.025
        usage_rows.append(pd.DataFrame(dict(
            customer_idx=act, month_idx=m,
            monthly_logins=np.where(miss_l, np.nan, logins),
            active_days=active_days,
            features_used=np.where(miss_f, np.nan, feats),
            feature_adoption_rate=np.where(miss_f, np.nan, np.round(feats / av, 4)),
            session_duration_minutes=np.where(miss_s, np.nan, np.round(session, 1)),
            usage_change_percentage=np.where(miss_l | np.isnan(usage_change), np.nan, np.round(usage_change, 1)),
        )))

        # ---- 3. support tickets ---------------------------------------------
        lam = 0.17 * (1 + 0.25 * plan[act]) * (1 + 0.7 * np.maximum(-sc, 0)) * np.exp(-0.15 * e) \
            * np.where(st < 3, 1.3, 1.0) * np.where(cu["size_idx"][act] >= 2, 1.2, 1.0)
        nt = rng.poisson(lam)
        if nt.sum():
            ci = np.repeat(act, nt)
            pri = rng.choice(4, size=len(ci), p=PRIORITY_P)
            sla = np.where(plan[ci] == 2, 0.6, np.where(plan[ci] == 1, 0.85, 1.0))
            res = PRIORITY_RES_HOURS[pri] * sla * np.exp(rng.normal(0, 0.7, len(ci)))
            unresolved = rng.random(len(ci)) < (0.035 + 0.03 * (pri >= 2))
            csat = 4.6 - 0.75 * np.log1p(res / 12.0) + rng.normal(0, 0.8, len(ci))
            csat = np.where(unresolved, rng.integers(1, 4, len(ci)), csat)
            csat = np.clip(np.round(csat), 1, 5)
            csat = np.where(rng.random(len(ci)) < 0.38, np.nan, csat)
            ticket_rows.append(pd.DataFrame(dict(
                customer_idx=ci,
                created_date=_rand_day_in_month(rng, np.full(len(ci), m)),
                issue_category=rng.choice(TICKET_CATEGORIES, size=len(ci), p=TICKET_CAT_P),
                priority=np.array(PRIORITIES)[pri],
                resolution_time_hours=np.where(unresolved, np.nan, np.round(res, 1)),
                customer_satisfaction_score=csat,
            )))
            np.add.at(unres_hist[:, m], ci, unresolved.astype(float))
            np.add.at(lowcsat_hist[:, m], ci, (csat <= 2).astype(float))

        # ---- 4. payment failures -----------------------------------------------
        lam_pf = 0.016 * np.where(cu["annual"][act], 0.35, 2.0) \
            * np.array([1.5, 1.2, 0.8, 0.6])[cu["size_idx"][act]] * np.exp(-0.2 * e) * (1 + 0.6 * np.maximum(-sc, 0))
        npf = rng.poisson(lam_pf)
        for k in np.where(npf > 0)[0]:
            i = act[k]
            for _ in range(npf[k]):
                event_rows.append((i, cur_sub[i], _rand_day_in_month(rng, np.array([m]))[0],
                                   "payment_failure", C.PLANS[plan[i]], C.PLANS[plan[i]], mrr[i]))
        pf_hist[act, m] = npf
        sub_pf[act] += npf

        # ---- 5. plan & seat changes ---------------------------------------------
        mrr_before = mrr[act].copy()
        pl = plan[act].copy()
        can_change = st > 0
        p_up = np.clip(0.006 * np.exp(0.55 * e), 0, 0.05) * (sc > -0.3)
        up = can_change & (pl < 2) & (rng.random(na) < p_up)
        p_dn = np.clip(0.005 * np.exp(-0.55 * e) * (1 + 1.5 * np.maximum(-sc, 0)), 0, 0.08)
        dn = can_change & (~up) & (pl > 0) & (rng.random(na) < p_dn)
        mrr_new = mrr_before.copy()
        ratio_up = base_price[np.minimum(pl + 1, 2)] / base_price[pl] * rng.uniform(0.85, 1.05, na)
        ratio_dn = base_price[np.maximum(pl - 1, 0)] / base_price[pl] * rng.uniform(0.9, 1.15, na)
        mrr_new = np.where(up, mrr_before * ratio_up, mrr_new)
        mrr_new = np.where(dn, mrr_before * ratio_dn, mrr_new)
        other = can_change & ~up & ~dn
        exp_seat = other & (rng.random(na) < 0.022 * np.exp(0.3 * e))
        jump = np.where(rng.random(na) < 0.01, rng.uniform(1.5, 2.2, na), 1 + rng.uniform(0.03, 0.12, na))
        mrr_new = np.where(exp_seat, mrr_before * jump, mrr_new)
        con_seat = other & ~exp_seat & (rng.random(na) < 0.007 * np.exp(-0.4 * e))
        mrr_new = np.where(con_seat, mrr_before * (1 - rng.uniform(0.03, 0.10, na)), mrr_new)
        mrr_new = np.round(mrr_new, 2)
        pl_new = pl + up.astype(int) - dn.astype(int)
        for k in np.where(up | dn)[0]:
            i = act[k]
            event_rows.append((i, cur_sub[i], _rand_day_in_month(rng, np.array([m]))[0],
                               "upgrade" if up[k] else "downgrade",
                               C.PLANS[pl[k]], C.PLANS[pl_new[k]], mrr_new[k]))
        sub_up[act] += up
        sub_dn[act] += dn
        dn_hist[act, m] = dn
        plan[act] = pl_new
        mrr[act] = mrr_new

        # ---- 6. revenue -------------------------------------------------------------
        delta = mrr_new - mrr_before
        recurring = mrr_before
        expansion = np.maximum(delta, 0.0)
        contraction = np.maximum(-delta, 0.0)
        gross = recurring + expansion - contraction
        discounts = gross * cu["disc"][act] / 100.0
        refund = np.where(rng.random(na) < 0.0035, rng.uniform(0.2, 0.8, na) * (gross - discounts), 0.0)
        recognized = gross - discounts - refund
        rev_rows.append(pd.DataFrame(dict(
            customer_idx=act, month_idx=m,
            recurring_revenue=np.round(recurring, 2), expansion_revenue=np.round(expansion, 2),
            contraction_revenue=np.round(contraction, 2), refunds=np.round(refund, 2),
            discounts=np.round(discounts, 2), recognized_revenue=np.round(recognized, 2),
        )))

        # ---- 7. churn decision (end of month) --------------------------------------
        lo = max(m - 2, 0)
        pf3 = pf_hist[act, lo:m + 1].sum(axis=1)
        un3 = unres_hist[act, lo:m + 1].sum(axis=1)
        cs3 = lowcsat_hist[act, lo:m + 1].sum(axis=1) > 0
        lo6 = max(m - 5, 0)
        dn6 = dn_hist[act, lo6:m + 1].sum(axis=1)
        tenure_effect = np.where(st < 3, 0.40, np.where(st < 6, 0.40, np.where(st < 12, 0.15, np.where(st >= 18, -0.25, 0.0))))
        logit = (
            HAZARD_INTERCEPT + PLAN_CHURN[plan[act]] + SIZE_CHURN[cu["size_idx"][act]]
            + cu["ind_effect"][act] + CHANNEL_CHURN[cu["chan_idx"][act]]
            - 0.55 * e + 1.1 * np.maximum(-sc, 0) + 0.55 * np.minimum(pf3, 3)
            + 0.45 * np.minimum(un3, 3) + 0.40 * cs3 + 0.60 * np.minimum(dn6, 2)
            + tenure_effect + CHURN_SEASON[cal] + 0.20 * (cu["disc"][act] >= 20)
            + cu["frailty"][act]
        )
        p = _sigmoid(logit)
        annual = cu["annual"][act]
        renewal = ((st + 1) % 12 == 0)
        eligible = (st >= 2) & ((~annual) | renewal)
        p = np.where(annual & renewal, 1 - (1 - p) ** 6, p)
        churned = eligible & (rng.random(na) < p)
        ch_idx = act[churned]
        if len(ch_idx):
            close_sub(ch_idx, m, "Cancelled")
            for i in ch_idx:
                event_rows.append((i, cur_sub[i], (C.FIRST_MONTH + m).end_time.normalize(),
                                   "cancellation", C.PLANS[plan[i]], None, 0.0))
            last_mrr[ch_idx] = mrr[ch_idx]
            active[ch_idx] = False
            # explicit post-cancellation adjustment (next month)
            has_adj = (rng.random(len(ch_idx)) < 0.06) & (m + 1 < M)
            adj_month[ch_idx[has_adj]] = m + 1
            adj_amount[ch_idx[has_adj]] = rng.uniform(0.3, 1.0, has_adj.sum()) * mrr[ch_idx[has_adj]] * (1 - cu["disc"][ch_idx[has_adj]] / 100)
            # win-back
            wb = rng.random(len(ch_idx)) < 0.07
            gap = rng.integers(2, 9, wb.sum())
            tgt = m + gap
            tgt = np.where(tgt < M, tgt, -1)
            react_idx[ch_idx[wb]] = tgt
        sub_tenure[act[~churned]] += 1

    # close still-active subscriptions
    close_sub(np.where(active)[0], M - 1, "Active")

    # ------------------------------------------------------------------------
    # Assemble tables
    # ------------------------------------------------------------------------
    cust_ids = np.array([f"C{i + 1:05d}" for i in range(n)])
    signup_date = _rand_day_in_month(rng, cu["signup_idx"])

    emp_out = cu["emp"].astype(float)
    emp_out[rng.random(n) < 0.012] = np.nan  # missing employee counts
    cac_out = cu["cac"].copy()
    cac_out[rng.random(n) < 0.015] = np.nan  # unattributed acquisition cost
    customers = pd.DataFrame(dict(
        customer_id=cust_ids, signup_date=signup_date, industry=cu["industry"],
        company_size=np.array(C.SIZE_BANDS)[cu["size_idx"]], employee_count=emp_out,
        acquisition_channel=np.array(C.CHANNELS)[cu["chan_idx"]], acquisition_cost=cac_out,
        country=cu["country"], sales_region=cu["region"],
    ))

    subs = pd.DataFrame(sub_rows).sort_values("_sid").reset_index(drop=True)
    subs["subscription_id"] = subs["_sid"].map(lambda x: f"S{x:06d}")
    subs["customer_id"] = cust_ids[subs["customer_idx"].values]
    start_m = subs["_start_m"].values
    starts_ts = _rand_day_in_month(rng, start_m)
    # first subscription starts on the customer's signup date
    first = ~subs.duplicated("customer_idx", keep="first")
    starts_ts = pd.Series(starts_ts, index=subs.index)
    starts_ts[first] = signup_date[subs.loc[first, "customer_idx"].values]
    subs["subscription_start_date"] = starts_ts
    subs = subs[["subscription_id", "customer_id", "subscription_plan", "billing_cycle",
                 "monthly_recurring_revenue", "subscription_start_date", "subscription_end_date",
                 "subscription_status", "upgrade_count", "downgrade_count",
                 "discount_percentage", "payment_failure_count"]]
    def month_end(idx):
        return pd.PeriodIndex(C.FIRST_MONTH + np.asarray(idx), freq="M").to_timestamp(how="end").normalize()

    usage = pd.concat(usage_rows, ignore_index=True)
    usage["customer_id"] = cust_ids[usage["customer_idx"].values]
    usage["observation_month"] = month_end(usage["month_idx"].values)
    usage = usage[["customer_id", "observation_month", "monthly_logins", "active_days", "features_used",
                   "feature_adoption_rate", "session_duration_minutes", "usage_change_percentage"]]

    rev = pd.concat(rev_rows, ignore_index=True)
    rev["customer_id"] = cust_ids[rev["customer_idx"].values]
    rev["month"] = month_end(rev["month_idx"].values)
    rev = rev[["customer_id", "month", "recurring_revenue", "expansion_revenue", "contraction_revenue",
               "refunds", "discounts", "recognized_revenue"]].sort_values(["customer_id", "month"]).reset_index(drop=True)
    usage = usage.sort_values(["customer_id", "observation_month"]).reset_index(drop=True)

    tick = pd.concat(ticket_rows, ignore_index=True) if ticket_rows else pd.DataFrame()
    tick["customer_id"] = cust_ids[tick["customer_idx"].values]
    tick = _clamp_to_signup(rng, tick, "created_date", signup_date)
    tick = tick.sort_values("created_date").reset_index(drop=True)
    tick["ticket_id"] = [f"T{i + 1:06d}" for i in range(len(tick))]
    tick = tick[["ticket_id", "customer_id", "created_date", "issue_category", "priority",
                 "resolution_time_hours", "customer_satisfaction_score"]]

    ev = pd.DataFrame(event_rows, columns=["customer_idx", "_sid", "event_date", "event_type",
                                           "plan_before", "plan_after", "mrr_after"])
    ev["customer_id"] = cust_ids[ev["customer_idx"].values]
    ev = _clamp_to_signup(rng, ev, "event_date", signup_date)
    first_new = (ev["event_type"] == "new_subscription")
    ev.loc[first_new, "event_date"] = signup_date[ev.loc[first_new, "customer_idx"].values]
    ev["subscription_id"] = ev["_sid"].map(lambda x: f"S{int(x):06d}")
    ev = ev.sort_values(["event_date", "customer_id"]).reset_index(drop=True)
    ev["event_id"] = [f"E{i + 1:07d}" for i in range(len(ev))]
    # payment failures keep the plan unchanged: plan_after is informational for upgrade/downgrade/new/react
    ev = ev[["event_id", "customer_id", "subscription_id", "event_date", "event_type",
             "plan_before", "plan_after", "mrr_after"]]

    marketing = _marketing_table(rng, customers, rev)
    return dict(customers=customers, subscriptions=subs, usage_monthly=usage, support_tickets=tick,
                revenue_monthly=rev, marketing_channels=marketing, billing_events=ev)


def _clamp_to_signup(rng, df: pd.DataFrame, col: str, signup_date: pd.DatetimeIndex) -> pd.DataFrame:
    """Events drawn on a random day of the signup month may precede the signup day;
    move those to a random day between signup and month end."""
    sd = pd.Series(signup_date)[df["customer_idx"].values].values
    early = df[col].values < sd
    if early.any():
        month_end = (pd.DatetimeIndex(sd[early]).to_period("M").to_timestamp(how="end").normalize()).values
        span = (month_end - sd[early]).astype("timedelta64[D]").astype(int)
        shift = (rng.random(early.sum()) * (span + 1)).astype(int)
        df.loc[early, col] = pd.DatetimeIndex(sd[early]) + pd.to_timedelta(shift, unit="D")
    return df


def _marketing_table(rng, customers: pd.DataFrame, revenue: pd.DataFrame) -> pd.DataFrame:
    """Campaign-level spend and *platform-attributed* results per channel-month.

    campaign_cost  = sum of customer acquisition costs for that cohort x overhead factor
                     (non-attributable program spend such as agency fees / content).
    attributed_new_customers = cohort size with +/- attribution slippage.
    attributed_revenue = recognised revenue in the cohort's first 3 months (incl. signup month).
    """
    c = customers.copy()
    c["campaign_month"] = c["signup_date"].dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()
    cac_fill = c.groupby("acquisition_channel")["acquisition_cost"].transform("median")
    c["cac_filled"] = c["acquisition_cost"].fillna(cac_fill)
    r = revenue.merge(c[["customer_id", "signup_date", "acquisition_channel", "campaign_month"]], on="customer_id")
    r["m_since"] = (r["month"].dt.to_period("M") - r["signup_date"].dt.to_period("M")).apply(lambda x: x.n)
    first3 = r[(r["m_since"] >= 0) & (r["m_since"] <= 2)]
    rev3 = first3.groupby(["acquisition_channel", "campaign_month"])["recognized_revenue"].sum().rename("attributed_revenue")
    g = c.groupby(["acquisition_channel", "campaign_month"]).agg(
        n=("customer_id", "count"), cost=("cac_filled", "sum")).reset_index()
    g = g.merge(rev3.reset_index(), on=["acquisition_channel", "campaign_month"], how="left")
    g["campaign_cost"] = np.round(g["cost"] * rng.uniform(1.0, 1.2, len(g)), 2)
    g["attributed_new_customers"] = np.maximum(
        0, g["n"] + np.where(g["acquisition_channel"].isin(["Paid Advertising", "Social Media"]),
                             rng.integers(-1, 3, len(g)), rng.integers(-1, 1, len(g))))
    g["attributed_revenue"] = np.round(g["attributed_revenue"].fillna(0.0), 2)
    return g[["acquisition_channel", "campaign_cost", "attributed_new_customers",
              "attributed_revenue", "campaign_month"]].sort_values(["campaign_month", "acquisition_channel"]).reset_index(drop=True)


def write_tables(tables: dict[str, pd.DataFrame], out_dir=C.DATA_RAW) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d")