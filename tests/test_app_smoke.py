"""Smoke tests: every dashboard page renders, filters change results, empty states do not crash."""
import pytest
from streamlit.testing.v1 import AppTest

from clv import config as C

pytestmark = pytest.mark.skipif(not (C.DB_PATH.exists() and (C.MODELS_DIR / "clv_model.joblib").exists()),
                                reason="run scripts/generate_data.py and scripts/run_pipeline.py first")

PAGES = ["overview", "clv", "churn", "marketing", "explorer"]


def _page_app(page: str):
    # self-contained: AppTest.from_function only ships this function's source (sys.path comes from conftest)
    import importlib
    from app.data import load_data, load_report
    from app.ui import build_context
    ctx = build_context(load_data(), load_report())
    importlib.import_module(f"app.views.{page}").render(ctx)


def _kpis(at) -> dict:
    """label -> value for every KPI card on the page."""
    import html as _h
    import re
    out = {}
    for m in at.markdown:
        for lab, val in re.findall(r'class="kpi-label">(.*?)</div><div class="kpi-value">(.*?)</div>', m.value):
            out[_h.unescape(lab)] = _h.unescape(val)
    return out


def _run(page, **widgets):
    at = AppTest.from_function(_page_app, kwargs={"page": page}, default_timeout=120).run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_error(page):
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_backtest_snapshot_renders(page):
    at = AppTest.from_function(_page_app, kwargs={"page": page}, default_timeout=120).run()
    at.sidebar.radio[0].set_value("backtest").run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_empty_filters_show_empty_state_not_crash(page):
    at = AppTest.from_function(_page_app, kwargs={"page": page}, default_timeout=120).run()
    at.sidebar.multiselect[0].set_value([]).run()   # no plans selected
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.info) >= 1


def test_filters_change_metrics():
    at = AppTest.from_function(_page_app, kwargs={"page": "overview"}, default_timeout=120).run()
    before = _kpis(at)
    at.sidebar.multiselect[0].set_value(["Enterprise"]).run()
    after = _kpis(at)
    assert not at.exception
    assert before["Active customers"] != after["Active customers"]
    assert before["Historical revenue"] != after["Historical revenue"]
    at.sidebar.multiselect[3].set_value(["Referral"]).run()   # + channel filter
    assert not at.exception
    assert _kpis(at)["Active customers"] != after["Active customers"]


def test_headline_metrics_are_computed_not_hardcoded():
    import pandas as pd
    from clv.storage import read_table
    at = AppTest.from_function(_page_app, kwargs={"page": "overview"}, default_timeout=120).run()
    snaps = read_table("customer_snapshots")
    cur = snaps[snaps["split"] == "current"]
    metrics = _kpis(at)
    assert metrics["Active customers"] == f"{len(cur):,}"


@pytest.mark.parametrize("idx,value", [(0, ["Basic"]), (1, ["Healthcare"]), (2, ["1000+"]), (3, ["Direct Sales"])])
def test_each_multiselect_filter_changes_results(idx, value):
    at = AppTest.from_function(_page_app, kwargs={"page": "overview"}, default_timeout=120).run()
    before = _kpis(at)
    at.sidebar.multiselect[idx].set_value(value).run()
    after = _kpis(at)
    assert not at.exception
    assert before["Active customers"] != after["Active customers"]
    assert before["Historical revenue"] != after["Historical revenue"]


def test_signup_date_and_period_filters_change_results():
    import datetime as dt
    at = AppTest.from_function(_page_app, kwargs={"page": "overview"}, default_timeout=120).run()
    before = _kpis(at)
    at.sidebar.date_input[0].set_value((dt.date(2023, 1, 1), dt.date(2023, 12, 31))).run()
    mid = _kpis(at)
    assert not at.exception and mid["Active customers"] != before["Active customers"]
    opts = at.sidebar.select_slider[0].options
    at.sidebar.select_slider[0].set_range(opts[0], opts[11]).run()
    late = _kpis(at)
    assert not at.exception and late["Historical revenue"] != mid["Historical revenue"]