"""Flowdesk CLV analytics - Streamlit entry point.

Run from the repository root:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Flowdesk CLV Analytics", page_icon="📈", layout="wide")

from app.data import artifacts_ready, load_data, load_report  # noqa: E402
from app.ui import CSS, build_context  # noqa: E402
from app.views import churn, clv, explorer, marketing, overview  # noqa: E402

st.markdown(CSS, unsafe_allow_html=True)

if not artifacts_ready():
    st.error("Model artifacts or the database are missing. From the repository root run:\n\n"
             "`python scripts/generate_data.py && python scripts/run_pipeline.py`")
    st.stop()

data, report = load_data(), load_report()
st.sidebar.title("Flowdesk CLV")
st.sidebar.caption("Synthetic data - fictional company")
ctx = build_context(data, report)
if ctx.filters_active:
    st.sidebar.info("Filters are active.")

pages = [
    st.Page(lambda: overview.render(ctx), title="Executive Overview", icon="📊", url_path="overview", default=True),
    st.Page(lambda: clv.render(ctx), title="CLV & Segmentation", icon="💎", url_path="clv"),
    st.Page(lambda: churn.render(ctx), title="Churn & Retention", icon="🛟", url_path="churn"),
    st.Page(lambda: marketing.render(ctx), title="Acquisition & Marketing ROI", icon="📣", url_path="marketing"),
    st.Page(lambda: explorer.render(ctx), title="Customer Explorer", icon="🔎", url_path="explorer"),
]
st.navigation(pages).run()