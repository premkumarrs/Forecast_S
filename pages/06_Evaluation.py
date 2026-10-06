"""
Model Evaluation page: stored out-of-sample results on the U.S. Census e-commerce series.

Loads saved experiment artifacts only (no database, GDELT, LLM or network access,
and no experiment is rerun).
"""

import streamlit as st

from pages.components.evaluation import (
    load_artifacts,
    render_baseline_section,
    render_bias_section,
    render_evaluation_downloads,
    render_indicator_section,
    render_methodology,
    render_missing,
    render_news_section,
    render_regime_section,
    render_summary,
)

st.set_page_config(page_title="Model Evaluation", page_icon="📐", layout="wide")

st.markdown(
    """
    <style>
      section.main h1 { font-size: 2rem; line-height: 1.2; margin-bottom: 0.25rem; }
      section.main h2 { font-size: 1.35rem; line-height: 1.25; margin-top: 1rem; margin-bottom: 0.35rem; }
      section.main h3 { font-size: 1.05rem; line-height: 1.3; margin-top: 0.8rem; margin-bottom: 0.25rem; }
      .block-container { padding-top: 1rem; padding-bottom: 2rem; }
      div[data-testid=\"stMetricValue\"] { font-size: 1.35rem; }
      div[data-testid=\"stMetricLabel\"] { font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📐 Model Evaluation")
st.caption("Out-of-sample walk-forward evaluation of the baseline models and the past-only indicator "
           "adjustment on real data. Results are loaded from stored experiment artifacts.")

artifacts = load_artifacts()
if not artifacts.baseline_available:
    render_missing(artifacts)
    st.stop()

render_summary(artifacts)
st.divider()
render_baseline_section(artifacts)
st.divider()
render_indicator_section(artifacts)
render_bias_section(artifacts)
st.divider()
render_regime_section(artifacts)
st.divider()
render_news_section()
render_methodology()

st.subheader("Download stored results")
render_evaluation_downloads(artifacts, key_prefix="evaluation_page")
