"""
Export page (minimal): exactly two outputs — Market (wide, single sheet) and News analysis.
"""

import streamlit as st
import pandas as pd
import io
from pages.helpers import build_market_wide_table, build_news_export
from pages.components.evaluation import load_artifacts, render_evaluation_downloads


st.set_page_config(page_title="Export — Minimal", page_icon="📤", layout="wide")

st.title("📤 Export Forecast Results — Minimal")

with st.expander("📐 Model evaluation results (stored, independent of the current forecast run)"):
    render_evaluation_downloads(load_artifacts(), key_prefix="export_page")
    st.caption("Saved outputs of the Census walk-forward evaluation. See the Model Evaluation page.")

# Preconditions
if 'forecast_result' not in st.session_state or not st.session_state.get('forecast_result'):
    st.warning("No forecast results available. Please complete forecasting first.")
    if st.button("Go to Forecasting"):
        st.switch_page("pages/03_Forecasting.py")
    st.stop()

forecast_result = st.session_state['forecast_result']
unified_data = st.session_state.get('unified_data', {})
meta = (unified_data or {}).get('market_value', {}).get('metadata', {}) if isinstance(unified_data, dict) else {}
kpi_key = meta.get('kpi_key', '')
market_name = meta.get('kpi_name', '')

st.write(f"Market: {market_name} | KPI: {kpi_key}")

fmt = st.selectbox("File format", ["Excel (.xlsx)", "CSV (.csv)"])
use_excel = fmt.startswith("Excel")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Market (wide)")
    if st.button("Download Market File", type="primary", use_container_width=True):
        df = build_market_wide_table(forecast_result, unified_data)
        if df is None or df.empty:
            st.error("Nothing to export.")
        else:
            if use_excel:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Market')
                st.download_button(
                    label="Save Market.xlsx",
                    data=buffer.getvalue(),
                    file_name="market_wide.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Save Market.csv",
                    data=csv,
                    file_name="market_wide.csv",
                    mime="text/csv",
                )

with col2:
    st.subheader("News Analysis")
    if st.button("Download News File", type="secondary", use_container_width=True):
        news_df = build_news_export(forecast_result, kpi_key)
        if news_df is None or news_df.empty:
            st.error("No news analysis available.")
        else:
            if use_excel:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    news_df.to_excel(writer, index=False, sheet_name='News')
                st.download_button(
                    label="Save News.xlsx",
                    data=buffer.getvalue(),
                    file_name="news_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                csv = news_df.to_csv(index=False)
                st.download_button(
                    label="Save News.csv",
                    data=csv,
                    file_name="news_analysis.csv",
                    mime="text/csv",
                )

st.caption("Exports include historical as-is and forecasts in future years only. Historical data is preserved unchanged.")
