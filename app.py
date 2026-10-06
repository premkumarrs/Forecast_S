import streamlit as st
from src.session import init_session_state
import logging

# ------------------------------------------------------------
# Basic setup
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

st.set_page_config(
    page_title="Forecasting App",
    page_icon="📊",
    layout="wide"
)

# Local style tweaks for cleaner layout on this page
st.markdown(
    """
    <style>
      section.main h1 { font-size: 2rem; line-height: 1.2; margin-bottom: 0.25rem; }
      section.main h2 { font-size: 1.35rem; line-height: 1.25; margin-top: 1rem; margin-bottom: 0.35rem; }
      section.main h3 { font-size: 1.05rem; line-height: 1.3; margin-top: 0.8rem; margin-bottom: 0.25rem; }
      .block-container { padding-top: 1rem; padding-bottom: 2rem; }
      div[data-testid=\"stMetricValue\"] { font-size: 1.6rem; }
      div[data-testid=\"stMetricLabel\"] { font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
init_session_state()

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("📊 Forecasting App")
st.markdown("#### A News-Driven market forecasting framework — data first, models second, optional signals last.")

st.markdown(
    """
**What this app does**  
- **Configure the run**: Pick approach (top-down / bottom-up / country-specific / global), baseline model (CAGR / ETS / Logistic), and weights.  
- **Run forecasts**: Generate baselines and structured country/regional outputs.  
- **Optionally adjust with signals**: Apply indicators and news-derived signals (GDELT). LLMs can be used *optionally* to suggest topics or help classify news — not required.  
- **Inspect & export**: Review charts, breakdowns, and export full data (CSV/Excel/JSON).
"""
)

# ------------------------------------------------------------
# Flow chart (Graphviz). If Graphviz isn't available, fall back to text.
# ------------------------------------------------------------
st.subheader("How it works (flow chart)")
flow_dot = r'''
digraph G {
  rankdir=LR;
  node [shape=box, style="rounded", fontsize=12];

  A [label="1) Data Sources\n(SQL)"];
  B [label="2) Extraction & Unification\n• validation\n• ISO3/country mapping"];
  C [label="3) Configuration\n• approach & model\n• categories & topics\n• indicator/news weights\n(optional: LLM topic help)"];
  D [label="4) Forecasting Engine\n• Baselines: CAGR / ETS / Logistic\n• Methods: Top-Down / Bottom-Up / Country-Specific / Global"];
  E [label="5) Adjustments (optional)\n• Indicators\n• News (GDELT)\n• Temporal decay & weighting"];
  F [label="6) Validation\n• requirements checks\n• sanity tests"];
  G [label="7) Insights & Visualization\n• charts, metrics, breakdowns"];
  H [label="8) Export\n• CSV / Excel / JSON"];

  A -> B -> C -> D -> E -> F -> G -> H;
}
'''
try:
    st.graphviz_chart(flow_dot, use_container_width=True)
except Exception:
    st.info("Flow: Data → Extraction → Configuration → Forecasting → (Optional) Adjustments → Validation → Insights → Export")

# ------------------------------------------------------------
# Method flows (concise per-approach diagrams)
# ------------------------------------------------------------
st.subheader("Method flows")
tabs = st.tabs(["Top-Down", "Bottom-Up", "Country-Specific", "Global"])

with tabs[0]:
    st.caption("Start global → allocate to regions/countries → LLM-gated news → apply signals.")
    top_down_dot = r'''
digraph TD {
  rankdir=LR; node [shape=box, style="rounded", fontsize=11];

  G [label="Global Baseline\n(CAGR / ETS / Logistic)"];
  R [label="Regional Allocation\n• shares / constraints"];
  C [label="Country Allocation\n• shares / constraints"];

  IND [label="Indicators\n• align + normalize\n• z-score/scale" shape=box];
  N0 [label="News Fetch (GDELT)"];
  N1 [label="Initial Filters\n• keywords\n• ISO3/country\n• date"];
  N2 [label="Deduplicate"];
  LLM [label="LLM Analysis\n• topic match\n• relevance 0–1\n• stance/sentiment"];
  GATE [label="Relevant?" shape=diamond];
  DROP [label="Discard non‑relevant" shape=note];
  SCORE [label="Score + Time Decay"];
  AGG [label="Aggregate\nregion/country/time"];
  NS [label="News Signal\n(0–1)"];

  ADJ [label="Apply Adjustments\n• news vs indicators\n• impact multiplier"];
  OUT [label="Outputs\nRegion + Country series"];

  G -> R -> C -> ADJ -> OUT;
  IND -> ADJ;
  N0 -> N1 -> N2 -> LLM -> GATE;
  GATE -> DROP [label="No"];
  GATE -> SCORE [label="Yes"];
  SCORE -> AGG -> NS -> ADJ;
}
'''
    try:
        st.graphviz_chart(top_down_dot, use_container_width=True)
    except Exception:
        st.info("Top-Down: Global → Allocation → Adjustments (News/Indicators) → Outputs")

with tabs[1]:
    st.caption("Start countries → aggregate up → LLM-gated news → apply signals.")
    bottom_up_dot = r'''
digraph BU {
  rankdir=LR; node [shape=box, style="rounded", fontsize=11];

  C0 [label="Country Baselines\n(per country)"];
  R0 [label="Aggregate to Regions\n• sum / rules"];
  G0 [label="Aggregate to Global"];

  IND [label="Indicators\n• align + normalize" shape=box];
  N0 [label="News Fetch (GDELT)"]; N1 [label="Initial Filters"]; N2 [label="Deduplicate"]; LLM [label="LLM Analysis\n• topic match\n• relevance\n• stance"]; GATE [label="Relevant?" shape=diamond]; DROP [label="Discard" shape=note]; SCORE [label="Score + Decay"]; AGG [label="Aggregate"];
  NS [label="News Signal (0–1)"];

  ADJ [label="Apply Adjustments\n• weights + multiplier"];
  OUT [label="Outputs\nGlobal + Region + Country"];

  C0 -> R0 -> G0 -> ADJ -> OUT;
  IND -> ADJ; N0 -> N1 -> N2 -> LLM -> GATE; GATE -> DROP [label="No"]; GATE -> SCORE [label="Yes"]; SCORE -> AGG -> NS -> ADJ;
}
'''
    try:
        st.graphviz_chart(bottom_up_dot, use_container_width=True)
    except Exception:
        st.info("Bottom-Up: Countries → Aggregate → Adjustments → Outputs")

with tabs[2]:
    st.caption("Select subset of countries → baselines → LLM-gated news → adjust.")
    country_dot = r'''
digraph CS {
  rankdir=LR; node [shape=box, style="rounded", fontsize=11];

  SEL [label="Select Countries"];
  C0 [label="Country Baselines"];

  IND [label="Indicators\n• align + normalize" shape=box];
  N0 [label="News Fetch (GDELT)"]; N1 [label="Initial Filters"]; N2 [label="Deduplicate"]; LLM [label="LLM Analysis\n• topic match\n• relevance\n• stance"]; GATE [label="Relevant?" shape=diamond]; DROP [label="Discard" shape=note]; SCORE [label="Score + Decay"]; AGG [label="Aggregate by country"]; NS [label="News Signal (0–1)"];

  ADJ [label="Apply Adjustments\n• weights + multiplier"];
  OUT [label="Outputs\nSelected countries (+ optional region/global)"];

  SEL -> C0 -> ADJ -> OUT;
  IND -> ADJ; N0 -> N1 -> N2 -> LLM -> GATE; GATE -> DROP [label="No"]; GATE -> SCORE [label="Yes"]; SCORE -> AGG -> NS -> ADJ;
}
'''
    try:
        st.graphviz_chart(country_dot, use_container_width=True)
    except Exception:
        st.info("Country-Specific: Select → Baselines → Adjust → Outputs")

with tabs[3]:
    st.caption("Single global series → LLM-gated news → adjustments.")
    global_dot = r'''
digraph GLB {
  rankdir=LR; node [shape=box, style="rounded", fontsize=11];

  G [label="Global Baseline"];
  IND [label="Indicators\n• align + normalize" shape=box];
  N0 [label="News Fetch (GDELT)"]; N1 [label="Initial Filters"]; N2 [label="Deduplicate"]; LLM [label="LLM Analysis\n• topic match\n• relevance\n• stance"]; GATE [label="Relevant?" shape=diamond]; DROP [label="Discard" shape=note]; SCORE [label="Score + Decay"]; AGG [label="Aggregate global"]; NS [label="News Signal (0–1)"];
  ADJ [label="Apply Adjustments\n• weights + multiplier"];
  OUT [label="Output\nGlobal series"];

  G -> ADJ -> OUT; IND -> ADJ; N0 -> N1 -> N2 -> LLM -> GATE; GATE -> DROP [label="No"]; GATE -> SCORE [label="Yes"]; SCORE -> AGG -> NS -> ADJ;
}
'''
    try:
        st.graphviz_chart(global_dot, use_container_width=True)
    except Exception:
        st.info("Global: Baseline → (Optional) Adjustments → Output")

# ------------------------------------------------------------
# LLM usage (concise explainer)
# ------------------------------------------------------------
st.subheader("How the LLM is used")
st.info(
    """
    - Fetch articles from GDELT using your configured topics and date window.  
    - Apply initial filters (keywords, ISO3/country mapping) and deduplicate items.  
    - For each article, an LLM analyzes topic match, relevance (0–1), and stance/sentiment.  
    - Non‑relevant items are discarded; relevant ones receive a time‑decayed relevance score.  
    - Scores are aggregated by country/region and time to form the News Signal used in adjustments.  
    - If LLM is disabled, only keyword filters and basic heuristics are used (no LLM gating).
    """
)

# ------------------------------------------------------------
# Quick start
# ------------------------------------------------------------
st.markdown("### Getting started")
st.markdown(
    """
1. **Data Extraction** – Upload/connect and unify your data.  
2. **Configuration** – Choose method, baseline model, topics & weights.  
3. **Forecasting** – Run the engine; optionally include indicators/news.  
4. **Insights** – Explore global/region/country charts & breakdowns.  
5. **Export** – Download full historical + forecast datasets.
6. **Model Evaluation** – Stored out-of-sample walk-forward results on real U.S. Census e-commerce data.
"""
)

st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Start Data Extraction", type="primary", use_container_width=True):
        st.switch_page("pages/01_Data_Extraction.py")

st.markdown("---")

# ------------------------------------------------------------
# System Status (retained)
# ------------------------------------------------------------
st.markdown("### System status")
if st.session_state.get("extraction_complete"):
    st.success("✅ Data extraction completed")
    if "unified_data" in st.session_state:
        unified_data = st.session_state["unified_data"]
        extraction_params = unified_data.get("extraction_params", {})

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Historical Cutoff", extraction_params.get("hist_cutoff", "N/A"))
        with c2:
            st.metric("Forecast Until", extraction_params.get("forecast_until", "N/A"))
        with c3:
            if "market_value" in unified_data:
                market_records = unified_data["market_value"]["metadata"].get("total_records", 0)
                st.metric("Total Records", market_records)
else:
    st.info("Ready to extract data")
