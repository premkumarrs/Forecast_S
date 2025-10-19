"""
Data Extraction Page - Clean service-based implementation.
"""

import streamlit as st
import pandas as pd
from src.services.data import DataExtractionService
from src.session import init_session_state
from src.constants import ForecastMode, FORECAST_MODE_LABELS
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Data Extraction - AI Market Forecaster",
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
      div[data-testid="stMetricValue"] { font-size: 1.6rem; }
      div[data-testid="stMetricLabel"] { font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
init_session_state()


def _apply_query_mode_override():
    """Allow optional ?mode= query parameter to set forecast mode."""
    try:
        # Use new st.query_params API (dictionary-like object)
        params = st.query_params
        mode_param = params.get('mode', None)
    except Exception:
        mode_param = None
    if not mode_param:
        return
    candidate = mode_param
    valid_values = {mode.value for mode in ForecastMode}
    if candidate in valid_values and st.session_state.get('forecast_mode') != candidate:
        st.session_state['forecast_mode'] = candidate


_apply_query_mode_override()

# Initialize service
extraction_service = DataExtractionService()

# Cache for queries
if 'query_cache' not in st.session_state:
    st.session_state.query_cache = {}

def get_cache_key(market_kpi, indicator_kpis, hist_cutoff, forecast_until):
    """Generate cache key for unified queries"""
    key_str = f"unified_{market_kpi}_{'-'.join(sorted(indicator_kpis))}_{hist_cutoff}_{forecast_until}"
    return hashlib.md5(key_str.encode()).hexdigest()

def execute_unified_extraction(market_kpi, indicator_kpis, year_params):
    """Execute unified data extraction using service."""
    cache_key = get_cache_key(market_kpi, indicator_kpis, year_params['hist_cutoff'], year_params['forecast_until'])
    
    if cache_key in st.session_state.query_cache:
        return st.session_state.query_cache[cache_key]
    
    try:
        # Use service for extraction
        if not extraction_service.has_database_connection():
            st.warning("⚠️ No database connection configured. Using mock data for demo.")
            result = extraction_service.create_mock_data(
                market_kpi, 
                indicator_kpis, 
                year_params['hist_cutoff'], 
                year_params['forecast_until']
            )
        else:
            result = extraction_service.extract_unified_data(
                market_kpi, 
                indicator_kpis, 
                year_params['hist_cutoff'], 
                year_params['forecast_until']
            )
        
        # Cache result if successful
        if 'error' not in result:
            st.session_state.query_cache[cache_key] = result
        
        return result
        
    except Exception as e:
        return {'error': f'Extraction failed: {str(e)}'}

# === HEADER SECTION ===
with st.container():
    st.title("📊 Data Extraction")
    st.info("""
    Configure inputs, validate parameters, and extract unified market and indicator data.
    Steps: 1) Configure, 2) Validate, 3) Extract & Preview.
    """)

mode_options = [ForecastMode.CLASSIC, ForecastMode.EXISTING_FORECAST_NEWS]
try:
    current_mode = ForecastMode(
        st.session_state.get('forecast_mode', ForecastMode.CLASSIC.value)
    )
except ValueError:
    current_mode = ForecastMode.CLASSIC
    st.session_state['forecast_mode'] = current_mode.value

with st.container():
    selected_mode = st.selectbox(
        "Mode",
        options=mode_options,
        index=mode_options.index(current_mode),
        format_func=lambda mode: FORECAST_MODE_LABELS.get(mode, mode.value),
        help="Choose between the classic forecast workflow and Existing Forecast + News Adjustment, which reuses provided market series and applies news-only adjustments.",
        key="forecast_mode_selector",
    )

if selected_mode != current_mode:
    st.session_state['forecast_mode'] = selected_mode.value

is_existing_mode = selected_mode == ForecastMode.EXISTING_FORECAST_NEWS

# === STEP 1: CONFIGURATION CARD ===
with st.container():
    st.subheader("STEP 1: Configure Parameters")

col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.write("**Market Configuration**")
        market_kpi = st.text_input(
            "Market KPI Key", 
            value=st.session_state.get('global_kpi_key', ''),
            placeholder="e.g., 16000_16110_1_1_revenue",
            help="Primary KPI that represents the market size/value"
        )

indicator_kpis_input = ""
with col2:
    if not is_existing_mode:
        with st.container():
            st.write("**Indicator Configuration**")
            indicator_kpis_input = st.text_area(
                "Indicator KPI Keys (one per line)",
                value=st.session_state.get('country_kpi_key', ''),
                placeholder="e.g.,\ninternetPenetration_kmi\ngdpTotalCurrentPricesCurrentFX_kmi",
                help="Leading indicators that might predict market changes",
                height=100
            )

# Parse indicator KPIs
indicator_kpis = []
if not is_existing_mode and indicator_kpis_input.strip():
    indicator_kpis = [kpi.strip() for kpi in indicator_kpis_input.split('\n') if kpi.strip()]

# Time Period Configuration
with st.container():
    st.write("**Time Period**")
    col1, col2 = st.columns(2)

with col1:
    if is_existing_mode:
        hist_cutoff = st.session_state.get('saved_hist_cutoff', st.session_state.get('hist_cutoff', 2024))
        st.session_state['hist_cutoff'] = hist_cutoff
        st.caption("Historical range automatically includes all available data in this mode.")
    else:
        hist_cutoff = st.number_input(
            "Historical Cutoff Year",
            min_value=2000,
            max_value=2030,
            value=st.session_state.get('saved_hist_cutoff', st.session_state.get('hist_cutoff', 2024)),
            help="Latest year of historical data to use",
            key="hist_cutoff"
        )

with col2:
    default_forecast = max(hist_cutoff + 5, st.session_state.get('saved_forecast_until', st.session_state.get('forecast_until', 2030)))
    forecast_until = st.number_input(
        "Forecast Until Year",
        min_value=hist_cutoff + 1,
        max_value=2040,
        value=default_forecast,
        help="Final year to forecast to",
        key="forecast_until"
    )

year_params = {
    'hist_cutoff': st.session_state.get('hist_cutoff', hist_cutoff),
    'forecast_until': st.session_state.get('forecast_until', forecast_until)
}

# Remove redundant state updates since widgets now use keys

# === STEP 2: VALIDATION & EXECUTION ===
with st.container():
    st.subheader("STEP 2: Validation & Execution")
    
    # Validation
    validation = extraction_service.validate_extraction_params(
        market_kpi, indicator_kpis, hist_cutoff, forecast_until
    )

    if is_existing_mode and validation['warnings']:
        filtered_warnings = [
            w for w in validation['warnings']
            if "No indicator KPIs selected" not in w
        ]
        if len(filtered_warnings) != len(validation['warnings']):
            validation = validation.copy()
            validation['warnings'] = filtered_warnings

    if validation['valid']:
        st.success("✅ All parameters are valid")
    
    if validation['errors']:
        for error in validation['errors']:
            st.error(f"❌ {error}")

    if validation['warnings']:
        for warning in validation['warnings']:
            st.warning(f"⚠️ {warning}")

# (Validation details shown above; avoid duplicate block to reduce clutter)

# === 3. DATA EXTRACTION ===
st.subheader("🔄 Data Extraction")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    extract_button = st.button(
        "🚀 Extract Data", 
        type="primary",
        disabled=not validation['valid'],
        use_container_width=True
    )

with col2:
    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.session_state.query_cache = {}
        st.success("Cache cleared!")
        st.rerun()

with col3:
    cache_info = f"📦 Cache: {len(st.session_state.query_cache)} items"
    st.info(cache_info)

# Execute extraction
if extract_button and validation['valid']:
    with st.spinner("Extracting data..."):
        unified_data = execute_unified_extraction(market_kpi, indicator_kpis, year_params)
        
        if 'error' in unified_data:
            st.error(f"❌ Extraction failed: {unified_data['error']}")
        else:
            # Save to session
            st.session_state['unified_data'] = unified_data
            st.session_state['data_loaded'] = True
            st.session_state['extraction_complete'] = True  # For compatibility
            # Save year parameters with different keys to avoid widget key conflicts
            st.session_state['saved_hist_cutoff'] = hist_cutoff
            st.session_state['saved_forecast_until'] = forecast_until
            
            # Show success and summary
            st.success("✅ Data extraction completed successfully!")
            
            # Generate and display summary
            summary = extraction_service.get_extraction_summary(unified_data)
            
            st.subheader("📋 Extraction Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Market Data", "Yes" if summary['has_market_data'] else "No")
                st.metric("Global Records", summary['global_market_records'])
            
            with col2:
                st.metric("Indicator Data", "Yes" if summary['has_indicator_data'] else "No") 
                st.metric("Global Indicators", summary['global_indicator_records'])
            
            with col3:
                st.metric("Country Records", summary['country_market_records'])
                st.metric("Country Indicators", summary['country_indicator_records'])
            
            with col4:
                st.metric("Countries", len(summary['countries_with_data']))
                if summary['countries_with_data']:
                    st.write("Countries with data:")
                    st.write(", ".join(summary['countries_with_data'][:5]))
                    if len(summary['countries_with_data']) > 5:
                        st.write(f"... and {len(summary['countries_with_data']) - 5} more")

# === 4. CURRENT DATA STATUS ===
if 'unified_data' in st.session_state and st.session_state['unified_data']:
    with st.container():
        st.subheader("STEP 3: Extracted Data Preview")
    
    unified_data = st.session_state['unified_data']
    summary = extraction_service.get_extraction_summary(unified_data)
    available_countries = extraction_service.get_available_countries(unified_data)
    
    # Show available countries and data samples using tabs
    if available_countries:
        tab1, tab2, tab3 = st.tabs([f"Countries ({len(available_countries)})", "Market Data", "Indicators"])
        
        with tab1:
            st.write("**Available Countries:**")
            # Display countries in columns
            countries_per_col = 8
            cols = st.columns(4)
            
            for i, country in enumerate(available_countries):
                col_idx = i // countries_per_col
                if col_idx < len(cols):
                    cols[col_idx].write(f"• {country}")
        
        with tab2:
            if summary['has_market_data']:
                market_data = unified_data.get('market_value', {})
                
                if 'global' in market_data and not market_data['global'].empty:
                    st.write("**Global Market Data (Latest 5 records):**")
                    st.dataframe(market_data['global'].tail(), use_container_width=True)
                
                if 'country' in market_data and not market_data['country'].empty:
                    st.write("**Country Market Data (Sample):**")
                    sample_df = market_data['country'].groupby('country').tail(1).head(5)
                    st.dataframe(sample_df, use_container_width=True)
            else:
                st.info("No market data available")
        
        with tab3:
            if summary['has_indicator_data']:
                indicator_data = unified_data.get('indicators', {})
                
                if 'global' in indicator_data and not indicator_data['global'].empty:
                    st.write("**Global Indicator Data (Sample):**")
                    st.dataframe(indicator_data['global'].head(), use_container_width=True)
            else:
                st.info("No indicator data available")
    
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("➡️ Continue to Configuration", type="primary", use_container_width=True):
            st.switch_page("pages/02_Configuration.py")
    
    with col2:
        st.info("💡 Data is ready for configuration and forecasting!")

else:
    st.info("👆 Extract data to begin the forecasting process.")
