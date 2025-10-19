"""
Redesigned Forecasting Page - Dynamic layout based on method.
Single view mode for Global/Single Country, Multi-country mode redirects to Insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import logging

from src.services.forecast import ForecastService
from src.services.configuration import ConfigurationService
from src.ui_components import display_forecast_chart, show_forecast_metrics
from pages.components.utils import format_value_intelligent, detect_data_unit
from ui.streamlit.components.calibration_card import render_calibration_cards, render_calibration_metrics
from src.constants import ForecastMode

st.set_page_config(page_title="Forecasting", page_icon="📈", layout="wide")

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

# Initialize services
forecast_service = ForecastService()
config_service = ConfigurationService()

def get_adjustment_weights(config, session_state):
    """Return weights for news and indicators."""
    weights = session_state.get('adjustment_weights', config.get('adjustment_weights', {}))
    news_w = float(weights.get('news_weight', 0.70))
    indicator_w = max(0.0, 1.0 - news_w)
    return {"news_weight": news_w, "indicator_weight": indicator_w}

# Check prerequisites
if 'unified_data' not in st.session_state or not st.session_state['unified_data']:
    st.error("❌ No data available. Please complete Data Extraction first.")
    if st.button("Go to Data Extraction"):
        st.switch_page("pages/01_Data_Extraction.py")
    st.stop()

if 'config' not in st.session_state or not st.session_state['config']:
    st.error("❌ No configuration found. Please complete Configuration first.")
    if st.button("Go to Configuration"):
        st.switch_page("pages/02_Configuration.py")
    st.stop()

# Load data and config
unified_data = st.session_state['unified_data']
config = st.session_state['config'].copy()
# Use saved year parameters (fallback to widget values, then defaults)
config['hist_cutoff'] = st.session_state.get('saved_hist_cutoff', st.session_state.get('hist_cutoff', 2020))
config['forecast_until'] = st.session_state.get('saved_forecast_until', st.session_state.get('forecast_until', 2025))

# Helper Functions
def calculate_impact_timelines(hist_cutoff, forecast_until, unified_data=None):
    """Calculate dynamic timelines for impact application"""
    # Get actual historical data range
    hist_start = hist_cutoff - 8  # Default fallback
    
    if unified_data:
        # Try global data first
        global_data = unified_data.get('market_value', {}).get('global', pd.DataFrame())
        if not global_data.empty and 'year' in global_data.columns:
            hist_start = int(global_data['year'].min())
        else:
            # Try country data if no global data
            country_data = unified_data.get('market_value', {}).get('country', pd.DataFrame())
            if not country_data.empty and 'year' in country_data.columns:
                hist_start = int(country_data['year'].min())
    
    forecast_years = forecast_until - hist_cutoff
    short_term_years = min(max(1, int(forecast_years * 0.25)), 3)
    
    short_term_start = hist_cutoff + 1
    short_term_end = hist_cutoff + short_term_years
    
    # Format short-term display properly
    if short_term_years == 1:
        short_term_display = f"{short_term_start}"
    else:
        short_term_display = f"{short_term_start}-{short_term_end}"
    
    return {
        'historical': f"{hist_start}-{hist_cutoff}",
        'forecast': f"{hist_cutoff+1}-{forecast_until}"
    }

def should_show_insights_page():
    """Determine if Insights page should be shown"""
    approach = config.get('forecast_approach', ['Global-level'])
    method = approach[0] if isinstance(approach, list) else approach
    
    if method in ['Top-down', 'Bottom-up']:
        return True
    elif method == 'Country-Specific':
        countries = config.get('selected_countries', [])
        return len(countries) > 1
    else:  # Global-level
        return False

def get_forecast_method_display():
    """Get method display string"""
    approach = config.get('forecast_approach', ['Global-level'])
    method = approach[0] if isinstance(approach, list) else approach
    
    if method == 'Country-Specific':
        countries = config.get('selected_countries', [])
        if len(countries) == 1:
            return f"Country-Specific: {countries[0]}"
        else:
            return f"Country-Specific: {len(countries)} countries"
    return method

def get_service_method_name(ui_method_name):
    """Convert UI method name to service method name"""
    method_mapping = {
        'Global-level': 'Global Only',
        'Top-down': 'Top-Down', 
        'Bottom-up': 'Bottom-Up',
        'Country-specific': 'Country-Specific'
    }
    return method_mapping.get(ui_method_name, ui_method_name)

def _recency_weighted_mean_list(rows, half_life_days):
    """Helper to calculate recency-weighted mean from list of dicts with growth_rate and date"""
    if not rows:
        return 0.0
    
    df = pd.DataFrame(rows)
    if df.empty or 'growth_rate' not in df.columns:
        return 0.0
    
    # Use same logic as in NewsAdjustment
    x = pd.to_numeric(df.get('growth_rate'), errors='coerce')
    d = pd.to_datetime(df.get('date'), errors='coerce')
    
    m = x.notna() & d.notna()
    if not m.any():
        # Fallback to simple mean if dates are missing
        return float(x.mean()) if x.notna().any() else 0.0
    
    age_days = (pd.Timestamp.utcnow() - d[m]).dt.days.clip(lower=0)
    # Ensure half_life is positive to avoid division issues
    half_life_days = max(1, abs(half_life_days))
    w = np.power(0.5, age_days / half_life_days)
    
    return float(np.average(x[m], weights=w))

def create_adjustment_breakdown_table(forecast_result, analyzed_news, config, unified_data):
    """Create comprehensive adjustment breakdown table showing progression from baseline to final forecast"""
    
    # Get the forecast data and filter to only FUTURE years
    hist_cutoff = config.get('hist_cutoff', 2020)
    # Support all methods: use 'forecast' when present, otherwise 'global_forecast'.
    # If still empty, try single-country (Country-Specific with one country selected).
    forecast_df = forecast_result.get('forecast', pd.DataFrame())
    if forecast_df.empty:
        forecast_df = forecast_result.get('global_forecast', pd.DataFrame())
    # Baseline may be stored under different keys
    baseline_df = forecast_result.get('baseline', pd.DataFrame())
    if baseline_df.empty:
        baseline_df = forecast_result.get('baseline_global_forecast', pd.DataFrame())

    single_country_mode = False
    single_country_name = None
    if forecast_df.empty:
        country_forecasts = forecast_result.get('country_forecasts', {}) or {}
        if isinstance(country_forecasts, dict) and len(country_forecasts) == 1:
            single_country_mode = True
            single_country_name = next(iter(country_forecasts.keys()))
            cval = country_forecasts[single_country_name]
            # Extract forecast/baseline from nested dict if present
            if isinstance(cval, dict):
                forecast_df = cval.get('forecast', pd.DataFrame())
                baseline_df = cval.get('baseline', pd.DataFrame()) if baseline_df.empty else baseline_df
            else:
                forecast_df = cval if isinstance(cval, pd.DataFrame) else pd.DataFrame()
            # Historical data should be country-level when in single-country mode
            historical_data = []
            country_market = unified_data.get('market_value', {}).get('country', pd.DataFrame())
            if not country_market.empty and 'country' in country_market.columns:
                hist_data = country_market[country_market['country'] == single_country_name]
                hist_data = hist_data[hist_data['year'] <= hist_cutoff]
                for _, row in hist_data.iterrows():
                    historical_data.append({
                        'Year': int(row['year']),
                        'Baseline': row.get('value', 0),
                        'Type': 'Historical'
                    })
    
    if forecast_df.empty:
        return pd.DataFrame()
    
    # Filter forecast data to ONLY future years (> hist_cutoff)
    forecast_df = forecast_df[forecast_df['year'] > hist_cutoff].copy()
    if not baseline_df.empty:
        baseline_df = baseline_df[baseline_df['year'] > hist_cutoff].copy()
    
    if forecast_df.empty:
        return pd.DataFrame()  # No future forecast data
    
    # Get historical data for context (global by default; country-level if single_country_mode)
    if not single_country_mode:
        historical_data = []
        global_data = unified_data.get('market_value', {}).get('global', pd.DataFrame())
        if not global_data.empty:
            hist_data = global_data[global_data['year'] <= hist_cutoff].copy()
            for _, row in hist_data.iterrows():
                historical_data.append({
                    'Year': int(row['year']),
                    'Baseline': row.get('value', 0),  # Keep original value
                    'Type': 'Historical'
                })
    # else historical_data already built in single-country branch above
    
    # Get actual adjustment details from forecast result
    adjustment_details = forecast_result.get('adjustment_details', {})
    year_adjustments = {}

    # Extract per-year adjustments if available
    if adjustment_details:
        if single_country_mode and single_country_name in adjustment_details:
            # Use the single country's adjustments directly
            year_adjustments = (adjustment_details.get(single_country_name, {}) or {}).get('year_adjustments', {})
        elif 'global' in adjustment_details:
            # Global forecast
            global_details = adjustment_details['global']
            year_adjustments = global_details.get('year_adjustments', {})
        else:
            # Aggregate country year adjustments to a global view using baseline shares
            baseline_country_forecasts = forecast_result.get('baseline_country_forecasts', {})
            if baseline_country_forecasts:
                agg_adjustments = {}
                years = forecast_df['year'].astype(int).tolist()
                for year in years:
                    # Sum baseline across countries for weighting
                    total_baseline = 0.0
                    country_baselines = {}
                    for ctry, bdf in baseline_country_forecasts.items():
                        try:
                            row = bdf[bdf['year'] == year]
                            if not row.empty:
                                val = float(row.iloc[0].get('value_hat', 0.0))
                                country_baselines[ctry] = val
                                total_baseline += val
                        except Exception:
                            continue
                    if total_baseline <= 0:
                        continue
                    # Weighted average of per-country percentages
                    news_acc = 0.0
                    ind_acc = 0.0
                    for ctry, base_val in country_baselines.items():
                        share = base_val / total_baseline if total_baseline else 0.0
                        c_details = adjustment_details.get(ctry, {})
                        y_adj = (c_details.get('year_adjustments', {}) or {}).get(year, {})
                        news_acc += share * float(y_adj.get('news_pct', 0.0) or 0.0)
                        ind_acc += share * float(y_adj.get('indicators_pct', 0.0) or 0.0)
                    agg_adjustments[year] = {
                        'news_pct': news_acc,
                        'indicators_pct': ind_acc
                    }
                year_adjustments = agg_adjustments
            else:
                # Fallback: pick first available country's details if aggregation is impossible
                first_country_details = next(iter(adjustment_details.values()), {})
                year_adjustments = first_country_details.get('year_adjustments', {})
    
    # Build forecast section - ONLY for future years
    forecast_data = []
    for _, row in forecast_df.iterrows():
        year = int(row['year'])
        
        # Get baseline value
        if not baseline_df.empty and year in baseline_df['year'].values:
            baseline_value = baseline_df.loc[baseline_df['year'] == year, 'value_hat'].iloc[0]
        else:
            baseline_value = row.get('value_hat', 0)
        
        # Keep original baseline value - no automatic conversion
        
        # Get final value
        final_value = row.get('value_hat', baseline_value)
        # Keep original final value - no automatic conversion
        
        # Get year-specific adjustments if available
        year_adj = year_adjustments.get(year, {})
        news_pct = year_adj.get('news_pct')
        indicators_pct = year_adj.get('indicators_pct')
        
        # Calculate final value from baseline + adjustments to ensure consistency
        calculated_final = baseline_value
        if news_pct is not None and news_pct != 0:
            calculated_final *= (1 + news_pct / 100)
        if indicators_pct is not None and indicators_pct != 0:
            calculated_final *= (1 + indicators_pct / 100)
        
        # Use calculated final if we have adjustments, otherwise use stored value
        if year_adj:
            final_value = calculated_final
        
        forecast_data.append({
            'Year': year,
            'Baseline': baseline_value,
            'News_Pct': news_pct if news_pct is not None and news_pct != 0 else None,
            'Indicators_Pct': indicators_pct if indicators_pct is not None and indicators_pct != 0 else None,
            'Final': final_value,
            'Type': 'Forecast'
        })
    
    # Combine historical and forecast data - NO DUPLICATION
    all_data = []
    
    # Add historical data (NO adjustments ever)
    for hist_row in historical_data:
        all_data.append({
            'Year': hist_row['Year'],
            'Baseline': hist_row['Baseline'],
            'News_Pct': None,  # Historical data NEVER has adjustments
            'Indicators_Pct': None,
            'Final': hist_row['Baseline'],  # Final = Baseline for historical
            'Type': 'Historical'
        })
    
    # Add forecast data (WITH adjustments)
    all_data.extend(forecast_data)
    
    # Convert to DataFrame and sort by year
    df = pd.DataFrame(all_data)
    df = df.sort_values('Year')
    
    # Format for display
    display_df = df.copy()
    
    # Fix value formatting - intelligent unit display
    display_df['Baseline'] = display_df['Baseline'].apply(format_value_intelligent)
    display_df['Final'] = display_df['Final'].apply(format_value_intelligent)
    
    # Format percentage columns
    def format_pct(x):
        if pd.isna(x) or x == 0:
            return ""
        return f"{x:+.1f}%"
    
    display_df['News (%)'] = display_df['News_Pct'].apply(format_pct)
    display_df['Indicators (%)'] = display_df['Indicators_Pct'].apply(format_pct)
    
    # Dynamic column filtering - only show columns with actual data
    columns_to_show = ['Year', 'Baseline']
    
    # Check if any forecast row has these adjustments
    has_news = any(pd.notna(row['News_Pct']) and row['News_Pct'] != 0 for row in forecast_data)
    has_indicators = any(pd.notna(row['Indicators_Pct']) and row['Indicators_Pct'] != 0 for row in forecast_data)
    
    if has_news:
        columns_to_show.append('News (%)')
    if has_indicators:
        columns_to_show.append('Indicators (%)')
        
    columns_to_show.append('Final')
    
    return display_df[columns_to_show]

# === PAGE HEADER ===
st.title("📈 Forecasting")
st.info("""
Run news analysis (if configured), generate the baseline and adjusted forecasts, and review results. Use the controls in each section to run steps in order.
""")

# Method display
method_display = get_forecast_method_display()
forecast_mode = config.get('forecast_mode', ForecastMode.CLASSIC.value)
if forecast_mode == ForecastMode.EXISTING_FORECAST_NEWS.value:
    baseline_method = "Existing forecast (news-only)"
else:
    baseline_method = config.get('forecast_method', '3-yr CAGR')

col1, col2 = st.columns([2, 1])
with col1:
    st.info(f"**Method:** {method_display}")
with col2:
    st.info(f"**Baseline:** {baseline_method}")

# === NEWS ANALYSIS SECTION ===
st.header("📰 News Analysis")

market_name = config.get('market_name', 'Global Market')
topics = config.get('topics', [])

if not topics:
    st.warning("⚠️ News analysis is enabled but no topics are configured.")
    st.stop()

# Determine if this is multi-country mode
is_multi_country = should_show_insights_page()

if is_multi_country:
    # Multi-country mode - show summary
    st.markdown("Processing news analysis for multiple countries...")
    
    if st.button("🔍 Run News Analysis", type="primary", use_container_width=True):
        # Get countries list
        approach = config.get('forecast_approach', ['Global-level'])
        method = approach[0] if isinstance(approach, list) else approach
        
        if method == 'Country-Specific':
            selected_countries = config.get('selected_countries', [])
        elif method in ['Top-down', 'Bottom-up']:
            country_market = unified_data.get('market_value', {}).get('country', pd.DataFrame())
            selected_countries = [c for c in country_market['country'].unique().tolist() if c and isinstance(c, str) and c.strip()] if not country_market.empty else []
        
        # Convert topics to strings
        topic_strings = []
        for topic in topics:
            if isinstance(topic, dict) and topic.get('active', True):
                topic_strings.append(topic.get('topic', ''))
            elif isinstance(topic, str):
                topic_strings.append(topic)
        topic_strings = [t for t in topic_strings if t.strip()]
        
        # Run analysis
        progress_placeholder = st.empty()
        
        def update_progress(message):
            with progress_placeholder.container():
                st.info(f"🔄 {message}")
        
        with st.spinner("Running news analysis..."):
            news_result = forecast_service.analyze_news(
                method=get_service_method_name(method),
                market_name=market_name,
                topics=topic_strings,
                unified_data=unified_data,
                countries=selected_countries,
                progress_callback=update_progress,
                config=st.session_state.get('config', {})
            )
        
        progress_placeholder.empty()
        
        if 'error' in news_result:
            st.error(f"❌ News analysis failed: {news_result['error']}")
            st.stop()
        
        # Store results
        st.session_state['analyzed_news'] = news_result

        # Render calibration cards (reactive to scope)
        st.subheader("🧪 Calibration (LLM)")
        cal = news_result.get('calibration') or {}
        ntype = news_result.get('type')
        # If analysis is global-only, show only global calibration.
        if ntype == 'global_news':
            render_calibration_cards(cal, show_global=True, show_countries=False)
        else:
            render_calibration_cards(cal)
        
        # Extract news DataFrames for Insights page
        news_dataframes = {}
        if news_result.get('type') == 'country_news':
            news_dataframes = news_result.get('data', {})
        elif news_result.get('type') == 'combined_news':
            news_dataframes['global'] = news_result.get('global_data')
            news_dataframes.update(news_result.get('country_data', {}))
        elif news_result.get('type') == 'global_news':
            news_dataframes['global'] = news_result.get('data')
        
        st.session_state['news_dataframes'] = news_dataframes
        
        # Show summary
        st.success("✅ News analysis complete for all countries")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Countries Analyzed", len(selected_countries))
        with col2:
            # Handle different news result structures
            news_type = news_result.get('type')
            if news_type == 'country_news':
                total_articles = news_result.get('count', 0)
            elif news_type == 'combined_news':
                total_articles = news_result.get('global_count', 0) + sum(news_result.get('country_counts', {}).values())
            else:
                total_articles = news_result.get('count', 0)
            st.metric("Total Headlines", total_articles)
        with col3:
            avg_sentiment = "Positive"  # Simplified for now
            st.metric("Average Sentiment", avg_sentiment)
    
    # Show generate forecast button if analysis is done
    if 'analyzed_news' in st.session_state:
        st.header("🚀 Generate Forecast")
        
        if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
            # Run forecast
            with st.spinner("Generating forecast..."):
                approach = config.get('forecast_approach', ['Global-level'])
                method = approach[0] if isinstance(approach, list) else approach
                
                if method == 'Country-Specific':
                    selected_countries = config.get('selected_countries', [])
                elif method in ['Top-down', 'Bottom-up']:
                    country_market = unified_data.get('market_value', {}).get('country', pd.DataFrame())
                    selected_countries = [c for c in country_market['country'].unique().tolist() if c and isinstance(c, str) and c.strip()] if not country_market.empty else []
                
                forecast_args = {
                    'method': get_service_method_name(method),
                    'unified_data': unified_data,
                    'config': config,
                    'selected_countries': selected_countries,
                    'analyzed_news': st.session_state['analyzed_news']
                }
                
                result = forecast_service.run_forecast(**forecast_args)
                st.session_state['forecast_result'] = result
                st.session_state['forecast_method'] = method
            
            if 'error' in result:
                st.error(f"❌ Forecast failed: {result['error']}")
            else:
                st.success("✅ Forecast complete! Redirecting to Insights...")
                time.sleep(2)
                st.switch_page("pages/04_Insights.py")

else:
    # Single view mode - show full analysis
    if st.button("🔍 Run News Analysis", type="primary", use_container_width=True, key="run_news_single"):
        # Get method for single view
        approach = config.get('forecast_approach', ['Global-level'])
        method = approach[0] if isinstance(approach, list) else approach
        
        if method == 'Country-Specific':
            selected_countries = config.get('selected_countries', [])
        else:
            selected_countries = None
        
        # Convert topics to strings
        topic_strings = []
        for topic in topics:
            if isinstance(topic, dict) and topic.get('active', True):
                topic_strings.append(topic.get('topic', ''))
            elif isinstance(topic, str):
                topic_strings.append(topic)
        topic_strings = [t for t in topic_strings if t.strip()]
        
        # Run analysis with progress
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        def update_progress(message):
            with progress_placeholder.container():
                if 'Fetching global' in message:
                    st.info("🌍 **Global News Collection**")
                elif 'Analyzing global' in message:
                    st.info("🤖 **Global LLM Analysis**")
                elif 'Fetching news for' in message:
                    st.info("🌐 **Country News Collection**")
                elif 'Analyzing' in message and ':' in message:
                    st.info("🤖 **Country LLM Analysis**")
                else:
                    st.info("🔄 **Processing**")
            
            with status_placeholder.container():
                st.caption(message)
        
        with st.spinner("Running news analysis..."):
            news_result = forecast_service.analyze_news(
                method=get_service_method_name(method),
                market_name=market_name,
                topics=topic_strings,
                unified_data=unified_data,
                countries=selected_countries,
                progress_callback=update_progress,
                config=st.session_state.get('config', {})
            )
        
        progress_placeholder.empty()
        status_placeholder.empty()
        
        if 'error' in news_result:
            st.error(f"❌ News analysis failed: {news_result['error']}")
        else:
            st.session_state['analyzed_news'] = news_result
            st.success("✅ Analysis Complete")
    
    # Show analysis results if available
    if 'analyzed_news' in st.session_state:
        analyzed_news = st.session_state['analyzed_news']
        
        # Show headlines dataframe
        with st.expander("📊 Analyzed Headlines", expanded=False):
            # Combine all dataframes for display
            all_headlines = []
            
            if analyzed_news.get('type') == 'combined_news':
                global_data = analyzed_news.get('global_data', pd.DataFrame())
                if not global_data.empty:
                    all_headlines.append(global_data)
                
                country_data = analyzed_news.get('country_data', {})
                for country_df in country_data.values():
                    if not country_df.empty:
                        all_headlines.append(country_df)
            
            elif analyzed_news.get('type') == 'global_news':
                data = analyzed_news.get('data', pd.DataFrame())
                if not data.empty:
                    all_headlines.append(data)
            
            elif analyzed_news.get('type') == 'country_news':
                data = analyzed_news.get('data', {})
                for country_df in data.values():
                    if not country_df.empty:
                        all_headlines.append(country_df)
            
            if all_headlines:
                combined_df = pd.concat(all_headlines, ignore_index=True)
                if not combined_df.empty:
                    # Add search and filter section
                    st.markdown("##### Search & Filters")
                    col_search, col_impact = st.columns([2, 1])
                    
                    with col_search:
                        search_term = st.text_input("🔍 Search headlines", "", key="forecast_search")
                    
                    with col_impact:
                        impact_range = st.slider(
                            "Impact Range (%)",
                            min_value=-100,
                            max_value=100,
                            value=(-100, 100),
                            key="forecast_impact_range"
                        )
                    
                    # Apply filters
                    filtered_df = combined_df.copy()
                    
                    # Search filter
                    if search_term:
                        if 'title' in filtered_df.columns:
                            mask = filtered_df['title'].str.contains(search_term, case=False, na=False)
                            filtered_df = filtered_df[mask]
                    

                    # Impact range filter
                    if 'growth_rate' in filtered_df.columns:
                        filtered_df = filtered_df[
                            (filtered_df['growth_rate'] >= impact_range[0]) & 
                            (filtered_df['growth_rate'] <= impact_range[1])
                        ]
                    
                    # Display filtered count
                    if len(filtered_df) < len(combined_df):
                        st.info(f"Showing {len(filtered_df)} of {len(combined_df)} headlines (filtered)")
                    
                    # Display relevant columns (including date for visibility)
                    display_cols = ['title', 'date', 'category', 'growth_rate', 'reason']
                    available_cols = [col for col in display_cols if col in filtered_df.columns]
                    
                    if available_cols and not filtered_df.empty:
                        # Show all rows with scrollable dataframe
                        st.dataframe(
                            filtered_df[available_cols],
                            use_container_width=True,
                            height=400,  # Fixed height with scrollbar
                            hide_index=True
                        )
                        
                        # Add summary statistics for filtered data
                        st.markdown("##### Impact Distribution (Filtered)")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_articles = len(filtered_df)
                            st.metric("Total Articles", total_articles)
                        
                        with col2:
                            if 'growth_rate' in filtered_df.columns:
                                positive_count = (filtered_df['growth_rate'] > 0).sum()
                                pct = positive_count/total_articles*100 if total_articles > 0 else 0
                                st.metric("Positive Impact", positive_count, 
                                         help=f"{pct:.1f}% of articles")
                        
                        with col3:
                            if 'growth_rate' in filtered_df.columns:
                                neutral_count = (filtered_df['growth_rate'] == 0).sum()
                                pct = neutral_count/total_articles*100 if total_articles > 0 else 0
                                st.metric("Neutral/Noise", neutral_count,
                                         help=f"{pct:.1f}% of articles")
                        
                        with col4:
                            if 'growth_rate' in filtered_df.columns:
                                negative_count = (filtered_df['growth_rate'] < 0).sum()
                                pct = negative_count/total_articles*100 if total_articles > 0 else 0
                                st.metric("Negative Impact", negative_count,
                                         help=f"{pct:.1f}% of articles")
                    elif filtered_df.empty:
                        st.warning("No headlines match the current filters")
        
        # Display news impact from backend MA calculation
        st.subheader("📊 News Impact Analysis")
        
        # Get the already-calculated MA impact and breakdown from the backend
        news_avg = 0.0
        category_breakdown = None
        breakdown_source = None
        
        if analyzed_news:
            # Get the calibration data which now includes MA calculation
            calibration = analyzed_news.get('calibration', {})
            
            # Get the forecasting method from config
            approach = config.get('forecast_approach', ['Global-level'])
            method = approach[0] if isinstance(approach, list) else approach
            
            # For country-specific and regional methods, try to get breakdown from countries
            if method in ["Country-Specific", "Bottom-Up", "Top-down"] and calibration.get('countries'):
                # Try to find a country with breakdown
                for country, country_cal in calibration['countries'].items():
                    if country_cal.get('category_breakdown'):
                        category_breakdown = country_cal.get('category_breakdown')
                        news_avg = country_cal.get('news_avg_pct', 0.0)
                        breakdown_source = f"Country: {country}"
                        break
                
                # If no country has breakdown, fall back to global
                if not category_breakdown:
                    global_cal = calibration.get('global', {})
                    news_avg = global_cal.get('news_avg_pct', 0.0)
                    category_breakdown = global_cal.get('category_breakdown', None)
                    breakdown_source = "Global" if category_breakdown else None
            else:
                # For global methods, use global calibration
                global_cal = calibration.get('global', {})
                news_avg = global_cal.get('news_avg_pct', 0.0)
                category_breakdown = global_cal.get('category_breakdown', None)
                breakdown_source = "Global" if category_breakdown else None
        
        # Display the detailed category breakdown with impact calculation
        with st.expander("📊 Detailed Category Impact Breakdown", expanded=True):
            if category_breakdown:
                # Show source of breakdown
                if breakdown_source:
                    st.caption(f"Showing breakdown for: **{breakdown_source}**")
                
                # Part 1: Summary Table with Visual Indicators
                st.subheader("Category Analysis")
                
                # Prepare data for display
                table_data = []
                for cat_name, info in category_breakdown.items():
                    window = info.get('window', 0)
                    days_with_data = info.get('days_with_data', 0)
                    coverage = (days_with_data / window * 100) if window > 0 else 0
                    
                    # Status icon based on coverage and status
                    if info.get('status') == 'Neutral (not contributing)':
                        status_icon = "⚪"
                    elif info.get('status') != 'Active':
                        status_icon = "❌"
                    elif coverage >= 75:
                        status_icon = "✅"
                    elif coverage >= 30:
                        status_icon = "⚠️"
                    else:
                        status_icon = "❌"
                    
                    table_data.append({
                        'Category': cat_name,
                        'MA Window': f"{window} days",
                        'Data Available': f"{days_with_data} days",
                        'Coverage': f"{coverage:.0f}%",
                        'MA Score': f"{info.get('ma_score', 0):+.2f}%" if info.get('status') == 'Active' else 'N/A',
                        'Status': status_icon
                    })
                
                # Display table with color coding
                df = pd.DataFrame(table_data)
                
                # Apply color coding based on coverage
                def color_coverage(row):
                    coverage_str = row['Coverage']
                    try:
                        coverage_val = float(coverage_str.replace('%', ''))
                    except:
                        coverage_val = 0
                    
                    if row['Status'] == "⚪":
                        return ['background-color: #f0f0f0'] * len(row)
                    elif row['Status'] == "❌" and row['MA Score'] == 'N/A':
                        return ['background-color: #f8d7da'] * len(row)
                    elif coverage_val >= 75:
                        return ['background-color: #d4f1d4'] * len(row)
                    elif coverage_val >= 30:
                        return ['background-color: #fff3cd'] * len(row)
                    else:
                        return ['background-color: #f8d7da'] * len(row)
                
                styled_df = df.style.apply(color_coverage, axis=1)
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # Part 2: Calculation Walkthrough
                st.subheader("Calculation Details")
                
                active_categories = [(name, info) for name, info in category_breakdown.items() 
                                    if info.get('status') == 'Active']
                
                if active_categories:
                    # Show the math
                    calc_lines = []
                    scores = []
                    for name, info in active_categories:
                        score = info.get('ma_score', 0)
                        scores.append(score)
                        days_used = info.get('days_with_data', 0)
                        window = info.get('window', 0)
                        # Pad category names for alignment
                        padded_name = f"{name:30s}"
                        calc_lines.append(f"{padded_name}: {score:+6.2f}% (using {days_used}/{window} days)")
                    
                    calc_lines.append("─" * 70)
                    calc_lines.append(f"Categories with data: {len(scores)}")
                    
                    # Show the averaging calculation
                    if scores:
                        avg = sum(scores) / len(scores)
                        total = sum(scores)
                        scores_str = " + ".join([f"({s:+.2f})" for s in scores])
                        calc_lines.append(f"Sum of scores: {scores_str} = {total:+.2f}%")
                        calc_lines.append(f"Final Impact: {total:+.2f}% ÷ {len(scores)} = {avg:+.2f}%")
                    
                    st.code("\n".join(calc_lines))
                else:
                    st.info("No categories with active data")
                
                # Part 3: Data Quality Assessment
                st.subheader("Data Quality Assessment")
                
                quality_messages = []
                for cat_name, info in category_breakdown.items():
                    window = info.get('window', 0)
                    days_with_data = info.get('days_with_data', 0)
                    coverage = (days_with_data / window * 100) if window > 0 else 0
                    status = info.get('status', '')
                    
                    if status == 'Neutral (not contributing)':
                        continue
                    elif status == 'No recent data' or status == 'No data in window':
                        quality_messages.append(f"ℹ️ '{cat_name}' has no recent news data")
                    elif coverage < 30 and window > 0:
                        quality_messages.append(f"⚠️ '{cat_name}' has only {coverage:.0f}% data coverage")
                    elif coverage >= 75 and status == 'Active':
                        quality_messages.append(f"✅ '{cat_name}' has good data coverage ({coverage:.0f}%)")
                
                if quality_messages:
                    for msg in quality_messages:
                        st.write(msg)
                else:
                    st.success("✅ All categories have sufficient data")
                
                # Legend for color coding
                st.markdown("---")
                st.markdown("**Legend:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("🟢 **Green**: >75% coverage")
                with col2:
                    st.markdown("🟡 **Yellow**: 30-75% coverage")
                with col3:
                    st.markdown("🔴 **Red**: <30% coverage")
                with col4:
                    st.markdown("⚪ **Gray**: Neutral/No impact")
                
                # Show final calculated impact prominently
                st.markdown("---")
                # Recalculate from active categories to ensure consistency
                if active_categories:
                    final_impact = sum([info.get('ma_score', 0) for _, info in active_categories]) / len(active_categories)
                    # Update news_avg to match the displayed value
                    news_avg = final_impact
                else:
                    news_avg = news_avg
                    
                # Get decay rate from calibration
                cal = (st.session_state.get('analyzed_news') or {}).get('calibration') or {}
                approach = config.get('forecast_approach', ['Global-level'])
                method = approach[0] if isinstance(approach, list) else approach
                gl = None
                if method == 'Country-Specific':
                    sel = config.get('selected_countries', []) or []
                    if len(sel) == 1:
                        gl = (cal.get('countries') or {}).get(sel[0])
                if not gl:
                    gl = cal.get('global')
                if not gl:
                    gl = {'news_avg_pct': news_avg}
                if 'news_avg_pct' not in gl:
                    gl['news_avg_pct'] = news_avg
                
                decay_rate = float(gl.get('long_term_decay_rate', 0.60))
                
                # Display both metrics in the same row
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📈 Final News Growth Impact", f"{news_avg:+.2f}%",
                             help="Calculated from category-specific moving averages")
                with col2:
                    st.metric("⏳ Decay Rate", f"{decay_rate:.2f}",
                             help="Rate at which news impact decays over forecast years")
            else:
                # No categories configured - impact is zero
                news_avg = 0.0
                st.warning("No categories configured. Please configure categories in the Configuration page to calculate news impact.")
        
        # Show LLM reasoning if available
        cal = (st.session_state.get('analyzed_news') or {}).get('calibration') or {}
        approach = config.get('forecast_approach', ['Global-level'])
        method = approach[0] if isinstance(approach, list) else approach
        gl = None
        if method == 'Country-Specific':
            sel = config.get('selected_countries', []) or []
            if len(sel) == 1:
                gl = (cal.get('countries') or {}).get(sel[0])
        if not gl:
            gl = cal.get('global')
        
        # Show reasoning if available
        reason = gl.get('reasoning', '') if gl else ''
        if reason:
            st.info(f"**LLM Recommendation:** {reason}")
        
        
        # Generate Forecast Button
        st.header("🚀 Generate Forecast")
        
        if st.button("🚀 Generate Forecast", type="primary", use_container_width=True, key="generate_forecast_single"):
            with st.spinner("Generating forecast..."):
                approach = config.get('forecast_approach', ['Global-level'])
                method = approach[0] if isinstance(approach, list) else approach
                
                if method == 'Country-Specific':
                    selected_countries = config.get('selected_countries', [])
                else:
                    selected_countries = None
                
                forecast_args = {
                    'method': get_service_method_name(method),
                    'unified_data': unified_data,
                    'config': config,
                    'selected_countries': selected_countries,
                    'analyzed_news': analyzed_news
                }
                
                result = forecast_service.run_forecast(**forecast_args)
                st.session_state['forecast_result'] = result
                st.session_state['forecast_method'] = method
            
            if 'error' in result:
                st.error(f"❌ Forecast failed: {result['error']}")
            else:
                st.success("✅ Forecast completed successfully!")
                st.rerun()
    
    # Show forecast results if available
    if 'forecast_result' in st.session_state and 'error' not in st.session_state['forecast_result']:
        result = st.session_state['forecast_result']
        
        st.header("📈 Forecast Results")
        
        # Split view: Chart + Adjustment Breakdown Table
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Main forecast chart
            try:
                display_forecast_chart(result)
            except:
                # Fallback simple chart display
                forecast = result.get('forecast', pd.DataFrame())
                if not forecast.empty and 'year' in forecast.columns and 'value_hat' in forecast.columns:
                    st.line_chart(data=forecast.set_index('year')['value_hat'], height=400)
                else:
                    st.info("Chart display not available")
        
        with col2:
            st.markdown("### 📊 Adjustment Breakdown")
            
            # Create comprehensive adjustment table
            adjustment_table = create_adjustment_breakdown_table(
                result, analyzed_news, config, unified_data
            )
            
            if not adjustment_table.empty:
                # Display the table
                st.dataframe(
                    adjustment_table,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                # Show fallback information if available
                metadata = result.get('metadata', {})
                fallback_countries = metadata.get('fallback_countries', {})
                if fallback_countries:
                    st.warning(f"⚠️ **Fallback Data Used:** {len(fallback_countries)} countries used global news as fallback due to insufficient country-specific data.")
                    with st.expander("View Fallback Details"):
                        for country, reason in fallback_countries.items():
                            st.write(f"- **{country}:** {reason}")
                
                # Show adjustment decay information briefly
                adjustment_details = result.get('adjustment_details', {})
                if adjustment_details:
                    first_country_details = next(iter(adjustment_details.values()), {})
                    year_adjustments = first_country_details.get('year_adjustments', {})
                    if year_adjustments:
                        st.info("📈 **Decay Applied:** News impact decays exponentially across the forecast horizon.")
                
                # Add enhanced explanation
                st.caption(f"""
                **Table Explanation**:
                • **Historical years**: Actual data (no adjustments)
                • **Baseline**: Unadjusted forecast from selected method  
                • **News**: News impact with temporal decay
                • **Indicators**: Economic indicator adjustments
                • **Final**: Fully adjusted forecast
                """)
            else:
                st.warning("No forecast data available for breakdown table")
        
        # Export section below
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            forecast = result.get('forecast', pd.DataFrame())
            if not forecast.empty:
                st.download_button(
                    "📥 Download Forecast CSV",
                    data=forecast.to_csv(index=False),
                    file_name=f"forecast_{method_display}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            if not adjustment_table.empty:
                st.download_button(
                    "📥 Download Breakdown CSV", 
                    data=adjustment_table.to_csv(index=False),
                    file_name=f"breakdown_{method_display}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# Navigation
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("⬅️ Back to Configuration"):
        st.switch_page("pages/02_Configuration.py")

with col2:
    if not is_multi_country:
        if st.button("➡️ Continue to Export"):
            st.switch_page("pages/05_Export.py")
    else:
        if 'forecast_result' in st.session_state:
            if st.button("➡️ Go to Insights"):
                st.switch_page("pages/04_Insights.py")
