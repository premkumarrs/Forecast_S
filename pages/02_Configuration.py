"""
Configuration page for forecasting setup.
Handles market categories, AI topic generation, and forecast options.
"""
import streamlit as st
import pandas as pd
from src.services.configuration import ConfigurationService
from src.services.data import DataExtractionService
from src.session import get_from_session, save_to_session
from src.constants import ForecastMode
from src.llm.categories.manager import load_market_categories, save_market_categories, load_categories
from src.llm.analyst import SimpleLLMAnalyst as LLM_Analyst
from src.compatibility import ensure_backward_compatibility, check_data_availability
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(page_title="Configuration – Forecasting App", page_icon="⚙️", layout="wide")

# Local style tweaks for cleaner layout on this page
st.markdown(
    """
    <style>
      /* Reduce heading sizes and spacing */
      section.main h1 { font-size: 2rem; line-height: 1.2; margin-bottom: 0.25rem; }
      section.main h2 { font-size: 1.35rem; line-height: 1.25; margin-top: 1rem; margin-bottom: 0.35rem; }
      section.main h3 { font-size: 1.05rem; line-height: 1.3; margin-top: 0.8rem; margin-bottom: 0.25rem; }

      /* Tighter block padding to reduce empty space */
      .block-container { padding-top: 1rem; padding-bottom: 2rem; }

      /* Slightly smaller metric value for better balance */
      div[data-testid="stMetricValue"] { font-size: 1.6rem; }
      div[data-testid="stMetricLabel"] { font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize services
config_service = ConfigurationService()
extraction_service = DataExtractionService()

# Helper Functions
def generate_topics_with_llm(market_name: str, num_topics: int = 7) -> List[str]:
    """Generate GDELT search topics using LLM"""
    try:
        return config_service.generate_topics_with_llm(market_name, num_topics)
    except Exception as e:
        st.error(f"Failed to generate topics: {str(e)}")
        return []

def get_available_countries_from_session() -> List[str]:
    """Extract country list from unified_data"""
    unified_data = get_from_session('unified_data', {})
    return extraction_service.get_available_countries(unified_data)

def validate_configuration() -> Tuple[bool, List[str]]:
    """Validate all configuration settings"""
    config_valid = True
    validation_messages = []
    
    # Check if categories exist
    categories = st.session_state.get('current_categories', [])
    if not categories:
        validation_messages.append("⚠️ No categories defined")
        config_valid = False
    
    # Check if topics exist  
    topics = st.session_state.get('current_topics', [])
    if not topics:
        validation_messages.append("⚠️ No search topics defined")
        config_valid = False
    
    return config_valid, validation_messages

# Check data availability
has_data, is_unified = check_data_availability()
if not has_data:
    st.warning("⚠️ No data available. Please complete Data Extraction first.")
    st.page_link("pages/01_Data_Extraction.py", label="Go to Data Extraction", icon="⬅️")
    st.stop()

# Ensure compatibility if using unified data
if is_unified:
    ensure_backward_compatibility()

st.title("⚙️ Configuration")
st.markdown("Configure your market forecasting settings")

# ===== 1. MARKET DEFINITION SECTION =====
st.header("📈 Market Definition")

# Pull market name from extracted data using service
unified_data = get_from_session('unified_data', {})
default_market_name = config_service.get_market_name_from_data(unified_data)

market_name = st.text_input(
    "Market Name", 
    value=get_from_session('market_name', default_market_name),
    help="This will be used for all forecasting operations",
    placeholder="e.g., Infrastructure as a Service (IaaS) Market",
    key="market_name")

# ===== 2. FORECASTING CONFIGURATION SECTION =====
st.header("⚙️ Forecasting Configuration")

mode_value = st.session_state.get('forecast_mode', ForecastMode.CLASSIC.value)
is_existing_mode = mode_value == ForecastMode.EXISTING_FORECAST_NEWS.value

# Forecast approach selection (single selection)
forecast_approaches = config_service.get_original_forecast_approaches()
if is_existing_mode:
    forecast_approaches = [approach for approach in forecast_approaches if approach != 'Top-down']
# Get default index for forecast approach
prev_approach = get_from_session('forecast_approach', ['Global-level'])
if isinstance(prev_approach, list) and prev_approach:
    default_approach = prev_approach[0]
else:
    default_approach = prev_approach if prev_approach else 'Global-level'

try:
    default_index = forecast_approaches.index(default_approach)
except ValueError:
    default_index = 0

forecast_approach = st.selectbox(
    "Forecast Approach",
    forecast_approaches,
    index=default_index,
    help="Choose your forecasting strategy",
    key="forecast_approach"
)

# Country selection if country-specific is chosen
if st.session_state.get('forecast_approach') == "Country-Specific":
    available_countries = get_available_countries_from_session()
    if available_countries:
        selected_countries = st.multiselect(
            "Select Countries for Forecasting",
            options=available_countries,
            default=get_from_session('selected_countries', available_countries[:5]),
            help="Choose specific countries to forecast",
            key="selected_countries"
        )
    else:
        st.warning("No countries available in extracted data")

# Forecast method selection
original_methods = config_service.get_original_forecast_methods()
if not is_existing_mode:
    forecast_method = st.selectbox(
        "Baseline Forecast Method",
        original_methods,
        index=0,
        help="Choose the primary forecasting algorithm",
        key="forecast_method"
    )
else:
    if not st.session_state.get('forecast_method'):
        st.session_state['forecast_method'] = original_methods[0]
    st.info("Baseline forecast type is fixed when applying existing forecasts.")

with st.expander("📊 Forecast Adjustments Configuration", expanded=False):
    if not is_existing_mode:
        # 1) Indicator weights section (with note)
        st.info("Adjust per-indicator weights (0.0–1.0) to control each indicator's contribution. Use Reset to return all to 0.5.")

        # Get available indicators from data using service
        market_countries = config_service.get_countries_with_market_data(unified_data)
        indicator_data = unified_data.get('indicators', {}).get('country', pd.DataFrame())

        if not indicator_data.empty and market_countries:
            indicator_data = indicator_data[indicator_data['country'].isin(market_countries)]
            indicator_col = config_service.get_indicator_column_name(unified_data)

            if indicator_col in indicator_data.columns:
                available_indicators = sorted(indicator_data[indicator_col].unique())

                if 'indicator_weights' not in st.session_state:
                    st.session_state.indicator_weights = config_service.configure_indicator_weights(
                        unified_data, market_countries)

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("**Indicator Weights (0.0 - 1.0):**")
                    for indicator in available_indicators:
                        current_weight = st.session_state.indicator_weights.get(indicator, 0.5)
                        _ = st.slider(
                            f"{indicator}",
                            min_value=0.0,
                            max_value=1.0,
                            value=current_weight,
                            step=0.1,
                            help=f"Weight for {indicator} in forecast adjustments",
                            key=f"indicator_weight_{indicator}"
                        )
                        st.session_state.indicator_weights[indicator] = st.session_state.get(
                            f"indicator_weight_{indicator}", current_weight)

                with col2:
                    st.markdown("**Global Settings:**")
                    if st.button("🔄 Reset All Weights", key="reset_weights_btn"):
                        st.session_state.indicator_weights = {ind: 0.5 for ind in available_indicators}
                        st.success("✅ Weights reset to 0.5")
            else:
                st.info("ℹ️ No economic indicators found in extracted data.")
                st.session_state.indicator_weights = {}
        else:
            st.info("ℹ️ No economic indicators available for configuration.")
            st.session_state.indicator_weights = {}
    else:
        st.session_state.indicator_weights = {}
        st.info("Indicator adjustments are disabled in this mode.")

    st.markdown("---")

    # 2) News/indicator influence section (with overview note)
    
    st.info("""
**Forecast adjustments overview**

- **News Influence (0–100%)** — share of total adjustment derived from news analysis.  
- **Indicator Influence** — automatically = (100% − News%).
""")

    if is_existing_mode:
        st.session_state['news_weight'] = 100

    col1, col2, col3 = st.columns(3)
    with col1:
        news_weight = st.slider(
            "📰 News Influence",
            0, 100, st.session_state.get('news_weight', 70),
            help="% of adjustment from news (remaining goes to indicators)",
            key="news_weight",
            disabled=is_existing_mode
        )
        if is_existing_mode:
            st.caption("Locked at 100% in this mode; indicators are not used.")
    with col2:
        indicator_weight = 100 - st.session_state.get('news_weight', 70)
        st.metric("📊 Indicator Influence", f"{0 if is_existing_mode else indicator_weight}%")
    with col3:
        st.info("Decay Rate is automatically calibrated by the LLM from current headlines and market context.")

    if is_existing_mode:
        forecast_until_year = st.session_state.get('saved_forecast_until', st.session_state.get('forecast_until', 2030))
        default_start = st.session_state.get('adjustment_start_year', st.session_state.get('hist_cutoff', 2020))
        default_end = st.session_state.get('adjustment_end_year', forecast_until_year)
        if default_start > forecast_until_year:
            default_start = forecast_until_year
        if default_end > forecast_until_year:
            default_end = forecast_until_year
        default_start = min(default_start, default_end)

        col_start, col_end = st.columns(2)
        with col_start:
            adjustment_start_year = st.number_input(
                "Adjustment Start Year",
                min_value=2000,
                max_value=forecast_until_year,
                value=default_start,
                step=1,
                key="adjustment_start_year",
                help="News impacts are applied only within this year range."
            )
        with col_end:
            adjustment_end_year = st.number_input(
                "Adjustment End Year",
                min_value=adjustment_start_year,
                max_value=forecast_until_year,
                value=default_end if default_end >= adjustment_start_year else adjustment_start_year,
                step=1,
                key="adjustment_end_year",
                help="News impacts are applied only within this year range."
            )

        derived_hist_cutoff = adjustment_start_year - 1 if adjustment_start_year is not None else forecast_until_year - 1
        st.session_state['hist_cutoff'] = derived_hist_cutoff
        st.session_state['saved_hist_cutoff'] = derived_hist_cutoff

# Set use_indicators and use_news to always align with selected mode
st.session_state['use_indicators'] = not is_existing_mode
st.session_state['use_news'] = True

# ===== 3. NEWS & AI SETTINGS SECTION =====
st.header("📰 News & AI Settings")

# Categories Management
st.subheader("📁 Market Categories")

# Check for existing categories using service
existing_categories = config_service.load_market_categories(market_name)
if existing_categories:
    st.success(f"✅ Loaded {len(existing_categories)} categories for market: **{market_name}**")
else:
    existing_categories = load_categories()  # Load default categories
    st.info(f"ℹ️ No custom categories found for **{market_name}**. Loaded {len(existing_categories)} default categories.")

# Initialize categories in session state
if 'current_categories' not in st.session_state:
    st.session_state.current_categories = existing_categories

# Convert to DataFrame for display
if st.session_state.current_categories:
    categories_df = pd.DataFrame(st.session_state.current_categories)
    if 'growth_constraint_min' not in categories_df.columns:
        categories_df['growth_constraint_min'] = -10.0
    if 'growth_constraint_max' not in categories_df.columns:
        categories_df['growth_constraint_max'] = 10.0
    if 'description' not in categories_df.columns:
        categories_df['description'] = ''
    if 'ma_window_days' not in categories_df.columns:
        categories_df['ma_window_days'] = 30
else:
    categories_df = pd.DataFrame(columns=['name', 'description', 'growth_constraint_min', 'growth_constraint_max', 'ma_window_days'])

# Display categories in expandable section
with st.expander("Current Categories", expanded=True):
    if not categories_df.empty:
        # Editable dataframe
        edited_categories = st.data_editor(
            categories_df,
            use_container_width=True,
            num_rows="dynamic",
            height=400,
            column_config={
                "name": st.column_config.TextColumn("Category Name", required=True),
                "description": st.column_config.TextColumn("Description"),
                "growth_constraint_min": st.column_config.NumberColumn("Min Growth (%)", format="%.1f"),
                "growth_constraint_max": st.column_config.NumberColumn("Max Growth (%)", format="%.1f"),
                "ma_window_days": st.column_config.NumberColumn(
                    "MA Window (days)", 
                    min_value=0, 
                    max_value=120, 
                    step=1,
                    help="Moving average window in days. 0 = neutral/no impact"
                )
            },
            key="categories_editor"
        )
        
        # Save button for categories
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("💾 Save Categories", key="save_categories_btn"):
                # Auto-fix categories before saving
                categories_to_save = edited_categories.to_dict('records')
                
                # Auto-fix invalid min/max growth values and MA windows
                for category in categories_to_save:
                    min_growth = category.get('growth_constraint_min', -10.0)
                    max_growth = category.get('growth_constraint_max', 10.0)
                    
                    # If min >= max, auto-correct them
                    if min_growth >= max_growth:
                        category['growth_constraint_min'] = -10.0
                        category['growth_constraint_max'] = 10.0
                    
                    # Ensure name is not empty
                    if not category.get('name', '').strip():
                        category['name'] = f"Category {categories_to_save.index(category) + 1}"
                    
                    # Ensure MA window is valid
                    ma_window = category.get('ma_window_days', 30)
                    if ma_window < 0:
                        category['ma_window_days'] = 0
                    elif ma_window > 120:
                        category['ma_window_days'] = 120
                
                # Save without validation (since we auto-fixed)
                st.session_state.current_categories = categories_to_save
                success = config_service.save_market_categories(market_name, categories_to_save)
                if success:
                    st.success("✅ Categories saved successfully!")
                else:
                    st.error("❌ Failed to save categories")
    else:
        st.info("No categories defined yet. Generate some below!")

# Category generation controls
st.markdown("**Generate New Categories**")
col1, col2, col3, col4 = st.columns(4)

with col1:
    num_categories = st.number_input("Number of Categories", 5, 20, 10, 1)

with col2:
    min_growth = st.number_input("Min Growth Rate (%)", -100.0, 100.0, -10.0)

with col3:
    max_growth = st.number_input("Max Growth Rate (%)", -100.0, 100.0, 10.0)

with col4:
    if st.button("🤖 Generate Categories"):
        try:
            with st.spinner(f"Generating {num_categories} categories..."):
                categories = config_service.generate_market_categories(market_name, num_categories, min_growth, max_growth)
                
                # Growth constraints should already be set by the LLM via the service
                # Only add fallback if completely missing (shouldn't happen with proper LLM response)
                for category in categories:
                    if 'growth_constraint_min' not in category:
                        category['growth_constraint_min'] = min_growth
                    if 'growth_constraint_max' not in category:
                        category['growth_constraint_max'] = max_growth
                
                if categories:
                    st.session_state.current_categories = categories
                    success = config_service.save_market_categories(market_name, categories)
                    if success:
                        st.success(f"Generated {len(categories)} categories!")
                        # Clear session state to force reload from file
                        del st.session_state['current_categories']
                        st.rerun()
                    else:
                        st.error("❌ Failed to save generated categories")
                else:
                    st.error("Failed to generate categories")
        except Exception as e:
            st.error(f"Error generating categories: {str(e)}")

# GDELT Search Topics
st.subheader("🔍 GDELT Search Topics")

# Initialize topics in session state - load existing topics for the market
if 'current_topics' not in st.session_state:
    # Try to load existing topics for this market
    existing_topics = config_service.load_market_topics(market_name)
    if existing_topics:
        st.session_state.current_topics = existing_topics
        st.success(f"✅ Loaded {len(existing_topics)} existing topics for market: **{market_name}**")
    else:
        st.session_state.current_topics = []
else:
    existing_topics = st.session_state.current_topics
    if existing_topics:
        st.success(f"✅ Loaded {len(existing_topics)} topics for market: **{market_name}**")
    else:
        st.info(f"ℹ️ No custom topics found for **{market_name}**. Generate some below!")

# Convert to DataFrame for display
topics_df = pd.DataFrame()
if st.session_state.current_topics:
    topics_df = pd.DataFrame(st.session_state.current_topics)
    # Ensure required columns exist
    if 'topic' not in topics_df.columns:
        topics_df['topic'] = ''
    if 'active' not in topics_df.columns:
        topics_df['active'] = True
else:
    topics_df = pd.DataFrame(columns=['topic', 'active'])



# Display topics in expandable section
with st.expander("Current Topics", expanded=True):
    if not topics_df.empty:
        # Editable dataframe
        edited_topics = st.data_editor(
            topics_df,
            use_container_width=True,
            num_rows="dynamic",
            height=400,
            column_config={
                "topic": st.column_config.TextColumn("Search Topic", required=True),
                "active": st.column_config.CheckboxColumn("Active", default=True)
            },
            key="topics_editor"
        )
        
        # Save button for topics
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("💾 Save Topics", key="save_topics_btn"):
                # Validate topics before saving
                topics_to_save = edited_topics.to_dict('records')
                # Filter out empty topics
                topics_to_save = [t for t in topics_to_save if t.get('topic', '').strip()]
                
                if topics_to_save:
                    st.session_state.current_topics = topics_to_save
                    success = config_service.save_market_topics(market_name, topics_to_save)
                    if success:
                        st.success("✅ Topics saved successfully!")
                    else:
                        st.error("❌ Failed to save topics")
                else:
                    st.error("❌ No valid topics to save")
    else:
        st.info("No topics defined yet. Generate some below!")

# Topic generation controls
st.markdown("**Generate New Topics**")
col1, col2 = st.columns(2)

with col1:
    num_topics = st.number_input("Number of Topics (max 10)", 1, 10, 10, 1)

with col2:
    if st.button("🤖 Generate Topics"):
        try:
            with st.spinner(f"Generating {num_topics} search topics..."):
                ai_topics = generate_topics_with_llm(market_name, num_topics)
                
                if ai_topics and len(ai_topics) > 0:
                    # Replace all topics with new AI-generated ones
                    new_topics = [{'topic': topic, 'active': True} for topic in ai_topics]
                    st.session_state.current_topics = new_topics
                    
                    success = config_service.save_market_topics(market_name, st.session_state.current_topics)
                    if success:
                        st.success(f"Generated {len(ai_topics)} topics!")
                        # Clear session state to force reload from file
                        del st.session_state['current_topics']
                        st.rerun()
                    else:
                        st.error("❌ Failed to save generated topics")
                        
                        # Show topics anyway in session state
                        st.info("Topics are available in current session (not saved to disk)")
                else:
                    st.error("Failed to generate topics - received empty list")
        except Exception as e:
            st.error(f"Error generating topics: {str(e)}")




# ===== 4. SAVE CONFIGURATION SECTION =====
st.markdown("---")

# Validation
config_valid, validation_messages = validate_configuration()

# Display validation messages
if validation_messages:
    for msg in validation_messages:
        st.warning(msg)

# Configuration summary
if config_valid:
    st.success("✅ Configuration is complete and valid")
    
    # Show summary
    with st.expander("Configuration Summary", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Market & Data:**")
            st.write(f"- Market: {market_name}")
            st.write(f"- Categories: {len(st.session_state.current_categories)}")
            st.write(f"- Topics: {len([t for t in st.session_state.current_topics if t.get('active', True)])}")
            
        with col2:
            st.markdown("**Forecast Settings:**")
            st.write(f"- Approach: {st.session_state.get('forecast_approach', 'Not Selected')}")
            st.write(f"- Method: {st.session_state.get('forecast_method', 'Not Selected')}")
            st.write(f"- Indicators: {'No' if is_existing_mode else 'Yes'}")
            st.write(f"- News Analysis: Yes")
            
        with col3:
            st.markdown("**Indicator Settings:**")
            indicator_count = len(st.session_state.get('indicator_weights', {}))
            st.write(f"- Configured: {indicator_count}")
            if indicator_count > 0:
                avg_weight = sum(st.session_state.get('indicator_weights', {}).values()) / indicator_count
                st.write(f"- Avg Weight: {avg_weight:.2f}")
            st.write(f"- News Weight: {st.session_state.get('news_weight', 70)}%")
            st.write(f"- Indicator Weight: {0 if is_existing_mode else 100 - st.session_state.get('news_weight', 70)}%")
            if is_existing_mode:
                st.write(f"- Adjustment Start: {st.session_state.get('adjustment_start_year', 'N/A')}")
                st.write(f"- Adjustment End: {st.session_state.get('adjustment_end_year', 'N/A')}")

# Save configuration button
if st.button("💾 Save Configuration & Continue", type="primary", disabled=not config_valid, use_container_width=True):
    # Prepare configuration data
    news_weight_pct = st.session_state.get('news_weight', 70)
    indicator_weight_pct = 100 - news_weight_pct if not is_existing_mode else 0

    config_data = {
        'market_name': st.session_state.get('market_name'),
        'categories': st.session_state.current_categories,
        'topics': [t for t in st.session_state.current_topics if t.get('active', True)],
        'forecast_approach': [st.session_state.get('forecast_approach', 'Global-level')],  # Convert to list
        'selected_countries': st.session_state.get('selected_countries', []),
        'forecast_method': st.session_state.get('forecast_method', ''),
        'use_indicators': not is_existing_mode,
        'use_news': True,
        'indicator_weights': st.session_state.get('indicator_weights', {}),
        'adjustment_weights': (
            {'news_weight': 1.0, 'indicator_weight': 0.0}
            if is_existing_mode else
            {
                'news_weight': news_weight_pct / 100,
                'indicator_weight': indicator_weight_pct / 100,
            }
        ),
        'adjustment_start_year': st.session_state.get('adjustment_start_year') if is_existing_mode else None,
        'adjustment_end_year': st.session_state.get('adjustment_end_year') if is_existing_mode else None,
        'configured_timestamp': pd.Timestamp.now(),
        'forecast_mode': st.session_state.get('forecast_mode', ForecastMode.CLASSIC.value),
    }
    
    # Save to session state using service
    success = config_service.save_configuration(config_data)
    save_to_session('configuration_complete', True)
    
    if success:
        st.success("🎉 Configuration saved successfully!")
        st.balloons()
    else:
        st.error("❌ Failed to save configuration")

# Navigation
st.markdown("---")
st.markdown("### 🧭 Navigation")
col1, col2 = st.columns(2)

with col1:
    if st.button("⬅️ Back to Data Extraction"):
        st.switch_page("pages/01_Data_Extraction.py")

with col2:
    if get_from_session('configuration_complete', False):
        if st.button("➡️ Continue to Forecasting"):
            st.switch_page("pages/03_Forecasting.py")
    else:
        st.button("➡️ Continue to Forecasting", disabled=True, help="Complete configuration first")
