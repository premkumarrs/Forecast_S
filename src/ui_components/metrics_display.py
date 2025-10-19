"""
Reusable metrics display components.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional

from src.constants import ForecastMode

def show_forecast_metrics(result: Dict):
    """Display forecast result metrics."""
    if 'error' in result:
        st.error(f"❌ {result['error']}")
        return
    
    method = result.get('method', 'unknown')
    metadata = result.get('metadata', {})
    adjustments = result.get('adjustments_applied', [])
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Data Quality",
            "Good" if metadata.get('data_points', 0) >= 5 else "Limited",
            f"{metadata.get('data_points', 'N/A')} historical points",
            help="Quality of historical data used for forecasting"
        )
    
    with col2:
        # Show news compatibility if available
        news_compatibility = metadata.get('news_compatibility', {})
        if news_compatibility:
            compatible = news_compatibility.get('compatible', True)
            optimal = news_compatibility.get('optimal', True)
            if compatible and optimal:
                status = "Optimal"
                color = "normal"
            elif compatible:
                status = "Compatible"
                color = "off"
            else:
                status = "Incompatible"
                color = "inverse"
            
            st.metric(
                "News Coverage",
                status,
                help="Compatibility of news analysis with selected method"
            )
        else:
            adjustment_text = ', '.join(adjustments) if adjustments else 'baseline only'
            st.metric(
                "Adjustments",
                len(adjustments),
                adjustment_text,
                help="Number and type of adjustments applied"
            )
    
    with col3:
        # Show news impact statistics if available
        news_coverage = metadata.get('news_coverage', {})
        if news_coverage:
            total_articles = news_coverage.get('total_articles', 0)
            if total_articles > 0:
                st.metric(
                    "News Articles",
                    total_articles,
                    help="Total number of news articles analyzed"
                )
            else:
                st.metric("News Articles", "None")
        else:
            data_points = metadata.get('data_points', 'N/A')
            st.metric(
                "Data Points",
                data_points,
                help="Number of historical data points used"
            )
    
    with col4:
        if method in ['country_specific', 'top_down', 'bottom_up']:
            countries_count = len(result.get('country_forecasts', {}))
            st.metric(
                "Countries",
                countries_count,
                help="Number of countries forecasted"
            )
        elif 'country_forecasts' in result:
            countries_count = len(result['country_forecasts'])
            st.metric(
                "Countries",
                countries_count,
                help="Number of countries in forecast"
            )
        else:
            forecast_years = metadata.get('forecast_years', [])
            years_count = len(forecast_years)
            st.metric(
                "Forecast Years",
                years_count,
                help="Number of years forecasted"
            )
    
    # Show temporal decay settings if news analysis is enabled
    config = st.session_state.get('config', {})
    if config.get('use_news') and 'news' in adjustments:
        st.markdown("**Temporal Decay Settings:**")
        col1, col2 = st.columns(2)
        with col1:
            decay_rate = config.get('long_term_decay_rate', 0.6)
            st.write(f"• Decay rate: {decay_rate:.2f}")
        with col2:
            half_life = config.get('news_half_life_days', 90)
            st.write(f"• News half-life: {half_life} days")
    
    # Warnings if any
    warnings = metadata.get('warnings', [])
    if warnings:
        for warning in warnings:
            st.warning(f"⚠️ {warning}")


def show_data_metrics(unified_data: Dict):
    """Display data extraction metrics."""
    if not unified_data:
        st.warning("No data available")
        return
    
    # Market data metrics
    market_data = unified_data.get('market_value', {})
    global_market = market_data.get('global', pd.DataFrame())
    country_market = market_data.get('country', pd.DataFrame())
    
    # Indicator data metrics
    indicator_data = unified_data.get('indicators', {})
    global_indicators = indicator_data.get('global', pd.DataFrame())
    country_indicators = indicator_data.get('country', pd.DataFrame())
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Global Market Data",
            len(global_market),
            "data points",
            help="Historical market data points for global forecast"
        )
    
    with col2:
        countries_count = len(country_market['country'].unique()) if not country_market.empty and 'country' in country_market.columns else 0
        st.metric(
            "Countries",
            countries_count,
            "with market data",
            help="Countries available for forecasting"
        )
    
    with col3:
        indicator_count = len(global_indicators['indicator_key'].unique()) if not global_indicators.empty and 'indicator_key' in global_indicators.columns else 0
        st.metric(
            "Global Indicators",
            indicator_count,
            "available",
            help="Economic indicators available for adjustments"
        )
    
    with col4:
        extraction_params = unified_data.get('extraction_params', {})
        hist_cutoff = extraction_params.get('hist_cutoff', 'N/A')
        forecast_until = extraction_params.get('forecast_until', 'N/A')
        st.metric(
            "Time Range",
            f"{hist_cutoff}→{forecast_until}",
            "years",
            help="Historical cutoff and forecast end year"
        )


def show_method_requirements(method: str, available_methods: list):
    """Display method requirements and availability."""
    from ..forecasting.validation import get_method_requirements
    
    requirements = get_method_requirements(method)
    is_available = method in available_methods
    
    status_color = "green" if is_available else "red"
    status_text = "✅ Available" if is_available else "❌ Not Available"
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown(f"**Status:** :{status_color}[{status_text}]")
    
    with col2:
        if requirements:
            description = requirements.get('description', 'No description available')
            st.markdown(f"**Requirements:** {description}")
        
        if not is_available:
            st.info("💡 Complete Data Extraction and Configuration to enable this method")


def show_configuration_summary(config: Dict):
    """Display configuration summary."""
    if not config:
        st.warning("No configuration found")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Basic Settings:**")
        st.write(f"• Market: {config.get('market_name', 'Not set')}")
        forecast_mode = config.get('forecast_mode', ForecastMode.CLASSIC.value)
        if forecast_mode == ForecastMode.EXISTING_FORECAST_NEWS.value:
            baseline_label = "Existing forecast (news-only)"
        else:
            baseline_label = config.get('forecast_method', 'Not set')
        st.write(f"• Baseline Method: {baseline_label}")
        st.write(f"• Approaches: {', '.join(config.get('forecast_approach', []))}")
    
    with col2:
        st.markdown("**Adjustments:**")
        indicators_enabled = config.get('use_indicators', False)
        news_enabled = config.get('use_news', False)
        
        indicator_text = f"✅ Enabled ({len(config.get('indicator_weights', {}))} weights)" if indicators_enabled else "❌ Disabled"
        news_text = f"✅ Enabled ({len(config.get('topics', []))} topics)" if news_enabled else "❌ Disabled"
        
        st.write(f"• Indicators: {indicator_text}")
        st.write(f"• News Analysis: {news_text}")


def show_news_impact_metrics(analyzed_news: pd.DataFrame):
    """Display news impact metrics with recency weighting information."""
    if analyzed_news.empty:
        return
    
    # Filter relevant news
    relevant_news = analyzed_news[analyzed_news.get('relevant', 0) == 1]
    if relevant_news.empty:
        st.info("No relevant news articles found")
        return
    
    st.markdown("### 📊 News Impact Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_articles = len(relevant_news)
        avg_growth = relevant_news['growth_rate'].mean()
        st.metric(
            "Relevant Articles",
            f"{total_articles}",
            f"avg: {avg_growth:+.1f}%",
            help="Articles with direct market impact"
        )
    
    with col2:
        positive_news = relevant_news[relevant_news['growth_rate'] > 0]
        pos_count = len(positive_news)
        pos_avg = positive_news['growth_rate'].mean() if not positive_news.empty else 0
        st.metric(
            "Positive Impact",
            f"{pos_count} articles",
            f"avg: {pos_avg:+.1f}%",
            help="Articles indicating market growth"
        )
    
    with col3:
        negative_news = relevant_news[relevant_news['growth_rate'] < 0]
        neg_count = len(negative_news)
        neg_avg = negative_news['growth_rate'].mean() if not negative_news.empty else 0
        st.metric(
            "Negative Impact",
            f"{neg_count} articles",
            f"avg: {neg_avg:+.1f}%",
            help="Articles indicating market decline"
        )


def show_news_analysis_details(result: Dict, method: str = None):
    """Display detailed news analysis from forecast results."""
    
    if 'news' not in result.get('adjustments_applied', []) and 'country_news' not in result.get('adjustments_applied', []):
        return
    
    # Get the analyzed news from session state for detailed analysis
    analyzed_news = st.session_state.get('analyzed_news', {})
    
    if not analyzed_news:
        return
    
    with st.expander("🔍 Detailed News Analysis", expanded=False):
        news_type = analyzed_news.get('type', 'unknown')
        
        
        # Show category distribution if available
        st.divider()
        st.subheader("Category Distribution")
        
        category_counts = {}
        
        # Aggregate categories from all news
        if news_type == 'combined_news':
            global_data = analyzed_news.get('global_data')
            if isinstance(global_data, pd.DataFrame) and 'category' in global_data.columns:
                for cat, count in global_data['category'].value_counts().to_dict().items():
                    category_counts[cat] = category_counts.get(cat, 0) + count
            
            country_data = analyzed_news.get('country_data', {})
            for df in country_data.values():
                if isinstance(df, pd.DataFrame) and 'category' in df.columns:
                    for cat, count in df['category'].value_counts().to_dict().items():
                        category_counts[cat] = category_counts.get(cat, 0) + count
                        
        elif news_type == 'global_news':
            data = analyzed_news.get('data')
            if isinstance(data, pd.DataFrame) and 'category' in data.columns:
                category_counts = data['category'].value_counts().to_dict()
                
        elif news_type == 'country_news':
            data = analyzed_news.get('data', {})
            for df in data.values():
                if isinstance(df, pd.DataFrame) and 'category' in df.columns:
                    for cat, count in df['category'].value_counts().to_dict().items():
                        category_counts[cat] = category_counts.get(cat, 0) + count
        
        if category_counts:
            # Sort by count
            sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Display as metrics
            cols = st.columns(min(4, len(sorted_cats)))
            for i, (cat, count) in enumerate(sorted_cats[:4]):
                cols[i].metric(cat, count)
    
    # Show country-specific news for bottom_up, country_specific, and top_down methods
    if method in ['bottom_up', 'country_specific', 'top_down'] and country_news:
        for country, country_data in country_news.items():
            if isinstance(country_data, pd.DataFrame) and not country_data.empty:
                with st.expander(f"🏳️ {country} News Analysis ({len(country_data)} articles)", expanded=False):
                    display_df = country_data[['title', 'category', 'temporal_impact', 'growth_rate', 'reason']].copy()
                    display_df.columns = ['Headline', 'Category', 'Temporal Impact', 'Growth Rate (%)', 'Reason']
                    display_df['Growth Rate (%)'] = display_df['Growth Rate (%)'].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "0.0")
                    st.dataframe(display_df, use_container_width=True, height=400)


def show_validation_results(validation: Dict):
    """Display validation results with clear status."""
    if validation.get('valid'):
        st.success("✅ Validation passed - ready to proceed")
    else:
        st.error(f"❌ Validation failed: {validation.get('message', 'Unknown error')}")
        return
    
    # Show warnings if any
    warnings = validation.get('warnings', [])
    if warnings:
        st.warning("⚠️ Warnings:")
        for warning in warnings:
            st.write(f"• {warning}")
    
    # Show additional info if available
    if 'countries_with_data' in validation:
        countries = validation['countries_with_data']
        if countries:
            st.info(f"✅ Countries with sufficient data: {len(countries)} ({', '.join(countries[:5])}{'...' if len(countries) > 5 else ''})")