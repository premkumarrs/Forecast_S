"""
Helper functions for Insights page.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional

from ..components.charts import (
    create_global_forecast_chart,
    create_regional_composition_chart,
    create_regional_forecast_chart,
    create_country_forecast_chart,
    create_yoy_growth_chart,
    create_multi_country_chart,
    create_cagr_comparison_chart,
    create_final_values_chart
)

from ..components.metrics import (
    calculate_global_metrics,
    calculate_regional_metrics,
    calculate_country_metrics
)

from ..components.layouts import (
    get_country_flag_emoji,
    render_country_pills,
    create_metric_card
)

from ..components.utils import (
    safe_get_value_column,
    safe_get_nested,
    get_hist_cutoff,
    get_forecast_until,
    get_available_countries,
    get_available_regions
)


def render_global_insights(forecast_result: Dict):
    """Render the global insights view."""
    st.markdown("### 🌍 Global Market Forecast")
    
    global_forecast = forecast_result.get('global_forecast', pd.DataFrame())
    
    if global_forecast.empty:
        st.warning("No global forecast data available")
        return
    
    # Display metrics
    metrics = calculate_global_metrics(global_forecast)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card(
            "Historical CAGR",
            f"{metrics.get('hist_cagr', 0):.1f}%"
        )
    
    with col2:
        create_metric_card(
            "Forecast CAGR",
            f"{metrics.get('fcst_cagr', 0):.1f}%"
        )
    
    with col3:
        create_metric_card(
            f"Final Value ({get_forecast_until()})",
            f"${metrics.get('final_value', 0):.1f}B"
        )
    
    with col4:
        create_metric_card(
            "Total Growth",
            f"{metrics.get('total_growth', 0):.1f}%"
        )
    
    # Display chart
    fig = create_global_forecast_chart(global_forecast, forecast_result)
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional composition if available
    region_forecasts = forecast_result.get('region_forecasts', {})
    if region_forecasts:
        st.markdown("#### Regional Market Composition")
        fig = create_regional_composition_chart(region_forecasts)
        st.plotly_chart(fig, use_container_width=True)


def render_regional_insights(forecast_result: Dict):
    """Render the regional insights view."""
    st.markdown("### 🗺️ Regional Forecast Analysis")
    
    region_forecasts = forecast_result.get('region_forecasts', {})
    
    if not region_forecasts:
        st.warning("No regional forecast data available")
        return
    
    regions = get_available_regions(forecast_result)
    
    if not regions:
        st.warning("No regions found in forecast data")
        return
    
    selected_region = st.selectbox(
        "Select Region",
        regions,
        format_func=lambda x: f"🗺️ {x}",
        key="region_selector"
    )
    
    if selected_region and selected_region in region_forecasts:
        region_data = region_forecasts[selected_region]
        
        # Display metrics
        metrics = calculate_regional_metrics(region_data)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_metric_card(
                "Historical CAGR",
                f"{metrics.get('hist_cagr', 0):.1f}%"
            )
        
        with col2:
            create_metric_card(
                "Forecast CAGR",
                f"{metrics.get('fcst_cagr', 0):.1f}%"
            )
        
        with col3:
            create_metric_card(
                f"Final Value ({get_forecast_until()})",
                f"${metrics.get('final_value', 0):.1f}B"
            )
        
        # Display chart
        fig = create_regional_forecast_chart(region_data, selected_region)
        st.plotly_chart(fig, use_container_width=True)


def render_country_insights(forecast_result: Dict):
    """Render the country insights view."""
    st.markdown("### 🏳️ Country-Level Forecast Analysis")
    
    countries = get_available_countries(forecast_result)
    
    if not countries:
        st.warning("No country data available")
        return
    
    # Country selector
    selected_country = st.selectbox(
        "Select Country",
        countries,
        format_func=lambda x: f"{get_country_flag_emoji(x)} {x}",
        key="country_selector"
    )
    
    if selected_country:
        # Render detailed analysis for selected country
        render_forecast_analysis(selected_country, forecast_result)
        render_yoy_growth_analysis(selected_country, forecast_result)
        render_news_analysis(selected_country)


def render_country_comparison(forecast_result: Dict, primary_country: str):
    """Render country comparison view."""
    st.markdown("### 📊 Multi-Country Comparison")
    
    countries = get_available_countries(forecast_result)
    
    # Select countries to compare
    selected_countries = st.multiselect(
        "Select countries to compare",
        [c for c in countries if c != primary_country],
        default=[countries[1]] if len(countries) > 1 else [],
        format_func=lambda x: f"{get_country_flag_emoji(x)} {x}"
    )
    
    # Include primary country
    comparison_countries = [primary_country] + selected_countries
    
    if len(comparison_countries) > 1:
        # Create comparison charts
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Forecast Comparison",
            "📊 CAGR Analysis", 
            "🎯 Final Values",
            "📋 Metrics Table"
        ])
        
        with tab1:
            fig = create_multi_country_chart(comparison_countries, forecast_result)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = create_cagr_comparison_chart(comparison_countries, forecast_result)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = create_final_values_chart(comparison_countries, forecast_result)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            from ..components.metrics import create_metrics_comparison_table
            df = create_metrics_comparison_table(comparison_countries, forecast_result)
            if not df.empty:
                st.dataframe(df, use_container_width=True, hide_index=True)


def render_news_analysis(country: str):
    """Render news analysis for a country."""
    news_data = get_country_news_data(country)
    
    if news_data.empty:
        st.info(f"No news data available for {country}")
        return
    
    st.markdown(f"#### 📰 News Impact Analysis - {country}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Articles", len(news_data))
    
    with col2:
        relevant = news_data['relevant'].sum() if 'relevant' in news_data.columns else 0
        st.metric("Relevant Articles", relevant)
    
    with col3:
        avg_impact = news_data['growth_rate'].mean() if 'growth_rate' in news_data.columns else 0
        st.metric("Avg Growth Impact", f"{avg_impact:.2f}%")
    
    # Show sample headlines
    if st.checkbox("Show sample headlines", key=f"news_{country}"):
        sample = news_data.head(10)[['title', 'category', 'growth_rate']]
        st.dataframe(sample, use_container_width=True, hide_index=True)


def render_forecast_analysis(country: str, forecast_result: Dict):
    """Render forecast analysis for a country."""
    combined_df = prepare_country_timeline(country, forecast_result)
    
    if combined_df.empty:
        st.warning(f"No forecast data available for {country}")
        return
    
    st.markdown(f"#### 📈 Forecast Analysis - {country}")
    
    # Display metrics
    metrics = calculate_country_metrics(combined_df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_metric_card(
            "Historical CAGR",
            f"{metrics.get('hist_cagr', 0):.1f}%"
        )
    
    with col2:
        create_metric_card(
            "Forecast CAGR",
            f"{metrics.get('fcst_cagr', 0):.1f}%"
        )
    
    with col3:
        create_metric_card(
            f"Final Value ({get_forecast_until()})",
            f"${metrics.get('final_value', 0):.1f}B"
        )
    
    # Display chart
    fig = create_country_forecast_chart(combined_df, country, forecast_result)
    st.plotly_chart(fig, use_container_width=True)


def render_yoy_growth_analysis(country: str, forecast_result: Dict):
    """Render year-over-year growth analysis."""
    combined_df = prepare_country_timeline(country, forecast_result)
    
    if combined_df.empty:
        return
    
    st.markdown(f"#### 📊 Year-over-Year Growth - {country}")
    
    fig = create_yoy_growth_chart(combined_df, country)
    st.plotly_chart(fig, use_container_width=True)


def create_adjustment_breakdown_table(country: str, forecast_result: Dict) -> pd.DataFrame:
    """Create adjustment breakdown table for a country."""
    adjustment_details = forecast_result.get('adjustment_details', {})
    
    if country not in adjustment_details:
        return pd.DataFrame()
    
    details = adjustment_details[country]
    
    if not isinstance(details, dict):
        return pd.DataFrame()
    
    breakdown_data = []
    
    # News adjustment
    if 'news_contribution' in details:
        breakdown_data.append({
            'Adjustment Type': 'News Impact',
            'Contribution (%)': f"{details['news_contribution']:.2f}",
            'Status': 'Applied' if details['news_contribution'] != 0 else 'Not Applied'
        })
    
    # Indicator adjustment
    if 'indicator_contribution' in details:
        breakdown_data.append({
            'Adjustment Type': 'Indicator Impact',
            'Contribution (%)': f"{details['indicator_contribution']:.2f}",
            'Status': 'Applied' if details['indicator_contribution'] != 0 else 'Not Applied'
        })
    
    # Temporal decay
    if 'temporal_decay_factor' in details:
        breakdown_data.append({
            'Adjustment Type': 'Temporal Decay',
            'Contribution (%)': f"{(1 - details['temporal_decay_factor']) * 100:.2f}",
            'Status': 'Applied'
        })
    
    # Fallback info
    if details.get('fallback_used'):
        breakdown_data.append({
            'Adjustment Type': 'Fallback Strategy',
            'Contribution (%)': details.get('fallback_reason', 'N/A'),
            'Status': 'Used'
        })
    
    return pd.DataFrame(breakdown_data)


def prepare_country_timeline(country: str, forecast_result: Dict) -> pd.DataFrame:
    """Prepare combined timeline for a country."""
    # Get unified data from session
    unified_data = st.session_state.get('unified_data', {})
    market_data = unified_data.get('market_value', {})
    country_market = market_data.get('country', pd.DataFrame())
    
    # Get forecast data
    country_forecasts = forecast_result.get('country_forecasts', {})
    
    if country not in country_forecasts:
        return pd.DataFrame()
    
    # Get country forecast
    if isinstance(country_forecasts[country], dict):
        forecast_df = country_forecasts[country].get('forecast', pd.DataFrame())
    else:
        forecast_df = country_forecasts[country]
    
    # Get historical data
    hist_df = pd.DataFrame()
    if not country_market.empty and 'country' in country_market.columns:
        hist_df = country_market[country_market['country'] == country].copy()
    
    # Combine historical and forecast
    if not hist_df.empty and not forecast_df.empty:
        hist_cutoff = get_hist_cutoff()
        hist_df = hist_df[hist_df['year'] <= hist_cutoff]
        forecast_df = forecast_df[forecast_df['year'] > hist_cutoff]
        
        # Ensure consistent columns
        if 'value' in hist_df.columns:
            hist_df['value_hat'] = hist_df['value']
        hist_df['type'] = 'historical'
        
        if 'type' not in forecast_df.columns:
            forecast_df['type'] = 'forecast'
        
        combined_df = pd.concat([hist_df, forecast_df], ignore_index=True)
        combined_df = combined_df.sort_values('year')
        
        return combined_df
    elif not forecast_df.empty:
        return forecast_df
    else:
        return hist_df


def get_country_news_data(country: str) -> pd.DataFrame:
    """Get news data for a specific country."""
    country_news = st.session_state.get('country_news', {})
    
    if isinstance(country_news, dict) and country in country_news:
        return country_news[country]
    
    return pd.DataFrame()