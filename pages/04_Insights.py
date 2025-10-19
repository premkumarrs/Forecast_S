"""
Multi-Country Insights Page - Hierarchical view with Global, Regional, and Country insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from pages.components.utils import format_value_intelligent, detect_data_unit
from ui.streamlit.components.calibration_card import render_calibration_cards, render_calibration_metrics

# Configure page
st.set_page_config(
    page_title="Insights - AI Market Forecaster",
    page_icon="💡",
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

# === UTILITY FUNCTIONS ===
def safe_get_value_column(df):
    """Safely get the value column from a DataFrame."""
    if df is None or df.empty:
        return None
    
    # Priority: value > value_hat
    if 'value' in df.columns:
        return 'value'
    elif 'value_hat' in df.columns:
        return 'value_hat'
    
    # Try common variations
    for col in df.columns:
        if 'value' in col.lower():
            return col
    
    return None

def safe_get_nested(data, *keys, default=None):
    """Safely get nested dictionary values."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
            if data is None:
                return default
        else:
            return default
    return data

def get_country_flag_emoji(country: str) -> str:
    """Get flag emoji for country (simplified mapping)."""
    flag_map = {
        'United States': '🇺🇸', 'USA': '🇺🇸', 'US': '🇺🇸',
        'United Kingdom': '🇬🇧', 'UK': '🇬🇧', 'Britain': '🇬🇧',
        'Germany': '🇩🇪', 'China': '🇨🇳', 'Japan': '🇯🇵', 
        'India': '🇮🇳', 'France': '🇫🇷', 'Italy': '🇮🇹',
        'Canada': '🇨🇦', 'Australia': '🇦🇺', 'Brazil': '🇧🇷',
        'Mexico': '🇲🇽', 'South Korea': '🇰🇷', 'Spain': '🇪🇸',
        'Netherlands': '🇳🇱', 'Switzerland': '🇨🇭', 'Sweden': '🇸🇪',
        'Singapore': '🇸🇬', 'South Africa': '🇿🇦', 'Russia': '🇷🇺',
        'Ukraine': '🇺🇦', 'Argentina': '🇦🇷', 'Belgium': '🇧🇪',
        'Austria': '🇦🇹', 'Poland': '🇵🇱', 'Turkey': '🇹🇷',
        'Saudi Arabia': '🇸🇦', 'UAE': '🇦🇪', 'Israel': '🇮🇱',
        'Egypt': '🇪🇬', 'Nigeria': '🇳🇬', 'Kenya': '🇰🇪',
        'Indonesia': '🇮🇩', 'Thailand': '🇹🇭', 'Vietnam': '🇻🇳',
        'Philippines': '🇵🇭', 'Malaysia': '🇲🇾', 'New Zealand': '🇳🇿'
    }
    return flag_map.get(country, '🌐')

def get_hist_cutoff():
    """Get the historical cutoff year consistently."""
    return st.session_state.get('saved_hist_cutoff', st.session_state.get('hist_cutoff', 2024))

def get_forecast_until():
    """Get the forecast until year consistently."""
    # Try unified_data first for most reliable source
    unified_data = st.session_state.get('unified_data', {})
    extraction_params = unified_data.get('extraction_params', {})
    if 'forecast_until' in extraction_params:
        return extraction_params['forecast_until']
    # Fallback to session state
    return st.session_state.get('saved_forecast_until', st.session_state.get('forecast_until', 2030))

def get_available_countries(forecast_result: Dict) -> List[str]:
    """Get all available countries from forecast result."""
    country_forecasts = forecast_result.get('country_forecasts', {})
    if not country_forecasts:
        return []
    return sorted(list(country_forecasts.keys()))

def get_available_regions(forecast_result: Dict) -> List[str]:
    """Get all available regions from forecast result."""
    region_forecasts = forecast_result.get('region_forecasts', {})
    if not region_forecasts:
        return []
    # Sort with Worldwide first if it exists
    regions = list(region_forecasts.keys())
    if 'Worldwide' in regions:
        regions.remove('Worldwide')
        return ['Worldwide'] + sorted(regions)
    return sorted(regions)

# === GLOBAL VIEW FUNCTIONS ===
def render_global_insights(forecast_result: Dict):
    """Render the Global/Worldwide insights view."""
    st.markdown("### 🌍 Global Forecast Overview")

    # Show only GLOBAL calibration here; country-level is shown in Country view
    cal = (st.session_state.get('analyzed_news') or {}).get('calibration') if 'analyzed_news' in st.session_state else None
    if cal:
        render_calibration_cards(cal, show_global=True, show_countries=False)
        
        # Show MA category breakdown if available
        global_cal = cal.get('global', {})
        category_breakdown = global_cal.get('category_breakdown')
        
        if category_breakdown:
            with st.expander("📊 Category Impact Breakdown", expanded=False):
                # Show active categories
                active_categories = [(name, info) for name, info in category_breakdown.items() 
                                   if info.get('status') == 'Active']
                
                if active_categories:
                    cols = st.columns(2)
                    for i, (name, info) in enumerate(active_categories):
                        with cols[i % 2]:
                            window = info.get('window', 0)
                            days_data = info.get('days_with_data', 0)
                            score = info.get('ma_score', 0)
                            coverage = info.get('coverage_pct', 0)
                            
                            # Color indicator based on coverage
                            if coverage >= 75:
                                indicator = "🟢"
                            elif coverage >= 30:
                                indicator = "🟡"
                            else:
                                indicator = "🔴"
                            
                            st.metric(
                                f"{indicator} {name}",
                                f"{score:+.2f}%",
                                f"{days_data}/{window} days ({coverage:.0f}%)"
                            )
                    
                    # Show final impact
                    avg_impact = sum(info.get('ma_score', 0) for _, info in active_categories) / len(active_categories)
                    st.success(f"**Final Global News Impact: {avg_impact:+.2f}%**")
                else:
                    st.info("No active categories with data")
    
    # Get global forecast data
    global_forecast = forecast_result.get('global_forecast')
    
    if global_forecast is None or global_forecast.empty:
        st.warning("No global forecast data available")
        return
    
    # Create two columns layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Create global forecast chart
        fig = create_global_forecast_chart(global_forecast, forecast_result)
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and display metrics
        metrics = calculate_global_metrics(global_forecast)
        
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("Historical CAGR", f"{metrics['hist_cagr']:.1f}%")
        with metric_cols[1]:
            st.metric("Forecast CAGR", f"{metrics['fcst_cagr']:.1f}%")
        with metric_cols[2]:
            st.metric(f"{get_forecast_until()} Value", format_value_intelligent(metrics['final_value']))
    
    with col2:
        # Show adjustment summary
        st.markdown("#### 📊 Forecast Method Summary")
        
        method = forecast_result.get('method', 'Unknown')
        adjustments = forecast_result.get('adjustments_applied', [])
        metadata = forecast_result.get('metadata', {})
        
        st.info(f"**Method:** {method.replace('_', ' ').title()}")
        
        if adjustments:
            st.success(f"**Adjustments Applied:** {', '.join(adjustments)}")
        else:
            st.info("**No adjustments applied** (baseline forecast)")
        
        # Show metadata statistics
        if metadata:
            st.markdown("#### 📈 Coverage Statistics")
            
            if 'countries_forecasted' in metadata:
                st.metric("Countries Forecasted", metadata['countries_forecasted'])
            
            if 'regions_aggregated' in metadata:
                st.metric("Regions Aggregated", metadata['regions_aggregated'])
            
            if 'countries_with_fallback' in metadata:
                fallback_count = metadata['countries_with_fallback']
                if fallback_count > 0:
                    st.warning(f"Countries using fallback data: {fallback_count}")

        # Calibration metrics (global)
        cal = (st.session_state.get('analyzed_news') or {}).get('calibration') or {}
        gl = cal.get('global') or {}
        if gl:
            render_calibration_metrics(gl, show_reason=True, note_title="LLM Calibration")
    
    # Add global adjustment breakdown if available
    if method == 'global_only':
        st.markdown("---")
        st.markdown("### 📊 Global Adjustment Breakdown")
        
        global_breakdown_df = create_global_adjustment_breakdown_table(forecast_result)
        if not global_breakdown_df.empty:
            st.dataframe(global_breakdown_df, use_container_width=True, hide_index=True, height=300)
            
            # Add subtle caption
            st.caption("Values shown with news and indicator adjustments applied")
            
    
    # Show regional composition if available
    if 'region_forecasts' in forecast_result:
        st.markdown("---")
        st.markdown("### 🗺️ Regional Composition")
        
        region_forecasts = forecast_result['region_forecasts']
        
        # Create regional breakdown chart
        fig = create_regional_composition_chart(region_forecasts)
        st.plotly_chart(fig, use_container_width=True)

def create_global_forecast_chart(global_forecast: pd.DataFrame, forecast_result: Dict) -> go.Figure:
    """Create the global forecast chart."""
    hist_cutoff = get_hist_cutoff()
    
    # Separate historical and forecast data
    hist_data = global_forecast[global_forecast['year'] <= hist_cutoff]
    fcst_data = global_forecast[global_forecast['year'] > hist_cutoff]
    
    # Get baseline forecast if available
    baseline_forecast = forecast_result.get('baseline', pd.DataFrame())
    if baseline_forecast.empty:
        baseline_forecast = forecast_result.get('baseline_global_forecast', pd.DataFrame())
    
    fig = go.Figure()
    
    # Helper to choose correct value columns explicitly
    def pick_hist_values(df: pd.DataFrame):
        col = 'value' if 'value' in df.columns else ('value_hat' if 'value_hat' in df.columns else None)
        return col, pd.to_numeric(df[col], errors='coerce') if col else (None, pd.Series(dtype=float))
    def pick_fore_values(df: pd.DataFrame):
        col = 'value_hat' if 'value_hat' in df.columns else ('value' if 'value' in df.columns else None)
        return col, pd.to_numeric(df[col], errors='coerce') if col else (None, pd.Series(dtype=float))

    # Pre-compute values for consistent scaling across all series
    col_h, values_h = pick_hist_values(hist_data) if not hist_data.empty else (None, pd.Series(dtype=float))
    col_f, values_f = pick_fore_values(fcst_data) if not fcst_data.empty else (None, pd.Series(dtype=float))
    baseline_fcst = baseline_forecast[baseline_forecast['year'] > hist_cutoff] if not baseline_forecast.empty else pd.DataFrame()
    col_b, values_b = pick_fore_values(baseline_fcst) if not baseline_fcst.empty else (None, pd.Series(dtype=float))

    all_vals = pd.concat([values_h, values_f, values_b], ignore_index=True) if not (values_h.empty and values_f.empty and values_b.empty) else pd.Series([0.0])
    scale_to_billion = (all_vals.max() if not all_vals.empty else 0) > 1e6
    def _scale(arr):
        # Accept list/ndarray/Series; always return a plain list for Plotly
        ser = arr if isinstance(arr, pd.Series) else pd.Series(arr)
        ser = pd.to_numeric(ser, errors='coerce')
        return ((ser / 1e9).tolist() if scale_to_billion else ser.tolist())

    # Historical line
    if col_h:
        fig.add_trace(go.Scatter(
            x=hist_data['year'],
            y=_scale(values_h),
            mode='lines+markers',
            name='Historical',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=6)
        ))
    
    # Baseline forecast line (if available and different from adjusted)
    if not baseline_forecast.empty:
        baseline_fcst = baseline_forecast[baseline_forecast['year'] > hist_cutoff]
        if not baseline_fcst.empty and col_b:
            # Connect last historical → first baseline forecast
            if col_h:
                lh_year = int(hist_data['year'].max())
                lh_val = float(hist_data.iloc[-1][col_h])
                fb_year = int(baseline_fcst['year'].min())
                fb_val = float(baseline_fcst[baseline_fcst['year'] == fb_year][col_b].iloc[0])
                fig.add_trace(go.Scatter(
                    x=[lh_year, fb_year],
                    y=_scale([lh_val, fb_val]),
                    mode='lines',
                    line=dict(color='#808080', width=2, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            fig.add_trace(go.Scatter(
                x=baseline_fcst['year'],
                y=_scale(values_b),
                mode='lines',
                name='Baseline Forecast',
                line=dict(color='#808080', width=2, dash='dot'),
                marker=dict(size=4),
                opacity=0.7
            ))
    
    # Adjusted forecast line
    if not fcst_data.empty and col_f:
        # Connect last historical → first adjusted forecast
        if col_h:
            lh_year = int(hist_data['year'].max())
            lh_val = float(hist_data.iloc[-1][col_h])
            fa_year = int(fcst_data['year'].min())
            fa_val = float(fcst_data[fcst_data['year'] == fa_year][col_f].iloc[0])
            fig.add_trace(go.Scatter(
                x=[lh_year, fa_year],
                y=_scale([lh_val, fa_val]),
                mode='lines',
                line=dict(color='#D4463D', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        # Check if there are adjustments to show different label
        adjustments = forecast_result.get('adjustments_applied', [])
        forecast_name = 'Adjusted Forecast' if adjustments else 'Forecast'
        fig.add_trace(go.Scatter(
            x=fcst_data['year'],
            y=_scale(values_f),
            mode='lines+markers',
            name=forecast_name,
            line=dict(color='#D4463D', width=3),
            marker=dict(size=6)
        ))
    
    # Add vertical line at cutoff
    fig.add_vline(x=hist_cutoff + 0.5, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Global Market Forecast",
        xaxis_title="Year",
        yaxis_title="Market Value ($B)" if scale_to_billion else "Market Value",
        template="plotly_white",
        height=400,
        showlegend=True
    )
    
    return fig

def calculate_global_metrics(global_forecast: pd.DataFrame) -> Dict:
    """Calculate key metrics for global forecast."""
    hist_cutoff = get_hist_cutoff()
    
    hist_data = global_forecast[global_forecast['year'] <= hist_cutoff]
    fcst_data = global_forecast[global_forecast['year'] > hist_cutoff]
    
    metrics = {}
    
    # Historical CAGR
    if len(hist_data) >= 2:
        value_col = safe_get_value_column(hist_data)
        if value_col:
            years = hist_data.iloc[-1]['year'] - hist_data.iloc[0]['year']
            if years > 0:
                start_val = hist_data.iloc[0][value_col]
                end_val = hist_data.iloc[-1][value_col]
                if start_val > 0:
                    metrics['hist_cagr'] = ((end_val / start_val) ** (1/years) - 1) * 100
                else:
                    metrics['hist_cagr'] = 0
            else:
                metrics['hist_cagr'] = 0
    else:
        metrics['hist_cagr'] = 0
    
    # Forecast CAGR
    if len(fcst_data) >= 2:
        value_col = safe_get_value_column(fcst_data)
        if value_col:
            years = fcst_data.iloc[-1]['year'] - fcst_data.iloc[0]['year']
            if years > 0:
                start_val = fcst_data.iloc[0][value_col]
                end_val = fcst_data.iloc[-1][value_col]
                if start_val > 0:
                    metrics['fcst_cagr'] = ((end_val / start_val) ** (1/years) - 1) * 100
                else:
                    metrics['fcst_cagr'] = 0
            else:
                metrics['fcst_cagr'] = 0
    else:
        metrics['fcst_cagr'] = 0
    
    # Final value
    if not fcst_data.empty:
        value_col = safe_get_value_column(fcst_data)
        if value_col:
            final_val = fcst_data.iloc[-1][value_col]
            metrics['final_value'] = final_val  # Keep original value
    else:
        metrics['final_value'] = 0
    
    return metrics

def create_regional_composition_chart(region_forecasts: Dict[str, pd.DataFrame]) -> go.Figure:
    """Create a chart showing regional composition."""
    hist_cutoff = get_hist_cutoff()
    
    # Get final year values for each region
    regional_values = {}
    
    for region, forecast_df in region_forecasts.items():
        if region == 'Worldwide':
            continue  # Skip worldwide as it's the total
        
        fcst_data = forecast_df[forecast_df['year'] > hist_cutoff]
        if not fcst_data.empty:
            value_col = safe_get_value_column(fcst_data)
            if value_col:
                final_val = fcst_data.iloc[-1][value_col]
                final_val = final_val / 1e9 if final_val > 1e6 else final_val
                regional_values[region] = final_val
    
    if not regional_values:
        fig = go.Figure()
        fig.add_annotation(text="No regional data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Sort by value
    sorted_regions = sorted(regional_values.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 10 regions for clarity
    if len(sorted_regions) > 10:
        top_regions = dict(sorted_regions[:10])
        other_value = sum([v for k, v in sorted_regions[10:]])
        top_regions['Others'] = other_value
    else:
        top_regions = dict(sorted_regions)
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=list(top_regions.keys()),
        values=list(top_regions.values()),
        hole=0.3
    )])
    
    fig.update_layout(
        title=f"Regional Market Share ({get_forecast_until()})",
        height=400,
        showlegend=True
    )
    
    return fig

# === REGIONAL VIEW FUNCTIONS ===
def render_regional_insights(forecast_result: Dict):
    """Render the Regional insights view."""
    st.markdown("### 🗺️ Regional Forecast Analysis")
    
    region_forecasts = forecast_result.get('region_forecasts', {})
    
    if not region_forecasts:
        st.warning("No regional forecast data available for this method")
        return
    
    # Region selector
    regions = get_available_regions(forecast_result)
    
    if not regions:
        st.warning("No regions found in forecast data")
        return
    
    # Use selectbox for region selection
    selected_region = st.selectbox(
        "Select Region",
        regions,
        format_func=lambda x: f"🗺️ {x}",
        key="region_selector"
    )
    
    if selected_region not in region_forecasts:
        st.error(f"No data available for {selected_region}")
        return
    
    # Get region data
    region_data = region_forecasts[selected_region]
    
    # Create layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Create regional forecast chart
        fig = create_regional_forecast_chart(region_data, selected_region)
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and display metrics
        metrics = calculate_regional_metrics(region_data)
        
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("Historical CAGR", f"{metrics['hist_cagr']:.1f}%")
        with metric_cols[1]:
            st.metric("Forecast CAGR", f"{metrics['fcst_cagr']:.1f}%")
        with metric_cols[2]:
            st.metric(f"{get_forecast_until()} Value", format_value_intelligent(metrics['final_value']))
    
    with col2:
        # Show region composition
        st.markdown("#### 📍 Region Composition")
        
        # Try to identify countries in this region
        countries_in_region = get_countries_in_region(selected_region, forecast_result)
        
        if countries_in_region:
            st.info(f"**Countries:** {len(countries_in_region)}")
            
            # Show country list in expandable section
            with st.expander("View Countries", expanded=False):
                # Create columns for country display
                cols = st.columns(3)
                for i, country in enumerate(countries_in_region):
                    with cols[i % 3]:
                        st.write(f"{get_country_flag_emoji(country)} {country}")
            
            # Add country comparison chart
            if len(countries_in_region) > 1:
                st.markdown(f"#### 📊 Top Countries by {get_forecast_until()} Value")
                country_values = get_country_final_values(countries_in_region, forecast_result)
                if country_values:
                    # Show top 5 countries
                    top_countries = dict(sorted(country_values.items(), key=lambda x: x[1], reverse=True)[:5])
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(top_countries.values()),
                            y=list(top_countries.keys()),
                            orientation='h',
                            marker_color='#2E86AB'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Top Countries in Region",
                        xaxis_title="Market Value ($B)",
                        yaxis_title="",
                        height=250,
                        template="plotly_white",
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Show sub-regions if this is a large region
        if selected_region in ['Worldwide', 'Americas', 'Europe', 'Asia', 'Africa']:
            st.markdown("#### 🗂️ Sub-regions")
            sub_regions = get_sub_regions(selected_region, region_forecasts)
            if sub_regions:
                for sub in sub_regions[:5]:  # Show top 5
                    st.write(f"• {sub}")

def create_regional_forecast_chart(region_data: pd.DataFrame, region_name: str) -> go.Figure:
    """Create forecast chart for a specific region."""
    hist_cutoff = get_hist_cutoff()
    
    # Separate historical and forecast data
    hist_data = region_data[region_data['year'] <= hist_cutoff]
    fcst_data = region_data[region_data['year'] > hist_cutoff]
    
    fig = go.Figure()
    
    # Historical line
    if not hist_data.empty:
        value_col = safe_get_value_column(hist_data)
        if value_col:
            values = hist_data[value_col].values
            values = values / 1e9 if (values > 1e6).any() else values
            
            fig.add_trace(go.Scatter(
                x=hist_data['year'],
                y=values,
                mode='lines+markers',
                name='Historical',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=6)
            ))
    
    # Forecast line
    if not fcst_data.empty:
        value_col = safe_get_value_column(fcst_data)
        if value_col:
            values = fcst_data[value_col].values
            values = values / 1e9 if (values > 1e6).any() else values
            
            fig.add_trace(go.Scatter(
                x=fcst_data['year'],
                y=values,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#D4463D', width=3, dash='dash'),
                marker=dict(size=6)
            ))
    
    # Add vertical line at cutoff
    fig.add_vline(x=hist_cutoff + 0.5, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"{region_name} Market Forecast",
        xaxis_title="Year",
        yaxis_title="Market Value ($B)",
        template="plotly_white",
        height=400,
        showlegend=True
    )
    
    return fig

def calculate_regional_metrics(region_data: pd.DataFrame) -> Dict:
    """Calculate key metrics for regional forecast."""
    return calculate_global_metrics(region_data)  # Same calculation logic

def get_countries_in_region(region: str, forecast_result: Dict) -> List[str]:
    """Get list of countries in a specific region."""
    import json
    import os
    
    # Load aggregation.json to get proper mappings
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '..', 'config', 'aggregation.json')
        
        with open(config_path, 'r') as f:
            data = json.load(f)
            region_mappings = data.get("region_mappings", {})
        
        # Get countries that have forecasts
        country_forecasts = forecast_result.get('country_forecasts', {})
        available_countries = set(country_forecasts.keys())
        
        if region == 'Worldwide':
            return sorted(list(available_countries))
        
        # Get countries in this region from mapping
        if region in region_mappings:
            region_members = region_mappings[region]
            # Filter to only countries that have forecasts
            countries_in_region = []
            
            for member in region_members:
                # Check if member is a country with forecast
                if member in available_countries:
                    countries_in_region.append(member)
                # If member is a sub-region, recursively get its countries
                elif member in region_mappings:
                    sub_countries = get_countries_in_subregion(member, region_mappings, available_countries)
                    countries_in_region.extend(sub_countries)
            
            return sorted(list(set(countries_in_region)))
        
        return []
        
    except Exception as e:
        # Fallback to simple approach
        country_forecasts = forecast_result.get('country_forecasts', {})
        if region == 'Worldwide':
            return sorted(list(country_forecasts.keys()))
        return []

def get_countries_in_subregion(subregion: str, region_mappings: Dict, available_countries: set) -> List[str]:
    """Recursively get countries in a subregion."""
    countries = []
    if subregion in region_mappings:
        for member in region_mappings[subregion]:
            if member in available_countries:
                countries.append(member)
            elif member in region_mappings:
                # Recursive call for nested regions
                sub_countries = get_countries_in_subregion(member, region_mappings, available_countries)
                countries.extend(sub_countries)
    return countries

def get_country_final_values(countries: List[str], forecast_result: Dict) -> Dict[str, float]:
    """Get final forecast values for a list of countries."""
    country_forecasts = forecast_result.get('country_forecasts', {})
    hist_cutoff = get_hist_cutoff()
    
    country_values = {}
    for country in countries:
        if country in country_forecasts:
            country_data = country_forecasts[country]
            
            if isinstance(country_data, dict):
                forecast_df = country_data.get('forecast', pd.DataFrame())
            else:
                forecast_df = country_data if isinstance(country_data, pd.DataFrame) else pd.DataFrame()
            
            if not forecast_df.empty:
                fcst_data = forecast_df[forecast_df['year'] > hist_cutoff]
                if not fcst_data.empty:
                    value_col = safe_get_value_column(fcst_data)
                    if value_col:
                        final_val = fcst_data.iloc[-1][value_col]
                        final_val = final_val / 1e9 if final_val > 1e6 else final_val
                        country_values[country] = final_val
    
    return country_values

def get_sub_regions(parent_region: str, region_forecasts: Dict) -> List[str]:
    """Get sub-regions of a parent region."""
    # Simple heuristic - look for regions that might be sub-regions
    sub_regions = []
    
    region_mapping = {
        'Worldwide': ['Americas', 'Europe', 'Asia', 'Africa', 'Australia & Oceania'],
        'Americas': ['North America', 'South America', 'Central America', 'Caribbean'],
        'Europe': ['Northern Europe', 'Southern Europe', 'Eastern Europe', 'Central & Western Europe'],
        'Asia': ['Eastern Asia', 'Southeast Asia', 'Southern Asia', 'Western Asia', 'Central Asia'],
        'Africa': ['Northern Africa', 'Southern Africa', 'Eastern Africa', 'Western Africa', 'Central Africa']
    }
    
    if parent_region in region_mapping:
        for sub in region_mapping[parent_region]:
            if sub in region_forecasts:
                sub_regions.append(sub)
    
    return sub_regions

# === COUNTRY VIEW FUNCTIONS ===
def render_country_insights(forecast_result: Dict):
    """Render the Country insights view with smart selector."""
    st.markdown("### 📍 Country-Level Analysis")
    
    countries = get_available_countries(forecast_result)
    
    if not countries:
        st.error("No country-level data available")
        return
    
    # Initialize selected country in session state
    if 'insights_selected_country' not in st.session_state:
        st.session_state['insights_selected_country'] = countries[0]
    
    # Add comparison mode toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        # Smart country selector based on count
        if len(countries) > 10:
            # Use searchable dropdown for many countries
            selected_country = st.selectbox(
                "🔍 Select Country",
                countries,
                index=countries.index(st.session_state['insights_selected_country']) if st.session_state['insights_selected_country'] in countries else 0,
                format_func=lambda x: f"{get_country_flag_emoji(x)} {x}",
                key="country_dropdown",
                help="Type to search for a country"
            )
        else:
            # Use pills for few countries (typically Country-Specific method)
            selected_country = render_country_pills(countries, st.session_state['insights_selected_country'])
    
    with col2:
        # Add comparison mode button
        if len(countries) > 1:
            compare_mode = st.checkbox("Compare Countries", key="compare_mode")
        else:
            compare_mode = False
    
    # Update session state
    st.session_state['insights_selected_country'] = selected_country
    
    if compare_mode:
        # Multi-country comparison mode
        render_country_comparison(forecast_result, selected_country)
    else:
        # Single country detailed view
        st.divider()
        
        # Render news analysis first
        render_news_analysis(selected_country)
        
        st.divider()
        
        # Create tabs for different analyses
        tab1, tab2 = st.tabs(["📈 Forecast Analysis", "📊 YoY Growth"])
        
        with tab1:
            render_forecast_analysis(selected_country, forecast_result)
        
        with tab2:
            render_yoy_growth_analysis(selected_country, forecast_result)

def render_country_comparison(forecast_result: Dict, primary_country: str):
    """Render multi-country comparison view."""
    st.divider()
    
    countries = get_available_countries(forecast_result)
    
    # Allow selection of countries to compare
    st.markdown("#### 📊 Select Countries to Compare")
    
    # Multi-select for comparison
    default_countries = [primary_country] if primary_country in countries else []
    
    # Add neighboring countries or top countries as suggestions
    if len(countries) > 5:
        # Suggest top 5 countries by final value
        country_values = get_country_final_values(countries, forecast_result)
        top_countries = sorted(country_values.items(), key=lambda x: x[1], reverse=True)[:5]
        suggested = [c[0] for c in top_countries]
        if primary_country not in suggested:
            suggested = [primary_country] + suggested[:4]
    else:
        suggested = countries
    
    selected_countries = st.multiselect(
        "Choose countries to compare",
        countries,
        default=suggested[:5],  # Default to top 5
        format_func=lambda x: f"{get_country_flag_emoji(x)} {x}",
        max_selections=10,
        help="Select up to 10 countries for comparison"
    )
    
    if not selected_countries:
        st.warning("Please select at least one country to display")
        return
    
    # Create comparison visualizations
    st.markdown("### 📈 Forecast Comparison")
    
    # Combined forecast chart
    fig = create_multi_country_chart(selected_countries, forecast_result)
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics comparison table
    st.markdown("### 📊 Key Metrics Comparison")
    metrics_df = create_metrics_comparison_table(selected_countries, forecast_result)
    
    if not metrics_df.empty:
        # Style the dataframe
        st.dataframe(
            metrics_df.style.format({
                'Historical CAGR': '{:.1f}%',
                'Forecast CAGR': '{:.1f}%',
                f'{get_forecast_until()} Value': '${:.1f}B',
                'Growth Δ': '{:+.1f}%'
            }).background_gradient(subset=[f'{get_forecast_until()} Value'], cmap='Blues'),
            use_container_width=True,
            hide_index=False
        )
    
    # Growth rate comparison
    st.markdown("### 📈 Growth Rate Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Historical vs Forecast CAGR comparison
        fig_cagr = create_cagr_comparison_chart(selected_countries, forecast_result)
        st.plotly_chart(fig_cagr, use_container_width=True)
    
    with col2:
        # Final values bar chart
        fig_values = create_final_values_chart(selected_countries, forecast_result)
        st.plotly_chart(fig_values, use_container_width=True)

def create_multi_country_chart(countries: List[str], forecast_result: Dict) -> go.Figure:
    """Create a multi-country forecast comparison chart."""
    fig = go.Figure()
    hist_cutoff = get_hist_cutoff()
    
    colors = px.colors.qualitative.Plotly
    
    for i, country in enumerate(countries):
        combined_df = prepare_country_timeline(country, forecast_result)
        
        if not combined_df.empty:
            hist_data = combined_df[combined_df['type'] == 'Historical']
            fcst_data = combined_df[combined_df['type'] == 'Forecast']
            
            color = colors[i % len(colors)]
            
            # Historical line
            if not hist_data.empty:
                fig.add_trace(go.Scatter(
                    x=hist_data['year'],
                    y=hist_data['value'],
                    mode='lines',
                    name=f"{country} (Historical)",
                    line=dict(color=color, width=2),
                    legendgroup=country,
                    showlegend=False
                ))
            
            # Forecast line
            if not fcst_data.empty:
                fig.add_trace(go.Scatter(
                    x=fcst_data['year'],
                    y=fcst_data['value'],
                    mode='lines',
                    name=country,
                    line=dict(color=color, width=2, dash='dash'),
                    legendgroup=country
                ))
    
    # Add vertical line at cutoff
    fig.add_vline(x=hist_cutoff + 0.5, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Multi-Country Forecast Comparison",
        xaxis_title="Year",
        yaxis_title="Market Value ($B)",
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_metrics_comparison_table(countries: List[str], forecast_result: Dict) -> pd.DataFrame:
    """Create a comparison table of key metrics for selected countries."""
    data = []
    
    for country in countries:
        combined_df = prepare_country_timeline(country, forecast_result)
        if not combined_df.empty:
            metrics = calculate_country_metrics(combined_df)
            
            data.append({
                'Country': f"{get_country_flag_emoji(country)} {country}",
                'Historical CAGR': metrics['hist_cagr'],
                'Forecast CAGR': metrics['fcst_cagr'],
                f'{get_forecast_until()} Value': metrics['final_value'],
                'Growth Δ': metrics['fcst_cagr'] - metrics['hist_cagr']
            })
    
    return pd.DataFrame(data).set_index('Country')

def create_cagr_comparison_chart(countries: List[str], forecast_result: Dict) -> go.Figure:
    """Create CAGR comparison chart."""
    hist_cagrs = []
    fcst_cagrs = []
    country_names = []
    
    for country in countries:
        combined_df = prepare_country_timeline(country, forecast_result)
        if not combined_df.empty:
            metrics = calculate_country_metrics(combined_df)
            country_names.append(country)
            hist_cagrs.append(metrics['hist_cagr'])
            fcst_cagrs.append(metrics['fcst_cagr'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Historical CAGR',
        x=country_names,
        y=hist_cagrs,
        marker_color='#2E86AB'
    ))
    
    fig.add_trace(go.Bar(
        name='Forecast CAGR',
        x=country_names,
        y=fcst_cagrs,
        marker_color='#D4463D'
    ))
    
    fig.update_layout(
        title="CAGR Comparison",
        xaxis_title="",
        yaxis_title="CAGR (%)",
        template="plotly_white",
        height=300,
        barmode='group'
    )
    
    return fig

def create_final_values_chart(countries: List[str], forecast_result: Dict) -> go.Figure:
    """Create final values comparison chart."""
    country_values = get_country_final_values(countries, forecast_result)
    
    if not country_values:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    sorted_data = sorted(country_values.items(), key=lambda x: x[1], reverse=True)
    
    fig = go.Figure(data=[
        go.Bar(
            x=[x[1] for x in sorted_data],
            y=[x[0] for x in sorted_data],
            orientation='h',
            marker_color='#2E86AB',
            text=[f"${x[1]:.1f}B" for x in sorted_data],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f"{get_forecast_until()} Market Value",
        xaxis_title="Market Value ($B)",
        yaxis_title="",
        template="plotly_white",
        height=300,
        margin=dict(l=100, r=50, t=50, b=50)
    )
    
    return fig

def render_country_pills(countries: List[str], selected_country: str) -> str:
    """Render country pills for small lists."""
    st.markdown("#### Select Country")
    
    # Create pills in rows
    countries_per_row = 5
    for i in range(0, len(countries), countries_per_row):
        row_countries = countries[i:i + countries_per_row]
        cols = st.columns(len(row_countries))
        
        for j, country in enumerate(row_countries):
            with cols[j]:
                flag = get_country_flag_emoji(country)
                label = f"{flag} {country}"
                
                if country == selected_country:
                    if st.button(label, key=f"country_{country}", type="primary", use_container_width=True):
                        selected_country = country
                else:
                    if st.button(label, key=f"country_{country}", use_container_width=True):
                        selected_country = country
    
    return selected_country

def render_news_analysis(country: str):
    """Render news analysis for a country."""
    st.markdown("### 📰 News Analysis")
    
    country_news_df = get_country_news_data(country)
    
    if country_news_df.empty:
        st.info(f"No news analysis available for {country}")
        return
    
    # Calculate metrics
    total_articles = len(country_news_df)
    
    # Growth impact averages
    growth_metrics = {}
    if 'growth_rate' in country_news_df.columns:
        # Calculate average news impact (excluding zero impact articles)
        relevant_news = country_news_df[country_news_df['growth_rate'] != 0]
        growth_metrics['news_avg'] = relevant_news['growth_rate'].mean() if not relevant_news.empty else 0
        growth_metrics['positive_count'] = (country_news_df['growth_rate'] > 0).sum()
        growth_metrics['negative_count'] = (country_news_df['growth_rate'] < 0).sum()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Headlines Analyzed", total_articles)
    with col2:
        news_impact = growth_metrics.get('news_avg', 0)
        st.metric("Average News Impact", f"{news_impact:+.2f}%",
                 help="Single recency-weighted average; impact decays over forecast horizon")
    with col3:
        positive = growth_metrics.get('positive_count', 0)
        negative = growth_metrics.get('negative_count', 0)
        st.metric("Sentiment", f"+{positive} / -{negative}",
                 help="Positive vs negative impact articles")
    
    # Add search and filter section
    st.markdown("#### Search & Filters")
    col_search, col_impact = st.columns([2, 1])
    
    with col_search:
        search_term = st.text_input("🔍 Search headlines", "", key=f"search_{country}")
    
    with col_impact:
        impact_range = st.slider(
            "Impact Range (%)",
            min_value=-100,
            max_value=100,
            value=(-100, 100),
            key=f"impact_{country}"
        )
    
    # Apply filters
    filtered_df = country_news_df.copy()
    
    # Search filter
    if search_term:
        headline_col = 'title' if 'title' in filtered_df.columns else 'headline' if 'headline' in filtered_df.columns else None
        if headline_col:
            mask = filtered_df[headline_col].str.contains(search_term, case=False, na=False)
            filtered_df = filtered_df[mask]
    

    # Impact range filter
    if 'growth_rate' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['growth_rate'] >= impact_range[0]) & 
            (filtered_df['growth_rate'] <= impact_range[1])
        ]
    
    # Display filtered count
    if len(filtered_df) < len(country_news_df):
        st.info(f"Showing {len(filtered_df)} of {len(country_news_df)} headlines (filtered)")
    
    # Display headlines table with all important columns
    headline_col = 'title' if 'title' in filtered_df.columns else 'headline' if 'headline' in filtered_df.columns else None
    
    if headline_col and not filtered_df.empty:
        display_df = pd.DataFrame()
        
        # Add date column if available
        if 'date' in filtered_df.columns:
            display_df['Date'] = pd.to_datetime(filtered_df['date']).dt.strftime('%Y-%m-%d')
        
        # Add headline
        display_df['Headline'] = filtered_df[headline_col].values
        
        # Add category
        if 'category' in filtered_df.columns:
            display_df['Category'] = filtered_df['category'].values
        

        # Add growth rate
        if 'growth_rate' in filtered_df.columns:
            display_df['Impact'] = filtered_df['growth_rate'].apply(lambda x: f"{x:+.1f}%").values
        
        # Add reason (truncated for display)
        if 'reason' in filtered_df.columns:
            display_df['Reason'] = filtered_df['reason'].str[:100].values
        
        st.dataframe(
            display_df, 
            use_container_width=True, 
            height=400,  # Increased height for better visibility
            hide_index=True
        )
    elif filtered_df.empty:
        st.warning("No headlines match the current filters")

def render_forecast_analysis(country: str, forecast_result: Dict):
    """Render forecast analysis for a country."""
    col1, col2 = st.columns([3, 2])
    
    with col1:
        
        # Get country timeline data
        combined_df = prepare_country_timeline(country, forecast_result)
        
        if combined_df.empty:
            st.warning(f"No forecast data available for {country}")
        else:
            # Create forecast chart with baseline comparison
            fig = create_country_forecast_chart(combined_df, country, forecast_result)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics
            metrics = calculate_country_metrics(combined_df)
            
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("Historical CAGR", f"{metrics['hist_cagr']:.1f}%")
            with metric_cols[1]:
                st.metric("Forecast CAGR", f"{metrics['fcst_cagr']:.1f}%")
            with metric_cols[2]:
                st.metric(f"{get_forecast_until()} Value", format_value_intelligent(metrics['final_value']))
    
    with col2:
        st.markdown(f"### 📊 Adjustment Breakdown")
        
        breakdown_df = create_adjustment_breakdown_table(country, forecast_result)
        
        if not breakdown_df.empty:
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True, height=400)
            
            # Add subtle caption with calibrated multiplier if available
            cal = (st.session_state.get('analyzed_news') or {}).get('calibration') or {}
            cm = (cal.get('countries') or {}).get(country, {})
            
            # Get country-specific or global MA breakdown
            category_breakdown = cm.get('category_breakdown')
            if not category_breakdown:
                # Try global if country-specific not available
                global_cal = cal.get('global', {})
                category_breakdown = global_cal.get('category_breakdown')
            
            # Display MA breakdown if available
            if category_breakdown:
                with st.expander("📊 Category Impact Breakdown", expanded=False):
                    # Show category analysis
                    active_categories = [(name, info) for name, info in category_breakdown.items() 
                                       if info.get('status') == 'Active']
                    
                    if active_categories:
                        st.write("**Active Categories:**")
                        for name, info in active_categories:
                            window = info.get('window', 0)
                            days_data = info.get('days_with_data', 0)
                            score = info.get('ma_score', 0)
                            coverage = info.get('coverage_pct', 0)
                            st.write(f"• {name}: {score:+.2f}% ({days_data}/{window} days, {coverage:.0f}% coverage)")
                        
                        # Calculate and show final impact
                        avg_impact = sum(info.get('ma_score', 0) for _, info in active_categories) / len(active_categories)
                        st.info(f"**Final News Impact: {avg_impact:+.2f}%** (equal-weighted average of categories)")
                    else:
                        st.info("No active categories with data")
            else:
                st.caption("Category-based MA breakdown using recency-weighted moving averages")
            
            st.info("✅ **News Impact Decay:** Category-specific moving averages with exponential decay over forecast horizon")

        # Calibration metrics (country)
        cal = (st.session_state.get('analyzed_news') or {}).get('calibration') or {}
        cm = (cal.get('countries') or {}).get(country, {})
        if cm:
            render_calibration_metrics(cm, show_reason=True, note_title=f"LLM Calibration — {country}")

def render_yoy_growth_analysis(country: str, forecast_result: Dict):
    """Render YoY growth analysis for a country."""
    combined_df = prepare_country_timeline(country, forecast_result)
    
    if combined_df.empty:
        st.warning(f"No data available for YoY growth analysis for {country}")
        return
    
    # Calculate YoY growth
    combined_df = combined_df.sort_values('year')
    combined_df['yoy_growth'] = combined_df['value'].pct_change() * 100
    
    # Create YoY growth chart
    fig = create_yoy_growth_chart(combined_df, country)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show growth statistics
    col1, col2, col3 = st.columns(3)
    
    hist_data = combined_df[combined_df['type'] == 'Historical']
    fcst_data = combined_df[combined_df['type'] == 'Forecast']
    
    with col1:
        if not hist_data.empty:
            avg_hist_growth = hist_data['yoy_growth'].mean()
            st.metric("Avg Historical YoY", f"{avg_hist_growth:.1f}%")
        else:
            st.metric("Avg Historical YoY", "N/A")
    
    with col2:
        if not fcst_data.empty:
            avg_fcst_growth = fcst_data['yoy_growth'].mean()
            st.metric("Avg Forecast YoY", f"{avg_fcst_growth:.1f}%")
        else:
            st.metric("Avg Forecast YoY", "N/A")
    
    with col3:
        if not hist_data.empty and not fcst_data.empty:
            transition_year = fcst_data['year'].min()
            transition_growth = fcst_data[fcst_data['year'] == transition_year]['yoy_growth'].iloc[0]
            st.metric(f"{transition_year} Growth", f"{transition_growth:.1f}%")
        else:
            st.metric("Transition Growth", "N/A")
    
    # Show detailed YoY table
    st.markdown("#### Year-over-Year Growth Details")
    
    display_df = combined_df[['year', 'value', 'yoy_growth', 'type']].copy()
    display_df['value'] = display_df['value'].apply(lambda x: f"${x:.1f}B")
    display_df['yoy_growth'] = display_df['yoy_growth'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "")
    display_df.columns = ['Year', 'Market Value', 'YoY Growth', 'Type']
    
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

def create_yoy_growth_chart(combined_df: pd.DataFrame, country: str) -> go.Figure:
    """Create YoY growth chart for a country."""
    hist_cutoff = get_hist_cutoff()
    
    hist_data = combined_df[combined_df['type'] == 'Historical']
    fcst_data = combined_df[combined_df['type'] == 'Forecast']
    
    fig = go.Figure()
    
    # Historical YoY growth
    if not hist_data.empty:
        hist_yoy = hist_data[hist_data['yoy_growth'].notna()]
        if not hist_yoy.empty:
            fig.add_trace(go.Scatter(
                x=hist_yoy['year'],
                y=hist_yoy['yoy_growth'],
                mode='lines+markers',
                name='Historical YoY',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8)
            ))
    
    # Forecast YoY growth
    if not fcst_data.empty:
        fcst_yoy = fcst_data[fcst_data['yoy_growth'].notna()]
        if not fcst_yoy.empty:
            fig.add_trace(go.Scatter(
                x=fcst_yoy['year'],
                y=fcst_yoy['yoy_growth'],
                mode='lines+markers',
                name='Forecast YoY',
                line=dict(color='#D4463D', width=3, dash='dash'),
                marker=dict(size=8)
            ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    
    # Add vertical line at cutoff
    fig.add_vline(x=hist_cutoff + 0.5, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"{get_country_flag_emoji(country)} {country} YoY Growth Rate",
        xaxis_title="Year",
        yaxis_title="YoY Growth (%)",
        template="plotly_white",
        height=400,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def prepare_country_timeline(country: str, forecast_result: Dict) -> pd.DataFrame:
    """Prepare combined historical + forecast timeline for a country."""
    try:
        unified_data = st.session_state.get('unified_data', {})
        hist_cutoff = get_hist_cutoff()
        
        if not unified_data:
            return pd.DataFrame()
        
        # Get historical data
        country_market = safe_get_nested(unified_data, 'market_value', 'country', default=pd.DataFrame())
        historical = pd.DataFrame()
        
        if not country_market.empty and 'country' in country_market.columns:
            hist_data = country_market[country_market['country'] == country].copy()
            if not hist_data.empty and 'year' in hist_data.columns:
                historical = hist_data[hist_data['year'] <= hist_cutoff].copy()
                historical['type'] = 'Historical'
        
        # Get forecast data
        forecast_data = safe_get_nested(forecast_result, 'country_forecasts', country, default=pd.DataFrame())
        
        if isinstance(forecast_data, dict):
            forecast = forecast_data.get('forecast', pd.DataFrame())
        else:
            forecast = forecast_data if isinstance(forecast_data, pd.DataFrame) else pd.DataFrame()
        
        if not forecast.empty and 'year' in forecast.columns:
            # Include the cutoff year in forecast for connection
            forecast = forecast[forecast['year'] >= hist_cutoff].copy()
            forecast['type'] = 'Forecast'
        else:
            forecast = pd.DataFrame()
        
        # Combine data
        combined_parts = []
        
        if not historical.empty:
            value_col = safe_get_value_column(historical)
            if value_col:
                hist_clean = historical[['year', value_col, 'type']].rename(columns={value_col: 'value'})
                # Keep original historical values
                combined_parts.append(hist_clean)
        
        if not forecast.empty:
            value_col = safe_get_value_column(forecast)
            if value_col:
                fcst_clean = forecast[['year', value_col, 'type']].rename(columns={value_col: 'value'})
                # Keep original forecast values
                # Remove duplicate 2024 if it exists in both
                if not hist_clean.empty and hist_cutoff in fcst_clean['year'].values:
                    fcst_clean = fcst_clean[fcst_clean['year'] > hist_cutoff]
                combined_parts.append(fcst_clean)
        
        if not combined_parts:
            return pd.DataFrame()
        
        combined = pd.concat(combined_parts, ignore_index=True).sort_values('year')
        return combined
        
    except Exception as e:
        st.error(f"Error preparing timeline for {country}: {str(e)}")
        return pd.DataFrame()

def create_country_forecast_chart(combined_df: pd.DataFrame, country: str, forecast_result: Dict = None) -> go.Figure:
    """Create forecast chart for a country."""
    if combined_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    hist_data = combined_df[combined_df['type'] == 'Historical']
    fcst_data = combined_df[combined_df['type'] == 'Forecast']
    hist_cutoff = get_hist_cutoff()
    
    fig = go.Figure()
    
    # Determine common scaling (billions when values are large)
    max_val = 0
    if not hist_data.empty:
        max_val = max(max_val, float(pd.to_numeric(hist_data['value'], errors='coerce').max()))
    if not fcst_data.empty:
        max_val = max(max_val, float(pd.to_numeric(fcst_data['value'], errors='coerce').max()))
    scale_to_billion = max_val > 1e6
    def _scale(v):
        return (pd.to_numeric(v, errors='coerce') / 1e9) if scale_to_billion else pd.to_numeric(v, errors='coerce')

    # Historical line
    if not hist_data.empty:
        fig.add_trace(go.Scatter(
            x=hist_data['year'],
            y=_scale(hist_data['value']),
            mode='lines+markers',
            name='Historical',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=6)
        ))
    
    # Try to add baseline forecast if available
    if forecast_result:
        baseline_forecasts = forecast_result.get('baseline_country_forecasts', {})
        country_forecasts = forecast_result.get('country_forecasts', {})
        
        # Get baseline data
        baseline_df = None
        if country in baseline_forecasts:
            baseline_df = baseline_forecasts[country]
        elif country in country_forecasts:
            country_data = country_forecasts[country]
            if isinstance(country_data, dict):
                baseline_df = country_data.get('baseline')
        
        # Plot baseline if available
        if baseline_df is not None and not baseline_df.empty:
            b_sorted = baseline_df.sort_values('year')
            baseline_fcst = b_sorted[b_sorted['year'] > hist_cutoff]
            if not baseline_fcst.empty:
                value_col = safe_get_value_column(baseline_fcst)
                if value_col:
                    # Connection line (hist → baseline first forecast)
                    if not hist_data.empty:
                        last_hist_year = int(hist_data['year'].max())
                        last_hist_value = float(hist_data[hist_data['year'] == last_hist_year]['value'].iloc[0])
                        first_b_year = int(baseline_fcst['year'].min())
                        first_b_value = float(baseline_fcst[baseline_fcst['year'] == first_b_year][value_col].iloc[0])
                        fig.add_trace(go.Scatter(
                            x=[last_hist_year, first_b_year],
                            y=[_scale(last_hist_value), _scale(first_b_value)],
                            mode='lines',
                            line=dict(color='#808080', width=2, dash='dot'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    fig.add_trace(go.Scatter(
                        x=baseline_fcst['year'],
                        y=_scale(baseline_fcst[value_col]),
                        mode='lines+markers',
                        name='Baseline Forecast',
                        line=dict(color='#808080', width=2, dash='dash'),
                        marker=dict(size=4),
                        opacity=0.9
                    ))
    
    # Adjusted forecast line
    if not fcst_data.empty:
        # Connect historical → adjusted first forecast
        if not hist_data.empty:
            last_hist_year = int(hist_data['year'].max())
            last_hist_value = float(hist_data[hist_data['year'] == last_hist_year]['value'].iloc[0])
            first_f_year = int(fcst_data['year'].min())
            first_f_value = float(fcst_data[fcst_data['year'] == first_f_year]['value'].iloc[0])
            fig.add_trace(go.Scatter(
                x=[last_hist_year, first_f_year],
                y=[_scale(last_hist_value), _scale(first_f_value)],
                mode='lines',
                line=dict(color='#D4463D', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        # Check if there are adjustments to show different label
        adjustments = forecast_result.get('adjustments_applied', []) if forecast_result else []
        forecast_name = 'Adjusted Forecast' if adjustments else 'Forecast'
        fig.add_trace(go.Scatter(
            x=fcst_data['year'],
            y=_scale(fcst_data['value']),
            mode='lines+markers',
            name=forecast_name,
            line=dict(color='#D4463D', width=3),
            marker=dict(size=6)
        ))
    
    # Add vertical line at cutoff
    fig.add_vline(x=hist_cutoff + 0.5, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"{get_country_flag_emoji(country)} {country} Market Forecast",
        xaxis_title="Year",
        yaxis_title="Market Value ($B)" if scale_to_billion else "Market Value",
        template="plotly_white",
        height=400,
        showlegend=True
    )
    
    return fig

def calculate_country_metrics(combined_df: pd.DataFrame) -> Dict:
    """Calculate metrics for a country."""
    if combined_df.empty:
        return {'hist_cagr': 0, 'fcst_cagr': 0, 'final_value': 0}
    
    hist_data = combined_df[combined_df['type'] == 'Historical']
    fcst_data = combined_df[combined_df['type'] == 'Forecast']
    
    metrics = {}
    
    # Historical CAGR
    if len(hist_data) >= 2:
        years = hist_data.iloc[-1]['year'] - hist_data.iloc[0]['year']
        if years > 0:
            start_val = hist_data.iloc[0]['value']
            end_val = hist_data.iloc[-1]['value']
            if start_val > 0:
                metrics['hist_cagr'] = ((end_val / start_val) ** (1/years) - 1) * 100
            else:
                metrics['hist_cagr'] = 0
        else:
            metrics['hist_cagr'] = 0
    else:
        metrics['hist_cagr'] = 0
    
    # Forecast CAGR
    if len(fcst_data) >= 2:
        years = fcst_data.iloc[-1]['year'] - fcst_data.iloc[0]['year']
        if years > 0:
            start_val = fcst_data.iloc[0]['value']
            end_val = fcst_data.iloc[-1]['value']
            if start_val > 0:
                metrics['fcst_cagr'] = ((end_val / start_val) ** (1/years) - 1) * 100
            else:
                metrics['fcst_cagr'] = 0
        else:
            metrics['fcst_cagr'] = 0
    else:
        metrics['fcst_cagr'] = 0
    
    # Final value
    metrics['final_value'] = fcst_data.iloc[-1]['value'] if not fcst_data.empty else 0
    
    return metrics

def create_global_adjustment_breakdown_table(forecast_result: Dict) -> pd.DataFrame:
    """Create adjustment breakdown table for global forecast."""
    try:
        hist_cutoff = get_hist_cutoff()
        
        # Get global forecast and baseline
        global_forecast = forecast_result.get('global_forecast', pd.DataFrame())
        # Baseline may be stored under 'baseline' or 'baseline_global_forecast'
        baseline_forecast = forecast_result.get('baseline', pd.DataFrame())
        if baseline_forecast.empty:
            baseline_forecast = forecast_result.get('baseline_global_forecast', pd.DataFrame())
        
        if global_forecast.empty:
            return pd.DataFrame()
        
        # Get adjustment details
        adjustment_details = forecast_result.get('adjustment_details', {})
        global_adj = adjustment_details.get('global', {})
        year_adjustments = global_adj.get('year_adjustments', {})

        # If no global details exist (e.g., Top-Down/Bottom-Up), aggregate per-country
        if not year_adjustments:
            baseline_country_forecasts = forecast_result.get('baseline_country_forecasts', {})
            if baseline_country_forecasts:
                agg_adjustments = {}
                years = [int(y) for y in global_forecast[global_forecast['year'] > hist_cutoff]['year'].tolist()]
                for year in years:
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
        
        # Build table
        table_data = []
        
        # Historical years
        hist_data = global_forecast[global_forecast['year'] <= hist_cutoff]
        for _, row in hist_data.iterrows():
            value = row.get('value_hat', 0)
            value = value / 1e9 if value > 1e6 else value
            table_data.append({
                'Year': int(row['year']),
                'Baseline': f"${value:,.1f}B",
                'News (%)': '',
                'Indicators (%)': '',
                'Final': f"${value:,.1f}B"
            })
        
        # Forecast years
        forecast_years = global_forecast[global_forecast['year'] > hist_cutoff]
        
        for _, row in forecast_years.iterrows():
            year = int(row['year'])
            adjusted_value = row.get('value_hat', 0)
            adjusted_value = adjusted_value / 1e9 if adjusted_value > 1e6 else adjusted_value
            
            # Get baseline value for this year
            baseline_value = adjusted_value  # Default to adjusted if no baseline found
            if baseline_forecast is not None and not baseline_forecast.empty:
                baseline_row = baseline_forecast[baseline_forecast['year'] == year]
                if not baseline_row.empty:
                    baseline_val = baseline_row.iloc[0].get('value_hat', adjusted_value)
                    baseline_value = baseline_val / 1e9 if baseline_val > 1e6 else baseline_val
            
            # If no adjustments, baseline equals adjusted
            if not year_adjustments or year not in year_adjustments:
                baseline_value = adjusted_value
            
            year_adj = year_adjustments.get(year, {})
            news = year_adj.get('news_pct', 0)
            indicators = year_adj.get('indicators_pct', 0)
            
            table_data.append({
                'Year': year,
                'Baseline': format_value_intelligent(baseline_value),
                'News (%)': f"{news:+.1f}%" if news != 0 else "",
                'Indicators (%)': f"{indicators:+.1f}%" if indicators != 0 else "",
                'Final': format_value_intelligent(adjusted_value)
            })
        
        return pd.DataFrame(table_data)
        
    except Exception as e:
        st.error(f"Error creating global breakdown: {str(e)}")
        return pd.DataFrame()

def create_adjustment_breakdown_table(country: str, forecast_result: Dict) -> pd.DataFrame:
    """Create adjustment breakdown table for a country."""
    try:
        unified_data = st.session_state.get('unified_data', {})
        hist_cutoff = get_hist_cutoff()
        
        # Get historical data
        country_market = safe_get_nested(unified_data, 'market_value', 'country', default=pd.DataFrame())
        historical_data = []
        
        if not country_market.empty and 'country' in country_market.columns:
            hist_data = country_market[country_market['country'] == country].copy()
            if not hist_data.empty:
                hist_data = hist_data[hist_data['year'] <= hist_cutoff].copy()
                for _, row in hist_data.iterrows():
                    value = row['value']  # Keep original value
                    historical_data.append({
                        'Year': int(row['year']),
                        'Baseline': format_value_intelligent(value),
                        'News (%)': '',
                        'Indicators (%)': '',
                        'Final': format_value_intelligent(value)
                    })
        
        # Get forecast data and baseline data
        country_forecasts = forecast_result.get('country_forecasts', {})
        baseline_country_forecasts = forecast_result.get('baseline_country_forecasts', {})
        
        if country not in country_forecasts:
            return pd.DataFrame(historical_data) if historical_data else pd.DataFrame()
        
        # Get adjusted forecast
        country_data = country_forecasts[country]
        if isinstance(country_data, dict):
            forecast_df = country_data.get('forecast', pd.DataFrame())
            # Try to get baseline from within country data structure first
            baseline_df = country_data.get('baseline', pd.DataFrame())
            if baseline_df.empty and baseline_country_forecasts:
                # Fallback to separate baseline_country_forecasts if available
                baseline_df = baseline_country_forecasts.get(country, pd.DataFrame())
        else:
            forecast_df = country_data if isinstance(country_data, pd.DataFrame) else pd.DataFrame()
            # Try to get baseline from the separate baseline_country_forecasts
            baseline_df = baseline_country_forecasts.get(country, pd.DataFrame())
        
        if forecast_df.empty:
            return pd.DataFrame(historical_data) if historical_data else pd.DataFrame()
        
        # Get adjustment details
        adjustment_details = forecast_result.get('adjustment_details', {})
        country_adj = adjustment_details.get(country, {})
        year_adjustments = country_adj.get('year_adjustments', {})
        
        # Build table
        table_data = historical_data.copy()
        
        # Add forecast years
        forecast_years = forecast_df[forecast_df['year'] > hist_cutoff]
        
        for _, row in forecast_years.iterrows():
            year = int(row['year'])
            adjusted_value = row.get('value_hat', 0)
            # Keep original adjusted value - no automatic conversion
            
            # Get baseline value for this year
            baseline_value = adjusted_value  # Default to adjusted if no baseline found
            if baseline_df is not None and not baseline_df.empty:
                baseline_row = baseline_df[baseline_df['year'] == year]
                if not baseline_row.empty:
                    baseline_val = baseline_row.iloc[0].get('value_hat', adjusted_value)
                    baseline_value = baseline_val  # Keep original value
            
            # If no adjustments, baseline equals adjusted
            if not year_adjustments or year not in year_adjustments:
                baseline_value = adjusted_value
            
            year_adj = year_adjustments.get(year, {})
            news = year_adj.get('news_pct', 0)
            indicators = year_adj.get('indicators_pct', 0)
            
            table_data.append({
                'Year': year,
                'Baseline': format_value_intelligent(baseline_value),
                'News (%)': f"{news:+.1f}%" if news != 0 else "",
                'Indicators (%)': f"{indicators:+.1f}%" if indicators != 0 else "",
                'Final': format_value_intelligent(adjusted_value)
            })
        
        return pd.DataFrame(table_data)
        
    except Exception as e:
        st.error(f"Error creating breakdown: {str(e)}")
        return pd.DataFrame()

def get_country_news_data(country: str) -> pd.DataFrame:
    """Get news data for a country."""
    try:
        news_dataframes = st.session_state.get('news_dataframes', {})
        
        if country in news_dataframes:
            news_df = news_dataframes[country]
            if isinstance(news_df, pd.DataFrame):
                return news_df
        
        # Fallback to global news
        if 'global' in news_dataframes:
            return news_dataframes['global']
        
        return pd.DataFrame()
        
    except Exception:
        return pd.DataFrame()

# === MAIN PAGE FUNCTION ===
def main():
    st.title("💡 Multi-Country Forecast Insights")
    st.info("""
    Explore global, regional, and country-level views. Use filters and tabs to compare baselines, adjustments, and final forecasts. Hover charts for details; use selectors to focus regions/countries.
    """)
    
    # Check if forecast exists
    if 'forecast_result' not in st.session_state:
        st.warning("No forecast has been generated yet. Please go to the Forecasting page first.")
        if st.button("Go to Forecasting"):
            st.switch_page("pages/03_Forecasting.py")
        return
    
    forecast_result = st.session_state['forecast_result']
    method = forecast_result.get('method', 'Unknown')
    
    # Display method info
    st.info(f"**Forecast Method:** {method.replace('_', ' ').title()}")
    
    # Determine available views based on method and data
    has_global = 'global_forecast' in forecast_result
    has_regional = 'region_forecasts' in forecast_result
    has_countries = 'country_forecasts' in forecast_result
    
    # Create view tabs based on available data
    available_views = []
    if has_global:
        available_views.append("🌍 Global")
    if has_regional:
        available_views.append("🗺️ Regional")
    if has_countries:
        available_views.append("📍 Country")
    
    if not available_views:
        st.error("No forecast data available to display")
        return
    
    # View selector
    if len(available_views) > 1:
        selected_view = st.radio(
            "Select View Level",
            available_views,
            horizontal=True,
            key="view_selector"
        )
    else:
        selected_view = available_views[0]
        st.markdown(f"**View:** {selected_view}")
    
    st.divider()
    
    # Render selected view
    if selected_view == "🌍 Global":
        render_global_insights(forecast_result)
    elif selected_view == "🗺️ Regional":
        render_regional_insights(forecast_result)
    elif selected_view == "📍 Country":
        render_country_insights(forecast_result)

# Run the app
if __name__ == "__main__":
    main()
