"""
Reusable chart components for forecast visualization.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional


def display_forecast_chart(result: Dict, title: Optional[str] = None):
    """Display forecast chart based on method type."""
    if 'error' in result:
        st.error(f"❌ {result['error']}")
        return
    
    method = result.get('method', 'unknown')
    
    if method == 'global_only':
        _display_global_chart(result, title or "Global Forecast")
    elif method == 'top_down':
        _display_top_down_charts(result, title or "Top-Down Forecast")
    elif method == 'bottom_up':
        _display_bottom_up_charts(result, title or "Bottom-Up Forecast")
    elif method == 'country_specific':
        _display_country_specific_charts(result, title or "Country-Specific Forecasts")
    else:
        st.error(f"Unknown method for chart display: {method}")


def _display_global_chart(result: Dict, title: str):
    """Display chart for global-only forecast."""
    forecast = result.get('forecast')
    baseline = result.get('baseline')
    
    if forecast is None or forecast.empty:
        st.warning("No forecast data to display")
        return
    
    fig = _create_single_forecast_chart(forecast, baseline, title, result.get('adjustments_applied', []))
    st.plotly_chart(fig, use_container_width=True)


def _display_top_down_charts(result: Dict, title: str):
    """Display charts for top-down forecast."""
    global_forecast = result.get('global_forecast')
    country_forecasts = result.get('country_forecasts', {})
    
    # Global chart
    if global_forecast is not None and not global_forecast.empty:
        st.subheader("Global Forecast")
        fig = _create_single_forecast_chart(
            global_forecast, 
            None, 
            "Global Forecast (Top-Down)", 
            result.get('adjustments_applied', [])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Country charts
    if country_forecasts:
        st.subheader("Country Distribution")
        
        # Show top countries in a combined chart
        top_countries = list(country_forecasts.keys())[:5]  # Show top 5
        if len(top_countries) > 1:
            fig = _create_multi_country_chart(country_forecasts, top_countries, "Top Countries Forecast")
            st.plotly_chart(fig, use_container_width=True)
        
        # Individual country details in expander
        with st.expander(f"Individual Country Details ({len(country_forecasts)} countries)"):
            selected_country = st.selectbox("Select Country", list(country_forecasts.keys()))
            if selected_country:
                # Try to find baseline for this country
                baseline_map = result.get('baseline_country_forecasts', {}) or {}
                baseline_df = None
                if selected_country in baseline_map:
                    baseline_df = baseline_map[selected_country]
                else:
                    sel = country_forecasts[selected_country]
                    if isinstance(sel, dict):
                        baseline_df = sel.get('baseline')
                fig = _create_single_forecast_chart(
                    country_forecasts[selected_country], 
                    baseline_df, 
                    f"{selected_country} Forecast",
                    result.get('adjustments_applied', [])
                )
                st.plotly_chart(fig, use_container_width=True)


def _display_bottom_up_charts(result: Dict, title: str):
    """Display charts for bottom-up forecast."""
    global_forecast = result.get('global_forecast')
    country_forecasts = result.get('country_forecasts', {})
    
    # Global aggregated chart
    if global_forecast is not None and not global_forecast.empty:
        st.subheader("Global Aggregated Forecast")
        fig = _create_single_forecast_chart(
            global_forecast, 
            None, 
            "Global Forecast (Bottom-Up Aggregation)", 
            result.get('adjustments_applied', [])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Country breakdown
    if country_forecasts:
        st.subheader("Country Breakdown")
        
        # Multi-country view
        if len(country_forecasts) > 1:
            fig = _create_multi_country_chart(country_forecasts, list(country_forecasts.keys()), "All Countries")
            st.plotly_chart(fig, use_container_width=True)


def _display_country_specific_charts(result: Dict, title: str):
    """Display charts for country-specific forecasts."""
    country_forecasts = result.get('country_forecasts', {})
    
    if not country_forecasts:
        st.warning("No country forecasts to display")
        return
    
    st.subheader(f"Country-Specific Forecasts ({len(country_forecasts)} countries)")
    
    # Show each country individually
    for country, country_result in country_forecasts.items():
        st.markdown(f"#### {country}")
        
        forecast = country_result.get('forecast')
        baseline = country_result.get('baseline')
        if forecast is not None and not forecast.empty:
            fig = _create_single_forecast_chart(
                forecast, 
                baseline, 
                f"{country} Detailed Forecast",
                result.get('adjustments_applied', [])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Show country metadata
        metadata = country_result.get('metadata', {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Points", metadata.get('data_points', 'N/A'))
        with col2:
            st.metric("Has Indicators", "Yes" if metadata.get('has_indicators') else "No")
        with col3:
            st.metric("Has News", "Yes" if metadata.get('has_country_news') else "No")


def _create_single_forecast_chart(forecast: pd.DataFrame, baseline: Optional[pd.DataFrame], title: str, adjustments: list) -> go.Figure:
    """Create a single forecast line chart."""
    fig = go.Figure()
    
    # Sort data by year to ensure proper line connections
    forecast_sorted = forecast.sort_values('year')
    
    # Historical data
    historical = forecast_sorted[forecast_sorted['type'] == 'Historical']
    if not historical.empty:
        fig.add_trace(go.Scatter(
            x=historical['year'],
            y=historical['value_hat'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
    
    # Forecast data
    forecast_data = forecast_sorted[forecast_sorted['type'] == 'Forecast']
    if not forecast_data.empty:
        # Use consistent colors: adjusted=red, baseline=gray
        color = '#D4463D' if adjustments else '#D4463D'
        name = 'Adjusted Forecast' if adjustments else 'Forecast'
        
        # Create connection point between historical and forecast
        connection_x = []
        connection_y = []
        
        if not historical.empty and not forecast_data.empty:
            # Add last historical point to connect the lines
            last_hist_year = historical['year'].max()
            last_hist_value = historical[historical['year'] == last_hist_year]['value_hat'].iloc[0]
            first_forecast_year = forecast_data['year'].min()
            first_forecast_value = forecast_data[forecast_data['year'] == first_forecast_year]['value_hat'].iloc[0]
            
            connection_x = [last_hist_year, first_forecast_year]
            connection_y = [last_hist_value, first_forecast_value]
        
        # Add connection line if needed
        if connection_x and len(connection_x) == 2:
            fig.add_trace(go.Scatter(
                x=connection_x,
                y=connection_y,
                mode='lines',
                name='Connection',
                line=dict(color=color, width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.add_trace(go.Scatter(
            x=forecast_data['year'],
            y=forecast_data['value_hat'],
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=6)
        ))
    
    # Baseline comparison if provided
    if baseline is not None and hasattr(baseline, 'empty') and not baseline.empty:
        b_sorted = baseline.sort_values('year')
        baseline_forecast = b_sorted[b_sorted['type'] == 'Forecast']
        if not baseline_forecast.empty:
            # Optional connection from last historical to first baseline forecast
            if not historical.empty:
                last_hist_year = historical['year'].max()
                last_hist_value = historical[historical['year'] == last_hist_year]['value_hat'].iloc[0]
                first_b_year = int(baseline_forecast['year'].min())
                first_b_value = baseline_forecast[baseline_forecast['year'] == first_b_year]['value_hat'].iloc[0]
                fig.add_trace(go.Scatter(
                    x=[last_hist_year, first_b_year],
                    y=[last_hist_value, first_b_value],
                    mode='lines',
                    name='Baseline Connection',
                    line=dict(color='#808080', width=2, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            fig.add_trace(go.Scatter(
                x=baseline_forecast['year'],
                y=baseline_forecast['value_hat'],
                mode='lines+markers',
                name='Baseline Forecast',
                line=dict(color='#808080', width=2, dash='dash'),
                marker=dict(size=5),
                opacity=0.9
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title='Value',
        hovermode='x unified',
        showlegend=True,
        height=400
    )
    
    return fig


def _create_multi_country_chart(country_forecasts: Dict, countries_to_show: list, title: str) -> go.Figure:
    """Create a multi-line chart for multiple countries."""
    fig = go.Figure()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, country in enumerate(countries_to_show):
        if country not in country_forecasts:
            continue
            
        forecast = country_forecasts[country]
        if isinstance(forecast, dict):
            forecast = forecast.get('forecast', pd.DataFrame())
        
        if forecast.empty:
            continue
        
        # Sort data by year
        forecast_sorted = forecast.sort_values('year')
        color = colors[i % len(colors)]
        
        # Historical
        historical = forecast_sorted[forecast_sorted['type'] == 'Historical']
        if not historical.empty:
            fig.add_trace(go.Scatter(
                x=historical['year'],
                y=historical['value_hat'],
                mode='lines',
                name=f'{country} (Historical)',
                line=dict(color=color, width=1),
                showlegend=False
            ))
        
        # Forecast
        forecast_data = forecast_sorted[forecast_sorted['type'] == 'Forecast']
        if not forecast_data.empty:
            # Add connection between historical and forecast
            if not historical.empty:
                last_hist_year = historical['year'].max()
                last_hist_value = historical[historical['year'] == last_hist_year]['value_hat'].iloc[0]
                first_forecast_year = forecast_data['year'].min()
                first_forecast_value = forecast_data[forecast_data['year'] == first_forecast_year]['value_hat'].iloc[0]
                
                # Connection line
                fig.add_trace(go.Scatter(
                    x=[last_hist_year, first_forecast_year],
                    y=[last_hist_value, first_forecast_value],
                    mode='lines',
                    line=dict(color=color, width=2, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            fig.add_trace(go.Scatter(
                x=forecast_data['year'],
                y=forecast_data['value_hat'],
                mode='lines+markers',
                name=country,
                line=dict(color=color, width=2, dash='dot'),
                marker=dict(size=4)
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title='Value',
        hovermode='x unified',
        showlegend=True,
        height=500
    )
    
    return fig
