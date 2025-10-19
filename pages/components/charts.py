"""
Chart creation functions for Streamlit pages.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional

from .utils import (
    safe_get_value_column,
    get_hist_cutoff,
    get_forecast_until,
    calculate_cagr
)


def create_global_forecast_chart(global_forecast: pd.DataFrame, 
                                forecast_result: Dict) -> go.Figure:
    """
    Create global forecast chart with historical and forecast data.
    
    Args:
        global_forecast: DataFrame with global forecast data
        forecast_result: Complete forecast result dictionary
        
    Returns:
        Plotly figure
    """
    hist_cutoff = get_hist_cutoff()
    
    # Split data
    hist_data = global_forecast[global_forecast['year'] <= hist_cutoff]
    fcst_data = global_forecast[global_forecast['year'] > hist_cutoff]
    
    # Get baseline forecast if available
    baseline_forecast = forecast_result.get('baseline_global_forecast', pd.DataFrame())
    if baseline_forecast.empty:
        baseline_forecast = forecast_result.get('baseline', pd.DataFrame())
    
    fig = go.Figure()
    
    # Historical line
    if not hist_data.empty:
        value_col = safe_get_value_column(hist_data)
        if value_col:
            values = hist_data[value_col].values
            # Keep original values - let axis formatting handle units
            
            fig.add_trace(go.Scatter(
                x=hist_data['year'],
                y=values,
                mode='lines+markers',
                name='Historical',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=6)
            ))
    
    # Baseline forecast line (if available and different from adjusted)
    if not baseline_forecast.empty:
        baseline_fcst = baseline_forecast[baseline_forecast['year'] > hist_cutoff]
        if not baseline_fcst.empty:
            value_col = safe_get_value_column(baseline_fcst)
            if value_col:
                values = baseline_fcst[value_col].values
                # Keep original values - let axis formatting handle units
                
                fig.add_trace(go.Scatter(
                    x=baseline_fcst['year'],
                    y=values,
                    mode='lines',
                    name='Baseline Forecast',
                    line=dict(color='#808080', width=2, dash='dot'),
                    marker=dict(size=4),
                    opacity=0.7
                ))
    
    # Adjusted forecast line
    if not fcst_data.empty:
        value_col = safe_get_value_column(fcst_data)
        if value_col:
            values = fcst_data[value_col].values
            # Keep original values - let axis formatting handle units
            
            # Check if there are adjustments to show different label
            adjustments = forecast_result.get('adjustments_applied', [])
            forecast_name = 'Adjusted Forecast' if adjustments else 'Forecast'
            
            fig.add_trace(go.Scatter(
                x=fcst_data['year'],
                y=values,
                mode='lines+markers',
                name=forecast_name,
                line=dict(color='#D4463D', width=3, dash='dash'),
                marker=dict(size=6)
            ))
    
    # Add vertical line at cutoff
    fig.add_vline(x=hist_cutoff + 0.5, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Global Market Forecast",
        xaxis_title="Year",
        yaxis_title="Market Value ($B)",
        template="plotly_white",
        height=400,
        showlegend=True
    )
    
    return fig


def create_regional_composition_chart(region_forecasts: Dict[str, pd.DataFrame]) -> go.Figure:
    """
    Create a chart showing regional composition.
    
    Args:
        region_forecasts: Dictionary of regional forecasts
        
    Returns:
        Plotly figure
    """
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
                # Keep original final value
                regional_values[region] = final_val
    
    if not regional_values:
        fig = go.Figure()
        fig.add_annotation(
            text="No regional data available", 
            xref="paper", yref="paper", 
            x=0.5, y=0.5
        )
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


def create_regional_forecast_chart(region_data: pd.DataFrame, 
                                  region_name: str) -> go.Figure:
    """
    Create forecast chart for a specific region.
    
    Args:
        region_data: DataFrame with regional data
        region_name: Name of the region
        
    Returns:
        Plotly figure
    """
    hist_cutoff = get_hist_cutoff()
    
    # Split data
    hist_data = region_data[region_data['year'] <= hist_cutoff]
    fcst_data = region_data[region_data['year'] > hist_cutoff]
    
    fig = go.Figure()
    
    # Historical line
    if not hist_data.empty:
        value_col = safe_get_value_column(hist_data)
        if value_col:
            values = hist_data[value_col].values
            # Keep original values - let axis formatting handle units
            
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
            # Keep original values - let axis formatting handle units
            
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


def create_multi_country_chart(countries: List[str], 
                              forecast_result: Dict) -> go.Figure:
    """
    Create comparison chart for multiple countries.
    
    Args:
        countries: List of country names
        forecast_result: Complete forecast result
        
    Returns:
        Plotly figure
    """
    hist_cutoff = get_hist_cutoff()
    country_forecasts = forecast_result.get('country_forecasts', {})
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1[:len(countries)]
    
    for i, country in enumerate(countries):
        if country not in country_forecasts:
            continue
        
        # Get country data
        if isinstance(country_forecasts[country], dict):
            country_df = country_forecasts[country].get('forecast', pd.DataFrame())
        else:
            country_df = country_forecasts[country]
        
        if country_df.empty:
            continue
        
        # Split into historical and forecast
        hist_data = country_df[country_df['year'] <= hist_cutoff]
        fcst_data = country_df[country_df['year'] > hist_cutoff]
        
        # Historical line
        if not hist_data.empty:
            value_col = safe_get_value_column(hist_data)
            if value_col:
                values = hist_data[value_col].values
                # Keep original values - let axis formatting handle units
                
                fig.add_trace(go.Scatter(
                    x=hist_data['year'],
                    y=values,
                    mode='lines+markers',
                    name=f"{country} (Historical)",
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=5),
                    legendgroup=country
                ))
        
        # Forecast line
        if not fcst_data.empty:
            value_col = safe_get_value_column(fcst_data)
            if value_col:
                values = fcst_data[value_col].values
                # Keep original values - let axis formatting handle units
                
                fig.add_trace(go.Scatter(
                    x=fcst_data['year'],
                    y=values,
                    mode='lines+markers',
                    name=f"{country} (Forecast)",
                    line=dict(color=colors[i], width=2, dash='dash'),
                    marker=dict(size=5),
                    legendgroup=country
                ))
    
    # Add vertical line at cutoff
    fig.add_vline(x=hist_cutoff + 0.5, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Multi-Country Forecast Comparison",
        xaxis_title="Year",
        yaxis_title="Market Value ($B)",
        template="plotly_white",
        height=500,
        hovermode='x unified'
    )
    
    return fig


def create_cagr_comparison_chart(countries: List[str], 
                                forecast_result: Dict) -> go.Figure:
    """
    Create CAGR comparison chart for countries.
    
    Args:
        countries: List of country names
        forecast_result: Complete forecast result
        
    Returns:
        Plotly figure
    """
    hist_cutoff = get_hist_cutoff()
    country_forecasts = forecast_result.get('country_forecasts', {})
    
    cagr_data = []
    
    for country in countries:
        if country not in country_forecasts:
            continue
        
        # Get country data
        if isinstance(country_forecasts[country], dict):
            country_df = country_forecasts[country].get('forecast', pd.DataFrame())
        else:
            country_df = country_forecasts[country]
        
        if country_df.empty:
            continue
        
        # Calculate historical CAGR
        hist_data = country_df[country_df['year'] <= hist_cutoff]
        hist_cagr = 0
        if len(hist_data) >= 2:
            value_col = safe_get_value_column(hist_data)
            if value_col:
                years = hist_data.iloc[-1]['year'] - hist_data.iloc[0]['year']
                if years > 0:
                    start_val = hist_data.iloc[0][value_col]
                    end_val = hist_data.iloc[-1][value_col]
                    hist_cagr = calculate_cagr(start_val, end_val, years)
        
        # Calculate forecast CAGR
        fcst_data = country_df[country_df['year'] > hist_cutoff]
        fcst_cagr = 0
        if len(fcst_data) >= 2:
            value_col = safe_get_value_column(fcst_data)
            if value_col:
                years = fcst_data.iloc[-1]['year'] - fcst_data.iloc[0]['year']
                if years > 0:
                    start_val = fcst_data.iloc[0][value_col]
                    end_val = fcst_data.iloc[-1][value_col]
                    fcst_cagr = calculate_cagr(start_val, end_val, years)
        
        cagr_data.append({
            'Country': country,
            'Historical CAGR': hist_cagr,
            'Forecast CAGR': fcst_cagr
        })
    
    if not cagr_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No CAGR data available", 
            xref="paper", yref="paper", 
            x=0.5, y=0.5
        )
        return fig
    
    df = pd.DataFrame(cagr_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Country'],
        y=df['Historical CAGR'],
        name='Historical CAGR',
        marker_color='#2E86AB'
    ))
    
    fig.add_trace(go.Bar(
        x=df['Country'],
        y=df['Forecast CAGR'],
        name='Forecast CAGR',
        marker_color='#D4463D'
    ))
    
    fig.update_layout(
        title="CAGR Comparison",
        xaxis_title="Country",
        yaxis_title="CAGR (%)",
        template="plotly_white",
        height=400,
        barmode='group'
    )
    
    return fig


def create_yoy_growth_chart(combined_df: pd.DataFrame, 
                           country: str) -> go.Figure:
    """
    Create year-over-year growth chart.
    
    Args:
        combined_df: Combined historical and forecast data
        country: Country name
        
    Returns:
        Plotly figure
    """
    if combined_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for YoY growth", 
            xref="paper", yref="paper", 
            x=0.5, y=0.5
        )
        return fig
    
    # Calculate YoY growth
    value_col = safe_get_value_column(combined_df)
    if not value_col:
        fig = go.Figure()
        fig.add_annotation(
            text="No value column found", 
            xref="paper", yref="paper", 
            x=0.5, y=0.5
        )
        return fig
    
    combined_df = combined_df.sort_values('year')
    combined_df['yoy_growth'] = combined_df[value_col].pct_change() * 100
    
    # Create chart
    fig = go.Figure()
    
    # Color by type
    colors = combined_df['type'].map({
        'historical': '#2E86AB',
        'forecast': '#D4463D'
    }).fillna('#808080')
    
    fig.add_trace(go.Bar(
        x=combined_df['year'],
        y=combined_df['yoy_growth'],
        marker_color=colors,
        text=combined_df['yoy_growth'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else ""),
        textposition='outside'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
    
    # Add vertical line at cutoff
    hist_cutoff = get_hist_cutoff()
    fig.add_vline(x=hist_cutoff + 0.5, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"Year-over-Year Growth - {country}",
        xaxis_title="Year",
        yaxis_title="YoY Growth (%)",
        template="plotly_white",
        height=400,
        showlegend=False
    )
    
    return fig


def create_country_forecast_chart(combined_df: pd.DataFrame, 
                                 country: str,
                                 forecast_result: Dict = None) -> go.Figure:
    """
    Create forecast chart for a specific country.
    
    Args:
        combined_df: Combined historical and forecast data
        country: Country name
        forecast_result: Optional forecast result for additional data
        
    Returns:
        Plotly figure
    """
    hist_cutoff = get_hist_cutoff()
    
    # Split data
    hist_data = combined_df[combined_df['year'] <= hist_cutoff]
    fcst_data = combined_df[combined_df['year'] > hist_cutoff]
    
    fig = go.Figure()
    
    # Historical line
    if not hist_data.empty:
        value_col = safe_get_value_column(hist_data)
        if value_col:
            values = hist_data[value_col].values
            # Keep original values - let axis formatting handle units
            
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
            # Keep original values - let axis formatting handle units
            
            fig.add_trace(go.Scatter(
                x=fcst_data['year'],
                y=values,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#D4463D', width=3, dash='dash'),
                marker=dict(size=6)
            ))
    
    # Add baseline if available
    if forecast_result:
        baseline_forecasts = forecast_result.get('baseline_country_forecasts', {})
        if country in baseline_forecasts:
            baseline_df = baseline_forecasts[country]
            baseline_fcst = baseline_df[baseline_df['year'] > hist_cutoff]
            if not baseline_fcst.empty:
                value_col = safe_get_value_column(baseline_fcst)
                if value_col:
                    values = baseline_fcst[value_col].values
                    # Keep original values - let axis formatting handle units
                    
                    fig.add_trace(go.Scatter(
                        x=baseline_fcst['year'],
                        y=values,
                        mode='lines',
                        name='Baseline',
                        line=dict(color='#808080', width=2, dash='dot'),
                        opacity=0.7
                    ))
    
    # Add vertical line at cutoff
    fig.add_vline(x=hist_cutoff + 0.5, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"{country} Market Forecast",
        xaxis_title="Year",
        yaxis_title="Market Value ($B)",
        template="plotly_white",
        height=400,
        showlegend=True
    )
    
    return fig


def create_final_values_chart(countries: List[str], 
                             forecast_result: Dict) -> go.Figure:
    """
    Create bar chart comparing final forecast values.
    
    Args:
        countries: List of country names
        forecast_result: Complete forecast result
        
    Returns:
        Plotly figure
    """
    country_forecasts = forecast_result.get('country_forecasts', {})
    
    final_values = []
    
    for country in countries:
        if country not in country_forecasts:
            continue
        
        # Get country data
        if isinstance(country_forecasts[country], dict):
            country_df = country_forecasts[country].get('forecast', pd.DataFrame())
        else:
            country_df = country_forecasts[country]
        
        if country_df.empty:
            continue
        
        # Get final value
        fcst_data = country_df[country_df['year'] > get_hist_cutoff()]
        if not fcst_data.empty:
            value_col = safe_get_value_column(fcst_data)
            if value_col:
                final_val = fcst_data.iloc[-1][value_col]
                # Keep original final value
                final_values.append({
                    'Country': country,
                    'Final Value': final_val
                })
    
    if not final_values:
        fig = go.Figure()
        fig.add_annotation(
            text="No final value data available", 
            xref="paper", yref="paper", 
            x=0.5, y=0.5
        )
        return fig
    
    df = pd.DataFrame(final_values)
    df = df.sort_values('Final Value', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Final Value'],
        y=df['Country'],
        orientation='h',
        marker_color='#2E86AB',
        text=df['Final Value'].apply(lambda x: f"${x:.1f}B"),
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Final Market Values ({get_forecast_until()})",
        xaxis_title="Market Value ($B)",
        yaxis_title="Country",
        template="plotly_white",
        height=max(400, len(df) * 40),
        margin=dict(l=100)
    )
    
    return fig