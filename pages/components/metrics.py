"""
Metric calculation functions for Streamlit pages.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from .utils import (
    safe_get_value_column,
    get_hist_cutoff,
    calculate_cagr,
    format_value_intelligent
)


def calculate_global_metrics(global_forecast: pd.DataFrame) -> Dict:
    """
    Calculate key metrics for global forecast.
    
    Args:
        global_forecast: DataFrame with global forecast data
        
    Returns:
        Dictionary containing metrics
    """
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
                metrics['hist_cagr'] = calculate_cagr(start_val, end_val, years)
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
                metrics['fcst_cagr'] = calculate_cagr(start_val, end_val, years)
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
    
    # Current value
    if not hist_data.empty:
        value_col = safe_get_value_column(hist_data)
        if value_col:
            current_val = hist_data.iloc[-1][value_col]
            metrics['current_value'] = current_val  # Keep original value
    else:
        metrics['current_value'] = 0
    
    # Growth percentage
    if metrics.get('current_value', 0) > 0 and metrics.get('final_value', 0) > 0:
        metrics['total_growth'] = (
            (metrics['final_value'] - metrics['current_value']) / metrics['current_value'] * 100
        )
    else:
        metrics['total_growth'] = 0
    
    return metrics


def calculate_regional_metrics(region_data: pd.DataFrame) -> Dict:
    """
    Calculate key metrics for regional forecast.
    
    Args:
        region_data: DataFrame with regional forecast data
        
    Returns:
        Dictionary containing metrics
    """
    # Reuse global metrics calculation as the logic is the same
    return calculate_global_metrics(region_data)


def calculate_country_metrics(combined_df: pd.DataFrame) -> Dict:
    """
    Calculate key metrics for country forecast.
    
    Args:
        combined_df: Combined historical and forecast data
        
    Returns:
        Dictionary containing metrics
    """
    # Reuse global metrics calculation as the logic is the same
    return calculate_global_metrics(combined_df)


def calculate_growth_impact_metrics(analyzed_news: pd.DataFrame) -> Dict:
    """
    Calculate growth impact metrics from analyzed news.
    
    Args:
        analyzed_news: DataFrame with analyzed news data
        
    Returns:
        Dictionary containing impact metrics
    """
    metrics = {
        'total_articles': 0,
        'relevant_articles': 0,
        'avg_growth_impact': 0,
        'positive_articles': 0,
        'negative_articles': 0,
        'neutral_articles': 0
    }
    
    if analyzed_news is None or analyzed_news.empty:
        return metrics
    
    metrics['total_articles'] = len(analyzed_news)
    
    # Count relevant articles
    if 'relevant' in analyzed_news.columns:
        metrics['relevant_articles'] = analyzed_news['relevant'].sum()
    
    # Calculate average growth impact
    if 'growth_rate' in analyzed_news.columns:
        growth_rates = pd.to_numeric(analyzed_news['growth_rate'], errors='coerce')
        metrics['avg_growth_impact'] = growth_rates.mean()
        
        # Count by sentiment
        metrics['positive_articles'] = (growth_rates > 0).sum()
        metrics['negative_articles'] = (growth_rates < 0).sum()
        metrics['neutral_articles'] = (growth_rates == 0).sum()
    
    # Category breakdown
    if 'category' in analyzed_news.columns:
        category_counts = analyzed_news['category'].value_counts().to_dict()
        metrics['category_breakdown'] = category_counts
    
    # Temporal impact breakdown
    if 'temporal_impact' in analyzed_news.columns:
        temporal_counts = analyzed_news['temporal_impact'].value_counts().to_dict()
        metrics['temporal_breakdown'] = temporal_counts
    
    return metrics


def create_metrics_comparison_table(countries: List[str], 
                                   forecast_result: Dict) -> pd.DataFrame:
    """
    Create comparison table of metrics for multiple countries.
    
    Args:
        countries: List of country names
        forecast_result: Complete forecast result
        
    Returns:
        DataFrame with comparison metrics
    """
    country_forecasts = forecast_result.get('country_forecasts', {})
    
    comparison_data = []
    
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
        
        # Calculate metrics
        metrics = calculate_country_metrics(country_df)
        
        comparison_data.append({
            'Country': country,
            'Current Value ($B)': f"{metrics.get('current_value', 0):.1f}",
            'Final Value ($B)': f"{metrics.get('final_value', 0):.1f}",
            'Total Growth (%)': f"{metrics.get('total_growth', 0):.1f}",
            'Historical CAGR (%)': f"{metrics.get('hist_cagr', 0):.1f}",
            'Forecast CAGR (%)': f"{metrics.get('fcst_cagr', 0):.1f}"
        })
    
    if not comparison_data:
        return pd.DataFrame()
    
    return pd.DataFrame(comparison_data)


def calculate_adjustment_impact(forecast_result: Dict) -> Dict:
    """
    Calculate the impact of adjustments on the forecast.
    
    Args:
        forecast_result: Complete forecast result
        
    Returns:
        Dictionary containing adjustment impact metrics
    """
    metrics = {
        'news_impact': 0,
        'indicator_impact': 0,
        'total_impact': 0,
        'adjustments_applied': []
    }
    
    # Get adjustment details
    adjustment_details = forecast_result.get('adjustment_details', {})
    
    if not adjustment_details:
        return metrics
    
    # Aggregate impacts
    total_news = 0
    total_indicator = 0
    count = 0
    
    for country, details in adjustment_details.items():
        if isinstance(details, dict):
            news_contrib = details.get('news_contribution', 0)
            indicator_contrib = details.get('indicator_contribution', 0)
            
            total_news += abs(news_contrib)
            total_indicator += abs(indicator_contrib)
            count += 1
    
    if count > 0:
        metrics['news_impact'] = total_news / count
        metrics['indicator_impact'] = total_indicator / count
        metrics['total_impact'] = metrics['news_impact'] + metrics['indicator_impact']
    
    metrics['adjustments_applied'] = forecast_result.get('adjustments_applied', [])
    
    return metrics


def calculate_forecast_accuracy_metrics(historical_df: pd.DataFrame,
                                       forecast_df: pd.DataFrame,
                                       overlap_years: List[int] = None) -> Dict:
    """
    Calculate accuracy metrics for forecasts with historical overlap.
    
    Args:
        historical_df: Historical data
        forecast_df: Forecast data
        overlap_years: Years where both exist for comparison
        
    Returns:
        Dictionary containing accuracy metrics
    """
    metrics = {
        'mape': None,  # Mean Absolute Percentage Error
        'rmse': None,  # Root Mean Square Error
        'mae': None,   # Mean Absolute Error
        'r_squared': None
    }
    
    if historical_df.empty or forecast_df.empty:
        return metrics
    
    if overlap_years is None:
        # Find overlapping years
        hist_years = set(historical_df['year'].unique())
        fcst_years = set(forecast_df['year'].unique())
        overlap_years = list(hist_years.intersection(fcst_years))
    
    if not overlap_years:
        return metrics
    
    # Get values for overlapping years
    hist_overlap = historical_df[historical_df['year'].isin(overlap_years)].sort_values('year')
    fcst_overlap = forecast_df[forecast_df['year'].isin(overlap_years)].sort_values('year')
    
    hist_value_col = safe_get_value_column(hist_overlap)
    fcst_value_col = safe_get_value_column(fcst_overlap)
    
    if not hist_value_col or not fcst_value_col:
        return metrics
    
    hist_values = hist_overlap[hist_value_col].values
    fcst_values = fcst_overlap[fcst_value_col].values
    
    if len(hist_values) != len(fcst_values) or len(hist_values) == 0:
        return metrics
    
    # Calculate metrics
    errors = fcst_values - hist_values
    
    # MAPE
    non_zero_mask = hist_values != 0
    if non_zero_mask.any():
        mape_values = np.abs(errors[non_zero_mask] / hist_values[non_zero_mask]) * 100
        metrics['mape'] = np.mean(mape_values)
    
    # RMSE
    metrics['rmse'] = np.sqrt(np.mean(errors ** 2))
    
    # MAE
    metrics['mae'] = np.mean(np.abs(errors))
    
    # R-squared
    if len(hist_values) > 1:
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((hist_values - np.mean(hist_values)) ** 2)
        if ss_tot != 0:
            metrics['r_squared'] = 1 - (ss_res / ss_tot)
    
    return metrics