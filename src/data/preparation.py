"""
Time series preparation functions.
"""

import pandas as pd
from typing import Optional, Tuple


def prepare_time_series(df: pd.DataFrame,
                       date_column: str = 'date',
                       value_column: str = 'value',
                       freq: str = 'M') -> pd.DataFrame:
    """Prepare time series data for forecasting.
    
    Args:
        df (pd.DataFrame): Input dataframe
        date_column (str): Name of date column
        value_column (str): Name of value column
        freq (str): Frequency for resampling ('M' for monthly, 'Q' for quarterly)
        
    Returns:
        pd.DataFrame: Prepared time series dataframe
    """
    if df.empty:
        return df
        
    df_ts = df.copy()
    
    # Ensure datetime index
    if date_column in df_ts.columns:
        df_ts[date_column] = pd.to_datetime(df_ts[date_column])
        df_ts = df_ts.set_index(date_column)
    
    # Resample to specified frequency
    if value_column in df_ts.columns:
        df_ts = df_ts.resample(freq)[value_column].mean().to_frame()
        df_ts = df_ts.dropna()
    
    return df_ts


def split_train_test(df: pd.DataFrame,
                    test_size: float = 0.2,
                    date_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into training and testing sets.
    
    Args:
        df (pd.DataFrame): Input dataframe
        test_size (float): Proportion of data for testing
        date_column (str, optional): Date column for time-based split
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing dataframes
    """
    if df.empty:
        return df, df
        
    if date_column and date_column in df.columns:
        # Time-based split
        df_sorted = df.sort_values(date_column)
        split_idx = int(len(df_sorted) * (1 - test_size))
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]
    else:
        # Random split
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
    
    return train_df, test_df


def create_lag_features(df: pd.DataFrame,
                       value_column: str,
                       lags: list = [1, 2, 3]) -> pd.DataFrame:
    """Create lagged features for time series analysis.
    
    Args:
        df: Input dataframe with time series data
        value_column: Name of the value column to create lags for
        lags: List of lag periods to create
        
    Returns:
        pd.DataFrame: Dataframe with lag features added
    """
    if df.empty or value_column not in df.columns:
        return df
    
    df_with_lags = df.copy()
    
    for lag in lags:
        lag_col = f'{value_column}_lag_{lag}'
        df_with_lags[lag_col] = df_with_lags[value_column].shift(lag)
    
    return df_with_lags


def create_rolling_features(df: pd.DataFrame,
                          value_column: str,
                          windows: list = [3, 6, 12]) -> pd.DataFrame:
    """Create rolling statistical features.
    
    Args:
        df: Input dataframe with time series data
        value_column: Name of the value column
        windows: List of window sizes for rolling statistics
        
    Returns:
        pd.DataFrame: Dataframe with rolling features added
    """
    if df.empty or value_column not in df.columns:
        return df
    
    df_with_rolling = df.copy()
    
    for window in windows:
        # Rolling mean
        df_with_rolling[f'{value_column}_roll_mean_{window}'] = (
            df_with_rolling[value_column].rolling(window=window).mean()
        )
        
        # Rolling standard deviation
        df_with_rolling[f'{value_column}_roll_std_{window}'] = (
            df_with_rolling[value_column].rolling(window=window).std()
        )
        
        # Rolling min and max
        df_with_rolling[f'{value_column}_roll_min_{window}'] = (
            df_with_rolling[value_column].rolling(window=window).min()
        )
        
        df_with_rolling[f'{value_column}_roll_max_{window}'] = (
            df_with_rolling[value_column].rolling(window=window).max()
        )
    
    return df_with_rolling


def resample_time_series(df: pd.DataFrame,
                        date_column: str,
                        freq: str,
                        agg_method: str = 'mean') -> pd.DataFrame:
    """Resample time series to different frequency.
    
    Args:
        df: Input dataframe
        date_column: Name of date column
        freq: Target frequency ('D', 'W', 'M', 'Q', 'Y')
        agg_method: Aggregation method ('mean', 'sum', 'min', 'max')
        
    Returns:
        pd.DataFrame: Resampled dataframe
    """
    if df.empty or date_column not in df.columns:
        return df
    
    df_resampled = df.copy()
    df_resampled[date_column] = pd.to_datetime(df_resampled[date_column])
    df_resampled = df_resampled.set_index(date_column)
    
    # Select aggregation method
    agg_func = {
        'mean': 'mean',
        'sum': 'sum',
        'min': 'min',
        'max': 'max',
        'first': 'first',
        'last': 'last'
    }.get(agg_method, 'mean')
    
    # Resample
    df_resampled = df_resampled.resample(freq).agg(agg_func)
    df_resampled = df_resampled.dropna(how='all')
    
    # Reset index to get date column back
    df_resampled = df_resampled.reset_index()
    
    return df_resampled