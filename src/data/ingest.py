"""
Data ingestion and normalization functions.
"""

import pandas as pd
import numpy as np
from typing import Optional


def normalize_data(df: pd.DataFrame, 
                  date_column: str = 'date',
                  value_column: str = 'value') -> pd.DataFrame:
    """Normalize time series data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        date_column (str): Name of date column
        value_column (str): Name of value column
        
    Returns:
        pd.DataFrame: Normalized dataframe
    """
    if df.empty:
        return df
        
    df_clean = df.copy()
    
    # Convert date column to datetime
    if date_column in df_clean.columns:
        df_clean[date_column] = pd.to_datetime(df_clean[date_column])
        df_clean = df_clean.sort_values(date_column)
    
    # Handle missing values in value column
    if value_column in df_clean.columns:
        # Forward fill then backward fill
        df_clean[value_column] = df_clean[value_column].fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers (values beyond 3 standard deviations)
        mean_val = df_clean[value_column].mean()
        std_val = df_clean[value_column].std()
        if std_val > 0:
            outlier_mask = np.abs(df_clean[value_column] - mean_val) > 3 * std_val
            df_clean.loc[outlier_mask, value_column] = np.nan
            df_clean[value_column] = df_clean[value_column].fillna(method='ffill').fillna(method='bfill')
    
    return df_clean


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize column names.
    
    Args:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with cleaned column names
    """
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Clean column names: lowercase, replace spaces with underscores
    df_clean.columns = [col.lower().replace(' ', '_').replace('-', '_') 
                       for col in df_clean.columns]
    
    return df_clean


def standardize_date_formats(df: pd.DataFrame, 
                           date_columns: Optional[list] = None) -> pd.DataFrame:
    """Standardize date formats across columns.
    
    Args:
        df: Input dataframe
        date_columns: List of date column names. If None, auto-detect.
        
    Returns:
        pd.DataFrame: Dataframe with standardized dates
    """
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    if date_columns is None:
        # Auto-detect date columns
        date_columns = []
        for col in df_clean.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_columns.append(col)
    
    # Convert to datetime
    for col in date_columns:
        if col in df_clean.columns:
            try:
                df_clean[col] = pd.to_datetime(df_clean[col])
            except (ValueError, TypeError):
                print(f"Warning: Could not convert column {col} to datetime")
    
    return df_clean


def handle_duplicates(df: pd.DataFrame, 
                     subset: Optional[list] = None,
                     keep: str = 'last') -> pd.DataFrame:
    """Handle duplicate rows in the dataset.
    
    Args:
        df: Input dataframe
        subset: Columns to consider for identifying duplicates
        keep: Which duplicate to keep ('first', 'last', False to remove all)
        
    Returns:
        pd.DataFrame: Dataframe with duplicates handled
    """
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Remove duplicates
    before_count = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=subset, keep=keep)
    after_count = len(df_clean)
    
    if before_count != after_count:
        print(f"Removed {before_count - after_count} duplicate rows")
    
    return df_clean