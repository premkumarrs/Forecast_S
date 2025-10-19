"""
Data transformation functions.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


def log_transform(df: pd.DataFrame,
                 columns: list,
                 add_constant: float = 1.0) -> pd.DataFrame:
    """Apply log transformation to specified columns.
    
    Args:
        df: Input dataframe
        columns: List of columns to transform
        add_constant: Constant to add before log (to handle zeros/negatives)
        
    Returns:
        pd.DataFrame: Dataframe with log-transformed columns
    """
    if df.empty:
        return df
    
    df_transformed = df.copy()
    
    for col in columns:
        if col in df_transformed.columns:
            # Add constant and take log
            df_transformed[f'{col}_log'] = np.log(df_transformed[col] + add_constant)
    
    return df_transformed


def difference_transform(df: pd.DataFrame,
                        columns: list,
                        periods: int = 1) -> pd.DataFrame:
    """Apply differencing transformation to make series stationary.
    
    Args:
        df: Input dataframe
        columns: List of columns to difference
        periods: Number of periods to difference
        
    Returns:
        pd.DataFrame: Dataframe with differenced columns
    """
    if df.empty:
        return df
    
    df_transformed = df.copy()
    
    for col in columns:
        if col in df_transformed.columns:
            df_transformed[f'{col}_diff_{periods}'] = df_transformed[col].diff(periods)
    
    return df_transformed


def normalize_columns(df: pd.DataFrame,
                     columns: list,
                     method: str = 'minmax') -> Tuple[pd.DataFrame, dict]:
    """Normalize specified columns.
    
    Args:
        df: Input dataframe
        columns: List of columns to normalize
        method: Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        Tuple[pd.DataFrame, dict]: Normalized dataframe and scaling parameters
    """
    if df.empty:
        return df, {}
    
    df_normalized = df.copy()
    scaling_params = {}
    
    for col in columns:
        if col not in df_normalized.columns:
            continue
            
        values = df_normalized[col].dropna()
        
        if method == 'minmax':
            min_val = values.min()
            max_val = values.max()
            if max_val != min_val:
                df_normalized[f'{col}_normalized'] = (df_normalized[col] - min_val) / (max_val - min_val)
                scaling_params[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}
            else:
                df_normalized[f'{col}_normalized'] = df_normalized[col]
                scaling_params[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}
                
        elif method == 'zscore':
            mean_val = values.mean()
            std_val = values.std()
            if std_val != 0:
                df_normalized[f'{col}_normalized'] = (df_normalized[col] - mean_val) / std_val
                scaling_params[col] = {'method': 'zscore', 'mean': mean_val, 'std': std_val}
            else:
                df_normalized[f'{col}_normalized'] = df_normalized[col] - mean_val
                scaling_params[col] = {'method': 'zscore', 'mean': mean_val, 'std': std_val}
                
        elif method == 'robust':
            median_val = values.median()
            mad_val = (values - median_val).abs().median()  # Median Absolute Deviation
            if mad_val != 0:
                df_normalized[f'{col}_normalized'] = (df_normalized[col] - median_val) / mad_val
                scaling_params[col] = {'method': 'robust', 'median': median_val, 'mad': mad_val}
            else:
                df_normalized[f'{col}_normalized'] = df_normalized[col] - median_val
                scaling_params[col] = {'method': 'robust', 'median': median_val, 'mad': mad_val}
    
    return df_normalized, scaling_params


def inverse_transform(df: pd.DataFrame,
                     columns: list,
                     scaling_params: dict) -> pd.DataFrame:
    """Inverse transform normalized columns back to original scale.
    
    Args:
        df: Input dataframe with normalized columns
        columns: List of columns to inverse transform
        scaling_params: Scaling parameters from normalize_columns
        
    Returns:
        pd.DataFrame: Dataframe with inverse transformed columns
    """
    if df.empty:
        return df
    
    df_inverse = df.copy()
    
    for col in columns:
        normalized_col = f'{col}_normalized'
        if normalized_col not in df_inverse.columns or col not in scaling_params:
            continue
        
        params = scaling_params[col]
        method = params['method']
        
        if method == 'minmax':
            min_val = params['min']
            max_val = params['max']
            df_inverse[f'{col}_original'] = (df_inverse[normalized_col] * (max_val - min_val)) + min_val
            
        elif method == 'zscore':
            mean_val = params['mean']
            std_val = params['std']
            df_inverse[f'{col}_original'] = (df_inverse[normalized_col] * std_val) + mean_val
            
        elif method == 'robust':
            median_val = params['median']
            mad_val = params['mad']
            df_inverse[f'{col}_original'] = (df_inverse[normalized_col] * mad_val) + median_val
    
    return df_inverse


def detect_outliers(df: pd.DataFrame,
                   columns: list,
                   method: str = 'iqr',
                   threshold: float = 1.5) -> pd.DataFrame:
    """Detect outliers in specified columns.
    
    Args:
        df: Input dataframe
        columns: List of columns to check for outliers
        method: Detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        pd.DataFrame: Dataframe with outlier flags added
    """
    if df.empty:
        return df
    
    df_with_outliers = df.copy()
    
    for col in columns:
        if col not in df_with_outliers.columns:
            continue
        
        values = df_with_outliers[col].dropna()
        outlier_col = f'{col}_outlier'
        
        if method == 'iqr':
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df_with_outliers[outlier_col] = (
                (df_with_outliers[col] < lower_bound) | 
                (df_with_outliers[col] > upper_bound)
            )
            
        elif method == 'zscore':
            mean_val = values.mean()
            std_val = values.std()
            if std_val > 0:
                z_scores = np.abs((df_with_outliers[col] - mean_val) / std_val)
                df_with_outliers[outlier_col] = z_scores > threshold
            else:
                df_with_outliers[outlier_col] = False
    
    return df_with_outliers