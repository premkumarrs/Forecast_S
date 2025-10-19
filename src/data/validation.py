"""
Data quality validation functions.
"""

import pandas as pd
from typing import Dict, List, Any


def validate_data_quality(df: pd.DataFrame, 
                         required_columns: List[str]) -> Dict[str, Any]:
    """Validate data quality and return quality metrics.
    
    Args:
        df (pd.DataFrame): Input dataframe
        required_columns (List[str]): List of required column names
        
    Returns:
        Dict[str, Any]: Quality metrics and validation results
    """
    quality_report = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'metrics': {}
    }
    
    # Check if dataframe is empty
    if df.empty:
        quality_report['is_valid'] = False
        quality_report['errors'].append('DataFrame is empty')
        return quality_report
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        quality_report['is_valid'] = False
        quality_report['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Calculate quality metrics
    quality_report['metrics']['total_rows'] = len(df)
    quality_report['metrics']['total_columns'] = len(df.columns)
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_percentage = (null_count / len(df)) * 100
        
        quality_report['metrics'][f'{col}_null_count'] = null_count
        quality_report['metrics'][f'{col}_null_percentage'] = null_percentage
        
        # Warning for high null percentage
        if null_percentage > 20:
            quality_report['warnings'].append(f'Column {col} has {null_percentage:.1f}% null values')
    
    return quality_report


def validate_time_series_data(df: pd.DataFrame,
                            date_column: str,
                            value_column: str) -> Dict[str, Any]:
    """Validate time series specific requirements.
    
    Args:
        df: Input time series dataframe
        date_column: Name of date column
        value_column: Name of value column
        
    Returns:
        Dict[str, Any]: Validation results
    """
    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'metrics': {}
    }
    
    # Check required columns exist
    if date_column not in df.columns:
        validation['is_valid'] = False
        validation['errors'].append(f'Date column "{date_column}" not found')
    
    if value_column not in df.columns:
        validation['is_valid'] = False
        validation['errors'].append(f'Value column "{value_column}" not found')
    
    if not validation['is_valid']:
        return validation
    
    # Check date column
    try:
        date_series = pd.to_datetime(df[date_column])
        validation['metrics']['date_range'] = {
            'start': date_series.min().isoformat(),
            'end': date_series.max().isoformat()
        }
        
        # Check for gaps in time series
        date_diffs = date_series.diff().dropna()
        if len(date_diffs.unique()) > 1:
            validation['warnings'].append('Irregular time intervals detected')
        
    except (ValueError, TypeError) as e:
        validation['is_valid'] = False
        validation['errors'].append(f'Date column cannot be converted to datetime: {e}')
    
    # Check value column
    try:
        value_series = pd.to_numeric(df[value_column])
        validation['metrics']['value_stats'] = {
            'min': float(value_series.min()),
            'max': float(value_series.max()),
            'mean': float(value_series.mean()),
            'std': float(value_series.std())
        }
        
        # Check for negative values that might be unexpected
        negative_count = (value_series < 0).sum()
        if negative_count > 0:
            validation['warnings'].append(f'{negative_count} negative values found')
        
    except (ValueError, TypeError) as e:
        validation['is_valid'] = False
        validation['errors'].append(f'Value column cannot be converted to numeric: {e}')
    
    return validation


def check_data_completeness(df: pd.DataFrame,
                          expected_columns: List[str]) -> Dict[str, Any]:
    """Check data completeness against expected schema.
    
    Args:
        df: Input dataframe
        expected_columns: List of expected column names
        
    Returns:
        Dict[str, Any]: Completeness report
    """
    report = {
        'completeness_score': 0.0,
        'missing_columns': [],
        'extra_columns': [],
        'column_match_rate': 0.0
    }
    
    actual_columns = set(df.columns)
    expected_columns_set = set(expected_columns)
    
    # Find missing and extra columns
    report['missing_columns'] = list(expected_columns_set - actual_columns)
    report['extra_columns'] = list(actual_columns - expected_columns_set)
    
    # Calculate match rate
    matching_columns = len(expected_columns_set & actual_columns)
    total_expected = len(expected_columns)
    
    if total_expected > 0:
        report['column_match_rate'] = matching_columns / total_expected
    
    # Calculate overall completeness score
    # Penalize both missing columns and excess nulls
    base_score = report['column_match_rate']
    
    if not df.empty and matching_columns > 0:
        # Calculate average null rate for matching columns
        matching_cols = list(expected_columns_set & actual_columns)
        null_rates = [df[col].isnull().sum() / len(df) for col in matching_cols]
        avg_null_rate = sum(null_rates) / len(null_rates)
        
        # Adjust score based on data completeness
        report['completeness_score'] = base_score * (1 - avg_null_rate)
    else:
        report['completeness_score'] = base_score
    
    return report


def validate_value_ranges(df: pd.DataFrame,
                        column_ranges: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Validate that numeric columns fall within expected ranges.
    
    Args:
        df: Input dataframe
        column_ranges: Dict mapping column names to {'min': val, 'max': val}
        
    Returns:
        Dict[str, Any]: Range validation results
    """
    validation = {
        'is_valid': True,
        'violations': [],
        'warnings': []
    }
    
    for column, range_spec in column_ranges.items():
        if column not in df.columns:
            continue
        
        try:
            values = pd.to_numeric(df[column], errors='coerce')
            min_val = range_spec.get('min', float('-inf'))
            max_val = range_spec.get('max', float('inf'))
            
            # Check violations
            below_min = (values < min_val).sum()
            above_max = (values > max_val).sum()
            
            if below_min > 0:
                validation['is_valid'] = False
                validation['violations'].append(
                    f'Column {column}: {below_min} values below minimum ({min_val})'
                )
            
            if above_max > 0:
                validation['is_valid'] = False
                validation['violations'].append(
                    f'Column {column}: {above_max} values above maximum ({max_val})'
                )
                
        except (ValueError, TypeError):
            validation['warnings'].append(f'Could not validate range for column {column}')
    
    return validation