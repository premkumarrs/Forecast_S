"""
Backwards compatibility utilities for handling old LT/ST format data.

DEPRECATED: This module provides compatibility with the old dual-rate (LT/ST) 
system. The system now uses a single-rate model. These functions are kept 
for backwards compatibility with cached results and old configurations.

The single-rate model replaces:
- lt_news_pct + st_news_pct → news_pct
- temporal_impact field → removed entirely
- Dual decay curves → single exponential decay

New implementations should use the single-rate model directly.
"""

import pandas as pd
from typing import Dict, Any, Optional


def get_news_adjustment_value(year_data: Dict[str, Any]) -> float:
    """
    DEPRECATED: Get news adjustment value from year data, handling both old and new formats.
    
    This function combines old LT/ST format data for backwards compatibility.
    New code should use the single 'news_pct' field directly.
    
    Args:
        year_data: Dictionary containing year adjustment data
        
    Returns:
        News adjustment percentage value
    """
    # Try new format first
    if 'news_pct' in year_data:
        return year_data['news_pct']
    
    # Fall back to old format (sum of LT and ST)
    lt_value = year_data.get('lt_news_pct', 0.0)
    st_value = year_data.get('st_news_pct', 0.0)
    
    # If both are present, sum them
    if lt_value or st_value:
        return lt_value + st_value
    
    # Default to 0
    return 0.0


def get_decay_value(year_data: Dict[str, Any]) -> float:
    """
    Get decay value from year data, handling both old and new formats.
    
    Args:
        year_data: Dictionary containing year adjustment data
        
    Returns:
        Decay value (0-1)
    """
    # Try new format first
    if 'decay' in year_data:
        return year_data['decay']
    
    # Fall back to old format (average of LT and ST decay)
    lt_decay = year_data.get('lt_decay', 1.0)
    st_decay = year_data.get('st_decay', 1.0)
    
    # Return average of both
    return (lt_decay + st_decay) / 2.0


def convert_old_details_to_new(old_details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert old LT/ST format details to new single-rate format.
    
    Args:
        old_details: Dictionary with old format adjustment details
        
    Returns:
        Dictionary with new format adjustment details
    """
    new_details = old_details.copy()
    
    # Convert top-level adjustments
    if 'lt_news_adjustment' in new_details or 'st_news_adjustment' in new_details:
        lt_adj = new_details.pop('lt_news_adjustment', 0.0)
        st_adj = new_details.pop('st_news_adjustment', 0.0)
        # Use average as the single news adjustment
        new_details['news_adjustment'] = (lt_adj + st_adj) / 2.0
    
    # Convert year adjustments
    if 'year_adjustments' in new_details:
        for year, year_data in new_details['year_adjustments'].items():
            # Convert news percentages
            if 'lt_news_pct' in year_data or 'st_news_pct' in year_data:
                lt_pct = year_data.pop('lt_news_pct', 0.0)
                st_pct = year_data.pop('st_news_pct', 0.0)
                year_data['news_pct'] = lt_pct + st_pct
            
            # Convert decay values
            if 'lt_decay' in year_data or 'st_decay' in year_data:
                lt_decay = year_data.pop('lt_decay', 1.0)
                st_decay = year_data.pop('st_decay', 1.0)
                year_data['decay'] = (lt_decay + st_decay) / 2.0
            
            # Update total_pct if needed
            if 'total_pct' not in year_data:
                news_pct = year_data.get('news_pct', 0.0)
                indicators_pct = year_data.get('indicators_pct', 0.0)
                year_data['total_pct'] = news_pct + indicators_pct
    
    return new_details


def ensure_dataframe_compatibility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has compatible column names for new format.
    
    Args:
        df: DataFrame with potentially old column names
        
    Returns:
        DataFrame with updated column names
    """
    if df.empty:
        return df
    
    # Copy to avoid modifying original
    result_df = df.copy()
    
    # Rename old columns to new format
    column_mapping = {
        'LT_News_Pct': 'News_Pct',
        'ST_News_Pct': None,  # Remove this column
        'lt_news_pct': 'news_pct',
        'st_news_pct': None,  # Remove this column
        'LT News (%)': 'News (%)',
        'ST News (%)': None,  # Remove this column
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in result_df.columns:
            if new_col:
                # If ST column exists too, combine them
                if old_col.startswith('LT') and old_col.replace('LT', 'ST') in result_df.columns:
                    st_col = old_col.replace('LT', 'ST')
                    result_df[new_col] = result_df[old_col].fillna(0) + result_df[st_col].fillna(0)
                    result_df = result_df.drop(columns=[old_col, st_col])
                else:
                    result_df = result_df.rename(columns={old_col: new_col})
            else:
                # Remove column
                result_df = result_df.drop(columns=[old_col])
    
    # Remove temporal_impact column if present
    if 'temporal_impact' in result_df.columns:
        result_df = result_df.drop(columns=['temporal_impact'])
    
    return result_df


def safe_get_adjustment_value(data: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """
    Safely get adjustment value from data dictionary.
    
    Args:
        data: Dictionary containing adjustment data
        key: Key to retrieve
        default: Default value if key not found
        
    Returns:
        Adjustment value or default
    """
    if key == 'news_pct':
        return get_news_adjustment_value(data)
    elif key == 'decay':
        return get_decay_value(data)
    else:
        return data.get(key, default)