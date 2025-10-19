"""
Export/import functionality for session data.
"""

import json
import pandas as pd
from typing import Dict, Any, Optional
from .serialization import export_session_state, import_session_state
from datetime import datetime


def export_to_json(filename: Optional[str] = None) -> str:
    """Export session state to JSON format.
    
    Args:
        filename (str, optional): Filename to save to. If None, returns JSON string.
        
    Returns:
        str: JSON string of session data
    """
    session_data = export_session_state()
    
    # Add export metadata
    export_data = {
        'metadata': {
            'export_timestamp': datetime.now().isoformat(),
            'export_version': '1.0'
        },
        'session_data': session_data
    }
    
    json_str = json.dumps(export_data, indent=2)
    
    if filename:
        with open(filename, 'w') as f:
            f.write(json_str)
    
    return json_str


def import_from_json(json_data: str) -> bool:
    """Import session state from JSON data.
    
    Args:
        json_data (str): JSON string containing session data
        
    Returns:
        bool: True if import successful, False otherwise
    """
    try:
        data = json.loads(json_data)
        
        # Handle both old format (direct session data) and new format (with metadata)
        if 'session_data' in data:
            session_data = data['session_data']
        else:
            session_data = data
        
        import_session_state(session_data)
        return True
        
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error importing session data: {e}")
        return False


def export_forecast_summary(forecasts: Dict[str, pd.Series]) -> Dict[str, Any]:
    """Export a summary of forecast results.
    
    Args:
        forecasts (Dict[str, pd.Series]): Forecast results
        
    Returns:
        Dict[str, Any]: Summary data
    """
    summary = {
        'export_timestamp': datetime.now().isoformat(),
        'total_forecasts': len(forecasts),
        'forecast_keys': list(forecasts.keys()),
        'summary_stats': {}
    }
    
    for key, series in forecasts.items():
        if len(series) > 0:
            summary['summary_stats'][key] = {
                'min': float(series.min()),
                'max': float(series.max()),
                'mean': float(series.mean()),
                'start_value': float(series.iloc[0]),
                'end_value': float(series.iloc[-1]),
                'total_change_pct': float(((series.iloc[-1] / series.iloc[0]) - 1) * 100)
            }
    
    return summary