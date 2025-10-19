"""
Session module for state management and data serialization.
Simplified - use st.session_state directly for basic operations.
"""

from .state import (
    init_session_state, get_session_summary,
    save_to_session, get_from_session, clear_session_data  # Deprecated
)
from .serialization import (
    save_dataframe, get_dataframe, save_forecast_results,
    get_forecast_results, get_forecast_metadata, 
    export_session_state, import_session_state
)
from .validation import (
    validate_required_data, validate_data_consistency,
    validate_forecast_readiness
)
from .exports import (
    export_to_json, import_from_json, export_forecast_summary
)

__all__ = [
    # Core functions
    'init_session_state', 'get_session_summary',
    
    # Deprecated (use st.session_state directly)
    'save_to_session', 'get_from_session', 'clear_session_data',
    
    # Serialization functions  
    'save_dataframe', 'get_dataframe', 'save_forecast_results',
    'get_forecast_results', 'get_forecast_metadata',
    'export_session_state', 'import_session_state',
    
    # Validation functions
    'validate_required_data', 'validate_data_consistency', 
    'validate_forecast_readiness',
    
    # Export functions
    'export_to_json', 'import_from_json', 'export_forecast_summary'
]