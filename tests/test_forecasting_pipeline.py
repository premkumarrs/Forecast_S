"""
End-to-end test for the forecasting pipeline in Phase 2.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Add project root to Python path to allow for module imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Mock Streamlit before importing app modules
# This prevents Streamlit from trying to run in a real session
mock_st = MagicMock()
sys.modules['streamlit'] = mock_st

from src.forecasting import generate_baseline_forecast, apply_indicator_adjustment, apply_temporal_news_adjustment
from src.llm import LLM_Analyst
from src.news import fetch_max_gdelt_articles

# --- Test Setup ---

@pytest.fixture
def mock_session_state():
    """Creates a mock session state with sample data from Page 1."""
    state = {
        'page1_ready': True,
        'hist_cutoff': 2023,
        'forecast_until': 2028,
        'global_ts': pd.DataFrame({
            'year': range(2018, 2023),
            'value': [100, 110, 125, 140, 160]
        }),
        'global_indicators': pd.DataFrame({
            'year': list(range(2018, 2029)) * 2,
            'indicator_key': ['gdp_growth'] * 11 + ['unemployment'] * 11,
            'value': np.random.rand(22) * 5
        }),
        'market_name': 'Global Test Market'
    }
    
    # Use patch to replace streamlit's session_state with our mock
    with patch('streamlit.session_state', new=state):
        yield state

@pytest.fixture
def mock_llm_analysis():
    """Provides a mock of the LLM analysis results."""
    return pd.DataFrame({
        'date': pd.to_datetime(['2024-01-15', '2024-02-20']),
        'year': [2024, 2024],
        'title': ['Major New Factory Announced', 'Supply Chain Hit By Strike'],
        'category': ['Strategic Investment', 'Supply Chain Disruption'],
        'magnitude': [1.8, -1.5],
        'growth_rate': [0.05, -0.03],
        'temporal_impact': ['long-term', 'short-term'],
        'relevant': [1, 1],
        'reason': ['Positive long-term impact', 'Negative short-term impact']
    })

# --- Test Cases ---

def test_pipeline_runs_without_errors(mock_session_state, mock_llm_analysis):
    """
    Tests that the full forecasting pipeline can be executed without raising errors.
    """
    try:
        # 1. Get data from session
        global_ts = mock_session_state['global_ts']
        global_indicators = mock_session_state['global_indicators']
        hist_cutoff = mock_session_state['hist_cutoff']
        forecast_until = mock_session_state['forecast_until']
        
        # 2. Define user inputs for Phase 2
        baseline_model_type = 'Damped ETS'
        indicator_weights = {'gdp_growth': 0.7, 'unemployment': -0.3}
        indicator_sensitivity = 1.0
        
        # 3. Generate Baseline Forecast
        baseline_df = generate_baseline_forecast(
            global_ts, baseline_model_type, hist_cutoff, forecast_until
        )
        
        # 4. Apply Indicator Weights
        indicators_only_df = apply_indicator_adjustment(
            baseline_df, global_indicators, indicator_weights, indicator_sensitivity
        )
        
        # 5. Apply News Shocks
        news_only_df = apply_temporal_news_adjustment(
            baseline_df, mock_llm_analysis
        )
        
        # 6. Create Combined Forecast
        combined_df = apply_temporal_news_adjustment(
            indicators_only_df, mock_llm_analysis
        )
        
        # 7. Assertions
        assert not baseline_df.empty
        assert not indicators_only_df.empty
        assert not news_only_df.empty
        assert not combined_df.empty
        
        print("Pipeline ran successfully. All forecast dataframes were generated.")

    except Exception as e:
        pytest.fail(f"The forecasting pipeline failed with an unexpected exception: {e}")

def test_forecast_outputs_have_correct_structure(mock_session_state, mock_llm_analysis):
    """
    Tests that the output DataFrames have the correct columns and types.
    """
    # --- Setup: Run the pipeline ---
    global_ts = mock_session_state['global_ts']
    global_indicators = mock_session_state['global_indicators']
    hist_cutoff = mock_session_state['hist_cutoff']
    forecast_until = mock_session_state['forecast_until']
    baseline_model_type = 'Damped ETS'
    indicator_weights = {'gdp_growth': 0.7, 'unemployment': -0.3}
    indicator_sensitivity = 1.0
    
    baseline_df = generate_baseline_forecast(global_ts, baseline_model_type, hist_cutoff, forecast_until)
    indicators_only_df = apply_indicator_adjustment(baseline_df, global_indicators, indicator_weights, indicator_sensitivity)
    news_only_df = apply_temporal_news_adjustment(baseline_df, mock_llm_analysis)
    combined_df = apply_temporal_news_adjustment(indicators_only_df, mock_llm_analysis)
    
    forecasts = {
        "Baseline": baseline_df,
        "Indicators Only": indicators_only_df,
        "News Only": news_only_df,
        "Combined": combined_df
    }
    
    # --- Assertions ---
    expected_columns = ['year', 'value_hat', 'type']
    
    for name, df in forecasts.items():
        print(f"Checking structure of '{name}' forecast...")
        assert isinstance(df, pd.DataFrame), f"{name} should be a pandas DataFrame"
        assert list(df.columns) == expected_columns, f"{name} has incorrect columns"
        
        # Check dtypes
        assert pd.api.types.is_integer_dtype(df['year']), f"{name} 'year' column should be integer"
        assert pd.api.types.is_numeric_dtype(df['value_hat']), f"{name} 'value_hat' column should be numeric"
        assert pd.api.types.is_string_dtype(df['type']), f"{name} 'type' column should be string"
        
        # Check for historical and forecast types
        assert 'Historical' in df['type'].values
        assert 'Forecast' in df['type'].values
        
        # Check that forecast values are not all zero
        assert df[df['type'] == 'Forecast']['value_hat'].sum() > 0, f"{name} forecast values are all zero"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
