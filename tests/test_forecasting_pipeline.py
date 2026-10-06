"""
End-to-end test for the forecasting pipeline: baseline forecast, indicator and news
adjustments, and the Top-Down method.
"""
import sys
import os
import types
import pytest
import pandas as pd
from unittest.mock import MagicMock

# Add project root to Python path to allow for module imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.forecasting import generate_baseline_forecast, calculate_unified_adjustment, forecast_top_down

# --- Test Setup ---

@pytest.fixture(autouse=True)
def plain_streamlit_session(monkeypatch):
    """calculate_unified_adjustment reads st.session_state; give it an empty, real dict."""
    monkeypatch.setitem(sys.modules, 'streamlit', types.SimpleNamespace(session_state={}))


@pytest.fixture
def mock_session_state():
    """Sample data as produced by the Data Extraction page."""
    return {
        'hist_cutoff': 2023,
        'forecast_until': 2028,
        'global_ts': pd.DataFrame({
            'year': range(2018, 2023),
            'value': [100, 110, 125, 140, 160]
        }),
        'global_indicators': pd.DataFrame({
            'year': list(range(2018, 2023)) * 2,
            'indicator_key': ['gdp_growth'] * 5 + ['unemployment'] * 5,
            'indicator_name': ['GDP'] * 5 + ['Unemployment'] * 5,
            'value': [1.00, 1.03, 1.07, 1.10, 1.15, 5.0, 5.2, 4.9, 4.7, 4.6]
        }),
        'market_name': 'Global Test Market'
    }

@pytest.fixture
def mock_llm_analysis():
    """Analyzed headlines in the schema returned by SimpleLLMAnalyst."""
    return pd.DataFrame({
        'date': pd.to_datetime(['2022-11-15', '2022-12-20']),
        'title': ['Major New Factory Announced', 'Supply Chain Hit By Strike'],
        'category': ['Infrastructure Investment', 'Supply Chain Disruption'],
        'growth_rate': [5.0, -3.0],
        'reason': ['Positive long-term impact', 'Negative short-term impact'],
        'relevant': [1, 1]
    })

@pytest.fixture
def adjustment_config(mock_session_state):
    return {
        'hist_cutoff': mock_session_state['hist_cutoff'],
        'forecast_until': mock_session_state['forecast_until'],
        'adjustment_weights': {'news_weight': 0.7, 'indicator_weight': 0.3},
        'indicator_weights': {'gdp_growth': 0.7, 'unemployment': 0.3},
    }


def _run_pipeline(state, news, config):
    baseline_df = generate_baseline_forecast(
        state['global_ts'], 'Damped ETS', state['hist_cutoff'], state['forecast_until']
    )
    indicators_only_df, _ = calculate_unified_adjustment(baseline_df, None, state['global_indicators'], config)
    news_only_df, _ = calculate_unified_adjustment(baseline_df, news, None, config)
    combined_df, _ = calculate_unified_adjustment(baseline_df, news, state['global_indicators'], config)
    return {
        "Baseline": baseline_df,
        "Indicators Only": indicators_only_df,
        "News Only": news_only_df,
        "Combined": combined_df
    }

# --- Test Cases ---

def test_pipeline_runs_without_errors(mock_session_state, mock_llm_analysis, adjustment_config):
    """
    Tests that the full forecasting pipeline can be executed without raising errors.
    """
    forecasts = _run_pipeline(mock_session_state, mock_llm_analysis, adjustment_config)

    for name, df in forecasts.items():
        assert not df.empty, f"{name} forecast is empty"

    baseline = forecasts["Baseline"]
    is_forecast = baseline['type'] == 'Forecast'
    # Adjustments change forecast years only, never the historical values
    for name in ("Indicators Only", "News Only", "Combined"):
        adjusted = forecasts[name]
        pd.testing.assert_series_equal(
            adjusted.loc[~is_forecast, 'value_hat'], baseline.loc[~is_forecast, 'value_hat']
        )
        assert not adjusted.loc[is_forecast, 'value_hat'].equals(baseline.loc[is_forecast, 'value_hat']), (
            f"{name} should differ from the baseline in forecast years"
        )

def test_forecast_outputs_have_correct_structure(mock_session_state, mock_llm_analysis, adjustment_config):
    """
    Tests that the output DataFrames have the correct columns and types.
    """
    forecasts = _run_pipeline(mock_session_state, mock_llm_analysis, adjustment_config)

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

def test_top_down_distributes_global_baseline_by_share():
    """Regression: classic Top-Down scales the global baseline by each country's historical share."""
    years = list(range(2015, 2023))
    global_values = [100.0, 110.0, 121.0, 133.0, 146.0, 161.0, 177.0, 195.0]
    country_market = pd.DataFrame(
        [(name, iso3, year, value * share)
         for name, iso3, share in [('United States', 'USA', 0.6), ('Germany', 'DEU', 0.4)]
         for year, value in zip(years, global_values)],
        columns=['country', 'iso3', 'year', 'value']
    )
    unified_data = {
        'market_value': {'global': pd.DataFrame({'year': years, 'value': global_values}), 'country': country_market},
        'indicators': {'global': pd.DataFrame(), 'country': pd.DataFrame()},
    }
    config = {
        'forecast_mode': 'classic',
        'forecast_method': 'Damped ETS',
        'hist_cutoff': 2022,
        'forecast_until': 2026,
        'adjustment_weights': {'news_weight': 0.7, 'indicator_weight': 0.3},
    }

    result = forecast_top_down(unified_data, config, analyzed_news=pd.DataFrame(), country_news={})

    assert 'error' not in result
    shares = result['metadata']['country_shares']
    assert shares == pytest.approx({'United States': 0.6, 'Germany': 0.4})
    global_baseline = result['baseline_global_forecast']['value_hat'].reset_index(drop=True)
    for country, share in shares.items():
        country_baseline = result['baseline_country_forecasts'][country]['value_hat'].reset_index(drop=True)
        pd.testing.assert_series_equal(country_baseline, global_baseline * share, check_names=False)
    assert set(result['country_forecasts']) == {'United States', 'Germany'}

@pytest.fixture
def export_helpers(monkeypatch):
    """Import the export helpers under a Streamlit mock and unload them afterwards."""
    monkeypatch.setitem(sys.modules, 'streamlit', MagicMock())
    loaded_before = set(sys.modules)
    from pages.helpers import export as export_module
    from pages.components import utils as component_utils
    monkeypatch.setattr(component_utils, 'st', types.SimpleNamespace(session_state={'hist_cutoff': 2020}))
    yield export_module
    for name in set(sys.modules) - loaded_before:
        if name == 'pages' or name.startswith('pages.'):
            del sys.modules[name]

def test_market_export_keeps_countries_without_idgeo(export_helpers):
    """Regression: country rows without idGeo (demo data) must not be dropped from the market export."""
    years = list(range(2018, 2023))
    country_market = pd.DataFrame(
        [(name, iso3, year, value)
         for name, iso3, base in [('United States', 'USA', 60.0), ('Germany', 'DEU', 40.0)]
         for year, value in zip(years, [base * (1.1 ** i) for i in range(len(years))])],
        columns=['country', 'iso3', 'year', 'value']
    )
    unified_data = {'market_value': {'country': country_market, 'global': pd.DataFrame(),
                                     'metadata': {'kpi_key': 'K1', 'kpi_name': 'Test market'}}}
    forecast_result = {
        'country_forecasts': {
            name: pd.DataFrame({'year': [2021, 2022, 2023], 'value_hat': [1.0, 2.0, 3.0],
                                'type': ['Forecast'] * 3})
            for name in ('United States', 'Germany')
        }
    }

    table = export_helpers.build_market_wide_table(forecast_result, unified_data)

    assert sorted(table['nameGeo']) == ['Germany', 'United States']
    us = table[table['nameGeo'] == 'United States'].iloc[0]
    assert us[2020] == pytest.approx(60.0 * 1.1 ** 2)  # historical value kept as-is
    assert us[2021] == 1.0 and us[2023] == 3.0  # forecast years after the cutoff

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
