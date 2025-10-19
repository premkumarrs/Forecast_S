"""
Unit test for the LLM headline analysis functionality.
"""
import sys
import os
import pytest
import json
import pandas as pd
from unittest.mock import patch, MagicMock

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Mock Streamlit before importing app modules
sys.modules['streamlit'] = MagicMock()

from src.llm import LLM_Analyst

@pytest.fixture
def sample_headlines():
    """Provides a sample list of headlines for testing."""
    return [
        {'title': 'New Factory Investment to Boost Production by 50%', 'date': '2024-01-10'},
        {'title': 'Geopolitical Tensions Halt Key Shipments', 'date': '2024-02-15'},
    ]

@pytest.fixture
def mock_api_responses():
    """Provides a mapping of headlines to their expected LLM API responses."""
    return {
        'New Factory Investment to Boost Production by 50%': {
            "category": "Strategic Investment",
            "magnitude": 1.8,
            "reason": "Significant capital expenditure indicates strong future growth."
        },
        'Geopolitical Tensions Halt Key Shipments': {
            "category": "Geopolitical Tension",
            "magnitude": -1.5,
            "reason": "Trade disruption creates near-term negative impact."
        }
    }

@patch.dict(os.environ, {'LLM_PROVIDER': 'ollama', 'OLLAMA_BASE_URL': 'http://localhost:11434', 'OLLAMA_MODEL': 'test'})
@patch('requests.post')
def test_analyze_headlines(mock_post, sample_headlines, mock_api_responses):
    """Tests the LLM_Analyst.analyze_headlines method by mocking the HTTP requests."""
    # --- Setup the Mock ---
    def get_mock_response(*args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        
        # Extract the prompt from the request
        payload = kwargs.get('json', {})
        prompt = payload.get('prompt', '')
        
        # Extract headline from prompt - look for title in the prompt
        headline = None
        for sample in sample_headlines:
            if sample['title'] in prompt:
                headline = sample['title']
                break
        
        # Get the expected response for this headline
        api_response_data = mock_api_responses.get(headline, {
            "category": "Neutral/Noise",
            "magnitude": 0,
            "reason": "Default response"
        })
        
        # Format as Ollama response
        ollama_response = {
            "response": json.dumps(api_response_data)
        }
        mock_resp.json.return_value = ollama_response
        
        return mock_resp

    mock_post.side_effect = get_mock_response

    # --- Execute the Test ---
    analyst = LLM_Analyst(market_name="Global Test Market")
    result_df = analyst.analyze_headlines(sample_headlines)

    # --- Assertions ---
    assert mock_post.call_count == len(sample_headlines)

    assert isinstance(result_df, pd.DataFrame)
    expected_columns = ['title', 'date', 'category', 'growth_rate', 'reason']
    assert all(col in result_df.columns for col in expected_columns)
    assert len(result_df) == len(sample_headlines)

    first_result = result_df.iloc[0]
    assert first_result['title'] == 'New Factory Investment to Boost Production by 50%'

    second_result = result_df.iloc[1]
    assert second_result['title'] == 'Geopolitical Tensions Halt Key Shipments'
    assert second_result['category'] == 'Geopolitical Tension'
    assert second_result['magnitude'] == -1.5
    
    print("\nHeadline analysis unit test passed successfully.")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])