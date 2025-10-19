"""Smoke tests to verify app imports and pages load without errors."""

import sys
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mock Streamlit before any imports that might use it
def setup_streamlit_mock():
    """Setup a comprehensive Streamlit mock"""
    mock_st = MagicMock()
    
    # Mock common Streamlit functions to return sensible defaults
    mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
    mock_st.tabs.return_value = [MagicMock(), MagicMock()]
    mock_st.expander.return_value = MagicMock()
    mock_st.container.return_value = MagicMock()
    mock_st.empty.return_value = MagicMock()
    mock_st.spinner.return_value = MagicMock()
    
    # Mock metric display
    mock_st.metric.return_value = None
    mock_st.success.return_value = None
    mock_st.error.return_value = None
    mock_st.warning.return_value = None
    mock_st.info.return_value = None
    
    # Mock input widgets
    mock_st.button.return_value = False
    mock_st.selectbox.return_value = "default"
    mock_st.multiselect.return_value = []
    mock_st.slider.return_value = 1
    mock_st.text_input.return_value = ""
    mock_st.number_input.return_value = 0
    
    # Create a comprehensive session_state mock
    mock_session_state = MagicMock()
    mock_session_state.query_cache = {}
    mock_session_state.unified_data = {}
    mock_session_state.config = {}
    mock_session_state.__contains__ = lambda self, key: True
    mock_session_state.__getitem__ = lambda self, key: {}
    mock_session_state.__setitem__ = lambda self, key, value: None
    mock_session_state.get = lambda key, default=None: default
    
    mock_st.session_state = mock_session_state
    sys.modules['streamlit'] = mock_st
    return mock_st

def test_app_imports():
    """Test that main app.py imports without errors."""
    # Setup Streamlit mock before import
    setup_streamlit_mock()
    try:
        import app
        assert True, "app.py imported successfully"
    except ImportError as e:
        pytest.fail(f"Failed to import app.py: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error importing app.py: {e}")

def test_page_imports():
    """Test that all page files import without errors."""
    # Setup Streamlit mock before any page imports
    setup_streamlit_mock()
    
    pages_dir = project_root / "pages"
    
    if not pages_dir.exists():
        pytest.fail("Pages directory does not exist")
    
    page_files = [
        "01_Data_Extraction.py",
        "02_Configuration.py", 
        "03_Forecasting.py",
        "04_Insights.py",
        "05_Export.py"
    ]
    
    for page_file in page_files:
        page_path = pages_dir / page_file
        
        if not page_path.exists():
            pytest.fail(f"Page file {page_file} does not exist")
        
        # Import the page module
        module_name = page_file.replace(".py", "").replace("-", "_").replace("(", "").replace(")", "")
        
        try:
            # Add pages directory to path temporarily
            sys.path.insert(0, str(pages_dir))
            
            # Import using exec to avoid module name conflicts
            with open(page_path, 'r', encoding='utf-8') as f:
                page_code = f.read()
            
            # Create a temporary namespace for execution
            page_namespace = {}
            exec(page_code, page_namespace)
            
            # Remove pages directory from path
            sys.path.remove(str(pages_dir))
            
        except ImportError as e:
            pytest.fail(f"Failed to import {page_file}: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error importing {page_file}: {e}")

def test_src_module_imports():
    """Test that all src modules import without errors."""
    # Setup Streamlit mock before any module imports
    setup_streamlit_mock()
    
    src_modules = [
        "src.data",
        "src.database", 
        "src.forecasting",
        "src.weights",
        "src.news",
        "src.llm",
        "src.session",
        "src.services",
        "src.ui_components"
    ]
    
    for module_name in src_modules:
        try:
            __import__(module_name)
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error importing {module_name}: {e}")

def test_config_files_exist():
    """Test that configuration files exist."""
    config_files = [
        "config/.env.example",
        "config/settings.toml"
    ]
    
    for config_file in config_files:
        config_path = project_root / config_file
        assert config_path.exists(), f"Configuration file {config_file} does not exist"
        assert config_path.is_file(), f"Configuration file {config_file} is not a file"

def test_requirements_file_exists():
    """Test that requirements.txt exists and contains expected packages."""
    requirements_path = project_root / "requirements.txt"
    
    assert requirements_path.exists(), "requirements.txt does not exist"
    
    with open(requirements_path, 'r') as f:
        requirements_content = f.read()
    
    expected_packages = [
        "streamlit",
        "pandas",
        "numpy",
        "sqlalchemy",
        "pmdarima",
        "scikit-learn",
        "altair",
        "python-dotenv"
    ]
    
    for package in expected_packages:
        assert package in requirements_content, f"Package {package} not found in requirements.txt"

def test_directory_structure():
    """Test that the expected directory structure exists."""
    expected_dirs = [
        "src",
        "pages",
        "config",
        "tests",
        "assets"
    ]
    
    for dir_name in expected_dirs:
        dir_path = project_root / dir_name
        if dir_name == "assets":
            # Assets directory is optional
            continue
        assert dir_path.exists(), f"Directory {dir_name} does not exist"
        assert dir_path.is_dir(), f"{dir_name} is not a directory"

def test_src_init_file():
    """Test that src/__init__.py exists and is valid."""
    init_path = project_root / "src" / "__init__.py"
    
    assert init_path.exists(), "src/__init__.py does not exist"
    
    # Try to import the src package
    try:
        import src
        assert hasattr(src, '__version__'), "src package should have __version__ attribute"
    except ImportError as e:
        pytest.fail(f"Failed to import src package: {e}")

def test_gitignore_exists():
    """Test that .gitignore exists and contains expected entries."""
    gitignore_path = project_root / ".gitignore"
    
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read()
        
        expected_entries = [
            "venv",
            "__pycache__",
            ".env",
            ".DS_Store"
        ]
        
        for entry in expected_entries:
            assert entry in gitignore_content, f"Entry {entry} not found in .gitignore"

if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])