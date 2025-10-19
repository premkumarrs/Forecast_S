"""
Simplified database query functionality.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv('config/.env')


def query(qry: str) -> pd.DataFrame:
    """Execute SQL query and return results as DataFrame.
    
    Args:
        qry (str): SQL query string
        
    Returns:
        pd.DataFrame: Query results
    """
    db_url = os.getenv('DB_URL')
    if not db_url:
        return pd.DataFrame()
    
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            return pd.read_sql(text(qry), conn)
    except Exception as e:
        raise RuntimeError(f"Query execution failed: {e}")


def get_engine():
    """Create and return SQLAlchemy engine for advanced usage."""
    db_url = os.getenv('DB_URL')
    if not db_url:
        raise ValueError("DB_URL not found in environment variables")
    return create_engine(db_url)


def test_connection() -> bool:
    """Test database connection.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    db_url = os.getenv('DB_URL')
    if not db_url:
        return False
        
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


# Backward compatibility class (deprecated)
class DatabaseConnection:
    """Deprecated: Use query() function directly."""
    
    @property
    def db_url(self):
        return os.getenv('DB_URL')
    
    def query(self, qry: str) -> pd.DataFrame:
        return query(qry)
    
    def get_engine(self):
        return get_engine()
    
    def test_connection(self) -> bool:
        return test_connection()