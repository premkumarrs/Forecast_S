"""
Category management for LLM-based analysis.
"""

from .manager import CategoryManager
from .constraints import CategoryConstraints
from .storage import CategoryStorage
from .validators import CategoryValidator

# Backward compatibility imports
from .manager import (
    load_categories,
    save_market_categories,
    save_market_topics,
    load_market_categories,
    load_market_topics,
    delete_market_categories,
    list_available_market_categories
)

from .constraints import (
    build_category_constraints,
    apply_category_constraints
)

from .validators import validate_category

__all__ = [
    # New modular components
    'CategoryManager',
    'CategoryConstraints',
    'CategoryStorage',
    'CategoryValidator',
    
    # Backward compatibility
    'load_categories',
    'save_market_categories',
    'save_market_topics',
    'load_market_categories',
    'load_market_topics',
    'delete_market_categories',
    'list_available_market_categories',
    'build_category_constraints',
    'apply_category_constraints',
    'validate_category'
]