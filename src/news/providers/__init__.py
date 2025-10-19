"""
News provider interfaces and implementations.
"""

from .base import NewsProvider
from .gdelt import GDELTProvider

__all__ = [
    'NewsProvider',
    'GDELTProvider'
]