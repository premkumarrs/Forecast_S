"""
Category management for market analysis.
"""

import json
import os
import re
import shutil
from typing import Dict, List, Any, Optional
import streamlit as st


def sanitize_market_name(market_name: str) -> str:
    """
    Sanitize market name for use as a directory name.
    
    Args:
        market_name: Original market name
        
    Returns:
        Sanitized market name safe for filesystem paths
    """
    # Strip leading/trailing whitespace
    sanitized = market_name.strip()
    
    # Replace problematic characters with underscores
    # Windows doesn't allow: < > : " | ? * \ /
    # Unix generally only disallows: / and null
    sanitized = re.sub(r'[<>:"|?*\\/]', '_', sanitized)
    
    # Replace multiple spaces with single underscore
    sanitized = re.sub(r'\s+', '_', sanitized)
    
    # Remove any remaining control characters
    sanitized = re.sub(r'[\x00-\x1f\x7f]', '', sanitized)
    
    # Ensure the name is not empty
    if not sanitized:
        sanitized = 'market_data'
    
    return sanitized


class CategoryManager:
    """Manages market categories for analysis."""
    
    def __init__(self, config_dir: str = 'config'):
        """
        Initialize category manager.
        
        Args:
            config_dir: Configuration directory path
        """
        self.config_dir = os.path.abspath(os.path.join(config_dir))
        self.categories_dir = os.path.join(self.config_dir, 'categories')
        self.category_file = os.path.join(self.config_dir, 'category.json')
        # st.write(self.categories_dir)
        # st.write(self.config_dir)
        # st.write(self.categories_dir)

    def load_default_categories(self) -> List[Dict[str, Any]]:
        """Load default category definitions."""
        try:
            with open(self.category_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading '{self.category_file}': {e}")
            return []
    
    def save_market_categories(
        self,
        market_name: str,
        categories: List[Dict[str, Any]]
    ) -> bool:
        """
        Save categories for a specific market.
        
        Args:
            market_name: Name of the market
            categories: List of category definitions
            
        Returns:
            True if successful
        """
        if not market_name or not categories:
            return False
        
        try:
            # Add MA window days to each category
            for category in categories:
                if 'ma_window_days' not in category:
                    # Intelligent defaults based on category name
                    name_lower = category['name'].lower()
                    if 'neutral' in name_lower or 'noise' in name_lower:
                        category['ma_window_days'] = 0
                    elif any(x in name_lower for x in ['price', 'trading', 'market', 'sentiment', 'volatility']):
                        category['ma_window_days'] = 14
                    elif any(x in name_lower for x in ['product', 'launch', 'release', 'announcement']):
                        category['ma_window_days'] = 30
                    elif any(x in name_lower for x in ['regulation', 'policy', 'infrastructure', 'legal', 'compliance']):
                        category['ma_window_days'] = 60
                    elif any(x in name_lower for x in ['strategic', 'investment', 'funding', 'acquisition']):
                        category['ma_window_days'] = 45
                    elif any(x in name_lower for x in ['research', 'development', 'innovation']):
                        category['ma_window_days'] = 45
                    else:
                        category['ma_window_days'] = 30  # Default
            
            # Sanitize market name for filesystem
            safe_name = sanitize_market_name(market_name)
            # st.write(f"Categories dory safe_name at: {safe_name}")
            # Create directory if it doesn't exist
            dir_path = os.path.join(self.categories_dir, safe_name)
            
            os.makedirs(dir_path, exist_ok=True)
            # st.write(f"Categories directory created at: {dir_path}")
            
            # Save categories to a JSON file
            file_path = os.path.join(dir_path, 'generated_categories.json')
            with open(file_path, 'w') as f:
                json.dump(categories, f, indent=2)
            
            return True
            
        except Exception as e:
            # st.write(f"Error saving categories for market '{market_name}': {e}")
            print(f"Error saving categories for market '{market_name}': {e}")
            return False
    
    def load_market_categories(
        self,
        market_name: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Load categories for a specific market.
        
        Args:
            market_name: Name of the market
            
        Returns:
            List of categories or None if not found
        """
        try:
            # Sanitize market name for filesystem
            safe_name = sanitize_market_name(market_name)
            
            file_path = os.path.join(
                self.categories_dir,
                safe_name,
                'generated_categories.json'
            )
            with open(file_path, 'r') as f:
                categories = json.load(f)
                
            # Ensure all categories have ma_window_days
            for category in categories:
                if 'ma_window_days' not in category:
                    # Apply same intelligent defaults as in save
                    name_lower = category['name'].lower()
                    if 'neutral' in name_lower or 'noise' in name_lower:
                        category['ma_window_days'] = 0
                    elif any(x in name_lower for x in ['price', 'trading', 'market', 'sentiment', 'volatility']):
                        category['ma_window_days'] = 14
                    elif any(x in name_lower for x in ['product', 'launch', 'release', 'announcement']):
                        category['ma_window_days'] = 30
                    elif any(x in name_lower for x in ['regulation', 'policy', 'infrastructure', 'legal', 'compliance']):
                        category['ma_window_days'] = 60
                    elif any(x in name_lower for x in ['strategic', 'investment', 'funding', 'acquisition']):
                        category['ma_window_days'] = 45
                    elif any(x in name_lower for x in ['research', 'development', 'innovation']):
                        category['ma_window_days'] = 45
                    else:
                        category['ma_window_days'] = 30  # Default
                        
            return categories
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    def save_market_topics(
        self,
        market_name: str,
        topics: List[str]
    ) -> bool:
        """
        Save topics for a specific market.
        
        Args:
            market_name: Name of the market
            topics: List of topics
            
        Returns:
            True if successful
        """
        if not market_name or not topics:
            return False
        
        try:
            # Sanitize market name for filesystem
            safe_name = sanitize_market_name(market_name)
            
            # Create directory if it doesn't exist
            dir_path = os.path.join(self.categories_dir, safe_name)
            os.makedirs(dir_path, exist_ok=True)
            
            # Save topics to a JSON file
            file_path = os.path.join(dir_path, 'generated_topics.json')
            with open(file_path, 'w') as f:
                json.dump(topics, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error saving topics for market '{market_name}': {e}")
            return False
    
    def load_market_topics(self, market_name: str) -> Optional[List[str]]:
        """
        Load topics for a specific market.
        
        Args:
            market_name: Name of the market
            
        Returns:
            List of topics or None if not found
        """
        try:
            # Sanitize market name for filesystem
            safe_name = sanitize_market_name(market_name)
            
            file_path = os.path.join(
                self.categories_dir,
                safe_name,
                'generated_topics.json'
            )
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    def delete_market_categories(self, market_name: str) -> bool:
        """
        Delete all saved data for a market.
        
        Args:
            market_name: Name of the market
            
        Returns:
            True if successful
        """
        try:
            # Sanitize market name for filesystem
            safe_name = sanitize_market_name(market_name)
            
            dir_path = os.path.join(self.categories_dir, safe_name)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            return True
        except Exception as e:
            print(f"Error deleting categories for market '{market_name}': {e}")
            return False
    
    def list_available_markets(self) -> List[str]:
        """
        List all markets with saved categories.
        
        Returns:
            List of market names
        """
        try:
            if not os.path.exists(self.categories_dir):
                return []
            
            markets = []
            for item in os.listdir(self.categories_dir):
                item_path = os.path.join(self.categories_dir, item)
                if os.path.isdir(item_path):
                    # Check if it has category files
                    cat_file = os.path.join(item_path, 'generated_categories.json')
                    if os.path.exists(cat_file):
                        markets.append(item)
            
            return sorted(markets)
            
        except Exception:
            return []


# Backward compatibility functions
def load_categories() -> List[Dict[str, Any]]:
    """Backward compatibility wrapper."""
    manager = CategoryManager()
    return manager.load_default_categories()


def save_market_categories(market_name: str, categories: list):
    """Backward compatibility wrapper."""
    manager = CategoryManager()
    manager.save_market_categories(market_name, categories)


def save_market_topics(market_name: str, topics: list):
    """Backward compatibility wrapper."""
    manager = CategoryManager()
    manager.save_market_topics(market_name, topics)


def load_market_categories(market_name: str) -> list:
    """Backward compatibility wrapper."""
    manager = CategoryManager()
    return manager.load_market_categories(market_name) or []


def load_market_topics(market_name: str) -> list:
    """Backward compatibility wrapper."""
    manager = CategoryManager()
    return manager.load_market_topics(market_name) or []


def delete_market_categories(market_name: str) -> bool:
    """Backward compatibility wrapper."""
    manager = CategoryManager()
    return manager.delete_market_categories(market_name)


def list_available_market_categories() -> List[str]:
    """Backward compatibility wrapper."""
    manager = CategoryManager()
    return manager.list_available_markets()