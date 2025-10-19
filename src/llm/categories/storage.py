"""
Storage management for categories and topics.
"""

import json
import os
import re
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path


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


class CategoryStorage:
    """Handles storage and retrieval of categories and topics."""
    
    def __init__(self, base_dir: str = 'config/categories'):
        """
        Initialize storage manager.
        
        Args:
            base_dir: Base directory for category storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_json(
        self,
        data: Any,
        market_name: str,
        filename: str
    ) -> bool:
        """
        Save data to JSON file.
        
        Args:
            data: Data to save
            market_name: Market name (subdirectory)
            filename: JSON filename
            
        Returns:
            True if successful
        """
        try:
            # Sanitize market name for filesystem
            safe_name = sanitize_market_name(market_name)
            
            market_dir = self.base_dir / safe_name
            market_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = market_dir / filename
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error saving {filename} for '{market_name}': {e}")
            return False
    
    def load_json(
        self,
        market_name: str,
        filename: str
    ) -> Optional[Any]:
        """
        Load data from JSON file.
        
        Args:
            market_name: Market name (subdirectory)
            filename: JSON filename
            
        Returns:
            Loaded data or None if not found
        """
        try:
            # Sanitize market name for filesystem
            safe_name = sanitize_market_name(market_name)
            
            file_path = self.base_dir / safe_name / filename
            with open(file_path, 'r') as f:
                return json.load(f)
                
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    def delete_market(self, market_name: str) -> bool:
        """
        Delete all data for a market.
        
        Args:
            market_name: Market name
            
        Returns:
            True if successful
        """
        try:
            # Sanitize market name for filesystem
            safe_name = sanitize_market_name(market_name)
            
            market_dir = self.base_dir / safe_name
            if market_dir.exists():
                shutil.rmtree(market_dir)
            return True
            
        except Exception as e:
            print(f"Error deleting market '{market_name}': {e}")
            return False
    
    def list_markets(self) -> List[str]:
        """
        List all markets with saved data.
        
        Returns:
            List of market names
        """
        try:
            markets = []
            for item in self.base_dir.iterdir():
                if item.is_dir():
                    # Check if it has any JSON files
                    json_files = list(item.glob('*.json'))
                    if json_files:
                        markets.append(item.name)
            
            return sorted(markets)
            
        except Exception:
            return []
    
    def market_exists(self, market_name: str) -> bool:
        """
        Check if market has saved data.
        
        Args:
            market_name: Market name
            
        Returns:
            True if market directory exists
        """
        # Sanitize market name for filesystem
        safe_name = sanitize_market_name(market_name)
        
        market_dir = self.base_dir / safe_name
        return market_dir.exists() and market_dir.is_dir()
    
    def get_market_files(self, market_name: str) -> List[str]:
        """
        List all files for a market.
        
        Args:
            market_name: Market name
            
        Returns:
            List of filenames
        """
        try:
            # Sanitize market name for filesystem
            safe_name = sanitize_market_name(market_name)
            
            market_dir = self.base_dir / safe_name
            if not market_dir.exists():
                return []
            
            return [f.name for f in market_dir.glob('*.json')]
            
        except Exception:
            return []
    
    def copy_market(
        self,
        source_market: str,
        target_market: str
    ) -> bool:
        """
        Copy all data from one market to another.
        
        Args:
            source_market: Source market name
            target_market: Target market name
            
        Returns:
            True if successful
        """
        try:
            source_dir = self.base_dir / source_market
            target_dir = self.base_dir / target_market
            
            if not source_dir.exists():
                return False
            
            if target_dir.exists():
                shutil.rmtree(target_dir)
            
            shutil.copytree(source_dir, target_dir)
            return True
            
        except Exception as e:
            print(f"Error copying market data: {e}")
            return False