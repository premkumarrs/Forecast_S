#!/usr/bin/env python3
"""
Migration script to remove impact_multiplier references and add MA windows to existing categories.

This script:
1. Adds ma_window_days to all existing category files
2. Removes any impact_multiplier references from configuration files
3. Logs all changes for audit purposes
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def add_ma_window_to_category(category: Dict[str, Any]) -> bool:
    """
    Add ma_window_days to a category if missing.
    
    Args:
        category: Category dictionary
        
    Returns:
        True if modified, False otherwise
    """
    if 'ma_window_days' in category:
        return False
    
    # Intelligent defaults based on category name
    name_lower = category.get('name', '').lower()
    
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
    
    logger.info(f"Added ma_window_days={category['ma_window_days']} to category '{category.get('name', 'Unknown')}'")
    return True


def migrate_categories_file(file_path: Path) -> int:
    """
    Migrate a single categories JSON file.
    
    Args:
        file_path: Path to the categories file
        
    Returns:
        Number of categories modified
    """
    try:
        with open(file_path, 'r') as f:
            categories = json.load(f)
        
        if not isinstance(categories, list):
            logger.warning(f"Skipping {file_path}: Not a list of categories")
            return 0
        
        modified_count = 0
        for cat in categories:
            if add_ma_window_to_category(cat):
                modified_count += 1
        
        if modified_count > 0:
            # Write back the modified categories
            with open(file_path, 'w') as f:
                json.dump(categories, f, indent=2)
            logger.info(f"Updated {file_path}: Modified {modified_count}/{len(categories)} categories")
        
        return modified_count
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return 0


def migrate_all_categories(config_dir: str = 'config') -> Dict[str, int]:
    """
    Migrate all category files in the config directory.
    
    Args:
        config_dir: Path to the config directory
        
    Returns:
        Dictionary with migration statistics
    """
    categories_dir = Path(config_dir) / 'categories'
    
    if not categories_dir.exists():
        logger.warning(f"Categories directory not found: {categories_dir}")
        return {'files': 0, 'categories': 0}
    
    stats = {'files': 0, 'categories': 0}
    
    # Find all generated_categories.json files
    for market_dir in categories_dir.iterdir():
        if market_dir.is_dir():
            cat_file = market_dir / 'generated_categories.json'
            if cat_file.exists():
                modified = migrate_categories_file(cat_file)
                if modified > 0:
                    stats['files'] += 1
                    stats['categories'] += modified
                logger.info(f"Processed market: {market_dir.name}")
    
    return stats


def remove_impact_multiplier_from_config(config_file: Path) -> bool:
    """
    Remove impact_multiplier references from a configuration file.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        True if file was modified
    """
    if not config_file.exists():
        return False
    
    try:
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Check if impact_multiplier is mentioned
        if 'impact_multiplier' in content:
            # For JSON files
            if config_file.suffix == '.json':
                data = json.loads(content)
                
                def remove_key(obj):
                    """Recursively remove impact_multiplier from nested structures."""
                    if isinstance(obj, dict):
                        if 'impact_multiplier' in obj:
                            del obj['impact_multiplier']
                        for value in obj.values():
                            remove_key(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            remove_key(item)
                
                remove_key(data)
                
                with open(config_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"Removed impact_multiplier from {config_file}")
                return True
            
            # For TOML files - we already handled settings.toml manually
            # so this is just for logging
            elif config_file.suffix == '.toml':
                logger.info(f"Found impact_multiplier in {config_file} - please review manually")
    
    except Exception as e:
        logger.error(f"Error processing config file {config_file}: {e}")
    
    return False


def main():
    """Run the complete migration."""
    logger.info("Starting migration: Remove impact_multiplier and add MA windows")
    logger.info("="*60)
    
    # Get the project root (assuming script is in src/migration/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    config_dir = project_root / 'config'
    
    # 1. Migrate all category files
    logger.info("\n1. Migrating category files...")
    stats = migrate_all_categories(config_dir)
    logger.info(f"   Modified {stats['files']} files, {stats['categories']} categories total")
    
    # 2. Check for impact_multiplier in other config files
    logger.info("\n2. Checking for impact_multiplier references...")
    config_files = [
        config_dir / 'aggregation.json',
        config_dir / 'category.json',
        config_dir / 'exclusions.json',
    ]
    
    modified_configs = 0
    for config_file in config_files:
        if remove_impact_multiplier_from_config(config_file):
            modified_configs += 1
    
    logger.info(f"   Modified {modified_configs} configuration files")
    
    # 3. Summary
    logger.info("\n" + "="*60)
    logger.info("Migration completed successfully!")
    logger.info(f"Total changes:")
    logger.info(f"  - Category files updated: {stats['files']}")
    logger.info(f"  - Categories with MA windows added: {stats['categories']}")
    logger.info(f"  - Config files cleaned: {modified_configs}")
    logger.info("\nPlease review the changes and test the system thoroughly.")
    

if __name__ == "__main__":
    main()
