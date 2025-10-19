"""
Regions module for loading mappings and aggregating forecast data.
"""

import json
import os
import pandas as pd
from typing import List, Dict, Optional, Any

# Global cache for region mappings
REGION_MAPPINGS: Optional[Dict[str, List[str]]] = None
COUNTRY_TO_REGION: Optional[Dict[str, str]] = None

def load_region_mappings() -> Dict[str, List[str]]:
    """Loads region mappings from config/aggregation.json."""
    global REGION_MAPPINGS
    if REGION_MAPPINGS is not None:
        return REGION_MAPPINGS

    try:
        # Correctly locate the config file relative to this file's location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '..', 'config', 'aggregation.json')
        
        with open(config_path, 'r') as f:
            data = json.load(f)
            REGION_MAPPINGS = data.get("region_mappings", {})
            return REGION_MAPPINGS
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading region mappings: {e}")
        REGION_MAPPINGS = {}
        return {}

def build_country_to_region_mapping() -> Dict[str, str]:
    """Builds a reverse mapping from a country to its immediate parent region."""
    global COUNTRY_TO_REGION
    if COUNTRY_TO_REGION is not None:
        return COUNTRY_TO_REGION

    mappings = load_region_mappings()
    country_map = {}
    for region, members in mappings.items():
        for member in members:
            # This creates a simple, non-nested mapping.
            # A country can only belong to one region in this map.
            if member not in country_map:
                country_map[member] = region
    COUNTRY_TO_REGION = country_map
    return COUNTRY_TO_REGION

def get_region_for_country(country: str) -> Optional[str]:
    """Gets the direct parent region for a given country."""
    country_map = build_country_to_region_mapping()
    return country_map.get(country)

def aggregate_countries_to_regions(
    country_forecasts: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Aggregates country-level forecasts into regional forecasts using nested mappings.

    Args:
        country_forecasts: A dictionary where keys are country names and values are their forecast DataFrames.

    Returns:
        A dictionary containing aggregated forecasts for all regions, including 'Worldwide'.
    """
    region_mappings = load_region_mappings()
    if not region_mappings:
        return {}

    # Memoization cache to store already computed region totals
    memo = {}

    def get_region_total(region_name: str) -> Optional[pd.DataFrame]:
        """Recursively calculates the total for a region."""
        if region_name in memo:
            return memo[region_name]

        if region_name in country_forecasts:
            # Base case: the "region" is actually a country
            memo[region_name] = country_forecasts[region_name]
            return country_forecasts[region_name]

        if region_name not in region_mappings:
            # This name is not a defined region and not a country with a forecast
            return None

        # This is a region that needs to be aggregated
        members = region_mappings[region_name]
        region_total_df = None

        for member_name in members:
            member_df = get_region_total(member_name)
            if member_df is not None:
                if region_total_df is None:
                    # Initialize with the structure of the first member
                    region_total_df = member_df.copy()
                else:
                    # Add the values
                    region_total_df['value_hat'] += member_df['value_hat']
        
        if region_total_df is not None:
            # Update the 'type' for historical vs. forecast rows
            hist_mask = region_total_df['type'] == 'Historical'
            fore_mask = region_total_df['type'] == 'Forecast'
            region_total_df.loc[hist_mask, 'type'] = 'Historical'
            region_total_df.loc[fore_mask, 'type'] = 'Forecast'

        memo[region_name] = region_total_df
        return region_total_df

    # Calculate totals for all defined regions
    all_region_forecasts = {}
    for region in region_mappings.keys():
        total_df = get_region_total(region)
        if total_df is not None:
            all_region_forecasts[region] = total_df

    return all_region_forecasts

# Initialize mappings when the module is loaded
load_region_mappings()
build_country_to_region_mapping()
