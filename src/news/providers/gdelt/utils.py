"""
Utilities for GDELT provider.
"""

import os
import csv
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Minimal fallback ISO3→FIPS map for common cases
FALLBACK_ISO3_TO_FIPS = {
    "USA": "US", "GBR": "UK", "DEU": "GM", "ESP": "SP", "CHN": "CH",
    "JPN": "JA", "KOR": "KS", "NLD": "NL", "CHE": "SZ", "AUS": "AS",
    "FRA": "FR", "ITA": "IT", "CAN": "CA", "BRA": "BR", "IND": "IN",
    "RUS": "RS", "MEX": "MX", "SGP": "SN", "HKG": "HK", "SWE": "SW",
    "NOR": "NO", "DNK": "DA", "FIN": "FI", "BEL": "BE", "AUT": "AU",
    "IRL": "EI", "POL": "PL", "CZE": "EZ", "HUN": "HU", "PRT": "PO",
    "GRC": "GR", "TUR": "TU", "ISR": "IS", "ARE": "AE", "SAU": "SA",
    "ZAF": "SF", "NZL": "NZ", "ARG": "AR", "CHL": "CI", "COL": "CO"
}


def load_iso3_fips_mapping() -> Dict[str, str]:
    """
    Load ISO3 to FIPS mapping from CSV file.
    
    Returns:
        Dictionary mapping ISO3 codes to FIPS codes
    """
    mapping = {}
    
    # Try to find the CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up to project root
    project_root = current_dir
    for _ in range(4):  # Go up 4 levels (gdelt/providers/news/src)
        project_root = os.path.dirname(project_root)
    
    csv_path = os.path.join(project_root, 'config', 'flat-ui__data-Sun Aug 17 2025.csv')
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                iso3 = row.get('ISO3', '').strip().upper()
                fips = row.get('FIPS', '').strip().upper()
                if iso3 and fips and fips.lower() != 'null':
                    mapping[iso3] = fips
        logger.info(f"Loaded {len(mapping)} ISO3->FIPS mappings from CSV")
    except FileNotFoundError:
        logger.warning(f"CSV file not found at {csv_path}, using fallback mappings")
    except Exception as e:
        logger.error(f"Error loading ISO3->FIPS mapping: {str(e)}")
    
    # Merge with fallback mappings
    for iso3, fips in FALLBACK_ISO3_TO_FIPS.items():
        if iso3 not in mapping:
            mapping[iso3] = fips
    
    return mapping


def iso3_to_fips(iso3: Optional[str], mapping: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Convert ISO3 country code to FIPS code.
    
    Args:
        iso3: ISO3 country code
        mapping: Optional pre-loaded mapping dictionary
        
    Returns:
        FIPS code or None if not found
    """
    if not iso3:
        return None
    
    iso3 = iso3.strip().upper()
    
    # Use provided mapping or load default
    if mapping is None:
        mapping = load_iso3_fips_mapping()
    
    return mapping.get(iso3)


def clean_topics(topics: List[str]) -> List[str]:
    """
    Clean and validate topic list.
    
    Args:
        topics: Raw topic list
        
    Returns:
        Cleaned topic list
    """
    if not topics:
        return []
    
    cleaned = []
    seen = set()
    
    for topic in topics:
        if topic and isinstance(topic, str):
            topic = topic.strip()
            if topic and topic not in seen:
                cleaned.append(topic)
                seen.add(topic)
    
    return cleaned


def calculate_fallback_confidence(
    article_count: int,
    min_threshold: int = 10,
    optimal_threshold: int = 50
) -> float:
    """
    Calculate confidence multiplier based on article count.
    
    Args:
        article_count: Number of articles found
        min_threshold: Minimum articles for any confidence
        optimal_threshold: Articles needed for full confidence
        
    Returns:
        Confidence multiplier between 0 and 1
    """
    if article_count <= 0:
        return 0.0
    
    if article_count >= optimal_threshold:
        return 1.0
    
    if article_count < min_threshold:
        # Linear scaling from 0 to 0.5 for very low counts
        return (article_count / min_threshold) * 0.5
    
    # Linear scaling from 0.5 to 1.0 for moderate counts
    range_size = optimal_threshold - min_threshold
    position = article_count - min_threshold
    return 0.5 + (position / range_size) * 0.5