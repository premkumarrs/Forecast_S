"""
SQL query templates and builders.
"""

from sqlalchemy import text


def build_kpi_query(kpi_key: str, geo_id: int = None) -> text:
    """Build KPI query with optional geo filtering."""
    base_query = """
    SELECT
      kv.idGeo, kv.idKpi, kv.valueTime, kv.value,
      k.id AS kpi_id, k.kpiKey, k.name AS kpi_name,
      g.name AS geo_name, g.isoCode3L AS iso3
    FROM smi_contentDev.kpisValues kv
    LEFT JOIN smi_contentDev.kpis k ON kv.idKpi = k.id
    LEFT JOIN smi_contentDev.geos g ON kv.idGeo = g.id
    WHERE k.kpiKey = :kpiKey_value
    """
    
    if geo_id is not None:
        if geo_id == 100:
            base_query += " AND kv.idGeo = 100"
        else:
            base_query += " AND kv.idGeo <> 100"
    
    base_query += " ORDER BY valueTime;"
    
    return text(base_query)


def build_multi_indicator_query(kpi_keys: list, geo_id: int = None) -> text:
    """Build query for multiple indicators."""
    placeholders = ', '.join([f':kpi_{i}' for i in range(len(kpi_keys))])
    
    base_query = f"""
    SELECT
      kv.idGeo,
      kv.valueTime,
      k.kpiKey AS indicator_key,
      k.name AS indicator_name,
      kv.value,
      g.name AS country,
      g.isoCode3L AS iso3
    FROM smi_contentDev.kpisValues kv
    LEFT JOIN smi_contentDev.kpis k ON kv.idKpi = k.id
    LEFT JOIN smi_contentDev.geos g ON kv.idGeo = g.id
    WHERE k.kpiKey IN ({placeholders})
    """
    
    if geo_id is not None:
        if geo_id == 100:
            base_query += " AND kv.idGeo = 100"
        else:
            base_query += " AND kv.idGeo <> 100"
    
    base_query += " ORDER BY indicator_key, valueTime DESC"
    
    return text(base_query)


def build_unified_market_query(kpi_key: str) -> text:
    """Build unified query for market value data (global + country)."""
    query = """
    SELECT
      kv.idGeo, kv.valueTime, kv.value,
      k.kpiKey, k.name AS kpi_name,
      g.name AS country, g.isoCode3L AS iso3
    FROM smi_contentDev.kpisValues kv
    LEFT JOIN smi_contentDev.kpis k ON kv.idKpi = k.id
    LEFT JOIN smi_contentDev.geos g ON kv.idGeo = g.id
    WHERE k.kpiKey = :market_kpi
    ORDER BY country, valueTime;
    """
    return text(query)


def build_unified_indicators_query(kpi_keys: list) -> text:
    """Build unified query for indicator data (global + country)."""
    placeholders = ', '.join([f':kpi_{i}' for i in range(len(kpi_keys))])
    
    query = f"""
    SELECT
      kv.idGeo,
      kv.valueTime,
      k.kpiKey AS indicator_key,
      k.name AS indicator_name,
      kv.value,
      g.name AS country,
      g.isoCode3L AS iso3
    FROM smi_contentDev.kpisValues kv
    LEFT JOIN smi_contentDev.kpis k ON kv.idKpi = k.id
    LEFT JOIN smi_contentDev.geos g ON kv.idGeo = g.id
    WHERE k.kpiKey IN ({placeholders})
    ORDER BY indicator_key, country, valueTime DESC
    """
    return text(query)


def build_query_params(kpi_keys: list, market_kpi: str = None) -> dict:
    """Build parameters dictionary for queries."""
    params = {}
    
    # For market query
    if market_kpi:
        params['market_kpi'] = market_kpi
    
    # For indicator queries
    for i, kpi_key in enumerate(kpi_keys):
        params[f'kpi_{i}'] = kpi_key
        
    return params