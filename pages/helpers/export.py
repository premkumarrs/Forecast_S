"""
Minimal export helpers: build a single wide market table and a news analysis table.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import streamlit as st
from ..components.utils import get_hist_cutoff, safe_get_value_column


def _get_market_metadata(unified_data: Dict) -> Dict[str, str]:
    meta = unified_data.get('market_value', {}).get('metadata', {}) if isinstance(unified_data, dict) else {}
    return {
        'kpi_key': meta.get('kpi_key', ''),
        'kpi_name': meta.get('kpi_name', '')
    }


def _years_union(hist_years: List[int], forecast_years: List[int]) -> List[int]:
    ys = sorted(set([int(y) for y in (hist_years or []) + (forecast_years or [])]))
    return ys


def _pivot_wide(df: pd.DataFrame, id_cols: List[str], value_col: str = 'value') -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=id_cols)
    # groupby(dropna=False) keeps rows whose idGeo is missing (demo data has no idGeo);
    # pivot_table would drop them, and its dropna=False adds a cartesian product of ids.
    p = (
        df.groupby(id_cols + ['year'], dropna=False)[value_col].last()
        .unstack('year')
        .dropna(axis=1, how='all')
    )
    p = p.reset_index()
    # Ensure numeric years ordering
    value_cols = [c for c in p.columns if isinstance(c, (int, np.integer)) or (isinstance(c, str) and c.isdigit())]
    value_cols_sorted = sorted([int(c) for c in value_cols])
    ordered = id_cols + value_cols_sorted
    # Coerce year column names to int
    p.columns = [int(c) if isinstance(c, str) and c.isdigit() else c for c in p.columns]
    return p.reindex(columns=ordered)


def build_market_wide_table(forecast_result: Dict, unified_data: Dict) -> pd.DataFrame:
    """Build a single wide-format market table with kpiKey/idGeo/nameGeo/nameVertical + year columns.

    Historical values (<= hist_cutoff) are preserved exactly from unified_data.
    Forecast values (> hist_cutoff) are filled from forecast_result.
    """
    hist_cutoff = get_hist_cutoff()
    meta = _get_market_metadata(unified_data)
    kpi_key = meta.get('kpi_key', '')
    name_vertical = meta.get('kpi_name', '')

    # 1) Determine scope
    country_forecasts = forecast_result.get('country_forecasts', {}) or {}
    global_forecast = forecast_result.get('global_forecast', pd.DataFrame())

    # Export ignores regional aggregation by design
    scope = 'country' if country_forecasts else 'global'

    # 2) Build historical base (never changed)
    market = unified_data.get('market_value', {}) if isinstance(unified_data, dict) else {}
    hist_country = market.get('country', pd.DataFrame())
    hist_global = market.get('global', pd.DataFrame())

    # Keep only <= hist_cutoff
    if not hist_country.empty and 'year' in hist_country.columns:
        hist_country = hist_country[hist_country['year'] <= hist_cutoff].copy()
    if not hist_global.empty and 'year' in hist_global.columns:
        hist_global = hist_global[hist_global['year'] <= hist_cutoff].copy()

    rows: pd.DataFrame
    if scope == 'country':
        # country index with idGeo if available, else synthesize None
        base_cols = ['idGeo', 'country', 'year', 'value']
        if hist_country.empty:
            # create empty base from forecast keys
            names = list(country_forecasts.keys())
            base = pd.DataFrame({'country': names, 'idGeo': [None]*len(names)})
        else:
            # ensure idGeo present
            if 'idGeo' not in hist_country.columns:
                hist_country['idGeo'] = None
            base = hist_country[['idGeo', 'country']].drop_duplicates()
        # pivot historical
        hist_wide = _pivot_wide(hist_country[['idGeo', 'country', 'year', 'value']], ['idGeo', 'country']) if not hist_country.empty else pd.DataFrame(columns=['idGeo', 'country'])

        # pivot forecasts per country
        f_blocks = []
        for cname, cdata in country_forecasts.items():
            df = cdata.get('forecast') if isinstance(cdata, dict) else cdata
            if df is None or df.empty:
                continue
            value_col = safe_get_value_column(df) or 'value_hat'
            tmp = df[['year', value_col]].copy()
            tmp['country'] = cname
            # idGeo: bring from base if exists
            id_geo = base.loc[base['country'] == cname, 'idGeo']
            tmp['idGeo'] = id_geo.iloc[0] if not id_geo.empty else None
            tmp.rename(columns={value_col: 'value'}, inplace=True)
            # keep only forecast years
            tmp = tmp[tmp['year'] > hist_cutoff]
            f_blocks.append(tmp)
        forecast_df = pd.concat(f_blocks, ignore_index=True) if f_blocks else pd.DataFrame(columns=['idGeo', 'country', 'year', 'value'])
        forecast_wide = _pivot_wide(forecast_df, ['idGeo', 'country']) if not forecast_df.empty else pd.DataFrame(columns=['idGeo', 'country'])

        # merge hist and forecast without overwriting historical
        rows = pd.merge(hist_wide, forecast_wide, on=['idGeo', 'country'], how='outer', suffixes=('', ''))
        rows.insert(0, 'nameGeo', rows.pop('country'))

    else:  # global
        # Historical
        hist_row = pd.DataFrame(columns=['idGeo', 'nameGeo'])
        if not hist_global.empty:
            hist_tmp = hist_global[['year', 'value']].copy()
            hist_tmp['idGeo'] = 100
            hist_tmp['nameGeo'] = 'Worldwide'
            hist_row = _pivot_wide(hist_tmp[['idGeo', 'nameGeo', 'year', 'value']], ['idGeo', 'nameGeo'])
        # Forecasts
        f_row = pd.DataFrame(columns=['idGeo', 'nameGeo'])
        if isinstance(global_forecast, pd.DataFrame) and not global_forecast.empty:
            value_col = safe_get_value_column(global_forecast) or 'value_hat'
            f_tmp = global_forecast[['year', value_col]].copy()
            f_tmp['idGeo'] = 100
            f_tmp['nameGeo'] = 'Worldwide'
            f_tmp.rename(columns={value_col: 'value'}, inplace=True)
            f_tmp = f_tmp[f_tmp['year'] > hist_cutoff]
            f_row = _pivot_wide(f_tmp[['idGeo', 'nameGeo', 'year', 'value']], ['idGeo', 'nameGeo'])
        rows = pd.merge(hist_row, f_row, on=['idGeo', 'nameGeo'], how='outer')

    # 3) Attach kpi columns and fill nameVertical
    if rows.empty:
        return pd.DataFrame(columns=['kpiKey', 'idGeo', 'nameGeo', 'nameVertical'])
    rows.insert(0, 'kpiKey', kpi_key)
    rows.insert(3, 'nameVertical', name_vertical)
    # Ensure year columns are numeric
    for c in rows.columns:
        if isinstance(c, int) or (isinstance(c, str) and c.isdigit()):
            try:
                rows[c] = pd.to_numeric(rows[c], errors='coerce')
            except Exception:
                pass
    return rows


def build_news_export(forecast_result: Dict, kpi_key: str) -> pd.DataFrame:
    """Flatten news analysis into a single table with minimal columns."""
    rows = []
    # Prefer news kept in session, but accept forecast_result copies
    news_package = st.session_state.get('analyzed_news', {}) or {}
    if not news_package:
        # try forecast_result passthrough
        if 'analyzed_news' in forecast_result:
            news_package = {'type': 'global_news', 'data': forecast_result.get('analyzed_news')}
        elif 'country_news' in forecast_result:
            news_package = {'type': 'country_news', 'data': forecast_result.get('country_news')}

    ntype = news_package.get('type')
    if ntype == 'global_news':
        df = news_package.get('data', pd.DataFrame())
        if not df.empty:
            tmp = df.copy()
            tmp['scope'] = 'Global'
            tmp['nameGeo'] = 'Worldwide'
            rows.append(tmp)
    elif ntype == 'country_news':
        country_data = news_package.get('data', {}) or {}
        for cname, cdf in country_data.items():
            if cdf is None or cdf.empty:
                continue
            tmp = cdf.copy()
            tmp['scope'] = 'Country'
            tmp['nameGeo'] = cname
            rows.append(tmp)
    elif ntype == 'combined_news':
        gdf = news_package.get('global_data', pd.DataFrame())
        if not gdf.empty:
            tmp = gdf.copy()
            tmp['scope'] = 'Global'
            tmp['nameGeo'] = 'Worldwide'
            rows.append(tmp)
        country_data = news_package.get('country_data', {}) or {}
        for cname, cdf in country_data.items():
            if cdf is None or cdf.empty:
                continue
            tmp = cdf.copy()
            tmp['scope'] = 'Country'
            tmp['nameGeo'] = cname
            rows.append(tmp)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if out.empty:
        return out
    # Keep essential columns
    keep = ['title', 'date', 'category', 'growth_rate', 'relevant', 'reason', 'scope', 'nameGeo']
    keep = [c for c in keep if c in out.columns] + [c for c in ['scope', 'nameGeo'] if c not in out.columns]
    out = out[keep]
    out.insert(0, 'kpiKey', kpi_key)
    # Normalize date
    if 'date' in out.columns:
        try:
            out['date'] = pd.to_datetime(out['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        except Exception:
            pass
    # Numeric growth rate
    if 'growth_rate' in out.columns:
        out['growth_rate'] = pd.to_numeric(out['growth_rate'], errors='coerce')
    # Relevant int 0/1
    if 'relevant' in out.columns:
        out['relevant'] = pd.to_numeric(out['relevant'], errors='coerce').fillna(0).astype(int)
    return out
