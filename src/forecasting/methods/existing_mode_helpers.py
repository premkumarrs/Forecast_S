"""Helper utilities for existing-forecast + news mode."""

from typing import Dict, Optional, Tuple
import pandas as pd


def derive_hist_cutoff(config: Dict) -> Optional[int]:
    """Derive the historical cutoff to separate baseline history vs adjustments."""
    start_year = config.get('adjustment_start_year')
    if start_year is not None:
        return int(start_year) - 1
    hist_cutoff = config.get('hist_cutoff')
    if hist_cutoff is not None:
        return int(hist_cutoff)
    forecast_until = config.get('forecast_until')
    if forecast_until is not None:
        return int(forecast_until) - 1
    return None


def build_baseline_dataframe(
    series_df: pd.DataFrame,
    forecast_until: int,
    derived_hist_cutoff: Optional[int],
    value_column: str = 'value'
) -> pd.DataFrame:
    """Create a baseline dataframe (year, value_hat, type) from a market series."""
    if series_df is None or series_df.empty or value_column not in series_df.columns:
        return pd.DataFrame(columns=['year', 'value_hat', 'type'])

    df = series_df.copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    df = df[df['year'] <= int(forecast_until)]
    df = df.sort_values('year')

    baseline = pd.DataFrame({
        'year': df['year'],
        'value_hat': pd.to_numeric(df[value_column], errors='coerce').fillna(0.0)
    })

    if derived_hist_cutoff is None:
        derived_hist_cutoff = int(baseline['year'].min()) - 1 if not baseline.empty else int(forecast_until) - 1

    baseline['type'] = baseline['year'].apply(
        lambda y: 'historical' if y <= derived_hist_cutoff else 'Forecast'
    )
    return baseline[['year', 'value_hat', 'type']].reset_index(drop=True)


def apply_adjustment_window(
    adjusted_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    adjustment_details: Optional[Dict],
    start_year: Optional[int],
    end_year: Optional[int]
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """Limit adjustments to the provided year window by reverting outside values to baseline."""
    if adjusted_df is None or adjusted_df.empty:
        return adjusted_df, adjustment_details

    if start_year is None or end_year is None:
        return adjusted_df, adjustment_details

    mask = ~adjusted_df['year'].between(start_year, end_year)
    if mask.any() and baseline_df is not None and not baseline_df.empty:
        baseline_map = baseline_df.set_index('year')['value_hat']
        adjusted_df.loc[mask, 'value_hat'] = adjusted_df.loc[mask, 'year'].map(baseline_map).fillna(
            adjusted_df.loc[mask, 'value_hat']
        )

    if adjustment_details and isinstance(adjustment_details, dict):
        year_adjustments = adjustment_details.get('year_adjustments', {})
        if isinstance(year_adjustments, dict):
            updated = {}
            for year_key, year_data in year_adjustments.items():
                try:
                    year_int = int(year_key)
                except Exception:
                    year_int = year_key
                if isinstance(year_int, int) and (year_int < start_year or year_int > end_year):
                    if isinstance(year_data, dict):
                        year_data = dict(year_data)
                        year_data['news_pct'] = 0.0
                        year_data['total_pct'] = 0.0
                        if 'indicators_pct' in year_data:
                            year_data['indicators_pct'] = 0.0
                    updated[year_key] = year_data
                else:
                    updated[year_key] = year_data
            adjustment_details['year_adjustments'] = updated
    return adjusted_df, adjustment_details
