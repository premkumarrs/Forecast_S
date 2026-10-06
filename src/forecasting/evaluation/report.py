"""
Aggregate backtest records into accuracy metrics.
"""

from typing import Sequence, Union

import numpy as np
import pandas as pd

from ..base.metrics import calculate_forecast_accuracy
from .backtest import BacktestResult, STATUS_FAILED, STATUS_INVALID, STATUS_OK

GROUPABLE_COLUMNS = ("model", "variant", "horizon", "origin", "year")
DEFAULT_GROUP_BY = ("model", "variant", "horizon")
ACCURACY_METRICS = ["mape", "rmse", "mae", "bias"]
REQUIRED_COLUMNS = {"status", "actual", "predicted"}

SUMMARY_COUNT_COLUMNS = ["n_forecasts", "n_valid", "n_failed", "n_invalid", "n_zero_actual"]
SUMMARY_METRIC_COLUMNS = ["mape", "rmse", "mae", "bias"]


def _summarize_group(group: pd.DataFrame) -> dict:
    actual_values = pd.to_numeric(group["actual"], errors="coerce")
    predicted_values = pd.to_numeric(group["predicted"], errors="coerce")
    finite = np.isfinite(actual_values) & np.isfinite(predicted_values)
    is_ok = group["status"] == STATUS_OK
    valid = group[is_ok & finite]
    row = {
        "n_forecasts": len(group),
        "n_valid": len(valid),
        "n_failed": int((group["status"] == STATUS_FAILED).sum()),
        "n_invalid": int((group["status"] == STATUS_INVALID).sum() + (is_ok & ~finite).sum()),
        "n_zero_actual": int((valid["actual"] == 0).sum()),
    }
    if valid.empty:
        row.update({metric: np.nan for metric in SUMMARY_METRIC_COLUMNS})
        return row

    actual = valid["actual"].astype(float).reset_index(drop=True)
    predicted = valid["predicted"].astype(float).reset_index(drop=True)
    metrics = calculate_forecast_accuracy(actual, predicted, metrics=ACCURACY_METRICS)
    # calculate_mape drops zero actuals and returns inf when none remain;
    # report that as undefined rather than an infinite error.
    if not np.isfinite(metrics["mape"]):
        metrics["mape"] = np.nan
    row.update({metric: float(metrics[metric]) for metric in SUMMARY_METRIC_COLUMNS})
    return row


def summarize_backtest(
    results: Union[BacktestResult, pd.DataFrame],
    group_by: Sequence[str] = DEFAULT_GROUP_BY,
) -> pd.DataFrame:
    """Compute MAPE, RMSE, MAE and bias per group of backtest records.

    Metrics use only records with ``status == "ok"`` whose actual and predicted
    values are finite numbers. Failed records are counted in ``n_failed``;
    records marked invalid, or marked ok but holding non-finite values, are
    counted in ``n_invalid``. Excluded records never enter the metrics, and a
    group with no valid predictions gets NaN metrics instead of a misleading
    value. MAPE excludes zero actuals (counted in ``n_zero_actual``) and is NaN
    when every valid actual is zero. Bias is ``mean(predicted - actual)``.

    Args:
        results: Output of ``run_backtest`` or its ``to_frame()`` DataFrame.
        group_by: Columns to group by; an empty sequence gives one overall row.

    Returns:
        DataFrame with the group columns, counts and metrics (MAPE in percent).

    Raises:
        ValueError: If required columns are missing or ``group_by`` contains an
            unsupported column.
    """
    frame = results.to_frame() if isinstance(results, BacktestResult) else results
    group_by = list(group_by)
    unsupported = [col for col in group_by if col not in GROUPABLE_COLUMNS]
    if unsupported:
        raise ValueError(
            f"Cannot group by {unsupported}. Supported: {list(GROUPABLE_COLUMNS)}"
        )
    missing = sorted((REQUIRED_COLUMNS | set(group_by)) - set(frame.columns))
    if missing:
        raise ValueError(f"Backtest records are missing required columns: {missing}")

    output_columns = group_by + SUMMARY_COUNT_COLUMNS + SUMMARY_METRIC_COLUMNS
    if frame.empty:
        return pd.DataFrame(columns=output_columns)

    if not group_by:
        return pd.DataFrame([_summarize_group(frame)], columns=output_columns)

    rows = []
    for keys, group in frame.groupby(group_by, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(group_by, keys)), **_summarize_group(group)})
    return pd.DataFrame(rows, columns=output_columns)
