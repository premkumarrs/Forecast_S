"""
CSV and JSON export of comparative experiment results.

Exports contain the configuration, metadata (data label, input fingerprints,
evaluated horizons, origins, limitations), per-horizon summaries for both
aggregations, benchmark-relative and adjustment comparisons, and the
fold-level records and failures needed to reproduce every number. Raw news
headlines and indicator values are not exported, only their fingerprints.
Non-finite numbers are written as empty CSV cells / JSON null.
"""

import csv
import json
import math
from typing import Any, Dict, List

from .comparison import (
    AGGREGATIONS,
    adjustment_effects,
    benchmark_improvements,
    summarize_experiment,
)
from .backtest import (
    FAILURE_COLUMNS,
    RECORD_COLUMNS,
    VARIANT_INDICATOR_ADJUSTED_PAST_ONLY,
    VARIANT_NEWS_ADJUSTED_HISTORICAL,
)
from .experiment import ExperimentResult

CSV_TABLES = ("summaries", "improvements", "records", "failures")
CONTEXT_COLUMNS = ["dataset_name", "data_label"]


def _plain(value: Any) -> Any:
    """JSON-safe primitive: numpy scalars unwrapped, NaN/inf -> None."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if hasattr(value, "item") and not isinstance(value, (list, dict, tuple)):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, int):
        return value
    return str(value)


def _improvement_rows(result: ExperimentResult) -> List[Dict[str, Any]]:
    rows = [dict(i.to_dict(), comparison="vs_benchmark") for i in benchmark_improvements(result)]
    for variant in (VARIANT_INDICATOR_ADJUSTED_PAST_ONLY, VARIANT_NEWS_ADJUSTED_HISTORICAL):
        rows += [dict(i.to_dict(), comparison="adjustment_vs_baseline")
                 for i in adjustment_effects(result, variant)]
    return rows


def experiment_to_dict(result: ExperimentResult) -> Dict[str, Any]:
    """Full machine-readable representation of an experiment."""
    return _plain({
        "config": result.config.to_dict(),
        "metadata": result.metadata,
        "summaries": [s.to_dict() for agg in AGGREGATIONS for s in summarize_experiment(result, agg)],
        "comparisons": _improvement_rows(result),
        "records": result.records_frame().to_dict("records"),
        "failures": result.failures_frame().to_dict("records"),
    })


def export_json(result: ExperimentResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(experiment_to_dict(result), f, indent=2, sort_keys=True)


def _table_rows(result: ExperimentResult, table: str) -> List[Dict[str, Any]]:
    if table == "summaries":
        return [s.to_dict() for agg in AGGREGATIONS for s in summarize_experiment(result, agg)]
    if table == "improvements":
        return _improvement_rows(result)
    if table == "records":
        return result.records_frame().to_dict("records")
    return result.failures_frame().to_dict("records")


def _table_columns(table: str, rows: List[Dict[str, Any]]) -> List[str]:
    if table == "records":
        return list(RECORD_COLUMNS)
    if table == "failures":
        return list(FAILURE_COLUMNS)
    return list(rows[0].keys()) if rows else []


def export_csv(result: ExperimentResult, path: str, table: str = "summaries") -> None:
    """Write one table as CSV.

    ``table`` is ``summaries`` (per model/variant/horizon/aggregation; horizon
    is written as ``all`` for the all-horizons rows), ``improvements``,
    ``records`` (fold-level forecasts) or ``failures``. Every row carries the
    dataset name and data label; the full configuration is in the JSON export.
    """
    if table not in CSV_TABLES:
        raise ValueError(f"table must be one of {CSV_TABLES}, got {table!r}")
    rows = _table_rows(result, table)
    columns = CONTEXT_COLUMNS + _table_columns(table, rows)
    context = {"dataset_name": result.config.dataset_name,
               "data_label": result.metadata.get("data_label", "")}
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {**context, **{k: _plain(v) for k, v in row.items()}}
            if table in ("summaries", "improvements") and out.get("horizon") is None:
                out["horizon"] = "all"
            writer.writerow({k: "" if out.get(k) is None else out.get(k) for k in columns})
