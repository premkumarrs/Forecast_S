"""
Loading of stored evaluation results for presentation.

Reads the artifacts written by ``scripts/run_census_baseline_experiment.py`` and
``scripts/run_census_indicator_experiment.py`` (which use
``run_comparative_experiment`` and ``export``). Nothing here fits a model, runs
an experiment or accesses the network; missing files are reported, never
replaced by generated values. Best-model picks are read from the saved Phase 4
comparison report rather than recomputed.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "results"
DEFAULT_INDICATOR_DIR = PROJECT_ROOT / "data" / "evaluation" / "indicators"
BASELINE_SUBDIR = "census_baseline"
INDICATOR_SUBDIR = Path("census_indicators") / "spec_a_goods"

BASELINE_FILES = {
    "summaries": "summaries.csv",
    "records": "records.csv",
    "metadata": "experiment_metadata.json",
    "report_pooled": "comparison_report_pooled.json",
    "period_breakdown": "breakdown_by_target_period.csv",
}
INDICATOR_FILES = {
    "summaries": "summaries.csv",
    "effects": "indicator_effect_vs_baseline.csv",
    "by_origin": "indicator_adjustment_by_origin.csv",
    "period_effects": "indicator_effect_by_target_period.csv",
    "metadata": "experiment_metadata.json",
}
INDICATOR_PROVENANCE_FILE = "us_pce_goods_provenance.json"
RUN_LOG_FILE = Path("census_indicators") / "run_log.json"

SUMMARY_COLUMNS = ["model", "variant", "horizon", "aggregation", "mape", "rmse", "mae",
                   "bias", "abs_bias", "successful_folds", "failed_folds", "coverage"]
EFFECT_COLUMNS = ["model", "horizon", "baseline_mape", "adjusted_mape", "baseline_rmse",
                  "adjusted_rmse", "baseline_mae", "adjusted_mae", "baseline_bias",
                  "adjusted_bias", "wins", "losses", "ties", "wins_on_baseline_underforecasts",
                  "constant_uplift_mape", "constant_uplift_rmse", "constant_uplift_mae"]
ALL_HORIZONS = "all"
METRIC_LABELS = {"mape": "MAPE", "rmse": "RMSE", "mae": "MAE"}


@dataclass
class EvaluationArtifacts:
    """Stored evaluation results; ``missing`` lists required files that were not found."""

    baseline: Dict[str, Any] = field(default_factory=dict)
    indicator: Dict[str, Any] = field(default_factory=dict)
    indicator_provenance: Optional[Dict[str, Any]] = None
    run_log: Optional[Dict[str, Any]] = None
    paths: Dict[str, Path] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)

    @property
    def baseline_available(self) -> bool:
        return all(key in self.baseline for key in BASELINE_FILES)

    @property
    def indicator_available(self) -> bool:
        return all(key in self.indicator for key in INDICATOR_FILES)


def _read(path: Path) -> Any:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return pd.read_csv(path)


def _load_group(directory: Path, files: Dict[str, str], prefix: str,
                artifacts: EvaluationArtifacts) -> Dict[str, Any]:
    loaded = {}
    for key, name in files.items():
        path = directory / name
        if path.is_file():
            loaded[key] = _read(path)
            artifacts.paths[f"{prefix}.{key}"] = path
        else:
            artifacts.missing.append(str(path))
    return loaded


def load_evaluation_artifacts(results_dir: Path = DEFAULT_RESULTS_DIR,
                              indicator_dir: Path = DEFAULT_INDICATOR_DIR) -> EvaluationArtifacts:
    """Load stored baseline and indicator results; never raises for missing files."""
    results_dir, indicator_dir = Path(results_dir), Path(indicator_dir)
    artifacts = EvaluationArtifacts()
    artifacts.baseline = _load_group(results_dir / BASELINE_SUBDIR, BASELINE_FILES, "baseline", artifacts)
    artifacts.indicator = _load_group(results_dir / INDICATOR_SUBDIR, INDICATOR_FILES, "indicator", artifacts)
    for target, path in (("indicator_provenance", indicator_dir / INDICATOR_PROVENANCE_FILE),
                         ("run_log", results_dir / RUN_LOG_FILE)):
        if path.is_file():
            setattr(artifacts, target, _read(path))
            artifacts.paths[target] = path
    return artifacts


def _require_columns(frame: pd.DataFrame, columns: List[str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns {missing}")


def _horizon_label(value: Any) -> str:
    return ALL_HORIZONS if str(value) == ALL_HORIZONS else f"{int(float(value))}Y"


def metric_table(summaries: pd.DataFrame, aggregation: str = "pooled",
                 variants: Tuple[str, ...] = ("baseline", "benchmark")) -> pd.DataFrame:
    """Model | Horizon | MAPE | RMSE | MAE | Bias | Abs bias | Coverage | Failed folds, from saved summaries."""
    _require_columns(summaries, SUMMARY_COLUMNS, "summaries")
    rows = summaries[(summaries["aggregation"] == aggregation) & summaries["variant"].isin(variants)].copy()
    rows["Horizon"] = rows["horizon"].map(_horizon_label)
    rows["_order"] = rows["Horizon"].map(lambda h: 99 if h == ALL_HORIZONS else int(h[:-1]))
    rows = rows.sort_values(["_order", "model"])
    return rows.rename(columns={
        "model": "Model", "mape": "MAPE", "rmse": "RMSE", "mae": "MAE", "bias": "Bias",
        "abs_bias": "Abs bias", "coverage": "Coverage", "failed_folds": "Failed folds",
    })[["Model", "Horizon", "MAPE", "RMSE", "MAE", "Bias", "Abs bias", "Coverage", "Failed folds"]] \
        .reset_index(drop=True)


def metric_by_horizon(summaries: pd.DataFrame, metric: str, aggregation: str = "pooled",
                      variants: Tuple[str, ...] = ("baseline", "benchmark")) -> pd.DataFrame:
    """Rows: horizons 1Y..nY; columns: models; values: the saved metric (all-horizon rows excluded)."""
    table = metric_table(summaries, aggregation, variants)
    table = table[table["Horizon"] != ALL_HORIZONS]
    column = METRIC_LABELS.get(metric, metric)
    return table.pivot(index="Horizon", columns="Model", values=column)


def best_models(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Best eligible entry per metric (all horizons) as saved by ``build_comparison_report``."""
    best = report.get("best_by_metric", {})
    return {metric: entry for metric, entry in best.items() if entry}


def best_models_by_horizon(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Best eligible entry by MAPE for each horizon, as saved by ``build_comparison_report``."""
    return {f"{int(h)}Y": entry for h, entry in report.get("best_by_horizon", {}).items() if entry}


def indicator_comparison(effects: pd.DataFrame, metric: str, horizon: str = ALL_HORIZONS) -> pd.DataFrame:
    """Model | Baseline | Adjusted | Change | Change % for one metric, from the saved effect table."""
    _require_columns(effects, EFFECT_COLUMNS, "indicator effects")
    rows = effects[effects["horizon"].astype(str) == str(horizon)]
    base, adj = rows[f"baseline_{metric}"], rows[f"adjusted_{metric}"]
    label = METRIC_LABELS.get(metric, metric)
    return pd.DataFrame({
        "Model": rows["model"].values,
        f"Baseline {label}": base.values,
        f"Adjusted {label}": adj.values,
        f"Change {label}": (adj - base).values,
        f"Change {label} %": ((adj - base) / base * 100).values,
    })


def bias_comparison(effects: pd.DataFrame, horizon: str = ALL_HORIZONS) -> pd.DataFrame:
    """Baseline bias, adjusted bias, bias change and win attribution per model."""
    _require_columns(effects, EFFECT_COLUMNS, "indicator effects")
    rows = effects[effects["horizon"].astype(str) == str(horizon)]
    return pd.DataFrame({
        "Model": rows["model"].values,
        "Baseline bias": rows["baseline_bias"].values,
        "Adjusted bias": rows["adjusted_bias"].values,
        "Bias change": (rows["adjusted_bias"] - rows["baseline_bias"]).values,
        "Wins": rows["wins"].values,
        "Losses": rows["losses"].values,
        "Wins on baseline under-forecasts": rows["wins_on_baseline_underforecasts"].values,
    })


def fixed_nudge_comparison(effects: pd.DataFrame, horizon: str = ALL_HORIZONS) -> pd.DataFrame:
    """Indicator-adjusted vs fixed-nudge (full-sample mean factor) errors per model."""
    _require_columns(effects, EFFECT_COLUMNS, "indicator effects")
    rows = effects[effects["horizon"].astype(str) == str(horizon)]
    return pd.DataFrame({
        "Model": rows["model"].values,
        "Indicator MAPE": rows["adjusted_mape"].values,
        "Fixed-nudge MAPE": rows["constant_uplift_mape"].values,
        "Indicator RMSE": rows["adjusted_rmse"].values,
        "Fixed-nudge RMSE": rows["constant_uplift_rmse"].values,
        "Indicator MAE": rows["adjusted_mae"].values,
        "Fixed-nudge MAE": rows["constant_uplift_mae"].values,
    })


def period_errors(breakdown: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """MAPE and share of absolute error by target period for one horizon."""
    _require_columns(breakdown, ["model", "horizon", "target_period", "n_forecasts", "mape_pct",
                                 "share_of_total_abs_error_pct"], "period breakdown")
    rows = breakdown[breakdown["horizon"] == horizon]
    return rows.rename(columns={
        "model": "Model", "target_period": "Target years", "n_forecasts": "Forecasts",
        "mape_pct": "MAPE", "share_of_total_abs_error_pct": "Share of abs. error %",
    })[["Model", "Target years", "Forecasts", "MAPE", "Share of abs. error %"]].reset_index(drop=True)


def download_files(artifacts: EvaluationArtifacts) -> List[Dict[str, Any]]:
    """Stored files offered for download (only those that exist)."""
    wanted = [
        ("baseline.summaries", "Baseline summary (CSV)", "census_baseline_summaries.csv", "text/csv"),
        ("baseline.records", "Baseline fold records (CSV)", "census_baseline_records.csv", "text/csv"),
        ("indicator.effects", "Indicator comparison (CSV)", "census_indicator_comparison.csv", "text/csv"),
        ("indicator.by_origin", "Indicator adjustment by origin (CSV)",
         "census_indicator_adjustment_by_origin.csv", "text/csv"),
        ("baseline.metadata", "Baseline experiment metadata (JSON)",
         "census_baseline_metadata.json", "application/json"),
        ("indicator.metadata", "Indicator experiment metadata (JSON)",
         "census_indicator_metadata.json", "application/json"),
    ]
    return [{"key": key, "label": label, "file_name": file_name, "mime": mime, "path": artifacts.paths[key]}
            for key, label, file_name, mime in wanted if key in artifacts.paths]
