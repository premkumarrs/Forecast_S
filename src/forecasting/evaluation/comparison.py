"""
Comparative analysis of experiment results.

Terminology used throughout:

* forecast: one (model, variant, origin, target year) record.
* fold: one forecast origin for a given model and variant. Within one horizon a
  fold holds exactly one forecast; across all horizons it holds ``horizon``.
  A fold is *successful* when all of its forecasts in the group are valid
  (``status == "ok"`` with finite actual and prediction); otherwise it counts
  as failed. ``coverage = successful_folds / attempted_folds * 100``.

Two aggregations are reported and always labelled:

* ``"pooled"``: metrics over every valid forecast in the group (the same
  computation as ``summarize_backtest``). Error statistics describe individual
  forecast errors.
* ``"fold_mean"``: metrics computed per fold, then averaged over folds with at
  least one valid forecast. Error statistics describe the per-fold mean errors.
  Pooled RMSE and fold-mean RMSE differ by construction.

Comparisons between two methods (benchmark-relative improvement, adjustment
effects) are paired: they use only forecasts that are valid for both methods
at the same origin and target year, so neither side benefits from folds the
other failed on. ``improvement = (reference_error - candidate_error) /
reference_error * 100``; positive means the candidate had lower error. The
result is undefined (None) when the reference error is zero. These are
measurements of out-of-sample predictive accuracy, not causal effects.
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..base.metrics import calculate_forecast_accuracy
from .backtest import (
    NAIVE_MODEL_NAME,
    STATUS_OK,
    VARIANT_BASELINE,
    VARIANT_BENCHMARK,
    VARIANT_INDICATOR_ADJUSTED_PAST_ONLY,
    VARIANT_NEWS_ADJUSTED_HISTORICAL,
    BacktestResult,
)
from .experiment import ExperimentResult
from .report import _summarize_group

AGGREGATIONS = ("pooled", "fold_mean")
RANKABLE_METRICS = ("mape", "rmse", "mae", "abs_bias")
IMPROVEMENT_METRICS = ("mape", "rmse", "mae")
ADJUSTED_VARIANTS = (VARIANT_INDICATOR_ADJUSTED_PAST_ONLY, VARIANT_NEWS_ADJUSTED_HISTORICAL)
ALL_HORIZONS = None

_METRIC_LABELS = {"mape": "MAPE", "rmse": "RMSE", "mae": "MAE", "abs_bias": "absolute bias"}
_ADJUSTMENT_LABELS = {
    VARIANT_INDICATOR_ADJUSTED_PAST_ONLY: "Indicator adjustment",
    VARIANT_NEWS_ADJUSTED_HISTORICAL: "News adjustment",
}

Results = Union[ExperimentResult, BacktestResult, pd.DataFrame]


def _clean(value) -> Optional[float]:
    """Convert to a plain float; NaN/inf become None."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _frame(results: Results) -> pd.DataFrame:
    if isinstance(results, ExperimentResult):
        return results.records_frame()
    if isinstance(results, BacktestResult):
        return results.to_frame()
    return results


def _valid_mask(frame: pd.DataFrame) -> pd.Series:
    actual = pd.to_numeric(frame["actual"], errors="coerce")
    predicted = pd.to_numeric(frame["predicted"], errors="coerce")
    return (frame["status"] == STATUS_OK) & np.isfinite(actual) & np.isfinite(predicted)


def _error_stats(errors: Sequence[float]) -> Dict[str, Optional[float]]:
    errors = np.asarray([e for e in errors if e is not None and math.isfinite(e)], dtype=float)
    if errors.size == 0:
        return {"error_mean": None, "error_median": None, "error_std": None}
    return {
        "error_mean": _clean(errors.mean()),
        "error_median": _clean(np.median(errors)),
        "error_std": _clean(errors.std(ddof=1)) if errors.size > 1 else None,
    }


@dataclass(frozen=True)
class MetricSummary:
    """Accuracy and coverage for one model/variant at one horizon (None = all horizons)."""

    model: str
    variant: str
    horizon: Optional[int]
    aggregation: str
    mape: Optional[float]
    rmse: Optional[float]
    mae: Optional[float]
    bias: Optional[float]
    abs_bias: Optional[float]
    error_mean: Optional[float]
    error_median: Optional[float]
    error_std: Optional[float]
    attempted_folds: int
    successful_folds: int
    failed_folds: int
    coverage: Optional[float]
    n_forecasts: int
    n_valid: int
    n_failed: int
    n_invalid: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _summarize(group: pd.DataFrame, model: str, variant: str, horizon: Optional[int],
               aggregation: str) -> MetricSummary:
    valid = _valid_mask(group)
    fold_valid = valid.groupby(group["origin"]).all()
    attempted = int(fold_valid.size)
    successful = int(fold_valid.sum())
    pooled = _summarize_group(group)

    if aggregation == "pooled":
        metrics = {m: _clean(pooled[m]) for m in ("mape", "rmse", "mae", "bias")}
        errors = (pd.to_numeric(group.loc[valid, "predicted"])
                  - pd.to_numeric(group.loc[valid, "actual"])).tolist()
    else:
        per_fold = [_summarize_group(sub) for _, sub in group.groupby("origin", sort=True)]
        per_fold = [row for row in per_fold if row["n_valid"] > 0]
        metrics = {}
        for m in ("mape", "rmse", "mae", "bias"):
            values = [row[m] for row in per_fold if math.isfinite(row[m])]
            metrics[m] = _clean(np.mean(values)) if values else None
        errors = [row["bias"] for row in per_fold]

    return MetricSummary(
        model=model,
        variant=variant,
        horizon=horizon,
        aggregation=aggregation,
        **metrics,
        abs_bias=abs(metrics["bias"]) if metrics["bias"] is not None else None,
        **_error_stats(errors),
        attempted_folds=attempted,
        successful_folds=successful,
        failed_folds=attempted - successful,
        coverage=successful / attempted * 100 if attempted else None,
        n_forecasts=int(pooled["n_forecasts"]),
        n_valid=int(pooled["n_valid"]),
        n_failed=int(pooled["n_failed"]),
        n_invalid=int(pooled["n_invalid"]),
    )


def summarize_experiment(
    results: Results,
    aggregation: str = "pooled",
    include_all_horizons: bool = True,
) -> List[MetricSummary]:
    """One summary per model, variant and horizon, plus an all-horizons row (horizon=None).

    Args:
        results: ExperimentResult, BacktestResult or records DataFrame.
        aggregation: ``"pooled"`` or ``"fold_mean"`` (see module docstring).
        include_all_horizons: Also add rows pooling every horizon.
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
    frame = _frame(results)
    summaries: List[MetricSummary] = []
    if frame.empty:
        return summaries
    for (model, variant), group in frame.groupby(["model", "variant"], sort=True):
        for horizon, sub in group.groupby("horizon", sort=True):
            summaries.append(_summarize(sub, model, variant, int(horizon), aggregation))
        if include_all_horizons:
            summaries.append(_summarize(group, model, variant, ALL_HORIZONS, aggregation))
    return summaries


@dataclass(frozen=True)
class RankedEntry:
    rank: Optional[int]
    model: str
    variant: str
    horizon: Optional[int]
    aggregation: str
    metric: str
    value: Optional[float]
    coverage: Optional[float]
    successful_folds: int
    attempted_folds: int
    eligible: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def rank_models(
    summaries: Iterable[MetricSummary],
    metric: str = "mape",
    horizon: Optional[int] = ALL_HORIZONS,
    aggregation: str = "pooled",
    min_coverage: float = 0.0,
) -> List[RankedEntry]:
    """Rank model/variant combinations by one metric (lower is better).

    ``metric`` is one of ``mape``, ``rmse``, ``mae``, ``abs_bias`` (``bias`` is
    accepted and ranked by its absolute value). Entries with an undefined
    metric or coverage below ``min_coverage`` are listed last with
    ``rank=None`` and ``eligible=False``, so a method that fails on most folds
    is never ranked as if it were robust. Ties share a rank. No composite
    score is computed.
    """
    metric = "abs_bias" if metric == "bias" else metric
    if metric not in RANKABLE_METRICS:
        raise ValueError(f"metric must be one of {RANKABLE_METRICS} or 'bias', got {metric!r}")
    selected = [s for s in summaries if s.horizon == horizon and s.aggregation == aggregation]

    def eligible(s: MetricSummary) -> bool:
        value = getattr(s, metric)
        return value is not None and s.coverage is not None and s.coverage >= min_coverage

    ranked = sorted((s for s in selected if eligible(s)),
                    key=lambda s: (getattr(s, metric), s.model, s.variant))
    entries: List[RankedEntry] = []
    previous_value, previous_rank = None, 0
    for position, s in enumerate(ranked, start=1):
        value = getattr(s, metric)
        rank = previous_rank if value == previous_value else position
        previous_value, previous_rank = value, rank
        entries.append(RankedEntry(rank, s.model, s.variant, s.horizon, aggregation, metric,
                                   value, s.coverage, s.successful_folds, s.attempted_folds, True))
    for s in sorted((s for s in selected if not eligible(s)), key=lambda s: (s.model, s.variant)):
        entries.append(RankedEntry(None, s.model, s.variant, s.horizon, aggregation, metric,
                                   getattr(s, metric), s.coverage, s.successful_folds,
                                   s.attempted_folds, False))
    return entries


@dataclass(frozen=True)
class ImprovementResult:
    """Paired accuracy comparison of a candidate method against a reference method."""

    model: str
    variant: str
    reference_model: str
    reference_variant: str
    horizon: Optional[int]
    n_paired: int
    candidate_mape: Optional[float]
    reference_mape: Optional[float]
    mape_improvement_pct: Optional[float]
    candidate_rmse: Optional[float]
    reference_rmse: Optional[float]
    rmse_improvement_pct: Optional[float]
    candidate_mae: Optional[float]
    reference_mae: Optional[float]
    mae_improvement_pct: Optional[float]
    abs_error_diff_mean: Optional[float]
    abs_error_diff_median: Optional[float]
    abs_error_diff_std: Optional[float]
    wins: int
    losses: int
    ties: int
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def improvement_pct(reference_error: Optional[float], candidate_error: Optional[float]) -> Optional[float]:
    """``(reference - candidate) / reference * 100``; None if undefined (e.g. zero reference)."""
    if reference_error is None or candidate_error is None:
        return None
    if not math.isfinite(reference_error) or not math.isfinite(candidate_error) or reference_error == 0:
        return None
    return (reference_error - candidate_error) / reference_error * 100


def describe_change(subject: str, reference: str, metric: str, pct: Optional[float],
                    context: str = "") -> str:
    """Plain-language statement of a measured change; never claims an unmeasured improvement.

    ``context`` (e.g. ``" (Damped ETS, horizon 1, 12 paired forecasts)"``) is
    appended to the reference phrase.
    """
    label = _METRIC_LABELS.get(metric, metric.upper())
    reference = reference + context
    if pct is None:
        return (f"{subject}: change in {label} relative to {reference} is undefined "
                "(no paired forecasts or zero reference error).")
    if abs(pct) < 1e-9:
        return f"{subject} left {label} unchanged relative to {reference}."
    if pct > 0:
        return f"{subject} reduced {label} by {pct:.1f}% relative to {reference}."
    return (f"{subject} increased {label} by {abs(pct):.1f}% relative to {reference}, "
            "indicating no improvement under this evaluation setup.")


def _paired_comparison(frame: pd.DataFrame, model: str, variant: str, reference_model: str,
                       reference_variant: str, horizon: Optional[int], subject: str,
                       reference_label: str) -> ImprovementResult:
    def select(m: str, v: str) -> pd.DataFrame:
        rows = frame[(frame["model"] == m) & (frame["variant"] == v)]
        if horizon is not None:
            rows = rows[rows["horizon"] == horizon]
        rows = rows[_valid_mask(rows)]
        return rows[["origin", "year", "actual", "predicted"]].astype(
            {"actual": float, "predicted": float})

    paired = select(model, variant).merge(
        select(reference_model, reference_variant), on=["origin", "year"],
        suffixes=("_c", "_r"), validate="one_to_one",
    ).sort_values(["origin", "year"]).reset_index(drop=True)

    values: Dict[str, Optional[float]] = {}
    for prefix, suffix in (("candidate", "_c"), ("reference", "_r")):
        if paired.empty:
            metrics = {m: None for m in IMPROVEMENT_METRICS}
        else:
            metrics = calculate_forecast_accuracy(paired["actual_r"], paired[f"predicted{suffix}"],
                                                  metrics=list(IMPROVEMENT_METRICS))
        for m in IMPROVEMENT_METRICS:
            values[f"{prefix}_{m}"] = _clean(metrics[m])
    for m in IMPROVEMENT_METRICS:
        values[f"{m}_improvement_pct"] = improvement_pct(values[f"reference_{m}"], values[f"candidate_{m}"])

    abs_c = (paired["predicted_c"] - paired["actual_r"]).abs()
    abs_r = (paired["predicted_r"] - paired["actual_r"]).abs()
    diff = (abs_c - abs_r).tolist()
    stats = _error_stats(diff)
    tolerance = 1e-12
    where = f" ({model}" + (f", horizon {horizon}" if horizon is not None else ", all horizons") \
        + f", {len(paired)} paired forecasts)"
    return ImprovementResult(
        model=model, variant=variant,
        reference_model=reference_model, reference_variant=reference_variant,
        horizon=horizon, n_paired=len(paired),
        **values,
        abs_error_diff_mean=stats["error_mean"],
        abs_error_diff_median=stats["error_median"],
        abs_error_diff_std=stats["error_std"],
        wins=sum(d < -tolerance for d in diff),
        losses=sum(d > tolerance for d in diff),
        ties=sum(abs(d) <= tolerance for d in diff),
        summary=describe_change(subject, reference_label, "mape", values["mape_improvement_pct"], where),
    )


def _horizons(frame: pd.DataFrame, include_all_horizons: bool) -> List[Optional[int]]:
    horizons: List[Optional[int]] = sorted(int(h) for h in frame["horizon"].unique())
    return horizons + ([ALL_HORIZONS] if include_all_horizons else [])


def benchmark_improvements(results: Results, include_all_horizons: bool = True) -> List[ImprovementResult]:
    """Paired comparison of every non-benchmark model/variant against the naive benchmark."""
    frame = _frame(results)
    if frame.empty:
        return []
    out = []
    candidates = frame.loc[frame["variant"] != VARIANT_BENCHMARK, ["model", "variant"]] \
        .drop_duplicates().sort_values(["model", "variant"])
    for horizon in _horizons(frame, include_all_horizons):
        for model, variant in candidates.itertuples(index=False):
            out.append(_paired_comparison(
                frame, model, variant, NAIVE_MODEL_NAME, VARIANT_BENCHMARK, horizon,
                subject=f"{model} [{variant}]", reference_label="the naive benchmark",
            ))
    return out


def adjustment_effects(results: Results, variant: str,
                       include_all_horizons: bool = True) -> List[ImprovementResult]:
    """Paired comparison of an adjusted variant against the same model's baseline."""
    if variant not in ADJUSTED_VARIANTS:
        raise ValueError(f"variant must be one of {ADJUSTED_VARIANTS}, got {variant!r}")
    frame = _frame(results)
    if frame.empty or variant not in set(frame["variant"]):
        return []
    models = sorted(frame.loc[frame["variant"] == variant, "model"].unique())
    return [
        _paired_comparison(frame, model, variant, model, VARIANT_BASELINE, horizon,
                           subject=f"{_ADJUSTMENT_LABELS[variant]} ({variant})",
                           reference_label="the baseline model")
        for horizon in _horizons(frame, include_all_horizons)
        for model in models
    ]


@dataclass
class ComparisonReport:
    """Structured answers to the comparative research questions."""

    data_label: str
    aggregation: str
    min_coverage: float
    summaries: List[MetricSummary]
    rankings: Dict[str, List[RankedEntry]]
    best_by_metric: Dict[str, Optional[RankedEntry]]
    best_by_horizon: Dict[int, Optional[RankedEntry]]
    benchmark_improvements: List[ImprovementResult]
    best_relative_to_benchmark: Optional[ImprovementResult]
    indicator_effects: List[ImprovementResult]
    news_effects: List[ImprovementResult]
    coverage: List[MetricSummary]
    limitations: List[str] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        def entries(items):
            return [item.to_dict() for item in items]
        return {
            "data_label": self.data_label,
            "aggregation": self.aggregation,
            "min_coverage": self.min_coverage,
            "summaries": entries(self.summaries),
            "rankings": {k: entries(v) for k, v in self.rankings.items()},
            "best_by_metric": {k: v.to_dict() if v else None for k, v in self.best_by_metric.items()},
            "best_by_horizon": {str(k): v.to_dict() if v else None for k, v in self.best_by_horizon.items()},
            "benchmark_improvements": entries(self.benchmark_improvements),
            "best_relative_to_benchmark": (self.best_relative_to_benchmark.to_dict()
                                           if self.best_relative_to_benchmark else None),
            "indicator_effects": entries(self.indicator_effects),
            "news_effects": entries(self.news_effects),
            "coverage": entries(self.coverage),
            "limitations": list(self.limitations),
            "findings": list(self.findings),
        }


def _first_eligible(entries: List[RankedEntry]) -> Optional[RankedEntry]:
    return next((e for e in entries if e.eligible), None)


def build_comparison_report(
    result: ExperimentResult,
    aggregation: str = "pooled",
    min_coverage: float = 80.0,
) -> ComparisonReport:
    """Summaries, rankings, benchmark-relative improvements and adjustment effects.

    "Best" picks consider only entries with coverage >= ``min_coverage``
    (percent); all entries remain visible in ``rankings`` and ``coverage``.
    """
    summaries = summarize_experiment(result, aggregation=aggregation)
    rankings = {m: rank_models(summaries, m, ALL_HORIZONS, aggregation, min_coverage)
                for m in RANKABLE_METRICS}
    best_by_metric = {m: _first_eligible(entries) for m, entries in rankings.items()}
    best_by_horizon = {
        h: _first_eligible(rank_models(summaries, "mape", h, aggregation, min_coverage))
        for h in result.evaluated_horizons
    }
    improvements = benchmark_improvements(result)
    overall = [i for i in improvements if i.horizon is None and i.mape_improvement_pct is not None]
    best_relative = max(overall, key=lambda i: (i.mape_improvement_pct, i.model, i.variant), default=None)
    indicator = adjustment_effects(result, VARIANT_INDICATOR_ADJUSTED_PAST_ONLY)
    news = adjustment_effects(result, VARIANT_NEWS_ADJUSTED_HISTORICAL)
    coverage = [s for s in summaries if s.horizon is None]

    findings: List[str] = []
    data_label = result.metadata.get("data_label", "")
    if result.config.is_synthetic:
        findings.append("All findings below are from DEMO/SYNTHETIC data.")
    for metric in ("mape", "rmse", "mae"):
        best = best_by_metric[metric]
        if best is not None:
            findings.append(
                f"Lowest {aggregation} {_METRIC_LABELS[metric]} (all horizons): {best.model} "
                f"[{best.variant}] = {best.value:.4g} (coverage {best.coverage:.0f}%)."
            )
        else:
            findings.append(f"No method met the {min_coverage:.0f}% coverage threshold for "
                            f"{_METRIC_LABELS[metric]}.")
    findings.extend(i.summary for i in overall)
    findings.extend(i.summary for i in indicator + news if i.horizon is None)
    for s in coverage:
        if s.failed_folds:
            findings.append(f"{s.model} [{s.variant}]: {s.failed_folds} of {s.attempted_folds} "
                            f"folds failed (coverage {s.coverage:.0f}%).")

    return ComparisonReport(
        data_label=data_label,
        aggregation=aggregation,
        min_coverage=min_coverage,
        summaries=summaries,
        rankings=rankings,
        best_by_metric=best_by_metric,
        best_by_horizon=best_by_horizon,
        benchmark_improvements=improvements,
        best_relative_to_benchmark=best_relative,
        indicator_effects=indicator,
        news_effects=news,
        coverage=coverage,
        limitations=list(result.metadata.get("limitations", [])),
        findings=findings,
    )
