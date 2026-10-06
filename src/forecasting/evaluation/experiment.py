"""
Reproducible comparative forecasting experiments.

``run_comparative_experiment`` is a thin, deterministic layer over
``run_backtest``: it resolves the model list, records the full configuration,
fingerprints the inputs and states the limitations that apply to the result.
It does not implement another backtesting loop, so every model and variant is
evaluated on exactly the same walk-forward folds (same origins, training
cutoffs, horizon and target series), and the indicator and news variants keep
the point-in-time guarantees of ``run_backtest``:

* market training data: ``year <= origin``;
* indicators: ``year <= origin``;
* historical news: dated on or before ``origin``-12-31 23:59:59 UTC and within
  the news lookback window.

The naive last-value benchmark never receives indicator or news adjustments.
Analysis of the result lives in ``comparison``; CSV/JSON export in ``export``.
"""

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ..adjustments.news_adjustment import DEFAULT_NEWS_LOOKBACK_DAYS
from ..config import MIN_HISTORICAL_YEARS
from .backtest import (
    DEFAULT_INDICATOR_WEIGHT,
    NAIVE_MODEL_NAME,
    VARIANT_BASELINE,
    VARIANT_BENCHMARK,
    VARIANT_INDICATOR_ADJUSTED_PAST_ONLY,
    VARIANT_NEWS_ADJUSTED_HISTORICAL,
    BacktestResult,
    run_backtest,
)

# Short experiment identifiers -> BaselineModelFactory names.
MODEL_ALIASES = {
    "naive_last_value": NAIVE_MODEL_NAME,
    "cagr": "3-yr CAGR",
    "damped_ets": "Damped ETS",
    "logistic_growth": "Logistic Growth",
}
DEFAULT_EXPERIMENT_MODELS = ("cagr", "damped_ets", "logistic_growth")
DEFAULT_EXPERIMENT_HORIZON = 3

SYNTHETIC_LIMITATION = (
    "DEMO/SYNTHETIC DATA: the target series is synthetic, so these results "
    "do not describe real market forecasting performance."
)
NEWS_LIMITATIONS = (
    "Historical news retrieval and recency/window calculations are "
    "point-in-time, but historical interpretation by a modern LLM may contain "
    "hindsight because the model may know events that occurred after the "
    "historical origin; this is a retrospective evaluation, not a perfect "
    "historical replay.",
    "Historical GDELT availability is limited (the DOC API officially covers "
    "only the most recent ~3 months); news results depend entirely on the "
    "supplied historical news data and its coverage of each fold.",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything needed to state how an experiment result was generated."""

    dataset_name: str
    is_synthetic: bool
    models: Tuple[str, ...]
    horizon: int
    min_train_years: int
    step: int
    origins: Optional[Tuple[int, ...]]
    include_indicators: bool
    indicator_weights: Optional[Dict[str, float]]
    indicator_weight: Optional[float]
    include_news: bool
    news_lookback_days: Optional[int]
    news_config: Optional[Dict[str, Any]]
    notes: str = ""

    @property
    def variants(self) -> Tuple[str, ...]:
        variants = [VARIANT_BENCHMARK, VARIANT_BASELINE]
        if self.include_indicators:
            variants.append(VARIANT_INDICATOR_ADJUSTED_PAST_ONLY)
        if self.include_news:
            variants.append(VARIANT_NEWS_ADJUSTED_HISTORICAL)
        return tuple(variants)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["models"] = list(self.models)
        data["origins"] = list(self.origins) if self.origins is not None else None
        data["variants"] = list(self.variants)
        return data


@dataclass
class ExperimentResult:
    """Backtest records plus the configuration and metadata that produced them."""

    config: ExperimentConfig
    backtest: BacktestResult
    metadata: Dict[str, Any] = field(default_factory=dict)

    def records_frame(self) -> pd.DataFrame:
        return self.backtest.to_frame()

    def failures_frame(self) -> pd.DataFrame:
        return self.backtest.failures_frame()

    @property
    def evaluated_horizons(self) -> List[int]:
        return list(self.metadata.get("evaluated_horizons", []))


def _resolve_models(models: Optional[Sequence[str]]) -> List[str]:
    if models is None:
        models = DEFAULT_EXPERIMENT_MODELS
    if isinstance(models, str):
        raise TypeError(f"models must be a sequence of model names, not a single string; use [{models!r}]")
    resolved = [MODEL_ALIASES.get(name, name) for name in models]
    resolved = list(dict.fromkeys(resolved))
    if NAIVE_MODEL_NAME not in resolved:
        resolved.append(NAIVE_MODEL_NAME)
    return resolved


def _fingerprint(frame: Optional[pd.DataFrame]) -> Optional[str]:
    """Order-independent SHA-256 of a DataFrame's contents (for provenance only)."""
    if frame is None:
        return None
    columns = sorted(map(str, frame.columns))
    as_text = frame.rename(columns=str)[columns].astype(str)
    as_text = as_text.sort_values(columns, kind="mergesort").reset_index(drop=True)
    return hashlib.sha256(as_text.to_csv(index=False).encode("utf-8")).hexdigest()


def _feasible_horizon(historical_data: pd.DataFrame, horizon: int, min_train_years: int) -> int:
    if not isinstance(historical_data, pd.DataFrame) or "year" not in historical_data.columns:
        return horizon
    n_years = int(pd.Series(historical_data["year"]).nunique())
    return min(horizon, n_years - min_train_years)


def run_comparative_experiment(
    historical_data: pd.DataFrame,
    *,
    dataset_name: str,
    is_synthetic: bool,
    models: Optional[Sequence[str]] = None,
    origins: Optional[Sequence[int]] = None,
    horizon: int = DEFAULT_EXPERIMENT_HORIZON,
    min_train_years: int = MIN_HISTORICAL_YEARS,
    step: int = 1,
    include_indicators: bool = False,
    indicators: Optional[pd.DataFrame] = None,
    indicator_weights: Optional[Mapping[str, float]] = None,
    indicator_weight: float = DEFAULT_INDICATOR_WEIGHT,
    include_news: bool = False,
    news: Optional[pd.DataFrame] = None,
    news_config: Optional[Mapping[str, Any]] = None,
    news_lookback_days: int = DEFAULT_NEWS_LOOKBACK_DAYS,
    run_timestamp: Optional[str] = None,
    notes: str = "",
) -> ExperimentResult:
    """Run every requested model and variant on one shared set of walk-forward folds.

    Args:
        historical_data: Target series with ``year`` and ``value`` columns.
        dataset_name: Identifier recorded with the result.
        is_synthetic: Must be stated explicitly; synthetic results are labelled
            DEMO/SYNTHETIC in the metadata and every export.
        models: Model identifiers (``"cagr"``, ``"damped_ets"``,
            ``"logistic_growth"``, ``"naive_last_value"``) or
            ``BaselineModelFactory`` names. Defaults to the three baseline
            models; the naive benchmark is always added.
        origins: Optional subset of fold origins to keep. Each must be a fold
            origin produced by the walk-forward split.
        horizon: Maximum forecast horizon; horizons ``1..horizon`` are each
            evaluated. If the series is too short, the largest feasible horizon
            is used and the reduction is recorded in the metadata.
        min_train_years, step: Passed to ``run_backtest``.
        include_indicators, indicators, indicator_weights, indicator_weight:
            Enable the ``indicator_adjusted_past_only`` variant.
        include_news, news, news_config, news_lookback_days: Enable the
            ``news_adjusted_historical`` variant from already-analysed
            historical headlines. No news is fetched or analysed here.
        run_timestamp: Recorded as metadata only (never used in any
            calculation); defaults to the current UTC time.
        notes: Free-text description recorded with the configuration.

    Returns:
        ExperimentResult with the backtest records, configuration and metadata.

    Raises:
        TypeError, ValueError: For invalid inputs (see ``run_backtest``), an
            empty ``dataset_name``, a non-boolean ``is_synthetic`` or unknown
            ``origins``.
    """
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string")
    if not isinstance(is_synthetic, bool):
        raise TypeError("is_synthetic must be True or False")
    model_names = _resolve_models(models)

    limitations: List[str] = []
    evaluated_horizon = horizon
    if isinstance(horizon, int) and not isinstance(horizon, bool) and horizon >= 1 \
            and isinstance(min_train_years, int) and not isinstance(min_train_years, bool):
        feasible = _feasible_horizon(historical_data, horizon, min_train_years)
        if 1 <= feasible < horizon:
            evaluated_horizon = feasible
            limitations.append(
                f"Requested horizon {horizon} is not supported by the data with "
                f"min_train_years={min_train_years}; evaluated horizons 1..{feasible} only."
            )

    backtest = run_backtest(
        historical_data,
        models=model_names,
        horizon=evaluated_horizon,
        min_train_years=min_train_years,
        step=step,
        include_naive=True,
        use_indicators=include_indicators,
        indicators=indicators,
        indicator_weights=indicator_weights,
        indicator_weight=indicator_weight,
        use_news=include_news,
        news=news,
        news_config=news_config,
        news_lookback_days=news_lookback_days,
    )

    kept_origins: Optional[Tuple[int, ...]] = None
    if origins is not None:
        requested = sorted(set(int(o) for o in origins))
        available = {fold.origin for fold in backtest.folds}
        unknown = [o for o in requested if o not in available]
        if unknown or not requested:
            raise ValueError(
                f"origins {unknown or requested} are not walk-forward fold origins; "
                f"available: {sorted(available)}"
            )
        keep = set(requested)
        backtest = BacktestResult(
            records=[r for r in backtest.records if r["origin"] in keep],
            failures=[f for f in backtest.failures if f["origin"] in keep],
            folds=[f for f in backtest.folds if f.origin in keep],
        )
        kept_origins = tuple(requested)

    if is_synthetic:
        limitations.insert(0, SYNTHETIC_LIMITATION)
    if include_news:
        limitations.extend(NEWS_LIMITATIONS)
    else:
        limitations.append("News variant not evaluated (no historical news data supplied).")
    if not include_indicators:
        limitations.append("Indicator variant not evaluated (no indicator data supplied).")

    config = ExperimentConfig(
        dataset_name=dataset_name,
        is_synthetic=is_synthetic,
        models=tuple(model_names),
        horizon=evaluated_horizon,
        min_train_years=min_train_years,
        step=step,
        origins=kept_origins,
        include_indicators=include_indicators,
        indicator_weights=dict(indicator_weights) if include_indicators else None,
        indicator_weight=float(indicator_weight) if include_indicators else None,
        include_news=include_news,
        news_lookback_days=news_lookback_days if include_news else None,
        news_config=dict(news_config or {}) if include_news else None,
        notes=notes,
    )
    metadata = {
        "data_label": "DEMO/SYNTHETIC" if is_synthetic else "REAL",
        "run_timestamp": run_timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requested_horizon": horizon,
        "evaluated_horizons": list(range(1, evaluated_horizon + 1)),
        "origins": [fold.origin for fold in backtest.folds],
        "n_folds": len(backtest.folds),
        "input_fingerprints": {
            "historical_data": _fingerprint(historical_data),
            "indicators": _fingerprint(indicators) if include_indicators else None,
            "news": _fingerprint(news) if include_news else None,
        },
        "limitations": limitations,
    }
    return ExperimentResult(config=config, backtest=backtest, metadata=metadata)
