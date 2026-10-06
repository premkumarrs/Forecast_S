"""
Walk-forward backtesting of baseline forecasting models.

Variants recorded per fold:

* ``baseline``: the model's forecast, fitted on market data up to the origin.
* ``benchmark``: the naive last-value forecast.
* ``indicator_adjusted_past_only`` (only with ``use_indicators=True``): the
  baseline forecast adjusted by the existing ``IndicatorAdjustment``, with the
  indicator adjustment computed using only indicator observations available
  at or before the fold origin. For a fold with origin Y, market training data
  is ``year <= Y`` and the indicator data passed to the adjustment is
  ``year <= Y``; indicator values after Y have no influence.

Production vs. backtest:

* Production currently receives indicator data extending into the forecast
  years (the extraction keeps ``year <= forecast_until``), and
  ``IndicatorAdjustment`` averages growth over every year it receives.
* The backtest intentionally does NOT use those future values: they are not
  as-of-origin data, and using today's values for years after the origin
  would leak the future.
* Therefore this is a point-in-time historical indicator evaluation, not an
  exact replay of the current production indicator pipeline. It measures the
  value of past-only indicator information, not the value of production's
  forward-looking indicator inputs. The adjustment formula and weighting are
  the unchanged production ``IndicatorAdjustment``; only its input differs.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ..adjustments.indicator_adjustment import IndicatorAdjustment
from ..base.models import BaseForecaster
from ..base.validators import validate_time_series
from ..config import MIN_HISTORICAL_YEARS
from ..models.baseline_factory import BaselineModelFactory
from .splits import Fold, walk_forward_folds

logger = logging.getLogger(__name__)

NAIVE_MODEL_NAME = "Naive (last value)"
DEFAULT_MODELS = ("3-yr CAGR", "Damped ETS", "Logistic Growth")
DEFAULT_INDICATOR_WEIGHT = 0.3

VARIANT_BASELINE = "baseline"
VARIANT_BENCHMARK = "benchmark"
VARIANT_INDICATOR_ADJUSTED_PAST_ONLY = "indicator_adjusted_past_only"

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_INVALID = "invalid_prediction"

RECORD_COLUMNS = [
    "origin", "model", "variant", "year", "horizon",
    "actual", "predicted", "error", "n_train", "status", "message",
]
FAILURE_COLUMNS = ["origin", "model", "variant", "error_type", "message"]


class InsufficientIndicatorData(ValueError):
    """Raised when a fold has no usable indicator growth at or before its origin."""


class NaiveLastValueForecaster(BaseForecaster):
    """Benchmark that repeats the last observed value for every forecast year."""

    def __init__(self):
        super().__init__(NAIVE_MODEL_NAME)
        self.last_value = None

    def fit(self, historical_data: pd.DataFrame) -> None:
        self.last_value = float(historical_data.iloc[-1]["value"])
        self._is_fitted = True

    def forecast(self, forecast_years: List[int]) -> List[float]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        return [self.last_value] * len(forecast_years)


@dataclass
class BacktestResult:
    """Per-forecast records plus a log of failed model/fold runs."""

    records: List[Dict] = field(default_factory=list)
    failures: List[Dict] = field(default_factory=list)
    folds: List[Fold] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records, columns=RECORD_COLUMNS)

    def failures_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.failures, columns=FAILURE_COLUMNS)


def _prepare_series(series: pd.DataFrame, min_length: int) -> pd.DataFrame:
    is_valid, message = validate_time_series(series, ["year", "value"], min_length=min_length)
    if not is_valid:
        raise ValueError(f"Invalid backtest series: {message}")
    return series[["year", "value"]].sort_values("year").reset_index(drop=True)


def _prepare_indicators(
    indicators: Optional[pd.DataFrame],
    indicator_weights: Optional[Mapping[str, float]],
    indicator_weight: float,
) -> Tuple[pd.DataFrame, str]:
    """Validate indicator inputs and return the data plus its indicator column name."""
    if indicators is None or not isinstance(indicators, pd.DataFrame) or indicators.empty:
        raise ValueError("use_indicators=True requires a non-empty indicators DataFrame")
    if not indicator_weights:
        raise ValueError("use_indicators=True requires non-empty indicator_weights")
    if isinstance(indicator_weight, bool) or not isinstance(indicator_weight, (int, float)) \
            or not math.isfinite(indicator_weight):
        raise ValueError(f"indicator_weight must be a finite number, got {indicator_weight!r}")

    indicator_col = "indicator_key" if "indicator_key" in indicators.columns else "indicator"
    missing = [col for col in ("year", "value", indicator_col) if col not in indicators.columns]
    if missing:
        raise ValueError(
            f"indicators is missing columns {missing}; expected year, value and "
            "indicator_key (or indicator)"
        )

    data = indicators.copy()
    years = pd.to_numeric(data["year"], errors="coerce")
    if years.isna().any() or (years % 1 != 0).any():
        raise ValueError("indicators 'year' must contain whole-number years")
    data["year"] = years.astype(int)
    values = pd.to_numeric(data["value"], errors="coerce")
    if not values.map(math.isfinite).all():
        raise ValueError("indicators 'value' must contain finite numbers only")
    data["value"] = values.astype(float)

    duplicated = data.duplicated(subset=[indicator_col, "year"])
    if duplicated.any():
        examples = data.loc[duplicated, [indicator_col, "year"]].head(3).values.tolist()
        raise ValueError(
            "indicators must have one row per indicator and year (a single geography); "
            f"duplicates found, e.g. {examples}"
        )

    for name, weight in indicator_weights.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
            raise ValueError(f"indicator_weights[{name!r}] must be a finite number, got {weight!r}")
    unknown = sorted(set(indicator_weights) - set(data[indicator_col]))
    if unknown:
        raise ValueError(f"indicator_weights references indicators not in the data: {unknown}")

    return data, indicator_col


def _create_model(model_name: str) -> BaseForecaster:
    if model_name == NAIVE_MODEL_NAME:
        return NaiveLastValueForecaster()
    return BaselineModelFactory.create_model(model_name)


def _forecast_rows(forecast_df: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    forecast_rows = forecast_df[forecast_df["type"] == "Forecast"]
    years = [int(y) for y in forecast_rows["year"]]
    if years != list(fold.test_years):
        raise ValueError(
            f"Model returned forecast years {years}, expected {list(fold.test_years)}"
        )
    return forecast_rows


def _forecast_fold(model_name: str, train: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    """Fit a fresh model on the training window and return its forecast DataFrame."""
    model = _create_model(model_name)
    forecast_df = model.generate_forecast_dataframe(
        train, fold.origin, fold.test_years[-1]
    )
    _forecast_rows(forecast_df, fold)
    return forecast_df


def _indicator_adjusted_values(
    baseline_df: pd.DataFrame,
    indicators: pd.DataFrame,
    indicator_col: str,
    indicator_weights: Mapping[str, float],
    indicator_weight: float,
    fold: Fold,
) -> List[float]:
    """Apply the existing IndicatorAdjustment using only indicator observations
    available at or before the fold origin.

    Rows are sorted by indicator and year so the year-over-year growth inside
    ``IndicatorAdjustment`` compares each year with its predecessor; because
    every row has ``year <= origin``, no growth term can involve a later year.
    """
    available = indicators[indicators["year"] <= fold.origin].sort_values(
        [indicator_col, "year"], kind="mergesort"
    )
    adjustment = IndicatorAdjustment(weight=indicator_weight)
    adjusted_df, _ = adjustment.apply(
        baseline_df, available, {"indicator_weights": dict(indicator_weights)}
    )

    observed = {name for signals in adjustment.indicator_signals.values() for name in signals}
    unobserved = sorted(set(indicator_weights) - observed)
    if unobserved:
        raise InsufficientIndicatorData(
            f"No year-over-year growth observable at or before origin {fold.origin} "
            f"for indicator(s) {unobserved}; each needs two observations through the origin"
        )
    return [float(v) for v in _forecast_rows(adjusted_df, fold)["value_hat"]]


def _log_failure(result: BacktestResult, fold: Fold, model_name: str, variant: str,
                 exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    logger.warning(
        "Backtest failed for model=%s variant=%s origin=%s: %s",
        model_name, variant, fold.origin, message,
    )
    result.failures.append({
        "origin": fold.origin,
        "model": model_name,
        "variant": variant,
        "error_type": type(exc).__name__,
        "message": str(exc),
    })
    return message


def _append_records(
    result: BacktestResult,
    fold: Fold,
    model_name: str,
    variant: str,
    predictions: Optional[List[float]],
    failure_message: str,
    actual_by_year: Dict[int, float],
    n_train: int,
) -> None:
    for step_index, year in enumerate(fold.test_years):
        actual = actual_by_year[year]
        predicted = float("nan")
        status = STATUS_FAILED
        message = failure_message
        if predictions is not None:
            predicted = predictions[step_index]
            if math.isfinite(predicted):
                status, message = STATUS_OK, ""
            else:
                status, message = STATUS_INVALID, f"Non-finite prediction: {predicted}"
        result.records.append({
            "origin": fold.origin,
            "model": model_name,
            "variant": variant,
            "year": year,
            "horizon": step_index + 1,
            "actual": actual,
            "predicted": predicted,
            "error": predicted - actual if status == STATUS_OK else float("nan"),
            "n_train": n_train,
            "status": status,
            "message": message,
        })


def run_backtest(
    series: pd.DataFrame,
    models: Sequence[str] = DEFAULT_MODELS,
    horizon: int = 3,
    min_train_years: int = MIN_HISTORICAL_YEARS,
    step: int = 1,
    include_naive: bool = True,
    use_indicators: bool = False,
    indicators: Optional[pd.DataFrame] = None,
    indicator_weights: Optional[Mapping[str, float]] = None,
    indicator_weight: float = DEFAULT_INDICATOR_WEIGHT,
) -> BacktestResult:
    """Evaluate baseline models with expanding-window walk-forward backtesting.

    For every fold, each model is fitted only on observations up to the fold
    origin and its forecasts are compared with the held-out actuals. A model
    that fails on a fold is logged in ``failures`` and its rows are kept with
    ``status="failed"`` and no prediction, so the rest of the backtest is
    unaffected.

    With ``use_indicators=True``, each non-benchmark model also produces an
    ``indicator_adjusted_past_only`` variant: its baseline forecast passed
    through the existing ``IndicatorAdjustment``, with the indicator adjustment
    computed using only indicator observations available at or before the fold
    origin. Production instead receives indicator data extending into the
    forecast years; this backtest intentionally does not use those future
    values, so it is a point-in-time historical indicator evaluation, not an
    exact replay of the production indicator pipeline (see the module
    docstring). Baseline records are unchanged. If a weighted indicator has no
    observable growth through the origin, or the baseline failed, that fold's
    adjusted rows are marked ``failed`` with the reason rather than being
    scored as a zero adjustment.

    Args:
        series: Annual series with ``year`` and ``value`` columns. Years must be
            unique and consecutive; values must be numeric, non-missing and
            non-negative (checked by ``validate_time_series``).
        models: Sequence of model names known to ``BaselineModelFactory``, or
            ``NAIVE_MODEL_NAME``. A single string is rejected.
        horizon: Years forecast per fold.
        min_train_years: Years in the first training window; at least
            ``MIN_HISTORICAL_YEARS``, the minimum the baseline models accept.
        step: Years the origin advances between folds.
        include_naive: Add the naive last-value benchmark.
        use_indicators: Also evaluate the ``indicator_adjusted_past_only``
            variant. Defaults to False, which gives the baseline-only results.
        indicators: Indicator observations for the same geography as
            ``series``, with ``year``, ``value`` and ``indicator_key`` (or
            ``indicator``) columns and one row per indicator and year.
            Required when ``use_indicators`` is True.
        indicator_weights: Weight per indicator, as in the app's
            ``indicator_weights`` config. Every key must exist in ``indicators``.
        indicator_weight: Overall indicator weight, as in the app's
            ``adjustment_weights['indicator_weight']``.

    Returns:
        BacktestResult with one record per model, variant, fold and forecast
        year. ``error`` is ``predicted - actual``.

    Raises:
        TypeError: If ``models`` is a single string or a parameter has the wrong type.
        ValueError: If the series, model names, indicator inputs or fold
            configuration are invalid.
    """
    if isinstance(models, str):
        raise TypeError(
            "models must be a sequence of model names, not a single string; "
            f"use [{models!r}]"
        )
    model_names = list(dict.fromkeys(models))
    available = set(BaselineModelFactory.get_available_models()) | {NAIVE_MODEL_NAME}
    unknown = [name for name in model_names if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown model(s) {unknown}. Available: {sorted(available)}"
        )
    if include_naive and NAIVE_MODEL_NAME not in model_names:
        model_names.append(NAIVE_MODEL_NAME)
    if not model_names:
        raise ValueError("No models to backtest")
    if (
        isinstance(min_train_years, int)
        and not isinstance(min_train_years, bool)
        and min_train_years < MIN_HISTORICAL_YEARS
    ):
        raise ValueError(
            f"min_train_years must be >= {MIN_HISTORICAL_YEARS}, got {min_train_years}; "
            "the baseline models need at least that many observations"
        )

    indicator_data: Optional[pd.DataFrame] = None
    indicator_col = ""
    if use_indicators:
        indicator_data, indicator_col = _prepare_indicators(
            indicators, indicator_weights, indicator_weight
        )
    elif indicators is not None or indicator_weights is not None:
        raise ValueError("indicators/indicator_weights were given but use_indicators is False")

    data = _prepare_series(series, min_length=1)
    folds = walk_forward_folds(
        years=data["year"].tolist(),
        min_train_years=min_train_years,
        horizon=horizon,
        step=step,
    )
    actual_by_year = dict(zip(data["year"].astype(int), data["value"].astype(float)))

    result = BacktestResult(folds=folds)
    for fold in folds:
        train = data[data["year"] <= fold.origin]
        for model_name in model_names:
            is_benchmark = model_name == NAIVE_MODEL_NAME
            variant = VARIANT_BENCHMARK if is_benchmark else VARIANT_BASELINE
            baseline_df: Optional[pd.DataFrame] = None
            predictions: Optional[List[float]] = None
            failure_message = ""
            try:
                baseline_df = _forecast_fold(model_name, train, fold)
                predictions = [float(v) for v in _forecast_rows(baseline_df, fold)["value_hat"]]
            except Exception as exc:
                failure_message = _log_failure(result, fold, model_name, variant, exc)
            _append_records(result, fold, model_name, variant, predictions,
                            failure_message, actual_by_year, len(train))

            if not use_indicators or is_benchmark:
                continue
            adjusted: Optional[List[float]] = None
            adjusted_message = ""
            if baseline_df is None:
                adjusted_message = f"Baseline forecast failed: {failure_message}"
                result.failures.append({
                    "origin": fold.origin,
                    "model": model_name,
                    "variant": VARIANT_INDICATOR_ADJUSTED_PAST_ONLY,
                    "error_type": "BaselineFailed",
                    "message": adjusted_message,
                })
            else:
                try:
                    adjusted = _indicator_adjusted_values(
                        baseline_df, indicator_data, indicator_col,
                        indicator_weights, indicator_weight, fold,
                    )
                except Exception as exc:
                    adjusted_message = _log_failure(
                        result, fold, model_name, VARIANT_INDICATOR_ADJUSTED_PAST_ONLY, exc
                    )
            _append_records(result, fold, model_name, VARIANT_INDICATOR_ADJUSTED_PAST_ONLY,
                            adjusted, adjusted_message, actual_by_year, len(train))
    return result
