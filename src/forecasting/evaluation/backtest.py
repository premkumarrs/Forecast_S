"""
Walk-forward backtesting of baseline forecasting models.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import pandas as pd

from ..base.models import BaseForecaster
from ..base.validators import validate_time_series
from ..config import MIN_HISTORICAL_YEARS
from ..models.baseline_factory import BaselineModelFactory
from .splits import Fold, walk_forward_folds

logger = logging.getLogger(__name__)

NAIVE_MODEL_NAME = "Naive (last value)"
DEFAULT_MODELS = ("3-yr CAGR", "Damped ETS", "Logistic Growth")

VARIANT_BASELINE = "baseline"
VARIANT_BENCHMARK = "benchmark"

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_INVALID = "invalid_prediction"

RECORD_COLUMNS = [
    "origin", "model", "variant", "year", "horizon",
    "actual", "predicted", "error", "n_train", "status", "message",
]
FAILURE_COLUMNS = ["origin", "model", "variant", "error_type", "message"]


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


def _create_model(model_name: str) -> BaseForecaster:
    if model_name == NAIVE_MODEL_NAME:
        return NaiveLastValueForecaster()
    return BaselineModelFactory.create_model(model_name)


def _forecast_fold(model_name: str, train: pd.DataFrame, fold: Fold) -> List[float]:
    """Fit a fresh model on the training window and forecast the fold's test years."""
    model = _create_model(model_name)
    forecast_df = model.generate_forecast_dataframe(
        train, fold.origin, fold.test_years[-1]
    )
    forecast_rows = forecast_df[forecast_df["type"] == "Forecast"]
    years = [int(y) for y in forecast_rows["year"]]
    if years != list(fold.test_years):
        raise ValueError(
            f"Model returned forecast years {years}, expected {list(fold.test_years)}"
        )
    return [float(v) for v in forecast_rows["value_hat"]]


def run_backtest(
    series: pd.DataFrame,
    models: Sequence[str] = DEFAULT_MODELS,
    horizon: int = 3,
    min_train_years: int = MIN_HISTORICAL_YEARS,
    step: int = 1,
    include_naive: bool = True,
) -> BacktestResult:
    """Evaluate baseline models with expanding-window walk-forward backtesting.

    For every fold, each model is fitted only on observations up to the fold
    origin and its forecasts are compared with the held-out actuals. A model
    that fails on a fold is logged in ``failures`` and its rows are kept with
    ``status="failed"`` and no prediction, so the rest of the backtest is
    unaffected.

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

    Returns:
        BacktestResult with one record per model, fold and forecast year.
        ``error`` is ``predicted - actual``.

    Raises:
        TypeError: If ``models`` is a single string or a parameter has the wrong type.
        ValueError: If the series, model names or fold configuration are invalid.
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
            variant = VARIANT_BENCHMARK if model_name == NAIVE_MODEL_NAME else VARIANT_BASELINE
            predictions: Optional[List[float]] = None
            failure_message = ""
            try:
                predictions = _forecast_fold(model_name, train, fold)
            except Exception as exc:
                failure_message = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Backtest failed for model=%s origin=%s: %s",
                    model_name, fold.origin, failure_message,
                )
                result.failures.append({
                    "origin": fold.origin,
                    "model": model_name,
                    "variant": variant,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                })

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
                    "n_train": len(train),
                    "status": status,
                    "message": message,
                })
    return result
