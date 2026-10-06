"""
Tests for the walk-forward backtesting framework.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.forecasting.base.models import BaseForecaster
from src.forecasting.evaluation import (
    NAIVE_MODEL_NAME,
    run_backtest,
    summarize_backtest,
    walk_forward_folds,
)
from src.forecasting.models.baseline_factory import BaselineModelFactory


def geometric_series(start_year=2010, n_years=12, start_value=100.0, growth=0.10):
    years = np.arange(start_year, start_year + n_years)
    return pd.DataFrame({
        "year": years,
        "value": start_value * (1 + growth) ** np.arange(n_years),
    })


def noisy_series(start_year=2010, n_years=14, seed=0):
    rng = np.random.default_rng(seed)
    steps = np.arange(n_years)
    return pd.DataFrame({
        "year": np.arange(start_year, start_year + n_years),
        "value": 100 * 1.08 ** steps * (1 + rng.normal(0, 0.03, n_years)),
    })


# --- walk_forward_folds ---

def test_walk_forward_folds_count_and_boundaries():
    folds = walk_forward_folds(range(2010, 2020), min_train_years=4, horizon=2, step=1)

    assert len(folds) == 5
    assert folds[0].origin == 2013
    assert folds[0].train_years == (2010, 2011, 2012, 2013)
    assert folds[0].test_years == (2014, 2015)
    assert folds[-1].origin == 2017
    assert folds[-1].train_years == tuple(range(2010, 2018))
    assert folds[-1].test_years == (2018, 2019)
    for fold in folds:
        assert fold.train_years[0] == 2010
        assert fold.train_years[-1] == fold.origin
        assert fold.test_years[0] == fold.origin + 1
        assert fold.horizon == 2


def test_walk_forward_folds_step_and_determinism():
    folds = walk_forward_folds(range(2010, 2020), min_train_years=4, horizon=2, step=2)
    assert [f.origin for f in folds] == [2013, 2015, 2017]

    shuffled = [2015, 2010, 2019, 2012, 2011, 2018, 2013, 2017, 2014, 2016]
    assert walk_forward_folds(shuffled, 4, 2, 2) == folds


@pytest.mark.parametrize("kwargs, error", [
    (dict(years=[2010, 2011, 2013, 2014, 2015], min_train_years=2, horizon=1), ValueError),
    (dict(years=range(2010, 2015), min_train_years=4, horizon=2), ValueError),
    (dict(years=range(2010, 2020), min_train_years=0, horizon=2), ValueError),
    (dict(years=range(2010, 2020), min_train_years=3, horizon=0), ValueError),
    (dict(years=range(2010, 2020), min_train_years=3, horizon=2, step=0), ValueError),
    (dict(years=range(2010, 2020), min_train_years=3, horizon=2, window="rolling"), ValueError),
    (dict(years=[], min_train_years=3, horizon=2), ValueError),
    (dict(years=range(2010, 2020), min_train_years=3.0, horizon=2), TypeError),
    (dict(years=range(2010, 2020), min_train_years=True, horizon=2), TypeError),
    (dict(years=[2010, 2011.5, 2012], min_train_years=1, horizon=1), TypeError),
    (dict(years=[2010, 2011, 2012, 2012, 2013], min_train_years=2, horizon=1), ValueError),
])
def test_walk_forward_folds_rejects_invalid_configuration(kwargs, error):
    with pytest.raises(error):
        walk_forward_folds(**kwargs)


# --- run_backtest ---

def test_geometric_series_has_near_zero_cagr_error():
    result = run_backtest(geometric_series(), models=["3-yr CAGR"], horizon=3, min_train_years=4)
    frame = result.to_frame()
    cagr = frame[frame["model"] == "3-yr CAGR"]

    assert not cagr.empty
    assert (cagr["status"] == "ok").all()
    assert (frame["horizon"] == frame["year"] - frame["origin"]).all()
    np.testing.assert_allclose(cagr["predicted"], cagr["actual"], rtol=1e-9)

    summary = summarize_backtest(result, group_by=["model"])
    assert summary.set_index("model").loc["3-yr CAGR", "mape"] < 1e-6


def test_values_after_origin_cannot_change_fold_predictions():
    original = noisy_series()
    models = ["3-yr CAGR", "Damped ETS", "Logistic Growth"]
    kwargs = dict(models=models, horizon=3, min_train_years=5)
    target_origin = 2016

    altered = original.copy()
    after_origin = altered["year"] > target_origin
    altered.loc[after_origin, "value"] = altered.loc[after_origin, "value"] * 3 + 500

    frame_a = run_backtest(original, **kwargs).to_frame()
    frame_b = run_backtest(altered, **kwargs).to_frame()
    fold_a = frame_a[frame_a["origin"] == target_origin].reset_index(drop=True)
    fold_b = frame_b[frame_b["origin"] == target_origin].reset_index(drop=True)

    assert set(fold_a["model"]) == set(models) | {NAIVE_MODEL_NAME}
    assert (fold_a["status"] == "ok").all()
    pd.testing.assert_frame_equal(
        fold_a[["model", "year", "predicted"]], fold_b[["model", "year", "predicted"]]
    )
    assert not np.allclose(fold_a["actual"], fold_b["actual"])


def test_naive_benchmark_repeats_last_training_value():
    series = geometric_series(growth=0.10)
    result = run_backtest(series, models=[], horizon=3, min_train_years=4)
    frame = result.to_frame()

    assert set(frame["model"]) == {NAIVE_MODEL_NAME}
    assert set(frame["variant"]) == {"benchmark"}
    last_value_by_year = series.set_index("year")["value"]
    for _, row in frame.iterrows():
        assert row["predicted"] == pytest.approx(last_value_by_year[row["origin"]])
        assert row["error"] == pytest.approx(row["predicted"] - row["actual"])

    summary = summarize_backtest(result, group_by=["horizon"]).set_index("horizon")
    for h in (1, 2, 3):
        expected_mape = (1 - 1.10 ** -h) * 100
        assert summary.loc[h, "mape"] == pytest.approx(expected_mape)
        assert summary.loc[h, "bias"] < 0


class AlwaysFailsForecaster(BaseForecaster):
    def __init__(self):
        super().__init__("Always Fails")

    def fit(self, historical_data):
        raise RuntimeError("deliberate failure")

    def forecast(self, forecast_years):
        raise AssertionError("unreachable")


class FailsOnShortHistoryForecaster(BaseForecaster):
    """Fails on early folds only, and emits NaN on one later fold."""

    def __init__(self):
        super().__init__("Flaky")
        self.n_train = 0
        self.last_value = None

    def fit(self, historical_data):
        self.n_train = len(historical_data)
        if self.n_train < 6:
            raise ValueError("not enough history")
        self.last_value = float(historical_data.iloc[-1]["value"])
        self._is_fitted = True

    def forecast(self, forecast_years):
        if self.n_train == 7:
            return [float("nan")] * len(forecast_years)
        return [self.last_value] * len(forecast_years)


def test_failed_model_does_not_crash_backtest(monkeypatch):
    monkeypatch.setitem(BaselineModelFactory._models, "Always Fails", AlwaysFailsForecaster)
    monkeypatch.setitem(BaselineModelFactory._models, "Flaky", FailsOnShortHistoryForecaster)

    result = run_backtest(
        noisy_series(n_years=12),
        models=["3-yr CAGR", "Always Fails", "Flaky"],
        horizon=2,
        min_train_years=4,
    )
    frame = result.to_frame()

    n_folds = len(result.folds)
    assert n_folds == 7
    for model in ["3-yr CAGR", "Always Fails", "Flaky", NAIVE_MODEL_NAME]:
        assert len(frame[frame["model"] == model]) == n_folds * 2

    assert (frame.loc[frame["model"] == "3-yr CAGR", "status"] == "ok").all()

    always = frame[frame["model"] == "Always Fails"]
    assert (always["status"] == "failed").all()
    assert always["predicted"].isna().all()
    assert always["message"].str.contains("deliberate failure").all()

    failures = result.failures_frame()
    assert (failures["model"] == "Always Fails").sum() == n_folds
    assert (failures["model"] == "Flaky").sum() == 2

    flaky = frame[frame["model"] == "Flaky"]
    assert (flaky["status"] == "failed").sum() == 4
    assert (flaky["status"] == "invalid_prediction").sum() == 2
    assert (flaky["status"] == "ok").sum() == 8

    summary = summarize_backtest(result, group_by=["model"]).set_index("model")
    assert summary.loc["Always Fails", "n_valid"] == 0
    assert summary.loc["Always Fails", "n_failed"] == n_folds * 2
    assert np.isnan(summary.loc["Always Fails", ["mape", "rmse", "mae", "bias"]].astype(float)).all()

    assert summary.loc["Flaky", "n_valid"] == 8
    assert summary.loc["Flaky", "n_failed"] == 4
    assert summary.loc["Flaky", "n_invalid"] == 2
    assert np.isfinite(summary.loc["Flaky", "mape"])
    assert np.isfinite(summary.loc["3-yr CAGR", "rmse"])


def test_run_backtest_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        run_backtest(noisy_series(), models=["Not A Model"])


def test_run_backtest_rejects_invalid_series():
    series = noisy_series()
    series.loc[3, "value"] = np.nan
    with pytest.raises(ValueError, match="Invalid backtest series"):
        run_backtest(series)


def test_run_backtest_rejects_duplicate_years():
    series = pd.concat([noisy_series(), noisy_series().iloc[[4]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate years"):
        run_backtest(series)


@pytest.mark.parametrize("min_train_years", [1, 2])
def test_run_backtest_rejects_min_train_below_model_minimum(min_train_years):
    with pytest.raises(ValueError, match="min_train_years must be >= 3"):
        run_backtest(noisy_series(), min_train_years=min_train_years)


def test_run_backtest_rejects_single_string_models():
    with pytest.raises(TypeError, match="not a single string"):
        run_backtest(noisy_series(), models="3-yr CAGR")


def test_naive_model_can_be_requested_by_name():
    explicit = run_backtest(
        noisy_series(), models=[NAIVE_MODEL_NAME], include_naive=False, horizon=2, min_train_years=4
    ).to_frame()
    implicit = run_backtest(noisy_series(), models=[], horizon=2, min_train_years=4).to_frame()

    assert set(explicit["model"]) == {NAIVE_MODEL_NAME}
    pd.testing.assert_frame_equal(explicit, implicit)

    combined = run_backtest(
        noisy_series(), models=[NAIVE_MODEL_NAME], include_naive=True, horizon=2, min_train_years=4
    ).to_frame()
    assert len(combined) == len(implicit)


# --- summarize_backtest ---

def test_summarize_backtest_grouping_and_validation():
    result = run_backtest(noisy_series(), horizon=2, min_train_years=5)

    default = summarize_backtest(result)
    assert list(default.columns[:3]) == ["model", "variant", "horizon"]
    assert set(default["horizon"]) == {1, 2}
    assert default["n_forecasts"].sum() == len(result.records)

    overall = summarize_backtest(result.to_frame(), group_by=[])
    assert len(overall) == 1
    assert overall.loc[0, "n_forecasts"] == len(result.records)

    with pytest.raises(ValueError, match="Cannot group by"):
        summarize_backtest(result, group_by=["actual"])


def test_mape_is_nan_when_all_actuals_are_zero():
    frame = pd.DataFrame({
        "origin": [2015, 2015], "model": ["m", "m"], "variant": ["baseline"] * 2,
        "year": [2016, 2017], "horizon": [1, 2], "actual": [0.0, 0.0],
        "predicted": [1.0, 2.0], "error": [1.0, 2.0], "n_train": [5, 5],
        "status": ["ok", "ok"], "message": ["", ""],
    })
    summary = summarize_backtest(frame, group_by=[])
    assert np.isnan(summary.loc[0, "mape"])
    assert summary.loc[0, "n_zero_actual"] == 2
    assert summary.loc[0, "mae"] == pytest.approx(1.5)


def _records(actual, predicted, status):
    n = len(actual)
    return pd.DataFrame({
        "origin": [2015] * n, "model": ["m"] * n, "variant": ["baseline"] * n,
        "year": list(range(2016, 2016 + n)), "horizon": list(range(1, n + 1)),
        "actual": actual, "predicted": predicted, "status": status,
    })


def test_ok_rows_with_non_finite_values_are_excluded_from_metrics():
    frame = _records(
        actual=[100.0, 100.0, 100.0, np.nan],
        predicted=[110.0, np.nan, np.inf, 90.0],
        status=["ok", "ok", "ok", "ok"],
    )
    summary = summarize_backtest(frame, group_by=[])

    assert summary.loc[0, "n_forecasts"] == 4
    assert summary.loc[0, "n_valid"] == 1
    assert summary.loc[0, "n_invalid"] == 3
    assert summary.loc[0, "mae"] == pytest.approx(10.0)
    assert summary.loc[0, "mape"] == pytest.approx(10.0)
    assert summary.loc[0, "bias"] == pytest.approx(10.0)


def test_non_numeric_values_are_invalid_not_valid():
    frame = _records(actual=[100.0, 100.0], predicted=[105.0, "bad"], status=["ok", "ok"])
    summary = summarize_backtest(frame, group_by=[])
    assert summary.loc[0, "n_valid"] == 1
    assert summary.loc[0, "n_invalid"] == 1
    assert summary.loc[0, "rmse"] == pytest.approx(5.0)


def test_failed_status_is_never_counted_even_with_finite_values():
    frame = _records(actual=[100.0, 100.0], predicted=[100.0, 500.0], status=["ok", "failed"])
    summary = summarize_backtest(frame, group_by=[])
    assert summary.loc[0, "n_valid"] == 1
    assert summary.loc[0, "n_failed"] == 1
    assert summary.loc[0, "mae"] == pytest.approx(0.0)


@pytest.mark.parametrize("dropped", ["status", "actual", "predicted"])
def test_summarize_backtest_requires_columns(dropped):
    frame = _records(actual=[100.0], predicted=[100.0], status=["ok"]).drop(columns=[dropped])
    with pytest.raises(ValueError, match="missing required columns"):
        summarize_backtest(frame, group_by=[])


def test_summarize_backtest_requires_group_columns_present():
    frame = _records(actual=[100.0], predicted=[100.0], status=["ok"]).drop(columns=["horizon"])
    with pytest.raises(ValueError, match="missing required columns"):
        summarize_backtest(frame, group_by=["horizon"])


# --- chronological ordering fix ---

def test_baseline_forecast_is_independent_of_input_row_order():
    series = noisy_series()
    shuffled = series.sample(frac=1, random_state=1)
    for model in ["3-yr CAGR", "Damped ETS", "Logistic Growth"]:
        ordered = BaselineModelFactory.generate_baseline_forecast(series, model, 2019, 2022)
        unordered = BaselineModelFactory.generate_baseline_forecast(shuffled, model, 2019, 2022)
        pd.testing.assert_frame_equal(ordered, unordered)
