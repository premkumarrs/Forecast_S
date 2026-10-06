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

from src.forecasting.adjustments import news_adjustment as news_adjustment_module
from src.forecasting.adjustments.indicator_adjustment import IndicatorAdjustment
from src.forecasting.adjustments.news_adjustment import NewsAdjustment
from src.forecasting.adjustments.temporal_decay import TemporalDecay
from src.forecasting.base.models import BaseForecaster
from src.forecasting.evaluation import (
    NAIVE_MODEL_NAME,
    run_backtest,
    summarize_backtest,
    walk_forward_folds,
)
from src.forecasting.evaluation import backtest as backtest_module
from src.forecasting.evaluation.news_history import (
    ANALYST_FAILURE_REASON,
    HeadlineAnalysisCache,
    HeadlineAnalysisError,
    analysis_fingerprint,
    analyze_headlines_cached,
    fetch_headlines_for_origins,
    news_as_of,
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


# --- Phase 2: leakage-controlled indicator evaluation ---
# "indicator_adjusted_past_only" = indicator adjustment computed using only
# indicator observations available at or before the fold origin. Unlike
# production, which receives indicator data extending into forecast years,
# this is a point-in-time historical evaluation, not a production replay.

MODELS = ["3-yr CAGR", "Damped ETS", "Logistic Growth"]
INDICATOR_WEIGHTS = {"gdp": 1.0, "internet": 0.5}
INDICATOR_WEIGHT = 0.3
TARGET_ORIGIN = 2016


def indicator_frame(start_year=2008, end_year=2023, seed=1):
    rng = np.random.default_rng(seed)
    years = np.arange(start_year, end_year + 1)
    steps = np.arange(len(years))
    gdp = 100 * 1.03 ** steps * (1 + rng.normal(0, 0.01, len(years)))
    internet = 50 * 1.06 ** steps * (1 + rng.normal(0, 0.01, len(years)))
    return pd.concat([
        pd.DataFrame({"year": years, "indicator_key": "gdp", "value": gdp}),
        pd.DataFrame({"year": years, "indicator_key": "internet", "value": internet}),
    ], ignore_index=True)


def indicator_backtest(indicators, series=None, models=MODELS, horizon=3, **kwargs):
    return run_backtest(
        noisy_series() if series is None else series,
        models=models,
        horizon=horizon,
        min_train_years=5,
        use_indicators=True,
        indicators=indicators,
        indicator_weights=INDICATOR_WEIGHTS,
        indicator_weight=INDICATOR_WEIGHT,
        **kwargs,
    )


def fold_rows(frame, origin, variant):
    rows = frame[(frame["origin"] == origin) & (frame["variant"] == variant)]
    return rows.sort_values(["model", "year"]).reset_index(drop=True)


def test_future_indicator_values_cannot_change_adjusted_prediction():
    original = indicator_frame()
    altered = original.copy()
    future = altered["year"] > TARGET_ORIGIN
    altered.loc[future, "value"] = altered.loc[future, "value"] * 5 + 1000

    a = fold_rows(indicator_backtest(original).to_frame(), TARGET_ORIGIN, "indicator_adjusted_past_only")
    b = fold_rows(indicator_backtest(altered).to_frame(), TARGET_ORIGIN, "indicator_adjusted_past_only")
    baseline = fold_rows(indicator_backtest(original).to_frame(), TARGET_ORIGIN, "baseline")

    assert set(a["model"]) == set(MODELS)
    assert (a["status"] == "ok").all()
    pd.testing.assert_series_equal(a["predicted"], b["predicted"])
    assert not np.allclose(a["predicted"], baseline["predicted"])


@pytest.mark.parametrize("changed_year", [TARGET_ORIGIN, TARGET_ORIGIN - 3])
def test_historical_indicator_values_do_change_adjusted_prediction(changed_year):
    original = indicator_frame()
    altered = original.copy()
    row = (altered["year"] == changed_year) & (altered["indicator_key"] == "gdp")
    altered.loc[row, "value"] = altered.loc[row, "value"] * 1.2

    a = fold_rows(indicator_backtest(original).to_frame(), TARGET_ORIGIN, "indicator_adjusted_past_only")
    b = fold_rows(indicator_backtest(altered).to_frame(), TARGET_ORIGIN, "indicator_adjusted_past_only")

    assert (a["status"] == "ok").all() and (b["status"] == "ok").all()
    assert (np.abs(a["predicted"] - b["predicted"]) > 1e-9).all()


def test_indicator_data_never_changes_baseline_or_benchmark_records():
    original = indicator_frame()
    altered = original.copy()
    altered["value"] = altered["value"] * np.linspace(0.5, 3.0, len(altered))

    def non_adjusted(frame):
        return frame[frame["variant"] != "indicator_adjusted_past_only"].reset_index(drop=True)

    plain = run_backtest(noisy_series(), models=MODELS, horizon=3, min_train_years=5).to_frame()
    with_original = non_adjusted(indicator_backtest(original).to_frame())
    with_altered = non_adjusted(indicator_backtest(altered).to_frame())

    pd.testing.assert_frame_equal(with_original, plain)
    pd.testing.assert_frame_equal(with_altered, plain)


def test_without_indicators_matches_phase1_baseline_forecasts():
    series = noisy_series()
    result = run_backtest(series, models=MODELS, horizon=3, min_train_years=5)
    frame = result.to_frame()

    assert set(frame["variant"]) == {"baseline", "benchmark"}
    pd.testing.assert_frame_equal(
        frame, run_backtest(series, models=MODELS, horizon=3, min_train_years=5,
                            use_indicators=False).to_frame()
    )
    for fold in result.folds:
        for model in MODELS:
            expected = BaselineModelFactory.generate_baseline_forecast(
                series[series["year"] <= fold.origin], model, fold.origin, fold.test_years[-1]
            )
            expected = expected[expected["type"] == "Forecast"]["value_hat"].to_numpy()
            got = frame[(frame["origin"] == fold.origin) & (frame["model"] == model)]
            np.testing.assert_allclose(got["predicted"].to_numpy(), expected, rtol=0, atol=0)


def test_past_only_variant_name_is_explicit():
    assert backtest_module.VARIANT_INDICATOR_ADJUSTED_PAST_ONLY == "indicator_adjusted_past_only"
    assert backtest_module.VARIANT_BASELINE == "baseline"
    assert backtest_module.VARIANT_BENCHMARK == "benchmark"
    frame = indicator_backtest(indicator_frame()).to_frame()
    assert "indicator_adjusted" not in set(frame["variant"])


def test_variants_are_separately_identifiable():
    result = indicator_backtest(indicator_frame())
    frame = result.to_frame()

    assert set(frame["variant"]) == {"baseline", "indicator_adjusted_past_only", "benchmark"}
    for model in MODELS:
        baseline = frame[(frame["model"] == model) & (frame["variant"] == "baseline")]
        adjusted = frame[(frame["model"] == model) & (frame["variant"] == "indicator_adjusted_past_only")]
        assert len(baseline) == len(adjusted) == len(result.folds) * 3
        assert list(baseline[["origin", "year"]].itertuples(index=False)) == \
            list(adjusted[["origin", "year"]].itertuples(index=False))
    naive = frame[frame["model"] == NAIVE_MODEL_NAME]
    assert set(naive["variant"]) == {"benchmark"}

    summary = summarize_backtest(result)
    pairs = set(zip(summary["model"], summary["variant"]))
    for model in MODELS:
        assert {(model, "baseline"), (model, "indicator_adjusted_past_only")} <= pairs
    assert (NAIVE_MODEL_NAME, "indicator_adjusted_past_only") not in pairs
    assert summary[["mape", "rmse", "mae", "bias"]].notna().all().all()


def test_indicator_adjustment_is_applied_at_every_horizon():
    indicators = indicator_frame()
    horizon = 4
    result = indicator_backtest(indicators, horizon=horizon)
    frame = result.to_frame()
    config = {"indicator_weights": INDICATOR_WEIGHTS}

    for fold in result.folds:
        available = indicators[indicators["year"] <= fold.origin].sort_values(["indicator_key", "year"])
        expected_factor = 1 + IndicatorAdjustment(weight=INDICATOR_WEIGHT).get_weighted_adjustment(
            IndicatorAdjustment(weight=INDICATOR_WEIGHT).calculate(available, config)
        )
        for model in MODELS:
            base = fold_rows(frame, fold.origin, "baseline")
            adj = fold_rows(frame, fold.origin, "indicator_adjusted_past_only")
            base, adj = base[base["model"] == model], adj[adj["model"] == model]
            assert list(adj["horizon"]) == list(range(1, horizon + 1))
            assert list(adj["year"]) == list(fold.test_years)
            np.testing.assert_allclose(
                adj["predicted"].to_numpy() / base["predicted"].to_numpy(), expected_factor, rtol=1e-12
            )
            np.testing.assert_allclose(adj["error"], adj["predicted"] - adj["actual"])


def test_past_only_adjustment_matches_hand_calculation():
    years = np.arange(2008, 2024)
    indicators = pd.DataFrame({"year": years, "indicator_key": "gdp",
                               "value": 100 * 1.10 ** np.arange(len(years))})
    result = run_backtest(
        noisy_series(), models=["3-yr CAGR"], horizon=2, min_train_years=5, include_naive=False,
        use_indicators=True, indicators=indicators, indicator_weights={"gdp": 1.0},
        indicator_weight=INDICATOR_WEIGHT,
    )
    frame = result.to_frame()
    base = fold_rows(frame, TARGET_ORIGIN, "baseline")
    adj = fold_rows(frame, TARGET_ORIGIN, "indicator_adjusted_past_only")

    # Years 2008..2016 are visible: 8 growth terms of 10% plus the first year,
    # which IndicatorAdjustment counts as a zero signal.
    n_years = TARGET_ORIGIN - 2008 + 1
    expected_factor = 1 + INDICATOR_WEIGHT * (0.10 * (n_years - 1) / n_years)
    np.testing.assert_allclose(adj["predicted"] / base["predicted"], expected_factor, rtol=1e-12)


def test_indicator_row_order_does_not_matter():
    indicators = indicator_frame()
    shuffled = indicators.sample(frac=1, random_state=3).reset_index(drop=True)
    a = indicator_backtest(indicators).to_frame()
    b = indicator_backtest(shuffled).to_frame()
    pd.testing.assert_frame_equal(a, b)


def test_folds_without_past_indicator_growth_fail_explicitly():
    late = indicator_frame(start_year=TARGET_ORIGIN)
    result = indicator_backtest(late)
    frame = result.to_frame()
    adjusted = frame[frame["variant"] == "indicator_adjusted_past_only"]

    early = adjusted[adjusted["origin"] <= TARGET_ORIGIN]
    later = adjusted[adjusted["origin"] > TARGET_ORIGIN]
    assert not early.empty and not later.empty
    assert (early["status"] == "failed").all()
    assert early["predicted"].isna().all()
    assert early["message"].str.contains("No year-over-year growth observable").all()
    assert (later["status"] == "ok").all()
    assert (frame.loc[frame["variant"] == "baseline", "status"] == "ok").all()

    failures = result.failures_frame()
    assert set(failures["variant"]) == {"indicator_adjusted_past_only"}
    assert set(failures["error_type"]) == {"InsufficientIndicatorData"}
    assert len(failures) == early["origin"].nunique() * len(MODELS)

    summary = summarize_backtest(result, group_by=["variant"]).set_index("variant")
    assert summary.loc["indicator_adjusted_past_only", "n_failed"] == len(early)
    assert summary.loc["indicator_adjusted_past_only", "n_valid"] == len(later)
    assert summary.loc["baseline", "n_failed"] == 0


def test_partially_missing_weighted_indicator_fails_the_fold():
    indicators = indicator_frame()
    late_internet = ~((indicators["indicator_key"] == "internet") & (indicators["year"] < 2018))
    frame = indicator_backtest(indicators[late_internet]).to_frame()
    adjusted = frame[frame["variant"] == "indicator_adjusted_past_only"]

    assert (adjusted.loc[adjusted["origin"] < 2019, "status"] == "failed").all()
    assert adjusted.loc[adjusted["origin"] < 2019, "message"].str.contains("internet").all()
    assert (adjusted.loc[adjusted["origin"] >= 2019, "status"] == "ok").all()


def test_baseline_failure_marks_adjusted_variant_failed(monkeypatch):
    monkeypatch.setitem(BaselineModelFactory._models, "Always Fails", AlwaysFailsForecaster)
    frame = indicator_backtest(indicator_frame(), models=["Always Fails"]).to_frame()
    adjusted = frame[frame["variant"] == "indicator_adjusted_past_only"]
    assert (adjusted["status"] == "failed").all()
    assert adjusted["message"].str.contains("Baseline forecast failed").all()


@pytest.mark.parametrize("kwargs, match", [
    (dict(indicators=None), "non-empty indicators"),
    (dict(indicators=pd.DataFrame()), "non-empty indicators"),
    (dict(indicator_weights=None), "non-empty indicator_weights"),
    (dict(indicator_weights={}), "non-empty indicator_weights"),
    (dict(indicator_weights={"gdp": 1.0, "unknown": 1.0}), "not in the data"),
    (dict(indicator_weights={"gdp": float("nan")}), "finite number"),
    (dict(indicator_weight=float("inf")), "finite number"),
])
def test_invalid_indicator_configuration_is_rejected(kwargs, match):
    params = dict(indicators=indicator_frame(), indicator_weights=INDICATOR_WEIGHTS,
                  indicator_weight=INDICATOR_WEIGHT)
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        run_backtest(noisy_series(), models=MODELS, horizon=3, min_train_years=5,
                     use_indicators=True, **params)


@pytest.mark.parametrize("mutate, match", [
    (lambda df: df.assign(value=df["value"].where(df.index != 3)), "finite numbers"),
    (lambda df: pd.concat([df, df.iloc[[0]]], ignore_index=True), "one row per indicator and year"),
    (lambda df: df.drop(columns=["value"]), "missing columns"),
])
def test_invalid_indicator_data_is_rejected(mutate, match):
    with pytest.raises(ValueError, match=match):
        indicator_backtest(mutate(indicator_frame()))


def test_indicator_inputs_without_use_indicators_are_rejected():
    with pytest.raises(ValueError, match="use_indicators is False"):
        run_backtest(noisy_series(), indicators=indicator_frame())
    with pytest.raises(ValueError, match="use_indicators is False"):
        run_backtest(noisy_series(), indicator_weights=INDICATOR_WEIGHTS)


# --- Phase 3: historical (point-in-time) news evaluation ---
# "news_adjusted_historical" = NewsAdjustment with as_of = Y-12-31 23:59:59 and
# only headlines dated in [as_of - 90 days, as_of]. Retrieval/windowing is
# point-in-time; the analysed growth rates here are synthetic, so no LLM (and
# no LLM hindsight) is involved in these tests.

NEWS = "news_adjusted_historical"
NEWS_CATEGORIES = [
    {"name": "Demand", "ma_window_days": 30},
    {"name": "Supply", "ma_window_days": 60},
    {"name": "Neutral/Noise", "ma_window_days": 0},
]
NEWS_CONFIG = {
    "categories": NEWS_CATEGORIES,
    "adjustment_weights": {"news_weight": 0.7},
    "long_term_decay_rate": 0.6,
}
LEGACY_NEWS_CONFIG = {"adjustment_weights": {"news_weight": 0.7}, "news_half_life_days": 90}
DAYS_BEFORE_YEAR_END = (2, 10, 25, 40, 55, 80)


def headline(date, growth_rate, category="Demand", title=None, relevant=1):
    return {
        "title": title or f"{category} headline {pd.Timestamp(date)}",
        "date": pd.Timestamp(date),
        "category": category,
        "growth_rate": float(growth_rate),
        "reason": "synthetic",
        "relevant": relevant,
    }


def news_frame(start_year=2012, end_year=2023, seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    for year in range(start_year, end_year + 1):
        for days_before in DAYS_BEFORE_YEAR_END:
            date = pd.Timestamp(year=year, month=12, day=31, hour=12) - pd.Timedelta(days=days_before)
            for category in ("Demand", "Supply"):
                growth = rng.uniform(0.5, 5.0) * rng.choice([-1.0, 1.0])
                rows.append(headline(date, round(growth, 3), category,
                                     title=f"{category} {year} -{days_before}d"))
    return pd.DataFrame(rows)


def news_backtest(news, models=MODELS, horizon=3, config=NEWS_CONFIG, **kwargs):
    return run_backtest(
        noisy_series(), models=models, horizon=horizon, min_train_years=5,
        use_news=True, news=news, news_config=config, **kwargs,
    )


def with_headlines(news, *rows):
    return pd.concat([news, pd.DataFrame(list(rows))], ignore_index=True)


def test_news_as_of_is_end_of_origin_year():
    assert news_as_of(2018) == pd.Timestamp("2018-12-31 23:59:59")
    with pytest.raises(TypeError):
        news_as_of("2018")


def test_future_news_cannot_change_historical_fold():
    original = news_frame()
    altered = original.copy()
    future = altered["date"] > news_as_of(TARGET_ORIGIN)
    altered.loc[future, "growth_rate"] = altered.loc[future, "growth_rate"] * -10 + 50
    altered = with_headlines(altered, headline("2017-01-15", 30.0), headline("2017-02-01", -25.0, "Supply"))

    a = fold_rows(news_backtest(original).to_frame(), TARGET_ORIGIN, NEWS)
    b = fold_rows(news_backtest(altered).to_frame(), TARGET_ORIGIN, NEWS)
    baseline = fold_rows(news_backtest(original).to_frame(), TARGET_ORIGIN, "baseline")

    assert set(a["model"]) == set(MODELS)
    assert (a["status"] == "ok").all()
    pd.testing.assert_series_equal(a["predicted"], b["predicted"])
    assert not np.allclose(a["predicted"], baseline["predicted"])


def test_headline_one_second_after_as_of_is_ignored_but_at_as_of_counts():
    as_of = news_as_of(TARGET_ORIGIN)
    original = news_frame()
    reference = fold_rows(news_backtest(original).to_frame(), TARGET_ORIGIN, NEWS)

    late = with_headlines(original, headline(as_of + pd.Timedelta(seconds=1), 40.0))
    late_day = with_headlines(original, headline(as_of + pd.Timedelta(days=1), 40.0))
    on_time = with_headlines(original, headline(as_of, 40.0))

    for news in (late, late_day):
        result = fold_rows(news_backtest(news).to_frame(), TARGET_ORIGIN, NEWS)
        pd.testing.assert_series_equal(result["predicted"], reference["predicted"])
    included = fold_rows(news_backtest(on_time).to_frame(), TARGET_ORIGIN, NEWS)
    assert (np.abs(included["predicted"] - reference["predicted"]) > 1e-9).all()


def test_historical_news_change_changes_fold():
    original = news_frame()
    altered = original.copy()
    row = altered["title"] == f"Demand {TARGET_ORIGIN} -2d"
    assert row.sum() == 1
    altered.loc[row, "growth_rate"] = altered.loc[row, "growth_rate"] + 20

    a = fold_rows(news_backtest(original).to_frame(), TARGET_ORIGIN, NEWS)
    b = fold_rows(news_backtest(altered).to_frame(), TARGET_ORIGIN, NEWS)
    assert (np.abs(a["predicted"] - b["predicted"]) > 1e-9).all()


@pytest.mark.parametrize("fake_now", ["2099-06-30", None])
def test_historical_news_does_not_depend_on_current_date(monkeypatch, fake_now):
    news = news_frame()
    reference = news_backtest(news).to_frame()

    def patched_now():
        if fake_now is None:
            raise AssertionError("current time consulted during historical evaluation")
        return pd.Timestamp(fake_now)

    monkeypatch.setattr(news_adjustment_module, "_utc_now", patched_now)
    pd.testing.assert_frame_equal(news_backtest(news).to_frame(), reference)


@pytest.mark.parametrize("config", [NEWS_CONFIG, LEGACY_NEWS_CONFIG], ids=["categories", "no_categories"])
def test_historical_news_never_uses_clock_based_calibrator_average(monkeypatch, config):
    news = news_frame()
    reference = news_backtest(news, config=config).to_frame()

    def clock_dependent(*args, **kwargs):
        raise AssertionError("recency_weighted_avg_pct used during historical evaluation")

    def clock(*args, **kwargs):
        raise AssertionError("current time consulted during historical evaluation")

    monkeypatch.setattr(
        "src.llm.calibrators.impact_decay_calibrator.recency_weighted_avg_pct", clock_dependent
    )
    monkeypatch.setattr(news_adjustment_module, "_utc_now", clock)
    frame = news_backtest(news, config=config).to_frame()

    pd.testing.assert_frame_equal(frame, reference)
    assert (frame.loc[frame["variant"] == NEWS, "status"] == "ok").all()


def test_category_windows_are_measured_from_as_of_not_today(monkeypatch):
    as_of = news_as_of(TARGET_ORIGIN)
    base = [headline(as_of - pd.Timedelta(days=5), 2.0, "Demand"),
            headline(as_of - pd.Timedelta(days=5), 1.0, "Supply")]

    def score(rows, as_of_value=as_of):
        adjustment = NewsAdjustment(as_of=as_of_value)
        value = adjustment.calculate(pd.DataFrame(rows), {"categories": NEWS_CATEGORIES})
        return value, adjustment.category_breakdown

    base_value, _ = score(base)
    inside, _ = score(base + [headline(as_of - pd.Timedelta(days=20), 10.0, "Demand")])
    outside_demand, _ = score(base + [headline(as_of - pd.Timedelta(days=45), 10.0, "Demand")])
    inside_supply, _ = score(base + [headline(as_of - pd.Timedelta(days=45), 10.0, "Supply")])

    assert inside != pytest.approx(base_value)
    assert outside_demand == pytest.approx(base_value)
    assert inside_supply != pytest.approx(base_value)

    monkeypatch.setattr(news_adjustment_module, "_utc_now", lambda: pd.Timestamp("2026-10-06"))
    historical_value, historical = score(base)
    production_value, production = score(base, as_of_value=None)
    assert historical["Demand"]["status"] == "Active"
    assert historical_value == pytest.approx(base_value)
    assert production["Demand"]["status"] == "No data in window"
    assert production_value == 0.0


def test_news_window_excludes_headlines_older_than_lookback():
    as_of = news_as_of(TARGET_ORIGIN)
    base = [headline(as_of - pd.Timedelta(days=10), 2.0)]

    def score(rows):
        return NewsAdjustment(as_of=as_of).calculate(pd.DataFrame(rows), LEGACY_NEWS_CONFIG)

    base_value = score(base)
    assert score(base + [headline(as_of - pd.Timedelta(days=100), 25.0)]) == pytest.approx(base_value)
    assert score(base + [headline(as_of - pd.Timedelta(days=90, seconds=1), 25.0)]) == pytest.approx(base_value)
    assert score(base + [headline(as_of - pd.Timedelta(days=90), 25.0)]) != pytest.approx(base_value)
    assert score(base + [headline(as_of - pd.Timedelta(days=80), 25.0)]) != pytest.approx(base_value)

    news = news_frame()
    reference = news_backtest(news, config=LEGACY_NEWS_CONFIG).to_frame()
    old = with_headlines(news, headline(as_of - pd.Timedelta(days=100), 25.0))
    pd.testing.assert_frame_equal(news_backtest(old, config=LEGACY_NEWS_CONFIG).to_frame(), reference)


def test_production_news_adjustment_without_as_of_is_unchanged(monkeypatch):
    monkeypatch.setattr(news_adjustment_module, "_utc_now", lambda: pd.Timestamp("2026-10-06"))
    rows = pd.DataFrame([headline("2026-10-06", 4.0), headline("2026-07-08", 1.0),
                         headline("2025-01-01", 3.0)])
    value = NewsAdjustment().calculate(rows, LEGACY_NEWS_CONFIG)

    ages = np.array([0, 90, (pd.Timestamp("2026-10-06") - pd.Timestamp("2025-01-01")).days])
    weights = 0.5 ** (ages / 90)
    expected = np.average([4.0, 1.0, 3.0], weights=weights) / 100
    assert value == pytest.approx(expected)
    assert NewsAdjustment().filter_point_in_time(rows) is rows


def test_news_data_never_changes_baseline_or_benchmark_records():
    plain = run_backtest(noisy_series(), models=MODELS, horizon=3, min_train_years=5).to_frame()
    altered = news_frame()
    altered["growth_rate"] = altered["growth_rate"] * 7 - 3

    for news in (news_frame(), altered):
        frame = news_backtest(news).to_frame()
        pd.testing.assert_frame_equal(frame[frame["variant"] != NEWS].reset_index(drop=True), plain)
        naive = frame[frame["model"] == NAIVE_MODEL_NAME]
        assert set(naive["variant"]) == {"benchmark"}


def test_news_variant_is_reported_separately():
    result = news_backtest(news_frame())
    frame = result.to_frame()
    assert backtest_module.VARIANT_NEWS_ADJUSTED_HISTORICAL == NEWS
    assert set(frame["variant"]) == {"baseline", NEWS, "benchmark"}
    for model in MODELS:
        base = frame[(frame["model"] == model) & (frame["variant"] == "baseline")]
        adj = frame[(frame["model"] == model) & (frame["variant"] == NEWS)]
        assert list(base[["origin", "year"]].itertuples(index=False)) == \
            list(adj[["origin", "year"]].itertuples(index=False))

    summary = summarize_backtest(result)
    pairs = set(zip(summary["model"], summary["variant"]))
    for model in MODELS:
        assert {(model, "baseline"), (model, NEWS)} <= pairs
    assert (NAIVE_MODEL_NAME, NEWS) not in pairs
    assert summary[["mape", "rmse", "mae", "bias"]].notna().all().all()


def test_news_adjustment_keeps_existing_temporal_decay_across_horizons():
    news = news_frame()
    horizon = 4
    result = news_backtest(news, horizon=horizon)
    frame = result.to_frame()
    decay = TemporalDecay(long_term_decay_rate=NEWS_CONFIG["long_term_decay_rate"])
    news_weight = NEWS_CONFIG["adjustment_weights"]["news_weight"]

    for fold in result.folds:
        news_avg = NewsAdjustment(as_of=news_as_of(fold.origin)).calculate(news, NEWS_CONFIG)
        assert news_avg != 0
        expected = np.array([1 + news_weight * news_avg * decay.calculate_news_decay(i, horizon)
                             for i in range(horizon)])
        for model in MODELS:
            base = fold_rows(frame, fold.origin, "baseline")
            adj = fold_rows(frame, fold.origin, NEWS)
            base, adj = base[base["model"] == model], adj[adj["model"] == model]
            assert list(adj["horizon"]) == list(range(1, horizon + 1))
            np.testing.assert_allclose(adj["predicted"].to_numpy() / base["predicted"].to_numpy(),
                                       expected, rtol=1e-12)


def test_irrelevant_headlines_in_window_give_zero_impact_not_failure():
    news = news_frame().assign(relevant=0)
    frame = news_backtest(news).to_frame()
    base = fold_rows(frame, TARGET_ORIGIN, "baseline")
    adj = fold_rows(frame, TARGET_ORIGIN, NEWS)
    assert (adj["status"] == "ok").all()
    np.testing.assert_allclose(adj["predicted"], base["predicted"])


def test_folds_without_news_in_window_fail_explicitly():
    result = news_backtest(news_frame(start_year=TARGET_ORIGIN + 1))
    frame = result.to_frame()
    adjusted = frame[frame["variant"] == NEWS]
    early = adjusted[adjusted["origin"] <= TARGET_ORIGIN]
    later = adjusted[adjusted["origin"] > TARGET_ORIGIN]

    assert not early.empty and not later.empty
    assert (early["status"] == "failed").all() and early["predicted"].isna().all()
    assert early["message"].str.contains("No analysed headlines").all()
    assert (later["status"] == "ok").all()
    assert (frame.loc[frame["variant"] == "baseline", "status"] == "ok").all()

    failures = result.failures_frame()
    assert set(failures["error_type"]) == {"InsufficientNewsData"}
    summary = summarize_backtest(result, group_by=["variant"]).set_index("variant")
    assert summary.loc[NEWS, "n_failed"] == len(early)
    assert summary.loc[NEWS, "n_valid"] == len(later)


def test_news_processing_error_marks_only_that_fold_failed(monkeypatch):
    original_apply = NewsAdjustment.apply

    def flaky_apply(self, baseline_df, data, config):
        if self.as_of.year == 2018:
            raise RuntimeError("simulated news failure")
        return original_apply(self, baseline_df, data, config)

    monkeypatch.setattr(NewsAdjustment, "apply", flaky_apply)
    result = news_backtest(news_frame())
    frame = result.to_frame()
    adjusted = frame[frame["variant"] == NEWS]

    failed = adjusted[adjusted["origin"] == 2018]
    assert (failed["status"] == "failed").all()
    assert failed["message"].str.contains("simulated news failure").all()
    assert failed["error"].isna().all()
    assert (adjusted.loc[adjusted["origin"] != 2018, "status"] == "ok").all()
    assert (frame.loc[frame["variant"] == "baseline", "status"] == "ok").all()
    assert set(result.failures_frame()["error_type"]) == {"RuntimeError"}


def test_baseline_failure_marks_news_variant_failed(monkeypatch):
    monkeypatch.setitem(BaselineModelFactory._models, "Always Fails", AlwaysFailsForecaster)
    frame = news_backtest(news_frame(), models=["Always Fails"]).to_frame()
    adjusted = frame[frame["variant"] == NEWS]
    assert (adjusted["status"] == "failed").all()
    assert adjusted["message"].str.contains("Baseline forecast failed").all()


def test_news_and_indicator_variants_are_independent():
    news, indicators = news_frame(), indicator_frame()
    both = run_backtest(
        noisy_series(), models=MODELS, horizon=3, min_train_years=5,
        use_indicators=True, indicators=indicators, indicator_weights=INDICATOR_WEIGHTS,
        indicator_weight=INDICATOR_WEIGHT, use_news=True, news=news, news_config=NEWS_CONFIG,
    ).to_frame()
    news_only = news_backtest(news).to_frame()
    indicators_only = indicator_backtest(indicators).to_frame()

    def variant(frame, name):
        return frame[frame["variant"] == name].reset_index(drop=True)

    pd.testing.assert_frame_equal(variant(both, NEWS), variant(news_only, NEWS))
    pd.testing.assert_frame_equal(variant(both, "indicator_adjusted_past_only"),
                                  variant(indicators_only, "indicator_adjusted_past_only"))


@pytest.mark.parametrize("kwargs, error, match", [
    (dict(news=None), ValueError, "non-empty DataFrame"),
    (dict(news=pd.DataFrame()), ValueError, "non-empty DataFrame"),
    (dict(news_config=["not", "a", "mapping"]), TypeError, "mapping"),
    (dict(news_lookback_days=0), ValueError, "positive integer"),
    (dict(news=news_frame().drop(columns=["category"])), ValueError, "missing columns"),
    (dict(news=news_frame().assign(date="not a date")), ValueError, "unparseable"),
    (dict(news=news_frame().assign(growth_rate=np.nan)), ValueError, "finite numbers"),
])
def test_invalid_news_configuration_is_rejected(kwargs, error, match):
    params = dict(news=news_frame(), news_config=NEWS_CONFIG)
    params.update(kwargs)
    with pytest.raises(error, match=match):
        run_backtest(noisy_series(), models=MODELS, horizon=3, min_train_years=5,
                     use_news=True, **params)


def test_news_inputs_without_use_news_are_rejected():
    with pytest.raises(ValueError, match="use_news is False"):
        run_backtest(noisy_series(), news=news_frame())
    with pytest.raises(ValueError, match="use_news is False"):
        run_backtest(noisy_series(), news_config=NEWS_CONFIG)


# --- headline-analysis cache (no LLM) ---

def fake_analyzer(calls, growth_rate=2.5):
    def analyze(headline_dict):
        calls.append(headline_dict["title"])
        return {"category": "Demand", "growth_rate": growth_rate, "reason": "fake", "relevant": 1}
    return analyze


def raw_headlines():
    return pd.DataFrame({
        "title": ["Chip demand surges", "Supply chain eases", "Chip demand surges"],
        "date": pd.to_datetime(["2018-12-01", "2018-12-02", "2018-12-01"], utc=True),
        "url": ["u1", "u2", "u1"],
    })


def test_same_headline_and_configuration_is_analysed_once():
    calls, cache = [], HeadlineAnalysisCache()
    fingerprint = analysis_fingerprint("prompt v1", model="model-a")

    first = analyze_headlines_cached(raw_headlines(), fake_analyzer(calls), cache, fingerprint)
    second = analyze_headlines_cached(raw_headlines(), fake_analyzer(calls), cache, fingerprint)

    assert calls == ["Chip demand surges", "Supply chain eases"]
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["title", "date", "category", "growth_rate", "reason", "relevant"]
    assert (first["date"] == raw_headlines()["date"]).all()


@pytest.mark.parametrize("changed", [
    dict(system_prompt="prompt v2", model="model-a"),
    dict(system_prompt="prompt v1", model="model-b"),
    dict(system_prompt="prompt v1", model="model-a", prompt_version="2"),
])
def test_changing_analysis_configuration_invalidates_cache(changed):
    calls, cache = [], HeadlineAnalysisCache()
    analyze_headlines_cached(raw_headlines(), fake_analyzer(calls), cache,
                             analysis_fingerprint("prompt v1", model="model-a"))
    calls.clear()
    out = analyze_headlines_cached(raw_headlines(), fake_analyzer(calls, growth_rate=-1.0), cache,
                                   analysis_fingerprint(**changed))
    assert calls == ["Chip demand surges", "Supply chain eases"]
    assert (out["growth_rate"] == -1.0).all()


def test_file_cache_round_trip_avoids_reanalysis(tmp_path):
    path = str(tmp_path / ".forecast_cache" / "headlines.json")
    fingerprint = analysis_fingerprint("prompt v1")
    calls = []
    cache = HeadlineAnalysisCache(path)
    expected = analyze_headlines_cached(raw_headlines(), fake_analyzer(calls), cache, fingerprint)
    cache.save()

    def must_not_be_called(_):
        raise AssertionError("cached headline re-analysed")

    reloaded = HeadlineAnalysisCache(path)
    got = analyze_headlines_cached(raw_headlines(), must_not_be_called, reloaded, fingerprint)
    pd.testing.assert_frame_equal(got, expected)


def test_failed_analysis_is_not_cached_or_scored_as_zero():
    cache = HeadlineAnalysisCache()
    fingerprint = analysis_fingerprint("prompt v1")

    def failing(headline_dict):
        return {"category": "Neutral/Noise", "growth_rate": 0.0,
                "reason": ANALYST_FAILURE_REASON, "relevant": 0}

    with pytest.raises(HeadlineAnalysisError, match="LLM analysis failed"):
        analyze_headlines_cached(raw_headlines(), failing, cache, fingerprint)
    assert len(cache) == 0

    calls = []
    analyze_headlines_cached(raw_headlines(), fake_analyzer(calls), cache, fingerprint)
    assert calls == ["Chip demand surges", "Supply chain eases"]


# --- historical GDELT fetching (no network) ---

def test_fetch_headlines_for_origins_uses_each_origin_as_of():
    seen = []

    def fake_fetch(days_back, end_date):
        seen.append((days_back, end_date))
        frame = pd.DataFrame({"title": [f"t{end_date.year}", "shared"],
                              "date": [end_date, end_date], "url": [f"u{end_date.year}", "shared"]})
        return frame, True

    out = fetch_headlines_for_origins([2017, 2016, 2017], fake_fetch)
    assert seen == [(90, news_as_of(2016)), (90, news_as_of(2017))]
    assert sorted(out["url"]) == ["shared", "u2016", "u2017"]


def _gdelt_module():
    pytest.importorskip("gdeltdoc")
    from src.news import gdelt
    return gdelt


def test_gdelt_historical_fetch_is_bounded_by_end_date(monkeypatch):
    gdelt = _gdelt_module()
    captured = []
    end = news_as_of(2018)

    def fake_batch(keywords, start_date, end_date, country_fips, max_records, batch_index):
        captured.append((start_date, end_date, country_fips))
        dates = ["2018-10-01 12:00:00", "2018-10-02 23:59:59", "2018-12-31 23:59:59",
                 "2019-01-01 00:00:00", "2019-03-01 00:00:00"]
        return pd.DataFrame({
            "title": [f"t{i}" for i in range(len(dates))],
            "date": pd.to_datetime(dates, utc=True),
            "url": [f"u{i}" for i in range(len(dates))],
        })

    monkeypatch.setattr(gdelt, "_search_articles_batch", fake_batch)
    df, ok = gdelt.fetch_max_gdelt_articles(["AI"], days_back=90, end_date=end)

    from datetime import datetime
    assert captured == [(datetime(2018, 10, 2, 23, 59, 59), datetime(2018, 12, 31, 23, 59, 59), None)]
    assert ok and sorted(df["url"]) == ["u1", "u2"]

    captured.clear()
    country_df, _ = gdelt.fetch_country_headlines("USA", ["AI"], end_date=end)
    assert captured[0][:2] == (datetime(2018, 10, 2, 23, 59, 59), datetime(2018, 12, 31, 23, 59, 59))
    assert sorted(country_df["url"]) == ["u1", "u2"]

    with pytest.raises(ValueError):
        gdelt.fetch_max_gdelt_articles(["AI"], end_date="not a date")


def test_gdelt_fetch_without_end_date_keeps_production_window(monkeypatch):
    gdelt = _gdelt_module()
    captured = []

    def fake_batch(keywords, start_date, end_date, country_fips, max_records, batch_index):
        captured.append((start_date, end_date))
        return pd.DataFrame({"title": ["a"], "date": pd.to_datetime(["2099-01-01"], utc=True), "url": ["u"]})

    monkeypatch.setattr(gdelt, "_search_articles_batch", fake_batch)
    df, _ = gdelt.fetch_max_gdelt_articles(["AI"], days_back=90)

    start, end = captured[0]
    assert isinstance(start, str) and isinstance(end, str)
    today = pd.Timestamp.now(tz="UTC")
    assert end in {today.strftime("%Y-%m-%d"), (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d")}
    assert list(df["url"]) == ["u"]
