"""
Tests for the comparative experiment, comparison and export layers (Phase 4).

All data here is deterministic synthetic fixture data.
"""
import csv
import json
import math
import os
import socket
import sys

import numpy as np
import pandas as pd
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.forecasting.base.models import BaseForecaster
from src.forecasting.evaluation import (
    NAIVE_MODEL_NAME,
    adjustment_effects,
    benchmark_improvements,
    build_comparison_report,
    experiment_to_dict,
    export_csv,
    export_json,
    rank_models,
    run_comparative_experiment,
    summarize_experiment,
)
from src.forecasting.evaluation.comparison import describe_change, improvement_pct
from src.forecasting.models.baseline_factory import BaselineModelFactory

BASELINE_MODELS = {"3-yr CAGR", "Damped ETS", "Logistic Growth"}
INDICATOR = "indicator_adjusted_past_only"
NEWS = "news_adjusted_historical"
NEWS_CONFIG = {
    "categories": [{"name": "Demand", "ma_window_days": 30}, {"name": "Supply", "ma_window_days": 60}],
    "adjustment_weights": {"news_weight": 0.7},
    "long_term_decay_rate": 0.6,
}
INDICATOR_WEIGHTS = {"gdp": 1.0}
TIMESTAMP = "2026-01-01T00:00:00+00:00"


def target_series(n_years=14, seed=0):
    rng = np.random.default_rng(seed)
    years = np.arange(2010, 2010 + n_years)
    values = 100 * 1.08 ** np.arange(n_years) * (1 + rng.normal(0, 0.03, n_years))
    return pd.DataFrame({"year": years, "value": values})


def indicator_data():
    years = np.arange(2008, 2024)
    rng = np.random.default_rng(1)
    return pd.DataFrame({"year": years, "indicator_key": "gdp",
                         "value": 100 * 1.03 ** np.arange(len(years)) * (1 + rng.normal(0, 0.01, len(years)))})


def news_data():
    rng = np.random.default_rng(2)
    rows = []
    for year in range(2012, 2024):
        for days_before in (3, 20, 45):
            for category in ("Demand", "Supply"):
                rows.append({
                    "title": f"{category} {year} -{days_before}d",
                    "date": pd.Timestamp(year=year, month=12, day=31, hour=12) - pd.Timedelta(days=days_before),
                    "category": category,
                    "growth_rate": round(float(rng.uniform(0.5, 4) * rng.choice([-1, 1])), 3),
                    "reason": "synthetic",
                    "relevant": 1,
                })
    return pd.DataFrame(rows)


def experiment(**kwargs):
    params = dict(dataset_name="fixture-series", is_synthetic=True, min_train_years=5,
                  run_timestamp=TIMESTAMP)
    params.update(kwargs)
    series = params.pop("series", None)
    return run_comparative_experiment(target_series() if series is None else series, **params)


def full_experiment(**kwargs):
    return experiment(include_indicators=True, indicators=indicator_data(),
                      indicator_weights=INDICATOR_WEIGHTS, include_news=True,
                      news=news_data(), news_config=NEWS_CONFIG, **kwargs)


def record(model, variant, origin, horizon, actual, predicted, status="ok"):
    return {
        "origin": origin, "model": model, "variant": variant, "year": origin + horizon,
        "horizon": horizon, "actual": float(actual), "predicted": float(predicted),
        "error": float(predicted - actual) if status == "ok" else float("nan"),
        "n_train": 5, "status": status, "message": "",
    }


# --- running the experiment ---

def test_experiment_runs_all_baseline_models_and_naive_benchmark():
    result = experiment()
    frame = result.records_frame()
    assert set(frame["model"]) == BASELINE_MODELS | {NAIVE_MODEL_NAME}
    assert set(result.config.models) == BASELINE_MODELS | {NAIVE_MODEL_NAME}
    assert set(frame.loc[frame["model"] == NAIVE_MODEL_NAME, "variant"]) == {"benchmark"}
    assert set(frame.loc[frame["model"] != NAIVE_MODEL_NAME, "variant"]) == {"baseline"}


def test_naive_benchmark_is_always_included():
    frame = experiment(models=["cagr"]).records_frame()
    assert set(frame["model"]) == {"3-yr CAGR", NAIVE_MODEL_NAME}


def test_indicator_and_news_variants_only_when_requested():
    assert set(experiment().records_frame()["variant"]) == {"baseline", "benchmark"}
    with_indicators = experiment(include_indicators=True, indicators=indicator_data(),
                                 indicator_weights=INDICATOR_WEIGHTS)
    assert set(with_indicators.records_frame()["variant"]) == {"baseline", "benchmark", INDICATOR}
    with_news = experiment(include_news=True, news=news_data(), news_config=NEWS_CONFIG)
    assert set(with_news.records_frame()["variant"]) == {"baseline", "benchmark", NEWS}
    assert set(full_experiment().config.variants) == {"baseline", "benchmark", INDICATOR, NEWS}

    with pytest.raises(ValueError):
        experiment(include_news=True)
    with pytest.raises(ValueError):
        experiment(include_indicators=True)
    with pytest.raises(ValueError):
        experiment(news=news_data())


def test_all_methods_share_the_same_folds():
    result = full_experiment()
    frame = result.records_frame()
    keys = {
        (model, variant): sorted(map(tuple, group[["origin", "year", "horizon", "n_train"]].values.tolist()))
        for (model, variant), group in frame.groupby(["model", "variant"])
    }
    assert len(set(map(tuple, keys.values()))) == 1
    summaries = summarize_experiment(result)
    assert {s.attempted_folds for s in summaries} == {result.metadata["n_folds"]}
    actuals = frame.groupby(["origin", "year"])["actual"].nunique()
    assert (actuals == 1).all()


def test_naive_benchmark_is_unaffected_by_indicators_and_news():
    def naive(result):
        frame = result.records_frame()
        return frame[frame["model"] == NAIVE_MODEL_NAME].reset_index(drop=True)
    pd.testing.assert_frame_equal(naive(experiment()), naive(full_experiment()))


def test_training_never_includes_target_values_after_origin():
    series = target_series()
    reference = experiment(series=series).records_frame()
    altered = series.copy()
    altered.loc[altered["year"] > 2016, "value"] *= 3
    changed = experiment(series=altered).records_frame()
    mask = reference["origin"] == 2016
    pd.testing.assert_series_equal(reference.loc[mask, "predicted"], changed.loc[mask, "predicted"])


def test_historical_news_remains_point_in_time():
    news = news_data()
    altered = news.copy()
    future = altered["date"] > pd.Timestamp("2016-12-31 23:59:59")
    altered.loc[future, "growth_rate"] = 40.0
    a = full_experiment().records_frame()
    b = experiment(include_indicators=True, indicators=indicator_data(), indicator_weights=INDICATOR_WEIGHTS,
                   include_news=True, news=altered, news_config=NEWS_CONFIG).records_frame()
    mask = (a["origin"] == 2016) & (a["variant"] == NEWS)
    assert (a.loc[mask, "status"] == "ok").all()
    pd.testing.assert_series_equal(a.loc[mask, "predicted"], b.loc[mask, "predicted"])


def test_indicators_remain_point_in_time():
    altered = indicator_data()
    altered.loc[altered["year"] > 2016, "value"] *= 5
    a = full_experiment().records_frame()
    b = experiment(include_indicators=True, indicators=altered, indicator_weights=INDICATOR_WEIGHTS,
                   include_news=True, news=news_data(), news_config=NEWS_CONFIG).records_frame()
    mask = (a["origin"] == 2016) & (a["variant"] == INDICATOR)
    assert (a.loc[mask, "status"] == "ok").all()
    pd.testing.assert_series_equal(a.loc[mask, "predicted"], b.loc[mask, "predicted"])


def test_no_live_network_or_llm_calls(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network/LLM access attempted during experiment")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr("src.llm.providers.provider_factory.call_llm", blocked)
    result = full_experiment()
    build_comparison_report(result)
    experiment_to_dict(result)


def test_experiment_is_deterministic():
    assert experiment_to_dict(full_experiment()) == experiment_to_dict(full_experiment())


# --- horizons ---

def test_horizon_results_are_separated():
    result = experiment()
    frame = result.records_frame()
    assert result.evaluated_horizons == [1, 2, 3]
    assert ((frame["year"] - frame["origin"]) == frame["horizon"]).all()
    summaries = summarize_experiment(result)
    horizons = {s.horizon for s in summaries}
    assert horizons == {1, 2, 3, None}
    for s in summaries:
        expected = result.metadata["n_folds"] * (3 if s.horizon is None else 1)
        assert s.n_forecasts == expected


def test_unsupported_horizons_are_reported_not_fabricated():
    result = experiment(series=target_series(n_years=8), min_train_years=6, horizon=3)
    assert result.metadata["requested_horizon"] == 3
    assert result.evaluated_horizons == [1, 2]
    assert set(result.records_frame()["horizon"]) == {1, 2}
    assert any("Requested horizon 3" in note for note in result.metadata["limitations"])
    with pytest.raises(ValueError):
        experiment(series=target_series(n_years=6), min_train_years=6)


# --- coverage and failures ---

class FailsOnShortHistory(BaseForecaster):
    def __init__(self):
        super().__init__("Fails Short")

    def fit(self, historical_data):
        if len(historical_data) < 7:
            raise ValueError("needs 7 years")
        self.last = float(historical_data.iloc[-1]["value"])
        self._is_fitted = True

    def forecast(self, forecast_years):
        return [self.last * 1.05 ** (i + 1) for i in range(len(forecast_years))]


def test_coverage_and_failed_folds_are_visible(monkeypatch):
    monkeypatch.setitem(BaselineModelFactory._models, "Fails Short", FailsOnShortHistory)
    result = experiment(models=["Fails Short"])
    summaries = [s for s in summarize_experiment(result) if s.model == "Fails Short"]

    # Origins 2014..2020; training windows of 5 and 6 years (2014, 2015) fail.
    for s in summaries:
        assert s.attempted_folds == 7
        assert s.successful_folds == 5
        assert s.failed_folds == 2
        assert s.coverage == pytest.approx(5 / 7 * 100)
    overall = next(s for s in summaries if s.horizon is None)
    assert overall.n_failed == 6 and overall.n_valid == 15
    failures = result.failures_frame()
    assert sorted(failures.loc[failures["model"] == "Fails Short", "origin"]) == [2014, 2015]

    report = build_comparison_report(result, min_coverage=80.0)
    ranking = report.rankings["mape"]
    fails = next(e for e in ranking if e.model == "Fails Short")
    assert fails.rank is None and not fails.eligible
    assert ranking[-1].model == "Fails Short"
    assert any("2 of 7 folds failed" in f for f in report.findings)


# --- aggregation ---

def test_pooled_and_fold_mean_metrics_are_distinct_and_labelled():
    frame = pd.DataFrame([
        record("M", "baseline", 2014, 1, 100, 101), record("M", "baseline", 2014, 2, 100, 101),
        record("M", "baseline", 2015, 1, 100, 105), record("M", "baseline", 2015, 2, 100, 105),
    ])
    pooled = next(s for s in summarize_experiment(frame, "pooled") if s.horizon is None)
    fold_mean = next(s for s in summarize_experiment(frame, "fold_mean") if s.horizon is None)

    assert pooled.aggregation == "pooled" and fold_mean.aggregation == "fold_mean"
    assert pooled.rmse == pytest.approx(math.sqrt(13))
    assert fold_mean.rmse == pytest.approx(3.0)
    assert pooled.mae == pytest.approx(3.0) and fold_mean.mae == pytest.approx(3.0)
    assert pooled.error_std == pytest.approx(np.std([1, 1, 5, 5], ddof=1))
    assert fold_mean.error_std == pytest.approx(np.std([1, 5], ddof=1))
    assert pooled.error_median == pytest.approx(3.0)
    with pytest.raises(ValueError):
        summarize_experiment(frame, "mixed")


# --- ranking ---

def test_rank_models_orders_by_metric_with_ties_and_coverage():
    frame = pd.DataFrame([
        record("A", "baseline", 2014, 1, 100, 110), record("A", "baseline", 2015, 1, 100, 110),
        record("B", "baseline", 2014, 1, 100, 95), record("B", "baseline", 2015, 1, 100, 95),
        record("C", "baseline", 2014, 1, 100, 105), record("C", "baseline", 2015, 1, 100, 95),
        record("D", "baseline", 2014, 1, 100, 100), record("D", "baseline", 2015, 1, 100, 0, "failed"),
    ])
    summaries = summarize_experiment(frame)

    by_mape = rank_models(summaries, "mape", horizon=1, min_coverage=100.0)
    assert [(e.model, e.rank) for e in by_mape] == [("B", 1), ("C", 1), ("A", 3), ("D", None)]
    assert not by_mape[-1].eligible and by_mape[-1].coverage == pytest.approx(50.0)

    by_bias = rank_models(summaries, "bias", horizon=1, min_coverage=100.0)
    assert [(e.model, e.value) for e in by_bias[:3]] == [("C", 0.0), ("B", 5.0), ("A", 10.0)]
    assert by_bias[0].metric == "abs_bias"

    with_low_coverage = rank_models(summaries, "mae", horizon=1)
    assert with_low_coverage[0].model == "D" and with_low_coverage[0].value == 0.0
    with pytest.raises(ValueError):
        rank_models(summaries, "r2")


# --- benchmark-relative improvement and adjustment effects ---

def comparison_frame(model_pred=95.0, naive_pred=90.0, adjusted_pred=None):
    rows = []
    for origin in (2014, 2015):
        rows.append(record(NAIVE_MODEL_NAME, "benchmark", origin, 1, 100, naive_pred))
        rows.append(record("M", "baseline", origin, 1, 100, model_pred))
        if adjusted_pred is not None:
            rows.append(record("M", NEWS, origin, 1, 100, adjusted_pred))
    return pd.DataFrame(rows)


def test_benchmark_relative_improvement_is_correct():
    better = next(i for i in benchmark_improvements(comparison_frame(95, 90)) if i.horizon == 1)
    assert better.n_paired == 2
    assert better.mape_improvement_pct == pytest.approx(50.0)
    assert better.rmse_improvement_pct == pytest.approx(50.0)
    assert better.mae_improvement_pct == pytest.approx(50.0)
    assert better.wins == 2 and better.losses == 0
    assert "reduced MAPE by 50.0%" in better.summary

    worse = next(i for i in benchmark_improvements(comparison_frame(80, 90)) if i.horizon == 1)
    assert worse.mape_improvement_pct == pytest.approx(-100.0)
    assert worse.losses == 2
    assert "increased MAPE by 100.0%" in worse.summary
    assert "no improvement" in worse.summary


def test_zero_benchmark_error_does_not_divide_by_zero():
    result = next(i for i in benchmark_improvements(comparison_frame(95, 100)) if i.horizon == 1)
    assert result.reference_mape == 0.0
    assert result.mape_improvement_pct is None
    assert result.rmse_improvement_pct is None
    assert "undefined" in result.summary
    assert improvement_pct(0.0, 0.0) is None
    assert improvement_pct(None, 1.0) is None


def test_improvements_are_paired_on_common_valid_forecasts():
    frame = comparison_frame(95, 90)
    frame.loc[(frame["model"] == "M") & (frame["origin"] == 2015), ["status", "predicted"]] = ["failed", np.nan]
    result = next(i for i in benchmark_improvements(frame) if i.horizon == 1)
    assert result.n_paired == 1


def test_adjustment_effect_reports_negative_results_honestly():
    worse = adjustment_effects(comparison_frame(95, 90, adjusted_pred=90), NEWS)
    worse = next(i for i in worse if i.horizon == 1)
    assert worse.reference_variant == "baseline" and worse.reference_model == "M"
    assert worse.mape_improvement_pct == pytest.approx(-100.0)
    assert "News adjustment" in worse.summary and "increased MAPE" in worse.summary

    better = next(i for i in adjustment_effects(comparison_frame(95, 90, adjusted_pred=98), NEWS)
                  if i.horizon == 1)
    assert better.mape_improvement_pct == pytest.approx(60.0)
    assert "reduced MAPE by 60.0%" in better.summary
    assert adjustment_effects(comparison_frame(), NEWS) == []
    with pytest.raises(ValueError):
        adjustment_effects(comparison_frame(), "baseline")


def test_describe_change_wording():
    assert describe_change("News adjustment", "baseline", "mape", 4.2) == \
        "News adjustment reduced MAPE by 4.2% relative to baseline."
    assert describe_change("News adjustment", "baseline", "mape", -1.7) == (
        "News adjustment increased MAPE by 1.7% relative to baseline, "
        "indicating no improvement under this evaluation setup.")


def test_comparison_report_answers_research_questions():
    result = full_experiment()
    report = build_comparison_report(result)
    assert report.data_label == "DEMO/SYNTHETIC"
    assert report.findings[0] == "All findings below are from DEMO/SYNTHETIC data."
    assert set(report.best_by_metric) == {"mape", "rmse", "mae", "abs_bias"}
    assert all(entry is not None for entry in report.best_by_metric.values())
    assert set(report.best_by_horizon) == {1, 2, 3}
    assert {i.model for i in report.indicator_effects} == BASELINE_MODELS
    assert {i.model for i in report.news_effects} == BASELINE_MODELS
    assert all(i.reference_variant == "baseline" for i in report.news_effects)
    assert report.best_relative_to_benchmark is not None
    assert any("hindsight" in note for note in report.limitations)
    assert any("GDELT" in note for note in report.limitations)
    json.dumps(report.to_dict())


# --- configuration and export ---

def test_configuration_is_preserved():
    result = full_experiment(horizon=2, step=1, notes="fixture run")
    config = result.config.to_dict()
    assert config["dataset_name"] == "fixture-series"
    assert config["is_synthetic"] is True
    assert config["horizon"] == 2 and config["min_train_years"] == 5 and config["step"] == 1
    assert config["include_indicators"] and config["indicator_weights"] == INDICATOR_WEIGHTS
    assert config["include_news"] and config["news_lookback_days"] == 90
    assert config["news_config"] == NEWS_CONFIG
    assert config["notes"] == "fixture run"
    assert result.metadata["run_timestamp"] == TIMESTAMP
    assert result.metadata["input_fingerprints"]["news"] is not None
    assert result.metadata["data_label"] == "DEMO/SYNTHETIC"


def test_origin_subset_and_validation():
    result = experiment(origins=[2016, 2018])
    assert result.metadata["origins"] == [2016, 2018]
    assert set(result.records_frame()["origin"]) == {2016, 2018}
    assert result.config.origins == (2016, 2018)
    with pytest.raises(ValueError, match="not walk-forward fold origins"):
        experiment(origins=[2030])
    with pytest.raises(TypeError):
        experiment(is_synthetic="yes")
    with pytest.raises(ValueError):
        experiment(dataset_name=" ")


def test_real_data_is_not_labelled_synthetic():
    result = experiment(is_synthetic=False, dataset_name="client-series")
    assert result.metadata["data_label"] == "REAL"
    assert not any("SYNTHETIC" in note for note in result.metadata["limitations"])


def test_export_csv_tables(tmp_path):
    result = full_experiment()
    summary_path = tmp_path / "summaries.csv"
    export_csv(result, str(summary_path))
    with open(summary_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {"dataset_name", "data_label", "model", "variant", "horizon", "aggregation", "mape",
            "rmse", "mae", "bias", "successful_folds", "failed_folds", "coverage"} <= set(rows[0])
    assert {r["aggregation"] for r in rows} == {"pooled", "fold_mean"}
    assert {r["horizon"] for r in rows} == {"1", "2", "3", "all"}
    assert {r["data_label"] for r in rows} == {"DEMO/SYNTHETIC"}

    records_path = tmp_path / "records.csv"
    export_csv(result, str(records_path), table="records")
    with open(records_path, newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    assert len(records) == len(result.records_frame())
    assert {"origin", "year", "horizon", "status"} <= set(records[0])

    improvements_path = tmp_path / "improvements.csv"
    export_csv(result, str(improvements_path), table="improvements")
    with open(improvements_path, newline="", encoding="utf-8") as f:
        comparisons = {r["comparison"] for r in csv.DictReader(f)}
    assert comparisons == {"vs_benchmark", "adjustment_vs_baseline"}

    with pytest.raises(ValueError):
        export_csv(result, str(tmp_path / "x.csv"), table="everything")


def test_export_json_round_trip(tmp_path):
    result = full_experiment()
    path = tmp_path / "experiment.json"
    export_json(result, str(path))

    def reject(constant):
        raise AssertionError(f"non-standard JSON constant {constant}")

    with open(path, encoding="utf-8") as f:
        data = json.loads(f.read(), parse_constant=reject)
    assert data["config"] == json.loads(json.dumps(result.config.to_dict()))
    assert data["metadata"]["data_label"] == "DEMO/SYNTHETIC"
    assert data["metadata"]["evaluated_horizons"] == [1, 2, 3]
    assert len(data["records"]) == len(result.records_frame())
    assert {s["aggregation"] for s in data["summaries"]} == {"pooled", "fold_mean"}
    assert "title" not in json.dumps(data["records"])
