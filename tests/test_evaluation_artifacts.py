"""Tests for loading stored evaluation artifacts (no Streamlit, no network)."""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.forecasting.evaluation.artifacts import (  # noqa: E402
    BASELINE_FILES,
    BASELINE_SUBDIR,
    DEFAULT_INDICATOR_DIR,
    DEFAULT_RESULTS_DIR,
    INDICATOR_FILES,
    INDICATOR_SUBDIR,
    best_models,
    best_models_by_horizon,
    bias_comparison,
    download_files,
    fixed_nudge_comparison,
    indicator_comparison,
    load_evaluation_artifacts,
    metric_by_horizon,
    metric_table,
    period_errors,
)

MODELS = {"Naive (last value)", "3-yr CAGR", "Damped ETS", "Logistic Growth"}
TREND_MODELS = {"3-yr CAGR", "Damped ETS", "Logistic Growth"}

stored = pytest.mark.skipif(
    not (DEFAULT_RESULTS_DIR / BASELINE_SUBDIR / "summaries.csv").is_file(),
    reason="stored evaluation artifacts not present",
)


@pytest.fixture(scope="module")
def artifacts():
    return load_evaluation_artifacts()


@pytest.fixture
def copied_results(tmp_path):
    """A writable copy of the stored results, for missing-file tests."""
    target = tmp_path / "results"
    shutil.copytree(DEFAULT_RESULTS_DIR, target)
    return target


# ---------------------------------------------------------------- stored results

@stored
def test_stored_artifacts_load_completely(artifacts):
    assert artifacts.baseline_available
    assert artifacts.indicator_available
    assert artifacts.missing == []
    assert artifacts.indicator_provenance["series_code"] == "DGDSRC"


@stored
def test_metadata_describes_the_census_experiment(artifacts):
    meta = artifacts.baseline["metadata"]
    assert meta["is_synthetic"] is False
    assert meta["data_label"] == "REAL / CURRENT-VINTAGE"
    assert (meta["first_year"], meta["last_year"], meta["n_observations"]) == (2000, 2025, 26)
    assert meta["origins"] == list(range(2004, 2023))
    assert meta["n_folds"] == 19
    assert meta["evaluated_horizons"] == [1, 2, 3]
    assert set(meta["models"]) == MODELS
    assert "not necessarily identical" in meta["current_vintage_status"]


@stored
def test_metric_table_columns_and_values_match_summaries(artifacts):
    summaries = artifacts.baseline["summaries"]
    table = metric_table(summaries)
    assert list(table.columns) == ["Model", "Horizon", "MAPE", "RMSE", "MAE", "Bias", "Abs bias",
                                   "Coverage", "Failed folds"]
    assert set(table["Model"]) == MODELS
    assert list(dict.fromkeys(table["Horizon"])) == ["1Y", "2Y", "3Y", "all"]
    row = table[(table["Model"] == "Damped ETS") & (table["Horizon"] == "all")].iloc[0]
    source = summaries[(summaries["model"] == "Damped ETS") & (summaries["horizon"] == "all")
                       & (summaries["aggregation"] == "pooled")].iloc[0]
    assert row["MAPE"] == source["mape"] and row["RMSE"] == source["rmse"] and row["Bias"] == source["bias"]
    assert (table["Coverage"] == 100.0).all()


@stored
def test_fold_mean_table_differs_from_pooled_only_where_expected(artifacts):
    summaries = artifacts.baseline["summaries"]
    pooled = metric_table(summaries, "pooled").set_index(["Model", "Horizon"])
    fold_mean = metric_table(summaries, "fold_mean").set_index(["Model", "Horizon"])
    assert fold_mean["MAPE"].reindex(pooled.index).to_numpy() == pytest.approx(pooled["MAPE"].to_numpy())
    assert not pooled.loc[(slice(None), "all"), "RMSE"].equals(fold_mean.loc[(slice(None), "all"), "RMSE"])


@stored
def test_metric_by_horizon_pivot(artifacts):
    pivot = metric_by_horizon(artifacts.baseline["summaries"], "mape")
    assert list(pivot.index) == ["1Y", "2Y", "3Y"]
    assert set(pivot.columns) == MODELS
    assert (pivot.diff().dropna() > 0).all().all()


@stored
def test_best_models_come_from_saved_report(artifacts):
    best = best_models(artifacts.baseline["report_pooled"])
    assert best["mape"]["model"] == "Damped ETS"
    assert best["rmse"]["model"] == "Logistic Growth"
    assert best["mae"]["model"] == "Logistic Growth"
    table = metric_table(artifacts.baseline["summaries"])
    overall = table[table["Horizon"] == "all"].set_index("Model")
    for metric, column in (("mape", "MAPE"), ("rmse", "RMSE"), ("mae", "MAE")):
        assert overall[column].idxmin() == best[metric]["model"]
        assert best[metric]["value"] == pytest.approx(overall[column].min())
    assert {e["model"] for e in best_models_by_horizon(artifacts.baseline["report_pooled"]).values()} \
        == {"Damped ETS"}


@stored
def test_indicator_comparison_deltas(artifacts):
    effects = artifacts.indicator["effects"]
    for metric in ("mape", "rmse", "mae"):
        table = indicator_comparison(effects, metric)
        label = metric.upper()
        assert set(table["Model"]) == TREND_MODELS
        assert (table[f"Change {label}"] == table[f"Adjusted {label}"] - table[f"Baseline {label}"]).all()
    mape = indicator_comparison(effects, "mape").set_index("Model")
    assert mape.loc["3-yr CAGR", "Change MAPE"] > 0
    assert mape.loc["Damped ETS", "Change MAPE"] < 0
    one_year = indicator_comparison(effects, "mape", "1")
    assert len(one_year) == 3


@stored
def test_indicator_values_match_summaries(artifacts):
    effects = indicator_comparison(artifacts.indicator["effects"], "mape").set_index("Model")
    summaries = artifacts.indicator["summaries"]
    pooled = summaries[(summaries["aggregation"] == "pooled") & (summaries["horizon"] == "all")]
    for model in TREND_MODELS:
        rows = pooled[pooled["model"] == model].set_index("variant")["mape"]
        assert effects.loc[model, "Baseline MAPE"] == pytest.approx(rows["baseline"])
        assert effects.loc[model, "Adjusted MAPE"] == pytest.approx(rows["indicator_adjusted_past_only"])


@stored
def test_bias_and_fixed_nudge_tables(artifacts):
    effects = artifacts.indicator["effects"]
    bias = bias_comparison(effects).set_index("Model")
    assert (bias["Bias change"] > 0).all()
    assert (bias["Wins"] == bias["Wins on baseline under-forecasts"]).all()
    nudge = fixed_nudge_comparison(effects).set_index("Model")
    assert (nudge["Fixed-nudge MAE"] <= nudge["Indicator MAE"]).all()


@stored
def test_period_errors(artifacts):
    table = period_errors(artifacts.baseline["period_breakdown"], 3)
    assert list(table.columns) == ["Model", "Target years", "Forecasts", "MAPE", "Share of abs. error %"]
    cagr = table[table["Model"] == "3-yr CAGR"].set_index("Target years")["MAPE"]
    assert cagr["2023-2025"] > cagr["2005-2019"]


@stored
def test_download_files_point_to_existing_stored_files(artifacts):
    files = download_files(artifacts)
    keys = {f["key"] for f in files}
    assert {"baseline.summaries", "baseline.records", "indicator.effects", "baseline.metadata"} <= keys
    assert all(Path(f["path"]).is_file() for f in files)


# ---------------------------------------------------------------- missing results

def test_missing_results_directory_is_reported_not_generated(tmp_path):
    result = load_evaluation_artifacts(tmp_path / "nope", tmp_path / "nope_ind")
    assert not result.baseline_available
    assert not result.indicator_available
    assert len(result.missing) == len(BASELINE_FILES) + len(INDICATOR_FILES)
    assert result.baseline == {} and result.indicator == {}
    assert download_files(result) == []


@stored
def test_missing_indicator_results_keep_baseline_available(copied_results):
    shutil.rmtree(copied_results / INDICATOR_SUBDIR)
    result = load_evaluation_artifacts(copied_results, DEFAULT_INDICATOR_DIR)
    assert result.baseline_available
    assert not result.indicator_available
    assert all(str(INDICATOR_SUBDIR.name) in m for m in result.missing)
    assert {f["key"] for f in download_files(result)} == {"baseline.summaries", "baseline.records",
                                                          "baseline.metadata"}


@stored
def test_single_missing_baseline_file_marks_baseline_unavailable(copied_results):
    (copied_results / BASELINE_SUBDIR / "comparison_report_pooled.json").unlink()
    result = load_evaluation_artifacts(copied_results, DEFAULT_INDICATOR_DIR)
    assert not result.baseline_available
    assert any(m.endswith("comparison_report_pooled.json") for m in result.missing)


# ---------------------------------------------------------------- parsing

def test_metric_table_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        metric_table(pd.DataFrame({"model": ["x"]}))


def test_parsing_minimal_files(tmp_path):
    base = tmp_path / BASELINE_SUBDIR
    base.mkdir(parents=True)
    rows = []
    for model, variant, mape in (("A", "baseline", 5.0), ("Naive (last value)", "benchmark", 9.0)):
        for horizon in ("1", "all"):
            rows.append({"model": model, "variant": variant, "horizon": horizon, "aggregation": "pooled",
                         "mape": mape, "rmse": 10.0, "mae": 8.0, "bias": -1.0, "abs_bias": 1.0,
                         "successful_folds": 2, "failed_folds": 0, "coverage": 100.0})
    pd.DataFrame(rows).to_csv(base / "summaries.csv", index=False)
    pd.DataFrame({"origin": [1]}).to_csv(base / "records.csv", index=False)
    pd.DataFrame({"model": ["A"], "horizon": [1], "target_period": ["p"], "n_forecasts": [1],
                  "mape_pct": [5.0], "share_of_total_abs_error_pct": [100.0]}) \
        .to_csv(base / "breakdown_by_target_period.csv", index=False)
    (base / "experiment_metadata.json").write_text(json.dumps({"n_folds": 2}), encoding="utf-8")
    (base / "comparison_report_pooled.json").write_text(json.dumps({
        "best_by_metric": {"mape": {"model": "A", "value": 5.0}, "rmse": None},
        "best_by_horizon": {"1": {"model": "A", "value": 5.0}},
    }), encoding="utf-8")

    result = load_evaluation_artifacts(tmp_path, tmp_path)
    assert result.baseline_available and not result.indicator_available
    table = metric_table(result.baseline["summaries"])
    assert list(table["Horizon"]) == ["1Y", "1Y", "all", "all"]
    assert best_models(result.baseline["report_pooled"]) == {"mape": {"model": "A", "value": 5.0}}
    assert best_models_by_horizon(result.baseline["report_pooled"]) == {"1Y": {"model": "A", "value": 5.0}}
    assert metric_by_horizon(result.baseline["summaries"], "mape").loc["1Y", "A"] == 5.0
