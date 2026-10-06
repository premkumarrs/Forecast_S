# Out-of-Sample Forecast Evaluation

This document describes how `src/forecasting/evaluation/` measures the
out-of-sample predictive accuracy of the baseline models and of the indicator
and news adjustments. It measures predictive performance only; it does not
establish causal effects.

Modules:

| Module | Role |
| --- | --- |
| `splits.py` | Expanding-window walk-forward folds |
| `backtest.py` | `run_backtest`: per-fold forecasts for every model and variant |
| `news_history.py` | Historical `as_of` definition, headline-analysis cache, per-origin fetching |
| `report.py` | `summarize_backtest`: grouped metrics |
| `experiment.py` | `run_comparative_experiment`: configuration, provenance, limitations |
| `comparison.py` | Summaries, rankings, benchmark-relative and adjustment comparisons |
| `export.py` | CSV / JSON export |

## 1. Why walk-forward evaluation

The application forecasts annual series forward from a historical cutoff.
Walk-forward (rolling-origin) evaluation reproduces that situation repeatedly:
at each historical origin a model is fitted only on data available at that
origin and its forecasts are compared with values observed afterwards. Unlike
a single train/test split, it yields many out-of-sample forecasts per
horizon and shows whether a method is consistently better or only on a few
origins.

## 2. Forecast origins

`walk_forward_folds(years, min_train_years, horizon, step)` builds expanding
windows over consecutive, unique years. The first origin is the
`min_train_years`-th year; each later origin advances by `step` years. A fold
exists only if all `horizon` test years after its origin are observed, so
every fold contributes exactly one forecast per horizon `1..horizon`. With
`step > 1` the final years may go untested.

`run_comparative_experiment` uses one set of folds for every model and
variant. An optional `origins` argument keeps a subset of those folds; it
cannot introduce origins the split did not produce.

## 3. Temporal isolation

For a fold with origin year `Y`:

| Input | Data allowed |
| --- | --- |
| Target series (training) | `year <= Y` |
| Target series (evaluation) | `Y+1 .. Y+horizon` |
| Indicators | `year <= Y` |
| Historical news | dated `<= Y-12-31 23:59:59` UTC and within the news lookback window |

A new model instance is fitted per fold, input rows are sorted
chronologically, and tests verify that altering any target, indicator or news
value after the origin leaves that fold's forecasts unchanged.

## 4. Naive benchmark

`Naive (last value)` forecasts the last observed training value for every
horizon. It is variant `benchmark` and never receives indicator or news
adjustments. A method is useful only if it beats this benchmark.

## 5. Model comparison

Baseline models: `3-yr CAGR`, `Damped ETS`, `Logistic Growth` (variant
`baseline`), produced by the unchanged `BaselineModelFactory`. All models see
the same training window, origin, horizon, target series and folds.

## 6. Indicator point-in-time handling

Variant `indicator_adjusted_past_only` applies the unchanged production
`IndicatorAdjustment` to the baseline forecast using only indicator rows with
`year <= Y`. Production receives indicator data extending into the forecast
years; the backtest intentionally does not use those future values, so this
is a point-in-time historical indicator evaluation, not a replay of the
production indicator pipeline. A fold where a weighted indicator has no
observable growth through the origin is marked `failed`.

## 7. Historical news point-in-time handling

Variant `news_adjusted_historical` applies the unchanged production
`NewsAdjustment` with `as_of = Y-12-31 23:59:59` UTC (`news_as_of(Y)`):

- only analysed headlines dated in `[as_of - news_lookback_days, as_of]`
  (90 days by default, mirroring production's GDELT fetch window) are used;
- recency weights and category moving-average windows are measured from
  `as_of`, never from the current date;
- a fold with no headlines in its window is marked `failed`, not scored as
  zero impact. Headlines present but irrelevant legitimately give zero impact.

Production runs `NewsAdjustment` without `as_of`, i.e. on the latest 90 days
relative to now. The experiment never fetches news or calls an LLM; it uses
supplied, already-analysed historical headlines. `analyze_headlines_cached`
analyses each headline at most once per analysis fingerprint (system prompt,
model, prompt version) and never caches failed analyses.

## 8. Metrics

Computed with `src/forecasting/base/metrics.py` on valid forecasts
(`status == "ok"`, finite actual and prediction):

- MAPE (percent; zero actuals excluded, undefined if all actuals are zero)
- RMSE, MAE
- bias = `mean(predicted - actual)`; ranking uses absolute bias
- error mean, median and standard deviation

Two aggregations are always labelled:

- **pooled**: metrics over every valid forecast in the group;
- **fold_mean**: metrics per fold, averaged over folds with at least one valid
  forecast.

They answer different questions and are never mixed; for example pooled RMSE
and fold-mean RMSE differ by construction.

## 9. Horizon evaluation

Every summary is reported per horizon (`1..horizon`) and, separately, pooled
over all horizons (`horizon = None`, exported as `all`). If the series is too
short for the requested horizon, the largest feasible horizon is evaluated
and the reduction is recorded in the experiment metadata; no results are
produced for unsupported horizons.

## 10. Failure and coverage reporting

A model or adjustment failure on one fold never stops the experiment. Its
rows are kept with `status = "failed"` (or `invalid_prediction`), a message,
and an entry in the failure log; they never enter the metrics.

For each model, variant and horizon: `attempted_folds`, `successful_folds`,
`failed_folds` and `coverage = successful_folds / attempted_folds * 100`. A
fold is successful only if all its forecasts in the group are valid.
`rank_models(..., min_coverage=...)` lists methods below the coverage threshold
last with `eligible = False`, so a method with good errors on few folds is not
presented as equivalent to a robust one. No composite score is computed.

## 11. Benchmark-relative improvement and adjustment effects

```
improvement = (reference_error - candidate_error) / reference_error * 100
```

Positive means lower error than the reference; negative means higher. It is
undefined (`None`) when the reference error is zero. Comparisons are paired:
only forecasts valid for both methods at the same origin and target year are
used, and wins/losses/ties and the mean, median and standard deviation of the
absolute-error difference are reported alongside.

- `benchmark_improvements`: each model/variant versus the naive benchmark.
- `adjustment_effects`: `indicator_adjusted_past_only` or
  `news_adjusted_historical` versus the same model's `baseline`.

Summaries state measured changes only, e.g. "News adjustment reduced MAPE by
4.2% ..." or "News adjustment increased MAPE by 1.7% ..., indicating no
improvement under this evaluation setup."

## 12. Historical GDELT availability

The GDELT fetchers accept an explicit historical `end_date`, but the GDELT DOC
API officially covers only the most recent ~3 months; older windows may return
partial or no data. Historical news results therefore depend entirely on the
coverage of the supplied historical news data, and folds without news are
reported as failed. The experiment does not make live GDELT calls.

## 13. LLM hindsight

This evaluation prevents temporal leakage in news retrieval and
recency/window calculations, but historical interpretation by a modern LLM may
contain hindsight because the model may know events that occurred after the
historical origin. News results are a retrospective evaluation, not a perfect
historical replay.

## 14. Data provenance and synthetic data

The repository contains no real historical market dataset; its mock data
generators are random demo data. `run_comparative_experiment` requires
`dataset_name` and an explicit `is_synthetic` flag. Synthetic runs are labelled
`DEMO/SYNTHETIC` in the metadata, findings and every export, and must not be
read as evidence about real markets. Exports also record SHA-256 fingerprints
of the target, indicator and news inputs (not the raw news or indicator
values) and the full configuration, so a result can be traced to its inputs.
The run timestamp is metadata only and never used in any calculation.

## Usage

```python
from src.forecasting.evaluation import (
    run_comparative_experiment, build_comparison_report, export_csv, export_json,
)

result = run_comparative_experiment(
    series,                       # DataFrame with year, value
    dataset_name="my-market-global",
    is_synthetic=False,
    horizon=3,
    min_train_years=5,
    include_indicators=True, indicators=ind_df, indicator_weights={"gdp": 1.0},
    include_news=True, news=analysed_news_df, news_config=app_config,
)
report = build_comparison_report(result, aggregation="pooled", min_coverage=80.0)
export_json(result, "results/experiment.json")
export_csv(result, "results/summaries.csv", table="summaries")
```

## Stored results and dashboard

The real-data experiments (`scripts/run_census_baseline_experiment.py`,
`scripts/run_census_indicator_experiment.py`) write their exports to
`data/evaluation/results/`. `src/forecasting/evaluation/artifacts.py` loads those
stored files for presentation (it never reruns an experiment or accesses the network),
and the Streamlit **Evaluation** page (`pages/06_Evaluation.py`) displays them.
