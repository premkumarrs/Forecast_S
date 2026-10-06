"""Run the baseline comparative experiment on the Census annual e-commerce series.

Uses ``run_comparative_experiment`` (no separate backtesting loop) and the Phase 4
export/comparison functions. The by-target-year and by-period tables are
descriptive breakdowns of the experiment's own fold-level records; they do not
refit any model and no years are removed from the experiment.

Usage:
    python scripts/run_census_baseline_experiment.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.evaluation import (  # noqa: E402
    build_comparison_report,
    export_csv,
    export_json,
    run_comparative_experiment,
)
from src.forecasting.evaluation.backtest import STATUS_OK  # noqa: E402

DATASET_PATH = ROOT / "data" / "evaluation" / "us_census_ecommerce_annual.csv"
DATASET_METADATA_PATH = ROOT / "data" / "evaluation" / "us_census_ecommerce_metadata.json"
RESULTS_DIR = ROOT / "data" / "evaluation" / "results" / "census_baseline"

DATASET_NAME = "U.S. Census Bureau annual retail e-commerce sales"
DATA_LABEL = "REAL / CURRENT-VINTAGE"
MODELS = ["naive_last_value", "cagr", "damped_ets", "logistic_growth"]
MIN_TRAIN_YEARS = 5
HORIZON = 3
STEP = 1
MIN_COVERAGE = 80.0
PANDEMIC_TARGET_YEARS = (2020, 2021, 2022)
PERIODS = {"2005-2019": (2005, 2019), "2020-2022": (2020, 2022), "2023-2025": (2023, 2025)}

METHODOLOGY_NOTES = [
    "Expanding-window walk-forward backtest; each fold trains on years <= origin and "
    "forecasts origin+1..origin+horizon. Only folds with a complete test horizon are used.",
    "All models are evaluated on exactly the same origins and target years.",
    "Annual value = sum of Q1-Q4 not seasonally adjusted quarterly e-commerce sales "
    "(millions of current US dollars).",
    "The experiment uses the currently available/revised Census historical series. Values "
    "are therefore not necessarily identical to the information that would have been "
    "available to a forecaster at each historical origin.",
    "Historical analyzed-news data sufficient for a comparable long-horizon experiment is "
    "unavailable; therefore the news-adjusted variant is excluded from the real-data results.",
    "Indicators are not evaluated in this experiment.",
    "Pooled metrics use every valid forecast; fold-mean metrics average per-fold metrics. "
    "Improvements vs naive are paired on identical (origin, target year) forecasts.",
    "No composite score and no statistical significance testing are used.",
    "2020 (pandemic jump) is kept in the data; target-year/period tables are descriptive only.",
]


def validate_dataset(df: pd.DataFrame) -> dict:
    years = df["year"].tolist()
    values = df["value"].tolist()
    checks = {
        "n_observations_is_26": len(df) == 26,
        "years_2000_to_2025": (min(years), max(years)) == (2000, 2025),
        "no_missing_years": years == list(range(min(years), max(years) + 1)),
        "no_duplicate_years": len(set(years)) == len(years),
        "no_missing_values": not df["value"].isna().any(),
        "all_values_finite": all(math.isfinite(v) for v in values),
        "all_values_non_negative": all(v >= 0 for v in values),
        "is_synthetic_false": True,
    }
    return checks


def records_with_errors(records: pd.DataFrame) -> pd.DataFrame:
    df = records.copy()
    actual = pd.to_numeric(df["actual"], errors="coerce")
    predicted = pd.to_numeric(df["predicted"], errors="coerce")
    df["valid"] = (df["status"] == STATUS_OK) & np.isfinite(actual) & np.isfinite(predicted)
    df["abs_error"] = (predicted - actual).abs().where(df["valid"])
    df["ape_pct"] = (df["abs_error"] / actual.abs() * 100).where(df["valid"])
    return df


def target_year_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    table = df.pivot_table(index=["horizon", "year"], columns="model", values="ape_pct",
                           aggfunc="first").reset_index()
    table.columns.name = None
    models = [c for c in table.columns if c not in ("horizon", "year")]
    table["lowest_ape_model"] = table[models].idxmin(axis=1)
    return table.rename(columns={"year": "target_year"})


def period_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, horizon), group in df.groupby(["model", "horizon"]):
        total_abs = group["abs_error"].sum()
        for label, (lo, hi) in PERIODS.items():
            sub = group[(group["year"] >= lo) & (group["year"] <= hi)]
            valid = sub[sub["valid"]]
            rows.append({
                "model": model, "horizon": int(horizon), "target_period": label,
                "n_forecasts": len(sub), "n_valid": len(valid),
                "mape_pct": valid["ape_pct"].mean() if len(valid) else None,
                "mae": valid["abs_error"].mean() if len(valid) else None,
                "share_of_total_abs_error_pct": (valid["abs_error"].sum() / total_abs * 100
                                                 if total_abs else None),
            })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    checks = validate_dataset(df)
    print("Dataset validation:")
    for name, ok in checks.items():
        print(f"  {name}: {'OK' if ok else 'FAILED'}")
    if not all(checks.values()):
        raise SystemExit("Dataset validation failed; experiment not run.")

    dataset_meta = json.loads(DATASET_METADATA_PATH.read_text(encoding="utf-8"))
    result = run_comparative_experiment(
        df,
        dataset_name=DATASET_NAME,
        is_synthetic=False,
        models=MODELS,
        horizon=HORIZON,
        min_train_years=MIN_TRAIN_YEARS,
        step=STEP,
        notes=" ".join(METHODOLOGY_NOTES),
    )
    result.metadata["api_data_label"] = result.metadata["data_label"]
    result.metadata["data_label"] = DATA_LABEL

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    export_json(result, str(RESULTS_DIR / "experiment_result.json"))
    for table in ("summaries", "improvements", "records", "failures"):
        export_csv(result, str(RESULTS_DIR / f"{table}.csv"), table=table)

    reports = {agg: build_comparison_report(result, aggregation=agg, min_coverage=MIN_COVERAGE)
               for agg in ("pooled", "fold_mean")}
    for agg, report in reports.items():
        (RESULTS_DIR / f"comparison_report_{agg}.json").write_text(
            json.dumps(report.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")

    records = records_with_errors(result.records_frame())
    target_year_breakdown(records).to_csv(RESULTS_DIR / "breakdown_by_target_year.csv", index=False)
    period_breakdown(records).to_csv(RESULTS_DIR / "breakdown_by_target_period.csv", index=False)

    metadata = {
        "dataset_name": DATASET_NAME,
        "data_label": DATA_LABEL,
        "is_synthetic": False,
        "source": dataset_meta["source"],
        "source_page": dataset_meta["source_page"],
        "source_vintage": {k: v["last_revised"] for k, v in dataset_meta["raw_files"].items()},
        "current_vintage_status": dataset_meta["vintage_statement"],
        "dataset_file": str(DATASET_PATH.relative_to(ROOT)).replace("\\", "/"),
        "dataset_unit": dataset_meta["unit"],
        "dataset_aggregation": dataset_meta["aggregation"],
        "n_observations": len(df),
        "first_year": int(df["year"].min()),
        "last_year": int(df["year"].max()),
        "dataset_validation": checks,
        "models": list(result.config.models),
        "min_train_years": MIN_TRAIN_YEARS,
        "horizon": result.config.horizon,
        "evaluated_horizons": result.evaluated_horizons,
        "step": STEP,
        "n_folds": result.metadata["n_folds"],
        "origins": result.metadata["origins"],
        "min_coverage_for_best": MIN_COVERAGE,
        "run_timestamp": result.metadata["run_timestamp"],
        "input_fingerprint": result.metadata["input_fingerprints"]["historical_data"],
        "methodology_notes": METHODOLOGY_NOTES,
        "limitations": result.metadata["limitations"],
        "files": sorted(p.name for p in RESULTS_DIR.iterdir()) + ["experiment_metadata.json"],
    }
    (RESULTS_DIR / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"\nFolds: {metadata['n_folds']}  origins {metadata['origins'][0]}..{metadata['origins'][-1]}")
    print(f"Failures: {len(result.failures_frame())}")
    print(f"Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
