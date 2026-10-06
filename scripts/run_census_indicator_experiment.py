"""Run the past-only indicator experiment on the Census annual e-commerce series.

Specification A: BEA personal consumption expenditures on goods (DGDSRC), indicator
weight 1.0, overall indicator weight 0.3. Specification B (goods 0.5 + lagged
internet users 0.5) runs only if the internet-user series passed the break
inspection in ``build_indicator_datasets.py``.

Every specification is one ``run_comparative_experiment`` call, so the baseline,
benchmark and ``indicator_adjusted_past_only`` variants share the same folds. All
other tables are descriptive analyses of the exported fold-level records; no model
is refitted outside the experiment API.

Usage:
    python scripts/run_census_indicator_experiment.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_census_baseline_experiment import (  # noqa: E402
    DATA_LABEL,
    DATASET_METADATA_PATH,
    DATASET_NAME,
    DATASET_PATH,
    HORIZON,
    MIN_COVERAGE,
    MIN_TRAIN_YEARS,
    MODELS,
    STEP,
    validate_dataset,
)
from src.forecasting.adjustments.indicator_adjustment import IndicatorAdjustment  # noqa: E402
from src.forecasting.evaluation import (  # noqa: E402
    build_comparison_report,
    export_csv,
    export_json,
    run_comparative_experiment,
)
from src.forecasting.evaluation.backtest import (  # noqa: E402
    STATUS_OK,
    VARIANT_BASELINE,
    VARIANT_BENCHMARK,
    VARIANT_INDICATOR_ADJUSTED_PAST_ONLY,
)

INDICATOR_DIR = ROOT / "data" / "evaluation" / "indicators"
BASELINE_RECORDS = ROOT / "data" / "evaluation" / "results" / "census_baseline" / "records.csv"
RESULTS_DIR = ROOT / "data" / "evaluation" / "results" / "census_indicators"

OVERALL_INDICATOR_WEIGHT = 0.3
GOODS_KEY = "us_pce_goods_nominal"
INTERNET_KEY = "us_internet_users_pct_lag1"
SPECS = {
    "spec_a_goods": {GOODS_KEY: 1.0},
    "spec_b_goods_internet": {GOODS_KEY: 0.5, INTERNET_KEY: 0.5},
}
PERIODS = {"2005-2019": (2005, 2019), "2020": (2020, 2020), "2021": (2021, 2021),
           "2022": (2022, 2022), "2023-2025": (2023, 2025)}

METHODOLOGY_NOTES = [
    "Current-vintage historical walk-forward evaluation, not a true real-time vintage backtest: "
    "the Census target and the BEA indicator are the latest revised series.",
    "For origin Y the market training data and the indicator data passed to the unchanged "
    "production IndicatorAdjustment are restricted to year <= Y.",
    "Indicator weights and the overall indicator weight (0.3) were fixed before the run and "
    "not tuned.",
    "IndicatorAdjustment applies one multiplicative factor (1 + 0.3 x average yearly weighted "
    "indicator growth, capped at +/-30% before weighting) to every forecast year of a fold.",
    "Effects are paired comparisons of the adjusted variant against the same model's baseline "
    "on identical forecasts. No composite score and no significance testing.",
    "The constant-uplift diagnostic applies the full-sample mean adjustment factor to every "
    "baseline forecast; it is NOT point-in-time and is used only to judge whether the "
    "indicator's variation across origins adds anything beyond a fixed level shift.",
    "News is disabled. 2020 is kept; period tables are descriptive only.",
]


def load_indicators(weights: dict) -> pd.DataFrame:
    frames = [pd.read_csv(INDICATOR_DIR / "us_pce_goods_annual.csv")]
    if INTERNET_KEY in weights:
        frames.append(pd.read_csv(INDICATOR_DIR / "us_internet_users_lag1_annual.csv"))
    return pd.concat(frames, ignore_index=True)


def valid_records(records: pd.DataFrame) -> pd.DataFrame:
    df = records[records["status"] == STATUS_OK].copy()
    df["error"] = df["predicted"] - df["actual"]
    return df


def check_baseline_reproduced(records: pd.DataFrame) -> dict:
    reference = pd.read_csv(BASELINE_RECORDS)
    keys = ["origin", "model", "variant", "year", "horizon"]
    mine = records[records["variant"].isin([VARIANT_BASELINE, VARIANT_BENCHMARK])]
    merged = mine.merge(reference, on=keys, suffixes=("", "_ref"), validate="one_to_one")
    max_diff = float((merged["predicted"] - merged["predicted_ref"]).abs().max())
    # The reference is read back from CSV, so allow float round-trip differences only.
    close = np.isclose(merged["predicted"], merged["predicted_ref"], rtol=1e-12, atol=0.0).all()
    return {"rows_compared": int(len(merged)), "rows_expected": int(len(reference)),
            "max_abs_prediction_diff": max_diff,
            "identical_within_csv_roundtrip_rtol_1e-12": bool(len(merged) == len(reference) and close)}


def adjustment_by_origin(records: pd.DataFrame, indicators: pd.DataFrame, weights: dict) -> pd.DataFrame:
    pairs = records[records["variant"] == VARIANT_BASELINE].merge(
        records[records["variant"] == VARIANT_INDICATOR_ADJUSTED_PAST_ONLY],
        on=["origin", "model", "year", "horizon"], suffixes=("_base", "_adj"))
    pairs = pairs[(pairs["status_base"] == STATUS_OK) & (pairs["status_adj"] == STATUS_OK)]
    pairs["ratio"] = pairs["predicted_adj"] / pairs["predicted_base"]
    observed = pairs.groupby("origin")["ratio"].agg(["min", "max", "count"])

    rows = []
    for origin in sorted(records["origin"].unique()):
        available = indicators[indicators["year"] <= origin].sort_values(["indicator_key", "year"])
        adjustment = IndicatorAdjustment(weight=OVERALL_INDICATOR_WEIGHT)
        raw_signal = adjustment.calculate(available, {"indicator_weights": weights})
        applied = adjustment.get_weighted_adjustment(raw_signal)
        rows.append({
            "origin": int(origin),
            "indicator_years_used": f"{int(available['year'].min())}-{int(available['year'].max())}",
            "n_years_averaged": int(available["year"].nunique()),
            "raw_signal_pct": raw_signal * 100,
            "applied_adjustment_pct": applied * 100,
            "adjustment_factor": 1 + applied,
            "observed_ratio_min": observed.loc[origin, "min"],
            "observed_ratio_max": observed.loc[origin, "max"],
            "n_forecasts_checked": int(observed.loc[origin, "count"]),
        })
    return pd.DataFrame(rows)


def _metrics(errors: pd.Series, actual: pd.Series) -> dict:
    e = errors.to_numpy(dtype=float)
    a = actual.to_numpy(dtype=float)
    return {
        "mape": float(np.mean(np.abs(e) / np.abs(a)) * 100),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
        "abs_bias": float(abs(np.mean(e))),
        "error_variance": float(np.var(e)),
        "mse": float(np.mean(e ** 2)),
    }


def effect_table(records: pd.DataFrame, mean_factor: float) -> pd.DataFrame:
    valid = valid_records(records)
    base = valid[valid["variant"] == VARIANT_BASELINE]
    adj = valid[valid["variant"] == VARIANT_INDICATOR_ADJUSTED_PAST_ONLY]
    paired = base.merge(adj, on=["origin", "model", "year", "horizon", "actual"], suffixes=("_b", "_a"))
    rows = []
    for (model, horizon), g in list(paired.groupby(["model", "horizon"])) + \
            [((m, "all"), g) for m, g in paired.groupby("model")]:
        b = _metrics(g["error_b"], g["actual"])
        a = _metrics(g["error_a"], g["actual"])
        uplift_err = g["predicted_b"] * mean_factor - g["actual"]
        c = _metrics(uplift_err, g["actual"])
        diff = (g["error_a"].abs() - g["error_b"].abs())
        wins = diff < -1e-9
        row = {"model": model, "horizon": horizon, "n_paired": len(g)}
        for k in ("mape", "rmse", "mae", "bias", "abs_bias"):
            row[f"baseline_{k}"] = b[k]
            row[f"adjusted_{k}"] = a[k]
        for k in ("mape", "rmse", "mae"):
            row[f"{k}_change_pct"] = (a[k] - b[k]) / b[k] * 100
        row["bias_change"] = a["bias"] - b["bias"]
        row["abs_bias_change"] = a["abs_bias"] - b["abs_bias"]
        row["baseline_bias_sq"] = b["bias"] ** 2
        row["adjusted_bias_sq"] = a["bias"] ** 2
        row["baseline_error_variance"] = b["error_variance"]
        row["adjusted_error_variance"] = a["error_variance"]
        row["mse_change"] = a["mse"] - b["mse"]
        row["mse_change_from_bias_sq"] = a["bias"] ** 2 - b["bias"] ** 2
        row["mse_change_from_variance"] = a["error_variance"] - b["error_variance"]
        row["wins"] = int(wins.sum())
        row["losses"] = int((diff > 1e-9).sum())
        row["ties"] = int((diff.abs() <= 1e-9).sum())
        row["baseline_underforecast_share_pct"] = float((g["error_b"] < 0).mean() * 100)
        row["wins_on_baseline_underforecasts"] = int((wins & (g["error_b"] < 0)).sum())
        row["mean_abs_error_diff"] = float(diff.mean())
        row["constant_uplift_mae"] = c["mae"]
        row["constant_uplift_rmse"] = c["rmse"]
        row["constant_uplift_mape"] = c["mape"]
        rows.append(row)
    return pd.DataFrame(rows)


def period_table(records: pd.DataFrame) -> pd.DataFrame:
    valid = valid_records(records)
    base = valid[valid["variant"] == VARIANT_BASELINE]
    adj = valid[valid["variant"] == VARIANT_INDICATOR_ADJUSTED_PAST_ONLY]
    paired = base.merge(adj, on=["origin", "model", "year", "horizon", "actual"], suffixes=("_b", "_a"))
    rows = []
    for (model, horizon), g in paired.groupby(["model", "horizon"]):
        for label, (lo, hi) in PERIODS.items():
            sub = g[(g["year"] >= lo) & (g["year"] <= hi)]
            if sub.empty:
                continue
            b, a = _metrics(sub["error_b"], sub["actual"]), _metrics(sub["error_a"], sub["actual"])
            rows.append({
                "model": model, "horizon": int(horizon), "target_period": label, "n": len(sub),
                "baseline_mape": b["mape"], "adjusted_mape": a["mape"],
                "baseline_mae": b["mae"], "adjusted_mae": a["mae"],
                "mae_change_pct": (a["mae"] - b["mae"]) / b["mae"] * 100,
                "baseline_bias": b["bias"], "adjusted_bias": a["bias"],
                "wins": int((sub["error_a"].abs() < sub["error_b"].abs() - 1e-9).sum()),
                "losses": int((sub["error_a"].abs() > sub["error_b"].abs() + 1e-9).sum()),
            })
    return pd.DataFrame(rows)


def run_spec(name: str, weights: dict, target: pd.DataFrame, dataset_meta: dict, checks: dict) -> dict:
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    indicators = load_indicators(weights)
    result = run_comparative_experiment(
        target,
        dataset_name=DATASET_NAME,
        is_synthetic=False,
        models=MODELS,
        horizon=HORIZON,
        min_train_years=MIN_TRAIN_YEARS,
        step=STEP,
        include_indicators=True,
        indicators=indicators,
        indicator_weights=weights,
        indicator_weight=OVERALL_INDICATOR_WEIGHT,
        notes=" ".join(METHODOLOGY_NOTES),
    )
    result.metadata["api_data_label"] = result.metadata["data_label"]
    result.metadata["data_label"] = DATA_LABEL

    export_json(result, str(out_dir / "experiment_result.json"))
    for table in ("summaries", "improvements", "records", "failures"):
        export_csv(result, str(out_dir / f"{table}.csv"), table=table)
    for agg in ("pooled", "fold_mean"):
        report = build_comparison_report(result, aggregation=agg, min_coverage=MIN_COVERAGE)
        (out_dir / f"comparison_report_{agg}.json").write_text(
            json.dumps(report.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")

    records = result.records_frame()
    reproduced = check_baseline_reproduced(records)
    by_origin = adjustment_by_origin(records, indicators, weights)
    by_origin.to_csv(out_dir / "indicator_adjustment_by_origin.csv", index=False)
    mean_factor = float(by_origin["adjustment_factor"].mean())
    effect_table(records, mean_factor).to_csv(out_dir / "indicator_effect_vs_baseline.csv", index=False)
    period_table(records).to_csv(out_dir / "indicator_effect_by_target_period.csv", index=False)

    metadata = {
        "specification": name,
        "dataset_name": DATASET_NAME,
        "data_label": DATA_LABEL,
        "is_synthetic": False,
        "source": dataset_meta["source"],
        "current_vintage_status": dataset_meta["vintage_statement"],
        "n_observations": len(target),
        "dataset_validation": checks,
        "models": list(result.config.models),
        "min_train_years": MIN_TRAIN_YEARS,
        "horizon": result.config.horizon,
        "step": STEP,
        "n_folds": result.metadata["n_folds"],
        "origins": result.metadata["origins"],
        "indicator_weights": weights,
        "overall_indicator_weight": OVERALL_INDICATOR_WEIGHT,
        "indicator_files": sorted({f"data/evaluation/indicators/{f}" for f in
                                   (["us_pce_goods_annual.csv", "us_pce_goods_provenance.json"]
                                    + (["us_internet_users_lag1_annual.csv"] if INTERNET_KEY in weights else []))}),
        "indicator_fingerprint": result.metadata["input_fingerprints"]["indicators"],
        "target_fingerprint": result.metadata["input_fingerprints"]["historical_data"],
        "baseline_reproduces_census_baseline_run": reproduced,
        "mean_adjustment_factor": mean_factor,
        "run_timestamp": result.metadata["run_timestamp"],
        "methodology_notes": METHODOLOGY_NOTES,
        "limitations": result.metadata["limitations"],
    }
    (out_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"result": result, "metadata": metadata, "by_origin": by_origin}


def main() -> None:
    target = pd.read_csv(DATASET_PATH)
    checks = validate_dataset(target)
    if not all(checks.values()):
        raise SystemExit("Target dataset validation failed; experiment not run.")
    dataset_meta = json.loads(DATASET_METADATA_PATH.read_text(encoding="utf-8"))
    goods_prov = json.loads((INDICATOR_DIR / "us_pce_goods_provenance.json").read_text(encoding="utf-8"))
    if not all(goods_prov["validation"].values()):
        raise SystemExit("Goods consumption validation failed; experiment not run.")
    internet = json.loads((INDICATOR_DIR / "us_internet_users_assessment.json").read_text(encoding="utf-8"))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_log = {"specifications": {}}
    for name, weights in SPECS.items():
        if INTERNET_KEY in weights and not internet["passes_inspection"]:
            run_log["specifications"][name] = {"status": "not run", "reason": internet["decision"]}
            print(f"{name}: not run ({internet['decision']})")
            continue
        out = run_spec(name, weights, target, dataset_meta, checks)
        meta = out["metadata"]
        run_log["specifications"][name] = {"status": "run", "weights": weights,
                                           "n_folds": meta["n_folds"],
                                           "failures": len(out["result"].failures_frame()),
                                           "baseline_reproduced": meta["baseline_reproduces_census_baseline_run"]}
        print(f"{name}: {meta['n_folds']} folds, failures {len(out['result'].failures_frame())}, "
              f"baseline reproduced {meta['baseline_reproduces_census_baseline_run']}")
        print(out["by_origin"].round(4).to_string(index=False))
    (RESULTS_DIR / "run_log.json").write_text(json.dumps(run_log, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
