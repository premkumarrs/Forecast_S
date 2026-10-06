"""Build and validate indicator inputs for the Census indicator experiment.

Reads the raw files saved by ``fetch_indicator_sources.py`` and writes:

* ``data/evaluation/indicators/us_pce_goods_annual.csv``: experiment input
  (``year``, ``indicator_key``, ``value``) for BEA series DGDSRC.
* ``data/evaluation/indicators/us_pce_goods_provenance.json``.
* ``data/evaluation/indicators/us_internet_users_assessment.json`` and
  ``us_internet_users_inspection.csv``: the year-by-year break inspection of
  World Bank/ITU IT.NET.USER.ZS. No experiment input is written for it when it
  fails the inspection.

Usage:
    python scripts/build_indicator_datasets.py
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "evaluation" / "raw" / "indicators"
OUT_DIR = ROOT / "data" / "evaluation" / "indicators"

GOODS_KEY = "us_pce_goods_nominal"
GOODS_SERIES = "DGDSRC"
INDICATOR_START_YEAR = 1999
REQUIRED_THROUGH = 2022
INTERNET_KEY = "us_internet_users_pct_lag1"
PUBLICATION_LAG_YEARS = 1
# A cumulative adoption share should not fall, or jump far above its own trend, from one
# year to the next unless the measurement changed.
MAX_DECLINE_PP = 1.0
MAX_JUMP_PP = 8.0

VINTAGE_CAVEAT = (
    "Current-vintage historical data: values are the latest revised estimates, not the "
    "figures available at each historical origin. The experiment is a current-vintage "
    "historical walk-forward evaluation, not a true real-time vintage backtest."
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_goods(fetch_log: dict) -> dict:
    register = pd.read_csv(RAW_DIR / "bea_SeriesRegister.txt", dtype=str)
    identity = register[register["%SeriesCode"] == GOODS_SERIES]
    if len(identity) != 1:
        raise ValueError(f"{GOODS_SERIES} not uniquely present in the BEA series register")
    identity = identity.iloc[0].to_dict()
    checks_identity = {
        "label_is_Goods": identity["SeriesLabel"] == "Goods",
        "metric_is_current_dollars": identity["MetricName"] == "Current Dollars",
        "calculation_is_level": identity["CalculationType"] == "Level",
        "scale_is_millions": identity["DefaultScale"] == "-6",
        "on_NIPA_table_2_3_5_line_2": "T20305:2" in identity["TableId:LineNo"].split("|"),
        "parent_is_total_PCE_DPCERC": identity["SeriesCodeParents"] == "DPCERC",
    }

    raw = gzip.decompress((RAW_DIR / "bea_NipaDataA.txt.gz").read_bytes())
    nipa = pd.read_csv(io.BytesIO(raw), dtype=str)
    rows = nipa[nipa["%SeriesCode"].str.strip() == GOODS_SERIES].copy()
    rows["year"] = rows["Period"].astype(int)
    rows["value"] = rows["Value"].str.replace(",", "", regex=False).str.strip().astype(float)
    series = rows[["year", "value"]].sort_values("year").reset_index(drop=True)

    fred_path = RAW_DIR / "fred_DGDSRC1A027NBEA.csv"
    fred_check = None
    if fred_path.exists():
        fred = pd.read_csv(fred_path)
        fred["year"] = pd.to_datetime(fred["observation_date"]).dt.year
        merged = series.merge(fred, on="year")
        diff = (merged["value"] - merged["DGDSRC1A027NBEA"] * 1000).abs().max()
        fred_check = {"years_compared": int(len(merged)), "max_abs_diff_millions": float(diff)}

    years = series["year"].tolist()
    selected = series[series["year"] >= INDICATOR_START_YEAR]
    validation = {
        **checks_identity,
        "annual": True,
        "no_duplicate_years": len(set(years)) == len(years),
        "no_gaps": years == list(range(years[0], years[-1] + 1)),
        "all_finite": all(math.isfinite(v) for v in series["value"]),
        "all_positive": bool((series["value"] > 0).all()),
        "covers_start_year": INDICATOR_START_YEAR in years,
        "covers_required_origins_through_2022": REQUIRED_THROUGH in years,
        "matches_fred_copy": fred_check is not None and fred_check["max_abs_diff_millions"] < 1.0,
    }

    out = selected.assign(indicator_key=GOODS_KEY)[["year", "indicator_key", "value"]]
    out.to_csv(OUT_DIR / "us_pce_goods_annual.csv", index=False)

    provenance = {
        "indicator_key": GOODS_KEY,
        "name": "Personal consumption expenditures: Goods (nominal)",
        "source": "U.S. Bureau of Economic Analysis (BEA), National Income and Product Accounts",
        "series_code": GOODS_SERIES,
        "series_register_entry": identity,
        "nipa_table": "Table 2.3.5 Personal Consumption Expenditures by Major Type of Product, line 2",
        "raw_file": {"name": "bea_NipaDataA.txt.gz", **fetch_log["bea_NipaDataA.txt.gz"]},
        "register_file": {"name": "bea_SeriesRegister.txt", **fetch_log["bea_SeriesRegister.txt"]},
        "crosscheck_file": {"name": "fred_DGDSRC1A027NBEA.csv", **fetch_log.get("fred_DGDSRC1A027NBEA.csv", {})},
        "fred_crosscheck": fred_check,
        "frequency": "annual",
        "unit": "millions of current US dollars (nominal, not inflation-adjusted)",
        "available_years": [years[0], years[-1]],
        "experiment_years": [int(out["year"].min()), int(out["year"].max())],
        "experiment_start_year_rationale": (
            "1999 (fixed before running the experiment) so year-over-year growth exists "
            "from 2000, the target's first year."
        ),
        "revisions": (
            "BEA revises NIPA annually (annual update each September; the 2026 update was "
            "published 2026-09-30, matching the file's Last-Modified) and restates history "
            "in comprehensive updates."
        ),
        "vintage_caveat": VINTAGE_CAVEAT,
        "point_in_time_treatment": (
            "Effective information year = observation year. BEA's first annual estimate for "
            "year Y is published in late January of Y+1, the same timing as the Census target's "
            "Q4; both are treated as known at origin Y. The backtest passes only rows with "
            "year <= origin to IndicatorAdjustment."
        ),
        "transformations": [
            "selected series DGDSRC from NipaDataA.txt",
            "removed thousands separators; values kept in millions of dollars",
            f"kept years >= {INDICATOR_START_YEAR}",
            "no interpolation, no imputation, no lag",
        ],
        "validation": validation,
    }
    (OUT_DIR / "us_pce_goods_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def assess_internet(fetch_log: dict) -> dict:
    payload = json.loads((RAW_DIR / "worldbank_IT.NET.USER.ZS_USA_footnotes.json").read_text(encoding="utf-8"))
    rows = [{"year": int(r["date"]), "value": r["value"], "footnote": (r.get("footnote") or "").strip()}
            for r in payload[1]]
    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    df = df[df["year"] >= INDICATOR_START_YEAR - PUBLICATION_LAG_YEARS].reset_index(drop=True)
    df["change_pp"] = df["value"].diff()
    df["source_label"] = (df["footnote"]
                          .str.replace("Source and notes:", "", regex=False)
                          .str.replace("Source:", "", regex=False)
                          .str.split(",").str[0].str.strip().str.rstrip(".")
                          .replace({"United States Census Bureau": "US Census Bureau"}))
    df["source_changed"] = df["source_label"] != df["source_label"].shift()
    df.loc[df.index[0], "source_changed"] = False
    df["flag"] = ""
    df.loc[df["value"].isna(), "flag"] = "missing"
    df.loc[df["change_pp"] < -MAX_DECLINE_PP, "flag"] = "decline"
    df.loc[df["change_pp"] > MAX_JUMP_PP, "flag"] = "jump"
    df.to_csv(OUT_DIR / "us_internet_users_inspection.csv", index=False)

    needed = df[(df["year"] >= INDICATOR_START_YEAR - PUBLICATION_LAG_YEARS)
                & (df["year"] <= REQUIRED_THROUGH - PUBLICATION_LAG_YEARS)]
    flagged = needed[needed["flag"] != ""]
    source_changes = needed[needed["source_changed"]]
    passes = flagged.empty and source_changes.empty
    assessment = {
        "indicator": "Individuals using the Internet (% of population), United States",
        "series_code": "IT.NET.USER.ZS",
        "source": "World Bank World Development Indicators, from the ITU World Telecommunication/ICT Indicators Database",
        "source_url": "https://data.worldbank.org/indicator/IT.NET.USER.ZS?locations=US",
        "raw_files": {name: fetch_log[name] for name in fetch_log if name.startswith("worldbank_")},
        "world_bank_last_updated": json.loads(
            (RAW_DIR / "worldbank_IT.NET.USER.ZS_USA.json").read_text(encoding="utf-8"))[0].get("lastupdated"),
        "frequency": "annual",
        "unit": "% of population",
        "available_years": [int(df.dropna(subset=["value"])["year"].min()),
                            int(df.dropna(subset=["value"])["year"].max())],
        "required_observation_years_with_lag": [INDICATOR_START_YEAR - PUBLICATION_LAG_YEARS,
                                                REQUIRED_THROUGH - PUBLICATION_LAG_YEARS],
        "missing_in_required_years": needed["value"].isna().sum().item(),
        "inspection_rules": {
            "max_decline_pp": MAX_DECLINE_PP,
            "max_jump_pp": MAX_JUMP_PP,
            "source_change_in_required_years": "fails",
        },
        "flagged_years": flagged[["year", "value", "change_pp", "flag", "source_label"]].to_dict("records"),
        "source_changes": source_changes[["year", "source_label"]].to_dict("records"),
        "publication_lag": (
            "ITU/World Bank publish year-Y values one to two years after Y. Planned "
            f"representation: effective_year = observation_year + {PUBLICATION_LAG_YEARS}, i.e. "
            "the input row labelled year Y holds the observation for Y-1, so the year <= origin "
            "filter admits at origin Y only observations through Y-1."
        ),
        "vintage_caveat": VINTAGE_CAVEAT,
        "passes_inspection": passes,
        "decision": ("included" if passes else
                     "EXCLUDED: the series fails the break inspection (documented source/"
                     "methodology changes and implausible year-to-year declines/jumps), so "
                     "specification B (goods 0.5 + internet users 0.5) is not run. Decision "
                     "made before any indicator-adjusted result was computed."),
    }
    (OUT_DIR / "us_internet_users_assessment.json").write_text(
        json.dumps(assessment, indent=2, default=str) + "\n", encoding="utf-8")
    if passes:
        lagged = df.dropna(subset=["value"]).assign(
            year=lambda x: x["year"] + PUBLICATION_LAG_YEARS, indicator_key=INTERNET_KEY)
        lagged[["year", "indicator_key", "value"]].to_csv(
            OUT_DIR / "us_internet_users_lag1_annual.csv", index=False)
    return {"assessment": assessment, "table": df}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fetch_log = json.loads((RAW_DIR / "fetch_log.json").read_text(encoding="utf-8"))

    goods = build_goods(fetch_log)
    print("Goods consumption (BEA DGDSRC) validation:")
    for name, ok in goods["validation"].items():
        print(f"  {name}: {'OK' if ok else 'FAILED'}")
    print(f"  available {goods['available_years']}, experiment input {goods['experiment_years']}, "
          f"FRED check {goods['fred_crosscheck']}")

    internet = assess_internet(fetch_log)
    table, assessment = internet["table"], internet["assessment"]
    print("\nInternet users (IT.NET.USER.ZS) year-by-year inspection:")
    print(table[["year", "value", "change_pp", "source_label", "source_changed", "flag"]]
          .round(2).to_string(index=False))
    print(f"\nPasses inspection: {assessment['passes_inspection']}")
    print(f"Decision: {assessment['decision']}")

    if not all(goods["validation"].values()):
        raise SystemExit("Goods consumption validation failed")


if __name__ == "__main__":
    main()
