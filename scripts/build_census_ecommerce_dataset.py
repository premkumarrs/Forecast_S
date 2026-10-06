"""Build the annual U.S. retail e-commerce evaluation series from official Census workbooks.

Source: U.S. Census Bureau, Quarterly Retail E-Commerce Sales Report
(https://www.census.gov/retail/ecommerce.html), time-series workbooks
``tsnotadjustedsales.xlsx`` and ``tsadjustedsales.xlsx`` saved unmodified in
``data/evaluation/raw/``.

annual_value(year) = Q1 + Q2 + Q3 + Q4 of the NOT seasonally adjusted quarterly
e-commerce sales (millions of dollars). The seasonally adjusted four-quarter sum
is recorded only as a cross-check and is never mixed into ``value``.

Usage:
    python scripts/build_census_ecommerce_dataset.py
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "evaluation"
RAW_DIR = DATA_DIR / "raw"

SOURCE_PAGE = "https://www.census.gov/retail/ecommerce.html"
RAW_BASE_URL = "https://www.census.gov/retail/mrts/www/data/excel/"
NSA_FILE = "tsnotadjustedsales.xlsx"
SA_FILE = "tsadjustedsales.xlsx"
RETRIEVAL_DATE = "2026-10-06"
SOURCE_LABEL = "U.S. Census Bureau Quarterly E-Commerce Report"
UNIT = "millions of US dollars (current prices)"
AGGREGATION = "sum of Q1-Q4 not seasonally adjusted quarterly e-commerce sales"

VINTAGE_STATEMENT = (
    "The experiment uses the currently available/revised Census historical series. "
    "Values are therefore not necessarily identical to the information that would "
    "have been available to a forecaster at each historical origin."
)
NEWS_STATEMENT = (
    "Historical analyzed-news data sufficient for a comparable long-horizon experiment "
    "is unavailable; therefore the news-adjusted variant is excluded from the "
    "real-data results."
)

QUARTER_LABEL = re.compile(r"^\s*([1-4])(?:st|nd|rd|th) quarter (\d{4})\s*(?:\((p|r)\))?\s*$")
ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
# Annual YoY growth outside this band is flagged for manual review, not altered.
GROWTH_REVIEW_BAND = (-0.05, 0.35)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_quarterly(path: Path) -> pd.DataFrame:
    """Parse one Census time-series workbook into (year, quarter, ecommerce, flag) rows."""
    raw = pd.read_excel(path, header=None)
    rows = []
    for _, row in raw.iterrows():
        label = row[1]
        if not isinstance(label, str):
            continue
        match = QUARTER_LABEL.match(label)
        if not match:
            continue
        quarter, year, flag = int(match.group(1)), int(match.group(2)), match.group(3) or ""
        rows.append({
            "year": year,
            "quarter": quarter,
            "ecommerce": float(row[3]),
            "flag": flag,
            "source_label": label.strip(),
        })
    df = pd.DataFrame(rows)
    if df.duplicated(["year", "quarter"]).any():
        raise ValueError(f"{path.name}: duplicate quarters")
    return df.sort_values(["year", "quarter"]).reset_index(drop=True)


def last_revised(path: Path) -> str:
    raw = pd.read_excel(path, header=None)
    for value in raw[1].dropna():
        if isinstance(value, str) and value.strip().startswith("Last Revised:"):
            return value.strip().removeprefix("Last Revised:").strip()
    return ""


def build() -> dict:
    nsa_path, sa_path = RAW_DIR / NSA_FILE, RAW_DIR / SA_FILE
    nsa = read_quarterly(nsa_path).rename(
        columns={"ecommerce": "nsa", "flag": "nsa_flag", "source_label": "nsa_label"})
    sa = read_quarterly(sa_path).rename(
        columns={"ecommerce": "sa", "flag": "sa_flag", "source_label": "sa_label"})
    quarterly = nsa.merge(sa, on=["year", "quarter"], how="outer", validate="one_to_one")
    if quarterly[["nsa", "sa"]].isna().any().any():
        raise ValueError("adjusted and not-adjusted workbooks cover different quarters")

    counts = quarterly.groupby("year")["quarter"].nunique()
    complete_years = counts[counts == 4].index
    incomplete_years = {int(y): int(c) for y, c in counts[counts < 4].items()}

    provenance_rows = []
    for year in complete_years:
        q = quarterly[quarterly["year"] == year].set_index("quarter")
        nsa_sum = float(sum(q.loc[i, "nsa"] for i in range(1, 5)))
        sa_sum = float(sum(q.loc[i, "sa"] for i in range(1, 5)))
        provenance_rows.append({
            "year": int(year),
            **{f"q{i}_nsa": q.loc[i, "nsa"] for i in range(1, 5)},
            **{f"q{i}_flag": q.loc[i, "nsa_flag"] for i in range(1, 5)},
            "annual_nsa_sum": nsa_sum,
            "annual_sa_sum_crosscheck": sa_sum,
            "sa_minus_nsa_pct": round((sa_sum - nsa_sum) / nsa_sum * 100, 4),
            "source": SOURCE_LABEL,
            "source_file": RAW_BASE_URL + NSA_FILE,
            "frequency": "quarterly -> annual",
            "unit": UNIT,
            "aggregation": AGGREGATION,
        })
    provenance = pd.DataFrame(provenance_rows)
    provenance["yoy_growth_pct"] = (provenance["annual_nsa_sum"].pct_change() * 100).round(3)
    annual = provenance[["year", "annual_nsa_sum"]].rename(columns={"annual_nsa_sum": "value"})

    tidy = quarterly[["year", "quarter", "nsa", "nsa_flag", "sa", "sa_flag", "nsa_label", "sa_label"]]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    annual.to_csv(DATA_DIR / "us_census_ecommerce_annual.csv", index=False)
    provenance.to_csv(DATA_DIR / "us_census_ecommerce_annual_provenance.csv", index=False)
    tidy.to_csv(DATA_DIR / "us_census_ecommerce_quarterly.csv", index=False)

    metadata = {
        "dataset_name": "us_census_ecommerce_annual",
        "is_synthetic": False,
        "source": SOURCE_LABEL,
        "source_page": SOURCE_PAGE,
        "raw_files": {
            NSA_FILE: {"url": RAW_BASE_URL + NSA_FILE, "sha256": sha256(nsa_path),
                       "last_revised": last_revised(nsa_path)},
            SA_FILE: {"url": RAW_BASE_URL + SA_FILE, "sha256": sha256(sa_path),
                      "last_revised": last_revised(sa_path)},
        },
        "retrieval_date": RETRIEVAL_DATE,
        "series": "Estimated quarterly U.S. retail e-commerce sales, not seasonally adjusted",
        "crosscheck_series": "Estimated quarterly U.S. retail e-commerce sales, seasonally adjusted",
        "original_frequency": "quarterly",
        "target_frequency": "annual",
        "unit": UNIT,
        "aggregation": AGGREGATION,
        "official_annual_total_in_source": False,
        "first_quarter_in_source": f"{int(quarterly.iloc[0]['year'])}Q{int(quarterly.iloc[0]['quarter'])}",
        "last_quarter_in_source": f"{int(quarterly.iloc[-1]['year'])}Q{int(quarterly.iloc[-1]['quarter'])}",
        "excluded_incomplete_years": incomplete_years,
        "first_year": int(annual["year"].min()),
        "last_year": int(annual["year"].max()),
        "n_years": int(len(annual)),
        "coverage_note": (
            "Estimates include only businesses with paid employees. The April 2025 benchmark "
            "restated the entire series (2017 NAICS, employer-only coverage aligned with the "
            "Annual Integrated Economic Survey, restated 2022 Annual Retail Trade Survey "
            "results); the workbooks contain that restated history, so no vintages are spliced."
        ),
        "restatement_crosscheck": {
            "source": "https://www.census.gov/retail/mrts/www/NAICS_Restatement_Summary.pdf",
            "2022_ecommerce_before_restatement": 1012636,
            "2022_ecommerce_restated": 997477,
            "2022_annual_nsa_sum_in_this_dataset": float(
                annual.loc[annual["year"] == 2022, "value"].iloc[0]),
        },
        "revision_notice": (
            "Census notice: as of 2026-09-28 the 2Q 2026 report no longer holds the most "
            "up-to-date estimates; revised figures are scheduled with the 3Q 2026 release on "
            "2026-11-19. Only 2026 quarters are flagged (p)/(r) in this vintage, and 2026 is "
            "excluded as incomplete."
        ),
        "vintage_statement": VINTAGE_STATEMENT,
        "news_statement": NEWS_STATEMENT,
        "indicators": "not included",
    }
    (DATA_DIR / "us_census_ecommerce_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"annual": annual, "provenance": provenance, "quarterly": quarterly, "metadata": metadata}


def validate(built: dict) -> list:
    annual, provenance, quarterly = built["annual"], built["provenance"], built["quarterly"]
    problems = []
    years = annual["year"].tolist()
    if len(set(years)) != len(years):
        problems.append("duplicate years")
    if years != list(range(years[0], years[-1] + 1)):
        problems.append("years are not consecutive")
    if annual["value"].isna().any():
        problems.append("missing values")
    if not all(math.isfinite(v) and v >= 0 for v in annual["value"]):
        problems.append("non-finite or negative values")
    for _, row in provenance.iterrows():
        if row["annual_nsa_sum"] != sum(row[f"q{i}_nsa"] for i in range(1, 5)):
            problems.append(f"{row['year']}: annual value != Q1+Q2+Q3+Q4")
    if (quarterly["nsa"] < 0).any() or (quarterly["sa"] < 0).any():
        problems.append("negative quarterly values")
    return problems


def main() -> None:
    built = build()
    annual, provenance, meta = built["annual"], built["provenance"], built["metadata"]
    problems = validate(built)

    print(f"Source: {meta['source']} ({meta['source_page']})")
    for name, info in meta["raw_files"].items():
        print(f"  {name}: last revised {info['last_revised']}, sha256 {info['sha256']}")
    print(f"Series: {meta['series']}")
    print(f"Unit: {meta['unit']}")
    print(f"Aggregation: {meta['aggregation']}")
    print(f"Quarters in source: {meta['first_quarter_in_source']} .. {meta['last_quarter_in_source']}")
    print(f"Excluded incomplete years (quarters present): {meta['excluded_incomplete_years']}")
    print(f"Years: {meta['first_year']}-{meta['last_year']} ({meta['n_years']} observations)")
    print("\nFirst 5 rows:\n" + annual.head(5).to_string(index=False))
    print("\nLast 5 rows:\n" + annual.tail(5).to_string(index=False))

    print("\nNSA sum vs SA sum cross-check (SA minus NSA, %):")
    diffs = provenance["sa_minus_nsa_pct"]
    print(f"  min {diffs.min():.3f}  max {diffs.max():.3f}  mean abs {diffs.abs().mean():.3f}")

    low, high = GROWTH_REVIEW_BAND
    growth = provenance.set_index("year")["yoy_growth_pct"].dropna()
    flagged = growth[(growth < low * 100) | (growth > high * 100)]
    print("\nYear-over-year growth (%):")
    print("  " + ", ".join(f"{y}: {g:.1f}" for y, g in growth.items()))
    print(f"  outside review band {low:.0%}..{high:.0%}: "
          + (", ".join(f"{y} ({g:.1f}%)" for y, g in flagged.items()) or "none"))

    flagged_quarters = built["quarterly"][(built["quarterly"]["nsa_flag"] != "")
                                          | (built["quarterly"]["sa_flag"] != "")]
    print("\nQuarters marked preliminary/revised in this vintage:")
    print(flagged_quarters[["year", "quarter", "nsa_flag", "sa_flag"]].to_string(index=False))

    print("\nValidation: " + ("PASSED" if not problems else "FAILED: " + "; ".join(problems)))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
