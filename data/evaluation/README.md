# Evaluation datasets

## `us_census_ecommerce_annual.csv`

Annual U.S. retail e-commerce sales built from the **U.S. Census Bureau Quarterly
E-Commerce Report**. This is real, official data (`is_synthetic=False`), used as the
target series for the real-data comparative experiment.

| Item | Value |
|---|---|
| Source page | https://www.census.gov/retail/ecommerce.html |
| Raw files | `tsnotadjustedsales.xlsx` (used), `tsadjustedsales.xlsx` (cross-check only), from https://www.census.gov/retail/mrts/www/data/excel/ |
| Workbook vintage | "Last Revised: August 18, 2026" (2Q 2026 release) |
| Retrieval date | 2026-10-06 |
| Series | Estimated quarterly U.S. retail e-commerce sales, **not seasonally adjusted** |
| Original frequency | Quarterly (1999Q4 to 2026Q2) |
| Unit | Millions of US dollars, current prices (not inflation-adjusted) |
| Aggregation | `value(year) = Q1 + Q2 + Q3 + Q4` (not seasonally adjusted) |
| Coverage | 2000 to 2025, 26 consecutive annual observations |
| Excluded | 1999 (only Q4 published); 2026 (only Q1 and Q2 published) |

The seasonally adjusted four-quarter sum is stored only as a cross-check
(`annual_sa_sum_crosscheck`). It is within -0.66% to +0.19% of the not-adjusted
sum and is never mixed into `value`. The quarterly workbooks do not publish an
official annual total, so there is no separate annual total to compare against.

### Files

- `raw/*.xlsx`: unmodified official workbooks; SHA-256 hashes are in the metadata file.
- `us_census_ecommerce_annual.csv`: clean experiment input, columns `year`, `value` only.
- `us_census_ecommerce_annual_provenance.csv`: per year, the four quarterly values and
  their revision flags, the not-adjusted sum, the adjusted-sum cross-check, YoY growth,
  source, frequency, unit and aggregation.
- `us_census_ecommerce_quarterly.csv`: tidy quarterly adjusted and not-adjusted values,
  with the original Census row labels.
- `us_census_ecommerce_metadata.json`: machine-readable provenance.

Regenerate everything with `python scripts/build_census_ecommerce_dataset.py`; the
script also runs the dataset validation. Experiment outputs must not be written into
these files.

### Methodology revision (April 2025 restatement)

The April 2025 benchmark restated the **entire** quarterly e-commerce history:
- 2017 NAICS;
- employer-only coverage aligned with the Annual Integrated Economic Survey
  (nonemployers removed);
- restated 2022 Annual Retail Trade Survey results.

The workbooks contain only this restated history, so no vintages are spliced. As a
check, the 2022 four-quarter sum here ($997,477M) equals the restated 2022 figure in
the [Census restatement summary](https://www.census.gov/retail/mrts/www/NAICS_Restatement_Summary.pdf)
(previously $1,012,636M, which is 1.5% lower after restatement). Year-over-year growth
shows no break around the restatement (2024: +7.6%, 2025: +5.4%). The only growth
value outside a -5%..35% review band is 2020 (+42.1%), the pandemic shift to online
retail. That is a genuine economic movement, not a methodology discontinuity.

### Revision status

In this vintage only 2Q 2025 (adjusted series only) and 2026 quarters carry
revised/preliminary flags. On 2026-09-28 Census noted that the 2Q 2026 report no
longer holds the most up-to-date estimates; revised figures are scheduled for the 3Q
2026 release on 2026-11-19. A later download may therefore differ slightly.

### Vintage caveat

The experiment uses the currently available/revised Census historical series. Values are therefore not necessarily identical to the information that would have been available to a forecaster at each historical origin.

This is **not** a real-time (vintage) backtest.

### News

Historical analyzed-news data sufficient for a comparable long-horizon experiment is unavailable; therefore the news-adjusted variant is excluded from the real-data results.

Indicators are not included in this dataset.

## Other contents

- `indicators/`: BEA goods-consumption experiment input with provenance, and the
  U.S. internet-users break assessment (excluded from the experiment).
- `raw/indicators/`: unmodified BEA and World Bank source files with `fetch_log.json`
  (URLs, retrieval times, SHA-256).
- `results/census_baseline/`, `results/census_indicators/`: stored experiment outputs
  shown on the Streamlit Evaluation page.
