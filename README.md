# Forecast_S

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![Version](https://img.shields.io/badge/version-2.0.0-2A9D8F)

A Streamlit market-forecasting framework that combines baseline time-series models, economic
indicators and GDELT/LLM news analysis, and evaluates forecast accuracy with leakage-controlled
walk-forward backtesting.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Repository Structure](#repository-structure)
7. [Configuration](#configuration)
8. [Data Requirements](#data-requirements)
9. [LLM Integration](#llm-integration)
10. [Development](#development)
11. [Troubleshooting](#troubleshooting)

---

## Overview

Forecast_S produces annual market forecasts at country, regional and global level, and measures
how accurate those forecasts are on data the models have never seen.

**Motivation.** Market forecasts are often adjusted with news and macroeconomic signals, but it is
rarely tested whether those adjustments improve accuracy. Forecast_S combines forecast generation
with an out-of-sample evaluation framework, so models and adjustments can be judged on measured
historical performance rather than assumed to help.

**How the parts fit together**

1. **Historical market data** is extracted from an SQL database (or demo data) for a market KPI.
2. A **baseline model** (3-yr CAGR, Damped ETS or Logistic Growth) is fitted on data up to a historical cutoff and projected forward.
3. **News analysis:** GDELT headlines for configured topics are classified by an LLM into categories with an estimated growth impact. They are turned into a decaying news adjustment.
4. **Indicator analysis:** economic indicators contribute a weighted adjustment based on their historical growth.
5. **Forecasting methods** apply this pipeline globally or per country and aggregate the results (Bottom-Up, Top-Down, Country-Specific, Global Only).
6. **Evaluation:** a walk-forward backtest refits the models at many historical origins, scores them against later actual values, and compares them with a naive benchmark. Adjustments are compared with each model's own unadjusted baseline. The evaluation does not assume that indicators or news improve accuracy.

```mermaid
flowchart TD
    A["Historical market data (SQL or demo)"] --> B["Data extraction and preparation"]
    N["GDELT news"] --> O["LLM headline analysis"]
    I["Economic indicators"] --> P["IndicatorAdjustment"]

    B --> C["BaselineModelFactory"]

    C --> D["3-yr CAGR"]
    C --> E["Damped ETS"]
    C --> F["Logistic Growth"]

    D --> G["Baseline forecast"]
    E --> G
    F --> G

    O --> H["NewsAdjustment"]

    G --> K["Forecasting methods"]
    H --> K
    P --> K

    K --> L["Bottom-Up / Top-Down / Country-Specific / Global Only"]
    L --> M["Streamlit dashboard"]

    M --> Q["CSV / Excel exports"]
    M --> R["Evaluation and walk-forward backtesting"]
    R --> S["MAPE / RMSE / MAE / Bias / Coverage"]
```

---

## Features

### Forecasting

- **Approaches:** Bottom-Up, Top-Down, Country-Specific and Global Only. Country results are aggregated into regions and `Worldwide` using the hierarchy in `config/aggregation.json`.
- **Forecast modes:**
  - *Forecast + News Adjustment* (classic): fitted baseline model plus optional news and indicator adjustments.
  - *Existing Forecast + News Adjustment*: the extracted market series is used as the baseline and only news adjustments are applied, within a chosen start/end year window (Top-Down is not available in this mode).
- **Baseline models** (`BaselineModelFactory`):
  - *3-yr CAGR*: compound annual growth over the last three periods, extrapolated from the last value.
  - *Damped ETS*: `statsmodels` exponential smoothing with an additive, damped trend.
  - *Logistic Growth*: S-curve fitted with `scipy`; falls back to 3-yr CAGR if the fit fails.

### News analysis (GDELT + LLM)

- **Retrieval:** GDELT headlines are retrieved per topic, globally or per country (ISO3 → FIPS mapping), over a 90-day window and rate-limited.
- **LLM classification:** each headline gets a category, a growth impact (%) and a short reason, constrained by per-category growth ranges.
- **Generation:** the LLM can generate market categories and GDELT search topics.
- **News adjustment:**
  - Category moving-average windows with recency weighting.
  - Temporal decay over the forecast horizon, with an LLM-calibrated decay rate.
- **Fallback:** countries with fewer than 5 analyzed headlines use global news at reduced confidence.

### Indicators

- **Inputs:** optional global and country indicators with configurable per-indicator weights.
- **Adjustment:** a bounded multiplicative adjustment from weighted year-over-year indicator growth, combined with news through a news/indicator weight split (default 70/30).

### Streamlit interface and exports

- **Workflow pages:** Data Extraction → Configuration → Forecasting → Insights → Export, plus a stored-results Evaluation page.
- **Insights:** global, regional and country charts, adjustment breakdowns and news analysis.
- **Exports:** the forecast table and analyzed news (CSV/Excel), and the stored evaluation result files.

### Evaluation and backtesting

- **Backtest design:**
  - Expanding-window walk-forward backtesting with a naive last-value benchmark on identical folds.
  - Separate 1-, 2- and 3-year horizon results plus an all-horizon pool.
  - Pooled and fold-mean metrics: MAPE, RMSE, MAE, bias, absolute bias, coverage and failure counts.
- **Temporal isolation:**
  - Training data, indicators (`year <= origin`) and historical news (`as_of` timestamp, GDELT `end_date`) are restricted to information available at each origin.
  - A cached historical headline-analysis helper never caches failed analyses.
- **Reporting:** paired comparisons against the benchmark and against each model's own baseline, and CSV/JSON export with configuration, provenance and input fingerprints.
- **Real-data experiment:** U.S. Census Bureau annual retail e-commerce sales, 2000–2025, with 19 forecast origins. The results are stored in the repository and shown offline on the Evaluation page.

Measured results (current-vintage data; descriptive, no significance testing):

- Damped ETS had the lowest pooled MAPE (11.48%). Logistic Growth had the lowest pooled RMSE and MAE. All three models beat the naive benchmark on MAPE and MAE at every horizon.
- The BEA goods-consumption indicator did not provide meaningful predictive improvement. Small MAPE/MAE changes were consistent with a near-constant ~1% upward bias correction.
- News adjustments were not evaluated empirically, because no suitable archived historical analyzed-news data was available. No claim is made that news improves accuracy.

See [CONTRIBUTION.md](CONTRIBUTION.md) for how the evaluation framework extends the original project.

---

## Prerequisites

| Requirement | Needed for |
|---|---|
| **Python 3.10+** (the Docker image uses 3.11; tests were last run on 3.12) | Required |
| **Git** and **pip** | Required |
| Packages in `requirements.txt` | Required |
| SQL database reachable via SQLAlchemy | Optional; without it the app uses random demo data |
| LLM provider (OpenAI-compatible API or local Ollama) | Optional; only for headline analysis, category/topic generation and decay calibration |
| Internet access to the GDELT API | Optional; only for news retrieval (no API key required) |

The Evaluation page and the stored evaluation results need no database, LLM or network access.

---

## Installation

Windows (PowerShell):

```powershell
git clone https://github.com/premkumarrs/Forecast_S.git
cd Forecast_S
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
git clone https://github.com/premkumarrs/Forecast_S.git
cd Forecast_S
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To use a database or an LLM, create `config/.env` from the template (see [Configuration](#configuration)):

```powershell
Copy-Item config\.env.example config\.env
```

---

## Quick Start

Run from the project root, because configuration paths such as `config/.env` are relative:

```powershell
streamlit run app.py
```

Open the URL printed by Streamlit (by default `http://localhost:8501`). The sidebar lists the
pages in workflow order:

| Page | Purpose |
|---|---|
| Home (`app.py`) | Overview and getting-started guide |
| Data Extraction | Choose the forecast mode, enter market/indicator KPI keys and years, extract data |
| Configuration | Market name, approach, baseline model, weights, categories, GDELT topics |
| Forecasting | Fetch and analyze news, then generate the forecast |
| Insights | Global, regional and country charts and breakdowns |
| Export | Download forecast and news tables, plus stored evaluation results |
| Evaluation | Stored out-of-sample evaluation results (works offline) |

**Without `DB_URL`:** if no database is configured, the Data Extraction page shows *"No database
connection configured. Using mock data for demo."* It then generates random demo data for eight
countries. This is useful for exploring the UI, but the demo data is unseeded and must not be read as
real market data.

---

## Repository Structure

```text
Forecast_S/
├── app.py                         # Streamlit entry point (home page)
├── requirements.txt               # Python dependencies (unpinned except openai>=1.56.0)
├── Dockerfile                     # Python 3.11 image with the dependencies installed
├── docker-compose.yml             # Dev/prod profiles running Streamlit on port 8501
├── README.md
├── CONTRIBUTION.md                # Original project vs. evaluation contribution
├── .gitignore
│
├── config/
│   ├── .env.example               # Environment variable template (copy to config/.env)
│   ├── settings.toml              # App settings; [gdelt] ssl_verify is read by the code
│   ├── category.json              # Default news categories and growth ranges
│   ├── categories/                # Generated categories/topics per market
│   ├── aggregation.json           # Region hierarchy for country -> region -> Worldwide
│   ├── exclusions.json            # Aggregate/sub-national names removed after extraction
│   └── flat-ui__data-Sun Aug 17 2025.csv   # ISO3 -> FIPS country codes for GDELT
│
├── pages/                         # Streamlit pages (sidebar order)
│   ├── 01_Data_Extraction.py
│   ├── 02_Configuration.py
│   ├── 03_Forecasting.py
│   ├── 04_Insights.py
│   ├── 05_Export.py
│   ├── 06_Evaluation.py           # Stored evaluation dashboard
│   ├── components/                # Charts, layouts, metrics, utilities, evaluation rendering
│   └── helpers/                   # Page logic: configuration, extraction, export, forecasting, insights
│
├── src/
│   ├── constants.py               # Forecast modes
│   ├── regions.py                 # Region mapping and country -> region aggregation
│   ├── compatibility.py
│   ├── config/security.py         # GDELT SSL verification setting
│   ├── data/                      # Country/ISO3 utilities and data helpers
│   ├── database/                  # SQLAlchemy connection (DB_URL) and query helpers
│   ├── forecasting/
│   │   ├── models/                # baseline_factory.py, cagr.py, ets.py, logistic.py
│   │   ├── methods/               # global_forecast.py, top_down.py, bottom_up.py, country_specific.py, existing_mode_helpers.py
│   │   ├── adjustments/           # News, indicator, temporal decay, fallback, unified adjustment
│   │   ├── base/                  # BaseForecaster, metrics, validators
│   │   ├── utils/                 # Aggregation and time-series helpers
│   │   ├── validation/            # Data/method validators
│   │   ├── evaluation/            # Walk-forward evaluation framework (see below)
│   │   ├── config.py
│   │   └── enhanced_fallback.py
│   ├── llm/
│   │   ├── analyst.py             # SimpleLLMAnalyst: headline analysis, category/topic generation
│   │   ├── providers/             # openai_provider.py, ollama_provider.py, provider_factory.py
│   │   ├── prompts/               # Analysis, category and decay-calibration prompts
│   │   ├── categories/            # Category management and storage
│   │   ├── calibrators/           # LLM decay-rate calibration
│   │   └── core/                  # Base LLM class and exceptions
│   ├── news/
│   │   ├── gdelt.py               # GDELT retrieval, rate limiting, ISO3 -> FIPS
│   │   ├── providers/gdelt/       # Alternative GDELT provider classes
│   │   └── utils/
│   ├── services/
│   │   ├── configuration/         # ConfigurationService (LLM generation with fallbacks)
│   │   ├── data/                  # DataExtractionService, SQL processor, demo-data generator
│   │   └── forecast/              # ForecastService, method router, news analysis service
│   ├── session/                   # Streamlit session state and serialization
│   ├── ui_components/             # Forecast charts and metric displays
│   ├── weights/
│   ├── utils/compatibility.py
│   └── migration/remove_multiplier_add_ma.py
│
├── ui/streamlit/components/calibration_card.py   # Decay-calibration display
│
├── scripts/                       # Evaluation dataset builds and experiments
│   ├── build_census_ecommerce_dataset.py
│   ├── run_census_baseline_experiment.py
│   ├── fetch_indicator_sources.py
│   ├── build_indicator_datasets.py
│   └── run_census_indicator_experiment.py
│
├── data/evaluation/               # Stored evaluation inputs and results
│   ├── us_census_ecommerce_annual.csv              # Target series (year, value)
│   ├── us_census_ecommerce_annual_provenance.csv
│   ├── us_census_ecommerce_quarterly.csv
│   ├── us_census_ecommerce_metadata.json
│   ├── indicators/                # BEA goods indicator input, provenance, internet-users assessment
│   ├── raw/                       # Official Census workbooks; raw/indicators/ BEA, World Bank, FRED files
│   └── results/
│       ├── census_baseline/       # Baseline experiment outputs
│       └── census_indicators/     # Indicator experiment outputs (spec_a_goods/) and run_log.json
│
└── tests/
    ├── test_backtest.py           # Folds, backtest, temporal isolation, indicator/news variants
    ├── test_experiment.py         # Comparative experiments, comparison, export
    ├── test_evaluation_artifacts.py   # Stored-result loading for the dashboard
    ├── test_forecasting_pipeline.py   # Legacy
    ├── test_headline_analysis.py      # Legacy
    └── test_smoke.py                  # Legacy import smoke tests
```

**Evaluation package** (`src/forecasting/evaluation/`):

| Module | Purpose |
|---|---|
| `splits.py` | `walk_forward_folds`: expanding-window folds over consecutive years |
| `backtest.py` | `run_backtest`: per-fold forecasts, naive benchmark, indicator/news variants, failure log |
| `news_history.py` | Historical `as_of` cutoff, headline-analysis cache, per-origin news fetching |
| `report.py` | `summarize_backtest`: grouped metrics |
| `experiment.py` | `run_comparative_experiment`, `ExperimentConfig`, `ExperimentResult` |
| `comparison.py` | Summaries, rankings, benchmark improvements, adjustment effects, comparison report |
| `export.py` | JSON/CSV export |
| `artifacts.py` | Read-only loading of stored results for the Evaluation page |

---

## Configuration

### Environment variables

Settings are read from `config/.env` (copy `config/.env.example`). The file is loaded relative to
the working directory and is ignored by Git.

| Variable | Purpose | Default in code |
|---|---|---|
| `DB_URL` | SQLAlchemy database URL, for example `mysql+pymysql://user:password@host:3306/db` | not set → demo data |
| `LLM_PROVIDER` | `openai` or `ollama` | `ollama` |
| `OPENAI_API_KEY` | API key for the `openai` provider | not set |
| `OPENAI_MODEL` | Model or Azure deployment name | `gpt-3.5-turbo` |
| `OPENAI_BASE_URL` | OpenAI-compatible or Azure OpenAI endpoint | not set (OpenAI default) |
| `OPENAI_API_VERSION`, `AZURE_OPENAI_ENDPOINT` | Azure OpenAI settings | not set |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama model | `llama2` |
| `LLM_MAX_CONCURRENT` | Parallel headline analyses | `10` |
| `LLM_REQUEST_DELAY` | Delay per headline request (seconds) | `0.05` |
| `NEWS_ARTICLE_THRESHOLD` | Article count used to flag low-coverage countries | `10` |
| `NEWS_HALF_LIFE_DAYS` | Recency half-life, used only when no categories are configured | `90` |
| `GDELT_SSL_VERIFY` | TLS certificate verification for GDELT requests | `true` |

`config/.env.example` also contains variables that the current code does not read
(`NEWS_API_ENABLED`, `LLM_DEBUG_LOGGING`, `DEFAULT_HIST_CUTOFF`, `DEFAULT_FORECAST_UNTIL`,
`DEFAULT_FORECAST_METHOD` and `DEBUG_MODE`).

### Database (optional)

`DB_URL` is passed directly to SQLAlchemy's `create_engine`. `pymysql` is included for MySQL URLs;
other databases need their own driver. If `DB_URL` is empty, or no engine can be created from it
(for example because of an invalid URL or a missing driver), the app falls back to demo data. If the
URL is valid but the database cannot be reached, extraction fails with an error instead.

### `config/settings.toml`

The only setting the code currently reads is `[gdelt] ssl_verify`. The other sections (`[app]`,
`[defaults]`, `[news]` and so on) are not loaded at runtime. Forecast settings are chosen on the
Configuration page instead.

**GDELT SSL verification.** Verification is on by default. `GDELT_SSL_VERIFY` takes precedence over
`[gdelt] ssl_verify`. When disabled, only requests to `*.gdeltproject.org` skip verification and a
warning is logged. The template sets `GDELT_SSL_VERIFY=false`; set it to `true` unless you are
troubleshooting certificate errors on a trusted network.

### Configuration page settings

- **Market and method:** market name, forecasting approach and baseline model.
- **Weights:** a news influence slider (default 70%; the indicator weight is the remainder) and per-indicator weights (0–1).
- **Categories:** name, description, minimum/maximum growth impact and a moving-average window (0–120 days; 0 = neutral). Defaults come from `config/category.json`. Generated categories and topics are saved to `config/categories/<market>/`.
- **GDELT topics:** short, news-relevant search terms; each topic is queried separately.

### LLM settings

See [LLM Integration](#llm-integration).

---

## Data Requirements

### Application input data

**SQL source.** Market and indicator values are extracted by KPI key
(`src/services/data/processor.py`). The query targets the original project's KPI schema
(`kpisValues`, `kpis` and `geos` tables); adapt that module to use a different database schema. The
extracted data is converted into these frames:

| Frame | Columns |
|---|---|
| Global market series | `year`, `value` |
| Country market series | `country`, `iso3`, `year`, `value` |
| Global indicators | `year`, `indicator_key`, `indicator_name`, `value` |
| Country indicators | `country`, `iso3`, `year`, `indicator_key`, `indicator_name`, `value` |

Data rules:

- **Rows:** one value per country and year; duplicates are resolved by keeping the last row. Values must be numeric.
- **History length:** baseline models need at least 3 years up to the cutoff. Missing years are not interpolated.
- **Exclusions:** names in `config/exclusions.json` (aggregates and sub-national regions) are removed after extraction.

**Demo data.** Without a database, `MockDataGenerator` creates a random global series and series for
eight countries (USA, DEU, JPN, GBR, FRA, CHN, IND, BRA), plus indicator series. It is for UI
demonstration only.

**News data (GDELT).**
- **Inputs:** a list of topics and, for country-level methods, valid ISO3 codes. Codes are converted to GDELT's FIPS codes through `config/flat-ui__data-Sun Aug 17 2025.csv`.
- **Coverage:** the GDELT DOC API covers roughly the last three months, which matches the default 90-day window.
- **Rate limit:** requests are throttled to one every 5–6 seconds.

**Indicator data.** Indicators are optional. Each row needs `year`, `indicator_key` and `value` (plus
`country`/`iso3` for country indicators). Without indicator rows or weights, the indicator adjustment
is zero.

### Stored evaluation datasets (included in the repository)

These files are inputs and outputs of the evaluation experiments, separate from the application's input data:

| Path | Content |
|---|---|
| `data/evaluation/us_census_ecommerce_annual.csv` | Census annual retail e-commerce sales, 2000–2025 (`year`, `value`, millions of current US$). Each value is the sum of the four not-seasonally-adjusted quarters; this is current-vintage data. |
| `data/evaluation/indicators/us_pce_goods_annual.csv` | BEA personal consumption expenditures on goods (`year`, `indicator_key`, `value`) |
| `data/evaluation/raw/` | Unmodified official Census, BEA, World Bank and FRED source files; `raw/indicators/fetch_log.json` records URLs and SHA-256 hashes |
| `data/evaluation/results/` | Stored experiment outputs (JSON/CSV) read by the Evaluation page |

The evaluation target must have unique, consecutive years with finite numeric values. The Census and
BEA series are revised (current-vintage) data, so the evaluation is a historical walk-forward test,
not a real-time vintage backtest.

---

## LLM Integration

### Supported providers

The provider is selected with `LLM_PROVIDER` (`src/llm/providers/provider_factory.py`):

| Provider | Covers | Configuration |
|---|---|---|
| `openai` | OpenAI, Azure OpenAI and any OpenAI-compatible endpoint | `OPENAI_API_KEY` (required), `OPENAI_MODEL`, optional `OPENAI_BASE_URL`; Azure also uses `OPENAI_API_VERSION` / `AZURE_OPENAI_ENDPOINT` |
| `ollama` (default) | Local Ollama server | `OLLAMA_BASE_URL`, `OLLAMA_MODEL`; no API key |

There are no dedicated integrations for Anthropic, Google, Perplexity or OpenRouter. A service that
offers an OpenAI-compatible API can be reached only through `OPENAI_BASE_URL` with
`LLM_PROVIDER=openai`. GDELT retrieval itself needs no key.

### What the LLM contributes

- **Headline analysis** (`SimpleLLMAnalyst.analyze_headlines`).
  - Each headline is classified with a global or a country-specific prompt that lists every category and its allowed growth range.
  - Headlines are processed concurrently (`LLM_MAX_CONCURRENT`, default 10 threads) with a short delay per request (`LLM_REQUEST_DELAY`).
  - The model must return:

    ```json
    {
      "category": "Infrastructure Investment",
      "growth_rate": 12.0,
      "reason": "Large data-center investment expands regional capacity"
    }
    ```

  - The app stores these fields with the headline's `title` and `date`, plus a derived `relevant` flag (1 when `growth_rate` is non-zero). These values feed the category-based news adjustment.
- **Category and topic generation:** market-specific news categories (with growth ranges) and GDELT search topics. Topic generation falls back to template topics if the LLM fails; category generation can return an empty list, in which case the defaults in `config/category.json` remain usable.
- **Decay calibration:** a market-specific decay rate for news impact over the forecast horizon (0.10–0.90). It is shrunk toward a prior of 0.65 before use.

### Failure behavior

LLM errors (missing key, unreachable server, invalid JSON) do not stop the app. The affected headline
is recorded as `Neutral/Noise` with 0% impact, so news adjustments become small or zero. There are
no automatic retries. The production analyst does not cache results; the evaluation module provides
a separate headline-analysis cache for historical experiments.

---

## Development

### Running tests

pytest is not listed in `requirements.txt`; install it first:

```powershell
pip install pytest
```

Full suite. The flag is needed because one legacy test file fails at collection:

```powershell
pytest -q --continue-on-collection-errors
```

Evaluation tests only (136 tests, all passing):

```powershell
pytest -q tests/test_backtest.py tests/test_experiment.py tests/test_evaluation_artifacts.py
```

**Current full-suite result:** 142 passed, 3 failed, 1 error. The failures are in legacy tests from the
original codebase and predate the evaluation work:

| Test | Failure |
|---|---|
| `tests/test_forecasting_pipeline.py` | Collection error: imports `apply_indicator_adjustment`, which `src.forecasting` does not export |
| `tests/test_smoke.py::test_app_imports`, `::test_page_imports` | Importing `pages/01_Data_Extraction.py` raises `'str' object has no attribute 'value'` |
| `tests/test_headline_analysis.py::test_analyze_headlines` | Mocks an older response schema (`KeyError: 'magnitude'`) |

### Reproducing the evaluation results

Run from the project root. Each step writes to `data/evaluation/`:

```powershell
python scripts/build_census_ecommerce_dataset.py   # Census target from the stored workbooks (offline)
python scripts/run_census_baseline_experiment.py   # -> data/evaluation/results/census_baseline/
python scripts/fetch_indicator_sources.py          # optional; re-downloads BEA/World Bank/FRED sources (network)
python scripts/build_indicator_datasets.py         # -> data/evaluation/indicators/
python scripts/run_census_indicator_experiment.py  # -> data/evaluation/results/census_indicators/
```

Re-running the experiments reproduces the stored metrics; only the `run_timestamp` metadata changes.

### Checks

No formatter, linter or type checker is configured in the repository. Before committing, run the
tests and check the diff:

```powershell
git diff --check
```

---

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| "No database connection configured. Using mock data for demo." | `DB_URL` is not set, or no engine could be created from it. Set `DB_URL` in `config/.env` and install the matching SQLAlchemy driver. |
| Data extraction fails with a query error | `DB_URL` is set but the database is unreachable, or it does not contain the expected KPI tables. Check connectivity, credentials and schema. |
| All headlines are `Neutral/Noise` with 0% impact | LLM calls are failing. Check `LLM_PROVIDER`, `OPENAI_API_KEY` / `OPENAI_BASE_URL`, or that the Ollama server is running at `OLLAMA_BASE_URL`, and check the logs. |
| LLM rate-limit errors | Lower `LLM_MAX_CONCURRENT` or raise `LLM_REQUEST_DELAY`. |
| `ModuleNotFoundError` on startup | Activate the virtual environment and run `pip install -r requirements.txt`. |
| `config/.env` or category files are not found | Start Streamlit from the project root. |
| `streamlit` is not recognised | The virtual environment is not active; activate it or run `python -m streamlit run app.py`. |
| GDELT requests fail with certificate errors | Check network access to `api.gdeltproject.org`. On a trusted network, temporarily set `GDELT_SSL_VERIFY=false`. |
| No news returned, or news fetching is slow | Use broader topics; GDELT covers only about the last three months. Requests are throttled to one every 5–6 seconds, so many topics take time. |
| A country receives no news | Check that it has a valid ISO3 code with a FIPS mapping. Countries with fewer than 5 analyzed headlines use global news at reduced confidence. |
| Top-Down returns "Forecast execution failed" | Known defect in classic mode: `src/forecasting/methods/top_down.py` uses an undefined variable `share` (the loop variable is `_share`). Use Bottom-Up, Country-Specific or Global-level until it is fixed. |
| The Evaluation page reports missing artifacts | Files in `data/evaluation/results/` are missing; regenerate them with the evaluation scripts. The page never generates results itself. |
| `pytest` stops with a collection error, or reports 3 failures | These are the known legacy test failures; use `--continue-on-collection-errors`. The evaluation tests are unaffected. |
