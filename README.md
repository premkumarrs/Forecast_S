# Forecast_S

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![Version](https://img.shields.io/badge/version-2.0.0-2A9D8F)

A Streamlit market-forecasting application that combines baseline time-series models, economic
indicators and GDELT/LLM news analysis, extended with a leakage-controlled walk-forward evaluation
framework that measures whether those models and adjustments actually work out of sample.

**Contents:** [Overview](#overview) · [Features](#features) · [Prerequisites](#prerequisites) ·
[Installation](#installation) · [Quick Start](#quick-start) ·
[Repository Structure](#repository-structure) · [Configuration](#configuration) ·
[Data Requirements](#data-requirements) · [LLM Integration](#llm-integration) ·
[Development](#development) · [Troubleshooting](#troubleshooting)

---

## Overview

Forecast_S produces annual market forecasts at country, regional and global level:

1. **Historical market data** for a market KPI (global and per-country series, plus optional indicators) is extracted from an SQL database, or generated as demo data when no database is configured.
2. A **baseline model** (3-yr CAGR, Damped ETS or Logistic Growth) is fitted on data up to a historical cutoff year and projected to the forecast horizon.
3. **GDELT news** is retrieved for configured topics, globally and per country. An **LLM** classifies each headline into a market category with an estimated growth impact and a reason.
4. **Forecast adjustments:** the classified news becomes a recency-weighted, time-decaying news adjustment. **Economic indicators** contribute a bounded, weighted adjustment from their year-over-year growth.
5. **Forecasting methods** (Bottom-Up, Top-Down, Country-Specific, Global Only) apply this pipeline and aggregate countries into regions and a worldwide total.
6. The **Streamlit application** guides the workflow, visualizes results and **exports** forecasts and analyzed news to CSV/Excel.

**Why the evaluation framework exists.** News and macroeconomic adjustments are easy to add to a
forecast and hard to justify. Forecast_S therefore does not stop at generating forecasts: it tests
the models and adjustments on historical observations they have never seen. The evaluation is built
on four principles:

- **No assumed benefit.** Neither news nor indicators are assumed to improve accuracy; both are measured.
- **Benchmark comparison.** Every model is compared with a naive last-value forecast on identical folds.
- **Own-baseline comparison.** An adjusted forecast is compared with the *same* model's unadjusted forecast, so the effect of the adjustment is isolated from the choice of model.
- **Temporal isolation.** At every historical forecast origin, the models only see data that would have been available at that point.

```mermaid
flowchart TD
    subgraph APP["Forecasting application"]
        A["Historical market data (SQL or demo)"] --> B["Data extraction"]
        B --> C["Baseline models: 3-yr CAGR / Damped ETS / Logistic Growth"]
        C --> D["Baseline forecast"]
        N["GDELT news"] --> O["LLM news analysis"] --> P["News adjustment"]
        I["Economic indicators"] --> Q["Indicator adjustment"]
        D --> K["Baseline + adjustments"]
        P --> K
        Q --> K
        K --> L["Forecasting methods: Bottom-Up / Top-Down / Country-Specific / Global Only"]
        L --> M["Streamlit dashboard / CSV and Excel exports"]
    end

    subgraph EVAL["Empirical evaluation"]
        H["Historical data"] --> W["Walk-forward evaluation (expanding window)"]
        W --> X["Naive benchmark + forecasting models"]
        X --> Y["MAPE / RMSE / MAE / Bias / Coverage"]
    end
```

---

## Features

### Forecasting

**Forecasting approaches** (selected on the Configuration page):

| Approach | How it works |
|---|---|
| **Global Only** (shown as *Global-level*) | Forecasts the global series only: baseline model plus global news and global indicator adjustments. |
| **Bottom-Up** | Forecasts each country with its own baseline and country news/indicator adjustments, then sums countries into regions and `Worldwide`. |
| **Top-Down** | Forecasts the global series and allocates it to countries by historical share, then applies country adjustments. Classic mode currently has a known defect (see [Troubleshooting](#troubleshooting)). |
| **Country-Specific** | Forecasts a user-selected set of countries individually. |

Regional aggregation uses the hierarchy in `config/aggregation.json`.

**Forecast modes** (chosen on the Data Extraction page, or with the URL parameter `?mode=existing_forecast_news`):

- **Forecast + News Adjustment** (default): a fitted baseline model with optional news and indicator adjustments.
- **Existing Forecast + News Adjustment:** the extracted market series is used as the baseline and only news adjustments are applied (news weight 100%) between a chosen adjustment start and end year. Top-Down is not available in this mode.

**Baseline models** (`src/forecasting/models/`, created through `BaselineModelFactory`):

| Model | Method |
|---|---|
| **3-yr CAGR** | Compound annual growth rate over the last three periods, extrapolated from the last observed value. |
| **Damped ETS** | `statsmodels` exponential smoothing with an additive, damped trend. |
| **Logistic Growth** | S-curve fitted with `scipy.optimize.curve_fit`; falls back to 3-yr CAGR if the fit fails. |
| **Naive (last value)** | Evaluation benchmark only: repeats the last observed value. It is not offered as a forecasting model in the application. |

### News Analysis

- **Retrieval:** GDELT DOC API via `gdeltdoc` (`src/news/gdelt.py`). Each topic is queried separately, globally and, for country-level methods, per country. Country ISO3 codes are mapped to GDELT FIPS codes through `config/flat-ui__data-Sun Aug 17 2025.csv`. The default window is the last 90 days, up to 250 articles per query.
- **Rate limiting:** a global limiter spaces GDELT requests 5–6 seconds apart.
- **LLM headline analysis:** each headline receives a `category`, a `growth_rate` (% impact, constrained by the category's allowed range) and a `reason`. Headlines are analyzed concurrently (`LLM_MAX_CONCURRENT` threads, `LLM_REQUEST_DELAY` between requests).
- **Category and topic generation:** the LLM can generate market-specific categories (with growth ranges) and GDELT search topics.
- **News adjustment** (`src/forecasting/adjustments/`):
  - Each category has a moving-average window (0–120 days; 0 = neutral). Headlines inside the window are recency-weighted with a half-life of `max(window / 3, 7)` days.
  - The resulting category signal is clamped to ±30%.
  - Over the forecast horizon the impact decays as `0.5^(i / (T × rate))`, where the decay `rate` is calibrated per market by the LLM (range 0.10–0.90, shrunk toward a prior of 0.65).
- **Country fallback:** a country with fewer than 5 analyzed headlines uses global news at reduced confidence (default multiplier 0.5).

### Indicators

- **Optional inputs:** global and country indicator series, each with a user-defined weight (0–1).
- **Signal:** the weighted mean of year-over-year indicator growth over the training period, clamped to ±30%.
- **Application:** a multiplicative adjustment, `value × (1 + signal × indicator_weight)`, applied to every forecast year.
- **News/indicator split:** a single slider sets the news weight (default 70%); the indicator weight is the remainder (default 30%).

The indicator mechanism is a heuristic growth adjustment, not a causal or econometric model.

### Streamlit Application

Pages in sidebar order:

| Page | File | Purpose |
|---|---|---|
| Forecasting App (home) | `app.py` | Overview and getting-started guide |
| Data Extraction | `pages/01_Data_Extraction.py` | Choose the forecast mode, enter market/indicator KPI keys and years, extract data (or demo data) |
| Configuration | `pages/02_Configuration.py` | Market name, approach, baseline model, news/indicator weights, categories and GDELT topics, LLM generation |
| Forecasting | `pages/03_Forecasting.py` | Fetch and analyze news, calibrate decay, generate the forecast |
| Insights | `pages/04_Insights.py` | Global, regional and country charts, adjustment breakdowns and analyzed news |
| Export | `pages/05_Export.py` | Download forecast and news tables, plus stored evaluation results |
| Model Evaluation | `pages/06_Evaluation.py` | Read-only dashboard of the stored out-of-sample evaluation results; works offline |

### Exports

- **Forecasts:** the forecast table as CSV or Excel.
- **News:** the analyzed headlines (category, growth impact, reason) as CSV or Excel.
- **Evaluation:** the stored experiment files from `data/evaluation/results/`, downloadable from the Export and Model Evaluation pages.

### Evaluation and Backtesting

The evaluation package (`src/forecasting/evaluation/`) runs a reproducible out-of-sample test of the
forecasting models:

- **Expanding-window walk-forward folds.** For each historical forecast origin *Y*, the models are trained on all years ≤ *Y* and forecast *Y*+1 … *Y*+3. Only folds with a complete test horizon are used, and all models are evaluated on exactly the same origins and target years.
- **Horizons.** Results are reported separately for 1-, 2- and 3-year horizons and for an all-horizon pool.
- **Models.** Naive last-value benchmark, 3-yr CAGR, Damped ETS and Logistic Growth, all produced by the same production model code.
- **Metrics.** MAPE, RMSE, MAE, bias (mean of forecast − actual; negative = under-forecast), absolute bias and coverage (share of attempted forecasts that produced a valid value).
- **Failure tracking.** Fitting errors and invalid forecasts are recorded per fold instead of being dropped silently; coverage reflects them.
- **Aggregation.** *Pooled* metrics use every valid forecast; *fold-mean* metrics average per-fold metrics.
- **Paired comparisons.** Improvements against the naive benchmark, and adjusted-vs-baseline effects for the same model, are computed on identical (origin, target year) pairs, with win/loss counts. No composite score and no significance tests are used.
- **Exports and provenance.** Results are exported to JSON and CSV (summaries, improvements, records, failures) together with the full configuration, dataset description, run timestamp and SHA-256 input fingerprints.

### Leakage Control

- **Target data:** each fold's training series is cut off at the origin year.
- **Indicators:** only indicator rows with `year <= origin` are passed to the adjustment.
- **Historical news:** news is filtered by an `as_of` timestamp at the origin, and GDELT retrieval uses a matching `end_date`.
- **LLM cache:** the historical headline-analysis cache stores only successful analyses; failed LLM calls are never cached as valid results.
- **Remaining limitations:**
  - *Current-vintage data.* The Census and BEA series are the latest revised values, not the figures a forecaster had at each origin, so the evaluation is a historical walk-forward test rather than a real-time vintage backtest.
  - *LLM hindsight.* Even with correctly cut-off headlines, a modern LLM may already know how later events unfolded, which can bias any historical news evaluation.

### Real-Data Evaluation

Dataset: **U.S. Census Bureau annual retail e-commerce sales**, 2000–2025 (26 years; each year is the
sum of four not-seasonally-adjusted quarters, millions of current US$). Design: minimum 5 training
years, **19 forecast origins** (2004–2022), horizons of 1, 2 and 3 years, 100% coverage for all
models. Results are stored in `data/evaluation/results/census_baseline/`.

Pooled results, all horizons (57 forecasts per model):

| Model | MAPE | RMSE | MAE | Bias |
|---|---:|---:|---:|---:|
| Damped ETS | **11.48%** | 111,072 | 64,054 | −6,995 |
| Logistic Growth | 13.02% | **93,546** | **62,109** | −23,405 |
| 3-yr CAGR | 13.51% | 148,744 | 77,939 | +33,741 |
| Naive (last value) | 23.36% | 154,231 | 111,167 | −111,167 |

Pooled MAPE by horizon:

| Model | 1 year | 2 years | 3 years |
|---|---:|---:|---:|
| Damped ETS | 5.30% | 11.92% | 17.23% |
| Logistic Growth | 7.19% | 12.72% | 19.16% |
| 3-yr CAGR | 6.20% | 13.41% | 20.93% |
| Naive (last value) | 13.21% | 23.96% | 32.90% |

Interpretation (descriptive, for this single series):

- Damped ETS had the lowest pooled MAPE; Logistic Growth had the lowest pooled RMSE and MAE.
- All three trend models outperformed the naive benchmark on MAPE at every horizon.
- Errors grow with the horizon; three-year forecasts are substantially harder.
- Regime changes dominate the errors: the 2020–2022 pandemic jump and its aftermath account for a large share of the trend models' error.
- These results do not show that any model is universally better; they describe one market series.

### Indicator Evaluation

The indicator adjustment was tested with **BEA Personal Consumption Expenditures: Goods** (indicator
weight 1.0, overall indicator weight 0.3, fixed before the run). Results are stored in
`data/evaluation/results/census_indicators/spec_a_goods/`.

Effect against each model's own baseline (pooled, all horizons):

| Model | MAPE | RMSE | MAE |
|---|---|---|---|
| 3-yr CAGR | 13.51% → 14.04% | 148,744 → 153,706 | 77,939 → 80,326 |
| Damped ETS | 11.48% → 11.27% | 111,072 → 113,054 | 64,054 → 64,384 |
| Logistic Growth | 13.02% → 12.86% | 93,546 → 94,157 | 62,109 → 61,384 |

**Conclusion:** under the existing adjustment formula, the tested indicator did not provide
meaningful or consistent predictive improvement. The adjustment raised forecasts by about 1% at every
origin (mean factor 1.011), so the small improvements for the models that under-forecast are
consistent with a near-constant upward bias correction rather than additional predictive information.
This does not show that indicators are useless in general.

A second candidate, *Individuals using the Internet (% of population)* from the World Bank, was
inspected and excluded before any adjusted result was computed: the historical series has
documented source changes and implausible year-to-year breaks
(`data/evaluation/indicators/us_internet_users_assessment.json`).

### News Evaluation Limitation

The infrastructure for historical news evaluation exists: `as_of` cutoffs, per-origin GDELT
retrieval, a cached headline analyzer and a news variant in the backtest, all covered by tests.
However, the real-data experiment does **not** establish whether news improves forecasting accuracy,
because no suitable archive of historical analyzed news covering 2004–2022 was available. No claim is
made that news adjustments improve accuracy.

See [CONTRIBUTION.md](CONTRIBUTION.md) for how the evaluation framework extends the original project.

---

## Prerequisites

| Requirement | Needed for |
|---|---|
| **Python 3.10+** (the Docker image uses 3.11; tests were last run on 3.12) | Required |
| **Git** and **pip** | Required |
| Packages in `requirements.txt` | Required |
| SQL database reachable via SQLAlchemy | Optional; without it the app uses demo data |
| LLM provider: OpenAI or an OpenAI-compatible endpoint (including Azure OpenAI), or a local Ollama server | Optional; only for headline analysis, category/topic generation and decay calibration |
| Internet access to the GDELT API | Optional; only for news retrieval (no API key required) |

The Model Evaluation page and the stored evaluation results need no database, LLM or network access.

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

To use a database or an LLM, create `config/.env` from the template and fill in the values you need
(see [Configuration](#configuration)):

```powershell
Copy-Item config\.env.example config\.env    # Windows
```

```bash
cp config/.env.example config/.env           # Linux/macOS
```

`config/.env` is ignored by Git. Never commit real credentials.

---

## Quick Start

Run from the project root, because configuration paths such as `config/.env` are relative:

```powershell
streamlit run app.py
```

Open the URL printed by Streamlit (by default `http://localhost:8501`), then follow the pages in order:

1. **Data Extraction:** choose the forecast mode, enter the market KPI key, optional indicator KPI keys, the historical cutoff and the forecast end year, and extract.
2. **Configuration:** set the market name, approach, baseline model, weights, categories and GDELT topics (optionally generated by the LLM).
3. **Forecasting:** fetch and analyze news, then generate the forecast.
4. **Insights** and **Export:** review and download the results.

**Database-backed mode.** With a valid `DB_URL`, market and indicator data are queried from the database.

**Demo mode.** Without `DB_URL`, the Data Extraction page shows *"No database connection configured.
Using mock data for demo."* and generates random demo data for eight countries. This is useful for
exploring the UI; it is not real market data.

**Offline evaluation.** The **Model Evaluation** page reads the stored results in
`data/evaluation/results/` and works without a database, LLM or network access.

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
│   ├── settings.toml              # App settings; only [gdelt] ssl_verify is read by the code
│   ├── category.json              # Default news categories and growth ranges
│   ├── categories/                # Generated categories/topics per market
│   ├── aggregation.json           # Region hierarchy: country -> region -> Worldwide
│   ├── exclusions.json            # Aggregate/sub-national names removed after extraction
│   └── flat-ui__data-Sun Aug 17 2025.csv   # ISO3 -> FIPS country codes for GDELT
│
├── pages/                         # Streamlit pages (sidebar order)
│   ├── 01_Data_Extraction.py
│   ├── 02_Configuration.py
│   ├── 03_Forecasting.py
│   ├── 04_Insights.py
│   ├── 05_Export.py
│   ├── 06_Evaluation.py           # Model Evaluation dashboard (stored results)
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
│   │   ├── adjustments/           # News, indicator, temporal decay, fallback and unified adjustment
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
│   ├── us_census_ecommerce_annual_provenance.csv   # Quarterly components of each annual value
│   ├── us_census_ecommerce_quarterly.csv
│   ├── us_census_ecommerce_metadata.json
│   ├── indicators/                # BEA goods indicator, provenance, internet-users assessment
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
| `news_history.py` | `as_of` news cutoff, cached historical headline analysis, per-origin GDELT retrieval |
| `report.py` | `summarize_backtest`: grouped metrics |
| `experiment.py` | `run_comparative_experiment`, `ExperimentConfig`, `ExperimentResult` |
| `comparison.py` | Summaries, rankings, benchmark improvements, adjustment effects, comparison report |
| `export.py` | JSON/CSV export of experiment results |
| `artifacts.py` | Read-only loading of stored results for the Model Evaluation page |

**Evaluation scripts** (`scripts/`):

| Script | Purpose |
|---|---|
| `build_census_ecommerce_dataset.py` | Builds the annual Census target, provenance and metadata from the stored official workbooks (offline) |
| `run_census_baseline_experiment.py` | Runs the baseline walk-forward experiment and writes `results/census_baseline/` |
| `fetch_indicator_sources.py` | Downloads the BEA, World Bank and FRED source files into `raw/indicators/` and logs URLs and SHA-256 hashes (network) |
| `build_indicator_datasets.py` | Builds the BEA goods indicator and the internet-users inspection/assessment |
| `run_census_indicator_experiment.py` | Runs the indicator experiment and writes `results/census_indicators/` |

---

## Configuration

### Environment variables

Settings are read from `config/.env`, loaded relative to the working directory.

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

The template's own values differ from the code defaults (for example `LLM_PROVIDER=openai`,
`OLLAMA_MODEL=llama3`, `LLM_MAX_CONCURRENT=5`, `LLM_REQUEST_DELAY=0.1`, `GDELT_SSL_VERIFY=false`).
Its Azure example endpoint is a placeholder; replace it with your own.

`config/.env.example` also contains variables that the current code does **not** read:
`NEWS_API_ENABLED`, `LLM_DEBUG_LOGGING`, `DEFAULT_HIST_CUTOFF`, `DEFAULT_FORECAST_UNTIL`,
`DEFAULT_FORECAST_METHOD` and `DEBUG_MODE` (and the commented-out `NEWS_API_KEY` / `NEWS_API_URL`).
Setting them has no effect.

### Database (optional)

`DB_URL` is passed directly to SQLAlchemy's `create_engine`. `pymysql` is included for MySQL URLs;
other databases need their own driver.

- `DB_URL` empty, or no engine can be created from it (invalid URL, missing driver) → the app uses demo data.
- `DB_URL` valid but the database unreachable, or missing the expected tables → extraction fails with an error. It does **not** silently fall back to demo data.

### `config/settings.toml`

The only setting the code reads is `[gdelt] ssl_verify`. The other sections (`[app]`, `[defaults]`,
`[news]` and so on) are not loaded at runtime; forecast settings are chosen on the Configuration page.

**GDELT SSL verification** (`src/config/security.py`): verification is on by default.
`GDELT_SSL_VERIFY` takes precedence over `[gdelt] ssl_verify`. When disabled, only requests to
`*.gdeltproject.org` skip verification and a warning is logged. Because the template sets
`GDELT_SSL_VERIFY=false`, change it to `true` unless you are troubleshooting certificate errors on a
trusted network.

### Configuration page settings

- **Market and method:** market name, forecasting approach and baseline model.
- **Weights:** news influence slider (default 70%; the indicator weight is the remainder) and per-indicator weights (0–1).
- **Categories:** name, description, minimum/maximum growth impact and moving-average window (0–120 days; 0 = neutral). Defaults come from `config/category.json`; generated categories and topics are saved under `config/categories/<market>/`.
- **GDELT topics:** short, news-relevant search terms; each topic is queried separately.

---

## Data Requirements

### Application input data

**SQL source.** Market and indicator values are extracted by KPI key
(`src/services/data/processor.py`). The query targets the original project's KPI schema (`kpisValues`,
`kpis` and `geos` tables); adapt that module for a different schema. The extracted data is converted
into these frames:

| Frame | Columns |
|---|---|
| Global market series | `year`, `value` |
| Country market series | `country`, `iso3`, `year`, `value` |
| Global indicators | `year`, `indicator_key`, `indicator_name`, `value` |
| Country indicators | `country`, `iso3`, `year`, `indicator_key`, `indicator_name`, `value` |

Data rules:

- **Years and values:** one numeric value per country and year; duplicates are resolved by keeping the last row.
- **ISO3:** country rows need a valid ISO3 code, used for regional aggregation and for mapping to GDELT FIPS codes.
- **History length:** baseline models need at least 3 years up to the historical cutoff.
- **Missing years:** not interpolated; models are fitted on the years present.
- **Exclusions:** names in `config/exclusions.json` (aggregates and sub-national regions) are removed after extraction.
- **Indicators:** optional. Each row needs `year`, `indicator_key` and `value` (plus `country`/`iso3` for country indicators). Without indicator rows or weights, the indicator adjustment is zero.

**Demo data.** Without a database, `MockDataGenerator` creates a random global series, series for
eight countries (USA, DEU, JPN, GBR, FRA, CHN, IND, BRA) and indicator series. It is unseeded and
intended only for UI testing.

**News data (GDELT).** Requires topics and, for country-level methods, valid ISO3 codes. The GDELT
DOC API covers roughly the last three months, which matches the default 90-day window.

### Stored evaluation datasets

These files are inputs and outputs of the evaluation experiments and are separate from the
application's input data:

| Path | Content |
|---|---|
| `data/evaluation/us_census_ecommerce_annual.csv` | Census annual retail e-commerce sales, 2000–2025 (`year`, `value`, millions of current US$); each value is the sum of the four not-seasonally-adjusted quarters |
| `data/evaluation/us_census_ecommerce_quarterly.csv` | The quarterly source series |
| `data/evaluation/us_census_ecommerce_annual_provenance.csv` | The quarterly components behind each annual value |
| `data/evaluation/us_census_ecommerce_metadata.json` | Source, vintage, units and validation checks |
| `data/evaluation/indicators/` | BEA PCE goods indicator (`us_pce_goods_annual.csv`) with provenance, and the internet-users inspection and assessment |
| `data/evaluation/raw/` | Unmodified official Census workbooks; `raw/indicators/` holds BEA, World Bank and FRED files, with `fetch_log.json` recording URLs and SHA-256 hashes |
| `data/evaluation/results/` | Stored experiment outputs (JSON/CSV) read by the Model Evaluation page |

The evaluation target must have unique, consecutive years with finite numeric values. Both the Census
and BEA series are current-vintage (revised) data; see [Leakage Control](#leakage-control).

---

## LLM Integration

### Architecture

| Component | Location | Role |
|---|---|---|
| Provider factory | `src/llm/providers/provider_factory.py` | Selects the provider from `LLM_PROVIDER` and reads its settings |
| Providers | `src/llm/providers/openai_provider.py`, `ollama_provider.py` | Chat-completion calls to OpenAI-compatible APIs or Ollama |
| Analyst | `src/llm/analyst.py` | `SimpleLLMAnalyst`: headline analysis, category and topic generation |
| Prompts | `src/llm/prompts/` | Global and country headline prompts, category generation, decay calibration |
| Categories | `src/llm/categories/` | Loading and saving default and market-specific categories |
| Calibration | `src/llm/calibrators/` | Market-specific decay-rate calibration |

### Supported providers

| Provider | Covers | Configuration |
|---|---|---|
| `openai` | OpenAI, Azure OpenAI and any OpenAI-compatible endpoint | `OPENAI_API_KEY` (required), `OPENAI_MODEL`, optional `OPENAI_BASE_URL`; Azure also uses `OPENAI_API_VERSION` / `AZURE_OPENAI_ENDPOINT` |
| `ollama` (default) | Local Ollama server | `OLLAMA_BASE_URL`, `OLLAMA_MODEL`; no API key |

There are no dedicated integrations for Anthropic, Google, Perplexity or OpenRouter. A service that
offers an OpenAI-compatible API can be used only through `OPENAI_BASE_URL` with `LLM_PROVIDER=openai`.

### What the LLM does

- **Headline classification:** each headline is sent with a global or a country-specific prompt listing every category and its allowed growth range. The model must return JSON:

  ```json
  {
    "category": "Infrastructure Investment",
    "growth_rate": 12.0,
    "reason": "Large data-center investment expands regional capacity"
  }
  ```

  The app stores these fields with the headline's `title` and `date`, plus a derived `relevant` flag (1 when `growth_rate` is non-zero).
- **Category generation:** market-specific categories with descriptions and growth ranges. If generation returns nothing, the defaults from `config/category.json` remain available.
- **Topic generation:** GDELT search topics for the market, with template topics as a fallback.
- **Decay calibration:** a decay rate (0.10–0.90) for how quickly news impact fades over the forecast horizon, shrunk toward a prior of 0.65.

### Failure behavior

LLM errors (missing key, unreachable server, invalid JSON) do not stop the app. The affected headline
is recorded as `Neutral/Noise` with 0% impact, so news adjustments become small or zero. There are no
automatic retries. The production analyst does not cache results; the evaluation package has a
separate cache that stores only successful analyses.

### Historical evaluation caveat

When the LLM is used on historical headlines, retrieval can be cut off correctly at the forecast
origin, but the model itself may know about later events from its training data. Any historical
news evaluation with a modern LLM is therefore subject to hindsight bias.

---

## Development

### Running tests

pytest is not listed in `requirements.txt`; install it in the virtual environment first:

```powershell
pip install pytest
```

Evaluation tests (136 tests, all passing):

```powershell
python -m pytest -q tests/test_backtest.py tests/test_experiment.py tests/test_evaluation_artifacts.py
```

Full suite:

```powershell
python -m pytest -q
```

Plain `python -m pytest -q` stops at the legacy collection error below. To run every collectable
test, add `--continue-on-collection-errors`; the current result is **142 passed, 3 failed, 1 error**.
The failures are legacy tests from the original codebase that predate the evaluation work:

| Test | Failure |
|---|---|
| `tests/test_forecasting_pipeline.py` | Collection error: imports `apply_indicator_adjustment`, which `src.forecasting` does not export |
| `tests/test_smoke.py::test_app_imports`, `::test_page_imports` | Importing `pages/01_Data_Extraction.py` raises `'str' object has no attribute 'value'` |
| `tests/test_headline_analysis.py::test_analyze_headlines` | Mocks an older response schema (`KeyError: 'magnitude'`) |

### Reproducing the evaluation results

Run from the project root:

```powershell
python scripts/build_census_ecommerce_dataset.py   # Census target from the stored workbooks (offline)
python scripts/run_census_baseline_experiment.py   # -> data/evaluation/results/census_baseline/

python scripts/fetch_indicator_sources.py          # optional; re-downloads BEA/World Bank/FRED sources (network)
python scripts/build_indicator_datasets.py         # -> data/evaluation/indicators/
python scripts/run_census_indicator_experiment.py  # -> data/evaluation/results/census_indicators/
```

With the stored source files, re-running the experiments reproduces the stored metrics; only the
`run_timestamp` metadata changes.

### Development principles

- **Temporal isolation:** any new model, adjustment or data source used in evaluation must only see data available at the forecast origin.
- **Reproducibility:** experiments are deterministic given their inputs, and results are written with their configuration.
- **Provenance:** raw sources are stored unmodified, with URLs and SHA-256 hashes; results carry input fingerprints.
- **No leakage:** temporal isolation is covered by tests in `tests/test_backtest.py`.
- **No unsupported claims:** performance statements must be backed by stored, reproducible results.

No formatter, linter or type checker is configured. Before committing, run the tests and
`git diff --check`.

---

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `pytest` is not recognized, or `No module named pytest` | pytest is not in `requirements.txt`. Activate the virtual environment, run `pip install pytest`, and use `python -m pytest`. |
| `streamlit` is not recognized | The virtual environment is not active; activate it or run `python -m streamlit run app.py`. |
| `ModuleNotFoundError` on startup | Activate the virtual environment and run `pip install -r requirements.txt`. |
| `config/.env` or category files are not found | Start Streamlit from the project root. |
| "No database connection configured. Using mock data for demo." | `DB_URL` is not set, or no engine could be created from it. Set `DB_URL` in `config/.env` and install the matching SQLAlchemy driver. |
| Data extraction fails with a query error | `DB_URL` is valid but the database is unreachable or lacks the expected KPI tables. Check connectivity, credentials and schema. |
| All headlines are `Neutral/Noise` with 0% impact | LLM calls are failing. Check `LLM_PROVIDER`, `OPENAI_API_KEY` / `OPENAI_BASE_URL`, or that Ollama is running at `OLLAMA_BASE_URL`, and check the logs. |
| LLM rate-limit errors | Lower `LLM_MAX_CONCURRENT` or raise `LLM_REQUEST_DELAY`. |
| GDELT requests fail with certificate errors | Check network access to `api.gdeltproject.org`. On a trusted network only, temporarily set `GDELT_SSL_VERIFY=false`. |
| No news returned, or news fetching is slow | Use broader topics; GDELT covers only about the last three months. Requests are spaced 5–6 seconds apart, so many topics take time. |
| A country receives no news | Check that it has a valid ISO3 code with a FIPS mapping. Countries with fewer than 5 analyzed headlines use global news at reduced confidence. |
| Top-Down returns "Forecast execution failed" | Known defect in classic mode: `src/forecasting/methods/top_down.py` uses an undefined variable `share` (the loop variable is `_share`). Use Bottom-Up, Country-Specific or Global-level until it is fixed. |
| The Model Evaluation page reports missing results | Files in `data/evaluation/results/` are missing; regenerate them with the evaluation scripts. The page never generates results itself. |
| Regenerated results differ from the stored ones | The source files were replaced with a newer release. Census and BEA revise historical values, so re-downloaded (current-vintage) data can change the metrics; keep the stored raw files to reproduce the published results. |
| `pytest` stops with a collection error, or reports 3 failures | These are the known legacy test failures; use `--continue-on-collection-errors`. The evaluation tests are unaffected. |
