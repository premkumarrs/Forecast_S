# XMI GDELT Forecasting Framework

A sophisticated market forecasting system that combines historical data analysis with real-time news sentiment and indicators to generate AI-enhanced market predictions.

## 🌟 Features

### Core Capabilities
- **Multi-Method Forecasting**: Support for multiple forecasting approaches (Bottom-Up, Top-Down, Country-Specific, Global Only, Middle-Out)
- **AI-Powered News Analysis**: Real-time news sentiment analysis using LLM integration with recency weighting
- **Indicators Integration**: Incorporation of indicators with flexible weighting
- **Temporal Decay Analysis**: News impact with exponential decay over forecast horizon
- **Country-Specific Analysis**: Detailed forecasts for individual countries with localized context
- **Interactive Visualizations**: Rich charts and insights powered by Streamlit
- **Concurrent Processing**: Fast parallel headline analysis with progress tracking

### Forecasting Methods
1. **Bottom-Up**: Aggregates country-level forecasts to global
2. **Top-Down**: Distributes global forecast to countries  
3. **Country-Specific**: Deep-dive analysis for selected countries
4. **Global Only**: Pure global market analysis

### Baseline Forecast Models
- **3-yr CAGR**: Compound Annual Growth Rate
- **Damped ETS**: Exponential Smoothing with damping
- **Logistic Growth**: S-curve growth modeling

### Data Sources
- **GDELT Project**: Real-time global news monitoring (250+ articles per batch)
- **SQL Databases**: Connect via SQLAlchemy

## 📋 Prerequisites

### Required Software
- Python 3.10 or higher
- Git

### API Keys Required
At least one of the following LLM providers:
- **OpenRouter API** (Recommended) - Access to multiple models
- **OpenAI API** - GPT-3.5/GPT-4
- **Anthropic API** - Claude models
- **Google AI API** - Gemini models
- **Perplexity API** (Recommended for research features)
- **Ollama** - For local LLM deployment

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/xmi_gdelt_forecasting_framework.git
cd xmi_gdelt_forecasting_framework
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```env
# LLM Configuration (at least one required)
OPENROUTER_API_KEY=your_openrouter_key_here
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
PERPLEXITY_API_KEY=your_perplexity_key_here

# LLM Settings
LLM_PROVIDER=openrouter  # or openai, ollama
LLM_MODEL=anthropic/claude-3.5-sonnet
LLM_MAX_CONCURRENT=10
LLM_REQUEST_DELAY=0.05

# News Settings
NEWS_ARTICLE_THRESHOLD=10  # Minimum articles for analysis
GDELT_RATE_LIMIT_MIN=1     # Min seconds between GDELT requests
GDELT_RATE_LIMIT_MAX=3     # Max seconds between GDELT requests

# Optional Settings
LOG_LEVEL=INFO
OLLAMA_BASE_URL=http://localhost:11434  # For local LLM
```

### 5. Configure LLM Provider
Create `config/.env` for additional LLM settings:
```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

## 🎯 Quick Start

### Launch the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Basic Workflow

1. **Home Page**
   - Overview of the framework
   - Navigation to different modules

2. **Configuration** (Page 1)
   - Set market name (e.g., "Cloud Computing Market")
   - Choose forecast mode (classic, or Existing Forecast + News Adjustment)
   - Classic mode: pick baseline model (CAGR / ETS / Logistic) and optional indicators
   - Existing forecast mode: reuse market series, lock news at 100%, set adjustment window
   - Configure data sources (indicators, news)
   - Generate or load market categories and topics

3. **Data Extraction** (Page 2)
   - connect to SQL database
   - Preview and validate data
   - Check data quality and completeness

4. **Forecasting** (Page 3)
   - Classic: run news analysis, generate model baseline, apply news/indicator adjustments
   - Existing forecast mode: skip model baseline, apply news-only adjustments within selected window
   - View results and progress

5. **Insights** (Page 4)
   - Interactive visualizations
   - News analysis results with categories
   - Forecast adjustments breakdown
   - Regional insights and country comparisons
   - Export results

## 📁 Project Structure

```
xmi_gdelt_forecasting_framework/
├── src/                          # Core business logic
│   ├── forecasting/             # Forecasting domain
│   │   ├── adjustment/         # News & indicator adjustments
│   │   ├── core/              # Core forecasting logic
│   │   ├── methods/           # Forecasting method implementations
│   │   │   ├── bottom_up.py
│   │   │   ├── top_down.py
│   │   │   ├── country_specific.py
│   │   │   └── global_only.py
│   │   └── models/            # Baseline forecast models
│   │       ├── baseline_factory.py
│   │       ├── cagr.py
│   │       ├── ets.py
│   │       └── logistic.py
│   │
│   ├── services/               # Application services layer
│   │   ├── base.py           # Base service class
│   │   ├── configuration/    # Configuration management
│   │   ├── data/             # Data extraction and processing
│   │   └── forecast/         # Forecast orchestration
│   │       ├── forecast_service.py
│   │       ├── news_analysis_service.py
│   │       └── method_router.py
│   │
│   ├── news/                   # News analysis domain
│   │   ├── gdelt.py          # GDELT API integration
│   │   ├── sentiment.py      # Sentiment analysis
│   │   └── aggregation.py    # News signal aggregation
│   │
│   ├── llm/                    # LLM integration
│   │   ├── analyst.py         # SimpleLLMAnalyst class
│   │   ├── prompts/           # Prompt templates
│   │   │   ├── analysis_prompts.py
│   │   │   └── category_prompts.py
│   │   ├── providers/         # LLM provider implementations
│   │   │   ├── provider_factory.py
│   │   │   ├── openai_provider.py
│   │   │   └── ollama_provider.py
│   │   └── categories/        # Category management
│   │
│   └── data/                   # Data processing
│       ├── extractors/        # Data extraction strategies
│       ├── processors/        # Data processing logic
│       └── country_utils.py  # Country/ISO3 utilities
│
├── pages/                      # Streamlit UI pages
│   ├── components/            # Reusable UI components
│   │   ├── charts.py         # Chart creation functions
│   │   ├── metrics.py        # Metric calculations
│   │   ├── layouts.py        # Layout helpers
│   │   └── utils.py          # Shared utilities
│   ├── helpers/               # Page-specific logic
│   │   ├── insights.py       # Insights page logic
│   │   ├── forecasting.py    # Forecasting helpers
│   │   └── configuration.py  # Config helpers
│   ├── 01_Configuration.py
│   ├── 02_Data_Extraction.py
│   ├── 03_Forecasting.py
│   └── 04_Insights.py
│
├── config/                     # Configuration files
│   ├── categories/            # Market category definitions
│   ├── exclusions.json       # Excluded regions list
│   └── flat-ui__data*.csv    # ISO3-FIPS mapping data
│
├── data/                       # Data storage
│   ├── raw/                   # Raw uploaded files
│   ├── processed/             # Processed data cache
│   └── sample/                # Sample datasets
│
├── Home.py                    # Application entry point
├── requirements.txt           # Python dependencies
├── ARCHITECTURE.md           # Detailed architecture docs
└── README.md                 # This file
```

## 🔧 Configuration

### Market Categories
Categories define how news articles are classified. Each category includes:

### GDELT SSL Verification
- Purpose: controls TLS certificate verification for connections to GDELT.
- Default: verification ON (secure). Recommended for all production deployments.
- Env var: set `GDELT_SSL_VERIFY=true|false` in `config/.env` or environment.
- Config file: in `config/settings.toml` set under `[gdelt]` → `ssl_verify = true|false`.
- Scope: when disabled, only requests to `*.gdeltproject.org` skip verification; a warning is logged.
- Security: disabling verification exposes you to MITM risks. Use only for troubleshooting in trusted networks when the GDELT endpoint presents temporary certificate issues. Re-enable as soon as possible.
- **Name**: Category identifier
- **Description**: What the category represents  
- **Growth Constraints**: Min/max impact percentages

Example categories for Cloud Computing:
```json
{
  "name": "Infrastructure Investment",
  "description": "Major cloud infrastructure investments and data center expansions",
  "growth_constraint_min": 10,
  "growth_constraint_max": 25
}
```

### News Topics
Topics are search terms for GDELT news fetching. Best practices:
- Use broad, news-relevant terms
- Include major companies and technologies
- Keep topics 1-2 words when possible
- Avoid special characters and quotes

Example topics:
```python
topics = [
    "cloud computing",
    "AWS",
    "Microsoft Azure",
    "Google Cloud",
    "data center",
    "SaaS",
    "serverless",
    "edge computing"
]
```

### Temporal Impact Classification
The system classifies news impacts as:
- **Long-term** (>2 years): Strategic investments, infrastructure, regulations
- **Short-term** (<2 years): Quarterly results, product launches, disruptions
- **No effect**: Noise, duplicates, irrelevant news

## 📊 Data Requirements

### Market Data Format (Excel/CSV)
Required columns:
- **country**: Country name (must match exactly)
- **year**: Year (YYYY format)
- **value**: Market value (numeric)
- **iso3**: ISO3 country code (optional but recommended)

Example:
```csv
country,year,value,iso3
India,2020,1000,IND
India,2021,1200,IND
United States,2020,5000,USA
United States,2021,5500,USA
```

### Indicator Data Format
Required columns:
- **country**: Country name
- **year**: Year
- **indicator_key**: Indicator name
- **value**: Indicator value

### Country-ISO3 Mapping
The system uses `config/flat-ui__data*.csv` for ISO3 to FIPS conversion needed by GDELT.

## 🤖 LLM Integration

### Response Parsing
The system expects JSON responses from LLMs with specific structure:
```json
{
  "category": "Infrastructure Investment",
  "temporal_impact": "long_term",
  "growth_rate": 15.0,
  "reason": "Major investment will boost capacity..."
}
```

### Prompt Engineering
Key prompt features:
- Explicit JSON format requirements
- Temporal impact guidelines
- Country-specific context when applicable
- Growth rate constraints per category

## 📈 Advanced Features

### Adjustment Mechanisms

#### News-Based Adjustments
- Temporal weighting (recent news weighted more)
- Category-specific growth constraints
- Confidence scoring based on article count
- Long-term vs short-term impact separation

#### Indicator-Based Adjustments
- Indicator correlation
- Weighted by indicator importance
- Historical trend analysis

### Fallback Strategies
When country-specific news is limited:
- Supplements with global news
- Adjusts confidence scores
- Applies weighted blending

### Performance Optimizations
- Concurrent LLM requests (10 parallel by default)
- Batch processing for GDELT queries
- Rate limiting with random delays
- Progress tracking for long operations

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black src/ pages/

# Lint
pylint src/ pages/

# Type checking
mypy src/
```

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### "Failed to generate categories"
- Verify LLM API key is correctly set
- Check API rate limits
- Ensure market name doesn't contain special characters

#### "No articles found"
- Check GDELT accessibility
- Verify topics are news-relevant
- Try broader search terms
- Check date range (default is 90 days)

#### All news showing "Neutral/Noise" with 0% impact
- LLM response parsing issue - check the fix in `src/llm/analyst.py`
- Ensure `parsed_response = llm_response.get('response', {})` is used

#### "Country-Specific forecast: 0 countries with news"
- Check article threshold setting (default 10)
- Verify ISO3 mapping in config files
- Ensure countries have sufficient news coverage

#### Rate Limiting Issues
- Adjust `LLM_REQUEST_DELAY` and `LLM_MAX_CONCURRENT`
- Increase GDELT rate limits in `.env`

### Performance Tips

#### For Large Datasets
- Reduce `LLM_MAX_CONCURRENT` to avoid rate limits
- Increase `NEWS_ARTICLE_THRESHOLD` to limit analysis
- Use sampling for very large news datasets

#### Memory Management
- Process countries in batches
- Clear session state between runs
- Use data caching for repeated operations

## 📚 API Reference

### Core Services

#### ForecastService
```python
from src.services.forecast import ForecastService

service = ForecastService()

# Run complete forecast
result = service.run_forecast(
    method="Bottom-Up",
    unified_data=data,
    config=config,
    analyzed_news=news_data,
    selected_countries=["India", "United States"]
)

# Analyze news
news_result = service.analyze_news(
    method="Country-Specific",
    market_name="Cloud Computing",
    topics=["cloud", "AWS"],
    unified_data=data,
    countries=["India"]
)
```

#### ConfigurationService
```python
from src.services.configuration import ConfigurationService

service = ConfigurationService()

# Generate with LLM
topics = service.generate_topics_with_llm("Cloud Computing", 50)
categories = service.generate_market_categories(
    "Cloud Computing",
    num_categories=8,
    min_growth=-10,
    max_growth=25
)
```

#### SimpleLLMAnalyst
```python
from src.llm.analyst import SimpleLLMAnalyst

analyst = SimpleLLMAnalyst("Cloud Computing")

# Analyze headlines
analyzed = analyst.analyze_headlines(
    headlines=[{"title": "...", "date": "..."}],
    country="India",  # Optional country context
    forecast_years=5
)
```

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact the development team
- Check documentation and troubleshooting guide

## 🔄 Version History

### Version 1.5.0
- Top Down 
- Bottom Up Forecasting 
- Regional Aggregation
- Insight Page
- Category generation with LLM
- Fallback strategies for limited data
- Enhanced temporal impact classification
- Improved error handling and logging

### Version 1.0.0
- Initial release
- Basic forecasting methods
- GDELT integration

---
