# Technical Documentation: Baseline + Adjustment Forecasting System

> **🔄 Updated for Moving Average (MA) System**: This documentation has been updated to reflect the new category-specific Moving Average (MA) system. The previous impact_multiplier has been completely removed, and news adjustments now use category-specific MA windows for more transparent and adaptive calculations.

## Table of Contents
1. [System Overview](#system-overview)
2. [Baseline Forecasting](#baseline-forecasting)
3. [GDELT News Search Process](#gdelt-news-search-process)
4. [News Analysis with LLM](#news-analysis-with-llm)
5. [Adjustment Calculations](#adjustment-calculations)
6. [Fallback Strategies](#fallback-strategies)
7. [Complete Flow Example](#complete-flow-example)
8. [Key Files and Formulas](#key-files-and-formulas)
9. [Configuration Parameters](#configuration-parameters)

---

## System Overview

The XMI GDELT Forecasting Framework uses a two-stage approach:
1. **Baseline Forecast**: Generate initial forecast using historical data
2. **Adjustments**: Apply news and indicator-based adjustments to the baseline

```
Historical Data → Baseline Model → Baseline Forecast
                                          ↓
News Data (GDELT) → LLM Analysis → News Adjustments
                                          ↓
Economic Indicators → Statistical Analysis → Indicator Adjustments
                                          ↓
                                    Final Adjusted Forecast
```

---

## Baseline Forecasting

### Available Models

**File**: `src/forecasting/models/baseline_factory.py`

Three baseline models are available:

#### 1. 3-Year CAGR (Compound Annual Growth Rate)
**File**: `src/forecasting/models/cagr.py` (Lines 33-45)

```python
# Formula: CAGR = (End Value / Start Value)^(1/periods) - 1
start_val = last_n_years.iloc[0]['value']
end_val = last_n_years.iloc[-1]['value']
periods = len(last_n_years) - 1

if start_val > 0:
    self.cagr = (end_val / start_val) ** (1 / periods) - 1

# Forecast: Value(t) = Last_Value × (1 + CAGR)^t
forecast_value = self.last_value * ((1 + self.cagr) ** (i + 1))
```

#### 2. Damped ETS (Exponential Smoothing)
**File**: `src/forecasting/models/ets.py`

Uses statsmodels ExponentialSmoothing with:
- Additive trend
- Damping factor (default 0.96)
- No seasonal component

#### 3. Logistic Growth
**File**: `src/forecasting/models/logistic.py`

S-curve growth model for markets approaching saturation.

### Baseline Generation Process

**File**: `src/forecasting/models/baseline_factory.py` (Lines 55-84)

```python
def generate_baseline_forecast(
    global_ts: pd.DataFrame,
    model_type: str,
    hist_cutoff: int,
    forecast_until: int
) -> pd.DataFrame:
    # Create model instance
    model = cls.create_model(model_type)
    
    # Generate forecast
    return model.generate_forecast_dataframe(
        global_ts,
        hist_cutoff,
        forecast_until
    )
```

---

## GDELT News Search Process

### 1. Country Mapping (ISO3 → FIPS)

**File**: `src/news/gdelt.py` (Lines 140-151)

GDELT requires FIPS-2 country codes, not ISO3. The system performs conversion:

```python
def iso3_to_fips(iso3: Optional[str]) -> Optional[str]:
    # Primary: Use CSV-loaded mappings
    if code in _CSV_ISO3_TO_FIPS:  # From config/flat-ui__data*.csv
        return _CSV_ISO3_TO_FIPS[code]
    
    # Fallback: Hardcoded common mappings
    return _FALLBACK_ISO3_TO_FIPS.get(code)

# Examples:
# IND → IN (India)
# USA → US (United States)
# DEU → GM (Germany)
```

**Mapping File**: `config/flat-ui__data-Sun Aug 17 2025.csv`

### 2. Topic Processing

**File**: `src/news/gdelt.py` (Lines 98-119)

Topics are cleaned before searching:
- Remove special characters: `()[]{}'"` 
- No quotes added (GDELT handles phrases)
- Each topic becomes separate batch

### 3. GDELT API Calls

**File**: `src/news/gdelt.py` (Lines 157-191)

```python
def _search_articles_batch(
    keywords: List[str],
    start_date: str,
    end_date: str,
    country_fips: Optional[str],
    max_records: int  # Max 250 per batch
):
    # Rate limiting (1-3 seconds random delay)
    _rate_limiter.acquire()
    
    # Create GDELT filter
    f = Filters(
        keyword=keywords[0],  # Single keyword per batch
        start_date=start_date,
        end_date=end_date,
        num_records=max_records,
        country=country_fips  # FIPS code for country
    )
    
    # Execute search
    df = _gd.article_search(f)
```

### 4. Multi-Country Fetching

**File**: `src/news/gdelt.py` (Lines 277-313)

```python
def fetch_multi_country_headlines(
    country_name_to_iso3: Dict[str, str],
    topics: Iterable[str],
    max_workers: int = 4  # Process 4 countries in parallel
):
    # Each country uses 5 parallel batch workers
    # Total: 4 countries × 5 batches = 20 concurrent requests max
```

### 5. Rate Limiting

**File**: `src/news/gdelt.py` (Lines 68-87)

```python
class GDELTRateLimiter:
    # Random delay between requests
    min_interval = 1  # seconds
    max_interval = 3  # seconds
    
    # Random interval for each call
    target_interval = random.uniform(min_interval, max_interval)
```

---

## News Analysis with LLM

### 1. Headline Analysis Flow

**File**: `src/llm/analyst.py` (Lines 46-94)

```python
def analyze_headlines(headlines, country=None):
    # Choose prompt type
    if country:
        system_prompt = build_country_system_prompt(...)  # Country-specific
    else:
        system_prompt = build_system_prompt(...)  # Global
    
    # Parallel processing (10 concurrent by default)
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Each headline analyzed separately
        analysis = _analyze_single_headline(headline, system_prompt)
```

### 2. LLM Response Structure

**File**: `src/llm/analyst.py` (Lines 105-128)

Expected response format:
```json
{
    "category": "Infrastructure Investment",
    "temporal_impact": "long_term",  // or "short_term", "no_effect"
    "growth_rate": 15.0,  // -100 to +100
    "reason": "Major cloud infrastructure investment..."
}
```

Response parsing:
```python
# LLM response is nested
llm_response = call_llm(title, system_prompt, max_tokens=150)
parsed_response = llm_response.get('response', {})  # Extract nested response

# Convert to analysis format
analysis = {
    'category': parsed_response.get('category', 'Neutral/Noise'),
    'temporal_impact': parsed_response.get('temporal_impact', 'no_effect'),
    'growth_rate': float(parsed_response.get('growth_rate', 0.0)),
    'reason': parsed_response.get('reason', 'Analysis completed')
}
```

---

## Moving Average (MA) System

### Overview

The MA system provides category-specific temporal windows for news impact calculation. Each news category can have its own MA window (0-120 days), allowing for nuanced impact assessment based on the nature of the news.

### Category-Specific Windows

Categories are automatically assigned default MA windows based on their characteristics:

- **Market/Trading/Sentiment** (14 days): Fast-moving market indicators
- **Product/Launch/Announcement** (30 days): Medium-term business events  
- **Strategic/Investment/Funding** (45 days): Longer-term strategic changes
- **Regulation/Policy/Infrastructure** (60 days): Slow-moving structural changes
- **Neutral/Noise** (0 days): Categories that don't contribute to adjustments

### MA Calculation Process

**File**: `src/forecasting/adjustments/news_adjustment.py`

1. **Category Filtering**: News is grouped by category
2. **Window Application**: Each category uses its configured MA window
3. **Dynamic Adaptation**: If actual data < window, uses available data
4. **Equal Weighting**: Active categories contribute equally to final impact
5. **Aggregation**: Simple average across all active categories

```python
# For each category with ma_window_days > 0:
cutoff_date = now - ma_window_days
recent_data = category_news[date >= cutoff_date]
daily_avg = recent_data.groupby(date).mean()
category_ma = daily_avg.mean()

# Final impact = mean(all category MAs)
```

### Benefits of MA System

- **Transparency**: Clear contribution from each category
- **Adaptability**: Handles sparse data gracefully
- **Customization**: Each category can be tuned independently
- **No Hidden Multipliers**: Direct calculation without opaque scaling factors

## Adjustment Calculations

### Complete Adjustment Formula

The final adjusted forecast value for any year is calculated as:

```
Adjusted_Value(year) = Baseline_Value(year) × (1 + Total_Adjustment(year))
```

Where Total_Adjustment is:

```
Total_Adjustment(year) = 
    News_Weight × Impact_Multiplier × News_Adjustment × Temporal_Decay(year) +
    Indicator_Weight × Indicator_Adjustment × Impact_Multiplier
```

And the news adjustment uses recency weighting:

```
News_Adjustment = Σ(growth_rate_i × weight_i) / Σ(weight_i) × Confidence
where weight_i = 0.5^(age_days_i / 90)
```

**Key Parameters**:
- News_Weight = 0.7 (70% influence)
- Indicator_Weight = 0.3 (30% influence)
- Impact_Multiplier = 1.5 (amplification factor, configurable 0x-10x)
- News_Half_Life = 90 days (for recency weighting)
- Temporal_Decay_Rate = 0.6 (exponential decay across forecast horizon)
- Confidence = 1.0 (native) or 0.5 (fallback)

### 1. Unified Adjustment Function

**File**: `src/forecasting/adjustments/unified.py` (Lines 12-62)

```python
def calculate_unified_adjustment(
    baseline_df: pd.DataFrame,
    news_data: pd.DataFrame,
    indicators_df: pd.DataFrame,
    config: Dict,
    news_confidence_multiplier: float = 1.0
) -> Tuple[pd.DataFrame, Dict]:
    # Extract weights from config
    weights = {
        'news_weight': 0.7,      # Default 70% news
        'indicator_weight': 0.3   # Default 30% indicators
    }
    
    # Apply all adjustments
    factory = AdjustmentFactory()
    adjusted_df, details = factory.apply_all_adjustments(...)
```

### 2. News-Based Adjustments (Single-Rate Model)

**File**: `src/forecasting/adjustments/news_adjustment.py` (Lines 35-110)

```python
def _recency_weighted_mean(df: pd.DataFrame, half_life_days: int) -> float:
    """Calculate recency-weighted mean using exponential decay."""
    x = pd.to_numeric(df.get('growth_rate'), errors='coerce')
    d = pd.to_datetime(df.get('date'), errors='coerce')
    
    # Filter valid data
    m = x.notna() & d.notna()
    if not m.any():
        return float('nan')
    
    # Calculate age in days from current time
    age_days = (pd.Timestamp.utcnow() - d[m]).dt.days.clip(lower=0)
    
    # Apply half-life formula: weight(t) = 0.5 ** (age_days / half_life)
    half_life_days = max(1, abs(half_life_days))
    w = np.power(0.5, age_days / half_life_days)
    
    return float(np.average(x[m], weights=w))

def calculate(data: pd.DataFrame, config: Dict) -> float:
    # Filter relevant articles only
    if 'relevant' in data.columns:
        data = data[data['relevant'] == 1].copy()
    
    # Filter non-zero growth rates
    data = data[data['growth_rate'] != 0]
    
    if data.empty:
        self.adjustment_value = 0.0
        return self.adjustment_value
    
    # Single recency half-life (days)
    half_life = int(config.get('news_half_life_days', 90))
    
    # Calculate single recency-weighted average
    news_avg_pct = self._recency_weighted_mean(data, half_life)
    if pd.isna(news_avg_pct):
        news_avg_pct = data['growth_rate'].mean()
    
    # Convert percentage to decimal and apply confidence multiplier
    self.adjustment_value = (news_avg_pct / 100.0) * self.confidence_multiplier
    
    # Bound to sane limits
    self.adjustment_value = self.validate_adjustment(self.adjustment_value, -0.3, 0.3)
    
    return self.adjustment_value
```

#### Recency Weighting Formula

The system uses **exponential decay based on article age**:

```
Weight(article) = 0.5 ^ (age_in_days / half_life_days)
```

Where:
- **News half-life**: 90 days (default, configurable)
- All relevant articles are combined into a single growth rate

#### Why 90-Day Half-Life for 90-Day Data Window

Since GDELT fetches 90 days of news data, the 90-day half-life provides optimal differentiation:
- Recent news (today): 100% weight (0.5^(0/90) = 1.00)
- Mid-range news (45 days): 71% weight (0.5^(45/90) = 0.71)  
- Oldest news (90 days): 50% weight (0.5^(90/90) = 0.50)

This ensures:
- Recent news has maximum influence on the forecast
- Older news still contributes but with diminishing weight
- No artificial separation between "long-term" and "short-term" categories

#### Detailed Calculation Example

Let's walk through a complete example of how recency-weighted adjustments are calculated:

**Scenario**: Analyzing news on August 21, 2025 with the following articles:

**Step 1: Calculate Individual Weights**

Using 90-day half-life for all articles:
```
Weight(A) = 0.5^(1/90) = 0.992    # Very recent article  
Weight(B) = 0.5^(6/90) = 0.975    # Recent article
Weight(C) = 0.5^(30/90) = 0.794   # Mid-range article
Weight(D) = 0.5^(90/90) = 0.500   # Oldest article
Weight(E) = 0.5^(11/90) = 0.919   # Recent article
```

**Step 2: Calculate Single Weighted Average**

Single news adjustment (all relevant articles combined):
```
Numerator = (20×0.992) + (15×0.975) + (-10×0.794) + (-15×0.500) + (30×0.919)
         = 19.84 + 14.63 - 7.94 - 7.50 + 27.57
         = 46.60

Denominator = 0.992 + 0.975 + 0.794 + 0.500 + 0.919 = 4.18

News_weighted_average = 46.60 / 4.18 = 11.15%
```

Compare to simple mean: (20 + 15 - 10 - 15 + 30) / 5 = 8.00%
**Impact**: Recent positive news (A, B, E) dominates, resulting in +11.15% vs +8.00%

**Step 3: Apply to Forecast with Temporal Decay**

For a 6-year forecast (2025-2030), using:
- News adjustment = 11.15% / 100 = 0.1115
- News weight = 0.7
- Impact multiplier = 1.5  
- Long-term decay rate = 0.6

```
Year 2025 (i=0):
  Decay = 0.5^(0 / (6 × 0.6)) = 1.00
  News_term = 0.7 × 1.5 × 0.1115 × 1.00 = 0.1171 (11.71%)
  Final_multiplier = 1 + 0.1171 = 1.1171
  
Year 2026 (i=1):
  Decay = 0.5^(1 / 3.6) = 0.83
  News_term = 0.7 × 1.5 × 0.1115 × 0.83 = 0.0973 (9.73%)
  Final_multiplier = 1 + 0.0973 = 1.0973
  
Year 2027 (i=2):
  Decay = 0.5^(2 / 3.6) = 0.69
  News_term = 0.7 × 1.5 × 0.1115 × 0.69 = 0.0808 (8.08%)
  Final_multiplier = 1 + 0.0808 = 1.0808
  
Year 2028 (i=3):
  Decay = 0.5^(3 / 3.6) = 0.58
  News_term = 0.7 × 1.5 × 0.1115 × 0.58 = 0.0679 (6.79%)
  Final_multiplier = 1 + 0.0679 = 1.0679
  
Year 2029 (i=4):
  Decay = 0.5^(4 / 3.6) = 0.48
  News_term = 0.7 × 1.5 × 0.1115 × 0.48 = 0.0562 (5.62%)
  Final_multiplier = 1 + 0.0562 = 1.0562
  
Year 2030 (i=5):
  Decay = 0.5^(5 / 3.6) = 0.40
  News_term = 0.7 × 1.5 × 0.1115 × 0.40 = 0.0469 (4.69%)
  Final_multiplier = 1 + 0.0469 = 1.0469
```

**Applied Adjustments**:
- 2025: Baseline × 1.1171 (+11.71%)
- 2026: Baseline × 1.0973 (+9.73%)
- 2027: Baseline × 1.0808 (+8.08%)
- 2028: Baseline × 1.0679 (+6.79%)
- 2029: Baseline × 1.0562 (+5.62%)
- 2030: Baseline × 1.0469 (+4.69%)
- 2030: Baseline × 1.0589

#### Why Recency Weighting Matters

**Without recency weighting** (simple average):
- Short-term: 2.50% (diluted by old negative news)
- Long-term: 21.67%
- Total: 16.67%

**With recency weighting**:
- Short-term: 8.40% (recent positive news dominates)
- Long-term: 22.90% (slight recent bias)
- Total: 18.55%

The difference is most pronounced for short-term impacts where news from 90 days ago gets only 25% weight compared to yesterday's news. This ensures the forecast reacts to current market conditions rather than stale information.

### 3. Temporal Decay

**File**: `src/forecasting/adjustments/temporal_decay.py`

#### Exponential Decay (Single Model)
**Lines 27-82**
```python
def calculate_temporal_decay(years_from_start, total_forecast_years, decay_rate=0.6):
    # Exponential decay with configurable half-life point
    decay_point = total_forecast_years * decay_rate  # Default: 60% point for half-life
    
    # Half-life formula: 0.5^(t/half_life)
    decay = 0.5 ** (years_from_start / decay_point)
    return decay

# Example (6-year forecast, decay_rate=0.6):
# Year 0: 100%, Year 1: 81%, Year 2: 66%, Year 3: 53%, Year 4: 43%, Year 5: 35%
```

### 4. Final Adjustment Application

**File**: `src/forecasting/adjustments/news_adjustment.py` (Lines 96-138)

```python
# For each forecast year
for i, year in enumerate(forecast_years):
    # Apply temporal decay
    decay = temporal_decay.calculate_temporal_decay(i, total_years, decay_rate=0.6)
    
    # Apply news adjustment with decay
    news_adj = news_adjustment * decay
    
    # Apply weights
    actual_news = news_weight * news_adj
    actual_indicators = indicator_weight * indicator_adjustment
    total_adj = actual_news + actual_indicators
    
    # Apply to baseline forecast
    adjustment_factor = 1 + total_adj
    adjusted_value = baseline_value * adjustment_factor
```

**Complete Formula**:
```
Adjusted_Value(year) = Baseline_Value(year) × (1 + Total_Adjustment(year))

where:
Total_Adjustment = News_Weight × Impact_Multiplier × News_Avg × Decay(year) × Confidence +
                  Indicator_Weight × Impact_Multiplier × Indicator_Adjustment
```

---

## Fallback Strategies

### When Fallbacks Trigger

**File**: `src/forecasting/methods/country_specific.py` (Lines 88-95)

Fallback triggers when:
- Country has no news articles (`country_news_df.empty`)
- Country has < 5 articles (`len(country_news_df) < 5`)

### Fallback Methods

**File**: `src/forecasting/adjustments/fallback_strategies.py`

#### 1. USE_GLOBAL_REDUCED (Default)
**Lines 93-101**
```python
def _use_global_reduced(global_data, confidence, country):
    # Use global news at reduced confidence (default 50%)
    reason = f"Using global news at {confidence:.0%} confidence"
    return global_data, confidence, reason
```

#### 2. SKIP_ADJUSTMENTS
**Lines 103-106**
```python
def _skip_adjustments(country):
    # No news adjustments applied
    return pd.DataFrame(), 0.0, reason
```

#### 3. USE_REGIONAL
**Lines 108-120**
```python
def _use_regional(regional_data, global_data, confidence, country):
    # Use regional data if available, else global at 80% confidence
    if regional_data and not regional_data.empty:
        return regional_data, confidence, reason
    else:
        return global_data, confidence * 0.8, reason
```

### GDELT Fallback (apply_country_fallback)

**File**: `src/news/gdelt.py` (Lines 346-418)

```python
def apply_country_fallback(
    country_name: str,
    country_articles: pd.DataFrame,
    global_articles: pd.DataFrame,
    min_threshold: int = 8,
    fill_limit: int = 50
) -> Tuple[pd.DataFrame, float, str]:
    
    n_native = len(country_articles)
    
    if n_native >= min_threshold:
        # Enough articles
        return country_articles, 1.0, "native"
    
    if global_articles is None:
        # No global fallback available
        weight = n_native / min_threshold
        return country_articles, weight, "low_coverage"
    
    # Top up with global articles
    need = min(min_threshold - n_native, fill_limit)
    supplement = global_articles.head(need)
    
    combined = pd.concat([country_articles, supplement])
    native_ratio = n_native / len(combined)
    
    return combined, native_ratio, "topped_up"
```

---

## Complete Flow Example

### Country-Specific Forecast with News

1. **Input**: India, Cloud Computing Market, 2025-2030 forecast

2. **Baseline Generation**
   ```python
   # File: src/forecasting/models/baseline_factory.py
   baseline = BaselineModelFactory.generate_baseline_forecast(
       india_historical_data,
       "3-yr CAGR",
       hist_cutoff=2024,
       forecast_until=2030
   )
   ```

3. **GDELT News Fetch**
   ```python
   # File: src/news/gdelt.py
   # Convert India → IND → IN (FIPS)
   fips = iso3_to_fips("IND")  # Returns "IN"
   
   # Fetch with topics
   articles = fetch_country_headlines(
       "IND",
       ["cloud computing", "AWS", "data center"],
       days_back=90
   )
   ```

4. **LLM Analysis**
   ```python
   # File: src/llm/analyst.py
   analyst = SimpleLLMAnalyst("Cloud Computing Market")
   analyzed = analyst.analyze_headlines(
       articles,
       country="India",  # Country-specific prompt
       forecast_years=6
   )
   ```

5. **Check for Fallback**
   ```python
   # File: src/forecasting/methods/country_specific.py (Lines 88-95)
   if len(analyzed) < 5:
       # Apply fallback
       analyzed, confidence, reason = apply_fallback_strategy(
           "India",
           global_news,  # Use global news
           config
       )
       # confidence = 0.5 (50% for global fallback)
   ```

6. **Calculate Adjustments**
   ```python
   # File: src/forecasting/adjustments/unified.py
   adjusted_forecast, details = calculate_unified_adjustment(
       baseline,
       analyzed,  # News data
       indicators,  # Economic indicators
       config,
       news_confidence_multiplier=confidence  # 0.5 if fallback, 1.0 if native
   )
   ```

7. **Apply Temporal Decay**
   ```python
   # For each year (2025-2030)
   Year 2025 (i=0): LT_decay=1.00, ST_decay=1.00
   Year 2026 (i=1): LT_decay=0.81, ST_decay=0.67
   Year 2027 (i=2): LT_decay=0.66, ST_decay=0.33
   Year 2028 (i=3): LT_decay=0.53, ST_decay=0.00
   Year 2029 (i=4): LT_decay=0.43, ST_decay=0.00
   Year 2030 (i=5): LT_decay=0.35, ST_decay=0.00
   ```

8. **Final Forecast**
   ```python
   # Example with +5% average news growth rate
   # News weight: 0.7, Impact multiplier: 1.5, Confidence: 0.5 (fallback)
   
   Year 2025:
   Adjustment = 0.7 × 1.5 × 0.05 × 1.00 × 0.5 = 2.6%
   Final = Baseline × 1.026
   
   Year 2030:
   Adjustment = 0.7 × 1.5 × 0.05 × 0.35 × 0.5 = 0.9%
   Final = Baseline × 1.009
   ```

---

## Key Files and Formulas

### Core Calculation Files

| Component | File | Key Lines | Formula/Purpose |
|-----------|------|-----------|-----------------|
| **Baseline CAGR** | `src/forecasting/models/cagr.py` | 33-45, 63-65 | `CAGR = (End/Start)^(1/n) - 1` |
| **Unified Adjustment** | `src/forecasting/adjustments/unified.py` | 12-62 | Orchestrates all adjustments |
| **News Adjustment** | `src/forecasting/adjustments/news_adjustment.py` | 34-74, 96-138 | `Adj = Weight × Impact × Growth × Decay × Confidence` |
| **Temporal Decay** | `src/forecasting/adjustments/temporal_decay.py` | 27-82 | Exponential decay across forecast horizon |
| **Fallback Strategy** | `src/forecasting/adjustments/fallback_strategies.py` | 38-91 | Confidence reduction logic |
| **GDELT Fallback** | `src/news/gdelt.py` | 346-418 | Article supplementation |

### News Processing Files

| Component | File | Key Lines | Purpose |
|-----------|------|-----------|---------|
| **ISO3→FIPS Mapping** | `src/news/gdelt.py` | 140-151 | Convert country codes |
| **Topic Cleaning** | `src/news/gdelt.py` | 98-119 | Sanitize search terms |
| **Rate Limiter** | `src/news/gdelt.py` | 68-87 | Random delay 1-3 seconds |
| **Article Search** | `src/news/gdelt.py` | 157-191 | GDELT API calls |
| **Multi-Country** | `src/news/gdelt.py` | 277-313 | Parallel country fetching |

### LLM Analysis Files

| Component | File | Key Lines | Purpose |
|-----------|------|-----------|---------|
| **Analyst Class** | `src/llm/analyst.py` | 23-144 | Main analysis orchestration |
| **Single Headline** | `src/llm/analyst.py` | 95-132 | Process one headline |
| **Response Parsing** | `src/llm/analyst.py` | 109-128 | Extract nested JSON response |
| **System Prompt** | `src/llm/prompts/analysis_prompts.py` | 11-81 | Global analysis prompt |
| **Country Prompt** | `src/llm/prompts/analysis_prompts.py` | 84-169 | Country-specific prompt |

---

## UI Display Improvements

### News Headlines Display

**File**: `pages/03_Forecasting.py` (Lines 594-634)

The analyzed headlines display now includes:

1. **Full scrollable dataframe** (400px height) showing all articles
2. **Date column** visible for transparency
3. **Impact distribution statistics**:
   - Total Articles count
   - Positive Impact count (% of total)
   - Neutral/Noise count (% of total)  
   - Negative Impact count (% of total)

This helps users understand why adjustments may be modest - typically most news is classified as "Neutral/Noise" with 0% growth impact.

---

## Configuration Parameters

### Default Values

**File**: Various locations

```python
# Adjustment Weights
NEWS_WEIGHT = 0.7              # 70% weight for news
INDICATOR_WEIGHT = 0.3         # 30% weight for indicators
IMPACT_MULTIPLIER = 1.5        # Amplification factor (configurable 0x-10x in UI)

# Recency Weighting (Single-rate model)
NEWS_HALF_LIFE_DAYS = 90       # 90-day half-life for news recency weighting (matches GDELT window)

# Temporal Decay
TEMPORAL_DECAY_RATE = 0.6      # 60% point for exponential decay across forecast horizon

# Fallback Thresholds
MIN_ARTICLES_THRESHOLD = 5     # Minimum for country analysis
FALLBACK_CONFIDENCE = 0.5      # 50% confidence for global fallback
GDELT_MIN_THRESHOLD = 8        # GDELT's article threshold
GDELT_FILL_LIMIT = 50          # Max global articles to supplement

# GDELT Settings
MAX_ARTICLES_PER_BATCH = 250   # GDELT API limit
BATCH_PARALLEL_WORKERS = 5     # Parallel batches per country
MULTI_COUNTRY_WORKERS = 4      # Parallel countries
RATE_LIMIT_MIN = 1             # Min seconds between requests
RATE_LIMIT_MAX = 3             # Max seconds between requests

# LLM Settings
LLM_MAX_CONCURRENT = 10        # Parallel headline analysis
LLM_REQUEST_DELAY = 0.05       # Delay between LLM calls
LLM_MAX_TOKENS = 150           # Max tokens per response
```

### Configuration Structure

```python
config = {
    'forecast_method': '3-yr CAGR',
    'hist_cutoff': 2024,
    'forecast_until': 2030,
    'adjustment_weights': {
        'news_weight': 0.7,
        'indicator_weight': 0.3
    },
    'temporal_decay': {
        'decay_rate': 0.6,
        'news_half_life_days': 90
    },
    'fallback_strategy': {
        'method': 'use_global_reduced',
        'confidence_multiplier': 0.5
    }
}
```

---

## Debugging Tips

### Check News Fetching
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check ISO3 mapping
from src.news.gdelt import iso3_to_fips
print(f"India ISO3→FIPS: IND → {iso3_to_fips('IND')}")  # Should be 'IN'
```

### Verify LLM Response
```python
# Check raw LLM response
from src.llm.providers.provider_factory import call_llm
response = call_llm("test headline", "test prompt", max_tokens=150)
print(f"Raw response: {response}")
print(f"Parsed: {response.get('response', {})}")  # Note nested structure
```

### Inspect Adjustments
```python
# Check adjustment details
adjusted, details = calculate_unified_adjustment(...)
print(f"LT News: {details['lt_news_adjustment']}%")
print(f"ST News: {details['st_news_adjustment']}%")
print(f"Confidence: {details['news_confidence_multiplier']}")
print(f"Year adjustments: {details['year_adjustments']}")
```

### Monitor Fallbacks
```python
# Check if fallback was used
if details.get('fallback_used'):
    print(f"Fallback: {details['fallback_reason']}")
    print(f"Confidence: {details['news_confidence_multiplier']}")
```

---

## Common Issues and Solutions

### Issue: All news showing 0% impact
**Solution**: Check LLM response parsing
```python
# File: src/llm/analyst.py (Line 110)
parsed_response = llm_response.get('response', {})  # Must extract nested response
```

### Issue: Country not getting news
**Solution**: Verify ISO3→FIPS mapping
```python
# Check config/flat-ui__data*.csv has country mapping
# Verify iso3_to_fips() returns correct FIPS code
```

### Issue: Adjustments not applying
**Solution**: Check confidence multiplier
```python
# If confidence = 0, no adjustments apply
# Check fallback_reason to understand why
```

### Issue: Temporal decay too aggressive
**Solution**: Adjust decay parameters
```python
# File: src/forecasting/adjustments/temporal_decay.py
TEMPORAL_DECAY_RATE = 0.6  # Decrease for slower decay (e.g., 0.8 for gentler decay)
```

### Issue: Recency weighting not working as expected
**Solution**: Check for dates in news data
```python
# Verify dates are present
print(f"Date column exists: {'date' in news_df.columns}")
print(f"Valid dates: {news_df['date'].notna().sum()}/{len(news_df)}")

# Adjust half-life if needed
config['news_half_life_days'] = 120  # 4 months for slower recency decay
```

### Issue: Adjustments seem too small
**Solution**: Check impact distribution
```python
# Most news may be classified as Neutral/Noise
print(f"Positive impact: {(news_df['growth_rate'] > 0).sum()}")
print(f"Neutral (0%): {(news_df['growth_rate'] == 0).sum()}")
print(f"Negative impact: {(news_df['growth_rate'] < 0).sum()}")

# If mostly neutral, adjustments will be small (working as intended)
```
