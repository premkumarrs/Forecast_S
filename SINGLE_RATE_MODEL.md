# Single-Rate Model Implementation

## Overview
This document describes the simplified single-rate model that replaces the previous dual LT/ST (Long-term/Short-term) approach. The new model uses a single news growth rate with exponential decay over the forecast horizon.

## Formula

### Total Adjustment Formula
```
Total_Adjustment(year) = 
    News_Weight × Impact_Multiplier × Temporal_Decay(year) × News_Average
  + Indicator_Weight × Impact_Multiplier × Indicator_Adjustment
```

Where:
- **News_Average**: Single recency-weighted average of news growth rates
- **Temporal_Decay(year)**: Exponential decay function over forecast horizon
- **News_Weight**: Weight for news adjustment (default: 0.7)
- **Impact_Multiplier**: Amplification factor (default: 1.5)
- **Indicator_Weight**: Weight for indicator adjustment (default: 0.3)

## Key Changes from LT/ST Model

### Removed
- ❌ `temporal_impact` field (long_term/short_term/no_effect)
- ❌ Separate LT and ST adjustments
- ❌ `lt_half_life_days` and `st_half_life_days` parameters
- ❌ `LT_WEIGHT` and `ST_WEIGHT` configuration
- ❌ Dual decay curves (exponential + linear)

### Added/Kept
- ✅ Single `news_pct` per year
- ✅ `news_half_life_days` for recency weighting (default: 90)
- ✅ `long_term_decay_rate` for horizon decay (default: 0.6)
- ✅ Single exponential decay over forecast period

## Configuration

### settings.toml
```toml
[news]
# Single recency weighting parameter
news_half_life_days = 90   # Half-life for article recency weighting

# Temporal decay over forecast horizon
long_term_decay_rate = 0.6  # Decay halves at 60% of forecast period
```

### Adjustment Weights
```python
adjustment_weights = {
    'news_weight': 0.7,         # Weight for news adjustment
    'indicator_weight': 0.3,    # Weight for indicator adjustment  
    'impact_multiplier': 1.5    # Amplification factor
}
```

## Implementation Details

### 1. News Analysis (LLM)
The LLM now returns a simplified JSON structure:
```json
{
  "category": "Growth",
  "growth_rate": 5.0,
  "reason": "Positive market expansion"
}
```
Note: No `temporal_impact` field

### 2. News Adjustment Calculation
```python
# Calculate single recency-weighted average
news_avg = recency_weighted_mean(news_data, news_half_life_days)

# Apply formula for each forecast year
for year in forecast_years:
    decay = exponential_decay(year_index, total_years, decay_rate)
    news_term = news_weight * impact_multiplier * news_avg * decay
    value_hat *= (1 + news_term)
```

### 3. Temporal Decay
Single exponential decay function:
```python
decay = 0.5 ** (year_index / (total_years * decay_rate))
```

At `decay_rate = 0.6`:
- Year 0: 100% impact
- Year 3 (60% of 5-year forecast): ~50% impact
- Year 5: ~31% impact

## Data Structure Changes

### Year Adjustments (New Format)
```python
year_adjustments = {
    2024: {
        'news_pct': 4.5,        # Single news percentage
        'news_avg_pct': 5.0,    # Average growth rate
        'decay': 1.0,           # Decay factor for this year
        'indicators_pct': 2.0,  # Indicator adjustment
        'total_pct': 6.5        # Total adjustment
    }
}
```

### Backwards Compatibility
The system handles old LT/ST format gracefully:
```python
# Old format
old_data = {'lt_news_pct': 3.0, 'st_news_pct': 2.0}

# Compatibility layer combines them
news_pct = old_data.get('news_pct', 
           old_data.get('lt_news_pct', 0) + 
           old_data.get('st_news_pct', 0))
# Result: news_pct = 5.0
```

## UI Changes

### Table Columns
- Before: "LT News (%)", "ST News (%)", "Indicators (%)"
- After: "News (%)", "Indicators (%)"

### Metrics
- Before: "Long-term Impact", "Short-term Impact"
- After: "Average News Impact" with tooltip explaining decay

### Filters
- Removed: "Temporal Impact" multiselect
- Kept: Search and Impact Range filters

## Testing

### Unit Tests
Run the test suite:
```bash
python tests/test_single_rate_model.py
```

Tests verify:
1. Parser ignores `temporal_impact` field
2. News adjustment formula is correct
3. Decay decreases over forecast years
4. No LT/ST references in output
5. Backwards compatibility with old data

### UI Smoke Test
Manual checklist:
- [ ] Tables show "News (%)" column
- [ ] No "Temporal Impact" filter
- [ ] Metrics show single news impact
- [ ] Tooltips mention recency-weighted average
- [ ] Old cached results don't crash

## Tuning Guidelines

### News Half-Life Days
Controls recency weighting for articles:
- **Lower (30-60)**: Recent news dominates
- **Default (90)**: Balanced weighting over 3 months
- **Higher (120-180)**: More equal weighting

### Long-Term Decay Rate
Controls decay over forecast horizon:
- **Lower (0.3-0.5)**: Faster decay, news impact fades quickly
- **Default (0.6)**: Balanced decay, 50% at 60% of horizon
- **Higher (0.7-0.9)**: Slower decay, sustained impact

### Example Configurations

**Conservative (rapid decay)**:
```toml
news_half_life_days = 60
long_term_decay_rate = 0.4
```

**Balanced (default)**:
```toml
news_half_life_days = 90
long_term_decay_rate = 0.6
```

**Aggressive (sustained impact)**:
```toml
news_half_life_days = 120
long_term_decay_rate = 0.8
```

## Migration Checklist

- [x] Update LLM prompts to remove temporal_impact
- [x] Update parser to exclude temporal_impact
- [x] Refactor NewsAdjustment to single rate
- [x] Update unified.py formatter
- [x] Fix UI components (tables, metrics, filters)
- [x] Update configuration files
- [x] Add backwards compatibility
- [x] Create unit tests
- [x] Document changes

## Benefits

1. **Simpler Model**: One growth rate instead of two
2. **Clearer Logic**: Single decay curve, easier to explain
3. **Fewer Parameters**: Reduced configuration complexity
4. **Better UX**: Cleaner UI without confusing LT/ST distinctions
5. **Maintained Accuracy**: Preserves recency weighting and horizon decay