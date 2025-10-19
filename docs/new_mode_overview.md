# Existing Forecast + News Adjustment Mode

Reuse an external or pre-existing baseline forecast and apply news-driven adjustments—no model fitting required inside the app.

## When to Use
- You already have a market baseline from another source.
- Indicators aren’t needed or available, but you want news impacts.
- You need a quick sensitivity run on existing forecasts.

## Differences vs. Classic Mode
| Capability | Classic Forecast + News Adjustment | Existing Forecast + News Adjustment |
|------------|------------------------------------|-------------------------------------|
| Baseline generation | Model-based (CAGR / ETS / Logistic) | Uses extracted market series exactly |
| Indicators | Optional (0–100% weighting) | Disabled (news locked at 100%) |
| Adjustment window | Full forecast horizon | User-defined start & end years |
| Methods available | Global, Top-Down, Bottom-Up, Country-Specific | Global, Bottom-Up, Country-Specific |

## Workflow
1. **Configuration**
   - Choose the new mode in the Mode dropdown.
   - Set `Adjustment Start Year` and `Adjustment End Year` for the news window.
   - Indicators are hidden; news weight is fixed at 100%.

2. **Data Extraction**
   - Run the unified extraction as usual. Market data will include all historical years plus forecast years up to the selected horizon.

3. **Forecasting**
   - The header shows `Existing forecast (news-only)` for the baseline.
   - Running the forecast applies news impact only within the chosen year window.

4. **Insights & Export**
   - Output schemas remain unchanged. Baseline values trace back to the extracted series.

## Validation Guarantees
- BaselineModelFactory is bypassed in this mode.
- Indicators are ignored.
- Adjustment weights are news-only, constrained to the selected year window.
- Top-Down is intentionally unavailable.
