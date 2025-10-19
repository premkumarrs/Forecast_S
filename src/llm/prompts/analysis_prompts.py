"""
Analysis prompt templates for market and news analysis.
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def build_system_prompt(market_name: str, categories: List[Dict[str, Any]], forecast_years: Optional[int] = None) -> str:
    """
    Builds the system prompt for the LLM analyst with category-specific growth constraints.
    
    Args:
        market_name: Name of the market being analyzed
        categories: List of category definitions with growth constraints
        forecast_years: Number of years in the forecast horizon
        
    Returns:
        System prompt string
    """
    
    # Build a clean category list for insertion into the prompt (no markdown)
    lines = []
    for cat in categories:
        name = str(cat.get('name', 'Unnamed')).strip()
        desc = str(cat.get('description', 'No description')).strip()
        cmin = cat.get('growth_constraint_min', -10)
        cmax = cat.get('growth_constraint_max', 10)
        lines.append(f"- {name}: {desc} (min={cmin}%, max={cmax}%)")
    category_list_here = "\n".join(lines) if lines else "- Neutral/Noise: No meaningful market impact (min=0%, max=0%)"

    # Optional horizon hint (kept short); this template focuses on calibrated output
    horizon_hint = ""
    if forecast_years:
        horizon_hint = f"\nNote: Forecast horizon is {forecast_years} years; calibrate annual growth impact accordingly.\n"

    # Compose calibrated global prompt using concatenation to avoid brace formatting issues
    prompt = "You are a senior market analyst for the global " + market_name + " market.\n\n"
    prompt += "TASK\n"
    prompt += "Given ONE news headline (and optional snippet/date), decide:\n"
    prompt += "1) which category it belongs to from the list below,\n"
    prompt += "2) the directional annual growth impact (growth_rate, in percent) for the GLOBAL market.\n\n"

    prompt += "HARD CONSTRAINTS\n"
    prompt += "- Select exactly one \"category\" from CATEGORIES below.\n"
    prompt += "- \"growth_rate\" MUST be a number within that category’s min/max bounds.\n"
    prompt += "- Return ONLY valid JSON with fields: category, growth_rate, reason.\n"
    prompt += "- Do NOT include a percent sign; use a number (e.g., 3.5 for +3.5%).\n"
    prompt += "- If the headline is irrelevant to " + market_name + ", use the \"Neutral/Noise\" category and set growth_rate at its allowed value (often 0).\n\n"

    prompt += "CALIBRATION HEURISTICS (combat underestimation)\n"
    prompt += "- If evidence is clearly positive but numeric size is uncertain, choose the MID–UPPER quartile of the allowed band.\n"
    prompt += "- Magnitude cues → map to allowed band:\n"
    prompt += "  • Global-scale capex, record earnings, major regulatory greenlights, multi-year mega-contracts, hyperscaler/infra expansions → UPPER quartile.\n"
    prompt += "  • Material partnerships, notable product launches, new regions, sizeable customer wins → MID band.\n"
    prompt += "  • Early pilots, hiring, soft signals → LOWER band (but not 0 if directional).\n"
    prompt += "- If signal is positive but weak/ambiguous, choose a SMALL positive in the 20–30th percentile of the band (not 0).\n"
    prompt += "- Negative news (bans, large outages, major price cuts, supply shocks, adverse regulation) → LOWER quartile of the negative direction within the category’s range.\n"
    prompt += "- Scope check (global realism): If impact is localized to a small region or niche provider, reduce magnitude one tier (e.g., from upper → mid band). If it plausibly affects top-share regions/providers, keep the higher tier.\n\n"

    prompt += "RECENCY & QUALITY CUES (tie-breakers only)\n"
    prompt += "- More recent, multi-source, or quantified items (e.g., “$X bn”) justify ticking magnitude upward within the chosen tier.\n"
    prompt += "- Opinion pieces or vague commentary justify ticking magnitude downward within the tier (but avoid defaulting to 0 if directional).\n\n"

    prompt += "CATEGORIES (allowed ranges)\n"
    prompt += category_list_here + "\n"
    prompt += "  - Each item has: name, description, growth_constraint_min, growth_constraint_max.\n"
    prompt += horizon_hint + "\n"

    prompt += "OUTPUT (JSON only; no markdown, no extra text)\n"
    prompt += "{\n"
    prompt += "  \"category\": \"<one name from the list>\",\n"
    prompt += "  \"growth_rate\": <number within that category’s range>,\n"
    prompt += "  \"reason\": \"<<=20 words citing the signal and why that magnitude tier was chosen>\"\n"
    prompt += "}\n"

    return prompt


def build_country_system_prompt(market_name: str, country: str, categories: List[Dict[str, Any]], forecast_years: Optional[int] = None) -> str:
    """
    Builds a country-specific system prompt for the LLM analyst.
    
    Args:
        market_name: Name of the market being analyzed
        country: Name of the country for localized context
        categories: List of category definitions
        forecast_years: Number of years in the forecast horizon
        
    Returns:
        Country-specific system prompt string
    """
    
    # Build a clean category list for insertion into the prompt
    lines = []
    for cat in categories:
        name = str(cat.get('name', 'Unnamed')).strip()
        desc = str(cat.get('description', 'No description')).strip()
        cmin = cat.get('growth_constraint_min', -10)
        cmax = cat.get('growth_constraint_max', 10)
        lines.append(f"- {name}: {desc} (min={cmin}%, max={cmax}%)")
    category_list_here = "\n".join(lines) if lines else "- Neutral/Noise: No meaningful market impact (min=0%, max=0%)"

    # Optional horizon hint for country calibration
    horizon_hint = ""
    if forecast_years:
        horizon_hint = f"\nNote: Calibrate annual growth impact for {country} across a {forecast_years}-year horizon.\n"

    # Calibrated country-specific prompt
    prompt = "You are a senior country analyst for " + market_name + " in " + country + ".\n\n"
    prompt += "TASK\n"
    prompt += "Given ONE news headline (and optional snippet/date), decide:\n"
    prompt += "1) which category it belongs to from the list below,\n"
    prompt += "2) the directional annual growth impact (growth_rate, in percent) for " + country + " specifically.\n\n"

    prompt += "COUNTRY CONTEXT (use for magnitude realism; do NOT invent numbers)\n"
    prompt += "- Market maturity: if known (e.g., nascent / developing / mature); otherwise be conservative.\n"
    prompt += "- Relevance hints: local regulation, domestic capex, facility openings, country-named contracts, macro shifts (inflation, FX, GDP), provider entries/expansions into " + country + ".\n"
    prompt += "- If the headline is global or regional only, infer plausible spillover to " + country + " (lower magnitude than a direct local event unless " + country + " is explicitly central).\n\n"

    prompt += "HARD CONSTRAINTS\n"
    prompt += "- Select exactly one \"category\" from CATEGORIES below.\n"
    prompt += "- \"growth_rate\" MUST be a number within that category’s min/max bounds.\n"
    prompt += "- Return ONLY valid JSON with fields: category, growth_rate, reason.\n"
    prompt += "- Do NOT include a percent sign; use a number (e.g., 2.0 for +2.0%).\n"
    prompt += "- If the headline is irrelevant to " + market_name + " or " + country + ", use \"Neutral/Noise\" and its allowed value (often 0).\n\n"

    prompt += "CALIBRATION HEURISTICS (country mapping; avoid underestimation)\n"
    prompt += "- Local, concrete positives (domestic regulatory approval, data center build, factory, spectrum award, major client win in " + country + ") → UPPER quartile of the allowed band.\n"
    prompt += "- Regional/global positives that plausibly benefit " + country + " (provider enters region, regional capex with probable local footprint) → MID band.\n"
    prompt += "- Soft/local signals (hiring, pilot, MoU) → LOWER band (not 0 if directional).\n"
    prompt += "- If signal is positive but weak/ambiguous, choose a SMALL positive in the 20–30th percentile of the band.\n"
    prompt += "- Clear local negatives (ban, tax burden, devaluation shock, outage, contract loss) → LOWER quartile of the negative direction within the category’s range.\n"
    prompt += "- Spillover rule: If the headline targets a neighboring country/region only, apply a SMALL spillover effect (lower band) unless explicitly connected to " + country + ".\n\n"

    prompt += "RECENCY & QUALITY CUES (tie-breakers only)\n"
    prompt += "- Recent, quantified, multi-source → tick magnitude upward within the chosen tier.\n"
    prompt += "- Vague, single-source opinion → tick downward within the tier.\n\n"

    prompt += "CATEGORIES (allowed ranges)\n"
    prompt += category_list_here + "\n"
    prompt += "  - Each item has: name, description, growth_constraint_min, growth_constraint_max.\n"
    prompt += horizon_hint + "\n"

    prompt += "OUTPUT (JSON only; no markdown, no extra text)\n"
    prompt += "{\n"
    prompt += "  \"category\": \"<one name from the list>\",\n"
    prompt += "  \"growth_rate\": <number within that category’s range>,\n"
    prompt += "  \"reason\": \"<<=20 words citing the country linkage and why that tier>\"\n"
    prompt += "}\n"

    return prompt