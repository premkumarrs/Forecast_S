"""
Decay Calibrator using existing LLM provider.

Uses ProviderFactory.call_llm(prompt, system_prompt, max_tokens) to produce
calibrated values for:
  - long_term_decay_rate (0.10–0.90, prior 0.60)

Also exposes a recency-weighted average helper consistent with existing pages.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.llm.providers.provider_factory import call_llm

logger = logging.getLogger(__name__)


GLOBAL_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'prompts', 'calibration', 'global_impact_decay.prompt'
)
COUNTRY_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'prompts', 'calibration', 'country_impact_decay.prompt'
)


def _load_template(path: str) -> str:
    try:
        with open(os.path.abspath(path), 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to load template at {path}: {e}")
        return ""


def _fill_template(tpl: str, values: Dict[str, object]) -> str:
    out = tpl
    for k, v in values.items():
        out = out.replace(f"{{{{{k}}}}}", str(v))
    return out


def recency_weighted_avg_pct(df: pd.DataFrame, half_life_days: int = 90) -> float:
    """Recency-weighted average of headline growth_rate with half-life in days.

    Returns a percentage value consistent with existing adjustment logic
    (e.g., 1.2 for +1.2%).
    """
    if df is None or df.empty or 'growth_rate' not in df.columns or 'date' not in df.columns:
        return 0.0
    # Align with NewsAdjustment: use relevant==1 if present and non-zero growth only
    filtered = df.copy()
    if 'relevant' in filtered.columns:
        try:
            filtered = filtered[filtered['relevant'] == 1]
        except Exception:
            pass
    filtered = filtered[filtered['growth_rate'] != 0]
    x = pd.to_numeric(filtered['growth_rate'], errors='coerce')
    d = pd.to_datetime(filtered['date'], errors='coerce')
    m = x.notna() & d.notna()
    if not m.any():
        return float(np.nan_to_num(x.mean(), nan=0.0))
    age = (pd.Timestamp.utcnow() - d[m]).dt.days.clip(lower=0)
    hl = max(1, int(half_life_days))
    w = np.power(0.5, age / hl)
    return float(np.average(x[m], weights=w))


class DecayCalibrator:
    PRIOR_DECAY = 0.65
    DECAY_BOUNDS = (0.10, 0.90)

    def __init__(self, default_forecast_years: int = 5):
        self.default_forecast_years = max(1, int(default_forecast_years))
        self._global_tpl = _load_template(GLOBAL_TEMPLATE_PATH)
        self._country_tpl = _load_template(COUNTRY_TEMPLATE_PATH)

    def _harmonize_reason(self, text: str, decay: float) -> str:
        """Replace any numeric mentions of decay with the applied values.

        This keeps the UI note consistent with the post-shrink, post-clamp
        values shown in metrics and used in adjustments.
        """
        try:
            import re
            t = text or ""
            # decay rate mentions
            t = re.sub(r"((?:long[\s-]*term\s+)?decay\s*rate\s*(?:of|=)?\s*)([-+]?\d+(?:\.\d+)?)",
                       lambda m: f"{m.group(1)}{decay:.2f}", t, flags=re.IGNORECASE)
            return t
        except Exception:
            return text

    def _postprocess(self, data: Dict[str, object]) -> Dict[str, object]:
        # Extract and coerce
        dr_raw = self._to_float(data.get('long_term_decay_rate'), self.PRIOR_DECAY)

        # Gentle shrink (10%) toward priors to keep LLM signal influential
        dr = 0.90 * dr_raw + 0.10 * self.PRIOR_DECAY

        # Clamp
        dr = float(min(max(dr, self.DECAY_BOUNDS[0]), self.DECAY_BOUNDS[1]))

        reason = str(data.get('reasoning', '')).strip()
        reason = self._harmonize_reason(reason, dr)
        return {
            'market_summary': str(data.get('market_summary', '')).strip(),
            'long_term_decay_rate': dr,
            'reasoning': reason
        }

    @staticmethod
    def _to_float(v, default: float) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)

    def _safe_call(self, prompt: str) -> Dict[str, object]:
        try:
            resp = call_llm(prompt=prompt, system_prompt="", max_tokens=350)
            data = resp.get('response', {}) if isinstance(resp, dict) else {}
            raw = resp.get('raw_response', '') if isinstance(resp, dict) else ''
            # Normalize string payloads
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except Exception:
                    logger.warning("Calibration LLM returned non-JSON string; falling back to priors.")
                    data = {}
            # If parsed dict lacks required keys, treat as failure
            if not isinstance(data, dict) or 'long_term_decay_rate' not in data:
                if isinstance(data, dict) and 'text' in data:
                    logger.warning("Calibration response had no structured fields; falling back to priors.")
                if raw:
                    logger.debug(f"Calibration raw response: {raw[:400]}")
                return {}
            return data
        except Exception as e:
            logger.error(f"LLM calibration call failed: {e}")
            return {}

    def calibrate_global(self, market: str, news_avg_pct: float, forecast_years: Optional[int] = None, method: Optional[str] = None) -> Dict[str, object]:
        tpl = self._global_tpl
        fy = int(forecast_years) if forecast_years and forecast_years > 0 else self.default_forecast_years
        # Convert decimal to percent for the prompt context
        prompt = _fill_template(tpl, {
            'market': market,
            'method': method or 'Global Only',
            'news_avg_pct': round(news_avg_pct, 2),
            'forecast_years': fy,
        })
        data = self._safe_call(prompt)
        if not data:
            data = {
                'market_summary': '',
                'long_term_decay_rate': self.PRIOR_DECAY,
                'reasoning': 'Fallback to priors due to missing response.'
            }
        return self._postprocess(data)

    def calibrate_country(self, market: str, country: str, news_avg_pct: float, forecast_years: Optional[int] = None, method: Optional[str] = None) -> Dict[str, object]:
        tpl = self._country_tpl
        fy = int(forecast_years) if forecast_years and forecast_years > 0 else self.default_forecast_years
        prompt = _fill_template(tpl, {
            'market': market,
            'method': method or 'Country-Specific',
            'country': country,
            'news_avg_pct': round(news_avg_pct, 2),
            'forecast_years': fy,
        })
        data = self._safe_call(prompt)
        if not data:
            data = {
                'market_summary': '',
                'long_term_decay_rate': self.PRIOR_DECAY,
                'reasoning': 'Fallback to priors due to missing response.'
            }
        return self._postprocess(data)
