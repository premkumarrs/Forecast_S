"""
Point-in-time helpers for historical news evaluation.

Historical as-of time: for an annual fold with origin year Y, news counts as
available if it is dated on or before ``Y-12-31 23:59:59`` (naive UTC; GDELT
dates have second precision). ``news_as_of`` is the single definition used by
the backtest and by ``fetch_headlines_for_origins``.

Two layers have very different leakage properties:

1. Historical news retrieval and windowing (point-in-time safe): only
   headlines dated within ``[as_of - lookback_days, as_of]`` are used, and
   recency weights and category windows are measured from ``as_of``, never
   from the current date.
2. Historical LLM interpretation (potentially hindsight-contaminated): the
   ``growth_rate``/``category`` of each headline comes from an LLM. A modern
   LLM analysing a 2018 headline may know what happened after 2018.

This evaluation prevents temporal leakage in news retrieval and
recency/window calculations, but historical interpretation by a modern LLM may
contain hindsight because the model may know events that occurred after the
historical origin. Results are therefore a retrospective evaluation, not a
perfect historical replay.

Headline analysis is cached so each headline is sent to the LLM at most once
per analysis configuration. The cache key includes an analysis fingerprint
(system prompt, model, prompt version), so changing any of them invalidates
earlier results instead of silently reusing them.
"""

import hashlib
import json
import math
import os
from typing import Callable, Dict, Iterable, Mapping, Optional

import pandas as pd

from ..adjustments.news_adjustment import DEFAULT_NEWS_LOOKBACK_DAYS

NEWS_ANALYSIS_COLUMNS = ["title", "date", "category", "growth_rate", "reason", "relevant"]

# SimpleLLMAnalyst returns this reason when an LLM call fails; such rows are
# defaults, not analyses, and must not be cached or scored as zero impact.
ANALYST_FAILURE_REASON = "Analysis failed - using defaults"


class HeadlineAnalysisError(RuntimeError):
    """Raised when a headline could not be analysed."""


def news_as_of(origin: int) -> pd.Timestamp:
    """Latest timestamp at which news is available for a fold with this origin year."""
    if isinstance(origin, bool) or not isinstance(origin, int):
        raise TypeError(f"origin must be an integer year, got {origin!r}")
    return pd.Timestamp(year=origin, month=12, day=31, hour=23, minute=59, second=59)


def analysis_fingerprint(system_prompt: str, model: str = "", prompt_version: str = "") -> str:
    """Identify an analysis configuration; any change yields a new fingerprint."""
    payload = json.dumps(
        {"system_prompt": system_prompt, "model": model, "prompt_version": prompt_version},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _date_key(value) -> str:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    return "" if pd.isna(ts) else ts.isoformat()


class HeadlineAnalysisCache:
    """Deterministic headline-analysis cache, in memory or backed by a JSON file.

    Keep cache files out of version control (e.g. under ``.forecast_cache/``).
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._entries: Dict[str, Dict] = {}
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self._entries = json.load(f)

    @staticmethod
    def make_key(headline: Mapping, fingerprint: str) -> str:
        payload = json.dumps(
            {
                "title": str(headline.get("title", "")),
                "url": str(headline.get("url", "") or ""),
                "date": _date_key(headline.get("date")),
                "analysis": fingerprint,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict]:
        entry = self._entries.get(key)
        return dict(entry) if entry is not None else None

    def put(self, key: str, analysis: Mapping) -> None:
        self._entries[key] = dict(analysis)

    def save(self) -> None:
        if not self.path:
            return
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, sort_keys=True)

    def __len__(self) -> int:
        return len(self._entries)


def _validated_analysis(raw, title: str) -> Dict:
    if not isinstance(raw, Mapping):
        raise HeadlineAnalysisError(f"Analysis for {title!r} is not a mapping: {raw!r}")
    if raw.get("reason") == ANALYST_FAILURE_REASON:
        raise HeadlineAnalysisError(f"LLM analysis failed for {title!r}")
    try:
        growth_rate = float(raw["growth_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HeadlineAnalysisError(f"Analysis for {title!r} has no numeric growth_rate") from exc
    if not math.isfinite(growth_rate):
        raise HeadlineAnalysisError(f"Analysis for {title!r} has non-finite growth_rate")
    return {
        "category": raw.get("category", ""),
        "growth_rate": growth_rate,
        "reason": raw.get("reason", ""),
        "relevant": int(raw.get("relevant", 1 if growth_rate != 0 else 0)),
    }


def analyze_headlines_cached(
    headlines: pd.DataFrame,
    analyze_fn: Callable[[Dict], Mapping],
    cache: HeadlineAnalysisCache,
    fingerprint: str,
) -> pd.DataFrame:
    """Analyse headlines, calling ``analyze_fn`` only for uncached ones.

    ``analyze_fn`` receives a headline dict (``title``, ``date``, ...) and
    returns the analyst's mapping (``category``, ``growth_rate``, ``reason``,
    ``relevant``); e.g.
    ``lambda h: analyst._analyze_single_headline(h, analyst.system_prompt)``
    with ``fingerprint=analysis_fingerprint(analyst.system_prompt, model)``.
    ``title`` and ``date`` always come from the source headline.

    Failed analyses are never cached and raise ``HeadlineAnalysisError``
    rather than being turned into zero impact; results analysed before the
    failure stay cached, so a retry does not repeat them.
    """
    if headlines is None or headlines.empty:
        return pd.DataFrame(columns=NEWS_ANALYSIS_COLUMNS)
    if "title" not in headlines.columns or "date" not in headlines.columns:
        raise ValueError("headlines must have 'title' and 'date' columns")

    rows = []
    for headline in headlines.to_dict("records"):
        key = cache.make_key(headline, fingerprint)
        analysis = cache.get(key)
        if analysis is None:
            analysis = _validated_analysis(analyze_fn(dict(headline)), headline["title"])
            cache.put(key, analysis)
        rows.append({"title": headline["title"], "date": headline["date"], **analysis})
    return pd.DataFrame(rows, columns=NEWS_ANALYSIS_COLUMNS)


def fetch_headlines_for_origins(
    origins: Iterable[int],
    fetch_fn: Callable[..., object],
    lookback_days: int = DEFAULT_NEWS_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Fetch the news window ending at each origin's ``news_as_of``.

    ``fetch_fn`` is called as ``fetch_fn(days_back=..., end_date=...)`` and may
    return a DataFrame or a ``(DataFrame, ok)`` tuple, e.g.
    ``functools.partial(fetch_max_gdelt_articles, topics)``. Results are
    concatenated and de-duplicated by URL (or title and date).
    """
    frames = []
    for origin in sorted(set(origins)):
        fetched = fetch_fn(days_back=lookback_days, end_date=news_as_of(origin))
        frame = fetched[0] if isinstance(fetched, tuple) else fetched
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["title", "date"])
    combined = pd.concat(frames, ignore_index=True)
    subset = ["url"] if "url" in combined.columns else ["title", "date"]
    return combined.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
