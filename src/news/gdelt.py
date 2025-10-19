# gdelt.py
# GDELT wrapper using gdeltdoc.
# - Global fetch (no country), country fetch (FIPS-2 via ISO3 conversion)
# - Multi-country fetch via small thread pool
# - Topic OR batching, de-dupe by URL
# - Fallback helper returns (df, weight, strategy)
#
# Deps: pip install gdeltdoc pandas
# Optional (no-hardcode ISO3->FIPS): pip install country_converter  (preferred) or countrycode

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import os
import csv
import random
import logging

import pandas as pd
from src.config.security import configure_requests_ssl_for_gdelt

# Apply SSL configuration for GDELT connections (if disabled by config)
configure_requests_ssl_for_gdelt()

from gdeltdoc import GdeltDoc, Filters

logger = logging.getLogger(__name__)

# -------------------
# Tunables / defaults
# -------------------
BATCH_SIZE = 1
MAX_ARTICLES_PER_BATCH = 250  # API max
MULTI_COUNTRY_MAX_WORKERS = 1  # Increased for parallel countries
GLOBAL_BATCH_WORKERS = 5  # Parallel batch workers for global
BATCH_PARALLEL_WORKERS = 5  # Parallel batches per country
MIN_RATE_LIMIT_SECONDS = 5 # GDELT: min 5 seconds
MAX_RATE_LIMIT_SECONDS = 6 # GDELT: max 10 seconds to appear more human

# Load ISO3 to FIPS mappings from CSV file
def _load_iso3_fips_mapping():
    mapping = {}
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    csv_path = os.path.join(project_root, 'config', 'flat-ui__data-Sun Aug 17 2025.csv')
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                iso3 = row.get('ISO3', '').strip().upper()
                fips = row.get('FIPS', '').strip().upper()
                if iso3 and fips and fips.lower() != 'null':
                    mapping[iso3] = fips
    except Exception:
        pass
    return mapping

_CSV_ISO3_TO_FIPS = _load_iso3_fips_mapping()

# Minimal fallback ISO3→FIPS map for common cases
_FALLBACK_ISO3_TO_FIPS = {
    "USA": "US", "GBR": "UK", "DEU": "GM", "ESP": "SP", "CHN": "CH",
    "JPN": "JA", "KOR": "KS", "NLD": "NL", "CHE": "SZ", "AUS": "AS",
}

# Global rate limiter instance
class GDELTRateLimiter:
    """Thread-safe rate limiter for GDELT API with random delays"""
    
    def __init__(self, min_seconds: float, max_seconds: float):
        self.min_interval = min_seconds
        self.max_interval = max_seconds
        self.last_call = 0
        self.lock = threading.Lock()
    
    def acquire(self):
        """Wait if necessary to respect rate limit with random interval"""
        with self.lock:
            # Determine a random interval for this specific call
            target_interval = random.uniform(self.min_interval, self.max_interval)
            
            elapsed = time.time() - self.last_call
            if elapsed < target_interval:
                wait_time = target_interval - elapsed
                logger.debug(f"Rate limiting - waiting {wait_time:.1f} seconds (randomized interval)")
                time.sleep(wait_time)
            self.last_call = time.time()

# Global rate limiter instance with randomized delay
_rate_limiter = GDELTRateLimiter(MIN_RATE_LIMIT_SECONDS, MAX_RATE_LIMIT_SECONDS)

_gd = GdeltDoc()


# -------------
# Topic helpers
# -------------
def _clean_topics(topics: Iterable[str]) -> List[str]:
    if not topics:
        return []
    
    cleaned = []
    for topic in topics:
        topic_str = str(topic).strip()  # Ensure it's a string
        if not topic_str:
            continue
        
        # Basic sanitization - remove all problematic characters
        sanitized = topic_str.replace('(', '').replace(')', '')
        sanitized = sanitized.replace('[', '').replace(']', '')
        sanitized = sanitized.replace('{', '').replace('}', '')
        sanitized = sanitized.replace('"', '').replace("'", '') # Remove existing quotes
        sanitized = ' '.join(sanitized.split())

        # DON'T add quotes - GDELT handles multi-word phrases in lists without quotes
        if sanitized:
            cleaned.append(sanitized)
    
    logger.debug(f"Cleaned topics: {cleaned}")
    return sorted(set(cleaned))


def _batch_topics(topics: Iterable[str], batch_size: int = BATCH_SIZE) -> List[List[str]]:
    cleaned = _clean_topics(topics)
    return [[topic] for topic in cleaned]


# -------------------------
# Dates (UTC, last N days)
# -------------------------
def _date_bounds_utc(days_back: int = 90) -> Tuple[str, str]:
    end_dt = datetime.now(tz=timezone.utc)
    start_dt = end_dt - timedelta(days=days_back)
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


# ----------------------------------
# ISO3 → FIPS-2 (prefer no hardcode)
# ----------------------------------
def iso3_to_fips(iso3: Optional[str]) -> Optional[str]:
    if not iso3:
        return None
    code = iso3.strip().upper()
    if len(code) == 2:
        return code

    # Primary: Use CSV-loaded mappings
    if code in _CSV_ISO3_TO_FIPS:
        return _CSV_ISO3_TO_FIPS[code]

    return _FALLBACK_ISO3_TO_FIPS.get(code)


# ------------------------
# Core article fetch logic
# ------------------------
def _search_articles_batch(
    keywords: List[str],
    start_date: str,
    end_date: str,
    country_fips: Optional[str],
    max_records: int,
    batch_index: int,
) -> pd.DataFrame:
    if not keywords:
        return pd.DataFrame()

    # Apply rate limiting before making the API call
    _rate_limiter.acquire()
    
    try:
        f = Filters(
            keyword=keywords[0] if len(keywords) == 1 else keywords,  # Single string or list
            start_date=start_date,
            end_date=end_date,
            num_records=max_records,
            country=country_fips,          # FIPS-2, or None for global
        )
        logger.info(f"GDELT query - keywords: {keywords}, country: {country_fips}")
        df = _gd.article_search(f)
    except ValueError as e:
        logger.error(f"GDELT API error: {e}")
        logger.error(f"Problematic keywords: {keywords}")
        # Return empty DataFrame on API error
        return pd.DataFrame()
    if df is None:
        df = pd.DataFrame()
    if df.empty:
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df.get("seendate", pd.Series(dtype=str)), errors="coerce", utc=True)
    df["batch"] = batch_index
    df["query"] = " OR ".join(keywords)
    if country_fips:
        df["country"] = country_fips

    keep = ["title", "date", "url", "sourcecountry", "language"] + (["country"] if country_fips else [])
    keep = [c for c in keep if c in df.columns]
    out = df[keep + ["batch", "query"]].drop_duplicates(subset="url", keep="first")
    return out


def _search_articles(
    topics: Iterable[str],
    days_back: int = 90,
    max_records: int = MAX_ARTICLES_PER_BATCH,
    country_fips: Optional[str] = None,
    max_workers: int = GLOBAL_BATCH_WORKERS,
    progress_callback=None,
) -> pd.DataFrame:
    batches = _batch_topics(topics)
    cols = ["title", "date", "url", "sourcecountry", "language"] + (["country"] if country_fips else [])
    if not batches:
        return pd.DataFrame(columns=cols)

    start_date, end_date = _date_bounds_utc(days_back)

    # Parallel batch processing
    frames: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches to thread pool
        future_to_batch = {
            executor.submit(
                _search_articles_batch,
                kw_batch, start_date, end_date, country_fips, max_records, i
            ): i for i, kw_batch in enumerate(batches)
        }
        
        # Collect results as they complete
        completed_batches = 0
        total_batches = len(batches)
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                part = future.result()
                if not part.empty:
                    frames.append(part)
                completed_batches += 1
                if progress_callback:
                    progress_callback(completed_batches, total_batches, len(frames), country_fips)
            except Exception as e:
                print(f"DEBUG: Batch {batch_idx} failed: {e}")
                completed_batches += 1

    if not frames:
        return pd.DataFrame(columns=cols)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.drop_duplicates(subset="url", keep="first")
    if "date" in all_df.columns:
        all_df = all_df.sort_values("date", ascending=False, na_position="last").reset_index(drop=True)
    return all_df


# -------------------------
# Public API (same as before)
# -------------------------
def fetch_max_gdelt_articles(topics: Iterable[str], days_back: int = 90, progress_callback=None) -> Tuple[pd.DataFrame, bool]:
    df = _search_articles(topics=topics, days_back=days_back, country_fips=None, progress_callback=progress_callback)
    return df, not df.empty


def fetch_country_headlines(
    iso3_code: str,
    topics: Iterable[str],
    days_back: int = 90,
    max_articles: int = MAX_ARTICLES_PER_BATCH,
) -> Tuple[pd.DataFrame, bool]:
    fips = iso3_to_fips(iso3_code)
    df = _search_articles(topics=topics, days_back=days_back, max_records=max_articles, country_fips=fips)
    return df, not df.empty


def fetch_multi_country_headlines(
    country_name_to_iso3: Dict[str, str],
    topics: Iterable[str],
    days_back: int = 90,
    max_articles: int = MAX_ARTICLES_PER_BATCH,
    max_workers: int = MULTI_COUNTRY_MAX_WORKERS,
    progress_callback=None,
) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}

    def _task(name: str, iso3: str) -> Tuple[str, pd.DataFrame]:
        fips = iso3_to_fips(iso3)
        # Use nested parallelism: each country uses parallel batch processing
        df = _search_articles(
            topics=topics, 
            days_back=days_back, 
            max_records=max_articles, 
            country_fips=fips,
            max_workers=BATCH_PARALLEL_WORKERS  # 5 parallel batches per country
        )
        if not df.empty:
            logger.info(f"Successfully fetched {len(df)} articles for {name}.")
            logger.info(f"First 5 articles for {name}:\n{df.head(5).to_string()}")
        else:
            logger.warning(f"No articles found for {name} with the given topics.")
        return name, df

    # Level 1: Process countries in parallel (4 at a time)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_task, name, iso3) for name, iso3 in country_name_to_iso3.items()]
        completed_countries = 0
        total_countries = len(country_name_to_iso3)
        for fut in as_completed(futs):
            try:
                name, df = fut.result()
                results[name] = df
                completed_countries += 1
                if progress_callback:
                    progress_callback(completed_countries, total_countries, name, len(df))
            except Exception as e:
                logger.error(f"Country fetch failed for {name}: {e}")
                results[name] = pd.DataFrame()
                completed_countries += 1

    return results


def get_country_article_summary(country_name: str, articles_df: pd.DataFrame) -> Dict[str, object]:
    if articles_df is None or articles_df.empty:
        return {"Country": country_name, "Articles": 0, "From": None, "To": None, "Unique Topics": 0}

    dates = pd.to_datetime(articles_df["date"], errors="coerce", utc=True) if "date" in articles_df.columns else pd.to_datetime([])
    return {
        "Country": country_name,
        "Articles": int(len(articles_df)),
        "From": (pd.to_datetime(dates.min()).isoformat() if len(dates) else None),
        "To": (pd.to_datetime(dates.max()).isoformat() if len(dates) else None),
        "Unique Topics": int(articles_df["query"].nunique()) if "query" in articles_df.columns else 0,
    }


# ----------------------------
# Fallback (returns 3-tuple)
# ----------------------------
def apply_country_fallback(
    country_name: str,
    country_articles: pd.DataFrame,
    global_articles: Optional[pd.DataFrame] = None,
    min_threshold: int = 8,
    fill_limit: int = 50,
) -> Tuple[pd.DataFrame, float, str]:
    """
    If a country's article set is small, either:
      - (2-arg mode) return (country_df, weight=<n/min_threshold>, strategy='native'|'low_coverage')
      - (3-arg mode) top up from global up to 'min_threshold' and return
        (combined_df, weight=<native_ratio>, strategy='native'|'topped_up'|'low_coverage')

    Weight is intended for downstream blending/penalization.

    Returns: (articles_df, weight, strategy)
    """
    ca = country_articles.copy() if isinstance(country_articles, pd.DataFrame) else pd.DataFrame()

    # Ensure URL and country columns exist for uniformity
    if "url" not in ca.columns and not ca.empty:
        ca["url"] = pd.NA
    if "country" not in ca.columns:
        ca["country"] = pd.NA

    # Sort by date if present
    if "date" in ca.columns:
        ca = ca.sort_values("date", ascending=False, na_position="last").reset_index(drop=True)

    n_native = len(ca)
    if global_articles is None:
        # Weight-only path (no topping up)
        if n_native >= min_threshold:
            return ca, 1.0, "native"
        weight = round(n_native / max(min_threshold, 1), 3)
        return ca, weight, "low_coverage"

    # Topping-up path
    ga = global_articles.copy() if isinstance(global_articles, pd.DataFrame) else pd.DataFrame()
    if "url" not in ga.columns and not ga.empty:
        ga["url"] = pd.NA
    if "date" in ga.columns:
        ga = ga.sort_values("date", ascending=False, na_position="last").reset_index(drop=True)

    # Determine target country code (if present)
    target_country = None
    non_null = ca.get("country", pd.Series(dtype=object)).dropna()
    if not non_null.empty:
        target_country = non_null.iloc[0]

    if n_native >= min_threshold:
        # Enough native: no supplement
        return ca.drop_duplicates(subset="url", keep="first"), 1.0, "native"

    need = max(0, min(min_threshold - n_native, fill_limit))
    # Choose recent globals not in native
    if "url" in ga.columns and "url" in ca.columns:
        mask_new = ~ga["url"].isin(ca["url"])
        supplement = ga[mask_new].head(need).copy()
    else:
        supplement = ga.head(need).copy()

    if target_country is not None and not supplement.empty:
        supplement["country"] = target_country

    out = pd.concat([ca, supplement], ignore_index=True)
    if "url" in out.columns:
        out = out.drop_duplicates(subset="url", keep="first")
    if "date" in out.columns:
        out = out.sort_values("date", ascending=False, na_position="last").reset_index(drop=True)

    total = len(out)
    native_ratio = round(n_native / max(total, 1), 3)

    if not supplement.empty:
        return out, native_ratio, "topped_up"
    else:
        # nothing to add from global
        strategy = "low_coverage"
        return out, native_ratio, strategy


# ------------- 
# Timeline shortcut
# ------------- 
def fetch_timeline(
    topics: Iterable[str],
    days_back: int = 90,
    iso3_code: Optional[str] = None,
    mode: str = "timelinevol",
) -> pd.DataFrame:
    start_date, end_date = _date_bounds_utc(days_back)
    fips = iso3_to_fips(iso3_code) if iso3_code else None

    f = Filters(
        keyword=_clean_topics(topics),
        start_date=start_date,
        end_date=end_date,
        country=fips,
    )
    df = _gd.timeline_search(mode, f)
    if df is None:
        df = pd.DataFrame()
    return df


# ------------- 
# CLI demo
# ------------- 
if __name__ == "__main__":
    demo_topics = ["AI", "GPU", "H100"]
    print("Global (last 90d):")
    gdf, gok = fetch_max_gdelt_articles(demo_topics, days_back=90)
    print(gdf.head(5))

    print("\nGermany (DEU) last 90d:")
    cdf, cok = fetch_country_headlines("DEU", demo_topics, days_back=90)
    print(cdf.head(5))

    print("\nTimeline (global):")
    tdf = fetch_timeline(demo_topics, days_back=30, iso3_code=None)
    print(tdf.head(5))
