"""Download the raw indicator source files for the Census indicator experiment.

Saves unmodified responses under ``data/evaluation/raw/indicators/`` and writes
``fetch_log.json`` with URL, retrieval time, HTTP Last-Modified and SHA-256 for
each file. The BEA NIPA annual flat file is stored gzip-compressed (the SHA-256
is of the uncompressed bytes as served).

Usage:
    python scripts/fetch_indicator_sources.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "evaluation" / "raw" / "indicators"
HEADERS = {"User-Agent": "Mozilla/5.0 (research data retrieval)"}

SOURCES = [
    ("bea_NipaDataA.txt.gz", "https://apps.bea.gov/national/Release/TXT/NipaDataA.txt", True, True),
    ("bea_SeriesRegister.txt", "https://apps.bea.gov/national/Release/TXT/SeriesRegister.txt", False, True),
    ("bea_TablesRegister.txt", "https://apps.bea.gov/national/Release/TXT/TablesRegister.txt", False, True),
    ("worldbank_IT.NET.USER.ZS_USA.json",
     "https://api.worldbank.org/v2/country/USA/indicator/IT.NET.USER.ZS?format=json&per_page=200", False, True),
    ("worldbank_IT.NET.USER.ZS_USA_footnotes.json",
     "https://api.worldbank.org/v2/country/USA/indicator/IT.NET.USER.ZS?format=json&per_page=200&footnote=y",
     False, True),
    ("worldbank_IT.NET.USER.ZS_metadata.json",
     "https://api.worldbank.org/v2/sources/2/series/IT.NET.USER.ZS/metadata?format=json", False, True),
    ("fred_DGDSRC1A027NBEA.csv",
     "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGDSRC1A027NBEA", False, False),
]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log = {}
    for name, url, compress, required in SOURCES:
        try:
            response = requests.get(url, timeout=180, headers=HEADERS)
            response.raise_for_status()
        except Exception as exc:
            if required:
                raise
            print(f"optional source skipped: {url} ({exc})")
            log[name] = {"url": url, "status": f"not retrieved: {type(exc).__name__}"}
            continue
        content = response.content
        path = RAW_DIR / name
        path.write_bytes(gzip.compress(content, mtime=0) if compress else content)
        log[name] = {
            "url": url,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "http_last_modified": response.headers.get("Last-Modified"),
            "bytes_as_served": len(content),
            "sha256_as_served": hashlib.sha256(content).hexdigest(),
            "stored_gzip": compress,
            "status": "retrieved",
        }
        print(f"{name}: {len(content)} bytes, sha256 {log[name]['sha256_as_served']}")
    (RAW_DIR / "fetch_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
