"""
Security-related configuration utilities.

Provides a toggle to enable/disable SSL verification for outbound
connections to the GDELT API, controlled via environment variable or
configuration file. Defaults to secure (verification ON).

Environment variable:
  - GDELT_SSL_VERIFY=true|false (default: true)

Optional config file (config/settings.toml):
  [gdelt]
  ssl_verify = true

When disabled, requests to *.gdeltproject.org are made with verify=False
and a clear WARNING is logged exactly once per process.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional


logger = logging.getLogger(__name__)

# Best-effort: load env file if present so GDELT_SSL_VERIFY in config/.env is honored
try:
    from dotenv import load_dotenv  # type: ignore
    if os.path.exists('config/.env'):
        load_dotenv('config/.env')
except Exception:
    pass


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def _read_settings_toml_boolean(section: str, key: str) -> Optional[bool]:
    """Very small TOML boolean reader to avoid new dependencies.

    Looks for a section header [section] and a line like:
      key = true|false

    Returns None if not found or on parse issues.
    """
    settings_path = os.path.join("config", "settings.toml")
    if not os.path.exists(settings_path):
        return None
    try:
        current_section = None
        section_pattern = re.compile(r"^\s*\[(?P<name>[^\]]+)\]\s*$")
        kv_pattern = re.compile(r"^\s*(?P<k>[A-Za-z0-9_]+)\s*=\s*(?P<v>.+?)\s*$")
        with open(settings_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                m = section_pattern.match(line)
                if m:
                    current_section = m.group("name").strip()
                    continue
                if current_section == section:
                    km = kv_pattern.match(line)
                    if km and km.group("k") == key:
                        raw_v = km.group("v").strip()
                        # Strip surrounding quotes if present
                        if raw_v.startswith('"') and raw_v.endswith('"') and len(raw_v) >= 2:
                            v = raw_v[1:-1]
                        elif raw_v.startswith("'") and raw_v.endswith("'") and len(raw_v) >= 2:
                            v = raw_v[1:-1]
                        else:
                            v = raw_v
                        v = v.strip()
                        parsed = _parse_bool(v)
                        return parsed
    except Exception:
        # Fail closed (secure-by-default) by returning None
        return None
    return None


def get_gdelt_ssl_verify() -> bool:
    """Return whether SSL verification should be enforced for GDELT.

    Priority:
      1) GDELT_SSL_VERIFY environment variable
      2) config/settings.toml [gdelt].ssl_verify
      3) default True (secure)
    """
    # Try env var first (dotenv may already be loaded by app code elsewhere)
    env_val = _parse_bool(os.getenv("GDELT_SSL_VERIFY"))
    if env_val is not None:
        return env_val

    # Fallback to config/settings.toml
    file_val = _read_settings_toml_boolean("gdelt", "ssl_verify")
    if file_val is not None:
        return file_val

    # Default secure
    return True


_PATCH_APPLIED = False


def configure_requests_ssl_for_gdelt():
    """If verification is disabled, patch requests to skip SSL verification
    only for hosts under *.gdeltproject.org.

    This keeps the scope as narrow as possible while meeting the need.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    verify = get_gdelt_ssl_verify()
    if verify:
        _PATCH_APPLIED = True
        return

    try:
        import requests  # type: ignore
        import urllib3  # type: ignore
        from requests.sessions import Session

        # Suppress urllib3 InsecureRequestWarning for the specific case
        try:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass

        original_request = Session.request

        def gdelt_aware_request(self, method, url, *args, **kwargs):
            try:
                # Only relax verification for GDELT domains
                if isinstance(url, str) and "gdeltproject.org" in url:
                    kwargs.setdefault("verify", False)
            except Exception:
                # On any error, fall back to original behavior
                pass
            return original_request(self, method, url, *args, **kwargs)

        Session.request = gdelt_aware_request  # type: ignore

        logger.warning(
            "SSL verification for GDELT is DISABLED via configuration. "
            "Requests to *.gdeltproject.org will not verify TLS certificates. "
            "Use only for troubleshooting in controlled environments."
        )
        _PATCH_APPLIED = True
    except Exception:
        # If anything goes wrong, leave default secure behavior in place
        _PATCH_APPLIED = True
        return
