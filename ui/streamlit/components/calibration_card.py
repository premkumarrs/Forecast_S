"""
Streamlit components to display calibration results (Impact & Decay).
"""

from __future__ import annotations

import streamlit as st


def _shorten_reason(text: str, max_words: int = 50) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "…"


def render_calibration_cards(calibration: dict, *, show_global: bool = True, show_countries: bool = True, countries_limit: int | None = None) -> None:
    """Render Global + per-country calibration cards if present.

    Expected structure:
      {
        "global": { long_term_decay_rate, news_avg_pct, reasoning, market_summary },
        "countries": { country: { ...same keys... } }
      }
    """
    if not calibration or not isinstance(calibration, dict):
        st.info("Calibration not available; using default decay rate (0.60).")
        return

    global_cal = calibration.get('global') or {}
    countries_cal = calibration.get('countries') or {}

    # Global card
    if show_global and global_cal:
        st.subheader("🌍 Global Calibration (Decay Rate Only)")
        _render_single_card("Global", global_cal)

    # Countries
    if show_countries and countries_cal:
        st.subheader("🏳️ Country Calibrations")
        items = list(countries_cal.items())
        if countries_limit is not None:
            items = items[:max(0, int(countries_limit))]
        for country, cal in items:
            _render_single_card(country, cal)


def render_calibration_metrics(cal: dict, show_reason: bool = True, note_title: str = "Calibration Reason") -> None:
    """Render a compact row of metrics (Avg Growth and Decay) and optional reason panel."""
    if not cal:
        st.info("Calibration not available; using default decay rate (0.60).")
        return
    dr = float(cal.get('long_term_decay_rate', 0.60))
    avg = float(cal.get('news_avg_pct', 0.0))
    reason_full = str(cal.get('reasoning', ''))

    c1, c2 = st.columns(2)
    with c1:
        st.metric("News Growth Impact", f"{avg:+.2f}%")
    with c2:
        st.metric("Decay Rate", f"{dr:.2f}")

    if show_reason and reason_full:
        # Show complete reasoning (no truncation) with an expander for long text
        if len(reason_full.split()) > 60:
            with st.expander(f"{note_title}"):
                st.write(reason_full)
        else:
            st.info(f"**{note_title}:** {reason_full}")


def _render_single_card(title: str, cal: dict) -> None:
    dr = float(cal.get('long_term_decay_rate', 0.60))
    avg = float(cal.get('news_avg_pct', 0.0))
    reason_full = str(cal.get('reasoning', ''))
    summary = str(cal.get('market_summary', ''))

    with st.container():
        st.markdown(f"**{title}**")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.metric("Avg Growth (%)", f"{avg:+.2f}%")
        with c2:
            st.metric("Decay (0–1)", f"{dr:.2f}")

        if reason_full:
            with st.expander("LLM reasoning", expanded=False):
                st.write(reason_full)
        if summary:
            if hasattr(st, 'popover'):
                with st.popover("Market summary"):
                    st.write(summary)
            else:
                with st.expander("Market summary"):
                    st.write(summary)
