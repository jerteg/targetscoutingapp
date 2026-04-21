"""
shared/season_filter.py
Herbruikbaar seizoen + minuten filter component.
Gebruik: season, min_minutes = render_season_filter()
         data = load_season_data(season, min_minutes)
"""

import streamlit as st
from shared.data_processing import SEASON_LABELS, load_season_data


# Standaard minimum minuten opties
MIN_MINUTES_OPTIONS = [0, 200, 400, 600, 800, 900, 1000, 1200, 1500, 2000]
MIN_MINUTES_DEFAULT = 900   # jij kiest zelf wat betrouwbaar is


def render_season_filter(
    key_prefix: str = "",
    default_season: str = "2025/26",
    default_min_min: int = MIN_MINUTES_DEFAULT,
    show_divider: bool = True,
) -> tuple:
    """
    Rendert seizoen-selector en minuten-slider in de sidebar.
    Geeft (season, min_minutes) terug.

    Parameters
    ----------
    key_prefix      : uniek prefix voor session_state keys
    default_season  : standaard seizoen
    default_min_min : standaard minimum minuten
    show_divider    : toon <hr> erboven
    """
    if show_divider:
        st.markdown(
            '<hr style="border:none;border-top:0.5px solid rgba(255,255,255,0.1);'
            'margin:4px 0 10px;">',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<span class="sb-section-label">Season</span>',
        unsafe_allow_html=True,
    )
    season = st.radio(
        f"{key_prefix}_season",
        SEASON_LABELS,
        index=SEASON_LABELS.index(default_season) if default_season in SEASON_LABELS else 0,
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown(
        '<span class="sb-section-label">Min. minutes played</span>',
        unsafe_allow_html=True,
    )
    min_minutes = st.select_slider(
        f"{key_prefix}_min_min",
        options=MIN_MINUTES_OPTIONS,
        value=default_min_min,
        label_visibility="collapsed",
        format_func=lambda x: f"{x}+" if x > 0 else "All",
    )

    return season, min_minutes


def render_season_filter_compare(key: str, label: str, default_season: str = "2025/26") -> str:
    """
    Compacte seizoen-selector voor de Compare pagina (per speler apart).
    Geeft het gekozen seizoen terug.
    """
    st.markdown(
        f'<span class="sb-section-label">{label}</span>',
        unsafe_allow_html=True,
    )
    return st.radio(
        f"season_{key}",
        SEASON_LABELS,
        index=SEASON_LABELS.index(default_season) if default_season in SEASON_LABELS else 0,
        horizontal=True,
        label_visibility="collapsed",
    )
