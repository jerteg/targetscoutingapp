"""
sidebar_nav.py — call render_sidebar_nav(active) inside every page's `with st.sidebar:` block.
Uses st.page_link() for real clickable navigation (requires Streamlit 1.27+).
"""

import streamlit as st
from shared.styles import sidebar_logo_html, sidebar_brand_block


def render_sidebar_nav(active: str = ""):
    """
    Renders logo, brand name, and clickable nav links via st.page_link().
    active: one of 'ranking', 'radar', 'card', 'scatter', 'dashboard'
    """
    logo = sidebar_logo_html(28)
    st.markdown(sidebar_brand_block(logo), unsafe_allow_html=True)

    # Nav links via st.page_link — these ARE clickable
    pages = [
        ("app",         "app.py",                 "🏠  Home"),
        ("ranking",     "pages/1_Ranking.py",     "   Ranking"),
        ("radar",       "pages/2_Radar.py",        "   Radar"),
        ("card",        "pages/3_Player_Card.py",  "   Player Card"),
        ("scatter",     "pages/4_Scatter.py",      "   Scatter"),
        ("dashboard",   "pages/5_Dashboard.py",    "   Dashboard"),
        ("database",    "pages/6_Database.py",     "   Database"),
        ("report",      "pages/7_Report.py",       "   Report Builder"),
        ("compare",     "pages/8_Compare.py",      "   Compare"),
        ("development", "pages/9_Development.py",  "   Development"),
        ("glossary",    "pages/10_Glossary.py",    "   Glossary"),
    ]

    for key, path, label in pages:
        st.page_link(path, label=label)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.markdown('<hr style="border:none;border-top:0.5px solid rgba(255,255,255,0.1);margin:4px 0 10px;">', unsafe_allow_html=True)
