import os
import sys

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd

from shared.data_processing import preprocess_data
from shared.templates import position_groups
from shared.styles import BASE_CSS, header_logo_html

st.set_page_config(
    page_title="Target Scouting",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(BASE_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
.block-container { padding-top: 2rem !important; max-width: 1060px !important; }

/* Equal-height metric cards */
.metric-card {
    background: #ffffff;
    border: 0.5px solid #e0d8cc;
    border-top: 2px solid #c9a84c;
    border-radius: 6px;
    padding: 16px 18px;
    height: 100%;
}
.metric-num {
    font-size: 28px;
    font-weight: 700;
    color: #111827;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 6px;
    font-family: 'DM Sans', sans-serif;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.metric-lbl {
    font-size: 9px;
    color: #111827;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 3px;
}
.metric-sub {
    font-size: 9px;
    color: #b0a898;
    font-family: 'JetBrains Mono', monospace;
}

/* Tool group label */
.tool-group-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #b0a898;
    margin: 24px 0 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.tool-group-label::after {
    content: '';
    flex: 1;
    height: 0.5px;
    background: #e0d8cc;
}

/* Tool cards */
.tool-card {
    background: #ffffff;
    border: 0.5px solid #e0d8cc;
    border-radius: 8px;
    padding: 14px 14px 0;
    display: flex;
    flex-direction: column;
    height: 100%;
}
.tool-card:hover { border-color: #c9a84c; }
.tool-icon-box {
    width: 30px; height: 30px;
    border-radius: 5px; background: #111827;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 8px;
}
.tool-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #b0a898; margin-bottom: 3px;
}
.tool-name {
    font-size: 13px; font-weight: 700; color: #111827;
    margin-bottom: 4px; letter-spacing: -0.01em;
}
.tool-desc {
    font-size: 11px; color: #7a7060;
    line-height: 1.5; flex: 1;
    margin-bottom: 10px;
}

/* Open buttons — gold, full width, flush to bottom of card */
div[data-testid="stButton"] > button {
    background: #c9a84c !important;
    color: #111827 !important;
    border: none !important;
    border-radius: 0 0 7px 7px !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    width: 100% !important;
    padding: 9px 0 !important;
    margin-top: 0 !important;
}
div[data-testid="stButton"] > button:hover {
    background: #b8963e !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "shared", "data.csv"))
    df = preprocess_data(df)
    df["Main Position"] = df["Position"].astype(str).str.split(",").str[0].str.strip()
    return df

data      = load_data()
total_pl  = len(data)
total_lg  = data["League"].nunique() if "League" in data.columns else 19
pos_count = len(position_groups)
data_date = "26 Mar 2026"

# ── Header ──────────────────────────────────────────────────────────────────
logo = header_logo_html(44)
st.markdown(f"""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:28px;
            padding-bottom:20px;border-bottom:0.5px solid #e0d8cc;">
    {logo}
    <div>
        <div style="font-size:20px;font-weight:700;color:#111827;letter-spacing:-0.01em;
                    font-family:'DM Sans',sans-serif;">Target Scouting</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#b0a898;
                    text-transform:uppercase;letter-spacing:0.1em;margin-top:3px;">
            Football intelligence · Wyscout {data_date}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Metric cards ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, num, lbl, sub in [
    (c1, f"{total_pl:,}", "Players",         "All competitions"),
    (c2, str(total_lg),   "Leagues",          "Top 5 + Next 14"),
    (c3, str(pos_count),  "Position groups",  "RB · CB · LB · DM · CM · AM · W · ST"),
    (c4, data_date,       "Dataset updated",  "Wyscout 2025/26"),
]:
    with col:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-num">{num}</div>
            <div class="metric-lbl">{lbl}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

# ── Tool definitions ──────────────────────────────────────────────────────────
ICON = lambda path: f'<svg width="16" height="16" viewBox="0 0 18 18" fill="none">{path}</svg>'

TOOL_GROUPS = [
    {
        "label": "Analysis",
        "tools": [
            ("Ranking",          "scout", "Rank players by role, custom stats and weights. Build shortlists.",
             "pages/1_Ranking.py", "btn_r",
             ICON('<rect x="2" y="5" width="14" height="2" rx="1" fill="#c9a84c"/><rect x="2" y="9" width="10" height="2" rx="1" fill="#c9a84c" opacity="0.65"/><rect x="2" y="13" width="6" height="2" rx="1" fill="#c9a84c" opacity="0.35"/>')),
            ("Player Dashboard", "scout", "Full scout view: bars, radar, similar players, scatter and trend.",
             "pages/5_Dashboard.py", "btn_db",
             ICON('<circle cx="9" cy="9" r="7" stroke="#c9a84c" stroke-width="1.5"/><circle cx="9" cy="9" r="2.5" fill="#c9a84c"/><line x1="9" y1="2" x2="9" y2="6.5" stroke="#c9a84c" stroke-width="1.2"/><line x1="9" y1="11.5" x2="9" y2="16" stroke="#c9a84c" stroke-width="1.2"/><line x1="2" y1="9" x2="6.5" y2="9" stroke="#c9a84c" stroke-width="1.2"/><line x1="11.5" y1="9" x2="16" y2="9" stroke="#c9a84c" stroke-width="1.2"/>')),
            ("Compare",          "scout", "Two players side-by-side: mirrored bars, key stats, overlay radar.",
             "pages/8_Compare.py", "btn_cmp",
             ICON('<rect x="1" y="3" width="6" height="12" rx="1.2" stroke="#c9a84c" stroke-width="1.4" fill="none"/><rect x="11" y="3" width="6" height="12" rx="1.2" stroke="#c9a84c" stroke-width="1.4" fill="none"/><line x1="8" y1="9" x2="10" y2="9" stroke="#c9a84c" stroke-width="1.4"/>')),
            ("Development",      "track", "Multi-season ratings, biggest risers & fallers, player trend profiles.",
             "pages/9_Development.py", "btn_dev",
             ICON('<polyline points="2,13 6,8 10,10 16,4" stroke="#c9a84c" stroke-width="1.5" fill="none" stroke-linejoin="round"/><circle cx="6" cy="8" r="1.3" fill="#c9a84c"/><circle cx="10" cy="10" r="1.3" fill="#c9a84c"/><circle cx="16" cy="4" r="1.3" fill="#c9a84c"/>')),
        ]
    },
    {
        "label": "Visualisation",
        "tools": [
            ("Radar",        "chart", "Performance as a radar chart. Single player or head-to-head.",
             "pages/2_Radar.py", "btn_rd",
             ICON('<polygon points="9,2 16,14 2,14" stroke="#c9a84c" stroke-width="1.5" fill="none"/><line x1="9" y1="2" x2="9" y2="9" stroke="#c9a84c" stroke-width="1" opacity="0.5"/><line x1="9" y1="9" x2="16" y2="14" stroke="#c9a84c" stroke-width="1" opacity="0.5"/><line x1="9" y1="9" x2="2" y2="14" stroke="#c9a84c" stroke-width="1" opacity="0.5"/>')),
            ("Player Card",  "chart", "Full percentile breakdown per category. Export as HTML card.",
             "pages/3_Player_Card.py", "btn_pc",
             ICON('<rect x="2" y="3" width="14" height="12" rx="1.5" stroke="#c9a84c" stroke-width="1.5"/><line x1="6" y1="3" x2="6" y2="15" stroke="#c9a84c" stroke-width="0.7" opacity="0.4"/><line x1="11" y1="3" x2="11" y2="15" stroke="#c9a84c" stroke-width="0.7" opacity="0.4"/><line x1="2" y1="8" x2="16" y2="8" stroke="#c9a84c" stroke-width="0.7" opacity="0.4"/>')),
            ("Scatter Plot", "chart", "Plot two stats against each other. Highlight any player.",
             "pages/4_Scatter.py", "btn_sc",
             ICON('<circle cx="4" cy="14" r="1.8" fill="#c9a84c"/><circle cx="9" cy="8.5" r="1.8" fill="#c9a84c"/><circle cx="14" cy="4" r="1.8" fill="#c9a84c"/><circle cx="6" cy="11" r="1.3" fill="#c9a84c" opacity="0.4"/><circle cx="12" cy="6.5" r="1.3" fill="#c9a84c" opacity="0.4"/>')),
        ]
    },
    {
        "label": "Data & Reports",
        "tools": [
            ("Database",       "data",   "Browse all 5,500+ players. Filter by position, club, league, age.",
             "pages/6_Database.py", "btn_dbase",
             ICON('<ellipse cx="9" cy="5" rx="7" ry="2.5" stroke="#c9a84c" stroke-width="1.4" fill="none"/><path d="M2 5v4c0 1.38 3.13 2.5 7 2.5s7-1.12 7-2.5V5" stroke="#c9a84c" stroke-width="1.4" fill="none"/><path d="M2 9v4c0 1.38 3.13 2.5 7 2.5s7-1.12 7-2.5V9" stroke="#c9a84c" stroke-width="1.4" fill="none"/>')),
            ("Report Builder", "report", "Build a custom scout report with radar, bars, scatter and notes.",
             "pages/7_Report.py", "btn_rep",
             ICON('<rect x="3" y="2" width="12" height="14" rx="1.5" stroke="#c9a84c" stroke-width="1.4" fill="none"/><line x1="6" y1="6" x2="12" y2="6" stroke="#c9a84c" stroke-width="1.1"/><line x1="6" y1="9" x2="12" y2="9" stroke="#c9a84c" stroke-width="1.1" opacity="0.6"/><line x1="6" y1="12" x2="9" y2="12" stroke="#c9a84c" stroke-width="1.1" opacity="0.35"/>')),
        ]
    },
]

# ── Render tool groups ────────────────────────────────────────────────────────
for group in TOOL_GROUPS:
    tools = group["tools"]
    st.markdown(f'<div class="tool-group-label">{group["label"]}</div>', unsafe_allow_html=True)
    cols = st.columns(len(tools))
    for col, (name, tag, desc, page, key, icon) in zip(cols, tools):
        with col:
            st.markdown(f"""<div class="tool-card">
                <div class="tool-icon-box">{icon}</div>
                <div class="tool-tag">{tag}</div>
                <div class="tool-name">{name}</div>
                <div class="tool-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)
            if st.button("Open →", key=key, use_container_width=True):
                st.switch_page(page)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:32px;padding-top:14px;border-top:0.5px solid #e0d8cc;
            display:flex;justify-content:space-between;align-items:center;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#b0a898;
                text-transform:uppercase;letter-spacing:0.1em;">
        Target Scouting · Data: Wyscout 26 Mar 2026
    </span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#b0a898;
                text-transform:uppercase;letter-spacing:0.1em;">
        Built with Streamlit
    </span>
</div>
""", unsafe_allow_html=True)

# Glossary as footer text link
_, mid, _ = st.columns([3, 1, 3])
with mid:
    if st.button("Glossary →", key="btn_gl", use_container_width=True):
        st.switch_page("pages/10_Glossary.py")
