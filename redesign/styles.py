"""
Shared styles — Redesigned for Target Scouting v2
Inspired by PVP Explorer editorial aesthetic:
  - DM Sans 400/500/700 + JetBrains Mono for labels/data
  - Warm off-white palette, navy sidebar, gold accent
  - Tighter, more intentional typographic hierarchy
"""

import os
import base64

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _logo_b64() -> str:
    path = os.path.join(BASE_DIR, "assets", "TS_Logo.png")
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

def header_logo_html(size: int = 52) -> str:
    b64 = _logo_b64()
    if b64:
        return (f'<img src="data:image/png;base64,{b64}" '
                f'style="width:{size}px;height:{size}px;border-radius:10px;object-fit:cover;flex-shrink:0;" />')
    return (f'<div style="width:{size}px;height:{size}px;background:#111827;border-radius:10px;'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:11px;font-weight:700;color:#c9a84c;flex-shrink:0;'
            f'font-family:\'JetBrains Mono\',monospace;letter-spacing:0.05em;">TS</div>')

def sidebar_logo_html(size: int = 28) -> str:
    b64 = _logo_b64()
    if b64:
        return (f'<img src="data:image/png;base64,{b64}" '
                f'style="width:{size}px;height:{size}px;border-radius:5px;object-fit:cover;flex-shrink:0;" />')
    return (f'<div style="width:{size}px;height:{size}px;background:#1a2240;border-radius:5px;'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:8px;font-weight:700;color:#c9a84c;flex-shrink:0;'
            f'font-family:\'JetBrains Mono\',monospace;">TS</div>')

BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* ── Page background ── */
[data-testid="stAppViewContainer"] { background-color: #faf7f2 !important; }
[data-testid="stHeader"]           { background-color: #faf7f2 !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #111827 !important;
    padding-top: 0 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* Sidebar text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label {
    color: rgba(255,255,255,0.7) !important;
    font-size: 13px !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #c9a84c !important; }

/* Sidebar dropdowns */
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="select"] > div > div,
section[data-testid="stSidebar"] [data-baseweb="select"] span {
    background-color: rgba(255,255,255,0.07) !important;
    color: rgba(255,255,255,0.85) !important;
    border-color: rgba(255,255,255,0.12) !important;
    border-radius: 4px !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
    color: rgba(255,255,255,0.85) !important;
}
section[data-testid="stSidebar"] input {
    background-color: rgba(255,255,255,0.07) !important;
    color: rgba(255,255,255,0.85) !important;
    border-color: rgba(255,255,255,0.12) !important;
    border-radius: 4px !important;
}

/* ── st.page_link nav styling ── */
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {
    background: transparent !important;
    border-radius: 0 4px 4px 0 !important;
    border-left: 2px solid transparent !important;
    padding: 6px 12px !important;
    color: rgba(255,255,255,0.45) !important;
    font-size: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    text-decoration: none !important;
    text-transform: uppercase !important;
    display: block !important;
    margin-bottom: 1px !important;
    transition: all 0.12s !important;
}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {
    background: rgba(255,255,255,0.05) !important;
    color: rgba(255,255,255,0.75) !important;
}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current="page"] {
    background: rgba(201,168,76,0.12) !important;
    border-left-color: #c9a84c !important;
    color: #c9a84c !important;
}

/* Hide default Streamlit nav */
[data-testid="stSidebarNav"] { display: none !important; }

/* ── Main content ── */
.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

/* ── Page title ── */
h1 {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #111827 !important;
    letter-spacing: -0.01em !important;
}
h2 {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #111827 !important;
}
h3 {
    font-size: 14px !important;
    font-weight: 600 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab"] {
    color: #7a7060 !important;
    font-size: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding: 10px 18px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
    color: #111827 !important;
    font-weight: 700 !important;
    border-bottom-color: #c9a84c !important;
}

/* ── Buttons ── */
div[data-testid="stButton"] > button {
    background: #111827 !important;
    color: #c9a84c !important;
    border: none !important;
    border-radius: 5px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    transition: background 0.12s !important;
}
div[data-testid="stButton"] > button:hover { background: #1a2240 !important; }

/* ── Home button (gold outline) ── */
.home-btn div[data-testid="stButton"] > button {
    background: transparent !important;
    border: 0.5px solid rgba(201,168,76,0.4) !important;
    color: #c9a84c !important;
    font-size: 11px !important;
    padding: 5px 10px !important;
    width: 100% !important;
    margin-bottom: 6px !important;
}
.home-btn div[data-testid="stButton"] > button:hover {
    background: rgba(201,168,76,0.1) !important;
}

/* ── Download buttons ── */
div[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: 0.5px solid #c9a84c !important;
    color: #c9a84c !important;
    border-radius: 5px !important;
    font-size: 11px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: rgba(201,168,76,0.1) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 0.5px solid #e0d8cc !important;
    border-radius: 6px !important;
}

/* ── Divider ── */
hr { border-color: #e0d8cc !important; }

/* ── Sidebar section label (mono, gold, uppercase) ── */
.sb-section-label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 9px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: rgba(201,168,76,0.55) !important;
    padding: 0 0 3px 0 !important;
    margin-top: 14px !important;
    display: block !important;
}
.sb-section-label::after {
    content: '';
    display: block;
    width: 20px;
    height: 1px;
    background: rgba(201,168,76,0.25);
    margin-top: 4px;
}

/* ── Sidebar brand block ── */
.sb-brand {
    padding: 16px 14px 12px;
    border-bottom: 0.5px solid rgba(255,255,255,0.07);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sb-brand-name {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #c9a84c !important;
    line-height: 1.35 !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.01em !important;
    text-transform: uppercase !important;
}

/* ── Metric card ── */
.metric-card {
    background: #ffffff;
    border: 0.5px solid #e0d8cc;
    border-top: 2px solid #c9a84c;
    border-radius: 6px;
    padding: 16px 18px;
}
.metric-num {
    font-size: 32px;
    font-weight: 700;
    color: #111827;
    letter-spacing: -0.02em;
    line-height: 1;
    margin-bottom: 6px;
    font-family: 'DM Sans', sans-serif;
}
.metric-lbl {
    font-size: 10px;
    color: #111827;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 3px;
}
.metric-sub {
    font-size: 10px;
    color: #b0a898;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Tool cards ── */
.tool-card {
    background: #ffffff;
    border: 0.5px solid #e0d8cc;
    border-radius: 8px;
    padding: 16px 16px 12px;
    display: flex;
    flex-direction: column;
    transition: border-color 0.15s;
}
.tool-card:hover { border-color: #c9a84c; }
.tool-icon-box {
    width: 32px; height: 32px;
    border-radius: 6px; background: #111827;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 10px; flex-shrink: 0;
}
.tool-name {
    font-size: 13px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 5px;
    letter-spacing: -0.01em;
}
.tool-desc {
    font-size: 11px;
    color: #7a7060;
    line-height: 1.55;
    flex: 1;
    font-family: 'DM Sans', sans-serif;
}
.tool-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #b0a898;
    margin-bottom: 6px;
    margin-top: -2px;
}

/* ── Section label (main content) ── */
.section-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    font-weight: 700;
    color: #b0a898;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-lbl::after {
    content: '';
    flex: 1;
    height: 0.5px;
    background: #e0d8cc;
}
</style>
"""

def sidebar_brand_block(logo_html: str) -> str:
    return f"""
<div class="sb-brand">
    {logo_html}
    <span class="sb-brand-name">Target<br>Scouting</span>
</div>
"""
