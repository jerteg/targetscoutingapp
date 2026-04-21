import os, sys, math
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from shared.data_processing import load_season_data, SEASON_LABELS
from shared.scoring import compute_scores
from shared.styles import BASE_CSS
from shared.sidebar_nav import render_sidebar_nav
from shared.templates import (
    position_groups, position_category_weights,
    LEAGUE_MULTIPLIERS_ALL, LEAGUE_DISPLAY_NAMES,
    TOP5_LEAGUES, NEXT14_LEAGUES,
)

st.set_page_config(
    page_title="Development · Target Scouting",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(BASE_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
.block-container { padding-top:1.6rem !important; max-width:1280px !important; }
.section-lbl {
    font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 700;
    color: #b0a898; text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 12px; display: flex; align-items: center; gap: 10px;
}
.section-lbl::after { content: ''; flex: 1; height: 0.5px; background: #e0d8cc; }
.delta-pos { font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #1a7a45; }
.delta-neg { font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #c0392b; }
.delta-neu { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #b0a898; }
.rf-card {
    background: #fff; border: 0.5px solid #e0d8cc; border-radius: 8px;
    padding: 11px 14px; margin-bottom: 7px;
    display: flex; justify-content: space-between; align-items: center;
}
.rf-name { font-size: 13px; font-weight: 700; color: #111827; }
.rf-meta { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #b0a898; margin-top: 2px; }
.rf-scores { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #7a7060; margin-top: 3px; }
</style>
""", unsafe_allow_html=True)

def _c(v):
    if v >= 75: return "#1a7a45"
    elif v >= 50: return "#91cf60"
    elif v >= 25: return "#f0a500"
    return "#d73027"

CATS = ["Goalscoring", "Chance creation", "Dribbling", "Passing", "Defending"]
CAT_SHORT = {
    "Goalscoring": "Goals", "Chance creation": "Chance cr.",
    "Dribbling": "Dribbling", "Passing": "Passing", "Defending": "Defending",
}

DECAY_WEIGHTS = {
    2: [0.65, 0.35], 3: [0.55, 0.30, 0.15],
    4: [0.50, 0.25, 0.15, 0.10], 5: [0.45, 0.25, 0.15, 0.10, 0.05],
}

@st.cache_data
def load_all_seasons(min_minutes: int = 900):
    return {s: load_season_data(s, min_minutes) for s in SEASON_LABELS}

@st.cache_data
def score_season(_data, pg, league_template, score_mode, season_key):
    return compute_scores(_data, pg, league_template, score_mode)

for k, v in [("dev_player", None), ("dev_team", None), ("dev_pg", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

with st.sidebar:
    render_sidebar_nav("development")
    st.markdown(
        '<hr style="border:none;border-top:0.5px solid rgba(255,255,255,0.1);margin:4px 0 10px;">',
        unsafe_allow_html=True,
    )
    st.markdown('<span class="sb-section-label">Min. minutes played</span>', unsafe_allow_html=True)
    min_minutes = st.select_slider(
        "dev_min_min",
        options=[0, 200, 400, 600, 800, 900, 1000, 1200, 1500, 2000],
        value=900, label_visibility="collapsed",
        format_func=lambda x: f"{x}+" if x > 0 else "All",
    )
    st.markdown('<span class="sb-section-label">League template</span>', unsafe_allow_html=True)
    league_template = st.radio(
        "dev_lt", ["Top 5 leagues", "Next 14 competitions", "Both"],
        label_visibility="collapsed",
    )
    st.markdown('<span class="sb-section-label">Score type</span>', unsafe_allow_html=True)
    score_mode = st.radio(
        "dev_sm", ["Adjusted (recommended)", "Model (raw)"],
        label_visibility="collapsed",
    )
    st.markdown('<span class="sb-section-label">Position group</span>', unsafe_allow_html=True)
    pg_list = list(position_groups.keys())
    pre_pg  = st.session_state.get("dev_pg") or pg_list[0]
    position_group = st.selectbox(
        "dev_pg_sel", pg_list,
        index=pg_list.index(pre_pg) if pre_pg in pg_list else 0,
        label_visibility="collapsed",
    )
    st.markdown('<span class="sb-section-label">Age range</span>', unsafe_allow_html=True)
    age_min, age_max = st.slider("dev_age", 15, 45, (16, 35), label_visibility="collapsed")
    st.markdown('<span class="sb-section-label">League filter</span>', unsafe_allow_html=True)
    if league_template == "Top 5 leagues":    lg_pool = sorted(TOP5_LEAGUES)
    elif league_template == "Next 14 competitions": lg_pool = sorted(NEXT14_LEAGUES)
    else: lg_pool = sorted(TOP5_LEAGUES | NEXT14_LEAGUES)
    league_filter = st.selectbox("dev_lgf", ["All"] + lg_pool, label_visibility="collapsed")

seasons_data = load_all_seasons(min_minutes)
n_seasons    = len(seasons_data)
weights      = DECAY_WEIGHTS.get(n_seasons, DECAY_WEIGHTS[2])

scored_by_season = {}
for s, df_s in seasons_data.items():
    sc = score_season(df_s, position_group, league_template, score_mode, season_key=f"{s}_{league_template}_{position_group}")
    if not sc.empty:
        sc["_season"] = s
        scored_by_season[s] = sc

if not scored_by_season:
    st.warning("No data available for the selected filters.")
    st.stop()

latest_season = SEASON_LABELS[0]
base_df = scored_by_season.get(latest_season, pd.DataFrame())
if base_df.empty:
    st.warning(f"No data for {latest_season}.")
    st.stop()

if league_filter != "All":
    base_df = base_df[base_df["League"] == league_filter]
base_df = base_df[base_df["Age"].between(age_min, age_max)]

def compute_multi_score(player, team, season_order, scored_by_season, weights):
    scores = []
    for s in season_order:
        pool = scored_by_season.get(s, pd.DataFrame())
        if pool.empty: scores.append(None); continue
        mask = (pool["Player"] == player) & (pool["Team within selected timeframe"] == team)
        scores.append(float(pool[mask]["overall_adj"].iloc[0]) if mask.any() else None)
    valid = [(w, s) for w, s in zip(weights, scores) if s is not None]
    if not valid: return None, {}
    total_w  = sum(w for w, _ in valid)
    ms_score = sum(w * s for w, s in valid) / total_w
    detail   = {SEASON_LABELS[i]: s for i, s in enumerate(scores) if s is not None}
    return round(ms_score, 1), detail

season_order = SEASON_LABELS
rows = []
for _, row in base_df.iterrows():
    ms, detail = compute_multi_score(
        row["Player"], row["Team within selected timeframe"],
        season_order, scored_by_season, weights,
    )
    r = row.to_dict()
    r["ms_score"]    = ms if ms is not None else row["overall_adj"]
    r["n_seasons"]   = len(detail)
    r["score_detail"] = detail
    rows.append(r)

display_df = pd.DataFrame(rows)
display_df = display_df.sort_values("ms_score", ascending=False).reset_index(drop=True)

prev_season = SEASON_LABELS[1] if len(SEASON_LABELS) > 1 else None
if prev_season and prev_season in scored_by_season:
    prev_pool = scored_by_season[prev_season]
    curr_pool = scored_by_season[latest_season]
    prev_dict = {(r["Player"], r["Team within selected timeframe"]): r["overall_adj"] for _, r in prev_pool.iterrows()}
    curr_dict = {(r["Player"], r["Team within selected timeframe"]): r["overall_adj"] for _, r in curr_pool.iterrows()}

    def _delta(row):
        key = (row["Player"], row["Team within selected timeframe"])
        c   = curr_dict.get(key); p = prev_dict.get(key)
        return round(c - p, 1) if c is not None and p is not None else None
    display_df["delta"] = display_df.apply(_delta, axis=1)
else:
    display_df["delta"] = None

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:20px;">
  <h1 style="font-size:20px;font-weight:700;letter-spacing:-0.01em;">Development & Multi-Season Rating</h1>
  <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#b0a898;
              text-transform:uppercase;letter-spacing:0.1em;margin-top:4px;">
    {position_group} · {" + ".join(season_order)} ·
    Weights: {" / ".join(f"{int(w*100)}%" for w in weights[:len(season_order)])} ·
    {len(display_df)} players
  </div>
</div>
""", unsafe_allow_html=True)

tab_ms, tab_dev, tab_player = st.tabs(["Multi-Season Ranking", "Top Risers & Fallers", "Player Profile"])

# ── Tab 1: Multi-Season Ranking ────────────────────────────────────────────────
with tab_ms:
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#7a7060;margin-bottom:14px;line-height:1.6;">'
        'Weighted average across available seasons. '
        '<span style="color:#1a7a45;">&#x2191;</span> positive delta = improved vs. previous season. '
        '<span style="color:#c0392b;">&#x2193;</span> negative delta = declined.'
        '</div>',
        unsafe_allow_html=True,
    )

    search = st.text_input("", placeholder="Search player…", key="dev_search", label_visibility="collapsed")
    show_df = display_df.copy()
    if search:
        show_df = show_df[show_df["Player"].str.contains(search, case=False, na=False)]

    rows_html = ""
    for i, (_, row) in enumerate(show_df.head(50).iterrows()):
        ms    = row["ms_score"]
        delta = row.get("delta")
        lg    = LEAGUE_DISPLAY_NAMES.get(row.get("League",""), row.get("League",""))
        age   = row.get("Age","-")
        team  = row.get("Team within selected timeframe","-")

        if delta is not None:
            if delta > 0:
                delta_html = f'<span class="delta-pos">&#x2191; +{delta:.1f}</span>'
            elif delta < 0:
                delta_html = f'<span class="delta-neg">&#x2193; {delta:.1f}</span>'
            else:
                delta_html = f'<span class="delta-neu">&#x2014;</span>'
        else:
            delta_html = '<span class="delta-neu">&#x2014;</span>'

        seasons_html = ""
        for sn, sv in row.get("score_detail", {}).items():
            seasons_html += (
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                f'color:#b0a898;margin-right:8px;">'
                f'{sn[:4]}: <b style="color:#111827;">{sv:.0f}</b></span>'
            )

        rows_html += f"""
        <tr>
          <td style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#b0a898;padding:9px 10px;">{i+1}</td>
          <td style="padding:9px 10px;">
            <div style="font-weight:700;font-size:13px;">{row['Player']}</div>
            <div style="margin-top:2px;">{seasons_html}</div>
          </td>
          <td style="color:#7a7060;font-size:12px;padding:9px 10px;">{team}</td>
          <td style="padding:9px 10px;"><span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#b0a898;">{lg}</span></td>
          <td style="font-family:'JetBrains Mono',monospace;font-size:12px;padding:9px 10px;">{age:.0f}</td>
          <td style="padding:9px 10px;">{delta_html}</td>
          <td style="padding:9px 10px;">
            <span style="background:{_c(ms)};color:white;padding:3px 9px;border-radius:5px;
                         font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;">{ms:.0f}</span>
          </td>
        </tr>"""

    table_html = f"""
    <div style="background:#fff;border:0.5px solid #e0d8cc;border-radius:8px;overflow:hidden;">
    <table style="width:100%;border-collapse:collapse;font-size:12px;font-family:'DM Sans',sans-serif;">
    <thead><tr style="background:#f0ebe2;">
      <th style="font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:700;text-transform:uppercase;
                 letter-spacing:0.1em;color:#b0a898;padding:9px 10px;text-align:left;border-bottom:1px solid #e0d8cc;">#</th>
      <th style="font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:700;text-transform:uppercase;
                 letter-spacing:0.1em;color:#b0a898;padding:9px 10px;text-align:left;border-bottom:1px solid #e0d8cc;">Player</th>
      <th style="font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:700;text-transform:uppercase;
                 letter-spacing:0.1em;color:#b0a898;padding:9px 10px;text-align:left;border-bottom:1px solid #e0d8cc;">Club</th>
      <th style="font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:700;text-transform:uppercase;
                 letter-spacing:0.1em;color:#b0a898;padding:9px 10px;text-align:left;border-bottom:1px solid #e0d8cc;">Competition</th>
      <th style="font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:700;text-transform:uppercase;
                 letter-spacing:0.1em;color:#b0a898;padding:9px 10px;text-align:left;border-bottom:1px solid #e0d8cc;">Age</th>
      <th style="font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:700;text-transform:uppercase;
                 letter-spacing:0.1em;color:#b0a898;padding:9px 10px;text-align:left;border-bottom:1px solid #e0d8cc;">&#x394; vs prev</th>
      <th style="font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:700;text-transform:uppercase;
                 letter-spacing:0.1em;color:#b0a898;padding:9px 10px;text-align:left;border-bottom:1px solid #e0d8cc;">MS Score</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
    </table></div>"""
    st.markdown(table_html, unsafe_allow_html=True)

# ── Tab 2: Risers & Fallers ────────────────────────────────────────────────────
with tab_dev:
    if "delta" not in display_df.columns or display_df["delta"].isna().all():
        st.info("Insufficient seasons available for delta calculation.")
    else:
        valid_delta = display_df.dropna(subset=["delta"])
        col_rise, col_fall = st.columns(2)

        with col_rise:
            st.markdown(
                '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;font-weight:700;'
                'color:#1a7a45;margin-bottom:12px;text-transform:uppercase;letter-spacing:0.08em;">'
                '&#x2191; Top 15 Risers</div>',
                unsafe_allow_html=True
            )
            for _, r in valid_delta.nlargest(15, "delta").iterrows():
                lg    = LEAGUE_DISPLAY_NAMES.get(r.get("League",""), r.get("League",""))
                ms    = r["ms_score"]
                delta = r["delta"]
                curr  = r.get("overall_adj", ms)
                prev_sc = curr - delta
                st.markdown(
                    f'<div class="rf-card">'
                    f'<div>'
                    f'<div class="rf-name">{r["Player"]}</div>'
                    f'<div class="rf-meta">{r["Team within selected timeframe"]} &middot; {lg} &middot; {r["Age"]:.0f} yrs</div>'
                    f'<div class="rf-scores">{prev_sc:.0f} &#x2192; <b style="color:#111827;">{curr:.0f}</b> (MS: {ms:.0f})</div>'
                    f'</div>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:18px;font-weight:700;color:#1a7a45;">+{delta:.1f}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with col_fall:
            st.markdown(
                '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;font-weight:700;'
                'color:#c0392b;margin-bottom:12px;text-transform:uppercase;letter-spacing:0.08em;">'
                '&#x2193; Top 15 Fallers</div>',
                unsafe_allow_html=True
            )
            for _, r in valid_delta.nsmallest(15, "delta").iterrows():
                lg    = LEAGUE_DISPLAY_NAMES.get(r.get("League",""), r.get("League",""))
                ms    = r["ms_score"]
                delta = r["delta"]
                curr  = r.get("overall_adj", ms)
                prev_sc = curr - delta
                st.markdown(
                    f'<div class="rf-card">'
                    f'<div>'
                    f'<div class="rf-name">{r["Player"]}</div>'
                    f'<div class="rf-meta">{r["Team within selected timeframe"]} &middot; {lg} &middot; {r["Age"]:.0f} yrs</div>'
                    f'<div class="rf-scores">{prev_sc:.0f} &#x2192; <b style="color:#111827;">{curr:.0f}</b> (MS: {ms:.0f})</div>'
                    f'</div>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:18px;font-weight:700;color:#c0392b;">{delta:.1f}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ── Tab 3: Player Profile ──────────────────────────────────────────────────────
with tab_player:
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#7a7060;'
        'margin-bottom:14px;line-height:1.6;">'
        'Select a player to view their season-by-season development. '
        'Current season is highlighted. Delta shown vs. previous season.</div>',
        unsafe_allow_html=True,
    )

    all_players = sorted(display_df["Player"].unique())
    pre_p       = st.session_state.get("dev_player") or (all_players[0] if all_players else None)
    sel_player  = st.selectbox("Player", all_players,
                               index=all_players.index(pre_p) if pre_p in all_players else 0,
                               key="dev_player_sel")

    player_rows = display_df[display_df["Player"] == sel_player]
    if player_rows.empty:
        st.warning("Player not found.")
        st.stop()

    p_row   = player_rows.iloc[0]
    p_team  = p_row["Team within selected timeframe"]
    p_lg    = LEAGUE_DISPLAY_NAMES.get(p_row.get("League",""), p_row.get("League",""))
    ms      = p_row["ms_score"]
    delta   = p_row.get("delta")
    dcol    = "#1a7a45" if (delta or 0) > 0 else ("#c0392b" if (delta or 0) < 0 else "#7a7060")
    dsign   = "+" if (delta or 0) > 0 else ""
    delta_str = f"{dsign}{delta:.1f}" if delta is not None else "&#x2014;"
    delta_arrow = "&#x2191;" if (delta or 0) > 0 else ("&#x2193;" if (delta or 0) < 0 else "")

    season_cards = []
    for s in SEASON_LABELS:
        pool = scored_by_season.get(s)
        if pool is None: continue
        mask = (pool["Player"] == sel_player) & (pool["Team within selected timeframe"] == p_team)
        if not mask.any(): continue
        row_s    = pool[mask].iloc[0]
        cats_s   = {c: float(row_s.get(f"{c}_score", 0)) for c in CATS}
        ov_s     = float(row_s.get("overall_adj", 0))
        mins_s   = row_s.get("Minutes played", 0)
        age_s_raw = row_s.get("Age","-")
        try:
            season_year = int(s.split("/")[0])
            age_s = str(int(float(age_s_raw)) + (season_year - 2025))
        except:
            age_s = str(age_s_raw)

        _LEAGUE_REF = {
            "Championship":4692,"Segunda Division":4606,"Pro League":4053,"Liga Pro":4042,
            "MLS":3968,"Serie A BRA":3812,"Premier League":3807,"La Liga":3797,
            "Swiss Super League":3722,"Italian Serie A":3707,"Super Lig":3591,
            "Eredivisie":3486,"Liga Profesional":3481,"Prva HNL":3403,"Ekstraklasa":3391,
            "Ligue 1":3389,"Bundesliga":3363,"Primeira Liga":3288,"Superligaen":3279,"Eliteserien":3004,
        }
        try:
            lg_ref     = _LEAGUE_REF.get(row_s.get("League",""), 3420)
            avail_pct  = float(mins_s) / max(lg_ref, 1)
            avail_col  = "#1a7a45" if avail_pct >= 0.60 else ("#f0a500" if avail_pct >= 0.35 else "#c0392b")
            avail_lbl  = "Available" if avail_pct >= 0.60 else ("Rotation" if avail_pct >= 0.35 else "Limited")
        except:
            avail_col, avail_lbl = "#f0a500", "&#x2014;"

        season_cards.append({
            "season": s, "ov": ov_s, "cats": cats_s,
            "mins": mins_s, "age": age_s,
            "avail_col": avail_col, "avail_lbl": avail_lbl,
            "is_current": (s == latest_season),
        })

    if not season_cards:
        st.info("No season data found for this player.")
        st.stop()

    # Player summary bar
    st.markdown(f"""
    <div style="background:#f0ebe2;border-radius:8px;padding:12px 16px;margin-bottom:16px;
                display:flex;justify-content:space-between;align-items:center;">
      <div>
        <div style="font-size:15px;font-weight:700;color:#111827;">{sel_player}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#7a7060;margin-top:2px;">
          {p_team} &middot; {p_lg} &middot; {position_group}
        </div>
      </div>
      <div style="text-align:center;">
        <div style="background:{_c(ms)};color:white;width:52px;height:52px;border-radius:8px;
                    display:flex;align-items:center;justify-content:center;
                    font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;">{ms:.0f}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#b0a898;margin-top:3px;">MS Score</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    def _bar_html(label, value):
        col = _c(value)
        return (
            f'<div style="display:flex;align-items:center;gap:5px;margin-bottom:4px;">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
            f'color:#7a7060;width:38px;flex-shrink:0;">{label}</span>'
            f'<div style="flex:1;height:5px;background:#e0d8cc;border-radius:2px;">'
            f'<div style="width:{value:.0f}%;height:5px;border-radius:2px;background:{col};"></div></div>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
            f'color:#111827;width:20px;text-align:right;font-weight:700;">{value:.0f}</span>'
            f'</div>'
        )

    def _radar_svg(cat_values, color="#c9a84c"):
        n  = 5
        cx = cy = 42
        r  = 32
        angles = [math.pi/2 + i * 2*math.pi/n for i in range(n)]
        pts    = []
        for i, v in enumerate(cat_values):
            frac = max(0, min(1, v / 100))
            pts.append(f"{cx + r*frac*math.cos(angles[i]):.1f},{cy - r*frac*math.sin(angles[i]):.1f}")
        outer  = " ".join(f"{cx + r*math.cos(a):.1f},{cy - r*math.sin(a):.1f}" for a in angles)
        mid    = " ".join(f"{cx + r*0.5*math.cos(a):.1f},{cy - r*0.5*math.sin(a):.1f}" for a in angles)
        spokes = "".join(
            f'<line x1="{cx}" y1="{cy}" x2="{cx + r*math.cos(a):.1f}" y2="{cy - r*math.sin(a):.1f}" stroke="#e0d8cc" stroke-width="0.5"/>'
            for a in angles
        )
        labels  = ["G","Ch","Dr","Pa","De"]
        lbl_html = ""
        for i, lbl in enumerate(labels):
            lx = cx + (r+8)*math.cos(angles[i])
            ly = cy - (r+8)*math.sin(angles[i])
            lbl_html += f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" dominant-baseline="middle" font-size="8" fill="#b0a898">{lbl}</text>'
        fh = color.lstrip("#")
        fr, fg, fb = int(fh[0:2],16), int(fh[2:4],16), int(fh[4:6],16)
        return (
            f'<svg viewBox="0 0 84 84" width="80" height="80" style="display:block;margin:6px auto 0;">'
            f'<polygon points="{outer}" fill="none" stroke="#e0d8cc" stroke-width="0.8"/>'
            f'<polygon points="{mid}" fill="none" stroke="#e0d8cc" stroke-width="0.4"/>'
            f'{spokes}'
            f'<polygon points="{" ".join(pts)}" fill="rgba({fr},{fg},{fb},0.18)" stroke="{color}" stroke-width="1.2"/>'
            f'{lbl_html}'
            f'</svg>'
        )

    cards_html = '<div style="display:flex;gap:12px;overflow-x:auto;padding-bottom:8px;">'
    for sc in season_cards:
        is_curr    = sc["is_current"]
        border     = "border:1.5px solid #c9a84c;" if is_curr else "border:0.5px solid #e0d8cc;"
        head_bg    = "background:#111827;" if is_curr else "background:#f0ebe2;"
        yr_col     = "#c9a84c" if is_curr else "#7a7060"
        name_col   = "white" if is_curr else "#111827"
        meta_col   = "rgba(255,255,255,0.4)" if is_curr else "#b0a898"
        ov_col     = _c(sc["ov"])
        cats       = sc["cats"]
        radar_col  = "#c9a84c" if is_curr else "#9aa5b4"
        radar_vals = [cats.get(c,0) for c in CATS]

        curr_label = " &#x2190; current" if is_curr else ""
        d_badge    = ""
        if is_curr and delta is not None:
            d_badge = f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;font-weight:700;color:{dcol};margin-left:6px;">{delta_arrow} {delta_str}</span>'

        bars   = "".join(_bar_html(CAT_SHORT[c], cats.get(c,0)) for c in CATS)
        radar  = _radar_svg(radar_vals, radar_col)

        try:   mins_disp = f'{int(float(sc["mins"])):,}'
        except: mins_disp = str(sc["mins"])

        cards_html += f"""
        <div style="min-width:155px;max-width:175px;flex-shrink:0;{border}border-radius:10px;overflow:hidden;">
          <div style="{head_bg}padding:10px 12px;border-bottom:0.5px solid rgba(201,168,76,0.15);">
            <div style="font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:600;color:{yr_col};">
              {sc["season"]}{curr_label}{d_badge}
            </div>
            <div style="font-size:12px;font-weight:600;color:{name_col};margin-top:3px;">{sel_player}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:{meta_col};margin-top:2px;">
              {p_team} &middot; {p_lg} &middot; {sc["age"]} yrs
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:{meta_col};margin-top:1px;">
              {mins_disp} min &nbsp;&middot;&nbsp;
              <span style="color:{sc['avail_col']};">&#x25CF;</span> {sc["avail_lbl"]}
            </div>
          </div>
          <div style="background:white;padding:10px 12px;">
            <div style="text-align:center;margin-bottom:8px;">
              <span style="background:{ov_col};color:white;padding:4px 10px;border-radius:6px;
                           font-family:'JetBrains Mono',monospace;font-size:15px;font-weight:700;">{sc["ov"]:.0f}</span>
            </div>
            {bars}
            {radar}
          </div>
        </div>"""

    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)
