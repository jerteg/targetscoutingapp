import os, sys, io, base64, math
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import percentileofscore

from shared.data_processing import load_season_data, SEASON_LABELS
from shared.season_filter import render_season_filter
from shared.styles import BASE_CSS
from shared.sidebar_nav import render_sidebar_nav
from shared.scoring import compute_scores
from shared.templates import (
    position_groups, report_template, position_category_weights,
    LEAGUE_MULTIPLIERS_ALL, TOP5_LEAGUES, NEXT14_LEAGUES,
    ALL_RADAR_STATS,
)
from shared.templates_extra import (
    DASHBOARD_BARS_PER_POSITION, DASHBOARD_SCATTER_AXES, DASHBOARD_SCATTER_AXES_2,
    POSITION_TO_ARCHETYPE_GROUP,
)
from shared.similarity import adjusted_similarity, tier_badge_color
from shared.archetypes import load_or_train_models, get_player_archetype, archetype_color
from radar_app.radar import create_radar, export_full

st.set_page_config(page_title="Dashboard · Target Scouting", layout="wide", initial_sidebar_state="expanded")
st.markdown(BASE_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
.block-container { padding-top:1rem !important; max-width:1400px !important; }

/* ── Player header ── */
.player-header {
    background:#111827; border-radius:10px; padding:10px 18px 0; margin-bottom:12px;
}
.player-name-lg {
    font-size:22px; font-weight:700; color:white;
    letter-spacing:-0.02em; line-height:1.1; font-family:'DM Sans',sans-serif;
}
.player-pos-line {
    font-family:'JetBrains Mono',monospace; font-size:9px;
    color:rgba(255,255,255,0.4); text-transform:uppercase;
    letter-spacing:0.1em; margin-top:2px;
}
.player-pill {
    display:inline-flex; flex-direction:column;
    background:rgba(255,255,255,0.07);
    border:0.5px solid rgba(255,255,255,0.12);
    border-radius:4px; padding:2px 8px; margin:2px 2px 0 0;
}
.player-pill .pl {
    font-family:'JetBrains Mono',monospace; font-size:7px;
    color:rgba(255,255,255,0.35); text-transform:uppercase; letter-spacing:0.06em;
}
.player-pill .pv { font-size:10px; font-weight:600; color:rgba(255,255,255,0.9); }
.arch-pill {
    display:inline-flex; align-items:center; gap:5px;
    border-radius:4px; padding:2px 8px; margin:2px 2px 0 0;
    font-family:'JetBrains Mono',monospace; font-size:9px; font-weight:700;
    text-transform:uppercase; letter-spacing:0.07em;
}

/* FIX 1 — category bars with score pill */
.cat-bars { margin-top:8px; display:grid; grid-template-columns:repeat(5,1fr); gap:5px; }
.cat-bar-lbl {
    font-family:'JetBrains Mono',monospace; font-size:8px;
    color:rgba(255,255,255,0.3); text-transform:uppercase;
    letter-spacing:0.06em;
}
.cat-bar-header {
    display:flex; align-items:center; gap:5px; margin-bottom:3px;
}
.cat-score-pill {
    border-radius:4px; padding:2px 8px;
    font-family:'JetBrains Mono',monospace;
    font-size:11px; font-weight:600; line-height:1;
}
.cat-bar-track { background:rgba(255,255,255,0.1); height:3px; border-radius:2px; }
.cat-bar-fill  { height:3px; border-radius:2px; }

/* FIX 6 — compact score box */
.score-num-lg {
    width:50px; height:50px; border-radius:8px;
    display:flex; align-items:center; justify-content:center;
    font-size:22px; font-weight:700; color:white; margin:0 auto;
    font-family:'JetBrains Mono',monospace;
}
.header-bottom-bar {
    background:rgba(255,255,255,0.04);
    border-top:0.5px solid rgba(255,255,255,0.07);
    margin:8px -18px 0; padding:4px 18px;
    border-radius:0 0 10px 10px;
    display:flex; gap:14px; align-items:center; flex-wrap:wrap;
}
.hbb-item {
    font-family:'JetBrains Mono',monospace; font-size:8px;
    color:rgba(255,255,255,0.22);
}
.hbb-label { color:rgba(201,168,76,0.55); font-weight:700; margin-right:3px; }

/* ── Style bars ── */
.section-lbl {
    font-family:'JetBrains Mono',monospace; font-size:9px; font-weight:700;
    color:#b0a898; text-transform:uppercase; letter-spacing:0.12em;
    margin-bottom:8px; display:flex; align-items:center; gap:8px;
}
.section-lbl::after { content:''; flex:1; height:0.5px; background:#e0d8cc; }
.pct-bar-row { margin-bottom:5px; }
.pct-bar-stat {
    font-size:10px; color:#111827; display:flex;
    justify-content:space-between; margin-bottom:2px;
}
.pct-bar-track { background:#f0ebe2; height:5px; border-radius:3px; }
.pct-bar-fill  { height:5px; border-radius:3px; }

/* FIX 2 — equal-height row for style profile + radar */
.equal-height-row {
    display:grid; grid-template-columns:1fr 1fr;
    align-items:stretch; gap:16px; margin-bottom:12px;
}
.eh-cell {
    display:flex; flex-direction:column; min-height:400px;
}

/* FIX 4 — Similar players minimal list */
.sim-list-row {
    display:flex; align-items:baseline; gap:8px;
    padding:5px 0; border-bottom:0.5px solid #f0ebe2;
}
.sim-list-row:last-child { border-bottom:none; }
.sim-list-rank {
    font-family:'JetBrains Mono',monospace; font-size:9px;
    font-weight:700; color:#c9a84c; width:16px; flex-shrink:0; text-align:right;
}
.sim-list-name {
    font-size:13px; font-weight:700; color:#111827; flex:1;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.sim-list-pct {
    font-family:'JetBrains Mono',monospace; font-size:11px;
    font-weight:700; color:#c9a84c; flex-shrink:0;
}

/* FIX 6 — smaller archetype badge */
.arch-pill { font-size:9px !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("dashboard_player", None), ("dashboard_team", None),
              ("dashboard_position_group", None), ("scout_notes", {}),
              ("dashboard_position_filter", "Right Winger")]:
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_data
def load_data(season="2025/26", min_minutes=0):
    return load_season_data(season, min_minutes)

@st.cache_data
def load_both_seasons(min_minutes=0):
    return {s: load_season_data(s, min_minutes) for s in SEASON_LABELS}

@st.cache_resource
def get_archetype_models(_data):
    return load_or_train_models(_data)

@st.cache_data
def compute_full_pct(_data, pos_group, league_template, score_mode, season_key):
    return compute_scores(_data, pos_group, league_template, score_mode)

if "_season" not in st.session_state: st.session_state["_season"] = "2025/26"
if "_min_min" not in st.session_state: st.session_state["_min_min"] = 900
data = load_data(st.session_state["_season"], st.session_state["_min_min"])

# Position labels for FIX 7
POSITION_OPTIONS = [
    "Centre-Back", "Right-Back", "Left-Back",
    "Defensive Midfielder", "Central Midfielder", "Attacking Midfielder",
    "Right Winger", "Left Winger", "Striker",
]

def _c(v):
    if v >= 75: return "#1a7a45"
    elif v >= 50: return "#91cf60"
    elif v >= 25: return "#f0a500"
    return "#d73027"

def _pct_bar_color(v):
    if v >= 80: return "#1a9850"
    elif v >= 60: return "#91cf60"
    elif v >= 40: return "#f0a500"
    return "#d73027"

# FIX 1 — pill color + text color
def _pill_color(v):
    if v >= 80:   return "#1a9850", "#111827"
    elif v >= 60: return "#91cf60", "#111827"
    elif v >= 40: return "#f0a500", "#111827"
    return "#d73027", "#ffffff"

STAT_SHORT = {
    "xG per 90":"xG /90","xG per shot":"xG /shot","Finishing":"Finishing",
    "Touches in box per 90":"Box touches /90","xA per 90":"xA /90",
    "Key passes per pass":"Key passes /pass","Through passes per pass":"Through /pass",
    "Accurate crosses per received pass":"Crosses /rec","Accurate crosses, %":"Crosses %",
    "Successful dribbles per received pass":"Dribbles /rec","Successful dribbles, %":"Dribbles %",
    "Progressive runs per received pass":"Prog. runs /rec",
    "Ball progression through passing":"Ball progression",
    "Passing accuracy (prog/1/3/forw)":"Pass accuracy",
    "PAdj Defensive duels won per 90":"Def duels /90","Defensive duels won, %":"Def duels won %",
    "PAdj Aerial duels won per 90":"Aerial /90","Aerial duels won, %":"Aerial won %",
    "PAdj Interceptions":"Interceptions","Fouls per 90":"Fouls /90",
    "Shots blocked per 90":"Shots blocked",
    "PAdj Successful defensive actions per 90":"Def actions /90",
    "Progressive runs per 90":"Prog. runs /90","Defensive duels per 90":"Def duels /90",
}
CAT_SHORT = {"Goalscoring":"Goals","Chance creation":"Chance cr.",
             "Dribbling":"Dribbling","Passing":"Passing","Defending":"Defending"}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav("dashboard")
    season, min_minutes = render_season_filter(key_prefix="5")
    if season != st.session_state.get("_season") or min_minutes != st.session_state.get("_min_min"):
        st.session_state["_season"] = season
        st.session_state["_min_min"] = min_minutes
        st.rerun()

    # FIX 7 — Positie-eerst: position filter before player selector
    st.markdown('<span class="sb-section-label">Position</span>', unsafe_allow_html=True)
    pos_filter_idx = POSITION_OPTIONS.index(st.session_state["dashboard_position_filter"]) \
                     if st.session_state["dashboard_position_filter"] in POSITION_OPTIONS else 6
    sel_position_filter = st.selectbox(
        "position_filter", POSITION_OPTIONS,
        index=pos_filter_idx,
        label_visibility="collapsed",
        key="db_pos_filter"
    )
    st.session_state["dashboard_position_filter"] = sel_position_filter

    # Filter players by selected position group
    pg_for_filter = sel_position_filter  # direct match to position_groups key
    pos_filtered_data = data[
        data["Main Position"].isin(position_groups.get(pg_for_filter, []))
    ] if pg_for_filter in position_groups else data
    filtered_players = sorted(pos_filtered_data["Player"].unique())
    if not filtered_players:
        filtered_players = sorted(data["Player"].unique())

    st.markdown('<span class="sb-section-label">Player</span>', unsafe_allow_html=True)
    pre = st.session_state.dashboard_player
    pi  = filtered_players.index(pre) if pre in filtered_players else 0
    sel_player = st.selectbox("player", filtered_players, index=pi, label_visibility="collapsed")

    p_rows = data[data["Player"] == sel_player]
    if p_rows.empty: st.warning("Player not found."); st.stop()
    teams    = p_rows["Team within selected timeframe"].unique()
    sel_team = st.selectbox("Team", teams) if len(teams) > 1 else teams[0]
    if len(teams) == 1: st.caption(f"🏟️ {sel_team}")
    p_row = p_rows[p_rows["Team within selected timeframe"] == sel_team].iloc[0]

    st.markdown('<span class="sb-section-label">Position group</span>', unsafe_allow_html=True)
    main_pos = str(p_row.get("Main Position",""))
    auto_pg  = next((pg for pg, pos in position_groups.items() if main_pos in pos),
                    list(position_groups.keys())[0])
    # Default position group to match the position filter
    pre_pg   = st.session_state.get("dashboard_position_group") or sel_position_filter or auto_pg
    pg_list  = list(position_groups.keys())
    position_group = st.selectbox("pg", pg_list,
                                  index=pg_list.index(pre_pg) if pre_pg in pg_list else
                                        pg_list.index(sel_position_filter) if sel_position_filter in pg_list else 0,
                                  label_visibility="collapsed")

    st.markdown(
        '<span class="sb-section-label" style="color:rgba(201,168,76,0.65);">'
        'Score basis'
        '<span style="background:rgba(201,168,76,0.15);border:0.5px solid rgba(201,168,76,0.4);'
        'border-radius:3px;padding:1px 5px;font-size:8px;color:#c9a84c;margin-left:5px;">score</span>'
        '</span>', unsafe_allow_html=True)
    league_template_score = st.radio("lts",
                                     ["Top 5 leagues","Next 14 competitions","Both"],
                                     label_visibility="collapsed", key="db_lt_score")
    score_mode = st.radio("sm", ["Adjusted (recommended)","Model (raw)"],
                          label_visibility="collapsed")
    multi_season_on = st.checkbox("Multi-season score (65/35)", value=False, key="db_multi")

    st.markdown(
        '<span class="sb-section-label" style="color:rgba(133,183,235,0.8);">'
        'Radar percentile basis'
        '<span style="background:rgba(41,128,185,0.15);border:0.5px solid rgba(41,128,185,0.4);'
        'border-radius:3px;padding:1px 5px;font-size:8px;color:#85B7EB;margin-left:5px;">radar</span>'
        '</span>', unsafe_allow_html=True)
    pct_basis = st.radio("pb", ["T5 only","Next 14 only","Own league"],
                         label_visibility="collapsed")

    st.markdown('<span class="sb-section-label">Similar players pool</span>',
                unsafe_allow_html=True)
    sim_pg = st.selectbox("Position group (similar)", pg_list,
                          index=pg_list.index(position_group), key="sim_pg")
    sim_leagues = st.radio("League filter (similar)", ["Top 5","Next 14","Both"],
                           key="sim_lg")

    st.session_state.dashboard_player         = sel_player
    st.session_state.dashboard_team           = sel_team
    st.session_state.dashboard_position_group = position_group
    st.button("Refresh", use_container_width=True)

# ── Archetype models ──────────────────────────────────────────────────────────
try:
    arch_models = get_archetype_models(data)
except Exception:
    arch_models = {}

# ── Compute scores ─────────────────────────────────────────────────────────────
active_player = sel_player; active_team = sel_team; active_pg = position_group
p_rows2 = data[data["Player"] == active_player]
p_row2  = p_rows2[p_rows2["Team within selected timeframe"] == active_team].iloc[0] \
          if not p_rows2.empty else None
if p_row2 is None: st.error("Player not found."); st.stop()

pos    = str(p_row2.get("Position","")).split(",")[0].strip()
age    = p_row2.get("Age","—");    mins     = p_row2.get("Minutes played","—")
foot   = p_row2.get("Foot","—");   height   = p_row2.get("Height","—")
nation = p_row2.get("Birth country","—"); league = p_row2.get("League","—")

pct_pool = compute_full_pct(data, active_pg, league_template_score, score_mode,
                            st.session_state["_season"])
p_mask   = (pct_pool["Player"]==active_player) & \
           (pct_pool["Team within selected timeframe"]==active_team)
ov_adj   = float(pct_pool[p_mask]["overall_adj"].iloc[0]) if p_mask.any() else 0.0

ov_display = ov_adj
if multi_season_on:
    try:
        _both  = load_both_seasons(st.session_state["_min_min"])
        _other = [s for s in _both if s != st.session_state["_season"]]
        if _other:
            _prev = compute_scores(_both[_other[0]], active_pg, league_template_score, score_mode)
            _mp   = (_prev["Player"]==active_player) & \
                    (_prev["Team within selected timeframe"]==active_team)
            if _mp.any():
                ov_display = round(0.65*ov_adj +
                                   0.35*float(_prev[_mp]["overall_adj"].iloc[0]), 1)
    except Exception:
        pass

ov_color   = _c(ov_display)
cat_scores = {c: float(pct_pool[p_mask][f"{c}_score"].iloc[0])
              if p_mask.any() else 0.0 for c in report_template}

# Rank
rank_str = ""
if p_mask.any():
    n_tot  = len(pct_pool)
    rank_v = int(pct_pool["overall_adj"].rank(ascending=False)
                 .loc[pct_pool[p_mask].index[0]])
    rank_str = f"Rank {rank_v}/{n_tot}"

# ── Archetype badge ──────────────────────────────────────────────────────────
try:
    primary_arch, secondary_arch, _, _ = get_player_archetype(
        p_row2, arch_models, active_pg)
except Exception:
    primary_arch, secondary_arch = None, None

# ── FIX 1: Category bars with score pill ─────────────────────────────────────
cat_bars_html = '<div class="cat-bars" style="margin-top:10px;">'
for cat in report_template:
    v = cat_scores.get(cat, 0)
    col = _c(v)
    lbl = CAT_SHORT.get(cat, cat)
    pill_bg, pill_txt = _pill_color(v)
    cat_bars_html += (
        f'<div>'
        f'  <div class="cat-bar-header">'
        f'    <span class="cat-bar-lbl">{lbl}</span>'
        f'    <span class="cat-score-pill" style="background:{pill_bg};color:{pill_txt};">'
        f'      {v:.0f}'
        f'    </span>'
        f'  </div>'
        f'  <div class="cat-bar-track">'
        f'    <div class="cat-bar-fill" style="width:{v:.0f}%;background:{col};"></div>'
        f'  </div>'
        f'</div>'
    )
cat_bars_html += '</div>'

# ── Pills row ────────────────────────────────────────────────────────────────
pills_html = ""
for lbl, val in [("Age", age), ("Min", mins), ("Foot", foot),
                 ("Height", f"{height} cm"), ("Nat.", nation)]:
    pills_html += (f'<div class="player-pill">'
                   f'<span class="pl">{lbl}</span>'
                   f'<span class="pv">{val}</span></div>')

# ── Archetype pill — inline with player name ──────────────────────────────────
arch_pill_html = ""
if primary_arch and primary_arch != "—":
    ac = archetype_color(primary_arch)
    try:
        r, g, b = int(ac[1:3], 16), int(ac[3:5], 16), int(ac[5:7], 16)
        arch_pill_html = (
            f'<div class="arch-pill" '
            f'style="background:rgba({r},{g},{b},0.15);'
            f'color:{ac};border:0.5px solid rgba({r},{g},{b},0.4);">'
            f'◆ {primary_arch}</div>'
        )
    except Exception:
        pass

# ── Player header ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="player-header">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px;">
    <div style="flex:1;">
      <div class="player-pos-line">{pos} · {active_team} · {league} · {st.session_state["_season"]}</div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:10px;">
        <div class="player-name-lg">{active_player}</div>
        {arch_pill_html}
      </div>
      <div style="display:flex;flex-wrap:wrap;margin-top:10px;">{pills_html}</div>
      {cat_bars_html}
    </div>
    <div style="text-align:center;flex-shrink:0;">
      <div class="score-num-lg" style="background:{ov_color};">{ov_display:.0f}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:7px;
                  color:rgba(255,255,255,0.3);text-transform:uppercase;margin-top:3px;">Overall</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:7px;
                  color:rgba(201,168,76,0.5);margin-top:1px;">{league_template_score.split()[0]} · {score_mode.split()[0]}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:7px;
                  color:rgba(255,255,255,0.2);margin-top:1px;">{rank_str}</div>
    </div>
  </div>
  <div class="header-bottom-bar">
    <span class="hbb-item"><span class="hbb-label">Score:</span>{league_template_score} · {score_mode.split()[0]}</span>
    <span class="hbb-item">|</span>
    <span class="hbb-item"><span class="hbb-label" style="color:rgba(133,183,235,0.55);">Radar:</span>{pct_basis}</span>
    <span class="hbb-item">|</span>
    <span class="hbb-item">{active_pg} · Wyscout 26 Mar 2026</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1: Style bars LEFT | Radar RIGHT  — FIX 2: gelijke hoogte
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1.3, 0.9])

with col_left:
    pos_label_key = str(p_row2.get("Position Label", active_pg))
    bar_stats     = DASHBOARD_BARS_PER_POSITION.get(
        pos_label_key, DASHBOARD_BARS_PER_POSITION.get(active_pg, []))
    pos_pool      = data[data["Main Position"].isin(position_groups[active_pg])].copy()
    avail_bars    = [s for s in bar_stats if s in pos_pool.columns and s in p_row2.index]

    st.markdown(f'<div class="section-lbl">Style profile — {pos_label_key}</div>',
                unsafe_allow_html=True)

    if avail_bars and not pos_pool.empty:
        half    = math.ceil(len(avail_bars) / 2)
        left_s  = avail_bars[:half]
        right_s = avail_bars[half:]

        bc1, bc2 = st.columns(2)
        for bar_col, stats_subset in [(bc1, left_s), (bc2, right_s)]:
            with bar_col:
                bars_html = ""
                for stat in stats_subset:
                    raw_val = p_row2.get(stat, None)
                    try:
                        raw_f   = float(str(raw_val).replace(",",".")) \
                                  if raw_val is not None else None
                        pct_val = float(percentileofscore(
                            pos_pool[stat].dropna().astype(float),
                            raw_f or 0, kind="rank"))
                    except Exception:
                        pct_val = 0.0; raw_f = None

                    if "fouls" in stat.lower():
                        pct_val = 100 - pct_val

                    col_fill = _pct_bar_color(pct_val)
                    short    = STAT_SHORT.get(stat, stat[:22])
                    # FIX 3 — alleen absolute waarde, geen percentiel-getal
                    raw_str  = f"{raw_f:.2f}" if raw_f is not None else "—"

                    bars_html += f"""
                    <div class="pct-bar-row">
                      <div class="pct-bar-stat">
                        <span>{short}</span>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#7a7060;">{raw_str}</span>
                      </div>
                      <div class="pct-bar-track">
                        <div class="pct-bar-fill" style="width:{pct_val:.0f}%;background:{col_fill};"></div>
                      </div>
                    </div>"""
                st.markdown(bars_html, unsafe_allow_html=True)
    else:
        st.info("No style profile available for this position.")

with col_right:
    st.markdown('<div class="section-lbl">Radar — Position Template</div>',
                unsafe_allow_html=True)
    try:
        with st.spinner("Building radar…"):
            fig_r, _ = create_radar(data, player_name=active_player,
                                    player_team=active_team,
                                    template_key=active_pg,
                                    percentile_basis=pct_basis,
                                    radar_type="Position Template",
                                    show_avg=True, compact=True)
        st.pyplot(fig_r, use_container_width=True)
    except Exception as e:
        st.error(f"Radar error: {e}")

st.markdown('<hr style="border:none;border-top:0.5px solid #e0d8cc;margin:6px 0 12px;">',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2: Similar players | Scatter 1 | Scatter 2
# FIX 4: compacte ranking-lijst  |  FIX 5: tweede scatter ipv season trend
# ══════════════════════════════════════════════════════════════════════════════
c_sim, c_scatter, c_scatter2 = st.columns([1, 1.1, 1.1])

# ── FIX 4 — Similar players as compact ranking list ───────────────────────────
with c_sim:
    st.markdown('<div class="section-lbl">Similar players</div>', unsafe_allow_html=True)
    try:
        sim_pool = data[data["Main Position"].isin(position_groups[sim_pg])].copy()
        if sim_leagues == "Top 5":
            sim_pool = sim_pool[sim_pool["League"].isin(TOP5_LEAGUES)]
        elif sim_leagues == "Next 14":
            sim_pool = sim_pool[sim_pool["League"].isin(NEXT14_LEAGUES)]

        sim_stats = [s for s in ALL_RADAR_STATS if s in sim_pool.columns]
        pl_league = str(p_row2.get("League",""))

        sim_cands = sim_pool[
            ~((sim_pool["Player"]==active_player) &
              (sim_pool["Team within selected timeframe"]==active_team))
        ].copy()

        sim_results = adjusted_similarity(
            target_row=p_row2, candidates_df=sim_cands,
            sim_stats=sim_stats, target_league=pl_league, min_minutes=600)

        top10 = sim_results.head(10)
        for i, (_, sr) in enumerate(top10.iterrows()):
            sim_name  = sr["Player"]
            sim_team  = sr["Team within selected timeframe"]
            adj_pct   = sr["adjusted_sim"] * 100

            st.markdown(f"""
            <div class="sim-list-row">
              <span class="sim-list-rank">{i+1}</span>
              <span class="sim-list-name">{sim_name}</span>
              <span class="sim-list-pct">{adj_pct:.0f}%</span>
            </div>""", unsafe_allow_html=True)

            if st.button("›", key=f"sim_{i}_{hash(sim_name+sim_team) % 99999}",
                         help=f"Open {sim_name}",
                         use_container_width=False):
                st.session_state.dashboard_player = sim_name
                st.session_state.dashboard_team   = sim_team
                st.session_state.dashboard_position_group = active_pg
                st.session_state.dashboard_position_filter = sel_position_filter
                st.rerun()

    except Exception as e:
        st.error(f"Similar players error: {e}")


def _render_scatter(col_ctx, x_stat, y_stat, active_player, active_pg, data,
                    position_groups, STAT_SHORT, scatter_key="sc1"):
    """Reusable scatter plot renderer."""
    x_s = STAT_SHORT.get(x_stat, x_stat)
    y_s = STAT_SHORT.get(y_stat, y_stat)

    with col_ctx:
        arch_group_label = POSITION_TO_ARCHETYPE_GROUP.get(active_pg, "W")
        st.markdown(
            f'<div class="section-lbl">Scatter — {x_s} vs {y_s}</div>',
            unsafe_allow_html=True)
        try:
            pool_sc = data[data["Main Position"].isin(position_groups[active_pg])] \
                          .dropna(subset=[x_stat, y_stat]).copy()
            if not pool_sc.empty:
                fig_s = go.Figure()
                for _, row in pool_sc.iterrows():
                    it = row["Player"] == active_player
                    fig_s.add_trace(go.Scatter(
                        x=[row[x_stat]], y=[row[y_stat]], mode="markers",
                        marker=dict(size=12 if it else 5,
                                    color="#c9a84c" if it else "#111827",
                                    opacity=1.0 if it else 0.18,
                                    symbol="star" if it else "circle",
                                    line=dict(width=1.5 if it else 0, color="#111827")),
                        hovertemplate=(f"<b>{row['Player']}</b><br>"
                                       f"{x_stat}: {row[x_stat]:.2f}<br>"
                                       f"{y_stat}: {row[y_stat]:.2f}<extra></extra>"),
                        showlegend=False))
                me = pool_sc[pool_sc["Player"]==active_player]
                if not me.empty:
                    fig_s.add_annotation(
                        x=me[x_stat].iloc[0], y=me[y_stat].iloc[0],
                        text=active_player, showarrow=True, arrowhead=2,
                        arrowcolor="#c9a84c", arrowwidth=1.5, ax=20, ay=-28,
                        font=dict(size=10, color="#111827"),
                        bgcolor="#f0ebe2", bordercolor="#c9a84c", borderpad=3)
                fig_s.add_vline(x=pool_sc[x_stat].mean(),
                                line=dict(color="rgba(17,24,39,0.12)", dash="dot", width=1))
                fig_s.add_hline(y=pool_sc[y_stat].mean(),
                                line=dict(color="rgba(17,24,39,0.12)", dash="dot", width=1))
                fig_s.update_layout(
                    paper_bgcolor="#faf7f2", plot_bgcolor="#faf7f2", height=320,
                    margin=dict(l=40, r=10, t=10, b=40),
                    xaxis=dict(title=x_s, tickfont=dict(color="#7a7060", size=9),
                               gridcolor="rgba(0,0,0,0.04)", zeroline=False),
                    yaxis=dict(title=y_s, tickfont=dict(color="#7a7060", size=9),
                               gridcolor="rgba(0,0,0,0.04)", zeroline=False),
                    hoverlabel=dict(bgcolor="#111827", font=dict(size=10, color="white")),
                )
                st.plotly_chart(fig_s, use_container_width=True, key=scatter_key)
                # Caption onder de scatter
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                    f'color:#b0a898;margin-top:-10px;text-align:center;">'
                    f'{active_pg.upper()} — {x_s.upper()} VS {y_s.upper()}</div>',
                    unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Scatter error: {e}")


# ── Scatter 1 (centre) ────────────────────────────────────────────────────────
arch_group = POSITION_TO_ARCHETYPE_GROUP.get(active_pg, "W")
x1, y1 = DASHBOARD_SCATTER_AXES.get(arch_group, ("xG per 90", "xA per 90"))
_render_scatter(c_scatter, x1, y1, active_player, active_pg, data,
                position_groups, STAT_SHORT, scatter_key="sc1")

# ── FIX 5: Scatter 2 (right, replaces season trend) ─────────────────────────
x2, y2 = DASHBOARD_SCATTER_AXES_2.get(arch_group, ("xG per 90", "xG per shot"))
_render_scatter(c_scatter2, x2, y2, active_player, active_pg, data,
                position_groups, STAT_SHORT, scatter_key="sc2")

# ── Download whole dashboard as HTML ─────────────────────────────────────────
st.markdown('<hr style="border:none;border-top:0.5px solid #e0d8cc;margin:12px 0 10px;">',
            unsafe_allow_html=True)

def build_dashboard_html():
    """Build a self-contained static HTML snapshot of the dashboard."""
    cats_html = ""
    for cat in report_template:
        v = cat_scores.get(cat, 0)
        col = _c(v)
        lbl = CAT_SHORT.get(cat, cat)
        pill_bg, pill_txt = _pill_color(v)
        cats_html += (
            f'<div style="background:#f0ebe2;border-left:3px solid {col};'
            f'border-radius:0 4px 4px 0;padding:5px 10px;margin-bottom:4px;">'
            f'<span style="font-size:11px;font-weight:700;color:#111;">{lbl}</span>'
            f'<span style="float:right;background:{pill_bg};color:{pill_txt};'
            f'padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;">{v:.0f}</span></div>'
        )
    bars_section = ""
    if avail_bars and not pos_pool.empty:
        bars_section = '<div style="margin-bottom:16px;">'
        for stat in avail_bars:
            raw_val = p_row2.get(stat, None)
            try:
                raw_f   = float(str(raw_val).replace(",",".")) if raw_val is not None else None
                pct_val = float(percentileofscore(
                    pos_pool[stat].dropna().astype(float), raw_f or 0, kind="rank"))
            except Exception:
                pct_val = 0.0; raw_f = None
            if "fouls" in stat.lower(): pct_val = 100 - pct_val
            col_fill = _pct_bar_color(pct_val)
            short    = STAT_SHORT.get(stat, stat[:28])
            raw_str  = f"{raw_f:.2f}" if raw_f is not None else "—"
            bars_section += (
                f'<div style="margin-bottom:6px;">'
                f'<div style="font-size:10px;color:#111;display:flex;'
                f'justify-content:space-between;margin-bottom:2px;">'
                f'<span>{short}</span>'
                f'<span style="font-family:JetBrains Mono,monospace;color:#7a7060;">{raw_str}</span>'
                f'</div>'
                f'<div style="background:#f0ebe2;height:5px;border-radius:3px;">'
                f'<div style="width:{pct_val:.0f}%;height:5px;border-radius:3px;'
                f'background:{col_fill};"></div></div></div>'
            )
        bars_section += '</div>'

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:'DM Sans',Arial,sans-serif;background:#faf7f2;color:#111827;
      -webkit-font-smoothing:antialiased;padding:24px;max-width:1000px;margin:0 auto;}}
.header{{background:#111827;border-radius:8px;padding:14px 18px;margin-bottom:14px;
          display:flex;justify-content:space-between;align-items:center;}}
.p-name{{font-size:20px;font-weight:700;color:white;letter-spacing:-0.01em;}}
.p-sub{{font-family:'JetBrains Mono',monospace;font-size:9px;color:rgba(255,255,255,0.4);
         text-transform:uppercase;letter-spacing:0.1em;margin-top:3px;}}
.score-box{{min-width:50px;height:50px;border-radius:8px;display:flex;align-items:center;
             justify-content:center;font-family:'JetBrains Mono',monospace;
             font-size:20px;font-weight:700;color:white;padding:0 10px;}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px;}}
.section-title{{font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:700;
                 text-transform:uppercase;letter-spacing:0.12em;color:#b0a898;
                 border-bottom:0.5px solid #e0d8cc;padding-bottom:4px;margin-bottom:8px;}}
.footer{{margin-top:18px;padding-top:10px;border-top:0.5px solid #e0d8cc;
          font-family:'JetBrains Mono',monospace;font-size:9px;color:#b0a898;
          display:flex;justify-content:space-between;}}
@media print{{body{{-webkit-print-color-adjust:exact;print-color-adjust:exact;}}
@page{{margin:1.5cm;}}}}
</style>
</head><body>
<div class="header">
  <div>
    <div class="p-name">{active_player}</div>
    <div class="p-sub">{pos} · {active_team} · {league} · {age} yrs · {mins} min</div>
    {f'<div style="margin-top:5px;font-family:JetBrains Mono,monospace;font-size:9px;color:rgba(201,168,76,0.8);">◆ {primary_arch}</div>' if primary_arch else ''}
  </div>
  <div class="score-box" style="background:{ov_color};">{ov_display:.0f}</div>
</div>

<div class="two-col">
  <div>
    <div class="section-title">Category scores</div>
    {cats_html}
  </div>
  <div>
    <div class="section-title">Rank</div>
    <div style="background:#f0ebe2;border-radius:6px;padding:12px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:8px;color:#b0a898;">Overall score</div>
      <div style="font-size:24px;font-weight:700;color:{ov_color};">{ov_display:.0f}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#b0a898;margin-top:6px;">{rank_str}</div>
    </div>
  </div>
</div>

<div class="section-title">Style profile — {pos_label_key}</div>
{bars_section}

<div class="footer">
  <span>Target Scouting · Player Dashboard · {active_player} · {active_team}</span>
  <span>Wyscout 26 Mar 2026 · {league_template_score} · {active_pg}</span>
</div>
</body></html>"""

dash_html = build_dashboard_html()
st.download_button(
    "⬇ Download Dashboard (HTML)",
    data=dash_html.encode("utf-8"),
    file_name=f"{active_player.replace(' ','_')}_dashboard.html",
    mime="text/html",
    use_container_width=False,
)
