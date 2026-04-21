import os, sys, math
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
    DASHBOARD_BARS_PER_POSITION, DASHBOARD_SCATTER_AXES,
    POSITION_TO_ARCHETYPE_GROUP,
)
from shared.similarity import adjusted_similarity, tier_badge_color
from shared.archetypes import (
    load_or_train_models, get_player_archetype, archetype_color,
)
from radar_app.radar import create_radar, export_full

st.set_page_config(page_title="Dashboard · Target Scouting", layout="wide", initial_sidebar_state="expanded")
st.markdown(BASE_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
.block-container { padding-top:1.6rem !important; max-width:1320px !important; }
.player-header { background:#111827; border-radius:10px; padding:20px 24px; margin-bottom:18px; }
.player-name-lg { font-size:26px; font-weight:700; color:white; letter-spacing:-0.02em; }
.player-pos-line { font-family:'JetBrains Mono',monospace; font-size:10px;
    color:rgba(255,255,255,0.4); text-transform:uppercase; letter-spacing:0.1em; margin-top:4px; }
.player-pill { display:inline-flex; flex-direction:column; background:rgba(255,255,255,0.07);
    border:0.5px solid rgba(255,255,255,0.12); border-radius:4px; padding:4px 10px; margin:3px 4px 0 0; }
.player-pill .pl { font-family:'JetBrains Mono',monospace; font-size:8px;
    color:rgba(255,255,255,0.35); text-transform:uppercase; letter-spacing:0.06em; }
.player-pill .pv { font-size:12px; font-weight:600; color:rgba(255,255,255,0.9); }
.cat-bars { margin-top:16px; display:grid; grid-template-columns:repeat(5,1fr); gap:8px; }
.cat-bar-lbl { font-family:'JetBrains Mono',monospace; font-size:8px;
    color:rgba(255,255,255,0.3); text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px; }
.cat-bar-track { background:rgba(255,255,255,0.1); height:4px; border-radius:2px; }
.cat-bar-fill { height:4px; border-radius:2px; }
.cat-bar-val { font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:700;
    color:rgba(255,255,255,0.75); margin-top:3px; }
.score-num-lg { width:72px; height:72px; border-radius:10px; display:flex; align-items:center;
    justify-content:center; font-size:28px; font-weight:700; color:white; margin:0 auto; }
.archetype-badge { display:inline-flex; align-items:center; gap:6px; padding:5px 12px;
    border-radius:5px; font-family:'JetBrains Mono',monospace; font-size:10px;
    font-weight:700; text-transform:uppercase; letter-spacing:0.08em; }
.tier-badge { display:inline-block; font-family:'JetBrains Mono',monospace; font-size:8px;
    font-weight:700; text-transform:uppercase; letter-spacing:0.06em;
    padding:2px 6px; border-radius:3px; }
.section-lbl { font-family:'JetBrains Mono',monospace; font-size:9px; font-weight:700;
    color:#b0a898; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:12px;
    display:flex; align-items:center; gap:10px; }
.section-lbl::after { content:''; flex:1; height:0.5px; background:#e0d8cc; }
.pct-bar-row { margin-bottom:9px; }
.pct-bar-stat { font-size:11px; color:#111827; display:flex; justify-content:space-between; margin-bottom:3px; }
.pct-bar-track { background:#f0ebe2; height:7px; border-radius:3px; }
.pct-bar-fill { height:7px; border-radius:3px; }
.sim-card { background:#fff; border:0.5px solid #e0d8cc; border-radius:8px;
    padding:11px 14px; margin-bottom:7px; }
.sim-name { font-size:13px; font-weight:700; color:#111827; }
.sim-meta { font-family:'JetBrains Mono',monospace; font-size:9px; color:#b0a898; margin-top:2px; }
.sim-adj { font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:700;
    color:#c9a84c; margin-top:5px; }
.tool-jump { background:#fff; border:0.5px solid #e0d8cc; border-radius:6px;
    padding:11px 14px; transition:border-color 0.12s; }
.tool-jump:hover { border-color:#c9a84c; }
.tj-tag { font-family:'JetBrains Mono',monospace; font-size:8px; font-weight:700;
    color:#c9a84c; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:3px; }
.tj-name { font-size:13px; font-weight:700; color:#111827; margin-bottom:2px; }
.tj-desc { font-size:11px; color:#7a7060; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("dashboard_player",None),("dashboard_team",None),
              ("dashboard_position_group",None),("scout_notes",{})]:
    if k not in st.session_state: st.session_state[k] = v

@st.cache_data
def load_data(season="2025/26", min_minutes=0):
    return load_season_data(season, min_minutes)

@st.cache_data
def load_both_seasons(min_minutes=0):
    return {s: load_season_data(s, min_minutes) for s in SEASON_LABELS}

@st.cache_resource
def get_archetype_models(_data):
    """Load or train archetype models. Cached as resource (not re-run per session)."""
    return load_or_train_models(_data)

@st.cache_data
def compute_full_pct(_data, pos_group, league_template, score_mode, season_key):
    pool = compute_scores(_data, pos_group, league_template, score_mode)
    ALL_STATS = [s for g in report_template.values() for s in g["stats"]]
    ex = [s for s in ALL_STATS if s in _data.columns]
    if pool.empty: return pool, pd.DataFrame()
    mdict = LEAGUE_MULTIPLIERS_ALL
    pct_raw = pool[ex].rank(pct=True) * 100
    mult = pool["League"].map(mdict).fillna(1.0)
    pct = pct_raw.multiply(mult.values, axis=0).clip(0, 100)
    return pool, pct

if "_season" not in st.session_state: st.session_state["_season"] = "2025/26"
if "_min_min" not in st.session_state: st.session_state["_min_min"] = 900
data = load_data(st.session_state["_season"], st.session_state["_min_min"])

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

STAT_SHORT = {
    "xG per 90":"xG /90","xG per shot":"xG /shot","Finishing":"Finishing",
    "Touches in box per 90":"Box touches /90","xA per 90":"xA /90",
    "Key passes per pass":"Key passes /pass","Through passes per pass":"Through passes /pass",
    "Accurate crosses per received pass":"Crosses /rec pass","Accurate crosses, %":"Crosses %",
    "Successful dribbles per received pass":"Dribbles /rec pass","Successful dribbles, %":"Dribbles %",
    "Progressive runs per received pass":"Prog. runs /rec pass",
    "Ball progression through passing":"Ball progression",
    "Passing accuracy (prog/1/3/forw)":"Pass accuracy",
    "PAdj Defensive duels won per 90":"Def duels /90","Defensive duels won, %":"Def duels won %",
    "PAdj Aerial duels won per 90":"Aerial duels /90","Aerial duels won, %":"Aerial duels won %",
    "PAdj Interceptions":"Interceptions","Fouls per 90":"Fouls /90",
    "Shots blocked per 90":"Shots blocked /90",
    "PAdj Successful defensive actions per 90":"Def actions /90",
    "Progressive runs per 90":"Prog. runs /90",
    "Defensive duels per 90":"Def duels /90",
}
CAT_SHORT = {"Goalscoring":"Goals","Chance creation":"Chance cr.",
             "Dribbling":"Dribbling","Passing":"Passing","Defending":"Defending"}

all_players = sorted(data["Player"].unique())

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav("dashboard")
    season, min_minutes = render_season_filter(key_prefix="5")
    if season != st.session_state.get("_season") or min_minutes != st.session_state.get("_min_min"):
        st.session_state["_season"] = season; st.session_state["_min_min"] = min_minutes; st.rerun()

    st.markdown('<span class="sb-section-label">Player</span>', unsafe_allow_html=True)
    pre = st.session_state.dashboard_player
    pi  = all_players.index(pre) if pre in all_players else 0
    sel_player = st.selectbox("player", all_players, index=pi, label_visibility="collapsed")
    p_rows = data[data["Player"] == sel_player]
    if p_rows.empty: st.warning("Player not found."); st.stop()
    teams    = p_rows["Team within selected timeframe"].unique()
    sel_team = st.selectbox("Team", teams) if len(teams) > 1 else teams[0]
    if len(teams) == 1: st.caption(f"🏟️ {sel_team}")
    p_row = p_rows[p_rows["Team within selected timeframe"] == sel_team].iloc[0]

    st.markdown('<span class="sb-section-label">Position group</span>', unsafe_allow_html=True)
    main_pos = str(p_row.get("Main Position",""))
    auto_pg  = next((pg for pg,pos in position_groups.items() if main_pos in pos), list(position_groups.keys())[0])
    pre_pg   = st.session_state.get("dashboard_position_group") or auto_pg
    pg_list  = list(position_groups.keys())
    position_group = st.selectbox("pg", pg_list, index=pg_list.index(pre_pg) if pre_pg in pg_list else 0, label_visibility="collapsed")

    st.markdown(
        '<span class="sb-section-label" style="color:rgba(201,168,76,0.65);">Overall score basis'
        '<span style="background:rgba(201,168,76,0.15);border:0.5px solid rgba(201,168,76,0.4);'
        'border-radius:3px;padding:1px 5px;font-size:8px;color:#c9a84c;margin-left:5px;">score</span></span>',
        unsafe_allow_html=True)
    league_template_score = st.radio("lts", ["Top 5 leagues","Next 14 competitions","Both"],
                                     label_visibility="collapsed", key="db_lt_score")
    score_mode = st.radio("sm", ["Adjusted (recommended)","Model (raw)"], label_visibility="collapsed")

    multi_season_on = st.checkbox("Multi-season score (65/35)", value=False, key="db_multi")

    st.markdown(
        '<span class="sb-section-label" style="color:rgba(133,183,235,0.8);">Radar percentile basis'
        '<span style="background:rgba(41,128,185,0.15);border:0.5px solid rgba(41,128,185,0.4);'
        'border-radius:3px;padding:1px 5px;font-size:8px;color:#85B7EB;margin-left:5px;">radar</span></span>',
        unsafe_allow_html=True)
    pct_basis = st.radio("pb", ["T5 only","Next 14 only","Own league"], label_visibility="collapsed")

    st.markdown('<span class="sb-section-label">Similar players pool</span>', unsafe_allow_html=True)
    sim_pg      = st.selectbox("Position group (similar)", pg_list, index=pg_list.index(position_group), key="sim_pg")
    sim_leagues = st.radio("League filter (similar)", ["Top 5","Next 14","Both"], key="sim_lg")

    st.session_state.dashboard_player         = sel_player
    st.session_state.dashboard_team           = sel_team
    st.session_state.dashboard_position_group = position_group
    st.button("Refresh Dashboard", use_container_width=True)

# ── Load archetype models ──────────────────────────────────────────────────────
try:
    arch_models = get_archetype_models(data)
except Exception:
    arch_models = {}

# ── Compute scores ─────────────────────────────────────────────────────────────
active_player = sel_player; active_team = sel_team; active_pg = position_group
p_rows2 = data[data["Player"] == active_player]
p_row2  = p_rows2[p_rows2["Team within selected timeframe"] == active_team].iloc[0] if not p_rows2.empty else None
if p_row2 is None: st.error("Player not found."); st.stop()

pos    = str(p_row2.get("Position","")).split(",")[0].strip()
age    = p_row2.get("Age","—"); mins = p_row2.get("Minutes played","—")
foot   = p_row2.get("Foot","—"); height = p_row2.get("Height","—")
nation = p_row2.get("Birth country","—"); league = p_row2.get("League","—")
value  = p_row2.get("Market value",None)
val_str = f"€{int(value):,}" if isinstance(value,(int,float)) and not np.isnan(float(value)) else "—"
contract = p_row2.get("Contract expires","—")

pct_pool, pct_df = compute_full_pct(data, active_pg, league_template_score, score_mode, st.session_state["_season"])
p_mask   = (pct_pool["Player"]==active_player)&(pct_pool["Team within selected timeframe"]==active_team)
ov_adj   = float(pct_pool[p_mask]["overall_adj"].iloc[0]) if p_mask.any() else 0.0

ov_display = ov_adj
if multi_season_on:
    try:
        _both = load_both_seasons(st.session_state["_min_min"])
        _other = [s for s in _both if s != st.session_state["_season"]]
        if _other:
            _prev = compute_scores(_both[_other[0]], active_pg, league_template_score, score_mode)
            _mp   = (_prev["Player"]==active_player)&(_prev["Team within selected timeframe"]==active_team)
            if _mp.any():
                ov_display = round(0.65*ov_adj + 0.35*float(_prev[_mp]["overall_adj"].iloc[0]), 1)
    except Exception: pass

ov_color = _c(ov_display)
cat_scores = {c: float(pct_pool[p_mask][f"{c}_score"].iloc[0]) if p_mask.any() else 0.0 for c in report_template}

# ── Archetype badges ──────────────────────────────────────────────────────────
try:
    primary_arch, secondary_arch, p_dist, s_dist = get_player_archetype(p_row2, arch_models, active_pg)
except Exception:
    primary_arch, secondary_arch = "—", None

arch_badges_html = ""
if primary_arch and primary_arch != "—":
    ac = archetype_color(primary_arch)
    arch_badges_html += (f'<span class="archetype-badge" style="background:rgba({int(ac[1:3],16)},'
                         f'{int(ac[3:5],16)},{int(ac[5:7],16)},0.15);'
                         f'color:{ac};border:0.5px solid {ac}40;">'
                         f'◆ {primary_arch}</span> ')
if secondary_arch:
    sc = archetype_color(secondary_arch)
    arch_badges_html += (f'<span class="archetype-badge" style="background:rgba({int(sc[1:3],16)},'
                         f'{int(sc[3:5],16)},{int(sc[5:7],16)},0.08);'
                         f'color:{sc};border:0.5px solid {sc}30;font-size:9px;padding:3px 9px;">'
                         f'◇ {secondary_arch}</span>')

# ── Player header ──────────────────────────────────────────────────────────────
pills_html = "".join(
    f'<div class="player-pill"><span class="pl">{l}</span><span class="pv">{v}</span></div>'
    for l,v in [("Age",age),("Minutes",mins),("Foot",foot),("Height",f"{height} cm"),("Nat.",nation),("Value",val_str)]
)
cat_bars_html = '<div class="cat-bars">'
for cat in report_template:
    v = cat_scores.get(cat,0); col = _c(v); lbl = CAT_SHORT.get(cat,cat)
    cat_bars_html += f'<div><div class="cat-bar-lbl">{lbl}</div><div class="cat-bar-track"><div class="cat-bar-fill" style="width:{v:.0f}%;background:{col};"></div></div><div class="cat-bar-val">{v:.0f}</div></div>'
cat_bars_html += '</div>'

rank_str = ""
if p_mask.any():
    n_tot  = len(pct_pool)
    rank_v = int(pct_pool["overall_adj"].rank(ascending=False).loc[pct_pool[p_mask].index[0]])
    rank_str = f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;color:rgba(255,255,255,0.2);margin-top:6px;">{active_pg} · Rank {rank_v}/{n_tot}</div>'

st.markdown(f"""
<div class="player-header">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:20px;">
    <div style="flex:1;">
      <div class="player-pos-line">{pos} · {active_team} · {league} · {st.session_state["_season"]}</div>
      <div class="player-name-lg">{active_player}</div>
      <div style="display:flex;flex-wrap:wrap;margin-top:10px;">{pills_html}</div>
      {cat_bars_html}
    </div>
    <div style="text-align:center;flex-shrink:0;">
      <div class="score-num-lg" style="background:{ov_color};">{ov_display:.0f}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:8px;color:rgba(255,255,255,0.3);
                  text-transform:uppercase;letter-spacing:0.08em;margin-top:6px;">Overall</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:8px;color:rgba(201,168,76,0.6);margin-top:2px;">
        {league_template_score.split()[0]} · {score_mode.split()[0]}
      </div>
      {rank_str}
    </div>
  </div>
  {f'<div style="margin-top:14px;display:flex;gap:6px;flex-wrap:wrap;">{arch_badges_html}</div>' if arch_badges_html else ''}
  <div style="background:rgba(255,255,255,0.04);border-top:0.5px solid rgba(255,255,255,0.07);
              margin:12px -24px -20px;padding:6px 24px;border-radius:0 0 10px 10px;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:rgba(201,168,76,0.55);">Score:</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:rgba(255,255,255,0.25);margin-left:4px;">{league_template_score} · {score_mode.split()[0]}</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:rgba(255,255,255,0.15);margin:0 8px;">|</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:rgba(133,183,235,0.6);">Radar:</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:rgba(255,255,255,0.25);margin-left:4px;">{pct_basis}</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:rgba(255,255,255,0.15);margin:0 8px;">|</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:rgba(255,255,255,0.25);">{active_pg} · Wyscout 26 Mar 2026</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tool jump buttons ──────────────────────────────────────────────────────────
st.markdown('<div class="section-lbl">Open in tools</div>', unsafe_allow_html=True)
tb1,tb2,tb3,tb4 = st.columns(4)
for col,(tag,name,desc,page,skey,extra) in zip([tb1,tb2,tb3,tb4],[
    ("chart","Radar Tool","Full radar chart","pages/2_Radar.py","jmp_radar",{}),
    ("scout","Ranking Tool","Player in ranking","pages/1_Ranking.py","jmp_rank",{}),
    ("chart","Player Card","Full stat breakdown","pages/3_Player_Card.py","jmp_card",{}),
    ("chart","Scatter Plot","Highlighted in scatter","pages/4_Scatter.py","jmp_scatter",
     {"scatter_highlight_player":active_player}),
]):
    with col:
        st.markdown(f'<div class="tool-jump"><div class="tj-tag">{tag}</div><div class="tj-name">{name}</div><div class="tj-desc">{desc}</div></div>',
                    unsafe_allow_html=True)
        if st.button("Open →", key=skey):
            st.session_state.dashboard_player         = active_player
            st.session_state.dashboard_team           = active_team
            st.session_state.dashboard_position_group = active_pg
            st.session_state["pre_select_player"]     = active_player
            for sk,sv in extra.items(): st.session_state[sk] = sv
            st.switch_page(page)

st.markdown('<hr style="border:none;border-top:0.5px solid #e0d8cc;margin:8px 0 20px;">', unsafe_allow_html=True)

# ── Row 2: Position bars (left) + Radar (right) ────────────────────────────────
col_bars, col_radar = st.columns([1, 1.1])

with col_bars:
    # Determine bar list from position label
    pos_label_key = str(p_row2.get("Position Label", active_pg))
    bar_stats = DASHBOARD_BARS_PER_POSITION.get(pos_label_key,
                DASHBOARD_BARS_PER_POSITION.get(active_pg, []))

    # Compute percentiles within position pool (no league multiplier — style profile)
    pos_pool = data[data["Main Position"].isin(position_groups[active_pg])].copy()
    avail_bar_stats = [s for s in bar_stats if s in pos_pool.columns and s in p_row2.index]

    st.markdown(f'<div class="section-lbl">Style profile — {pos_label_key}</div>', unsafe_allow_html=True)

    if avail_bar_stats and not pos_pool.empty:
        bars_html = ""
        for stat in avail_bar_stats:
            raw_val = p_row2.get(stat, None)
            try:
                raw_f   = float(str(raw_val).replace(",",".")) if raw_val is not None else None
                pct_val = float(percentileofscore(
                    pos_pool[stat].dropna().astype(float), raw_f or 0, kind="rank"
                ))
            except Exception:
                pct_val = 0.0; raw_f = None

            col_fill = _pct_bar_color(pct_val)
            short    = STAT_SHORT.get(stat, stat[:28])
            raw_str  = f"{raw_f:.2f}" if raw_f is not None else "—"

            # Fouls are negative: flip percentile display
            if "fouls" in stat.lower():
                pct_val = 100 - pct_val
                col_fill = _pct_bar_color(pct_val)

            bars_html += f"""
            <div class="pct-bar-row">
              <div class="pct-bar-stat">
                <span>{short}</span>
                <span style="color:#7a7060;">{raw_str} <span style="color:{col_fill};font-weight:700;">{pct_val:.0f}</span></span>
              </div>
              <div class="pct-bar-track">
                <div class="pct-bar-fill" style="width:{pct_val:.0f}%;background:{col_fill};"></div>
              </div>
            </div>"""
        st.markdown(bars_html, unsafe_allow_html=True)
    else:
        st.info("No stat profile available for this position.")

with col_radar:
    st.markdown('<div class="section-lbl">Radar — Universal</div>', unsafe_allow_html=True)
    try:
        with st.spinner("Building radar…"):
            fig_r, _ = create_radar(data, player_name=active_player, player_team=active_team,
                                    template_key=active_pg, percentile_basis=pct_basis,
                                    radar_type="Universal Radar", show_avg=True, compact=True)
        st.pyplot(fig_r, use_container_width=True)
        st.download_button("Download radar (PNG)", export_full(fig_r),
                           f"{active_player.replace(' ','_')}_radar.png",
                           mime="image/png", key="dl_radar")
    except Exception as e:
        st.error(f"Radar error: {e}")

st.markdown('<hr style="border:none;border-top:0.5px solid #e0d8cc;margin:4px 0 20px;">', unsafe_allow_html=True)

# ── Row 3: Similar players (left) + Fixed scatter (right) ──────────────────────
col_sim, col_scatter = st.columns([1, 1.1])

with col_sim:
    st.markdown('<div class="section-lbl">Similar players — tier-adjusted</div>', unsafe_allow_html=True)
    try:
        sim_pool = data[data["Main Position"].isin(position_groups[sim_pg])].copy()
        if sim_leagues == "Top 5":   sim_pool = sim_pool[sim_pool["League"].isin(TOP5_LEAGUES)]
        elif sim_leagues == "Next 14": sim_pool = sim_pool[sim_pool["League"].isin(NEXT14_LEAGUES)]

        sim_stats = [s for s in ALL_RADAR_STATS if s in sim_pool.columns]
        player_league = str(p_row2.get("League",""))

        # Exclude self
        sim_candidates = sim_pool[
            ~((sim_pool["Player"]==active_player)&(sim_pool["Team within selected timeframe"]==active_team))
        ].copy()

        sim_results = adjusted_similarity(
            target_row=p_row2,
            candidates_df=sim_candidates,
            sim_stats=sim_stats,
            target_league=player_league,
            min_minutes=600,
        )

        top5 = sim_results.head(5)

        # Pool info banner
        st.markdown(f"""
        <div style="background:#f0ebe2;border-left:3px solid #c9a84c;border-radius:0 6px 6px 0;
                    padding:7px 12px;margin-bottom:12px;font-family:'JetBrains Mono',monospace;
                    font-size:9px;color:#7a7060;">
            Similar to <b style="color:#111827;">{active_player}</b> ·
            {sim_pg} · {sim_leagues} · {len(sim_pool)} players · tier-adjusted
        </div>""", unsafe_allow_html=True)

        for _, sr in top5.iterrows():
            sim_name = sr["Player"]
            sim_team = sr["Team within selected timeframe"]
            sim_lg   = sr.get("League","")
            sim_age  = sr.get("Age","—")
            sim_pos  = str(sr.get("Position","")).split(",")[0].strip()
            adj_pct  = sr["adjusted_sim"] * 100
            raw_pct  = sr["raw_sim"] * 100
            badge    = sr.get("tier_badge","Same tier")
            badge_c  = tier_badge_color(badge)

            st.markdown(f"""
            <div class="sim-card">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                  <div class="sim-name">{sim_name}</div>
                  <div class="sim-meta">{sim_pos} · {sim_team} · {sim_lg} · {sim_age} yrs</div>
                  <div class="sim-adj">{adj_pct:.0f}% adj. match
                    <span style="font-size:9px;color:#b0a898;font-weight:400;"> raw {raw_pct:.0f}%</span>
                  </div>
                </div>
                <span class="tier-badge" style="background:{badge_c}22;color:{badge_c};border:0.5px solid {badge_c}44;">
                  {badge}
                </span>
              </div>
            </div>""", unsafe_allow_html=True)
            if st.button(f"Open {sim_name} →", key=f"sim_{sim_name[:8]}"):
                st.session_state.dashboard_player = sim_name
                st.session_state.dashboard_team   = sim_team
                st.session_state.dashboard_position_group = active_pg
                st.rerun()

    except Exception as e:
        st.error(f"Similar players error: {e}")

with col_scatter:
    arch_group = POSITION_TO_ARCHETYPE_GROUP.get(active_pg, "W")
    x_stat, y_stat = DASHBOARD_SCATTER_AXES.get(arch_group, ("xG per 90","xA per 90"))
    st.markdown(f'<div class="section-lbl">Scatter — {STAT_SHORT.get(x_stat,x_stat)} vs {STAT_SHORT.get(y_stat,y_stat)}</div>',
                unsafe_allow_html=True)
    try:
        pool_sc = data[data["Main Position"].isin(position_groups[active_pg])].dropna(subset=[x_stat,y_stat]).copy()
        if not pool_sc.empty:
            fig_s = go.Figure()
            for _, row in pool_sc.iterrows():
                it = row["Player"] == active_player
                fig_s.add_trace(go.Scatter(
                    x=[row[x_stat]], y=[row[y_stat]], mode="markers",
                    marker=dict(size=12 if it else 5, color="#c9a84c" if it else "#111827",
                                opacity=1.0 if it else 0.18,
                                symbol="star" if it else "circle",
                                line=dict(width=1.5 if it else 0, color="#111827")),
                    hovertemplate=f"<b>{row['Player']}</b><br>{x_stat}: {row[x_stat]:.2f}<br>{y_stat}: {row[y_stat]:.2f}<extra></extra>",
                    showlegend=False))
            me = pool_sc[pool_sc["Player"]==active_player]
            if not me.empty:
                fig_s.add_annotation(x=me[x_stat].iloc[0], y=me[y_stat].iloc[0],
                                     text=active_player, showarrow=True, arrowhead=2,
                                     arrowcolor="#c9a84c", arrowwidth=1.5, ax=20, ay=-30,
                                     font=dict(size=11, color="#111827"),
                                     bgcolor="#f0ebe2", bordercolor="#c9a84c", borderpad=4)
            fig_s.add_vline(x=pool_sc[x_stat].mean(), line=dict(color="rgba(17,24,39,0.12)",dash="dot",width=1))
            fig_s.add_hline(y=pool_sc[y_stat].mean(), line=dict(color="rgba(17,24,39,0.12)",dash="dot",width=1))
            fig_s.update_layout(
                paper_bgcolor="#faf7f2", plot_bgcolor="#faf7f2", height=420,
                margin=dict(l=50,r=20,t=30,b=50),
                title=dict(text=f"<sup>{active_pg} · {len(pool_sc)} players</sup>",
                           font=dict(size=11,color="#b0a898"),x=0.01),
                xaxis=dict(title=STAT_SHORT.get(x_stat,x_stat), tickfont=dict(color="#7a7060"),
                           gridcolor="rgba(0,0,0,0.04)", zeroline=False),
                yaxis=dict(title=STAT_SHORT.get(y_stat,y_stat), tickfont=dict(color="#7a7060"),
                           gridcolor="rgba(0,0,0,0.04)", zeroline=False),
                hoverlabel=dict(bgcolor="#111827", font=dict(size=11,color="white")),
            )
            st.plotly_chart(fig_s, use_container_width=True)
    except Exception as e:
        st.error(f"Scatter error: {e}")

st.markdown('<hr style="border:none;border-top:0.5px solid #e0d8cc;margin:4px 0 20px;">', unsafe_allow_html=True)

# ── Row 4: Trend + context (left) + Scout notes (right) ───────────────────────
col_trend, col_notes = st.columns([1, 1.1])

with col_trend:
    st.markdown('<div class="section-lbl">Season trend</div>', unsafe_allow_html=True)
    try:
        both_seasons = load_both_seasons(st.session_state["_min_min"])
        trend_points = []
        for s, df_s in sorted(both_seasons.items(), reverse=True):
            pool_s = compute_scores(df_s, active_pg, league_template_score, score_mode)
            m = (pool_s["Player"]==active_player)&(pool_s["Team within selected timeframe"]==active_team)
            if m.any():
                trend_points.append({"season":s, "score":float(pool_s[m]["overall_adj"].iloc[0])})

        if len(trend_points) >= 2:
            seasons_x  = [p["season"] for p in reversed(trend_points)]
            scores_y   = [p["score"]  for p in reversed(trend_points)]
            delta      = scores_y[-1] - scores_y[-2]
            dcol       = "#1a7a45" if delta > 0 else ("#c0392b" if delta < 0 else "#7a7060")
            dsign      = "+" if delta > 0 else ""
            darrow     = "↑" if delta > 0 else ("↓" if delta < 0 else "—")

            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=seasons_x, y=scores_y, mode="lines+markers+text",
                line=dict(color="#c9a84c", width=2),
                marker=dict(size=10, color="#c9a84c", line=dict(width=2, color="#111827")),
                text=[f"{s:.0f}" for s in scores_y],
                textposition="top center",
                textfont=dict(family="JetBrains Mono", size=11, color="#111827"),
            ))
            fig_t.update_layout(
                paper_bgcolor="#faf7f2", plot_bgcolor="#faf7f2", height=200,
                margin=dict(l=40,r=20,t=20,b=30), showlegend=False,
                xaxis=dict(tickfont=dict(color="#7a7060", size=10), gridcolor="rgba(0,0,0,0.04)"),
                yaxis=dict(tickfont=dict(color="#7a7060", size=10), gridcolor="rgba(0,0,0,0.04)",
                           range=[max(0,min(scores_y)-10), min(100,max(scores_y)+10)]),
            )
            st.plotly_chart(fig_t, use_container_width=True)
            st.markdown(f"""
            <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#b0a898;margin-top:-8px;margin-bottom:14px;">
              {seasons_x[-2]} → {seasons_x[-1]} &nbsp;
              <span style="color:{dcol};font-weight:700;">{darrow} {dsign}{delta:.1f}</span>
            </div>""", unsafe_allow_html=True)
        elif len(trend_points) == 1:
            p = trend_points[0]
            st.markdown(f"""<div style="background:#f0ebe2;border-radius:6px;padding:12px 16px;text-align:center;">
              <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#b0a898;margin-bottom:4px;">{p['season']}</div>
              <div style="font-size:28px;font-weight:700;color:{_c(p['score'])};">{p['score']:.0f}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#b0a898;margin-top:4px;">Single season available</div>
            </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.info(f"Trend unavailable: {e}")

    # Contract + value context
    st.markdown(f"""
    <div style="background:#f0ebe2;border-radius:6px;padding:12px 14px;margin-top:8px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#b0a898;
                  text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Player context</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:8px;color:#b0a898;margin-bottom:2px;">Market value</div>
          <div style="font-size:14px;font-weight:700;color:#111827;">{val_str}</div>
        </div>
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:8px;color:#b0a898;margin-bottom:2px;">Contract expires</div>
          <div style="font-size:14px;font-weight:700;color:#111827;">{contract}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

with col_notes:
    st.markdown('<div class="section-lbl">Scout report notes</div>', unsafe_allow_html=True)
    note_key = f"{active_player}__{active_team}"
    existing = st.session_state.scout_notes.get(note_key,"")
    notes = st.text_area("Notes", value=existing, height=220,
                         placeholder="Strengths:\n—\n\nWeaknesses:\n—\n\nConclusion:\n—",
                         label_visibility="collapsed", key=f"notes_{note_key}")
    n1,n2 = st.columns(2)
    with n1:
        if st.button("Save notes", key="save_n"):
            st.session_state.scout_notes[note_key] = notes
            st.success("Notes saved.")
    with n2:
        if notes.strip():
            st.download_button("Export (TXT)",
                               f"Scout Report — {active_player} ({active_team})\n{'='*50}\n\n{notes}".encode(),
                               f"{active_player.replace(' ','_')}_scout_report.txt",
                               mime="text/plain", key="dl_n")

st.markdown("""
<div style="margin-top:20px;padding-top:12px;border-top:0.5px solid #e0d8cc;
            display:flex;justify-content:space-between;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#b0a898;
               text-transform:uppercase;letter-spacing:0.08em;">Target Scouting · Player Dashboard</span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#b0a898;
               text-transform:uppercase;letter-spacing:0.08em;">Wyscout 26 Mar 2026</span>
</div>
""", unsafe_allow_html=True)
