import os, sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.data_processing import preprocess_data, load_season_data, SEASON_LABELS
from shared.season_filter import render_season_filter_compare
from shared.styles import BASE_CSS
from shared.sidebar_nav import render_sidebar_nav
from shared.scoring import compute_scores
from shared.templates import (
    position_groups, position_map, report_template,
    position_category_weights, LEAGUE_MULTIPLIERS_ALL,
    LEAGUE_DISPLAY_NAMES, TOP5_LEAGUES, NEXT14_LEAGUES,
)
from radar_app.radar import (
    create_comparison_radar, export_full,
    _build_pool, _compute_percentiles,
)

st.set_page_config(
    page_title="Compare · Target Scouting",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(BASE_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
.block-container { padding-top:1.6rem !important; max-width:1280px !important; }

/* Player cards — gold (P1) and steel blue (P2) */
.p1-card {
    background: rgba(201,168,76,0.07);
    border: 1px solid rgba(201,168,76,0.3);
    border-radius: 8px; padding: 14px 18px;
}
.p2-card {
    background: rgba(74,127,165,0.07);
    border: 1px solid rgba(74,127,165,0.3);
    border-radius: 8px; padding: 14px 18px;
}
.p-name  { font-size: 18px; font-weight: 700; letter-spacing: -0.01em; }
.p1-name { color: #c9a84c; }
.p2-name { color: #4a7fa5; }
.p-meta  { font-family: 'JetBrains Mono', monospace; font-size: 10px;
           color: #7a7060; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.06em; }
.p-season-badge {
    font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 700;
    padding: 2px 7px; border-radius: 3px; margin-left: 6px;
}
.p1-season-badge { background: rgba(201,168,76,0.15); color: #c9a84c; }
.p2-season-badge { background: rgba(74,127,165,0.15); color: #4a7fa5; }

/* Section label */
.section-lbl {
    font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 700;
    color: #b0a898; text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 10px; display: flex; align-items: center; gap: 10px;
}
.section-lbl::after { content: ''; flex: 1; height: 0.5px; background: #e0d8cc; }

/* Mirror bars */
.mirror-legend {
    display: grid; grid-template-columns: 1fr 80px 1fr;
    gap: 8px; margin-bottom: 10px;
}
.mirror-row {
    display: grid; grid-template-columns: 1fr 80px 1fr;
    gap: 8px; align-items: center; margin-bottom: 11px;
}
.mirror-lbl {
    text-align: center; font-family: 'JetBrains Mono', monospace;
    font-size: 9px; font-weight: 600; color: #7a7060;
    text-transform: uppercase; letter-spacing: 0.06em;
    display: flex; align-items: center; justify-content: center; gap: 4px;
}
.win-badge {
    font-family: 'JetBrains Mono', monospace; font-size: 8px; font-weight: 700;
    padding: 1px 5px; border-radius: 3px;
}
.win1-badge { background: rgba(201,168,76,0.2); color: #c9a84c; }
.win2-badge { background: rgba(74,127,165,0.2); color: #4a7fa5; }
.mirror-left  { display: flex; align-items: center; gap: 6px; justify-content: flex-end; }
.mirror-right { display: flex; align-items: center; gap: 6px; }
.mirror-track { flex: 1; height: 9px; background: #f0ebe2; border-radius: 4px; overflow: hidden; }
.mirror-fill  { height: 9px; border-radius: 4px; }

/* Stat table */
.stat-tbl { width: 100%; border-collapse: collapse; font-size: 12px; }
.stat-tbl th {
    font-family: 'JetBrains Mono', monospace; font-size: 8px; text-transform: uppercase;
    letter-spacing: 0.08em; color: #b0a898; padding: 8px 10px;
    border-bottom: 1px solid #e0d8cc; text-align: center;
}
.stat-tbl th.sn { text-align: left; }
.stat-tbl td { padding: 7px 10px; border-bottom: 0.5px solid #f0ebe2; text-align: center; }
.stat-tbl td.sn { text-align: left; color: #7a7060; font-size: 11px; }
.win  { font-weight: 700; }
.win1 { color: #c9a84c; }
.win2 { color: #4a7fa5; }
.lose { color: #b0a898; }
.draw { color: #111827; }
</style>
""", unsafe_allow_html=True)

# Player colors
C1 = "#c9a84c"   # gold — Player 1
C2 = "#4a7fa5"   # steel blue — Player 2

def _c(v):
    if v >= 75: return "#1a7a45"
    elif v >= 50: return "#91cf60"
    elif v >= 25: return "#f0a500"
    return "#d73027"

def _wt(stats, weights, pct):
    tw = sum(weights.get(s, 0) for s in stats if s in pct)
    return 0 if tw == 0 else sum(pct[s]*weights[s] for s in stats if s in pct and s in weights) / tw

def _overall_adj(cat_scores, weights, score_mode="Adjusted (recommended)"):
    tw  = sum(weights.values())
    raw = 0 if tw == 0 else sum(cat_scores.get(c,0)*w for c,w in weights.items()) / tw
    if score_mode == "Adjusted (recommended)":
        return (raw / 100) ** 0.45 * 100
    return raw

STAT_SHORT = {
    "Non-penalty goals per 90":"NP Goals /90","xG per 90":"xG /90","xG per shot":"xG /shot",
    "Finishing":"Finishing","Shots per 90":"Shots /90","Shots on target, %":"On target %",
    "Touches in box per 90":"Box touches /90","Assists per 90":"Assists /90","xA per 90":"xA /90",
    "Shot assists per 90":"Shot ast. /90","Key passes per pass":"Key passes /pass",
    "Through passes per pass":"Through passes /pass",
    "Accurate crosses per received pass":"Crosses /rec pass","Accurate crosses, %":"Crosses %",
    "Successful dribbles per received pass":"Dribbles /rec pass","Successful dribbles, %":"Dribbles %",
    "Offensive duels won, %":"Off. duels won %",
    "Progressive runs per received pass":"Prog. runs /rec pass",
    "Completed progressive passes per 90":"Prog. passes /90","Accurate progressive passes, %":"Prog. pass %",
    "Completed passes to final third per 90":"Passes F3 /90","Accurate passes to final third, %":"Passes F3 %",
    "Completed passes to penalty area per 90":"Passes PA /90","Accurate passes to penalty area, %":"Passes PA %",
    "Deep completions per 90":"Deep compl. /90","PAdj Defensive duels won per 90":"Def duels /90",
    "Defensive duels won, %":"Def duels won %","PAdj Aerial duels won per 90":"Aerial duels /90",
    "Aerial duels won, %":"Aerial duels won %","PAdj Interceptions":"Interceptions",
    "PAdj Successful defensive actions per 90":"Def actions /90","Fouls per 90":"Fouls /90",
}

ALL_STATS   = [s for g in report_template.values() for s in g["stats"]]
CATS        = list(report_template.keys())
CAT_SHORT   = {
    "Goalscoring":    "Goals",
    "Chance creation":"Chance cr.",
    "Dribbling":      "Dribbling",
    "Passing":        "Passing",
    "Defending":      "Defending",
}

@st.cache_data
def load_data(season: str = "2025/26", min_minutes: int = 0):
    return load_season_data(season, min_minutes)

@st.cache_data
def load_both_seasons(min_minutes: int = 0):
    return {s: load_season_data(s, min_minutes) for s in SEASON_LABELS}

if "_cmp_min_min" not in st.session_state:
    st.session_state["_cmp_min_min"] = 900

seasons_data = load_both_seasons(st.session_state["_cmp_min_min"])
data = seasons_data["2025/26"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav("compare")

    st.markdown(
        '<hr style="border:none;border-top:0.5px solid rgba(255,255,255,0.1);margin:4px 0 10px;">',
        unsafe_allow_html=True,
    )
    st.markdown('<span class="sb-section-label">Min. minutes played</span>', unsafe_allow_html=True)
    cmp_min_min = st.select_slider(
        "cmp_min_min_sl",
        options=[0, 200, 400, 600, 800, 900, 1000, 1200, 1500, 2000],
        value=st.session_state["_cmp_min_min"],
        label_visibility="collapsed",
        format_func=lambda x: f"{x}+" if x > 0 else "All",
    )
    if cmp_min_min != st.session_state["_cmp_min_min"]:
        st.session_state["_cmp_min_min"] = cmp_min_min
        st.rerun()

    st.markdown('<span class="sb-section-label">Position group</span>', unsafe_allow_html=True)
    pg_list = list(position_groups.keys())
    pre_pg  = st.session_state.get("dashboard_position_group") or pg_list[0]
    position_group = st.selectbox(
        "pg", pg_list,
        index=pg_list.index(pre_pg) if pre_pg in pg_list else 0,
        label_visibility="collapsed",
    )

    st.markdown('<span class="sb-section-label">League</span>', unsafe_allow_html=True)
    league_filter = st.radio("lf", ["Top 5", "Next 14", "Both"],
                             horizontal=True, label_visibility="collapsed")

    st.markdown('<span class="sb-section-label">Age range</span>', unsafe_allow_html=True)
    age_min, age_max = st.slider("age_range", 15, 45, (16, 40), label_visibility="collapsed")

    st.markdown('<span class="sb-section-label">Foot</span>', unsafe_allow_html=True)
    foot_filter = st.multiselect("foot", ["left", "right", "both"], label_visibility="collapsed")

    st.markdown(
        '<span class="sb-section-label" style="color:rgba(201,168,76,0.65);">'
        'Overall score basis'
        '<span style="background:rgba(201,168,76,0.15);border:0.5px solid rgba(201,168,76,0.4);'
        'border-radius:3px;padding:1px 5px;font-size:8px;color:#c9a84c;margin-left:5px;">score</span>'
        '</span>',
        unsafe_allow_html=True
    )
    league_template_score = st.radio(
        "lts", ["Top 5 leagues", "Next 14 competitions", "Both"],
        label_visibility="collapsed", key="cmp_lt_score"
    )
    st.markdown(
        '<div style="font-size:9px;color:rgba(255,255,255,0.25);margin:2px 0 8px;line-height:1.4;">'
        'Affects: category scores, overall rating, stat table</div>',
        unsafe_allow_html=True
    )

    st.markdown('<span class="sb-section-label">Score type</span>', unsafe_allow_html=True)
    score_mode = st.radio("sm", ["Adjusted (recommended)", "Model (raw)"], label_visibility="collapsed")

    st.markdown(
        '<span class="sb-section-label" style="color:rgba(133,183,235,0.8);">'
        'Radar percentile basis'
        '<span style="background:rgba(41,128,185,0.15);border:0.5px solid rgba(41,128,185,0.4);'
        'border-radius:3px;padding:1px 5px;font-size:8px;color:#85B7EB;margin-left:5px;">radar</span>'
        '</span>',
        unsafe_allow_html=True
    )
    pct_basis = st.radio("pb", ["T5 only", "Next 14 only", "Own league"], label_visibility="collapsed")
    st.markdown(
        '<div style="font-size:9px;color:rgba(255,255,255,0.25);margin:2px 0 8px;line-height:1.4;">'
        'Affects: overlay radar chart only</div>',
        unsafe_allow_html=True
    )

    # Build filtered pool
    all_data  = pd.concat(seasons_data.values(), ignore_index=True)
    positions = position_groups[position_group]
    filtered  = all_data[all_data["Main Position"].isin(positions)].copy()
    if league_filter == "Top 5":   filtered = filtered[filtered["League"].isin(TOP5_LEAGUES)]
    elif league_filter == "Next 14": filtered = filtered[filtered["League"].isin(NEXT14_LEAGUES)]
    filtered = filtered[filtered["Age"].between(age_min, age_max)]
    if foot_filter: filtered = filtered[filtered["Foot"].isin(foot_filter)]
    filtered_players = sorted(filtered["Player"].unique())

    if len(filtered_players) < 2:
        st.warning(f"Only {len(filtered_players)} player(s) match. Loosen filters.")
        st.stop()

    st.markdown(f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;color:#b0a898;">{len(filtered_players)} players match filters</span>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<span class="sb-section-label">Player 1</span>', unsafe_allow_html=True)
    pre1   = st.session_state.get("pre_select_player") or st.session_state.get("dashboard_player")
    pi1    = filtered_players.index(pre1) if pre1 in filtered_players else 0
    p1_name = st.selectbox("p1", filtered_players, index=pi1, label_visibility="collapsed")

    p1_seasons_avail = sorted(filtered[filtered["Player"] == p1_name]["_season"].unique(), reverse=True)
    p1_season = render_season_filter_compare("p1", "Season · Player 1",
                                             default_season=p1_seasons_avail[0] if p1_seasons_avail else "2025/26")

    p1_rows = filtered[(filtered["Player"] == p1_name) & (filtered["_season"] == p1_season)]
    if p1_rows.empty:
        st.warning(f"{p1_name} has no data in {p1_season} with current filters.")
        st.stop()
    p1_teams = p1_rows["Team within selected timeframe"].unique()
    p1_team  = (st.selectbox("t1", p1_teams, label_visibility="collapsed")
                if len(p1_teams) > 1 else p1_teams[0])
    if len(p1_teams) == 1: st.caption(f"🏟️ {p1_team}  ·  {p1_season}")
    p1_row = p1_rows[p1_rows["Team within selected timeframe"] == p1_team].iloc[0]

    st.markdown('<span class="sb-section-label">Player 2</span>', unsafe_allow_html=True)
    p2_name = st.selectbox("p2", filtered_players, label_visibility="collapsed")

    p2_seasons_avail = sorted(filtered[filtered["Player"] == p2_name]["_season"].unique(), reverse=True)
    if p2_name == p1_name:
        other_season = next((s for s in p2_seasons_avail if s != p1_season), p2_seasons_avail[0] if p2_seasons_avail else "2024/25")
        p2_default = other_season
    else:
        p2_default = p1_season
    p2_season = render_season_filter_compare("p2", "Season · Player 2", default_season=p2_default)

    p2_rows = filtered[(filtered["Player"] == p2_name) & (filtered["_season"] == p2_season)]
    if p2_rows.empty:
        st.warning(f"{p2_name} has no data in {p2_season} with current filters.")
        st.stop()
    p2_teams = p2_rows["Team within selected timeframe"].unique()
    p2_team  = (st.selectbox("t2", p2_teams, label_visibility="collapsed")
                if len(p2_teams) > 1 else p2_teams[0])
    if len(p2_teams) == 1: st.caption(f"🏟️ {p2_team}  ·  {p2_season}")
    p2_row = p2_rows[p2_rows["Team within selected timeframe"] == p2_team].iloc[0]

    st.markdown("---")
    st.markdown('<span class="sb-section-label">Key stats to compare</span>', unsafe_allow_html=True)
    avail_stats = [s for s in ALL_STATS if s in data.columns]
    safe_defaults = [s for s in [
        "xG per 90","xA per 90","Successful dribbles per received pass",
        "Offensive duels won, %","Progressive runs per received pass",
        "Defensive duels won, %","Accurate crosses per received pass","PAdj Interceptions",
    ] if s in avail_stats]
    key_stats = st.multiselect(
        "ks", avail_stats, default=safe_defaults, max_selections=10,
        label_visibility="collapsed",
    )

    generate = st.button("Compare", use_container_width=True)

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:20px;">
  <h1 style="font-size:20px;font-weight:700;letter-spacing:-0.01em;">Compare Players</h1>
  <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#b0a898;text-transform:uppercase;letter-spacing:0.08em;">
    Head-to-head analysis
  </span>
</div>
""", unsafe_allow_html=True)

if not generate:
    st.markdown(f"""
    <div style="background:#f0ebe2;border:1.5px dashed #c9a84c;border-radius:10px;
                padding:2.5rem;text-align:center;">
        <div style="font-size:16px;font-weight:700;color:#111827;margin-bottom:8px;">
            Select two players in the sidebar, then click <em>Compare</em>
        </div>
        <div style="font-size:13px;color:#7a7060;margin-top:8px;">
            Pool: <b>{len(filtered_players)}</b> {position_group}s ·
            <span style="color:{C1};font-weight:700;">{p1_name}</span> vs
            <span style="color:{C2};font-weight:700;">{p2_name}</span>
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Compute scores ─────────────────────────────────────────────────────────────
def get_scores(p_row, p_name, p_team, season):
    season_df = seasons_data.get(season, seasons_data["2025/26"])
    scored = compute_scores(season_df, position_group, league_template_score, score_mode)
    p_mask = (scored["Player"] == p_name) & (scored["Team within selected timeframe"] == p_team)
    cw     = position_category_weights.get(position_group, {})
    if not p_mask.any():
        return pd.Series(dtype=float), {c: 0.0 for c in report_template}, 0.0
    row = scored[p_mask].iloc[0]
    cat_scores = {c: float(row.get(f"{c}_score", 0)) for c in report_template}
    ov = float(row.get("overall_adj", 0))
    avail = [s for s in [s2 for g in report_template.values() for s2 in g["stats"]] if s in season_df.columns]
    try:
        pool = _build_pool(season_df, position_group, pct_basis, p_row.get("League", None))
        pct  = _compute_percentiles(pool, p_name, p_team, avail)
    except:
        pct = pd.Series(dtype=float)
    return pct, cat_scores, ov

with st.spinner("Computing scores…"):
    pct1, cat1, ov1 = get_scores(p1_row, p1_name, p1_team, p1_season)
    pct2, cat2, ov2 = get_scores(p2_row, p2_name, p2_team, p2_season)

# ── Player meta ────────────────────────────────────────────────────────────────
def p_meta(row, season):
    pos  = str(row.get("Position","")).split(",")[0].strip()
    base_age = row.get("Age","—")
    try:
        base_age_int = int(float(base_age))
        season_year  = int(season.split("/")[0])
        age_display  = str(base_age_int + (season_year - 2025))
    except:
        age_display = str(base_age)
    mins   = row.get("Minutes played","—")
    lg     = LEAGUE_DISPLAY_NAMES.get(row.get("League",""), row.get("League",""))
    return pos, age_display, mins, lg

pos1,age1,mins1,lg1 = p_meta(p1_row, p1_season)
pos2,age2,mins2,lg2 = p_meta(p2_row, p2_season)

# ── Header cards ──────────────────────────────────────────────────────────────
h1, vs_col, h2 = st.columns([10,1,10])
with h1:
    st.markdown(f"""
    <div class="p1-card">
      <div style="font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.1em;color:rgba(201,168,76,0.6);margin-bottom:6px;">
        Player 1 <span class="p1-season-badge">{p1_season}</span>
      </div>
      <div class="p-name p1-name">{p1_name}</div>
      <div class="p-meta">{pos1} · {p1_team} · {lg1} · {age1} yrs · {mins1} min</div>
      <div style="display:flex;align-items:center;gap:10px;margin-top:10px;">
        <div style="background:{_c(ov1)};color:white;width:42px;height:42px;border-radius:6px;
                    display:flex;align-items:center;justify-content:center;
                    font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;">{ov1:.0f}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#b0a898;">
          Overall · {league_template_score.split()[0]} · {score_mode.split()[0]}
        </div>
      </div>
    </div>""", unsafe_allow_html=True)
with vs_col:
    st.markdown('<div style="display:flex;align-items:center;justify-content:center;height:100%;font-family:\'JetBrains Mono\',monospace;font-size:11px;font-weight:700;color:#b0a898;">VS</div>', unsafe_allow_html=True)
with h2:
    st.markdown(f"""
    <div class="p2-card">
      <div style="font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.1em;color:rgba(74,127,165,0.8);margin-bottom:6px;">
        Player 2 <span class="p2-season-badge">{p2_season}</span>
      </div>
      <div class="p-name p2-name">{p2_name}</div>
      <div class="p-meta">{pos2} · {p2_team} · {lg2} · {age2} yrs · {mins2} min</div>
      <div style="display:flex;align-items:center;gap:10px;margin-top:10px;">
        <div style="background:{_c(ov2)};color:white;width:42px;height:42px;border-radius:6px;
                    display:flex;align-items:center;justify-content:center;
                    font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;">{ov2:.0f}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#b0a898;">
          Overall · {league_template_score.split()[0]} · {score_mode.split()[0]}
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── Left: mirror bars + stat table | Right: radar ─────────────────────────────
col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.markdown('<div class="section-lbl">Category scores</div>', unsafe_allow_html=True)
    # Legend
    st.markdown(f"""
    <div class="mirror-legend">
      <div style="text-align:right;font-size:12px;font-weight:700;color:{C1};">
        {p1_name.split()[-1]}
      </div>
      <div></div>
      <div style="font-size:12px;font-weight:700;color:{C2};">
        {p2_name.split()[-1]}
      </div>
    </div>""", unsafe_allow_html=True)

    for cat in CATS:
        v1    = cat1.get(cat, 0)
        v2    = cat2.get(cat, 0)
        short = CAT_SHORT.get(cat, cat)
        win1  = v1 > v2
        win2  = v2 > v1
        c1t   = C1 if win1 else ("#b0a898" if win2 else "#111827")
        c2t   = C2 if win2 else ("#b0a898" if win1 else "#111827")
        wb    = ""
        if win1: wb = '<span class="win-badge win1-badge">↑</span>'
        elif win2: wb = '<span class="win-badge win2-badge">↑</span>'

        st.markdown(f"""
        <div class="mirror-row">
          <div class="mirror-left">
            <span style="font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;
                         color:{c1t};min-width:26px;text-align:right;">{v1:.0f}</span>
            <div class="mirror-track">
              <div class="mirror-fill" style="width:{v1:.0f}%;background:{C1};opacity:0.8;float:right;"></div>
            </div>
          </div>
          <div class="mirror-lbl">{short} {wb}</div>
          <div class="mirror-right">
            <div class="mirror-track">
              <div class="mirror-fill" style="width:{v2:.0f}%;background:{C2};opacity:0.8;"></div>
            </div>
            <span style="font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;
                         color:{c2t};min-width:26px;">{v2:.0f}</span>
          </div>
        </div>""", unsafe_allow_html=True)

    # Stat table
    if key_stats:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-lbl">Key stat comparison</div>', unsafe_allow_html=True)
        p1s = p1_name.split()[-1]
        p2s = p2_name.split()[-1]
        tbl = f"""<table class="stat-tbl">
        <thead><tr>
          <th class="sn">Stat</th>
          <th style="color:{C1};">{p1s}</th>
          <th style="color:{C2};">{p2s}</th>
        </tr></thead><tbody>"""

        for stat in key_stats:
            short = STAT_SHORT.get(stat, stat)
            raw1  = p1_row.get(stat, None)
            raw2  = p2_row.get(stat, None)
            pv1   = float(pct1.get(stat, 50)) if stat in pct1.index else 50
            pv2   = float(pct2.get(stat, 50)) if stat in pct2.index else 50
            try:    d1 = f"{float(raw1):.2f}" if raw1 is not None else "—"
            except: d1 = str(raw1) if raw1 is not None else "—"
            try:    d2 = f"{float(raw2):.2f}" if raw2 is not None else "—"
            except: d2 = str(raw2) if raw2 is not None else "—"
            if abs(pv1 - pv2) < 2:
                cls1 = cls2 = "draw"
            elif pv1 > pv2:
                cls1, cls2 = "win win1", "lose"
            else:
                cls1, cls2 = "lose", "win win2"
            tbl += (f'<tr><td class="sn">{short}</td>'
                    f'<td class="{cls1}">{d1}</td>'
                    f'<td class="{cls2}">{d2}</td></tr>')
        tbl += "</tbody></table>"
        st.markdown(tbl, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-lbl">Radar overlay</div>', unsafe_allow_html=True)
    try:
        with st.spinner("Building overlay radar…"):
            import pandas as _pd
            data_p1 = seasons_data.get(p1_season, seasons_data["2025/26"])
            data_p2 = seasons_data.get(p2_season, seasons_data["2025/26"])

            if p1_name == p2_name and p1_team == p2_team and p1_season != p2_season:
                _d2_mod = data_p2.copy()
                _SUFFIX = "__prev__"
                mask_p2 = (
                    (_d2_mod["Player"] == p2_name) &
                    (_d2_mod["Team within selected timeframe"] == p2_team)
                )
                _d2_mod.loc[mask_p2, "Team within selected timeframe"] = p2_team + _SUFFIX
                radar_data    = _pd.concat([data_p1, _d2_mod], ignore_index=True)
                _p2_team_radar = p2_team + _SUFFIX
            else:
                radar_data    = _pd.concat([data_p1, data_p2], ignore_index=True).drop_duplicates(
                    subset=["Player", "Team within selected timeframe"], keep="first"
                )
                _p2_team_radar = p2_team

            fig_cmp, _ = create_comparison_radar(
                radar_data,
                player1_name=p1_name, player1_team=p1_team,
                player2_name=p2_name, player2_team=_p2_team_radar,
                template_key=position_group,
                percentile_basis=pct_basis,
                radar_type="Universal Radar",
                mode="overlay",
                show_avg=False,
                compact=True,
            )
        st.pyplot(fig_cmp, use_container_width=True)
    except Exception as e:
        st.error(f"Radar error: {e}")

st.markdown(f"""
<div style="background:#f0ebe2;border-radius:6px;padding:8px 16px;margin-top:20px;
            display:flex;justify-content:space-between;align-items:center;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#b0a898;">
        Target Scouting · {p1_name} vs {p2_name}
    </span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#b0a898;">
        <span style="color:#c9a84c;">Score:</span> {league_template_score} · {score_mode.split()[0]} &nbsp;|&nbsp;
        <span style="color:#85B7EB;">Radar:</span> {pct_basis} &nbsp;|&nbsp;
        {position_group} · Wyscout 26 Mar 2026
    </span>
</div>""", unsafe_allow_html=True)
