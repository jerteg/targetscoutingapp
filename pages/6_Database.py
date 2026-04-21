import os, sys, datetime
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np

from shared.data_processing import load_season_data, SEASON_LABELS
from shared.styles import BASE_CSS
from shared.sidebar_nav import render_sidebar_nav
from shared.scoring import compute_scores
from shared.templates import (
    position_groups, LEAGUE_DISPLAY_NAMES,
    TOP5_LEAGUES, NEXT14_LEAGUES,
)

POS_SHORT = {
    "Striker":"ST","Left Wing":"LW","Right Wing":"RW","Attacking Midfielder":"AM",
    "Central Midfielder":"CM","Defensive Midfielder":"DM","Left-Back":"LB",
    "Right-Back":"RB","Centre-Back":"CB","Goalkeeper":"GK",
}
LEAGUE_REF_MINUTES = {
    "Championship":4692,"Segunda Division":4606,"Pro League":4053,"Liga Pro":4042,
    "MLS":3968,"Serie A BRA":3812,"Premier League":3807,"La Liga":3797,
    "Swiss Super League":3722,"Italian Serie A":3707,"Super Lig":3591,
    "Eredivisie":3486,"Liga Profesional":3481,"Prva HNL":3403,"Ekstraklasa":3391,
    "Ligue 1":3389,"Bundesliga":3363,"Primeira Liga":3288,"Superligaen":3279,"Eliteserien":3004,
}
AVAIL_COLOR = {"green":"#1a7a45","orange":"#f0a500","red":"#d73027"}
AVAIL_LABEL = {"green":"Available","orange":"Rotation","red":"Limited"}

def compute_availability(df):
    import pandas as _pd
    mins    = _pd.to_numeric(df["Minutes played"], errors="coerce").fillna(0)
    lg_ref  = df["League"].map(LEAGUE_REF_MINUTES).fillna(3420)
    pct     = mins / lg_ref.clip(lower=1)
    avail   = _pd.Series("orange", index=df.index)
    avail[pct >= 0.60] = "green"
    avail[pct < 0.35]  = "red"
    return avail

def _c(v):
    if v >= 75: return "#1a7a45"
    elif v >= 50: return "#91cf60"
    elif v >= 25: return "#f0a500"
    return "#d73027"

st.set_page_config(page_title="Database · Target Scouting", layout="wide", initial_sidebar_state="expanded")
st.markdown(BASE_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
.block-container { padding-top:1.6rem !important; max-width:1400px !important; }
.db-table { width:100%; border-collapse:collapse; }
.db-table thead th { font-family:'JetBrains Mono',monospace; font-size:8px; font-weight:700;
    text-transform:uppercase; letter-spacing:0.1em; color:#b0a898; padding:10px 8px;
    border-bottom:1px solid #e0d8cc; background:#f0ebe2; text-align:center; white-space:nowrap; }
.db-table thead th.l { text-align:left; padding-left:12px; }
.db-table tbody tr { cursor:pointer; transition:background 0.1s; }
.db-table tbody tr:hover { background:#f0ebe2; }
.db-table tbody tr:hover .row-action { opacity:1; }
.db-table tbody td { padding:9px 8px; border-bottom:0.5px solid #e0d8cc;
    font-size:12px; text-align:center; vertical-align:middle; }
.db-table tbody td.l { text-align:left; padding-left:12px; }
.score-chip { display:inline-flex; align-items:center; justify-content:center;
    width:34px; height:22px; border-radius:3px; font-family:'JetBrains Mono',monospace;
    font-size:11px; font-weight:700; color:white; }
.score-ov { display:inline-flex; align-items:center; justify-content:center;
    min-width:36px; height:26px; border-radius:5px; font-family:'JetBrains Mono',monospace;
    font-size:13px; font-weight:700; color:white; padding:0 6px; }
.pos-chip { background:#f0ebe2; color:#7a7060; border-radius:3px; padding:2px 6px;
    font-family:'JetBrains Mono',monospace; font-size:8px; font-weight:700; text-transform:uppercase; }
.row-action { opacity:0; font-family:'JetBrains Mono',monospace; font-size:9px; font-weight:700;
    color:#c9a84c; text-transform:uppercase; letter-spacing:0.06em;
    background:rgba(201,168,76,0.1); border:0.5px solid rgba(201,168,76,0.3);
    padding:3px 8px; border-radius:3px; white-space:nowrap; }
.avail-dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
section[data-testid="stSidebar"] input[type="number"] { color:white !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(season="2025/26", min_minutes=0):
    return load_season_data(season, min_minutes)

@st.cache_data
def load_both_seasons_cached(min_minutes=0):
    return {s: load_season_data(s, min_minutes) for s in SEASON_LABELS}

@st.cache_data
def compute_database(_data, league_template, score_mode, season_key="2025/26"):
    results = []
    for pg in position_groups.keys():
        scored = compute_scores(_data, pg, league_template, score_mode)
        if not scored.empty:
            scored["_pg"] = pg
            results.append(scored)
    if not results: return pd.DataFrame()
    out = pd.concat(results, ignore_index=True)
    out = out.sort_values("overall_adj", ascending=False).drop_duplicates(
        subset=["Player","Team within selected timeframe"], keep="first"
    )
    return out

@st.cache_data
def compute_database_multiseasons(_data_curr, _data_prev, league_template, score_mode,
                                   season_key_curr="2025/26", season_key_prev="2024/25"):
    curr = compute_database(_data_curr, league_template, score_mode, season_key=season_key_curr)
    prev = compute_database(_data_prev, league_template, score_mode, season_key=season_key_prev)
    if curr.empty: return curr
    prev_lookup = prev.set_index(["Player","Team within selected timeframe"])["overall_adj"].to_dict()
    def _ms(row):
        key = (row["Player"], row["Team within selected timeframe"])
        p   = prev_lookup.get(key)
        return round(0.65*row["overall_adj"]+0.35*p, 2) if p is not None else row["overall_adj"]
    out = curr.copy()
    out["overall_adj"] = out.apply(_ms, axis=1)
    return out

if "_season" not in st.session_state: st.session_state["_season"] = "2025/26"
if "_min_min" not in st.session_state: st.session_state["_min_min"] = 900
data = load_data(st.session_state["_season"], st.session_state["_min_min"])

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav("database")
    st.markdown('<span class="sb-section-label">Season</span>', unsafe_allow_html=True)
    season_mode = st.radio("db_season_mode",["Multi-season","2025/26","2024/25"],
                           horizontal=True, label_visibility="collapsed", key="db_season_mode")
    multi_season_on = (season_mode == "Multi-season")
    _active_season  = "2025/26" if season_mode in ["Multi-season","2025/26"] else "2024/25"
    if _active_season != st.session_state.get("_season"):
        st.session_state["_season"] = _active_season; st.rerun()

    st.markdown('<span class="sb-section-label">Min. minutes played</span>', unsafe_allow_html=True)
    min_minutes = st.select_slider("db_min_min",
                                   options=[0,200,400,600,800,900,1000,1200,1500,2000],
                                   value=st.session_state.get("_min_min",900),
                                   label_visibility="collapsed",
                                   format_func=lambda x: f"{x}+" if x>0 else "All")
    if min_minutes != st.session_state.get("_min_min"):
        st.session_state["_min_min"] = min_minutes; st.rerun()

    st.markdown('<span class="sb-section-label">League template</span>', unsafe_allow_html=True)
    league_template = st.radio("lt",["Top 5 leagues","Next 14 competitions","Both"],label_visibility="collapsed")
    st.markdown('<span class="sb-section-label">Score type</span>', unsafe_allow_html=True)
    score_mode = st.radio("sm",["Adjusted (recommended)","Model (raw)"],label_visibility="collapsed")
    st.markdown('<span class="sb-section-label">Position</span>', unsafe_allow_html=True)
    pos_opts   = ["All positions"] + list(position_groups.keys())
    pos_filter = st.selectbox("pos", pos_opts, label_visibility="collapsed")

    if league_template=="Top 5 leagues": league_pool = sorted(TOP5_LEAGUES)
    elif league_template=="Next 14 competitions": league_pool = sorted(NEXT14_LEAGUES)
    else: league_pool = sorted(TOP5_LEAGUES|NEXT14_LEAGUES)

    st.markdown('<span class="sb-section-label">Competition</span>', unsafe_allow_html=True)
    comp_filter = st.selectbox("comp",["All competitions"]+league_pool,label_visibility="collapsed")

    if league_template=="Top 5 leagues": club_data = data[data["League"].isin(TOP5_LEAGUES)]
    elif league_template=="Next 14 competitions": club_data = data[data["League"].isin(NEXT14_LEAGUES)]
    else: club_data = data[data["League"].isin(TOP5_LEAGUES|NEXT14_LEAGUES)]
    if comp_filter != "All competitions": club_data = club_data[club_data["League"]==comp_filter]
    clubs = sorted(club_data["Team within selected timeframe"].dropna().unique())

    st.markdown('<span class="sb-section-label">Club</span>', unsafe_allow_html=True)
    club_filter = st.selectbox("club",["All clubs"]+clubs,label_visibility="collapsed")

    st.markdown('<span class="sb-section-label">Age</span>', unsafe_allow_html=True)
    age_min,age_max = st.slider("age_range",15,45,(15,40),label_visibility="collapsed")
    st.markdown('<span class="sb-section-label">Min. minutes</span>', unsafe_allow_html=True)
    min_mins = st.slider("mm",0,3000,500,step=50,label_visibility="collapsed")
    st.markdown('<span class="sb-section-label">Foot</span>', unsafe_allow_html=True)
    foot_filter = st.multiselect("foot",["left","right","both"],label_visibility="collapsed")
    st.markdown('<span class="sb-section-label">Contract expires</span>', unsafe_allow_html=True)
    contract_filter = st.selectbox("contract_exp",["Any","< 6 months","< 12 months","< 18 months","< 24 months"],label_visibility="collapsed")

    sort_opts = {"Overall":"overall_adj","Goalscoring":"Goalscoring_score",
                 "Chance cr.":"Chance creation_score","Dribbling":"Dribbling_score",
                 "Passing":"Passing_score","Defending":"Defending_score"}
    st.markdown('<span class="sb-section-label">Sort by</span>', unsafe_allow_html=True)
    sort_lbl = st.selectbox("sort",list(sort_opts.keys()),label_visibility="collapsed")
    sort_col = sort_opts[sort_lbl]
    st.markdown('<span class="sb-section-label">Show rows</span>', unsafe_allow_html=True)
    max_rows = st.select_slider("rows",[25,50,100,200],value=50,label_visibility="collapsed")

# ── Load + score ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:16px;">
  <h1 style="font-size:20px;font-weight:700;letter-spacing:-0.01em;">Player Database</h1>
</div>""", unsafe_allow_html=True)

search = st.text_input("", placeholder="Search player name…", label_visibility="collapsed")

with st.spinner("Computing scores…"):
    _curr = st.session_state["_season"]
    if multi_season_on:
        _both = load_both_seasons_cached(st.session_state["_min_min"])
        _seasons = list(_both.keys())
        _prev    = [s for s in _seasons if s != _curr]
        if _prev:
            scored_df = compute_database_multiseasons(data, _both[_prev[0]], league_template, score_mode,
                                                      season_key_curr=_curr, season_key_prev=_prev[0])
        else:
            scored_df = compute_database(data, league_template, score_mode, season_key=_curr)
    else:
        scored_df = compute_database(data, league_template, score_mode, season_key=_curr)

if scored_df.empty: st.warning("No data available."); st.stop()

df = scored_df.copy()
if search:         df = df[df["Player"].str.contains(search, case=False, na=False)]
if pos_filter != "All positions": df = df[df["_pg"]==pos_filter]
if comp_filter != "All competitions": df = df[df["League"]==comp_filter]
if club_filter != "All clubs":    df = df[df["Team within selected timeframe"]==club_filter]
df = df[df["Age"].between(age_min,age_max)]
if "Minutes played" in df.columns: df = df[df["Minutes played"]>=min_mins]
if foot_filter: df = df[df["Foot"].isin(foot_filter)]
if contract_filter != "Any" and "Contract expires" in df.columns:
    _pd2 = pd
    threshold_map = {"< 6 months":6,"< 12 months":12,"< 18 months":18,"< 24 months":24}
    threshold  = threshold_map[contract_filter]
    today_ts   = _pd2.Timestamp(datetime.date.today())
    df["_contract_exp"] = _pd2.to_datetime(df["Contract expires"], errors="coerce")
    df["_months_left"]  = ((df["_contract_exp"]-today_ts).dt.days/30)
    df = df[df["_months_left"].between(0,threshold)]
if sort_col in df.columns: df = df.sort_values(sort_col, ascending=False)
df = df.reset_index(drop=True)
total = len(df)

if not df.empty: df["_avail"] = compute_availability(df)

# ── Stat/filter bar ────────────────────────────────────────────────────────────
ms_lbl = " · multi-season (65/35)" if multi_season_on else ""
score_lbl = "adjusted" if score_mode=="Adjusted (recommended)" else "raw"
st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;flex-wrap:wrap;
            font-family:'JetBrains Mono',monospace;font-size:9px;color:#b0a898;">
  <span><b style="color:#111827;">{total:,}</b> players</span>
  <span>·</span>
  <span><b style="color:#111827;">{league_template}</b></span>
  <span>·</span>
  <span><b style="color:#111827;">{score_lbl}</b> scores{ms_lbl}</span>
  <span>·</span>
  <span>sorted by <b style="color:#111827;">{sort_lbl}</b></span>
  <div style="margin-left:auto;display:flex;align-items:center;gap:10px;">
    <span><span class="avail-dot" style="background:#1a7a45;"></span> Available ≥60%</span>
    <span><span class="avail-dot" style="background:#f0a500;"></span> Rotation 35–59%</span>
    <span><span class="avail-dot" style="background:#d73027;"></span> Limited &lt;35%</span>
  </div>
</div>""", unsafe_allow_html=True)

# ── Open in Dashboard row ──────────────────────────────────────────────────────
display_df = df.head(max_rows)
if not display_df.empty:
    player_options = [f"{r['Player']} — {r['Team within selected timeframe']}" for _,r in display_df.iterrows()]
    col_sel, col_btn = st.columns([4,1])
    with col_sel:
        selected_str = st.selectbox("Select a player",["— select a player —"]+player_options,label_visibility="collapsed")
    with col_btn:
        if st.button("Open Dashboard →", key="db_open"):
            if selected_str != "— select a player —":
                pname = selected_str.split(" — ")[0]; tname = selected_str.split(" — ")[1]
                match = display_df[(display_df["Player"]==pname)&(display_df["Team within selected timeframe"]==tname)]
                if not match.empty:
                    st.session_state.dashboard_player         = pname
                    st.session_state.dashboard_team           = tname
                    st.session_state.dashboard_position_group = match.iloc[0]["_pg"]
                    st.session_state["pre_select_player"]     = pname
                    st.switch_page("pages/5_Dashboard.py")
            else:
                st.warning("Select a player first.")

# ── Build table ────────────────────────────────────────────────────────────────
CAT_COLS  = ["Goalscoring","Chance creation","Dribbling","Passing","Defending"]
CAT_SHORT = {"Goalscoring":"Goals","Chance creation":"Chance","Dribbling":"Drib.","Passing":"Pass.","Defending":"Def."}
header_cats = "".join(f"<th>{CAT_SHORT[c]}</th>" for c in CAT_COLS)
table_html  = f"""
<div style="background:#fff;border:0.5px solid #e0d8cc;border-radius:8px;overflow:hidden;">
<table class="db-table">
<thead><tr>
  <th class="l" style="width:28px;">#</th>
  <th class="l">Player</th>
  <th style="width:40px;">Pos.</th>
  <th class="l">Club</th>
  <th style="width:48px;">League</th>
  <th style="width:32px;">Age</th>
  <th title="Availability: Green ≥60% · Orange 35–59% · Red &lt;35% of league max minutes">Avail.</th>
  {header_cats}
  <th>Overall</th>
  <th></th>
</tr></thead><tbody>"""

for i, (_, row) in enumerate(display_df.iterrows()):
    player   = row.get("Player","")
    pos_full = str(row.get("Position Label", row.get("Main Position",""))).split(",")[0].strip()
    pos_lbl  = POS_SHORT.get(pos_full, pos_full[:2].upper())
    club     = row.get("Team within selected timeframe","")
    league   = LEAGUE_DISPLAY_NAMES.get(row.get("League",""), row.get("League",""))
    age      = row.get("Age","")
    try:    age_str = str(int(float(age)))
    except: age_str = "—"
    ov  = row.get("overall_adj",0)
    try: ov_val = int(round(float(ov))) if not (isinstance(ov,float) and np.isnan(ov)) else 0
    except: ov_val = 0
    ov_c = _c(ov_val)

    avail_key  = row.get("_avail","green")
    avail_col  = AVAIL_COLOR.get(avail_key,"#1a7a45")
    avail_cell = (f'<td title="{AVAIL_LABEL.get(avail_key,"")}" style="text-align:center;">'
                  f'<span class="avail-dot" style="background:{avail_col};"></span></td>')

    cat_cells = ""
    for cat in CAT_COLS:
        raw = row.get(f"{cat}_score",0)
        try: v = int(round(float(raw))) if not (isinstance(raw,float) and np.isnan(raw)) else 0
        except: v = 0
        cat_cells += f'<td><span class="score-chip" style="background:{_c(v)};">{v}</span></td>'

    table_html += f"""<tr>
      <td class="l" style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#b0a898;">{i+1}</td>
      <td class="l"><span style="font-weight:700;">{player}</span></td>
      <td><span class="pos-chip">{pos_lbl}</span></td>
      <td class="l" style="color:#7a7060;">{club}</td>
      <td style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#b0a898;">{league}</td>
      <td style="font-family:'JetBrains Mono',monospace;font-size:12px;">{age_str}</td>
      {avail_cell}
      {cat_cells}
      <td><span class="score-ov" style="background:{ov_c};">{ov_val}</span></td>
      <td><span class="row-action">Open →</span></td>
    </tr>"""

table_html += "</tbody></table></div>"

if display_df.empty:
    st.info("No players found with current filters.")
else:
    st.markdown(table_html, unsafe_allow_html=True)
    if total > max_rows:
        st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;color:#b0a898;margin-top:8px;">Showing {max_rows} of {total:,} players. Increase \'Show rows\' in the sidebar.</div>',
                    unsafe_allow_html=True)
