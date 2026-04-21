import os, sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd

from radar_app.radar import create_radar, create_comparison_radar, export_full, export_circle
from shared.templates import template_config, role_config, TOP5_LEAGUES, NEXT14_LEAGUES, LEAGUE_DISPLAY_NAMES
from shared.data_processing import preprocess_data, load_season_data
from shared.season_filter import render_season_filter
from shared.styles import BASE_CSS
from shared.sidebar_nav import render_sidebar_nav

st.set_page_config(page_title="Radar · Target Scouting", layout="wide", initial_sidebar_state="expanded")
st.markdown(BASE_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
.block-container { padding-top:1.6rem !important; max-width:1200px !important; }

/* Toolbar */
.radar-toolbar {
    background: #fff; border: 0.5px solid #e0d8cc; border-radius: 8px;
    padding: 12px 16px; margin-bottom: 20px;
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
    flex-wrap: wrap;
}
.seg-control {
    display: flex; background: #f0ebe2; border-radius: 6px; padding: 3px; gap: 2px;
}
.seg-btn {
    font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.07em;
    padding: 5px 12px; border: none; border-radius: 4px; cursor: pointer;
    color: #7a7060; background: transparent; transition: all 0.12s;
}
.seg-btn.active { background: #111827; color: #c9a84c; }

/* Chart wrapper */
.radar-chart-wrap {
    background: #fff; border: 0.5px solid #e0d8cc; border-radius: 8px; overflow: hidden;
}
.radar-chart-header {
    background: #111827; padding: 14px 20px;
    display: flex; justify-content: space-between; align-items: center;
}
.rch-name { font-size: 16px; font-weight: 700; color: white; letter-spacing: -0.01em; }
.rch-meta {
    font-family: 'JetBrains Mono', monospace; font-size: 9px;
    color: rgba(255,255,255,0.4); text-transform: uppercase;
    letter-spacing: 0.08em; margin-top: 2px;
}
.rch-score {
    font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 700;
    color: white; padding: 6px 14px; border-radius: 6px;
}
.radar-actions {
    padding: 12px 20px; border-top: 0.5px solid #e0d8cc; background: #fff;
    display: flex; gap: 8px; align-items: center;
}
.radar-pool-meta {
    font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #b0a898;
    text-transform: uppercase; letter-spacing: 0.08em; margin-left: auto;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(season: str = "2025/26", min_minutes: int = 0):
    return load_season_data(season, min_minutes)

if "_season" not in st.session_state: st.session_state["_season"] = "2025/26"
if "_min_min" not in st.session_state: st.session_state["_min_min"] = 900
data = load_data(st.session_state["_season"], st.session_state["_min_min"])

pre_player = st.session_state.get("pre_select_player") or st.session_state.get("dashboard_player")
pre_team   = st.session_state.get("dashboard_team")
pre_pg     = st.session_state.get("dashboard_position_group")

with st.sidebar:
    render_sidebar_nav()
    season, min_minutes = render_season_filter(key_prefix="2")
    if season != st.session_state.get("_season") or min_minutes != st.session_state.get("_min_min"):
        st.session_state["_season"] = season
        st.session_state["_min_min"] = min_minutes
        st.rerun()

    st.markdown('<div class="home-btn">', unsafe_allow_html=True)
    if st.button("← Home", key="home_rd"): st.switch_page("app.py")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<span class="sb-section-label">Mode</span>', unsafe_allow_html=True)
    mode = st.radio("mode", ["Single Radar", "Comparison"], horizontal=True, label_visibility="collapsed")

    st.markdown('<span class="sb-section-label">Radar type</span>', unsafe_allow_html=True)
    radar_type = st.radio("rt", ["Universal Radar", "Position Template", "Role Radar"], label_visibility="collapsed")

    st.markdown('<span class="sb-section-label">League template</span>', unsafe_allow_html=True)
    league_template = st.radio("lt", ["Top 5 leagues", "Next 14 competitions", "Both"], label_visibility="collapsed")

    st.markdown('<span class="sb-section-label">Percentile basis</span>', unsafe_allow_html=True)
    if league_template == "Top 5 leagues":         pct_opts = ["T5 only", "Own league"]
    elif league_template == "Next 14 competitions": pct_opts = ["Next 14 only", "Own league"]
    else:                                            pct_opts = ["T5 + Next 14", "T5 only", "Next 14 only", "Own league"]
    percentile_basis = st.radio("pb", pct_opts, label_visibility="collapsed")

    st.markdown('<span class="sb-section-label">Position group</span>', unsafe_allow_html=True)
    pg_list = list(template_config.keys())
    pg_idx  = pg_list.index(pre_pg) if pre_pg in pg_list else 0
    position_group = st.selectbox("pg", pg_list, index=pg_idx, label_visibility="collapsed")
    positions = template_config[position_group]["positions"]

    role_name = None
    if radar_type == "Role Radar":
        avail = list(role_config.get(position_group, {}).keys())
        if avail: role_name = st.selectbox("Role", avail)
        else: st.warning(f"No roles for {position_group}.")

    filtered = data[data["Main Position"].isin(positions)].copy()
    if league_template == "Top 5 leagues":         filtered = filtered[filtered["League"].isin(TOP5_LEAGUES)]
    elif league_template == "Next 14 competitions": filtered = filtered[filtered["League"].isin(NEXT14_LEAGUES)]

    all_lg   = sorted(filtered["League"].dropna().unique())
    disp_opts = {LEAGUE_DISPLAY_NAMES.get(l,l): l for l in all_lg}
    sel_lg   = st.selectbox("League filter", ["All"] + list(disp_opts.keys()))
    if sel_lg != "All": filtered = filtered[filtered["League"] == disp_opts[sel_lg]]

    clubs    = sorted(filtered["Team within selected timeframe"].dropna().unique())
    sel_club = st.selectbox("Club filter", ["All"] + clubs)
    if sel_club != "All": filtered = filtered[filtered["Team within selected timeframe"] == sel_club]

    if not filtered.empty:
        mn, mx = int(filtered["Age"].min()), int(filtered["Age"].max())
        if mn != mx: ar = st.slider("Age", mn, mx, (mn, mx))
        else: ar = (mn, mx)
        filtered = filtered[filtered["Age"].between(*ar)]
        mmx = int(filtered["Minutes played"].max())
        mm  = st.slider("Min. minutes", 0, mmx, min(500, mmx), step=50)
        filtered = filtered[filtered["Minutes played"] >= mm]

    if filtered.empty: st.warning("No players found."); st.stop()

    st.markdown('<span class="sb-section-label">Player 1</span>', unsafe_allow_html=True)
    players = sorted(filtered["Player"].unique())
    p1_idx  = players.index(pre_player) if pre_player in players else 0
    player1 = st.selectbox("p1", players, index=p1_idx, label_visibility="collapsed")
    if st.session_state.get("pre_select_player"): st.session_state["pre_select_player"] = ""
    team1   = filtered[filtered["Player"] == player1]["Team within selected timeframe"].iloc[0]
    st.caption(f"🏟️ {team1}")

    benchmark_player = None
    if mode == "Single Radar":
        st.markdown('<span class="sb-section-label">Benchmark (optional)</span>', unsafe_allow_html=True)
        bench = st.selectbox("bench", ["None"] + players, label_visibility="collapsed")
        if bench != "None":
            bt = filtered[filtered["Player"] == bench]["Team within selected timeframe"].iloc[0]
            benchmark_player = (bench, bt)

    if mode == "Comparison":
        st.markdown('<span class="sb-section-label">Player 2</span>', unsafe_allow_html=True)
        player2 = st.selectbox("p2", players, index=min(1,len(players)-1), label_visibility="collapsed")
        team2   = filtered[filtered["Player"] == player2]["Team within selected timeframe"].iloc[0]
        st.caption(f"🏟️ {team2}")
        cmode = st.radio("cm", ["Side by side","Overlay"], label_visibility="collapsed")

    st.markdown('<span class="sb-section-label">Options</span>', unsafe_allow_html=True)
    show_avg    = st.checkbox("Show pool average", value=True)
    export_type = st.radio("et", ["Full figure","Circle only"], label_visibility="collapsed")
    generate    = st.button("Generate Radar", use_container_width=True)

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:20px;">
  <h1 style="font-size:20px;font-weight:700;letter-spacing:-0.01em;">Radar Tool</h1>
  <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#b0a898;
               text-transform:uppercase;letter-spacing:0.08em;">{mode} · {radar_type}</span>
</div>
""", unsafe_allow_html=True)

# ── Mode/type toolbar ──────────────────────────────────────────────────────────
m_active  = ["active" if m == mode else "" for m in ["Single Radar","Comparison"]]
rt_active = ["active" if r == radar_type else "" for r in ["Universal Radar","Position Template","Role Radar"]]
pb_active = ["active" if p == percentile_basis else "" for p in pct_opts]

st.markdown(f"""
<div class="radar-toolbar">
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
    <div class="seg-control">
      <button class="seg-btn {m_active[0]}">Single Radar</button>
      <button class="seg-btn {m_active[1]}">Comparison</button>
    </div>
    <div class="seg-control">
      {''.join(f'<button class="seg-btn {rt_active[i]}">{r}</button>' for i,r in enumerate(["Universal Radar","Position Template","Role Radar"]))}
    </div>
  </div>
  <div class="seg-control">
    {''.join(f'<button class="seg-btn {pb_active[i]}">{p}</button>' for i,p in enumerate(pct_opts))}
  </div>
</div>""", unsafe_allow_html=True)

# ── Generate ───────────────────────────────────────────────────────────────────
if generate:
    try:
        if mode == "Single Radar":
            with st.spinner("Generating…"):
                fig, ax = create_radar(data, player_name=player1, player_team=team1,
                                       template_key=position_group, percentile_basis=percentile_basis,
                                       radar_type=radar_type, role_name=role_name,
                                       show_avg=show_avg, benchmark_player=benchmark_player)

            pool_size = len(filtered)
            st.markdown(f"""
            <div class="radar-chart-wrap">
              <div class="radar-chart-header">
                <div>
                  <div class="rch-name">{player1}</div>
                  <div class="rch-meta">{position_group} · {team1} · {radar_type} · {percentile_basis}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            st.pyplot(fig, use_container_width=True)

            st.markdown(f"""
            <div class="radar-actions">
              <div class="radar-pool-meta">{position_group} · {pool_size} players · {league_template}</div>
            </div>""", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button("Download — Full PNG", export_full(fig),
                                   f"{player1.replace(' ','_')}_radar.png", mime="image/png")
            with c2:
                if export_type == "Circle only":
                    st.download_button("Download — Circle", export_circle(fig, ax),
                                       f"{player1.replace(' ','_')}_radar_circle.png", mime="image/png")
        else:
            cm = "side_by_side" if cmode == "Side by side" else "overlay"
            with st.spinner("Generating comparison…"):
                fig, ax = create_comparison_radar(data,
                                                  player1_name=player1, player1_team=team1,
                                                  player2_name=player2, player2_team=team2,
                                                  template_key=position_group,
                                                  percentile_basis=percentile_basis,
                                                  radar_type=radar_type, role_name=role_name,
                                                  mode=cm, show_avg=show_avg)

            st.markdown(f"""
            <div class="radar-chart-wrap">
              <div class="radar-chart-header">
                <div>
                  <div class="rch-name">{player1} <span style="color:#c9a84c;margin:0 8px;">vs</span> {player2}</div>
                  <div class="rch-meta">{position_group} · {radar_type} · {percentile_basis} · {cmode}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            st.pyplot(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button("Download — Full PNG", export_full(fig),
                                   f"{player1.replace(' ','_')}_vs_{player2.replace(' ','_')}_radar.png",
                                   mime="image/png")
            with c2:
                if export_type == "Circle only" and ax:
                    st.download_button("Download — Circle", export_circle(fig, ax),
                                       f"{player1.replace(' ','_')}_vs_{player2.replace(' ','_')}_circle.png",
                                       mime="image/png")
    except Exception as e:
        st.error(f"❌ {e}")
else:
    player_label = player1 if mode == "Single Radar" else f"{player1} vs {player2}"
    st.markdown(f"""
    <div class="radar-chart-wrap">
      <div class="radar-chart-header">
        <div>
          <div class="rch-name" style="color:rgba(255,255,255,0.3);">No radar generated yet</div>
          <div class="rch-meta">{position_group} · {radar_type} · {percentile_basis}</div>
        </div>
      </div>
      <div style="padding:60px 20px;text-align:center;background:#faf7f2;">
        <div style="font-size:15px;font-weight:700;color:#111827;margin-bottom:8px;">
          Click <em>Generate Radar</em> in the sidebar
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#b0a898;">
          {player_label} · {position_group} · {len(filtered)} players in pool
        </div>
      </div>
    </div>""", unsafe_allow_html=True)
