import os, sys, io, base64
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.data_processing import load_season_data
from shared.season_filter import render_season_filter
from shared.styles import BASE_CSS, header_logo_html
from shared.sidebar_nav import render_sidebar_nav
from shared.templates import (
    position_groups, report_template, position_category_weights,
    LEAGUE_DISPLAY_NAMES, TOP5_LEAGUES, NEXT14_LEAGUES, role_config,
)
from radar_app.radar import create_radar_compact, _build_pool, _compute_percentiles

st.set_page_config(page_title="Report Builder · Target Scouting", layout="wide", initial_sidebar_state="expanded")
st.markdown(BASE_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
.block-container { padding-top:1.6rem !important; max-width:1280px !important; }
.block-lbl { font-family:'JetBrains Mono',monospace; font-size:9px; font-weight:700;
    text-transform:uppercase; letter-spacing:0.1em; color:#b0a898;
    margin-bottom:8px; display:flex; align-items:center; gap:8px; }
.block-lbl::after { content:''; flex:1; height:0.5px; background:#e0d8cc; }
.block-config-card { background:#fff; border:0.5px solid #e0d8cc; border-radius:8px; overflow:hidden; }
.bcc-header { background:#f0ebe2; padding:8px 14px; border-bottom:0.5px solid #e0d8cc;
    display:flex; align-items:center; justify-content:space-between; }
.bcc-label { font-family:'JetBrains Mono',monospace; font-size:9px; font-weight:700;
    text-transform:uppercase; letter-spacing:0.1em; color:#b0a898; }
.bcc-type { font-family:'JetBrains Mono',monospace; font-size:9px; font-weight:700;
    color:#c9a84c; text-transform:uppercase; letter-spacing:0.06em; }
</style>
""", unsafe_allow_html=True)

BG = "#faf7f2"

for k, v in [("rpt_generated",False),("rpt_player",None),("rpt_team",None),
              ("rpt_pg",None),("rpt_pct","T5 only"),
              ("rpt_ba_type","Radar chart"),("rpt_ba_cfg",{}),
              ("rpt_bb_type","Percentile bars"),("rpt_bb_cfg",{}),
              ("rpt_bc_type","Scatter plot"),("rpt_bc_cfg",{}),
              ("rpt_notes",{})]:
    if k not in st.session_state: st.session_state[k] = v

def _c(v):
    if v>=75: return "#1a9850"
    elif v>=50: return "#91cf60"
    elif v>=25: return "#f0a500"
    return "#d73027"

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
    "Ball progression through passing":"Ball progression","Passing accuracy (prog/1/3/forw)":"Pass accuracy",
}

ALL_REPORT_STATS = [s for g in report_template.values() for s in g["stats"]]

@st.cache_data
def load_data(season="2025/26", min_minutes=0):
    return load_season_data(season, min_minutes)

if "_season" not in st.session_state: st.session_state["_season"] = "2025/26"
if "_min_min" not in st.session_state: st.session_state["_min_min"] = 900
data = load_data(st.session_state["_season"], st.session_state["_min_min"])
avail_cols    = set(data.columns)
all_players   = sorted(data["Player"].unique())
all_selectable = [s for s in ALL_REPORT_STATS if s in avail_cols]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav("report")
    season, min_minutes = render_season_filter(key_prefix="7")
    if season != st.session_state.get("_season") or min_minutes != st.session_state.get("_min_min"):
        st.session_state["_season"] = season; st.session_state["_min_min"] = min_minutes; st.rerun()

    st.markdown('<span class="sb-section-label">Player</span>', unsafe_allow_html=True)
    pre = st.session_state.get("pre_select_player") or st.session_state.get("dashboard_player")
    pi  = all_players.index(pre) if pre in all_players else 0
    sel_player = st.selectbox("player", all_players, index=pi, label_visibility="collapsed")
    p_rows = data[data["Player"]==sel_player]
    if p_rows.empty: st.warning("Player not found."); st.stop()
    teams    = p_rows["Team within selected timeframe"].unique()
    sel_team = st.selectbox("Team", teams) if len(teams)>1 else teams[0]
    if len(teams)==1: st.caption(f"🏟️ {sel_team}")
    p_row = p_rows[p_rows["Team within selected timeframe"]==sel_team].iloc[0]

    st.markdown('<span class="sb-section-label">Position group</span>', unsafe_allow_html=True)
    main_pos = str(p_row.get("Main Position",""))
    auto_pg  = next((pg for pg,pos in position_groups.items() if main_pos in pos), list(position_groups.keys())[0])
    pre_pg   = st.session_state.get("dashboard_position_group") or auto_pg
    pg_list  = list(position_groups.keys())
    position_group = st.selectbox("pg", pg_list, index=pg_list.index(pre_pg) if pre_pg in pg_list else 0, label_visibility="collapsed")

    st.markdown('<span class="sb-section-label">Percentile basis</span>', unsafe_allow_html=True)
    pct_basis = st.radio("pb",["T5 only","Next 14 only","Own league"],label_visibility="collapsed")

    st.markdown("---")
    BLOCK_TYPES = ["Radar chart","Percentile bars","Scatter plot","Empty"]

    def block_controls(prefix, default_type, default_stats=None):
        btype = st.selectbox(f"{prefix}_type", BLOCK_TYPES,
                             index=BLOCK_TYPES.index(default_type),
                             label_visibility="collapsed", key=f"{prefix}_t")
        cfg = {}
        if btype == "Radar chart":
            cfg["radar_type"] = st.radio(f"{prefix}_rt",
                ["Universal Radar","Position Template","Role Radar"],
                label_visibility="collapsed", key=f"{prefix}_rt")
            if cfg["radar_type"] == "Role Radar":
                roles = list(role_config.get(position_group,{}).keys())
                cfg["role"] = st.selectbox("Role", roles, key=f"{prefix}_role") if roles else None
        elif btype == "Percentile bars":
            cfg["stats"] = st.multiselect("Stats", all_selectable,
                default=[s for s in (default_stats or ["xG per 90","xA per 90",
                    "Successful dribbles per received pass","Progressive runs per received pass",
                    "Defensive duels won, %"]) if s in all_selectable],
                max_selections=12, label_visibility="collapsed", key=f"{prefix}_stats")
        elif btype == "Scatter plot":
            avail = [s for s in all_selectable if s in avail_cols]
            cc1,cc2 = st.columns(2)
            with cc1: cfg["x"] = st.selectbox("X", avail, key=f"{prefix}_x", label_visibility="collapsed")
            with cc2: cfg["y"] = st.selectbox("Y", avail, index=min(1,len(avail)-1), key=f"{prefix}_y", label_visibility="collapsed")
        return btype, cfg

    st.markdown('<span class="sb-section-label">Block A — left column</span>', unsafe_allow_html=True)
    ba_type, ba_cfg = block_controls("ba","Radar chart")
    st.markdown('<span class="sb-section-label">Block B — right column</span>', unsafe_allow_html=True)
    bb_type, bb_cfg = block_controls("bb","Percentile bars",[
        "PAdj Defensive duels won per 90","Defensive duels won, %",
        "PAdj Aerial duels won per 90","Aerial duels won, %","PAdj Interceptions"])
    st.markdown('<span class="sb-section-label">Block C — full width</span>', unsafe_allow_html=True)
    bc_type, bc_cfg = block_controls("bc","Scatter plot")
    st.markdown("---")

    if st.button("Generate Report", use_container_width=True):
        st.session_state.update({"rpt_generated":True,"rpt_player":sel_player,
            "rpt_team":sel_team,"rpt_pg":position_group,"rpt_pct":pct_basis,
            "rpt_ba_type":ba_type,"rpt_ba_cfg":dict(ba_cfg),
            "rpt_bb_type":bb_type,"rpt_bb_cfg":dict(bb_cfg),
            "rpt_bc_type":bc_type,"rpt_bc_cfg":dict(bc_cfg)})
        st.rerun()

    if st.session_state["rpt_generated"]:
        if st.button("← Edit report", use_container_width=True, key="reset_rpt"):
            st.session_state["rpt_generated"] = False; st.rerun()

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:20px;">
  <h1 style="font-size:20px;font-weight:700;letter-spacing:-0.01em;">Scout Report Builder</h1>
</div>""", unsafe_allow_html=True)

if not st.session_state["rpt_generated"]:
    # Config state — show block layout preview
    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#7a7060;margin-bottom:20px;">Configure your report in the sidebar, then click <b style=\'color:#111827;\'>Generate Report</b>.</div>',
                unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col, lbl, btype in [(c1,"Block A",ba_type),(c2,"Block B",bb_type),(c3,"Block C",bc_type)]:
        with col:
            icon = {"Radar chart":"◎","Percentile bars":"▤","Scatter plot":"⋯","Empty":"○"}.get(btype,"○")
            st.markdown(
                f'<div class="block-config-card">'
                f'<div class="bcc-header"><span class="bcc-label">{lbl}</span><span class="bcc-type">{btype}</span></div>'
                f'<div style="padding:24px;text-align:center;">'
                f'<div style="font-size:28px;color:#c9a84c;margin-bottom:6px;">{icon}</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;color:#b0a898;text-transform:uppercase;">{btype}</div>'
                f'</div></div>',
                unsafe_allow_html=True)
    st.stop()

# ── Report view ────────────────────────────────────────────────────────────────
rpt_player  = st.session_state["rpt_player"]
rpt_team    = st.session_state["rpt_team"]
rpt_pg      = st.session_state["rpt_pg"]
rpt_pct     = st.session_state["rpt_pct"]
rpt_ba_type = st.session_state["rpt_ba_type"]
rpt_ba_cfg  = st.session_state["rpt_ba_cfg"]
rpt_bb_type = st.session_state["rpt_bb_type"]
rpt_bb_cfg  = st.session_state["rpt_bb_cfg"]
rpt_bc_type = st.session_state["rpt_bc_type"]
rpt_bc_cfg  = st.session_state["rpt_bc_cfg"]

p_rows2 = data[data["Player"]==rpt_player]
if p_rows2.empty: st.error(f"Player {rpt_player!r} not found."); st.stop()
p_row2  = p_rows2[p_rows2["Team within selected timeframe"]==rpt_team]
p_row2  = p_row2.iloc[0] if not p_row2.empty else p_rows2.iloc[0]

pos    = str(p_row2.get("Position","")).split(",")[0].strip()
age    = p_row2.get("Age","—"); mins = p_row2.get("Minutes played","—")
foot   = p_row2.get("Foot","—"); height = p_row2.get("Height","—")
nation = p_row2.get("Birth country","—"); league = p_row2.get("League","—")
league_disp = LEAGUE_DISPLAY_NAMES.get(league, league)
value  = p_row2.get("Market value",None)
val_str = f"€{int(value):,}" if isinstance(value,(int,float)) and not np.isnan(float(value)) else "—"
player_league = p_row2.get("League",None)
pool = _build_pool(data, rpt_pg, rpt_pct, player_league)

meta_items = [("Position",pos),("Club",rpt_team),("League",league_disp),
              ("Age",str(age)),("Minutes",str(mins)),("Foot",foot),
              ("Height",f"{height} cm"),("Nat.",nation),("Value",val_str)]

_logo_html = header_logo_html(44)
meta_html  = "".join(
    f'<div style="margin-right:16px;"><div style="font-size:9px;color:rgba(255,255,255,0.35);'
    f'text-transform:uppercase;letter-spacing:0.05em;">{l}</div>'
    f'<div style="font-size:12px;font-weight:600;color:white;">{v}</div></div>'
    for l,v in meta_items
)

st.markdown(f"""
<div style="background:#111827;border-radius:8px 8px 0 0;padding:18px 22px;
            display:flex;justify-content:space-between;align-items:center;">
  <div>
    <div style="font-size:20px;font-weight:700;color:#c9a84c;letter-spacing:-0.01em;">{rpt_player}</div>
    <div style="display:flex;flex-wrap:wrap;margin-top:8px;">{meta_html}</div>
  </div>
  {_logo_html}
</div>
<div style="background:#f0ebe2;padding:5px 22px;margin-bottom:16px;border-radius:0 0 6px 6px;
            font-family:'JetBrains Mono',monospace;font-size:9px;color:#7a7060;">
  {rpt_pg} · {rpt_pct} · Wyscout 26 Mar 2026
</div>
""", unsafe_allow_html=True)

def fig_to_b64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def render_radar_st(radar_type, role_name=None):
    try:
        with st.spinner("Rendering radar…"):
            fig, _ = create_radar_compact(data, rpt_player, rpt_team, rpt_pg, rpt_pct, radar_type, role_name, show_avg=True)
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    except Exception as e: st.error(f"Radar error: {e}")

def render_bars_st(stats):
    if not stats: st.info("Select at least one stat."); return
    avail = [s for s in stats if s in pool.columns]
    if not avail: st.warning("Stats not available."); return
    try: pct = _compute_percentiles(pool, rpt_player, rpt_team, avail)
    except Exception as e: st.error(f"Percentile error: {e}"); return
    for stat in avail:
        if stat not in pct: continue
        v = float(pct[stat]); fc = _c(v); short = STAT_SHORT.get(stat, stat)
        raw = p_row2.get(stat)
        try:    raw_str = f"{float(raw):.2f}" if raw is not None else ""
        except: raw_str = ""
        st.markdown(
            f'<div style="margin-bottom:9px;">'
            f'<div style="font-size:12px;color:#111827;display:flex;justify-content:space-between;margin-bottom:3px;">'
            f'<span>{short}</span><span style="color:#7a7060;">{raw_str}</span></div>'
            f'<div style="background:#f0ebe2;height:8px;border-radius:4px;">'
            f'<div style="width:{v:.0f}%;height:8px;border-radius:4px;background:{fc};"></div></div></div>',
            unsafe_allow_html=True)

def make_scatter_fig(x_stat, y_stat, w=5, h=5):
    if x_stat not in avail_cols or y_stat not in avail_cols: return None
    pool_sc = data[data["Main Position"].isin(position_groups[rpt_pg])].dropna(subset=[x_stat,y_stat]).copy()
    if pool_sc.empty: return None
    fig, ax = plt.subplots(figsize=(w,h)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    others = pool_sc[pool_sc["Player"]!=rpt_player]
    ax.scatter(others[x_stat], others[y_stat], c="#111827", alpha=0.18, s=22, zorder=2)
    me = pool_sc[pool_sc["Player"]==rpt_player]
    if not me.empty:
        ax.scatter(me[x_stat], me[y_stat], c="#c9a84c", s=90, zorder=5, marker="*")
        ax.annotate(rpt_player, (me[x_stat].iloc[0], me[y_stat].iloc[0]),
                    xytext=(8,8), textcoords="offset points", fontsize=8, color="#111827",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0ebe2", edgecolor="#c9a84c", linewidth=0.8))
    ax.axvline(pool_sc[x_stat].mean(), color="#111827", alpha=0.15, linewidth=0.8, linestyle="--")
    ax.axhline(pool_sc[y_stat].mean(), color="#111827", alpha=0.15, linewidth=0.8, linestyle="--")
    ax.set_xlabel(STAT_SHORT.get(x_stat,x_stat), fontsize=9, color="#7a7060")
    ax.set_ylabel(STAT_SHORT.get(y_stat,y_stat), fontsize=9, color="#7a7060")
    ax.tick_params(colors="#b0a898", labelsize=7)
    for spine in ax.spines.values(): spine.set_color("#e0d8cc"); spine.set_linewidth(0.5)
    ax.grid(True, color="#e0d8cc", linewidth=0.4, alpha=0.6); fig.tight_layout()
    return fig

def render_scatter_st(x_stat, y_stat):
    fig = make_scatter_fig(x_stat, y_stat)
    if fig: st.pyplot(fig, use_container_width=True); plt.close(fig)
    else: st.warning("Not enough data.")

def render_block(btype, bcfg, label):
    st.markdown(f'<div class="block-lbl">{label} — {btype}</div>', unsafe_allow_html=True)
    if btype == "Radar chart":
        render_radar_st(bcfg.get("radar_type","Universal Radar"), bcfg.get("role"))
    elif btype == "Percentile bars":
        render_bars_st(bcfg.get("stats",[]))
    elif btype == "Scatter plot":
        render_scatter_st(bcfg.get("x",""), bcfg.get("y",""))

# Row 1: notes + Block A
row1_l, row1_r = st.columns(2)
with row1_l:
    st.markdown('<div class="block-lbl">Scout notes</div>', unsafe_allow_html=True)
    top_note_key = f"{rpt_player}__{rpt_team}__top"
    existing_top = st.session_state["rpt_notes"].get(top_note_key,"")
    top_notes = st.text_area("", value=existing_top, height=340,
                             placeholder="Strengths:\n—\n\nWeaknesses:\n—\n\nConclusion:\n—",
                             label_visibility="collapsed", key="top_notes_area")
    if st.button("Save notes", key="save_top"):
        st.session_state["rpt_notes"][top_note_key] = top_notes
        if "scout_notes" not in st.session_state: st.session_state["scout_notes"] = {}
        st.session_state["scout_notes"][f"{rpt_player}__{rpt_team}"] = top_notes
        st.success("Notes saved.")
with row1_r:
    render_block(rpt_ba_type, rpt_ba_cfg, "Block A")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# Row 2: Block B + Block C
row2_l, row2_r = st.columns(2)
with row2_l: render_block(rpt_bb_type, rpt_bb_cfg, "Block B")
with row2_r: render_block(rpt_bc_type, rpt_bc_cfg, "Block C")

st.markdown(f"""
<div style="background:#f0ebe2;border-radius:0 0 6px 6px;padding:8px 22px;
            display:flex;justify-content:space-between;margin-top:12px;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#b0a898;">
    Target Scouting · Scout Report · {rpt_player} · {rpt_team}
  </span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#b0a898;">
    Wyscout 26 Mar 2026 · {rpt_pct} · {rpt_pg}
  </span>
</div>""", unsafe_allow_html=True)

# ── Export ─────────────────────────────────────────────────────────────────────
st.markdown('<hr style="border:none;border-top:0.5px solid #e0d8cc;margin:20px 0 14px;">', unsafe_allow_html=True)
st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;font-weight:700;color:#b0a898;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;">Export</div>',
            unsafe_allow_html=True)

def block_to_html(btype, bcfg):
    if btype == "Empty": return ""
    if btype == "Radar chart":
        try:
            fig, _ = create_radar_compact(data, rpt_player, rpt_team, rpt_pg, rpt_pct,
                                          bcfg.get("radar_type","Universal Radar"), bcfg.get("role"), show_avg=True)
            b64 = fig_to_b64(fig, dpi=200); plt.close(fig)
            return f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:6px;"/>'
        except Exception as e: return f'<p style="color:#d73027;font-size:11px;">Radar error: {e}</p>'
    elif btype == "Percentile bars":
        stats = bcfg.get("stats",[]); avail = [s for s in stats if s in pool.columns]
        if not avail: return ""
        try: pct = _compute_percentiles(pool, rpt_player, rpt_team, avail)
        except: return ""
        out = ""
        for stat in avail:
            if stat not in pct: continue
            v = float(pct[stat]); fc = _c(v); short = STAT_SHORT.get(stat,stat)
            raw = p_row2.get(stat)
            try: raw_str = f"{float(raw):.2f}" if raw is not None else ""
            except: raw_str = ""
            out += (f'<div style="margin-bottom:8px;"><div style="font-size:11px;color:#111;'
                    f'display:flex;justify-content:space-between;margin-bottom:3px;">'
                    f'<span>{short}</span><span style="color:#777;">{raw_str}</span></div>'
                    f'<div style="background:#f0ebe2;height:7px;border-radius:3px;">'
                    f'<div style="width:{v:.0f}%;height:7px;border-radius:3px;background:{fc};"></div></div></div>')
        return out
    elif btype == "Scatter plot":
        x = bcfg.get("x",""); y = bcfg.get("y","")
        if not x or not y: return ""
        try:
            fig = make_scatter_fig(x, y)
            if not fig: return ""
            b64 = fig_to_b64(fig, dpi=200); plt.close(fig)
            return f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:6px;"/>'
        except Exception as e: return f'<p style="color:#d73027;font-size:11px;">Scatter error: {e}</p>'
    return ""

def build_full_html():
    notes_top_html = (top_notes.replace("\n","<br>") if top_notes.strip()
                      else "<em style='color:#b0a898;'>No notes added.</em>")
    html_a = block_to_html(rpt_ba_type, rpt_ba_cfg)
    html_b = block_to_html(rpt_bb_type, rpt_bb_cfg)
    html_c = block_to_html(rpt_bc_type, rpt_bc_cfg)
    meta_e = "".join(
        f'<div style="margin-right:14px;"><div style="font-size:8px;color:rgba(255,255,255,0.35);'
        f'text-transform:uppercase;">{l}</div>'
        f'<div style="font-size:11px;font-weight:600;color:white;">{v}</div></div>'
        for l,v in meta_items
    )
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:Arial,sans-serif;background:#faf7f2;color:#111;}}
.page{{max-width:900px;margin:0 auto;padding:0;}}
.header{{background:#111827;padding:16px 22px;display:flex;justify-content:space-between;align-items:center;}}
.pname{{font-size:20px;font-weight:700;color:#c9a84c;margin-bottom:8px;}}
.meta-row{{display:flex;flex-wrap:wrap;}}
.sub{{background:#f0ebe2;padding:5px 22px;font-size:9px;color:#7a7060;margin-bottom:14px;}}
.body{{padding:0 20px 20px;}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:14px;}}
.blbl{{font-size:8px;text-transform:uppercase;letter-spacing:0.06em;color:#b0a898;
       border-bottom:0.5px solid #e0d8cc;padding-bottom:4px;margin-bottom:10px;}}
.notes{{background:#faf7f2;border:0.5px solid #e0d8cc;border-radius:6px;
        padding:12px;font-size:12px;color:#111;line-height:1.7;min-height:180px;}}
.footer{{background:#f0ebe2;padding:8px 22px;display:flex;justify-content:space-between;
         font-size:9px;color:#b0a898;margin-top:14px;}}
@media print{{body{{-webkit-print-color-adjust:exact;print-color-adjust:exact;}}
@page{{margin:1cm;}}}}
</style></head><body><div class="page">
<div class="header">
  <div><div class="pname">{rpt_player}</div><div class="meta-row">{meta_e}</div></div>
</div>
<div class="sub">{rpt_pg} · {rpt_pct} · Wyscout 26 Mar 2026</div>
<div class="body">
  <div class="two-col">
    <div><div class="blbl">Scout notes</div><div class="notes">{notes_top_html}</div></div>
    <div><div class="blbl">Block A — {rpt_ba_type}</div>{html_a}</div>
  </div>
  <div class="two-col">
    <div><div class="blbl">Block B — {rpt_bb_type}</div>{html_b}</div>
    <div><div class="blbl">Block C — {rpt_bc_type}</div>{html_c}</div>
  </div>
</div>
<div class="footer">
  <span>Target Scouting · Scout Report · {rpt_player} · {rpt_team}</span>
  <span>Wyscout 26 Mar 2026 · {rpt_pct} · {rpt_pg}</span>
</div>
</div></body></html>"""

ec1, ec2, ec3 = st.columns([1,1,2])
with ec1:
    if st.button("Build export", key="build_exp"):
        with st.spinner("Building export…"):
            html_out = build_full_html()
        st.session_state["_rpt_html_out"] = html_out

if "_rpt_html_out" in st.session_state:
    _html = st.session_state["_rpt_html_out"]
    with ec1:
        st.download_button("Download HTML", data=_html.encode("utf-8"),
                           file_name=f"{rpt_player.replace(' ','_')}_scout_report.html",
                           mime="text/html", key="dl_html")
    with ec2:
        try:
            from weasyprint import HTML, CSS
            pdf = HTML(string=_html).write_pdf(stylesheets=[CSS(string="@page{size:A4;margin:12mm;}")])
            st.download_button("Download PDF", data=pdf,
                               file_name=f"{rpt_player.replace(' ','_')}_scout_report.pdf",
                               mime="application/pdf", key="dl_pdf")
        except ImportError:
            st.caption("Install weasyprint for PDF export")
    with ec3:
        st.markdown('<div style="background:#f0ebe2;border-radius:6px;padding:10px 14px;font-size:11px;color:#7a7060;">✓ Export ready. Download as HTML or PDF.</div>',
                    unsafe_allow_html=True)
