import os, sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import percentileofscore

from shared.data_processing import load_season_data
from shared.season_filter import render_season_filter
from shared.styles import BASE_CSS
from shared.sidebar_nav import render_sidebar_nav
from shared.templates import position_groups

st.set_page_config(page_title="Scatter · Target Scouting", layout="wide", initial_sidebar_state="expanded")
st.markdown(BASE_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
.block-container { padding-top:1.6rem !important; max-width:1280px !important; }
.scatter-bar { background:#111827; border-radius:8px 8px 0 0; padding:12px 18px;
    display:flex; align-items:center; justify-content:space-between; gap:16px; }
.scatter-axis-lbl { font-family:'JetBrains Mono',monospace; font-size:8px;
    color:rgba(255,255,255,0.4); text-transform:uppercase; letter-spacing:0.1em; }
.scatter-axis-val { font-family:'JetBrains Mono',monospace; font-size:11px;
    font-weight:600; color:white; }
.scatter-hl { background:rgba(201,168,76,0.15); border:0.5px solid rgba(201,168,76,0.4);
    border-radius:4px; padding:4px 10px; font-family:'JetBrains Mono',monospace;
    font-size:10px; font-weight:600; color:#c9a84c; }
.scatter-meta { font-family:'JetBrains Mono',monospace; font-size:9px;
    color:rgba(255,255,255,0.3); text-transform:uppercase; letter-spacing:0.08em; }
</style>
""", unsafe_allow_html=True)

BG = "#faf7f2"
REPORT_STATS = {
    "Goalscoring":    ["Non-penalty goals per 90","xG per 90","xG per shot","Finishing","Shots per 90","Shots on target, %","Touches in box per 90"],
    "Chance creation":["Assists per 90","xA per 90","Shot assists per 90","Key passes per pass","Through passes per pass","Accurate crosses per received pass","Accurate crosses, %"],
    "Dribbling":      ["Successful dribbles per received pass","Successful dribbles, %","Offensive duels won, %","Progressive runs per received pass"],
    "Passing":        ["Completed progressive passes per 90","Accurate progressive passes, %","Completed passes to final third per 90","Accurate passes to final third, %","Completed passes to penalty area per 90","Accurate passes to penalty area, %","Deep completions per 90"],
    "Defending":      ["PAdj Defensive duels won per 90","Defensive duels won, %","PAdj Aerial duels won per 90","Aerial duels won, %","PAdj Interceptions","PAdj Successful defensive actions per 90","Fouls per 90"],
}
PRESETS = {"Default (16:9)":(1200,675),"Square (1:1)":(800,800),"Portrait (3:4)":(675,900),"Wide (2:1)":(1200,600)}

@st.cache_data
def load_data(season="2025/26", min_minutes=0):
    return load_season_data(season, min_minutes)

def stat_opts(cols):
    out = []
    for cat, stats in REPORT_STATS.items():
        out.append((f"── {cat} ──", None))
        for s in stats:
            if s in cols: out.append((s, s))
    return out

def pct_score(ref, row, x, y):
    return float(np.mean([percentileofscore(ref[s].dropna(), row[s], kind="rank")
                          if not ref[s].dropna().empty else 50.0 for s in [x,y]]))

def pclr(v):
    if v<=50: t=v/50; r=int(220+t*35); g=int(50+t*115); b=int(50-t*50)
    else: t=(v-50)/50; r=int(255-t*205); g=int(165+t*35); b=int(t*50)
    return f"rgb({r},{g},{b})"

if "_season" not in st.session_state: st.session_state["_season"] = "2025/26"
if "_min_min" not in st.session_state: st.session_state["_min_min"] = 900
df = load_data(st.session_state["_season"], st.session_state["_min_min"])

pre_player = st.session_state.get("scatter_highlight_player","None")
pre_x      = st.session_state.get("scatter_x_stat",None)
pre_y      = st.session_state.get("scatter_y_stat",None)
pre_pg     = st.session_state.get("dashboard_position_group",None)

with st.sidebar:
    render_sidebar_nav()
    season, min_minutes = render_season_filter(key_prefix="4")
    if season != st.session_state.get("_season") or min_minutes != st.session_state.get("_min_min"):
        st.session_state["_season"] = season; st.session_state["_min_min"] = min_minutes; st.rerun()

    st.markdown('<div class="home-btn">', unsafe_allow_html=True)
    if st.button("← Home", key="home_sc"): st.switch_page("app.py")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<span class="sb-section-label">Position</span>', unsafe_allow_html=True)
    pg_list = list(position_groups.keys())
    pg_idx  = pg_list.index(pre_pg) if pre_pg in pg_list else 0
    sel_pos = st.selectbox("pos", pg_list, index=pg_idx, label_visibility="collapsed")
    df_pos  = df[df["Main Position"].isin(position_groups[sel_pos])].copy()

    st.markdown('<span class="sb-section-label">Filters</span>', unsafe_allow_html=True)
    comps = sorted(df_pos["League"].dropna().unique())
    sel_comps = st.multiselect("Competition", comps, default=comps)
    df_pos = df_pos[df_pos["League"].isin(sel_comps)]

    clubs = ["All clubs"] + sorted(df_pos["Team within selected timeframe"].dropna().unique())
    sel_club = st.selectbox("Club", clubs)
    if sel_club != "All clubs": df_pos = df_pos[df_pos["Team within selected timeframe"] == sel_club]

    p_opts = ["None"] + sorted(df_pos["Player"].dropna().unique())
    pre_idx = p_opts.index(pre_player) if pre_player in p_opts else 0
    hl = st.selectbox("Highlight player", p_opts, index=pre_idx)

    lo, hi = int(df_pos["Age"].min()), int(df_pos["Age"].max())
    if lo < hi: ar = st.slider("Age", lo, hi, (lo, hi))
    else: ar = (lo, hi)
    df_pos = df_pos[df_pos["Age"].between(*ar)]

    mx = int(df_pos["Minutes played"].max())
    mm = st.slider("Min. minutes", 0, mx, min(500,mx), step=50)
    df_pos = df_pos[df_pos["Minutes played"] >= mm]

    st.markdown(f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;color:#b0a898;">{len(df_pos)} players after filtering</span>',
                unsafe_allow_html=True)

    st.markdown('<span class="sb-section-label">Axes</span>', unsafe_allow_html=True)
    avail = set(df_pos.columns)
    opts  = stat_opts(avail)
    lbls  = [l for l,_ in opts]
    keys  = [k for _,k in opts]
    ridx  = [i for i,k in enumerate(keys) if k is not None]

    def _sb(label, default_pos, pre_stat, skey):
        if pre_stat and pre_stat in keys and keys.index(pre_stat) in ridx:
            default_pos = ridx.index(keys.index(pre_stat))
        i = st.selectbox(label, ridx, index=min(default_pos,len(ridx)-1),
                         format_func=lambda i: lbls[i], key=skey)
        return keys[i]

    x_stat = _sb("X-axis", 0, pre_x, "sc_x")
    y_stat = _sb("Y-axis", min(1,len(ridx)-1), pre_y, "sc_y")

    sc_cands = ["Minutes played","Shots per 90","Passes per 90"]
    sc_opts  = [None] + [c for c in sc_cands if c in avail]
    sz_stat  = st.selectbox("Bubble size", sc_opts, format_func=lambda x: x or "Equal")

    show_pct  = st.checkbox("Colour by percentile", value=False)
    show_topn = st.checkbox("Label top N", value=False)
    top_n     = st.slider("N", 3, 30, 10) if show_topn else 0
    show_all  = st.checkbox("Label all players", value=False)

    st.markdown('<span class="sb-section-label">Export</span>', unsafe_allow_html=True)
    preset = st.selectbox("Dimensions", list(PRESETS.keys()))
    pw, ph = PRESETS[preset]

# ── Chart title bar ────────────────────────────────────────────────────────────
hl_badge = f'<span class="scatter-hl">★ {hl}</span>' if hl != "None" else ""
st.markdown(f"""
<div class="scatter-bar">
  <div style="display:flex;align-items:center;gap:16px;">
    <div><div class="scatter-axis-lbl">X</div><div class="scatter-axis-val">{x_stat}</div></div>
    <div style="color:rgba(255,255,255,0.2);font-size:18px;">·</div>
    <div><div class="scatter-axis-lbl">Y</div><div class="scatter-axis-val">{y_stat}</div></div>
  </div>
  <div style="display:flex;align-items:center;gap:12px;">
    {hl_badge}
    <span class="scatter-meta">{sel_pos} · {len(df_pos)} players</span>
  </div>
</div>""", unsafe_allow_html=True)

# ── Build chart ────────────────────────────────────────────────────────────────
plot_df = df_pos.dropna(subset=[x_stat, y_stat]).copy()
if plot_df.empty:
    st.warning("No data."); st.stop()

plot_df["_pct"] = plot_df.apply(lambda r: pct_score(plot_df, r, x_stat, y_stat), axis=1)

if sz_stat and sz_stat in plot_df.columns:
    raw   = plot_df[sz_stat].fillna(0)
    sizes = ((raw-raw.min())/(raw.max()-raw.min()+1e-9))*28+7
else:
    sizes = pd.Series([10.0]*len(plot_df), index=plot_df.index)

is_hl = hl != "None"
fig   = go.Figure()

for idx, row in plot_df.iterrows():
    pn = row["Player"]; it = pn == hl; pct = row["_pct"]; bs = float(sizes.loc[idx])
    if is_hl:
        if it: color="#c9a84c"; op=1.0; ms=bs*1.9; lw,lc=2.5,"#111827"; sym="star"
        else:  color="#111827"; op=0.18; ms=bs; lw,lc=0,"rgba(0,0,0,0)"; sym="circle"
    else:
        color=pclr(pct) if show_pct else "#111827"; op=0.72; ms=bs; lw,lc=0.5,"rgba(255,255,255,0.3)"; sym="circle"
    hover = [f"<b>{pn}</b>",
             f"{row.get('Team within selected timeframe','')} · {row.get('League','')}",
             f"{x_stat}: {row[x_stat]:.2f}", f"{y_stat}: {row[y_stat]:.2f}"]
    fig.add_trace(go.Scatter(x=[row[x_stat]], y=[row[y_stat]], mode="markers",
        marker=dict(size=ms, color=color, opacity=op, line=dict(width=lw,color=lc), symbol=sym),
        hovertemplate="<br>".join(hover)+"<extra></extra>", showlegend=False))

# Always label the highlighted player
if is_hl:
    hl_row = plot_df[plot_df["Player"] == hl]
    if not hl_row.empty:
        r = hl_row.iloc[0]
        fig.add_annotation(
            x=r[x_stat], y=r[y_stat], text=hl,
            showarrow=True, arrowhead=2, arrowcolor="#c9a84c",
            arrowwidth=1.5, ax=20, ay=-30,
            font=dict(size=11, color="#111827", family="DM Sans"),
            bgcolor="#f0ebe2", bordercolor="#c9a84c", borderpad=4)

if show_topn:
    for _, row in plot_df.nlargest(top_n,"_pct").iterrows():
        if is_hl and row["Player"]==hl: continue
        fig.add_annotation(x=row[x_stat], y=row[y_stat], text=row["Player"], showarrow=False,
                           yshift=13, font=dict(size=9,color="white"), bgcolor="rgba(17,24,39,0.82)", borderpad=3)
if show_all and not show_topn:
    for _, row in plot_df.iterrows():
        if is_hl and row["Player"]==hl: continue
        fig.add_annotation(x=row[x_stat], y=row[y_stat], text=row["Player"], showarrow=False,
                           yshift=13, font=dict(size=8,color="white"), bgcolor="rgba(17,24,39,0.7)", borderpad=2)

fig.add_vline(x=plot_df[x_stat].mean(), line=dict(color="rgba(17,24,39,0.18)", dash="dot", width=1))
fig.add_hline(y=plot_df[y_stat].mean(), line=dict(color="rgba(17,24,39,0.18)", dash="dot", width=1))

fig.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG, width=pw, height=ph,
    margin=dict(l=70,r=40,t=40,b=60),
    xaxis=dict(title=dict(text=x_stat, font=dict(color="#111827",size=13)),
               tickfont=dict(color="#7a7060"), gridcolor="rgba(0,0,0,0.05)", zeroline=False),
    yaxis=dict(title=dict(text=y_stat, font=dict(color="#111827",size=13)),
               tickfont=dict(color="#7a7060"), gridcolor="rgba(0,0,0,0.05)", zeroline=False),
    hoverlabel=dict(bgcolor="#111827", font=dict(size=12,color="white")),
)
st.plotly_chart(fig, use_container_width=False)

# ── Top players strip ──────────────────────────────────────────────────────────
with st.expander("Top 10 players by combined percentile"):
    top10 = plot_df.nlargest(10,"_pct")[["Player","Team within selected timeframe","League",x_stat,y_stat,"_pct"]]
    top10 = top10.rename(columns={"Team within selected timeframe":"Team","_pct":"Pct (X+Y avg)"}).round(2)
    st.dataframe(top10, use_container_width=True, hide_index=True)
