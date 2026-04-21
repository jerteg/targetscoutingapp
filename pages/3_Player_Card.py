import os, sys, math
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd

from shared.data_processing import load_season_data
from shared.season_filter import render_season_filter
from shared.styles import BASE_CSS
from shared.sidebar_nav import render_sidebar_nav
from shared.templates import (
    report_template, position_groups, position_category_weights,
    LEAGUE_MULTIPLIERS_ALL, LEAGUE_MULTIPLIERS_NEXT14,
    TOP5_LEAGUES, NEXT14_LEAGUES,
)

st.set_page_config(page_title="Player Card · Target Scouting", layout="wide", initial_sidebar_state="expanded")
st.markdown(BASE_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
.block-container { padding-top:1.6rem !important; max-width:1100px !important; }
.section-lbl { font-family:'JetBrains Mono',monospace; font-size:9px; font-weight:700;
    color:#b0a898; text-transform:uppercase; letter-spacing:0.12em;
    margin-bottom:12px; display:flex; align-items:center; gap:10px; }
.section-lbl::after { content:''; flex:1; height:0.5px; background:#e0d8cc; }
.cat-header { display:flex; justify-content:space-between; align-items:center;
    background:#f0ebe2; border-left:3px solid var(--cc); border-radius:0 6px 6px 0;
    padding:9px 14px; margin-bottom:10px; }
.cat-name { font-size:13px; font-weight:700; color:#111827; }
.cat-score-badge { color:white; padding:3px 10px; border-radius:5px;
    font-family:'JetBrains Mono',monospace; font-size:13px; font-weight:700; }
.stat-bar-row { margin-bottom:9px; }
.stat-bar-label { font-size:12px; color:#111827; display:flex;
    justify-content:space-between; margin-bottom:3px; }
.stat-bar-track { background:#e0d8cc; height:7px; border-radius:3px; }
.stat-bar-fill { height:7px; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

def wt_score(stats, weights, pct):
    tw = sum(weights.get(s,0) for s in stats if s in pct)
    return 0 if tw == 0 else sum(pct[s]*weights[s] for s in stats if s in pct and s in weights) / tw

def overall(cat_scores, weights):
    tw = sum(weights.values())
    return 0 if tw == 0 else sum(cat_scores.get(c,0)*w for c,w in weights.items()) / tw

def clr(v):
    if v >= 75: return "#1a7a45"
    elif v >= 50: return "#91cf60"
    elif v >= 25: return "#f0a500"
    return "#d73027"

@st.cache_data
def load_data(season="2025/26", min_minutes=0):
    return load_season_data(season, min_minutes)

if "_season" not in st.session_state: st.session_state["_season"] = "2025/26"
if "_min_min" not in st.session_state: st.session_state["_min_min"] = 900
data = load_data(st.session_state["_season"], st.session_state["_min_min"])
ALL_STATS = [s for g in report_template.values() for s in g["stats"]]

pre_player = st.session_state.get("pre_select_player") or st.session_state.get("dashboard_player")
pre_team   = st.session_state.get("dashboard_team")

with st.sidebar:
    render_sidebar_nav()
    season, min_minutes = render_season_filter(key_prefix="3")
    if season != st.session_state.get("_season") or min_minutes != st.session_state.get("_min_min"):
        st.session_state["_season"] = season; st.session_state["_min_min"] = min_minutes; st.rerun()

    st.markdown('<div class="home-btn">', unsafe_allow_html=True)
    if st.button("← Home", key="home_pc"): st.switch_page("app.py")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<span class="sb-section-label">League template</span>', unsafe_allow_html=True)
    league_template = st.radio("lt", ["Top 5 leagues","Next 14 competitions","Both"],
                               horizontal=True, label_visibility="collapsed")

    st.markdown('<span class="sb-section-label">Position group</span>', unsafe_allow_html=True)
    position_group = st.selectbox("pg", list(position_groups.keys()), label_visibility="collapsed")
    positions      = position_groups[position_group]

    st.markdown('<span class="sb-section-label">Score type</span>', unsafe_allow_html=True)
    score_mode = st.radio("sm", ["Adjusted (recommended)","Model (raw)"], label_visibility="collapsed")

    st.markdown('<span class="sb-section-label">Filters</span>', unsafe_allow_html=True)
    countries     = ["All"] + sorted(data["Birth country"].dropna().unique())
    country       = st.selectbox("Country", countries)
    league_opts   = ["All"] + sorted(data["League"].dropna().unique())
    league_filter = st.selectbox("League", league_opts)
    age_range     = st.slider("Age", int(data["Age"].min()), int(data["Age"].max()),
                               (int(data["Age"].min()), int(data["Age"].max())))

    st.markdown('<span class="sb-section-label">Score filters (0–100)</span>', unsafe_allow_html=True)
    overall_range     = st.slider("Overall",       0,100,(0,100),key="pc_ov")
    goalscoring_range = st.slider("Goalscoring",   0,100,(0,100),key="pc_gs")
    chance_range      = st.slider("Chance creation",0,100,(0,100),key="pc_cc")
    dribbling_range   = st.slider("Dribbling",     0,100,(0,100),key="pc_dr")
    passing_range     = st.slider("Passing",       0,100,(0,100),key="pc_pa")
    defending_range   = st.slider("Defending",     0,100,(0,100),key="pc_de")

# ── Score computation ──────────────────────────────────────────────────────────
if league_template == "Top 5 leagues":
    allowed = TOP5_LEAGUES; mdict = LEAGUE_MULTIPLIERS_ALL
elif league_template == "Next 14 competitions":
    allowed = NEXT14_LEAGUES; mdict = LEAGUE_MULTIPLIERS_NEXT14
else:
    allowed = set(LEAGUE_MULTIPLIERS_ALL.keys()); mdict = LEAGUE_MULTIPLIERS_ALL

pct_data = data[data["Main Position"].isin(positions) & data["League"].isin(allowed)].copy()
ex_stats = [s for s in ALL_STATS if s in pct_data.columns]
pct_raw  = pct_data[ex_stats].rank(pct=True) * 100
pct_raw  = pct_raw.fillna(50)
lg_mult  = pct_data["League"].map(mdict).fillna(1.0)
pct      = pct_raw.multiply(lg_mult.values, axis=0).clip(0, 100)

for cat, grp in report_template.items():
    stats = [s for s in grp["stats"] if s in pct.columns]
    neg   = grp.get("negative_stats",[])
    def _sr(row, stats=stats, weights=grp.get("weights",{}), neg=neg):
        adj = row.copy()
        for ns in neg:
            if ns in adj: adj[ns] = 100 - adj[ns]
        return wt_score(stats, weights, adj)
    pct_data[f"{cat}_score"] = pct[stats].apply(_sr, axis=1)

cw = position_category_weights.get(position_group,{})
pct_data["overall_score"] = pct_data.apply(
    lambda r: overall({c: r.get(f"{c}_score",0) for c in report_template}, cw), axis=1)

fd = pct_data.copy()
if country != "All":       fd = fd[fd["Birth country"] == country]
if league_filter != "All": fd = fd[fd["League"] == league_filter]
fd = fd[fd["Age"].between(*age_range)]

ov_s_raw = (fd["overall_score"]/100)**0.45*100 if score_mode=="Adjusted (recommended)" else fd["overall_score"]
fd["_ov_display"] = ov_s_raw
fd = fd[fd["_ov_display"].between(*overall_range)]
fd = fd[fd["Goalscoring_score"].between(*goalscoring_range)]
fd = fd[fd["Chance creation_score"].between(*chance_range)]
fd = fd[fd["Dribbling_score"].between(*dribbling_range)]
fd = fd[fd["Passing_score"].between(*passing_range)]
fd = fd[fd["Defending_score"].between(*defending_range)]

if fd.empty: st.warning("No players found with current filters."); st.stop()
fd["rank"] = fd["_ov_display"].rank(method="first", ascending=False)

all_players = sorted(fd["Player"].unique())
pre_idx = all_players.index(pre_player) if pre_player in all_players else 0
player  = st.sidebar.selectbox("Player", all_players, index=pre_idx)
if st.session_state.get("pre_select_player"): st.session_state["pre_select_player"] = ""

pr         = fd[fd["Player"] == player].iloc[0]
team       = pr["Team within selected timeframe"]
cat_scores = {c: pr.get(f"{c}_score",0) for c in report_template}
ov         = overall(cat_scores, cw)
if score_mode == "Adjusted (recommended)": ov = (ov/100)**0.45*100

player_pct = pct[(pct_data["Player"]==player)&(pct_data["Team within selected timeframe"]==team)].iloc[0].copy()
for grp in report_template.values():
    for ns in grp.get("negative_stats",[]):
        if ns in player_pct: player_pct[ns] = 100 - player_pct[ns]

# ── Player header ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:#111827;border-radius:10px;padding:18px 22px;margin-bottom:20px;
            display:flex;justify-content:space-between;align-items:center;">
  <div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700;
                text-transform:uppercase;letter-spacing:0.12em;color:rgba(201,168,76,0.5);margin-bottom:5px;">
      {position_group} · {league_template} · {score_mode.split()[0]}
    </div>
    <div style="font-size:22px;font-weight:700;color:white;letter-spacing:-0.01em;">{player}</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:rgba(255,255,255,0.4);margin-top:4px;">
      {pr['Main Position']} · {team} · {pr['Age']} yrs · {pr['League']} · {pr.get('Minutes played','—')} mins ·
      Rank {int(pr['rank']) if not math.isnan(float(pr['rank'])) else '—'} / {len(fd)}
    </div>
  </div>
  <div style="background:{clr(ov)};color:white;width:64px;height:64px;border-radius:8px;
              display:flex;align-items:center;justify-content:center;
              font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;flex-shrink:0;">
    {ov:.0f}
  </div>
</div>
""", unsafe_allow_html=True)

# Export HTML
def _build_html(player, pr, team, ov, player_pct, pct_data_row):
    def _c(v):
        if v>=75: return "#1a9850"
        if v>=50: return "#91cf60"
        if v>=25: return "#f0a500"
        return "#d73027"
    def _bar(label, value, raw=None):
        ll=label.lower(); ip="per pass" in ll and "received" not in ll; ir="per received pass" in ll
        if ip: label=label.replace(" per pass","")
        elif ir: label=label.replace(" per received pass","")
        sfx=""; d=raw
        if (ip or ir) and raw is not None:
            try: d=float(str(raw).replace(",","."))*100; sfx=" per 100 passes" if ip else " per 100 received passes"
            except: pass
        rt=f"{d:.1f}{sfx}" if isinstance(d,(int,float)) else (str(d).replace(",",".") if d is not None else "")
        c=_c(value)
        return (f'<div style="margin-bottom:7px"><div style="font-size:11px;color:#111;display:flex;'
                f'justify-content:space-between;"><span>{label}</span><span style="color:#777">{rt}</span></div>'
                f'<div style="background:#e0d8cc;height:5px;border-radius:3px;margin-top:2px;">'
                f'<div style="width:{value:.1f}%;background:{c};height:5px;border-radius:3px;"></div></div></div>')
    hdr=(f'<div style="background:#111827;border-radius:8px;padding:14px 18px;margin-bottom:14px;'
         f'display:flex;justify-content:space-between;align-items:center;">'
         f'<div><div style="font-size:20px;font-weight:700;color:white;">{player}</div>'
         f'<div style="font-size:11px;color:rgba(255,255,255,0.4);">{pr["Main Position"]} · {team} · {pr["Age"]} yrs · {pr["League"]}</div></div>'
         f'<div style="background:{_c(ov)};color:white;padding:8px 14px;font-size:18px;font-weight:700;border-radius:6px;">{ov:.0f}</div></div>')
    cats=""
    for cat, grp in report_template.items():
        stats=[s for s in grp["stats"] if s in player_pct.index]
        sc=wt_score(stats, grp.get("weights",{}), player_pct); c=_c(sc)
        half=math.ceil(len(stats)/2)
        L="".join(_bar(s,player_pct[s],pct_data_row.get(s)) for s in stats[:half] if s in player_pct)
        R="".join(_bar(s,player_pct[s],pct_data_row.get(s)) for s in stats[half:] if s in player_pct)
        cats+=(f'<div style="background:#f0ebe2;padding:7px 10px;margin:10px 0 7px;display:flex;'
               f'justify-content:space-between;align-items:center;border-radius:4px;border-left:3px solid {c};">'
               f'<b style="font-size:12px;color:#111;">{cat}</b>'
               f'<span style="background:{c};color:white;padding:2px 7px;border-radius:6px;font-size:11px;">{sc:.0f}</span></div>'
               f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0 16px;"><div>{L}</div><div>{R}</div></div>')
    return (f'<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<style>body{{font-family:"DM Sans",Arial,sans-serif;background:#faf7f2;color:#111;padding:18px;max-width:860px;margin:0 auto;}}</style>'
            f'</head><body>{hdr}{cats}'
            f'<div style="font-size:9px;color:#b0a898;text-align:right;margin-top:10px;">Target Scouting · Player Card</div>'
            f'</body></html>')

card_html = _build_html(player, pr, team, ov, player_pct, pr)
st.download_button("Download Player Card (HTML)", card_html.encode("utf-8"),
                   f"{player.replace(' ','_')}_card.html", mime="text/html")

# ── Category bars ──────────────────────────────────────────────────────────────
for cat, grp in report_template.items():
    stats = [s for s in grp["stats"] if s in player_pct.index]
    sc    = wt_score(stats, grp.get("weights",{}), player_pct)
    cc    = clr(sc)

    st.markdown(f"""
    <div style="background:#f0ebe2;border-left:3px solid {cc};border-radius:0 8px 8px 0;
        padding:10px 14px;margin-top:16px;margin-bottom:12px;
        display:flex;justify-content:space-between;align-items:center;">
        <b style="color:#111827;font-size:13px;">{cat}</b>
        <span style="background:{cc};color:white;padding:3px 10px;border-radius:6px;
              font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;">{sc:.0f}</span>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    half = math.ceil(len(stats)/2)

    def _stat_bar(label, value, raw=None):
        ll = label.lower()
        is_pp = "per pass" in ll and "received" not in ll
        is_pr = "per received pass" in ll
        if is_pp:   label = label.replace(" per pass","")
        elif is_pr: label = label.replace(" per received pass","")
        sfx = ""; disp = raw
        if (is_pp or is_pr) and raw is not None:
            try: disp = float(str(raw).replace(",","."))*100; sfx = " /100 passes" if is_pp else " /100 rec."
            except: disp = raw
        rt = f"{disp:.1f}{sfx}" if isinstance(disp,(int,float)) else str(disp).replace(",",".")
        bc = clr(value)
        st.markdown(f"""<div class="stat-bar-row">
            <div class="stat-bar-label">
                <span>{label}</span><span style="color:#7a7060;">{rt}</span>
            </div>
            <div class="stat-bar-track">
                <div class="stat-bar-fill" style="width:{value:.1f}%;background:{bc};"></div>
            </div></div>""", unsafe_allow_html=True)

    with c1:
        for s in stats[:half]:
            if s in player_pct: _stat_bar(s, player_pct[s], pr.get(s))
    with c2:
        for s in stats[half:]:
            if s in player_pct: _stat_bar(s, player_pct[s], pr.get(s))
