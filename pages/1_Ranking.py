import os, sys
from io import BytesIO
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd

from shared.data_processing import preprocess_data, load_season_data
from shared.season_filter import render_season_filter
from shared.styles import BASE_CSS
from shared.sidebar_nav import render_sidebar_nav
from shared.templates import (
    template_config, role_config, position_groups,
    position_to_template, position_map,
    TOP5_LEAGUES, NEXT14_LEAGUES,
    LEAGUE_MULTIPLIERS_ALL, LEAGUE_MULTIPLIERS_NEXT14,
)

def _zscores(data, stats):
    df = data.copy()
    for s in stats:
        df[s] = pd.to_numeric(df[s], errors="coerce")
        m, sd = df[s].mean(), df[s].std()
        df[f"{s}_z"] = 0 if sd == 0 else (df[s] - m) / sd
    return df

def _scale(df, stats):
    df = df.copy()
    for s in stats:
        df[f"{s}_score"] = df[f"{s}_z"].rank(pct=True) * 100
        df[f"{s}_score"] = df[f"{s}_score"].fillna(50)
    return df

def _filter_scores(df, stats, filters):
    f = df.copy()
    for s in stats:
        lo, hi = filters[s]
        f = f[f[f"{s}_score"].between(lo, hi)]
    return f

def _rating(df, stats, weights):
    df = df.copy()
    tw = sum(weights.values())
    df["Rating"] = sum(df[f"{s}_score"] * w for s, w in weights.items()) / tw
    return df

def _league_adj(df, tmpl):
    df = df.copy()
    def _m(lg):
        if tmpl == "Top 5":   return LEAGUE_MULTIPLIERS_ALL.get(lg, 1.0) if lg in TOP5_LEAGUES else 1.0
        if tmpl == "Next 14": return LEAGUE_MULTIPLIERS_NEXT14.get(lg, 1.0) if lg in NEXT14_LEAGUES else 1.0
        return LEAGUE_MULTIPLIERS_ALL.get(lg, 1.0)
    df["Rating"] = df.apply(lambda r: r["Rating"] * _m(r["League"]), axis=1)
    return df

def _c(v):
    if v >= 75: return "#1a7a45"
    elif v >= 50: return "#91cf60"
    elif v >= 25: return "#f0a500"
    return "#d73027"

st.set_page_config(page_title="Ranking · Target Scouting", layout="wide", initial_sidebar_state="expanded")
st.markdown(BASE_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
.block-container { padding-top:1.6rem !important; max-width:1280px !important; }

/* Role callout */
.role-callout {
    background: #111827; border-radius: 8px;
    padding: 16px 20px; margin-bottom: 20px;
    display: flex; justify-content: space-between; align-items: flex-start; gap: 20px;
}
.rc-name { font-size: 17px; font-weight: 700; color: #c9a84c; letter-spacing: -0.01em; }
.rc-pg {
    font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em; color: rgba(201,168,76,0.5);
    margin-bottom: 5px;
}
.rc-desc { font-size: 12px; color: rgba(255,255,255,0.5); margin-top: 4px; line-height: 1.5; }
.rc-stat-pills { display: flex; gap: 5px; flex-wrap: wrap; margin-top: 10px; }
.rc-stat-pill {
    font-family: 'JetBrains Mono', monospace; font-size: 8px; font-weight: 700;
    padding: 3px 8px; background: rgba(201,168,76,0.12);
    color: #c9a84c; border: 0.5px solid rgba(201,168,76,0.3);
    border-radius: 3px; text-transform: uppercase; letter-spacing: 0.05em;
}

/* Section label */
.section-lbl {
    font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 700;
    color: #b0a898; text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 10px; display: flex; align-items: center; gap: 10px;
}
.section-lbl::after { content: ''; flex: 1; height: 0.5px; background: #e0d8cc; }

/* Ranking table */
.rank-table { width: 100%; border-collapse: collapse; }
.rank-table thead th {
    font-family: 'JetBrains Mono', monospace; font-size: 8px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em; color: #b0a898;
    padding: 10px 10px; border-bottom: 1px solid #e0d8cc; background: #f0ebe2;
    text-align: left; white-space: nowrap;
}
.rank-table thead th.r { text-align: right; }
.rank-table tbody tr { cursor: pointer; transition: background 0.1s; }
.rank-table tbody tr:hover { background: #f0ebe2; }
.rank-table tbody td { padding: 10px 10px; border-bottom: 0.5px solid #e0d8cc; font-size: 13px; }
.rank-table tbody td.r { text-align: right; }
.rank-badge {
    width: 26px; height: 26px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700;
}
.rank-badge.gold   { background: #c9a84c; color: #111827; }
.rank-badge.silver { background: #9aa5b4; color: #fff; }
.rank-badge.bronze { background: #a0674a; color: #fff; }
.rank-badge.plain  { background: transparent; color: #b0a898; font-size: 11px; }
.pos-tag {
    font-family: 'JetBrains Mono', monospace; font-size: 8px; font-weight: 700;
    background: #111827; color: #c9a84c; padding: 2px 6px; border-radius: 3px;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.rating-bar-wrap { display: flex; align-items: center; gap: 8px; }
.rating-bar { height: 6px; border-radius: 3px; background: #f0ebe2; flex: 1; overflow: hidden; }
.rating-bar-fill { height: 100%; border-radius: 3px; }
.rating-val {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    font-weight: 700; min-width: 34px; text-align: right;
}
.sl-btn {
    background: transparent; border: 0.5px solid #e0d8cc; color: #b0a898;
    font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em; padding: 4px 8px;
    border-radius: 3px; cursor: pointer;
}
.sl-btn:hover { border-color: #c9a84c; color: #c9a84c; }
</style>
""", unsafe_allow_html=True)

for k, v in [("shortlist", []), ("dashboard_player", None), ("dashboard_team", None), ("dashboard_position_group", None)]:
    if k not in st.session_state: st.session_state[k] = v

@st.cache_data
def load_data(season: str = "2025/26", min_minutes: int = 0):
    return load_season_data(season, min_minutes)

if "_season" not in st.session_state: st.session_state["_season"] = "2025/26"
if "_min_min" not in st.session_state: st.session_state["_min_min"] = 900
data = load_data(st.session_state["_season"], st.session_state["_min_min"])

with st.sidebar:
    render_sidebar_nav("ranking")
    season, min_minutes = render_season_filter(key_prefix="1")
    if season != st.session_state.get("_season") or min_minutes != st.session_state.get("_min_min"):
        st.session_state["_season"] = season
        st.session_state["_min_min"] = min_minutes
        st.rerun()

    st.markdown('<span class="sb-section-label">League</span>', unsafe_allow_html=True)
    league_template = st.radio("lt", ["Top 5", "Next 14", "Both"], horizontal=True, label_visibility="collapsed")

    st.markdown('<span class="sb-section-label">Position</span>', unsafe_allow_html=True)
    position_group = st.selectbox("pg", list(position_groups.keys()), label_visibility="collapsed")
    roles_for_pos  = role_config.get(position_group, {})
    role_options   = ["— Custom —"] + list(roles_for_pos.keys())
    selected_role  = st.selectbox("role", role_options, label_visibility="collapsed")

    st.markdown('<span class="sb-section-label">Filters</span>', unsafe_allow_html=True)
    age_range     = st.slider("Age", 16, 40, (16, 40))
    minutes_range = st.slider("Minutes", 0, 10_000, (0, 10_000), step=100)
    foot          = st.multiselect("Foot", ["left", "right", "both"])
    contract_filter = st.selectbox(
        "Contract expires",
        ["Any", "< 6 months", "< 12 months", "< 18 months", "< 24 months"],
        key="rk_contract",
    )

    st.markdown('<span class="sb-section-label">Performance</span>', unsafe_allow_html=True)
    template_stats = template_config[position_to_template[position_group]]["stats"]
    use_role       = selected_role != "— Custom —"

    if use_role:
        role_data   = roles_for_pos[selected_role]
        stats       = list(role_data["stats"].keys())
        raw_weights = role_data["stats"]
        weights     = {s: st.slider(s, 0.0, 1.0, float(raw_weights[s]), step=0.025) for s in stats}
        score_filters = {s: (0, 100) for s in stats}
    else:
        stats = st.multiselect("Stats", template_stats, default=[])
        if not stats:
            st.info("Select stats to generate ranking")
            st.stop()
        score_filters = {s: st.slider(s, 0, 100, (0, 100)) for s in stats}
        default_w     = 1 / len(stats)
        weights       = {s: st.slider(f"{s} weight", 0.0, 1.0, default_w) for s in stats}

    tw = sum(weights.values())
    if tw == 0: st.warning("Total weight is 0"); st.stop()
    norm_w = {s: w / tw for s, w in weights.items()}

# ── Data pipeline ─────────────────────────────────────────────────────────────
positions = position_groups[position_group]
pos_data  = data[data["Main Position"].isin(positions)].copy()
if league_template == "Top 5":   pos_data = pos_data[pos_data["League"].isin(TOP5_LEAGUES)]
elif league_template == "Next 14": pos_data = pos_data[pos_data["League"].isin(NEXT14_LEAGUES)]

fil = pos_data[pos_data["Age"].between(*age_range) & pos_data["Minutes played"].between(*minutes_range)].copy()
if foot: fil = fil[fil["Foot"].isin(foot)]
if contract_filter != "Any" and "Contract expires" in fil.columns:
    import pandas as _pd2, datetime
    threshold_map = {"< 6 months":6,"< 12 months":12,"< 18 months":18,"< 24 months":24}
    threshold  = threshold_map[contract_filter]
    today_ts   = _pd2.Timestamp(datetime.date.today())
    fil["_contract_exp"] = _pd2.to_datetime(fil["Contract expires"], errors="coerce")
    fil["_months_left"]  = ((fil["_contract_exp"] - today_ts).dt.days / 30)
    fil = fil[fil["_months_left"].between(0, threshold)]

idx     = ["Player","Team within selected timeframe"]
scored  = _scale(_zscores(pos_data, stats), stats)
fscored = _filter_scores(scored, stats, score_filters)
final   = fscored[fscored.set_index(idx).index.isin(fil.set_index(idx).index)]
ranking = _league_adj(_rating(final, stats, norm_w), league_template)

if ranking.empty:
    st.markdown(f"## {position_group} Ranking")
    st.warning("No players found with current filters.")
    st.stop()

ranking["Rank"] = ranking["Rating"].rank(method="first", ascending=False).astype(int)
ranking = ranking.sort_values("Rating", ascending=False).reset_index(drop=True)

# ── Page header ────────────────────────────────────────────────────────────────
role_label = f" · {selected_role}" if use_role else ""
st.markdown(f"""
<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:20px;">
  <h1 style="font-size:20px;font-weight:700;letter-spacing:-0.01em;">{position_group} Ranking{role_label}</h1>
  <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#b0a898;text-transform:uppercase;letter-spacing:0.08em;">
    {league_template} · {len(ranking)} players
  </span>
</div>
""", unsafe_allow_html=True)

# ── Role callout (if role selected) ───────────────────────────────────────────
if use_role:
    role_data = roles_for_pos[selected_role]
    stat_pills = "".join(f'<span class="rc-stat-pill">{s} &#xD7;{w:.2f}</span>' for s,w in raw_weights.items())
    st.markdown(f"""
    <div class="role-callout">
      <div>
        <div class="rc-pg">{position_group} · {league_template}</div>
        <div class="rc-name">{selected_role}</div>
        <div class="rc-desc">{role_data.get("description","")}</div>
        <div class="rc-stat-pills">{stat_pills}</div>
      </div>
    </div>""", unsafe_allow_html=True)

tab_r, tab_c, tab_s = st.tabs(["Ranking", "Compare", "Shortlist"])

with tab_r:
    pre_search = st.session_state.get("pre_select_player","") or ""
    search = st.text_input("", placeholder="Search player, club or nationality…",
                           label_visibility="collapsed", value=pre_search)
    if pre_search: st.session_state["pre_select_player"] = ""

    dr = ranking.copy()
    if search: dr = dr[dr["Player"].str.contains(search, case=False, na=False)]

    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#b0a898;margin-bottom:12px;">
      <b style="color:#111827;">{len(dr)}</b> players &nbsp;·&nbsp;
      <b style="color:#111827;">{league_template}</b> &nbsp;·&nbsp;
      <b style="color:#111827;">{selected_role}</b>
    </div>""", unsafe_allow_html=True)

    # Build HTML ranking table
    rows_html = ""
    for i, (_, row) in enumerate(dr.head(30).iterrows()):
        rank = int(row["Rank"])
        if rank == 1:   badge = '<span class="rank-badge gold">1</span>'
        elif rank == 2: badge = '<span class="rank-badge silver">2</span>'
        elif rank == 3: badge = '<span class="rank-badge bronze">3</span>'
        else:           badge = f'<span class="rank-badge plain">{rank}</span>'

        rating  = float(row["Rating"])
        col     = _c(rating)
        pct     = min(100, rating)
        pos_lbl = str(row.get("Position Label", row.get("Main Position",""))).split(",")[0].strip()[:2].upper()
        team    = row.get("Team within selected timeframe","")
        league  = row.get("League","")
        age     = row.get("Age","")
        mins    = row.get("Minutes played","")
        try: age_str  = str(int(float(age)))
        except: age_str = "—"
        try: mins_str = f"{int(float(mins)):,}"
        except: mins_str = "—"

        rows_html += f"""<tr>
          <td style="width:42px;">{badge}</td>
          <td>
            <div style="font-weight:700;">{row["Player"]}</div>
            <div style="margin-top:2px;"><span class="pos-tag">{pos_lbl}</span></div>
          </td>
          <td style="color:#7a7060;">{team}</td>
          <td style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#b0a898;">{league}</td>
          <td class="r" style="font-family:'JetBrains Mono',monospace;font-size:12px;">{age_str}</td>
          <td class="r" style="font-family:'JetBrains Mono',monospace;font-size:12px;">{mins_str}</td>
          <td style="min-width:160px;">
            <div class="rating-bar-wrap">
              <div class="rating-bar"><div class="rating-bar-fill" style="width:{pct:.0f}%;background:{col};"></div></div>
              <span class="rating-val" style="color:{col};">{rating:.1f}</span>
            </div>
          </td>
          <td><button class="sl-btn">+ List</button></td>
        </tr>"""

    st.markdown(f"""
    <div style="background:#fff;border:0.5px solid #e0d8cc;border-radius:8px;overflow:hidden;">
    <table class="rank-table">
      <thead><tr>
        <th>#</th><th>Player</th><th>Club</th><th>League</th>
        <th class="r">Age</th><th class="r">Min</th>
        <th style="min-width:160px;">Rating</th><th></th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-lbl">Open in Dashboard</div>', unsafe_allow_html=True)
    dash_p = st.selectbox("Select player for Dashboard", ["—"] + dr["Player"].head(30).tolist(), key="dash_sel")
    if st.button("Open Dashboard →", key="open_dash"):
        if dash_p != "—":
            row = dr[dr["Player"] == dash_p].iloc[0]
            st.session_state.dashboard_player         = dash_p
            st.session_state.dashboard_team           = row["Team within selected timeframe"]
            st.session_state.dashboard_position_group = position_group
            st.session_state.dashboard_generated      = True
            st.switch_page("pages/5_Dashboard.py")
        else:
            st.warning("Select a player first.")

    st.markdown('<div class="section-lbl">Add to shortlist</div>', unsafe_allow_html=True)
    to_add = st.selectbox("Select player for shortlist", ["—"] + dr["Player"].head(30).tolist(), key="sl_add")
    if st.button("Add to shortlist", key="btn_sl_add"):
        if to_add != "—":
            if to_add not in st.session_state.shortlist:
                st.session_state.shortlist.append(to_add)
                st.success(f"{to_add} added.")
            else:
                st.info("Already in shortlist.")

    st.markdown('<div class="section-lbl">Export</div>', unsafe_allow_html=True)
    disp = (dr[["Rank","Player","Age","Position Label","Foot","Team within selected timeframe","League","Rating"]]
            .head(30)
            .rename(columns={"Team within selected timeframe":"Team","Position Label":"Position"})
            .round(1))
    disp["Rank"]   = disp["Rank"].astype(str)
    disp["Rating"] = disp["Rating"].map(lambda x: f"{x:.1f}")
    disp["Age"]    = disp["Age"].astype(str)

    ce1, ce2 = st.columns(2)
    with ce1:
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            disp.to_excel(w, index=False, sheet_name="Ranking")
        st.download_button("Download Excel", buf.getvalue(),
                           f"{position_group}_ranking.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with ce2:
        hr = "".join(f"<tr>{''.join(f'<td>{v}</td>' for v in row)}</tr>" for row in disp.values.tolist())
        hh = "".join(f"<th>{c}</th>" for c in disp.columns)
        html = (f"<html><head><style>body{{font-family:'DM Sans',Arial,sans-serif;background:#faf7f2;padding:20px;}}"
                f"table{{border-collapse:collapse;width:100%;}}"
                f"th{{background:#111827;color:#c9a84c;padding:8px 12px;text-align:left;font-family:'JetBrains Mono',monospace;font-size:11px;}}"
                f"td{{padding:7px 12px;border-bottom:1px solid #e0d8cc;}}"
                f"tr:nth-child(even){{background:#f0ebe2;}}"
                f"</style></head><body><h2 style='font-family:DM Sans;margin-bottom:16px;'>{position_group} – {selected_role}</h2>"
                f"<table><thead><tr>{hh}</tr></thead><tbody>{hr}</tbody></table></body></html>")
        st.download_button("Download HTML", html.encode(),
                           f"{position_group}_ranking.html", mime="text/html")

with tab_c:
    st.markdown('<div class="section-lbl">Compare two players</div>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca: pa = st.selectbox("Player A", ["—"] + ranking["Player"].tolist(), key="cmp_a")
    with cb: pb = st.selectbox("Player B", ["—"] + ranking["Player"].tolist(), key="cmp_b")
    if pa != "—" and pb != "—" and pa != pb:
        ra, rb = ranking[ranking["Player"] == pa].iloc[0], ranking[ranking["Player"] == pb].iloc[0]
        cols_c = ["Player","Age","Position Label","Foot","Team within selected timeframe","League","Rating"] + [f"{s}_score" for s in stats]
        def _f(r, k): v = r.get(k,""); return f"{v:.1f}" if isinstance(v, float) else str(v)
        rows = [{"Metric": k.replace("_score"," (score)").replace("Team within selected timeframe","Team"),
                 pa: _f(ra,k), pb: _f(rb,k)} for k in cols_c]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    elif pa == pb and pa != "—": st.info("Select two different players.")
    else: st.info("Select two players to compare.")

with tab_s:
    st.markdown('<div class="section-lbl">Shortlist</div>', unsafe_allow_html=True)
    if not st.session_state.shortlist:
        st.info("Your shortlist is empty. Add players from the Ranking tab.")
    else:
        sl = ranking[ranking["Player"].isin(st.session_state.shortlist)][
            ["Rank","Player","Age","Position Label","Foot","Team within selected timeframe","League","Rating"]
        ].rename(columns={"Team within selected timeframe":"Team","Position Label":"Position"}).round(1)
        sl["Rating"] = sl["Rating"].map(lambda x: f"{x:.1f}")
        st.dataframe(sl, use_container_width=True, hide_index=True)
        rm = st.selectbox("Remove player", ["—"] + st.session_state.shortlist, key="rm")
        if st.button("Remove", key="btn_rm") and rm != "—":
            st.session_state.shortlist.remove(rm); st.rerun()
        buf2 = BytesIO()
        with pd.ExcelWriter(buf2, engine="openpyxl") as w:
            sl.to_excel(w, index=False, sheet_name="Shortlist")
        st.download_button("Export Shortlist (Excel)", buf2.getvalue(), "shortlist.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
