"""
pages/1_Ranking.py — Versie 2

Twee modi:
  A. Find by role  → dimensie-gebaseerde rol-rankings via shared/roles_v2.py
  B. Similar to player → archetype-match + tier-adjusted cosine similarity
                         (zoals in Dashboard, maar als zelfstandige zoek-pagina)

Nieuwe filters: leeftijd, voet, contract, minuten, league, market value.
Behoud: shortlist, compare, export.
"""
import os, sys, datetime
from io import BytesIO

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np

from shared.data_processing import preprocess_data, load_season_data
from shared.season_filter import render_season_filter
from shared.styles import BASE_CSS
from shared.sidebar_nav import render_sidebar_nav
from shared.templates import (
    position_groups, position_to_template, position_map,
    TOP5_LEAGUES, NEXT14_LEAGUES,
)
from shared.roles_v2 import (
    ROLE_CONFIG_V2, POSITION_DIMENSIONS,
    compute_dimension_scores, compute_role_score,
    get_role_options,
)
from shared.scoring import compute_role_ranking
from shared.similarity import adjusted_similarity, tier_badge_color, LEAGUE_TIERS
from shared.archetypes import (
    train_or_load_models, get_player_archetype, ARCHETYPE_COLORS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ranking · Target Scouting", layout="wide", initial_sidebar_state="expanded")
st.markdown(BASE_CSS, unsafe_allow_html=True)

# Page-specifieke CSS — mode toggle, ranking row, etc
st.markdown("""
<style>
.mode-toggle {
    display: flex; gap: 0; margin-bottom: 14px;
    border: 0.5px solid #e0d8cc; border-radius: 6px; overflow: hidden;
    background: #fff;
}
.mode-toggle div {
    flex: 1; padding: 10px 16px; text-align: center; cursor: pointer;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.06em;
    color: #7a7060; transition: all 0.12s;
}
.mode-toggle div.active {
    background: #111827; color: #c9a84c; font-weight: 600;
}
.mode-info {
    background: #f0ebe2; border-left: 3px solid #c9a84c;
    border-radius: 0 6px 6px 0; padding: 10px 14px; margin-bottom: 14px;
    font-size: 12px; color: #7a7060;
}
.role-desc {
    font-size: 11px; color: #7a7060; font-style: italic;
    margin: 6px 0 10px 0;
}
.archetype-pill {
    display: inline-block; padding: 4px 10px; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    font-weight: 600; letter-spacing: 0.04em;
    color: #faf7f2; margin-left: 8px;
}
.dim-row {
    display: grid; grid-template-columns: 160px 1fr 40px;
    gap: 8px; align-items: center; margin-bottom: 4px;
    font-size: 11px;
}
.dim-track {
    background: #f0ebe2; height: 6px; border-radius: 2px;
}
.dim-fill { height: 6px; border-radius: 2px; }
.tier-badge {
    display: inline-block; padding: 2px 7px; border-radius: 3px;
    font-family: 'JetBrains Mono', monospace; font-size: 9px;
    font-weight: 500; letter-spacing: 0.04em;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Init session state
for k, v in [
    ("shortlist", []),
    ("dashboard_player", None), ("dashboard_team", None),
    ("dashboard_position_group", None),
    ("ranking_mode", "Find by role"),
]:
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_data
def load_data(season: str = "2025/26", min_minutes: int = 0):
    return load_season_data(season, min_minutes)


# Season/minutes komen uit de sidebar
if "_season" not in st.session_state:
    st.session_state["_season"] = "2025/26"
if "_min_min" not in st.session_state:
    st.session_state["_min_min"] = 900

data = load_data(st.session_state["_season"], st.session_state["_min_min"])


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav("ranking")

    season, min_minutes = render_season_filter(key_prefix="1")
    if season != st.session_state.get("_season") or min_minutes != st.session_state.get("_min_min"):
        st.session_state["_season"] = season
        st.session_state["_min_min"] = min_minutes
        st.rerun()

    # ── Mode selector ──
    st.markdown('<span class="sb-section-label">Search mode</span>', unsafe_allow_html=True)
    mode = st.radio(
        "mode",
        ["Find by role", "Similar to player"],
        label_visibility="collapsed",
        key="rk_mode",
    )

    st.markdown('<span class="sb-section-label">League pool</span>', unsafe_allow_html=True)
    league_template = st.radio(
        "lt", ["Top 5", "Next 14", "Both"],
        horizontal=True, label_visibility="collapsed",
    )

    # ── Mode-specific UI ──
    if mode == "Find by role":
        st.markdown('<span class="sb-section-label">Position</span>', unsafe_allow_html=True)
        pos_options = list(ROLE_CONFIG_V2.keys())
        # Filter "Winger" alias eruit (alleen Right/Left tonen)
        pos_options = [p for p in pos_options if p != "Winger"]
        position_label = st.selectbox(
            "pos", pos_options,
            label_visibility="collapsed",
            key="rk_pos_role",
        )

        st.markdown('<span class="sb-section-label">Role</span>', unsafe_allow_html=True)
        role_names = get_role_options(position_label)
        if not role_names:
            st.warning(f"No roles defined for {position_label}")
            st.stop()
        selected_role = st.selectbox(
            "role", role_names,
            label_visibility="collapsed",
            key="rk_role",
        )

        # Show role description
        role_def = ROLE_CONFIG_V2[position_label][selected_role]
        st.markdown(
            f'<div class="role-desc">{role_def["description"]}</div>',
            unsafe_allow_html=True,
        )

    else:  # Similar to player
        st.markdown('<span class="sb-section-label">Reference player</span>', unsafe_allow_html=True)
        all_players = sorted(data["Player"].unique())
        # Pre-fill from session_state
        pre = st.session_state.get("ranking_ref_player", "")
        idx_p = all_players.index(pre) if pre in all_players else 0
        ref_player = st.selectbox(
            "ref",
            all_players,
            index=idx_p,
            label_visibility="collapsed",
            key="rk_ref_player",
        )

        # Determine team if multiple
        ref_rows = data[data["Player"] == ref_player]
        if ref_rows.empty:
            st.warning("Player not found")
            st.stop()
        teams_ref = ref_rows["Team within selected timeframe"].unique()
        if len(teams_ref) > 1:
            ref_team = st.selectbox("Team", teams_ref, key="rk_ref_team")
        else:
            ref_team = teams_ref[0]
        ref_row = ref_rows[ref_rows["Team within selected timeframe"] == ref_team].iloc[0]

        ref_pos = str(ref_row.get("Main Position", ""))
        # Auto-detect positiegroep
        ref_pg = next(
            (pg for pg, ps in position_groups.items() if ref_pos in ps),
            list(position_groups.keys())[0],
        )

        st.caption(f"Ref pos: **{ref_pg}** · {ref_team} · {ref_row.get('League','—')}")

    # ── Universele filters ──
    st.markdown('<span class="sb-section-label">Filters</span>', unsafe_allow_html=True)
    age_range = st.slider("Age", 16, 40, (16, 40), key="rk_age")
    minutes_range = st.slider("Minutes", 0, 5000, (600, 5000), step=100, key="rk_min")
    foot_filter = st.multiselect("Foot", ["left", "right", "both"], key="rk_foot")

    contract_filter = st.selectbox(
        "Contract expires",
        ["Any", "< 6 months", "< 12 months", "< 18 months", "< 24 months"],
        key="rk_contract",
    )

    # Market value filter (optioneel)
    if "Market value" in data.columns:
        mv_max = int(data["Market value"].fillna(0).max() / 1_000_000) + 1
        mv_range = st.slider(
            "Market value (€M)", 0, max(mv_max, 100), (0, max(mv_max, 100)),
            step=1, key="rk_mv",
        )
    else:
        mv_range = None


# ──────────────────────────────────────────────────────────────────────────────
# Page title
# ──────────────────────────────────────────────────────────────────────────────
if mode == "Find by role":
    st.title(f"{position_label} · {selected_role}")
    st.markdown(
        f'<div class="mode-info">Mode <b>Find by role</b> · '
        f'Score gebaseerd op gewogen dimensies '
        f'<span style="color:#c9a84c;">(VIF-vrij)</span> · '
        f'{league_template} leagues</div>',
        unsafe_allow_html=True,
    )
else:
    arch_models = train_or_load_models(data)
    primary, secondary, p_dist, s_dist = get_player_archetype(
        ref_row, arch_models, ref_pg
    )
    arch_color = ARCHETYPE_COLORS.get(primary, "#7a7060")
    sec_html = ""
    if secondary:
        sec_color = ARCHETYPE_COLORS.get(secondary, "#7a7060")
        sec_html = f'<span class="archetype-pill" style="background:{sec_color};">{secondary}</span>'
    st.title(f"Similar to {ref_player}")
    st.markdown(
        f'<div class="mode-info">Mode <b>Similar to player</b> · '
        f'Reference: <b>{ref_player}</b> ({ref_team}) · '
        f'<span class="archetype-pill" style="background:{arch_color};">{primary}</span>{sec_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Data pipeline — algemene filters
# ──────────────────────────────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Pas algemene filters toe die voor beide modi gelden."""
    out = df.copy()

    # League
    if league_template == "Top 5":
        out = out[out["League"].isin(TOP5_LEAGUES)]
    elif league_template == "Next 14":
        out = out[out["League"].isin(NEXT14_LEAGUES)]

    # Age
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out = out[out["Age"].between(*age_range)]

    # Minutes
    out["Minutes played"] = pd.to_numeric(out["Minutes played"], errors="coerce")
    out = out[out["Minutes played"].between(*minutes_range)]

    # Foot
    if foot_filter:
        out = out[out["Foot"].isin(foot_filter)]

    # Contract
    if contract_filter != "Any" and "Contract expires" in out.columns:
        thresholds = {
            "< 6 months": 6, "< 12 months": 12,
            "< 18 months": 18, "< 24 months": 24,
        }
        thr = thresholds[contract_filter]
        today_ts = pd.Timestamp(datetime.date.today())
        out["_contract_exp"] = pd.to_datetime(out["Contract expires"], errors="coerce")
        out["_months_left"] = (out["_contract_exp"] - today_ts).dt.days / 30
        out = out[out["_months_left"].between(0, thr)]

    # Market value
    if mv_range is not None and "Market value" in out.columns:
        mv = pd.to_numeric(out["Market value"], errors="coerce").fillna(0) / 1_000_000
        out = out[mv.between(*mv_range)]

    return out


# ──────────────────────────────────────────────────────────────────────────────
# RANKING COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────
if mode == "Find by role":
    # Map naar league_template format dat compute_role_ranking verwacht
    lt_map = {"Top 5": "Top 5 leagues", "Next 14": "Next 14 competitions", "Both": "Both"}
    pool_with_dims = compute_role_ranking(
        data, position_label, selected_role,
        league_template=lt_map[league_template],
        apply_league_multiplier=True,
        apply_shrinkage=True,
    )

    # Apply user filters
    pool_with_dims = apply_filters(pool_with_dims)
    if pool_with_dims.empty:
        st.warning("No players match the filters.")
        st.stop()

    pool_with_dims = pool_with_dims.sort_values("role_score", ascending=False).reset_index(drop=True)
    pool_with_dims["Rank"] = pool_with_dims.index + 1
    ranking = pool_with_dims
    score_col = "role_score"

else:  # Similar to player mode
    # Bouw similarity pool — zelfde positiegroep als referentie-speler
    pos_codes = position_groups.get(ref_pg, [])
    sim_pool = data[data["Main Position"].isin(pos_codes)].copy()
    sim_pool = apply_filters(sim_pool)

    if sim_pool.empty:
        st.warning("No players match the filters.")
        st.stop()

    # Use radar stats for similarity (consistent with Dashboard)
    from shared.templates import ALL_RADAR_STATS
    sim_stats = [s for s in ALL_RADAR_STATS if s in sim_pool.columns]
    if len(sim_stats) < 5:
        st.error("Not enough stats available for similarity")
        st.stop()

    ref_league = ref_row.get("League", "")
    sim_results = adjusted_similarity(
        target_row=ref_row,
        candidates_df=sim_pool,
        sim_stats=sim_stats,
        target_league=ref_league,
        min_minutes=minutes_range[0],
    )

    if sim_results.empty:
        st.warning("No similar players found.")
        st.stop()

    # Filter target player out
    sim_results = sim_results[
        ~((sim_results["Player"] == ref_player) &
          (sim_results["Team within selected timeframe"] == ref_team))
    ].reset_index(drop=True)

    sim_results["Rank"] = sim_results.index + 1
    sim_results["match_pct"] = (sim_results["adjusted_sim"] * 100).round(1)
    ranking = sim_results
    score_col = "match_pct"


# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab_r, tab_c, tab_s = st.tabs(["Ranking", "Compare", "Shortlist"])


with tab_r:
    # Search box
    search = st.text_input(
        "search",
        placeholder="Search player by name…",
        label_visibility="collapsed",
        key="rk_search",
    )
    dr = ranking.copy()
    if search:
        dr = dr[dr["Player"].str.contains(search, case=False, na=False)]

    n_total = len(dr)
    st.caption(f"{n_total} players · {league_template} leagues · {len(data['League'].unique())} leagues in dataset")

    # Build display dataframe
    if mode == "Find by role":
        cols_to_show = ["Rank", "Player", "Age", "Position Label", "Foot",
                        "Team within selected timeframe", "League", "role_score"]
        col_renames = {
            "Team within selected timeframe": "Team",
            "Position Label": "Position",
            "role_score": "Score",
        }
    else:
        cols_to_show = ["Rank", "Player", "Age", "Position Label", "Foot",
                        "Team within selected timeframe", "League",
                        "match_pct", "tier_badge"]
        col_renames = {
            "Team within selected timeframe": "Team",
            "Position Label": "Position",
            "match_pct": "Match %",
            "tier_badge": "Tier",
        }

    # Add Market value if exists
    if "Market value" in dr.columns:
        dr["MV"] = pd.to_numeric(dr["Market value"], errors="coerce").fillna(0) / 1_000_000
        dr["MV"] = dr["MV"].apply(lambda x: f"€{x:.1f}M" if x > 0 else "—")
        cols_to_show.insert(-1 if mode == "Find by role" else -2, "MV")

    available_cols = [c for c in cols_to_show if c in dr.columns]
    disp = dr[available_cols].head(50).rename(columns=col_renames).copy()

    # Format numeric columns
    if "Score" in disp.columns:
        disp["Score"] = disp["Score"].map(lambda x: f"{x:.1f}")
    if "Match %" in disp.columns:
        disp["Match %"] = disp["Match %"].map(lambda x: f"{x:.1f}%")
    disp["Age"] = disp["Age"].astype("Int64").astype(str)
    disp["Rank"] = disp["Rank"].astype(str)

    st.dataframe(disp, use_container_width=True, hide_index=True, height=540)

    st.markdown("---")

    # Action buttons
    ac1, ac2, ac3 = st.columns([2, 2, 1])
    with ac1:
        st.markdown("**Open in Dashboard**")
        dash_p = st.selectbox(
            "Select player",
            ["—"] + dr["Player"].head(50).tolist(),
            key="dash_sel",
            label_visibility="collapsed",
        )
        if st.button("Open Dashboard →", key="open_dash", use_container_width=True):
            if dash_p != "—":
                row = dr[dr["Player"] == dash_p].iloc[0]
                st.session_state.dashboard_player = dash_p
                st.session_state.dashboard_team = row["Team within selected timeframe"]
                # Map naar Dashboard's positiegroep nomenclatuur
                if mode == "Find by role":
                    st.session_state.dashboard_position_group = position_label
                else:
                    st.session_state.dashboard_position_group = ref_pg
                st.session_state["pre_select_player"] = dash_p
                st.switch_page("pages/5_Dashboard.py")
            else:
                st.warning("Select a player first.")

    with ac2:
        st.markdown("**Add to shortlist**")
        sl_p = st.selectbox(
            "Select for shortlist",
            ["—"] + dr["Player"].head(50).tolist(),
            key="sl_add",
            label_visibility="collapsed",
        )
        if st.button("Add to shortlist", key="btn_sl_add", use_container_width=True):
            if sl_p != "—" and sl_p not in st.session_state.shortlist:
                st.session_state.shortlist.append(sl_p)
                st.success(f"{sl_p} added.")
            elif sl_p in st.session_state.shortlist:
                st.info("Already in shortlist.")

    with ac3:
        st.markdown("**Export**")
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            disp.to_excel(w, index=False, sheet_name="Ranking")
        fname = (
            f"{position_label}_{selected_role}".replace(" ", "_")
            if mode == "Find by role"
            else f"similar_to_{ref_player.replace(' ', '_')}"
        )
        st.download_button(
            "Excel",
            buf.getvalue(),
            f"{fname}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# COMPARE TAB
# ──────────────────────────────────────────────────────────────────────────────
with tab_c:
    st.markdown("### Compare two players")

    if mode == "Find by role":
        # Vergelijk dimensie-scores
        ca, cb = st.columns(2)
        with ca:
            pa = st.selectbox("Player A", ["—"] + ranking["Player"].head(50).tolist(), key="cmp_a")
        with cb:
            pb = st.selectbox("Player B", ["—"] + ranking["Player"].head(50).tolist(), key="cmp_b")

        if pa != "—" and pb != "—" and pa != pb:
            ra = ranking[ranking["Player"] == pa].iloc[0]
            rb = ranking[ranking["Player"] == pb].iloc[0]

            # Dimensie-vergelijking
            dims = list(POSITION_DIMENSIONS[position_label].keys())
            comparison_rows = []
            for dim in dims:
                va = ra.get(f"dim_{dim}", 0)
                vb = rb.get(f"dim_{dim}", 0)
                weight = ROLE_CONFIG_V2[position_label][selected_role]["weights"].get(dim, 0)
                comparison_rows.append({
                    "Dimension": dim,
                    "Weight in role": f"{weight*100:.0f}%",
                    pa: f"{va:.0f}",
                    pb: f"{vb:.0f}",
                })
            comparison_rows.append({
                "Dimension": "ROLE SCORE",
                "Weight in role": "—",
                pa: f"{ra['role_score']:.1f}",
                pb: f"{rb['role_score']:.1f}",
            })
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Select two different players to compare.")
    else:
        st.info("Compare-tab is in role-mode beschikbaar (dimensie-vergelijking). Switch naar Find by role om te vergelijken.")


# ──────────────────────────────────────────────────────────────────────────────
# SHORTLIST TAB
# ──────────────────────────────────────────────────────────────────────────────
with tab_s:
    st.markdown("### Shortlist")
    if not st.session_state.shortlist:
        st.info("Your shortlist is empty. Add players from the Ranking tab.")
    else:
        sl_data = data[data["Player"].isin(st.session_state.shortlist)].copy()
        sl_disp_cols = [
            "Player", "Age", "Main Position", "Foot",
            "Team within selected timeframe", "League",
        ]
        if "Market value" in sl_data.columns:
            sl_data["MV"] = pd.to_numeric(sl_data["Market value"], errors="coerce").fillna(0) / 1_000_000
            sl_data["MV"] = sl_data["MV"].apply(lambda x: f"€{x:.1f}M" if x > 0 else "—")
            sl_disp_cols.append("MV")
        if "Contract expires" in sl_data.columns:
            sl_disp_cols.append("Contract expires")

        sl_avail = [c for c in sl_disp_cols if c in sl_data.columns]
        sl_disp = sl_data[sl_avail].rename(columns={
            "Team within selected timeframe": "Team",
            "Main Position": "Position",
        })
        st.dataframe(sl_disp, use_container_width=True, hide_index=True)

        rm = st.selectbox("Remove player", ["—"] + st.session_state.shortlist, key="rm")
        if st.button("Remove", key="btn_rm") and rm != "—":
            st.session_state.shortlist.remove(rm)
            st.rerun()

        # Export
        buf2 = BytesIO()
        with pd.ExcelWriter(buf2, engine="openpyxl") as w:
            sl_disp.to_excel(w, index=False, sheet_name="Shortlist")
        st.download_button(
            "Export Shortlist (Excel)",
            buf2.getvalue(),
            "shortlist.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
