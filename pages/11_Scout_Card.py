"""
pages/11_Scout_Card.py
======================
Dedicated page for generating downloadable Target Scouting report cards.

Self-contained: no changes needed to other files except adding `packages.txt`
in the repo root with the two font packages (see install instructions).
"""
import os, sys, io
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon

from shared.data_processing import load_season_data
from shared.season_filter import render_season_filter
from shared.styles import BASE_CSS
from shared.sidebar_nav import render_sidebar_nav
from shared.scoring import compute_scores
from shared.similarity import adjusted_similarity
from shared.templates import (
    position_groups, RADAR_CATEGORIES, ALL_RADAR_STATS,
    report_template,
)

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Scout Card · Target Scouting",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(BASE_CSS, unsafe_allow_html=True)
render_sidebar_nav()


# ─────────────────────────────────────────────────────────────────────────────
# Brand palette + fonts (with fallbacks if Inter/JetBrains Mono not installed)
# ─────────────────────────────────────────────────────────────────────────────
BG        = "#000000"
PANEL     = "#0c0c0c"
PANEL_BR  = "#1c1c1c"
GOLD      = "#b89243"
WHITE     = "#e8e8e8"
TEXT      = "#c4c4c4"
MUTED     = "#6e6e6e"
TICK      = "#2a2a2a"
TRACK     = "#161616"

FONT_SANS = ["Inter", "DejaVu Sans", "Arial", "sans-serif"]
FONT_MONO = ["JetBrains Mono", "DejaVu Sans Mono", "Courier New", "monospace"]

STAT_FORMATS = {
    "xG per 90": "{:.2f}",
    "xG per shot": "{:.2f}",
    "Finishing": "{:+.2f}",
    "Touches in box per 90": "{:.2f}",
    "xA per 90": "{:.2f}",
    "Shot assists per 90": "{:.2f}",
    "Key passes per pass": "{:.3f}",
    "Accurate crosses per received pass": "{:.3f}",
    "Successful dribbles per received pass": "{:.3f}",
    "Successful dribbles, %": "{:.1f}%",
    "Offensive duels won, %": "{:.1f}%",
    "Progressive runs per received pass": "{:.3f}",
    "Completed progressive passes per 90": "{:.2f}",
    "Completed passes to final third per 90": "{:.2f}",
    "Deep completions per 90": "{:.2f}",
    "Passing accuracy (prog/1/3/forw)": "{:.1f}%",
    "PAdj Defensive duels won per 90": "{:.2f}",
    "Defensive duels won, %": "{:.1f}%",
    "PAdj Aerial duels won per 90": "{:.2f}",
    "Aerial duels won, %": "{:.1f}%",
    "PAdj Interceptions": "{:.2f}",
}

SHORT_NAMES = {
    "Successful dribbles per received pass": "Succ. dribbles per rec.",
    "Progressive runs per received pass":    "Prog. runs per rec.",
    "Accurate crosses per received pass":    "Acc. crosses per rec.",
    "Completed progressive passes per 90":   "Prog. passes",
    "Completed passes to final third per 90":"Passes to final third",
    "Deep completions per 90":               "Deep completions",
    "Passing accuracy (prog/1/3/forw)":      "Pass acc. (prog/1/3)",
    "Touches in box per 90":                 "Touches in box",
    "Key passes per pass":                   "Key passes",
    "PAdj Defensive duels won per 90":       "Def. duels won (PAdj)",
    "Defensive duels won, %":                "Defensive duels %",
    "PAdj Aerial duels won per 90":          "Aerial duels won (PAdj)",
    "Aerial duels won, %":                   "Aerial duels %",
    "PAdj Interceptions":                    "Interceptions (PAdj)",
    "Successful dribbles, %":                "Successful dribbles %",
    "Offensive duels won, %":                "Offensive duels %",
}

COUNTRY_CODES = {
    "Belgium": "BEL", "Netherlands": "NED", "England": "ENG", "Spain": "ESP",
    "France": "FRA", "Germany": "GER", "Italy": "ITA", "Portugal": "POR",
    "Brazil": "BRA", "Argentina": "ARG", "Croatia": "CRO", "Norway": "NOR",
    "Denmark": "DEN", "Sweden": "SWE", "Türkiye": "TUR", "Turkey": "TUR",
    "Mali": "MLI", "Nigeria": "NGA", "Ghana": "GHA", "Morocco": "MAR",
    "Senegal": "SEN", "Cameroon": "CMR", "Côte d'Ivoire": "CIV",
    "Algeria": "ALG", "Egypt": "EGY", "Tunisia": "TUN", "Uruguay": "URU",
    "Chile": "CHI", "Colombia": "COL", "Mexico": "MEX", "USA": "USA",
    "Canada": "CAN", "Scotland": "SCO", "Wales": "WAL", "Ireland": "IRL",
    "Northern Ireland": "NIR", "Switzerland": "SUI", "Austria": "AUT",
    "Slovenia": "SVN", "Slovakia": "SVK", "Poland": "POL",
    "Czech Republic": "CZE", "Hungary": "HUN", "Greece": "GRE",
    "Serbia": "SRB", "Romania": "ROU", "Bulgaria": "BUL", "Ukraine": "UKR",
    "Russia": "RUS", "Japan": "JPN", "South Korea": "KOR",
    "Korea Republic": "KOR", "Australia": "AUS", "Iceland": "ISL",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _percentile(pool_series, value, negative=False):
    s = pd.to_numeric(pool_series, errors="coerce").dropna()
    if pd.isna(value) or len(s) == 0:
        return 50.0
    pct = (s <= value).mean() * 100
    if negative:
        pct = 100 - pct
    return pct


def _category_score(row, pool, category_name):
    rt_name = "Chance creation" if category_name == "Chance Creation" else category_name
    cfg = report_template.get(rt_name)
    if cfg is None:
        stats = RADAR_CATEGORIES[category_name]["stats"]
        vals = []
        for stat in stats:
            if stat in row and not pd.isna(row[stat]) and stat in pool.columns:
                vals.append(_percentile(pool[stat], float(row[stat])))
        return sum(vals) / len(vals) if vals else 0.0
    weights = cfg["weights"]
    neg = set(cfg.get("negative_stats", []))
    score = 0; total_w = 0
    for stat, w in weights.items():
        if stat not in row or pd.isna(row[stat]) or stat not in pool.columns:
            continue
        pct = _percentile(pool[stat], float(row[stat]), negative=(stat in neg))
        score += w * pct; total_w += w
    return (score / total_w) if total_w > 0 else 0.0


def _panel(ax, x, y, w, h, zorder=1):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=PANEL,
                           edgecolor=PANEL_BR, linewidth=0.6, zorder=zorder))


def _draw_pentagon(ax, values_dict, x, y, radius):
    categories = ["Goalscoring", "Chance Creation", "Dribbling",
                  "Passing", "Defending"]
    angles = np.linspace(np.pi/2, np.pi/2 - 2*np.pi, 5, endpoint=False)

    for r_frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
        pts = [(x + radius * r_frac * np.cos(a),
                y + radius * r_frac * np.sin(a)) for a in angles]
        ax.add_patch(Polygon(pts, closed=True, fill=False,
                              edgecolor=TICK if r_frac < 1.0 else MUTED,
                              linewidth=0.4 if r_frac < 1.0 else 0.6,
                              zorder=3))
    for a in angles:
        ax.plot([x, x + radius * np.cos(a)], [y, y + radius * np.sin(a)],
                color=TICK, linewidth=0.4, zorder=3)

    values = [values_dict.get(c, 0) for c in categories]
    pts = [(x + radius * (v/100) * np.cos(a),
            y + radius * (v/100) * np.sin(a))
            for v, a in zip(values, angles)]
    ax.add_patch(Polygon(pts, closed=True,
                          facecolor=GOLD, alpha=0.20,
                          edgecolor=GOLD, linewidth=1.2, zorder=4))
    for px, py in pts:
        ax.add_patch(Circle((px, py), 2.5, facecolor=GOLD,
                             edgecolor="none", zorder=5))

    label_r = radius + 26
    for cat, a, v in zip(categories, angles, values):
        lx = x + label_r * np.cos(a); ly = y + label_r * np.sin(a)
        ha = "center" if abs(np.cos(a)) < 0.3 else ("left" if np.cos(a) > 0 else "right")
        ax.text(lx, ly + 6, cat.upper(), fontsize=8, color=GOLD,
                family=FONT_MONO, ha=ha, va="bottom")
        ax.text(lx, ly - 4, f"{v:.0f}", fontsize=9, color=GOLD,
                family=FONT_MONO, ha=ha, va="top")


def _draw_bar_row(ax, x, y, w, stat_name, value_str, pct,
                  label_w=240, value_w=70, bar_w=110):
    ax.text(x, y, stat_name, fontsize=9, color=TEXT,
            family=FONT_SANS, va="center")
    value_right_x = x + label_w + value_w
    ax.text(value_right_x, y, value_str, fontsize=9, color=WHITE,
            family=FONT_MONO, va="center", ha="right")
    bar_x = value_right_x + 12
    bar_h = 4
    ax.add_patch(Rectangle((bar_x, y - bar_h/2), bar_w, bar_h,
                           facecolor=TRACK, edgecolor="none", zorder=2))
    fill_w = bar_w * (pct / 100)
    ax.add_patch(Rectangle((bar_x, y - bar_h/2), fill_w, bar_h,
                           facecolor=GOLD, edgecolor="none", zorder=3))
    ax.plot([bar_x + bar_w/2, bar_x + bar_w/2],
            [y - bar_h/2 - 1.5, y + bar_h/2 + 1.5],
            color=TICK, linewidth=0.4, zorder=4)
    ax.text(bar_x + bar_w + 12, y, f"{pct:.0f}", fontsize=9,
            color=GOLD, family=FONT_MONO, va="center")


# ─────────────────────────────────────────────────────────────────────────────
# Main card generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_scout_card(df, player_name, position_label, season_label="2025/26"):
    pool_all = df[df["Main Position"].isin(position_groups[position_label])].copy()
    pool_all = pool_all[pool_all["Minutes played"] >= 600]
    target = pool_all[pool_all["Player"] == player_name]
    if target.empty:
        raise ValueError(f"{player_name} not in {position_label} pool")
    target = target.iloc[0]

    scored = compute_scores(df, position_label, "Top 5 leagues + Next 14")
    scored = scored[scored["Minutes played"] >= 600].copy()
    scored = scored.sort_values("overall_adj", ascending=False).reset_index(drop=True)
    if player_name not in scored["Player"].values:
        scored = compute_scores(df, position_label, "All leagues")
        scored = scored[scored["Minutes played"] >= 600].copy()
        scored = scored.sort_values("overall_adj", ascending=False).reset_index(drop=True)
    s_row = scored[scored["Player"] == player_name]
    overall_score = float(s_row["overall_adj"].iloc[0]) if not s_row.empty else 0.0
    overall_rank = scored[scored["Player"] == player_name].index[0] + 1 \
                   if not s_row.empty else 0
    pool_size = len(scored)

    cat_scores = {cat: _category_score(target, pool_all, cat)
                  for cat in RADAR_CATEGORIES.keys()}

    pool_sim = pool_all[pool_all["Player"] != player_name]
    sim = adjusted_similarity(
        target_row=target, candidates_df=pool_sim,
        sim_stats=ALL_RADAR_STATS, target_league=target.get("League", ""),
        min_minutes=600,
    )
    similar = sim.head(4)

    # Canvas
    W, H = 1240, 1700
    fig = plt.figure(figsize=(W/150, H/150), facecolor=BG, dpi=150)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, facecolor=BG, zorder=0))

    PADDING = 80

    # ── HEADER ───────────────────────────────────────────────────────────────
    pos_short = {
        "Central Midfielder": "CENTRAL MID",
        "Attacking Midfielder": "ATTACKING MID",
        "Defensive Midfielder": "DEFENSIVE MID",
    }.get(position_label, position_label.upper())

    eyebrow_y = 1655
    name_y    = 1605
    bar_top   = 1640
    bar_bot   = 1580
    ax.text(PADDING + 16, eyebrow_y,
            f"{pos_short} REPORT  ·  {season_label}",
            fontsize=9, color=GOLD, family=FONT_MONO, va="center")
    ax.add_patch(Rectangle((PADDING, bar_bot), 3, bar_top - bar_bot,
                           facecolor=GOLD, zorder=3))
    ax.text(PADDING + 16, name_y, player_name, fontsize=28, color=WHITE,
            family=FONT_SANS, va="center")

    age = int(target.get("Age", 0))
    nation = str(target.get("Passport country", "") or target.get("Birth country", ""))
    nation_c = COUNTRY_CODES.get(nation, nation[:3].upper() if nation else "")
    team = str(target.get("Team within selected timeframe", ""))
    pos = str(target.get("Position", "")).split(",")[0].strip()
    foot = str(target.get("Foot", "")).upper()

    # Multi-line team for long names
    team_upper = team.upper()
    if len(team_upper) > 10 and " " in team_upper:
        words = team_upper.split(" ")
        if len(words) == 2:
            team_display = f"{words[0]}\n{words[1]}"
        else:
            mid = len(words) // 2
            team_display = f"{' '.join(words[:mid])}\n{' '.join(words[mid:])}"
    else:
        team_display = team_upper

    meta_blocks = [
        ("AGE", str(age)),
        ("NATION", nation_c),
        ("CLUB  ·  POS", f"{team_display}  ·  {pos}"),
        ("FOOT", foot),
    ]

    # Dynamic block widths
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    block_widths = []
    for label, value in meta_blocks:
        t1 = ax.text(0, -1000, label, fontsize=8, family=FONT_MONO)
        test_text = max(value.split("\n"), key=len) if "\n" in value else value
        t2 = ax.text(0, -1000, test_text, fontsize=11, family=FONT_SANS)
        w1 = t1.get_window_extent(renderer=renderer)
        w2 = t2.get_window_extent(renderer=renderer)
        b1 = w1.transformed(ax.transData.inverted())
        b2 = w2.transformed(ax.transData.inverted())
        block_widths.append(max(b1.x1 - b1.x0, b2.x1 - b2.x0) + 16)
        t1.remove(); t2.remove()

    gap = 36
    total_meta_w = sum(block_widths) + 3 * gap
    meta_start_x = W - PADDING - total_meta_w

    block_xs = []
    cursor = meta_start_x
    for w in block_widths:
        block_xs.append(cursor + w/2)
        cursor += w + gap

    for (label, value), bx, w in zip(meta_blocks, block_xs, block_widths):
        ax.text(bx, 1645, label, fontsize=8, color=MUTED, family=FONT_MONO,
                ha="center", va="center")
        ax.text(bx, 1610, value, fontsize=11, color=WHITE, family=FONT_SANS,
                ha="center", va="center", linespacing=1.1)

    # ── KEY NUMBERS ──────────────────────────────────────────────────────────
    z2_top, z2_bot = 1540, 1410
    nums = [
        ("GAMES",   f"{int(target.get('Matches played', 0))}"),
        ("MINUTES", f"{int(target.get('Minutes played', 0)):,}"),
        ("GOALS",   f"{int(target.get('Goals', 0))}"),
        ("ASSISTS", f"{int(target.get('Assists', 0))}"),
        ("XG",      f"{float(target.get('xG per 90', 0))*int(target.get('Minutes played',0))/90:.1f}"),
        ("XA",      f"{float(target.get('xA per 90', 0))*int(target.get('Minutes played',0))/90:.1f}"),
    ]
    g = 12
    card_w = (W - 2*PADDING - 5*g) / 6
    card_h = z2_top - z2_bot
    for i, (label, val) in enumerate(nums):
        cx = PADDING + i * (card_w + g)
        _panel(ax, cx, z2_bot, card_w, card_h, zorder=2)
        ax.text(cx + 18, z2_top - 22, label, fontsize=8, color=MUTED,
                family=FONT_MONO, va="center")
        ax.text(cx + 18, z2_top - 68, val, fontsize=24, color=WHITE,
                family=FONT_SANS, va="center")

    # ── SCORE + RADAR ────────────────────────────────────────────────────────
    z3_top, z3_bot = 1395, 1075
    panel_gap = 16
    score_w = (W - 2*PADDING - panel_gap) * 0.38
    radar_w = (W - 2*PADDING - panel_gap) - score_w
    score_x = PADDING
    radar_x = PADDING + score_w + panel_gap

    _panel(ax, score_x, z3_bot, score_w, z3_top - z3_bot, zorder=2)
    _panel(ax, radar_x, z3_bot, radar_w, z3_top - z3_bot, zorder=2)

    ax.text(score_x + 24, z3_top - 22, "OVERALL SCORE",
            fontsize=10, color=GOLD, family=FONT_MONO, va="top")

    score_baseline_y = z3_top - 130
    score_text = ax.text(score_x + 24, score_baseline_y,
                          f"{overall_score:.1f}",
                          fontsize=58, color=GOLD,
                          family=FONT_SANS, va="baseline")
    fig.canvas.draw()
    bbox = score_text.get_window_extent(renderer=fig.canvas.get_renderer())
    bbox_data = bbox.transformed(ax.transData.inverted())
    ax.text(bbox_data.x1 + 8, score_baseline_y + 6, "/100",
            fontsize=13, color=MUTED, family=FONT_SANS, va="baseline")

    pbar_x = score_x + 24
    pbar_y = z3_top - 160
    pbar_w = score_w - 48
    pbar_h = 3
    ax.add_patch(Rectangle((pbar_x, pbar_y), pbar_w, pbar_h,
                           facecolor=TRACK, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((pbar_x, pbar_y), pbar_w * (overall_score/100),
                           pbar_h, facecolor=GOLD, edgecolor="none", zorder=4))

    pct_rank = ((pool_size - overall_rank + 1) / pool_size) * 100 if pool_size > 0 else 0
    ax.text(pbar_x, pbar_y - 26, f"#{overall_rank} OF {pool_size:,}",
            fontsize=10, color=MUTED, family=FONT_MONO, va="center")
    ax.text(pbar_x + pbar_w, pbar_y - 26, f"TOP {max(100-pct_rank, 0):.1f}%",
            fontsize=10, color=MUTED, family=FONT_MONO,
            ha="right", va="center")
    ax.text(pbar_x, pbar_y - 56,
            f"POOL: {pool_size:,} {position_label.upper()}S  ·  MIN. 600 MINS",
            fontsize=8, color=MUTED, family=FONT_MONO, va="center")

    ax.text(radar_x + 24, z3_top - 22, "CATEGORY PROFILE",
            fontsize=10, color=GOLD, family=FONT_MONO, va="top")
    ax.text(radar_x + radar_w - 24, z3_top - 22, "PERCENTILE",
            fontsize=9, color=MUTED, family=FONT_MONO,
            ha="right", va="top")

    radar_cx = radar_x + radar_w / 2
    radar_cy = (z3_top + z3_bot) / 2 - 12
    _draw_pentagon(ax, cat_scores, radar_cx, radar_cy, 95)

    # ── PER-90 PANELS ────────────────────────────────────────────────────────
    z4_top, z4_bot = 1060, 540
    left_panel_w = (W - 2*PADDING - panel_gap) * 0.50
    right_panel_w = (W - 2*PADDING - panel_gap) - left_panel_w
    left_x = PADDING
    right_x = PADDING + left_panel_w + panel_gap

    _panel(ax, left_x, z4_bot, left_panel_w, z4_top - z4_bot, zorder=2)
    _panel(ax, right_x, z4_bot, right_panel_w, z4_top - z4_bot, zorder=2)

    left_cats  = ["Goalscoring", "Chance Creation", "Dribbling"]
    right_cats = ["Passing", "Defending"]

    def draw_stat_section(panel_x, panel_w, top_y, cats):
        cursor = top_y - 22
        for cat_name in cats:
            stats = RADAR_CATEGORIES[cat_name]["stats"]
            neg = set(RADAR_CATEGORIES[cat_name].get("negative_stats", []))
            ax.text(panel_x + 24, cursor, cat_name.upper(),
                    fontsize=9, color=GOLD, family=FONT_MONO, va="center")
            cursor -= 26
            for stat in stats:
                if stat not in target.index or pd.isna(target[stat]):
                    continue
                if stat not in pool_all.columns:
                    continue
                value = float(target[stat])
                pct = _percentile(pool_all[stat], value, negative=(stat in neg))
                fmt = STAT_FORMATS.get(stat, "{:.2f}")
                try:
                    value_str = fmt.format(value)
                except Exception:
                    value_str = f"{value:.2f}"
                display_name = SHORT_NAMES.get(stat, stat)
                _draw_bar_row(ax, panel_x + 24, cursor,
                              panel_w - 48, display_name, value_str, pct)
                cursor -= 23
            cursor -= 12

    draw_stat_section(left_x, left_panel_w, z4_top, left_cats)
    draw_stat_section(right_x, right_panel_w, z4_top, right_cats)

    # ── STYLE-MATCHED ────────────────────────────────────────────────────────
    z5_top, z5_bot = 525, 360
    _panel(ax, PADDING, z5_bot, W - 2*PADDING, z5_top - z5_bot, zorder=2)

    ax.text(PADDING + 24, z5_top - 22,
            "STYLE-MATCHED PROFILES  ·  TIER-ADJUSTED COSINE SIMILARITY",
            fontsize=9, color=GOLD, family=FONT_MONO, va="center")

    inner_w = W - 2*PADDING - 48
    item_w = inner_w / 4

    for i, (_, sp) in enumerate(similar.iterrows()):
        ix = PADDING + 24 + i * item_w
        adj_sim = float(sp.get("adjusted_sim", 0)) * 100

        ax.text(ix, z5_top - 60, f"{adj_sim:.0f}%", fontsize=20,
                color=GOLD, family=FONT_MONO, va="center")

        sp_name = str(sp["Player"])
        if len(sp_name) > 22: sp_name = sp_name[:21] + "…"
        ax.text(ix, z5_top - 92, sp_name, fontsize=11,
                color=WHITE, family=FONT_SANS, va="center")

        team_str = str(sp.get("Team within selected timeframe", ""))
        if len(team_str) > 14: team_str = team_str[:13] + "…"
        meta = f"{team_str}  ·  {int(sp.get('Age', 0))}y"
        ax.text(ix, z5_top - 112, meta,
                fontsize=8.5, color=MUTED, family=FONT_MONO, va="center")

    # ── FOOTER ───────────────────────────────────────────────────────────────
    from datetime import datetime
    today = datetime.now().strftime("%d %b %Y").upper()
    ax.add_patch(Rectangle((PADDING, 90), W - 2*PADDING, 0.4,
                           facecolor=PANEL_BR, zorder=3))
    ax.text(PADDING, 60, "TARGET SCOUTING", fontsize=10, color=GOLD,
            family=FONT_MONO, va="center")
    ax.text(PADDING + 230, 60, f"WYSCOUT  ·  {today}",
            fontsize=9, color=MUTED, family=FONT_MONO, va="center")
    ax.text(PADDING + 530, 60, "@TARGET_SCOUTING",
            fontsize=9, color=MUTED, family=FONT_MONO, va="center")
    ax.text(W - PADDING, 60, f"{pos_short} TEMPLATE",
            fontsize=9, color=GOLD, family=FONT_MONO,
            ha="right", va="center")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, facecolor=BG, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf, overall_score, overall_rank, pool_size


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## Scout Card")
st.caption(
    "Generate a downloadable, print-ready Target Scouting report card. "
    "Black/gold dashboard layout, all 21 RADAR_CATEGORIES stats, "
    "Player Card overall score, and tier-adjusted style matches."
)

# Season filter
season_label = render_season_filter()

# Load data
df = load_season_data(season_label, 0)
df["Main Position"] = df["Position"].astype(str).str.split(",").str[0].str.strip()

# Selectors
col1, col2 = st.columns([1, 2])
with col1:
    position_label = st.selectbox(
        "Position",
        list(position_groups.keys()),
        index=list(position_groups.keys()).index("Winger") if "Winger" in position_groups else 0,
    )

# Filter players for this position with min minutes
pool = df[df["Main Position"].isin(position_groups[position_label])].copy()
pool = pool[pool["Minutes played"] >= 600]
pool = pool.sort_values(["League", "Minutes played"], ascending=[True, False])
players = pool["Player"].dropna().unique().tolist()

with col2:
    if not players:
        st.warning(f"No {position_label}s with ≥600 minutes found in {season_label}.")
        st.stop()
    player_name = st.selectbox(
        "Player",
        players,
        index=0,
    )

st.markdown("---")

# Generate + preview
try:
    with st.spinner("Generating scout card..."):
        buf, score, rank, pool_size = generate_scout_card(
            df, player_name, position_label, season_label=season_label
        )

    # Preview
    st.image(buf.getvalue(), use_container_width=True)

    # Stats summary line
    st.caption(
        f"**{player_name}** · {position_label} · Overall {score:.1f} "
        f"(#{rank} of {pool_size:,} · {season_label})"
    )

    # Download
    safe_name = player_name.replace(" ", "_").replace(".", "").replace("/", "-")
    filename = f"target_scouting_{safe_name}_{position_label.replace(' ', '_').lower()}.png"
    st.download_button(
        label="Download PNG",
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png",
        type="primary",
    )

except Exception as e:
    st.error(f"Failed to generate scout card: {e}")
    st.exception(e)
