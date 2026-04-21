import importlib
import shared.templates
importlib.reload(shared.templates)

import io
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from shared.templates import (
    template_config, role_config,
    TOP5_LEAGUES, NEXT14_LEAGUES,
    RADAR_CATEGORIES, ALL_RADAR_STATS,
    STAT_CATEGORY_COLORS, STAT_TO_CATEGORY,
    LEAGUE_DISPLAY_NAMES,
)

# ── Logo ──────────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(__file__)
_LOGO_PATH = os.path.join(_BASE, "target_scouting_black.png")
try:
    TS_LOGO = Image.open(_LOGO_PATH)
except FileNotFoundError:
    TS_LOGO = None

# ── Colors ────────────────────────────────────────────────────────────────────
BG_COLOR    = "#faf7f2"   # Variant D off-white
DARK_TEXT   = "#111827"
GRAY_TEXT   = "#7a7060"
AVG_COLOR   = "#c9a84c"   # TS gold — pool average line
BENCH_COLOR = "#aaaaaa"

# Minimum stats per category to show a category arc + label
MIN_STATS_FOR_CATEGORY_LABEL = 3


# ── Percentile color ──────────────────────────────────────────────────────────

def _percentile_color(v):
    if v >= 90:   return "#0d9313"
    elif v >= 70: return "#3ebe43"
    elif v >= 50: return "#f39c12"
    elif v >= 25: return "#e67e22"
    else:         return "#e74c3c"


# ── Short labels (keyed on original stat name) ────────────────────────────────

STAT_SHORT_LABELS = {
    "Passing accuracy (prog/1/3/forw)":        "Prog/F3/forw\npass acc.",
    "PAdj Defensive duels won per 90":          "Def duels won\n/90",
    "PAdj Aerial duels won per 90":             "Aerial duels won\n/90",
    "PAdj Interceptions":                       "Interceptions",
    "Defensive duels won, %":                   "Def duels\nwon %",
    "Aerial duels won, %":                      "Aerial duels\nwon %",
    "Successful dribbles per received pass":    "Drib.\n/rec pass",
    "Successful dribbles, %":                   "Drib. %",
    "Offensive duels won, %":                   "Off. duels\n won, %",
    "Progressive runs per received pass":       "Prog. runs\n/rec pass",
    "Completed progressive passes per 90":      "Prog. passes\n/90",
    "Completed passes to final third per 90":   "Passes F3\n/90",
    "Deep completions per 90":                  "Deep compl.\n/90",
    "Accurate crosses per received pass":       "Crosses\n/rec pass",
    "Shot assists per 90":                      "Shot ast.\n/90",
    "Key passes per pass":                      "Key passes\n/pass",
    "Touches in box per 90":                    "Box touches\n/90",
    "xG per shot":                              "xG/shot",
    "xG per 90":                                "xG/90",
    "xA per 90":                                "xA/90",
    "Finishing":                                "Finishing",
    "Accurate crosses, %":                      "Crosses %",
    "Ball progression through passing":         "Ball prog.\npassing",
    "PAdj Successful defensive actions per 90": "Succ. def. act.\n/90",
    "Through passes per pass":                  "Through passes\n/pass",
    "Non-penalty goals per 90":                 "NP goals\n/90",
    "Shots per 90":                             "Shots/90",
    "Shots on target, %":                       "Shots on\ntarget %",
    "Assists per 90":                           "Assists/90",
    "Accurate progressive passes, %":           "Prog. passes %",
    "Completed passes to penalty area per 90":  "Passes PA\n/90",
    "Accurate passes to final third, %":        "Passes F3 %",
    "Fouls per 90":                             "Fouls/90",
}

def _stat_short_label(stat: str) -> str:
    return STAT_SHORT_LABELS.get(stat, stat)


# ── Rotation ──────────────────────────────────────────────────────────────────

def _label_rotation(angle_rad: float, flip: bool) -> float:
    deg = np.degrees(angle_rad) % 360
    rot = deg - 90
    if flip:
        rot += 180
    return rot


def _compute_flip_set(stats: list) -> set:
    n      = len(stats)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    flip   = set()
    for angle, stat in zip(angles, stats):
        deg = np.degrees(angle) % 360
        if deg > 180:
            flip.add(stat)
    return flip


def _cat_label_rotation(mid_rad: float) -> float:
    """
    Tangential rotation for category arc labels on a polar axis.
    Labels run along the arc direction and are always readable.
    The +180 corrects for polar axis orientation (theta_zero=N, clockwise).
    """
    mid_deg = np.degrees(mid_rad) % 360
    rot = mid_deg + 90 + 180   # +180 corrects for polar coordinate orientation
    if 180 < mid_deg <= 360:
        rot += 180
    return rot % 360


# ── Pool / percentile helpers ─────────────────────────────────────────────────

def _compute_percentiles(pool, player_name, player_team, stats, neg_stats=None):
    available = [s for s in stats if s in pool.columns]
    pct = pool[available].rank(pct=True) * 100
    for s in (neg_stats or []):
        if s in pct.columns:
            pct[s] = 100 - pct[s]
    mask = (
        (pool["Player"] == player_name) &
        (pool["Team within selected timeframe"] == player_team)
    )
    if not mask.any():
        raise ValueError(f"Player '{player_name}' not found in pool.")
    return pct[mask].iloc[0]


def _compute_pool_averages(pool, stats, neg_stats=None):
    available = [s for s in stats if s in pool.columns]
    pct = pool[available].rank(pct=True) * 100
    for s in (neg_stats or []):
        if s in pct.columns:
            pct[s] = 100 - pct[s]
    return pct.mean()


def _build_pool(data, template_key, percentile_basis, player_league=None):
    pool = data.copy()
    pool["Main Position"] = pool["Position"].astype(str).str.split(",").str[0].str.strip()
    positions = template_config[template_key]["positions"]
    pool = pool[pool["Main Position"].isin(positions)]
    if percentile_basis == "T5 only":
        pool = pool[pool["League"].isin(TOP5_LEAGUES)]
    elif percentile_basis == "Next 14 only":
        pool = pool[pool["League"].isin(NEXT14_LEAGUES)]
    elif percentile_basis == "Own league" and player_league:
        pool = pool[pool["League"] == player_league]
    return pool


# ── Legend ────────────────────────────────────────────────────────────────────

def _add_legend(fig, rect=(0.00, 0.022, 0.85, 0.07)):
    legend_items = [
        (">= 90\n(Top 10%)",           "#0d9313"),
        ("70 – 89\n(Well above avg)",  "#3ebe43"),
        ("50 – 69\n(Above average)",   "#f39c12"),
        ("25 – 49\n(Below average)",   "#e67e22"),
        ("< 25\n(Bottom 25%)",         "#e74c3c"),
    ]
    ax_leg = fig.add_axes(rect)
    ax_leg.set_axis_off()
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    n, w, gap, start_x = len(legend_items), 0.10, 0.07, 0.02
    for i, (label, color) in enumerate(legend_items):
        x = start_x + i * (w + gap)
        ax_leg.add_patch(mpatches.FancyBboxPatch(
            (x, 0.05), w, 0.85,
            boxstyle="round,pad=0.02",
            facecolor=color, alpha=0.35,
            edgecolor=color, linewidth=1.2,
        ))
        ax_leg.text(x + w / 2, 0.52, label,
                    ha="center", va="center",
                    fontsize=7.5, color=DARK_TEXT, fontweight="bold")


# ── Core draw ─────────────────────────────────────────────────────────────────

def _draw_radar(
    ax, stats, angles, values,
    flip_stats,
    avg_values=None,
    benchmark_values=None,
    show_avg=True,
    bar_width_ratio=0.75,
    label_fontsize=8.5,
    show_category_labels=True,
):
    n         = len(stats)
    bar_width = (2 * np.pi / n) * bar_width_ratio
    half_w    = bar_width / 2

    ARC_R  = 108
    CAT_R  = 120
    STAT_R = 136

    # ── Benchmark (grey background bars) ─────────────────────────────────────
    if benchmark_values:
        for angle, bval in zip(angles, benchmark_values):
            ax.bar(angle, bval, width=bar_width, bottom=0,
                   color=BENCH_COLOR, alpha=0.25, edgecolor="none", zorder=1)

    # ── Main bars ────────────────────────────────────────────────────────────
    for angle, value, stat in zip(angles, values, stats):
        ax.bar(angle, value, width=bar_width, bottom=0,
               color=_percentile_color(value),
               alpha=0.55, edgecolor="black", linewidth=0.6, zorder=2)
        if value > 6:
            ax.text(angle, value / 1.18, str(int(round(value))),
                    ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold",
                    bbox=dict(facecolor="black", edgecolor="none",
                              boxstyle="round,pad=0.25"),
                    zorder=3)

    # ── Pool average dotted line ──────────────────────────────────────────────
    if show_avg and avg_values is not None:
        avg_closed = list(avg_values) + [avg_values[0]]
        ang_closed = angles + [angles[0]]
        ax.plot(ang_closed, avg_closed,
                color=AVG_COLOR, linewidth=1.2,
                linestyle="--", zorder=4, alpha=0.7)

    # ── Category arcs + labels ────────────────────────────────────────────────
    cat_angle_map = {}
    for angle, stat in zip(angles, stats):
        cat = STAT_TO_CATEGORY.get(stat, "Misc")
        cat_angle_map.setdefault(cat, []).append(angle)

    for cat, cat_angles in cat_angle_map.items():
        show_label = show_category_labels and len(cat_angles) >= MIN_STATS_FOR_CATEGORY_LABEL

        cat_color = RADAR_CATEGORIES.get(cat, {}).get("color", "#888888")
        a_start   = min(cat_angles) - half_w * 0.9
        a_end     = max(cat_angles) + half_w * 0.9
        arc       = np.linspace(a_start, a_end, 80)

        ax.plot(arc, [ARC_R] * 80,
                color=cat_color, linewidth=6,
                solid_capstyle="round", zorder=5)

        if show_label:
            mid_rad = (a_start + a_end) / 2
            rot     = _cat_label_rotation(mid_rad)
            ax.text(mid_rad, CAT_R, cat.upper(),
                    ha="center", va="center",
                    fontsize=label_fontsize - 1.5, fontweight="bold",
                    color=cat_color,
                    rotation=rot, rotation_mode="anchor", zorder=6)

    # ── Stat labels ───────────────────────────────────────────────────────────
    for angle, stat in zip(angles, stats):
        label = _stat_short_label(stat)
        flip  = stat in flip_stats
        rot   = _label_rotation(angle, flip)
        ax.text(angle, STAT_R, label,
                ha="center", va="center",
                fontsize=label_fontsize, color=DARK_TEXT,
                rotation=rot, rotation_mode="anchor",
                linespacing=1.4, zorder=6)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 158)
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor(BG_COLOR)


# ── Figure helpers ────────────────────────────────────────────────────────────

def _make_figure():
    fig = plt.figure(figsize=(11, 12.5))
    fig.patch.set_facecolor(BG_COLOR)
    return fig


def _add_header(fig, player_name, age, team, pos, mins, foot, league, pool_label):
    fig.text(0.5, 0.965, f"{player_name}  ({age})  —  {team}",
             ha="center", fontsize=18, fontweight="bold", color=DARK_TEXT)
    fig.text(0.5, 0.946,
             f"{pos}  |  {mins} mins  |  {foot}  |  {league}  |  Percentile basis: {pool_label}",
             ha="center", fontsize=8.5, color=GRAY_TEXT)


def _add_footer(fig):
    fig.text(0.01, 0.007, "Data: Wyscout (26/03/2026)", fontsize=7.5, color=GRAY_TEXT)
    if TS_LOGO:
        ax_logo = fig.add_axes([0.865, 0.012, 0.09, 0.065])
        ax_logo.imshow(TS_LOGO)
        ax_logo.axis("off")


# ── Export helpers ────────────────────────────────────────────────────────────

def _fig_to_bytes(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


def _crop_circle_only(fig, ax, dpi=180):
    fig.canvas.draw()
    buf = io.BytesIO()
    extent = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    margin = 40
    bbox = matplotlib.transforms.Bbox([
        [(extent.x0 - margin) / fig.dpi,
         (extent.y0 - margin) / fig.dpi],
        [(extent.x1 + margin) / fig.dpi,
         (extent.y1 + margin) / fig.dpi],
    ])
    fig.savefig(buf, format="png", dpi=dpi,
                bbox_inches=bbox, facecolor="none", transparent=True)
    buf.seek(0)
    return buf.read()


def export_full(fig, dpi=180):
    return _fig_to_bytes(fig, dpi=dpi)


def export_circle(fig, ax, dpi=180):
    if ax is None:
        return _fig_to_bytes(fig, dpi=dpi)
    return _crop_circle_only(fig, ax, dpi=dpi)


# ── Stats / pool builder ──────────────────────────────────────────────────────

def _resolve_stats(radar_type, template_key, role_name):
    """Returns (stats, neg_stats)."""
    if radar_type == "Universal Radar":
        stats     = ALL_RADAR_STATS
        neg_stats = []
        for cat_data in RADAR_CATEGORIES.values():
            neg_stats.extend(cat_data.get("negative_stats", []))
    elif radar_type == "Position Template":
        cfg       = template_config[template_key]
        stats     = cfg["stats"]
        neg_stats = cfg.get("negative_stats", [])
    elif radar_type == "Role Radar":
        if role_name is None:
            raise ValueError("role_name required for Role Radar.")
        role_cfg  = role_config[template_key][role_name]
        stats     = list(role_cfg["stats"].keys())
        neg_stats = []
    else:
        raise ValueError(f"Unknown radar_type: {radar_type}")
    return stats, neg_stats


# ── Public API ────────────────────────────────────────────────────────────────

def create_radar(
    data,
    player_name,
    player_team,
    template_key,
    percentile_basis="T5 + Next 14",
    radar_type="Universal Radar",
    role_name=None,
    show_avg=True,
    benchmark_player=None,
    compact=False,
):
    stats, neg_stats = _resolve_stats(radar_type, template_key, role_name)
    angles     = np.linspace(0, 2 * np.pi, len(stats), endpoint=False).tolist()
    flip_stats = _compute_flip_set(stats)

    show_cat_labels = (radar_type == "Universal Radar")

    player_row = data[
        (data["Player"] == player_name) &
        (data["Team within selected timeframe"] == player_team)
    ]
    if player_row.empty:
        raise ValueError(f"Player '{player_name}' not found.")
    player_row    = player_row.iloc[0]
    player_league = player_row.get("League", None)

    pool   = _build_pool(data, template_key, percentile_basis, player_league)
    pct    = _compute_percentiles(pool, player_name, player_team, stats, neg_stats)
    values = [float(pct.get(s, 0)) for s in stats]

    avg_values = None
    if show_avg:
        avg_pct    = _compute_pool_averages(pool, stats, neg_stats)
        avg_values = [float(avg_pct.get(s, 50)) for s in stats]

    bench_values = None
    if benchmark_player:
        bp_name, bp_team = benchmark_player
        try:
            bp_pct       = _compute_percentiles(pool, bp_name, bp_team, stats, neg_stats)
            bench_values = [float(bp_pct.get(s, 0)) for s in stats]
        except ValueError:
            bench_values = None

    fig = _make_figure()
    ax  = fig.add_axes([0.01, 0.12, 0.98, 0.78], projection="polar")

    _draw_radar(
        ax, stats, angles, values,
        flip_stats=flip_stats,
        avg_values=avg_values,
        benchmark_values=bench_values,
        show_avg=show_avg,
        label_fontsize=8.5,
        show_category_labels=show_cat_labels,
    )

    pos    = str(player_row.get("Position", "")).split(",")[0].strip()
    age    = player_row.get("Age", "")
    mins   = player_row.get("Minutes played", "")
    foot   = player_row.get("Foot", "")
    league = LEAGUE_DISPLAY_NAMES.get(player_league, player_league or "")
    type_label = radar_type if radar_type != "Role Radar" else f"Role: {role_name}"

    if not compact:
        _add_header(fig, player_name, age, player_team, pos, mins, foot, league,
                    f"{percentile_basis} | {type_label}")
        _add_legend(fig, rect=(0.00, 0.022, 0.85, 0.07))
        _add_footer(fig)
        if show_avg:
            fig.text(0.01, 0.093, "— — Pool average (percentile)",
                     fontsize=7.5, color=AVG_COLOR)
        if bench_values and benchmark_player:
            fig.text(0.01, 0.105,
                     f"░ Benchmark: {benchmark_player[0]} ({benchmark_player[1]})",
                     fontsize=7.5, color=BENCH_COLOR)

    return fig, ax


def create_radar_compact(
    data,
    player_name,
    player_team,
    template_key,
    percentile_basis="T5 only",
    radar_type="Universal Radar",
    role_name=None,
    show_avg=True,
):
    """
    Compact radar for report export: circle + stat labels + category arcs only.
    No header, no legend, no footer, no avg text. Returns fig, ax.
    """
    stats, neg_stats = _resolve_stats(radar_type, template_key, role_name)
    angles     = np.linspace(0, 2 * np.pi, len(stats), endpoint=False).tolist()
    flip_stats = _compute_flip_set(stats)
    show_cat_labels = (radar_type == "Universal Radar")

    player_row = data[
        (data["Player"] == player_name) &
        (data["Team within selected timeframe"] == player_team)
    ]
    if player_row.empty:
        raise ValueError(f"Player '{player_name}' not found.")
    player_row    = player_row.iloc[0]
    player_league = player_row.get("League", None)

    pool   = _build_pool(data, template_key, percentile_basis, player_league)
    pct    = _compute_percentiles(pool, player_name, player_team, stats, neg_stats)
    values = [float(pct.get(s, 0)) for s in stats]

    avg_values = None
    if show_avg:
        avg_pct    = _compute_pool_averages(pool, stats, neg_stats)
        avg_values = [float(avg_pct.get(s, 50)) for s in stats]

    # Compact figure — tighter, circle fills the whole canvas
    fig = plt.figure(figsize=(7, 7))
    fig.patch.set_facecolor(BG_COLOR)
    # Circle fills almost the full figure
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], projection="polar")

    _draw_radar(
        ax, stats, angles, values,
        flip_stats=flip_stats,
        avg_values=avg_values,
        show_avg=show_avg,
        label_fontsize=7.5,
        show_category_labels=show_cat_labels,
    )

    return fig, ax


def create_comparison_radar(
    data,
    player1_name, player1_team,
    player2_name, player2_team,
    template_key,
    percentile_basis="T5 + Next 14",
    radar_type="Universal Radar",
    role_name=None,
    mode="side_by_side",
    show_avg=True,
    compact=False,
):
    stats, neg_stats = _resolve_stats(radar_type, template_key, role_name)
    angles     = np.linspace(0, 2 * np.pi, len(stats), endpoint=False).tolist()
    flip_stats = _compute_flip_set(stats)
    show_cat_labels = (radar_type == "Universal Radar")

    COLOR1, COLOR2 = "#2980b9", "#c0392b"

    def _get_row(name, team):
        row = data[(data["Player"] == name) &
                   (data["Team within selected timeframe"] == team)]
        if row.empty:
            raise ValueError(f"Player '{name}' not found.")
        return row.iloc[0]

    p1 = _get_row(player1_name, player1_team)
    p2 = _get_row(player2_name, player2_team)

    pool1  = _build_pool(data, template_key, percentile_basis, p1.get("League"))
    pool2  = _build_pool(data, template_key, percentile_basis, p2.get("League"))
    vals1  = [float(_compute_percentiles(pool1, player1_name, player1_team, stats, neg_stats).get(s, 0)) for s in stats]
    vals2  = [float(_compute_percentiles(pool2, player2_name, player2_team, stats, neg_stats).get(s, 0)) for s in stats]

    avg1 = avg2 = None
    if show_avg:
        a1   = _compute_pool_averages(pool1, stats, neg_stats)
        a2   = _compute_pool_averages(pool2, stats, neg_stats)
        avg1 = [float(a1.get(s, 50)) for s in stats]
        avg2 = [float(a2.get(s, 50)) for s in stats]

    p1_disp = LEAGUE_DISPLAY_NAMES.get(p1.get("League"), p1.get("League", ""))
    p2_disp = LEAGUE_DISPLAY_NAMES.get(p2.get("League"), p2.get("League", ""))

    # ── Side by side ──────────────────────────────────────────────────────────
    if mode == "side_by_side":
        fig, axes = plt.subplots(1, 2, figsize=(22, 12.5),
                                 subplot_kw={"projection": "polar"})
        fig.patch.set_facecolor(BG_COLOR)

        for ax, vals, name, team, row, avg_v in [
            (axes[0], vals1, player1_name, player1_team, p1, avg1),
            (axes[1], vals2, player2_name, player2_team, p2, avg2),
        ]:
            _draw_radar(ax, stats, angles, vals,
                        flip_stats=flip_stats,
                        avg_values=avg_v,
                        show_avg=show_avg,
                        label_fontsize=8.5,
                        show_category_labels=show_cat_labels)
            pos   = str(row.get("Position", "")).split(",")[0].strip()
            age   = row.get("Age", "")
            mins  = row.get("Minutes played", "")
            ldisp = LEAGUE_DISPLAY_NAMES.get(row.get("League"), row.get("League", ""))
            ax.set_title(
                f"{name}  ({age})  —  {team}\n{pos}  |  {mins} mins  |  {ldisp}",
                fontsize=12, fontweight="bold", color=DARK_TEXT, pad=75,
            )

        if not compact:
            fig.suptitle("Player Comparison", fontsize=20,
                         fontweight="bold", color=DARK_TEXT, y=0.98)
            _add_legend(fig, rect=(0.20, 0.022, 0.55, 0.065))
            fig.text(0.02, 0.007, "Data: Wyscout (26/03/2026)", fontsize=7.5, color=GRAY_TEXT)
            if TS_LOGO:
                ax_logo = fig.add_axes([0.91, 0.010, 0.06, 0.055])
                ax_logo.imshow(TS_LOGO)
                ax_logo.axis("off")
        return fig, None

    # ── Overlay ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 13.5))
    fig.patch.set_facecolor(BG_COLOR)

    if not compact:
        fig.text(0.5, 0.972, "Player Comparison — Overlay",
                 ha="center", fontsize=18, fontweight="bold", color=DARK_TEXT)
        patch1 = mpatches.Patch(color=COLOR1, alpha=0.8,
            label=f"{player1_name} ({p1.get('Age')}) — {player1_team} | {p1_disp}")
        patch2 = mpatches.Patch(color=COLOR2, alpha=0.8,
            label=f"{player2_name} ({p2.get('Age')}) — {player2_team} | {p2_disp}")
        fig.legend(
            handles=[patch1, patch2],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            fontsize=9, ncol=2,
            frameon=True, facecolor=BG_COLOR, edgecolor="#e0c8be",
        )

    ax = fig.add_axes([0.01, 0.11, 0.98, 0.80], projection="polar")
    ax.set_facecolor(BG_COLOR)

    n      = len(stats)
    slot_w = (2 * np.pi / n)
    bar_w  = slot_w * 0.72
    sub_w  = bar_w * 0.48
    gap    = bar_w * 0.04
    offset = sub_w / 2 + gap / 2
    half_w = bar_w / 2

    for i, (angle, v1, v2) in enumerate(zip(angles, vals1, vals2)):
        ax.bar(angle - offset, v1, width=sub_w, bottom=0, color=COLOR1, alpha=0.65,
               edgecolor="black", linewidth=0.5, zorder=2)
        ax.bar(angle + offset, v2, width=sub_w, bottom=0, color=COLOR2, alpha=0.65,
               edgecolor="black", linewidth=0.5, zorder=2)
        if v1 > 6:
            ax.text(angle - offset, v1 / 1.18, str(int(round(v1))),
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold",
                    bbox=dict(facecolor="black", edgecolor="none", boxstyle="round,pad=0.2"), zorder=3)
        if v2 > 6:
            ax.text(angle + offset, v2 / 1.18, str(int(round(v2))),
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold",
                    bbox=dict(facecolor="black", edgecolor="none", boxstyle="round,pad=0.2"), zorder=3)

    if show_avg and avg1 and avg2:
        ang_c = angles + [angles[0]]
        ax.plot(ang_c, avg1 + [avg1[0]], color=COLOR1, linewidth=1.0, linestyle="--", alpha=0.6, zorder=4)
        ax.plot(ang_c, avg2 + [avg2[0]], color=COLOR2, linewidth=1.0, linestyle="--", alpha=0.6, zorder=4)

    ARC_R, CAT_R, STAT_R = 108, 120, 136
    cat_angle_map = {}
    for angle, stat in zip(angles, stats):
        cat = STAT_TO_CATEGORY.get(stat, "Misc")
        cat_angle_map.setdefault(cat, []).append(angle)

    for cat, cat_angles in cat_angle_map.items():
        show_label = show_cat_labels and len(cat_angles) >= MIN_STATS_FOR_CATEGORY_LABEL
        cat_color  = RADAR_CATEGORIES.get(cat, {}).get("color", "#888")
        a_start    = min(cat_angles) - half_w * 0.9
        a_end      = max(cat_angles) + half_w * 0.9
        arc        = np.linspace(a_start, a_end, 80)
        ax.plot(arc, [ARC_R] * 80, color=cat_color, linewidth=6,
                solid_capstyle="round", zorder=5)
        if show_label:
            mid_rad = (a_start + a_end) / 2
            rot     = _cat_label_rotation(mid_rad)
            ax.text(mid_rad, CAT_R, cat.upper(),
                    ha="center", va="center",
                    fontsize=7, fontweight="bold", color=cat_color,
                    rotation=rot, rotation_mode="anchor", zorder=6)

    for angle, stat in zip(angles, stats):
        label = _stat_short_label(stat)
        flip  = stat in flip_stats
        rot   = _label_rotation(angle, flip)
        ax.text(angle, STAT_R, label,
                ha="center", va="center",
                fontsize=8.5, color=DARK_TEXT,
                rotation=rot, rotation_mode="anchor",
                linespacing=1.4, zorder=6)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 158)
    ax.spines["polar"].set_visible(False)

    if not compact:
        _add_legend(fig, rect=(0.00, 0.022, 0.85, 0.065))
        fig.text(0.01, 0.007, "Data: Wyscout (26/03/2026)", fontsize=7.5, color=GRAY_TEXT)
        if TS_LOGO:
            ax_logo = fig.add_axes([0.865, 0.012, 0.09, 0.060])
            ax_logo.imshow(TS_LOGO)
            ax_logo.axis("off")

    return fig, ax
