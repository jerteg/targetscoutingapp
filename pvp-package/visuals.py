"""
visuals.py — Visualisaties voor PVP model.

Radar charts voor enkele spelers of vergelijkingen, en scatter plots voor
shortlist-analyse.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


# Kleuren: palette dat leesbaar is voor voetbal-analyses
COLOR_MAIN    = "#1f77b4"
COLOR_COMPARE = "#d62728"
COLOR_REF     = "#888888"


def player_radar(model, player_name, comparison=None, out_path=None, figsize=(8, 8)):
    """Radar chart voor één speler, optioneel met vergelijking.

    Args:
        model: PVPModel instance
        player_name: string, substring match
        comparison: optionele lijst van andere spelernamen om te vergelijken
    """
    def _get(name):
        m = model.df[model.df['Player'].str.contains(name, case=False, na=False)]
        if len(m) == 0:
            raise ValueError(f"Speler '{name}' niet gevonden")
        return m.iloc[0]

    p = _get(player_name)

    # Metrics voor radar — we nemen wat meer dimensies dan alleen de 3 z-scores
    # om een rijker beeld te geven. Alle op 0-100 percentile schaal binnen positiegroep.
    raw_metrics = [
        ('Progression',       'PROG'),
        ('Creation',          'CREATE'),
        ('Defense',           'DEF'),
        ('xG/90',             'xG per 90'),
        ('xA/90',             'xA per 90'),
        ('Dribble succ.',     None),  # custom
        ('Pass accuracy',     'Accurate passes, %'),
        ('Duels won %',       'Duels won, %'),
    ]

    def percentile_within_group(col, value, pos_group):
        """Return 0-100 percentile van value binnen posgroep voor col."""
        from pvp import to_num
        pool = model.df[model.df['PosGroup'] == pos_group]
        vals = to_num(pool[col]) if isinstance(col, str) else col
        return (vals < value).sum() / len(vals) * 100

    def _value(player, label, col):
        from pvp import to_num, safe
        if label == 'Dribble succ.':
            # kwantiteit × kwaliteit
            vol = safe(model.df[model.df['PosGroup'] == player['PosGroup']]['Dribbles per 90'])
            qua = safe(model.df[model.df['PosGroup'] == player['PosGroup']]['Successful dribbles, %']) / 100
            pool_score = vol * qua
            player_score = safe_scalar(player['Dribbles per 90']) * safe_scalar(player['Successful dribbles, %']) / 100
            return (pool_score < player_score).sum() / len(pool_score) * 100
        pool = model.df[model.df['PosGroup'] == player['PosGroup']]
        vals = to_num(pool[col])
        v = safe_scalar(player[col])
        return (vals < v).sum() / len(vals) * 100

    # Setup radar
    labels = [m[0] for m in raw_metrics]
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Grid
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(['20', '40', '60', '80'], fontsize=7, color='gray')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(color='lightgray', linewidth=0.5)
    ax.spines['polar'].set_color('lightgray')

    def plot_player(player, color, label):
        vals = [_value(player, m[0], m[1]) for m in raw_metrics]
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=2, label=label)
        ax.fill(angles, vals, color=color, alpha=0.18)

    plot_player(p, COLOR_MAIN,
                f"{p['Player']} ({p['Team']}, {p['Age']}y)")

    if comparison:
        if isinstance(comparison, str):
            comparison = [comparison]
        colors = [COLOR_COMPARE, "#2ca02c", "#9467bd"]
        for i, cname in enumerate(comparison[:3]):
            try:
                c = _get(cname)
                plot_player(c, colors[i],
                            f"{c['Player']} ({c['Team']}, {c['Age']}y)")
            except ValueError as e:
                print(f"Waarschuwing: {e}")

    title = f"PVP profile — {p['Player']}"
    if comparison:
        title += " vs. " + ", ".join(comparison if isinstance(comparison, list) else [comparison])
    plt.title(title, y=1.08, fontsize=13, fontweight='bold')

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    # Caption
    fig.text(0.5, 0.02,
             f"Alle assen: percentiel binnen positiegroep '{p['PosGroup']}' "
             f"(n={(model.df['PosGroup']==p['PosGroup']).sum()}). "
             f"Hoger = beter.",
             ha='center', fontsize=8, color='gray', style='italic')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Opgeslagen: {out_path}")
    return fig


def safe_scalar(v):
    """Convert single value to float, handling European decimals."""
    if isinstance(v, (int, float)):
        return float(v) if not np.isnan(v) else 0.0
    try:
        return float(str(v).replace(',', '.'))
    except (ValueError, TypeError):
        return 0.0


def shortlist_scatter(model, position_group, x_component='PROG_z',
                      y_component='CREATE_z', max_age=None,
                      min_minutes=1000, highlight=None, out_path=None,
                      figsize=(12, 9)):
    """Scatter plot van spelers in een positiegroep op 2 componenten.

    Nuttig voor het visueel spotten van outliers: bijv. alle CM's op
    PROG vs CREATE, met je targets highlighted.
    """
    pool = model.df[
        (model.df['PosGroup'] == position_group) &
        (model.df['Minutes played'] >= min_minutes)
    ].copy()
    if max_age:
        pool = pool[pool['Age'] <= max_age]

    fig, ax = plt.subplots(figsize=figsize)

    # Background scatter
    ax.scatter(pool[x_component], pool[y_component],
               s=30, alpha=0.3, color='lightgray',
               edgecolors='none', label=f'Andere {position_group}s')

    # Quadrant lijnen
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Color gradient op PVP
    top = pool.nlargest(20, 'PVP')
    sc = ax.scatter(top[x_component], top[y_component],
                    s=90, c=top['PVP'], cmap='viridis',
                    edgecolors='black', linewidth=0.5, zorder=3,
                    label='Top 20 op PVP')

    # Labels voor top 20
    for _, row in top.iterrows():
        ax.annotate(row['Player'],
                    xy=(row[x_component], row[y_component]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.9)

    # Highlights
    if highlight:
        if isinstance(highlight, str):
            highlight = [highlight]
        for name in highlight:
            m = pool[pool['Player'].str.contains(name, case=False, na=False)]
            if len(m) > 0:
                r = m.iloc[0]
                ax.scatter(r[x_component], r[y_component],
                           s=250, marker='*', color='red',
                           edgecolors='black', linewidth=1, zorder=4)
                ax.annotate(f"★ {r['Player']}",
                            xy=(r[x_component], r[y_component]),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=10, fontweight='bold', color='red')

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[1]*0.95, ylim[1]*0.95, "High / High", fontsize=10,
            color='#2ca02c', style='italic', ha='right', va='top')
    ax.text(xlim[0]*0.95, ylim[1]*0.95, "Low / High", fontsize=10,
            color='gray', style='italic', ha='left', va='top')
    ax.text(xlim[1]*0.95, ylim[0]*0.95, "High / Low", fontsize=10,
            color='gray', style='italic', ha='right', va='bottom')
    ax.text(xlim[0]*0.95, ylim[0]*0.95, "Low / Low", fontsize=10,
            color='#d62728', style='italic', ha='left', va='bottom')

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label('PVP (league-adjusted)', fontsize=9)

    label_map = {'PROG_z': 'Ball Progression (z)',
                 'CREATE_z': 'Chance Creation (z)',
                 'DEF_z': 'Defensive Value (z)'}
    ax.set_xlabel(label_map.get(x_component, x_component), fontsize=11)
    ax.set_ylabel(label_map.get(y_component, y_component), fontsize=11)

    title = f"{position_group}s — {label_map.get(x_component)} vs {label_map.get(y_component)}"
    if max_age:
        title += f" (U{max_age})"
    ax.set_title(title, fontsize=13, fontweight='bold')

    ax.grid(True, alpha=0.2)
    ax.legend(loc='lower right', fontsize=9)

    fig.text(0.5, 0.01,
             f"n={len(pool)} spelers | min {min_minutes} minuten | "
             f"z-scores binnen positiegroep genormaliseerd",
             ha='center', fontsize=8, color='gray', style='italic')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Opgeslagen: {out_path}")
    return fig
