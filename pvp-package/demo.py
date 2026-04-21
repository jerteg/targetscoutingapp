"""
demo.py — Demonstratie van het PVP package.

Draait vier voorbeelden:
1. Speler-profiel (Pedri) als markdown rapport
2. Shortlist (U23 CMs) als markdown rapport
3. Radar chart van Pedri vs Bellingham vs Kimmich
4. Scatter plot van alle CMs met highlights
"""

import sys
sys.path.insert(0, '/home/claude/pvp_package')

from pvp import PVPModel
from report import generate_player_report, generate_shortlist_report
from visuals import player_radar, shortlist_scatter

OUT = '/mnt/user-data/outputs'

# Load model
print("▶ Laden van model...")
m = PVPModel.from_csv('/mnt/user-data/uploads/data.csv')
print(f"  {len(m.df)} spelers verwerkt, {len(m.df['PosGroup'].unique())} posgroepen\n")

# Export full results
m.export(f'{OUT}/pvp_full_results.csv')
print(f"▶ Volledige resultaten opgeslagen: pvp_full_results.csv\n")

# 1. Speler-rapport
print("▶ Genereer speler-rapport voor Pedri...")
report = generate_player_report(m, 'Pedri', out_path=f'{OUT}/report_pedri.md')
print(f"  Opgeslagen: report_pedri.md\n")

# 2. Shortlist-rapport: jonge CMs
print("▶ Genereer shortlist: CM's onder de 23...")
generate_shortlist_report(
    m, position_group='CM', max_age=23, min_pvp_percentile=85,
    top_n=20, title="Shortlist: Jonge middenvelders (U23)",
    out_path=f'{OUT}/shortlist_cm_u23.md'
)
print(f"  Opgeslagen: shortlist_cm_u23.md\n")

# 3. Shortlist: defensieve CBs onder 25 uit tier-2 competities
print("▶ Genereer shortlist: jonge CBs uit tier-2 competities...")
generate_shortlist_report(
    m, position_group='CB', max_age=25,
    min_league_coef=0.70, min_pvp_percentile=85, top_n=20,
    title="Shortlist: Jonge CBs uit tier-2 competities",
    out_path=f'{OUT}/shortlist_cb_tier2.md'
)
print(f"  Opgeslagen: shortlist_cb_tier2.md\n")

# 4. Radar: Pedri vs anderen
print("▶ Radar chart: Pedri vs Kimmich vs Fernandes...")
fig = player_radar(m, 'Pedri', comparison=['Kimmich', 'Bruno Fernandes'],
                   out_path=f'{OUT}/radar_pedri_vs.png')
import matplotlib.pyplot as plt
plt.close(fig)

# 5. Scatter: CMs PROG vs CREATE
print("▶ Scatter plot: CMs op PROG vs CREATE (U25)...")
fig = shortlist_scatter(m, position_group='CM', max_age=25,
                        x_component='PROG_z', y_component='CREATE_z',
                        highlight=['Pedri', 'Yamal', 'Cherki'],
                        out_path=f'{OUT}/scatter_cm_u25.png')
plt.close(fig)

# 6. Scatter: CBs DEF vs PROG (ball-playing vs stopper)
print("▶ Scatter plot: CBs op PROG vs DEF...")
fig = shortlist_scatter(m, position_group='CB',
                        x_component='DEF_z', y_component='PROG_z',
                        highlight=['Senesi', 'Bastoni', 'Eric García'],
                        out_path=f'{OUT}/scatter_cb_all.png')
plt.close(fig)

print("\n✓ Alle output opgeslagen in /mnt/user-data/outputs/")
