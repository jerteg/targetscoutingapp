"""
report.py — Genereer markdown scouting rapporten op basis van PVP.

Usage:
    from pvp import PVPModel
    from report import generate_player_report, generate_shortlist_report

    m = PVPModel.from_csv("data.csv")
    generate_player_report(m, "Pedri", out_path="pedri.md")
    generate_shortlist_report(m, position_group="CM", max_age=23,
                              out_path="shortlist_cm_u23.md")
"""

from datetime import date


def _bar(z, width=30):
    """ASCII-bar voor z-score op -3 tot +3 schaal."""
    z_clipped = max(-3, min(3, z))
    pos = int((z_clipped + 3) / 6 * width)
    bar = ['·'] * width
    center = width // 2
    bar[center] = '│'
    if pos > center:
        for i in range(center+1, min(pos+1, width)):
            bar[i] = '█'
    elif pos < center:
        for i in range(max(0, pos), center):
            bar[i] = '█'
    return ''.join(bar)


def _interpret_z(z):
    """Vertaal z-score naar label."""
    if z >= 2.0: return "**elite**"
    if z >= 1.0: return "duidelijk bovengemiddeld"
    if z >= 0.3: return "bovengemiddeld"
    if z >= -0.3: return "gemiddeld"
    if z >= -1.0: return "ondergemiddeld"
    if z >= -2.0: return "zwak"
    return "**zeer zwak**"


def _format_value(v):
    """Formatteer market value (euro)."""
    try:
        v = float(v)
    except (ValueError, TypeError):
        return "onbekend"
    if v == 0:
        return "onbekend"
    if v >= 1_000_000:
        return f"€{v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"€{v/1_000:.0f}K"
    return f"€{v:.0f}"


def generate_player_report(model, player_name, out_path=None, n_similar=10):
    """Maak een uitgebreid Markdown-rapport voor één speler."""
    match = model.df[model.df['Player'].str.contains(player_name, case=False, na=False)]
    if len(match) == 0:
        raise ValueError(f"Geen speler gevonden: '{player_name}'")
    if len(match) > 1:
        print(f"Let op: {len(match)} matches gevonden, eerste gebruikt: "
              f"{match.iloc[0]['Player']} ({match.iloc[0]['Team']})")
    p = match.iloc[0]

    target, similar = model.find_similar(p['Player'], n=n_similar, same_position=True)

    lines = []
    lines.append(f"# Scouting Rapport — {p['Player']}")
    lines.append("")
    lines.append(f"*Gegenereerd op {date.today().isoformat()} · Model: PVP (Possession Value Proxy)*")
    lines.append("")
    lines.append("> **Disclaimer:** PVP is een composite score geïnspireerd op xT, VAEP en OBV, "
                 "maar is **geen** implementatie daarvan. Het model gebruikt per-90 aggregated "
                 "stats en kan geen context-aware evaluaties maken (druk, passlijnen, ploeggenoten). "
                 "Gebruik altijd in combinatie met videoanalyse.")
    lines.append("")

    # --- Basisinfo ---
    lines.append("## Basisinformatie")
    lines.append("")
    lines.append(f"| Veld | Waarde |")
    lines.append(f"|---|---|")
    lines.append(f"| Team | {p['Team']} |")
    lines.append(f"| Competitie | {p['League']} (league coef {p['LeagueCoef']:.2f}) |")
    lines.append(f"| Positie | {p['Position']} (groep: **{p['PosGroup']}**) |")
    lines.append(f"| Leeftijd | {p['Age']} |")
    lines.append(f"| Minuten | {int(p['Minutes played'])} |")
    lines.append(f"| Marktwaarde | {_format_value(p.get('Market value', 0))} |")
    lines.append(f"| Contract expires | {p.get('Contract expires', 'onbekend')} |")
    lines.append("")

    # --- Overall PVP ---
    lines.append("## Overall Possession Value Proxy")
    lines.append("")
    lines.append(f"- **PVP score:** {p['PVP']:+.2f}")
    lines.append(f"- **Percentiel binnen positiegroep ({p['PosGroup']}):** "
                 f"{p['PVP_percentile']:.1f}%")
    lines.append("")
    if p['PVP_percentile'] >= 95:
        lines.append("→ Behoort tot de **top 5%** van spelers in zijn positiegroep.")
    elif p['PVP_percentile'] >= 85:
        lines.append("→ Behoort tot de **top 15%** van spelers in zijn positiegroep.")
    elif p['PVP_percentile'] >= 70:
        lines.append("→ Bovengemiddelde profiel binnen positiegroep.")
    elif p['PVP_percentile'] >= 30:
        lines.append("→ Gemiddeld profiel binnen positiegroep.")
    else:
        lines.append("→ Ondergemiddeld profiel binnen positiegroep.")
    lines.append("")

    # --- Component breakdown ---
    lines.append("## Profiel-breakdown (z-scores binnen positiegroep)")
    lines.append("")
    lines.append("```")
    lines.append(f"                    -3       -2       -1        0       +1       +2       +3")
    lines.append(f"                     ↓        ↓        ↓        │        ↓        ↓        ↓")
    for comp_label, z in [("BALL PROGRESSION", p['PROG_z']),
                          ("CHANCE CREATION ", p['CREATE_z']),
                          ("DEFENSIVE VALUE ", p['DEF_z'])]:
        lines.append(f"{comp_label}  {_bar(z)}  {z:+.2f}")
    lines.append("```")
    lines.append("")
    lines.append(f"- **Ball Progression** ({p['PROG_z']:+.2f}): {_interpret_z(p['PROG_z'])} — "
                 "hoe goed de speler de bal naar dreigende zones beweegt via progressive passes, "
                 "runs, through balls en deep completions.")
    lines.append(f"- **Chance Creation** ({p['CREATE_z']:+.2f}): {_interpret_z(p['CREATE_z'])} — "
                 "directe offensieve dreiging via xG, xA, key passes en box-entries.")
    lines.append(f"- **Defensive Value** ({p['DEF_z']:+.2f}): {_interpret_z(p['DEF_z'])} — "
                 "defensieve bijdrage via PAdj interceptions/tackles, blocks en duels.")
    lines.append("")

    # --- Rol-interpretatie ---
    lines.append("## Profiel-classificatie")
    lines.append("")
    lines.append(_classify_profile(p))
    lines.append("")

    # --- Vergelijkbare spelers ---
    lines.append(f"## Vergelijkbare profielen (top {n_similar})")
    lines.append("")
    lines.append("Gesorteerd op Euclidische afstand in (PROG, CREATE, DEF) z-score ruimte.")
    lines.append("")
    lines.append("| Speler | Team | Competitie | Leeftijd | Min | PROG | CREATE | DEF | PVP | Similarity |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, row in similar.iterrows():
        lines.append(f"| {row['Player']} | {row['Team']} | {row['League']} | {row['Age']} | "
                     f"{int(row['Minutes played'])} | {row['PROG_z']:+.2f} | "
                     f"{row['CREATE_z']:+.2f} | {row['DEF_z']:+.2f} | "
                     f"{row['PVP']:+.2f} | {row['Similarity']:.3f} |")
    lines.append("")

    # --- Caveats ---
    lines.append("## Caveats")
    lines.append("")
    lines.append("1. League strength coefficient is een benadering. Kalibreer tegen je "
                 "interne beoordelingen voor betere cross-competitie vergelijkingen.")
    lines.append("2. De model-gewichten (per rol) zijn een keuze. Speelt jouw team in een "
                 "systeem dat andere eigenschappen prioriteert, overweeg dan ROLE_WEIGHTS aan "
                 "te passen.")
    lines.append("3. Aggregated stats missen context: pressure, pass difficulty, kwaliteit "
                 "van teamgenoten. Echte xT/VAEP/OBV missen dit óók deels (alleen tracking "
                 "data lost dit volledig op), maar zij wegen tenminste per-actie context mee.")
    lines.append("4. Sample-size: minimum 1000 minuten is hier gebruikt. Voor hele jonge "
                 "spelers met beperkte speeltijd kan één goede periode het gemiddelde "
                 "vertekenen.")

    report = "\n".join(lines)
    if out_path:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(report)
    return report


def _classify_profile(p):
    """Korte interpretatie op basis van welke componenten domineren."""
    prog, create, defn = p['PROG_z'], p['CREATE_z'], p['DEF_z']
    pos = p['PosGroup']

    tags = []
    if prog >= 1.5: tags.append("sterke progressor")
    if create >= 1.5: tags.append("gevaarlijke creator")
    if defn >= 1.5: tags.append("defensief dominant")
    if prog <= -1.0: tags.append("beperkt in progressie")
    if create <= -1.0: tags.append("beperkt in creatie")
    if defn <= -1.0: tags.append("defensief zwak")

    if not tags:
        return f"Gebalanceerd {pos}-profiel zonder duidelijke uitschieters (positief of negatief)."

    # Archetype heuristiek
    archetype = ""
    if pos == 'CM':
        if prog > 1 and create > 1: archetype = " — klassieke **deep-lying playmaker / regista**"
        elif prog > 1 and defn > 1: archetype = " — **box-to-box midfielder**"
        elif create > 1.5 and defn < 0: archetype = " — **attackende nummer 10**"
        elif defn > 1.5 and create < 0: archetype = " — **pure 6 / holdend**"
    elif pos == 'CF':
        if create > 1.5 and prog > 0.5: archetype = " — **complete striker / false 9**"
        elif create > 1.5: archetype = " — **pure finisher / penalty box predator**"
    elif pos == 'WG':
        if prog > 1 and create > 1: archetype = " — **inverted winger / dreigende breeker**"
    elif pos == 'FB':
        if prog > 1 and create > 0.5: archetype = " — **offensieve fullback / wing-back**"
        elif defn > 1 and prog < 0: archetype = " — **traditionele defensieve back**"
    elif pos == 'CB':
        if prog > 1: archetype = " — **ball-playing defender**"
        elif defn > 1.5 and prog < 0: archetype = " — **traditionele stopper**"

    return f"Archetype tags: _{', '.join(tags)}_{archetype}."


def generate_shortlist_report(model, position_group=None, max_age=None,
                               min_minutes=1000, min_pvp_percentile=85,
                               leagues=None, min_league_coef=None,
                               top_n=25, title="Scouting Shortlist",
                               out_path=None):
    """Genereer een shortlist-rapport met top-N spelers volgens filters."""
    sl = model.shortlist(position_group=position_group, max_age=max_age,
                         min_minutes=min_minutes,
                         min_pvp_percentile=min_pvp_percentile,
                         leagues=leagues, min_league_coef=min_league_coef,
                         top_n=top_n)

    lines = []
    lines.append(f"# {title}")
    lines.append(f"\n*Gegenereerd op {date.today().isoformat()}*\n")

    # Filters
    lines.append("## Filters")
    lines.append("")
    if position_group: lines.append(f"- Positiegroep: **{position_group}**")
    if max_age: lines.append(f"- Max leeftijd: **{max_age}**")
    lines.append(f"- Min minuten: **{min_minutes}**")
    lines.append(f"- Min PVP-percentiel: **{min_pvp_percentile}**")
    if min_league_coef: lines.append(f"- Min league coef: **{min_league_coef}**")
    if leagues: lines.append(f"- Competities: {', '.join(leagues)}")
    lines.append("")

    # Tabel
    lines.append(f"## Top {len(sl)} spelers")
    lines.append("")
    lines.append("| # | Speler | Team | Comp | Pos | Age | Min | Market | PROG | CREATE | DEF | PVP |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, (_, row) in enumerate(sl.iterrows(), 1):
        lines.append(f"| {i} | {row['Player']} | {row['Team']} | {row['League']} | "
                     f"{row['Position']} | {row['Age']} | {int(row['Minutes played'])} | "
                     f"{_format_value(row.get('Market value', 0))} | "
                     f"{row['PROG_z']:+.1f} | {row['CREATE_z']:+.1f} | "
                     f"{row['DEF_z']:+.1f} | **{row['PVP']:+.2f}** |")
    lines.append("")
    lines.append("> Z-scores zijn binnen positiegroep genormaliseerd.")

    report = "\n".join(lines)
    if out_path:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(report)
    return report
