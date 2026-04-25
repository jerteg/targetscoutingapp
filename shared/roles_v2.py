"""
shared/roles_v2.py

Dimensie-gebaseerd rol-scoring systeem voor Target Scouting.
Vervangt de individuele-stat aanpak in role_config (zie templates.py).

Architectuur:
  1. POSITION_DIMENSIONS — per positie: dict van {dimensie_naam: [stats]}
  2. ROLE_CONFIG_V2     — per positie: dict van {rol_naam: {description, weights}}
                          waarbij weights de dimensies wegen (niet stats)
  3. compute_dimension_scores() — voor elke speler: percentile per stat → gemiddelde per dimensie
  4. compute_role_score()        — voor elke speler+rol: gewogen gemiddelde van dimensie-scores

Voordelen tov v1:
  • Geen multicollineariteit binnen een rol (gecorreleerde stats samen in één dimensie)
  • Geen dubbeltelling van Ball progression through passing en zijn componenten
  • Transparanter: rol-definities zijn dimensie-gewichten ipv stat-gewichten
  • Stabieler: dimensies zijn data-gedreven, rollen zijn meningen
"""

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# DIMENSIES PER POSITIE
#
# Belangrijke regels die in deze definities zijn toegepast:
# 1. Ball progression through passing (BPTP) is een SOM van vier componenten:
#    Deep completions + Completed passes to penalty area + Completed passes to
#    final third + Completed progressive passes (zie data_processing.py).
#    Daarom MAG geen van deze 4 componenten samen in dezelfde dimensie zitten
#    als BPTP — dat zou dubbeltelling zijn.
# 2. Aerial duels en interceptions zijn formeel verschillende fysieke acties:
#    aerial = 50/50 luchtduel, interceptie = anticipatie. Niet samenvoegen.
# 3. % stats meten kwaliteit, per 90 stats meten volume. Beide kunnen samen
#    in één dimensie als ze hetzelfde concept meten (bv. luchtduels), maar
#    NIET als ze verschillende facetten meten (bv. dribbel-kwaliteit vs
#    dribbel-volume).
# ──────────────────────────────────────────────────────────────────────────────

POSITION_DIMENSIONS = {

    # ════════════════════════════════════════════════════════════
    "Centre-Back": {
        "Defensive Duelling": [
            "PAdj Defensive duels won per 90",
            "Defensive duels won, %",
        ],
        "Aerial Dominance": [
            "PAdj Aerial duels won per 90",
            "Aerial duels won, %",
        ],
        "Anticipation": [
            "PAdj Interceptions",
        ],
        "Box Defending": [
            "Shots blocked per 90",
        ],
        "Discipline": [
            "Fouls per 90",  # negative
        ],
        "Ball Progression": [
            # Volume + accuracy samen — maar GEEN BPTP-componenten apart
            "Ball progression through passing",
            "Passing accuracy (prog/1/3/forw)",
        ],
        "Carrying": [
            "Progressive runs per received pass",
            "Successful dribbles per received pass",
        ],
        "Creation": [
            "xA per 90",
        ],
    },

    # ════════════════════════════════════════════════════════════
    "Right-Back": {  # ← identiek aan Left-Back
        "Defensive Duelling": [
            "PAdj Defensive duels won per 90",
            "Defensive duels won, %",
        ],
        "Aerial Dominance": [
            "PAdj Aerial duels won per 90",
            "Aerial duels won, %",
        ],
        "Anticipation": [
            "PAdj Interceptions",
        ],
        "Discipline": [
            "Fouls per 90",  # negative
        ],
        "Chance Creation": [  # hernoemd van "Crossing Output" — xA is breder
            "xA per 90",
            "Shot assists per 90",
            "Key passes per pass",
            "Accurate crosses per received pass",
        ],
        "Cross Accuracy": [
            "Accurate crosses, %",
        ],
        "Ball Progression": [
            "Ball progression through passing",
            "Passing accuracy (prog/1/3/forw)",
        ],
        "Dribble Volume": [  # gescheiden van quality
            "Successful dribbles per received pass",
            "Progressive runs per received pass",
        ],
        "Quality of Dribbles": [
            "Successful dribbles, %",
            "Offensive duels won, %",
        ],
        "Goal Threat": [
            "xG per 90",
        ],
    },
    "Left-Back": {},  # gevuld via copy hieronder

    # ════════════════════════════════════════════════════════════
    "Defensive Midfielder": {
        "Ground Defense Quality": [
            "Defensive duels won, %",
        ],
        "PAdj Defensive Volume": [
            "PAdj Defensive duels won per 90",
            "PAdj Interceptions",
            "PAdj Successful defensive actions per 90",
        ],
        "Aerial Dominance": [
            "PAdj Aerial duels won per 90",
            "Aerial duels won, %",
        ],
        "Discipline": [
            "Fouls per 90",  # negative
        ],
        "Ball Progression": [
            "Ball progression through passing",
            "Passing accuracy (prog/1/3/forw)",
        ],
        "Carrying": [
            "Progressive runs per received pass",
            "Successful dribbles per received pass",
        ],
        "Quality of Dribbles": [
            "Successful dribbles, %",
        ],
        "Creation": [
            "xA per 90",
            "Key passes per pass",
            "Through passes per pass",
        ],
    },

    # ════════════════════════════════════════════════════════════
    "Central Midfielder": {
        "Ground Defense Quality": [
            "Defensive duels won, %",
        ],
        "PAdj Defensive Volume": [
            "PAdj Defensive duels won per 90",
            "PAdj Interceptions",
        ],
        "Aerial Dominance": [
            "PAdj Aerial duels won per 90",
            "Aerial duels won, %",
        ],
        "Ball Progression": [
            "Ball progression through passing",
            "Passing accuracy (prog/1/3/forw)",
        ],
        "Carrying": [
            "Progressive runs per received pass",
            "Successful dribbles per received pass",
        ],
        "Quality of Dribbles": [
            "Successful dribbles, %",
            "Offensive duels won, %",
        ],
        "Box Threat": [
            "xG per 90",
            "xG per shot",
            "Touches in box per 90",
        ],
        "Finishing": [
            "Finishing",
        ],
        "Creation": [
            "xA per 90",
            "Key passes per pass",
            "Through passes per pass",
        ],
    },

    # ════════════════════════════════════════════════════════════
    "Attacking Midfielder": {
        "Box Threat": [
            "xG per 90",
            "xG per shot",
            "Touches in box per 90",
        ],
        "Finishing": [
            "Finishing",
        ],
        "Creation": [
            "xA per 90",
            "Key passes per pass",
            "Through passes per pass",
            "Completed passes to penalty area per 90",  # mag bij creation, niet bij BPTP
        ],
        "Ball Progression": [
            "Ball progression through passing",
            "Passing accuracy (prog/1/3/forw)",
        ],
        "Carrying": [
            "Successful dribbles per received pass",
            "Progressive runs per received pass",
        ],
        "Quality of Dribbles": [
            "Successful dribbles, %",
            "Offensive duels won, %",
        ],
        "Defensive Volume": [
            "PAdj Defensive duels won per 90",
            "PAdj Interceptions",
        ],
    },

    # ════════════════════════════════════════════════════════════
    "Right Winger": {  # ← identiek aan Left Winger
        "Goal Output": [
            "xG per 90",
            "xG per shot",
        ],
        "Finishing": [
            "Finishing",
        ],
        "Box Presence": [
            "Touches in box per 90",
        ],
        "Creation": [
            "xA per 90",
            "Key passes per pass",
            "Through passes per pass",
            "Accurate crosses per received pass",
        ],
        "Cross Accuracy": [
            "Accurate crosses, %",
        ],
        "Ball Progression": [
            "Ball progression through passing",
            "Passing accuracy (prog/1/3/forw)",
        ],
        "Dribble Volume": [
            "Successful dribbles per received pass",
            "Progressive runs per received pass",
        ],
        "Quality of Dribbles": [
            "Successful dribbles, %",
            "Offensive duels won, %",
        ],
        "Defensive Volume": [
            "PAdj Defensive duels won per 90",
            "PAdj Interceptions",
        ],
    },
    "Left Winger": {},  # gevuld via copy hieronder
    "Winger": {},  # alias voor Left/Right Winger

    # ════════════════════════════════════════════════════════════
    "Striker": {
        "Shooting Volume": [
            "xG per 90",
            "Shots per 90",
        ],
        "Shot Quality": [
            "xG per shot",
        ],
        "Finishing": [
            "Finishing",
            "Shots on target, %",
        ],
        "Box Presence": [
            "Touches in box per 90",
        ],
        "Aerial Threat": [
            "Aerial duels won, %",
            "PAdj Aerial duels won per 90",
            "Head goals per 90",
        ],
        "Target Reception": [
            "Received long passes per 90",
            "Received passes per 90",
        ],
        "Link-up Play": [
            "xA per 90",
            "Key passes per pass",
            "Offensive duels won, %",  # hold-up indicator
        ],
        "Carrying": [
            "Successful dribbles per received pass",
            "Progressive runs per received pass",
        ],
        "Quality of Dribbles": [
            "Successful dribbles, %",
        ],
        "Defensive Volume": [
            "PAdj Defensive duels won per 90",
            "PAdj Interceptions",
        ],
    },
}

# Spiegel posities (LB = RB, LW = RW)
POSITION_DIMENSIONS["Left-Back"]   = POSITION_DIMENSIONS["Right-Back"].copy()
POSITION_DIMENSIONS["Left Winger"] = POSITION_DIMENSIONS["Right Winger"].copy()
POSITION_DIMENSIONS["Winger"]      = POSITION_DIMENSIONS["Right Winger"].copy()


# Negatieve stats — bij percentile-berekening 100−pct nemen
NEGATIVE_STATS = {"Fouls per 90"}


# ──────────────────────────────────────────────────────────────────────────────
# ROL-DEFINITIES (gewichten op dimensies, niet op stats)
# Alle gewichten tellen op tot 1.0 per rol
# ──────────────────────────────────────────────────────────────────────────────

ROLE_CONFIG_V2 = {

    # ──────── CB ────────
    "Centre-Back": {
        "Stopper": {
            "description": "Klassieke no-nonsense CB. Duelzuchtig en defensief dominant. Levert minimal offensieve input.",
            "weights": {
                "Defensive Duelling": 0.25,
                "Aerial Dominance":   0.25,
                "Anticipation":       0.20,
                "Box Defending":      0.10,
                "Discipline":         0.10,
                "Ball Progression":   0.10,
            },
        },
        "Ball-Playing CB": {
            "description": "Moderne opbouw-CB. Speelt vaak vanaf eigen helft door de linies.",
            "weights": {
                "Defensive Duelling": 0.12,
                "Aerial Dominance":   0.13,
                "Anticipation":       0.10,
                "Discipline":         0.05,
                "Ball Progression":   0.40,
                "Carrying":           0.15,
                "Creation":           0.05,
            },
        },
        "Sweeper": {
            "description": "Anticipatie en positiespel boven duels. Leest het spel vanaf hoge lijn.",
            "weights": {
                "Defensive Duelling": 0.15,
                "Aerial Dominance":   0.15,
                "Anticipation":       0.30,
                "Box Defending":      0.05,
                "Discipline":         0.05,
                "Ball Progression":   0.20,
                "Carrying":           0.10,
            },
        },
        "Aggressor": {
            "description": "Stept uit, gaat mee in duels, dribbelt eruit. Vaak een outside-CB in een drieman.",
            "weights": {
                "Defensive Duelling": 0.30,
                "Aerial Dominance":   0.15,
                "Anticipation":       0.10,
                "Discipline":         0.05,
                "Ball Progression":   0.15,
                "Carrying":           0.25,
            },
        },
    },

    # ──────── FB (RB en LB delen rollen) ────────
    "Right-Back": {
        "Defensive Full-Back": {
            "description": "Stopt eerst, valt niet aan. Klassieke 1v1 verdediger op de flank.",
            "weights": {
                "Defensive Duelling": 0.25,
                "Aerial Dominance":   0.20,
                "Anticipation":       0.20,
                "Discipline":         0.10,
                "Ball Progression":   0.15,
                "Dribble Volume":     0.05,
                "Chance Creation":    0.05,
            },
        },
        "Attacking Full-Back": {
            "description": "Loopt door naar de achterlijn. Levert crosses, combineert in laatste 1/3.",
            "weights": {
                "Defensive Duelling": 0.10,
                "Aerial Dominance":   0.05,
                "Anticipation":       0.05,
                "Discipline":         0.05,
                "Chance Creation":    0.30,
                "Cross Accuracy":     0.10,
                "Dribble Volume":     0.15,
                "Quality of Dribbles": 0.10,
                "Goal Threat":        0.05,
                "Ball Progression":   0.05,
            },
        },
        "Inverted Full-Back": {
            "description": "Schuift naar binnen in opbouw, werkt als extra middenvelder. Stuurt het tempo.",
            "weights": {
                "Defensive Duelling": 0.10,
                "Anticipation":       0.10,
                "Discipline":         0.05,
                "Ball Progression":   0.45,
                "Dribble Volume":     0.10,
                "Chance Creation":    0.10,
                "Cross Accuracy":     0.10,
            },
        },
        "Wing-Back (complete)": {
            "description": "Alle facetten: verdedigt de flank én levert output. Vereist in 3-5-2 systemen.",
            "weights": {
                "Defensive Duelling": 0.15,
                "Aerial Dominance":   0.10,
                "Anticipation":       0.10,
                "Chance Creation":    0.20,
                "Cross Accuracy":     0.05,
                "Dribble Volume":     0.15,
                "Quality of Dribbles": 0.05,
                "Ball Progression":   0.15,
                "Goal Threat":        0.05,
            },
        },
    },

    # ──────── DM ────────
    "Defensive Midfielder": {
        "Ball-Winner": {
            "description": "Breekt het spel. Hoog volume tackles, interceptions, fouls. Klassieke destroyer.",
            "weights": {
                "Ground Defense Quality":  0.15,
                "PAdj Defensive Volume":   0.40,
                "Aerial Dominance":        0.20,
                "Discipline":              0.10,
                "Ball Progression":        0.10,
                "Carrying":                0.05,
            },
        },
        "Deep-Lying Playmaker": {
            "description": "Regisseur van voor de verdediging. Pass accuracy > duelkracht.",
            "weights": {
                "Ground Defense Quality":  0.10,
                "PAdj Defensive Volume":   0.10,
                "Aerial Dominance":        0.10,
                "Discipline":              0.05,
                "Ball Progression":        0.40,
                "Carrying":                0.10,
                "Creation":                0.15,
            },
        },
        "Anchor": {
            "description": "Simpel, veilig, positioneel. Houdt positie, speelt korte passes, wint luchtduels.",
            "weights": {
                "Ground Defense Quality":  0.15,
                "PAdj Defensive Volume":   0.20,
                "Aerial Dominance":        0.30,
                "Discipline":              0.10,
                "Ball Progression":        0.20,
                "Carrying":                0.05,
            },
        },
        "Box-to-Box DM": {
            "description": "Diepste middenvelder die óók ruimte inloopt en soms in de zestien komt.",
            "weights": {
                "Ground Defense Quality":  0.10,
                "PAdj Defensive Volume":   0.20,
                "Aerial Dominance":        0.10,
                "Discipline":              0.05,
                "Ball Progression":        0.20,
                "Carrying":                0.15,
                "Creation":                0.15,
                "Quality of Dribbles":     0.05,
            },
        },
    },

    # ──────── CM ────────
    "Central Midfielder": {
        "Ball-Winning Midfielder": {
            "description": "Wint ballen centraal, begint counters. Mix van destroyer en progressor.",
            "weights": {
                "Ground Defense Quality":  0.15,
                "PAdj Defensive Volume":   0.20,
                "Aerial Dominance":        0.20,
                "Ball Progression":        0.25,
                "Carrying":                0.10,
                "Creation":                0.10,
            },
        },
        "Deep-Lying Playmaker (8)": {
            "description": "CM die diep zakt en passes verdeelt. Meer een '8 op de 6-lijn'.",
            "weights": {
                "Ground Defense Quality":  0.10,
                "Aerial Dominance":        0.10,
                "PAdj Defensive Volume":   0.10,
                "Ball Progression":        0.40,
                "Carrying":                0.10,
                "Creation":                0.20,
            },
        },
        "Box-to-Box Midfielder": {
            "description": "All-rounder. Aanvallend én verdedigend betrokken, loopt ruimtes in.",
            "weights": {
                "PAdj Defensive Volume":   0.15,
                "Aerial Dominance":        0.10,
                "Ball Progression":        0.20,
                "Carrying":                0.15,
                "Quality of Dribbles":     0.05,
                "Box Threat":              0.15,
                "Finishing":               0.05,
                "Creation":                0.15,
            },
        },
        "Advanced Playmaker": {
            "description": "CM met hoge creatieve output. Komt dicht bij AM, creëert vanuit de 8-positie.",
            "weights": {
                "Ground Defense Quality":  0.05,
                "PAdj Defensive Volume":   0.10,
                "Ball Progression":        0.20,
                "Carrying":                0.10,
                "Quality of Dribbles":     0.05,
                "Box Threat":              0.10,
                "Creation":                0.40,
            },
        },
        "Goalscoring Midfielder": {
            "description": "8 die in de box komt (Lampard/De Bruyne-achtig). Hoge xG voor een CM.",
            "weights": {
                "PAdj Defensive Volume":   0.10,
                "Aerial Dominance":        0.05,
                "Ball Progression":        0.15,
                "Carrying":                0.10,
                "Box Threat":              0.30,
                "Finishing":               0.15,
                "Creation":                0.15,
            },
        },
    },

    # ──────── AM ────────
    "Attacking Midfielder": {
        "Classic 10": {
            "description": "Creëert vanuit de zone achter de spits. Hoge assist-output.",
            "weights": {
                "Box Threat":              0.10,
                "Finishing":               0.05,
                "Creation":                0.40,
                "Ball Progression":        0.15,
                "Carrying":                0.15,
                "Quality of Dribbles":     0.05,
                "Defensive Volume":        0.10,
            },
        },
        "Shadow Striker": {
            "description": "AM die uit zijn positie zakt richting de box. Scoort zelf veel.",
            "weights": {
                "Box Threat":              0.30,
                "Finishing":               0.20,
                "Creation":                0.15,
                "Ball Progression":        0.05,
                "Carrying":                0.10,
                "Quality of Dribbles":     0.10,
                "Defensive Volume":        0.10,
            },
        },
        "Dribbling 10": {
            "description": "Creëert door eigen actie. Dribbels en runs voor de creatie van kansen.",
            "weights": {
                "Box Threat":              0.15,
                "Finishing":               0.05,
                "Creation":                0.15,
                "Ball Progression":        0.05,
                "Carrying":                0.30,
                "Quality of Dribbles":     0.20,
                "Defensive Volume":        0.10,
            },
        },
        "Pressing 10": {
            "description": "Modern werkpaardje. Creëert, maar levert ook defensief werk.",
            "weights": {
                "Box Threat":              0.10,
                "Finishing":               0.05,
                "Creation":                0.30,
                "Ball Progression":        0.10,
                "Carrying":                0.10,
                "Defensive Volume":        0.35,
            },
        },
    },

    # ──────── W ────────
    "Right Winger": {
        "Goalscoring Winger": {
            "description": "Snijdt naar binnen en scoort. Hoog xG volume en afwerking kritisch.",
            "weights": {
                "Goal Output":         0.30,
                "Finishing":           0.20,
                "Box Presence":        0.15,
                "Creation":            0.10,
                "Dribble Volume":      0.10,
                "Quality of Dribbles": 0.05,
                "Ball Progression":    0.05,
                "Defensive Volume":    0.05,
            },
        },
        "Creative Winger": {
            "description": "Inverted playmaker. Levert de assist, kijkt meer dan hij schiet.",
            "weights": {
                "Goal Output":         0.10,
                "Finishing":           0.05,
                "Box Presence":        0.05,
                "Creation":            0.40,
                "Cross Accuracy":      0.05,
                "Dribble Volume":      0.10,
                "Quality of Dribbles": 0.10,
                "Ball Progression":    0.10,
                "Defensive Volume":    0.05,
            },
        },
        "Direct Winger": {
            "description": "Klassieke buitenkant-speler. Gaat door tot de achterlijn en levert crosses.",
            "weights": {
                "Goal Output":         0.05,
                "Finishing":           0.05,
                "Box Presence":        0.05,
                "Creation":            0.20,
                "Cross Accuracy":      0.15,
                "Dribble Volume":      0.25,
                "Quality of Dribbles": 0.15,
                "Defensive Volume":    0.10,
            },
        },
        "Complete Forward": {
            "description": "Top-tier elite winger: scoort, creëert én draagt. Yamal/Saka/Olise-achtig.",
            "weights": {
                "Goal Output":         0.20,
                "Finishing":           0.15,
                "Box Presence":        0.10,
                "Creation":            0.20,
                "Dribble Volume":      0.15,
                "Quality of Dribbles": 0.10,
                "Ball Progression":    0.05,
                "Defensive Volume":    0.05,
            },
        },
        "Pressing Winger": {
            "description": "Houdt de flank dicht én valt aan. Essentieel in intense pressing systemen.",
            "weights": {
                "Goal Output":         0.15,
                "Finishing":           0.10,
                "Box Presence":        0.05,
                "Creation":            0.10,
                "Dribble Volume":      0.10,
                "Quality of Dribbles": 0.05,
                "Defensive Volume":    0.40,
                "Ball Progression":    0.05,
            },
        },
    },

    # ──────── ST ────────
    "Striker": {
        "Poacher": {
            "description": "Leeft in de zestien. Scoort vanuit goede posities, low-volume maar high-quality.",
            "weights": {
                "Shooting Volume":     0.25,
                "Shot Quality":        0.20,
                "Finishing":           0.25,
                "Box Presence":        0.15,
                "Aerial Threat":       0.05,
                "Link-up Play":        0.05,
                "Defensive Volume":    0.05,
            },
        },
        "Target Man": {
            "description": "Kopspelend referentiepunt. Houdt ballen vast, wint luchtduels, scoort met hoofd.",
            "weights": {
                "Shooting Volume":     0.10,
                "Shot Quality":        0.10,
                "Finishing":           0.05,
                "Box Presence":        0.10,
                "Aerial Threat":       0.30,
                "Target Reception":    0.20,
                "Link-up Play":        0.10,
                "Defensive Volume":    0.05,
            },
        },
        "Deep-Lying Forward": {
            "description": "Zakt diep. Combineert, creëert, scoort. Kane/Firmino/Benzema-type.",
            "weights": {
                "Shooting Volume":     0.10,
                "Shot Quality":        0.10,
                "Finishing":           0.10,
                "Box Presence":        0.05,
                "Link-up Play":        0.30,
                "Carrying":            0.15,
                "Quality of Dribbles": 0.10,
                "Defensive Volume":    0.05,
                "Target Reception":    0.05,
            },
        },
        "Pressing Forward": {
            "description": "Eerste verdediger van het team. Hoog duelwerk, gaat mee in pressing.",
            "weights": {
                "Shooting Volume":     0.15,
                "Shot Quality":        0.10,
                "Finishing":           0.10,
                "Box Presence":        0.05,
                "Aerial Threat":       0.10,
                "Link-up Play":        0.10,
                "Carrying":            0.05,
                "Defensive Volume":    0.35,
            },
        },
        "Complete Forward": {
            "description": "Alle facetten. Moderne topspits (Haaland, Lewandowski-type).",
            "weights": {
                "Shooting Volume":     0.20,
                "Shot Quality":        0.20,
                "Finishing":           0.15,
                "Box Presence":        0.10,
                "Aerial Threat":       0.10,
                "Target Reception":    0.05,
                "Link-up Play":        0.10,
                "Carrying":            0.05,
                "Defensive Volume":    0.05,
            },
        },
    },
}

# Spiegel: LB krijgt zelfde rollen als RB, LW als RW
ROLE_CONFIG_V2["Left-Back"]   = ROLE_CONFIG_V2["Right-Back"].copy()
ROLE_CONFIG_V2["Left Winger"] = ROLE_CONFIG_V2["Right Winger"].copy()
ROLE_CONFIG_V2["Winger"]      = ROLE_CONFIG_V2["Right Winger"].copy()


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATIE — bij module-load checken of elke rol's gewichten op 1.0 sommeren
# en alle gebruikte dimensies bestaan
# ──────────────────────────────────────────────────────────────────────────────
def _validate_role_config():
    errors = []
    for pos, roles in ROLE_CONFIG_V2.items():
        dims_for_pos = set(POSITION_DIMENSIONS.get(pos, {}).keys())
        for role_name, role_def in roles.items():
            w_sum = sum(role_def["weights"].values())
            if abs(w_sum - 1.0) > 0.01:
                errors.append(f"{pos} → {role_name}: weights sum to {w_sum:.3f}, not 1.0")
            for dim in role_def["weights"]:
                if dim not in dims_for_pos:
                    errors.append(f"{pos} → {role_name}: unknown dimension '{dim}'")
    return errors


# ──────────────────────────────────────────────────────────────────────────────
# SCORING FUNCTIES
# ──────────────────────────────────────────────────────────────────────────────

def compute_dimension_scores(
    pool: pd.DataFrame,
    position_label: str,
) -> pd.DataFrame:
    """
    Bereken per speler in `pool` een score per dimensie.

    Methodologie:
      - Voor elke stat in een dimensie: percentile rank binnen de pool (0-100)
      - Negatieve stats (Fouls per 90): 100 - percentile
      - Dimensie-score = ongewogen gemiddelde van zijn stat-percentielen
      - NaN's: ingevuld als 50ste percentiel (neutraal)

    Parameters
    ----------
    pool           : DataFrame van spelers (al gefilterd op positie + competitie)
    position_label : key uit POSITION_DIMENSIONS (bv. "Centre-Back")

    Returns
    -------
    DataFrame met kolommen `dim_<dimensienaam>` per dimensie, plus de
    originele kolommen uit `pool`. Index identiek aan `pool`.
    """
    if pool.empty or position_label not in POSITION_DIMENSIONS:
        return pool.copy()

    out = pool.copy()
    dims = POSITION_DIMENSIONS[position_label]

    for dim_name, stats in dims.items():
        # Filter naar stats die echt in pool zitten
        avail = [s for s in stats if s in pool.columns]
        if not avail:
            out[f"dim_{dim_name}"] = 50.0  # geen data = neutraal
            continue

        # Percentile rank per stat binnen pool
        pcts = []
        for s in avail:
            vals = pd.to_numeric(pool[s], errors="coerce")
            pct = vals.rank(pct=True) * 100
            if s in NEGATIVE_STATS:
                pct = 100 - pct
            pct = pct.fillna(50)  # neutraal voor missing
            pcts.append(pct)

        # Ongewogen gemiddelde van de stat-percentielen
        dim_score = pd.concat(pcts, axis=1).mean(axis=1)
        out[f"dim_{dim_name}"] = dim_score

    return out


def compute_role_score(
    pool_with_dims: pd.DataFrame,
    position_label: str,
    role_name: str,
) -> pd.Series:
    """
    Bereken role-score per speler op basis van gewogen dimensie-scores.

    Parameters
    ----------
    pool_with_dims : output van compute_dimension_scores()
    position_label : moet bestaan in ROLE_CONFIG_V2
    role_name      : moet bestaan binnen die positie

    Returns
    -------
    Series met role-score per speler (0-100), index van pool_with_dims.
    """
    if position_label not in ROLE_CONFIG_V2:
        raise KeyError(f"Position '{position_label}' niet in ROLE_CONFIG_V2")
    roles = ROLE_CONFIG_V2[position_label]
    if role_name not in roles:
        raise KeyError(f"Role '{role_name}' niet beschikbaar voor {position_label}")

    weights = roles[role_name]["weights"]
    score = pd.Series(0.0, index=pool_with_dims.index)

    for dim_name, weight in weights.items():
        col = f"dim_{dim_name}"
        if col not in pool_with_dims.columns:
            continue
        score = score + pool_with_dims[col].fillna(50) * weight

    return score.clip(0, 100)


def get_role_options(position_label: str) -> list:
    """Geef lijst van rol-namen voor een positie."""
    return list(ROLE_CONFIG_V2.get(position_label, {}).keys())


def get_role_description(position_label: str, role_name: str) -> str:
    """Geef de description string voor een rol."""
    return ROLE_CONFIG_V2.get(position_label, {}).get(role_name, {}).get("description", "")


def get_dimensions_for_position(position_label: str) -> dict:
    """Geef de dimensie → stats mapping voor een positie."""
    return POSITION_DIMENSIONS.get(position_label, {})


def get_role_weights(position_label: str, role_name: str) -> dict:
    """Geef de dimensie-gewichten voor een specifieke rol."""
    return ROLE_CONFIG_V2.get(position_label, {}).get(role_name, {}).get("weights", {})


# ──────────────────────────────────────────────────────────────────────────────
# Run validatie bij import
# ──────────────────────────────────────────────────────────────────────────────
_errors = _validate_role_config()
if _errors:
    import warnings as _w
    for e in _errors:
        _w.warn(f"[roles_v2] {e}")
