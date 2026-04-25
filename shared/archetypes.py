"""
shared/archetypes.py — PATCH

Voeg "Winger" toe aan SCORING_TO_ARCHETYPE.

Vervang het hele SCORING_TO_ARCHETYPE blok (regels ~26-36 in je huidige bestand) met:
"""

SCORING_TO_ARCHETYPE = {
    "Centre-Back":           "CB",
    "Right-Back":            "FB",
    "Left-Back":             "FB",
    "Defensive Midfielder":  "MID",
    "Central Midfielder":    "MID",
    "Attacking Midfielder":  "MID",
    "Right Winger":          "W",
    "Left Winger":           "W",
    "Winger":                "W",   # ← NIEUW: fix voor Dashboard-archetype-bug
    "Striker":               "ST",
}
