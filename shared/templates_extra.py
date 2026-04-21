"""
shared/templates_extra.py
New dicts to add alongside the existing templates.py.
Import from here in pages that need archetype or dashboard-bar data.
"""

# ── Dashboard percentile bars per position ─────────────────────────────────────
DASHBOARD_BARS_PER_POSITION = {
    "Right-Back": [
        "xA per 90", "Accurate crosses per received pass", "Accurate crosses, %",
        "Successful dribbles per received pass",
        "Progressive runs per received pass",
        "Ball progression through passing",
        "Passing accuracy (prog/1/3/forw)",
        "PAdj Defensive duels won per 90", "Defensive duels won, %",
        "PAdj Aerial duels won per 90", "Aerial duels won, %",
        "PAdj Interceptions",
    ],
    "Left-Back": [
        "xA per 90", "Accurate crosses per received pass", "Accurate crosses, %",
        "Successful dribbles per received pass",
        "Progressive runs per received pass",
        "Ball progression through passing",
        "Passing accuracy (prog/1/3/forw)",
        "PAdj Defensive duels won per 90", "Defensive duels won, %",
        "PAdj Aerial duels won per 90", "Aerial duels won, %",
        "PAdj Interceptions",
    ],
    "Centre-Back": [
        "xG per 90",
        "Successful dribbles per received pass",
        "Progressive runs per received pass",
        "Ball progression through passing",
        "Passing accuracy (prog/1/3/forw)",
        "PAdj Defensive duels won per 90", "Defensive duels won, %",
        "PAdj Aerial duels won per 90", "Aerial duels won, %",
        "PAdj Interceptions",
        "Fouls per 90",
        "Shots blocked per 90",
    ],
    "Defensive Midfielder": [
        "xA per 90",
        "Successful dribbles, %",
        "Progressive runs per received pass",
        "Ball progression through passing",
        "Passing accuracy (prog/1/3/forw)",
        "PAdj Defensive duels won per 90", "Defensive duels won, %",
        "PAdj Aerial duels won per 90", "Aerial duels won, %",
        "PAdj Successful defensive actions per 90",
        "PAdj Interceptions",
        "Fouls per 90",
    ],
    "Central Midfielder": [
        "xG per 90", "Touches in box per 90", "xA per 90",
        "Successful dribbles per received pass",
        "Progressive runs per received pass",
        "Ball progression through passing",
        "Passing accuracy (prog/1/3/forw)",
        "PAdj Defensive duels won per 90", "Defensive duels won, %",
        "PAdj Aerial duels won per 90", "Aerial duels won, %",
        "PAdj Interceptions",
    ],
    "Attacking Midfielder": [
        "xG per 90", "xG per shot", "Finishing",
        "Touches in box per 90", "xA per 90",
        "Key passes per pass",
        "Successful dribbles per received pass",
        "Progressive runs per received pass",
        "Ball progression through passing",
        "Passing accuracy (prog/1/3/forw)",
        "Defensive duels won, %", "Aerial duels won, %",
    ],
    "Right Winger": [
        "Finishing", "xG per 90", "xG per shot",
        "Touches in box per 90", "xA per 90",
        "Key passes per pass",
        "Accurate crosses per received pass",
        "Successful dribbles per received pass",
        "Progressive runs per received pass",
        "Ball progression through passing",
        "Passing accuracy (prog/1/3/forw)",
        "Defensive duels won, %",
    ],
    "Left Winger": [
        "Finishing", "xG per 90", "xG per shot",
        "Touches in box per 90", "xA per 90",
        "Key passes per pass",
        "Accurate crosses per received pass",
        "Successful dribbles per received pass",
        "Progressive runs per received pass",
        "Ball progression through passing",
        "Passing accuracy (prog/1/3/forw)",
        "Defensive duels won, %",
    ],
    "Striker": [
        "xG per 90", "xG per shot", "Finishing",
        "Touches in box per 90", "xA per 90",
        "Key passes per pass",
        "Successful dribbles per received pass", "Successful dribbles, %",
        "Progressive runs per received pass",
        "Passing accuracy (prog/1/3/forw)",
        "Aerial duels won, %",
        "PAdj Interceptions",
    ],
    # Winger fallback
    "Winger": [
        "Finishing", "xG per 90", "xG per shot",
        "Touches in box per 90", "xA per 90",
        "Key passes per pass",
        "Accurate crosses per received pass",
        "Successful dribbles per received pass",
        "Progressive runs per received pass",
        "Ball progression through passing",
        "Passing accuracy (prog/1/3/forw)",
        "Defensive duels won, %",
    ],
}

# Fixed scatter axes per archetype position group
DASHBOARD_SCATTER_AXES = {
    "CB":  ("Ball progression through passing",  "PAdj Defensive duels won per 90"),
    "FB":  ("Progressive runs per 90",            "Accurate crosses per received pass"),
    "MID": ("Defensive duels per 90",             "Ball progression through passing"),
    "W":   ("Progressive runs per 90",            "xA per 90"),
    "ST":  ("xG per 90",                          "xA per 90"),
}

# ── Archetype position group mapping (also in archetypes.py) ─────────────────
ARCHETYPE_POSITION_GROUPS = {
    "CB":  ["CB", "LCB", "RCB"],
    "FB":  ["LB", "RB", "LWB", "RWB"],
    "MID": ["DMF", "LDMF", "RDMF", "LCMF", "RCMF", "AMF"],
    "W":   ["LW", "RW", "LWF", "RWF", "LAMF", "RAMF"],
    "ST":  ["CF"],
}

# Map scoring position_group label → archetype group → scatter axis key
POSITION_TO_ARCHETYPE_GROUP = {
    "Centre-Back":           "CB",
    "Right-Back":            "FB",
    "Left-Back":             "FB",
    "Defensive Midfielder":  "MID",
    "Central Midfielder":    "MID",
    "Attacking Midfielder":  "MID",
    "Right Winger":          "W",
    "Left Winger":           "W",
    "Winger":                "W",
    "Striker":               "ST",
}
