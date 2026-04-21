import pandas as pd

# ── Template configs per position group ───────────────────────────────────────

template_config = {
    "Right-Back": {
        "positions": ['RB', 'RWB'],
        "stats": [
            'xG per 90', 'xA per 90',
            'Accurate crosses per received pass', 'Accurate crosses, %',
            'Shot assists per 90',
            'Successful dribbles per received pass',
            'Progressive runs per received pass',
            'Ball progression through passing',
            'Passing accuracy (prog/1/3/forw)',
            'PAdj Defensive duels won per 90', 'Defensive duels won, %',
            'PAdj Aerial duels won per 90', 'Aerial duels won, %',
            'PAdj Interceptions'
        ],
        "label": "RB Template",
    },
    "Centre-Back": {
        "positions": ['CB', 'RCB', 'LCB'],
        "stats": [
            'Progressive runs per received pass',
            'Ball progression through passing',
            'Passing accuracy (prog/1/3/forw)',
            'PAdj Defensive duels won per 90', 'Defensive duels won, %',
            'PAdj Aerial duels won per 90', 'Aerial duels won, %',
            'PAdj Interceptions',
            'Fouls per 90'
        ],
        "label": "CB Template",
        "negative_stats": ['Fouls per 90'],
    },
    "Left-Back": {
        "positions": ['LB', 'LWB'],
        "stats": [
            'xG per 90', 'xA per 90',
            'Accurate crosses per received pass', 'Accurate crosses, %',
            'Shot assists per 90',
            'Successful dribbles per received pass',
            'Progressive runs per received pass',
            'Ball progression through passing',
            'Passing accuracy (prog/1/3/forw)',
            'PAdj Defensive duels won per 90', 'Defensive duels won, %',
            'PAdj Aerial duels won per 90', 'Aerial duels won, %',
            'PAdj Interceptions'
        ],
        "label": "LB Template",
    },
    "Defensive Midfielder": {
        "positions": ['RDMF', 'LDMF', 'DMF'],
        "stats": [
            'xA per 90',
            'Successful dribbles, %',
            'Progressive runs per received pass',
            'Ball progression through passing',
            'Passing accuracy (prog/1/3/forw)',
            'PAdj Defensive duels won per 90', 'Defensive duels won, %',
            'PAdj Aerial duels won per 90', 'Aerial duels won, %',
            'PAdj Successful defensive actions per 90',
            'PAdj Interceptions',
            'Fouls per 90'
        ],
        "label": "DM Template",
        "negative_stats": ['Fouls per 90'],
    },
    "Central Midfielder": {
        "positions": ['RDMF', 'LDMF', 'RCMF', 'LCMF'],
        "stats": [
            'xG per 90', 'Touches in box per 90', 'xA per 90',
            'Key passes per pass', 'Through passes per pass',
            'Successful dribbles per received pass', 'Successful dribbles, %',
            'Progressive runs per received pass',
            'Ball progression through passing',
            'Passing accuracy (prog/1/3/forw)',
            'PAdj Defensive duels won per 90', 'Defensive duels won, %',
            'PAdj Aerial duels won per 90', 'Aerial duels won, %',
            'PAdj Successful defensive actions per 90',
            'PAdj Interceptions'
        ],
        "label": "CM Template",
    },
    "Attacking Midfielder": {
        "positions": ['RCMF', 'LCMF', 'AMF'],
        "stats": [
            'xG per 90', 'xG per shot', 'Finishing',
            'Touches in box per 90', 'xA per 90',
            'Key passes per pass', 'Through passes per pass',
            'Successful dribbles per received pass', 'Successful dribbles, %',
            'Progressive runs per received pass',
            'Ball progression through passing',
            'Passing accuracy (prog/1/3/forw)',
            'Defensive duels won, %', 'Aerial duels won, %',
            'PAdj Interceptions'
        ],
        "label": "AM Template",
    },
    "Winger": {
        "positions": ['LWF', 'LAMF', 'LW', 'RWF', 'RAMF', 'RW'],
        "stats": [
            'Finishing', 'xG per 90', 'xG per shot',
            'Touches in box per 90', 'xA per 90',
            'Shot assists per 90', 'Key passes per pass',
            'Accurate crosses per received pass', 'Accurate crosses, %',
            'Successful dribbles per received pass',
            'Progressive runs per received pass',
            'Ball progression through passing',
            'Passing accuracy (prog/1/3/forw)',
            'Defensive duels won, %', 'Aerial duels won, %',
            'PAdj Interceptions'
        ],
        "label": "Winger Template",
    },
    "Striker": {
        "positions": ['CF'],
        "stats": [
            'xG per 90', 'xG per shot', 'Finishing',
            'Touches in box per 90', 'xA per 90',
            'Key passes per pass',
            'Successful dribbles per received pass', 'Successful dribbles, %',
            'Progressive runs per received pass',
            'Passing accuracy (prog/1/3/forw)',
            'Defensive duels won, %', 'Aerial duels won, %',
            'PAdj Interceptions'
        ],
        "label": "Striker Template",
    },
}

# ── Roles per position group ───────────────────────────────────────────────────

role_config = {
    "Right-Back": {
        "Defensive Full-Back": {
            "description": "Defensive-minded full-back focused on stopping direct opponent",
            "stats": {
                'PAdj Defensive duels won per 90': 0.150,
                'Defensive duels won, %':          0.175,
                'PAdj Aerial duels won per 90':    0.150,
                'Aerial duels won, %':             0.175,
                'PAdj Interceptions':              0.150,
                'Completed progressive passes per 90': 0.100,
                'Progressive runs per received pass':  0.100,
            },
        },
        "Attacking Full-Back": {
            "description": "Attacking-minded full-back focused on contributing to the attacking play",
            "stats": {
                'Defensive duels won, %':              0.100,
                'Aerial duels won, %':                 0.100,
                'Completed progressive passes per 90': 0.175,
                'Progressive runs per received pass':  0.175,
                'xA per 90':                           0.200,
                'Accurate crosses per received pass':  0.125,
                'Deep completions per 90':             0.125,
            },
        },
        "Inverted Full-Back": {
            "description": "Full-back focused on adding value in the build-up play",
            "stats": {
                'Defensive duels won, %':                  0.125,
                'PAdj Interceptions':                      0.125,
                'Completed progressive passes per 90':     0.250,
                'Completed passes to final third per 90':  0.250,
                'Progressive runs per received pass':      0.100,
                'Accurate progressive passes, %':          0.075,
                'Accurate passes to final third, %':       0.075,
            },
        },
    },
    "Left-Back": {
        "Defensive Full-Back": {
            "description": "Defensive-minded full-back focused on stopping direct opponent",
            "stats": {
                'PAdj Defensive duels won per 90': 0.150,
                'Defensive duels won, %':          0.175,
                'PAdj Aerial duels won per 90':    0.150,
                'Aerial duels won, %':             0.175,
                'PAdj Interceptions':              0.150,
                'Completed progressive passes per 90': 0.100,
                'Progressive runs per received pass':  0.100,
            },
        },
        "Attacking Full-Back": {
            "description": "Attacking-minded full-back focused on contributing to the attacking play",
            "stats": {
                'Defensive duels won, %':              0.100,
                'Aerial duels won, %':                 0.100,
                'Completed progressive passes per 90': 0.175,
                'Progressive runs per received pass':  0.175,
                'xA per 90':                           0.200,
                'Accurate crosses per received pass':  0.125,
                'Deep completions per 90':             0.125,
            },
        },
        "Inverted Full-Back": {
            "description": "Full-back focused on adding value in the build-up play",
            "stats": {
                'Defensive duels won, %':                  0.125,
                'PAdj Interceptions':                      0.125,
                'Completed progressive passes per 90':     0.250,
                'Completed passes to final third per 90':  0.250,
                'Progressive runs per received pass':      0.100,
                'Accurate progressive passes, %':          0.075,
                'Accurate passes to final third, %':       0.075,
            },
        },
    },
    "Centre-Back": {
        "Ball-Playing CB": {
            "description": "Centre-back focused on playing out from the back",
            "stats": {
                'Defensive duels won, %':                  0.125,
                'Aerial duels won, %':                     0.125,
                'Completed progressive passes per 90':     0.250,
                'Completed passes to final third per 90':  0.250,
                'Progressive runs per received pass':      0.100,
                'Accurate progressive passes, %':          0.075,
                'Accurate passes to final third, %':       0.075,
            },
        },
        "Sweeper": {
            "description": "Centre-back focused on winning duels in and around the box",
            "stats": {
                'PAdj Defensive duels won per 90': 0.200,
                'Defensive duels won, %':          0.200,
                'PAdj Aerial duels won per 90':    0.200,
                'Aerial duels won, %':             0.200,
                'PAdj Interceptions':              0.200,
            },
        },
    },
    "Defensive Midfielder": {
        "Ball-Winning Midfielder": {
            "description": "Defensive midfielder focused on intercepting the ball and winning duels",
            "stats": {
                'PAdj Defensive duels won per 90':         0.125,
                'Defensive duels won, %':                  0.125,
                'PAdj Aerial duels won per 90':            0.125,
                'Aerial duels won, %':                     0.125,
                'PAdj Interceptions':                      0.250,
                'Completed progressive passes per 90':     0.125,
                'Completed passes to final third per 90':  0.125,
            },
        },
        "Deep-Lying Playmaker": {
            "description": "Defensive midfielder focused on progressing play",
            "stats": {
                'Defensive duels won, %':                  0.125,
                'Aerial duels won, %':                     0.125,
                'Completed progressive passes per 90':     0.250,
                'Completed passes to final third per 90':  0.250,
                'Progressive runs per received pass':      0.100,
                'Accurate progressive passes, %':          0.075,
                'Accurate passes to final third, %':       0.075,
            },
        },
    },
    "Central Midfielder": {
        "Advanced Playmaker": {
            "description": "Central midfielder focused on creating opportunities",
            "stats": {
                'xG per 90':                           0.100,
                'xA per 90':                           0.250,
                'Successful dribbles, %':              0.125,
                'Offensive duels won, %':              0.125,
                'Completed passes to penalty area per 90': 0.100,
                'Key passes per pass':                 0.200,
                'Through passes per pass':             0.100,
            },
        },
        "Deep-Lying Playmaker": {
            "description": "Central midfielder focused on progressing play",
            "stats": {
                'Defensive duels won, %':                  0.125,
                'Aerial duels won, %':                     0.125,
                'Completed progressive passes per 90':     0.250,
                'Completed passes to final third per 90':  0.250,
                'Progressive runs per received pass':      0.100,
                'Accurate progressive passes, %':          0.075,
                'Accurate passes to final third, %':       0.075,
            },
        },
        "Box-to-Box Midfielder": {
            "description": "All-action central midfielder",
            "stats": {
                'PAdj Defensive duels won per 90':         0.125,
                'PAdj Aerial duels won per 90':            0.125,
                'PAdj Interceptions':                      0.125,
                'Completed progressive passes per 90':     0.125,
                'Completed passes to final third per 90':  0.125,
                'Progressive runs per received pass':      0.125,
                'xG per 90':                               0.125,
                'xA per 90':                               0.125,
            },
        },
        "Ball-Winning Midfielder": {
            "description": "Central midfielder focused on intercepting the ball and winning duels",
            "stats": {
                'PAdj Defensive duels won per 90':         0.125,
                'Defensive duels won, %':                  0.125,
                'PAdj Aerial duels won per 90':            0.125,
                'Aerial duels won, %':                     0.125,
                'PAdj Interceptions':                      0.250,
                'Completed progressive passes per 90':     0.125,
                'Completed passes to final third per 90':  0.125,
            },
        },
    },
    "Attacking Midfielder": {
        "Advanced Playmaker": {
            "description": "Attacking midfielder focused on creating opportunities",
            "stats": {
                'xG per 90':                               0.100,
                'xA per 90':                               0.250,
                'Successful dribbles, %':                  0.125,
                'Offensive duels won, %':                  0.125,
                'Completed passes to penalty area per 90': 0.100,
                'Key passes per pass':                     0.200,
                'Through passes per pass':                 0.100,
            },
        },
        "Box Crasher": {
            "description": "Attacking midfielder focused on getting into the box and scoring",
            "stats": {
                'xG per 90':            0.200,
                'xG per shot':          0.300,
                'Finishing':            0.300,
                'Touches in box per 90': 0.150,
                'xA per 90':            0.050,
            },
        },
    },
    "Winger": {
        "Goalscoring Winger": {
            "description": "Winger focused on getting into the box and scoring",
            "stats": {
                'xG per 90':             0.200,
                'xG per shot':           0.300,
                'Finishing':             0.300,
                'Touches in box per 90': 0.150,
                'xA per 90':             0.050,
            },
        },
        "Creative Winger": {
            "description": "Winger focused on creating opportunities",
            "stats": {
                'xG per 90':                               0.100,
                'xA per 90':                               0.250,
                'Successful dribbles, %':                  0.250,
                'Completed passes to penalty area per 90': 0.100,
                'Key passes per pass':                     0.200,
                'Through passes per pass':                 0.100,
            },
        },
        "Direct Winger": {
            "description": "Winger focused on taking on opponents",
            "stats": {
                'Progressive runs per received pass':     0.225,
                'Successful dribbles per received pass':  0.225,
                'Successful dribbles, %':                 0.075,
                'Offensive duels won, %':                 0.075,
                'xG per 90':                              0.150,
                'xA per 90':                              0.150,
                'Accurate crosses per received pass':     0.100,
            },
        },
    },
    "Striker": {
        "Poacher": {
            "description": "Striker focused on scoring goals only",
            "stats": {
                'xG per 90':             0.200,
                'xG per shot':           0.250,
                'Finishing':             0.400,
                'Touches in box per 90': 0.150,
            },
        },
        "Target Man": {
            "description": "Striker focused on holding up play and scoring goals",
            "stats": {
                'xG per 90':                    0.100,
                'xG per shot':                  0.150,
                'Finishing':                    0.150,
                'PAdj Aerial duels won per 90': 0.200,
                'Aerial duels won, %':          0.200,
                'xA per 90':                    0.100,
                'Touches in box per 90':        0.100,
            },
        },
        "Deep-Lying Forward": {
            "description": "Striker focused on contributing on the ball and scoring goals",
            "stats": {
                'xG per 90':                         0.150,
                'xG per shot':                       0.150,
                'Finishing':                         0.150,
                'xA per 90':                         0.250,
                'Key passes per pass':               0.200,
                'Successful dribbles per received pass': 0.100,
            },
        },
    },
}

report_template = {

    "Goalscoring": {
        "stats": [
            "Non-penalty goals per 90",
            "xG per 90", 
            "xG per shot", 
            "Finishing", 
            "Shots per 90", 
            "Shots on target, %",
            "Touches in box per 90"
        ],
        "weights": {
            "Non-penalty goals per 90": 0.20,
            "xG per 90": 0.20, 
            "xG per shot": 0.20, 
            "Finishing": 0.25, 
            "Shots per 90": 0.05, 
            "Shots on target, %": 0.05,
            "Touches in box per 90": 0.05
        }
    },

    "Chance creation": {
        "stats": [
            "Assists per 90", 
            "xA per 90", 
            "Shot assists per 90", 
            "Key passes per pass", 
            "Through passes per pass",
            "Accurate crosses per received pass", 
            "Accurate crosses, %"
        ],
        "weights": {
            "Assists per 90": 0.2, 
            "xA per 90": 0.3, 
            "Shot assists per 90": 0.2, 
            "Key passes per pass": 0.1, 
            "Through passes per pass": 0.1,
            "Accurate crosses per received pass": 0.05, 
            "Accurate crosses, %": 0.05
        }
    },

    "Dribbling": {
        "stats": [
            "Successful dribbles per received pass", 
            "Successful dribbles, %",
            "Offensive duels won, %",
            "Progressive runs per received pass"
        ],
        "weights": {
            "Successful dribbles per received pass": 0.25, 
            "Successful dribbles, %": 0.25,
            "Offensive duels won, %": 0.25,
            "Progressive runs per received pass": 0.25
        }
    },

    "Passing": {
        "stats": [
            "Completed progressive passes per 90", 
            "Accurate progressive passes, %",
            "Completed passes to final third per 90", 
            "Accurate passes to final third, %",
            "Completed passes to penalty area per 90", 
            "Accurate passes to penalty area, %", 
            "Deep completions per 90"
        ],
        "weights": {
            "Completed progressive passes per 90": 0.15, 
            "Accurate progressive passes, %": 0.15,
            "Completed passes to final third per 90": 0.15, 
            "Accurate passes to final third, %": 0.15,
            "Completed passes to penalty area per 90": 0.15, 
            "Accurate passes to penalty area, %": 0.15, 
            "Deep completions per 90": 0.1
        }
    },

    "Defending": {
        "stats": [
            "PAdj Defensive duels won per 90",
            "Defensive duels won, %",
            "PAdj Aerial duels won per 90",
            "Aerial duels won, %",
            "PAdj Interceptions",
            "PAdj Successful defensive actions per 90",
            "Fouls per 90"
            ],
        "negative_stats": ['Fouls per 90'],
        "weights": {
            "PAdj Defensive duels won per 90": 0.15,
            "Defensive duels won, %": 0.25,
            "PAdj Aerial duels won per 90": 0.15,
            "Aerial duels won, %": 0.25,
            "PAdj Interceptions": 0.1,
            "PAdj Successful defensive actions per 90": 0.05,
            "Fouls per 90": 0.05
        }
    }
}

position_category_weights = {

    "Striker": {
        "Goalscoring": 0.55,
        "Chance creation": 0.2,
        "Dribbling": 0.1,
        "Passing": 0.1,
        "Defending": 0.05
    },

    "Winger": {
        "Goalscoring": 0.35,
        "Chance creation": 0.3,
        "Dribbling": 0.15,
        "Passing": 0.15,
        "Defending": 0.05
    },

    "Attacking Midfielder": {
        "Goalscoring": 0.25,
        "Chance creation": 0.35,
        "Dribbling": 0.1,
        "Passing": 0.2,
        "Defending": 0.1
    },

    "Central Midfielder": {
        "Goalscoring": 0.20,
        "Chance creation": 0.25,
        "Dribbling": 0.15,
        "Passing": 0.25,
        "Defending": 0.15
    },

    "Defensive Midfielder": {
        "Goalscoring": 0.05,
        "Chance creation": 0.1,
        "Dribbling": 0.1,
        "Passing": 0.40,
        "Defending": 0.35
    },

    "Centre-Back": {
        "Goalscoring": 0.05,
        "Chance creation": 0.05,
        "Dribbling": 0.05,
        "Passing": 0.35,
        "Defending": 0.5
    },

    "Right-Back": {
        "Goalscoring": 0.05,
        "Chance creation": 0.25,
        "Dribbling": 0.1,
        "Passing": 0.3,
        "Defending": 0.3
    },

    "Left-Back": {
        "Goalscoring": 0.05,
        "Chance creation": 0.25,
        "Dribbling": 0.1,
        "Passing": 0.3,
        "Defending": 0.3
    }
}


# ── Position groups & mappings ────────────────────────────────────────────────

position_groups = {
    "Right-Back":           ['RB', 'RWB'],
    "Centre-Back":          ['CB', 'RCB', 'LCB'],
    "Left-Back":            ['LB', 'LWB'],
    "Defensive Midfielder": ['DMF', 'RDMF', 'LDMF'],
    "Central Midfielder":   ['RDMF', 'LDMF', 'RCMF', 'LCMF'],
    "Attacking Midfielder": ['RCMF', 'LCMF', 'AMF'],
    "Winger":               ['LW', 'RW', 'LWF', 'RWF', 'LAMF', 'RAMF'],
    "Striker":              ['CF'],
}

position_to_template = {pg: pg for pg in position_groups}

position_labels = {
    "Left Wing":            ['LWF', 'LAMF', 'LW'],
    "Right Wing":           ['RWF', 'RAMF', 'RW'],
    "Striker":              ['CF'],
    "Right-Back":           ['RB', 'RWB'],
    "Left-Back":            ['LB', 'LWB'],
    "Centre-Back":          ['CB', 'RCB', 'LCB'],
    "Central Midfielder":   ['RDMF', 'RCMF', 'LDMF', 'LCMF'],
    "Attacking Midfielder": ['AMF'],
    "Defensive Midfielder": ['DMF'],
}

position_map = {
    pos: label
    for label, positions in position_labels.items()
    for pos in positions
}

# ── League multipliers ────────────────────────────────────────────────────────

TOP5_LEAGUES = {"Premier League", "La Liga", "Italian Serie A", "Bundesliga", "Ligue 1"}

NEXT14_LEAGUES = {
    "Pro League", "Primeira Liga", "Liga Profesional", "Serie A BRA",
    "Championship", "Superligaen", "Ekstraklasa", "MLS", "Prva HNL",
    "Eliteserien", "Super Lig", "Eredivisie", "Liga Pro", "Swiss Super League",
    "Segunda Division",
}

LEAGUE_MULTIPLIERS_ALL = {
    # Scenario A — proportioneel herschaald van [0.836, 1.000] naar [0.720, 1.000]
    # Methode: t = (v - 0.836) / (1.000 - 0.836); new_v = 0.720 + t * (1.000 - 0.720)
    "Premier League":     1.0000,
    "La Liga":            0.9129,
    "Italian Serie A":    0.8907,
    "Bundesliga":         0.8890,
    "Ligue 1":            0.8737,
    "Pro League":         0.8020,
    "Primeira Liga":      0.7968,
    "Liga Profesional":   0.7968,
    "Serie A BRA":        0.7917,
    "Championship":       0.7900,
    "Superligaen":        0.7746,
    "Ekstraklasa":        0.7610,
    "MLS":                0.7576,
    "Prva HNL":           0.7559,
    "Eliteserien":        0.7524,
    "Super Lig":          0.7439,
    "Eredivisie":         0.7439,
    "Liga Pro":           0.7422,
    "Segunda Division":   0.7388,
    "Swiss Super League": 0.7200,
}

LEAGUE_MULTIPLIERS_NEXT14 = {
    # Scenario A — proportioneel herschaald van [0.945, 1.000] naar [0.720, 1.000]
    "Pro League":         1.0000,
    "Primeira Liga":      0.9796,
    "Liga Profesional":   0.9796,
    "Serie A BRA":        0.9644,
    "Championship":       0.9542,
    "Superligaen":        0.9033,
    "Ekstraklasa":        0.8625,
    "MLS":                0.8473,
    "Prva HNL":           0.8422,
    "Eliteserien":        0.8269,
    "Super Lig":          0.8015,
    "Eredivisie":         0.8015,
    "Liga Pro":           0.7964,
    "Segunda Division":   0.7862,
    "Swiss Super League": 0.7200,
}


# ── Radar categorieën ─────────────────────────────────────────────────────────
# Goalscoring:      4 stats
# Chance Creation:  4 stats
# Dribbling:        3 stats
# Passing:          4 stats
# Defending:        5 stats
# Totaal:          20 stats

RADAR_CATEGORIES = {
    "Goalscoring": {
        "stats": [
            "xG per 90",
            "xG per shot",
            "Finishing",
            "Touches in box per 90",
        ],
        "negative_stats": [],
        "color": "#c0392b",
    },
    "Chance Creation": {
        "stats": [
            "xA per 90",
            "Shot assists per 90",
            "Key passes per pass",
            "Accurate crosses per received pass",
        ],
        "negative_stats": [],
        "color": "#e67e22",
    },
    "Dribbling": {
        "stats": [
            "Successful dribbles per received pass",
            "Successful dribbles, %",
            "Offensive duels won, %",
            "Progressive runs per received pass",
        ],
        "negative_stats": [],
        "color": "#d4ac0d",
    },
    "Passing": {
        "stats": [
            "Completed progressive passes per 90",
            "Completed passes to final third per 90",
            "Deep completions per 90",
            "Passing accuracy (prog/1/3/forw)",
        ],
        "negative_stats": [],
        "color": "#27ae60",
    },
    "Defending": {
        "stats": [
            "PAdj Defensive duels won per 90",
            "Defensive duels won, %",
            "PAdj Aerial duels won per 90",
            "Aerial duels won, %",
            "PAdj Interceptions",
        ],
        "negative_stats": [],
        "color": "#2980b9",
    },
}

# ── Flat geordende lijst ──────────────────────────────────────────────────────
ALL_RADAR_STATS = [
    stat
    for cat_data in RADAR_CATEGORIES.values()
    for stat in cat_data["stats"]
]

# ── Kleur per stat (op basis van categorie) ───────────────────────────────────
STAT_CATEGORY_COLORS = {
    stat: cat_data["color"]
    for cat_data in RADAR_CATEGORIES.values()
    for stat in cat_data["stats"]
}

# ── Categorie per stat ────────────────────────────────────────────────────────
STAT_TO_CATEGORY = {
    stat: cat_name
    for cat_name, cat_data in RADAR_CATEGORIES.items()
    for stat in cat_data["stats"]
}

# ── League display names ──────────────────────────────────────────────────────
LEAGUE_DISPLAY_NAMES = {
    # Top 5
    "Premier League":   "ENG-1",
    "La Liga":          "ESP-1",
    "Italian Serie A":  "ITA-1",
    "Bundesliga":       "GER-1",
    "Ligue 1":          "FRA-1",
    # Next 14
    "Pro League":           "BEL-1",
    "Primeira Liga":        "POR-1",
    "Liga Profesional":     "ARG-1",
    "Serie A BRA":          "BRA-1",
    "Championship":         "ENG-2",
    "Superligaen":          "DEN-1",
    "Ekstraklasa":          "POL-1",
    "MLS":                  "USA-1",
    "Prva HNL":             "CRO-1",
    "Eliteserien":          "NOR-1",
    "Super Lig":            "TUR-1",
    "Eredivisie":           "NED-1",
    "Liga Pro":             "ECU-1",
    "Segunda Division":     "ESP-2",
    "Swiss Super League":   "SUI-1",
}

FORCE_FLIP_STATS = {
    # DEFENDING
    "Def duels won /90",
    "Def duels won %",
    "Aerial duels won /90",

    # DRIBBLING
    "Drib.\n/rec pass",
    "Drib. %",
    "Prog. runs\n/rec pass",

    # CHANCE CREATION (links van shot assists)
    "Crosses\n/rec pass",
    "Deep compl.\n/90",
}

