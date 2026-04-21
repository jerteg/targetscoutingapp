"""
pvp.py — Possession Value Proxy model voor Wyscout aggregated stats

Een xT/VAEP/OBV-geïnspireerde scoring composite voor scouting, gebouwd op
per-90 aggregated statistieken (geen event data vereist).

LET OP: dit is GEEN echte xT, VAEP of OBV. Voor die modellen heb je
Wyscout Events Pack (of vergelijkbaar) nodig plus socceraction library.
PVP imiteert de filosofie ("waardeer acties naar hun waarschijnlijke
impact op scoren/tegenscoren") met wat we wél hebben.

Gebruik:
    from pvp import PVPModel
    m = PVPModel.from_csv("data.csv")
    m.profile("Pedri")
    m.find_similar("Pedri", n=10, max_age=25)
    m.shortlist(position="CM", min_pvp_percentile=90, max_age=23)
"""

import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATIE
# ============================================================

POSITION_GROUPS = {
    'GK': ['GK'],
    'CB': ['LCB', 'RCB', 'CB'],
    'FB': ['LB', 'RB', 'LWB', 'RWB'],
    'CM': ['LCMF', 'RCMF', 'DMF', 'LDMF', 'RDMF', 'AMF', 'LAMF', 'RAMF'],
    'WG': ['LW', 'RW', 'LWF', 'RWF'],
    'CF': ['CF', 'SS'],
}

# Rol-gewichten: (PROG, CREATE, DEF)
# Verdedigers: def-heavy. Spitsen: create-heavy. Middenvelders: progression.
ROLE_WEIGHTS = {
    'CB': (0.35, 0.10, 0.55),
    'FB': (0.40, 0.25, 0.35),
    'CM': (0.45, 0.30, 0.25),
    'WG': (0.30, 0.55, 0.15),
    'CF': (0.15, 0.70, 0.15),
}

# League strength — ruwe schatting op basis van UEFA/FIFA coefficients +
# transfer market realiteit. Kalibreer dit op jouw interne beoordelingen.
LEAGUE_STRENGTH = {
    # Tier 1
    'Premier League': 1.00, 'La Liga': 0.95,
    'Bundesliga': 0.92, 'Italian Serie A': 0.92, 'Ligue 1': 0.88,
    # Tier 2
    'Eredivisie': 0.78, 'Primeira Liga': 0.78, 'Championship': 0.78,
    'Super Lig': 0.75, 'Pro League': 0.74,
    # Tier 3
    'Liga Profesional': 0.72, 'Serie A BRA': 0.72,
    'Swiss Super League': 0.70, 'Segunda Division': 0.70,
    'MLS': 0.68, 'Superligaen': 0.65, 'Ekstraklasa': 0.65,
    # Tier 4
    'Eliteserien': 0.60, 'Liga Pro': 0.55,
}
DEFAULT_LEAGUE_COEF = 0.60


# ============================================================
# HELPERS
# ============================================================

def to_num(series):
    """Robuuste numeric conversie. Handelt Europese komma-decimalen af. NaN → 0."""
    s = series.astype(str).str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce').fillna(0.0)

def safe(series):
    """to_num + clip negatieve waarden (voor counts die niet negatief horen)."""
    return to_num(series).clip(lower=0)

def primary_position(pos_str):
    if pd.isna(pos_str):
        return "UNK"
    return pos_str.split(",")[0].strip()

def assign_group(p):
    for g, members in POSITION_GROUPS.items():
        if p in members:
            return g
    return 'CM'


# ============================================================
# COMPONENT CALCULATIONS
# ============================================================

def component_progression(df):
    """Ball Progression: beweeg bal naar gevaarlijker zones.
    Kern van xT. Gewichten prefereren acties dichter bij het doel."""
    prog_pass = safe(df['Progressive passes per 90']) * safe(df['Accurate progressive passes, %'])/100
    prog_run  = safe(df['Progressive runs per 90'])
    deep_comp = safe(df['Deep completions per 90'])
    through   = safe(df['Through passes per 90']) * safe(df['Accurate through passes, %'])/100
    final3rd  = safe(df['Passes to final third per 90']) * safe(df['Accurate passes to final third, %'])/100
    dribbles  = safe(df['Dribbles per 90']) * safe(df['Successful dribbles, %'])/100
    return (3.0 * deep_comp + 2.5 * through + 1.5 * prog_pass
          + 1.5 * prog_run + 1.0 * final3rd + 1.2 * dribbles)

def component_create(df):
    """Chance Creation: directe dreiging (eigen schoten + creatie voor teamgenoten).
    xG/xA × 10 om magnitude te matchen met andere per-90 counts."""
    return (10 * safe(df['xG per 90'])
          + 10 * safe(df['xA per 90'])
          + 1.5 * safe(df['Key passes per 90'])
          + 1.0 * safe(df['Passes to penalty area per 90']) * safe(df['Accurate passes to penalty area, %'])/100
          + 0.8 * safe(df['Touches in box per 90'])
          + 1.0 * safe(df['Smart passes per 90']) * safe(df['Accurate smart passes, %'])/100)

def component_defense(df):
    """Defensive Value: verlaag P(concede).
    PAdj (possession-adjusted) voorkomt dat spelers in slechte teams onterecht hoog scoren."""
    return (3.0 * safe(df['Shots blocked per 90'])
          + 1.5 * safe(df['PAdj Interceptions'])
          + 1.2 * safe(df['PAdj Sliding tackles'])
          + 1.0 * safe(df['Defensive duels per 90']) * safe(df['Defensive duels won, %'])/100
          + 0.8 * safe(df['Aerial duels per 90']) * safe(df['Aerial duels won, %'])/100)

def component_gk(df):
    """GK-specifieke waarde. Prevented goals is de VAEP-achtige metric hier."""
    return (15 * to_num(df['Prevented goals per 90'])   # kan negatief zijn!
          + 3  * safe(df['Save rate, %'])/100
          + 1.5 * safe(df['Accurate long passes, %'])/100 * safe(df['Long passes per 90'])
          + 0.5 * safe(df['Exits per 90']))


# ============================================================
# MAIN CLASS
# ============================================================

class PVPModel:
    """Possession Value Proxy model.

    Attributes:
        df (pd.DataFrame): dataframe met alle spelers + berekende PVP-kolommen.
    """

    def __init__(self, df):
        self.df = df.copy()
        self._build()

    @classmethod
    def from_csv(cls, path):
        return cls(pd.read_csv(path))

    def _build(self):
        df = self.df

        # Positie-indeling
        df['PrimaryPos'] = df['Position'].apply(primary_position)
        df['PosGroup'] = df['PrimaryPos'].apply(assign_group)

        # Componenten (raw)
        df['PROG']     = component_progression(df)
        df['CREATE']   = component_create(df)
        df['DEF']      = component_defense(df)
        df['GK_VALUE'] = component_gk(df)

        # Z-scores binnen positiegroep
        for col in ['PROG', 'CREATE', 'DEF', 'GK_VALUE']:
            df[f'{col}_z'] = df.groupby('PosGroup')[col].transform(
                lambda g: (g - g.mean()) / g.std() if g.std() > 0 else g * 0
            )

        # Composite
        def composite(row):
            if row['PosGroup'] == 'GK':
                return row['GK_VALUE_z']
            w = ROLE_WEIGHTS.get(row['PosGroup'], (0.4, 0.35, 0.25))
            return w[0]*row['PROG_z'] + w[1]*row['CREATE_z'] + w[2]*row['DEF_z']
        df['PVP_raw'] = df.apply(composite, axis=1)

        # League adjustment
        df['LeagueCoef'] = df['League'].apply(lambda L: LEAGUE_STRENGTH.get(L, DEFAULT_LEAGUE_COEF))
        df['PVP'] = df['PVP_raw'] * df['LeagueCoef']

        # Percentile rank binnen positiegroep
        df['PVP_percentile'] = df.groupby('PosGroup')['PVP'].rank(pct=True) * 100

        self.df = df

    # -------------------- SCOUTING API --------------------

    def profile(self, name_contains, position_group=None):
        """Return dataframe met profiel(en) die matchen op naam."""
        mask = self.df['Player'].str.contains(name_contains, case=False, na=False)
        if position_group:
            mask &= self.df['PosGroup'] == position_group
        cols = ['Player', 'Team', 'League', 'Position', 'PosGroup', 'Age',
                'Minutes played', 'LeagueCoef',
                'PROG_z', 'CREATE_z', 'DEF_z', 'PVP', 'PVP_percentile']
        return self.df[mask][cols]

    def find_similar(self, player_name, n=10, same_position=True,
                     min_minutes=1000, max_age=None, min_age=None,
                     exclude_same_team=False, min_league_coef=None):
        """Vind vergelijkbare profielen in z-score ruimte (PROG, CREATE, DEF)."""
        match = self.df[self.df['Player'].str.contains(player_name, case=False, na=False)]
        if len(match) == 0:
            raise ValueError(f"Speler '{player_name}' niet gevonden")
        target = match.iloc[0]

        pool = self.df[self.df['Minutes played'] >= min_minutes].copy()
        if same_position:
            pool = pool[pool['PosGroup'] == target['PosGroup']]
        if max_age is not None:
            pool = pool[pool['Age'] <= max_age]
        if min_age is not None:
            pool = pool[pool['Age'] >= min_age]
        if exclude_same_team:
            pool = pool[pool['Team'] != target['Team']]
        if min_league_coef is not None:
            pool = pool[pool['LeagueCoef'] >= min_league_coef]
        pool = pool[pool['Player'] != target['Player']]

        dist = np.sqrt(
            (pool['PROG_z']   - target['PROG_z'])**2 +
            (pool['CREATE_z'] - target['CREATE_z'])**2 +
            (pool['DEF_z']    - target['DEF_z'])**2
        )
        pool = pool.assign(Similarity=1/(1+dist)).sort_values('Similarity', ascending=False)
        return target, pool.head(n)[['Player', 'Team', 'League', 'Age',
                                      'Minutes played', 'Market value',
                                      'PROG_z', 'CREATE_z', 'DEF_z', 'PVP',
                                      'Similarity']]

    def shortlist(self, position_group=None, min_pvp_percentile=80,
                  max_age=None, min_minutes=1000, leagues=None,
                  min_league_coef=None, top_n=30):
        """Genereer een scouting shortlist op basis van filters."""
        pool = self.df[self.df['Minutes played'] >= min_minutes].copy()
        if position_group:
            pool = pool[pool['PosGroup'] == position_group]
        if max_age is not None:
            pool = pool[pool['Age'] <= max_age]
        if leagues:
            pool = pool[pool['League'].isin(leagues)]
        if min_league_coef is not None:
            pool = pool[pool['LeagueCoef'] >= min_league_coef]
        pool = pool[pool['PVP_percentile'] >= min_pvp_percentile]
        cols = ['Player', 'Team', 'League', 'Position', 'Age', 'Minutes played',
                'Market value', 'PROG_z', 'CREATE_z', 'DEF_z',
                'PVP', 'PVP_percentile']
        return pool.nlargest(top_n, 'PVP')[cols]

    def export(self, path):
        """Schrijf alle resultaten naar CSV."""
        cols = ['Player', 'Team', 'League', 'Position', 'PosGroup', 'Age',
                'Minutes played', 'Market value', 'LeagueCoef',
                'PROG', 'CREATE', 'DEF', 'GK_VALUE',
                'PROG_z', 'CREATE_z', 'DEF_z', 'GK_VALUE_z',
                'PVP_raw', 'PVP', 'PVP_percentile']
        self.df[cols].sort_values('PVP', ascending=False).to_csv(path, index=False)
