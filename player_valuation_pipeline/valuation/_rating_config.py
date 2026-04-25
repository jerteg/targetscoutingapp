"""
FASE 1 + 2 resultaat: rating_config_v2

Doel: Schone, niet-overlappende rating configuratie.

Composites:
  - Creative_Threat = avg(z(xA/90), z(Shot assists/90))
  - Defensive_Actions_Composite = weighted avg van subcategorieën (interceptions, def duels, etc.)

Algoritmische VIF-filtering:
  - VIF > 10 → automatisch verwijderen
  - Near-duplicaten (|r| > 0.85) → kies één via info-criterium

Elke metric is geclassificeerd als 'production', 'quality', of 'negative'.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from ._templates import template_config
from ._data_processing import load_season_data


# ─────────────────────────────────────────────────────────────────────────────
# METRIC CLASSIFICATIE (Production / Quality / Negative)
# ─────────────────────────────────────────────────────────────────────────────

METRIC_TYPES = {
    # PRODUCTION (volume × kwaliteit, per 90)
    'xG per 90':                                'production',
    'xA per 90':                                'production',
    'Non-penalty goals per 90':                 'production',
    'Assists per 90':                           'production',
    'Shot assists per 90':                      'production',
    'Touches in box per 90':                    'production',
    'Completed progressive passes per 90':      'production',
    'Completed passes to final third per 90':   'production',
    'Completed passes to penalty area per 90':  'production',
    'Deep completions per 90':                  'production',
    'PAdj Defensive duels won per 90':          'production',
    'PAdj Aerial duels won per 90':             'production',
    'PAdj Interceptions':                       'production',
    'PAdj Successful defensive actions per 90': 'production',
    'Ball progression through passing':         'production',

    # QUALITY (efficiency / accuracy / %-based)
    'Finishing':                                'quality',
    'xG per shot':                              'quality',
    'Successful dribbles, %':                   'quality',
    'Successful dribbles per received pass':    'quality',  # per-touch = quality
    'Progressive runs per received pass':       'quality',  # per-touch = quality
    'Accurate crosses, %':                      'quality',
    'Accurate crosses per received pass':       'quality',  # per-touch = quality
    'Accurate progressive passes, %':           'quality',
    'Accurate passes to final third, %':        'quality',
    'Accurate passes to penalty area, %':       'quality',
    'Accurate forward passes, %':               'quality',
    'Passing accuracy (prog/1/3/forw)':         'quality',
    'Defensive duels won, %':                   'quality',
    'Aerial duels won, %':                      'quality',
    'Offensive duels won, %':                   'quality',
    'Shots on target, %':                       'quality',
    'Key passes per pass':                      'quality',
    'Through passes per pass':                  'quality',

    # NEGATIVE
    'Fouls per 90':                             'negative',
}


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE METRICS (berekend uit bestaande metrics)
# ─────────────────────────────────────────────────────────────────────────────

def build_creative_threat(df: pd.DataFrame) -> pd.Series:
    """
    Creative_Threat = avg(z(xA/90), z(Shot assists/90))
    
    Per LIGA berekend, dus z-scores zijn binnen-liga.
    Dit is een PRODUCTION composite.
    """
    result = pd.Series(np.nan, index=df.index, name='Creative_Threat')
    
    for league, group in df.groupby('League'):
        xa = group['xA per 90']
        sa = group['Shot assists per 90']
        
        # Z-scores binnen league
        xa_z = (xa - xa.mean()) / xa.std()
        sa_z = (sa - sa.mean()) / sa.std()
        
        # Gemiddelde van de twee z-scores
        composite = (xa_z + sa_z) / 2
        result.loc[group.index] = composite
    
    return result


# Positie-specifieke weights voor Defensive_Actions_Composite
# Redenering:
#  - CB:  aerial cruciaal (kopduels in eigen box), interceptions belangrijk, def duels minder (staan dieper)
#  - FB:  alle drie ongeveer even belangrijk, maar aerial iets minder (niet hoofdtaak)
#  - DM:  interceptions + def duels meest cruciaal (ball-winning rol), aerial minder
#  - CM:  interceptions belangrijk voor transitie, duels secundair
#  - AM/W/ST: defensief werk = bonus, niet differentiatie — gelijke weights prima

POSITION_DEFENSIVE_WEIGHTS = {
    'Centre-Back': {
        'PAdj Interceptions':              0.30,
        'PAdj Defensive duels won per 90': 0.25,
        'PAdj Aerial duels won per 90':    0.45,
    },
    'Right-Back': {
        'PAdj Interceptions':              0.35,
        'PAdj Defensive duels won per 90': 0.40,
        'PAdj Aerial duels won per 90':    0.25,
    },
    'Left-Back': {
        'PAdj Interceptions':              0.35,
        'PAdj Defensive duels won per 90': 0.40,
        'PAdj Aerial duels won per 90':    0.25,
    },
    'Defensive Midfielder': {
        'PAdj Interceptions':              0.50,
        'PAdj Defensive duels won per 90': 0.35,
        'PAdj Aerial duels won per 90':    0.15,
    },
    'Central Midfielder': {
        'PAdj Interceptions':              0.40,
        'PAdj Defensive duels won per 90': 0.40,
        'PAdj Aerial duels won per 90':    0.20,
    },
    'Attacking Midfielder': {
        'PAdj Interceptions':              0.35,
        'PAdj Defensive duels won per 90': 0.40,
        'PAdj Aerial duels won per 90':    0.25,
    },
    'Winger': {
        'PAdj Interceptions':              0.35,
        'PAdj Defensive duels won per 90': 0.40,
        'PAdj Aerial duels won per 90':    0.25,
    },
    'Striker': {
        'PAdj Interceptions':              0.20,
        'PAdj Defensive duels won per 90': 0.40,
        'PAdj Aerial duels won per 90':    0.40,  # pressing/aerial duels belangrijker voor spits
    },
}


def build_defensive_composite(df: pd.DataFrame,
                               position_weights: dict = None) -> pd.Series:
    """
    Defensive_Actions_Composite = gewogen avg van z-scores van:
      - PAdj Interceptions
      - PAdj Defensive duels won per 90  
      - PAdj Aerial duels won per 90
    
    Weights verschillen PER POSITIE (POSITION_DEFENSIVE_WEIGHTS).
    Z-scores worden berekend BINNEN LIGA.
    
    Let op: Position Label moet al bestaan in df (via position_map uit templates.py).
    """
    if position_weights is None:
        position_weights = POSITION_DEFENSIVE_WEIGHTS
    
    result = pd.Series(np.nan, index=df.index, name='Defensive_Actions_Composite')
    
    # Default weights voor spelers wiens positie niet in de map staat
    default_weights = {
        'PAdj Interceptions': 0.40,
        'PAdj Defensive duels won per 90': 0.40,
        'PAdj Aerial duels won per 90': 0.20,
    }
    
    # Stap 1: bereken z-scores binnen liga voor elk van de 3 metrics
    z_scores = {}
    for metric in ['PAdj Interceptions', 'PAdj Defensive duels won per 90', 'PAdj Aerial duels won per 90']:
        if metric in df.columns:
            z = df.groupby('League')[metric].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else pd.Series(np.nan, index=x.index)
            )
            z_scores[metric] = z
    
    # Stap 2: pas positie-specifieke weights toe
    for pos_label, weights in position_weights.items():
        mask = df['Position Label'] == pos_label
        if not mask.any():
            continue
        
        weighted_sum = pd.Series(0.0, index=df.index[mask])
        total_weight = 0.0
        for metric, w in weights.items():
            if metric in z_scores:
                vals = z_scores[metric].loc[mask].fillna(0)
                weighted_sum = weighted_sum + vals * w
                total_weight += w
        
        if total_weight > 0:
            result.loc[mask] = weighted_sum / total_weight
    
    # Spelers zonder gematchte Position Label → default weights
    unmatched = result.isna()
    if unmatched.any():
        weighted_sum = pd.Series(0.0, index=df.index[unmatched])
        total_weight = 0.0
        for metric, w in default_weights.items():
            if metric in z_scores:
                vals = z_scores[metric].loc[unmatched].fillna(0)
                weighted_sum = weighted_sum + vals * w
                total_weight += w
        if total_weight > 0:
            result.loc[unmatched] = weighted_sum / total_weight
    
    return result


def add_composite_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Voeg composite metrics toe aan de DataFrame."""
    df = df.copy()
    df['Creative_Threat'] = build_creative_threat(df)
    df['Defensive_Actions_Composite'] = build_defensive_composite(df)
    return df


# Composites toevoegen aan metric types
METRIC_TYPES['Creative_Threat'] = 'production'
METRIC_TYPES['Defensive_Actions_Composite'] = 'production'


# ─────────────────────────────────────────────────────────────────────────────
# HERZIENE TEMPLATES — Na handmatige cleanup op basis van VIF bevindingen
# ─────────────────────────────────────────────────────────────────────────────

# Principes:
# 1. xA + Shot assists → Creative_Threat composite
# 2. PAdj Successful defensive actions + interceptions + def duels → Defensive_Actions_Composite
# 3. Near-duplicaten met r > 0.85 worden samengevoegd of één wordt geschrapt
# 4. Nog te valideren met VIF op composite-set

rating_config_v2 = {
    "Striker": {
        "positions": ['CF'],
        "stats": [
            # PRODUCTION
            'xG per 90',             # expected goals (mathematisch: NPG ≈ xG + Finishing, dus NPG zou redundant zijn)
            'Creative_Threat',
            'Touches in box per 90',
            # QUALITY
            'Finishing',             # actual goals - xG (de "overperformance")
            'xG per shot',
            'Successful dribbles per received pass',
            'Aerial duels won, %',
            'Offensive duels won, %',
            # DEFENSIVE (minor voor striker)
            'Defensive_Actions_Composite',
        ],
        "label": "Striker Template v2",
    },
    "Winger": {
        "positions": ['LWF', 'LAMF', 'LW', 'RWF', 'RAMF', 'RW'],
        "stats": [
            # PRODUCTION
            'xG per 90',
            'Creative_Threat',
            'Touches in box per 90',
            # QUALITY - creativiteit & dribbling
            'Finishing',
            'xG per shot',
            'Key passes per pass',
            'Accurate crosses per received pass',
            'Successful dribbles per received pass',  # 1 van de 2 (dribbling + progressive runs r=0.83)
            'Offensive duels won, %',
            # PASSING
            'Passing accuracy (prog/1/3/forw)',
            # DEFENSIVE (minor voor winger)
            'Defensive_Actions_Composite',
        ],
        "label": "Winger Template v2",
    },
    "Attacking Midfielder": {
        "positions": ['RCMF', 'LCMF', 'AMF'],
        "stats": [
            # PRODUCTION
            'xG per 90',
            'Creative_Threat',
            # QUALITY - finishing & creativiteit  
            'Finishing',
            'xG per shot',
            'Key passes per pass',
            'Through passes per pass',
            'Successful dribbles per received pass',  # 1 van 2
            'Successful dribbles, %',
            # PASSING
            'Passing accuracy (prog/1/3/forw)',
            # DEFENSIVE
            'Defensive_Actions_Composite',
        ],
        "label": "AM Template v2",
    },
    "Central Midfielder": {
        "positions": ['RDMF', 'LDMF', 'RCMF', 'LCMF'],
        "stats": [
            # PRODUCTION
            'xG per 90',
            'Creative_Threat',
            'Completed progressive passes per 90',
            # QUALITY
            'Key passes per pass',
            'Through passes per pass',
            'Successful dribbles per received pass',
            'Passing accuracy (prog/1/3/forw)',
            # DEFENSIVE (composite vervangt de 3 losse metrics)
            'Defensive_Actions_Composite',
            'Defensive duels won, %',   # kwaliteit blijft apart
            'Aerial duels won, %',      # kwaliteit blijft apart
        ],
        "label": "CM Template v2",
    },
    "Defensive Midfielder": {
        "positions": ['RDMF', 'LDMF', 'DMF'],
        "stats": [
            # PRODUCTION
            'Completed progressive passes per 90',
            'Creative_Threat',
            # PASSING QUALITY
            'Passing accuracy (prog/1/3/forw)',
            'Successful dribbles, %',
            'Progressive runs per received pass',
            # DEFENSIVE
            'Defensive_Actions_Composite',  # production
            'Defensive duels won, %',       # quality
            'Aerial duels won, %',          # quality
            # NEGATIVE
            'Fouls per 90',
        ],
        "negative_stats": ['Fouls per 90'],
        "label": "DM Template v2",
    },
    "Centre-Back": {
        "positions": ['CB', 'RCB', 'LCB'],
        "stats": [
            # DEFENSIVE PRODUCTION
            'Defensive_Actions_Composite',
            # DEFENSIVE QUALITY
            'Defensive duels won, %',
            'Aerial duels won, %',
            # PASSING PRODUCTION
            'Completed progressive passes per 90',
            # PASSING QUALITY (Accurate progressive passes % is al vervat in de composite)
            'Passing accuracy (prog/1/3/forw)',
            # ON-BALL (ball carrying for CB's)
            'Progressive runs per received pass',
            # NEGATIVE
            'Fouls per 90',
        ],
        "negative_stats": ['Fouls per 90'],
        "label": "CB Template v2",
    },
    "Right-Back": {
        "positions": ['RB', 'RWB'],
        "stats": [
            # PRODUCTION
            'xG per 90',
            'Creative_Threat',
            'Completed progressive passes per 90',
            # QUALITY - offensief
            'Accurate crosses per received pass',
            'Accurate crosses, %',
            'Successful dribbles per received pass',  # keep 1 of 2 duplicates
            # QUALITY - passing
            'Passing accuracy (prog/1/3/forw)',
            # DEFENSIVE
            'Defensive_Actions_Composite',
            'Defensive duels won, %',
            'Aerial duels won, %',
        ],
        "label": "RB Template v2",
    },
    "Left-Back": {
        "positions": ['LB', 'LWB'],
        "stats": [
            # PRODUCTION
            'xG per 90',
            'Creative_Threat',
            'Completed progressive passes per 90',
            # QUALITY - offensief
            'Accurate crosses per received pass',
            'Accurate crosses, %',
            'Successful dribbles per received pass',
            # QUALITY - passing
            'Passing accuracy (prog/1/3/forw)',
            # DEFENSIVE
            'Defensive_Actions_Composite',
            'Defensive duels won, %',
            'Aerial duels won, %',
        ],
        "label": "LB Template v2",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITMISCHE VIF-FILTERING (check of onze templates nu gezond zijn)
# ─────────────────────────────────────────────────────────────────────────────

def validate_templates_vif(df: pd.DataFrame, 
                            config: dict = None,
                            min_minutes: int = 900,
                            vif_threshold: float = 10.0) -> dict:
    """
    Valideert dat rating_config_v2 geen VIF problemen meer heeft.
    Als er nog metrics zijn met VIF > threshold, worden die voorgesteld om
    te verwijderen.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    if config is None:
        config = rating_config_v2
    
    results = {}
    
    for pos_name, cfg in config.items():
        stats = cfg['stats']
        positions = cfg['positions']
        
        pos_df = df[
            df['Main Position'].isin(positions) &
            (df['Minutes played'] >= min_minutes)
        ].copy()
        
        available = [s for s in stats if s in pos_df.columns]
        analysis = pos_df[available].dropna()
        
        if len(analysis) < 30:
            results[pos_name] = {'error': f'Te weinig data: {len(analysis)}'}
            continue
        
        # VIF berekenen
        X = analysis.values
        X_std = (X - X.mean(axis=0)) / X.std(axis=0)
        
        vif_scores = {}
        for i, metric in enumerate(available):
            try:
                vif = variance_inflation_factor(X_std, i)
                vif_scores[metric] = round(float(vif), 2) if np.isfinite(vif) else float('inf')
            except Exception:
                vif_scores[metric] = None
        
        # Correlaties
        corr = analysis.corr()
        high_corrs = []
        for i in range(len(corr)):
            for j in range(i+1, len(corr)):
                r = corr.iloc[i, j]
                if abs(r) > 0.7:
                    high_corrs.append({
                        'm1': corr.index[i], 'm2': corr.columns[j], 'r': round(r, 3)
                    })
        
        problematic = {k: v for k, v in vif_scores.items() 
                       if v is not None and isinstance(v, (int, float)) and v > vif_threshold}
        
        results[pos_name] = {
            'n_players': len(analysis),
            'n_metrics': len(available),
            'vif_scores': vif_scores,
            'problematic_vif': problematic,
            'high_correlations': high_corrs,
            'clean': len(problematic) == 0 and not high_corrs,
        }
    
    return results


def print_validation_summary(results: dict):
    """Print leesbare samenvatting of rating_config_v2 gezond is."""
    
    print("=" * 80)
    print(" VALIDATIE rating_config_v2")
    print("=" * 80)
    
    total_clean = 0
    total_positions = 0
    
    for pos_name, res in results.items():
        total_positions += 1
        print(f"\n▶ {pos_name}")
        
        if 'error' in res:
            print(f"  ✗ {res['error']}")
            continue
        
        print(f"  Spelers: {res['n_players']}, Metrics: {res['n_metrics']}")
        
        if res['clean']:
            total_clean += 1
            print(f"  ✓ CLEAN — geen VIF > 10, geen |r| > 0.7")
        else:
            if res['problematic_vif']:
                print(f"  💥 VIF problemen ({len(res['problematic_vif'])}):")
                for m, v in sorted(res['problematic_vif'].items(), key=lambda x: -x[1]):
                    print(f"     {m}: VIF = {v:.1f}")
            
            if res['high_correlations']:
                print(f"  ⚠ Resterende correlaties:")
                for hc in res['high_correlations'][:5]:
                    marker = '⚠⚠' if abs(hc['r']) > 0.85 else '⚠ '
                    print(f"     {marker} {hc['m1']:40s} ↔ {hc['m2']:40s} r={hc['r']:+.2f}")
        
        # Max VIF als algemene health check
        vif_vals = [v for v in res['vif_scores'].values() 
                    if v is not None and isinstance(v, (int, float))]
        if vif_vals:
            print(f"  Max VIF: {max(vif_vals):.2f}  (threshold: 10)")
    
    print(f"\n{'=' * 80}")
    print(f" RESULTAAT: {total_clean}/{total_positions} posities CLEAN")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    print("Laden 2025/26 data met composite metrics...")
    df = load_season_data(season="2025/26", min_minutes=0)
    df = add_composite_metrics(df)
    print(f"Dataset: {len(df):,} rows, composites toegevoegd\n")
    
    print("Valideren rating_config_v2...")
    results = validate_templates_vif(df, rating_config_v2, min_minutes=900)
    print_validation_summary(results)
