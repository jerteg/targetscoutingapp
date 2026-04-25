"""
Rating module: Fase 1-3 combined.

Public API:
  - compute_rating(df, config): bereken category-weighted ratings voor alle spelers

Dit is de "skill" rating, nog VOOR league multiplier toepassing.
"""

import pandas as pd
import numpy as np
from typing import Optional

from ._rating_config import rating_config_v2, add_composite_metrics
from ._templates import position_category_weights, STAT_TO_CATEGORY


# Mapping van composite/extra metrics naar categorieën
EXTENDED_STAT_TO_CATEGORY = {
    **STAT_TO_CATEGORY,
    'Creative_Threat': 'Chance Creation',
    'Defensive_Actions_Composite': 'Defending',
    'Through passes per pass': 'Chance Creation',
    'Accurate crosses, %': 'Chance Creation',
    'Fouls per 90': 'Defending',
}


def compute_rating(df: pd.DataFrame, 
                    min_minutes: int = 900,
                    z_clip: float = 3.0,
                    verbose: bool = False) -> pd.DataFrame:
    """
    Bereken category-weighted skill rating per speler per template.
    
    Pipeline:
      1. Add composite metrics (Creative_Threat, Defensive_Actions_Composite)
      2. Voor elke template: bereken z-scores binnen liga voor alle metrics
      3. Aggregeer z-scores per categorie (Goalscoring, Chance Creation, etc.)
      4. Gewogen som over categorieën met expert weights (position_category_weights)
    
    Parameters
    ----------
    df : Wyscout DataFrame (output van load_wyscout_data)
    min_minutes : minimum minuten filter
    z_clip : z-score outlier clip (±z_clip)
    verbose : print weight-mappings per template
    
    Returns
    -------
    DataFrame met één rij per (speler, template), kolommen:
      - Player, Team within selected timeframe, League, Main Position, Age, Minutes played
      - rating (category-weighted skill rating)
      - _template (welk template)
      - z_<Category> (per categorie z-score voor transparantie)
    """
    # Composite metrics
    df = add_composite_metrics(df)
    
    all_results = []
    
    for template_name, cfg in rating_config_v2.items():
        stats = cfg['stats']
        positions = cfg['positions']
        negative = set(cfg.get('negative_stats', []))
        
        # Categorieën in template
        cat_to_stats = _get_categories_for_template(stats)
        
        # Expert weights
        expert_weights = position_category_weights.get(template_name, {})
        if not expert_weights:
            if verbose:
                print(f"  ⚠ Geen expert weights voor {template_name}, gebruik gelijke weights")
            expert_weights = {c: 1.0/len(cat_to_stats) for c in cat_to_stats}
        
        # Case-insensitive match: 'Chance creation' (expert) → 'Chance Creation' (stats)
        expert_lower_to_orig = {c.lower(): c for c in expert_weights}
        weights = {}
        for cat in cat_to_stats.keys():
            if cat.lower() in expert_lower_to_orig:
                weights[cat] = expert_weights[expert_lower_to_orig[cat.lower()]]
        
        total = sum(weights.values())
        if total == 0:
            if verbose:
                print(f"  ✗ Geen weights voor {template_name}")
            continue
        normalized_weights = {c: w/total for c, w in weights.items()}
        
        if verbose:
            print(f"\n  {template_name} weights:")
            for cat, w in sorted(normalized_weights.items(), key=lambda x: -x[1]):
                print(f"    {cat:<20} {w*100:5.1f}%")
        
        # Filter data
        pos_df = df[df['Main Position'].isin(positions)].copy()
        if min_minutes > 0:
            pos_df = pos_df[pos_df['Minutes played'] >= min_minutes]
        
        if len(pos_df) == 0:
            continue
        
        # Z-scores per metric BINNEN liga
        for metric in stats:
            if metric not in pos_df.columns:
                continue
            col = f'_z_{metric}'
            z = pos_df.groupby('League')[metric].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else pd.Series(0, index=x.index)
            )
            if metric in negative:
                z = -z
            z = z.clip(-z_clip, z_clip)
            pos_df[col] = z
        
        # Per categorie: gemiddelde van z-scores
        category_columns = {}
        for cat, metrics_in_cat in cat_to_stats.items():
            z_cols = [f'_z_{m}' for m in metrics_in_cat if f'_z_{m}' in pos_df.columns]
            if not z_cols:
                continue
            cat_col = f'z_{cat.replace(" ", "_")}'
            pos_df[cat_col] = pos_df[z_cols].mean(axis=1)
            category_columns[cat] = cat_col
        
        # Gewogen rating
        pos_df['rating'] = 0.0
        for cat, cat_col in category_columns.items():
            w = normalized_weights.get(cat, 0)
            pos_df['rating'] = pos_df['rating'] + w * pos_df[cat_col]
        
        pos_df['_template'] = template_name
        
        out_cols = ['Player', 'Team within selected timeframe', 'League',
                    'Main Position', 'Age', 'Minutes played', 'rating', '_template']
        out_cols += list(category_columns.values())
        out_cols = [c for c in out_cols if c in pos_df.columns]
        
        all_results.append(pos_df[out_cols])
    
    if not all_results:
        raise ValueError("Geen ratings berekend — controleer input data")
    
    return pd.concat(all_results, ignore_index=True)


def _get_categories_for_template(stats: list) -> dict:
    """Map metrics naar hun categorieën."""
    cat_to_stats = {}
    for s in stats:
        cat = EXTENDED_STAT_TO_CATEGORY.get(s)
        if cat is None:
            continue
        cat_to_stats.setdefault(cat, []).append(s)
    return cat_to_stats


def apply_league_multipliers(ratings: pd.DataFrame, multipliers: dict) -> pd.DataFrame:
    """
    Pas league multipliers toe.
    
    Formule: adjusted = raw + 3 * ln(multiplier)
    
    Dit is additive shift in z-space, wat equivalent is aan multiplicatief effect
    op underlying performance metrics.
    """
    result = ratings.copy()
    result['league_multiplier'] = result['League'].map(multipliers).fillna(1.0)
    result['rating_adjusted'] = result['rating'] + 3 * np.log(result['league_multiplier'])
    return result
