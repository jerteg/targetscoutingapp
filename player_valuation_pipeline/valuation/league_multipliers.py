"""
League multipliers module — Fase 4 + 5 combined.

Public API:
  - derive_migration_multipliers(wy_current, wy_previous, config):
      Berekent migration-based league multipliers met Bayesian shrinkage.
  - apply_bayesian_shrinkage(data_results, prior): Bayesian combine.

Methodologie:
  Cross-season migration regression (ElHabr-stijl APM voor leagues):
    delta_rating = beta_to - beta_from + ε
  
  Daarna Bayesian combine met expert prior:
    posterior_beta = w_data * data_beta + w_prior * prior_beta
    waarbij w_data = n / (n + tau)
"""

import pandas as pd
import numpy as np
from typing import Optional
from sklearn.linear_model import Ridge

from ._templates import LEAGUE_MULTIPLIERS_ALL


def _simple_rating_for_migrations(df: pd.DataFrame, config: dict, min_minutes: int) -> pd.DataFrame:
    """
    Eenvoudige rating voor migration analyse — mean z-score (NIET category-weighted).
    We gebruiken simple rating voor delta-migration omdat:
      - We migraties meten per template, dus category weights zouden shifts introduceren
      - Simple rating is stabieler voor kleine sample (migratie N=373)
    """
    all_results = []
    
    for tpl_name, cfg in config.items():
        stats = cfg['stats']
        positions = cfg['positions']
        negative = set(cfg.get('negative_stats', []))
        
        pos_df = df[df['Main Position'].isin(positions)].copy()
        if min_minutes > 0:
            pos_df = pos_df[pos_df['Minutes played'] >= min_minutes]
        
        if len(pos_df) == 0:
            continue
        
        z_cols = []
        for metric in stats:
            if metric not in pos_df.columns:
                continue
            col = f'_z_{metric}'
            z = pos_df.groupby('League')[metric].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else pd.Series(0, index=x.index)
            )
            if metric in negative:
                z = -z
            z = z.clip(-3, 3)
            pos_df[col] = z
            z_cols.append(col)
        
        pos_df['rating'] = pos_df[z_cols].mean(axis=1)
        pos_df['_template'] = tpl_name
        
        all_results.append(pos_df[['Player', 'Team within selected timeframe', 'League',
                                    'Main Position', 'Age', 'Minutes played',
                                    'rating', '_template']])
    
    return pd.concat(all_results, ignore_index=True)


def _build_migrations(wy_current: pd.DataFrame, wy_previous: pd.DataFrame,
                      config: dict, min_minutes_per_season: int = 600) -> pd.DataFrame:
    """Detect migrations: spelers die tussen seizoenen van liga wisselden."""
    
    # Composite metrics + rating per seizoen
    from ._rating_config import add_composite_metrics
    
    wy_curr = add_composite_metrics(wy_current.copy())
    wy_prev = add_composite_metrics(wy_previous.copy())
    
    r_curr = _simple_rating_for_migrations(wy_curr, config, min_minutes=min_minutes_per_season)
    r_prev = _simple_rating_for_migrations(wy_prev, config, min_minutes=min_minutes_per_season)
    
    # Per speler: pak rij met meeste minuten per seizoen
    def consolidate(rating_df):
        idx = rating_df.groupby('Player')['Minutes played'].idxmax()
        return rating_df.loc[idx].copy()
    
    r_curr_c = consolidate(r_curr)
    r_prev_c = consolidate(r_prev)
    
    merged = r_prev_c.merge(r_curr_c, on='Player', suffixes=('_prev', '_curr'))
    
    migrations = merged[
        (merged['League_prev'] != merged['League_curr']) &
        (merged['_template_prev'] == merged['_template_curr']) &
        (merged['Minutes played_prev'] >= min_minutes_per_season) &
        (merged['Minutes played_curr'] >= min_minutes_per_season)
    ].copy()
    
    migrations['delta_rating'] = migrations['rating_curr'] - migrations['rating_prev']
    migrations['template'] = migrations['_template_prev']
    migrations['Age'] = migrations['Age_prev']
    migrations['age_diff_from_peak'] = migrations['Age'] - 27
    
    return migrations[['Player', 'Age', 'template', 'League_prev', 'League_curr',
                       'rating_prev', 'rating_curr', 'delta_rating',
                       'age_diff_from_peak']].reset_index(drop=True)


def _fit_migration_regression(migrations: pd.DataFrame, 
                               reference_league: str,
                               ridge_alpha: float = 0.1) -> dict:
    """Fit ridge regression over migrations."""
    all_leagues = sorted(set(migrations['League_prev'].unique()) | 
                          set(migrations['League_curr'].unique()))
    non_ref = [lg for lg in all_leagues if lg != reference_league]
    
    # Design matrix: -1 source league, +1 destination league
    X = np.zeros((len(migrations), len(non_ref)))
    for idx, (_, row) in enumerate(migrations.iterrows()):
        for j, lg in enumerate(non_ref):
            if row['League_prev'] == lg:
                X[idx, j] = -1
            elif row['League_curr'] == lg:
                X[idx, j] = +1
    
    # Age control
    age = migrations['age_diff_from_peak'].values.reshape(-1, 1)
    X = np.column_stack([X, age])
    y = migrations['delta_rating'].values
    
    model = Ridge(alpha=ridge_alpha, fit_intercept=False)
    model.fit(X, y)
    
    coefficients = {reference_league: 0.0}
    for j, lg in enumerate(non_ref):
        coefficients[lg] = float(model.coef_[j])
    
    # Bootstrap SE's
    rng = np.random.default_rng(42)
    boot = {lg: [] for lg in all_leagues}
    for _ in range(200):
        idx = rng.integers(0, len(migrations), len(migrations))
        try:
            m_b = Ridge(alpha=ridge_alpha, fit_intercept=False)
            m_b.fit(X[idx], y[idx])
            for j, lg in enumerate(non_ref):
                boot[lg].append(m_b.coef_[j])
            boot[reference_league].append(0.0)
        except Exception:
            continue
    
    std_errors = {lg: float(np.std(v)) if len(v) > 10 else None for lg, v in boot.items()}
    
    n_migrations = {lg: int(((migrations['League_prev'] == lg) | 
                              (migrations['League_curr'] == lg)).sum())
                    for lg in all_leagues}
    
    return {
        'coefficients': coefficients,
        'n_migrations': n_migrations,
        'std_errors': std_errors,
        'n_total': len(migrations),
        'r_squared': float(model.score(X, y)),
        'age_coefficient': float(model.coef_[-1]),
    }


def apply_bayesian_shrinkage(data_results: dict,
                              priors: dict = None,
                              tau: float = 15,
                              k_scaling: float = 1.5,
                              multiplier_min: float = 0.20,
                              multiplier_max: float = 1.20,
                              reference_league: str = "Premier League") -> dict:
    """
    Combine data-driven betas met expert prior via Bayesian shrinkage.
    
    Parameters
    ----------
    data_results : output van _fit_migration_regression
    priors : dict {league: multiplier}, default uses LEAGUE_MULTIPLIERS_ALL
    tau : prior strength. Lower = data wins more.
    k_scaling : multiplier = exp(-beta / k_scaling)
    multiplier_min, multiplier_max : clip range
    
    Returns
    -------
    dict met:
      - posterior_multipliers: {league: multiplier}
      - posterior_betas
      - shrinkage_weights (per league)
      - data_betas, prior_betas
    """
    if priors is None:
        priors = LEAGUE_MULTIPLIERS_ALL
    
    data_betas = data_results['coefficients']
    n_per_league = data_results['n_migrations']
    
    # Compute prior betas (relative to reference league)
    prior_ref_mult = priors.get(reference_league, 1.0)
    prior_betas = {}
    for lg in priors:
        if lg == reference_league:
            prior_betas[lg] = 0.0
        else:
            rel_mult = priors[lg] / prior_ref_mult
            prior_betas[lg] = -k_scaling * np.log(rel_mult)
    
    # Shrinkage combine
    posterior_betas = {}
    posterior_multipliers = {}
    shrinkage_weights = {}
    
    all_leagues = set(data_betas.keys()) | set(prior_betas.keys())
    
    for lg in all_leagues:
        n = n_per_league.get(lg, 0)
        data_beta = data_betas.get(lg)
        prior_beta = prior_betas.get(lg, 0.0)
        
        w_data = n / (n + tau) if data_beta is not None else 0.0
        w_prior = 1.0 - w_data
        
        if data_beta is None:
            posterior = prior_beta
        else:
            posterior = w_data * data_beta + w_prior * prior_beta
        
        posterior_betas[lg] = posterior
        
        # Convert beta to multiplier
        m = np.exp(-posterior / k_scaling)
        m = max(multiplier_min, min(multiplier_max, m))
        posterior_multipliers[lg] = round(float(m), 3)
        
        shrinkage_weights[lg] = {
            'w_data': round(w_data, 3),
            'w_prior': round(w_prior, 3),
            'n': n,
        }
    
    return {
        'posterior_betas': posterior_betas,
        'posterior_multipliers': posterior_multipliers,
        'shrinkage_weights': shrinkage_weights,
        'prior_betas': prior_betas,
        'data_betas': data_betas,
    }


def derive_migration_multipliers(wy_current: pd.DataFrame, 
                                  wy_previous: pd.DataFrame,
                                  config: dict,
                                  reference_league: str = "Premier League",
                                  min_minutes_per_season: int = 600,
                                  bayesian_tau: float = 15,
                                  k_scaling: float = 1.5,
                                  multiplier_min: float = 0.20,
                                  multiplier_max: float = 1.20) -> dict:
    """
    End-to-end: derive migration-based league multipliers.
    
    Returns dict met alle outputs (multipliers, shrinkage weights, raw data).
    """
    migrations = _build_migrations(wy_current, wy_previous, config, min_minutes_per_season)
    
    data_results = _fit_migration_regression(migrations, reference_league)
    
    bayesian = apply_bayesian_shrinkage(
        data_results,
        tau=bayesian_tau,
        k_scaling=k_scaling,
        multiplier_min=multiplier_min,
        multiplier_max=multiplier_max,
        reference_league=reference_league,
    )
    
    return {
        'multipliers': bayesian['posterior_multipliers'],
        'bayesian': bayesian,
        'data_regression': data_results,
        'migrations': migrations,
    }
