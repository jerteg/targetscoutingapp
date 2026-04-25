"""
Market value model (Fase 7).

Public API:
  - train_market_value_model(...): train model op gematchte spelers met TM waardes
  - predict_market_values(...): pas getraind model toe op nieuwe spelers

Model:
  log(TM_value) = intercept
                + beta_rating * rating_adjusted
                + beta_age * age_effect (per-template curve)
                + beta_club * club_tier (log mean TM waarde van club)
                + beta_contract * log(1 + contract_months_left)
                + beta_league * log(fee_league_multiplier)
                + gamma_template * template_dummies
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr


def _compute_age_curves(df: pd.DataFrame, 
                        poly_degree: int = 3) -> dict:
    """
    Per template: fit log(TM) = polynomial(age - 27).
    
    Returns dict {template: {'coefficients': [...], 'intercept': ..., 'peak_age': ...}}
    """
    curves = {}
    
    for tpl in df['_template'].unique():
        subset = df[df['_template'] == tpl].copy()
        if len(subset) < 50:
            continue
        
        age_c = (subset['Age'] - 27).values
        X = np.column_stack([age_c**(i+1) for i in range(poly_degree)])
        y = subset['log_tm_value'].values
        
        m = Ridge(alpha=0.1).fit(X, y)
        
        # Peak age
        ages_range = np.arange(16, 40)
        age_c_range = ages_range - 27
        X_range = np.column_stack([age_c_range**(i+1) for i in range(poly_degree)])
        log_pred = m.predict(X_range) + m.intercept_
        peak_age = int(ages_range[np.argmax(log_pred)])
        
        curves[tpl] = {
            'coefficients': m.coef_.tolist(),
            'intercept': float(m.intercept_),
            'peak_age': peak_age,
            'r_squared': float(m.score(X, y)),
            'poly_degree': poly_degree,
        }
    
    return curves


def _apply_age_curve(df: pd.DataFrame, age_curves: dict) -> pd.DataFrame:
    """Bereken age_effect voor elke speler o.b.v. curves per template."""
    df = df.copy()
    df['age_effect'] = 0.0
    
    for tpl, curve in age_curves.items():
        mask = df['_template'] == tpl
        if not mask.any():
            continue
        
        age_c = df.loc[mask, 'Age'].values - 27
        poly_d = curve['poly_degree']
        X = np.column_stack([age_c**(i+1) for i in range(poly_d)])
        coefs = np.array(curve['coefficients'])
        df.loc[mask, 'age_effect'] = X @ coefs
    
    return df


def _compute_club_tier(df: pd.DataFrame, min_players_per_club: int = 3) -> pd.DataFrame:
    """
    Bereken club_tier = log(mean TM value of club).
    Voor clubs met <min_players: fallback naar liga-mediaan.
    
    Idempotent: als club_tier al bestaat, overschrijft niet.
    """
    df = df.copy()
    
    # Als club_tier al bestaat, return as-is
    if 'club_tier' in df.columns and df['club_tier'].notna().all():
        return df
    
    # Per club: mean TM value + count
    club_stats = df.groupby('current_club_name')['tm_value'].agg(['mean', 'count']).reset_index()
    club_stats.columns = ['current_club_name', 'club_mean_value', 'club_n_players']
    
    reliable = club_stats[club_stats['club_n_players'] >= min_players_per_club].copy()
    reliable['club_tier'] = np.log(reliable['club_mean_value'])
    
    # Liga-median als fallback
    liga_medians = df.groupby('League')['tm_value'].median().reset_index()
    liga_medians.columns = ['League', 'liga_median_tm']
    
    # Als club_tier bestaat (maar niet compleet), drop eerst
    if 'club_tier' in df.columns:
        df = df.drop(columns=['club_tier'])
    
    df = df.merge(reliable[['current_club_name', 'club_tier']], 
                  on='current_club_name', how='left')
    df = df.merge(liga_medians, on='League', how='left')
    df['club_tier'] = df['club_tier'].fillna(np.log(df['liga_median_tm']))
    df.drop(columns=['liga_median_tm'], inplace=True, errors='ignore')
    
    return df


def _build_features(df: pd.DataFrame, 
                     age_curves: dict,
                     club_tiers: dict,
                     fee_multipliers: dict,
                     min_players_per_club: int = 3) -> pd.DataFrame:
    """Bereken alle features nodig voor prediction."""
    df = df.copy()
    
    # Age effect
    df = _apply_age_curve(df, age_curves)
    
    # Club tier (use stored tiers if prediction mode, compute if training mode)
    if club_tiers:
        df['club_tier'] = df['current_club_name'].map(club_tiers)
        # Fallback voor unknown clubs
        if df['club_tier'].isna().any():
            # Gebruik mediaan van bekende clubs in dezelfde liga
            df['club_tier'] = df.groupby('League')['club_tier'].transform(
                lambda x: x.fillna(x.median())
            )
            # Finale fallback: globale mediaan
            df['club_tier'] = df['club_tier'].fillna(df['club_tier'].median())
    else:
        df = _compute_club_tier(df, min_players_per_club)
    
    # Contract
    today = pd.Timestamp(datetime.now())
    df['contract_days_left'] = (df['contract_expires'] - today).dt.days
    df['contract_months_left'] = (df['contract_days_left'] / 30.44).clip(lower=0)
    df['contract_months_left'] = df['contract_months_left'].fillna(
        df['contract_months_left'].median()
    )
    df['log_contract'] = np.log1p(df['contract_months_left'])
    
    # League fee multiplier (log)
    df['league_fee_mult'] = df['League'].map(fee_multipliers).fillna(1.0)
    df['league_fee_mult_log'] = np.log(df['league_fee_mult'])
    
    return df


def train_market_value_model(matched_df: pd.DataFrame,
                              tm_profiles: pd.DataFrame,
                              tm_market_values: pd.DataFrame,
                              ratings: pd.DataFrame,
                              fee_multipliers: dict,
                              min_match_confidence: float = 85,
                              test_size: float = 0.20,
                              random_state: int = 42,
                              ridge_alpha: float = 0.5,
                              min_players_per_club: int = 3,
                              age_poly_degree: int = 3,
                              min_tm_value: float = 100000,
                              min_tm_value_date: str = '2024-01-01',
                              verbose: bool = True) -> dict:
    """
    Train het €-model.
    
    Parameters
    ----------
    matched_df : Wyscout × TM matching output
    tm_profiles : TM player profiles (voor contract + club)
    tm_market_values : meest recente TM market values
    ratings : league-adjusted ratings (output Fase 3+5)
    fee_multipliers : {league: multiplier} uit Fase 6 (combined)
    
    Returns
    -------
    dict met:
      - model: de getrainde Ridge
      - feature_cols: welke features
      - template_dummies_cols: one-hot kolommen
      - age_curves: per-template curves
      - club_tiers: {club_name: tier}
      - fee_multipliers
      - metrics: train/test R², Spearman, % accuracy
      - training_data: de data gebruikt voor training (voor audit)
    """
    # Join alles
    hc = matched_df[matched_df['match_confidence'] >= min_match_confidence].copy()
    hc['tm_player_id'] = hc['tm_player_id'].astype('Int64')
    hc_lookup = hc[['Player', 'Team within selected timeframe', 'tm_player_id', 'match_confidence']].drop_duplicates(
        subset=['Player', 'Team within selected timeframe']
    )
    
    df = ratings.merge(hc_lookup, on=['Player', 'Team within selected timeframe'], how='inner')
    
    df = df.merge(
        tm_profiles[['player_id', 'current_club_name', 'date_of_birth', 'contract_expires']],
        left_on='tm_player_id', right_on='player_id', how='left'
    )
    
    df = df.merge(
        tm_market_values,
        left_on='tm_player_id', right_on='player_id', how='inner',
        suffixes=('', '_mv')
    )
    
    # Filters
    df = df[df['tm_value'].notna() & (df['tm_value'] >= min_tm_value)].copy()
    df = df[df['tm_value_date'] >= pd.Timestamp(min_tm_value_date)].copy()
    df['log_tm_value'] = np.log(df['tm_value'])
    
    if verbose:
        print(f"Training set: {len(df)} spelers")
    
    # Step 1: age curves
    age_curves = _compute_age_curves(df, poly_degree=age_poly_degree)
    if verbose:
        print(f"Age curves voor {len(age_curves)} templates")
    
    # Step 2: club tiers (compute once, use throughout)
    df = _compute_club_tier(df, min_players_per_club)
    club_tiers = df.drop_duplicates('current_club_name').set_index('current_club_name')['club_tier'].to_dict()
    
    # Step 3: apply age curve + other features (but club_tier already set)
    df = _apply_age_curve(df, age_curves)
    
    # Contract
    today = pd.Timestamp(datetime.now())
    df['contract_days_left'] = (df['contract_expires'] - today).dt.days
    df['contract_months_left'] = (df['contract_days_left'] / 30.44).clip(lower=0)
    df['contract_months_left'] = df['contract_months_left'].fillna(
        df['contract_months_left'].median()
    )
    df['log_contract'] = np.log1p(df['contract_months_left'])
    
    # League fee multiplier (log)
    df['league_fee_mult'] = df['League'].map(fee_multipliers).fillna(1.0)
    df['league_fee_mult_log'] = np.log(df['league_fee_mult'])
    
    # Feature matrix
    feature_cols = ['rating_adjusted', 'age_effect', 'club_tier', 
                    'log_contract', 'league_fee_mult_log']
    template_dummies = pd.get_dummies(df['_template'], prefix='tpl', drop_first=True)
    
    X_cont = df[feature_cols].values
    X = np.hstack([X_cont, template_dummies.values])
    y = df['log_tm_value'].values
    
    # Train/test split (stratified op template)
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=test_size, random_state=random_state,
        stratify=df['_template']
    )
    
    # Fit
    model = Ridge(alpha=ridge_alpha).fit(X_train, y_train)
    
    # Metrics
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # €-space accuracy
    pct_error_test = np.abs(np.exp(y_pred_test) - np.exp(y_test)) / np.exp(y_test) * 100
    
    rho_test, _ = spearmanr(y_test, y_pred_test)
    
    metrics = {
        'r2_train': float(r2_score(y_train, y_pred_train)),
        'r2_test': float(r2_score(y_test, y_pred_test)),
        'spearman_test': float(rho_test),
        'mae_test_log': float(mean_absolute_error(y_test, y_pred_test)),
        'median_pct_error': float(np.median(pct_error_test)),
        'mean_pct_error': float(np.mean(pct_error_test)),
        'pct_within_50': float((pct_error_test <= 50).mean() * 100),
        'pct_within_100': float((pct_error_test <= 100).mean() * 100),
        'n_train': len(X_train),
        'n_test': len(X_test),
    }
    
    if verbose:
        print(f"\nModel performance:")
        print(f"  Spearman ρ (test): {metrics['spearman_test']:.3f}")
        print(f"  R² (test):         {metrics['r2_test']:.3f}")
        print(f"  Median % error:    {metrics['median_pct_error']:.1f}%")
        print(f"  Within 50%:        {metrics['pct_within_50']:.1f}%")
        print(f"  Within 100%:       {metrics['pct_within_100']:.1f}%")
    
    return {
        'model': model,
        'feature_cols': feature_cols,
        'template_dummies_cols': list(template_dummies.columns),
        'age_curves': age_curves,
        'club_tiers': club_tiers,
        'fee_multipliers': fee_multipliers,
        'metrics': metrics,
        'training_data': df,
    }


def predict_market_values(model_artifacts: dict,
                           ratings: pd.DataFrame,
                           matched_df: pd.DataFrame,
                           tm_profiles: pd.DataFrame,
                           min_match_confidence: float = 85) -> pd.DataFrame:
    """
    Pas getraind model toe om €-waardes te voorspellen.
    
    Parameters
    ----------
    model_artifacts : output van train_market_value_model
    ratings : league-adjusted ratings (nieuwe spelers)
    matched_df : Wyscout × TM matching
    tm_profiles : TM profiles
    
    Returns
    -------
    DataFrame met:
      - alle input kolommen
      - predicted_value (in €)
      - predicted_log_value
    """
    hc = matched_df[matched_df['match_confidence'] >= min_match_confidence].copy()
    hc['tm_player_id'] = hc['tm_player_id'].astype('Int64')
    hc_lookup = hc[['Player', 'Team within selected timeframe', 'tm_player_id']].drop_duplicates(
        subset=['Player', 'Team within selected timeframe']
    )
    
    df = ratings.merge(hc_lookup, on=['Player', 'Team within selected timeframe'], how='inner')
    df = df.merge(
        tm_profiles[['player_id', 'current_club_name', 'contract_expires']],
        left_on='tm_player_id', right_on='player_id', how='left'
    )
    
    # Build features using stored artifacts
    df = _build_features(
        df,
        age_curves=model_artifacts['age_curves'],
        club_tiers=model_artifacts['club_tiers'],
        fee_multipliers=model_artifacts['fee_multipliers'],
    )
    
    # Feature matrix — moet EXACT dezelfde kolommen hebben als training
    feature_cols = model_artifacts['feature_cols']
    template_dummies_cols = model_artifacts['template_dummies_cols']
    
    X_cont = df[feature_cols].values
    
    # One-hot encode template, zorg dat alle training kolommen aanwezig zijn
    template_dummies = pd.get_dummies(df['_template'], prefix='tpl', drop_first=True)
    # Voeg ontbrekende kolommen toe (voor templates niet in prediction data)
    for col in template_dummies_cols:
        if col not in template_dummies.columns:
            template_dummies[col] = 0
    # Reorder naar training volgorde
    template_dummies = template_dummies[template_dummies_cols]
    
    X = np.hstack([X_cont, template_dummies.values])
    
    model = model_artifacts['model']
    log_pred = model.predict(X)
    df['predicted_log_value'] = log_pred
    df['predicted_value'] = np.exp(log_pred)
    
    return df


def save_model(model_artifacts: dict, path: str) -> None:
    """Save model artifacts to pickle."""
    import pickle
    # Filter out pandas DataFrames (kunnen groot zijn)
    to_save = {k: v for k, v in model_artifacts.items() if k != 'training_data'}
    with open(path, 'wb') as f:
        pickle.dump(to_save, f)


def load_model(path: str) -> dict:
    """Load model artifacts from pickle."""
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)
