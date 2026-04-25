"""
Fee-based league multipliers (Fase 6).

Public API:
  - derive_fee_multipliers(matched_df, transfers_df, tm_profiles, config):
      Berekent from/to/combined fee multipliers via log-lineaire regressie
      op cross-league paid transfers.

NOT USED for rating adjustment — we use these for €-value model (Fase 7).
Rating adjustment uses migration multipliers.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.linear_model import Ridge


def _build_team_to_league(matched_df: pd.DataFrame, 
                           tm_profiles: pd.DataFrame) -> dict:
    """
    Bouw team→liga mapping uit gematchte spelers.
    """
    # Meest voorkomende Wyscout liga per TM club_name
    hc = matched_df[matched_df['match_confidence'] >= 85].copy()
    hc['tm_player_id'] = hc['tm_player_id'].astype('Int64')
    
    hc_with_club = hc.merge(
        tm_profiles[['player_id', 'current_club_name']].rename(
            columns={'player_id': 'tm_player_id'}
        ),
        on='tm_player_id',
        how='left'
    )
    
    team_votes = defaultdict(Counter)
    for _, row in hc_with_club.iterrows():
        tm_club = row.get('current_club_name')
        wy_league = row.get('League')
        if pd.notna(tm_club) and pd.notna(wy_league):
            team_votes[tm_club][wy_league] += 1
    
    team_to_league = {team: votes.most_common(1)[0][0] 
                      for team, votes in team_votes.items()}
    
    # Voeg Wyscout team names toe (voor cases waar TM en Wyscout verschillend zijn)
    wy_votes = defaultdict(Counter)
    for _, row in hc_with_club.iterrows():
        wy_team = row.get('Team within selected timeframe')
        wy_league = row.get('League')
        if pd.notna(wy_team) and pd.notna(wy_league):
            wy_votes[wy_team][wy_league] += 1
    
    for team, votes in wy_votes.items():
        if team not in team_to_league:
            team_to_league[team] = votes.most_common(1)[0][0]
    
    return team_to_league


def _build_cross_league_transfers(transfers_df: pd.DataFrame,
                                   matched_ids: set,
                                   team_to_league: dict,
                                   tm_profiles: pd.DataFrame) -> pd.DataFrame:
    """Construct cross-league transfers from gematchte spelers."""
    relevant = transfers_df[transfers_df['player_id'].isin(matched_ids)].copy()
    
    # Map teams naar liga's
    relevant['from_league'] = relevant['from_team_name'].map(team_to_league)
    relevant['to_league'] = relevant['to_team_name'].map(team_to_league)
    
    both_known = relevant[
        relevant['from_league'].notna() & 
        relevant['to_league'].notna()
    ].copy()
    
    cross = both_known[both_known['from_league'] != both_known['to_league']].copy()
    
    # Voeg leeftijd toe
    profiles_dob = tm_profiles[['player_id', 'date_of_birth']]
    cross = cross.merge(profiles_dob, on='player_id', how='left')
    cross['age_at_transfer'] = (
        (cross['transfer_date'] - cross['date_of_birth']).dt.days / 365.25
    ).round(1)
    
    cross = cross[(cross['age_at_transfer'] >= 16) & 
                  (cross['age_at_transfer'] <= 40)].copy()
    
    cross['log_fee'] = np.log(cross['transfer_fee'])
    
    return cross


def _fit_fee_regression(cross_df: pd.DataFrame, 
                         reference_league: str = "Premier League",
                         ridge_alpha: float = 0.5) -> dict:
    """Log-lineaire regressie op cross-league fees."""
    df = cross_df.copy()
    df['age_centered'] = df['age_at_transfer'] - 25
    df['age_centered_sq'] = df['age_centered'] ** 2
    df['year_centered'] = df['year'] - 2020
    
    all_leagues = sorted(set(df['from_league'].unique()) | 
                          set(df['to_league'].unique()))
    non_ref = [lg for lg in all_leagues if lg != reference_league]
    
    # Features
    X_cont = df[['age_centered', 'age_centered_sq', 'year_centered']].values
    X_from = np.zeros((len(df), len(non_ref)))
    X_to = np.zeros((len(df), len(non_ref)))
    for j, lg in enumerate(non_ref):
        X_from[:, j] = (df['from_league'] == lg).astype(int)
        X_to[:, j] = (df['to_league'] == lg).astype(int)
    
    X = np.hstack([X_cont, X_from, X_to])
    y = df['log_fee'].values
    
    model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    model.fit(X, y)
    
    n_cont = X_cont.shape[1]
    n_leagues = len(non_ref)
    
    from_coef = {reference_league: 0.0}
    to_coef = {reference_league: 0.0}
    for j, lg in enumerate(non_ref):
        from_coef[lg] = float(model.coef_[n_cont + j])
        to_coef[lg] = float(model.coef_[n_cont + n_leagues + j])
    
    n_from = {lg: int((df['from_league'] == lg).sum()) for lg in all_leagues}
    n_to = {lg: int((df['to_league'] == lg).sum()) for lg in all_leagues}
    
    return {
        'from_coefficients': from_coef,
        'to_coefficients': to_coef,
        'r_squared': float(model.score(X, y)),
        'n_from': n_from,
        'n_to': n_to,
        'n_total': len(df),
    }


def derive_fee_multipliers(matched_df: pd.DataFrame,
                            transfers_df: pd.DataFrame,
                            tm_profiles: pd.DataFrame,
                            reference_league: str = "Premier League",
                            ridge_alpha: float = 0.5) -> dict:
    """
    Derive fee-based league multipliers from cross-league paid transfers.
    
    Parameters
    ----------
    matched_df : output van matching (Wyscout × TM)
    transfers_df : TM paid transfers (met year, transfer_fee, from/to_team_name)
    tm_profiles : TM player profiles (voor age)
    reference_league : PL default = 1.00
    
    Returns
    -------
    dict met:
      - from_multipliers: {league: mult} — hoe duur zijn uitgaande spelers
      - to_multipliers: {league: mult} — hoeveel betaalt deze liga
      - combined_multipliers: gemiddelde (meest stabiel)
      - n_per_league: counts
    """
    team_to_league = _build_team_to_league(matched_df, tm_profiles)
    
    hc = matched_df[matched_df['match_confidence'] >= 85].copy()
    hc_ids = set(hc['tm_player_id'].dropna().astype(int))
    
    cross = _build_cross_league_transfers(
        transfers_df, hc_ids, team_to_league, tm_profiles
    )
    
    regression = _fit_fee_regression(cross, reference_league, ridge_alpha)
    
    # Convert log-coefficients to multipliers (exp clipped)
    def _to_mult(coefs):
        out = {}
        for lg, beta in coefs.items():
            m = np.exp(beta)
            m = max(0.05, min(1.50, m))
            out[lg] = round(float(m), 3)
        return out
    
    from_mult = _to_mult(regression['from_coefficients'])
    to_mult = _to_mult(regression['to_coefficients'])
    combined = {
        lg: round(float(np.exp(
            (regression['from_coefficients'].get(lg, 0) + 
             regression['to_coefficients'].get(lg, 0)) / 2
        )), 3)
        for lg in set(from_mult.keys()) | set(to_mult.keys())
    }
    
    return {
        'from_multipliers': from_mult,
        'to_multipliers': to_mult,
        'combined_multipliers': combined,
        'n_transfers_used': regression['n_total'],
        'n_per_league_from': regression['n_from'],
        'n_per_league_to': regression['n_to'],
        'r_squared': regression['r_squared'],
        'cross_league_df': cross,
    }
