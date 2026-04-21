"""
shared/similarity.py
Tier-adjusted cosine similarity for similar player finding.
Replaces the plain cosine similarity in Dashboard.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ── League tier weights ───────────────────────────────────────────────────────
LEAGUE_TIERS = {
    "Premier League":       1.00,
    "La Liga":              0.95,
    "Bundesliga":           0.92,
    "Italian Serie A":      0.92,
    "Ligue 1":              0.88,
    "Eredivisie":           0.78,
    "Primeira Liga":        0.78,
    "Pro League":           0.74,
    "Championship":         0.72,
    "Super Lig":            0.68,
    "Serie A BRA":          0.70,
    "Liga Profesional":     0.68,
    "MLS":                  0.62,
    "Eliteserien":          0.55,
    "Ekstraklasa":          0.55,
    "Superligaen":          0.55,
    "Prva HNL":             0.52,
    "Liga Pro":             0.50,
    "Segunda Division":     0.55,
    "Swiss Super League":   0.60,
}

DEFAULT_TIER = 0.55  # fallback for unknown leagues


def _get_tier(league: str) -> float:
    return LEAGUE_TIERS.get(league, DEFAULT_TIER)


def tier_badge(target_league: str, candidate_league: str) -> str:
    """Return tier label for UI badge."""
    t_mult = _get_tier(target_league)
    c_mult = _get_tier(candidate_league)
    diff   = c_mult - t_mult
    if diff >= -0.05:
        return "Same tier"
    elif diff >= -0.20:
        return "Tier below"
    else:
        return "Lower tier"


def tier_badge_color(badge: str) -> str:
    return {
        "Same tier":   "#1a7a45",
        "Tier below":  "#f0a500",
        "Lower tier":  "#c0392b",
    }.get(badge, "#b0a898")


def adjusted_similarity(
    target_row: pd.Series,
    candidates_df: pd.DataFrame,
    sim_stats: list,
    target_league: str = "",
    min_minutes: int = 600,
) -> pd.DataFrame:
    """
    Compute tier-adjusted cosine similarity.

    Parameters
    ----------
    target_row       : Series for the selected player (must include sim_stats)
    candidates_df    : DataFrame of candidate players (must include sim_stats + League + Minutes played)
    sim_stats        : list of stat column names to use for similarity
    target_league    : league of the target player (for tier adjustment)
    min_minutes      : minimum playing time to include candidates

    Returns
    -------
    DataFrame with original columns plus:
        raw_sim, tier_factor, adjusted_sim, tier_badge
    Sorted by adjusted_sim descending.
    """
    # Filter candidates by minutes
    if "Minutes played" in candidates_df.columns:
        mins = pd.to_numeric(candidates_df["Minutes played"], errors="coerce").fillna(0)
        candidates_df = candidates_df[mins >= min_minutes].copy()
    else:
        candidates_df = candidates_df.copy()

    if candidates_df.empty:
        return candidates_df

    # Available stats
    avail = [s for s in sim_stats if s in candidates_df.columns and s in target_row.index]
    if len(avail) < 3:
        # Fallback: return unsorted candidates with zero similarity
        candidates_df["raw_sim"]     = 0.0
        candidates_df["tier_factor"] = 1.0
        candidates_df["adjusted_sim"] = 0.0
        candidates_df["tier_badge"]  = "Same tier"
        return candidates_df

    # Compute percentiles within candidate pool (includes target for consistent ranking)
    all_vals = candidates_df[avail].apply(pd.to_numeric, errors="coerce").fillna(0)
    pct_pool = all_vals.rank(pct=True) * 100  # shape (n_candidates, n_features)

    # Target percentile vector (rank relative to candidates)
    target_vals = pd.to_numeric(
        pd.Series([target_row.get(s, 0) for s in avail], index=avail),
        errors="coerce"
    ).fillna(0)

    target_pct = np.array([
        float(np.mean(all_vals[s] <= target_vals[s]) * 100) for s in avail
    ]).reshape(1, -1)

    cand_pct = pct_pool.fillna(50).values

    # Raw cosine similarity
    raw_sims = cosine_similarity(target_pct, cand_pct)[0]

    # Tier factor
    t_tier = _get_tier(target_league) if target_league else DEFAULT_TIER
    if "League" in candidates_df.columns:
        c_tiers = candidates_df["League"].map(lambda l: _get_tier(l))
    else:
        c_tiers = pd.Series(DEFAULT_TIER, index=candidates_df.index)

    tier_factors = c_tiers.values / (t_tier + 1e-9)
    tier_factors = np.clip(tier_factors, 0.0, 1.0)

    # Adjusted similarity
    adjusted = raw_sims * tier_factors

    out = candidates_df.copy()
    out["raw_sim"]      = raw_sims
    out["tier_factor"]  = tier_factors
    out["adjusted_sim"] = adjusted

    if "League" in candidates_df.columns and target_league:
        out["tier_badge"] = out["League"].map(lambda l: tier_badge(target_league, l))
    else:
        out["tier_badge"] = "Same tier"

    return out.sort_values("adjusted_sim", ascending=False).reset_index(drop=True)
