"""
shared/scoring.py — Single source of truth for player score computation.

All pages import compute_scores() so ratings are identical everywhere.
Logic mirrors the Player Card exactly:
  1. Filter pool by position group + league template
  2. Rank each stat percentile within the pool
  3. Apply league multiplier (per mdict) and clip to [0, 100]
  4. Compute weighted category scores (with negative stats flipped)
  5. Compute overall = weighted sum of category scores
  6. Apply adjustment: (overall/100)^0.45 * 100  (if mode='adjusted')
"""

import pandas as pd
import numpy as np

from shared.templates import (
    report_template, position_category_weights, position_groups,
    LEAGUE_MULTIPLIERS_ALL, LEAGUE_MULTIPLIERS_NEXT14,
    TOP5_LEAGUES, NEXT14_LEAGUES,
)


def _wt(stats, weights, row):
    tw = sum(weights.get(s, 0) for s in stats if s in row.index)
    if tw == 0:
        return 0.0
    return sum(row[s] * weights[s] for s in stats if s in row.index and s in weights) / tw


def _overall_raw(cat_scores, cw):
    tw = sum(cw.values())
    if tw == 0:
        return 0.0
    return sum(cat_scores.get(c, 0) * w for c, w in cw.items()) / tw


def compute_scores(
    data: pd.DataFrame,
    position_group: str,
    league_template: str = "Top 5 leagues",
    score_mode: str = "Adjusted (recommended)",
) -> pd.DataFrame:
    """
    Compute category scores + overall for all players in the pool.

    Parameters
    ----------
    data            : full preprocessed dataframe
    position_group  : one of position_groups.keys()
    league_template : 'Top 5 leagues' | 'Next 14 competitions' | 'Both'
    score_mode      : 'Adjusted (recommended)' | 'Model (raw)'

    Returns
    -------
    DataFrame with added columns:
        {cat}_score   for each category in report_template
        overall_score (raw)
        overall_adj   (adjusted, same as overall_score if mode='Model (raw)')
    """
    positions = position_groups.get(position_group, [])

    # ── 1. Select pool + multiplier dict ──────────────────────────────────────
    if league_template == "Top 5 leagues":
        allowed = TOP5_LEAGUES
        mdict   = LEAGUE_MULTIPLIERS_ALL
    elif league_template == "Next 14 competitions":
        allowed = NEXT14_LEAGUES
        mdict   = LEAGUE_MULTIPLIERS_NEXT14
    else:  # Both
        allowed = TOP5_LEAGUES | NEXT14_LEAGUES
        mdict   = LEAGUE_MULTIPLIERS_ALL

    pool = data[
        data["Main Position"].isin(positions) &
        data["League"].isin(allowed)
    ].copy()

    if pool.empty:
        return pool

    ALL_STATS = [s for g in report_template.values() for s in g["stats"]]
    ex_stats  = [s for s in ALL_STATS if s in pool.columns]

    # ── 2. Percentile rank within pool ────────────────────────────────────────
    # Fill NaN with 50th percentile so missing stats don't zero out category scores
    pct_raw = pool[ex_stats].rank(pct=True) * 100
    pct_raw = pct_raw.fillna(50)

    # ── 3. Apply league multiplier and clip ───────────────────────────────────
    lg_mult = pool["League"].map(mdict).fillna(1.0)
    pct     = pct_raw.multiply(lg_mult.values, axis=0).clip(0, 100)

    # ── 4. Category scores ────────────────────────────────────────────────────
    cw = position_category_weights.get(position_group, {})

    for cat, grp in report_template.items():
        stats  = [s for s in grp["stats"] if s in pct.columns]
        neg    = grp.get("negative_stats", [])
        w      = grp.get("weights", {})

        def _sr(row, stats=stats, weights=w, neg=neg):
            adj = row.copy()
            for ns in neg:
                if ns in adj:
                    adj[ns] = 100 - adj[ns]
            return _wt(stats, weights, adj)

        pool[f"{cat}_score"] = pct[stats].apply(_sr, axis=1)

    # ── 5. Overall raw ────────────────────────────────────────────────────────
    pool["overall_score"] = pool.apply(
        lambda r: _overall_raw(
            {c: r.get(f"{c}_score", 0) for c in report_template}, cw
        ),
        axis=1,
    )

    # ── 6. Adjusted overall ───────────────────────────────────────────────────
    if score_mode == "Adjusted (recommended)":
        pool["overall_adj"] = (pool["overall_score"] / 100) ** 0.45 * 100
    else:
        pool["overall_adj"] = pool["overall_score"]

    # ── 7. Bayesian shrinkage (reliability correction, k=1200) ───────────────
    # Applied ONLY to players with fewer than 900 minutes (below the reliability
    # threshold established in the literature — StatsBomb, Mackay 2017, Caley 2013).
    # Players with 900+ minutes have a statistically sufficient sample; their score
    # is left unchanged. Players below 900 min are pulled towards the pool mean.
    # Formula: adjusted = w × score + (1 − w) × pool_mean
    # where w = minutes / (minutes + k), k = 1200
    # At 300 min: w = 0.20 | At 600 min: w = 0.33 | At 899 min: w = 0.43
    _k                 = 1200
    _RELIABILITY_THRESHOLD = 900
    if "Minutes played" in pool.columns:
        _mins = pd.to_numeric(pool["Minutes played"], errors="coerce").fillna(0)
        _mean = pool["overall_adj"].mean()
        _w    = _mins / (_mins + _k)
        # Only apply shrinkage to players below the reliability threshold
        _needs_shrinkage = _mins < _RELIABILITY_THRESHOLD
        pool.loc[_needs_shrinkage, "overall_adj"] = (
            _w[_needs_shrinkage] * pool.loc[_needs_shrinkage, "overall_adj"]
            + (1 - _w[_needs_shrinkage]) * _mean
        ).clip(0, 100)

    # Also store the per-player percentile rows for bars rendering
    pool["_pct_index"] = pool.index  # keep track for join
    pool["_league_template"] = league_template
    pool["_score_mode"]      = score_mode

    return pool
