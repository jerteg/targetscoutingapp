"""
shared/scoring.py — Single source of truth for player score computation.

Updated v2: ondersteunt naast het oude category-based score model
ook het nieuwe dimensie-based role-scoring (zie shared/roles_v2.py).

Backwards compatible: compute_scores() blijft hetzelfde signature.
Nieuw: compute_role_ranking() voor de Ranking-pagina.
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
    LEGACY: berekent category scores + overall via report_template.
    Voor nieuwe rol-gebaseerde rankings: gebruik compute_role_ranking().
    """
    positions = position_groups.get(position_group, [])

    if league_template == "Top 5 leagues":
        allowed = TOP5_LEAGUES
        mdict   = LEAGUE_MULTIPLIERS_ALL
    elif league_template == "Next 14 competitions":
        allowed = NEXT14_LEAGUES
        mdict   = LEAGUE_MULTIPLIERS_NEXT14
    else:
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

    pct_raw = pool[ex_stats].rank(pct=True) * 100
    pct_raw = pct_raw.fillna(50)

    lg_mult = pool["League"].map(mdict).fillna(1.0)
    pct     = pct_raw.multiply(lg_mult.values, axis=0).clip(0, 100)

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

    pool["overall_score"] = pool.apply(
        lambda r: _overall_raw(
            {c: r.get(f"{c}_score", 0) for c in report_template}, cw
        ),
        axis=1,
    )

    if score_mode == "Adjusted (recommended)":
        pool["overall_adj"] = (pool["overall_score"] / 100) ** 0.45 * 100
    else:
        pool["overall_adj"] = pool["overall_score"]

    # Bayesian shrinkage (alleen <900 min)
    _k                     = 1200
    _RELIABILITY_THRESHOLD = 900
    if "Minutes played" in pool.columns:
        _mins = pd.to_numeric(pool["Minutes played"], errors="coerce").fillna(0)
        _mean = pool["overall_adj"].mean()
        _w    = _mins / (_mins + _k)
        _needs_shrinkage = _mins < _RELIABILITY_THRESHOLD
        pool.loc[_needs_shrinkage, "overall_adj"] = (
            _w[_needs_shrinkage] * pool.loc[_needs_shrinkage, "overall_adj"]
            + (1 - _w[_needs_shrinkage]) * _mean
        ).clip(0, 100)

    pool["_pct_index"] = pool.index
    pool["_league_template"] = league_template
    pool["_score_mode"]      = score_mode

    return pool


# ──────────────────────────────────────────────────────────────────────────────
# NEW: dimensie-gebaseerde rol-ranking
# ──────────────────────────────────────────────────────────────────────────────

def compute_role_ranking(
    data: pd.DataFrame,
    position_label: str,
    role_name: str,
    league_template: str = "Top 5 leagues",
    apply_league_multiplier: bool = True,
    apply_shrinkage: bool = True,
) -> pd.DataFrame:
    """
    Bereken role-score via dimensie-systeem (roles_v2).

    Parameters
    ----------
    data            : volledige preprocessed dataframe (uit load_season_data)
    position_label  : key uit ROLE_CONFIG_V2 (bv. "Centre-Back", "Right Winger")
    role_name       : rol-naam binnen die positie
    league_template : 'Top 5 leagues' | 'Next 14 competitions' | 'Both'
    apply_league_multiplier : zo ja: tier-correctie op eindscore
    apply_shrinkage : zo ja: Bayesian shrinkage voor spelers <900 min

    Returns
    -------
    DataFrame met:
      - originele kolommen
      - dim_<dimensienaam> kolommen voor elke dimensie
      - role_score (0-100), de uiteindelijke gerangschikte score
    Gesorteerd op role_score descending.
    """
    from shared.roles_v2 import (
        compute_dimension_scores, compute_role_score,
        ROLE_CONFIG_V2, POSITION_DIMENSIONS,
    )

    if position_label not in ROLE_CONFIG_V2:
        raise KeyError(f"Position '{position_label}' niet in ROLE_CONFIG_V2")
    if role_name not in ROLE_CONFIG_V2[position_label]:
        raise KeyError(f"Role '{role_name}' niet in {position_label}")

    # ── 1. Pool selecteren ──
    # Map position_label naar Wyscout MainPos codes (uit templates.position_groups)
    # Let op: position_groups in templates.py kent niet exact dezelfde keys als
    # ROLE_CONFIG_V2 (bv. "Right Winger" vs "Winger"). Deze mapping is robuust:
    POS_LABEL_TO_GROUP = {
        "Centre-Back":           "Centre-Back",
        "Right-Back":            "Right-Back",
        "Left-Back":             "Left-Back",
        "Defensive Midfielder":  "Defensive Midfielder",
        "Central Midfielder":    "Central Midfielder",
        "Attacking Midfielder":  "Attacking Midfielder",
        "Right Winger":          "Right Winger",
        "Left Winger":           "Left Winger",
        "Winger":                "Winger",
        "Striker":               "Striker",
    }
    positions = position_groups.get(POS_LABEL_TO_GROUP.get(position_label, position_label), [])
    if not positions:
        # fallback: probeer direct
        positions = position_groups.get(position_label, [])

    if league_template == "Top 5 leagues":
        allowed = TOP5_LEAGUES
        mdict   = LEAGUE_MULTIPLIERS_ALL
    elif league_template == "Next 14 competitions":
        allowed = NEXT14_LEAGUES
        mdict   = LEAGUE_MULTIPLIERS_NEXT14
    else:
        allowed = TOP5_LEAGUES | NEXT14_LEAGUES
        mdict   = LEAGUE_MULTIPLIERS_ALL

    pool = data[
        data["Main Position"].isin(positions) &
        data["League"].isin(allowed)
    ].copy()
    pool = pool.replace([np.inf, -np.inf], np.nan)

    if pool.empty:
        return pool

    # ── 2. Dimensie-scores berekenen ──
    pool_with_dims = compute_dimension_scores(pool, position_label)

    # ── 3. Rol-score berekenen ──
    pool_with_dims["role_score"] = compute_role_score(
        pool_with_dims, position_label, role_name
    )

    # ── 4. League multiplier (optioneel) ──
    if apply_league_multiplier:
        lg_mult = pool_with_dims["League"].map(mdict).fillna(1.0)
        pool_with_dims["role_score"] = (
            pool_with_dims["role_score"] * lg_mult
        ).clip(0, 100)

    # ── 5. Bayesian shrinkage (optioneel) ──
    if apply_shrinkage and "Minutes played" in pool_with_dims.columns:
        k       = 1200
        thresh  = 900
        mins    = pd.to_numeric(pool_with_dims["Minutes played"], errors="coerce").fillna(0)
        mean    = pool_with_dims["role_score"].mean()
        w       = mins / (mins + k)
        needs   = mins < thresh
        pool_with_dims.loc[needs, "role_score"] = (
            w[needs] * pool_with_dims.loc[needs, "role_score"]
            + (1 - w[needs]) * mean
        ).clip(0, 100)

    return pool_with_dims.sort_values("role_score", ascending=False).reset_index(drop=True)
