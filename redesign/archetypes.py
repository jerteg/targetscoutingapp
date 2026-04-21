"""
shared/archetypes.py
K-Means clustering per position group. Persists trained models as pickles.
Provides assign_archetype() and get_player_archetype() functions.
"""
import os
import pickle
import hashlib
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "shared", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Position groups for archetypes (different from scoring position_groups!) ──
ARCHETYPE_POSITION_GROUPS = {
    "CB":  ["CB", "LCB", "RCB"],
    "FB":  ["LB", "RB", "LWB", "RWB"],
    "MID": ["DMF", "LDMF", "RDMF", "LCMF", "RCMF", "AMF"],
    "W":   ["LW", "RW", "LWF", "RWF", "LAMF", "RAMF"],
    "ST":  ["CF"],
}

# Map scoring position_groups keys → archetype group
SCORING_TO_ARCHETYPE = {
    "Centre-Back":           "CB",
    "Right-Back":            "FB",
    "Left-Back":             "FB",
    "Defensive Midfielder":  "MID",
    "Central Midfielder":    "MID",
    "Attacking Midfielder":  "MID",
    "Right Winger":          "W",
    "Left Winger":           "W",
    "Striker":               "ST",
}

# ── Training competitions ─────────────────────────────────────────────────────
ARCHETYPE_LEAGUES = {
    "Premier League", "La Liga", "Bundesliga", "Italian Serie A", "Ligue 1",
    "Eredivisie", "Primeira Liga", "Pro League", "Championship",
    "Liga Profesional", "Serie A BRA", "MLS", "Super Lig",
}
ARCHETYPE_MIN_MINUTES = 1000

# ── Features per position group ───────────────────────────────────────────────
ARCHETYPE_FEATURES = {
    "CB": [
        "Defensive duels per 90", "Aerial duels per 90", "PAdj Sliding tackles",
        "Shots blocked per 90", "PAdj Interceptions", "Fouls per 90",
        "Passes per 90", "Forward passes per 90", "Long passes per 90",
        "Progressive passes per 90", "Average pass length, m",
        "Passes to final third per 90", "Dribbles per 90",
        "Progressive runs per 90",
    ],
    "FB": [
        "Crosses per 90", "Touches in box per 90", "Dribbles per 90",
        "Progressive runs per 90", "Offensive duels per 90", "Shots per 90",
        "Key passes per 90", "Passes to penalty area per 90",
        "Defensive duels per 90", "Aerial duels per 90", "PAdj Interceptions",
        "Forward passes per 90", "Long passes per 90", "Average pass length, m",
    ],
    "MID": [
        "Defensive duels per 90", "PAdj Interceptions", "PAdj Sliding tackles",
        "Aerial duels per 90", "Fouls per 90", "Average pass length, m",
        "Through passes per 90", "Smart passes per 90", "Key passes per 90",
        "Passes to final third per 90", "Dribbles per 90",
        "Progressive runs per 90", "Shots per 90", "xG per shot", "xG per 90",
    ],
    "W": [
        "Crosses per 90", "Dribbles per 90", "Accelerations per 90",
        "Progressive runs per 90", "Touches in box per 90", "Shots per 90",
        "Offensive duels per 90", "Key passes per 90", "Through passes per 90",
        "Smart passes per 90", "Passes to penalty area per 90",
        "Defensive duels per 90", "PAdj Interceptions", "Average pass length, m",
    ],
    "ST": [
        "Head goals per 90", "Aerial duels per 90", "Received long passes per 90",
        "Key passes per 90", "Smart passes per 90", "Through passes per 90",
        "Passes to penalty area per 90", "Forward passes per 90",
        "Received passes per 90", "Dribbles per 90", "Progressive runs per 90",
        "Accelerations per 90", "Touches in box per 90", "Shots per 90",
        "Defensive duels per 90", "PAdj Interceptions",
    ],
}

# ── Cluster counts ─────────────────────────────────────────────────────────────
ARCHETYPE_K = {"CB": 4, "FB": 4, "MID": 7, "W": 4, "ST": 4}

# ── Archetype names per position ───────────────────────────────────────────────
ARCHETYPE_NAMES = {
    "CB":  ["Blocker", "Aggressor", "Sweeper-Distributor", "Ball-Playing CB"],
    "FB":  ["Wide Deliverer", "Inverted Modern FB", "Defensive Anchor", "Overlapping Attacker"],
    "MID": ["Space Invader", "Destroyer", "Box-Crashing Finisher", "Efficient Shooter",
            "Elite Playmaker", "Deep Balancer", "All-Rounder"],
    "W":   ["Elite Creator-Finisher", "Inverted Playmaker", "Defensive Winger", "Direct Runner"],
    "ST":  ["Pure Finisher", "False 9", "Target Man", "Pressing Forward"],
}

# ── Identification: which features most distinguish each archetype ─────────────
# Each entry = list of (feature_name, high_or_low) pairs that characterise the type
ARCHETYPE_SIGNATURES = {
    "CB": {
        "Blocker":              [("Shots blocked per 90", "high"), ("Defensive duels per 90", "high")],
        "Aggressor":            [("Aerial duels per 90", "high"), ("Fouls per 90", "high"), ("PAdj Sliding tackles", "high")],
        "Sweeper-Distributor":  [("PAdj Interceptions", "high"), ("Progressive passes per 90", "high")],
        "Ball-Playing CB":      [("Passes to final third per 90", "high"), ("Average pass length, m", "high"), ("Progressive runs per 90", "high")],
    },
    "FB": {
        "Wide Deliverer":       [("Crosses per 90", "high"), ("Passes to penalty area per 90", "high")],
        "Inverted Modern FB":   [("Dribbles per 90", "high"), ("Key passes per 90", "high"), ("Average pass length, m", "high")],
        "Defensive Anchor":     [("Defensive duels per 90", "high"), ("Aerial duels per 90", "high"), ("PAdj Interceptions", "high")],
        "Overlapping Attacker": [("Progressive runs per 90", "high"), ("Touches in box per 90", "high"), ("Shots per 90", "high")],
    },
    "MID": {
        "Space Invader":        [("Progressive runs per 90", "high"), ("Dribbles per 90", "high"), ("Touches in box per 90", "high") if "Touches in box per 90" in ARCHETYPE_FEATURES["MID"] else ("Shots per 90", "high")],
        "Destroyer":            [("Defensive duels per 90", "high"), ("PAdj Sliding tackles", "high"), ("Aerial duels per 90", "high")],
        "Box-Crashing Finisher":[("Shots per 90", "high"), ("xG per 90", "high"), ("xG per shot", "high")],
        "Efficient Shooter":    [("xG per shot", "high"), ("Shots per 90", "med")],
        "Elite Playmaker":      [("Key passes per 90", "high"), ("Through passes per 90", "high"), ("Smart passes per 90", "high")],
        "Deep Balancer":        [("Average pass length, m", "high"), ("PAdj Interceptions", "high"), ("Passes to final third per 90", "high")],
        "All-Rounder":          [("Fouls per 90", "low")],
    },
    "W": {
        "Elite Creator-Finisher":[("Touches in box per 90", "high"), ("Shots per 90", "high"), ("Progressive runs per 90", "high")],
        "Inverted Playmaker":   [("Through passes per 90", "high"), ("Smart passes per 90", "high"), ("Key passes per 90", "high")],
        "Defensive Winger":     [("Defensive duels per 90", "high"), ("PAdj Interceptions", "high")],
        "Direct Runner":        [("Dribbles per 90", "high"), ("Accelerations per 90", "high"), ("Crosses per 90", "low")],
    },
    "ST": {
        "Pure Finisher":        [("Shots per 90", "high"), ("xG per 90", "high"), ("Touches in box per 90", "high")],
        "False 9":              [("Key passes per 90", "high"), ("Smart passes per 90", "high"), ("Through passes per 90", "high")],
        "Target Man":           [("Aerial duels per 90", "high"), ("Head goals per 90", "high"), ("Received long passes per 90", "high")],
        "Pressing Forward":     [("Defensive duels per 90", "high"), ("PAdj Interceptions", "high"), ("Progressive runs per 90", "high")],
    },
}


def _available_features(df: pd.DataFrame, pos_group: str) -> list:
    """Return only features that exist as columns in df."""
    return [f for f in ARCHETYPE_FEATURES[pos_group] if f in df.columns]


def _score_cluster_for_archetype(centroid_zscores: dict, archetype: str, pos_group: str) -> float:
    """Score a centroid against an archetype signature. Higher = better match."""
    sigs = ARCHETYPE_SIGNATURES[pos_group].get(archetype, [])
    score = 0.0
    for feat, direction in sigs:
        if feat not in centroid_zscores:
            continue
        z = centroid_zscores[feat]
        if direction == "high":
            score += z
        elif direction == "low":
            score -= z
        # "med" → neutral
    return score


def _assign_names_to_clusters(kmeans, scaler, pos_group: str, features: list) -> dict:
    """Map cluster IDs → archetype names using centroid z-scores."""
    from sklearn.preprocessing import StandardScaler
    centroids_scaled = kmeans.cluster_centers_                        # shape (k, n_features)
    centroids_orig   = scaler.inverse_transform(centroids_scaled)     # back to original space
    # Re-zscore centroids relative to each other
    c_mean = centroids_orig.mean(axis=0)
    c_std  = centroids_orig.std(axis=0) + 1e-9
    centroids_z = (centroids_orig - c_mean) / c_std

    archetype_names = ARCHETYPE_NAMES[pos_group]
    k = kmeans.n_clusters

    # Build score matrix: shape (k, n_archetypes)
    scores = np.zeros((k, len(archetype_names)))
    for ci in range(k):
        cz = {features[fi]: centroids_z[ci, fi] for fi in range(len(features))}
        for ai, aname in enumerate(archetype_names):
            scores[ci, ai] = _score_cluster_for_archetype(cz, aname, pos_group)

    # Greedy assignment: each cluster gets its best unused archetype
    assignment = {}
    used_archetypes = set()
    # Sort clusters by their max score descending for stable assignment
    cluster_order = sorted(range(k), key=lambda ci: scores[ci].max(), reverse=True)
    for ci in cluster_order:
        best_ai = int(np.argmax([s if a not in used_archetypes else -999
                                 for a, s in zip(archetype_names, scores[ci])]))
        assignment[ci] = archetype_names[best_ai]
        used_archetypes.add(archetype_names[best_ai])

    # If k < n_archetypes some names won't appear; that's fine
    return assignment


def train_archetypes(data: pd.DataFrame) -> dict:
    """
    Train K-Means models for all position groups.
    Returns dict: {pos_group: {"kmeans": model, "scaler": scaler,
                                "features": list, "cluster_names": dict}}
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Filter training pool
    train_data = data.copy()
    if "League" in train_data.columns:
        train_data = train_data[train_data["League"].isin(ARCHETYPE_LEAGUES)]
    if "Minutes played" in train_data.columns:
        train_data = train_data[
            pd.to_numeric(train_data["Minutes played"], errors="coerce") >= ARCHETYPE_MIN_MINUTES
        ]

    models = {}
    for pos_group, positions in ARCHETYPE_POSITION_GROUPS.items():
        pos_col = "Main Position" if "Main Position" in train_data.columns else "Position"
        pool = train_data[train_data[pos_col].isin(positions)].copy()

        features = _available_features(pool, pos_group)
        if len(features) < 4 or len(pool) < ARCHETYPE_K[pos_group] * 5:
            print(f"[archetypes] Skipping {pos_group}: insufficient data ({len(pool)} rows, {len(features)} features)")
            continue

        X = pool[features].apply(pd.to_numeric, errors="coerce").fillna(0).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        km = KMeans(n_clusters=ARCHETYPE_K[pos_group], random_state=42, n_init=20, max_iter=500)
        km.fit(X_scaled)

        cluster_names = _assign_names_to_clusters(km, scaler, pos_group, features)

        models[pos_group] = {
            "kmeans":        km,
            "scaler":        scaler,
            "features":      features,
            "cluster_names": cluster_names,
        }
        print(f"[archetypes] Trained {pos_group}: {len(pool)} players → {cluster_names}")

    return models


def _pickle_path(pos_group: str) -> str:
    return os.path.join(MODELS_DIR, f"archetype_{pos_group}.pkl")


def _data_hash(data: pd.DataFrame) -> str:
    """Quick hash of data shape + first/last row to detect dataset changes."""
    sig = f"{data.shape}_{data.columns.tolist()}_{data.iloc[0].tolist() if len(data) else ''}"
    return hashlib.md5(sig.encode()).hexdigest()[:12]


def _hash_path() -> str:
    return os.path.join(MODELS_DIR, "archetype_data_hash.txt")


def load_or_train_models(data: pd.DataFrame, force_retrain: bool = False) -> dict:
    """
    Load pickled models if available and data hasn't changed.
    Otherwise train and persist.
    """
    current_hash = _data_hash(data)

    # Check if we should retrain
    saved_hash = ""
    if os.path.exists(_hash_path()):
        with open(_hash_path(), "r") as f:
            saved_hash = f.read().strip()

    all_exist = all(os.path.exists(_pickle_path(pg)) for pg in ARCHETYPE_POSITION_GROUPS)

    if not force_retrain and all_exist and saved_hash == current_hash:
        # Load from pickle
        models = {}
        for pos_group in ARCHETYPE_POSITION_GROUPS:
            p = _pickle_path(pos_group)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    models[pos_group] = pickle.load(f)
        return models

    # Train fresh
    print("[archetypes] Training models…")
    models = train_archetypes(data)

    # Persist
    for pos_group, model_data in models.items():
        with open(_pickle_path(pos_group), "wb") as f:
            pickle.dump(model_data, f)

    with open(_hash_path(), "w") as f:
        f.write(current_hash)

    return models


def assign_archetype(
    player_row: pd.Series,
    models: dict,
    pos_group: str,
) -> tuple:
    """
    Returns: (primary_name, secondary_name_or_None, primary_dist, secondary_dist)
    Secondary only returned if secondary_dist / primary_dist < 1.4.
    """
    if pos_group not in models:
        return "Unknown", None, 0.0, 0.0

    m         = models[pos_group]
    kmeans    = m["kmeans"]
    scaler    = m["scaler"]
    features  = m["features"]
    cnames    = m["cluster_names"]

    # Build feature vector — use 0 for missing features
    vec = np.array([pd.to_numeric(player_row.get(f, 0), errors="coerce") or 0.0
                    for f in features]).reshape(1, -1)
    vec_scaled = scaler.transform(vec)

    # Distances to all cluster centres
    dists = np.linalg.norm(kmeans.cluster_centers_ - vec_scaled, axis=1)
    order = np.argsort(dists)

    primary_id   = int(order[0])
    primary_dist = float(dists[order[0]])
    primary_name = cnames.get(primary_id, f"Type {primary_id+1}")

    secondary_name = None
    secondary_dist = float(dists[order[1]]) if len(order) > 1 else 999.0

    if primary_dist > 0 and secondary_dist / (primary_dist + 1e-9) < 1.4:
        secondary_id   = int(order[1])
        secondary_name = cnames.get(secondary_id, f"Type {secondary_id+1}")

    return primary_name, secondary_name, primary_dist, secondary_dist


def get_player_archetype(
    player_row: pd.Series,
    models: dict,
    scoring_position_group: str,
) -> tuple:
    """
    Convenience wrapper. Maps scoring position_group → archetype group, then calls assign_archetype.
    Returns: (primary, secondary_or_None, p_dist, s_dist)
    """
    arch_group = SCORING_TO_ARCHETYPE.get(scoring_position_group)
    if arch_group is None:
        return "Unknown", None, 0.0, 0.0
    return assign_archetype(player_row, models, arch_group)


# ── Archetype colour map (for UI badges) ─────────────────────────────────────
ARCHETYPE_COLORS = {
    # CB
    "Blocker":              "#4a7fa5",
    "Aggressor":            "#c0392b",
    "Sweeper-Distributor":  "#1a7a45",
    "Ball-Playing CB":      "#c9a84c",
    # FB
    "Wide Deliverer":       "#c9a84c",
    "Inverted Modern FB":   "#8e44ad",
    "Defensive Anchor":     "#4a7fa5",
    "Overlapping Attacker": "#1a7a45",
    # MID
    "Space Invader":        "#e67e22",
    "Destroyer":            "#c0392b",
    "Box-Crashing Finisher":"#1a7a45",
    "Efficient Shooter":    "#c9a84c",
    "Elite Playmaker":      "#8e44ad",
    "Deep Balancer":        "#4a7fa5",
    "All-Rounder":          "#7a7060",
    # W
    "Elite Creator-Finisher":"#c9a84c",
    "Inverted Playmaker":   "#8e44ad",
    "Defensive Winger":     "#4a7fa5",
    "Direct Runner":        "#1a7a45",
    # ST
    "Pure Finisher":        "#c0392b",
    "False 9":              "#8e44ad",
    "Target Man":           "#4a7fa5",
    "Pressing Forward":     "#e67e22",
}

def archetype_color(name: str) -> str:
    return ARCHETYPE_COLORS.get(name, "#7a7060")
