"""
transfermodel.py
================
Volledig transferwaarderingsmodel met gelaagde multipliers.

Architectuur:
    Transferwaarde = Globale basiswaarde
                   × Positie/Leeftijd multiplier   (globaal, veel data)
                   × Competitiemultiplier           (per competitie, gewogen)
                   × Clubmultiplier                 (alle posities, gewogen)
                   × Kwaliteitsmultiplier            (eigen model score 0-100)

Gebruik:
    python transfermodel.py
    → print alle multipliers
    → bereken modelwaarde voor testcases
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paden (pas aan naar jouw lokale situatie) ─────────────────────────────────
DATA_DIR  = Path(__file__).parent / "data"
TRANSFERS = DATA_DIR / "transfers.csv.gz"
PLAYERS   = DATA_DIR / "players.csv.gz"
CLUBS     = DATA_DIR / "clubs.csv.gz"
COMPS     = DATA_DIR / "competitions.csv.gz"

# ── Constanten ────────────────────────────────────────────────────────────────
KWALITEIT_EXPONENT = 1.256   # (score/50)^1.256
MIN_TRANSFERS_CLUB = 5       # minimum transfers voor betrouwbare clubmultiplier
MIN_TRANSFERS_COMP = 20      # minimum transfers voor betrouwbare competitiemultiplier
HALF_LIFE_JAREN    = 3       # exponentieel gewicht: halvering per 3 jaar

# Geschatte multipliers voor competities zonder data
GESCHATTE_COMP_MULTIPLIERS = {
    "ENG-2": 2.23,   # 60% van ENG-1 (3.71)
    "ESP-2": 1.03,   # 60% van ESP-1 (1.72)
    "ECU-1": 0.45,   # schatting op basis van vergelijkbare competities
}

# Wyscout competitienamen → Transfermarkt namen
COMP_MAP = {
    "Premier League":              "premier-league",
    "La Liga":                     "laliga",
    "Italian Serie A":             "serie-a",
    "Bundesliga":                  "bundesliga",
    "Ligue 1":                     "ligue-1",
    "Eredivisie":                  "eredivisie",
    "Pro League":                  "jupiler-pro-league",
    "Serie A BRA":                 "campeonato-brasileiro-serie-a",
    "Liga Profesional":            "torneo-apertura",
    "Primeira Liga":               "liga-portugal",
    "Ekstraklasa":                 "pko-bp-ekstraklasa",
    "MLS":                         "major-league-soccer",
    "Super Lig":                   "super-lig",
    "Prva HNL":                    "supersport-hnl",
    "Eliteserien":                 "eliteserien",
    "Superligaen":                 "superliga",
    "Swiss Super League":          "super-league",
    "Championship":                "ENG-2",   # geschat
    "Segunda Division":            "ESP-2",   # geschat
    "Liga Pro":                    "ECU-1",   # geschat
}

# ── Helper functies ───────────────────────────────────────────────────────────
def gewogen_mediaan(values, weights):
    """Bereken gewogen mediaan."""
    if len(values) == 0:
        return np.nan
    arr = np.array(values, dtype=float)
    w   = np.array(weights, dtype=float)
    mask = ~np.isnan(arr) & ~np.isnan(w) & (w > 0)
    if mask.sum() == 0:
        return np.nan
    arr, w = arr[mask], w[mask]
    idx    = np.argsort(arr)
    arr, w = arr[idx], w[idx]
    cumsum = np.cumsum(w)
    cutoff = cumsum[-1] / 2
    return float(arr[np.searchsorted(cumsum, cutoff)])


def leeftijd_categorie(leeftijd):
    """Zet leeftijd om naar categorie."""
    if leeftijd < 18:   return "<18"
    if leeftijd < 22:   return "19-21"
    if leeftijd < 25:   return "22-24"
    if leeftijd < 28:   return "25-27"
    if leeftijd < 31:   return "28-30"
    return "31+"


def kwaliteit_multiplier(score: float) -> float:
    """
    Zet eigen model score (0-100) om naar kwaliteitsmultiplier.
    Score 50 = mediaan = 1.0x
    Score 75 = 1.66x
    Score 90 = 2.09x
    Score 95 = 2.24x
    """
    score = max(1, min(99, score))
    return (score / 50) ** KWALITEIT_EXPONENT


# ── Data laden en voorbereiden ────────────────────────────────────────────────
def laad_werkset():
    """Laad en combineer alle Transfermarkt bestanden."""
    DATA_DIR  = Path(__file__).parent / "data"
    transfers = pd.read_csv(TRANSFERS, compression="gzip")
    players   = pd.read_csv(PLAYERS,   compression="gzip")
    clubs     = pd.read_csv(CLUBS,     compression="gzip")
    comps     = pd.read_csv(COMPS,     compression="gzip")

    # Filter: betaalde transfers vanaf 2018
    t = transfers.copy()
    t["transfer_date"] = pd.to_datetime(t["transfer_date"], errors="coerce")
    t["year"]          = t["transfer_date"].dt.year
    t = t[(t["transfer_fee"] > 0) & (t["year"] >= 2018) & (t["year"] < 2026)]

    # Spelerinformatie
    p = players[["player_id", "date_of_birth", "position"]].copy()
    p["date_of_birth"] = pd.to_datetime(p["date_of_birth"], errors="coerce")
    t = t.merge(p, on="player_id", how="left")
    t["age_at_transfer"] = (t["transfer_date"] - t["date_of_birth"]).dt.days / 365.25

    # Positie mapping
    pos_map = {"Attack": "Attack", "Midfield": "Midfield",
               "Defender": "Defender", "Goalkeeper": "Goalkeeper"}
    t["positie"] = t["position"].map(pos_map).fillna("Unknown")

    # Leeftijdscategorie
    bins   = [0, 18, 21, 24, 27, 30, 100]
    labels = ["<18", "19-21", "22-24", "25-27", "28-30", "31+"]
    t["leeftijd_categorie"] = pd.cut(
        t["age_at_transfer"], bins=bins, labels=labels).astype(str)

    # Clubinformatie → competitie
    c = clubs[["club_id", "domestic_competition_id"]].copy()
    c.columns = ["from_club_id", "comp_id"]
    t = t.merge(c, on="from_club_id", how="left")

    # Competitienaam + sub_type
    comp_slim = comps[["competition_id", "name", "sub_type"]].copy()
    comp_slim.columns = ["comp_id", "competitie_naam", "sub_type"]
    t = t.merge(comp_slim, on="comp_id", how="left")

    # Alleen first-tier
    t = t[t["sub_type"] == "first_tier"].copy()

    # Exponentieel tijdsgewicht (recente transfers zwaarder)
    t["gewicht"] = np.exp((t["year"] - 2018) * np.log(2) / HALF_LIFE_JAREN)

    return t


# ── Multipliers berekenen ─────────────────────────────────────────────────────
def bereken_multipliers(df):
    """
    Bereken alle vier de multiplier-lagen op basis van de werkset.
    Geeft een dict terug met alle tabellen en de globale basiswaarde.
    """

    # ── Laag 1: Globale basiswaarde ───────────────────────────────────────────
    globale_basis = gewogen_mediaan(df["transfer_fee"], df["gewicht"])

    # ── Laag 2: Positie / Leeftijd multiplier ─────────────────────────────────
    pos_leeft = {}
    for (pos, leeft), grp in df.groupby(["positie", "leeftijd_categorie"]):
        if len(grp) < 5:
            continue
        m = gewogen_mediaan(grp["transfer_fee"], grp["gewicht"])
        pos_leeft[(pos, leeft)] = m / globale_basis

    # Residuen na pos/leeft correctie
    df = df.copy()
    df["mult_pl"] = df.apply(
        lambda r: pos_leeft.get((r["positie"], r["leeftijd_categorie"]), 1.0), axis=1)
    df["verwacht_2"] = globale_basis * df["mult_pl"]
    df["residu_2"]   = df["transfer_fee"] / df["verwacht_2"]

    # ── Laag 3: Competitiemultiplier ──────────────────────────────────────────
    comp_mult = {}
    comp_stats = []
    for comp, grp in df.groupby("competitie_naam"):
        if len(grp) < MIN_TRANSFERS_COMP:
            continue
        m = gewogen_mediaan(grp["residu_2"], grp["gewicht"])
        gew_fee = gewogen_mediaan(grp["transfer_fee"], grp["gewicht"])

        # Trend: 2018-2020 vs 2023-2025
        vroeg = grp[grp["year"].isin([2018, 2019, 2020])]["transfer_fee"].median()
        laat  = grp[grp["year"].isin([2023, 2024, 2025])]["transfer_fee"].median()
        trend = laat / vroeg if vroeg > 0 else 1.0

        comp_mult[comp] = m
        comp_stats.append({
            "competitie": comp, "multiplier": m,
            "gew_mediaan_fee": gew_fee, "n": len(grp),
            "trend_2018_vs_2025": trend,
        })

    comp_df = pd.DataFrame(comp_stats).sort_values("multiplier", ascending=False)

    # Voeg geschatte competities toe
    for code, mult in GESCHATTE_COMP_MULTIPLIERS.items():
        comp_df = pd.concat([comp_df, pd.DataFrame([{
            "competitie": code, "multiplier": mult,
            "gew_mediaan_fee": np.nan, "n": 0,
            "trend_2018_vs_2025": np.nan,
        }])], ignore_index=True)

    # Residuen na competitiecorrectie
    df["mult_comp"] = df["competitie_naam"].map(comp_mult).fillna(1.0)
    df["verwacht_3"] = df["verwacht_2"] * df["mult_comp"]
    df["residu_3"]   = df["transfer_fee"] / df["verwacht_3"]

    # ── Laag 4: Clubmultiplier ────────────────────────────────────────────────
    club_mult  = {}
    club_stats = []
    for club, grp in df.groupby("from_club_name"):
        if len(grp) < MIN_TRANSFERS_CLUB:
            continue
        m = gewogen_mediaan(grp["residu_3"], grp["gewicht"])
        club_mult[club] = m
        club_stats.append({
            "club": club,
            "competitie": grp["competitie_naam"].iloc[0],
            "multiplier": m,
            "n": len(grp),
        })

    club_df = pd.DataFrame(club_stats).sort_values("multiplier", ascending=False)

    return {
        "globale_basis":    globale_basis,
        "pos_leeft":        pos_leeft,
        "comp_mult":        comp_mult,
        "comp_df":          comp_df,
        "club_mult":        club_mult,
        "club_df":          club_df,
        "df_met_residuen":  df,
    }


# ── Modelberekening voor één speler ──────────────────────────────────────────
def bereken_transferwaarde(
    multipliers: dict,
    positie: str,          # "Attack" / "Midfield" / "Defender" / "Goalkeeper"
    leeftijd: float,       # leeftijd in jaren
    competitie: str,       # Wyscout competitienaam (bijv. "Eredivisie")
    club: str,             # clubnaam zoals in Transfermarkt
    model_score: float,    # eigen model score 0-100
    verbose: bool = True,
) -> dict:
    """
    Bereken transferwaarde voor één speler.
    Geeft dict terug met alle tussenstappen en het eindresultaat.
    """
    gb      = multipliers["globale_basis"]
    pl_map  = multipliers["pos_leeft"]
    cm_map  = multipliers["comp_mult"]
    club_map= multipliers["club_mult"]

    leeft_cat = leeftijd_categorie(leeftijd)

    # Laag 2: pos/leeft
    pl_mult = pl_map.get((positie, leeft_cat), 1.0)

    # Laag 3: competitie
    tm_comp = COMP_MAP.get(competitie)
    if tm_comp in GESCHATTE_COMP_MULTIPLIERS:
        c_mult = GESCHATTE_COMP_MULTIPLIERS[tm_comp]
        c_bron = "schatting"
    elif tm_comp in cm_map:
        c_mult = cm_map[tm_comp]
        c_bron = "data"
    else:
        c_mult = 1.0
        c_bron = "default"

    # Laag 4: club
    cl_mult = club_map.get(club, 1.0)
    cl_bron = "data" if club in club_map else "default 1.0"

    # Laag 5: kwaliteit
    kw_mult = kwaliteit_multiplier(model_score)

    # Eindresultaat
    waarde = gb * pl_mult * c_mult * cl_mult * kw_mult

    result = {
        "globale_basis":        gb,
        "positie":              positie,
        "leeftijd_categorie":   leeft_cat,
        "pl_multiplier":        pl_mult,
        "comp_multiplier":      c_mult,
        "comp_bron":            c_bron,
        "club_multiplier":      cl_mult,
        "club_bron":            cl_bron,
        "kwaliteit_multiplier": kw_mult,
        "model_score":          model_score,
        "transferwaarde":       waarde,
    }

    if verbose:
        print(f"\n{'='*55}")
        print(f"  {positie} | {leeftijd:.0f} jr ({leeft_cat}) | {competitie} | {club}")
        print(f"  Model score: {model_score:.0f}/100")
        print(f"{'='*55}")
        print(f"  Globale basis:        €{gb/1e6:>6.2f}M")
        print(f"  × Pos/leeftijd:       {pl_mult:>7.3f}×")
        print(f"  × Competitie ({c_bron:<9}): {c_mult:>7.3f}×")
        print(f"  × Club ({cl_bron:<14}): {cl_mult:>7.3f}×")
        print(f"  × Kwaliteit:          {kw_mult:>7.3f}×")
        print(f"{'─'*55}")
        print(f"  Transferwaarde:  €{waarde/1e6:>7.1f}M")
        print(f"{'='*55}")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("Laden van data...")
    df = laad_werkset()
    print(f"Werkset: {len(df):,} transfers")

    print("\nBerekenen van multipliers...")
    m = bereken_multipliers(df)

    print(f"\nGlobale basiswaarde: €{m['globale_basis']/1e6:.2f}M")

    # ── Competitiemultipliers ──────────────────────────────────────────────────
    print("\n" + "="*70)
    print("COMPETITIEMULTIPLIERS")
    print("="*70)
    print(f"{'Competitie':<40} {'Mult':>7} {'Gew. fee':>10} {'n':>6} {'Trend':>8}")
    print("-"*70)
    for _, row in m["comp_df"].sort_values("multiplier", ascending=False).iterrows():
        fee_str   = f"€{row['gew_mediaan_fee']/1e6:.1f}M" if not pd.isna(row["gew_mediaan_fee"]) else "schatting"
        trend_str = f"{row['trend_2018_vs_2025']:.2f}×"   if not pd.isna(row["trend_2018_vs_2025"]) else "—"
        print(f"{row['competitie']:<40} {row['multiplier']:>7.3f}× {fee_str:>10} {int(row['n']):>6} {trend_str:>8}")

    # ── Clubmultipliers (top 40) ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("CLUBMULTIPLIERS (top 40, min 5 transfers)")
    print("="*70)
    print(f"{'Club':<30} {'Competitie':<25} {'Mult':>7} {'n':>6}")
    print("-"*70)
    for _, row in m["club_df"].head(40).iterrows():
        print(f"{row['club']:<30} {row['competitie']:<25} {row['multiplier']:>7.3f}× {int(row['n']):>6}")

    # ── Positie/leeftijd multipliers ──────────────────────────────────────────
    print("\n" + "="*55)
    print("POSITIE/LEEFTIJD MULTIPLIERS")
    print("="*55)
    pl_df = pd.DataFrame([
        {"positie": k[0], "leeftijd": k[1], "multiplier": v}
        for k, v in m["pos_leeft"].items()
    ]).sort_values(["positie", "leeftijd"])
    print(pl_df.to_string(index=False))

    # ── Kwaliteitsmultiplier curve ────────────────────────────────────────────
    print("\n" + "="*45)
    print("KWALITEITSMULTIPLIER CURVE")
    print("="*45)
    print(f"  Formule: (score / 50) ^ {KWALITEIT_EXPONENT}")
    print(f"  {'Score':>6} {'Multiplier':>12}")
    print(f"  {'─'*20}")
    for s in [10, 25, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99]:
        print(f"  {s:>6}   {kwaliteit_multiplier(s):>10.3f}×")

    # ── Testcases ─────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("VALIDATIE OP BEKENDE TRANSFERS")
    print("="*55)

    testcases = [
        # naam,              pos,        leeft, competitie,          club,       score, echte_fee
        ("Frenkie de Jong",  "Midfield", 21.5,  "Eredivisie",        "Ajax",      92,   86_000_000),
        ("Timber",           "Defender", 21.8,  "Eredivisie",        "Ajax",      78,   40_000_000),
        ("Bellingham",       "Midfield", 19.4,  "Bundesliga",        "Dortmund",  95,  103_000_000),
        ("Dumfries",         "Defender", 25.2,  "Eredivisie",        "PSV",       65,   14_000_000),
        ("Koopmeiners",      "Midfield", 25.6,  "Italian Serie A",   "Atalanta",  82,   59_000_000),
        ("Neres",            "Attack",   25.8,  "Eredivisie",        "Ajax",      68,   15_000_000),
        ("Gravenberch",      "Midfield", 20.1,  "Eredivisie",        "Ajax",      85,   18_500_000),
    ]

    print(f"\n{'Speler':<20} {'Model':>9} {'Echt':>9} {'Ratio':>7}")
    print("-"*48)
    for naam, pos, leeft, comp, club, score, echt in testcases:
        r = bereken_transferwaarde(
            m, pos, leeft, comp, club, score, verbose=False)
        ratio = r["transferwaarde"] / echt
        print(f"{naam:<20} €{r['transferwaarde']/1e6:>6.1f}M €{echt/1e6:>6.1f}M {ratio:>7.2f}×")

    # ── Voorbeeld verbose berekening ──────────────────────────────────────────
    print("\n\nVOORBEELD: GEDETAILLEERDE BEREKENING")
    bereken_transferwaarde(
        m,
        positie="Attack",
        leeftijd=22,
        competitie="Eredivisie",
        club="Ajax",
        model_score=78,
    )
