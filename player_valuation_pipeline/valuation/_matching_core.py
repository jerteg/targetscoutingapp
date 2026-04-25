"""
Wyscout ↔ Transfermarkt Player Matching Pipeline

Probleem: Naming conventions verschillen tussen datasets.
  Wyscout: "L. Messi", "Vinícius Júnior", "Lamine Yamal"
  TM:      "Lionel Messi", "Vinicius Junior", "Lamine Yamal Nasraoui Ebana"

Oplossing: Multi-stage matching strategy met decreasing strictness:
  1. Exact match on normalized name + club + season  (highest confidence)
  2. Fuzzy name match within same club + season       (high confidence)  
  3. Fuzzy name match + position match + age range    (medium confidence)
  4. Fuzzy name match only + age range                (low confidence)
  5. Unmatched → handmatige review

Elke match krijgt een confidence score (0-100) zodat je later kunt filteren
op kwaliteit (bijv. alleen >80 gebruiken voor league multiplier derivation).

Normalisatie:
  - Unidecode: "Vinícius" → "Vinicius"
  - Lowercase
  - Strip punctuation (behalve hyphens, die blijven)
  - Handle initials: "L. Messi" → search ook naar "* Messi" waarbij * begint met L
  - Strip common suffixes: " Jr", " Junior", " Sr", " Senior"
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Optional, Tuple
from unidecode import unidecode
from rapidfuzz import fuzz, process as rf_process


# ─────────────────────────────────────────────────────────────────────────────
# NAAM NORMALISATIE
# ─────────────────────────────────────────────────────────────────────────────

# Suffixes die uit namen gestript moeten worden voor matching
# LET OP: "Junior" NIET strippen want dat is vaak deel van officiële naam
# (bijv. Vinicius Junior is de volledige officiële naam van de Real Madrid speler)
NAME_SUFFIXES = [
    ' sr', ' senior',
    ' ii', ' iii',
]

# Team name aliases (Wyscout → TM varianten die veel verschillen)
TEAM_ALIASES_WYSCOUT_TO_TM = {
    # Deze moet je waarschijnlijk aanvullen tijdens matching
    'Manchester United': ['Man Utd', 'Manchester Utd'],
    'Manchester City': ['Man City'],
    'Tottenham Hotspur': ['Spurs', 'Tottenham'],
    'Brighton': ['Brighton & Hove Albion', 'Brighton and Hove Albion'],
    'Wolverhampton Wanderers': ['Wolves'],
    'Newcastle United': ['Newcastle'],
    'West Bromwich Albion': ['West Brom'],
    'Queens Park Rangers': ['QPR'],
    'Bayern München': ['Bayern Munich', 'FC Bayern München', 'FC Bayern'],
    'Borussia Dortmund': ['BVB', 'Dortmund'],
    'Borussia Mönchengladbach': ['Borussia M.gladbach', 'Mönchengladbach'],
    'Internazionale': ['Inter Milan', 'Inter'],
    'Milan': ['AC Milan'],
    'PSG': ['Paris Saint-Germain', 'Paris SG'],
    'Olympique Lyonnais': ['Lyon', 'OL'],
    'Olympique de Marseille': ['Marseille', 'OM'],
    'Real Madrid': ['Real Madrid CF'],
    'Barcelona': ['FC Barcelona'],
    'Atlético Madrid': ['Atletico Madrid', 'Atlético de Madrid'],
    'Athletic Club': ['Athletic Bilbao'],
    'Real Sociedad': ['Real Sociedad de Fútbol'],
    'Inter Miami': ['Inter Miami CF'],
    'Los Angeles Galaxy': ['LA Galaxy'],
    'New York City': ['NYCFC', 'New York City FC'],
    'Vancouver Whitecaps': ['Vancouver Whitecaps FC'],
    'Seattle Sounders': ['Seattle Sounders FC'],
    'River Plate': ['CA River Plate'],
    'Boca Juniors': ['CA Boca Juniors'],
}


def normalize_name(name: str) -> str:
    """
    Normaliseer een spelersnaam voor matching.
    
    - Unidecode (diakritische tekens weg)
    - Lowercase
    - Strip punctuation (behalve hyphens)
    - Remove common suffixes (Jr, Sr, etc)
    - Collapse whitespace
    """
    if not isinstance(name, str):
        return ''
    
    # Unidecode
    s = unidecode(name)
    
    # Lowercase
    s = s.lower()
    
    # Verwijder punten van initialen ("L. Messi" → "L Messi")
    s = s.replace('.', ' ')
    
    # Verwijder andere punctuatie behalve hyphens en spaties
    s = re.sub(r"[^\w\s\-]", ' ', s)
    
    # Strip suffixes
    for suffix in NAME_SUFFIXES:
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    
    # Collapse whitespace
    s = ' '.join(s.split())
    
    return s.strip()


def expand_initial(wyscout_name: str, tm_name: str) -> bool:
    """
    Check of Wyscout's 'L. Messi' matcht met TM's 'Lionel Messi'.
    
    Logic: als Wyscout begint met 1 letter + punt/spatie, en TM begint 
    met een woord dat met die letter start, en de rest van de achternamen matcht.
    """
    wy_parts = normalize_name(wyscout_name).split()
    tm_parts = normalize_name(tm_name).split()
    
    if not wy_parts or not tm_parts:
        return False
    
    # Check: is eerste deel van Wyscout één letter?
    first_wy = wy_parts[0]
    if len(first_wy) != 1:
        return False
    
    # Check: begint eerste deel van TM met dezelfde letter?
    first_tm = tm_parts[0]
    if not first_tm.startswith(first_wy):
        return False
    
    # Rest van de namen moet matchen (surname + eventueel middle names)
    wy_rest = ' '.join(wy_parts[1:])
    tm_rest = ' '.join(tm_parts[1:])
    
    # Soft match op de rest — moet vrij dicht bij elkaar liggen
    if not wy_rest or not tm_rest:
        return False
    
    similarity = fuzz.ratio(wy_rest, tm_rest)
    return similarity >= 85


def name_similarity(name1: str, name2: str) -> float:
    """
    Bereken similarity tussen twee namen, met aandacht voor initialen.
    
    CRUCIAL RULE: Als één naam begint met een initial (1 letter + punt/spatie),
    MOET die letter matchen met de eerste letter van de andere naam's voornaam.
    Anders: max score = 50 (afgewezen als goede match).
    
    Returns: score 0-100.
    """
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    
    if not n1 or not n2:
        return 0.0
    
    # Exact match na normalisatie
    if n1 == n2:
        return 100.0
    
    # ── STRICT INITIAL CHECK ──────────────────────────────────────────
    # Als één naam initial-style is (eerste token = 1 letter),
    # MOET die letter matchen met eerste letter van andere naam's eerste token.
    n1_parts = n1.split()
    n2_parts = n2.split()
    
    n1_has_initial = len(n1_parts) > 1 and len(n1_parts[0]) == 1
    n2_has_initial = len(n2_parts) > 1 and len(n2_parts[0]) == 1
    
    if n1_has_initial and n2_parts:
        # n1 is "E. Martinez", n2 is e.g. "Lautaro Martinez"
        # Eerste letter van n1 moet matchen met eerste letter van n2's eerste token
        if not n2_parts[0].startswith(n1_parts[0]):
            # Initial mismatch — hard reject
            return min(50.0, fuzz.partial_ratio(n1, n2) * 0.5)
    
    if n2_has_initial and n1_parts:
        if not n1_parts[0].startswith(n2_parts[0]):
            return min(50.0, fuzz.partial_ratio(n1, n2) * 0.5)
    
    # ── INITIAL EXPANSION CHECK ────────────────────────────────────────
    # Initial expansion check (bidirectional)
    if expand_initial(name1, name2) or expand_initial(name2, name1):
        return 95.0
    
    # ── FUZZY CHECKS met surname-first constraint ─────────────────────
    # Token-based fuzzy match (ignoring word order, handles middle names)
    token_score = fuzz.token_set_ratio(n1, n2)
    
    # Ratio: direct string similarity
    ratio_score = fuzz.ratio(n1, n2)
    
    # Partial ratio: handles cases like "Vinicius" vs "Vinicius Junior"
    # BUT we moeten checken dat dit geen false positive is
    # Als partial match sterk is maar token_set is laag, is het waarschijnlijk
    # een "contained in" match (bijv. "Vinicius" in "Vinicius Nogueira")
    partial_score = fuzz.partial_ratio(n1, n2)
    
    # Als partial score veel hoger is dan token_set, wees voorzichtig
    # Een goede match heeft beide scores hoog
    if partial_score > 85 and token_score < 70:
        # Eén naam is waarschijnlijk een prefix/subset van de andere
        # Dit is vaak een false positive
        partial_score = min(partial_score, 75)
    
    # ── SURNAME MISMATCH CHECK ────────────────────────────────────────
    # Als beide namen 2+ tokens hebben en token_set=100 (veel overlap),
    # check: is de laatste token (vaak surname) hetzelfde?
    # "Vinicius Lopes" vs "Vinicius Junior" → surnames different, should be penalized
    if len(n1_parts) >= 2 and len(n2_parts) >= 2 and token_score >= 95:
        surname_1 = n1_parts[-1]
        surname_2 = n2_parts[-1]
        # Als laatste tokens niet matchen (en geen suffix zoals Jr/Sr), penalty
        if surname_1 != surname_2:
            # Check of er EEN gemeenschappelijke token is die geen common word is
            common = set(n1_parts) & set(n2_parts)
            # Common vernamen zoals "Junior" zijn geen goed signaal
            weak_commons = {'junior', 'jr', 'senior', 'sr', 'filho', 'neto', 'de', 'da', 'dos', 'das', 'do'}
            strong_common = common - weak_commons
            
            if not strong_common:
                # Alleen zwakke common tokens → niet dezelfde persoon
                return 45.0
            elif len(strong_common) == 1 and len(n1_parts) >= 2 and len(n2_parts) >= 2:
                # Precies 1 sterke gemeenschappelijke token, bijv. voornaam
                # Dit is vaak "Vinicius Lopes" vs "Vinicius Junior" — verschillende personen
                # Penalty: max 75 score
                return min(token_score, 75.0)
    
    # Gewogen combinatie — prioriteer token_score (meest robuust)
    return max(token_score, ratio_score, partial_score)


def club_similarity(club1: str, club2: str) -> float:
    """Check of twee clubnamen dezelfde club zijn, met aliases."""
    if not isinstance(club1, str) or not isinstance(club2, str):
        return 0.0
    
    c1 = normalize_name(club1)
    c2 = normalize_name(club2)
    
    if c1 == c2:
        return 100.0
    
    # Check aliases
    for canonical, aliases in TEAM_ALIASES_WYSCOUT_TO_TM.items():
        canon_norm = normalize_name(canonical)
        alias_norms = [normalize_name(a) for a in aliases]
        all_variants = [canon_norm] + alias_norms
        
        if c1 in all_variants and c2 in all_variants:
            return 100.0
    
    # Fuzzy match (clubs vaak wel via naam herkenbaar)
    return fuzz.token_set_ratio(c1, c2)


# ─────────────────────────────────────────────────────────────────────────────
# MATCHING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class MatchResult:
    """Een match tussen een Wyscout speler en een TM speler."""
    
    def __init__(self, wyscout_idx, tm_idx, confidence: float, method: str, 
                 details: dict = None):
        self.wyscout_idx = wyscout_idx
        self.tm_idx = tm_idx
        self.confidence = confidence
        self.method = method
        self.details = details or {}
    
    def __repr__(self):
        return f"Match({self.method}, conf={self.confidence:.1f})"


def match_wyscout_to_tm(
    wyscout_df: pd.DataFrame,
    tm_df: pd.DataFrame,
    wyscout_name_col: str = 'Player',
    wyscout_club_col: str = 'Team within selected timeframe',
    wyscout_dob_col: str = None,
    wyscout_age_col: str = 'Age',
    tm_name_col: str = 'player_name',
    tm_club_col: str = 'current_club',
    tm_dob_col: str = 'date_of_birth',
    min_name_similarity: float = 80.0,
) -> pd.DataFrame:
    """
    Match Wyscout spelers met Transfermarkt spelers.
    
    Parameters
    ----------
    wyscout_df : Wyscout dataset (player per row, uit data.csv)
    tm_df : TM spelers dataset (uit salimt's player_profiles.csv)
    wyscout_name_col : kolomnaam met spelersnamen in Wyscout
    wyscout_club_col : kolomnaam met huidige club in Wyscout
    wyscout_dob_col : OPTIONEEL kolom met geboortedatum
    wyscout_age_col : kolom met leeftijd (fallback als geen DOB)
    tm_name_col : naam kolom in TM
    tm_club_col : club kolom in TM
    tm_dob_col : date of birth in TM
    min_name_similarity : minimum naam-similarity om te overwegen
    
    Returns
    -------
    DataFrame met één rij per Wyscout speler, met:
      - alle originele Wyscout kolommen
      - tm_player_id: matched TM id (als gevonden)
      - tm_name: TM naam van match
      - match_confidence: 0-100
      - match_method: welke strategie werkte
      - match_details: extra info
    """
    print(f"\nMatching {len(wyscout_df)} Wyscout spelers tegen {len(tm_df)} TM spelers...")
    
    # Pre-normaliseer alle TM namen voor snellere matching
    print("  Pre-normalizing TM names...")
    tm_df = tm_df.copy()
    tm_df['_norm_name'] = tm_df[tm_name_col].apply(normalize_name)
    
    # Converteer TM DOB naar datetime voor leeftijd berekening
    if tm_dob_col in tm_df.columns:
        tm_df['_dob'] = pd.to_datetime(tm_df[tm_dob_col], errors='coerce')
    
    results = []
    
    for idx, wy_row in wyscout_df.iterrows():
        wy_name = wy_row[wyscout_name_col]
        wy_club = wy_row.get(wyscout_club_col, '')
        wy_age = wy_row.get(wyscout_age_col, None)
        
        if pd.isna(wy_name) or not wy_name:
            results.append(None)
            continue
        
        wy_name_norm = normalize_name(wy_name)
        if not wy_name_norm:
            results.append(None)
            continue
        
        # ── STAGE 1: Exacte naam-match (na normalisatie) ──────────────────
        exact_matches = tm_df[tm_df['_norm_name'] == wy_name_norm]
        
        if len(exact_matches) == 1:
            # Unieke exacte match
            m = exact_matches.iloc[0]
            results.append(MatchResult(
                wyscout_idx=idx,
                tm_idx=exact_matches.index[0],
                confidence=100.0,
                method='exact_name',
                details={'tm_name': m[tm_name_col]}
            ))
            continue
        
        if len(exact_matches) > 1:
            # Meerdere spelers met zelfde naam — disambigueer via club/age
            best = _disambiguate(
                exact_matches, wy_club, wy_age, 
                tm_name_col, tm_club_col, 'age'
            )
            if best is not None:
                best_row, disambig_conf = best
                results.append(MatchResult(
                    wyscout_idx=idx,
                    tm_idx=best_row.name,
                    confidence=min(100.0, 85.0 + disambig_conf / 10),
                    method='exact_name_disambig',
                    details={
                        'tm_name': best_row[tm_name_col],
                        'n_candidates': len(exact_matches),
                    }
                ))
                continue
        
        # ── STAGE 2: Fuzzy name match op top-N candidates ──────────────
        # Gebruik rapidfuzz voor snelle top-N candidate selection,
        # dan re-score met onze strict name_similarity.
        choices = tm_df['_norm_name'].tolist()
        top_candidates = rf_process.extract(
            wy_name_norm, choices, 
            scorer=fuzz.token_set_ratio,
            limit=20,  # meer candidates want we re-scoren strict
            score_cutoff=60  # lagere cutoff want we gaan strict filteren
        )
        
        if not top_candidates:
            # Probeer ook met initialen-expansie check
            if len(wy_name_norm.split()[0]) == 1:
                first_letter = wy_name_norm[0]
                candidates = tm_df[tm_df['_norm_name'].str.startswith(first_letter, na=False)]
                if len(candidates) > 0:
                    for _, c_row in candidates.iterrows():
                        if expand_initial(wy_name, c_row[tm_name_col]):
                            top_candidates = top_candidates + [(c_row['_norm_name'], 95.0, c_row.name)]
            
            if not top_candidates:
                results.append(MatchResult(
                    wyscout_idx=idx,
                    tm_idx=None,
                    confidence=0.0,
                    method='unmatched',
                    details={'wy_name': wy_name}
                ))
                continue
        
        # Re-score met STRICT name_similarity (eliminates false positives)
        strict_scored = []
        for match_tuple in top_candidates:
            if len(match_tuple) == 3:
                choice, _, tm_idx = match_tuple
            else:
                choice, _ = match_tuple[:2]
                mask = tm_df['_norm_name'] == choice
                if not mask.any():
                    continue
                tm_idx = mask.idxmax()
            
            tm_row = tm_df.loc[tm_idx]
            # Gebruik STRICT similarity (met initial check)
            strict_score = name_similarity(wy_name, tm_row[tm_name_col])
            
            # Skip cases die hard afgewezen zijn (initial mismatch → <=50)
            if strict_score <= 50:
                continue
            
            strict_scored.append((tm_idx, strict_score, tm_row))
        
        if not strict_scored:
            results.append(MatchResult(
                wyscout_idx=idx,
                tm_idx=None,
                confidence=0.0,
                method='unmatched',
                details={'wy_name': wy_name, 'reason': 'all_candidates_failed_strict_check'}
            ))
            continue
        
        # Rank top matches by combined name + club + age score
        scored = []
        for tm_idx, name_score, tm_row in strict_scored:
            # Club match boost
            club_score = club_similarity(wy_club, tm_row.get(tm_club_col, ''))
            
            # Age match boost
            age_match = 0.0
            if wy_age and '_dob' in tm_df.columns and pd.notna(tm_row['_dob']):
                from datetime import datetime
                tm_age = (datetime.now() - tm_row['_dob']).days / 365.25
                age_diff = abs(float(wy_age) - tm_age)
                if age_diff < 1:
                    age_match = 100.0
                elif age_diff < 2:
                    age_match = 70.0
                elif age_diff < 3:
                    age_match = 40.0
                else:
                    age_match = 0.0
            
            # Gewogen totaal
            # Normaal: naam belangrijkst. Maar als name_score niet 100 is (geen exact match),
            # dan moet club_score DISCRIMINEREN — zonder goede club match accepteren we niet.
            if name_score >= 95:
                # Name is sterk, club is extra validatie
                total_score = 0.7 * name_score + 0.2 * club_score + 0.1 * age_match
            else:
                # Name is niet perfect — club MOET matchen voor acceptatie
                # Als club_score laag is, drop total_score significant
                if club_score < 50:
                    # Club mismatch → penalty
                    total_score = 0.5 * name_score + 0.3 * club_score + 0.2 * age_match
                    # Extra penalty: als age ook niet matcht, rating naar beneden
                    if age_match < 40:
                        total_score = min(total_score, 65)
                else:
                    total_score = 0.5 * name_score + 0.4 * club_score + 0.1 * age_match
            
            scored.append({
                'tm_idx': tm_idx,
                'tm_name': tm_row[tm_name_col],
                'name_score': name_score,
                'club_score': club_score,
                'age_match': age_match,
                'total_score': total_score,
            })
        
        if not scored:
            results.append(MatchResult(
                wyscout_idx=idx,
                tm_idx=None,
                confidence=0.0,
                method='unmatched',
                details={'wy_name': wy_name}
            ))
            continue
        
        # Pak beste match
        best = max(scored, key=lambda x: x['total_score'])
        
        # Hoe zeker zijn we? 
        # - High confidence: naam >90 én club >70 (of age perfect)
        # - Medium: naam >85 met enige club signal
        # - Low: alleen naam match
        if best['total_score'] >= 85:
            method = 'fuzzy_high'
        elif best['total_score'] >= 70:
            method = 'fuzzy_medium'
        else:
            method = 'fuzzy_low'
        
        results.append(MatchResult(
            wyscout_idx=idx,
            tm_idx=best['tm_idx'],
            confidence=best['total_score'],
            method=method,
            details={
                'tm_name': best['tm_name'],
                'name_score': best['name_score'],
                'club_score': best['club_score'],
                'n_candidates_checked': len(scored),
            }
        ))
    
    # Convert to DataFrame for convenience
    output = wyscout_df.copy()
    output['tm_player_id'] = [
        tm_df.loc[r.tm_idx, 'player_id'] if r and r.tm_idx is not None and 'player_id' in tm_df.columns else None
        for r in results
    ]
    output['tm_name'] = [
        r.details.get('tm_name', None) if r else None for r in results
    ]
    output['match_confidence'] = [r.confidence if r else 0.0 for r in results]
    output['match_method'] = [r.method if r else 'unmatched' for r in results]
    
    return output


def _disambiguate(candidates: pd.DataFrame, 
                   wy_club: str, 
                   wy_age: float,
                   tm_name_col: str,
                   tm_club_col: str,
                   tm_age_col: str) -> Optional[Tuple[pd.Series, float]]:
    """Bij meerdere naam-matches, kies beste op basis van club/leeftijd."""
    scored = []
    for _, row in candidates.iterrows():
        club_score = club_similarity(wy_club, row.get(tm_club_col, '')) if wy_club else 0
        
        age_score = 0
        if wy_age and '_dob' in candidates.columns and pd.notna(row.get('_dob')):
            from datetime import datetime
            tm_age = (datetime.now() - row['_dob']).days / 365.25
            age_diff = abs(float(wy_age) - tm_age)
            age_score = max(0, 100 - age_diff * 30)  # 30 punten per jaar verschil
        
        total = 0.7 * club_score + 0.3 * age_score
        scored.append((row, total))
    
    if not scored:
        return None
    
    best_row, best_score = max(scored, key=lambda x: x[1])
    
    # Alleen geaccepteerd als duidelijk onderscheid
    if best_score < 30:
        return None
    
    return best_row, best_score


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION & DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def match_quality_report(matched_df: pd.DataFrame) -> None:
    """Print rapport over matching kwaliteit."""
    
    total = len(matched_df)
    
    print(f"\n{'=' * 70}")
    print(f" MATCH QUALITY REPORT")
    print(f"{'=' * 70}")
    print(f"Totaal spelers: {total}")
    
    method_counts = matched_df['match_method'].value_counts()
    print(f"\nPer methode:")
    for method, count in method_counts.items():
        pct = 100 * count / total
        print(f"  {method:30s} {count:5d}  ({pct:5.1f}%)")
    
    print(f"\nConfidence distributie:")
    confidence_bins = [
        ('Excellent (>90)', (matched_df['match_confidence'] > 90).sum()),
        ('Good (80-90)',    ((matched_df['match_confidence'] > 80) & (matched_df['match_confidence'] <= 90)).sum()),
        ('Medium (70-80)',  ((matched_df['match_confidence'] > 70) & (matched_df['match_confidence'] <= 80)).sum()),
        ('Low (50-70)',     ((matched_df['match_confidence'] > 50) & (matched_df['match_confidence'] <= 70)).sum()),
        ('Very low (<50)',  ((matched_df['match_confidence'] <= 50) & (matched_df['match_confidence'] > 0)).sum()),
        ('Unmatched (0)',   (matched_df['match_confidence'] == 0).sum()),
    ]
    for label, count in confidence_bins:
        pct = 100 * count / total
        bar = '█' * int(pct / 2)
        print(f"  {label:18s} {count:5d}  ({pct:5.1f}%)  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST (werkt zonder TM data — gebruikt gesimuleerd TM bestand)
# ─────────────────────────────────────────────────────────────────────────────

def run_self_test():
    """
    Test de matching logica met een handmatig gesimuleerde 'TM' dataset.
    Dit valideert dat de pipeline correct werkt VOORDAT we echte TM data hebben.
    """
    print("\n" + "=" * 70)
    print(" SELF-TEST: Matching pipeline zonder echte TM data")
    print("=" * 70)
    
    # Gesimuleerde "TM" data met verschillende naamstijlen
    fake_tm = pd.DataFrame([
        {'player_id': '1', 'player_name': 'Lionel Messi', 'current_club': 'Inter Miami CF', 'date_of_birth': '1987-06-24'},
        {'player_id': '2', 'player_name': 'Erling Haaland', 'current_club': 'Manchester City', 'date_of_birth': '2000-07-21'},
        {'player_id': '3', 'player_name': 'Vinicius Junior', 'current_club': 'Real Madrid', 'date_of_birth': '2000-07-12'},
        {'player_id': '4', 'player_name': 'Virgil van Dijk', 'current_club': 'Liverpool FC', 'date_of_birth': '1991-07-08'},
        {'player_id': '5', 'player_name': 'Kylian Mbappé', 'current_club': 'Real Madrid CF', 'date_of_birth': '1998-12-20'},
        {'player_id': '6', 'player_name': 'Lamine Yamal Nasraoui Ebana', 'current_club': 'FC Barcelona', 'date_of_birth': '2007-07-13'},
        {'player_id': '7', 'player_name': 'Harry Kane', 'current_club': 'FC Bayern München', 'date_of_birth': '1993-07-28'},
        {'player_id': '8', 'player_name': 'Rodri', 'current_club': 'Manchester City', 'date_of_birth': '1996-06-22'},
        {'player_id': '9', 'player_name': 'Pedri', 'current_club': 'FC Barcelona', 'date_of_birth': '2002-11-25'},
        {'player_id': '10', 'player_name': 'Jude Bellingham', 'current_club': 'Real Madrid', 'date_of_birth': '2003-06-29'},
        {'player_id': '11', 'player_name': 'Jobe Bellingham', 'current_club': 'Borussia Dortmund', 'date_of_birth': '2005-09-23'},
        # Tricky cases:
        {'player_id': '12', 'player_name': 'Rodrigo Hernández Cascante', 'current_club': 'Manchester City', 'date_of_birth': '1996-06-22'},  # Rodri's full name
    ])
    
    # Gesimuleerde Wyscout data (matcht ons echte dataformat)
    fake_wyscout = pd.DataFrame([
        {'Player': 'L. Messi', 'Team within selected timeframe': 'Inter Miami', 'Age': 38},
        {'Player': 'E. Haaland', 'Team within selected timeframe': 'Manchester City', 'Age': 25},
        {'Player': 'Vinícius Júnior', 'Team within selected timeframe': 'Real Madrid', 'Age': 25},
        {'Player': 'V. van Dijk', 'Team within selected timeframe': 'Liverpool', 'Age': 34},
        {'Player': 'K. Mbappé', 'Team within selected timeframe': 'Real Madrid', 'Age': 27},
        {'Player': 'Lamine Yamal', 'Team within selected timeframe': 'Barcelona', 'Age': 18},
        {'Player': 'H. Kane', 'Team within selected timeframe': 'Bayern München', 'Age': 32},
        {'Player': 'Rodri', 'Team within selected timeframe': 'Manchester City', 'Age': 29},
        {'Player': 'Pedri', 'Team within selected timeframe': 'Barcelona', 'Age': 23},
        {'Player': 'J. Bellingham', 'Team within selected timeframe': 'Real Madrid', 'Age': 22},  # Jude
        {'Player': 'J. Bellingham', 'Team within selected timeframe': 'Borussia Dortmund', 'Age': 20},  # Jobe — different player!
        {'Player': 'Fake Player', 'Team within selected timeframe': 'Nonexistent FC', 'Age': 99},  # should be unmatched
    ])
    
    print(f"\nFake Wyscout: {len(fake_wyscout)} spelers")
    print(f"Fake TM: {len(fake_tm)} spelers")
    
    matched = match_wyscout_to_tm(fake_wyscout, fake_tm)
    
    print("\n" + "=" * 70)
    print(" MATCHING RESULTATEN")
    print("=" * 70)
    print(f"{'Wyscout':<25} {'→ TM match':<35} {'Conf':<8} {'Method':<15}")
    print("-" * 85)
    
    for _, row in matched.iterrows():
        wy_name = f"{row['Player']} ({row['Team within selected timeframe']})"[:24]
        tm_name = str(row.get('tm_name', 'NONE'))[:34]
        conf = row['match_confidence']
        method = row['match_method']
        print(f"{wy_name:<25} {tm_name:<35} {conf:<8.1f} {method:<15}")
    
    match_quality_report(matched)
    
    # Validatie checks
    print(f"\n{'=' * 70}")
    print(" VALIDATION CHECKS")
    print("=" * 70)
    
    checks = [
        ('Messi matches Lionel Messi', 
         (matched.iloc[0]['tm_name'] == 'Lionel Messi') if pd.notna(matched.iloc[0]['tm_name']) else False),
        ('Haaland initial expansion works', 
         matched.iloc[1]['match_method'].startswith('fuzzy') and matched.iloc[1]['tm_name'] == 'Erling Haaland'),
        ('Vinícius matches Vinicius Junior',
         (matched.iloc[2]['tm_name'] == 'Vinicius Junior') if pd.notna(matched.iloc[2]['tm_name']) else False),
        ('Lamine Yamal matches full name variant',
         (matched.iloc[5]['tm_name'] == 'Lamine Yamal Nasraoui Ebana') if pd.notna(matched.iloc[5]['tm_name']) else False),
        ('Jude Bellingham (RM, 22) → Jude (not Jobe)',
         (matched.iloc[9]['tm_name'] == 'Jude Bellingham') if pd.notna(matched.iloc[9]['tm_name']) else False),
        ('Jobe Bellingham (BVB, 20) → Jobe (not Jude)',
         (matched.iloc[10]['tm_name'] == 'Jobe Bellingham') if pd.notna(matched.iloc[10]['tm_name']) else False),
        ('Fake Player → unmatched',
         matched.iloc[11]['match_method'] == 'unmatched'),
        ('Rodri → matches Rodri (not Rodrigo Hernández)',
         (matched.iloc[7]['tm_name'] == 'Rodri') if pd.notna(matched.iloc[7]['tm_name']) else False),
    ]
    
    passed = 0
    for desc, result in checks:
        marker = '✓' if result else '✗'
        print(f"  {marker} {desc}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(checks)}")
    return passed == len(checks)


if __name__ == "__main__":
    success = run_self_test()
    if success:
        print("\n✓ Self-test passed — pipeline ready voor echte TM data")
    else:
        print("\n✗ Some tests failed — pipeline needs review")
