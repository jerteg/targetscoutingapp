"""
Ultra-fast matching: exact-lookup first, fuzzy only for failures.

Strategie:
  1. Index TM op surname (laatste token)
  2. Voor elke Wyscout: lookup surname directly
  3. Als 0 matches: unmatched
  4. Als 1 match: direct match
  5. Als meerdere: disambigueer op voornaam-initial / club / age
  6. Alleen falle-back naar fuzzy als surname niet direct matched
"""

import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process as rf_process
from pathlib import Path
import sys
import time
from datetime import datetime
from collections import defaultdict



from ._matching_core import (
    normalize_name,
    name_similarity,
    club_similarity,
    expand_initial,
    MatchResult,
)


def get_surname_tokens(norm_name: str) -> list:
    """
    Voor 'vinicius junior' return ['junior', 'vinicius'].
    Voor 'l messi' return ['messi'].  (initial = skip)
    Voor 'pedri' return ['pedri'].
    
    We return alle tokens die GEEN single letter zijn en GEEN weak common.
    """
    weak = {'junior', 'jr', 'senior', 'sr', 'filho', 'neto', 'de', 'da', 'dos', 'das', 'do', 'el', 'al'}
    tokens = [t for t in norm_name.split() if len(t) > 1 and t not in weak]
    return tokens


def match_ultra_fast(
    wyscout_df: pd.DataFrame,
    tm_df: pd.DataFrame,
    wyscout_name_col: str = 'Player',
    wyscout_club_col: str = 'Team within selected timeframe',
    wyscout_age_col: str = 'Age',
    tm_name_col: str = 'player_name',
    tm_club_col: str = 'current_club_name',
    tm_dob_col: str = 'date_of_birth',
    tm_id_col: str = 'player_id',
) -> pd.DataFrame:
    """
    Ultra-fast matching via surname-exact lookup.
    """
    print(f"Matching {len(wyscout_df):,} vs {len(tm_df):,}...")
    t_start = time.time()
    
    # Pre-process TM
    print("  Indexing TM by surname tokens...")
    tm_df = tm_df.copy()
    tm_df['_norm_name'] = tm_df[tm_name_col].apply(normalize_name)
    
    if tm_dob_col in tm_df.columns:
        tm_df['_dob'] = pd.to_datetime(tm_df[tm_dob_col], errors='coerce')
    
    # Index: elke TM speler op elke surname-token in zijn naam
    surname_index = defaultdict(list)
    for idx, row in tm_df.iterrows():
        tokens = get_surname_tokens(row['_norm_name'])
        for t in tokens:
            surname_index[t].append(idx)
    
    print(f"  Index heeft {len(surname_index):,} unieke surname tokens")
    print(f"  Grootste bucket: {max(len(v) for v in surname_index.values())} spelers")
    
    now = datetime.now()
    results = []
    progress_every = 500
    
    for wy_idx, wy_row in wyscout_df.iterrows():
        wy_name = wy_row[wyscout_name_col]
        wy_club = wy_row.get(wyscout_club_col, '')
        wy_age = wy_row.get(wyscout_age_col, None)
        
        if pd.isna(wy_name) or not wy_name:
            results.append({'method': 'empty_name', 'tm_idx': None, 'conf': 0, 'tm_name': None})
            continue
        
        wy_norm = normalize_name(wy_name)
        wy_surnames = get_surname_tokens(wy_norm)
        wy_tokens = wy_norm.split()
        
        if not wy_surnames:
            results.append({'method': 'no_surname', 'tm_idx': None, 'conf': 0, 'tm_name': None})
            continue
        
        # STAGE 1: Exacte normalized naam match
        exact_mask = tm_df['_norm_name'] == wy_norm
        if exact_mask.any():
            matches_idx = tm_df[exact_mask].index.tolist()
            
            if len(matches_idx) == 1:
                results.append({
                    'method': 'exact_name', 'tm_idx': matches_idx[0], 'conf': 100,
                    'tm_name': tm_df.loc[matches_idx[0], tm_name_col]
                })
                continue
            
            # Multiple exacte matches: disambigueer
            best = _pick_best(matches_idx, wy_club, wy_age, tm_df, tm_club_col, now)
            if best:
                results.append({
                    'method': 'exact_name_disambig', 'tm_idx': best['idx'], 
                    'conf': min(100, 85 + best['score']/10),
                    'tm_name': tm_df.loc[best['idx'], tm_name_col]
                })
                continue
        
        # STAGE 2: Surname-based lookup
        # Zoek alle TM spelers die tenminste één surname-token delen met Wyscout speler
        candidate_set = set()
        for sn in wy_surnames:
            candidate_set.update(surname_index.get(sn, []))
        
        if not candidate_set:
            results.append({'method': 'no_candidates', 'tm_idx': None, 'conf': 0, 'tm_name': None})
            continue
        
        # Voor elke candidate: score via onze strict name_similarity
        # Maar eerst: filter op INITIAL letter (als Wyscout initial-style is)
        wy_has_initial = len(wy_tokens) > 1 and len(wy_tokens[0]) == 1
        wy_initial = wy_tokens[0] if wy_has_initial else None
        
        filtered_candidates = []
        for tm_idx in candidate_set:
            tm_row = tm_df.loc[tm_idx]
            tm_norm = tm_row['_norm_name']
            tm_tokens = tm_norm.split()
            
            # Initial filter: als Wyscout heeft "L. Messi", dan moet TM's eerste token met L beginnen
            if wy_has_initial and tm_tokens:
                if not tm_tokens[0].startswith(wy_initial):
                    continue
            
            # Check ook andersom: als TM initial-style is
            tm_has_initial = len(tm_tokens) > 1 and len(tm_tokens[0]) == 1
            if tm_has_initial and wy_tokens:
                if not wy_tokens[0].startswith(tm_tokens[0]):
                    continue
            
            filtered_candidates.append(tm_idx)
        
        if not filtered_candidates:
            results.append({'method': 'initial_mismatch', 'tm_idx': None, 'conf': 0, 'tm_name': None})
            continue
        
        # Score elk filtered candidate
        scored = []
        for tm_idx in filtered_candidates:
            tm_row = tm_df.loc[tm_idx]
            
            name_score = name_similarity(wy_name, tm_row[tm_name_col])
            if name_score <= 50:
                continue  # strict filter
            
            club_score = club_similarity(wy_club, tm_row.get(tm_club_col, '')) if wy_club else 0
            
            age_match = 0
            if wy_age and pd.notna(tm_row.get('_dob')):
                tm_age = (now - tm_row['_dob']).days / 365.25
                age_diff = abs(float(wy_age) - tm_age)
                if age_diff < 1: age_match = 100
                elif age_diff < 2: age_match = 70
                elif age_diff < 3: age_match = 40
            
            if name_score >= 95:
                total = 0.7 * name_score + 0.2 * club_score + 0.1 * age_match
            else:
                if club_score < 50:
                    total = 0.5 * name_score + 0.3 * club_score + 0.2 * age_match
                    if age_match < 40:
                        total = min(total, 65)
                else:
                    total = 0.5 * name_score + 0.4 * club_score + 0.1 * age_match
            
            scored.append({
                'idx': tm_idx, 'name': tm_row[tm_name_col],
                'name_score': name_score, 'club_score': club_score,
                'age_match': age_match, 'total': total
            })
        
        if not scored:
            results.append({'method': 'unmatched', 'tm_idx': None, 'conf': 0, 'tm_name': None})
            continue
        
        best = max(scored, key=lambda x: x['total'])
        
        if best['total'] >= 85:
            method = 'fuzzy_high'
        elif best['total'] >= 70:
            method = 'fuzzy_medium'
        else:
            method = 'fuzzy_low'
        
        results.append({
            'method': method, 'tm_idx': best['idx'],
            'conf': best['total'], 'tm_name': best['name']
        })
        
        # Progress
        if len(results) % progress_every == 0:
            elapsed = time.time() - t_start
            pct = 100 * len(results) / len(wyscout_df)
            rate = len(results) / elapsed
            eta = (len(wyscout_df) - len(results)) / rate if rate > 0 else 0
            print(f"  {len(results):>5}/{len(wyscout_df)} ({pct:.1f}%) — {elapsed:.0f}s, ETA {eta:.0f}s")
    
    # Bouw output DataFrame
    output = wyscout_df.copy()
    output['tm_player_id'] = [
        tm_df.loc[r['tm_idx'], tm_id_col] if r['tm_idx'] is not None else None
        for r in results
    ]
    output['tm_name'] = [r['tm_name'] for r in results]
    output['match_confidence'] = [r['conf'] for r in results]
    output['match_method'] = [r['method'] for r in results]
    
    print(f"\n⏱ Total: {time.time()-t_start:.0f}s")
    return output


def _pick_best(candidate_idxs, wy_club, wy_age, tm_df, tm_club_col, now):
    """Disambigueer tussen meerdere exact name matches via club+age."""
    best_score = -1
    best_idx = None
    for tm_idx in candidate_idxs:
        tm_row = tm_df.loc[tm_idx]
        cs = club_similarity(wy_club, tm_row.get(tm_club_col, '')) if wy_club else 0
        
        age_s = 0
        if wy_age and pd.notna(tm_row.get('_dob')):
            tm_age = (now - tm_row['_dob']).days / 365.25
            diff = abs(float(wy_age) - tm_age)
            age_s = max(0, 100 - diff * 30)
        
        combined = 0.7 * cs + 0.3 * age_s
        if combined > best_score:
            best_score = combined
            best_idx = tm_idx
    
    if best_idx is not None and best_score > 30:
        return {'idx': best_idx, 'score': best_score}
    return None


if __name__ == "__main__":
    print("Ultra-fast matching module loaded")
