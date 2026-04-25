"""
End-to-end pipeline orchestrator.

Public API:
  - train_full_pipeline(config): volledige training, saved model + multipliers
  - predict_for_players(config, model_path): inference mode

Training pipeline (in volgorde):
  1. Load Wyscout current + previous season
  2. Derive migration-based league multipliers (Fase 4-5)
  3. Compute category-weighted ratings + apply migration multipliers (Fase 1-3+5)
  4. Load TM profiles
  5. Match Wyscout to TM (matching module)
  6. Load TM transfer history
  7. Derive fee-based league multipliers (Fase 6)
  8. Load TM market values
  9. Train €-model (Fase 7)
  10. Save alles naar disk

Inference pipeline:
  1. Load Wyscout (new season)
  2. Load getraind model + multipliers
  3. Compute ratings + apply saved multipliers
  4. Load TM profiles (for new season), match
  5. Predict €-values using saved model
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Optional

from . import data
from . import rating
from . import league_multipliers
from . import matching
from . import fee_multipliers
from . import market_value
from ._rating_config import rating_config_v2


def train_full_pipeline(config, output_dir: Optional[str] = None,
                         verbose: bool = True) -> dict:
    """
    End-to-end training pipeline.
    
    Parameters
    ----------
    config : Config object (uit config.load_config)
    output_dir : waar output te schrijven (default: config.output.base_dir)
    verbose : print progress
    
    Returns
    -------
    dict met alle artifacts (model, multipliers, ratings, etc.)
    """
    if output_dir is None:
        output_dir = config.output.base_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = Path(config.output.model_dir)
    if not model_dir.is_absolute():
        model_dir = output_dir / model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    
    log = print if verbose else (lambda *a, **k: None)
    
    # ── STEP 1: Load Wyscout ──────────────────────────────────────────
    log("\n[1/10] Load Wyscout data...")
    wy_current = data.load_wyscout(
        config.data.wyscout_current_season,
        config.data.current_season_label
    )
    wy_previous = data.load_wyscout(
        config.data.wyscout_previous_season,
        config.data.previous_season_label
    )
    log(f"  Current ({config.data.current_season_label}): {len(wy_current)} rows")
    log(f"  Previous ({config.data.previous_season_label}): {len(wy_previous)} rows")
    
    # ── STEP 2: Migration-based league multipliers ───────────────────
    log("\n[2/10] Derive migration-based league multipliers...")
    mig_results = league_multipliers.derive_migration_multipliers(
        wy_current=wy_current,
        wy_previous=wy_previous,
        config=rating_config_v2,
        reference_league=config.league_multipliers.reference_league,
        min_minutes_per_season=config.league_multipliers.min_minutes_per_season,
        bayesian_tau=config.league_multipliers.bayesian_tau,
        k_scaling=config.league_multipliers.k_scaling,
        multiplier_min=config.league_multipliers.multiplier_min,
        multiplier_max=config.league_multipliers.multiplier_max,
    )
    migration_multipliers = mig_results['multipliers']
    log(f"  {len(migration_multipliers)} leagues, {mig_results['data_regression']['n_total']} migraties used")
    
    # Save
    mig_df = pd.DataFrame([
        {'League': lg, 'Multiplier': m, 
         'n_migrations': mig_results['bayesian']['shrinkage_weights'][lg]['n'],
         'w_data': mig_results['bayesian']['shrinkage_weights'][lg]['w_data']}
        for lg, m in migration_multipliers.items()
    ]).sort_values('Multiplier', ascending=False)
    mig_df.to_csv(output_dir / 'migration_multipliers.csv', index=False)
    
    # ── STEP 3: Compute ratings (Fase 1-3) + apply migration multipliers ──
    log("\n[3/10] Compute category-weighted ratings...")
    ratings = rating.compute_rating(
        wy_current,
        min_minutes=config.rating.min_minutes,
        z_clip=config.rating.z_clip,
    )
    log(f"  {len(ratings)} player-template combinations")
    
    log("      Apply migration multipliers...")
    ratings_adj = rating.apply_league_multipliers(ratings, migration_multipliers)
    ratings_adj.to_csv(output_dir / 'ratings_league_adjusted.csv', index=False)
    
    # ── STEP 4: Load TM profiles ──────────────────────────────────────
    log("\n[4/10] Load TM profiles...")
    tm_profiles = data.load_tm_profiles(config.data.tm_player_profiles)
    log(f"  {len(tm_profiles)} spelers")
    
    # ── STEP 5: Match Wyscout ↔ TM ────────────────────────────────────
    log("\n[5/10] Match Wyscout to TM (~2 min)...")
    matched = matching.match_ultra_fast(
        wy_current, tm_profiles,
        tm_name_col='player_name',
        tm_club_col='current_club_name',
        tm_dob_col='date_of_birth',
    )
    matched.to_csv(output_dir / 'wyscout_tm_matched.csv', index=False)
    
    n_high = (matched['match_confidence'] >= 85).sum()
    log(f"  High-confidence (≥85): {n_high} / {len(matched)} ({100*n_high/len(matched):.1f}%)")
    
    # ── STEP 6: Load transfers ────────────────────────────────────────
    log("\n[6/10] Load TM transfer history...")
    transfers = data.load_tm_transfers(
        config.data.tm_transfer_history,
        min_year=config.fee_multipliers.min_year,
    )
    log(f"  {len(transfers)} paid transfers from {config.fee_multipliers.min_year}+")
    
    # ── STEP 7: Fee-based multipliers (Fase 6) ────────────────────────
    log("\n[7/10] Derive fee-based league multipliers...")
    fee_results = fee_multipliers.derive_fee_multipliers(
        matched_df=matched,
        transfers_df=transfers,
        tm_profiles=tm_profiles,
        reference_league=config.league_multipliers.reference_league,
        ridge_alpha=config.fee_multipliers.ridge_alpha,
    )
    fee_mult = fee_results['combined_multipliers']
    log(f"  {len(fee_mult)} leagues with fee multipliers")
    log(f"  {fee_results['n_transfers_used']} cross-league transfers used")
    
    fee_df = pd.DataFrame([
        {'League': lg, 'From_Multiplier': fee_results['from_multipliers'].get(lg),
         'To_Multiplier': fee_results['to_multipliers'].get(lg),
         'Combined_Multiplier': fee_mult.get(lg),
         'N_From': fee_results['n_per_league_from'].get(lg, 0),
         'N_To': fee_results['n_per_league_to'].get(lg, 0)}
        for lg in fee_mult
    ]).sort_values('Combined_Multiplier', ascending=False)
    fee_df.to_csv(output_dir / 'fee_multipliers.csv', index=False)
    
    # ── STEP 8: Load market values ────────────────────────────────────
    log("\n[8/10] Load TM market values...")
    tm_mv = data.load_tm_market_values(
        config.data.tm_market_values,
        min_date=config.market_value.min_tm_value_date,
    )
    log(f"  {len(tm_mv)} recent values")
    
    # ── STEP 9: Train €-model (Fase 7) ────────────────────────────────
    log("\n[9/10] Train market value model...")
    model_artifacts = market_value.train_market_value_model(
        matched_df=matched,
        tm_profiles=tm_profiles,
        tm_market_values=tm_mv,
        ratings=ratings_adj,
        fee_multipliers=fee_mult,
        min_match_confidence=config.matching.min_confidence,
        test_size=config.market_value.test_size,
        random_state=config.market_value.random_state,
        ridge_alpha=config.market_value.ridge_alpha,
        min_players_per_club=config.market_value.min_players_per_club,
        age_poly_degree=config.market_value.age_poly_degree,
        min_tm_value=config.market_value.min_tm_value,
        min_tm_value_date=config.market_value.min_tm_value_date,
        verbose=verbose,
    )
    
    # ── STEP 10: Save artifacts ───────────────────────────────────────
    log("\n[10/10] Save all artifacts...")
    
    # Model (voor inference)
    model_path = model_dir / 'market_value_model.pkl'
    market_value.save_model(model_artifacts, model_path)
    log(f"  Model → {model_path}")
    
    # Multipliers JSON (voor snelle lookup)
    multipliers_all = {
        'migration_multipliers': migration_multipliers,
        'fee_multipliers': fee_mult,
        'reference_league': config.league_multipliers.reference_league,
    }
    with open(model_dir / 'multipliers.json', 'w') as f:
        json.dump(multipliers_all, f, indent=2)
    log(f"  Multipliers → {model_dir / 'multipliers.json'}")
    
    # Metrics (voor dashboard/reporting)
    with open(output_dir / 'model_metrics.json', 'w') as f:
        json.dump(model_artifacts['metrics'], f, indent=2)
    
    # Final predictions voor alle spelers
    log("  Predict for all training players...")
    all_predictions = market_value.predict_market_values(
        model_artifacts,
        ratings=ratings_adj,
        matched_df=matched,
        tm_profiles=tm_profiles,
        min_match_confidence=config.matching.min_confidence,
    )
    all_predictions.to_csv(output_dir / 'player_predictions.csv', index=False)
    log(f"  Predictions → {output_dir / 'player_predictions.csv'}")
    
    # Top 20 per position
    top_frames = []
    for tpl in all_predictions['_template'].unique():
        sub = all_predictions[all_predictions['_template'] == tpl]
        if len(sub) < 20:
            continue
        top = sub.nlargest(20, 'predicted_value').copy()
        top.insert(0, 'Rank', range(1, len(top)+1))
        top_frames.append(top)
    if top_frames:
        pd.concat(top_frames, ignore_index=True).to_csv(
            output_dir / 'top20_per_position.csv', index=False
        )
    
    log(f"\n✓ Pipeline complete. Results in {output_dir}")
    log(f"  Model: {model_path}")
    log(f"  Metrics: Spearman ρ = {model_artifacts['metrics']['spearman_test']:.3f}")
    
    return {
        'migration_multipliers': migration_multipliers,
        'fee_multipliers': fee_mult,
        'ratings': ratings_adj,
        'matched': matched,
        'model_artifacts': model_artifacts,
        'predictions': all_predictions,
    }


def predict_for_players(config,
                         model_path: Optional[str] = None,
                         wyscout_path: Optional[str] = None,
                         season_label: Optional[str] = None,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Inference: gebruik getraind model om nieuwe spelers te waarderen.
    
    Parameters
    ----------
    config : Config object
    model_path : pad naar model.pkl (default: config.output.model_dir/market_value_model.pkl)
    wyscout_path : Wyscout CSV (default: config.data.wyscout_current_season)
    season_label : bijv "2025/26"
    
    Returns
    -------
    DataFrame met predicted_value per speler-template
    """
    if model_path is None:
        model_dir = Path(config.output.model_dir)
        if not model_dir.is_absolute():
            model_dir = Path(config.output.base_dir) / model_dir
        model_path = model_dir / 'market_value_model.pkl'
    
    if wyscout_path is None:
        wyscout_path = config.data.wyscout_current_season
        season_label = config.data.current_season_label
    
    log = print if verbose else (lambda *a, **k: None)
    
    # Load artifacts
    log(f"[1/5] Load model from {model_path}")
    model_artifacts = market_value.load_model(model_path)
    
    # Load multipliers
    model_dir = Path(model_path).parent
    mult_path = model_dir / 'multipliers.json'
    with open(mult_path) as f:
        all_mults = json.load(f)
    migration_multipliers = all_mults['migration_multipliers']
    
    # Load Wyscout
    log(f"[2/5] Load Wyscout: {wyscout_path}")
    wy = data.load_wyscout(wyscout_path, season_label)
    
    # Rating + apply saved multipliers
    log("[3/5] Compute ratings with saved multipliers...")
    ratings = rating.compute_rating(
        wy, 
        min_minutes=config.rating.min_minutes,
        z_clip=config.rating.z_clip,
    )
    ratings_adj = rating.apply_league_multipliers(ratings, migration_multipliers)
    
    # Load TM profiles + match
    log("[4/5] Match Wyscout to TM...")
    tm_profiles = data.load_tm_profiles(config.data.tm_player_profiles)
    matched = matching.match_ultra_fast(
        wy, tm_profiles,
        tm_name_col='player_name',
        tm_club_col='current_club_name',
        tm_dob_col='date_of_birth',
    )
    
    # Predict
    log("[5/5] Predict values...")
    predictions = market_value.predict_market_values(
        model_artifacts,
        ratings=ratings_adj,
        matched_df=matched,
        tm_profiles=tm_profiles,
        min_match_confidence=config.matching.min_confidence,
    )
    
    return predictions
