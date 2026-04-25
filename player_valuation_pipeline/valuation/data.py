"""
Data loading and preprocessing module.

Public API:
  - load_wyscout(path, season_label): laad één seizoen Wyscout data + preprocess
  - load_tm_profiles(path): laad TM player profiles
  - load_tm_transfers(path, min_year): laad betaalde transfers
  - load_tm_market_values(path, min_date): laad meest recente market values
"""

import pandas as pd

from ._data_processing import preprocess_data
from ._templates import position_map


def load_wyscout(path: str, season_label: str, min_minutes: int = 0) -> pd.DataFrame:
    """
    Laad een Wyscout CSV vanaf willekeurig pad + pre-process.
    
    Parameters
    ----------
    path : pad naar CSV
    season_label : "2025/26", "2024/25"
    min_minutes : filter
    
    Returns
    -------
    Preprocessed DataFrame klaar voor rating computation
    """
    df = pd.read_csv(path)
    
    # League-label correcties (alleen voor 2025/26 — Wyscout export artefact)
    if season_label == "2025/26":
        _corrections = {
            ("K. Wagner", "Birmingham City", "MLS"): "Championship",
            ("A. Malanda", "Middlesbrough", "MLS"): "Championship",
            ("G. Campbell", "West Bromwich Albion", "MLS"): "Championship",
            ("P. Agyemang", "Derby County", "MLS"): "Championship",
            ("C. Awaziem", "Nantes", "MLS"): "Ligue 1",
            ("B. Utvik", "Sarpsborg 08", "MLS"): "Eliteserien",
            ("A. Moreno", "River Plate", "Serie A BRA"): "Liga Profesional",
            ("L. Acosta", "Fluminense", "MLS"): "Serie A BRA",
            ("Janderson", "Göztepe", "Serie A BRA"): "Super Lig",
            ("Marcos Felipe", "Eyüpspor", "Serie A BRA"): "Super Lig",
            ("Wesley", "Roma", "Serie A BRA"): "Italian Serie A",
        }
        for (player, team, wrong_lg), correct_lg in _corrections.items():
            mask = (
                (df["Player"] == player) &
                (df["Team within selected timeframe"] == team) &
                (df["League"] == wrong_lg)
            )
            if mask.any():
                df.loc[mask, "League"] = correct_lg
    
    if min_minutes > 0 and "Minutes played" in df.columns:
        df["Minutes played"] = pd.to_numeric(
            df["Minutes played"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        df = df[df["Minutes played"] >= min_minutes].copy()
    
    df = preprocess_data(df)
    
    df["Main Position"] = df["Position"].astype(str).str.split(",").str[0].str.strip()
    df["Position Label"] = df["Main Position"].map(position_map).fillna(df["Main Position"])
    df["_season"] = season_label
    
    return df


def load_tm_profiles(path: str) -> pd.DataFrame:
    """Laad Transfermarkt player profiles."""
    df = pd.read_csv(path, usecols=['player_id', 'player_name', 'name_in_home_country',
                                      'date_of_birth', 'current_club_name', 'main_position',
                                      'contract_expires'])
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
    df['contract_expires'] = pd.to_datetime(df['contract_expires'], errors='coerce')
    return df


def load_tm_transfers(path: str, min_year: int = 2015) -> pd.DataFrame:
    """Laad betaalde TM transfers."""
    df = pd.read_csv(path)
    df = df[(df['transfer_fee'] > 0) & (df['transfer_type'] == 'Transfer')].copy()
    df['transfer_date'] = pd.to_datetime(df['transfer_date'], errors='coerce')
    df['year'] = df['transfer_date'].dt.year
    df = df[df['year'] >= min_year].copy()
    return df


def load_tm_market_values(path: str, min_date: str = '2024-01-01') -> pd.DataFrame:
    """Laad meest recente TM market value per speler."""
    mv = pd.read_csv(path)
    mv['date_unix'] = pd.to_datetime(mv['date_unix'], errors='coerce')
    
    latest = mv.sort_values('date_unix').groupby('player_id').tail(1)
    latest = latest.rename(columns={'value': 'tm_value', 'date_unix': 'tm_value_date'})
    latest = latest[latest['tm_value_date'] >= pd.Timestamp(min_date)]
    
    return latest[['player_id', 'tm_value', 'tm_value_date']].reset_index(drop=True)
