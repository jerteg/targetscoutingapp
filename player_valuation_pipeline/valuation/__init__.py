"""
Player Valuation Pipeline
=========================

Een end-to-end pipeline voor het waarderen van voetballers op basis van
Wyscout performance data en Transfermarkt marktwaarde data.

Gebruik (training):
    from valuation.config import load_config
    from valuation.pipeline import train_full_pipeline
    
    config = load_config()
    results = train_full_pipeline(config)

Gebruik (inference):
    from valuation.pipeline import predict_for_players
    predictions = predict_for_players(config)

Individuele modules:
    from valuation import data, rating, league_multipliers, matching
    from valuation import fee_multipliers, market_value
"""

from .config import load_config, Config
from .pipeline import train_full_pipeline, predict_for_players

__all__ = [
    'load_config',
    'Config',
    'train_full_pipeline',
    'predict_for_players',
]

__version__ = '1.0.0'
