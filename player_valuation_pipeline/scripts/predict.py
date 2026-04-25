#!/usr/bin/env python3
"""
Inference script — pas getraind model toe op nieuwe Wyscout data.

Gebruik:
    python scripts/predict.py
    python scripts/predict.py --wyscout /path/to/new_season.csv --season "2026/27"
    python scripts/predict.py --output predictions_new.csv

Gebruikt het model dat eerder getraind is met scripts/train.py.
"""

import argparse
import sys
from pathlib import Path

_PIPELINE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PIPELINE_ROOT))

from valuation.config import load_config
from valuation.pipeline import predict_for_players


def main():
    parser = argparse.ArgumentParser(
        description="Voorspel marktwaardes voor spelers met getraind model"
    )
    parser.add_argument('--config', type=str, default=None,
                        help="Config YAML pad")
    parser.add_argument('--model-path', type=str, default=None,
                        help="Override model path")
    parser.add_argument('--wyscout', type=str, default=None,
                        help="Override Wyscout CSV pad")
    parser.add_argument('--season', type=str, default=None,
                        help="Season label, bijv '2026/27'")
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help="Output CSV pad")
    parser.add_argument('--top-n', type=int, default=None,
                        help="Toon top N per positie in stdout")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    predictions = predict_for_players(
        config,
        model_path=args.model_path,
        wyscout_path=args.wyscout,
        season_label=args.season,
    )
    
    predictions.to_csv(args.output, index=False)
    print(f"\n✓ {len(predictions)} voorspellingen → {args.output}")
    
    if args.top_n:
        print(f"\nTop {args.top_n} per positie (by predicted_value):")
        for tpl in predictions['_template'].unique():
            sub = predictions[predictions['_template'] == tpl]
            top = sub.nlargest(args.top_n, 'predicted_value')
            print(f"\n── {tpl} ──")
            for i, (_, row) in enumerate(top.iterrows(), 1):
                print(f"  {i:<3} {str(row['Player'])[:22]:<22} "
                      f"{str(row['Team within selected timeframe'])[:22]:<22} "
                      f"€{row['predicted_value']/1e6:.1f}M")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
