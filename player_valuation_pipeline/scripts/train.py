#!/usr/bin/env python3
"""
Training script — draai de volledige pipeline.

Gebruik:
    python scripts/train.py
    python scripts/train.py --config configs/custom.yaml
    python scripts/train.py --output-dir /tmp/results

Output:
    - model.pkl + multipliers.json (voor inference)
    - CSV's met ratings, predictions, multipliers (voor reporting)
    - JSON met model metrics
"""

import argparse
import sys
from pathlib import Path

# Maak pipeline package vindbaar
_PIPELINE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PIPELINE_ROOT))

from valuation.config import load_config
from valuation.pipeline import train_full_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Train het end-to-end player valuation model"
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help="Pad naar config.yaml (default: configs/default.yaml)"
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help="Override output directory (default: uit config)"
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help="Suppress progress output"
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    results = train_full_pipeline(
        config,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
    
    print("\n" + "="*60)
    print(" SAMENVATTING")
    print("="*60)
    metrics = results['model_artifacts']['metrics']
    print(f"  Training spelers:  {metrics['n_train']}")
    print(f"  Test spelers:      {metrics['n_test']}")
    print(f"  Spearman ρ (test): {metrics['spearman_test']:.3f}")
    print(f"  R² (test):         {metrics['r2_test']:.3f}")
    print(f"  Median % error:    {metrics['median_pct_error']:.1f}%")
    print(f"  Within 50% van TM: {metrics['pct_within_50']:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
