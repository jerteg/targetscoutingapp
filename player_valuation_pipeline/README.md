# Player Valuation Pipeline

Een end-to-end pipeline voor het waarderen van voetballers op basis van
Wyscout performance data en Transfermarkt marktwaarde data.

## Wat doet dit

**Input:** Wyscout CSV (huidige + vorig seizoen) + Transfermarkt datasets
**Output:** Per speler: rating (skill) + voorspelde marktwaarde (€)

**Methodologie** (7 fases):
1. **Fase 1-2:** VIF-schone positie templates met composite metrics
2. **Fase 3:** Category-weighted rating (Goalscoring/Chance Creation/etc.)
3. **Fase 4-5:** Migration-based league multipliers met Bayesian shrinkage
4. **Fase 6:** Fee-based league multipliers uit werkelijke transfer fees
5. **Fase 7:** Marktwaarde model (rating + age + club + contract + league → €)

**Performance:** Spearman ρ = 0.87 tegen Transfermarkt (academic grade).

## Structuur

```
player_valuation_pipeline/
├── configs/
│   └── default.yaml              # alle paths + hyperparameters
├── valuation/                    # library modules
│   ├── __init__.py
│   ├── config.py                 # config loader
│   ├── data.py                   # data loading
│   ├── rating.py                 # Fase 1-3
│   ├── league_multipliers.py     # Fase 4-5
│   ├── matching.py               # Wyscout ↔ TM matching
│   ├── fee_multipliers.py        # Fase 6
│   ├── market_value.py           # Fase 7
│   └── pipeline.py               # orchestrator
├── scripts/
│   ├── train.py                  # full training
│   └── predict.py                # inference
└── README.md
```

## Gebruik

### Setup

```bash
pip install pandas numpy scikit-learn scipy rapidfuzz unidecode pyyaml
```

### Config aanpassen

Bewerk `configs/default.yaml` met je paden:

```yaml
data:
  wyscout_current_season: "/path/to/current_season.csv"
  wyscout_previous_season: "/path/to/previous_season.csv"
  tm_player_profiles: "/path/to/tm_profiles.csv"
  tm_transfer_history: "/path/to/tm_transfers.csv"
  tm_market_values: "/path/to/tm_market_values.csv"

output:
  base_dir: "/path/to/output"
  model_dir: "./models"
```

TM data is te downloaden van [salimt/football-datasets](https://github.com/salimt/football-datasets).

### Training

```bash
cd player_valuation_pipeline
python scripts/train.py
```

Output in `{base_dir}/`:
- `models/market_value_model.pkl` — getraind model (voor inference)
- `models/multipliers.json` — league multipliers
- `ratings_league_adjusted.csv` — ratings per speler
- `player_predictions.csv` — voorspelde €-waardes
- `top20_per_position.csv` — top 20 per positie
- `migration_multipliers.csv` — multipliers met metadata
- `fee_multipliers.csv` — fee-based multipliers
- `model_metrics.json` — performance metrics

### Inference

```bash
# Gebruik default config
python scripts/predict.py --output predictions.csv

# Top 5 per positie tonen
python scripts/predict.py --output predictions.csv --top-n 5

# Andere Wyscout CSV gebruiken (ander seizoen)
python scripts/predict.py --wyscout /path/new_season.csv --season "2026/27"
```

### Python API

```python
from valuation.config import load_config
from valuation.pipeline import train_full_pipeline, predict_for_players

# Training
config = load_config("configs/default.yaml")
results = train_full_pipeline(config)

# Inference
predictions = predict_for_players(config)

# Of module-by-module voor custom workflows:
from valuation import data, rating, league_multipliers, matching, fee_multipliers, market_value
```

## Belangrijkste design beslissingen

1. **Rating vs Market Value gescheiden.** Rating meet puur skill (Fase 3+5).
 Market value combineert skill met leeftijd, club, contract en liga-markt (Fase 7).
 Scouts willen beide apart weten.

2. **Migration-based multipliers voor rating.** Fee-multipliers zijn te agressief
 (een Bundesliga speler wordt gestraft voor markt-artefacten), dus we gebruiken
 alleen prestatieverschillen bij league-overstappen.

3. **Fee-based multipliers voor €-model.** Fee-ratio's meten markt-realiteit, 
 perfect voor €-waardebepaling.

4. **Bayesian shrinkage.** Voor liga's met weinig data (MLS, Eliteserien) 
 vallen we gedeeltelijk terug op expert prior. Tau=15 betekent: 15 migraties 
 wegen evenveel als de prior.

5. **Log-lineaire €-regressie.** Geen machine learning — transparant, 
 interpreteerbaar. Elke coefficient is direct begrijpelijk.

## Performance benchmarks

- **Spearman ρ = 0.87** op hold-out test set (4613 spelers)
- **R² = 0.78** op log-waarde
- **60% predictions binnen 50% van TM waarde**
- **86% predictions binnen 100% van TM waarde**

Vergelijkbaar met academische benchmarks (CIES, SciSports) maar met minder 
input features (alleen aggregaat Wyscout, geen event data).

## Bekende limitaties

- **Top-tier underestimate:** Haaland/Mbappé krijgen €100-125M vs TM €180M.
 Superstar premiums volgen niet-lineair, model pakt ~60-70%.
- **Liga Pro, Eliteserien, MLS:** data dun, multipliers prior-gedomineerd.
- **Matching** faalt voor ~24% spelers in Zuid-Amerikaanse liga's (data gap in TM).
- **Leeftijd peak R²:** per-template curves hebben R² 0.05-0.11 (age alleen 
 verklaart maar 5-11% van waarde; skill en club doen de rest).
- **Geen transfer market timing.** Winter vs zomer, aflopende contracten 
 < 6 maanden (Bosman) — niet apart gemodelleerd.
