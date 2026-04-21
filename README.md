# Target Scouting — Integrated App

## Project structure

```
project/
├── app.py                    ← Homepage (dashboard) — run this
├── pages/
│   ├── 1_Ranking.py          ← Ranking Tool
│   ├── 2_Radar.py            ← Radar Tool
│   ├── 3_Player_Card.py      ← Player Card
│   └── 4_Scatter.py          ← Scatter Plot
├── shared/                   ← UNCHANGED — copy as-is from your old project
│   ├── data.csv
│   ├── templates.py
│   └── data_processing.py
└── radar_app/                ← UNCHANGED — copy as-is from your old project
    ├── radar.py
    └── target_scouting_black.png   ← add when ready
```

## Setup

1. Copy your existing `shared/` folder (data.csv, templates.py, data_processing.py) into the project root.
2. Copy your existing `radar_app/radar.py` into `radar_app/`.
3. Optionally add `radar_app/target_scouting_black.png` for the logo in radar exports.
4. Install dependencies:

```bash
pip install streamlit pandas numpy matplotlib plotly scipy openpyxl pillow
```

5. Run the app:

```bash
streamlit run app.py
```

## Navigation

Streamlit automatically generates a top navigation bar from the `pages/` folder.
The pages appear in order: Home → Ranking → Radar → Player Card → Scatter.

## Notes

- `shared/` and `radar_app/radar.py` are completely unchanged from your original code.
- The `ranking.py` helper functions have been inlined into `pages/1_Ranking.py` to reduce file count.
- All pages share the same colour palette: dark navy (#1a2240), gold (#c9a84c), warm beige (#f5f0e8).
- Logo placeholder shows "TS" until you add target_scouting_black.png.
