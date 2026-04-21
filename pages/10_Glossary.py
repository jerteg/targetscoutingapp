import os, sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import streamlit as st
from shared.styles import BASE_CSS
from shared.sidebar_nav import render_sidebar_nav

st.set_page_config(page_title="Glossary · Target Scouting", layout="wide", initial_sidebar_state="expanded")
st.markdown(BASE_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
.block-container { padding-top:1.6rem !important; max-width:1200px !important; }

/* Search bar */
.gl-search-wrap { display:flex; align-items:center; gap:10px; margin-bottom:20px; }

/* Category header */
.gl-cat-header { display:flex; align-items:center; gap:12px; margin:22px 0 10px; }
.gl-cat-badge { background:#111827; color:#c9a84c; font-family:'JetBrains Mono',monospace;
    font-size:8px; font-weight:700; padding:3px 9px; border-radius:3px;
    text-transform:uppercase; letter-spacing:0.06em; }
.gl-cat-line { flex:1; height:0.5px; background:#e0d8cc; }

/* Stat rows */
.gl-row { display:grid; grid-template-columns:220px 1fr; gap:16px;
    padding:8px 0; border-bottom:0.5px solid #e0d8cc; }
.gl-stat-name { font-size:12px; font-weight:700; color:#111827; }
.gl-stat-tag { font-family:'JetBrains Mono',monospace; font-size:8px; color:#b0a898;
    margin-top:2px; text-transform:uppercase; letter-spacing:0.05em; }
.gl-def { font-size:12px; color:#7a7060; line-height:1.6; }
.gl-formula { font-family:'JetBrains Mono',monospace; font-size:10px; color:#b0a898;
    background:#f0ebe2; padding:3px 8px; border-radius:3px;
    display:inline-block; margin-top:4px; }

/* Methodology cards */
.meth-card { background:#fff; border:0.5px solid #e0d8cc; border-radius:8px;
    margin-bottom:10px; overflow:hidden; }
.meth-header { background:#f0ebe2; padding:10px 16px; border-bottom:0.5px solid #e0d8cc;
    display:flex; align-items:center; gap:10px; }
.meth-num-badge { background:#111827; color:#c9a84c; font-family:'JetBrains Mono',monospace;
    font-size:8px; font-weight:700; padding:2px 8px; border-radius:3px; }
.meth-title { font-size:13px; font-weight:700; color:#111827; }
.meth-body { padding:14px 16px; }
.meth-step { display:flex; gap:10px; margin-bottom:10px; align-items:flex-start; }
.meth-step-num { width:22px; height:22px; border-radius:50%; background:#111827; color:#c9a84c;
    font-family:'JetBrains Mono',monospace; font-size:9px; font-weight:700;
    display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:1px; }
.meth-step-text { font-size:12px; color:#7a7060; line-height:1.6; }
.meth-step-text b { color:#111827; }

/* League table */
.league-table { width:100%; border-collapse:collapse; font-size:11px; }
.league-table tr { border-bottom:0.5px solid #e0d8cc; }
.league-table td { padding:6px 8px; }
.lt-name { color:#111827; font-weight:500; }
.lt-bar-cell { width:130px; padding:6px 8px; }
.lt-bar-track { background:#f0ebe2; height:5px; border-radius:2px; }
.lt-bar-fill { height:5px; border-radius:2px; background:#c9a84c; }
.lt-mult { font-family:'JetBrains Mono',monospace; font-weight:700; font-size:11px; text-align:right; padding:6px 8px; }

/* Position weights grid */
.pw-grid { display:grid; grid-template-columns:140px repeat(5,1fr);
    border:0.5px solid #e0d8cc; border-radius:6px; overflow:hidden; font-size:11px; }
.pw-head { font-family:'JetBrains Mono',monospace; font-size:8px; font-weight:700;
    text-transform:uppercase; letter-spacing:0.08em; color:#b0a898;
    padding:8px; background:#f0ebe2; border-bottom:0.5px solid #e0d8cc; text-align:center; }
.pw-head.l { text-align:left; }
.pw-cell { padding:7px 8px; border-bottom:0.5px solid #e0d8cc; text-align:center;
    font-family:'JetBrains Mono',monospace; font-size:11px; }
.pw-cell.l { text-align:left; color:#111827; font-family:'DM Sans',sans-serif; font-size:12px; font-weight:500; }
.pw-high { font-weight:700; color:#1a7a45; }
.pw-med  { color:#f0a500; }
.pw-low  { color:#b0a898; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    render_sidebar_nav("glossary")

st.markdown("""
<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px;">
  <h1 style="font-size:20px;font-weight:700;letter-spacing:-0.01em;">Glossary & Methodology</h1>
</div>
<div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#b0a898;margin-bottom:20px;">
  All stats from <b style="color:#111827;">Wyscout</b> · Season 2025/26 · Cut-off 26 March 2026 ·
  Per-90 stats normalised to 90 minutes of playing time
</div>""", unsafe_allow_html=True)

tab_stats, tab_meth = st.tabs(["Stat definitions", "Methodology"])

# ── GLOSSARY DATA ──────────────────────────────────────────────────────────────
GLOSSARY = {
    "Goalscoring": [
        ("Non-penalty goals per 90", "per 90",
         "Goals scored excluding penalties, per 90 minutes.",
         "Goals − Pen. goals / (mins / 90)"),
        ("xG per 90", "model",
         "Expected goals per 90. Model estimate of how many goals a player should score based on shot quality.",
         "Sum of shot xG / (mins / 90)"),
        ("xG per shot", "derived",
         "Average shot quality — how dangerous each attempt is on average.",
         "Total xG / Total shots"),
        ("Finishing", "derived",
         "Over- or under-performance vs xG. Positive = scores more than expected.",
         "NP Goals/90 − xG/90"),
        ("Shots per 90", "volume", "Total shot attempts per 90 minutes.", ""),
        ("Shots on target, %", "percentage",
         "Percentage of shots that hit the frame (on target).",
         "On-target shots / Total shots × 100"),
        ("Touches in box per 90", "volume",
         "Times a player received the ball inside the opposition penalty area, per 90 minutes.", ""),
    ],
    "Chance creation": [
        ("Assists per 90", "per 90", "Goal assists per 90 minutes.", ""),
        ("xA per 90", "model",
         "Expected assists per 90. Sum of xG value of shots created by passes.", ""),
        ("Shot assists per 90", "per 90",
         "Passes that directly led to a shot (including assists), per 90 minutes.", ""),
        ("Key passes per pass", "ratio",
         "Share of passes that were key passes (directly created a shot opportunity).",
         "Key passes / Total passes"),
        ("Through passes per pass", "ratio",
         "Share of passes that were through balls.",
         "Through passes / Total passes"),
        ("Accurate crosses per received pass", "ratio",
         "How often a player crosses when receiving a pass — captures crossing frequency relative to touches.",
         "Accurate crosses/90 / Received passes/90"),
        ("Accurate crosses, %", "percentage",
         "Percentage of crosses that reached a teammate.",
         "Accurate crosses / Total crosses × 100"),
    ],
    "Dribbling": [
        ("Successful dribbles per received pass", "ratio",
         "Successful dribbles relative to how often a player has the ball — captures dribbling intent.",
         "Successful dribbles/90 / Received passes/90"),
        ("Successful dribbles, %", "percentage",
         "Percentage of attempted dribbles that were completed.",
         "Successful dribbles / Attempted dribbles × 100"),
        ("Offensive duels won, %", "percentage",
         "Percentage of offensive duels (1v1 attacking situations) won.",
         "Offensive duels won / Total offensive duels × 100"),
        ("Progressive runs per received pass", "ratio",
         "Ball carries that advance significantly towards goal, relative to touches received.",
         "Progressive runs/90 / Received passes/90"),
    ],
    "Passing": [
        ("Completed progressive passes per 90", "per 90",
         "Passes that move the ball significantly towards the opposition goal, per 90 minutes.", ""),
        ("Accurate progressive passes, %", "percentage",
         "Completion rate on progressive passes.",
         "Completed prog. passes / Total prog. passes × 100"),
        ("Ball progression through passing", "derived",
         "Sum of deep completions, passes to penalty area, passes to final third, and progressive passes — all completed per 90.",
         "Deep compl. + PA passes + F3 passes + prog. passes (all per 90)"),
        ("Passing accuracy (prog/1/3/forw)", "derived",
         "Average of three pass accuracy metrics: progressive, final-third, and forward passes.",
         "(Prog % + F3 % + Forw %) / 3"),
        ("Completed passes to final third per 90", "per 90",
         "Passes completed into the final third of the pitch, per 90 minutes.", ""),
        ("Completed passes to penalty area per 90", "per 90",
         "Passes completed into the opposition box, per 90 minutes.", ""),
        ("Deep completions per 90", "per 90",
         "Passes completed into the zone within 20m of the goal, per 90 minutes.", ""),
    ],
    "Defending": [
        ("PAdj Defensive duels won per 90", "possession-adj",
         "Defensive duels won, adjusted for opponent possession. More possession by opponent = more actions expected.",
         "Def. duels won/90 / Opponent possession %"),
        ("Defensive duels won, %", "percentage",
         "Percentage of defensive duels (1v1 defending) won.",
         "Def. duels won / Total def. duels × 100"),
        ("PAdj Aerial duels won per 90", "possession-adj",
         "Aerial duels won, possession-adjusted.",
         "Aerial duels won/90 / Opponent possession %"),
        ("Aerial duels won, %", "percentage",
         "Percentage of aerial duels (headers) won.",
         "Aerial duels won / Total aerial duels × 100"),
        ("PAdj Interceptions", "possession-adj",
         "Interceptions per 90, possession-adjusted. Captures how often a player reads and cuts out passes.",
         "Interceptions/90 / Opponent possession %"),
        ("PAdj Successful defensive actions per 90", "possession-adj",
         "All successful defensive actions (duels won + interceptions + clearances) per 90, possession-adjusted.", ""),
        ("Fouls per 90", "volume",
         "Fouls committed per 90 minutes. Used as a negative stat — fewer fouls is better.", ""),
    ],
}

LEAGUE_MULTS = [
    ("Premier League",1.000),("La Liga",0.913),("Italian Serie A",0.891),
    ("Bundesliga",0.889),("Ligue 1",0.874),("Pro League",0.802),
    ("Primeira Liga",0.797),("Serie A BRA",0.792),("Championship",0.790),
    ("Superligaen",0.775),("Ekstraklasa",0.761),("MLS",0.758),
    ("Prva HNL",0.756),("Eliteserien",0.752),("Super Lig",0.744),
    ("Eredivisie",0.744),("Liga Pro",0.742),("Segunda Division",0.739),
    ("Swiss Super League",0.720),
]

POS_WEIGHTS = [
    ("Striker",           55,20,10,10, 5),
    ("Winger",            35,30,15,15, 5),
    ("Att. Midfielder",   25,35,10,20,10),
    ("Central Midfielder",20,25,15,25,15),
    ("Def. Midfielder",    5,10,10,40,35),
    ("Full-back",          5,25,10,30,30),
    ("Centre-Back",        5, 5, 5,35,50),
]

SCORING_STEPS = [
    ("Pool selection",       "Players compared only within the same position group and league template."),
    ("Percentile ranking",   "Each stat is ranked within the pool. Score of 80 = better than 80% of peers on that stat."),
    ("League multiplier",    "Percentile scores scaled by league difficulty. Premier League ×1.00 → Swiss SL ×0.720. Calibrated from OPTA Power Rankings."),
    ("Category score",       "Stats within each category combined using weighted averages. Negative stats (e.g. Fouls/90) are flipped: 100 − percentile."),
    ("Overall (raw)",        "Category scores combined using position-specific weights (see table below)."),
    ("Adjusted score",       "(raw/100)^0.45 × 100 — compresses towards the middle, making 80+ genuinely rare."),
    ("Bayesian shrinkage k=1200", "Players with <900 min pulled towards pool mean: w = minutes / (minutes + 1200). At 600 min: w=0.33; at 1200 min: w=0.50."),
]

# ── TAB 1: Stat definitions ────────────────────────────────────────────────────
with tab_stats:
    # Two-column layout: stats left, methodology compact right
    col_gl, col_meth = st.columns([1.1, 0.9])

    with col_gl:
        search = st.text_input("", placeholder="Search stat or keyword…", label_visibility="collapsed", key="gl_search")

        for cat, entries in GLOSSARY.items():
            filtered = [(s,t,d,f) for s,t,d,f in entries
                        if not search or search.lower() in s.lower() or search.lower() in d.lower()]
            if not filtered: continue

            st.markdown(f"""
            <div class="gl-cat-header">
              <span class="gl-cat-badge">{cat}</span>
              <div class="gl-cat-line"></div>
            </div>""", unsafe_allow_html=True)

            for stat_name, tag, definition, formula in filtered:
                fml_html = f'<div class="gl-formula">{formula}</div>' if formula else ""
                st.markdown(f"""
                <div class="gl-row">
                  <div>
                    <div class="gl-stat-name">{stat_name}</div>
                    <div class="gl-stat-tag">{tag}</div>
                  </div>
                  <div>
                    <div class="gl-def">{definition}</div>
                    {fml_html}
                  </div>
                </div>""", unsafe_allow_html=True)

    with col_meth:
        st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;font-weight:700;color:#b0a898;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;">Quick Reference</div>',
                    unsafe_allow_html=True)

        # Scoring model compact
        st.markdown("""
        <div class="meth-card">
          <div class="meth-header">
            <span class="meth-num-badge">7 steps</span>
            <span class="meth-title">Scoring model</span>
          </div>
          <div class="meth-body">""", unsafe_allow_html=True)
        steps_html = "".join(
            f'<div class="meth-step"><div class="meth-step-num">{i+1}</div>'
            f'<div class="meth-step-text"><b>{title}</b> — {body}</div></div>'
            for i,(title,body) in enumerate(SCORING_STEPS)
        )
        st.markdown(steps_html + "</div></div>", unsafe_allow_html=True)

        # League multipliers
        st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;font-weight:700;color:#b0a898;text-transform:uppercase;letter-spacing:0.1em;margin:16px 0 8px;">League multipliers</div>',
                    unsafe_allow_html=True)
        rows_html = ""
        for name, m in LEAGUE_MULTS:
            w   = int(((m-0.720)/(1.000-0.720))*100)
            col = "#1a7a45" if m>=0.87 else ("#f0a500" if m>=0.77 else "#c0392b")
            rows_html += (f'<tr><td class="lt-name">{name}</td>'
                          f'<td class="lt-bar-cell"><div class="lt-bar-track">'
                          f'<div class="lt-bar-fill" style="width:{w}%;"></div></div></td>'
                          f'<td class="lt-mult" style="color:{col};">{m:.3f}</td></tr>')
        st.markdown(f'<div style="border:0.5px solid #e0d8cc;border-radius:6px;overflow:hidden;">'
                    f'<table class="league-table">{rows_html}</table></div>',
                    unsafe_allow_html=True)

        # Position weights
        st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;font-weight:700;color:#b0a898;text-transform:uppercase;letter-spacing:0.1em;margin:16px 0 8px;">Position category weights</div>',
                    unsafe_allow_html=True)
        header = '<div class="pw-head l">Position</div><div class="pw-head">Goals</div><div class="pw-head">Chance</div><div class="pw-head">Drib.</div><div class="pw-head">Pass.</div><div class="pw-head">Def.</div>'
        cells  = ""
        for pos, *vals in POS_WEIGHTS:
            mx = max(vals)
            cells += f'<div class="pw-cell l">{pos}</div>'
            cells += "".join(
                f'<div class="pw-cell {"pw-high" if v==mx else "pw-med" if v>=25 else "pw-low"}">{v}%</div>'
                for v in vals
            )
        st.markdown(f'<div class="pw-grid">{header}{cells}</div>', unsafe_allow_html=True)

# ── TAB 2: Methodology ─────────────────────────────────────────────────────────
with tab_meth:
    METH_SECTIONS = [
        ("Scoring model — how ratings are calculated",
         "7 steps",
         "The overall rating in Player Card, Dashboard, Database and Compare is computed identically across all tools:",
         [(t,b) for t,b in SCORING_STEPS]),
        ("Radar chart — percentile basis", "3 options",
         "The radar shows 20 universal stats as percentiles. Three comparison pools:",
         [
             ("T5 only", "Compared against same-position players in the 5 major European leagues. Players from weaker leagues score higher here."),
             ("Next 14 only", "Compared against same-position players in the 14 second-tier leagues."),
             ("Own league / Both", "All leagues combined — broadest comparison."),
         ]),
        ("Ranking tool — role scoring", "2 modes",
         "Rank by a pre-defined role (fixed stat weights) or custom stat selection.",
         [
             ("Role scoring", "Each role has fixed stats and weights. League adjustment applied on top."),
             ("Custom scoring", "Select any stats and assign weights manually."),
         ]),
        ("Similar players — cosine similarity", "tier-adjusted",
         "Dashboard uses tier-adjusted cosine similarity on 20 universal radar stat percentiles.",
         [
             ("Raw similarity", "Cosine similarity of percentile vectors — 100% = identical radar profiles."),
             ("Tier factor", "candidate_league_mult / target_league_mult, clipped to [0, 1]."),
             ("Adjusted similarity", "raw_sim × tier_factor. Rewards finding quality in comparable leagues."),
             ("Tier badges", "Same tier (diff ≥ −0.05) · Tier below (≥ −0.20) · Lower tier (< −0.20)."),
         ]),
        ("Archetype clustering — K-Means", "23 archetypes",
         "Players are clustered into playing-style archetypes using K-Means on position-specific features.",
         [
             ("CB (4)", "Blocker · Aggressor · Sweeper-Distributor · Ball-Playing CB"),
             ("FB (4)", "Wide Deliverer · Inverted Modern FB · Defensive Anchor · Overlapping Attacker"),
             ("MID (7)", "Space Invader · Destroyer · Box-Crashing Finisher · Efficient Shooter · Elite Playmaker · Deep Balancer · All-Rounder"),
             ("W (4)", "Elite Creator-Finisher · Inverted Playmaker · Defensive Winger · Direct Runner"),
             ("ST (4)", "Pure Finisher · False 9 · Target Man · Pressing Forward"),
         ]),
        ("Multi-season rating — exponential decay", "2+ seasons",
         "Available in Database, Dashboard and Development. Combines scores across seasons with decay weights.",
         [
             ("2 seasons", "65% current + 35% previous"),
             ("3 seasons", "55% / 30% / 15%"),
             ("4 seasons", "50% / 25% / 15% / 10%"),
             ("Stability", "Year-on-year Pearson r = 0.634 (n=1,504). Reduces SD by 8.6% vs single-season."),
         ]),
    ]

    for title, badge, intro, steps in METH_SECTIONS:
        steps_html = "".join(
            f'<div class="meth-step"><div class="meth-step-num" style="width:auto;padding:0 8px;border-radius:4px;font-size:8px;">{i+1}</div>'
            f'<div class="meth-step-text"><b>{t}</b> — {b}</div></div>'
            for i,(t,b) in enumerate(steps)
        )
        st.markdown(f"""
        <div class="meth-card">
          <div class="meth-header">
            <span class="meth-num-badge">{badge}</span>
            <span class="meth-title">{title}</span>
          </div>
          <div class="meth-body">
            <div style="font-size:12px;color:#7a7060;margin-bottom:10px;">{intro}</div>
            {steps_html}
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:20px;padding:14px 16px;background:#f0ebe2;border-radius:8px;
                font-family:'JetBrains Mono',monospace;font-size:10px;color:#7a7060;line-height:1.7;">
      <b style="color:#c0392b;">Known caveats</b><br>
      · Current Team ≠ where stats came from (transfers mid-season).<br>
      · Cross-league consistency imperfect, especially for defensive tags.<br>
      · Neither model captures off-ball value (positioning, pressing choices).<br>
      · A score drop after moving to a stronger league is expected — not a performance decline.<br>
      · Video remains the final judge. Use scores as a shortlist filter, then watch the players.
    </div>""", unsafe_allow_html=True)
