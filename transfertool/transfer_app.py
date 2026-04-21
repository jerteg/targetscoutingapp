"""
transfer_app.py — Transfer Valuation Tool
Plaatsen in: transfertool/transfer_app.py
Uitvoeren:   streamlit run transfer_app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Paden ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
WYSCOUT_CSV = BASE_DIR.parent / "shared" / "data.csv"
TRANSFERS   = DATA_DIR / "transfers.csv.gz"
CLUBS_TM    = DATA_DIR / "clubs.csv.gz"
COMPS_TM    = DATA_DIR / "competitions.csv.gz"

sys.path.insert(0, str(BASE_DIR.parent))

# ── Kleuren ────────────────────────────────────────────────────────────────────
NAVY  = "#111827"
GOLD  = "#c9a84c"
CREAM = "#faf7f2"
SAND  = "#f0ebe2"
MUTED = "#6b6356"
GREEN = "#22c55e"
AMBER = "#f59e0b"

# ── Model constanten ───────────────────────────────────────────────────────────
GLOBALE_BASIS = 2_000_000
B_KWALITEIT   = 3.323

LEEFTIJD_FACTOR = {
    (0, 19): 1.20, (20, 21): 1.10, (22, 25): 1.00,
    (26, 27): 0.90, (28, 29): 0.75, (30, 31): 0.65,
    (32, 33): 0.50, (34, 99): 0.30,
}

COMP_MULT = {
    "Premier League": 3.714, "La Liga": 1.724, "Italian Serie A": 1.806,
    "Bundesliga": 1.286, "Ligue 1": 1.667, "Eredivisie": 0.600,
    "Pro League": 0.857, "Serie A BRA": 1.429, "Liga Profesional": 1.140,
    "Primeira Liga": 0.812, "Ekstraklasa": 0.521, "MLS": 0.625,
    "Super Lig": 0.571, "Prva HNL": 0.550, "Eliteserien": 0.429,
    "Superligaen": 0.338, "Swiss Super League": 0.833,
    "Championship": 2.230, "Segunda Division": 1.030, "Liga Pro": 0.450,
}

COMP_LABEL = {k: f"{k}  ({v:.3f}×)" for k, v in sorted(COMP_MULT.items())}

POS_MATRIX = {
    ("GK","<18"):0.30,  ("GK","19-21"):0.25, ("GK","22-24"):0.60,
    ("GK","25-27"):0.75,("GK","28-30"):0.60, ("GK","31+"):0.625,
    ("CB","<18"):1.80,  ("CB","19-21"):1.75, ("CB","22-24"):1.25,
    ("CB","25-27"):1.00,("CB","28-30"):0.60, ("CB","31+"):0.75,
    ("LB","<18"):1.50,  ("LB","19-21"):1.50, ("LB","22-24"):1.25,
    ("LB","25-27"):0.75,("LB","28-30"):0.75, ("LB","31+"):0.25,
    ("RB","<18"):1.50,  ("RB","19-21"):1.50, ("RB","22-24"):0.90,
    ("RB","25-27"):1.25,("RB","28-30"):0.50, ("RB","31+"):0.825,
    ("DM","<18"):1.50,  ("DM","19-21"):1.50, ("DM","22-24"):1.25,
    ("DM","25-27"):1.07,("DM","28-30"):0.80, ("DM","31+"):0.975,
    ("CM","<18"):0.665, ("CM","19-21"):1.30, ("CM","22-24"):1.00,
    ("CM","25-27"):1.00,("CM","28-30"):1.00, ("CM","31+"):0.95,
    ("AM","<18"):0.665, ("AM","19-21"):1.25, ("AM","22-24"):1.15,
    ("AM","25-27"):1.10,("AM","28-30"):0.90, ("AM","31+"):1.20,
    ("LW","<18"):1.50,  ("LW","19-21"):1.50, ("LW","22-24"):1.56,
    ("LW","25-27"):1.18,("LW","28-30"):1.25, ("LW","31+"):0.60,
    ("RW","<18"):1.50,  ("RW","19-21"):2.50, ("RW","22-24"):1.00,
    ("RW","25-27"):1.25,("RW","28-30"):1.25, ("RW","31+"):0.90,
    ("CF","<18"):1.75,  ("CF","19-21"):1.50, ("CF","22-24"):1.50,
    ("CF","25-27"):1.175,("CF","28-30"):1.15,("CF","31+"):0.80,
}

POS_GROUP_MAP = {
    "GK":"Goalkeeper","CB":"Centre-Back","LCB":"Centre-Back","RCB":"Centre-Back",
    "LB":"Full-Back","RB":"Full-Back","LWB":"Full-Back","RWB":"Full-Back",
    "DMF":"Defensive Midfielder","LDMF":"Defensive Midfielder","RDMF":"Defensive Midfielder",
    "CMF":"Central Midfielder","LCMF":"Central Midfielder","RCMF":"Central Midfielder",
    "AMF":"Attacking Midfielder","LW":"Winger","RW":"Winger",
    "LWF":"Winger","RWF":"Winger","LAMF":"Winger","RAMF":"Winger",
    "CF":"Striker","SS":"Striker",
}

POS_KORT = {
    "Centre-Back":"CB","Full-Back":"LB","Goalkeeper":"GK",
    "Defensive Midfielder":"DM","Central Midfielder":"CM",
    "Attacking Midfielder":"AM","Winger":"LW","Striker":"CF",
}

UEFA_SCORE = {
    "Real Madrid":98,"Man City":99,"Bayern Munich":97,"Liverpool":96,
    "PSG":88,"Chelsea":88,"Dortmund":87,"Inter":87,"Leverkusen":86,
    "Arsenal":85,"Atlético":84,"Atletico Madrid":84,"Barcelona":95,
    "Juventus":82,"Napoli":80,"Tottenham":79,"Benfica":77,"Porto":75,
    "Sevilla FC":72,"AC Milan":71,"Leipzig":70,"Ajax":68,"Sporting CP":66,
    "Sporting":66,"Frankfurt":63,"Villarreal":62,"PSV":60,"Feyenoord":57,
    "Palmeiras":52,"Celtic":51,"Flamengo":51,"Monaco":47,"Marseille":46,
    "Galatasaray":45,"Atalanta":44,"Lille":43,"Club Brugge":43,
    "Fenerbahçe":40,"Fluminense":40,"Mönchengladbach":37,"AZ Alkmaar":33,
    "Shakhtar D.":31,"Young Boys":31,"Red Star":29,"Genk":26,
    "Dinamo Zagreb":24,"Midtjylland":22,"Bodø/Glimt":22,"Nordsjaelland":18,
    "Rosenborg BK":15,"Molde FK":10,"Man Utd":93,"Manchester United":93,
    "Aston Villa":47,"Newcastle United":43,"Brighton":37,"West Ham United":40,
    "Wolves":27,"Nottingham Forest":30,"Stade Rennais":22,"Al-Hilal":14,
    "Udinese":12,"Lecce":10,"Bayer Leverkusen":86,"RB Leipzig":70,
    "Borussia Dortmund":87,"Atletico":84,"Inter Miami":6,
}

ACADEMIE_SCORE = {
    "Barcelona":100,"Ajax":98,"Benfica":92,"Porto":90,"Bayern Munich":88,
    "Real Madrid":86,"Sporting CP":80,"Sporting":80,"PSV":75,"Dortmund":74,
    "Borussia Dortmund":74,"Lyon":72,"Liverpool":72,"Chelsea":70,
    "Man City":70,"Arsenal":70,"Juventus":65,"AC Milan":65,"Inter":63,
    "PSG":62,"Atlético":62,"Atletico Madrid":62,"Atletico":62,
    "Leipzig":62,"RB Leipzig":62,"Atalanta":60,"Celtic":55,"Leverkusen":60,
    "Bayer Leverkusen":60,"Man Utd":60,"Manchester United":60,"Tottenham":55,
    "Marseille":54,"Napoli":54,"Sevilla FC":54,"Galatasaray":45,
    "Fenerbahçe":44,"Palmeiras":52,"Flamengo":42,"River Plate":45,
    "Nordsjaelland":55,"Midtjylland":45,"Genk":45,"Club Brugge":45,
    "Shakhtar D.":42,"Red Star":35,"Dinamo Zagreb":32,"Molde FK":30,
    "Bodø/Glimt":30,"Young Boys":30,"AZ Alkmaar":45,"Mönchengladbach":42,
    "Brighton":42,"Fluminense":32,"Monaco":52,"Lille":42,"Frankfurt":42,
    "Villarreal":42,"Newcastle United":32,"Aston Villa":30,
    "West Ham United":30,"Wolves":22,"Nottingham Forest":20,"Feyenoord":65,
    "Inter Miami":8,"Al-Hilal":10,"Udinese":15,"Lecce":12,"Nice":22,
    "Stade Rennais":32,
}

# ── Helper functies ─────────────────────────────────────────────────────────────
def leeft_cat(l):
    if l < 18: return "<18"
    if l < 22: return "19-21"
    if l < 25: return "22-24"
    if l < 28: return "25-27"
    if l < 31: return "28-30"
    return "31+"

def leeft_factor(score, leeftijd):
    base = next((f for (lo, hi), f in LEEFTIJD_FACTOR.items()
                 if lo <= leeftijd <= hi), 0.30)
    if base >= 1.0:
        return base
    premium = max(0, (score - 50) / 50)
    return base ** (1.0 - premium * 0.7)

def kw_mult(score, leeftijd):
    return (max(1, min(99, score)) / 50) ** B_KWALITEIT * leeft_factor(score, leeftijd)

# ── Data laden ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Transferdata laden…")
def laad_koopkracht():
    try:
        t = pd.read_csv(TRANSFERS, compression="gzip")
        c = pd.read_csv(CLUBS_TM,  compression="gzip")
        co= pd.read_csv(COMPS_TM,  compression="gzip")
        t["transfer_date"] = pd.to_datetime(t["transfer_date"], errors="coerce")
        t["year"] = t["transfer_date"].dt.year
        t = t[(t["transfer_fee"] > 0) & t["year"].between(2020, 2025)]
        c = c[["club_id","domestic_competition_id"]].rename(
            columns={"club_id":"to_club_id","domestic_competition_id":"comp_id"})
        co = co[["competition_id","sub_type"]].rename(columns={"competition_id":"comp_id"})
        t = t.merge(c, on="to_club_id", how="left").merge(co, on="comp_id", how="left")
        kk = t[t["sub_type"]=="first_tier"].groupby("to_club_name")["transfer_fee"].sum().to_dict()
        mediaan = float(np.median([v for v in kk.values() if v > 1e7]))
        return kk, mediaan
    except Exception:
        return {}, 50_000_000.0

@st.cache_data(show_spinner="Scores berekenen…")
def laad_scores():
    try:
        from shared.data_processing import preprocess_data
        from shared.scoring import compute_scores

        df = pd.read_csv(WYSCOUT_CSV)
        df = preprocess_data(df)
        df["Main Position"] = df["Position"].astype(str).str.split(",").str[0].str.strip()
        df["pos_group"] = df["Main Position"].map(POS_GROUP_MAP).fillna("Winger")

        all_scores = []
        for pg in ["Centre-Back","Defensive Midfielder","Central Midfielder",
                   "Attacking Midfielder","Winger","Striker"]:
            try:
                s = compute_scores(df, pg)
                s["pg_used"] = pg
                all_scores.append(
                    s[["Player","Team within selected timeframe","overall_adj","pg_used"]])
            except Exception:
                pass

        if not all_scores:
            return pd.DataFrame()

        combined = pd.concat(all_scores, ignore_index=True)
        df = df.merge(combined.rename(columns={"pg_used":"pos_group"}),
                      on=["Player","Team within selected timeframe","pos_group"], how="left")
        return df
    except Exception:
        return pd.DataFrame()

def club_mult(club, kk, mediaan_kk):
    u = (UEFA_SCORE.get(club, 15) / 50) ** 1.2
    a = (ACADEMIE_SCORE.get(club, 20) / 50) ** 0.9
    k = (kk.get(club, mediaan_kk * 0.05) / mediaan_kk) ** 0.35
    return round(0.40 * u + 0.35 * a + 0.25 * k, 3)

# ── App layout ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Transfer Valuation",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&display=swap');
html, body, [class*="css"] {{ font-family:'DM Sans',sans-serif !important; }}
[data-testid="stAppViewContainer"] {{ background:{CREAM} !important; }}
[data-testid="stHeader"] {{ background:{CREAM} !important; }}
section[data-testid="stSidebar"] {{ background:{NAVY} !important; }}
section[data-testid="stSidebar"] * {{ color:rgba(255,255,255,0.85) !important; }}
section[data-testid="stSidebar"] label {{
    color:{GOLD} !important; font-size:11px !important;
    text-transform:uppercase; letter-spacing:0.07em; font-weight:500 !important;
}}
.block-container {{ padding-top:1.5rem !important; max-width:1200px !important; }}
div[data-testid="stButton"]>button {{
    background:{NAVY} !important; color:{GOLD} !important;
    border:none !important; border-radius:6px !important; font-weight:500 !important;
}}
h4 {{ color:{NAVY} !important; font-size:14px !important;
      text-transform:uppercase; letter-spacing:0.06em; font-weight:600 !important; }}
</style>
""", unsafe_allow_html=True)

# ── Data ────────────────────────────────────────────────────────────────────────
kk, mediaan_kk = laad_koopkracht()
ws_df = laad_scores()
heeft_scores = not ws_df.empty and "overall_adj" in ws_df.columns

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:12px 0 20px;">
      <div style="font-size:16px;font-weight:600;color:{GOLD};">Transfer Valuation</div>
      <div style="font-size:11px;color:rgba(255,255,255,0.35);margin-top:3px;">Target Scouting · v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    if heeft_scores:
        spelers = ["— Handmatig —"] + sorted(ws_df["Player"].dropna().unique())
        keuze   = st.selectbox("Speler", spelers)
        gebruik_auto = keuze != "— Handmatig —"
    else:
        gebruik_auto = False
        keuze = "— Handmatig —"

    # Auto-fill vanuit Wyscout
    if gebruik_auto:
        r_all = ws_df[ws_df["Player"] == keuze]
        teams  = r_all["Team within selected timeframe"].unique()
        if len(teams) > 1:
            team  = st.selectbox("Team", teams)
            r_all = r_all[r_all["Team within selected timeframe"] == team]
        r = r_all.nlargest(1, "overall_adj").iloc[0]

        leeftijd_def   = int(float(r.get("Age", 24)))
        comp_def       = str(r.get("League", list(COMP_MULT.keys())[0]))
        club_def       = str(r.get("Team within selected timeframe", ""))
        pos_group_def  = str(r.get("pos_group", "Winger"))
        score_def      = float(r["overall_adj"]) if pd.notna(r.get("overall_adj")) else 65.0
        pos_kort_def   = POS_KORT.get(pos_group_def, "LW")
        speler_naam    = keuze
    else:
        leeftijd_def, comp_def, club_def = 24, "Eredivisie", ""
        score_def, pos_kort_def = 65.0, "LW"
        speler_naam = "Speler"

    st.markdown("---")

    pos_opties = ["GK","CB","LB","RB","DM","CM","AM","LW","RW","CF"]
    pos_idx    = pos_opties.index(pos_kort_def) if pos_kort_def in pos_opties else 7
    positie    = st.selectbox("Positie", pos_opties, index=pos_idx)

    leeftijd = st.slider("Leeftijd", 15, 42, leeftijd_def)

    comp_opties = sorted(COMP_MULT.keys())
    comp_idx    = comp_opties.index(comp_def) if comp_def in comp_opties else 0
    competitie  = st.selectbox("Competitie", comp_opties, index=comp_idx)

    club = st.text_input("Club (exact zoals Transfermarkt)", value=club_def)

    # Score slider — automatisch gevuld, aanpasbaar
    score_int = int(round(score_def))
    if gebruik_auto and pd.notna(r.get("overall_adj")):
        st.markdown(f"""
        <div style="margin-top:4px;margin-bottom:-8px;font-size:10px;
                    color:{GOLD};text-transform:uppercase;letter-spacing:0.07em;
                    font-weight:500;">
          Score  <span style="color:rgba(255,255,255,0.45);font-weight:400;">
          (automatisch: {score_def:.1f})</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="margin-top:4px;margin-bottom:-8px;font-size:10px;
                    color:{GOLD};text-transform:uppercase;letter-spacing:0.07em;
                    font-weight:500;">Score  <span style="color:rgba(255,255,255,0.45);
                    font-weight:400;">(handmatig)</span></div>
        """, unsafe_allow_html=True)

    score = st.slider("Score", 1, 99, score_int, label_visibility="collapsed")

# ── Berekeningen ────────────────────────────────────────────────────────────────
lc     = leeft_cat(leeftijd)
pl     = POS_MATRIX.get((positie, lc), 1.0)
cm     = COMP_MULT.get(competitie, 1.0)
clm    = club_mult(club, kk, mediaan_kk)
lf     = leeft_factor(score, leeftijd)
kw     = kw_mult(score, leeftijd)
waarde = GLOBALE_BASIS * pl * cm * clm * kw

# Heeft club herkenning?
club_bekend = club in UEFA_SCORE or club in kk

waarde_kleur = GREEN if waarde >= 50e6 else (GOLD if waarde >= 10e6 else AMBER)

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{NAVY};border-radius:12px;padding:22px 28px;
            display:flex;justify-content:space-between;align-items:center;
            margin-bottom:24px;">
  <div>
    <div style="font-size:22px;font-weight:600;color:white;">{speler_naam}</div>
    <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap;">
      <span style="background:rgba(255,255,255,0.10);color:rgba(255,255,255,0.75);
                   padding:4px 12px;border-radius:20px;font-size:12px;">{positie}</span>
      <span style="background:rgba(255,255,255,0.10);color:rgba(255,255,255,0.75);
                   padding:4px 12px;border-radius:20px;font-size:12px;">{leeftijd} jaar</span>
      <span style="background:rgba(255,255,255,0.10);color:rgba(255,255,255,0.75);
                   padding:4px 12px;border-radius:20px;font-size:12px;">{club or '—'}</span>
      <span style="background:rgba(255,255,255,0.10);color:rgba(255,255,255,0.75);
                   padding:4px 12px;border-radius:20px;font-size:12px;">{competitie}</span>
      <span style="background:rgba(201,168,76,0.20);color:{GOLD};
                   padding:4px 12px;border-radius:20px;font-size:12px;font-weight:500;">
                   Score {score}</span>
    </div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:38px;font-weight:700;color:{waarde_kleur};line-height:1.1;">
      €{waarde/1e6:.1f}M
    </div>
    <div style="font-size:11px;color:rgba(255,255,255,0.35);margin-top:4px;">
      geschatte transferwaarde
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Twee kolommen ───────────────────────────────────────────────────────────────
col_l, col_r = st.columns([1, 1.5], gap="large")

with col_l:
    st.markdown("#### Berekening")

    def rij(label, waarde_txt, toelichting, is_basis=False):
        bg = SAND if is_basis else "white"
        st.markdown(f"""
        <div style="background:{bg};padding:11px 16px;border-radius:7px;
                    margin-bottom:4px;display:flex;justify-content:space-between;
                    align-items:center;">
          <div>
            <div style="font-size:13px;color:{NAVY};font-weight:500;">{label}</div>
            <div style="font-size:10px;color:{MUTED};margin-top:2px;">{toelichting}</div>
          </div>
          <div style="font-size:14px;font-weight:600;color:{NAVY};
                      white-space:nowrap;margin-left:16px;">{waarde_txt}</div>
        </div>
        """, unsafe_allow_html=True)

    rij("Globale basiswaarde",
        f"€{GLOBALE_BASIS/1e6:.2f}M",
        "Gewogen mediaan alle first-tier transfers 2018–2025",
        is_basis=True)
    rij(f"× Positie / Leeftijd",
        f"{pl:.3f}×",
        f"{positie} · leeftijdscategorie {lc}")
    rij(f"× Competitie",
        f"{cm:.3f}×",
        f"{competitie}")
    rij(f"× Club",
        f"{clm:.3f}×",
        f"{'⚠ onbekend — schatting' if not club_bekend else 'UEFA ' + str(UEFA_SCORE.get(club,'?')) + ' · Academie ' + str(ACADEMIE_SCORE.get(club,'?'))}")
    rij(f"× Kwaliteitsscore  ({score}/99)",
        f"{(score/50)**B_KWALITEIT:.3f}×",
        f"(score / 50) ^ {B_KWALITEIT}")
    rij(f"× Leeftijdsfactor  ({leeftijd} jr)",
        f"{lf:.3f}×",
        "Score-afhankelijk — hogere score = minder leeftijdsafstraf")

    st.markdown(f"""
    <div style="background:{NAVY};border-radius:7px;padding:14px 16px;
                display:flex;justify-content:space-between;align-items:center;
                margin-top:4px;">
      <div style="color:{GOLD};font-size:13px;font-weight:600;">
        Transferwaarde
      </div>
      <div style="color:white;font-size:24px;font-weight:700;">
        €{waarde/1e6:.1f}M
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not club_bekend and club:
        st.warning(f"**{club}** niet herkend — standaard multiplier ({clm:.2f}×) toegepast. "
                   "Controleer de exacte naam zoals op Transfermarkt.")

with col_r:
    # Grafiek 1: waarde per score
    st.markdown("#### Waarde per score")
    scores_range = list(range(1, 100))
    waarden_score = [
        GLOBALE_BASIS * pl * cm * clm * kw_mult(s, leeftijd) / 1e6
        for s in scores_range
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scores_range, y=waarden_score,
        fill="tozeroy", fillcolor="rgba(201,168,76,0.10)",
        line=dict(color=GOLD, width=2.5),
        hovertemplate="<b>Score %{x}</b><br>€%{y:.1f}M<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[score], y=[waarde / 1e6],
        mode="markers+text",
        marker=dict(color=NAVY, size=12, symbol="circle",
                    line=dict(color=GOLD, width=2.5)),
        text=[f"€{waarde/1e6:.1f}M"],
        textposition="top center",
        textfont=dict(size=12, color=NAVY, family="DM Sans"),
        hovertemplate=f"<b>Score {score}</b><br>€{waarde/1e6:.1f}M<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor=CREAM, paper_bgcolor=CREAM,
        margin=dict(l=10, r=10, t=10, b=40), height=270,
        xaxis=dict(title=dict(text="Score", font=dict(size=12, color=MUTED)),
                   gridcolor="#e0d8cc", showline=False,
                   tickfont=dict(size=11, family="DM Sans", color=MUTED)),
        yaxis=dict(title=dict(text="Transferwaarde (€M)", font=dict(size=12, color=MUTED)),
                   gridcolor="#e0d8cc", showline=False,
                   tickfont=dict(size=11, family="DM Sans", color=MUTED),
                   tickprefix="€", ticksuffix="M"),
        showlegend=False, font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Grafiek 2: waarde per leeftijd
    st.markdown("#### Waarde per leeftijd  (huidige score)")
    leeftijden = list(range(16, 40))
    waarden_lft = [
        GLOBALE_BASIS * pl * cm * clm * kw_mult(score, l) / 1e6
        for l in leeftijden
    ]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=leeftijden, y=waarden_lft,
        fill="tozeroy", fillcolor="rgba(17,24,39,0.06)",
        line=dict(color=NAVY, width=2),
        hovertemplate="<b>Leeftijd %{x}</b><br>€%{y:.1f}M<extra></extra>",
    ))
    fig2.add_trace(go.Scatter(
        x=[leeftijd], y=[waarde / 1e6],
        mode="markers",
        marker=dict(color=GOLD, size=12, symbol="circle",
                    line=dict(color=NAVY, width=2)),
        hovertemplate=f"<b>Leeftijd {leeftijd}</b><br>€{waarde/1e6:.1f}M<extra></extra>",
    ))
    fig2.update_layout(
        plot_bgcolor=CREAM, paper_bgcolor=CREAM,
        margin=dict(l=10, r=10, t=10, b=40), height=220,
        xaxis=dict(title=dict(text="Leeftijd", font=dict(size=12, color=MUTED)),
                   gridcolor="#e0d8cc", showline=False,
                   tickfont=dict(size=11, family="DM Sans", color=MUTED)),
        yaxis=dict(title=dict(text="€M", font=dict(size=12, color=MUTED)),
                   gridcolor="#e0d8cc", showline=False,
                   tickfont=dict(size=11, family="DM Sans", color=MUTED),
                   tickprefix="€", ticksuffix="M"),
        showlegend=False, font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Factorbijdrage ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Bijdrage per factor")

factors = ["Pos/Leeftijd", "Competitie", "Club", "Kwaliteitsscore", "Leeftijdsfactor"]
values  = [pl, cm, clm, (score/50)**B_KWALITEIT, lf]
kleuren = [GREEN if v > 1.0 else (GOLD if v == 1.0 else "#d1c4a8") for v in values]

fig3 = go.Figure(go.Bar(
    x=factors, y=values,
    marker_color=kleuren,
    text=[f"{v:.3f}×" for v in values],
    textposition="outside",
    textfont=dict(size=12, color=NAVY, family="DM Sans"),
))
fig3.add_hline(y=1.0, line_dash="dot", line_color="#9e9485", line_width=1.5,
               annotation_text="1.0× (neutraal)",
               annotation_font=dict(size=10, color=MUTED),
               annotation_position="bottom right")
fig3.update_layout(
    plot_bgcolor=CREAM, paper_bgcolor=CREAM,
    margin=dict(l=10, r=10, t=30, b=10), height=230,
    yaxis=dict(gridcolor="#e0d8cc", ticksuffix="×",
               range=[0, max(values) * 1.35],
               tickfont=dict(size=11, family="DM Sans", color=MUTED)),
    xaxis=dict(showgrid=False,
               tickfont=dict(size=12, family="DM Sans", color=NAVY)),
    font=dict(family="DM Sans"),
    showlegend=False,
)
st.plotly_chart(fig3, use_container_width=True)

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{SAND};border-radius:8px;padding:14px 18px;
            font-size:11px;color:{MUTED};line-height:1.9;margin-top:4px;">
  <b style="color:{NAVY};">Model</b> &nbsp;
  Globale basiswaarde €2M × Positie/Leeftijdscategorie × Competitie × Club × Kwaliteit × Leeftijdsfactor<br>
  <b style="color:{NAVY};">Kwaliteitscurve</b> &nbsp;
  (score/50)^{B_KWALITEIT} — gefit op handmatige dataset van bekende spelers<br>
  <b style="color:{NAVY};">Clubmultiplier</b> &nbsp;
  40% UEFA-coëfficiënt · 35% academie/exportreputatie · 25% koopkracht 2020–2025<br>
  <b style="color:{NAVY};">Beperking</b> &nbsp;
  Topclubs die zelden verkopen (Real Madrid, Bayern) worden onderschat — bekend probleem
  in alle transferwaarderingsmodellen
</div>
""", unsafe_allow_html=True)
