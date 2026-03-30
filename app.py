"""
⚽ BUNDESLIGA ANALYTICS PRO v2
================================
Professional Football Analytics Platform for Scouts, Sporting Directors & Squad Planners.
LLM-powered KI-Agent + Scouting Tools + Player Profiling.

Built by Francois 🦾 for Nico
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import re
import os
import glob
import json
import math
import base64 as b64

# ── Anthropic SDK ──
try:
    import anthropic
except ImportError:
    anthropic = None

# ── Similarity Engine ──
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ── Kaggle ──
try:
    import kagglehub
    HAS_KAGGLE = True
except ImportError:
    HAS_KAGGLE = False


# ═══════════════════════════════════════════════════════════
# 🎨 THEME & CONSTANTS
# ═══════════════════════════════════════════════════════════

T = {
    "PRIMARY": "#1768AC", "DARK": "#0B3D6B", "LIGHT": "#4A9FD9",
    "SKY": "#7EC8E3", "WHITE": "#FFFFFF",
    "BG": "#0A1628", "SURFACE": "#0F2136", "CARD": "#122A45",
    "TEXT": "#a8c4df", "TEXT_LIGHT": "#eaf2fb",
    "TEXT_MID": "#7ba3c4", "TEXT_DIM": "#4A6F8F",
    "GRID": "rgba(23, 104, 172, 0.1)",
    "BORDER": "rgba(23, 104, 172, 0.25)",
    "BORDER_H": "rgba(23, 104, 172, 0.6)",
    "GLOW": "rgba(23, 104, 172, 0.15)",
    "SUCCESS": "#34d399", "WARNING": "#fbbf24", "DANGER": "#f87171",
    "ACCENT2": "#a78bfa", "ACCENT3": "#f472b6",
}

def _rgb(h):
    h = h.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


# ═══════════════════════════════════════════════════════════
# 📊 METRICS & LABELS
# ═══════════════════════════════════════════════════════════

RANKABLE = [
    "Gls", "xG", "Ast", "xAG", "G+A", "G-PK", "npxG", "npxG+xAG",
    "Min", "MP", "Starts", "90s", "PrgC", "PrgP", "PrgR",
    "CrdY", "CrdR", "PK", "PKatt",
    "Gls_p90", "xG_p90", "Ast_p90", "xAG_p90", "GA_p90", "npxG_p90",
    "Sh", "SoT", "SoT%", "Sh/90", "SoT/90", "G/Sh", "G/SoT",
    "Int", "TklW", "Fls", "Fld", "Crs",
    "GA", "GA90", "Saves", "Save%", "CS", "CS%",
    "Sh_p90", "SoT_p90", "Int_p90", "TklW_p90",
]

LABELS = {
    "Gls": "Tore", "xG": "xG", "Ast": "Assists", "xAG": "xA",
    "G+A": "G+A", "G-PK": "Non-Pen Goals", "npxG": "npxG",
    "npxG+xAG": "npxG+xA", "Min": "Minuten", "MP": "Spiele",
    "Starts": "Starts", "90s": "90er", "PrgC": "Prog. Carries",
    "PrgP": "Prog. Pässe", "PrgR": "Prog. Empfänge",
    "CrdY": "Gelbe Karten", "CrdR": "Rote Karten",
    "Gls_p90": "Tore/90", "xG_p90": "xG/90", "Ast_p90": "Assists/90",
    "xAG_p90": "xA/90", "GA_p90": "G+A/90", "npxG_p90": "npxG/90",
    "Sh": "Schüsse", "SoT": "Schüsse aufs Tor", "SoT%": "Schussgenauigkeit",
    "Sh/90": "Schüsse/90", "SoT/90": "SoT/90", "G/Sh": "Tore/Schuss", "G/SoT": "Tore/SoT",
    "Int": "Interceptions", "TklW": "Tackles Won", "Fls": "Fouls",
    "Fld": "Fouls erlitten", "Crs": "Flanken",
    "GA": "Gegentore", "GA90": "Gegentore/90", "Saves": "Paraden",
    "Save%": "Paraden %", "CS": "Weiße Westen", "CS%": "Clean Sheet %",
    "Sh_p90": "Schüsse/90", "SoT_p90": "SoT/90",
    "Int_p90": "Int./90", "TklW_p90": "Tackles/90",
    "Age": "Alter", "Player": "Spieler", "Nation": "Nation",
    "Pos": "Position", "Squad": "Verein", "Comp": "Liga",
}

# Position-specific key metrics for scouting
POS_METRICS = {
    "FW": ["Gls_p90", "xG_p90", "npxG_p90", "Ast_p90", "SoT_p90", "G/SoT", "PrgC", "PrgR"],
    "MF": ["PrgP", "PrgC", "Ast_p90", "xAG_p90", "Gls_p90", "PrgR", "Int_p90", "TklW_p90"],
    "DF": ["TklW_p90", "Int_p90", "PrgP", "PrgC", "Fls", "CrdY"],
    "GK": ["Save%", "CS%", "GA90", "Saves"],
}

# Scouting score weights by position
SCOUT_WEIGHTS = {
    "FW": {"Gls_p90": 0.25, "xG_p90": 0.20, "npxG_p90": 0.15, "Ast_p90": 0.15, "SoT_p90": 0.10, "PrgC": 0.08, "PrgR": 0.07},
    "MF": {"PrgP": 0.20, "PrgC": 0.18, "Ast_p90": 0.15, "xAG_p90": 0.15, "Gls_p90": 0.10, "TklW_p90": 0.12, "Int_p90": 0.10},
    "DF": {"TklW_p90": 0.25, "Int_p90": 0.25, "PrgP": 0.20, "PrgC": 0.15, "Fls": -0.15},
    "GK": {"Save%": 0.40, "CS%": 0.30, "GA90": -0.30},
}

# Squad logos
SQUAD_LOGOS = {
    "Augsburg": "FC_Augsburg.svg.png", "Bayern Munich": "FC_Bayern_Muenchen.png",
    "Bochum": "VfL_Bochum.svg", "Dortmund": "Borussia_Dortmund.png",
    "Eint Frankfurt": "Logo_Eintracht_Frankfurt.svg.png", "Freiburg": "SC_Freiburg.svg",
    "Gladbach": "Borussia_Moenchengladbach.png", "Heidenheim": "1._FC_Heidenheim_1846.svg.png",
    "Hoffenheim": "TSGHoffenheim.png", "Holstein Kiel": "Holstein_Kiel.svg.png",
    "Leverkusen": "Bayer_Leverkusen.svg.png", "Mainz 05": "FSV_Mainz_05.png",
    "RB Leipzig": "rb_leipzig.png", "St. Pauli": "Fc_st_pauli.svg.png",
    "Stuttgart": "VfB_Stuttgart.svg.png", "Union Berlin": "1._FC_Union_Berlin.svg.png",
    "Werder Bremen": "SV_Werder_Bremen.svg.png", "Wolfsburg": "VfL_Wolfsburg.svg.png",
}

_IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Images")
PLAYER_PLACEHOLDER = "player_placeholder.png"

TEAM_ALIASES = {
    "hoffenheim": "Hoffenheim", "tsg": "Hoffenheim",
    "bayern": "Bayern Munich", "fcb": "Bayern Munich", "münchen": "Bayern Munich",
    "bvb": "Dortmund", "dortmund": "Dortmund",
    "leverkusen": "Leverkusen", "bayer": "Leverkusen",
    "leipzig": "RB Leipzig", "rbl": "RB Leipzig",
    "frankfurt": "Eint Frankfurt", "eintracht": "Eint Frankfurt",
    "freiburg": "Freiburg", "wolfsburg": "Wolfsburg",
    "gladbach": "Gladbach", "mönchengladbach": "Gladbach",
    "stuttgart": "Stuttgart", "vfb": "Stuttgart",
    "mainz": "Mainz 05", "union": "Union Berlin",
    "augsburg": "Augsburg", "werder": "Werder Bremen", "bremen": "Werder Bremen",
    "bochum": "Bochum", "heidenheim": "Heidenheim",
    "kiel": "Holstein Kiel", "st. pauli": "St. Pauli", "pauli": "St. Pauli",
}


# ═══════════════════════════════════════════════════════════
# 🖼️ IMAGE HELPERS
# ═══════════════════════════════════════════════════════════

def _img_to_b64(filename):
    path = os.path.join(_IMG_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return b64.b64encode(f.read()).decode()
    import urllib.request
    url = f"https://raw.githubusercontent.com/FrancoisLeGee/website-1/main/Images/{filename}"
    try:
        with urllib.request.urlopen(url) as resp:
            return b64.b64encode(resp.read()).decode()
    except:
        return ""

def get_squad_logo_html(squad, size=30):
    filename = SQUAD_LOGOS.get(squad, "")
    if filename:
        data = _img_to_b64(filename)
        if data:
            ext = filename.rsplit(".", 1)[-1].lower()
            mime = "image/svg+xml" if ext == "svg" else "image/png"
            return f'<img src="data:{mime};base64,{data}" style="height:{size}px;width:{size}px;object-fit:contain;">'
    return f'<span style="font-size:{size}px">⚽</span>'


# ═══════════════════════════════════════════════════════════
# 📂 DATA LOADING
# ═══════════════════════════════════════════════════════════


def _setup_kaggle():
    """Set Kaggle credentials from Streamlit secrets."""
    token = st.secrets.get("KAGGLE_API_TOKEN", os.environ.get("KAGGLE_API_TOKEN", ""))
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token
    # Legacy format
    username = st.secrets.get("KAGGLE_USERNAME", os.environ.get("KAGGLE_USERNAME", ""))
    key = st.secrets.get("KAGGLE_KEY", os.environ.get("KAGGLE_KEY", ""))
    if username and key:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key


@st.cache_data(ttl=3600)
def load_data():
    _setup_kaggle()
    """Load FBref data via kagglehub."""
    if not HAS_KAGGLE:
        st.error("kagglehub nicht installiert")
        return pd.DataFrame()
    try:
        path = kagglehub.dataset_download("sujithmandala/fbref-bundesliga-player-stats-2425-season")
        files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
        if not files:
            st.error("Keine CSV-Dateien gefunden")
            return pd.DataFrame()
        df = pd.read_csv(files[0])
    except Exception as e:
        st.error(f"Datenlade-Fehler: {e}")
        return pd.DataFrame()

    # Clean
    if "Comp" in df.columns:
        df = df[df["Comp"].str.contains("Bundesliga", case=False, na=False)]
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"].astype(str).str[:2], errors="coerce")

    # Ensure numeric
    for c in RANKABLE:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Per-90 calculations
    if "90s" in df.columns:
        for total, p90 in [("Gls","Gls_p90"),("xG","xG_p90"),("Ast","Ast_p90"),
                           ("xAG","xAG_p90"),("G+A","GA_p90"),("npxG","npxG_p90"),
                           ("Sh","Sh_p90"),("SoT","SoT_p90"),("Int","Int_p90"),("TklW","TklW_p90")]:
            if total in df.columns:
                df[p90] = (df[total] / df["90s"].replace(0, np.nan)).round(2)

    # Clean player names
    if "Player" in df.columns:
        df["Player"] = df["Player"].str.strip()

    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════
# 📈 PERCENTILE ENGINE
# ═══════════════════════════════════════════════════════════

def compute_percentiles(df, player_idx, metrics, pos_filter=True):
    """Compute percentile ranks for a player vs peers."""
    player = df.loc[player_idx]
    pos = str(player.get("Pos", "")).split(",")[0].strip()

    if pos_filter and pos in ["FW", "MF", "DF", "GK"]:
        peers = df[df["Pos"].str.contains(pos, na=False)]
    else:
        peers = df

    # Min minutes filter
    if "Min" in peers.columns:
        peers = peers[peers["Min"] >= 300]

    result = {}
    for m in metrics:
        if m not in df.columns:
            continue
        vals = peers[m].dropna()
        if len(vals) < 5:
            continue
        player_val = player.get(m, np.nan)
        if pd.isna(player_val):
            continue
        pct = (vals < player_val).sum() / len(vals) * 100
        result[m] = {"value": round(player_val, 2), "percentile": round(pct, 1)}

    return result


def find_similar_players(df, player_idx, n=5, min_minutes=300):
    """Find similar players using cosine similarity."""
    if not HAS_SKLEARN:
        return pd.DataFrame()

    player = df.loc[player_idx]
    pos = str(player.get("Pos", "")).split(",")[0].strip()

    # Use position-specific metrics
    metrics = POS_METRICS.get(pos, POS_METRICS["MF"])
    available = [m for m in metrics if m in df.columns]
    if len(available) < 3:
        available = [m for m in ["Gls_p90","Ast_p90","PrgP","PrgC","TklW_p90","Int_p90"] if m in df.columns]

    # Filter: same position, minimum minutes
    peers = df.copy()
    if pos in ["FW","MF","DF","GK"]:
        peers = peers[peers["Pos"].str.contains(pos, na=False)]
    if "Min" in peers.columns:
        peers = peers[peers["Min"] >= min_minutes]

    if len(peers) < 3:
        return pd.DataFrame()

    # Normalize
    data = peers[available].fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    # Player vector
    player_vec_idx = peers.index.get_loc(player_idx) if player_idx in peers.index else None
    if player_vec_idx is None:
        return pd.DataFrame()

    # Similarity
    sims = cosine_similarity([scaled[player_vec_idx]], scaled)[0]
    peers = peers.copy()
    peers["Similarity"] = sims
    peers = peers[peers.index != player_idx]
    peers = peers.nlargest(n, "Similarity")

    show_cols = ["Player", "Squad", "Pos", "Age", "Similarity"] + available[:4]
    show_cols = [c for c in show_cols if c in peers.columns]
    result = peers[show_cols].copy()
    result["Similarity"] = (result["Similarity"] * 100).round(1).astype(str) + "%"
    return result.reset_index(drop=True)


def compute_scouting_score(df, min_minutes=450):
    """Compute weighted scouting score by position."""
    result = df.copy()
    if "Min" in result.columns:
        result = result[result["Min"] >= min_minutes]

    scores = []
    for idx, row in result.iterrows():
        pos = str(row.get("Pos", "")).split(",")[0].strip()
        weights = SCOUT_WEIGHTS.get(pos, SCOUT_WEIGHTS.get("MF"))

        score = 0
        total_weight = 0
        for metric, weight in weights.items():
            if metric in result.columns:
                val = row.get(metric, np.nan)
                if pd.notna(val):
                    # Percentile-based scoring
                    col_vals = result[metric].dropna()
                    if len(col_vals) > 0:
                        pct = (col_vals < val).sum() / len(col_vals)
                        if weight < 0:  # Inverted metrics (fouls, goals against)
                            pct = 1 - pct
                            weight = abs(weight)
                        score += pct * weight
                        total_weight += weight

        scores.append(round(score / total_weight * 100, 1) if total_weight > 0 else 0)

    result["Scout Score"] = scores
    return result


# ═══════════════════════════════════════════════════════════
# 🎨 CSS
# ═══════════════════════════════════════════════════════════

def inject_css():
    st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp {{ background-color: {T['BG']}; font-family: 'Inter', sans-serif; }}
    [data-testid="stSidebar"] {{ background-color: {T['SURFACE']}; border-right: 1px solid {T['BORDER']}; }}

    .main-header {{
        font-size: 2rem; font-weight: 800;
        background: linear-gradient(135deg, {T['PRIMARY']}, {T['SKY']});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 20px 0 5px;
    }}
    .sub-header {{ color: {T['TEXT_DIM']}; font-size: 0.85rem; text-align: center; letter-spacing: 4px; text-transform: uppercase; }}
    .divider {{ width: 60px; height: 2px; background: {T['PRIMARY']}; margin: 15px auto; }}

    .metric-card {{
        background: {T['CARD']}; border: 1px solid {T['BORDER']};
        border-radius: 12px; padding: 16px; text-align: center;
    }}
    .metric-card:hover {{ border-color: {T['BORDER_H']}; }}
    .metric-val {{ font-size: 1.8rem; font-weight: 800; color: {T['SKY']}; }}
    .metric-label {{ font-size: 0.7rem; color: {T['TEXT_DIM']}; letter-spacing: 2px; text-transform: uppercase; margin-top: 4px; }}

    .player-card {{
        background: {T['CARD']}; border: 1px solid {T['BORDER']};
        border-radius: 12px; padding: 18px; margin-bottom: 12px;
        transition: all 0.2s;
    }}
    .player-card:hover {{ border-color: {T['BORDER_H']}; transform: translateY(-2px); }}
    .player-name {{ font-size: 1.1rem; font-weight: 700; color: {T['TEXT_LIGHT']}; }}
    .player-meta {{ font-size: 0.8rem; color: {T['TEXT_DIM']}; margin-top: 4px; }}
    .player-stat {{ display: inline-block; margin: 4px 6px 4px 0; padding: 3px 10px; background: {T['SURFACE']}; border-radius: 6px; font-size: 0.78rem; }}
    .player-stat-val {{ color: {T['SKY']}; font-weight: 700; }}
    .player-stat-label {{ color: {T['TEXT_DIM']}; }}

    .section-title {{
        font-size: 1.3rem; font-weight: 700; color: {T['TEXT_LIGHT']};
        border-left: 4px solid {T['PRIMARY']}; padding-left: 12px;
        margin: 25px 0 12px;
    }}

    .pct-bar-container {{ margin: 4px 0; display: flex; align-items: center; gap: 8px; }}
    .pct-bar-label {{ min-width: 120px; font-size: 0.78rem; color: {T['TEXT_MID']}; text-align: right; }}
    .pct-bar-track {{ flex: 1; height: 12px; background: {T['SURFACE']}; border-radius: 6px; overflow: hidden; }}
    .pct-bar-fill {{ height: 100%; border-radius: 6px; transition: width 0.5s; }}
    .pct-bar-value {{ min-width: 45px; font-size: 0.75rem; color: {T['TEXT']}; font-weight: 600; }}

    .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }}
    .tag-fw {{ background: rgba(239,68,68,0.15); color: #ef4444; }}
    .tag-mf {{ background: rgba(59,130,246,0.15); color: #3b82f6; }}
    .tag-df {{ background: rgba(34,197,94,0.15); color: #22c55e; }}
    .tag-gk {{ background: rgba(234,179,8,0.15); color: #eab308; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {T['SURFACE']}; border-radius: 8px; color: {T['TEXT_DIM']};
        border: 1px solid {T['BORDER']}; font-size: 0.85rem;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: rgba(23,104,172,0.2) !important;
        color: {T['SKY']} !important; border-color: {T['PRIMARY']} !important;
    }}

    .chat-msg {{ padding: 12px 16px; border-radius: 12px; margin: 8px 0; font-size: 0.9rem; line-height: 1.6; }}
    .chat-user {{ background: {T['SURFACE']}; color: {T['TEXT']}; border: 1px solid {T['BORDER']}; }}
    .chat-agent {{ background: rgba(23,104,172,0.1); color: {T['TEXT_LIGHT']}; border: 1px solid rgba(23,104,172,0.2); }}

    .footer {{ text-align: center; padding: 30px; font-size: 0.7rem; color: {T['TEXT_DIM']}; letter-spacing: 2px; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# 📊 VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════

def pct_color(pct):
    if pct >= 75: return T["SUCCESS"]
    if pct >= 50: return T["WARNING"]
    if pct >= 25: return "#f97316"
    return T["DANGER"]

def render_percentile_bars(percentiles, title="Scouting Report"):
    """Render FBref-style percentile bars."""
    html = f'<div class="section-title">{title}</div>'
    for metric, data in percentiles.items():
        label = LABELS.get(metric, metric)
        pct = data["percentile"]
        val = data["value"]
        color = pct_color(pct)
        html += f'''
        <div class="pct-bar-container">
            <div class="pct-bar-label">{label}</div>
            <div class="pct-bar-track">
                <div class="pct-bar-fill" style="width:{pct}%;background:{color};"></div>
            </div>
            <div class="pct-bar-value">{val} <span style="color:{color}">({pct:.0f}%)</span></div>
        </div>'''
    st.markdown(html, unsafe_allow_html=True)

def pos_tag(pos):
    p = str(pos).split(",")[0].strip()
    cls = {"FW":"tag-fw","MF":"tag-mf","DF":"tag-df","GK":"tag-gk"}.get(p, "tag-mf")
    return f'<span class="tag {cls}">{pos}</span>'

def make_radar(player_data, labels, name, color=T["PRIMARY"]):
    """Create a radar chart for a player."""
    values = list(player_data.values())
    labels_list = list(player_data.keys())
    values += values[:1]
    labels_list += labels_list[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=[LABELS.get(l,l) for l in labels_list],
        fill="toself", name=name,
        line=dict(color=color, width=2),
        fillcolor=f"rgba({_rgb(color)},0.15)"
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=T["GRID"]),
            angularaxis=dict(gridcolor=T["GRID"])
        ),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=400,
        font=dict(color=T["TEXT"], size=11), showlegend=True,
        margin=dict(l=60, r=60, t=30, b=30)
    )
    return fig

def plotly_defaults(fig, height=400):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=height,
        font=dict(color=T["TEXT"], family="Inter"),
        margin=dict(l=20, r=20, t=30, b=40)
    )
    return fig


# ═══════════════════════════════════════════════════════════
# 🤖 AGENT TOOLS
# ═══════════════════════════════════════════════════════════

def tool_top_players(df, column, n=10, ascending=False):
    if column not in df.columns: return pd.DataFrame()
    work = df.dropna(subset=[column])
    if "Min" in work.columns: work = work[work["Min"] >= 200]
    work = work.nsmallest(n, column) if ascending else work.nlargest(n, column)
    show = ["Player", "Squad", "Pos", "Age", column]
    extras = {"Gls": ["xG","Min"], "xG": ["Gls","Min"], "Ast": ["xAG","Min"], "xAG": ["Ast","Min"],
              "G+A": ["Gls","Ast","Min"], "PrgP": ["PrgC","Min"], "PrgC": ["PrgP","Min"],
              "TklW": ["Int","Min"], "Int": ["TklW","Min"]}
    for e in extras.get(column, ["Min"]):
        if e in df.columns and e not in show: show.append(e)
    return work[[c for c in show if c in work.columns]].reset_index(drop=True)

def tool_player_info(df, name):
    mask = df["Player"].str.contains(name, case=False, na=False)
    r = df[mask]
    if r.empty: return pd.DataFrame()
    show = ["Player","Squad","Pos","Age","MP","Starts","Min","Gls","Ast","G+A","xG","xAG","PrgC","PrgP"]
    return r[[c for c in show if c in r.columns]].sort_values("Min", ascending=False).reset_index(drop=True)

def tool_team_overview(df, team):
    team_lower = team.lower().strip()
    actual = TEAM_ALIASES.get(team_lower, team)
    mask = df["Squad"].str.contains(actual, case=False, na=False)
    r = df[mask]
    show = ["Player","Pos","Age","MP","Starts","Min","Gls","Ast","G+A","xG","xAG"]
    return r[[c for c in show if c in r.columns]].sort_values("Min", ascending=False).reset_index(drop=True)

def tool_compare_players(df, names, columns=None):
    masks = [df["Player"].str.contains(n, case=False, na=False) for n in names]
    combined = pd.concat([df[m] for m in masks if m.any()])
    if combined.empty: return pd.DataFrame()
    if columns:
        show = ["Player","Squad","Pos","Age"] + [c for c in columns if c in combined.columns]
    else:
        show = ["Player","Squad","Pos","Age","Gls","Ast","G+A","xG","xAG","PrgC","PrgP","Min"]
    return combined[[c for c in show if c in combined.columns]].reset_index(drop=True)

def tool_overperformers(df, metric_actual, metric_expected, n=10, direction="over", max_age=None):
    if metric_actual not in df.columns or metric_expected not in df.columns: return pd.DataFrame()
    w = df.dropna(subset=[metric_actual, metric_expected]).copy()
    if "Min" in w.columns: w = w[w["Min"] >= 300]
    if max_age and "Age" in w.columns: w = w[w["Age"] <= max_age]
    w["Diff"] = (w[metric_actual] - w[metric_expected]).round(2)
    w = w.nlargest(n, "Diff") if direction == "over" else w.nsmallest(n, "Diff")
    show = ["Player","Squad","Pos","Age",metric_actual,metric_expected,"Diff","Min"]
    return w[[c for c in show if c in w.columns]].reset_index(drop=True)

def tool_position_filter(df, position, column, n=10):
    if column not in df.columns: return pd.DataFrame()
    w = df[df["Pos"].str.contains(position, case=False, na=False)]
    return tool_top_players(w, column, n)

def tool_young_talents(df, max_age=23, column="G+A", n=10):
    if "Age" not in df.columns or column not in df.columns: return pd.DataFrame()
    w = df[df["Age"] <= max_age]
    if "Min" in w.columns: w = w[w["Min"] >= 200]
    w = w.dropna(subset=[column]).nlargest(n, column)
    show = ["Player","Squad","Pos","Age",column,"Min"]
    return w[[c for c in show if c in w.columns]].reset_index(drop=True)

def tool_similar_players(df, name, n=5):
    """Find players with a similar statistical profile."""
    mask = df["Player"].str.contains(name, case=False, na=False)
    matches = df[mask]
    if matches.empty: return pd.DataFrame()
    idx = matches.index[0]
    return find_similar_players(df, idx, n=n)

def tool_scouting_shortlist(df, position="FW", max_age=25, min_minutes=450, metric="Gls_p90", n=10):
    """Create a scouting shortlist based on criteria."""
    w = df.copy()
    if position and "Pos" in w.columns:
        w = w[w["Pos"].str.contains(position, case=False, na=False)]
    if max_age and "Age" in w.columns:
        w = w[w["Age"] <= max_age]
    if min_minutes and "Min" in w.columns:
        w = w[w["Min"] >= min_minutes]
    if metric not in w.columns:
        metric = "G+A"
    w = w.dropna(subset=[metric]).nlargest(n, metric)
    show = ["Player","Squad","Pos","Age",metric,"Min","Gls","Ast","xG"]
    return w[[c for c in show if c in w.columns]].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════
# 🤖 AGENT (LLM)
# ═══════════════════════════════════════════════════════════

CLAUDE_FAST = "claude-haiku-4-5-20251001"
CLAUDE_SMART = "claude-sonnet-4-6"
MAX_AGENT_STEPS = 5

def get_llm_client():
    key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY"))
    if not key or not anthropic: return None
    return anthropic.Anthropic(api_key=key)

def _call_llm(system, user, model=CLAUDE_FAST, max_tokens=500):
    client = get_llm_client()
    if not client: return ""
    try:
        r = client.messages.create(model=model, max_tokens=max_tokens,
                                    system=system, messages=[{"role":"user","content":user}])
        return r.content[0].text.strip()
    except: return ""

INTENT_PROMPT = """Du bist ein Intent-Parser für eine Fußball-Analyse-App. Antworte NUR mit JSON.

TOOLS: top_players, player_info, team_overview, compare_players, overperformers, position_filter, young_talents, similar_players, scouting_shortlist

SPALTEN: Gls, xG, Ast, xAG, G+A, G-PK, npxG, Min, MP, PrgC, PrgP, PrgR, Gls_p90, xG_p90, Ast_p90, xAG_p90, GA_p90, npxG_p90, Sh, SoT, Int, TklW, Save%, CS%

TEAMS: Hoffenheim, Bayern Munich, Dortmund, Leverkusen, RB Leipzig, Eint Frankfurt, Freiburg, Wolfsburg, Gladbach, Stuttgart, Mainz 05, Union Berlin, Augsburg, Werder Bremen, Bochum, Heidenheim, Holstein Kiel, St. Pauli

FORMAT: {"tool":"...","params":{...},"explanation":"Emoji + Deutsch","reasoning":"...","complex":false}

Setze complex:true bei analytischen/vergleichenden Fragen."""

AGENT_PROMPT = """Du bist ein professioneller Bundesliga-Analyst und Scout. Du analysierst Daten fundiert und gibst Einordnungen wie ein erfahrener Kaderplaner.

TOOLS: top_players, player_info, team_overview, compare_players, overperformers, position_filter, young_talents, similar_players, scouting_shortlist

FORMAT für Tool-Aufruf: {"action":"tool_call","tool":"...","params":{...},"thought":"..."}
FORMAT für Antwort: {"action":"final_answer","summary_title":"Emoji + Titel","narrative":"4-8 Sätze, Deutsch, konkrete Namen und Zahlen"}

Maximal 5 Tool-Aufrufe, dann final_answer. Bewerte und interpretiere die Daten wie ein Scout."""

def parse_intent_llm(question):
    raw = _call_llm(INTENT_PROMPT, question, model=CLAUDE_FAST, max_tokens=400)
    try:
        cleaned = re.sub(r'^[^{]*', '', raw)
        cleaned = re.sub(r'[^}]*$', '', cleaned) + '}'
        return json.loads(cleaned)
    except:
        return _fallback_parse(question)

def _fallback_parse(question):
    q = question.lower()
    for alias, team in TEAM_ALIASES.items():
        if alias in q:
            return {"tool":"team_overview","params":{"team":team},"explanation":f"🏟️ {team}","reasoning":"Team erkannt","complex":False}
    if any(w in q for w in ["vergleich","vs","gegen"]):
        return {"tool":"top_players","params":{"column":"G+A","n":10},"explanation":"⚔️ Vergleich","reasoning":"Vergleichs-Frage","complex":True}
    return {"tool":"top_players","params":{"column":"Gls","n":10},"explanation":"🏆 Top Torschützen","reasoning":"Fallback","complex":False}

def agent_step(client, messages, max_tokens=800):
    try:
        r = client.messages.create(model=CLAUDE_SMART, max_tokens=max_tokens,
                                    system=AGENT_PROMPT, messages=messages)
        raw = r.content[0].text.strip()
        cleaned = re.sub(r'^[^{]*', '', raw)
        cleaned = re.sub(r'[^}]*$', '', cleaned) + '}'
        return json.loads(cleaned)
    except:
        return {"action":"final_answer","summary_title":"⚠️ Fehler","narrative":"Konnte keine Analyse erstellen."}


class FootballAgent:
    def __init__(self, df):
        self.df = df
        self.tools = {
            "top_players": tool_top_players,
            "player_info": tool_player_info,
            "team_overview": tool_team_overview,
            "compare_players": tool_compare_players,
            "overperformers": tool_overperformers,
            "position_filter": tool_position_filter,
            "young_talents": tool_young_talents,
            "similar_players": lambda df, **p: tool_similar_players(df, **p),
            "scouting_shortlist": tool_scouting_shortlist,
        }

    def _exec_tool(self, tool_name, params):
        if tool_name not in self.tools: return pd.DataFrame()
        try: return self.tools[tool_name](self.df, **params)
        except: return pd.DataFrame()

    def run(self, question):
        intent = parse_intent_llm(question)
        is_complex = intent.get("complex", False)

        if is_complex and get_llm_client():
            return self.run_multi(question)

        tool_name = intent.get("tool", "top_players")
        params = intent.get("params", {})
        if tool_name not in self.tools:
            tool_name = "top_players"
            params = {"column": "Gls", "n": 10}
        try:
            result = self._exec_tool(tool_name, params)
        except:
            result = pd.DataFrame()
        return {"mode":"single","explanation":intent.get("explanation",""),"result":result,
                "tool":tool_name,"params":params,"reasoning":intent.get("reasoning","")}

    def run_multi(self, question):
        client = get_llm_client()
        messages = [{"role":"user","content":question}]
        steps = []
        narrative = ""
        summary_title = ""

        for _ in range(MAX_AGENT_STEPS):
            action = agent_step(client, messages)
            if action.get("action") == "final_answer":
                narrative = action.get("narrative", "")
                summary_title = action.get("summary_title", "Analyse")
                break
            if action.get("action") == "tool_call":
                tool_name = action.get("tool", "")
                params = action.get("params", {})
                result_df = self._exec_tool(tool_name, params)
                result_text = result_df.to_string(index=False, max_rows=15) if not result_df.empty else "(Keine Ergebnisse)"
                steps.append({"tool":tool_name,"params":params,"thought":action.get("thought",""),"result_df":result_df})
                messages.append({"role":"assistant","content":json.dumps(action, ensure_ascii=False)})
                messages.append({"role":"user","content":f"TOOL_RESULT ({tool_name}):\n{result_text}"})
            else:
                break

        if not narrative and steps:
            data_summary = "\n".join([f"--- {s['tool']} ---\n{s['result_df'].head(8).to_string(index=False)}" for s in steps if not s['result_df'].empty])
            messages.append({"role":"user","content":f"Daten:\n{data_summary}\n\nSchreibe analytische Zusammenfassung. JSON: {{\"action\":\"final_answer\",\"summary_title\":\"...\",\"narrative\":\"...\"}}"})
            action = agent_step(client, messages, max_tokens=1200)
            narrative = action.get("narrative", "")
            summary_title = action.get("summary_title", "📊 Analyse")

        return {"mode":"multi","steps":steps,"narrative":narrative,"summary_title":summary_title}


# ═══════════════════════════════════════════════════════════
# 🏠 TAB: DASHBOARD
# ═══════════════════════════════════════════════════════════

def tab_dashboard(df):
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    total_goals = int(df["Gls"].sum()) if "Gls" in df.columns else 0
    total_players = len(df)
    avg_age = round(df["Age"].mean(), 1) if "Age" in df.columns else 0
    teams = df["Squad"].nunique() if "Squad" in df.columns else 0

    for col, (val, label) in zip([c1,c2,c3,c4], [
        (total_goals, "Tore gesamt"), (total_players, "Spieler"),
        (avg_age, "Ø Alter"), (teams, "Teams")
    ]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">🏆 Top Torschützen</div>', unsafe_allow_html=True)
        top_scorers = tool_top_players(df, "Gls", n=10)
        if not top_scorers.empty:
            st.dataframe(top_scorers, use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="section-title">🎯 Top Assists</div>', unsafe_allow_html=True)
        top_assists = tool_top_players(df, "Ast", n=10)
        if not top_assists.empty:
            st.dataframe(top_assists, use_container_width=True, hide_index=True)

    # xG Chart
    st.markdown('<div class="section-title">📈 xG vs. Tatsächliche Tore (Top 20)</div>', unsafe_allow_html=True)
    if "xG" in df.columns and "Gls" in df.columns:
        top20 = df.nlargest(20, "Gls").dropna(subset=["xG","Gls"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=top20["xG"], y=top20["Gls"], mode="markers+text",
                                  text=top20["Player"].apply(lambda x: x.split()[-1]),
                                  textposition="top center", textfont=dict(size=9, color=T["TEXT_MID"]),
                                  marker=dict(size=10, color=T["PRIMARY"], line=dict(width=1, color=T["SKY"]))))
        max_val = max(top20["xG"].max(), top20["Gls"].max()) + 2
        fig.add_trace(go.Scatter(x=[0,max_val], y=[0,max_val], mode="lines",
                                  line=dict(dash="dash", color=T["TEXT_DIM"]), showlegend=False))
        fig.update_layout(xaxis_title="Expected Goals (xG)", yaxis_title="Tatsächliche Tore", showlegend=False)
        st.plotly_chart(plotly_defaults(fig, 450), use_container_width=True)


# ═══════════════════════════════════════════════════════════
# 🔍 TAB: SPIELER-SUCHE
# ═══════════════════════════════════════════════════════════

def tab_search(df):
    st.markdown('<div class="section-title">🔍 Spieler-Suche & Scouting</div>', unsafe_allow_html=True)

    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        positions = st.multiselect("Position", ["FW", "MF", "DF", "GK"], default=["FW", "MF", "DF", "GK"])
    with col2:
        age_range = st.slider("Alter", 16, 40, (16, 40))
    with col3:
        min_minutes = st.number_input("Min. Minuten", 0, 3000, 300, 50)
    with col4:
        sort_by = st.selectbox("Sortieren nach", ["Scout Score", "Gls_p90", "xG_p90", "Ast_p90", "G+A", "PrgP", "PrgC", "TklW_p90", "Age"])

    # Team filter
    all_teams = sorted(df["Squad"].unique()) if "Squad" in df.columns else []
    selected_teams = st.multiselect("Teams", all_teams, default=[])

    # Apply filters
    filtered = df.copy()
    if positions and "Pos" in filtered.columns:
        mask = filtered["Pos"].str.contains("|".join(positions), case=False, na=False)
        filtered = filtered[mask]
    if "Age" in filtered.columns:
        filtered = filtered[(filtered["Age"] >= age_range[0]) & (filtered["Age"] <= age_range[1])]
    if min_minutes and "Min" in filtered.columns:
        filtered = filtered[filtered["Min"] >= min_minutes]
    if selected_teams and "Squad" in filtered.columns:
        filtered = filtered[filtered["Squad"].isin(selected_teams)]

    # Compute scouting scores
    if sort_by == "Scout Score":
        filtered = compute_scouting_score(filtered, min_minutes=0)
        filtered = filtered.sort_values("Scout Score", ascending=False)
    elif sort_by in filtered.columns:
        filtered = filtered.dropna(subset=[sort_by]).sort_values(sort_by, ascending=(sort_by == "Age"))

    st.markdown(f"**{len(filtered)} Spieler** gefunden")

    # Render cards
    for i in range(0, min(len(filtered), 30), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(filtered): break
            row = filtered.iloc[idx]
            real_idx = filtered.index[idx]
            pos = str(row.get("Pos",""))
            primary_pos = pos.split(",")[0].strip()

            # Get key metrics for position
            key_metrics = POS_METRICS.get(primary_pos, POS_METRICS["MF"])[:5]

            with col:
                logo = get_squad_logo_html(row.get("Squad",""), size=24)
                stats_html = ""
                for m in key_metrics:
                    if m in df.columns and pd.notna(row.get(m)):
                        stats_html += f'<span class="player-stat"><span class="player-stat-val">{row[m]:.1f}</span> <span class="player-stat-label">{LABELS.get(m,m)}</span></span>'

                scout_score = f'<span class="player-stat"><span class="player-stat-val" style="color:{T["SUCCESS"]}">{row.get("Scout Score",0):.0f}</span> <span class="player-stat-label">Scout</span></span>' if "Scout Score" in row.index and pd.notna(row.get("Scout Score")) else ""

                st.markdown(f'''
                <div class="player-card">
                    <div style="display:flex;align-items:center;gap:8px;">
                        {logo}
                        <div>
                            <div class="player-name">{row.get("Player","?")}</div>
                            <div class="player-meta">{row.get("Squad","")} · {pos_tag(pos)} · {int(row.get("Age",0))} Jahre · {int(row.get("Min",0))} Min</div>
                        </div>
                    </div>
                    <div style="margin-top:10px;">{stats_html}{scout_score}</div>
                </div>
                ''', unsafe_allow_html=True)

                if st.button(f"👤 Profil", key=f"profile_{real_idx}", use_container_width=True):
                    st.session_state["profile_player_idx"] = real_idx
                    st.session_state["active_tab"] = "profile"
                    st.rerun()


# ═══════════════════════════════════════════════════════════
# 👤 TAB: SPIELER-PROFIL
# ═══════════════════════════════════════════════════════════

def tab_profile(df):
    st.markdown('<div class="section-title">👤 Spieler-Profil</div>', unsafe_allow_html=True)

    # Player selection
    player_idx = st.session_state.get("profile_player_idx", None)
    all_players = df["Player"].dropna().unique().tolist() if "Player" in df.columns else []

    selected = st.selectbox("Spieler auswählen", all_players,
                            index=all_players.index(df.loc[player_idx, "Player"]) if player_idx is not None and player_idx in df.index else 0)

    mask = df["Player"] == selected
    if not mask.any():
        st.warning("Spieler nicht gefunden")
        return

    player_idx = df[mask].index[0]
    player = df.loc[player_idx]
    pos = str(player.get("Pos","")).split(",")[0].strip()

    # Header
    logo = get_squad_logo_html(player.get("Squad",""), size=40)
    st.markdown(f'''
    <div style="display:flex;align-items:center;gap:16px;padding:20px;background:{T["CARD"]};border-radius:12px;border:1px solid {T["BORDER"]};">
        {logo}
        <div>
            <div style="font-size:1.8rem;font-weight:800;color:{T["TEXT_LIGHT"]}">{player.get("Player","")}</div>
            <div style="color:{T["TEXT_MID"]}">{player.get("Squad","")} · {pos_tag(player.get("Pos",""))} · {int(player.get("Age",0))} Jahre · {player.get("Nation","")}</div>
            <div style="color:{T["TEXT_DIM"]};font-size:0.85rem;margin-top:4px;">{int(player.get("MP",0))} Spiele · {int(player.get("Starts",0))} Starts · {int(player.get("Min",0))} Minuten</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("")

    # Key stats
    c1, c2, c3, c4, c5 = st.columns(5)
    key_stats = [("Gls","Tore"),("Ast","Assists"),("G+A","G+A"),("xG","xG"),("xAG","xA")]
    for col, (metric, label) in zip([c1,c2,c3,c4,c5], key_stats):
        val = player.get(metric, 0)
        if pd.notna(val):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{val:.1f}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns([3, 2])

    with col1:
        # Percentile bars
        metrics_for_report = POS_METRICS.get(pos, POS_METRICS["MF"])
        # Add common metrics
        all_report_metrics = metrics_for_report + ["Gls_p90", "Ast_p90", "GA_p90", "PrgP", "PrgC", "Min"]
        all_report_metrics = list(dict.fromkeys(all_report_metrics))  # dedupe

        percentiles = compute_percentiles(df, player_idx, all_report_metrics)
        if percentiles:
            render_percentile_bars(percentiles, "📊 Scouting Report (Percentile vs. Position)")

    with col2:
        # Radar
        radar_metrics = POS_METRICS.get(pos, POS_METRICS["MF"])[:8]
        radar_data = {}
        for m in radar_metrics:
            pct = percentiles.get(m, {}).get("percentile")
            if pct is not None:
                radar_data[m] = pct

        if len(radar_data) >= 3:
            fig = make_radar(radar_data, list(radar_data.keys()), player.get("Player",""), T["PRIMARY"])
            st.plotly_chart(fig, use_container_width=True)

        # xG Analysis
        gls = player.get("Gls", 0)
        xg = player.get("xG", 0)
        if pd.notna(gls) and pd.notna(xg) and xg > 0:
            diff = gls - xg
            color = T["SUCCESS"] if diff > 0 else T["DANGER"]
            label = "Overperformer" if diff > 0 else "Underperformer"
            st.markdown(f'''
            <div style="background:{T["CARD"]};border:1px solid {T["BORDER"]};border-radius:12px;padding:16px;text-align:center;">
                <div style="font-size:0.75rem;color:{T["TEXT_DIM"]};letter-spacing:2px;text-transform:uppercase;">xG-Analyse</div>
                <div style="font-size:1.5rem;font-weight:800;color:{color};margin:8px 0;">{diff:+.1f}</div>
                <div style="font-size:0.85rem;color:{T["TEXT_MID"]}">{gls:.0f} Tore vs {xg:.1f} xG · {label}</div>
            </div>
            ''', unsafe_allow_html=True)

    # Similar players
    if HAS_SKLEARN:
        st.markdown('<div class="section-title">🔗 Ähnliche Spieler</div>', unsafe_allow_html=True)
        similar = find_similar_players(df, player_idx, n=5)
        if not similar.empty:
            st.dataframe(similar, use_container_width=True, hide_index=True)
        else:
            st.info("Nicht genug Daten für Ähnlichkeitsanalyse.")


# ═══════════════════════════════════════════════════════════
# ⚔️ TAB: VERGLEICH
# ═══════════════════════════════════════════════════════════

def tab_compare(df):
    st.markdown('<div class="section-title">⚔️ Spieler-Vergleich</div>', unsafe_allow_html=True)

    all_players = df["Player"].dropna().unique().tolist()
    selected = st.multiselect("Spieler auswählen (2-4)", all_players, max_selections=4)

    if len(selected) < 2:
        st.info("Wähle mindestens 2 Spieler zum Vergleichen.")
        return

    # Get player data
    players = []
    for name in selected:
        mask = df["Player"] == name
        if mask.any():
            players.append(df[mask].iloc[0])

    if len(players) < 2: return

    # Radar overlay
    colors = [T["PRIMARY"], T["ACCENT2"], T["ACCENT3"], T["SUCCESS"]]
    pos = str(players[0].get("Pos","")).split(",")[0].strip()
    metrics = POS_METRICS.get(pos, POS_METRICS["MF"])[:8]

    fig = go.Figure()
    for i, p in enumerate(players):
        p_idx = df[df["Player"] == p["Player"]].index[0]
        pcts = compute_percentiles(df, p_idx, metrics)
        values = [pcts.get(m, {}).get("percentile", 0) for m in metrics]
        values += values[:1]
        theta = [LABELS.get(m,m) for m in metrics] + [LABELS.get(metrics[0], metrics[0])]

        fig.add_trace(go.Scatterpolar(
            r=values, theta=theta, fill="toself",
            name=p["Player"], line=dict(color=colors[i % len(colors)], width=2),
            fillcolor=f"rgba({_rgb(colors[i % len(colors)])},0.1)"
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,100], gridcolor=T["GRID"]),
                   angularaxis=dict(gridcolor=T["GRID"])),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        height=450, font=dict(color=T["TEXT"]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Bar comparison
    compare_metrics = st.multiselect("Metriken vergleichen",
                                      [m for m in RANKABLE if m in df.columns],
                                      default=["Gls", "Ast", "xG", "xAG", "PrgP", "PrgC"])

    if compare_metrics:
        fig_bar = go.Figure()
        for i, p in enumerate(players):
            vals = [float(p.get(m, 0) or 0) for m in compare_metrics]
            fig_bar.add_trace(go.Bar(
                name=p["Player"], x=[LABELS.get(m,m) for m in compare_metrics],
                y=vals, marker_color=colors[i % len(colors)]
            ))
        fig_bar.update_layout(barmode="group")
        st.plotly_chart(plotly_defaults(fig_bar, 400), use_container_width=True)

    # Stat table
    show_cols = ["Player","Squad","Pos","Age","Gls","Ast","G+A","xG","xAG","PrgP","PrgC","TklW","Int","Min"]
    show_cols = [c for c in show_cols if c in df.columns]
    compare_df = pd.DataFrame(players)[show_cols]
    st.dataframe(compare_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# 🤖 TAB: KI-AGENT
# ═══════════════════════════════════════════════════════════

def tab_agent(df):
    st.markdown('<div class="section-title">🤖 KI-Analyst</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{T["TEXT_DIM"]};font-size:0.85rem;">Stelle Fragen zur Bundesliga — der Agent analysiert die Daten für dich.</p>', unsafe_allow_html=True)

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display history
    for msg in st.session_state.chat_history:
        cls = "chat-user" if msg["role"] == "user" else "chat-agent"
        icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f'<div class="chat-msg {cls}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)
        if "dataframe" in msg and msg["dataframe"] is not None:
            st.dataframe(msg["dataframe"], use_container_width=True, hide_index=True)

    # Input
    question = st.chat_input("Frage stellen... (z.B. 'Wer sind die besten jungen Stürmer?')")

    if question:
        st.session_state.chat_history.append({"role":"user","content":question})

        agent = FootballAgent(df)
        with st.spinner("🤖 Analysiere..."):
            result = agent.run(question)

        if result["mode"] == "single":
            response = result.get("explanation", "")
            st.session_state.chat_history.append({
                "role":"agent", "content":response,
                "dataframe": result.get("result") if not result.get("result", pd.DataFrame()).empty else None
            })
        else:
            title = result.get("summary_title", "📊 Analyse")
            narrative = result.get("narrative", "")
            content = f"**{title}**\n\n{narrative}"

            # Include step results
            result_df = None
            steps = result.get("steps", [])
            if steps:
                last_df = steps[-1].get("result_df", pd.DataFrame())
                if not last_df.empty:
                    result_df = last_df

            st.session_state.chat_history.append({"role":"agent","content":content,"dataframe":result_df})

        st.rerun()

    # Example questions
    if not st.session_state.chat_history:
        st.markdown(f'<div style="color:{T["TEXT_DIM"]};font-size:0.8rem;margin-top:20px;">Beispiele:</div>', unsafe_allow_html=True)
        examples = [
            "Wer hat die meisten Tore?",
            "Vergleiche Wirtz und Musiala",
            "Beste junge Talente unter 23",
            "Wie gut performt Hoffenheims Sturm?",
            "Overperformer bei Toren",
            "Finde Spieler ähnlich wie Harry Kane",
        ]
        cols = st.columns(3)
        for i, ex in enumerate(examples):
            with cols[i % 3]:
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role":"user","content":ex})
                    st.rerun()


# ═══════════════════════════════════════════════════════════
# 📊 TAB: TEAM-ANALYSE
# ═══════════════════════════════════════════════════════════

def tab_team(df):
    st.markdown('<div class="section-title">📊 Team-Analyse</div>', unsafe_allow_html=True)

    teams = sorted(df["Squad"].unique()) if "Squad" in df.columns else []
    selected_team = st.selectbox("Team auswählen", teams)

    if not selected_team: return

    team_df = df[df["Squad"] == selected_team].copy()
    logo = get_squad_logo_html(selected_team, size=40)

    st.markdown(f'''
    <div style="display:flex;align-items:center;gap:16px;padding:20px;background:{T["CARD"]};border-radius:12px;border:1px solid {T["BORDER"]};">
        {logo}
        <div>
            <div style="font-size:1.5rem;font-weight:800;color:{T["TEXT_LIGHT"]}">{selected_team}</div>
            <div style="color:{T["TEXT_MID"]}">{len(team_df)} Spieler im Kader</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        goals = int(team_df["Gls"].sum()) if "Gls" in team_df.columns else 0
        st.markdown(f'<div class="metric-card"><div class="metric-val">{goals}</div><div class="metric-label">Team Tore</div></div>', unsafe_allow_html=True)
    with c2:
        avg_age = round(team_df["Age"].mean(), 1) if "Age" in team_df.columns else 0
        st.markdown(f'<div class="metric-card"><div class="metric-val">{avg_age}</div><div class="metric-label">Ø Alter</div></div>', unsafe_allow_html=True)
    with c3:
        xg = round(team_df["xG"].sum(), 1) if "xG" in team_df.columns else 0
        st.markdown(f'<div class="metric-card"><div class="metric-val">{xg}</div><div class="metric-label">Team xG</div></div>', unsafe_allow_html=True)
    with c4:
        assists = int(team_df["Ast"].sum()) if "Ast" in team_df.columns else 0
        st.markdown(f'<div class="metric-card"><div class="metric-val">{assists}</div><div class="metric-label">Team Assists</div></div>', unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        # Kader
        st.markdown('<div class="section-title">📋 Kader</div>', unsafe_allow_html=True)
        show_cols = ["Player","Pos","Age","MP","Min","Gls","Ast","G+A","xG"]
        show_cols = [c for c in show_cols if c in team_df.columns]
        st.dataframe(team_df[show_cols].sort_values("Min", ascending=False), use_container_width=True, hide_index=True)

    with col2:
        # Age distribution
        st.markdown('<div class="section-title">📊 Altersstruktur</div>', unsafe_allow_html=True)
        if "Age" in team_df.columns:
            fig_age = px.histogram(team_df, x="Age", nbins=12, color_discrete_sequence=[T["PRIMARY"]])
            fig_age.update_layout(xaxis_title="Alter", yaxis_title="Anzahl Spieler")
            st.plotly_chart(plotly_defaults(fig_age, 300), use_container_width=True)

        # Position distribution
        if "Pos" in team_df.columns:
            pos_counts = team_df["Pos"].str.split(",").explode().str.strip().value_counts()
            fig_pos = px.pie(values=pos_counts.values, names=pos_counts.index,
                            color_discrete_sequence=[T["PRIMARY"], T["SKY"], T["SUCCESS"], T["WARNING"]])
            fig_pos.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=250)
            st.plotly_chart(fig_pos, use_container_width=True)

    # Team vs League comparison
    st.markdown('<div class="section-title">📈 Team vs. Liga-Durchschnitt (per Spieler)</div>', unsafe_allow_html=True)
    compare_metrics = ["Gls_p90", "xG_p90", "Ast_p90", "PrgP", "PrgC", "TklW_p90"]
    compare_metrics = [m for m in compare_metrics if m in df.columns]

    if compare_metrics:
        team_avg = team_df[compare_metrics].mean()
        league_avg = df[compare_metrics].mean()

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name=selected_team, x=[LABELS.get(m,m) for m in compare_metrics],
                                   y=team_avg.values, marker_color=T["PRIMARY"]))
        fig_comp.add_trace(go.Bar(name="Liga-Ø", x=[LABELS.get(m,m) for m in compare_metrics],
                                   y=league_avg.values, marker_color=T["TEXT_DIM"]))
        fig_comp.update_layout(barmode="group")
        st.plotly_chart(plotly_defaults(fig_comp, 350), use_container_width=True)


# ═══════════════════════════════════════════════════════════
# 🚀 MAIN APP
# ═══════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Bundesliga Analytics Pro", page_icon="⚽", layout="wide")
    inject_css()

    # Header
    st.markdown('<div class="main-header">⚽ BUNDESLIGA ANALYTICS PRO</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Professional Football Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Load data
    with st.spinner("📂 Lade Bundesliga-Daten..."):
        df = load_data()

    if df.empty:
        st.error("Keine Daten verfügbar. Bitte Kaggle-Credentials prüfen.")
        return

    # Tabs
    tab_names = ["🏠 Dashboard", "🔍 Spieler-Suche", "👤 Profil", "⚔️ Vergleich", "🤖 KI-Agent", "📊 Team-Analyse"]
    tabs = st.tabs(tab_names)

    with tabs[0]: tab_dashboard(df)
    with tabs[1]: tab_search(df)
    with tabs[2]: tab_profile(df)
    with tabs[3]: tab_compare(df)
    with tabs[4]: tab_agent(df)
    with tabs[5]: tab_team(df)

    # Footer
    st.markdown(f'<div class="footer">BUNDESLIGA ANALYTICS PRO © 2025 · BUILT BY FRANCOIS 🦾 FOR NICO · POWERED BY CLAUDE + STREAMLIT</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
