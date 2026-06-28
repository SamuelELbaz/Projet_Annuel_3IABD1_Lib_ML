"""
♻️ Classification de déchets — Application web
Projet Annuel 3IABD — 2025/2026

Navigation :
  Page 0 → Accueil (choix du modèle)
  Page 1 → Prédiction avec le modèle sélectionné
"""

import sys, os, ctypes, subprocess, random
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ─── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE_DIR, "dataset.csv")
C_SRC     = os.path.join(BASE_DIR, "model_lineaire", "model_lineaire.c")
SO_PATH   = os.path.join(BASE_DIR, "model_lineaire", "model_lineaire.so")

sys.path.insert(0, os.path.join(BASE_DIR, "pmc"))
from pmc import PMC  # noqa: E402

# ─── RBF (lib externe, branche rbf) ─────────────────────────────────────
# rbf_main charge la lib C au moment de l'import : on la compile avant si besoin.
RBF_DIR   = os.path.join(BASE_DIR, "rbf")
RBF_SO    = os.path.join(RBF_DIR, "rbf_linux.so")
RBF_C_SRC = os.path.join(RBF_DIR, "main.c")
try:
    if not os.path.exists(RBF_SO) and os.path.exists(RBF_C_SRC):
        subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-lm", "-o", RBF_SO, RBF_C_SRC],
                       capture_output=True, text=True)
    sys.path.insert(0, RBF_DIR)
    from rbf_main import RBF, one_hot, set_seed  # noqa: E402
    RBF_AVAILABLE = True
except Exception:
    RBF_AVAILABLE = False

# ─── Constantes ─────────────────────────────────────────────────────────
NB_FEATURES     = 3      # features couleur globales utilisées par le PMC (R, G, B)
NB_FEATURES_LIN = 29     # features du modèle linéaire : grille 3×3 RGB (27) + gradient (2)
NB_FEATURES_RBF = 29     # features du RBF : mêmes 29 que le linéaire (grille 3×3 + gradient)
NB_CLASSES      = 3

LINEAR_FEATURE_MODE = "grille3x3+grad"
LINEAR_BASE_MODEL   = "pseudo_inverse"
# Biais de classe (recalibré pour les 29 features) — quasi-nul : la grille spatiale
# a rééquilibré le modèle, le rappel n'a plus besoin d'un gros correctif.
LINEAR_CLASS_BIAS   = np.array([0.0, 0.0, 0.0], dtype=np.float64)

# ─── Rejet « je ne sais pas » ──────────────────────────────────────────
# Si la confiance (max softmax) est sous le seuil, le modèle préfère dire « je sais pas »
# plutôt que de risquer une erreur. Seuil linéaire calibré : ~76 % de précision sur
# les images acceptées (au lieu de 63 % en forçant une réponse sur tout).
REJECT_THRESHOLD = {"linear": 0.40, "pmc": 0.45, "rbf": 0.45}

MESSAGES_JE_SAIS_PAS = [
    ("🤷", "Aucune idée !",        "Cette image me laisse perplexe — je préfère ne pas inventer."),
    ("🙅", "Je sèche…",          "Pas assez sûr de moi pour me prononcer sur ce déchet."),
    ("🤔", "Hmm, mystère…",      "Je ne reconnais pas vraiment ça. Une photo plus nette m'aiderait !"),
    ("😅", "Je passe mon tour",   "Je préfère me taire plutôt que de dire une bêtise."),
    ("🫠", "Boule de gomme…",     "Trop incertain pour trancher entre les catégories."),
    ("🔮", "Ma boule de cristal bug", "Je ne sais pas classer cette image avec assez de confiance."),
]

CATEGORIES = {
    0: ("Compost",    "Déchets organiques et matières compostables", "#1a7a4a", "#34d399"),
    1: ("Chimique",   "Piles, batteries et déchets dangereux",       "#dc2626", "#f87171"),
    2: ("Recyclable", "Verre, plastique, papier, métal recyclable",  "#2563eb", "#60a5fa"),
}

# ─── Config page ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Classification de déchets — IA",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, .stApp { font-family: 'Inter', sans-serif !important; }

    /* ── FOND GLOBAL — maillage géométrique subtil ── */
    .stApp {
        background-color: #0d1117;
        background-image:
            linear-gradient(rgba(99,102,241,.06) 1px, transparent 1px),
            linear-gradient(90deg, rgba(99,102,241,.06) 1px, transparent 1px);
        background-size: 40px 40px;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2.5rem !important; max-width: 1080px !important; }

    /* ── HERO ── */
    .hero {
        position: relative;
        overflow: hidden;
        text-align: center;
        padding: 4rem 2rem 3rem;
        border-radius: 28px;
        margin-bottom: 2.5rem;
        background: linear-gradient(135deg, #0f172a 0%, #1a1f3c 40%, #0f2d4a 100%);
        border: 1px solid rgba(99,102,241,.25);
        box-shadow: 0 0 80px rgba(99,102,241,.12), 0 32px 64px rgba(0,0,0,.4);
    }
    /* orbes lumineux en fond */
    .hero::before {
        content:'';
        position:absolute; top:-60px; right:-60px;
        width:340px; height:340px; border-radius:50%;
        background: radial-gradient(circle, rgba(52,211,153,.18) 0%, transparent 70%);
        pointer-events:none;
    }
    .hero::after {
        content:'';
        position:absolute; bottom:-80px; left:-40px;
        width:300px; height:300px; border-radius:50%;
        background: radial-gradient(circle, rgba(99,102,241,.2) 0%, transparent 70%);
        pointer-events:none;
    }
    .hero h1 {
        position: relative; z-index:1;
        font-size: 3.2rem;
        font-weight: 900;
        color: #f8fafc;
        margin: 0 0 1rem;
        letter-spacing: -.03em;
        line-height: 1.1;
    }
    .hero h1 span { color: #34d399; }
    .hero p {
        position: relative; z-index:1;
        color: #94a3b8;
        font-size: 1rem;
        margin: 0;
        line-height: 1.7;
    }

    /* ── SECTION LABEL ── */
    .section-label {
        font-size: .7rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .14em;
        color: #4b5563;
        margin-bottom: .75rem;
    }

    /* ── CARDS MODÈLES ── */
    .model-card {
        position: relative;
        overflow: hidden;
        background: #161b2e;
        border-radius: 22px;
        padding: 2.5rem 2rem 2.2rem;
        text-align: center;
        border: 1px solid rgba(99,102,241,.2);
        height: 100%;
        transition: transform .3s ease, box-shadow .3s ease, border-color .3s ease;
        box-shadow: 0 4px 24px rgba(0,0,0,.3);
    }
    .model-card::before {
        content:'';
        position:absolute; top:0; left:0; right:0; height:2px;
        border-radius:22px 22px 0 0;
    }
    .mc-lin::before  { background: linear-gradient(90deg, #6366f1, #818cf8); }
    .mc-pmc::before  { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
    .mc-rbf::before  { background: linear-gradient(90deg, #14b8a6, #2dd4bf); }
    .model-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 60px rgba(0,0,0,.5);
        border-color: rgba(99,102,241,.5);
    }
    .mc-icon {
        width: 76px; height: 76px;
        border-radius: 20px;
        display: flex; align-items: center; justify-content: center;
        font-size: 2.4rem;
        margin: 0 auto 1.4rem;
    }
    .mc-lin .mc-icon { background: linear-gradient(135deg, #1e2a5e, #2d3a8c); box-shadow: 0 8px 24px rgba(99,102,241,.25); }
    .mc-pmc .mc-icon { background: linear-gradient(135deg, #2d1a5e, #4c1d95); box-shadow: 0 8px 24px rgba(139,92,246,.25); }
    .mc-rbf .mc-icon { background: linear-gradient(135deg, #0f3d3a, #115e59); box-shadow: 0 8px 24px rgba(20,184,166,.25); }
    .model-card h2 {
        margin: 0 0 .7rem;
        font-size: 1.3rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -.01em;
    }
    .model-card p { color: #94a3b8; font-size: .88rem; line-height: 1.7; margin: 0; }
    .model-card code {
        background: rgba(99,102,241,.15);
        color: #a5b4fc;
        padding: .15rem .5rem;
        border-radius: 6px;
        font-size: .83rem;
        border: 1px solid rgba(99,102,241,.25);
    }

    /* ── STAT CARDS DATASET ── */
    .stat-row { display:flex; gap:.9rem; margin-top:1rem; }
    .stat-card {
        flex: 1;
        background: #161b2e;
        border-radius: 16px;
        padding: 1.3rem 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,.07);
        box-shadow: 0 2px 12px rgba(0,0,0,.25);
    }
    .stat-card .sc-val { font-size: 2rem; font-weight: 900; line-height: 1; }
    .stat-card .sc-lbl { font-size: .8rem; color: #64748b; margin-top: .35rem; font-weight: 500; }

    /* ── PREDICT HEADER ── */
    .predict-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1.1rem 1.5rem;
        background: #161b2e;
        border-radius: 18px;
        border: 1px solid rgba(99,102,241,.2);
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,.2);
    }
    .ph-badge {
        border-radius: 14px;
        width: 52px; height: 52px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.6rem; flex-shrink: 0;
    }
    .ph-lin { background: linear-gradient(135deg, #1e2a5e, #2d3a8c); }
    .ph-pmc { background: linear-gradient(135deg, #2d1a5e, #4c1d95); }
    .ph-rbf { background: linear-gradient(135deg, #0f3d3a, #115e59); }
    .predict-header h2 { margin: 0; font-size: 1.2rem; font-weight: 700; color: #f1f5f9; }
    .predict-header p  { margin: 0; font-size: .82rem; color: #64748b; }

    /* ── RÉSULTAT ── */
    .result-hero {
        border-radius: 22px;
        padding: 2.5rem 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,.08);
        box-shadow: 0 8px 40px rgba(0,0,0,.3);
    }
    .result-hero::after {
        content:'';
        position:absolute; bottom:-50px; right:-50px;
        width:180px; height:180px; border-radius:50%;
        opacity:.12;
    }
    .rh-emoji { font-size: 5rem; line-height: 1; margin-bottom: .6rem; filter: drop-shadow(0 4px 12px rgba(0,0,0,.3)); }
    .result-hero h1 {
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: -.03em;
        margin: 0 0 .4rem;
    }
    .result-hero p { font-size: .92rem; color: #94a3b8; margin: 0 0 1.2rem; }
    .result-badge {
        display: inline-block;
        border-radius: 99px;
        padding: .45rem 1.4rem;
        font-weight: 700;
        font-size: .88rem;
        letter-spacing: .04em;
    }

    /* ── PROBA BARS ── */
    .proba-section { margin-top: 1.4rem; }
    .proba-row {
        display: flex;
        align-items: center;
        gap: .75rem;
        margin-bottom: .7rem;
    }
    .proba-label {
        min-width: 115px;
        font-size: .85rem;
        font-weight: 600;
        color: #cbd5e1;
    }
    .proba-label.active { color: #f1f5f9; }
    .proba-bar-bg {
        flex: 1;
        height: 8px;
        background: rgba(255,255,255,.07);
        border-radius: 99px;
        overflow: hidden;
    }
    .proba-bar-fill { height: 100%; border-radius: 99px; }
    .proba-pct {
        min-width: 46px;
        text-align: right;
        font-size: .83rem;
        font-weight: 700;
        color: #94a3b8;
    }
    .proba-pct.active { color: #f1f5f9; }

    /* ── FEATURE CHIPS ── */
    .feat-chip {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        background: rgba(255,255,255,.05);
        border: 1px solid rgba(255,255,255,.1);
        border-radius: 10px;
        padding: .45rem 1rem;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: .84rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: .25rem .15rem;
    }
    .feat-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink:0; }

    /* ── TIP BOX ── */
    .tip-box {
        display: flex;
        align-items: flex-start;
        gap: .9rem;
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        margin-top: 1.5rem;
        border: 1px solid rgba(250,204,21,.2);
        background: rgba(250,204,21,.05);
        color: #fef9c3;
        font-size: .9rem;
        line-height: 1.6;
    }
    .tip-box b { color: #fde047; }
    .tip-icon { font-size: 1.3rem; flex-shrink:0; margin-top:.05rem; }

    /* ── UPLOAD PLACEHOLDER ── */
    .upload-placeholder {
        border: 2px dashed rgba(99,102,241,.3);
        border-radius: 22px;
        padding: 4.5rem 2rem;
        text-align: center;
        background: rgba(99,102,241,.04);
        margin-top: 1rem;
    }
    .up-icon { font-size: 4rem; margin-bottom: .8rem; }
    .upload-placeholder p { margin: 0; color: #64748b; }
    .upload-placeholder .up-title { font-size: 1.05rem; font-weight: 600; color: #94a3b8; margin-bottom: .3rem; }

    /* ── DIVIDER ── */
    .st-emotion-cache-1qd0mnu, hr { border-color: rgba(255,255,255,.06) !important; }

    /* Bouton Streamlit override */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        letter-spacing: .01em !important;
        transition: transform .2s, box-shadow .2s !important;
    }
    .stButton > button:hover { transform: translateY(-2px) !important; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #161b2e;
        border-radius: 16px;
        padding: .5rem;
        border: 1px solid rgba(99,102,241,.2);
    }
</style>
""", unsafe_allow_html=True)

# ─── Session state ─────────────────────────────────────────────────────────────
if "page"  not in st.session_state: st.session_state.page  = "home"
if "model" not in st.session_state: st.session_state.model = None


# ══════════════════════════════════════════════════════════════════════════════
# ML
# ══════════════════════════════════════════════════════════════════════════════

def _grille_moyennes(canal: np.ndarray, g: int = 3) -> list:
    """Moyenne de chaque case d'une grille g×g (ordre ligne par ligne)."""
    H, W = canal.shape
    ys = np.linspace(0, H, g + 1).astype(int)
    xs = np.linspace(0, W, g + 1).astype(int)
    return [float(canal[ys[i]:ys[i+1], xs[j]:xs[j+1]].mean())
            for i in range(g) for j in range(g)]


def preprocess(img: Image.Image) -> np.ndarray:
    """Extrait 32 features, identiques à main.py (contenu seul, sans padding) :
        [0:3]   moyennes globales R, G, B           → entrée du PMC
        [3:30]  grille 3×3 des moyennes R, G, B (27) → répartition spatiale
        [30:32] moyenne / écart-type du gradient     → texture / bords
    Le modèle linéaire utilise features[3:32] (29) ; le PMC utilise features[:3].
    """
    img = img.convert("RGB")
    img.thumbnail((224, 224), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0          # [0–1]
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    grille = _grille_moyennes(r, 3) + _grille_moyennes(g, 3) + _grille_moyennes(b, 3)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy)
    return np.array([float(r.mean()), float(g.mean()), float(b.mean())]
                    + grille
                    + [float(grad.mean()), float(grad.std())], dtype=np.float64)


@st.cache_resource(show_spinner=False)
def load_linear(_dataset_version: float = 0.0):
    if not os.path.exists(SO_PATH):
        r = subprocess.run(
            ["gcc", "-shared", "-fPIC", "-DNO_MAIN", "-O2", "-lm", "-o", SO_PATH, C_SRC],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return None
    df = pd.read_csv(CSV_PATH)
    # 29 features linéaires = grille 3×3 RGB (27, en [0–255] → /255) + gradient (2, déjà [0–1]).
    grille_cols = ([f"r_{i}" for i in range(9)]
                   + [f"g_{i}" for i in range(9)]
                   + [f"b_{i}" for i in range(9)])
    Xgrille = df[grille_cols].values.astype(np.float64) / 255.0
    Xgrad   = df[["grad_mean", "grad_std"]].values.astype(np.float64)
    X = np.hstack([Xgrille, Xgrad])                  # [n, 29] (= preprocess[3:32])
    y = df["label"].values.astype(int)
    Y = np.eye(NB_CLASSES, dtype=np.float64)[y]
    # Ajout du biais x₀=1 (cours : X = [x₀, x₁, …, xₙ] avec x₀=1)
    n   = len(X)
    X_b = np.hstack([np.ones((n, 1)), X])            # [n, 30]
    W   = np.linalg.pinv(X_b) @ Y  # [30, 3]
    return W


@st.cache_resource(show_spinner=False)
def load_pmc(_dataset_version: float = 0.0, epochs: int = 300, lr: float = 0.05):
    df  = pd.read_csv(CSV_PATH)
    # Même normalisation /255 que pour preprocess()
    X   = df[["r_mean","g_mean","b_mean"]].values.astype(np.float64) / 255.0
    y   = df["label"].values.astype(int)
    Y   = np.eye(NB_CLASSES, dtype=np.float64)[y]
    pmc = PMC(layer_sizes=[NB_FEATURES, 32, 16, NB_CLASSES], learning_rate=lr, is_regression=False)
    for _ in range(epochs):
        pmc.train(X, Y)
    return pmc


@st.cache_resource(show_spinner=False)
def load_rbf(_dataset_version: float = 0.0):
    if not RBF_AVAILABLE:
        return None
    df = pd.read_csv(CSV_PATH)
    # Mêmes 29 features que le modèle linéaire : grille 3×3 RGB (27, /255) + gradient (2).
    grille_cols = ([f"r_{i}" for i in range(9)]
                   + [f"g_{i}" for i in range(9)]
                   + [f"b_{i}" for i in range(9)])
    Xgrille = df[grille_cols].values.astype(np.float64) / 255.0
    Xgrad   = df[["grad_mean", "grad_std"]].values.astype(np.float64)
    X = np.hstack([Xgrille, Xgrad])                  # [n, 29] (= preprocess[3:32])
    y = df["label"].values.astype(int)
    set_seed(42)
    m = RBF(nb_centres=100, gamma=0.5)               # K=100 centres, noyau γ=0.5 (calibré)
    m.train(X, one_hot(y, NB_CLASSES))
    return m


def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════

def page_home():
    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>Classification de <span>Déchets</span></h1>
        <p>Sélectionnez un modèle, importez une photo et obtenez la catégorie de votre déchet.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Cartes modèles ────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown("""
        <div class="model-card mc-lin">
            <div class="mc-icon"><span style="font-size:1.4rem; font-weight:900; letter-spacing:-.05em; color:#a5b4fc">LIN</span></div>
            <h2>Modèle Linéaire</h2>
            <p>
                Algorithme <b style="color:#a5b4fc">pseudo-inverse</b> &mdash; solution analytique directe.<br><br>
                Calcul en une seule étape :<br>
                <code>W&nbsp;=&nbsp;(XᵀX)⁻¹&nbsp;Xᵀ&nbsp;Y</code>
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
        if st.button("Utiliser le Modèle Linéaire", key="btn_lin", use_container_width=True, type="primary"):
            st.session_state.model = "linear"
            st.session_state.page  = "predict"
            st.rerun()

    with c2:
        st.markdown("""
        <div class="model-card mc-pmc">
            <div class="mc-icon"><span style="font-size:1.4rem; font-weight:900; letter-spacing:-.05em; color:#c4b5fd">PMC</span></div>
            <h2>Réseau de Neurones &mdash; PMC</h2>
            <p>
                <b style="color:#c4b5fd">Perceptron Multi-Couches</b> entraîné par rétropropagation.<br><br>
                Architecture :<br>
                <code>3 &rarr; 32 &rarr; 16 &rarr; 3</code>
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
        if st.button("Utiliser le PMC", key="btn_pmc", use_container_width=True, type="primary"):
            st.session_state.model = "pmc"
            st.session_state.page  = "predict"
            st.rerun()

    with c3:
        st.markdown("""
        <div class="model-card mc-rbf">
            <div class="mc-icon"><span style="font-size:1.4rem; font-weight:900; letter-spacing:-.05em; color:#5eead4">RBF</span></div>
            <h2>Réseau à Fonctions Radiales</h2>
            <p>
                <b style="color:#5eead4">k-means + noyau gaussien</b> &mdash; centres puis pseudo-inverse.<br><br>
                Noyau de chaque centre :<br>
                <code>φ(x)&nbsp;=&nbsp;exp(−γ&middot;‖x−c‖²)</code>
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
        if st.button("Utiliser le RBF", key="btn_rbf", use_container_width=True, type="primary"):
            st.session_state.model = "rbf"
            st.session_state.page  = "predict"
            st.rerun()

    # ── Stats dataset ─────────────────────────────────────────────────────────
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Dataset d’entraînement</p>', unsafe_allow_html=True)

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        counts = df["label"].value_counts().sort_index()
        total  = len(df)

        cat_colors = {0: "#34d399", 1: "#f87171", 2: "#60a5fa"}
        items = ""
        for lbl, cnt in counts.items():
            nm, *_ = CATEGORIES[lbl]
            clr = cat_colors[lbl]
            pct = cnt / total * 100
            items += (f'<div class="stat-card">'
                      f'<div class="sc-val" style="color:{clr}">{cnt}</div>'
                      f'<div class="sc-lbl">{nm}<br>'
                      f'<span style="color:#4b5563;font-size:.74rem">{pct:.0f}&thinsp;%</span></div></div>')
        total_card = (f'<div class="stat-card">'
                     f'<div class="sc-val" style="color:#a5b4fc">{total}</div>'
                     f'<div class="sc-lbl">Échantillons<br><span style="color:#4b5563;font-size:.74rem">total</span></div></div>')
        st.markdown(f'<div class="stat-row">{total_card}{items}</div>', unsafe_allow_html=True)
    else:
        st.warning("dataset.csv introuvable — lancez `python main.py` pour générer le dataset.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE PRÉDICTION
# ══════════════════════════════════════════════════════════════════════════════

def page_predict():
    model_key   = st.session_state.model
    model_label = {
        "linear": "Modèle Linéaire",
        "pmc":    "PMC — Perceptron Multi-Couches",
        "rbf":    "Réseau à Fonctions Radiales — RBF",
    }[model_key]

    # ── Header ────────────────────────────────────────────────────────────────
    icon_text = {"linear": "LIN", "pmc": "PMC", "rbf": "RBF"}[model_key]
    ph_class  = {"linear": "ph-lin", "pmc": "ph-pmc", "rbf": "ph-rbf"}[model_key]
    subtitle  = {
        "linear": "Solution analytique · pseudo-inverse · implémenté en C",
        "pmc":    "Rétropropagation du gradient · 3→32→16→3 · implémenté en C",
        "rbf":    "k-means + noyau gaussien · pseudo-inverse · implémenté en C",
    }[model_key]

    hc, bc = st.columns([8, 2])
    with hc:
        st.markdown(f"""
        <div class="predict-header">
            <div class="ph-badge {ph_class}"><span style="font-weight:900; font-size:.95rem; letter-spacing:-.03em">{icon_text}</span></div>
            <div>
                <h2>{model_label}</h2>
                <p>{subtitle}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with bc:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        if st.button("Changer de modèle", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()

    # Chargement modèle
    dataset_version = os.path.getmtime(CSV_PATH) if os.path.exists(CSV_PATH) else 0.0

    with st.spinner("Chargement du modèle en cours…"):
        if model_key == "linear":
            W = load_linear(dataset_version)
            ok = W is not None
        elif model_key == "rbf":
            rbf = load_rbf(dataset_version)
            ok  = rbf is not None
        else:
            pmc = load_pmc(dataset_version)
            ok  = pmc is not None

    if not ok:
        st.error("Impossible de charger le modèle. Vérifiez que dataset.csv et les sources C sont présents.")
        return

    # Upload
    uploaded = st.file_uploader(
        "Sélectionnez une image (JPG, PNG, WEBP, BMP)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=False,
    )

    if not uploaded:
        st.markdown("""
        <div class="upload-placeholder">
            <div style="font-size:2.5rem; font-weight:800; color:#334155; letter-spacing:.05em; margin-bottom:.8rem">IMG</div>
            <p style="font-size:1.05rem; font-weight:600; color:#94a3b8;">Aucune image sélectionnée</p>
            <p style="font-size:.88rem; margin-top:.4rem; color:#4b5563;">JPG &middot; PNG &middot; WEBP &middot; BMP acceptés</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Prédiction
    img      = Image.open(uploaded).convert("RGB")
    features = preprocess(img)

    if model_key == "linear":
        # Modèle linéaire : 29 features (grille 3×3 + gradient) + biais x0=1
        features_lin = features[3:32]
        features_b   = np.hstack([[1.0], features_lin])   # [30]
        scores = (features_b @ W) + LINEAR_CLASS_BIAS
    elif model_key == "rbf":
        scores = rbf.forward(features[3:32])              # RBF : 29 features (grille 3×3 + gradient)
    else:
        scores = pmc.forward(features[:NB_FEATURES])      # PMC : 3 features couleur (R, G, B)

    proba = softmax(np.array(scores, dtype=np.float64))
    pred  = int(np.argmax(scores))
    conf  = float(proba.max())
    # Rejet « je ne sais pas » si la confiance est sous le seuil du modèle
    rejected = conf < REJECT_THRESHOLD.get(model_key, 0.0)

    name, desc, color_fg, color_bar = CATEGORIES[pred]

    # ── Affichage ─────────────────────────────────────────────────────────────
    img_col, res_col = st.columns([4, 6], gap="large")

    with img_col:
        st.image(img, caption=uploaded.name, use_container_width=True)
        st.markdown('<p class="section-label" style="margin:.9rem 0 .4rem">Caractéristiques extraites</p>', unsafe_allow_html=True)
        chips = f"""
        <span class="feat-chip"><span class="feat-dot" style="background:#ef4444"></span>R&nbsp;=&nbsp;{features[0]:.3f}</span>
        <span class="feat-chip"><span class="feat-dot" style="background:#22c55e"></span>G&nbsp;=&nbsp;{features[1]:.3f}</span>
        <span class="feat-chip"><span class="feat-dot" style="background:#3b82f6"></span>B&nbsp;=&nbsp;{features[2]:.3f}</span>"""
        if model_key in ("linear", "rbf"):
            chips += f"""
        <span class="feat-chip"><span class="feat-dot" style="background:#f59e0b"></span>∇̄&nbsp;=&nbsp;{features[30]:.3f}</span>
        <span class="feat-chip"><span class="feat-dot" style="background:#a855f7"></span>σ∇&nbsp;=&nbsp;{features[31]:.3f}</span>
        <span class="feat-chip"><span class="feat-dot" style="background:#64748b"></span>grille&nbsp;3×3&nbsp;(27)</span>"""
            st.markdown(chips, unsafe_allow_html=True)

    with res_col:
        bg_dark    = {0: "rgba(52,211,153,.08)",  1: "rgba(248,113,113,.08)",  2: "rgba(96,165,250,.08)"}
        border_col = {0: "rgba(52,211,153,.25)",  1: "rgba(248,113,113,.25)",  2: "rgba(96,165,250,.25)"}
        if rejected:
            emoji, titre, sous = random.choice(MESSAGES_JE_SAIS_PAS)
            st.markdown(f"""
            <div class="result-hero" style="background:rgba(148,163,184,.06); border-color:rgba(148,163,184,.25);">
                <div class="rh-emoji">{emoji}</div>
                <h1 style="color:#cbd5e1;">{titre}</h1>
                <p>{sous}</p>
                <span class="result-badge"
                      style="background:rgba(148,163,184,.12); color:#cbd5e1;
                             border:1px solid rgba(148,163,184,.3);">
                    Confiance&nbsp;{conf*100:.1f}&thinsp;%&ensp;&middot;&ensp;sous le seuil de&nbsp;{REJECT_THRESHOLD.get(model_key, 0.0)*100:.0f}&thinsp;%
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-hero" style="background:{bg_dark[pred]}; border-color:{border_col[pred]};">
                <div style="width:16px; height:16px; border-radius:50%; background:{color_bar};
                            margin:0 auto .8rem; box-shadow:0 0 20px {color_bar}60;"></div>
                <h1 style="color:{color_bar};">{name}</h1>
                <p>{desc}</p>
                <span class="result-badge"
                      style="background:{color_bar}18; color:{color_bar};
                             border:1px solid {color_bar}40;">
                    Classe&nbsp;{pred}&ensp;&middot;&ensp;Confiance&nbsp;{proba[pred]*100:.1f}&thinsp;%
                </span>
            </div>
            """, unsafe_allow_html=True)

        # Barres de probabilité
        st.markdown('<p class="section-label" style="margin:1.4rem 0 .7rem">Probabilités par catégorie</p>', unsafe_allow_html=True)
        bar_colors = {0: "#34d399", 1: "#f87171", 2: "#60a5fa"}
        for i, p in enumerate(proba):
            nm_i, *_ = CATEGORIES[i]
            pct  = p * 100
            bc_i = bar_colors[i]
            active = "active" if i == pred else ""
            st.markdown(f"""
            <div class="proba-row">
                <div class="proba-label {active}">
                    <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                                 background:{bc_i};margin-right:.4rem;"></span>{nm_i}</div>
                <div class="proba-bar-bg">
                    <div class="proba-bar-fill" style="width:{pct:.1f}%; background:{bc_i};"></div>
                </div>
                <div class="proba-pct {active}">{pct:.1f}&thinsp;%</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Conseil de tri (seulement si le modèle s'est prononcé) ───────────────────
    if not rejected:
        tips = {
            0: ("Bac compost (marron)",    "Déposez dans le bac marron ou votre composteur. Pas de plastique ni d'emballage."),
            1: ("Déchetterie obligatoire",  "Ne jetez jamais dans la poubelle ordinaire. Rapportez en déchetterie ou point de collecte spécialisé."),
            2: ("Bac de tri (jaune)",       "Videz et rincez l'emballage avant de le déposer dans le bac jaune."),
        }
        tip_title, tip_text = tips[pred]
        tip_color = {0: "#34d399", 1: "#f87171", 2: "#60a5fa"}[pred]
        st.markdown(f"""
        <div class="tip-box" style="border-color:{tip_color}30; background:{tip_color}06;">
            <div style="width:3px; border-radius:2px; background:{tip_color}; flex-shrink:0; align-self:stretch;"></div>
            <div style="color:#cbd5e1;"><b style="color:{tip_color}">{tip_title}</b> &mdash; {tip_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    if st.button("Analyser une autre image", use_container_width=True):
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Routeur
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "home":
    page_home()
else:
    page_predict()
