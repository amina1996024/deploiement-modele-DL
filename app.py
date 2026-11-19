import joblib
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

# Configuration en mode large
st.set_page_config(
    page_title="Plateforme IA & Deep Learning",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Palette claire, lumineuse et moderne : fond pastel/lumineux avec touches corail, violet et bleu ciel
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Fond clair aéré avec dégradés doux colorés */
    .stApp {
        background:
            radial-gradient(circle at 8% 12%, rgba(251, 113, 133, 0.16) 0%, transparent 38%),
            radial-gradient(circle at 92% 14%, rgba(56, 189, 248, 0.20) 0%, transparent 40%),
            radial-gradient(circle at 50% 95%, rgba(167, 139, 250, 0.16) 0%, transparent 45%),
            linear-gradient(180deg, #F8FAFF 0%, #F1F5FF 55%, #FAF5FF 100%);
        color: #1E293B;
    }

    /* Barre latérale claire et bien contrastée pour une lecture parfaite */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%) !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    [data-testid="stSidebar"] * {
        color: #1E293B;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] .stCaption {
        color: #475569 !important;
    }

    /* Bandeau principal lumineux et coloré */
    .hero-banner {
        background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 48%, #EC4899 100%);
        padding: 2.1rem 2.4rem;
        border-radius: 22px;
        margin-bottom: 1.8rem;
        box-shadow: 0 16px 35px -12px rgba(79, 70, 229, 0.35);
        color: #FFFFFF;
    }
    .hero-kicker {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #FFFFFF;
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.35);
        padding: 0.28rem 0.75rem;
        border-radius: 999px;
        margin-bottom: 0.8rem;
    }
    .hero-banner h1 {
        color: #FFFFFF !important;
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        margin: 0 0 0.45rem 0 !important;
        padding: 0 !important;
    }
    .hero-banner p {
        color: #F8FAFC !important;
        font-size: 1.04rem;
        margin: 0;
        line-height: 1.5;
    }

    /* Cartes blanches propres avec ombre douce */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 18px !important;
        padding: 1.15rem !important;
        box-shadow: 0 8px 24px -10px rgba(15, 23, 42, 0.08);
    }

    /* Carte de résultat claire, fraîche et très lisible */
    .result-box {
        background: linear-gradient(135deg, #ECFDF5 0%, #EFF6FF 100%);
        border: 1px solid #A7F3D0;
        border-left: 6px solid #10B981;
        border-radius: 18px;
        padding: 1.5rem 2rem;
        margin: 1rem 0 1.4rem 0;
        box-shadow: 0 10px 25px -10px rgba(16, 185, 129, 0.2);
    }
    .result-label {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #047857;
        margin-bottom: 0.3rem;
    }
    .result-main {
        font-size: 2rem;
        font-weight: 800;
        color: #0F172A;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.8rem;
    }
    .confidence-pill {
        background: #D1FAE5;
        color: #065F46;
        border: 1px solid #6EE7B7;
        padding: 0.28rem 0.85rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 700;
    }
    .unit-text {
        font-size: 1.02rem;
        font-weight: 600;
        color: #475569;
    }

    /* Badges colorés dans la barre latérale */
    .chip {
        display: inline-block;
        padding: 0.28rem 0.72rem;
        border-radius: 8px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-right: 0.4rem;
        margin-bottom: 0.5rem;
        background: #EEF2FF !important;
        color: #4338CA !important;
        border: 1px solid #C7D2FE;
    }
    .chip-pink {
        background: #FDF2F8 !important;
        color: #BE185D !important;
        border: 1px solid #FBCFE8;
    }

    /* Bouton principal vif et moderne */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4F46E5 0%, #EC4899 100%);
        color: #FFFFFF !important;
        font-weight: 700;
        font-size: 1.05rem;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.8rem;
        width: 100%;
        box-shadow: 0 10px 22px -6px rgba(79, 70, 229, 0.4);
        transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 26px -6px rgba(236, 72, 153, 0.45);
        filter: brightness(1.04);
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# REGISTRE DES 5 TP
# ---------------------------------------------------------------
MODELS = {
    "DL1_BANK": {
        "label": "🏦 Prédiction Souscription Bancaire (DL1)",
        "desc": "Évalue la probabilité qu'un client souscrive à un dépôt à terme selon son profil.",
        "input": "tabulaire", "task": "binaire", "path": "models/DL1_BANK/DNN_bank_model.h5",
        "encoders": "models/DL1_BANK/encoders.joblib", "scaler": "models/DL1_BANK/scaler.joblib", "unique": "models/DL1_BANK/uniques.joblib",
        "classes": ["Souscrire", "Pas souscrire"],
        "columns": ["age", "job", "marital", "education", "default",
                    "housing", "loan", "contact", "month", "day_of_week",
                    "duration", "campaign", "pdays", "previous", "poutcome"],
        "categorical": ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"],
    },
    "DL2_STUDENT": {
        "label": "🎓 Performance Académique Étudiant (DL2)",
        "desc": "Estime l'indice de performance d'un étudiant à partir de ses habitudes d'étude.",
        "input": "tabulaire", "task": "regression", "path": "models/DL2_STUDENT/DNN_model_student.h5",
        "encoders": "models/DL2_STUDENT/encoder.joblib", "scaler": "models/DL2_STUDENT/scaler.joblib",
        "target_scaler": None, "unit": "(Performance Index)",
        "columns": ["Hours Studied", "Previous Scores", "Extracurricular Activities",
                    "Sleep Hours", "Sample Question Papers Practiced"],
        "categorical": ["Extracurricular Activities"],
    },
    "DL3_CIFAR10": {
        "label": "🖼️ Classification d'Images CIFAR-10 (DL3)",
        "desc": "Reconnaissance visuelle d'objets et d'animaux parmi 10 catégories distinctes.",
        "input": "image", "task": "multiclasse", "path": "models/DL3_CIFAR10/cifar10_model.h5",
        "size": (32, 32), "mode": "RGB",
        "classes": ["avion", "automobile", "oiseau", "chat", "cerf", "chien", "grenouille", "cheval", "bateau", "camion"],
    },
    "DL5_TUBERCULOSIS": {
        "label": "🫁 Détection de la Tuberculose (DL5)",
        "desc": "Analyse de radiographies thoraciques pour détecter des signes de tuberculose.",
        "input": "image", "task": "binaire", "path": "models/DL5_TUBERCULOSIS/model_tuberculosis.h5",
        "size": (150, 150), "mode": "RGB",
        "classes": ["Normal", "Tuberculosis"],
    },
    "PROJET_PNEUMONIA": {
        "label": "🩺 Détection de la Pneumonie (Projet)",
        "desc": "Analyse de radiographies pulmonaires pour identifier la présence d'une pneumonie.",
        "input": "image", "task": "binaire", "path": "models/PROJET_PNEUMONIA/model_pneumonia.h5",
        "size": (150, 150), "mode": "RGB",
        "classes": ["Normal", "Pneumonia"],
    },
}


@st.cache_resource(show_spinner="Chargement du modèle...")
def load_model(name):
    return keras.models.load_model(MODELS[name]["path"], compile=False)


@st.cache_resource
def load_file(path):
    return joblib.load(path)


def show_result(cfg, out):
    out = np.atleast_1d(out).astype("float64")

    if cfg["task"] == "regression":
        val = out
        if cfg.get("target_scaler"):
            val = load_file(cfg["target_scaler"]).inverse_transform(val.reshape(1, -1))[0]
        st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Valeur prédite</div>
                <div class="result-main">
                    {val[0]:.2f}
                    <span class="unit-text">{cfg.get('unit', '')}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        return

    if len(out) == 1:
        out = np.array([1 - out[0], out[0]])
    classes = cfg["classes"]
    best = int(np.argmax(out))

    st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Résultat de l'inférence</div>
            <div class="result-main">
                {classes[best].capitalize()}
                <span class="confidence-pill">{out[best]:.1%}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.bar_chart({c.capitalize(): float(p) for c, p in zip(classes, out)}, color="#4F46E5")


def tabular_input(cfg):
    enc = load_file(cfg["encoders"])
    if isinstance(enc, (list, tuple)):
        enc = dict(zip(cfg["categorical"], enc))
    elif not isinstance(enc, dict):
        enc = {cfg["categorical"][0]: enc}
    uniq = load_file(cfg["unique"]) if cfg.get("unique") else {}
    if not isinstance(uniq, dict):
        uniq = {}
    cols = cfg["columns"]

    st.markdown("### ⚙️ Paramètres d'entrée")
    grid = st.columns(3)
    row = {}
    for i, col in enumerate(cols):
        with grid[i % 3]:
            label_txt = col.replace("_", " ").capitalize()
            if col in enc:
                choices = list(uniq.get(col, enc[col].classes_))
                row[col] = st.selectbox(label_txt, choices, key=f"tab_{col}")
            else:
                row[col] = st.number_input(label_txt, value=0.0, key=f"tab_{col}")

    for col, le in enc.items():
        if col in row:
            row[col] = le.transform([row[col]])[0]

    x = np.array([[row[c] for c in cols]], dtype="float64")
    sc = load_file(cfg["scaler"])
    if getattr(sc, "n_features_in_", len(cols)) == len(cols):
        x = sc.transform(x)
    else:
        idx = [i for i, c in enumerate(cols) if c not in enc]
        x[:, idx] = sc.transform(x[:, idx])
    return x.astype("float32")


# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## ✦ Deep Learning Hub")
    st.caption("Sélectionnez un modèle déployé")
    choice = st.selectbox(
        "Choisissez un modèle",
        list(MODELS.keys()),
        format_func=lambda k: MODELS[k]["label"],
        label_visibility="collapsed"
    )
    cfg = MODELS[choice]

    st.markdown("---")
    st.markdown(
        f'<span class="chip">{cfg["input"].upper()}</span>'
        f'<span class="chip chip-pink">{cfg["task"].upper()}</span>',
        unsafe_allow_html=True
    )
    st.write(cfg["desc"])
    if cfg["input"] == "image":
        st.caption(f"Format cible : **{cfg['size'][0]}×{cfg['size'][1]}** · **{cfg['mode']}**")


# --- BANNIÈRE PRINCIPALE ---
st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-kicker">Déploiement de modèles · {choice}</div>
        <h1>{cfg['label']}</h1>
        <p>{cfg['desc']}</p>
    </div>
""", unsafe_allow_html=True)

x = None

if cfg["input"] == "image":
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("### 📤 Image d'entrée")
            file = st.file_uploader("Importez une image", type=["jpg", "jpeg", "png"])
            if "TUBERCULOSIS" in choice or "PNEUMONIA" in choice:
                st.caption("⚠️ Outil pédagogique : ne remplace pas un diagnostic médical.")
            if file:
                img = Image.open(file)
                arr = np.asarray(img.convert(cfg["mode"]).resize(cfg["size"]), dtype="float32") / 255.0
                if arr.ndim == 2:
                    arr = arr[..., None]
                x = arr[None, ...]
                predict_btn = st.button("Prédire")
            else:
                predict_btn = False

    with right:
        with st.container(border=True):
            st.markdown("### 🖼️ Aperçu")
            if file:
                st.image(img, caption=file.name, use_container_width=True)
            else:
                st.info("Chargez une image JPG, JPEG ou PNG pour afficher l'aperçu.")

    if x is not None and predict_btn:
        show_result(cfg, load_model(choice).predict(x, verbose=0)[0])

else:
    with st.container(border=True):
        x = tabular_input(cfg)
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        predict_btn = st.button("Prédire")
    if x is not None and predict_btn:
        show_result(cfg, load_model(choice).predict(x, verbose=0)[0])