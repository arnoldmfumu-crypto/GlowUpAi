import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

import base64

st.set_page_config(page_title="Skinmatch", layout="centered")

ACNE_API_URL = os.getenv("ACNE_API_URL", "http://acne:8000/predict")
OILY_API_URL = os.getenv("OILY_API_URL", "http://oily:8000/predict")
PRODUCT_API_URL = os.getenv("PRODUCT_API_URL", "http://product:8000/recommend")

PRODUCT_TYPE_OPTIONS = {
    "Peu importe": None,
    "Crème / Hydratant": "moisturizer",
    "Sérum": "serum",
    "Nettoyant": "cleanser",
    "Tonique": "toner",
    "Masque": "mask",
    "Huile": "oil",
    "Exfoliant": "exfoliant",
    "Contour des yeux": "eye_care",
}

PRICE_BAND_OPTIONS = {
    "Peu importe": "any",
    "Budget (< 15€)": "budget",
    "Milieu de gamme (15–50€)": "mid",
    "Premium (> 50€)": "premium",
}

## Style

# Pour intégrer l'image en fond transparent dans votre header HTML
# (en attendant d'avoir un hébergement pour l'image)
def get_image_as_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Téléchargez le PNG ci-dessous et placez-le dans le même dossier
# que votre script Streamlit sous le nom "logo_glowup.png"
# logo_base64 = get_image_as_base64("logo-glowup-ai.png")

# Injection de CSS personnalisé

st.markdown("""
    <style>
    /* 1. Importation des polices Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400..700;1,400..700&family=Montserrat:wght@300;400;500;600&display=swap');        
    
    /* On cible le conteneur spécifique du file uploader */
    [data-testid="stFileUploader"] {
        background-color: white;
        border-radius: 20px;
        padding: 40px;
        border: 1px solid #F0E6E1;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        max-width: 600px;
        margin: 0 auto 30px;
    }
    
    /* On stylise la zone de drop (le rectangle en pointillés) */
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #E5B1B6 !important;
        background-color: #FFF9F4 !important;
    }
            

    .custom-section {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #F0E6E1;
        margin-bottom: 25px;
    }

    /* Supprimer les marges par défaut de Streamlit en haut */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }

    .full-width-gradient {
        /* On force la largeur sur 100% de la fenêtre */
        width: 100vw;
        position: relative;
        left: 50%;
        right: 50%;
        margin-left: -50vw;
        margin-right: -50vw;
        text-align: center;
        
        /* Votre dégradé */
        background: linear-gradient(180deg, #FAD0C4 0%, #FFF9F4 100%);
        
        /* Espacement interne */
        padding: 60px 0 30px;
        text-align: center;
        margin-bottom: 40px;
    }

    .header-content {
        max-width: 800px; /* Pour que le texte ne soit pas trop étalé */
        margin: 0 auto;
        padding: 0 20px;
    }

    p {
        color: #3C2A21;
    }
    
    .st-emotion-cache-1rsqh2s p {
        color: #fff; !important        
    }

    /* 1. Correction du fond global */
    [data-testid="stAppViewContainer"] {
        background-color: #FFF9F4;
    }

    /* 2. Style pour le titre principal */
    .main-title {
        color: #3C2A21 !important;
        font-family: 'Lora', serif !important;
        font-weight: 600;
        font-size: 42px !important;
        text-align: center;
        font-weight: 500;
        margin-bottom: 0px;
    }
            
    /* 3. Style pour le sous-titre */
    .subtitle {
        color: #3C2A21 !important;
        font-family: 'serif';
        font-size: 24px !important;
        text-align: center;
        font-weight: 400;
        margin-top: -10px;
    }

    /* Correction de la zone d'upload */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #ffffff !important;
        border: 1px dashed #E5B1B6 !important;
    }

    /* Style du bouton */
    div.stButton > button {
        background-color: #E5B1B6 !important;
        color: #3C2A21 !important;
        border-radius: 25px !important;
        border: none !important;
        width: 100%;
    }
    
    .st-emotion-cache-kt79cc .st-emotion-cache-1x4hur2:first-child {
        color: #3C2A21;      
    }
    
    /* cases à cocher */
    [data-testid="stCheckbox"] p,
    [data-testid="stMarkdownContainer"] p {
        color: #3C2A21; !important;
        text-align: center !important;
    }
            
    /* Centre le texte des étiquettes de checkbox */
    .stCheckbox {
        display: flex;
        justify-content: center;
        margin: 0 auto;
        text-align: center;
    }
            
    .headings {
        font-family: "Lora", serif;        
    }
    
    /* Centre la checkbox "Tout me convient" spécifiquement */
    .centered-checkbox {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
            
    /* Type de produits */
    .st.selectbox {
        text-align: center;        
    }
    </style>
    """, unsafe_allow_html=True)

# Création d'une section "Header"
st.markdown("""
    <div class="full-width-gradient">
        <div class="header-content">
            <p style="letter-spacing: 2px; font-size: 12px; color: #3C2A21; opacity: 0.7;">✨ DIAGNOSTIC PEAU • IA</p>
            <h1 class="main-title">
                GlowUp AI
            </h1>
            <img src="data:image/png;base64,{logo_base64}" width="250" class="logo-img">
            <p style="color: #3C2A21; font-size: 18px; max-width: 600px; margin: 0 auto;">
                Un selfie, vos préférences, et nous vous recommandons le produit cosmétique fait pour votre peau.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Etapes 1 et 2
st.markdown('<p class="headings" style="text-align:center; letter-spacing:2px; color:#B8A39A;">01 — VOTRE SELFIE</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Déposez votre selfie", type=["jpg", "jpeg", "png"])

st.markdown('<p class="headings" style="text-align:center; letter-spacing:2px; color:#B8A39A;">02 — VOS PREFERENCES</p>', unsafe_allow_html=True)

# --- Formulation (clean / bio / vegan) ---
st.markdown('<p style="text-align: center: font-weight: bold">Formulation</p>', unsafe_allow_html=True)


def handle_formulation_change(source):
    if source == "no_pref":
        if st.session_state.no_pref:
            st.session_state.clean = False
            st.session_state.bio = False
            st.session_state.vegan = False
    else:
        if st.session_state[source]:
            st.session_state.no_pref = False

## previous version
# col1, col2, col3 = st.columns(3)
# with col1:
#     clean = st.checkbox("Clean", key="clean", on_change=handle_formulation_change, args=("clean",))
# with col2:
#     bio = st.checkbox("Bio", key="bio", on_change=handle_formulation_change, args=("bio",))
# with col3:
#     vegan = st.checkbox("Vegan", key="vegan", on_change=handle_formulation_change, args=("vegan",))

# no_pref = st.checkbox("Tout me convient !", key="no_pref", on_change=handle_formulation_change, args=("no_pref",))

c1, c2, c3, c4 = st.columns([1, 1, 1, 1.5])

with c1:
    clean = st.checkbox("Clean", key="clean", on_change=handle_formulation_change, args=("clean",))
with c2:
    bio = st.checkbox("Bio", key="bio", on_change=handle_formulation_change, args=("bio",))
with c3:
    vegan = st.checkbox("Vegan", key="vegan", on_change=handle_formulation_change, args=("vegan",))
with c4:
    no_pref = st.checkbox("Tout me convient.", key="no_pref", on_change=handle_formulation_change, args=("no_pref",))

st.divider()

# --- Autres préférences ---
french = st.checkbox("Privilégier le savoir-faire français 🇫🇷", key="french")

product_type_label = st.selectbox(
    "Type de produit",
    options=list(PRODUCT_TYPE_OPTIONS.keys()),
)
product_type = PRODUCT_TYPE_OPTIONS[product_type_label]

price_band_label = st.radio(
    "Budget",
    options=list(PRICE_BAND_OPTIONS.keys()),
    horizontal=True,
)
price_band = PRICE_BAND_OPTIONS[price_band_label]

st.divider()

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(BytesIO(image_bytes))
    st.image(image, caption="Image uploadée", use_container_width=True)

    if st.button("Analyser et recommander", type="primary"):
        try:
            with st.spinner("Analyse acné..."):
                acne_response = requests.post(
                    ACNE_API_URL,
                    files={"file": (uploaded_file.name, image_bytes, uploaded_file.type)},
                    timeout=60,
                )
                acne_response.raise_for_status()
                acne_result = acne_response.json()

            with st.spinner("Analyse type de peau..."):
                oily_response = requests.post(
                    OILY_API_URL,
                    files={"file": (uploaded_file.name, image_bytes, uploaded_file.type)},
                    timeout=60,
                )
                oily_response.raise_for_status()
                oily_result = oily_response.json()

            formulation = []
            if not no_pref:
                if clean:
                    formulation.append("clean")
                if bio:
                    formulation.append("bio")
                if vegan:
                    formulation.append("vegan")

            payload = {
                "skin_type": oily_result["prediction"],
                "acne": acne_result["prediction"] == "acne",
                "preferences": {
                    "formulation": formulation,
                    "origin": ["french"] if french else [],
                    "product_type": product_type,
                    "price_band": price_band,
                },
            }

            with st.spinner("Recherche du meilleur produit..."):
                product_response = requests.post(
                    PRODUCT_API_URL,
                    json=payload,
                    timeout=60,
                )
                product_response.raise_for_status()
                recommendation = product_response.json()

            st.success("Analyse terminée !")

            col_skin, col_acne = st.columns(2)
            with col_skin:
                st.metric("Type de peau", oily_result["prediction"].capitalize())
            with col_acne:
                st.metric("Acné", "Oui" if acne_result["prediction"] == "acne" else "Non")

            st.subheader("Produit recommandé")

            if "error" in recommendation:
                st.warning(recommendation["error"])
            else:
                st.markdown(f"### {recommendation['product_name']}")

                meta_parts = []
                if recommendation.get("brand"):
                    meta_parts.append(recommendation["brand"])
                if recommendation.get("price_display"):
                    meta_parts.append(recommendation["price_display"])
                if meta_parts:
                    st.caption(" · ".join(meta_parts))

                badges = []
                if recommendation.get("is_french"):
                    badges.append("Marque française")
                if recommendation.get("is_vegan"):
                    badges.append("Vegan")
                if recommendation.get("is_clean"):
                    badges.append("Clean")
                if badges:
                    st.write(" · ".join(badges))

                st.info(recommendation["explanation"])

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur de communication avec l'API : {e}")
        except Exception as e:
            st.error(f"Erreur inattendue : {e}")
