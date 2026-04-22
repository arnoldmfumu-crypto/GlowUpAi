import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="GlowUp AI", layout="centered")

ACNE_API_URL = os.getenv("ACNE_API_URL", "http://acne:8000/predict")
OILY_API_URL = os.getenv("OILY_API_URL", "http://oily:8000/predict")
PRODUCT_API_URL = os.getenv("PRODUCT_API_URL", "http://product:8000/recommend")

PRODUCT_TYPE_OPTIONS = {
    "Tout": None,
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
    "Tout": "any",
    "Budget (< 15€)": "budget",
    "Milieu de gamme (15–50€)": "mid",
    "Premium (> 50€)": "premium",
}

## Style

st.markdown("""
    <style>
    /* Importation des polices Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400..700;1,400..700&family=Oxygen:wght@300;400;700&display=swap');

    /* ─── LAYOUT GLOBAL ─────────────────────────────────────────────── */

    [data-testid="stAppViewContainer"] {
        background-color: #FFF9F4;
    }

    .block-container {
        max-width: 700px !important;
        margin: 0 auto !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        justify-items: center !important;
    }

    /* ─── HEADER ────────────────────────────────────────────────────── */

    .full-width-gradient {
        width: 100vw;
        position: relative;
        left: 50%;
        margin-left: -50vw;
        background: linear-gradient(180deg, #FAD0C4 0%, #FFF9F4 100%);
        padding: 60px 0 30px;
        text-align: center;
        margin-bottom: 40px;
    }

    .header-content {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 20px;
    }

    #glow-up-ai span {
        font-family: 'Lora', serif !important;
        font-size: 42px;
        font-weight: 500;
        color: #3C2A21 !important;
        text-align: center;
        margin-bottom: 0;
    }

    /* ─── TYPOGRAPHIE ───────────────────────────────────────────────── */

    p {
        color: #3C2A21 !important;
    }

    [data-testid="stMarkdownContainer"] p.headings {
        font-family: 'Lora', serif !important;
        color: #B8A39A !important;
        text-align: center;
        letter-spacing: 2px;
        font-size: 1.3rem;
    }

    .subtitle {
        font-family: serif;
        font-size: 24px;
        font-weight: 400;
        color: #3C2A21 !important;
        text-align: center;
        margin-top: -10px;
    }
            
    [data-testid="stElementContainer"] label, [data-testid="stElementContainer"] span, [data-testid="stElementContainer"] p, [data-testid="stTooltipHoverTarget"] {
        font-family: "Oxygen", sans-serif;        
    }
    
    [data-testid="stWidgetLabel"] > span > div > p {
        font-size: 1.2rem;        
    }
            

    /* ─── COMPOSANTS ────────────────────────────────────────────────── */

    /* Carte générique */
    .custom-section {
        background-color: #fff;
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #F0E6E1;
        margin-bottom: 25px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #fff;
        border-radius: 20px;
        padding: 20px;
        border: 1px solid #F0E6E1;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        max-width: 600px;
        margin: 0 auto 50px;
        justify-items: center;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #FFF9F4 !important;
        border: 2px dashed #E5B1B6 !important;
    }

    /* Bouton */
    div.stButton {
        display: flex;
        justify-content: center;
    }
    
    [data-testid="stElementContainer"] {
        align-self: center;
        color: #3C2A21;
        text-align: center;
    }

    div.stButton > button {
        background-color: #3C2A21 !important;
        color: #ffffff !important;
        border: none !important;
        width: 100%;
    }
            
    [data-testid="stButton"] > button > div > p {
        color: #ffffff !important;        
    }
            
    [data-testid="stFileUploaderDropzoneInstructions"] > div > span {            
        color: #3C2A21 !important;
    }
            
    [data-testid="stWidgetLabel"] > span > [data-testid="stMarkdownContainer"] > p {
        color: #3C2A21 !important;
        text-align: center;
        font-size: 1rem;
        font-weight: 700;
        margin: 20px auto 0;
    }
    
    [data-testid="stBaseButton-secondary"] p, [data-testid="stMarkdownContainer"] span {
        color: #fff !important;
        font-family: inherit !important;
    }
    
    [data-testid="stWidgetLabel"] {
        color: #3C2A21 !important;
    }
            
    

    /* ─── FORMULAIRE ────────────────────────────────────────────────── */

    /* Checkboxes */
    [data-testid="stCheckbox"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }

    [data-testid="stCheckbox"] p,
    [data-testid="stMarkdownContainer"] p {
        color: #3C2A21;
        text-align: center;
    }
            
    [data-testid="stMarkdownContainer"] p.ingredients {
        text-align: center;
        font-size: 1rem;
        font-weight: 700;
    }

    [data-testid="stHorizontalBlock"] {
        justify-content: center !important;
    }

    /* Selectbox */
    [data-testid="stSelectbox"],
    [data-testid="stSelectbox"] label {
        text-align: center;
        margin: 0 auto;
        justify-content: center;
        font-family: 'Oxygen', sans serif;
    }

    /* Selector */          
    .st-c4 {
        width: 80%;
        align-self: center;
        margin: 15px auto 0;
    }

    /* Radio */
    [data-testid="stRadio"] {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    [data-testid="stBaseButton-primary"] > div > span > div > p {
        color: #fff !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding: 0 25px;
    }

    [data-testid="stRadio"] > label {
        text-align: center;
    }

    [data-testid="stRadio"] [role="radiogroup"] {
        justify-content: center;
    }

    /* ─── RÉSULTATS ─────────────────────────────────────────────────── */

    [data-testid="stMetric"] {
        text-align: center;
    }

    [data-testid="stImage"] {
        display: flex;
        justify-content: center;
    }

    hr {
        margin: 20px auto !important;
    }
    
    [data-testid="stLayoutWrapper"] {
        margin-bottom: 3%;        
    }
    
    [data-testid="stMetricValue"] p, [data-testid="stAlertContentInfo"] p {
        text-align: justify;
    }
            
    [data-testid="stHeadingWithActionElements"] > h3 {
        padding-bottom: 0;        
    }
            
    [data-testid="stHeadingWithActionElements"] > h3 > span {
        color: #3C2A21 !important;
        font-family: 'Oxygen', sans-serif !important;
        font-size: 1.5rem;
        padding: 2% 0;
    }
    
    [data-testid="stBaseButton-primary"] [data-testid="stCaptionContainer"] p, [data-testid="stMarkdown"] [data-testid="stCaptionContainer"] p {
        text-transform: uppercase;        
    }
            
    /* other */
    [data-testid="stIconMaterial"] {
        font-family: inherit;        
    }

    """, unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="full-width-gradient">
        <div class="header-content">
            <p style="letter-spacing: 2px; font-size: 12px; color: #3C2A21; opacity: 0.7;">DIAGNOSTIC PEAU • IA</p>
            <h1 class="main-title">
                GlowUp AI
            </h1>            
            <p style="color: #3C2A21; font-size: 18px; max-width: 600px; margin: 0 auto;">
                Un selfie, vos préférences, et nous vous recommandons les produits faits pour votre peau.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Etape 1 : SELFIE
st.markdown('<p class="headings">01 — VOTRE SELFIE</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Importez votre selfie", type=["jpg", "jpeg", "png"])

# Etape 2 : PREFERENCES
st.markdown('<p class="headings">02 — VOS PREFERENCES</p>', unsafe_allow_html=True)

# --- Formulation (clean / bio / vegan) ---
st.markdown('<p class="ingredients">Formulation</p>', unsafe_allow_html=True)


def handle_formulation_change(source):
    if source == "no_pref":
        if st.session_state.no_pref:
            st.session_state.clean = False
            st.session_state.bio = False
            st.session_state.vegan = False
    else:
        if st.session_state[source]:
            st.session_state.no_pref = False

c1, c2, c3, c4 = st.columns([1, 1, 1, 1.5])

with c1:
    clean = st.checkbox("Clean", key="clean", on_change=handle_formulation_change, args=("clean",))
with c2:
    bio = st.checkbox("Bio", key="bio", on_change=handle_formulation_change, args=("bio",))
with c3:
    vegan = st.checkbox("Vegan", key="vegan", on_change=handle_formulation_change, args=("vegan",))
with c4:
    no_pref = st.checkbox("Tout me convient", key="no_pref", on_change=handle_formulation_change, args=("no_pref",))

#st.divider()

# --- Autres préférences ---
french = st.checkbox("Privilégier le savoir-faire français 🇫🇷", key="french")

product_type_label = st.selectbox(
    "Type de produit",
    options=list(PRODUCT_TYPE_OPTIONS.keys()),
)
product_type = PRODUCT_TYPE_OPTIONS[product_type_label]

price_band_label = st.radio(
    "Votre budget",
    options=list(PRICE_BAND_OPTIONS.keys()),
    horizontal=True,
)
price_band = PRICE_BAND_OPTIONS[price_band_label]

st.divider()


# --- Résultats ---

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(BytesIO(image_bytes))
    st.image(image, caption="Photo importée", use_container_width=True)

    if st.button("Lancer l'analyse", type="primary"):
        try:
            with st.spinner("Analyse de votre acné..."):
                acne_response = requests.post(
                    ACNE_API_URL,
                    files={"file": (uploaded_file.name, image_bytes, uploaded_file.type)},
                    timeout=60,
                )
                acne_response.raise_for_status()
                acne_result = acne_response.json()

            with st.spinner("Analyse de votre type de peau..."):
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

            with st.spinner("Nous recherchons les meilleurs produits..."):
                product_response = requests.post(
                    PRODUCT_API_URL,
                    json=payload,
                    timeout=120,
                )
                product_response.raise_for_status()
                recommendation = product_response.json()

            st.markdown("<p class='headings' style='margin-top: 10%'>03 — RESULTATS DE L'ANALYSE</p>", unsafe_allow_html=True)

            col_skin, col_acne = st.columns(2)
            with col_skin:
                st.metric("Votre type de peau :", oily_result["prediction"].capitalize())
            with col_acne:
                st.metric("Est-elle acnéïque ?", "Oui" if acne_result["prediction"] == "acne" else "Non")

            st.markdown('<p style="font-size: 1rem ; padding: 0; margin-bottom: 0">Nous vous recommandons ce produit :</p>', unsafe_allow_html=True)

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
