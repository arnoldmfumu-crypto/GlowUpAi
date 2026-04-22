import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

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

st.title("Skinmatch")
st.write("Uploadez une photo de votre visage, renseignez vos préférences et obtenez une recommandation produit.")

uploaded_file = st.file_uploader("Uploadez une image", type=["jpg", "jpeg", "png"])

st.subheader("Vos préférences")

# --- Formulation (clean / bio / vegan) ---
st.write("**Formulation**")


def handle_formulation_change(source):
    if source == "no_pref":
        if st.session_state.no_pref:
            st.session_state.clean = False
            st.session_state.bio = False
            st.session_state.vegan = False
    else:
        if st.session_state[source]:
            st.session_state.no_pref = False


col1, col2, col3 = st.columns(3)
with col1:
    clean = st.checkbox("Clean", key="clean", on_change=handle_formulation_change, args=("clean",))
with col2:
    bio = st.checkbox("Bio", key="bio", on_change=handle_formulation_change, args=("bio",))
with col3:
    vegan = st.checkbox("Vegan", key="vegan", on_change=handle_formulation_change, args=("vegan",))

no_pref = st.checkbox("Tout me convient !", key="no_pref", on_change=handle_formulation_change, args=("no_pref",))

st.divider()

# --- Autres préférences ---
french = st.checkbox("Produits de marque française uniquement", key="french")

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
                    timeout=120,
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
