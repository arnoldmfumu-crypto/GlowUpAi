import streamlit as st
from PIL import Image
import numpy as np
import time

# ---------------- INIT ----------------
if "step" not in st.session_state:
    st.session_state.step = 1

if "result" not in st.session_state:
    st.session_state.result = None

# ---------------- FAKE MODEL (à remplacer) ----------------
def predict_skin(image):
    """
    Remplace cette fonction par ton vrai modèle ML
    """
    time.sleep(2)  # simulation

    return {
        "acne": 0.78,
        "wrinkles": 0.34,
        "spots": 0.65,
        "hydration": "Faible"
    }

# ---------------- ÉTAPE 1 ----------------
if st.session_state.step == 1:
    st.title("Analyse de peau")

    uploaded_file = st.file_uploader("Importer une photo", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file)

        if st.button("Lancer le diagnostic"):
            st.session_state.image = uploaded_file
            st.session_state.step = 2
            st.rerun()

# ---------------- ÉTAPE 2 : IA ----------------
elif st.session_state.step == 2:
    st.title("Analyse en cours...")

    image = Image.open(st.session_state.image)

    with st.spinner("Analyse de la peau par IA..."):
        result = predict_skin(image)
        st.session_state.result = result

    st.session_state.step = 3
    st.rerun()

# ---------------- ÉTAPE 3 : FORMULAIRE ADDITIONNEL ----------------
elif st.session_state.step == 3:
    if "filters" not in st.session_state:
      st.session_state.filters = []

    st.title("Avez-vous des préférences ?")

    natural_ing = st.checkbox("Clean")
    organic_ing = st.checkbox("Bio")
    vegan_ing = st.checkbox("Vegan")
    no_pref_ing = st.checkbox("Non, tout me convient !")

    validate = st.button("Valider les filtres")

    if validate:
        st.session_state.step = 4
        st.rerun()


# ---------------- ÉTAPE 4 : RÉSULTATS ----------------
elif st.session_state.step == 4:
    st.title("Résultats du diagnostic")

    result = st.session_state.result

    st.write("### Voici le produit que nous vous recommandons :")

    st.write(f"🔴 V : {result['acne']*100:.0f}%")

    if st.button("Recommencer"):
        st.session_state.step = 1
        st.session_state.result = None
        st.rerun()