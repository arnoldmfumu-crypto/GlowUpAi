import os
import requests
import streamlit as st
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Skin Product Recommender", layout="centered")

ACNE_API_URL = os.getenv("ACNE_API_URL", "http://acne:8000/predict")
OILY_API_URL = os.getenv("OILY_API_URL", "http://oily:8000/predict")
PRODUCT_API_URL = os.getenv("PRODUCT_API_URL", "http://product:8000/recommend")

st.title("Skinmatch")

st.write("Upload a face image, then complete the form to get a product recommendation.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# cocher préférences
st.write("Avez-vous des préférences ?")

# 1. Fonction pour gérer la logique de décochage
def handle_change(source):
    if source == 'no_pref':
        # Si "Tout me convient" est coché, on décoche les autres
        if st.session_state.no_pref:
            st.session_state.clean = False
            st.session_state.bio = False
            st.session_state.vegan = False
    else:
        # Si une option spécifique est cochée, on décoche "Tout me convient"
        if st.session_state[source]:
            st.session_state.no_pref = False

# 2. Création des checkboxes avec clés et callbacks
col1, col2, col3, col4 = st.columns(4)

with col1:
    clean = st.checkbox("Clean", key="clean", on_change=handle_change, args=('clean',))
with col2:
    bio = st.checkbox("Bio", key="bio", on_change=handle_change, args=('bio',))
with col3:
    vegan = st.checkbox("Vegan", key="vegan", on_change=handle_change, args=('vegan',))

st.divider()

no_pref = st.checkbox("Non, tout me convient !", key="no_pref", on_change=handle_change, args=('no_pref',))

# Utilisation des variables
if no_pref:
    st.info("Vous n'avez pas de préférences particulières.")
else:
    preferences = []
    if clean: preferences.append("Clean")
    if bio: preferences.append("Bio")
    if vegan: preferences.append("Vegan")
    
    if preferences:
        st.success(f"Vos préférences : {', '.join(preferences)}")

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(BytesIO(image_bytes))
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Analyze and Recommend"):
        try:
            with st.spinner("Calling acne model..."):
                acne_response = requests.post(
                    ACNE_API_URL,
                    files={"file": (uploaded_file.name, image_bytes, uploaded_file.type)},
                    timeout=60
                )
                acne_response.raise_for_status()
                acne_result = acne_response.json()

            with st.spinner("Calling oily model..."):
                oily_response = requests.post(
                    OILY_API_URL,
                    files={"file": (uploaded_file.name, image_bytes, uploaded_file.type)},
                    timeout=60
                )
                oily_response.raise_for_status()
                oily_result = oily_response.json()

            # payload = {
            #     "acne_prediction": acne_result["prediction"],
            #     "acne_confidence": acne_result["confidence"],
            #     "oily_prediction": oily_result["prediction"],
            #     "oily_confidence": oily_result["confidence"],
            #     "clean": clean,
            #     "bio": bio,
            #     "vegan": vegan,
            #     "no_preference": no_pref
            # }

            # with st.spinner("Searching for the best product..."):
            #     product_response = requests.post(
            #         PRODUCT_API_URL,
            #         json=payload,
            #         timeout=60
            #     )
            #     product_response.raise_for_status()
            #     recommendation = product_response.json()

            st.success("Analysis completed")
            
            # st.subheader("Oily Model Result")
            # st.write(f"Prediction: {oily_result['prediction']}")
            # st.write(f"Confidence: {oily_result['confidence']:.2f}")
            st.subheader("Model outputs")
            st.json({
                "acne": acne_result,
                "oily": oily_result
            })

            # st.subheader("Recommended product")
            # st.write(f"**Product:** {recommendation['product_name']}")
            # st.write(f"**Brand:** {recommendation['brand']}")
            # st.write(f"**Format:** {recommendation['format']}")
            # st.write(f"**Price:** {recommendation['price']} €")
            # st.write(f"**Why this product?** {recommendation['reason']}")

        except requests.exceptions.RequestException as e:
            st.error(f"API communication error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")