import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os

# Page setup
st.set_page_config(page_title="DeepFish", layout="centered")
st.markdown("<h1 style='text-align: center;'>🐟 DeepFish</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Multiclass Fish Image Classifier</h4>", unsafe_allow_html=True)

# Class labels
CLASS_NAMES = {
    "animal_fish_bass": "animal fish bass",
    "fish_sea_food_black_sea_sprat": "fish sea_food black_sea_sprat",
    "fish_sea_food_gilt_head_bream": "fish sea_food gilt_head_bream",
    "fish_sea_food_hourse_mackerel": "fish sea_food hourse_mackerel",
    "fish_sea_food_red_mullet": "fish sea_food red_mullet",
    "fish_sea_food_red_sea_bream": "fish sea_food red_sea_bream",
    "fish_sea_food_sea_bass": "fish sea_food sea_bass",
    "fish_sea_food_shrimp": "fish sea_food shrimp",
    "fish_sea_food_striped_red_mullet": "fish sea_food striped_red_mullet",
    "fish_sea_food_trout": "fish sea_food trout"
}

# Download model from Google Drive
@st.cache_resource
def load_fish_model():
    model_path = "fine_tuned_fish_model.h5"
    if not os.path.exists(model_path):
        file_id = "1JaED2j1HjifDV82M9haFbmTu8thUlbn3"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    model = load_model(model_path)
    return model

model = load_fish_model()

# Upload image
st.markdown("---")
uploaded_file = st.file_uploader("📤 Upload a fish image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        # Preprocessing
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediction
        prediction = model.predict(img_array)
        predicted_key = list(CLASS_NAMES.keys())[np.argmax(prediction)]
        predicted_class = CLASS_NAMES[predicted_key]
        confidence = np.max(prediction) * 100

    st.success("✅ Prediction complete!")
    st.markdown(f"<h3>🎯 Predicted Class: <span style='color:#4CAF50'>{predicted_class}</span></h3>", unsafe_allow_html=True)
    st.markdown(f"<h4>🔍 Confidence: <span style='color:#2196F3'>{confidence:.2f}%</span></h4>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and TensorFlow")
