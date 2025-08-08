import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import tempfile
import gdown
import os

st.set_page_config(page_title="DeepFish", layout="centered")
st.title("🐟 DeepFish - Multiclass Fish Image Classifier")

class_names = {
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

# Download model from Google Drive if not already downloaded
@st.cache_resource
def load_fish_model():
    model_path = "densenet_finetuned.h5"
    if not os.path.exists(model_path):
        # Replace this with YOUR Google Drive shareable file ID
        file_id = "1Mq7y85ZHaciK1_6KzVPv-X9FLVAxJMKh"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    model = load_model(model_path)
    return model

model = load_fish_model()

# Image upload
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown(f"### 🎯 Prediction: `{predicted_class}`")
    st.markdown(f"### 🔍 Confidence: `{confidence:.2f}%`")
