import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os

st.set_page_config(page_title="DeepFish", layout="centered")
st.title("🐟 DeepFish - Multiclass Fish Image Classifier")

# --- Load model from Google Drive if not present ---
@st.cache_resource
def load_fish_model():
    model_path = "densenet_finetuned.h5"
    if not os.path.exists(model_path):
        file_id = "1Mq7y85ZHaciK1_6KzVPv-X9FLVAxJMKh"  # <-- your file ID
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    return load_model(model_path)

model = load_fish_model()

# --- Class labels from training (must match class_indices order exactly) ---
CLASS_NAMES = [
    "animal_fish", 
    "fish_sea_food_black_sea_sprat", 
    "fish_sea_food_gilt_head_bream", 
    "fish_sea_food_hourse_mackerel", 
    "fish_sea_food_red_mullet", 
    "fish_sea_food_red_sea_bream", 
    "fish_sea_food_sea_bass", 
    "fish_sea_food_shrimp", 
    "fish_sea_food_striped_red_mullet", 
    "fish_sea_food_trout",
    "animal_fish_bass"
]

uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # --- Preprocessing (must match training) ---
    img = img.resize((224, 224))                         # Resize to training size
    img_array = image.img_to_array(img)                  # Convert to array
    img_array = img_array / 255.0                        # Rescale to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)        # Add batch dimension

    # --- Predict ---
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(np.max(predictions)) * 100

    # --- Display results ---
    st.markdown(f"### 🎯 Predicted Class: `{predicted_label}`")
    st.markdown(f"### 🔍 Confidence: `{confidence:.2f}%`")
