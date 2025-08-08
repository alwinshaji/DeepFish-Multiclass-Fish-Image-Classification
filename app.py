import streamlit as st
import numpy as np
import tensorflow as tf
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Set title
st.set_page_config(page_title="DeepFish 🐟", layout="centered")
st.title("🐟 DeepFish: Fish Image Classifier")
st.write("Upload a fish image to identify its category.")

# Class labels
class_labels = [
    "animal_fish",
    "fish_sea_food_black_sea_sprat",
    "fish_sea_food_trout",
    "fish_sea_food_shrimp",
    "fish_sea_food_gilt_head_bream",
    "fish_sea_food_sea_bass",
    "fish_sea_food_red_mullet",
    "fish_sea_food_red_sea_bream",
    "fish_sea_food_hourse_mackerel",
    "fish_sea_food_striped_red_mullet",
    "animal_fish_bass"
]

# Function to download model from Google Drive
def download_model():
    file_id = '1Mq7y85ZHaciK1_6KzVPv-X9FLVAxJMKh'  # Replace with your correct file ID
    model_path = 'densenet_finetuned.h5'
    if not os.path.exists(model_path):
        gdown.download(f'https://drive.google.com/uc?id={file_id}', model_path, quiet=False)
    return model_path

# Load model
@st.cache_resource
def load_fish_model():
    model_file = download_model()
    model = load_model(model_file)
    return model

model = load_fish_model()

# Prediction function
def predict_image(image: Image.Image):
    image = image.resize((224, 224))  # Adjust size based on your model
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_index]
    confidence = predictions[0][predicted_index]
    return predicted_class, confidence

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        label, confidence = predict_image(image)

    st.markdown(f"### 🧠 Prediction: `{label}`")
    st.markdown(f"### 🔍 Confidence: `{confidence:.2f}`")
