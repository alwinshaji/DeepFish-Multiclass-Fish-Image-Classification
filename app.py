import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os

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

# Download model from Google Drive if not present
MODEL_PATH = "densenet_finetuned.h5"
DRIVE_FILE_ID = "1Mq7y85ZHaciK1_6KzVPv-X9FLVAxJMKh"  # replace with your actual file ID

def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

# Preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))  # match model input size
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # normalize
    return np.expand_dims(img_array, axis=0)

# Prediction
def predict_image(model, img):
    processed = preprocess_image(img)
    predictions = model.predict(processed)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    return class_labels[predicted_class], confidence

# Streamlit app UI
st.title("Fish Image Classifier 🐟")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Loading model..."):
        model = download_model()

    with st.spinner("Classifying..."):
        label, confidence = predict_image(model, image)
        st.success(f"Predicted: {label} (Confidence: {confidence:.2f})")
