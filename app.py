import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Set the page title and layout
st.set_page_config(page_title="DeepFish - Fish Image Classifier", layout="centered")

# Load model
@st.cache_resource
def load_cnn_model():
    return load_model("models/efficientnet_model.h5")

model = load_cnn_model()

# Class labels
class_labels = [
    "Animal Fish",
    "Black Sea Sprat",
    "Trout",
    "Shrimp",
    "Gilt Head Bream",
    "Sea Bass",
    "Red Mullet",
    "Red Sea Bream",
    "Horse Mackerel",
    "Striped Red Mullet",
    "Bass"
]

# Image preprocessing
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction function
def predict(image_array):
    predictions = model.predict(image_array)[0]
    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index]
    return class_labels[predicted_index], confidence

# UI
st.title("🐟 DeepFish")
st.subheader("Multiclass Fish Image Classifier")

uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Classifying..."):
            processed_image = preprocess_image(uploaded_file)
            label, confidence = predict(processed_image)
            st.success(f"**Prediction:** {label}")
            st.info(f"**Confidence:** {confidence:.2%}")
    except Exception as e:
        st.error("Something went wrong. Please upload a valid fish image.")
        st.text(f"Error: {str(e)}")
