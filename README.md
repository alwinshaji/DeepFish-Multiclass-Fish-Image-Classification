# 🐟 Multiclass Fish Image Classification

A deep learning project to classify fish species into **11 classes** using **DenseNet201** with fine-tuning.  
[🎯 Try it on Streamlit](https://deepfish-multiclass-fish-image-classification.streamlit.app)

---

## 📌 Overview
- **Model:** DenseNet201 (ImageNet pretrained)
- **Training:** 3 stages  
  1. Train classifier head  
  2. Unfreeze top 30% layers  
  3. Full fine-tuning  
- **Features:** Image augmentation, class weight balancing, saved model loading
- **Note:** `densenet_finetuned.h5` is not uploaded here due to GitHub’s file size limit (25 MB).
---

## 📂 Structure

├── Multiclass_Fish_Image_Classification.ipynb # Training notebook

├── app.py # Streamlit app

├── requirements.txt # Dependencies

├── runtime.txt # Runtime config

└── README.md

---

## 📊 Dataset
Organized into `train` and `val` folders:
fish_data/

├── train/class_1 ... class_11

└── val/class_1 ... class_11

---

## 🏋️‍♂️ Training
Run the notebook:
bash
jupyter notebook Multiclass_Fish_Image_Classification.ipynb
Model auto-saves to Google Drive after training.

---

🌐 Run Locally
pip install -r requirements.txt
streamlit run app.py

---

🙌 Acknowledgements 

DenseNet201 from Keras Applications

Streamlit for deployment
---
👤 Author

[Alwin Shaji](https://www.linkedin.com/in/alwnshaji)

---


