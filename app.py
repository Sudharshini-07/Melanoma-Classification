# app.py

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Melanoma Classifier", layout="centered")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/melanoma_model.h5")

model = load_model()


def preprocess(img):
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


st.title("🔬 Melanoma Skin Cancer Detection")
st.write("Upload a skin lesion image to detect if it's **benign** or **malignant**.")

uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        input_data = preprocess(image)
        pred = model.predict(input_data)[0][0]
        label = "Malignant (Cancerous)" if pred > 0.5 else "Benign (Non-cancerous)"
        confidence = pred if pred > 0.5 else 1 - pred

        st.subheader("🧠 Model Prediction:")
        st.success(f"{label} ({confidence*100:.2f}% confidence)")
