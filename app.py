import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Model Load
model = tf.keras.models.load_model("gender_classifier.keras")

# Class Names
classes = ["Female", "Male"]

st.title("Male and Female Image Classification")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Image Preprocessing
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    result = classes[np.argmax(prediction)]

    st.subheader(f"Prediction: {result}")
