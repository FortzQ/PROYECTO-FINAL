# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# -----------------------------
# Rutas de archivos
# -----------------------------
MODEL_H5 = "model.h5"
LABELS_TXT = "labels.txt"

# -----------------------------
# Cargar modelo
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_H5)
    return model

model = load_model()

# -----------------------------
# Cargar labels
# -----------------------------
def load_labels():
    if not os.path.exists(LABELS_TXT):
        st.error(f"No se encontró {LABELS_TXT}")
        return []
    with open(LABELS_TXT, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

labels = load_labels()

# -----------------------------
# Preprocesar imagen
# -----------------------------
def preprocess_image(image: Image.Image):
    # Ajustar tamaño del modelo (normalmente 224x224 para TM)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    # Convertir a array y normalizar
    image_array = np.asarray(image) / 255.0
    # Asegurar que tenga la forma (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# -----------------------------
# Interfaz Streamlit
# -----------------------------
st.title("Moodify 🎵 — Reconocimiento de emociones")

st.write("Sube una foto y detecta tu emoción")

uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagen cargada", use_column_width=True)

        # Preprocesar y predecir
        input_data = preprocess_image(image)
        prediction = model.predict(input_data)
        predicted_index = np.argmax(prediction)
        predicted_label = labels[predicted_index] if labels else "Desconocida"
        confidence = prediction[0][predicted_index]

        st.success(f"Emoción detectada: **{predicted_label}** ({confidence*100:.2f}%)")
    except Exception as e:
        st.error(f"Ocurrió un error al procesar la imagen: {e}")
