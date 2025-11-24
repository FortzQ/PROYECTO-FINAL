# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import pandas as pd
from modify.src.reco_engine import recomendar_canciones
import io
from urllib.parse import quote_plus

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
# Función para normalizar labels del modelo a claves de perfil
# -----------------------------
def normalize_label(raw_label: str) -> str:
    """Convierte la etiqueta del modelo a una clave esperada por el reco_engine.
    Ejemplos de entrada: '0 triste', '1 happy', 'neutral', 'angry' -> devuelve 'sad','happy','neutral','angry'
    """
    if not raw_label:
        return "neutral"
    l = raw_label.lower()
    if "triste" in l or "sad" in l:
        return "sad"
    if "happy" in l or "feliz" in l:
        return "happy"
    if "neutral" in l or "neutro" in l:
        return "neutral"
    if "angry" in l or "enoj" in l or "ira" in l:
        return "angry"
    # fallback
    return "neutral"

# -----------------------------
# Preprocesar imagen
# -----------------------------
def preprocess_image(image: Image.Image):
    size = (224, 224)
    image = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def load_image_from_upload(uploaded):
    """Carga una imagen PIL desde un UploadedFile de Streamlit o bytes robustamente."""
    if uploaded is None:
        return None
    try:
        # UploadedFile tiene read() en Streamlit; camera_input también devuelve un UploadedFile
        if hasattr(uploaded, 'read'):
            data = uploaded.read()
        elif hasattr(uploaded, 'getvalue'):
            data = uploaded.getvalue()
        elif isinstance(uploaded, (bytes, bytearray)):
            data = bytes(uploaded)
        else:
            # Fallback a intentar abrir directamente
            return Image.open(uploaded).convert('RGB')

        return Image.open(io.BytesIO(data)).convert('RGB')
    except Exception:
        return None

# ==========================================================
# 🎵 DISEÑO PREMIUM - MOODIFY
# ==========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
body, html, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
}
.appbar {
    width: 100vw;
    background: rgba(16,185,129,0.12);
    box-shadow: 0 2px 16px rgba(16,185,129,0.10);
    display: flex;
    align-items: center;
    padding: 0.7rem 2.5vw 0.7rem 2vw;
    position: sticky;
    top: 0;
    z-index: 100;
}
.appbar-logo {
    height: 48px;
    margin-right: 1.2rem;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(16,185,129,0.18);
}
.appbar-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #10b981, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    letter-spacing: -0.02em;
}
.appbar-sub {
    font-size: 1.1rem;
    color: #b0e0c8;
    margin-left: 2.5vw;
    font-weight: 500;
}
.main-card {
    background: rgba(16,185,129,0.08);
    border-radius: 24px;
    box-shadow: 0 8px 32px rgba(16,185,129,0.10);
    padding: 2.5rem 2rem 2rem 2rem;
    margin: 2.5rem auto 2rem auto;
    max-width: 700px;
}
.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #10b981;
    margin-bottom: 1.2rem;
    text-align: left;
}
.section-sub {
    font-size: 1.05rem;
    color: #b0e0c8;
    margin-bottom: 1.5rem;
    text-align: left;
}
.image-preview-container {
    background: rgba(16,185,129,0.10);
    border-radius: 18px;
    padding: 1.5rem;
    margin: 1.5rem auto 2rem auto;
    max-width: 420px;
    border: 1px solid rgba(16,185,129,0.18);
    box-shadow: 0 4px 18px rgba(16,185,129,0.10);
}
.result-card {
    background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(52,211,153,0.10));
    border: 1px solid rgba(16,185,129,0.22);
    border-radius: 20px;
    padding: 2.2rem 1.5rem 1.5rem 1.5rem;
    margin: 2rem auto 1.5rem auto;
    max-width: 520px;
    box-shadow: 0 8px 32px rgba(16,185,129,0.10);
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #10b981, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1.2rem;
}
.confidence-badge {
    display: inline-block;
    background: rgba(16,185,129,0.18);
    color: #10b981;
    padding: 0.7rem 1.7rem;
    border-radius: 50px;
    font-size: 1.1rem;
    font-weight: 700;
    border: 2px solid rgba(16,185,129,0.22);
    box-shadow: 0 2px 8px rgba(16,185,129,0.10);
    margin-bottom: 0.7rem;
}
.reco-container {
    background: rgba(8,12,20,0.7);
    border-radius: 18px;
    padding: 18px 18px 10px 18px;
    margin-top: 1.2rem;
    box-shadow: 0 4px 18px rgba(16,185,129,0.10);
}
.reco-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #10b981;
    margin-bottom: 0.7rem;
    text-align: left;
}
.reco-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 6px;
}
.reco-table th {
    text-align: left;
    padding: 10px;
    color: #10b981;
    font-weight:700;
    font-size:0.97rem;
    background: rgba(16,185,129,0.08);
}
.reco-table td {
    padding: 10px;
    color: #e6fff4;
    border-bottom: 1px solid rgba(16,185,129,0.06);
    font-size:0.97rem;
}
.reco-table tr:nth-child(even) {
    background: rgba(16,185,129,0.03);
}
.mini-thumb {
    width:38px;height:38px;border-radius:50%;background:linear-gradient(135deg,#10b981,#34d399);display:inline-flex;align-items:center;justify-content:center;margin-right:10px;color:#012;font-weight:800;font-size:0.97rem;box-shadow:0 2px 8px rgba(16,185,129,0.10);}
.small-btn { background: linear-gradient(135deg, #10b981, #34d399); color: #012 !important; padding: 0.7rem 1.2rem; border-radius: 12px; border: none; font-weight:800; font-size:1rem; white-space:nowrap; display:inline-block; }
.stButton > button { background: linear-gradient(135deg, #10b981, #34d399); color: white; border: none; border-radius: 12px; padding: 0.8rem 2rem; font-weight: 700; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(16,185,129,0.18); }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(16,185,129,0.22); background: linear-gradient(135deg, #34d399, #10b981); }
.footer { margin-top: 3.5rem; text-align: center; color: #88c9b0; font-size: 0.95rem; padding: 1.5rem; border-top: 1px solid rgba(16,185,129,0.18); background: rgba(16,185,129,0.05); border-radius: 18px 18px 0 0; }
</style>
""", unsafe_allow_html=True)
# ==========================================================
# SIDEBAR PREMIUM
# ==========================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 2rem 1rem;'>
        <h2 style='background: linear-gradient(135deg, #10b981, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 1.5rem; font-size: 2rem;'>🎧 Moodify</h2>
        <p style='color: #e0f2fe; font-size: 1rem; line-height: 1.6; margin-bottom: 2rem;'>Convierte tu emoción en música.<br>Sube una foto y deja que la IA haga magia.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='border-top: 1px solid rgba(16, 185, 129, 0.3); margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='padding: 1rem;'>
        <h4 style='color: #10b981; margin-bottom: 1.5rem; text-align: center;'> Características</h4>
        <ul style='color: #b0e0c8; font-size: 0.95rem; line-height: 1.8; list-style: none; padding: 0;'>
        <li style='margin-bottom: 0.8rem;'> Análisis emocional avanzado</li>
        <li style='margin-bottom: 0.8rem;'> Recomendaciones musicales precisas</li>
        <li style='margin-bottom: 0.8rem;'> Interfaz elegante e intuitiva</li>
        <li style='margin-bottom: 0.8rem;'> Tecnología IA de vanguardia</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='border-top: 1px solid rgba(16, 185, 129, 0.3); margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: rgba(16, 185, 129, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.2);'>
        <p style='color: #34d399; font-size: 0.9rem; text-align: center; margin: 0;'>
         Desarrollado con TensorFlow, Streamlit, pasión por la IA y
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# HERO SECTION PREMIUM - COMPLETAMENTE CENTRADO
# ==========================================================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    
    
    # Logo más grande - 400px
    logo = Image.open("img/logo.png")
    st.image(logo, width=400, output_format="PNG")
    
    st.markdown("""
    <div style="text-align:center; margin-top:30px;">

    <h1 style="
        font-size:3rem;
        font-weight:900;
        background: linear-gradient(90deg, #1DB954, #4DF5C2);
        -webkit-background-clip: text;
        color: transparent;
        margin-bottom:10px;
    ">Moodify</h1>

    <p style="
        font-size:1.2rem;
        font-weight:300;
        color:#e0e0e0;
        line-height:1.5;
    ">
        Descubre la banda sonora de tus emociones<br>
        Donde cada sentimiento encuentra su melodía perfecta
    </p>

    <div style="
        width:60%;
        height:3px;
        margin:20px auto;
        border-radius:10px;
        background: linear-gradient(90deg, #1DB954, transparent);
    "></div>

    </div>
""", unsafe_allow_html=True)


# ==========================================================
# UPLOAD SECTION ELEGANTE
# ==========================================================
st.markdown('<div style="max-width:760px; margin: 1.2rem auto; text-align:center">', unsafe_allow_html=True)
st.markdown('<div style="font-size:1.3rem; font-weight:700; color:#10b981;">🎵 Selecciona una imagen o toma una foto</div>', unsafe_allow_html=True)
st.markdown('<div style="color:#b0e0c8; margin-bottom:0.8rem;">Analizaremos tu expresión facial para recomendar la música perfecta</div>', unsafe_allow_html=True)

if "uploader_idx" not in st.session_state:
    st.session_state["uploader_idx"] = 0
if "camera_idx" not in st.session_state:
    st.session_state["camera_idx"] = 0

# Claves dinámicas para forzar reinicio de widgets
uploader_key = f"uploaded_file_{st.session_state['uploader_idx']}"
camera_key = f"camera_input_{st.session_state['camera_idx']}"

# Subir archivo (seguimos permitiendo opcionalmente)
uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed", key=uploader_key)

# Texto elegante para la cámara con botones de reinicio
st.markdown("""
<div style='margin-top:0.6rem; color:#b0e0c8; font-size:0.95rem;'>
    <strong style='color:#10b981;'>📷 Toma una foto</strong> — captura un momento y deja que Moodify lo convierta en música.
</div>
""", unsafe_allow_html=True)
col_cam, col_cam_btn, col_clear = st.columns([3,1,1])
with col_cam:
    camera_file = st.camera_input("", key=camera_key)
with col_cam_btn:
    if st.button("Limpiar foto"):
        # Incrementar camera_idx para crear un nuevo widget vacío
        st.session_state["camera_idx"] += 1
with col_clear:
    if st.button("Limpiar todo"):
        # Incrementar ambos índices para reiniciar uploader y cámara
        st.session_state["camera_idx"] += 1
        st.session_state["uploader_idx"] += 1

st.markdown('</div>', unsafe_allow_html=True)

# Selección de imagen a procesar (prioriza la cámara si se ha tomado foto)
image_to_process = camera_file if camera_file else uploaded_file

# ==========================================================
# PREDICCIÓN ELEGANTE CORREGIDA
# ==========================================================

if image_to_process:
    try:
        # Abrir imagen desde archivo o cámara (robusto)
        image = load_image_from_upload(image_to_process)
        if image is None:
            raise ValueError("No se pudo cargar la imagen desde la entrada proporcionada.")
        
        # Preview elegante
        st.markdown("""
        <div class="image-preview-container">
            <div style='text-align: center; color: #10b981; margin-bottom: 1.5rem; font-size: 1.3rem; font-weight: 600;'>
                📸 Imagen lista
            </div>
            <div class="image-preview">
        """, unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

        # Procesamiento y predicción
        with st.spinner('🎭 Analizando tus emociones...'):
            input_data = preprocess_image(image)
            prediction = model.predict(input_data)
            predicted_index = np.argmax(prediction)
            predicted_label = labels[predicted_index] if labels else "Desconocida"
            confidence = prediction[0][predicted_index]

        # Mostrar resultado elegante
        st.markdown(f"""
        <div class="result-card">
            <div style='font-size: 4.5rem; margin-bottom: 1.5rem;'>🎭</div>
            <div class="result-title">{predicted_label}</div>
            <div class="confidence-badge">
                Confianza: {confidence*100:.1f}%
            </div>
            <div style='margin-top: 2.5rem; color: #b0e0c8; font-size: 1.1rem; line-height: 1.6;'>
                Basado en nuestro análisis avanzado, hemos identificado tu estado emocional principal.<br>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # -----------------------------
        # Recomendaciones musicales - UI mejorada
        # -----------------------------
        mood_key = normalize_label(predicted_label)
        try:
            recomendaciones = recomendar_canciones(mood_key, n=8)
        except Exception as e:
            recomendaciones = None
            st.warning(f"No se pudieron obtener recomendaciones: {e}")

        if recomendaciones is not None and not recomendaciones.empty:
            

            recomendaciones = recomendaciones.reset_index(drop=True)

            # Mostrar todas las canciones dentro de una sola tarjeta elegante (dentro de reco-container)
            st.markdown(f"""
            <div class='reco-container'>
            <div class='result-card' style='max-width:820px; margin:1rem auto; padding:1rem 1.2rem;'>
                <div style='font-size:1.6rem; font-weight:700; color:#10b981; text-align:center; margin-bottom:6px;'>Lista recomendada</div>
                <div style='color:#bfeadf; text-align:center; margin-bottom:12px;'>Selecciona la canción y sube un trozo para reproducirlo, o usa "Buscar" para abrir la canción en la web.</div>
                <div style='padding:6px 4px;'>
            """, unsafe_allow_html=True)

            # (No uploader global: usamos uploaders por fila dentro de la tarjeta)

            for idx, row in recomendaciones.iterrows():
                artist = row.get('artist_name', '')
                track = row.get('track_name', '')
                genre = row.get('genre', '')
                energy = row.get('energy', '')
                valence = row.get('valence', '')
                tempo = row.get('tempo', '')
                # Layout de fila: miniatura | metadata | botón Buscar | (reproductor si corresponde)
                cols = st.columns([1,6,1,3])
                # generar iniciales para miniatura
                initials = ''
                if artist:
                    parts = artist.split()
                    initials = ''.join([p[0].upper() for p in parts[:2]])
                else:
                    initials = (track[:2] if track else '??').upper()

                thumb_html = f"<div style='width:48px;height:48px;border-radius:50%;background:linear-gradient(135deg,#10b981,#34d399);display:inline-flex;align-items:center;justify-content:center;margin-right:10px;color:#012;font-weight:800;font-size:0.95rem;'>{initials}</div>"
                with cols[0]:
                    st.markdown(thumb_html, unsafe_allow_html=True)
                with cols[1]:
                    st.markdown(f"<div style='font-weight:700; color:#e6fff4; font-size:1.05rem;'>{track}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='color:#bfeadf; margin-top:3px;'>{artist} • {genre}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='color:#9fdcc6; font-size:0.85rem; margin-top:6px;'>Energy: {energy} · Valence: {valence} · Tempo: {tempo}</div>", unsafe_allow_html=True)

                # botón de búsqueda (abre YouTube para mejor experiencia)
                query = quote_plus(f"{artist} {track} youtube")
                search_url = f"https://www.youtube.com/results?search_query={query}"
                with cols[2]:
                    st.markdown(f'<a href="{search_url}" target="_blank" class="small-btn" style="text-decoration:none; padding:0.35rem 0.6rem; display:inline-block;">Buscar</a>', unsafe_allow_html=True)

                
                # separación entre filas
                st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

            st.markdown("</div></div>", unsafe_allow_html=True)

            # Botón global de descarga
            all_csv = recomendaciones.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar todas las recomendaciones (CSV)", data=all_csv, file_name=f"recomendaciones_{mood_key}.csv", mime='text/csv')
        else:
            st.info("No se encontraron recomendaciones para este estado anímico.")

    except Exception as e:
        st.error(f"Ocurrió un error al procesar la imagen: {e}")
else:
    st.markdown("""
    <div style='text-align: center; color: #88c9b0; margin-top: 3rem; font-size: 1.2rem; font-style: italic;'>
         Selecciona una imagen o toma una foto para comenzar tu experiencia musical única
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# FOOTER PREMIUM
# ==========================================================
st.markdown("""
<div class="footer">
    <div style='font-size: 1.3rem; color: #10b981; margin-bottom: 1rem; font-weight: 600;'>
        Moodify Experience 2025
    </div>
    <div style='margin-bottom: 1.5rem; color: #e0f2fe;'>
        Creado por Sergio López, Julián Sacristán y Joel Forteza
    </div>
    <div style='font-size: 0.9rem; opacity: 0.8; color: #b0e0c8;'>
        Donde la inteligencia artificial se encuentra con la expresión humana<br>
        <span style='color: #34d399; font-weight: 600;'>IA + Música = Moodify </span>
    </div>
</div>
""", unsafe_allow_html=True)