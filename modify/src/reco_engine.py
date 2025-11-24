import sqlite3
import pandas as pd
import os
import json

# Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Rutas relevantes
db_path = os.path.join(BASE_DIR, "db", "moodify.db")
profile_path = os.path.join(BASE_DIR, "data", "processed", "mood_profiles.json")

# Cargar perfiles emocionales
with open(profile_path, "r") as f:
    mood_profiles = json.load(f)

def recomendar_canciones(emocion, n=5):
    """
    Devuelve n canciones cuyo energy/valence/tempo estén cerca del perfil emocional.
    """

    # Seleccionamos el perfil medio (energy, valence, tempo)
    target_energy   = mood_profiles["energy"][emocion]
    target_valence  = mood_profiles["valence"][emocion]
    target_tempo    = mood_profiles["tempo"][emocion]

    # Conectar a la base de datos
    conn = sqlite3.connect(db_path)

    # Cargar tabla completa a un DataFrame
    df = pd.read_sql_query("SELECT * FROM songs", conn)
    conn.close()

    # Calcular diferencia absoluta respecto al perfil emocional
    df["score"] = (
        (df["energy"] - target_energy).abs() * 0.4 +
        (df["valence"] - target_valence).abs() * 0.4 +
        (df["tempo"] - target_tempo).abs() * 0.2
    )

    # Ordenar de menor (más parecida) a mayor y tomar un pool más grande
    pool_size = max(30, n*3)  # puedes ajustar el tamaño del pool
    pool = df.sort_values("score").head(pool_size)
    # Eliminar canciones repetidas por nombre
    pool = pool.drop_duplicates(subset=["track_name"])
    # Si hay menos canciones únicas que n, ajusta n
    n_final = min(n, len(pool))
    recomendaciones = pool.sample(n=n_final, random_state=None)
    return recomendaciones[["artist_name", "track_name", "genre", "energy", "valence", "tempo"]]
