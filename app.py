# app.py
import streamlit as st
import pandas as pd
from scripts.predict import predict_headlines

st.set_page_config(page_title="Spanish Headline Classifier", layout="centered")

# =========================
# Sidebar: settings
# =========================
st.sidebar.header("Settings")
model_dir = st.sidebar.text_input("Model directory", value="models/beto-targeted")
max_len = st.sidebar.slider("Max tokenized length", min_value=32, max_value=512, value=256, step=32)
st.sidebar.caption("Tip: leave defaults unless you know you need to change them.")

# =========================
# Title
# =========================
st.title("📰 Spanish Headline Classifier")
st.write("Clasifica titulares en **REAL / FAKE / AI_GENERATED** usando un modelo BETO fine-tuned.")

# =========================
# Single prediction
# =========================
st.subheader("Clasificación individual")
single_text = st.text_area(
    "Escribe un titular en español:",
    height=100,
    placeholder="El mundo termina mañana",
    key="single_text",
)

col1, col2 = st.columns([1, 1])
with col1:
    run_single = st.button("Clasificar", key="run_single")
with col2:
    load_examples = st.button("Cargar ejemplos", key="load_examples")

if load_examples:
    st.session_state.single_text = "\n".join([
        "El Gobierno presenta un plan de inversión en infraestructuras para 2026",
        "El mundo termina mañana",
        "Expertos afirman que la situación actual requiere medidas coordinadas a nivel regional",
    ])

def show_prob_bars(labels, probs_row):
    st.write("**Probabilidades**")
    for lbl, p in zip(labels, probs_row):
        st.progress(float(p), text=f"{lbl}: {float(p):.3f}")

if run_single and single_text.strip():
    df_single, probs = predict_headlines([single_text.strip()], model_dir=model_dir, max_len=max_len)
    st.write(f"**Predicción:** {df_single.label[0]}  (confianza {df_single.confidence[0]:.3f})")
    show_prob_bars(["REAL", "FAKE", "AI_GENERATED"], probs[0])
    st.download_button(
        "Descargar CSV (este titular)",
        df_single.to_csv(index=False).encode("utf-8"),
        file_name="prediccion_unica.csv",
        mime="text/csv",
    )

st.divider()

# =========================
# Batch classification
# =========================
st.subheader("Clasificación por lotes")
multi = st.text_area(
    "Pega varios titulares (uno por línea):",
    height=180,
    placeholder="El Gobierno presenta un plan de inversión...\nEl mundo termina mañana\nExpertos afirman que la situación actual...",
    key="batch_text",
)

if st.button("Clasificar en lote", key="run_batch") and multi.strip():
    rows = [t.strip() for t in multi.splitlines() if t.strip()]
    df_batch, probs = predict_headlines(rows, model_dir=model_dir, max_len=max_len)

    # Añadimos columnas con probabilidades por clase para cada fila
    labels = ["REAL", "FAKE", "AI_GENERATED"]
    for i, lbl in enumerate(labels):
        df_batch[f"p_{lbl}"] = [float(p[i]) for p in probs]

    st.write("Resultados")
    st.dataframe(df_batch, use_container_width=True)

    st.download_button(
        "Descargar CSV (lote)",
        df_batch.to_csv(index=False).encode("utf-8"),
        file_name="predicciones_lote.csv",
        mime="text/csv",
    )

st.caption("⚠️ Prototipo: detecta patrones lingüísticos; no verifica hechos. Usa criterio humano.")
