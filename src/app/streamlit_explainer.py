import streamlit as st
import pandas as pd
import joblib
import shap
import os
import numpy as np
from llm_interface import ask_llm_with_shap

# === Streamlit Grundkonfiguration ===
st.set_page_config(page_title="CostPilot – Explainable AI", layout="wide")

# === Styling ===
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #e4e4e4;
        }
        .stApp { background-color: #121212; }
        h1, h2, h3, label { color: #e4e4e4 !important; }
        .stTextInput, .stNumberInput, .stSelectbox {
            background-color: #1e1e1e !important; color: white !important;
        }
        div[data-baseweb="input"] input { color: white !important; }
        .chat-bubble {
            background-color: #1e1e1e; border-radius: 10px;
            padding: 12px 18px; margin-bottom: 10px;
        }
        .user { text-align: right; background-color: #343541; }
        .assistant { text-align: left; background-color: #2c2c2c; }
    </style>
""", unsafe_allow_html=True)

# === Titel ===
st.title("CostPilot – Explainable Cost Prediction + Chat")
st.markdown("Gib unten deine Eingabedaten ein, um eine Vorhersage zu erhalten – und sprich danach direkt mit dem KI-Assistenten über das Ergebnis.")

# === Modell & Daten laden ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "models", "linear", "optimized", "ridge_optimized_model.pkl"))
DATA_PATH_ORIGINAL = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "data", "processed", "cleaned_data.csv"))
DATA_PATH_ENCODED = os.path.normpath(os.path.join(BASE_DIR, "..", "..","models", "linear", "optimized", "cleaned_data_one_hot_encoding.csv"))

if not os.path.exists(MODEL_PATH):
    st.error("❌ Kein Modell gefunden! Bitte zuerst das Ridge-Modell trainieren und speichern.")
    st.stop()

model = joblib.load(MODEL_PATH)
data_original = pd.read_csv(DATA_PATH_ORIGINAL)
data_encoded = pd.read_csv(DATA_PATH_ENCODED)

# === Zielvariable entfernen ===
TARGET_COL = "Preis"
for df in [data_original, data_encoded]:
    if TARGET_COL in df.columns:
        df.drop(columns=[TARGET_COL], inplace=True)

# === Spaltenarten erkennen ===
categorical_cols = data_original.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = data_original.select_dtypes(include=[np.number]).columns.tolist()

# === Eingabeformular ===
st.subheader("Eingabedaten")
user_inputs = {}

with st.form("eingabeformular"):
    # numerische Felder
    for col in numerical_cols:
        default_val = float(data_original[col].median()) if not data_original[col].isnull().all() else 0.0
        user_inputs[col] = st.number_input(f"{col}", value=default_val)

    # kategorische Felder
    for col in categorical_cols:
        options = sorted(data_original[col].dropna().unique().tolist())
        selected = st.selectbox(f"{col}", ["- Bitte auswählen -"] + options)
        user_inputs[col] = selected

    submitted = st.form_submit_button("Vorhersage starten 🚀")

# === Prediction + Chat-Start ===
if submitted:
    try:
        # Eingaben als DataFrame
        df_input = pd.DataFrame([user_inputs])
        df_input.replace("- Bitte auswählen -", np.nan, inplace=True)

        # Gleiche One-Hot-Encoding-Logik wie beim Training
        df_encoded = pd.get_dummies(df_input, drop_first=False)

        # Fehlende Spalten aus Trainingsdaten hinzufügen
        for col in data_encoded.columns:
          if col not in df_encoded.columns:
               df_encoded[col] = 0

        # Überflüssige Spalten entfernen (falls vorhanden)
        df_encoded = df_encoded[data_encoded.columns]
        df_encoded = df_encoded.reset_index(drop=True)
        # st.write("✅ Finales Input fürs Modell:", df_encoded)
        try:
            # Falls Modell eine Pipeline ist → Regressor extrahieren
            base_model = model.named_steps["ridge"] if hasattr(model, "named_steps") else model
            background = data_encoded.sample(min(200, len(data_encoded)), random_state=42)

            explainer = shap.LinearExplainer(
                base_model,
                background,
                feature_perturbation="interventional"
            )

        except Exception as e:
            st.error(f"SHAP konnte nicht initialisiert werden: {e}")
            st.stop()

        # SHAP-Werte für aktuelle Eingabe berechnen
        shap_values = explainer.shap_values(df_encoded)
        instance_shap = shap_values[0]  # rohwerte für dieses sample

        instance = df_encoded.iloc[0]

        # Ergebnis-DatenFrame
        shap_df = pd.DataFrame({
            "feature": df_encoded.columns,
            "value": instance.values,
            "shap_value": instance_shap
        })

        prediction = float(model.predict(df_encoded)[0])

        # In Session speichern, um später an LLM zu übergeben
        st.session_state["prediction"] = prediction
        st.session_state["shap_df"] = shap_df
        st.session_state["user_inputs"] = user_inputs

        st.markdown("---")
        st.success(f"Geschätzte Kosten: **{prediction:,.2f} €**")


    except Exception as e:
        st.error(f"Fehler bei der Vorhersage: {e}")

# === Chatbereich nach der Vorhersage ===
if "prediction" in st.session_state:
    st.markdown("---")
    st.header("💬 CostPilot Chat")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": f"Ich habe eine Kostenschätzung von **{st.session_state['prediction']:,.2f} €** berechnet. Was möchtest du darüber wissen?"}
        ]

    for msg in st.session_state["messages"]:
        css_class = "user" if msg["role"] == "user" else "assistant"
        st.markdown(f"<div class='chat-bubble {css_class}'>{msg['content']}</div>", unsafe_allow_html=True)

    user_input = st.text_input(
        "Deine Frage:",
        placeholder="Frag mich etwas zur Vorhersage oder zu den Einflussfaktoren...",
        key="chat_input"
    )

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})

        try:
            shap_df = st.session_state.get("shap_df")
            prediction = st.session_state.get("prediction")
            answer = ask_llm_with_shap(shap_df, prediction, user_input)
            st.session_state["messages"].append({"role": "assistant", "content": answer})

        except Exception as e:
            st.session_state["messages"].append({
                "role": "assistant",
                "content": f"⚠️ Fehler bei der Analyse: {e}"
            })

        # statt st.session_state["chat_input"] = ""
        st.session_state.pop("chat_input", None)
        st.rerun()
