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
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "models", "linear", "pipeline", "elasticnet_pipeline.pkl"))
DATA_PATH_ORIGINAL = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "data", "processed", "cleaned_data.csv"))

if not os.path.exists(MODEL_PATH):
    st.error("❌ Kein Modell gefunden! Bitte zuerst das Ridge-Modell trainieren und speichern.")
    st.stop()

model = joblib.load(MODEL_PATH)
data_original = pd.read_csv(DATA_PATH_ORIGINAL)

# === Zielvariable entfernen ===
TARGET_COL = "Preis"
if TARGET_COL in data_original.columns:
    data_original = data_original.drop(columns=[TARGET_COL])


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
        user_inputs[col] = st.selectbox(col, ["- Bitte auswählen -"] + options)

    submitted = st.form_submit_button("Vorhersage starten 🚀")

# === Prediction + Chat-Start ===
if submitted:
    try:
        # Eingaben als DataFrame
        df_input = pd.DataFrame([user_inputs])
        df_input.replace("- Bitte auswählen -", np.nan, inplace=True)

        prediction = float(model.predict(df_input)[0])

        # Pipeline-Preprocessing separat ausführen
        preprocessor = model.named_steps["prep"]
        regressor = model.named_steps["model"]

        background = data_original.sample(min(300, len(data_original)), random_state=42)

        X_background = preprocessor.transform(background)
        if hasattr(X_background, "toarray"):
            X_background = X_background.toarray()
        X_input = preprocessor.transform(df_input)
        if hasattr(X_input, "toarray"):
            X_input = X_input.toarray()

        explainer = shap.LinearExplainer(regressor, X_background)
        shap_values = explainer(X_input)

        instance_shap = shap_values.values[0]

        shap_df = pd.DataFrame({
            "feature": preprocessor.get_feature_names_out(),
            "value": X_input[0].tolist(),
            "shap_value": instance_shap.tolist()
        })

        st.session_state["prediction"] = prediction
        st.session_state["shap_df"] = shap_df.to_dict(orient="records")
        st.session_state["user_inputs"] = user_inputs

        st.markdown("---")
        st.success(f"Geschätzte Kosten: **{prediction:,.2f} €** ✅")



    except Exception as e:
        st.error(f"Fehler bei der Vorhersage: {e}")

# === Chatbereich nach der Vorhersage ===
# === Chatbereich nach der Vorhersage ===
if "prediction" in st.session_state:
    st.markdown("---")
    st.header("💬 CostPilot Chat")

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant",
             "content": f"Ich habe eine Kostenschätzung von **{st.session_state['prediction']:,.2f} €** berechnet. Was möchtest du darüber wissen?"}
        ]

    # Show pending loader
    if st.session_state.get("waiting_for_response", False):
        st.info("⏳ KI denkt nach...")

    # Render messages
    for msg in st.session_state["messages"]:
        bubble = "user" if msg["role"] == "user" else "assistant"
        st.markdown(f"<div class='chat-bubble {bubble}'>{msg['content']}</div>", unsafe_allow_html=True)

    # CASE 1: We still need to process a pending message (after rerun)
    if st.session_state.get("waiting_for_response") and "pending_user_message" in st.session_state:
        question = st.session_state.pop("pending_user_message")
        shap_df = pd.DataFrame(st.session_state.get("shap_df"))
        prediction = st.session_state.get("prediction")

        try:
            answer = ask_llm_with_shap(shap_df, prediction, question)
            st.session_state["messages"].append({"role": "assistant", "content": answer})
        except Exception as e:
            st.session_state["messages"].append({
                "role": "assistant",
                "content": f"⚠️ Fehler bei der Analyse: {e}"
            })

        st.session_state["waiting_for_response"] = False
        st.rerun()

    # CASE 2: No pending message → show input box
    user_input = st.text_input(
        "Deine Frage:",
        placeholder="Frag mich etwas zur Vorhersage oder zu den Einflussfaktoren...",
        key="chat_input"
    )

    if user_input:
        # Add user message ONCE
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # Store & trigger processing
        st.session_state["pending_user_message"] = user_input
        st.session_state["waiting_for_response"] = True

        # Clear text input box cleanly
        st.session_state.pop("chat_input", None)
        st.rerun()
