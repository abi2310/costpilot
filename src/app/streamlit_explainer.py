import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

from src.explainability.shap_explainer import explain_with_shap
from src.explainability.llm_interface import ask_llm_with_shap  # <-- neue Funktion mit LangChain

# --- Streamlit Setup ---
st.set_page_config(page_title="CostPilot – Explainable AI", layout="wide")
st.title("💸 CostPilot – Kostenanalyse & Erklärbares ML-Modell")
