from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
try:
    load_dotenv()
except ImportError:
    pass


llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=1
)


SYSTEM_PROMPT = (
    "Du bist ein KI-Assistent für Fertigungskosten. "
    "Du bekommst IMMER eine Preisvorhersage und SHAP-Werte. "
    "Deine Aufgabe ist es, wie ein erfahrener Fertigungs- oder Kosteningenieur zu sprechen – "
    "klar, praxisnah und umsetzungsorientiert.\n\n"

    "Du erkennst automatisch, was der Nutzer will:\n"
    "- Wenn er nach Gründen, Einsparungen oder Kostentreibern fragt → gib eine konkrete Kostenanalyse basierend auf den SHAP-Werten.\n"
    "- Wenn er allgemein über Fertigung, Materialien oder Konstruktion spricht → verhalte dich wie ein normaler Ingenieur-Kollege.\n"
    "- Wenn er beides mischt, erkläre kurz die Kostenseite und geh dann auf die restliche Frage ein.\n\n"

    "Richtlinien für SHAP-basierte Antworten:\n"
    "- Maximal 8 Sätze.\n"
    "- Kein Data-Science-, Modell- oder Statistik-Jargon.\n"
    "- Nur Informationen nutzen, die direkt aus den SHAP-Werten kommen.\n"
    "- Sprich wie in einer technischen Besprechung: klar, direkt, mit Zahlen oder Maßnahmen.\n"
    "- Gib realistische, messbare Empfehlungen (z. B. 'Materialfläche um 10 % reduzieren' oder 'Fräszeit durch einfachere Geometrie senken').\n"
    "- Wenn der Nutzer ein Einsparziel nennt (z. B. 10 %), rechne gedanklich rückwärts und gib konkrete Handlungsoptionen.\n\n"

    "Richtlinien für normale Gespräche:\n"
    "- Sei fachlich, aber locker – wie ein Kollege mit Produktionserfahrung.\n"
    "- Du darfst Rückfragen stellen oder Ideen diskutieren.\n"
    "- Antworte verständlich, ohne unnötige Theorie.\n"
)

HUMAN_TEMPLATE = (
    "Vorhersage: {prediction} €\n"
    "SHAP:\n```json\n{shap_json}\n```\n"
    "Frage des Nutzers:\n{question}\n\n"
    "Antwortformat:\n"
    "- Kurzfazit (1–2 Sätze)\n"
    "- Konkrete Maßnahmen oder Antwort in Alltagssprache (je nach Fragetyp)"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_TEMPLATE),
])

explain_chain = prompt | llm | StrOutputParser()




def ask_llm_with_shap(shap_df: pd.DataFrame, prediction: float, question: str) -> str:
    """
    Übergibt SHAP-Werte + Modellvorhersage + Nutzerfrage an das LLM und gibt die Antwort zurück.
    """
    shap_json = shap_df.to_dict(orient="records")
    shap_json_str = json.dumps(shap_json, indent=2, ensure_ascii=False)
    print("SHAP JSON:", shap_json_str)  # Debug-Ausgabe
    print("Prediction:", prediction)  # Debug-Ausgabe
    print("Question:", question)  # Debug-Ausgabe   
    response = explain_chain.invoke({
        "prediction": prediction,
        "shap_json": shap_json_str,
        "question": question
    })
    print(response)

    return response.strip()

