from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import pandas as pd
from dotenv import load_dotenv

try:
    load_dotenv()
except ImportError:
    pass


llm = ChatOpenAI(
    model="gpt-5-nano"
)


SYSTEM_PROMPT = (
    "Du bist ein KI-Assistent für Kostenanalyse, der ein Kostenprognosemodell (Ridge Regression) erklärt. "
    "Du erhältst eine Modellvorhersage (Preis in €) und SHAP-Werte pro Feature. "
    "Ziel: verständliche, präzise, datenbasierte Erklärung für Ingenieur:innen ohne Data-Science-Hintergrund. "
    "Nutze Fachsprache aus Fertigung/Kostenanalyse, aber ohne Jargon-Overkill. "
    "Nenne keine Annahmen, die nicht aus den Daten ableitbar sind. "
    "Wenn nach Optimierung/Reduktion gefragt wird, gib konkrete Vorschläge basierend auf den einflussstärksten SHAP-Werten. "
    "Antworte ausschließlich auf Basis der gelieferten SHAP-Daten und Frage."
)

HUMAN_TEMPLATE = (
    "Vorhersage (Preis): {prediction} €\n\n"
    "SHAP-Werte (JSON, eine Zeile pro Beobachtung/Feature-Set):\n```json\n{shap_json}\n```\n\n"
    "Nutzerfrage:\n{question}\n\n"
    "Antworte im folgenden Format:\n"
    "1) Kurzfazit (1–2 Sätze)\n"
    "2) Wichtigste Kostentreiber (Top 5; jeweils: Feature, SHAP-Wert, Richtung ↑/↓, kurzer Effekt)\n"
    "3) Begründung (wie SHAP interpretiert wurde; keine Methodenerklärung, nur das Nötigste)\n"
    "4) Handlungsempfehlungen (konkret, priorisiert; falls Reduktionsziel genannt: Bezug zur Prozentzahl)\n"
    "5) Grenzen/Unsicherheiten (max. 2 Bullet Points)\n"
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

