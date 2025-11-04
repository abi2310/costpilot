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
    "Du bist ein KI-Assistent für Fertigungskosten. "
    "Du bekommst eine Preisvorhersage und SHAP-Werte. "
    "Ziel: sehr klare, kurze, praxistaugliche Erklärungen für Ingenieure. "
    "Keine Data-Science-Fachsprache, keine Theorie. "
    "Sprich wie ein Kosten-/Fertigungsingenieur.\n\n"

    "Richtlinien:\n"
    "- Maximal 8 Sätze.\n"
    "- Fokus auf: was treibt Kosten, was senkt sie.\n"
    "- Bei SHAP: nur Richtung (↑ teurer / ↓ günstiger) und grobe Stärke.\n"
    "- Empfehlungen müssen realistisch und umsetzbar sein: Materialwahl, Geometrie, Design-Optimierung, Fertigung.\n"
    "- Wenn Nutzer nach Einsparung fragt (z. B. 10 %), gib konkrete Handlungsschritte.\n"
    "- Kein Modell-, SHAP- oder Statistik-Jargon. Keine Formeln.\n"
    "- Nur Informationen nutzen, die direkt aus SHAP kommen.\n"
)

HUMAN_TEMPLATE = (
    "Vorhersage: {prediction} €\n"
    "SHAP:\n```json\n{shap_json}\n```\n"
    "Frage des Nutzers:\n{question}\n\n"

    "Antwortformat:\n"
    "- Kurzfazit (1–2 Sätze)\n"
    "- Top-Kostentreiber (max. 3; Format: Feature – ↑/↓ – kurzer Effekt)\n"
    "- Konkrete Maßnahmen (3 Punkte, einfache Sprache)\n"
    "- Ein Satz zu Unsicherheit/Technik-Machbarkeit\n"
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

