import os
import json
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from dotenv import load_dotenv

try:
    load_dotenv()
except ImportError:
    pass


llm = ChatOpenAI(
    model="gpt-5-nano",      
    temperature=0.2,
    max_retries=2
)


instruction_deutsch = (
    "Du bist ein KI-Assistent, der ein Kostenprognosemodell (Ridge Regression) erklärt. "
    "Du erhältst eine Modellvorhersage (z. B. Preis) und eine Liste mit SHAP-Werten, "
    "die zeigen, welche Merkmale welchen Einfluss auf den Preis hatten. "
    "Deine Aufgabe ist es, diese Werte zu interpretieren und die Nutzerfrage verständlich, "
    "präzise und datenbasiert zu beantworten. "
    "Verwende dabei Fachsprache aus der Fertigung und Kostenanalyse, "
    "aber erkläre sie so, dass auch ein Ingenieur ohne Data-Science-Hintergrund sie versteht. "
    "Wenn nach Optimierung gefragt wird (z. B. 'Wie kann ich Kosten um 10 % senken?'), "
    "gib konkrete Vorschläge auf Basis der einflussreichsten SHAP-Werte."
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", instruction_deutsch),
    (
        "human",
        "Vorhersage: {prediction} €\n\n"
        "SHAP-Werte (Einfluss jedes Features):\n{shap_json}\n\n"
        "Nutzerfrage:\n{question}\n"
        "Bitte antworte klar, fachlich fundiert und datenbasiert."
    ),
])


explain_chain = prompt_template | llm | StrOutputParser()



def ask_llm_with_shap(shap_df: pd.DataFrame, prediction: float, question: str) -> str:
    """
    Nimmt SHAP-Werte + Modellvorhersage + Nutzerfrage und liefert eine
    natürlichsprachliche Antwort vom LLM.
    """

    shap_json = shap_df.to_dict(orient="records")
    shap_json = json.dumps(shap_json, indent=2, ensure_ascii=False)


    answer = explain_chain.invoke({
        "prediction": prediction,
        "shap_json": shap_json,
        "question": question
    })

    return answer.strip()
