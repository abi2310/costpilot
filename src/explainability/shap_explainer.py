import joblib
import shap
import pandas as pd

def explain_with_shap(model_path: str, input_df: pd.DataFrame):
    model = joblib.load(model_path)
    explainer = shap.Explainer(model, input_df)
    shap_values = explainer(input_df)
    shap_df = pd.DataFrame({
        "Feature": input_df.columns,
        "Wert": input_df.iloc[0].values,
        "SHAP-Wert": shap_values.values[0]
    }).sort_values("SHAP-Wert", key=abs, ascending=False)
    return shap_df, shap_values, explainer
