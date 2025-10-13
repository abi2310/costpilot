from src.data.load_data import load_data, split_data
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import joblib


data = load_data()

data_encoded = pd.get_dummies(data, drop_first=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
output_dir = os.path.join(project_root, "data", "processed")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "cleaned_data_one_hot_encoding.csv")
data_encoded.to_csv(output_path, sep=",", index=False)

X_train, X_test, y_train, y_test = split_data(data_encoded, test_size=0.2, random_state=42)


models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

output_dir = "models/linear"
os.makedirs(output_dir, exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
   
    print(f"\n Ergebnisse {name} Regression:")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    coef_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Coefficient": model.coef_
        }).sort_values(by="Coefficient", ascending=False)
    
    coef_path = os.path.join(output_dir, f"{name.lower()}_coefficients.csv")
    coef_df.to_csv(coef_path, index=False)

    output_dir = "models/linear"
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f"{name.lower()}_model.pkl")
    joblib.dump(model, model_path)

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": model_path,
        "metrics": {
            "MAE": mae,
            "MSE": mse,
            "R2": r2
        },
        "model_type": name
    }

    metrics_path = os.path.join(output_dir, f"{name.lower()}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n Modell & Metriken gespeichert unter: {output_dir}")
