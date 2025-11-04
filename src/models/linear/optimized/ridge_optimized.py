import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from src.data.load_data import load_data, split_data

output_dir = "models/linear/optimized"
os.makedirs(output_dir, exist_ok=True)

data = load_data()
data_encoded = pd.get_dummies(data, drop_first=True)

# Daten speichern (nach One-Hot-Encoding)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
output_dir = os.path.join(project_root, "models", "liear", "optimized")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "cleaned_data_one_hot_encoding.csv")
data_encoded.to_csv(output_path, sep=",", index=False)

# Train/Test Split
X_train, X_test, y_train, y_test = split_data(data_encoded, test_size=0.2, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

param_grid = {
    'ridge__alpha': np.logspace(-3, 3, 100),
    'ridge__solver': ['auto', 'cholesky', 'saga']  # effiziente Solver
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("\n Beste Ergebnisse aus Cross Validation:")
print("Bestes R² (CV):", round(grid.best_score_, 4))
print("Beste Parameter:", grid.best_params_)

best_ridge = grid.best_estimator_

y_pred = best_ridge.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Testergebnisse:")
print("MAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("R²:", round(r2, 4))

# Ridge hat kein feature_importances_, sondern coef_
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": best_ridge.named_steps['ridge'].coef_
}).sort_values(by="Importance", ascending=False)

importance_path = os.path.join(output_dir, f"ridge_optimizied_feature_importance.csv")
feature_importance.to_csv(importance_path, index=False)

model_path = os.path.join(output_dir, f"ridge_optimizied_model.pkl")
joblib.dump(best_ridge, model_path)

results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_path": model_path,
    "metrics": {
        "MAE": mae,
        "MSE": mse,
        "R2": r2
    },
    "model_type": "ridge_optimizied_model.pkl"
}

metrics_path = os.path.join(output_dir, f"ridge_optimizied_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(results, f, indent=4)

print("\n🔍 Zusammenfassung:")
print(f"Optimales Modell (Ridge Regression):")
print(f"Alpha = {grid.best_params_['ridge__alpha']}")
print(f"Solver = {grid.best_params_['ridge__solver']}")
print(f"Test-R² = {round(r2, 3)} | MAE = {round(mae, 3)} | MSE = {round(mse, 3)}")
