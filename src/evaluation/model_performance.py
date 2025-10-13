import json, glob, pandas as pd

rows = []
for file in glob.glob("models/**/ *_metrics.json", recursive=True):
    with open(file, "r") as f:
        data = json.load(f)
        row = {
            "Model": data["model_type"],
            "MAE": data["metrics"]["MAE"],
            "MSE": data["metrics"]["MSE"],
            "R2": data["metrics"]["R2"],
            "Timestamp": data["timestamp"]
        }
        rows.append(row)

df = pd.DataFrame(rows).sort_values(by="R2", ascending=False)
print(df)
