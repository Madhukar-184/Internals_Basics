import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/training_data.csv")
X = df.drop("compound_yield_mg", axis=1)
y = df["compound_yield_mg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("biosynth-compound-yield-mg")

results = []

# --- Lasso ---
with mlflow.start_run(run_name="Lasso"):
    mlflow.set_tag("project_phase", "model_selection")
    params = {"alpha": 1.0}
    mlflow.log_params(params)

    model = Lasso(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, "model")

    lasso_run_id = mlflow.active_run().info.run_id
    results.append({"name": "Lasso", "mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4), "run_id": lasso_run_id})
    print(f"Lasso  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

# --- RandomForest ---
with mlflow.start_run(run_name="RandomForest"):
    mlflow.set_tag("project_phase", "model_selection")
    params = {"n_estimators": 100, "max_depth": None}
    mlflow.log_params(params)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, "model")

    rf_run_id = mlflow.active_run().info.run_id
    results.append({"name": "RandomForest", "mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4), "run_id": rf_run_id})
    print(f"RandomForest  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

# Pick best by MAE
best = min(results, key=lambda x: x["mae"])
print(f"\nBest model: {best['name']} (MAE={best['mae']})")

# Save best model to disk
if best["name"] == "Lasso":
    best_model = Lasso(alpha=1.0, random_state=42)
else:
    best_model = RandomForestRegressor(n_estimators=100, random_state=42)

best_model.fit(X_train, y_train)
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save best run_id for later tasks
with open("models/best_run_id.txt", "w") as f:
    f.write(f"{best['run_id']}\n{best['name']}\n{best['mae']}")

# Write step1 result
step1 = {
    "experiment_name": "biosynth-compound-yield-mg",
    "models": [{"name": r["name"], "mae": r["mae"], "rmse": r["rmse"], "r2": r["r2"]} for r in results],
    "best_model": best["name"],
    "best_metric_name": "mae",
    "best_metric_value": best["mae"]
}
with open("results/step1_s1.json", "w") as f:
    json.dump(step1, f, indent=2)

print("step1_s1.json written.")