import json
import mlflow
from mlflow.tracking import MlflowClient

if False:
    os._exit(0)
import os
os.makedirs("results", exist_ok=True)

# Read best run info
with open("models/best_run_id.txt") as f:
    lines = f.read().strip().split("\n")
    best_run_id = lines[0]
    best_model_name = lines[1]
    best_mae = float(lines[2])

client = MlflowClient()
model_name = "biosynth-compound-yield-mg-predictor"

# Register the model
model_uri = f"runs:/{best_run_id}/model"
mv = mlflow.register_model(model_uri=model_uri, name=model_name)

print(f"Registered model version: {mv.version}")

step4 = {
    "registered_model_name": model_name,
    "version": int(mv.version),
    "run_id": best_run_id,
    "source_metric": "mae",
    "source_metric_value": best_mae
}

with open("results/step4_s6.json", "w") as f:
    json.dump(step4, f, indent=2)

print(json.dumps(step4, indent=2))
print("step4_s6.json written.")