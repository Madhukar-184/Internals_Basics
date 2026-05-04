import json
import pandas as pd
import os

os.makedirs("results", exist_ok=True)

# Load training data
train_df = pd.read_csv("data/training_data.csv")
train_catalyst_mean = train_df["catalyst_concentration"].mean()
train_time_mean = train_df["reaction_time_hours"].mean()

# Load prediction logs
logs = []
with open("logs/predictions.jsonl") as f:
    for line in f:
        logs.append(json.loads(line.strip()))

total_predictions = len(logs)
predictions = [l["prediction"] for l in logs]
mean_prediction = round(sum(predictions) / len(predictions), 4)

live_catalyst = [l["input"]["catalyst_concentration"] for l in logs]
live_time = [l["input"]["reaction_time_hours"] for l in logs]

live_catalyst_mean = round(sum(live_catalyst) / len(live_catalyst), 4)
live_time_mean = round(sum(live_time) / len(live_time), 4)

catalyst_shift = round(abs(live_catalyst_mean - train_catalyst_mean), 4)
time_shift = round(abs(live_time_mean - train_time_mean), 4)

alerts = []
drift_detected = False

if catalyst_shift > 0.25:
    drift_detected = True
    alerts.append({
        "feature": "catalyst_concentration",
        "train_mean": round(train_catalyst_mean, 4),
        "live_mean": live_catalyst_mean,
        "shift": catalyst_shift,
        "threshold": 0.25,
        "status": "ALERT"
    })

if time_shift > 5.35:
    drift_detected = True
    alerts.append({
        "feature": "reaction_time_hours",
        "train_mean": round(train_time_mean, 4),
        "live_mean": live_time_mean,
        "shift": time_shift,
        "threshold": 5.35,
        "status": "ALERT"
    })

step3 = {
    "total_predictions": total_predictions,
    "mean_prediction": mean_prediction,
    "drift_detected": drift_detected,
    "alerts": alerts
}

with open("results/step3_s5.json", "w") as f:
    json.dump(step3, f, indent=2)

print(json.dumps(step3, indent=2))
print("step3_s5.json written.")