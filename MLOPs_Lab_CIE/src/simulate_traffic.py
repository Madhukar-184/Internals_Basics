import requests
import json
import random
import os

os.makedirs("results", exist_ok=True)

BASE_URL = "http://localhost:8000"

# Check health
health = requests.get(f"{BASE_URL}/status").json()
print("Health:", health)

# 40 normal requests (within training distribution)
normal_inputs = []
random.seed(42)
for _ in range(40):
    inp = {
        "reaction_temp_c": round(random.uniform(20, 80), 1),
        "catalyst_concentration": round(random.uniform(0.01, 1.0), 2),
        "reaction_time_hours": round(random.uniform(1, 24), 1),
        "ph_level": round(random.uniform(4, 9), 1)
    }
    normal_inputs.append(inp)

# 10 drifted requests (shifted catalyst_concentration and reaction_time_hours)
drifted_inputs = []
for _ in range(10):
    inp = {
        "reaction_temp_c": round(random.uniform(20, 80), 1),
        "catalyst_concentration": round(min(random.uniform(0.6, 1.0), 1.0), 2),  # higher end but capped at 1.0
        "reaction_time_hours": round(min(random.uniform(18, 24), 24.0), 1),       # higher end but capped at 24
        "ph_level": round(random.uniform(4, 9), 1)
    }
    drifted_inputs.append(inp)

all_inputs = normal_inputs + drifted_inputs
predictions = []

for inp in all_inputs:
    resp = requests.post(f"{BASE_URL}/infer", json=inp)
    if resp.status_code == 200:
        predictions.append(resp.json()["prediction"])
    else:
        print(f"Error: {resp.status_code} for input {inp}")

print(f"Sent {len(predictions)} requests successfully.")

# Test input from question paper
test_input = {
    "reaction_temp_c": 60.0,
    "catalyst_concentration": 0.6,
    "reaction_time_hours": 16.8,
    "ph_level": 6.7
}
test_resp = requests.post(f"{BASE_URL}/infer", json=test_input).json()
print("Test prediction:", test_resp)

step2 = {
    "health_endpoint": "/status",
    "predict_endpoint": "/infer",
    "port": 8000,
    "health_response": health,
    "test_input": test_input,
    "prediction": test_resp["prediction"]
}
with open("results/step2_s4.json", "w") as f:
    json.dump(step2, f, indent=2)

print("step2_s4.json written.")