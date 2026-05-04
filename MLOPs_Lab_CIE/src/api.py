import os
import json
import pickle
import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="BioSynth compound_yield_mg API")

class InputFeatures(BaseModel):
    reaction_temp_c: float = Field(..., ge=20.0, le=80.0)
    catalyst_concentration: float = Field(..., ge=0.01, le=1.0)
    reaction_time_hours: float = Field(..., ge=1.0, le=24.0)
    ph_level: float = Field(..., ge=4.0, le=9.0)

@app.get("/status")
def status():
    return {"alive": True, "service": "BioSynth compound_yield_mg API"}

@app.post("/infer")
def infer(features: InputFeatures):
    X = [[
        features.reaction_temp_c,
        features.catalyst_concentration,
        features.reaction_time_hours,
        features.ph_level
    ]]
    prediction = float(model.predict(X)[0])

    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "input": features.dict(),
        "prediction": round(prediction, 4)
    }
    with open("logs/predictions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"prediction": round(prediction, 4)}

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)