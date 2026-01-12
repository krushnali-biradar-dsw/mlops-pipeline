from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

app = FastAPI(title="ML Inference API")

# Load model
model = joblib.load("model/model.pkl")

class PredictRequest(BaseModel):
    features: List[float]

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictRequest):
    data = np.array(request.features).reshape(1, -1)
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
