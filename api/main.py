from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from pathlib import Path

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError("Trained model not found. Run training first.")

model = joblib.load(MODEL_PATH)

print("Model path:", MODEL_PATH)

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------

app = FastAPI(
    title="Fraud Detection API",
    description="API for predicting fraudulent credit card transactions",
    version="1.0"
)

# --------------------------------------------------
# Request schema
# --------------------------------------------------

class Transaction(BaseModel):
    features: List[float]

# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(transaction: Transaction):

    features_array = np.array(transaction.features).reshape(1, -1)

    prediction = model.predict(features_array)

    return {
        "fraud_prediction": int(prediction[0])
    }