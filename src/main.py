from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(
    title="Customer Conversion API",
    description="API for predicting customer conversion",
    version="1.0"
)

MODEL_PATH = "artifacts/customer_conversion_model.pkl"
SCALER_PATH = "artifacts/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

class CustomerData(BaseModel):
    ads_clicks: int
    time_on_site: float
    pages_visited: int

@app.post("/predict")
def predict(data: CustomerData):
    input_data = np.array([[data.ads_clicks, data.time_on_site, data.pages_visited]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }