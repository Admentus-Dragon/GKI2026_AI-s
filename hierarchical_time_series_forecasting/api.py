"""
FastAPI endpoint for hot water demand forecasting.

Run locally:
  python api.py
Server: http://0.0.0.0:8080
"""
from __future__ import annotations

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional
import numpy as np

from model import predict

HOST = "0.0.0.0"
PORT = 8080

HISTORY_LENGTH = 672
HORIZON = 72
N_SENSORS = 45


class PredictRequest(BaseModel):
    sensor_history: List[List[float]] = Field(..., description="(672,45) hourly sensor values")
    timestamp: str = Field(..., description="ISO datetime string for forecast start")
    weather_forecast: Optional[List[List[Any]]] = Field(None, description="(72, n_features) optional")
    weather_history: Optional[List[List[Any]]] = Field(None, description="(672, n_features) optional")


class PredictResponse(BaseModel):
    predictions: List[List[float]]


app = FastAPI(title="Hot Water Demand Forecast API")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest) -> PredictResponse:
    sensor_history = np.asarray(request.sensor_history, dtype=float)
    if sensor_history.shape != (HISTORY_LENGTH, N_SENSORS):
        raise HTTPException(
            status_code=400,
            detail=f"sensor_history must have shape ({HISTORY_LENGTH}, {N_SENSORS}); got {sensor_history.shape}",
        )

    weather_forecast = None
    if request.weather_forecast is not None:
        weather_forecast = np.asarray(request.weather_forecast)  # mixed dtype is OK

    weather_history = None
    if request.weather_history is not None:
        weather_history = np.asarray(request.weather_history)  # mixed dtype is OK

    preds = predict(
        sensor_history=sensor_history,
        timestamp=request.timestamp,
        weather_forecast=weather_forecast,
        weather_history=weather_history,
    )

    preds = np.asarray(preds, dtype=float)
    if preds.shape != (HORIZON, N_SENSORS):
        raise HTTPException(
            status_code=500,
            detail=f"Model returned wrong shape. Expected ({HORIZON}, {N_SENSORS}), got {preds.shape}",
        )
    if not np.all(np.isfinite(preds)):
        raise HTTPException(status_code=500, detail="Model returned non-finite values (NaN/Inf).")

    return PredictResponse(predictions=preds.tolist())


if __name__ == "__main__":
    print(f"Starting server on http://{HOST}:{PORT}")
    print(f"Sensor history shape: ({HISTORY_LENGTH}, {N_SENSORS})")
    print(f"Predictions shape: ({HORIZON}, {N_SENSORS})")
    uvicorn.run("api:app", host=HOST, port=PORT, reload=False)
