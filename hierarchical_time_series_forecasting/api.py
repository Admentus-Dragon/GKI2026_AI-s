"""
FastAPI endpoint for hot water demand forecasting.

Usage:
    python api.py

The server will start on http://0.0.0.0:8080
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from model import predict

HOST = "0.0.0.0"
PORT = 8080
HISTORY_LENGTH = 672  # 4 weeks of hourly data
HORIZON = 72  # 3 days ahead
N_SENSORS = 45


class PredictRequest(BaseModel):
    """Request body for /predict endpoint."""
    sensor_history: List[List[float]]  # 672 x 45 array
    timestamp: str  # ISO format datetime
    weather_forecast: Optional[List[List[float]]] = None  # 72 x n_features (optional)
    weather_history: Optional[List[List[float]]] = None  # 672 x n_features (optional)


class PredictResponse(BaseModel):
    """Response body from /predict endpoint."""
    predictions: List[List[float]]  # 72 x 45 array


app = FastAPI(
    title="Hot Water Forecasting API",
    description="Predict hot water demand 72 hours ahead for 45 sensors",
    version="1.0.0"
)


@app.get("/")
def index():
    """Health check endpoint."""
    return {"status": "running", "message": "Hot Water Forecasting API"}


@app.get("/api")
def api_info():
    """API information endpoint."""
    return {
        "service": "hot-water-forecasting",
        "version": "1.0.0",
        "input": {
            "sensor_history": [HISTORY_LENGTH, N_SENSORS],
            "timestamp": "ISO format datetime string",
            "weather_forecast": "optional, (72, n_features)",
            "weather_history": "optional, (672, n_features)"
        },
        "output": {
            "predictions": [HORIZON, N_SENSORS]
        },
        "endpoints": {
            "/": "Health check",
            "/api": "API information",
            "/predict": "POST - Predict future demand"
        }
    }


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    """
    Predict hot water demand for all sensors, 72 hours ahead.

    Input:
        - sensor_history: 672 x 45 array (4 weeks of sensor data)
        - timestamp: ISO format datetime of first forecast hour
        - weather_forecast: 72 x n array (optional)
        - weather_history: 672 x n array (optional)

    Output:
        - predictions: 72 x 45 array
    """
    # Convert to numpy arrays
    sensor_history = np.array(request.sensor_history)

    weather_forecast = None
    if request.weather_forecast is not None:
        weather_forecast = np.array(request.weather_forecast)

    weather_history = None
    if request.weather_history is not None:
        weather_history = np.array(request.weather_history)

    # Validate sensor history shape
    if sensor_history.shape != (HISTORY_LENGTH, N_SENSORS):
        raise HTTPException(
            status_code=400,
            detail=f"Expected sensor_history shape ({HISTORY_LENGTH}, {N_SENSORS}), got {sensor_history.shape}"
        )

    # Get prediction from model
    predictions = predict(
        sensor_history,
        request.timestamp,
        weather_forecast,
        weather_history
    )

    # Validate output shape
    if predictions.shape != (HORIZON, N_SENSORS):
        raise HTTPException(
            status_code=500,
            detail=f"Model returned wrong shape. Expected ({HORIZON}, {N_SENSORS}), got {predictions.shape}"
        )

    return PredictResponse(predictions=predictions.tolist())


if __name__ == "__main__":
    print(f"Starting server on http://{HOST}:{PORT}")
    print(f"Sensor history shape: ({HISTORY_LENGTH}, {N_SENSORS})")
    print(f"Predictions shape: ({HORIZON}, {N_SENSORS})")
    uvicorn.run("api:app", host=HOST, port=PORT, reload=False)
