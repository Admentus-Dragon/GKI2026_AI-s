"""
Model for hot water demand forecasting.

This file contains the predict() function that will be called by the API.
Replace baseline_model() with your own implementation.

Input:
  - sensor_history: (672, 45) array - 4 weeks of hourly sensor data
  - timestamp: datetime string (ISO format) - when the forecast starts
  - weather_forecast: (72, n_features) array - weather forecasts for next 72h (optional)
  - weather_history: (672, n_features) array - weather observations for past 672h (optional)

Output:
  - predictions: (72, 45) array - 3 days of predictions for 45 sensors
"""

import numpy as np
from datetime import datetime
from typing import Optional

HISTORY_LENGTH = 672  # 4 weeks
HORIZON = 72  # 3 days
N_SENSORS = 45


def predict(
    sensor_history: np.ndarray,
    timestamp: str,
    weather_forecast: Optional[np.ndarray] = None,
    weather_history: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Predict hot water demand for all 45 sensors, 72 hours ahead.

    Args:
        sensor_history: (672, 45) array of past sensor readings
        timestamp: ISO format datetime string for the first forecast hour
        weather_forecast: (72, n) array of weather forecasts (optional)
        weather_history: (672, n) array of weather observations (optional)

    Returns:
        (72, 45) array of predictions
    """
    # TODO: Replace with your own model
    return baseline_model(sensor_history)


def baseline_model(sensor_history: np.ndarray) -> np.ndarray:
    """
    Baseline: Use values from 72 hours ago.

    For predicting hour t+h, use the value from hour t+h-72.
    This assumes the pattern from 3 days ago will repeat.
    """
    predictions = np.zeros((HORIZON, N_SENSORS))

    for h in range(HORIZON):
        idx = HISTORY_LENGTH - HORIZON + h
        predictions[h] = sensor_history[idx]

    return predictions
