"""
Utility functions for hot water demand forecasting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

HISTORY_LENGTH = 672  # 4 weeks of hourly data
HORIZON = 72  # 3 days ahead
N_SENSORS = 45


def load_training_data(
    data_dir: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training data for model development.

    Returns:
        X_train: (N, 672, 45) - sensor history for each sample
        y_train: (N, 72, 45) - target values to predict
        timestamps: (N,) - datetime of first forecast hour for each sample
        sensor_names: (45,) - names of sensors
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    else:
        data_dir = Path(data_dir)

    data = np.load(data_dir / "train.npz", allow_pickle=True)

    return (
        data['X_train'],
        data['y_train'],
        data['timestamps'],
        data['sensor_names']
    )


def load_weather_data(data_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load weather data.

    Returns:
        weather_forecasts: DataFrame with weather forecasts
        weather_observations: DataFrame with weather observations
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    else:
        data_dir = Path(data_dir)

    forecasts = pd.read_csv(data_dir / "weather_forecasts.csv")
    observations = pd.read_csv(data_dir / "weather_observations.csv")

    return forecasts, observations


def compute_baseline_predictions(X: np.ndarray) -> np.ndarray:
    """
    Generate baseline predictions using lag-72.

    Args:
        X: Input history, shape (n_samples, 672, 45) or (672, 45)

    Returns:
        Predictions, shape (n_samples, 72, 45) or (72, 45)
    """
    single_sample = X.ndim == 2

    if single_sample:
        X = X[np.newaxis, :]

    n_samples = X.shape[0]
    predictions = np.zeros((n_samples, HORIZON, N_SENSORS))

    for h in range(HORIZON):
        idx = HISTORY_LENGTH - HORIZON + h
        predictions[:, h, :] = X[:, idx, :]

    if single_sample:
        return predictions[0]

    return predictions


def compute_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray
) -> float:
    """
    Compute the competition score.

    The score is a weighted average of per-sensor skill scores.
    Skill = 1 - (RMSE_model / RMSE_baseline)

    Sensors are weighted by sqrt(mean_flow).

    Args:
        y_true: Ground truth, shape (n_samples, 72, 45)
        y_pred: Predictions, shape (n_samples, 72, 45)
        X: Input history, shape (n_samples, 672, 45)

    Returns:
        Score (0 = baseline performance, higher = better)
    """
    # Generate baseline predictions
    y_baseline = compute_baseline_predictions(X)

    # Flatten to (n_samples * horizon, n_sensors)
    y_true_flat = y_true.reshape(-1, N_SENSORS)
    y_pred_flat = y_pred.reshape(-1, N_SENSORS)
    y_baseline_flat = y_baseline.reshape(-1, N_SENSORS)

    # Compute weights based on sqrt of mean flow
    mean_flows = np.abs(y_true_flat).mean(axis=0)
    sqrt_flows = np.sqrt(mean_flows + 1e-6)
    weights = sqrt_flows / sqrt_flows.sum()

    # Compute skill per sensor
    skills = []
    for s in range(N_SENSORS):
        rmse_model = np.sqrt(np.mean((y_true_flat[:, s] - y_pred_flat[:, s]) ** 2))
        rmse_baseline = np.sqrt(np.mean((y_true_flat[:, s] - y_baseline_flat[:, s]) ** 2))

        if rmse_baseline > 1e-6:
            skill = 1 - rmse_model / rmse_baseline
        else:
            skill = 0.0

        skills.append(skill)

    # Weighted average (clipped to 0 minimum)
    score = max(0, np.sum(np.array(skills) * weights))

    return score


def evaluate_model(predict_fn, X: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Evaluate a prediction function.

    Args:
        predict_fn: Function that takes (672, 45) and returns (72, 45)
        X: Input history, shape (n_samples, 672, 45)
        y_true: Ground truth, shape (n_samples, 72, 45)

    Returns:
        Dictionary with evaluation metrics
    """
    # Generate predictions
    y_pred = np.array([predict_fn(x, "", None, None) for x in X])

    # Compute score
    score = compute_score(y_true, y_pred, X)

    # Overall RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return {
        'score': score,
        'rmse': rmse,
        'n_samples': len(X)
    }
