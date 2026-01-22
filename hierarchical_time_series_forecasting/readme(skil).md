# Hot Water Demand Forecasting

Predict how much hot water Reykjavik will need over the next **72 hours** using historical sensor data and optional weather information.

---

## Overview

Reykjavik heats its buildings using geothermal hot water. The distribution network is monitored by **44 flow sensors** plus a **total network flow** sensor.

This repository contains a **full forecasting system** that:

- Builds supervised training data from raw sensor time series
- Constructs leakage-safe baselines and residual targets
- Optionally incorporates weather forecasts and observations
- Trains a deep residual forecasting model
- Serves predictions through a FastAPI endpoint compatible with the competition

---

## The Problem

**Goal:** Given **4 weeks (672 hours)** of historical data, predict the next **72 hours** for **all 45 sensors**.

### Sensors

| Count | Description |
|------|-------------|
| 44 | Individual network flow sensors (M01–M44) |
| 1 | Total network flow (`FRAMRENNSLI_TOTAL`) |

---

## Input Specification

Your model receives:

1. **Sensor history** – `(672, 45)`
2. **Timestamp** – ISO 8601 string for forecast start
3. **Weather forecasts** – optional `(N, 11)` per-station rows
4. **Weather history** – optional `(N, 21)` per-station rows

### Sensor History

Shape: `(672, 45)`

```
       M01    M02   ...   M44   FRAMRENNSLI_TOTAL
t-672  523    891   ...   234   12543
t-671  518    887   ...   231   12489
...
t-1    508    871   ...   225   11023
```

### Timestamp

ISO 8601 format:

```
"2025-01-15T08:00:00"
```

---

## Weather Data (Optional)

Weather data **matches the training CSVs exactly** and is provided **per station**, with no aggregation required by the user.

### Weather Forecasts

Shape: `(≈72 × stations, 11)`

| Column | Description |
|------|-------------|
| date_time | Forecast target time |
| station_id | Station ID |
| temperature | °C |
| windspeed | m/s |
| cloud_coverage | % |
| gust | m/s |
| humidity | % |
| winddirection | Compass |
| dewpoint | °C |
| rain_accumulated | mm |
| value_date | Forecast issue time |

### Weather Observations

Shape: `(N, 21)`

Includes wind, temperature, humidity, pressure, precipitation, and metadata fields.

> **Important:** Weather arrays have variable row counts. Filter by station if needed:

```python
def filter_by_station(weather_history, station_id=1):
    if weather_history is None:
        return None
    return weather_history[weather_history[:, 0] == station_id]
```

---

## Output Specification

Your model must return:

- Shape: `(72, 45)`

```
       M01    M02   ...   M44   FRAMRENNSLI_TOTAL
t+1    510    873   ...   227   11150
t+2    515    878   ...   229   11200
...
t+72   495    858   ...   215   10850
```

---

## Data

### Training Files

| File | Description |
|----|-------------|
| `train.npz` | Supervised training samples |
| `weather_forecasts.csv` | Forecast weather data |
| `weather_observations.csv` | Historical weather data |
| `sensor_timeseries.csv` | Raw hourly sensor data |

Example:

```python
from utils import load_training_data, load_weather_data

X_train, y_train, timestamps, sensors = load_training_data()
print(X_train.shape)  # (N, 672, 45)
print(y_train.shape)  # (N, 72, 45)

wf, wo = load_weather_data()
```

### Data Notes

- Training data may contain missing values
- Validation and test data are clean
- Weather data comes from multiple stations
- Forecasts span up to 10 days ahead

---

## Scoring

Your score is computed as:

```
Score = Σ w_s × (1 - RMSE_model_s / RMSE_baseline_s)
```

Where:

- Baseline = weekly-blend persistence
- `w_s = sqrt(mean_flow_s) / Σ sqrt(mean_flow)`

| Score | Meaning |
|------|--------|
| 0 | Same as baseline |
| > 0 | Better than baseline |
| < 0 | Worse than baseline |
| 1 | Best possible |

---

## Repository Structure

```
.
├── api.py                     # FastAPI inference server
├── model.py                   # Competition-compatible predict() logic
├── model_good.py              # Residual CNN (HistConvResidual)
├── train_histconv_good.py     # Model training script
├── pipeline.py                # End-to-end data & feature pipeline
├── build_train_npz.py         # Build train.npz from raw sensors
├── build_r_hist168_4ch.py     # Residual history features (4-channel)
├── utils.py                   # Data loading & scoring helpers
├── memmaps/                   # Generated features & baselines
├── data/                      # Raw & processed data
└── requirements.txt
```

---

## How the System Works

### 1. Baseline

A **weekly-blend baseline** is used everywhere:

```
Y_base = α × last_72h + (1 − α) × same_72h_last_week
```

This baseline:

- Matches the competition scoring baseline exactly
- Is used both for scoring and for residual learning

---

### 2. Residual Learning

The neural network predicts **residuals** on top of the baseline:

```
Y_hat = Y_base + Residual_model
```

Residual history is encoded using **168-hour convolutional features** per sensor.

---

### 3. Model Architecture

- `HistConvResidual` (PyTorch)
- Per-sensor embeddings
- Horizon embeddings (1–72)
- Weather projections
- Convolutional encoder over residual history
- Outputs normalized residuals `(72, 45)`

---

### 4. Weather Handling

Weather is:

- Aggregated safely by issue time (no leakage)
- Optional at inference
- Robust to mixed dtypes
- Zero-filled if missing or malformed

---

## Getting Started

### 1. Setup

```bash
git clone <repo-url>
cd hot-water-forecasting

# Using uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Or using pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Build Training Data

```bash
python build_train_npz.py
python pipeline.py train-npz
python pipeline.py sensor-residual-memmaps
python pipeline.py weather-hourly-forecast
python pipeline.py weather-forecast-memmap
python pipeline.py weather-obs-summary
python pipeline.py residual-hist-features
python build_r_hist168_4ch.py
```

---

### 3. Train Model

```bash
python train_histconv_good.py
```

This produces:

- `histconv.pt` – trained model
- Normalization statistics
- Baseline parameters

---

### 4. Run API Locally

```bash
python api.py
```

Visit:

```
http://localhost:8080
```

---

## API

### POST /predict

**Request**

```json
{
  "sensor_history": [[...]],
  "timestamp": "2025-01-15T08:00:00",
  "weather_forecast": [[...]],
  "weather_history": [[...]]
}
```

**Response**

```json
{
  "predictions": [[...]]
}
```

---

## Submission

1. Deploy to a VM (AWS / Azure / GCP)
2. Install dependencies
3. Place trained artifacts on disk
4. Run:

```bash
python api.py
```

5. Submit VM IP to the competition portal

---

## Good Luck 🚿🔥

May your fo