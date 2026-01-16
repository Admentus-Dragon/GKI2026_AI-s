# Hot Water Demand Forecasting

Predict how much hot water a city will need over the next 3 days.

## The Problem

Reykjavik heats its buildings using geothermal hot water. The distribution network has **44 sensors** measuring flow in different parts of the network, plus a **total flow** measurement.

**Your task:** Given 4 weeks of historical data, predict the next 72 hours for all 45 sensors.

## Input

Your model receives:

1. **Sensor history** - 672 hours (4 weeks) of measurements for 45 sensors
2. **Timestamp** - The datetime of the first hour you need to predict
3. **Weather forecasts** - Forecasts for the next 72 hours (optional)
4. **Weather observations** - Past 672 hours of weather data (optional)

### Sensor History Format

```
Shape: (672, 45)

       M01    M02    M03   ...   M44    FRAMRENNSLI_TOTAL
t-672  523    891    45    ...   234    12543
t-671  518    887    42    ...   231    12489
...
t-2    512    876    48    ...   228    11089
t-1    508    871    51    ...   225    11023  <- most recent
```

### Timestamp Format

ISO 8601 format: `"2025-01-15T08:00:00"`

## Output

Your model returns predictions for all 45 sensors for the next 72 hours:

```
Shape: (72, 45)

       M01    M02    M03   ...   M44    FRAMRENNSLI_TOTAL
t+1    510    873    49    ...   227    11150   <- 1 hour ahead
t+2    515    878    47    ...   229    11200
...
t+71   498    862    52    ...   218    10890
t+72   495    858    54    ...   215    10850   <- 72 hours ahead
```

## Data

### Training Data

| File | Description |
|------|-------------|
| `train.npz` | Sensor history, targets, and timestamps |
| `weather_forecasts.csv` | Weather forecasts (temperature, wind, etc.) |
| `weather_observations.csv` | Historical weather observations |
| `sensor_timeseries.csv` | Raw sensor time series (alternative format) |

```python
from utils import load_training_data, load_weather_data

# Load sensor data
X_train, y_train, timestamps, sensor_names = load_training_data()
print(f"Samples: {X_train.shape[0]}")
print(f"Input: {X_train.shape}")      # (N, 672, 45)
print(f"Output: {y_train.shape}")     # (N, 72, 45)

# Load weather data
weather_forecasts, weather_obs = load_weather_data()
```

### Data Notes

- Training data may contain missing values and quality issues typical of real-world sensor data
- Validation and test sets are cleaned and do not contain missing values
- Weather forecast data starts mid-2022
- Weather data comes from multiple stations

## Scoring

Your score is computed as:

```
Score = weighted_average_over_sensors(1 - RMSE_model / RMSE_baseline)
```

Where:
- `RMSE_model` is your prediction error
- `RMSE_baseline` is the baseline prediction error
- Sensors are weighted by `sqrt(mean_flow) / sum(sqrt(mean_flows))`

| Score | Meaning |
|-------|---------|
| 0 | Same as baseline |
| > 0 | Better than baseline |
| < 0 | Worse than baseline |
| 1 | Perfect prediction |

## Getting Started

### 1. Setup

```bash
git clone <repo-url>
cd hot-water-forecasting

# Using uv
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Or using pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Explore the Data

```python
from utils import load_training_data
import matplotlib.pyplot as plt

X_train, y_train, timestamps, sensors = load_training_data()

# Plot one sample
sample_idx = 0
plt.figure(figsize=(12, 4))
plt.plot(range(-672, 0), X_train[sample_idx, :, -1], label='History')
plt.plot(range(0, 72), y_train[sample_idx, :, -1], label='Target')
plt.axvline(0, color='red', linestyle='--')
plt.xlabel('Hours from prediction time')
plt.ylabel('Flow')
plt.legend()
plt.show()
```

### 3. Develop Your Model

Edit `model.py`. Your `predict()` function receives:
- `sensor_history`: (672, 45) array
- `timestamp`: ISO format string
- `weather_forecast`: (72, n) array (optional)
- `weather_history`: (672, n) array (optional)

And must return a (72, 45) array.

### 4. Test Locally

```bash
python api.py
```

Visit http://localhost:8080 to verify.

### 5. Submit

1. Deploy to a VM (Azure, AWS, etc.)
2. Run `python api.py`
3. Submit your VM's IP to the competition portal

## API Endpoint

POST `/predict`

**Request:**
```json
{
  "sensor_history": [[...], [...], ...],
  "timestamp": "2025-01-15T08:00:00",
  "weather_forecast": [[...], [...], ...],
  "weather_history": [[...], [...], ...]
}
```

**Response:**
```json
{
  "predictions": [[...], [...], ...]
}
```

## File Structure

```
starter_repo/
├── api.py           # API endpoint
├── model.py         # Your model (edit this)
├── utils.py         # Helper functions
├── requirements.txt
├── data/
│   ├── train.npz
│   ├── weather_forecasts.csv
│   └── weather_observations.csv
└── README.md
```

Good luck!
