# Icelandic Habitat Classification

Classify Icelandic landscapes from satellite imagery using machine learning 🌋❄️

![Fjallahveravist - Geothermal Alpine Habitat](photo_fjallahveravist.jpg)

*Fjallahveravist (geothermal alpine habitat) – Photo: Náttúrufræðistofnun Íslands*

---

## Overview

Iceland has one of the most detailed habitat mapping systems in the world, with **71 habitat types** (*vistgerðir*) grouped into **13 habitat categories** (*vistlendi*).

This repository contains a **complete, end‑to‑end pipeline** to:

- Train habitat classification models from satellite imagery
- Serve predictions through a FastAPI endpoint
- Submit the model to the competition infrastructure

Each input sample is a **35×35 pixel satellite patch** covering **350×350 meters**, captured by **Sentinel‑2** and enriched with terrain data.

---

## The Challenge

**Goal:** Given a satellite image patch, predict which of the **71 habitat types** it belongs to.

### Input

A numpy array of shape `(15, 35, 35)`:

| Channels | Description |
|--------|-------------|
| 0–11 | Sentinel‑2 spectral bands (coastal aerosol → SWIR) |
| 12 | Elevation (meters) |
| 13 | Slope (degrees) |
| 14 | Aspect (direction) |

### Output

- Integer `0–70` representing the predicted habitat type (*vistgerð*)

### Example Satellite Patch

![Satellite Example](example.png)

---

## Data

The data comes from summer (July–August) satellite imagery over Iceland combined with high‑resolution terrain models.

| Dataset | Samples | Purpose |
|-------|---------|--------|
| Training | 5,186 | Model training |
| Validation | 799 | Local evaluation |
| Test | 1,998 | Final competition score |

Class distributions are preserved across splits.

### Training Files

```
data/train/patches.npy   # (N, 15, 35, 35)
data/train.csv           # labels (vistgerd_idx)
```

Example:

```python
from utils import load_training_data

patches, labels = load_training_data()
print(patches.shape)  # (5186, 15, 35, 35)
print(labels.shape)   # (5186,)
```

### All Habitat Types

![All Classes RGB](all_classes_rgb.png)

---

## Scoring

Models are evaluated using **Weighted F1 Score**:

```
F1_weighted = Σ (n_c / N) × F1_c
```

- `n_c`: samples in class `c`
- `N`: total samples

Baseline (random stratified): **~4% weighted F1**

---

## Repository Structure

```
.
├── api.py                 # FastAPI inference server
├── model.py               # Prediction logic & backend selection
├── train_models.py        # LightGBM training script
├── utils.py               # Data loading & encoding helpers
├── feature_registry.py    # Feature version switch (v36 / v60)
├── features_v36.py        # 36‑feature extractor
├── features_v60.py        # 60‑feature extractor
├── artifacts/             # Saved models & scaler (generated)
├── data/                  # Training data (provided)
└── requirements.txt
```

---

## How the Code Works

### Feature Extraction

Two feature sets are supported:

| Version | Features | Description |
|-------|----------|-------------|
| `v36` | 36 | Mean & std of spectral + terrain |
| `v60` | 60 | Mean, std, min, max of spectral + terrain |

Controlled via environment variable:

```bash
export HABITAT_FEATURES=v60
```

---

### Model Training (LightGBM)

Training produces **three artifacts**:

- `artifacts/scaler.joblib` – StandardScaler
- `artifacts/coarse_model.joblib` – 13‑class habitat group model
- `artifacts/fine_model.joblib` – 71‑class habitat model

Run training:

```bash
HABITAT_FEATURES=v60 python train_models.py
```

This:

1. Loads training patches & labels
2. Extracts features
3. Scales features
4. Trains LightGBM classifiers
5. Saves artifacts to `artifacts/`

---

### Prediction Backends

`model.py` supports **multiple inference backends**:

| Backend | Description |
|-------|-------------|
| `lgbm` | LightGBM (fast, default) |
| `cnn` | ResNet‑18 (Torch + timm) |
| `baseline` | Random weighted baseline |
| `auto` | Chooses best available |

Select manually if needed:

```bash
export HABITAT_BACKEND=lgbm
```

---

### API Server

Predictions are served using **FastAPI**.

Start locally:

```bash
python api.py
```

Server runs at:

```
http://localhost:4321
```

#### Endpoints

| Route | Method | Description |
|-----|-------|-------------|
| `/` | GET | Health check |
| `/api` | GET | API info |
| `/predict` | POST | Habitat prediction |

#### `/predict` Payload

```json
{
  "patch": "<base64-encoded float32 array>"
}
```

Response:

```json
{
  "prediction": 42
}
```

---

## Setup Instructions

### 1. Clone & Install

```bash
git clone <repo-url>
cd habitat-classification

# Using uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Or using pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train Model

```bash
HABITAT_FEATURES=v60 python train_models.py
```

### 3. Run API

```bash
python api.py
```

---

## Submission Workflow

1. Create a VM (AWS / Azure / GCP)
2. Clone this repo on the VM
3. Install requirements
4. Train or upload artifacts
5. Run:

```bash
python api.py
```

6. Submit VM IP + API key in the competition portal

⚠️ **Test set submission allowed only once**

---

## About the Data

- **Sentinel‑2 Level‑2A** surface reflectance
- **Cloud Score Plus** filtering (≤ 0.6)
- **Summer median composite** (July–August 2023–2025)
- **IslandsDEM v1** terrain model

Habitat labels provided by the **Icelandic Institute of Natural History** (*Náttúrufræðistofnun Íslands*).

---

## Good Luck 🇮🇸

May your F1 be high and your models converge fast 🚀

