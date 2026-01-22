"""
Model for habitat classification.

Keeps the competition contract:
    predict(patch: np.ndarray) -> int

Backends:
- LightGBM: uses artifacts/scaler.joblib + artifacts/fine_model.joblib
- CNN ResNet18: uses a timm resnet18 checkpoint saved by cnn_torchgeo_train_full_compatible.py

Select with env var:
    HABITAT_BACKEND = "auto" | "lgbm" | "cnn" | "baseline"
Default: auto (prefers lgbm if artifacts exist, else cnn if checkpoint exists, else baseline)

Extra toggles added (safe, does not affect CNN torch.load behavior):
- HABITAT_ART_DIR: path to artifact directory (default: ./artifacts)
- HABITAT_FEATURES: "v36" | "v60" (must match what was used during training)
- HABITAT_CNN_CKPTS: "path1,path2,..." (optional CNN ensemble)
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np

from feature_registry import get_feature_fns

# ------------------------------------------------------------
# Baseline fallback (your original)
# ------------------------------------------------------------
import pandas as pd


ROOT = Path(__file__).parent
ART_DIR = Path(os.getenv("HABITAT_ART_DIR", str(ROOT / "artifacts")))

# You can place your CNN checkpoint anywhere; we also auto-search common spots.
DEFAULT_CNN_SEARCH_DIRS = [
    ROOT,
    ROOT / "runs",
    ROOT / "checkpoints",
]

# -----------------------------
# Lazy globals
# -----------------------------
_BACKEND = None

_LGBM = None
_SCALER = None

_CNN_MODELS = None  # list[torch.nn.Module]
_CNN_MEAN = None
_CNN_STD = None
_CNN_DEVICE = None

# Cache feature fns so we don't re-import on every predict()
_FE_SINGLE = None
_FE_BATCH = None
_FE_DIM = None


def _baseline_model(patch: np.ndarray) -> int:
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    class_counts = train_df["vistgerd_idx"].value_counts(normalize=True).sort_index()
    return int(np.random.choice(class_counts.index.values, p=class_counts.values))


# ------------------------------------------------------------
# Feature registry (LGBM only)
# ------------------------------------------------------------
def _load_feature_fns():
    global _FE_SINGLE, _FE_BATCH, _FE_DIM
    if _FE_SINGLE is not None:
        return
    _FE_SINGLE, _FE_BATCH, _FE_DIM = get_feature_fns()
    # quick sanity
    _ = _FE_SINGLE(np.zeros((15, 35, 35), dtype=np.float32))


# ------------------------------------------------------------
# LightGBM backend
# ------------------------------------------------------------
def _load_lgbm():
    global _LGBM, _SCALER
    if _LGBM is not None and _SCALER is not None:
        return

    import joblib

    _load_feature_fns()

    scaler_path = ART_DIR / "scaler.joblib"
    model_path = ART_DIR / "fine_model.joblib"

    if not scaler_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Missing LightGBM artifacts. Expected:\n"
            f"- {scaler_path}\n"
            f"- {model_path}\n"
            f"Train with: HABITAT_FEATURES=v36|v60 HABITAT_ART_DIR=... python train_lgbm_artifacts.py"
        )

    _SCALER = joblib.load(scaler_path)
    _LGBM = joblib.load(model_path)

    # sanity on feature dim vs scaler
    if hasattr(_SCALER, "n_features_in_") and int(_SCALER.n_features_in_) != int(_FE_DIM):
        raise RuntimeError(
            f"Feature dim mismatch: scaler expects {_SCALER.n_features_in_} features, "
            f"but HABITAT_FEATURES produced {_FE_DIM}. "
            f"Fix by setting HABITAT_FEATURES to match the artifacts used in {ART_DIR}."
        )


def _predict_lgbm(patch: np.ndarray) -> int:
    # IMPORTANT: use the feature variant selected by HABITAT_FEATURES
    _extract_features, _, feat_dim = get_feature_fns()

    feats = _extract_features(patch).astype(np.float32)  # (F,)
    X = feats.reshape(1, -1)

    # Guardrail: make mismatch obvious
    expected = getattr(_SCALER, "n_features_in_", None)
    if expected is not None and X.shape[1] != int(expected):
        raise ValueError(
            f"Feature dim mismatch at inference: got {X.shape[1]} features, "
            f"but scaler expects {int(expected)}. "
            f"Did you train with HABITAT_FEATURES=v60 but run API with v36 (or vice versa)?"
        )

    Xs = _SCALER.transform(X)
    pred = _LGBM.predict(Xs)[0]
    return int(pred)



def _find_cnn_checkpoints() -> list[Path]:
    """
    Return a list of CNN checkpoints for ensembling.

    Priority:
    1) HABITAT_CNN_CKPTS="path1,path2,..."  (must all exist)
    2) HABITAT_CNN_CKPT="path"             (single)
    3) auto-search                          (single most recent)
    """
    env_paths = os.getenv("HABITAT_CNN_CKPTS", "").strip()
    if env_paths:
        paths = [Path(x.strip()) for x in env_paths.split(",") if x.strip()]
        if not paths:
            raise ValueError("HABITAT_CNN_CKPTS was set but no valid paths were parsed.")
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"HABITAT_CNN_CKPTS contains missing files: {missing}")
        return paths

    # Fallback to the existing single-checkpoint behavior
    return [_find_cnn_checkpoint()]


# ------------------------------------------------------------
# CNN backend (timm ResNet18)
# ------------------------------------------------------------
def _find_cnn_checkpoint() -> Path:
    # explicit override
    env_path = os.getenv("HABITAT_CNN_CKPT", "").strip()
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"HABITAT_CNN_CKPT set but file not found: {p}")

    # auto: prefer EMA then non-EMA if present in common dirs
    candidates: list[Path] = []
    for d in DEFAULT_CNN_SEARCH_DIRS:
        if not d.exists():
            continue
        candidates += list(d.rglob("best_resnet18_ema.pth"))
        candidates += list(d.rglob("best_resnet18.pth"))
        candidates += list(d.rglob("final_resnet18_ema.pth"))
        candidates += list(d.rglob("final_resnet18.pth"))

    if candidates:
        # pick most recently modified
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    raise FileNotFoundError(
        "Could not find CNN checkpoint. Expected something like final_resnet18_ema.pth or final_resnet18.pth.\n"
        "Either set HABITAT_CNN_CKPT=/path/to/final_resnet18_ema.pth\n"
        "or place it under ./runs/... as produced by cnn_torchgeo_train_full_compatible.py"
    )


def _load_cnn():
    global _CNN_MODELS, _CNN_MEAN, _CNN_STD, _CNN_DEVICE

    if _CNN_MODELS is not None:
        return

    import torch
    import timm

    ckpt_paths = _find_cnn_checkpoints()  # list[Path]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []

    mean = None
    std = None
    in_chans = None
    num_classes = None

    for p in ckpt_paths:
        # KEEP your safe load setting exactly
        ckpt = torch.load(p, map_location="cpu", weights_only=False)

        # Read meta once and enforce consistency across ensemble
        this_in_chans = int(ckpt.get("in_chans", 13))
        this_num_classes = int(ckpt.get("num_classes", 71))

        if in_chans is None:
            in_chans = this_in_chans
            num_classes = this_num_classes
            mean = np.asarray(ckpt["mean"], dtype=np.float32).reshape(1, -1, 1, 1)
            std = np.asarray(ckpt["std"], dtype=np.float32).reshape(1, -1, 1, 1)
            std = np.clip(std, 1e-6, None)
        else:
            if this_in_chans != in_chans or this_num_classes != num_classes:
                raise RuntimeError(
                    "CNN ensemble checkpoints must match in_chans/num_classes. "
                    f"Got {p} with in_chans={this_in_chans}, num_classes={this_num_classes}, "
                    f"expected in_chans={in_chans}, num_classes={num_classes}."
                )

        m = timm.create_model("resnet18", in_chans=in_chans, num_classes=num_classes)
        m.load_state_dict(ckpt["state_dict"], strict=True)
        m.eval()
        m.to(device)
        models.append(m)

    _CNN_MODELS = models
    _CNN_DEVICE = device
    _CNN_MEAN = torch.tensor(mean, device=device)
    _CNN_STD = torch.tensor(std, device=device)

    if len(ckpt_paths) == 1:
        print(f"[model.py] Loaded CNN checkpoint: {ckpt_paths[0]} on {device}")
    else:
        print(f"[model.py] Loaded CNN ensemble ({len(ckpt_paths)} models) on {device}:")
        for p in ckpt_paths:
            print(f"  - {p}")


def _prep_cnn_input(patch: np.ndarray):
    """
    patch: (15,35,35)
    CNN expects (1,13,35,35): 12 spectral + dummy
    """
    import torch

    patch = np.asarray(patch, dtype=np.float32)

    x = patch[:12]  # (12,35,35)
    z = np.zeros((1, 35, 35), dtype=np.float32)
    x = np.concatenate([x, z], axis=0)  # (13,35,35)

    x = torch.from_numpy(x).unsqueeze(0)  # (1,13,35,35)
    return x


def _predict_cnn(patch: np.ndarray) -> int:
    import torch

    x = _prep_cnn_input(patch).to(_CNN_DEVICE)

    # normalize + safety (matches your training script intent)
    x = (x - _CNN_MEAN) / _CNN_STD
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp(-10.0, 10.0)

    with torch.no_grad():
        # Average logits across ensemble (or single model)
        logits_sum = None
        for m in _CNN_MODELS:
            out = m(x)
            logits_sum = out if logits_sum is None else (logits_sum + out)
        logits = logits_sum / float(len(_CNN_MODELS))

        pred = int(torch.argmax(logits, dim=1).item())
    return pred


# ------------------------------------------------------------
# Backend selection
# ------------------------------------------------------------
def _select_backend() -> str:
    """
    Returns one of: 'lgbm', 'cnn', 'baseline'
    """
    forced = os.getenv("HABITAT_BACKEND", "auto").strip().lower()

    if forced in ("lgbm", "cnn", "baseline"):
        return forced

    # auto
    if (ART_DIR / "scaler.joblib").exists() and (ART_DIR / "fine_model.joblib").exists():
        return "lgbm"

    # if any cnn checkpoint exists
    try:
        _ = _find_cnn_checkpoint()
        return "cnn"
    except Exception:
        return "baseline"


def predict(patch: np.ndarray) -> int:
    """
    Predict habitat class for a single patch (15,35,35).
    """
    global _BACKEND

    if _BACKEND is None:
        _BACKEND = _select_backend()
        print(f"[model.py] Using backend: {_BACKEND}")

        if _BACKEND == "lgbm":
            _load_lgbm()
        elif _BACKEND == "cnn":
            _load_cnn()

    if _BACKEND == "lgbm":
        return _predict_lgbm(patch)
    if _BACKEND == "cnn":
        return _predict_cnn(patch)

    return _baseline_model(patch)
