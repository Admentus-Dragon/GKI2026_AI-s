"""
Utility functions for habitat classification.
"""

import numpy as np
import pandas as pd
import base64
from pathlib import Path


def decode_patch(encoded: str) -> np.ndarray:
    """
    Decode base64 string to numpy array.

    Args:
        encoded: base64-encoded string representing a (15, 35, 35) float32 array

    Returns:
        numpy array of shape (15, 35, 35)
    """
    patch_bytes = base64.b64decode(encoded)
    patch = np.frombuffer(patch_bytes, dtype=np.float32)
    return patch.reshape(15, 35, 35)


def encode_patch(patch: np.ndarray) -> str:
    """
    Encode numpy array to base64 string.

    Args:
        patch: numpy array of shape (15, 35, 35)

    Returns:
        base64-encoded string
    """
    patch_float32 = patch.astype(np.float32)
    return base64.b64encode(patch_float32.tobytes()).decode("utf-8")


def load_training_data():
    """
    Load training patches and labels.

    Returns:
        patches: numpy array of shape (N, 15, 35, 35)
        labels: numpy array of shape (N,) with class indices 0-70
    """
    data_dir = Path(__file__).parent / "data"
    patches = np.load(data_dir / "train" / "patches.npy")
    labels_df = pd.read_csv(data_dir / "train.csv")
    return patches, labels_df["vistgerd_idx"].values


def load_class_names():
    """
    Load class name mappings.

    Returns:
        dict with 'vistgerd' and 'vistlendi' mappings
    """
    import json
    data_dir = Path(__file__).parent / "data"
    with open(data_dir / "class_names.json", "r") as f:
        return json.load(f)


def load_hierarchy():
    """
    Load hierarchy mapping (vistgerd -> vistlendi).

    Returns:
        dict mapping vistgerd index to vistlendi index
    """
    import json
    data_dir = Path(__file__).parent / "data"
    with open(data_dir / "hierarchy.json", "r") as f:
        return json.load(f)


# -------------------------------------------------------------------
# Features (UPDATED): 60 features total
#   Spectral (0-11): mean, std, min, max  -> 48
#   Terrain  (12-14): mean, std, min, max -> 12
#   Total: 60
# -------------------------------------------------------------------
def extract_features(patch: np.ndarray) -> np.ndarray:
    """
    Extract fast, stable features from a SINGLE patch.

    Input:
        patch: (15, 35, 35)

    Output:
        (60,) float32
    """
    patch = np.asarray(patch, dtype=np.float32)

    if patch.shape != (15, 35, 35):
        raise ValueError(f"Expected patch shape (15, 35, 35), got {patch.shape}")

    spectral = patch[:12]
    spec_mean = spectral.mean(axis=(1, 2))
    spec_std  = spectral.std(axis=(1, 2))
    spec_min  = spectral.min(axis=(1, 2))
    spec_max  = spectral.max(axis=(1, 2))

    terrain = patch[12:15]
    terr_mean = terrain.mean(axis=(1, 2))
    terr_std  = terrain.std(axis=(1, 2))
    terr_min  = terrain.min(axis=(1, 2))
    terr_max  = terrain.max(axis=(1, 2))

    feats = np.concatenate(
        [spec_mean, spec_std, spec_min, spec_max,
         terr_mean, terr_std, terr_min, terr_max]
    )
    return feats.astype(np.float32, copy=False)


def extract_features_batch(patches: np.ndarray) -> np.ndarray:
    """
    Vectorized feature extraction for a BATCH of patches.

    Input:
        patches: (N, 15, 35, 35)

    Output:
        (N, 60) float32
    """
    patches = np.asarray(patches, dtype=np.float32)

    if patches.ndim != 4 or patches.shape[1:] != (15, 35, 35):
        raise ValueError(f"Expected patches shape (N, 15, 35, 35), got {patches.shape}")

    spectral = patches[:, :12]
    spec_mean = spectral.mean(axis=(2, 3))
    spec_std  = spectral.std(axis=(2, 3))
    spec_min  = spectral.min(axis=(2, 3))
    spec_max  = spectral.max(axis=(2, 3))

    terrain = patches[:, 12:15]
    terr_mean = terrain.mean(axis=(2, 3))
    terr_std  = terrain.std(axis=(2, 3))
    terr_min  = terrain.min(axis=(2, 3))
    terr_max  = terrain.max(axis=(2, 3))

    feats = np.concatenate(
        [spec_mean, spec_std, spec_min, spec_max,
         terr_mean, terr_std, terr_min, terr_max],
        axis=1
    )
    return feats.astype(np.float32, copy=False)
