"""
Train LightGBM models + scaler artifacts.

Artifacts produced:
- artifacts/scaler.joblib
- artifacts/coarse_model.joblib
- artifacts/fine_model.joblib

Uses feature variant controlled by:
  HABITAT_FEATURES=v36|v60

Run:
  HABITAT_FEATURES=v60 python train_models.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

from utils import extract_features_batch, load_hierarchy, load_training_data


ROOT = Path(__file__).parent
ART_DIR = ROOT / "artifacts"


def _safe_int_keys(d: Dict) -> Dict[int, int]:
    return {int(k): int(v) for k, v in d.items()}


def main() -> None:
    # Optional: cap BLAS threads to avoid oversubscription stalls
    threads = os.getenv("HABITAT_THREADS", "").strip()
    if threads:
        os.environ.setdefault("OMP_NUM_THREADS", threads)
        os.environ.setdefault("MKL_NUM_THREADS", threads)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", threads)
        os.environ.setdefault("NUMEXPR_NUM_THREADS", threads)

    feat_variant = os.getenv("HABITAT_FEATURES", "v36").strip().lower()
    print("HABITAT_FEATURES =", feat_variant)

    print("Loading training data...")
    patches, y_fine = load_training_data()
    y_fine = np.asarray(y_fine, dtype=np.int64)

    print("Loading hierarchy...")
    hierarchy = _safe_int_keys(load_hierarchy())
    y_coarse = np.array([hierarchy[int(c)] for c in y_fine], dtype=np.int64)

    print("Extracting features...")
    X = extract_features_batch(patches).astype(np.float32)

    if X.ndim != 2 or X.shape[0] != y_fine.shape[0]:
        raise ValueError(f"Bad feature matrix shape: X={X.shape}, y={y_fine.shape}")

    print(f"Features: N={X.shape[0]} F={X.shape[1]}")

    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # ------------------------------------------------------------
    # LightGBM configuration (your best-known baseline params)
    # ------------------------------------------------------------
    lgb_common_params = dict(
        boosting_type="gbdt",
        learning_rate=0.05,
        n_estimators=300,
        num_leaves=31,
        max_depth=12,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multiclass",
        n_jobs=int(os.getenv("HABITAT_LGBM_THREADS", "-1")),
        random_state=42,
        verbosity=-1,
        force_col_wise=True,
    )

    print("Training coarse (13-class) LightGBM model...")
    coarse_model = lgb.LGBMClassifier(
        num_class=len(np.unique(y_coarse)),
        class_weight="balanced",
        **lgb_common_params,
    )
    coarse_model.fit(Xs, y_coarse)

    print("Training fine (71-class) LightGBM model...")
    fine_model = lgb.LGBMClassifier(
        num_class=len(np.unique(y_fine)),
        class_weight="balanced",
        **lgb_common_params,
    )
    fine_model.fit(Xs, y_fine)

    ART_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving artifacts to: {ART_DIR}")

    joblib.dump(scaler, ART_DIR / "scaler.joblib")
    joblib.dump(coarse_model, ART_DIR / "coarse_model.joblib")
    joblib.dump(fine_model, ART_DIR / "fine_model.joblib")

    print("Done.")


if __name__ == "__main__":
    main()
