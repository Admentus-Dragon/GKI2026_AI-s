from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

from utils import load_hierarchy, load_training_data
from feature_registry import get_feature_fns

ROOT = Path(__file__).parent

def _safe_int_keys(d: Dict) -> Dict[int, int]:
    return {int(k): int(v) for k, v in d.items()}

def main() -> None:
    # Variant directories
    feat_variant = os.getenv("HABITAT_FEATURES", "v36").strip().lower()
    out_dir = os.getenv("HABITAT_ART_DIR", str(ROOT / "artifacts" / feat_variant))
    ART_DIR = Path(out_dir)

    print("Loading training data...")
    patches, y_fine = load_training_data()
    y_fine = np.asarray(y_fine, dtype=np.int64)

    print("Loading hierarchy...")
    hierarchy = _safe_int_keys(load_hierarchy())
    y_coarse = np.array([hierarchy[int(c)] for c in y_fine], dtype=np.int64)

    extract_features, extract_features_batch, F = get_feature_fns()

    print(f"Extracting features ({feat_variant}) ...")
    X = extract_features_batch(patches).astype(np.float32)
    if X.ndim != 2 or X.shape[0] != len(y_fine):
        raise ValueError(f"Bad features shape: X={X.shape}, y={y_fine.shape}")
    if X.shape[1] != F:
        raise ValueError(f"Feature dim mismatch: expected F={F}, got {X.shape[1]}")

    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

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
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )

    print("Training coarse (13-class) LightGBM model...")
    coarse_model = lgb.LGBMClassifier(
        num_class=len(np.unique(y_coarse)),
        class_weight="balanced",
        force_col_wise=True,
        **lgb_common_params,
    )
    coarse_model.fit(Xs, y_coarse)

    print("Training fine (71-class) LightGBM model...")
    fine_model = lgb.LGBMClassifier(
        num_class=len(np.unique(y_fine)),
        class_weight="balanced",
        force_col_wise=True,
        **lgb_common_params,
    )
    fine_model.fit(Xs, y_fine)

    ART_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving artifacts to: {ART_DIR}")

    joblib.dump(scaler, ART_DIR / "scaler.joblib")
    joblib.dump(coarse_model, ART_DIR / "coarse_model.joblib")
    joblib.dump(fine_model, ART_DIR / "fine_model.joblib")

    # Save metadata so you know what’s inside
    meta = {
        "features": feat_variant,
        "feature_dim": int(F),
        "params": lgb_common_params,
    }
    joblib.dump(meta, ART_DIR / "meta.joblib")

    print("Done.")

if __name__ == "__main__":
    main()
