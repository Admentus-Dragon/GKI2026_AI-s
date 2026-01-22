"""Leakage-free local validation for habitat classification.

IMPORTANT:
- Models are trained ONLY on the training split
- Validation split is NEVER seen during training
- Artifacts are NOT used
"""

from __future__ import annotations

from typing import Dict
import numpy as np
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from utils import (
    extract_features,
    load_hierarchy,
    load_training_data,
)

from sklearn.metrics import classification_report


def _safe_int_keys(d: Dict) -> Dict[int, int]:
    return {int(k): int(v) for k, v in d.items()}


def main() -> None:
    print("Loading data...")
    patches, y_fine = load_training_data()
    hierarchy = _safe_int_keys(load_hierarchy())
    y_coarse = np.array([hierarchy[int(c)] for c in y_fine], dtype=np.int64)

    print("Extracting features...")
    X = extract_features(patches).astype(np.float32)

    # ------------------------------------------------------------
    # STRATIFIED SPLIT ON COARSE CLASSES
    # ------------------------------------------------------------
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=42,
    )
    train_idx, val_idx = next(splitter.split(X, y_coarse))

    X_train, X_val = X[train_idx], X[val_idx]
    y_train_fine, y_val_fine = y_fine[train_idx], y_fine[val_idx]
    y_train_coarse = y_coarse[train_idx]

    print(f"Train samples: {len(train_idx)}")
    print(f"Val samples  : {len(val_idx)}")

    # ------------------------------------------------------------
    # SCALING (FIT ON TRAIN ONLY)
    # ------------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # ------------------------------------------------------------
    # LIGHTGBM CONFIG
    # ------------------------------------------------------------
    lgb_params = dict(
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
        force_col_wise=True,
    )

    print("Training coarse model...")
    coarse_model = lgb.LGBMClassifier(
        num_class=len(np.unique(y_train_coarse)),
        class_weight="balanced",
        **lgb_params,
    )
    coarse_model.fit(X_train, y_train_coarse)

    print("Training fine model...")
    fine_model = lgb.LGBMClassifier(
        num_class=len(np.unique(y_train_fine)),
        class_weight="balanced",
        **lgb_params,
    )
    fine_model.fit(X_train, y_train_fine)

    # ------------------------------------------------------------
    # HIERARCHY-AWARE VALIDATION
    # ------------------------------------------------------------
    coarse_proba = coarse_model.predict_proba(X_val)
    fine_proba = fine_model.predict_proba(X_val)

    coarse_classes = coarse_model.classes_
    fine_classes = fine_model.classes_
    coarse_class_to_index = {int(c): i for i, c in enumerate(coarse_classes)}

    y_pred = np.zeros(len(y_val_fine), dtype=np.int64)

    for i in range(len(y_val_fine)):
        top_idx = np.argpartition(coarse_proba[i], -2)[-2:]
        top_coarse = {int(coarse_classes[j]) for j in top_idx}

        weights = np.zeros_like(fine_proba[i], dtype=np.float32)
        for k, fine_cls in enumerate(fine_classes):
            coarse = hierarchy.get(int(fine_cls))
            if coarse in top_coarse:
                weights[k] = coarse_proba[i, coarse_class_to_index[coarse]]

        weighted = fine_proba[i] * weights
        y_pred[i] = (
            fine_classes[np.argmax(weighted)]
            if np.any(weighted)
            else fine_classes[np.argmax(fine_proba[i])]
        )

    print(
        classification_report(
            y_val_fine,
            y_pred,
            digits=3,
            zero_division=0,
        )
    )

    # ------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------
    f1_weighted = f1_score(y_val_fine, y_pred, average="weighted")
    f1_macro = f1_score(y_val_fine, y_pred, average="macro")

    print("\n========== LEAKAGE-FREE VALIDATION ==========")
    print(f"Weighted F1 : {f1_weighted:.4f}")
    print(f"Macro F1    : {f1_macro:.4f}")
    print("============================================\n")


if __name__ == "__main__":
    main()
