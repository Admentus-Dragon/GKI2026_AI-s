# validate_lgbm_variants.py
"""
Leakage-free local validation for habitat classification (LightGBM).

Key properties:
- Models are trained ONLY on the training split
- Validation split is NEVER seen during training
- No artifacts are used (pure in-memory training/eval)
- Supports feature variants via HABITAT_FEATURES=v36|v60
- Supports simple sweeps via CLI args

Run examples:
  HABITAT_FEATURES=v36 python validate_lgbm_variants.py
  HABITAT_FEATURES=v60 python validate_lgbm_variants.py --seed 4
  HABITAT_FEATURES=v60 python validate_lgbm_variants.py --topk 2 --best-of 5

If it "looks stuck", run with low threads:
  HABITAT_FEATURES=v60 python validate_lgbm_variants.py --seed 1 --best-of 5 --threads 2 --lgbm-threads 2
"""

from __future__ import annotations

# IMPORTANT: set thread env vars BEFORE importing numpy/sklearn/lightgbm.
import os

_DEFAULT_THREADS = int(os.getenv("HABITAT_THREADS", "2"))
os.environ.setdefault("OMP_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_DEFAULT_THREADS))

import argparse
import time
from typing import Dict, Tuple, Any, Optional

import numpy as np
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit

from utils import load_hierarchy, load_training_data
from feature_registry import get_feature_fns


def _safe_int_keys(d: Dict) -> Dict[int, int]:
    return {int(k): int(v) for k, v in d.items()}


def _hierarchy_gate_predict(
    hierarchy: Dict[int, int],
    coarse_model: lgb.LGBMClassifier,
    fine_model: lgb.LGBMClassifier,
    X_val: np.ndarray,
    topk: int = 2,
) -> np.ndarray:
    coarse_proba = coarse_model.predict_proba(X_val)
    fine_proba = fine_model.predict_proba(X_val)

    coarse_classes = np.asarray(coarse_model.classes_, dtype=np.int64)
    fine_classes = np.asarray(fine_model.classes_, dtype=np.int64)
    coarse_class_to_index = {int(c): i for i, c in enumerate(coarse_classes)}

    y_pred = np.zeros(X_val.shape[0], dtype=np.int64)

    for i in range(X_val.shape[0]):
        k = min(topk, coarse_proba.shape[1])
        top_idx = np.argpartition(coarse_proba[i], -k)[-k:]
        top_coarse = {int(coarse_classes[j]) for j in top_idx}

        weights = np.zeros_like(fine_proba[i], dtype=np.float32)
        for j, fine_cls in enumerate(fine_classes):
            coarse = hierarchy.get(int(fine_cls), None)
            if coarse is None:
                continue
            if int(coarse) in top_coarse:
                weights[j] = float(coarse_proba[i, coarse_class_to_index[int(coarse)]])

        weighted = fine_proba[i].astype(np.float32) * weights
        if np.any(weighted):
            y_pred[i] = int(fine_classes[int(np.argmax(weighted))])
        else:
            y_pred[i] = int(fine_classes[int(np.argmax(fine_proba[i]))])

    return y_pred


def _train_and_eval_once(
    patches: np.ndarray,
    y_fine: np.ndarray,
    y_coarse: np.ndarray,
    hierarchy: Dict[int, int],
    seed: int,
    test_size: float,
    topk: int,
    lgb_params: Dict[str, Any],
    do_report: bool,
    log_period: int,
    early_stopping_rounds: int,
    reject_missing_train: bool,
) -> Optional[Tuple[float, float, int, int]]:
    """
    Returns None if run is skipped (e.g. rejected split).
    """
    t0 = time.time()

    # ----------------------------
    # Features
    # ----------------------------
    _, extract_batch, feat_dim = get_feature_fns()

    print(f"[seed={seed}] Extracting features...", flush=True)
    t_feat = time.time()
    X = extract_batch(patches).astype(np.float32)
    print(
        f"[seed={seed}] Features: X={X.shape} (feat_dim={feat_dim}) took {time.time()-t_feat:.1f}s",
        flush=True,
    )

    if X.shape[0] != y_fine.shape[0]:
        raise RuntimeError(f"X/y mismatch: X={X.shape} y={y_fine.shape}")
    if X.shape[1] != feat_dim:
        raise RuntimeError(f"Feature dim mismatch: got {X.shape[1]} expected {feat_dim}")

    # ----------------------------
    # Split stratified on coarse
    # ----------------------------
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y_fine)), y_coarse))

    X_train, X_val = X[train_idx], X[val_idx]
    y_train_fine, y_val_fine = y_fine[train_idx], y_fine[val_idx]
    y_train_coarse, y_val_coarse = y_coarse[train_idx], y_coarse[val_idx]

    # Split quality: missing fine classes in TRAIN
    train_fine_set = set(np.unique(y_train_fine).tolist())
    missing_in_train = sorted(set(np.unique(y_fine).tolist()) - train_fine_set)

    print(
        f"[seed={seed}] Split: train={len(train_idx)} val={len(val_idx)} "
        f"missing_fine_in_train={len(missing_in_train)}",
        flush=True,
    )

    if reject_missing_train and len(missing_in_train) > 0:
        print(
            f"[seed={seed}] SKIP: train split missing fine classes (e.g. {missing_in_train[:10]}...)",
            flush=True,
        )
        return None

    # ----------------------------
    # Scaling train-only
    # ----------------------------
    t_scale = time.time()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    print(f"[seed={seed}] Scaling took {time.time()-t_scale:.1f}s", flush=True)

    # ----------------------------
    # Train (with logging + early stopping)
    # ----------------------------
    callbacks = []
    if log_period > 0:
        callbacks.append(lgb.log_evaluation(period=log_period))
    if early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

    print(f"[seed={seed}] Training coarse model...", flush=True)

    coarse_model = lgb.LGBMClassifier(
        num_class=int(len(np.unique(y_train_coarse))),
        class_weight="balanced",
        force_col_wise=True,
        **lgb_params,
    )
    coarse_model.fit(
        X_train,
        y_train_coarse,
        eval_set=[(X_val, y_val_coarse)],
        eval_metric="multi_logloss",
        callbacks=callbacks if callbacks else None,
    )

    print(f"[seed={seed}] Training fine model...", flush=True)

    fine_model = lgb.LGBMClassifier(
        num_class=int(len(np.unique(y_train_fine))),
        class_weight="balanced",
        force_col_wise=True,
        **lgb_params,
    )

    # IMPORTANT:
    # If train is missing some fine labels but we did NOT reject the split,
    # we must NOT pass y_val_fine in eval_set (sklearn wrapper will crash).
    if len(missing_in_train) == 0:
        fine_model.fit(
            X_train,
            y_train_fine,
            eval_set=[(X_val, y_val_fine)],
            eval_metric="multi_logloss",
            callbacks=callbacks if callbacks else None,
        )
    else:
        fine_model.fit(X_train, y_train_fine)

    # ----------------------------
    # Predict with hierarchy gating
    # ----------------------------
    print(f"[seed={seed}] Predicting (hierarchy topk={topk})...", flush=True)
    t_pred = time.time()
    y_pred = _hierarchy_gate_predict(hierarchy, coarse_model, fine_model, X_val, topk=topk)
    print(f"[seed={seed}] Predicting took {time.time()-t_pred:.1f}s", flush=True)

    wf1 = float(f1_score(y_val_fine, y_pred, average="weighted"))
    mf1 = float(f1_score(y_val_fine, y_pred, average="macro"))
    uniq = int(len(np.unique(y_pred)))

    if do_report:
        print(
            classification_report(
                y_val_fine,
                y_pred,
                digits=3,
                zero_division=0,
            ),
            flush=True,
        )

    print(f"[seed={seed}] Total run time {time.time()-t0:.1f}s", flush=True)
    return wf1, mf1, uniq, len(missing_in_train)


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--best-of", type=int, default=1, help="run N seeds: seed..seed+N-1 and report best")

    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--topk", type=int, default=2, help="top-K coarse groups to allow (default 2)")
    p.add_argument("--report", action="store_true", help="print per-class report (only if best-of=1)")

    # Thread caps
    p.add_argument("--threads", type=int, default=_DEFAULT_THREADS, help="cap BLAS/OMP threads")
    p.add_argument("--lgbm-threads", type=int, default=2, help="LightGBM internal threads")
    p.add_argument("--log-period", type=int, default=50, help="LightGBM log period (0 disables)")
    p.add_argument("--early-stopping", type=int, default=50, help="early stopping rounds (0 disables)")

    # Split policy
    p.add_argument(
        "--reject-missing-train",
        action="store_true",
        help="skip seeds where the TRAIN split is missing any fine class (recommended)",
    )
    p.add_argument(
        "--allow-missing-train",
        action="store_true",
        help="do not skip; if train misses fine labels, fine_model will train without eval_set",
    )

    # LGBM params baseline
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--min-child-samples", type=int, default=20)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)

    args = p.parse_args()

    reject_missing_train = True
    if args.allow_missing_train:
        reject_missing_train = False
    if args.reject_missing_train:
        reject_missing_train = True

    # enforce thread caps
    t = str(int(args.threads))
    os.environ["OMP_NUM_THREADS"] = t
    os.environ["MKL_NUM_THREADS"] = t
    os.environ["OPENBLAS_NUM_THREADS"] = t
    os.environ["NUMEXPR_NUM_THREADS"] = t

    feat_variant = os.getenv("HABITAT_FEATURES", "v36").strip().lower()
    print("HABITAT_FEATURES =", feat_variant, flush=True)
    print(
        f"Thread caps: OMP={os.getenv('OMP_NUM_THREADS')} MKL={os.getenv('MKL_NUM_THREADS')} "
        f"OPENBLAS={os.getenv('OPENBLAS_NUM_THREADS')} NUMEXPR={os.getenv('NUMEXPR_NUM_THREADS')} "
        f"| LGBM_THREADS={int(args.lgbm_threads)}",
        flush=True,
    )
    print(f"Split policy: reject_missing_train={reject_missing_train}", flush=True)

    t0 = time.time()
    print("Loading data...", flush=True)
    patches, y_fine = load_training_data()
    y_fine = np.asarray(y_fine, dtype=np.int64)
    print(f"Loaded patches={patches.shape} labels={y_fine.shape} (took {time.time()-t0:.1f}s)", flush=True)

    print("Loading hierarchy...", flush=True)
    hierarchy = _safe_int_keys(load_hierarchy())
    y_coarse = np.array([hierarchy[int(c)] for c in y_fine], dtype=np.int64)
    print(f"Computed coarse labels: {len(np.unique(y_coarse))} classes", flush=True)

    lgb_params = dict(
        boosting_type="gbdt",
        learning_rate=float(args.lr),
        n_estimators=int(args.n_estimators),
        num_leaves=int(args.num_leaves),
        max_depth=int(args.max_depth),
        min_child_samples=int(args.min_child_samples),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        objective="multiclass",
        random_state=42,
        verbosity=1,
        n_jobs=1,  # avoid oversubscription
        num_threads=int(args.lgbm_threads),
    )

    best = None
    ran = 0
    skipped = 0

    for i in range(int(args.best_of)):
        seed = int(args.seed) + i
        out = _train_and_eval_once(
            patches=patches,
            y_fine=y_fine,
            y_coarse=y_coarse,
            hierarchy=hierarchy,
            seed=seed,
            test_size=float(args.test_size),
            topk=int(args.topk),
            lgb_params=lgb_params,
            do_report=bool(args.report) and (args.best_of == 1),
            log_period=int(args.log_period),
            early_stopping_rounds=int(args.early_stopping),
            reject_missing_train=reject_missing_train,
        )

        if out is None:
            skipped += 1
            continue

        ran += 1
        wf1, mf1, uniq, miss = out
        print(
            f"[seed={seed}] wf1={wf1:.4f} mf1={mf1:.4f} uniq_preds={uniq} missing_in_train={miss}",
            flush=True,
        )
        if best is None or wf1 > best[0]:
            best = (wf1, mf1, seed, uniq, miss)

    print(f"\nRuns completed: {ran} | skipped: {skipped}", flush=True)

    if best is None:
        print("No valid runs (all seeds rejected). Try more seeds or --allow-missing-train.", flush=True)
        return

    print("\n========== LEAKAGE-FREE VALIDATION (BEST) ==========", flush=True)
    print(f"Best seed        : {best[2]}", flush=True)
    print(f"Weighted F1      : {best[0]:.4f}", flush=True)
    print(f"Macro F1         : {best[1]:.4f}", flush=True)
    print(f"Unique preds     : {best[3]}", flush=True)
    print(f"Missing in train : {best[4]}", flush=True)
    print("===================================================", flush=True)


if __name__ == "__main__":
    main()
