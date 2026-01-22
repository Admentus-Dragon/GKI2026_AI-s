# lgbm_sweep.py
"""
Simple, safe sweep runner for LightGBM settings + feature variants.

What it does:
- Runs leakage-free validation for each config
- Supports HABITAT_FEATURES=v36|v60
- Writes results to sweep_results.jsonl (append) so you can stop/resume
- Keeps the split logic consistent (stratified on coarse)

Run examples:
  HABITAT_FEATURES=v36 python lgbm_sweep.py
  HABITAT_FEATURES=v60 python lgbm_sweep.py --seeds 42,43,44 --topk 2
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, List, Tuple

import numpy as np

from utils import load_hierarchy, load_training_data
from validate_lgbm_variants import _safe_int_keys, _train_and_eval_once


def _grid() -> List[Dict[str, Any]]:
    """
    A small grid focused on leaf/learning-rate stability.
    Keep it small; you can expand later.
    """
    base = dict(
        boosting_type="gbdt",
        objective="multiclass",
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )

    grid = []
    for lr in [0.03, 0.05, 0.07]:
        for leaves in [23, 31, 47]:
            for depth in [10, 12, 14]:
                cfg = dict(base)
                cfg.update(
                    learning_rate=lr,
                    n_estimators=300,          # keep constant initially
                    num_leaves=leaves,
                    max_depth=depth,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                )
                grid.append(cfg)

    # add a couple “slower but stronger” options
    grid.append(dict(base, learning_rate=0.03, n_estimators=600, num_leaves=31, max_depth=12,
                     min_child_samples=20, subsample=0.8, colsample_bytree=0.8))
    grid.append(dict(base, learning_rate=0.02, n_estimators=900, num_leaves=31, max_depth=12,
                     min_child_samples=20, subsample=0.8, colsample_bytree=0.8))
    return grid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="42", help="comma-separated seeds, e.g. 42,43,44")
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--out", type=str, default="sweep_results.jsonl")
    p.add_argument("--max-runs", type=int, default=0, help="0 = run all configs, else stop after N")
    args = p.parse_args()

    feat_variant = os.getenv("HABITAT_FEATURES", "v36").strip().lower()
    print("HABITAT_FEATURES =", feat_variant)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("No seeds provided")

    print("Loading data...")
    patches, y_fine = load_training_data()
    y_fine = np.asarray(y_fine, dtype=np.int64)

    print("Loading hierarchy...")
    hierarchy = _safe_int_keys(load_hierarchy())
    y_coarse = np.array([hierarchy[int(c)] for c in y_fine], dtype=np.int64)

    grid = _grid()
    print(f"Grid size: {len(grid)} configs")
    if args.max_runs and args.max_runs > 0:
        grid = grid[: int(args.max_runs)]
        print(f"Truncated grid to {len(grid)} configs (max-runs)")

    best: Tuple[float, Dict[str, Any], int] | None = None  # (wf1, cfg, seed)

    with open(args.out, "a", encoding="utf-8") as f:
        for idx, cfg in enumerate(grid, 1):
            for seed in seeds:
                wf1, mf1, uniq, miss = _train_and_eval_once(
                    patches=patches,
                    y_fine=y_fine,
                    y_coarse=y_coarse,
                    hierarchy=hierarchy,
                    seed=seed,
                    test_size=float(args.test_size),
                    topk=int(args.topk),
                    lgb_params=cfg,
                    do_report=False,
                )

                row = {
                    "features": feat_variant,
                    "seed": seed,
                    "topk": int(args.topk),
                    "test_size": float(args.test_size),
                    "wf1": float(wf1),
                    "mf1": float(mf1),
                    "uniq_preds": int(uniq),
                    "missing_in_train": int(miss),
                    "params": cfg,
                }
                f.write(json.dumps(row) + "\n")
                f.flush()

                print(
                    f"[{idx:03d}/{len(grid):03d}] seed={seed} "
                    f"wf1={wf1:.4f} mf1={mf1:.4f} uniq={uniq} miss={miss} "
                    f"lr={cfg['learning_rate']} leaves={cfg['num_leaves']} depth={cfg['max_depth']} est={cfg['n_estimators']}"
                )

                if best is None or wf1 > best[0]:
                    best = (wf1, cfg, seed)

    if best is not None:
        wf1, cfg, seed = best
        print("\n========== BEST FOUND ==========")
        print("HABITAT_FEATURES:", feat_variant)
        print("Best seed:", seed)
        print("Best weighted F1:", wf1)
        print("Params:", cfg)
        print("Output log:", args.out)
        print("================================\n")


if __name__ == "__main__":
    main()
