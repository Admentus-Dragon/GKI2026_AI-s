#!/usr/bin/env python3
"""
Build memmaps/R_hist168_norm.dat as (N,168,S,4):

ch0: weekly_diff = X[t] - X[t-168], normalized by TRAIN-ONLY mean/std (per sensor)
ch1: weekly_blend_resid = X[t] - (alpha*X[t-1] + (1-alpha)*X[t-168]),
     normalized by TRAIN-ONLY mean/std (per sensor)  [FIXED: was using r_mean/r_std]
ch2: level = log1p(X[t]), normalized by TRAIN-ONLY mean/std (per sensor)
ch3: trend24 = X[t] - X[t-24], normalized by TRAIN-ONLY mean/std (per sensor)

Also writes:
- memmaps/X_last.dat  (N,S) where X_last = X[:, 671, :]
- stats npys for ch0,ch1,ch2,ch3
- R_hist168_meta.npy updated with C=4 + descriptions
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

HISTORY_LENGTH = 672
RH_LEN = 168
WEEK_LAG = 168
DAY_LAG = 24


def nanmean_std_over_train(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    arr: (Ntrain, T, S) float32
    returns mean/std per sensor: (S,), (S,)
    """
    x = arr.reshape(-1, arr.shape[-1])  # (Ntrain*T, S)
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = np.nanstd(x, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def load_alpha(mem: Path, N: int, S: int) -> np.ndarray:
    """
    Returns alpha broadcastable to (N,1,S).
    Supports:
      - scalar
      - (S,) per-sensor
      - (N,) per-sample
      - (N,S) per-sample-per-sensor
    """
    p = mem / "alpha.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")

    a = np.load(p).astype(np.float32)

    # scalar
    if a.ndim == 0 or a.size == 1:
        return np.asarray(a).reshape(1, 1, 1).astype(np.float32)

    # per-sensor
    if a.ndim == 1 and a.size == S:
        return a.reshape(1, 1, S)

    # per-sample
    if a.ndim == 1 and a.size == N:
        return a.reshape(N, 1, 1)

    # per-sample-per-sensor
    if a.ndim == 2 and a.shape == (N, S):
        return a.reshape(N, 1, S)

    raise ValueError(f"alpha.npy has unsupported shape {a.shape} (N={N}, S={S})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", type=str, default="data/train.npz")
    ap.add_argument("--memmap-dir", type=str, default="memmaps")
    ap.add_argument(
        "--fit-end",
        type=int,
        default=None,
        help="Raw index fit_end. If omitted, uses 32634.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing R_hist168_norm.dat")
    args = ap.parse_args()

    mem = Path(args.memmap_dir)

    meta = np.load(mem / "meta.npy", allow_pickle=True).item()
    N = int(meta["N"])
    S = int(meta["S"])

    fit_end = int(args.fit_end) if args.fit_end is not None else 32634
    fit_end = max(1, min(fit_end, N))

    data = np.load(args.train_npz, allow_pickle=True)
    X = data["X_train"].astype(np.float32)  # (N,672,S)
    if X.shape != (N, HISTORY_LENGTH, S):
        raise ValueError(f"Expected X shape {(N,HISTORY_LENGTH,S)} got {X.shape}")

    # Save X_last (absolute)
    x_last = X[:, HISTORY_LENGTH - 1, :]  # (N,S)
    xlast_path = mem / "X_last.dat"
    mm_last = np.memmap(xlast_path, dtype="float32", mode="w+", shape=(N, S))
    mm_last[:] = np.nan_to_num(x_last, nan=0.0, posinf=0.0, neginf=0.0)
    mm_last.flush()

    # Indices for last 168 hours: t=504..671
    idx = np.arange(HISTORY_LENGTH - RH_LEN, HISTORY_LENGTH, dtype=np.int64)  # 504..671
    idx_w = idx - WEEK_LAG  # 336..503
    idx_d = idx - DAY_LAG   # 480..647

    alpha = load_alpha(mem, N=N, S=S)  # broadcastable

    # ----- Build raw channels (N,T,S) -----
    x_t = X[:, idx, :]            # (N,T,S)
    x_tm1 = X[:, idx - 1, :]      # (N,T,S)
    x_tw = X[:, idx_w, :]         # (N,T,S)

    # ch0 weekly diff
    ch0_raw = x_t - x_tw

    # ch1 weekly blend residual (X-space)
    blend = alpha * x_tm1 + (1.0 - alpha) * x_tw
    ch1_raw = x_t - blend

    # ch2 level
    ch2_raw = np.log1p(np.clip(x_t, a_min=0.0, a_max=None)).astype(np.float32)

    # ch3 24h trend
    ch3_raw = x_t - X[:, idx_d, :]

    # ----- Train-only stats for ALL channels (including ch1) -----
    train_mask = (np.arange(N, dtype=np.int64) < fit_end)

    ch0_mean, ch0_std = nanmean_std_over_train(ch0_raw[train_mask])
    ch1_mean, ch1_std = nanmean_std_over_train(ch1_raw[train_mask])  # FIXED
    ch2_mean, ch2_std = nanmean_std_over_train(ch2_raw[train_mask])
    ch3_mean, ch3_std = nanmean_std_over_train(ch3_raw[train_mask])

    np.save(mem / "R_hist168_ch0_mean.npy", ch0_mean)
    np.save(mem / "R_hist168_ch0_std.npy", ch0_std)
    np.save(mem / "R_hist168_ch1_mean.npy", ch1_mean)
    np.save(mem / "R_hist168_ch1_std.npy", ch1_std)
    np.save(mem / "R_hist168_ch2_mean.npy", ch2_mean)
    np.save(mem / "R_hist168_ch2_std.npy", ch2_std)
    np.save(mem / "R_hist168_ch3_mean.npy", ch3_mean)
    np.save(mem / "R_hist168_ch3_std.npy", ch3_std)

    # ----- Normalize (force finite) -----
    ch0 = (ch0_raw - ch0_mean[None, None, :]) / ch0_std[None, None, :]
    ch1 = (ch1_raw - ch1_mean[None, None, :]) / ch1_std[None, None, :]  # FIXED
    ch2 = (ch2_raw - ch2_mean[None, None, :]) / ch2_std[None, None, :]
    ch3 = (ch3_raw - ch3_mean[None, None, :]) / ch3_std[None, None, :]

    RH = np.stack([ch0, ch1, ch2, ch3], axis=-1).astype(np.float32)

    # make absolutely sure it’s finite for the model
    RH = np.nan_to_num(RH, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    out_path = mem / "R_hist168_norm.dat"
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists. Re-run with --overwrite to replace it.")

    mm = np.memmap(out_path, dtype="float32", mode="w+", shape=(N, RH_LEN, S, 4))
    mm[:] = RH
    mm.flush()

    hmeta = {
        "N": int(N),
        "T": int(RH_LEN),
        "S": int(S),
        "C": 4,
        "week_lag": int(WEEK_LAG),
        "day_lag": int(DAY_LAG),
        "fit_end": int(fit_end),
        "ch0": "weekly_diff: X[t]-X[t-168], norm=train-only mean/std",
        "ch1": "weekly_blend_resid: X[t]-(alpha*X[t-1]+(1-alpha)*X[t-168]), norm=train-only mean/std",
        "ch2": "level: log1p(X[t]), norm=train-only mean/std",
        "ch3": "trend24: X[t]-X[t-24], norm=train-only mean/std",
        "x_last": "X_last.dat contains X[:,671,:] (absolute)",
    }
    np.save(mem / "R_hist168_meta.npy", hmeta)

    print("[DONE] wrote", out_path, "shape=", (N, RH_LEN, S, 4))
    print("[DONE] wrote", xlast_path, "shape=", (N, S))
    print("[DONE] wrote", mem / "R_hist168_meta.npy")

    print("[STATS] ch0 mean/std:", float(ch0_mean.mean()), float(ch0_std.mean()))
    print("[STATS] ch1 mean/std:", float(ch1_mean.mean()), float(ch1_std.mean()))
    print("[STATS] ch2 mean/std:", float(ch2_mean.mean()), float(ch2_std.mean()))
    print("[STATS] ch3 mean/std:", float(ch3_mean.mean()), float(ch3_std.mean()))


if __name__ == "__main__":
    main()
