# score_local.py
import argparse
import numpy as np
import hashlib


HISTORY_LENGTH = 672
HORIZON = 72
N_SENSORS = 45

def _timestamps_fingerprint(timestamps) -> str:
    # Stable fingerprint: convert to unicode strings, join with null separators, hash
    ts = np.asarray(timestamps).astype(str)
    blob = ("\0".join(ts.tolist())).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

def baseline_predict(sensor_history: np.ndarray) -> np.ndarray:
    return sensor_history[HISTORY_LENGTH - HORIZON : HISTORY_LENGTH].copy()


def rmse_per_sensor(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    denom = mask.sum(axis=(0, 1)).astype(np.float64)

    yt = np.nan_to_num(y_true, nan=0.0).astype(np.float64)
    yp = np.nan_to_num(y_pred, nan=0.0).astype(np.float64)

    err2 = (yp - yt) ** 2
    num = (err2 * mask).sum(axis=(0, 1)).astype(np.float64)

    mse = np.where(denom > 0, num / denom, np.nan)
    return np.sqrt(mse).astype(np.float64)


def compute_weights(y_true: np.ndarray) -> np.ndarray:
    mean_flow = np.nanmean(y_true, axis=(0, 1))
    w = np.sqrt(np.maximum(mean_flow, 0.0))
    w_sum = np.nansum(w)
    if not np.isfinite(w_sum) or w_sum <= 0:
        return np.ones_like(w) / len(w)
    return (w / w_sum).astype(np.float64)


def score_from_rmses(rmse_model: np.ndarray, rmse_base: np.ndarray, weights: np.ndarray) -> float:
    eps = 1e-8
    valid = np.isfinite(rmse_model) & np.isfinite(rmse_base) & np.isfinite(weights) & (rmse_base > eps)
    if not np.any(valid):
        return float("nan")
    return float(np.sum(weights[valid] * (1.0 - rmse_model[valid] / rmse_base[valid])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=float, default=0.2)
    ap.add_argument("--purge", type=int, default=672)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--model", choices=["baseline", "lowrank"], default="baseline")
    ap.add_argument("--ckpt", type=str, default="lowrank_resid_best.pt")
    ap.add_argument("--memmap-dir", type=str, default="memmaps")
    ap.add_argument("--mode", choices=["residual", "direct"], default="residual")
    args = ap.parse_args()

    data = np.load("data/train.npz", allow_pickle=True)
    weights_path = f"{args.memmap_dir}/score_weights.npy"
    weights = np.load(weights_path).astype(np.float64)
    X_all = data["X_train"].astype(np.float32)  # (N,672,45)
    Y_all = data["y_train"].astype(np.float32)  # (N,72,45)
    timestamps_all = data["timestamps"]
    N0 = len(timestamps_all)

    # filter samples where targets are fully finite
    # keep samples even if partially missing targets; rmse_per_sensor already masks NaNs
    good = np.isfinite(Y_all).all(axis=(1, 2))

    X = X_all[good]
    Y = Y_all[good]
    ts_len = int(len(timestamps_all))
    ts_hash = _timestamps_fingerprint(timestamps_all)

    # ---- load weather forecast memmap aligned to ORIGINAL N0, then apply good mask ----
    wmeta = np.load(f"{args.memmap_dir}/weather_meta.npy", allow_pickle=True).item()

    # Guardrail: verify weather built from same train.npz timestamps
    meta_len = int(wmeta.get("timestamps_len", -1))
    meta_hash = wmeta.get("timestamps_sha256", None)

    if meta_len != -1 and meta_len != ts_len:
        raise ValueError(
            f"Timestamp length mismatch: train.npz has {ts_len}, weather_meta has {meta_len}. "
            f"Rebuild weather memmap."
        )
    if meta_hash is not None and meta_hash != ts_hash:
        raise ValueError(
            f"Timestamp hash mismatch between train.npz and weather_meta. "
            f"Rebuild weather memmap."
    )

    # ---- load weather forecast memmap aligned to ORIGINAL N0, then apply good mask ----
    wmeta = np.load(f"{args.memmap_dir}/weather_meta.npy", allow_pickle=True).item()
    w_dim = int(wmeta["w_dim"])
    Wf_all = np.memmap(
        f"{args.memmap_dir}/W_forecast.dat",
        dtype="float32",
        mode="r",
        shape=(N0, HORIZON, w_dim),
    )
    Wf = Wf_all[good]

    # ---- optional obs summary (aligned the same way) ----
    Wo = None
    try:
        ometa = np.load(f"{args.memmap_dir}/weather_obs_meta.npy", allow_pickle=True).item()
        d_obs = int(ometa["d_obs"])
        Wo_all = np.memmap(
            f"{args.memmap_dir}/W_obs_summary.dat",
            dtype="float32",
            mode="r",
            shape=(N0, d_obs),
        )
        Wo = Wo_all[good]
    except FileNotFoundError:
        Wo = None

    N = len(X)
    if N < 100:
        raise RuntimeError("Too few clean samples after filtering NaNs in targets.")

    # ---- time split with purge ----
    n_val = int(round(N * args.split))
    train_end = N - n_val
    val_start = min(train_end + args.purge, N)

    X_val = X[val_start:]
    Y_val = Y[val_start:]
    Wf_val = Wf[val_start:]
    Wo_val = Wo[val_start:] if Wo is not None else None

    if args.max_samples and args.max_samples > 0:
        X_val = X_val[: args.max_samples]
        Y_val = Y_val[: args.max_samples]
        Wf_val = Wf_val[: args.max_samples]
        if Wo_val is not None:
            Wo_val = Wo_val[: args.max_samples]

    print(f"[SPLIT] N={N} train=[0:{train_end}) purge=[{train_end}:{val_start}) val=[{val_start}:{N})")
    print(f"Eval samples: {len(X_val)} (split={args.split}, purge={args.purge})")

    # baseline predictions
    alpha = np.load(f"{args.memmap_dir}/alpha.npy").astype(np.float32)  # (45,)

    P = X_val[:, HISTORY_LENGTH - HORIZON : HISTORY_LENGTH, :].copy()
    W = X_val[:, HISTORY_LENGTH - HORIZON - 168 : HISTORY_LENGTH - 168, :].copy()

    Yb = alpha[None, None, :] * P + (1.0 - alpha[None, None, :]) * W


    # model predictions
    if args.model == "baseline":
        Ym = Yb
    else:
        from lowrank_infer import LowRankPredictor
        predictor = LowRankPredictor(ckpt_path=args.ckpt, memmap_dir=args.memmap_dir, mode=args.mode)

        BATCH = 512  # try 256, 512, 1024 depending on GPU/CPU memory

        preds = []
        for i0 in range(0, len(X_val), BATCH):
            i1 = min(i0 + BATCH, len(X_val))

            x_b = X_val[i0:i1]       # (B,672,45)
            wf_b = Wf_val[i0:i1]     # (B,72,w_dim)
            wo_b = Wo_val[i0:i1] if Wo_val is not None else None  # (B,d_obs) or None

            yhat_b = predictor.predict_batch(x_b, wf_b, wo_b)     # (B,72,45)
            preds.append(yhat_b)

        Ym = np.concatenate(preds, axis=0)


    # metrics / score
    rmse_base = rmse_per_sensor(Y_val, Yb)
    rmse_model = rmse_per_sensor(Y_val, Ym)
    score = score_from_rmses(rmse_model, rmse_base, weights)

    print(f"Score: {score:.6f}")
    print(f"Mean RMSE baseline: {rmse_base.mean():.3f}")
    print(f"Mean RMSE model:    {rmse_model.mean():.3f}")

    contrib = weights * (1.0 - rmse_model / np.maximum(rmse_base, 1e-8))
    top = np.argsort(-weights)[:10]
    print("\nTop sensors by weight:")
    for s in top:
        print(
            f"  s={s:02d} w={weights[s]:.4f}  "
            f"rmse_base={rmse_base[s]:.2f}  rmse_model={rmse_model[s]:.2f}  contrib={contrib[s]:+.4f}"
        )


if __name__ == "__main__":
    main()
