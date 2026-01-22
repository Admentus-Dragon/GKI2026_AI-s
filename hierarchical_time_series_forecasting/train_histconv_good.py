#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from model_good import HistConvResidual


HORIZON = 72
RH_LEN = 168

# Set to 2 if your R_hist168_norm.dat is (N,168,S,2); else 1 for (N,168,S)
RH_CHANNELS = 4


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Loss + scoring helpers
# -------------------------
def weighted_masked_mse(y_hat: torch.Tensor, y: torch.Tensor, w_s: torch.Tensor) -> torch.Tensor:
    """
    y_hat, y: (B,H,S) in normalized residual space
    w_s: (S,)
    """
    mask = torch.isfinite(y)
    y0 = torch.nan_to_num(y, nan=0.0)
    yhat0 = torch.nan_to_num(y_hat, nan=0.0)

    diff2 = (yhat0 - y0) ** 2
    diff2 = diff2 * mask
    diff2 = diff2 * w_s.view(1, 1, -1)

    denom = (mask * w_s.view(1, 1, -1)).sum().clamp_min(1e-8)
    return diff2.sum() / denom


def resid_norm_to_raw(r_norm: torch.Tensor, r_mean: torch.Tensor, r_std: torch.Tensor) -> torch.Tensor:
    return r_norm * r_std.view(1, 1, -1) + r_mean.view(1, 1, -1)


def accum_sse_counts(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Global RMSE per sensor, accumulated over batches.
    """
    mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
    t0 = torch.nan_to_num(y_true, nan=0.0)
    p0 = torch.nan_to_num(y_pred, nan=0.0)
    err2 = (p0 - t0) ** 2
    err2 = err2 * mask
    sse = err2.sum(dim=(0, 1))
    cnt = mask.sum(dim=(0, 1)).to(err2.dtype)
    return sse, cnt


def score_from_rmses_torch(rmse_model: torch.Tensor, rmse_base: torch.Tensor, weights: torch.Tensor) -> float:
    eps = 1e-8
    valid = torch.isfinite(rmse_model) & torch.isfinite(rmse_base) & torch.isfinite(weights) & (rmse_base > eps)
    if not torch.any(valid):
        return float("nan")
    return float(torch.sum(weights[valid] * (1.0 - rmse_model[valid] / rmse_base[valid])).detach().cpu())


def compute_weights_from_train_loader(dl_train, r_mean, r_std, device, S):
    """
    Competition weights computed from TRAIN ONLY in absolute units:
      w_s = sqrt(mean_flow_s) / sum sqrt(mean_flow_s)
    """
    sum_flow = torch.zeros((S,), device=device)
    cnt_flow = torch.zeros((S,), device=device)

    with torch.no_grad():
        for _w_in, r_norm, _rh, y_base, _yb_s_norm in dl_train:
            r_norm = r_norm.to(device, non_blocking=True)
            y_base = y_base.to(device, non_blocking=True)

            r_true_raw = resid_norm_to_raw(r_norm, r_mean, r_std)
            y_true = y_base + r_true_raw

            mask = torch.isfinite(y_true)
            y0 = torch.nan_to_num(y_true, nan=0.0)

            sum_flow += (y0 * mask).sum(dim=(0, 1))
            cnt_flow += mask.sum(dim=(0, 1)).to(sum_flow.dtype)

    mean_flow = sum_flow / torch.clamp(cnt_flow, min=1.0)
    w = torch.sqrt(torch.clamp(mean_flow, min=0.0))
    w = w / torch.clamp(w.sum(), min=1e-12)
    return w



class DS(Dataset):
    """
    Returns:
      w_in      : (H, w_in_dim) float32
      r_norm    : (H, S) float32  (target residuals, normalized)
      rh        : (RH_LEN, S, C) float32 (history features)
      y_base    : (H, S) float32  (baseline, absolute)
      yb_s_norm : (H, S) float32  (per-sensor baseline feature, normalized train-only)

    Indexing:
      - All memmaps are aligned to RAW row index 0..N-1
      - DS filters to "good rows" where R_norm is fully finite
      - IMPORTANT: time features are derived from start_hour/start_dow saved by pipeline.py,
        not from (row % 24) assumptions.
    """

    def __init__(
        self,
        memmap_dir: str,
        *,
        add_time_feats: bool = True,
        add_jump_feats: bool = True,
        yb_feat_mean: np.ndarray | None = None,     # (3,)
        yb_feat_std: np.ndarray | None = None,      # (3,)
        yb_s_mean: np.ndarray | None = None,        # (S,)
        yb_s_std: np.ndarray | None = None,         # (S,)
    ):
        super().__init__()

        meta = np.load(f"{memmap_dir}/meta.npy", allow_pickle=True).item()
        wmeta = np.load(f"{memmap_dir}/weather_meta.npy", allow_pickle=True).item()
        ometa = np.load(f"{memmap_dir}/weather_obs_meta.npy", allow_pickle=True).item()
        hmeta = np.load(f"{memmap_dir}/R_hist168_meta.npy", allow_pickle=True).item()

        self.H = int(meta["H"])
        self.S = int(meta["S"])
        self.N0 = int(meta["N"])

        self.w_dim = int(wmeta["w_dim"])
        self.d_obs = int(ometa["d_obs"])

        assert self.H == HORIZON, f"Expected H={HORIZON}, got {self.H}"
        assert int(hmeta["T"]) == RH_LEN and int(hmeta["S"]) == self.S, f"Bad RH meta: {hmeta}"

        # RH channels must match file
        file_C = int(hmeta.get("C", 1))
        if file_C != RH_CHANNELS:
            raise ValueError(f"RH_CHANNELS={RH_CHANNELS} but R_hist168_meta.npy says C={file_C}")

        self.add_time_feats = bool(add_time_feats)
        self.add_jump_feats = bool(add_jump_feats)

        # dims of extra features
        self.extra_yb_feats = 3
        self.extra_time_feats = 5 if self.add_time_feats else 0
        self.extra_jump_feats = 2 if self.add_jump_feats else 0

        self.w_in_dim = self.w_dim + self.d_obs + self.extra_yb_feats + self.extra_time_feats + self.extra_jump_feats

        # ----- memmaps -----
        self.Wf = np.memmap(
            f"{memmap_dir}/W_forecast.dat",
            dtype="float32",
            mode="r",
            shape=(self.N0, self.H, self.w_dim),
        )
        self.Wo = np.memmap(
            f"{memmap_dir}/W_obs_summary.dat",
            dtype="float32",
            mode="r",
            shape=(self.N0, self.d_obs),
        )
        self.RH = np.memmap(
            f"{memmap_dir}/R_hist168_norm.dat",
            dtype="float32",
            mode="r",
            shape=(self.N0, RH_LEN, self.S, RH_CHANNELS),
        )
        self.R = np.memmap(
            f"{memmap_dir}/R_norm.dat",
            dtype="float32",
            mode="r",
            shape=(self.N0, self.H, self.S),
        )
        self.Yb = np.memmap(
            f"{memmap_dir}/Y_base.dat",
            dtype="float32",
            mode="r",
            shape=(self.N0, self.H, self.S),
        )
        self.X_last = np.memmap(
            f"{memmap_dir}/X_last.dat",
            dtype="float32",
            mode="r",
            shape=(self.N0, self.S),
        )

        self.r_mean = np.load(f"{memmap_dir}/r_mean.npy").astype(np.float32)
        self.r_std = np.load(f"{memmap_dir}/r_std.npy").astype(np.float32)

        # --- REAL time features saved by pipeline.py ---
        # start_hour/start_dow align each raw row index to the true wall-clock time.
        # If missing (older memmaps), we fall back to synthetic row%24 features.
        try:
            self.start_hour = np.load(f"{memmap_dir}/start_hour.npy").astype(np.int16, copy=False)
            self.start_dow  = np.load(f"{memmap_dir}/start_dow.npy").astype(np.int16, copy=False)
        except FileNotFoundError:
            self.start_hour = None
            self.start_dow = None

        # train-only normalization params passed from training script
        self.yb_feat_mean = yb_feat_mean.astype(np.float32) if yb_feat_mean is not None else None
        self.yb_feat_std = yb_feat_std.astype(np.float32) if yb_feat_std is not None else None
        self.yb_s_mean = yb_s_mean.astype(np.float32) if yb_s_mean is not None else None
        self.yb_s_std = yb_s_std.astype(np.float32) if yb_s_std is not None else None

        # Filter to “good” rows based on residual memmap (authoritative)
        good = np.isfinite(self.R).all(axis=(1, 2))
        self._idx = np.nonzero(good)[0].astype(np.int64)
        self.ds_raw_idx = self._idx  # raw row index for each kept sample
        self.N = len(self._idx)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, i: int):
        row = int(self._idx[i])  # raw memmap row index == time index
        raw_t0 = row

        wf = self.Wf[row]  # (H,w_dim)
        wo = self.Wo[row]  # (d_obs,)
        woH = np.broadcast_to(wo[None, :], (self.H, wo.shape[0])).astype(np.float32, copy=False)

        y_base = np.nan_to_num(self.Yb[row], nan=0.0).astype(np.float32, copy=False)      # (H,S)
        x_last = np.nan_to_num(self.X_last[row], nan=0.0).astype(np.float32, copy=False) # (S,)

        # ----- baseline summary feats (H,3) -----
        yb_mean = np.mean(y_base, axis=1, keepdims=True).astype(np.float32, copy=False)
        yb_std  = np.std(y_base, axis=1, keepdims=True).astype(np.float32, copy=False)
        topk = np.sort(y_base, axis=1)[:, -5:]
        yb_top5 = np.mean(topk, axis=1, keepdims=True).astype(np.float32, copy=False)

        yb_feats = np.concatenate([yb_mean, yb_std, yb_top5], axis=1).astype(np.float32, copy=False)  # (H,3)
        if (self.yb_feat_mean is not None) and (self.yb_feat_std is not None):
            yb_feats = (yb_feats - self.yb_feat_mean[None, :]) / self.yb_feat_std[None, :]

        feat_list = [wf, woH, yb_feats]

        # ----- time feats (H,5) -----
        if self.add_time_feats:
            h = np.arange(self.H, dtype=np.int64)

            if (getattr(self, "start_hour", None) is not None) and (getattr(self, "start_dow", None) is not None):
                hour0 = int(self.start_hour[row])
                day0  = int(self.start_dow[row])
            else:
                # fallback (older memmaps) - may be phase-shifted
                hour0 = raw_t0 % 24
                day0  = (raw_t0 // 24) % 7

            hour = (hour0 + h) % 24
            day  = (day0 + (hour0 + h) // 24) % 7

            hod = hour.astype(np.float32)
            dow = day.astype(np.float32)

            hod_sin = np.sin(2*np.pi*hod/24.0).reshape(self.H,1).astype(np.float32)
            hod_cos = np.cos(2*np.pi*hod/24.0).reshape(self.H,1).astype(np.float32)
            dow_sin = np.sin(2*np.pi*dow/7.0).reshape(self.H,1).astype(np.float32)
            dow_cos = np.cos(2*np.pi*dow/7.0).reshape(self.H,1).astype(np.float32)
            is_weekend = ((day == 5) | (day == 6)).astype(np.float32).reshape(self.H,1)

            time_feats = np.concatenate([hod_sin, hod_cos, dow_sin, dow_cos, is_weekend], axis=1)
            feat_list.append(time_feats)

        # ----- baseline jump vs last obs (H,2) -----
        if self.add_jump_feats:
            jump = (y_base - x_last[None, :]).astype(np.float32, copy=False)  # (H,S)
            jump_mean = np.mean(jump, axis=1, keepdims=True).astype(np.float32, copy=False)
            jump_std  = np.std(jump, axis=1, keepdims=True).astype(np.float32, copy=False)
            jump_feats = np.concatenate([jump_mean, jump_std], axis=1).astype(np.float32, copy=False)
            feat_list.append(jump_feats)

        w_in = np.concatenate(feat_list, axis=1).astype(np.float32, copy=False)
        w_in = np.nan_to_num(w_in, nan=0.0).astype(np.float32, copy=False)

        # targets + history
        r_norm = self.R[row].astype(np.float32, copy=False)  # (H,S)
        rh = np.nan_to_num(self.RH[row], nan=0.0).astype(np.float32, copy=False)  # (RH_LEN,S,C)

        # per-sensor baseline normalized (H,S)
        yb_s_norm = y_base
        if (self.yb_s_mean is not None) and (self.yb_s_std is not None):
            yb_s_norm = (y_base - self.yb_s_mean[None, :]) / self.yb_s_std[None, :]

        return (
    np.array(w_in, copy=True),
    np.array(r_norm, copy=True),
    np.array(rh, copy=True),
    np.array(y_base, copy=True),
    np.array(yb_s_norm, copy=True),
)








# -------------------------
# Train
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap-dir", type=str, default="memmaps")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.03)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--h-dim", type=int, default=64)
    ap.add_argument("--ctx-dim", type=int, default=64)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", type=str, default="bench")
    ap.add_argument("--resid-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    mem = args.memmap_dir
    print(f"[SEED] {args.seed}")

    # --- load meta only (do NOT use good_idx.npy anymore) ---
    meta = np.load(f"{mem}/meta.npy", allow_pickle=True).item()
    raw_N = int(meta["N"])
    S = int(meta["S"])

    # ---- RAW time boundaries ----
    RAW_TRAIN_END = 32634
    RAW_PURGE_END = 33306
    RAW_VAL_END   = 40792

    # -------------------------
    # Pass 1: create DS without stats, get time split indices
    # -------------------------
    ds0 = DS(mem)  # no stats yet
    train_idx0 = np.nonzero(ds0.ds_raw_idx < RAW_TRAIN_END)[0].astype(np.int64)
    val_idx0   = np.nonzero((ds0.ds_raw_idx >= RAW_PURGE_END) & (ds0.ds_raw_idx < RAW_VAL_END))[0].astype(np.int64)

    print(f"[META] raw_N={raw_N} kept_N={len(ds0)} baseline={meta.get('baseline')}")
    print(f"[TIME SPLIT] raw train=[0:{RAW_TRAIN_END}) purge=[{RAW_TRAIN_END}:{RAW_PURGE_END}) val=[{RAW_PURGE_END}:{RAW_VAL_END})")
    print(f"[TIME SPLIT] kept train={len(train_idx0)} kept val={len(val_idx0)}")

    # -------------------------
    # Compute TRAIN-ONLY stats using RAW indices
    # -------------------------
    raw_train = ds0.ds_raw_idx[train_idx0]  # raw memmap rows for training samples

    Yb = np.memmap(f"{mem}/Y_base.dat", dtype="float32", mode="r", shape=(raw_N, HORIZON, S))
    yb_train = np.asarray(Yb[raw_train])  # (Ntrain,H,S)

    # summary feats (mean/std/top5 over sensors)
    yb_mean = np.mean(yb_train, axis=2)                           # (Ntrain,H)
    yb_std  = np.std(yb_train, axis=2)                            # (Ntrain,H)
    topk    = np.sort(yb_train, axis=2)[:, :, -5:]                # (Ntrain,H,5)
    yb_top5 = np.mean(topk, axis=2)                               # (Ntrain,H)

    feat = np.stack([yb_mean, yb_std, yb_top5], axis=-1).astype(np.float32)  # (Ntrain,H,3)
    feat2 = feat.reshape(-1, 3)                                   # (Ntrain*H,3)

    yb_feat_mean = feat2.mean(axis=0).astype(np.float32)          # (3,)
    yb_feat_std  = feat2.std(axis=0).astype(np.float32)           # (3,)
    yb_feat_std  = np.where(yb_feat_std == 0.0, 1.0, yb_feat_std).astype(np.float32)

    print("[YB FEATS] mean=", yb_feat_mean, "std=", yb_feat_std)

    # per-sensor baseline stats
    yb_s_flat = yb_train.reshape(-1, S).astype(np.float32)         # (Ntrain*H,S)
    yb_s_mean = yb_s_flat.mean(axis=0).astype(np.float32)          # (S,)
    yb_s_std  = yb_s_flat.std(axis=0).astype(np.float32)           # (S,)
    yb_s_std  = np.where(yb_s_std == 0.0, 1.0, yb_s_std).astype(np.float32)

    print("[YB_S] mean(mean)=", float(yb_s_mean.mean()), "std(mean)=", float(yb_s_std.mean()))

    # -------------------------
    # Pass 2: create DS with stats
    # -------------------------
    ds = DS(
        mem,
        yb_feat_mean=yb_feat_mean,
        yb_feat_std=yb_feat_std,
        yb_s_mean=yb_s_mean,
        yb_s_std=yb_s_std,
        add_time_feats=True,
        add_jump_feats=True,
    )

    # same split logic, but recompute indices on the real ds
    train_idx = np.nonzero(ds.ds_raw_idx < RAW_TRAIN_END)[0].astype(np.int64)
    val_idx   = np.nonzero((ds.ds_raw_idx >= RAW_PURGE_END) & (ds.ds_raw_idx < RAW_VAL_END))[0].astype(np.int64)

    r_mean = torch.from_numpy(ds.r_mean).to(device)
    r_std  = torch.from_numpy(ds.r_std).to(device)

    print(f"[DATA] H={ds.H} S={ds.S} w_in_dim={ds.w_in_dim} RH_LEN={RH_LEN} RH_CHANNELS={RH_CHANNELS}")

    # loaders
    dl = DataLoader(
        Subset(ds, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    dlv = DataLoader(
        Subset(ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    # weights from train only
    weights = compute_weights_from_train_loader(dl, r_mean, r_std, device, ds.S)
    print("[WEIGHTS] sum=", float(weights.sum().detach().cpu()))

    print(
        "[CONFIG] "
        f"tag={args.tag} seed={args.seed} batch={args.batch_size} lr={args.lr} wd={args.weight_decay} "
        f"h_dim={args.h_dim} ctx_dim={args.ctx_dim} RH_CHANNELS={RH_CHANNELS} "
        f"yb_feats=on yb_per_sensor=on time_feats=on jump_feats=on"
    )

    # model/opt
    model = HistConvResidual(
        n_sensors=ds.S,
        w_dim=ds.w_in_dim,
        h_dim=args.h_dim,
        ctx_dim=args.ctx_dim,
        rh_channels=RH_CHANNELS,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    h_idx_base = torch.arange(ds.H, device=device)[None, :]
    s_idx_base = torch.arange(ds.S, device=device)[None, None, :]

    run_dir = Path("runs") / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    best_score = -1e9
    best_epoch = -1
    bad = 0

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        tr_loss = 0.0
        steps = 0

        for w_in, r_norm, rh, _y_base, yb_s_norm in dl:
            B = r_norm.shape[0]
            w_in = w_in.to(device, non_blocking=True)
            r_norm = r_norm.to(device, non_blocking=True)
            rh = rh.to(device, non_blocking=True)
            yb_s_norm = yb_s_norm.to(device, non_blocking=True)

            h = h_idx_base.expand(B, -1)
            s = s_idx_base.expand(B, ds.H, ds.S)

            r_hat_norm = model(h, s, w_in, rh, yb_s_norm=yb_s_norm)
            if args.resid_scale != 1.0:
                r_hat_norm = r_hat_norm * float(args.resid_scale)

            loss = weighted_masked_mse(r_hat_norm, r_norm, weights)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            tr_loss += float(loss.detach().cpu())
            steps += 1

        # ---- val ----
        model.eval()
        sse_m = torch.zeros((ds.S,), device=device)
        cnt_m = torch.zeros((ds.S,), device=device)
        sse_b = torch.zeros((ds.S,), device=device)
        cnt_b = torch.zeros((ds.S,), device=device)

        with torch.no_grad():
            for w_in, r_norm, rh, y_base, yb_s_norm in dlv:
                B = r_norm.shape[0]
                w_in = w_in.to(device, non_blocking=True)
                r_norm = r_norm.to(device, non_blocking=True)
                rh = rh.to(device, non_blocking=True)
                y_base = y_base.to(device, non_blocking=True)
                yb_s_norm = yb_s_norm.to(device, non_blocking=True)

                h = h_idx_base.expand(B, -1)
                s = s_idx_base.expand(B, ds.H, ds.S)

                r_hat_norm = model(h, s, w_in, rh, yb_s_norm=yb_s_norm)
                if args.resid_scale != 1.0:
                    r_hat_norm = r_hat_norm * float(args.resid_scale)

                r_true_raw = resid_norm_to_raw(r_norm, r_mean, r_std)
                r_pred_raw = resid_norm_to_raw(r_hat_norm, r_mean, r_std)

                y_true = y_base + r_true_raw
                y_pred = y_base + r_pred_raw

                a_sse, a_cnt = accum_sse_counts(y_true, y_pred)
                b_sse, b_cnt = accum_sse_counts(y_true, y_base)

                sse_m += a_sse
                cnt_m += a_cnt
                sse_b += b_sse
                cnt_b += b_cnt

        nanS = torch.tensor(float("nan"), device=device)
        rmse_model = torch.sqrt(torch.where(cnt_m > 0, sse_m / cnt_m, nanS))
        rmse_base  = torch.sqrt(torch.where(cnt_b > 0, sse_b / cnt_b, nanS))
        val_score = score_from_rmses_torch(rmse_model, rmse_base, weights)

        if epoch == 1:
            baseline_score = score_from_rmses_torch(rmse_base, rmse_base, weights)
            print(f"[SANITY] baseline-as-model score (should be ~0): {baseline_score:+.6f}")

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={tr_loss/max(steps,1):.6f} | "
            f"val_score={val_score:+.6f} | "
            f"steps={steps}"
        )

        improved = (val_score - best_score) > args.min_delta
        if improved:
            best_score = val_score
            best_epoch = epoch
            bad = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1

        if bad >= args.patience:
            print(f"[EARLY STOP] epoch={epoch} best_epoch={best_epoch} best_val_score={best_score:+.6f}")
            break

    torch.save(model.state_dict(), last_path)
    print("Saved:", last_path)
    print("Saved best:", best_path, "best_val_score=", best_score, "best_epoch=", best_epoch)

    print(
        "[RESULT] "
        f"best_val_score={best_score:.9f} "
        f"best_epoch={best_epoch} "
        f"tag={args.tag} "
        f"seed={args.seed}"
    )



if __name__ == "__main__":
    main()