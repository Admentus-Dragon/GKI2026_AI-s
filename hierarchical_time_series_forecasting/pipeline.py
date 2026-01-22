#!/usr/bin/env python3
"""
pipeline.py — single-file refactor of your weather + sensor preprocessing scripts.

Subcommands:
  - train-npz                 : build data/train.npz from data/sensor_timeseries.csv
  - sensor-residual-memmaps   : build memmaps/ residual targets from data/train.npz
  - weather-hourly-forecast   : build memmaps/weather_hourly.parquet + memmaps/weather_meta.npy from data/weather_forecasts.csv
  - weather-forecast-memmap   : build memmaps/W_forecast.dat aligned to train.npz timestamps (uses weather_hourly.parquet)
  - weather-obs-summary       : build memmaps/W_obs_summary.dat + memmaps/weather_obs_meta.npy from data/weather_observations.csv
  - residual-hist-features    : build memmaps/R_hist_feat.dat from X_train

Key guarantee:
  - Y_base.dat is the SAME weekly-blend baseline used in score_local.py:
      Yb = alpha * P + (1-alpha) * W
    where P = last 72h in history, W = same 72h last week.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


# =============================================================================
# Schema / column indices (forecast/obs CSVs)
# =============================================================================

# ---- Forecast column indices ----
FC_STATION = 1
FC_TEMP = 2
FC_WIND = 3
FC_CLOUD = 4
FC_DEW = 8
FC_RAINACC = 9
FC_VALUE_DT = 10

# Keep only numeric, non-NaN columns:
FORECAST_FEATURES: Dict[str, int] = {
    "temp": FC_TEMP,
    "wind": FC_WIND,
    "cloud": FC_CLOUD,
    "dew": FC_DEW,
    "rainacc": FC_RAINACC,
}

# ---- Observation vars (by name; we use pandas columns) ----
OBS_VARS = ["t", "f", "rh", "td", "tx", "tn"]  # keep same as your script
WIN_24, WIN_72, WIN_168 = 24, 72, 168


# =============================================================================
# Shared constants / paths
# =============================================================================

DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUT_DIR = Path("memmaps")

HISTORY_LENGTH = 672
HORIZON = 72
N_SENSORS = 45
WEEK_LAG = 168


# =============================================================================
# Helpers
# =============================================================================

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_naive_hour_series(ts: pd.Series) -> pd.Series:
    """Parse timestamps as UTC, convert to naive, floor to hour."""
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    t = t.dt.tz_convert(None)
    return t.dt.floor("h")


def _to_naive_hour(ts) -> pd.Timestamp:
    """Parse timestamp as UTC (if tz-aware or naive), convert to tz-naive, floor to hour."""
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    if t is pd.NaT:
        raise ValueError(f"Bad timestamp: {ts!r}")
    t = t.tz_convert(None)
    return pd.Timestamp(t).floor("h")


def _timestamps_fingerprint(timestamps) -> str:
    ts = np.asarray(timestamps).astype(str)
    blob = ("\0".join(ts.tolist())).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# =============================================================================
# Weekly-blend baseline helpers (matches score_local.py)
# =============================================================================

def fit_alpha_per_sensor(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Fit alpha per sensor by minimizing MSE between Y and alpha*P + (1-alpha)*W
    using finite target entries only.

    X: (N,672,S)
    Y: (N,72,S)
    Returns alpha: (S,) clipped to [0,1]
    """
    P = X[:, HISTORY_LENGTH - HORIZON : HISTORY_LENGTH, :]
    W = X[:, HISTORY_LENGTH - HORIZON - WEEK_LAG : HISTORY_LENGTH - WEEK_LAG, :]
    D = P - W
    T = Y - W

    mask = np.isfinite(Y) & np.isfinite(P) & np.isfinite(W)
    D0 = np.nan_to_num(D, nan=0.0).astype(np.float64)
    T0 = np.nan_to_num(T, nan=0.0).astype(np.float64)
    m = mask.astype(np.float64)

    num = (D0 * T0 * m).sum(axis=(0, 1))             # (S,)
    den = ((D0 * D0) * m).sum(axis=(0, 1)) + 1e-12   # (S,)

    a = num / den
    a = np.clip(a, 0.0, 1.0).astype(np.float32)
    return a


def build_weekly_blend_baseline_from_X(X: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    X: (N, 672, S)
    alpha: (S,)
    Returns Y_base: (N, 72, S)
    """
    P = X[:, HISTORY_LENGTH - HORIZON : HISTORY_LENGTH, :]                        # (N,72,S)
    W = X[:, HISTORY_LENGTH - HORIZON - WEEK_LAG : HISTORY_LENGTH - WEEK_LAG, :]  # (N,72,S)
    return alpha[None, None, :] * P + (1.0 - alpha[None, None, :]) * W


# =============================================================================
# residual-hist-features
# =============================================================================

def cmd_residual_history_features(
    data_dir: Path,
    out_dir: Path,
    train_npz: str = "train.npz",
):
    """
    Build per-sensor residual-history features from X_train only (no Y needed).

    Residual is X[t] - X[t-168] inside the 672-hour history window.
    Features per sensor (F=5):
      last, mean6, mean24, slope6, std24
    Output:
      memmaps/R_hist_feat.dat: (N, S, F) float32
    """
    out_dir = _ensure_dir(out_dir)
    train_path = data_dir / train_npz

    data = np.load(train_path, allow_pickle=True)
    X = data["X_train"].astype(np.float32)  # (N,672,45)
    N, T, S = X.shape
    assert T == HISTORY_LENGTH and S == N_SENSORS

    idx24 = np.arange(HISTORY_LENGTH - 24, HISTORY_LENGTH, dtype=np.int64)
    idx6 = np.arange(HISTORY_LENGTH - 6, HISTORY_LENGTH, dtype=np.int64)

    r24 = X[:, idx24, :] - X[:, idx24 - WEEK_LAG, :]   # (N,24,S)
    r6  = X[:, idx6,  :] - X[:, idx6  - WEEK_LAG, :]   # (N,6,S)

    r_last = r24[:, -1, :]                             # (N,S)
    r_mean6 = np.nanmean(r6, axis=1)                   # (N,S)
    r_mean24 = np.nanmean(r24, axis=1)                 # (N,S)
    r_std24 = np.nanstd(r24, axis=1)                   # (N,S)

    x = np.arange(6, dtype=np.float32)
    x = x - x.mean()
    denom = float((x ** 2).sum())
    r_slope6 = (r6 * x[None, :, None]).sum(axis=1) / denom  # (N,S)

    feats = np.stack([r_last, r_mean6, r_mean24, r_slope6, r_std24], axis=2).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0).astype(np.float32)

    F = feats.shape[2]
    path = out_dir / "R_hist_feat.dat"
    mm = np.memmap(path, dtype="float32", mode="w+", shape=(N, S, F))
    mm[:] = feats
    mm.flush()

    meta = dict(F=int(F), names=["r_last", "r_mean6", "r_mean24", "r_slope6", "r_std24"], week_lag=int(WEEK_LAG))
    np.save(out_dir / "R_hist_feat_meta.npy", meta)

    print("[DONE] Wrote", path, "shape=", (N, S, F))
    print("[DONE] Wrote", out_dir / "R_hist_feat_meta.npy")


# =============================================================================
# weather-hourly-forecast
# =============================================================================

def cmd_weather_hourly_forecast(
    data_dir: Path,
    out_dir: Path,
    forecasts_csv: str = "weather_forecasts.csv",
):
    """
    Build an hourly *lookup table* of forecasts keyed by:
      (target_hour = date_time floored to hour, issue_hour = value_date floored to hour)

    This is later consumed by cmd_weather_forecast_memmap(), which will, for each training
    sample at time t0, select the **latest issue_hour <= t0** for each target hour in
    [t0, t0+H-1]. This avoids leakage and fixes the prior misalignment that used value_date
    as the time index.
    """
    out_dir = _ensure_dir(out_dir)
    csv_path = data_dir / forecasts_csv

    print(f"[INFO] Loading {csv_path}")
    df = pd.read_csv(csv_path)

    # Column 0 is `date_time` (forecast target time) per README_data.md.
    # Column FC_VALUE_DT is `value_date` (when the forecast was issued).
    df["target_hour"] = (
        pd.to_datetime(df.iloc[:, 0], utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dt.floor("h")
    )
    df["issue_hour"] = (
        pd.to_datetime(df.iloc[:, FC_VALUE_DT], utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dt.floor("h")
    )
    df["station"] = pd.to_numeric(df.iloc[:, FC_STATION], errors="coerce")

    # Keep only rows with valid times
    df = df.dropna(subset=["target_hour", "issue_hour", "station"])
    if len(df) == 0:
        raise RuntimeError("No valid forecast rows after parsing date_time/value_date/station_id.")

    main_station = int(df["station"].value_counts(dropna=True).index[0])
    print(f"[INFO] Main station inferred: {main_station}")

    for name, col in FORECAST_FEATURES.items():
        df[name] = pd.to_numeric(df.iloc[:, col], errors="coerce")

    # Aggregate to hourly per (target_hour, issue_hour)
    records = []
    gb = df.groupby(["target_hour", "issue_hour"], sort=True)
    for (tgt, iss), g in gb:
        row = {"target_hour": tgt, "issue_hour": iss}
        g_main = g[g["station"] == main_station]
        g_other = g[g["station"] != main_station]

        for name in FORECAST_FEATURES:
            row[f"{name}_main"] = float(g_main[name].mean()) if len(g_main) else np.nan
            row[f"{name}_other"] = float(g_other[name].mean()) if len(g_other) else np.nan
        records.append(row)

    hourly = pd.DataFrame.from_records(records)

    # MultiIndex: level0=target_hour, level1=issue_hour
    hourly = hourly.set_index(["target_hour", "issue_hour"]).sort_index()

    # Fill remaining NaNs with 0 so downstream never sees NaN/Inf
    hourly = hourly.fillna(0.0)

    parquet_path = out_dir / "weather_hourly.parquet"
    hourly.to_parquet(parquet_path)

    meta = dict(
        vars=list(FORECAST_FEATURES.keys()),
        w_dim=int(hourly.shape[1]),
        main_station=int(main_station),
        columns=list(hourly.columns),
        index_levels=["target_hour", "issue_hour"],
        source=str(csv_path),
    )
    np.save(out_dir / "weather_meta.npy", meta)

    print("[DONE] Saved hourly weather lookup:", hourly.shape)
    print("       ->", parquet_path)
    print("       ->", out_dir / "weather_meta.npy")

def cmd_weather_forecast_sanity(
    data_dir: Path,
    out_dir: Path,
    train_npz: str = "train.npz",
    n: int = 6,
    seed: int = 0,
    ks: tuple[int, ...] = (0, 6, 24, 48, 71),
):
    """
    Sanity-check that W_forecast.dat was built with rule:
      for each target hour (t0+k), pick latest issue_hour <= t0.

    What it does:
      - Loads train.npz timestamps
      - Loads memmaps/W_forecast.dat and memmaps/weather_hourly.parquet
      - Randomly samples n rows i, and checks a few horizon steps k
      - Recomputes the "expected" selected forecast from parquet and compares
        to what's stored in W_forecast.dat (max abs diff).
      - Prints the chosen issue_hour and verifies it's <= t0 (no leakage).
    """
    out_dir = _ensure_dir(out_dir)
    train_path = data_dir / train_npz

    data = np.load(train_path, allow_pickle=True)
    if "timestamps" not in data:
        raise KeyError(f"{train_path} must contain key 'timestamps'")
    timestamps = data["timestamps"]
    N = len(timestamps)
    if N == 0:
        raise RuntimeError("train.npz has 0 timestamps?")

    # Load weather meta + memmap
    wmeta_path = out_dir / "weather_meta.npy"
    if not wmeta_path.exists():
        raise FileNotFoundError(f"Missing {wmeta_path} (run weather-forecast-memmap first).")
    wmeta = np.load(wmeta_path, allow_pickle=True).item()
    w_dim = int(wmeta["w_dim"])

    W_path = out_dir / "W_forecast.dat"
    if not W_path.exists():
        raise FileNotFoundError(f"Missing {W_path} (run weather-forecast-memmap first).")

    W = np.memmap(W_path, dtype="float32", mode="r", shape=(N, HORIZON, w_dim))

    hourly_path = out_dir / "weather_hourly.parquet"
    if not hourly_path.exists():
        raise FileNotFoundError(f"Missing {hourly_path} (run weather-hourly-forecast first).")

    hourly = pd.read_parquet(hourly_path)

    # Expect MultiIndex (target_hour, issue_hour)
    if not isinstance(hourly.index, pd.MultiIndex) or hourly.index.nlevels != 2:
        raise ValueError(
            "weather_hourly.parquet must have MultiIndex (target_hour, issue_hour). "
            "Rebuild it with the updated cmd_weather_hourly_forecast()."
        )

    # Normalize index types (naive hourly)
    tgt = pd.DatetimeIndex(hourly.index.get_level_values(0)).tz_localize(None).floor("h")
    iss = pd.DatetimeIndex(hourly.index.get_level_values(1)).tz_localize(None).floor("h")
    hourly.index = pd.MultiIndex.from_arrays([tgt, iss], names=["target_hour", "issue_hour"])
    hourly = hourly.sort_index()

    rng = np.random.default_rng(seed)
    idxs = rng.choice(np.arange(N), size=min(n, N), replace=False)

    print("=" * 88)
    print("[SANITY] Checking W_forecast.dat alignment / no-leakage")
    print(f"  N={N} | HORIZON={HORIZON} | w_dim={w_dim}")
    print(f"  samples={len(idxs)} | seed={seed} | ks={ks}")
    print("=" * 88)

    for i in idxs:
        t0 = _to_naive_hour(timestamps[i])
        print(f"\n[i={i}] t0={t0}")

        for k in ks:
            if not (0 <= k < HORIZON):
                continue

            tgt_hour = t0 + pd.Timedelta(hours=int(k))

            # Get all issues for this target hour
            try:
                sub = hourly.xs(tgt_hour, level=0)  # index: issue_hour
            except KeyError:
                print(f"  k={k:2d} target={tgt_hour} | NO DATA for target hour")
                continue

            issue_hours = pd.DatetimeIndex(sub.index).tz_localize(None).floor("h")
            issue_eh = (issue_hours.view("int64") // (10**9 * 3600)).astype(np.int64)
            vals = sub.to_numpy(dtype=np.float32, copy=False)

            t0_eh = int(pd.Timestamp(t0).value // (10**9 * 3600))

            j = int(np.searchsorted(issue_eh, t0_eh, side="right") - 1)
            if j < 0:
                j = 0  # earliest available if nothing <= t0

            chosen_issue = issue_hours[j]
            leak = chosen_issue > t0

            expected = vals[j]             # (w_dim,)
            stored = np.asarray(W[i, k])   # (w_dim,)

            max_abs = float(np.max(np.abs(stored - expected)))

            # Find the first issue strictly after t0 (if any) to show that we didn't pick it
            j_future = int(np.searchsorted(issue_eh, t0_eh, side="right"))
            future_issue = None
            future_diff = None
            if 0 <= j_future < len(issue_hours):
                future_issue = issue_hours[j_future]
                future_diff = float(np.max(np.abs(vals[j_future] - expected)))

            msg = (
                f"  k={k:2d} target={tgt_hour} | chosen_issue={chosen_issue} "
                f"| leak?={'YES' if leak else 'no'} | max_abs(stored-expected)={max_abs:.6g}"
            )
            if future_issue is not None:
                msg += f" | next_issue>{t0} is {future_issue} (diff_vs_chosen={future_diff:.6g})"
            print(msg)

    print("\n[NOTE] max_abs(stored-expected) should be ~0 (float rounding ok).")
    print("[NOTE] leak? should always be 'no' (unless no <=t0 exists and earliest > t0, which indicates data issue).")


def cmd_weather_forecast_memmap(
    data_dir: Path,
    out_dir: Path,
    train_npz: str = "train.npz",
):
    """
    Build W_forecast.dat of shape (N, HORIZON, w_dim).

    For each training sample i with start timestamp t0 (naive hour),
    and for each horizon step k (target hour = t0 + k),
    select the latest forecast with issue_hour <= t0 for that target hour.

    IMPORTANT POLICY (no leakage):
      If there is NO issue_hour <= t0 available for a given target hour,
      we write zeros for that horizon step (do NOT use a future issuance).
    """
    out_dir = _ensure_dir(out_dir)
    train_path = data_dir / train_npz

    data = np.load(train_path, allow_pickle=True)
    if "timestamps" not in data:
        raise KeyError(f"{train_path} must contain key 'timestamps'")
    timestamps = data["timestamps"]
    N = len(timestamps)
    print(f"[INFO] N samples: {N}")

    ts_hash = _timestamps_fingerprint(timestamps)
    ts_first = str(timestamps[0]) if N else ""
    ts_last = str(timestamps[-1]) if N else ""

    hourly_path = out_dir / "weather_hourly.parquet"
    if not hourly_path.exists():
        raise FileNotFoundError(f"Missing {hourly_path}. Run: weather-hourly-forecast first.")

    print(f"[INFO] Loading hourly parquet (target_hour, issue_hour): {hourly_path}")
    t_load = time.time()
    hourly = pd.read_parquet(hourly_path)

    if not isinstance(hourly.index, pd.MultiIndex) or hourly.index.nlevels != 2:
        raise ValueError(
            "weather_hourly.parquet must have a 2-level MultiIndex: (target_hour, issue_hour). "
            "Re-run cmd_weather_hourly_forecast() after updating pipeline.py."
        )

    tgt = pd.DatetimeIndex(hourly.index.get_level_values(0)).tz_localize(None).floor("h")
    iss = pd.DatetimeIndex(hourly.index.get_level_values(1)).tz_localize(None).floor("h")
    hourly.index = pd.MultiIndex.from_arrays([tgt, iss], names=["target_hour", "issue_hour"])
    hourly = hourly.sort_index()

    w_dim = int(hourly.shape[1])
    print(f"[INFO] Hourly loaded in {time.time()-t_load:.1f}s | rows={len(hourly):,} | w_dim={w_dim}")

    # Build per-target lookup: target_eh -> (issue_eh_sorted, values_sorted)
    print("[INFO] Building per-target forecast lookup (may take a bit on first run)...")
    lookup: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for target_hour, sub in hourly.groupby(level=0, sort=False):
        sub = sub.droplevel(0).sort_index()  # index now issue_hour
        issue_hours = pd.DatetimeIndex(sub.index).tz_localize(None).floor("h")
        issue_eh = (issue_hours.view("int64") // (10**9 * 3600)).astype(np.int64)
        vals = sub.to_numpy(dtype=np.float32, copy=False)
        target_eh = int(pd.Timestamp(target_hour).value // (10**9 * 3600))
        lookup[target_eh] = (issue_eh, vals)

    W_path = out_dir / "W_forecast.dat"
    W_mm = np.memmap(W_path, dtype="float32", mode="w+", shape=(N, HORIZON, w_dim))

    meta_path = out_dir / "weather_meta.npy"
    meta = dict(
        w_dim=int(w_dim),
        horizon=int(HORIZON),
        parquet=str(hourly_path),
        selection_rule="latest issue_hour <= t0 (sample start); if none, zeros",
        timestamps_len=int(N),
        timestamps_sha256=ts_hash,
        timestamps_first=ts_first,
        timestamps_last=ts_last,
    )
    np.save(meta_path, meta)

    start = time.time()
    for i, ts in enumerate(timestamps):
        t0 = _to_naive_hour(ts)
        t0_eh = int(pd.Timestamp(t0).value // (10**9 * 3600))

        out = np.zeros((HORIZON, w_dim), dtype=np.float32)

        for k in range(HORIZON):
            tgt_eh = t0_eh + k
            item = lookup.get(tgt_eh)
            if item is None:
                continue

            issue_eh, vals = item
            j = int(np.searchsorted(issue_eh, t0_eh, side="right") - 1)

            if j < 0:
                # No issuance at or before t0 for this target hour -> leave zeros (no leakage)
                continue

            out[k] = vals[j]

        W_mm[i] = out

        if (i + 1) % 500 == 0 or i == 0 or (i + 1) == N:
            now = time.time()
            elapsed = now - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            remaining = (N - (i + 1)) / rate if rate > 0 else float("inf")
            print(
                f"[{i+1:6d}/{N}] {rate:6.2f} samples/s | elapsed {elapsed/60:6.1f} min | ETA {remaining/60:6.1f} min"
            )

    W_mm.flush()
    print(f"[DONE] W_forecast.dat written: shape=({N},{HORIZON},{w_dim}) -> {W_path}")
    print(f"[DONE] weather_meta.npy written -> {meta_path}")


# =============================================================================
# weather-obs-summary
# =============================================================================

def _safe_mean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else np.nan

def _safe_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.std()) if x.size else np.nan

def _safe_last(x: np.ndarray) -> float:
    for v in x[::-1]:
        if np.isfinite(v):
            return float(v)
    return np.nan

def build_one_summary(obs_hist: np.ndarray) -> np.ndarray:
    feats: List[float] = []
    for j in range(obs_hist.shape[1]):
        x = obs_hist[:, j]
        feats.append(_safe_last(x))
        feats.append(_safe_mean(x[-WIN_24:]))
        feats.append(_safe_mean(x[-WIN_72:]))
        feats.append(_safe_mean(x[-WIN_168:]))
        feats.append(_safe_std(x[-WIN_24:]))
    return np.asarray(feats, dtype=np.float32)

def cmd_weather_obs_summary(
    data_dir: Path,
    out_dir: Path,
    train_npz: str = "train.npz",
    observations_csv: str = "weather_observations.csv",
):
    out_dir = _ensure_dir(out_dir)
    train_path = data_dir / train_npz

    data = np.load(train_path, allow_pickle=True)
    ts_raw = data["timestamps"]
    N = len(ts_raw)

    ts_hash = _timestamps_fingerprint(ts_raw)
    ts_arr = np.asarray(ts_raw).astype(str)
    ts_first = str(ts_arr[0]) if N > 0 else ""
    ts_last = str(ts_arr[-1]) if N > 0 else ""

    timestamps = pd.Series(pd.to_datetime(ts_raw, errors="coerce"))
    if timestamps.isna().any():
        raise RuntimeError(f"Found {int(timestamps.isna().sum())} unparsable timestamps in {train_path}")

    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize("UTC")
    else:
        timestamps = timestamps.dt.tz_convert("UTC")

    obs_path = data_dir / observations_csv
    obs = pd.read_csv(obs_path, low_memory=False)

    obs["timi"] = pd.to_datetime(obs["timi"], utc=True, errors="coerce")
    obs = obs.dropna(subset=["timi"])

    main_station = int(obs["stod"].value_counts().idxmax())
    print("[obs] main_station:", main_station)

    obs = obs[obs["stod"] == main_station].copy()

    keep = ["timi"] + OBS_VARS
    missing = [c for c in keep if c not in obs.columns]
    if missing:
        raise KeyError(f"Missing required columns in {obs_path}: {missing}")

    obs = obs[keep].sort_values("timi")
    obs["timi_hour"] = obs["timi"].dt.floor("h")
    obs = obs.groupby("timi_hour", as_index=True)[OBS_VARS].mean()

    t_min = (timestamps.min() - pd.Timedelta(hours=HISTORY_LENGTH)).floor("h")
    t_max = timestamps.max().ceil("h")
    hourly_index = pd.date_range(start=t_min, end=t_max, freq="h", tz="UTC")

    obs = obs.reindex(hourly_index)
    obs_np = obs.to_numpy(dtype=np.float32)

    idx_map = pd.Series(np.arange(len(hourly_index), dtype=np.int64), index=hourly_index)

    d_obs = len(OBS_VARS) * 5
    W_obs_path = out_dir / "W_obs_summary.dat"
    W_obs = np.memmap(W_obs_path, dtype="float32", mode="w+", shape=(N, d_obs))

    t0_hours = timestamps.dt.floor("h")
    for i, t0 in enumerate(t0_hours):
        start = t0 - pd.Timedelta(hours=HISTORY_LENGTH)
        end = t0 - pd.Timedelta(hours=1)

        if (start not in idx_map.index) or (end not in idx_map.index):
            W_obs[i] = np.full((d_obs,), np.nan, dtype=np.float32)
        else:
            a = int(idx_map.loc[start])
            b = int(idx_map.loc[end])
            hist = obs_np[a:b + 1]
            if hist.shape[0] != HISTORY_LENGTH:
                W_obs[i] = np.full((d_obs,), np.nan, dtype=np.float32)
            else:
                W_obs[i] = build_one_summary(hist)

        if (i + 1) % 2000 == 0:
            print(f"[obs] done {i+1}/{N}")

    W_obs.flush()

    meta = dict(
        main_station=main_station,
        obs_vars=OBS_VARS,
        d_obs=int(d_obs),
        windows=dict(last=1, mean24=24, mean72=72, mean168=168, std24=24),
        timestamps_len=int(N),
        timestamps_sha256=ts_hash,
        timestamps_first=ts_first,
        timestamps_last=ts_last,
        source=str(obs_path),
    )
    np.save(out_dir / "weather_obs_meta.npy", meta)

    print("Saved", W_obs_path, "shape=", (N, d_obs))
    print("Saved", out_dir / "weather_obs_meta.npy")


# =============================================================================
# train-npz
# =============================================================================

def cmd_train_npz(
    data_dir: Path,
    sensor_csv: str = "sensor_timeseries.csv",
    out_npz: str = "train.npz",
    hist: int = HISTORY_LENGTH,
    hor: int = HORIZON,
):
    csv_path = data_dir / sensor_csv
    out_path = data_dir / out_npz

    df = pd.read_csv(csv_path, parse_dates=["CTime"])
    df = df.sort_values("CTime")

    ctime = pd.to_datetime(df["CTime"], errors="coerce")
    if ctime.dt.tz is None:
        ctime = ctime.dt.tz_localize("UTC")
    else:
        ctime = ctime.dt.tz_convert("UTC")
    df["CTime"] = ctime

    sensors = [c for c in df.columns if c.startswith("M")] + ["FRAMRENNSLI_TOTAL"]
    arr = df[sensors].to_numpy(np.float32)

    X, Y, T = [], [], []
    start_epoch_hour = []
    start_hour = []
    start_dow = []

    for i in range(len(arr) - hist - hor):
        X.append(arr[i : i + hist])
        Y.append(arr[i + hist : i + hist + hor])

        t0 = df["CTime"].iloc[i + hist]
        t0h = t0.floor("h")
        T.append(t0h.to_pydatetime())

        eh = int(t0h.value // (10**9 * 3600))
        start_epoch_hour.append(eh)
        start_hour.append(int(t0h.hour))
        start_dow.append(int(t0h.dayofweek))

    np.savez(
        out_path,
        X_train=np.stack(X),
        y_train=np.stack(Y),
        timestamps=np.array(T, dtype=object),
        sensor_names=np.array(sensors),
        start_epoch_hour=np.asarray(start_epoch_hour, dtype=np.int64),
        start_hour=np.asarray(start_hour, dtype=np.int16),
        start_dow=np.asarray(start_dow, dtype=np.int16),
    )

    print("[DONE] Saved", out_path, "samples=", len(X))


# =============================================================================
# sensor-residual-memmaps  (THIS is the important one)
# =============================================================================

def cmd_sensor_residual_memmaps(
    data_dir: Path,
    out_dir: Path,
    train_npz: str = "train.npz",
    fit_end: int = 0,
    refit_alpha: bool = False,
    refit_stats: bool = False,
):
    out_dir = _ensure_dir(out_dir)
    train_path = data_dir / train_npz

    data = np.load(train_path, allow_pickle=True)
    X = data["X_train"].astype(np.float32)  # (N,672,45)
    Y = data["y_train"].astype(np.float32)  # (N,72,45)

    start_epoch_hour = data["start_epoch_hour"].astype(np.int64)
    start_hour = data["start_hour"].astype(np.int16)
    start_dow = data["start_dow"].astype(np.int16)

    N = X.shape[0]
    assert X.shape[1:] == (HISTORY_LENGTH, N_SENSORS), f"X shape mismatch: {X.shape}"
    assert Y.shape[1:] == (HORIZON, N_SENSORS), f"Y shape mismatch: {Y.shape}"

    # --- IMPORTANT: avoid leakage in baseline/stat fitting ---
    # Fit alpha and residual normalization stats ONLY on the first `fit_end` samples (train split).
    # If fit_end<=0, fall back to using the full dataset (old/leaky behavior).
    fit_end_eff = int(fit_end) if int(fit_end) > 0 else int(N)
    fit_end_eff = max(1, min(fit_end_eff, int(N)))
    fit_slice = slice(0, fit_end_eff)

    # Fit or load alpha for weekly-blend baseline (train-only if refit or missing)
    alpha_path = out_dir / "alpha.npy"
    alpha_meta_path = out_dir / "alpha_meta.npy"
    if alpha_path.exists() and (not refit_alpha):
        alpha = np.load(alpha_path).astype(np.float32)
        if alpha.shape != (N_SENSORS,):
            raise ValueError(f"alpha.npy bad shape {alpha.shape}, expected {(N_SENSORS,)}")
        print("[ALPHA] loaded:", alpha_path, "min/mean/max:", float(alpha.min()), float(alpha.mean()), float(alpha.max()))
    else:
        alpha = fit_alpha_per_sensor(X[fit_slice], Y[fit_slice])
        np.save(alpha_path, alpha)
        np.save(alpha_meta_path, {"fit_end": int(fit_end_eff)})
        print("[ALPHA] fit+saved (train-only):", alpha_path, "fit_end=", int(fit_end_eff),
              "min/mean/max:", float(alpha.min()), float(alpha.mean()), float(alpha.max()))

    # Weekly-blend baseline for ALL samples (matches score_local.py)
    Y_base = build_weekly_blend_baseline_from_X(X, alpha).astype(np.float32)

    # Residuals in original units (ALL samples)
    R_all = Y - Y_base  # may contain NaNs where Y is NaN

    # Residual normalization stats per sensor computed on TRAIN-ONLY slice
    R_fit = R_all[fit_slice]
    mask_fit = np.isfinite(R_fit)
    denom = mask_fit.sum(axis=(0, 1)).astype(np.float64)
    denom = np.maximum(denom, 1.0)

    R0_fit = np.nan_to_num(R_fit, nan=0.0).astype(np.float64)
    r_mean = (R0_fit * mask_fit).sum(axis=(0, 1)) / denom

    R_center = (R0_fit - r_mean[None, None, :]) * mask_fit
    r_var = (R_center ** 2).sum(axis=(0, 1)) / denom
    r_std = np.sqrt(np.maximum(r_var, 1e-8))

    r_mean = r_mean.astype(np.float32)
    r_std = r_std.astype(np.float32)

    # For completeness/debugging
    np.save(out_dir / "stats_meta.npy", {"fit_end": int(fit_end_eff)})

    # Mask for ALL samples (used below to keep NaNs where targets are missing)
    mask = np.isfinite(R_all)
    R0 = np.nan_to_num(R_all, nan=0.0).astype(np.float64)
    # Normalize residuals, keep NaNs where original target is missing
    R_norm = ((R0.astype(np.float32) - r_mean[None, None, :]) / r_std[None, None, :]).astype(np.float32)
    R_norm[~mask] = np.nan

    # Save good_idx like score_local's "good = isfinite(Y).all(...)"
    good = np.isfinite(Y).all(axis=(1, 2))
    good_idx = np.nonzero(good)[0].astype(np.int64)
    np.save(out_dir / "good_idx.npy", good_idx)

    # Minimal placeholder (kept for compatibility)
    X_hist_feat = np.zeros((N, 5), dtype=np.float32)

    # Write memmaps
    meta = dict(
        N=int(N),
        H=int(HORIZON),
        S=int(N_SENSORS),
        baseline="weekly_blend",
        week_lag=int(WEEK_LAG),
    )
    np.save(out_dir / "meta.npy", meta)

    X_mm = np.memmap(out_dir / "X_hist.dat", dtype="float32", mode="w+", shape=X_hist_feat.shape)
    X_mm[:] = X_hist_feat
    X_mm.flush()

    Yb_mm = np.memmap(out_dir / "Y_base.dat", dtype="float32", mode="w+", shape=Y_base.shape)
    Yb_mm[:] = np.nan_to_num(Y_base, nan=0.0).astype(np.float32)
    Yb_mm.flush()

    Rn_mm = np.memmap(out_dir / "R_norm.dat", dtype="float32", mode="w+", shape=R_norm.shape)
    Rn_mm[:] = R_norm
    Rn_mm.flush()

    np.save(out_dir / "r_mean.npy", r_mean)
    np.save(out_dir / "r_std.npy", r_std)
    np.save(out_dir / "start_epoch_hour.npy", start_epoch_hour)
    np.save(out_dir / "start_hour.npy", start_hour)
    np.save(out_dir / "start_dow.npy", start_dow)

    print("[DONE] Memmaps written to", out_dir)
    print("       meta:", meta)
    print("       X_hist:", X_hist_feat.shape)
    print("       Y_base:", Y_base.shape)
    print("       R_norm:", R_norm.shape)
    print("       Saved:", out_dir / "alpha.npy")
    print("       Saved:", out_dir / "r_mean.npy", "and", out_dir / "r_std.npy")
    print("       Saved:", out_dir / "good_idx.npy", "kept", len(good_idx), "of", N)


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pipeline.py")
    ap.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))

    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train-npz", help="Build data/train.npz from data/sensor_timeseries.csv")
    p.add_argument("--sensor-csv", type=str, default="sensor_timeseries.csv")
    p.add_argument("--out-npz", type=str, default="train.npz")
    p.add_argument("--hist", type=int, default=HISTORY_LENGTH)
    p.add_argument("--hor", type=int, default=HORIZON)

    p = sub.add_parser("sensor-residual-memmaps", help="Build residual memmaps from data/train.npz")
    p.add_argument("--train-npz", type=str, default="train.npz")
    p.add_argument("--fit-end", type=int, default=0, help="Number of initial samples to fit baseline/stats on (use your train split end). If <=0, uses all samples (leaky).")
    p.add_argument("--refit-alpha", action="store_true", help="Recompute alpha even if alpha.npy exists.")

    p = sub.add_parser("weather-hourly-forecast", help="Build hourly forecast parquet from weather_forecasts.csv")
    p.add_argument("--forecasts-csv", type=str, default="weather_forecasts.csv")

    p = sub.add_parser("weather-forecast-memmap", help="Build W_forecast.dat aligned to train.npz timestamps")
    p.add_argument("--train-npz", type=str, default="train.npz")

    p = sub.add_parser("weather-obs-summary", help="Build W_obs_summary.dat from weather_observations.csv aligned to train.npz timestamps")
    p.add_argument("--train-npz", type=str, default="train.npz")
    p.add_argument("--observations-csv", type=str, default="weather_observations.csv")

    p = sub.add_parser("residual-hist-features", help="Build per-sensor residual history feature memmap from X_train")
    p.add_argument("--train-npz", type=str, default="train.npz")

    p = sub.add_parser("weather-forecast-sanity", help="Sanity-check W_forecast.dat alignment and no-leakage")
    p.add_argument("--train-npz", type=str, default="train.npz")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)


    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    if args.cmd == "train-npz":
        cmd_train_npz(
            data_dir=data_dir,
            sensor_csv=args.sensor_csv,
            out_npz=args.out_npz,
            hist=args.hist,
            hor=args.hor,
        )
        return

    if args.cmd == "sensor-residual-memmaps":
        cmd_sensor_residual_memmaps(
            data_dir=data_dir,
            out_dir=out_dir,
            train_npz=args.train_npz,
            fit_end=args.fit_end,
            refit_alpha=args.refit_alpha,
        )
        return

    if args.cmd == "weather-hourly-forecast":
        cmd_weather_hourly_forecast(
            data_dir=data_dir,
            out_dir=out_dir,
            forecasts_csv=args.forecasts_csv,
        )
        return

    if args.cmd == "weather-forecast-memmap":
        cmd_weather_forecast_memmap(
            data_dir=data_dir,
            out_dir=out_dir,
            train_npz=args.train_npz,
        )
        return

    if args.cmd == "weather-obs-summary":
        cmd_weather_obs_summary(
            data_dir=data_dir,
            out_dir=out_dir,
            train_npz=args.train_npz,
            observations_csv=args.observations_csv,
        )
        return

    if args.cmd == "residual-hist-features":
        cmd_residual_history_features(
            data_dir=data_dir,
            out_dir=out_dir,
            train_npz=args.train_npz,
        )
        return
    if args.cmd == "weather-forecast-sanity":
        cmd_weather_forecast_sanity(
            data_dir=data_dir,
            out_dir=out_dir,
            train_npz=args.train_npz,
            n=args.n,
            seed=args.seed,
        )
        return


    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
