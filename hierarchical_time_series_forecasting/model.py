"""
Model for hot water demand forecasting.

Contract (must match api.py / leaderboard):
Inputs:
  - sensor_history: (672, 45) float array
  - timestamp: ISO datetime string - forecast start
  - weather_forecast: optional array-like (72, n_features) (often mixed dtype)
  - weather_history: optional array-like (672, n_features) (often mixed dtype)

Output:
  - predictions: (72, 45) float array
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from datetime import datetime, timezone

HISTORY_LENGTH = 672
HORIZON = 72
N_SENSORS = 45
WEEK_LAG = 168  # hours
DAY_LAG = 72    # hours

from pathlib import Path
import numpy as np

def load_weather_meta(root: Path):
    meta_path = root / "memmaps" / "weather_meta.npy"
    if not meta_path.exists():
        return None
    return np.load(meta_path, allow_pickle=True).item()

def coerce_weather_forecast_to_wdim(weather_forecast, wmeta):
    """
    Convert API-provided weather_forecast (mixed dtype) to float32 (72, w_dim),
    using the wmeta["columns"] layout if available.
    """
    if weather_forecast is None or wmeta is None:
        return None

    a = np.asarray(weather_forecast)
    if a.ndim != 2 or a.shape[0] != 72:
        # Don’t crash; just ignore weather if shape is unexpected.
        return None

    w_dim = int(wmeta.get("w_dim", a.shape[1]))
    # Best case: already numeric and correct width
    if np.issubdtype(a.dtype, np.number):
        x = a.astype(np.float32, copy=False)
        if x.shape[1] < w_dim:
            x = np.pad(x, ((0,0),(0, w_dim - x.shape[1])), mode="constant")
        elif x.shape[1] > w_dim:
            x = x[:, :w_dim]
        return x

    # Mixed dtype: keep numeric-coercible columns in-order
    cols = []
    for j in range(a.shape[1]):
        col = a[:, j]
        out = np.full((a.shape[0],), np.nan, dtype=np.float32)
        ok_any = False
        for i, v in enumerate(col):
            try:
                if v is None:
                    continue
                out[i] = float(v)
                ok_any = True
            except Exception:
                continue
        if ok_any:
            m = np.nanmean(out)
            if not np.isfinite(m):
                m = 0.0
            out = np.where(np.isfinite(out), out, m).astype(np.float32)
            cols.append(out[:, None])

    if not cols:
        return None

    x = np.concatenate(cols, axis=1).astype(np.float32, copy=False)

    # Match training w_dim
    if x.shape[1] < w_dim:
        x = np.pad(x, ((0,0),(0, w_dim - x.shape[1])), mode="constant")
    elif x.shape[1] > w_dim:
        x = x[:, :w_dim]

    return x


def _parse_iso(ts: str) -> datetime:
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _coerce_weather(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Convert weather array-like to float features, dropping non-numeric cols.

    The API may pass a mixed dtype array (timestamps/strings + numbers).
    Strategy:
      - ensure 2D
      - keep only columns that can be coerced to float for at least one row
      - coerce with NaN for failures
      - fill NaN with column means (fallback 0)
    """
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim != 2:
        return None

    if np.issubdtype(a.dtype, np.number):
        return a.astype(np.float32, copy=False)

    cols = []
    for j in range(a.shape[1]):
        col = a[:, j]
        out_col = np.full((a.shape[0],), np.nan, dtype=np.float32)
        ok_any = False
        for i, v in enumerate(col):
            try:
                if v is None:
                    continue
                out_col[i] = float(v)
                ok_any = True
            except Exception:
                continue
        if ok_any:
            m = np.nanmean(out_col)
            if not np.isfinite(m):
                m = 0.0
            out_col = np.where(np.isfinite(out_col), out_col, m).astype(np.float32)
            cols.append(out_col[:, None])

    if not cols:
        return None
    return np.concatenate(cols, axis=1).astype(np.float32, copy=False)


def _weekly_blend_baseline(sensor_history: np.ndarray, alpha: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Weekly-blend baseline:
      P = last 72h
      W = same 72h last week
      Yb = alpha*P + (1-alpha)*W
    alpha: (45,) in [0,1]. If None, use 0.5.
    """
    X = sensor_history.astype(np.float32, copy=False)
    P = X[HISTORY_LENGTH - HORIZON : HISTORY_LENGTH, :]
    W = X[HISTORY_LENGTH - HORIZON - WEEK_LAG : HISTORY_LENGTH - WEEK_LAG, :]
    if alpha is None:
        alpha = np.full((N_SENSORS,), 0.5, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32).reshape(1, -1)
    return alpha * P + (1.0 - alpha) * W


def baseline_model(sensor_history: np.ndarray) -> np.ndarray:
    """Simple baseline: repeat the last 72 hours in the window."""
    X = sensor_history.astype(np.float32, copy=False)
    return X[HISTORY_LENGTH - HORIZON : HISTORY_LENGTH, :].copy()


@dataclass
class ResidualArtifacts:
    alpha: Optional[np.ndarray]
    r_mu: Optional[np.ndarray]
    r_sigma: Optional[np.ndarray]
    w_mu: Optional[np.ndarray]
    w_sigma: Optional[np.ndarray]
    w_dim: Optional[int]


class _ResidualModelWrapper:
    """
    Optional learned-model runner.

    If the required artifacts aren't present, this stays disabled and we fall back to baseline.
    """
    def __init__(self, root: Path):
        self.root = root
        self.enabled = False
        self.art = ResidualArtifacts(None, None, None, None, None, None)
        self.model = None
        self.device = None
        self.t_hist = 168
        self._try_load()

    def _try_load(self) -> None:
        try:
            import torch  # noqa
            from model_good import HistConvResidual  # noqa
        except Exception:
            return

        pt = self.root / "histconv.pt"
        if not pt.exists():
            return

        meta = {}
        meta_path = self.root / "meta.json"
        if meta_path.exists():
            try:
                import json
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}

        self.art.w_dim = int(meta.get("w_dim")) if "w_dim" in meta else None
        t_hist = int(meta.get("t_hist", 168))
        self.t_hist = max(1, min(t_hist, HISTORY_LENGTH - HORIZON))

        def _load_npy(name: str) -> Optional[np.ndarray]:
            p = self.root / name
            if p.exists():
                try:
                    return np.load(p)
                except Exception:
                    return None
            return None

        self.art.alpha = _load_npy("alpha.npy")
        self.art.r_mu = _load_npy("r_mu.npy")
        self.art.r_sigma = _load_npy("r_sigma.npy")
        self.art.w_mu = _load_npy("weather_mu.npy")
        self.art.w_sigma = _load_npy("weather_sigma.npy")

        import torch
        ckpt = torch.load(pt, map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        # Infer w_dim from weights if possible
        w_dim = None
        for k, v in state.items():
            if k.endswith("w_proj.weight"):
                w_dim = v.shape[1]
                break
        if w_dim is None:
            w_dim = self.art.w_dim
        if w_dim is None:
            return
        self.art.w_dim = int(w_dim)

        # Infer h_dim
        h_dim = None
        for k, v in state.items():
            if k.endswith("h_emb.weight"):
                h_dim = v.shape[1]
                break
        if h_dim is None:
            return

        # Infer input channels to in_proj (c_dim)
        c_dim = None
        for k, v in state.items():
            if k.endswith("in_proj.weight"):
                c_dim = v.shape[1]
                break
        if c_dim is None:
            c_dim = 1

        from model_good import HistConvResidual
        self.model = HistConvResidual(
            n_sensors=N_SENSORS,
            horizon=HORIZON,
            h_dim=int(h_dim),
            w_dim=int(w_dim),
            c_dim=int(c_dim),
        )
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.enabled = True

    def __call__(self, sensor_history: np.ndarray, timestamp: str,
                 weather_forecast: Optional[np.ndarray], weather_history: Optional[np.ndarray]) -> np.ndarray:
        import torch

        # Baseline for the forecast horizon
        yb = _weekly_blend_baseline(sensor_history, self.art.alpha)  # (72,45)

        # Residual history for last t_hist hours (ending at forecast start)
        X = sensor_history.astype(np.float32, copy=False)
        alpha = self.art.alpha
        if alpha is None:
            alpha = np.full((N_SENSORS,), 0.5, dtype=np.float32)
        alpha = alpha.astype(np.float32)

        end = HISTORY_LENGTH - HORIZON
        start = end - self.t_hist
        Yt = X[start:end, :]
        P = X[start - DAY_LAG : end - DAY_LAG, :]
        W = X[start - WEEK_LAG : end - WEEK_LAG, :]
        base_hist = alpha[None, :] * P + (1.0 - alpha[None, :]) * W
        r_hist = Yt - base_hist

        r_mu = self.art.r_mu
        r_sigma = self.art.r_sigma
        if r_mu is None or r_sigma is None:
            r_mu = np.nanmean(r_hist, axis=0)
            r_sigma = np.nanstd(r_hist, axis=0) + 1e-6
        r_mu = np.asarray(r_mu, dtype=np.float32)
        r_sigma = np.where(np.asarray(r_sigma, dtype=np.float32) > 0, np.asarray(r_sigma, dtype=np.float32), 1.0)

        r_hist_norm = (r_hist - r_mu[None, :]) / r_sigma[None, :]
        r_hist_norm = np.nan_to_num(r_hist_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Weather features for horizon (robust coercion)
        Wf = _coerce_weather(weather_forecast)
        if Wf is None:
            Wf = np.zeros((HORIZON, self.art.w_dim), dtype=np.float32)

        # Match model w_dim
        if Wf.shape[1] < self.art.w_dim:
            pad = np.zeros((HORIZON, self.art.w_dim - Wf.shape[1]), dtype=np.float32)
            Wf = np.concatenate([Wf, pad], axis=1)
        elif Wf.shape[1] > self.art.w_dim:
            Wf = Wf[:, : self.art.w_dim]

        # Normalize weather if stats exist
        if self.art.w_mu is not None and self.art.w_sigma is not None:
            mu = np.asarray(self.art.w_mu, dtype=np.float32)
            sig = np.where(np.asarray(self.art.w_sigma, dtype=np.float32) > 0, np.asarray(self.art.w_sigma, dtype=np.float32), 1.0)
            if mu.shape[0] == Wf.shape[1]:
                Wf = (Wf - mu[None, :]) / sig[None, :]

        # Indices expected by HistConvResidual
        h_idx = np.arange(HORIZON, dtype=np.int64)[None, :]
        s_idx = np.arange(N_SENSORS, dtype=np.int64)[None, None, :].repeat(HORIZON, axis=1)

        # Baseline injection normalized (simple z-score from sensor history window)
        yb_mu = np.nanmean(X, axis=0).astype(np.float32)
        yb_sigma = (np.nanstd(X, axis=0).astype(np.float32) + 1e-6)
        yb_s_norm = (yb - yb_mu[None, :]) / yb_sigma[None, :]
        yb_s_norm = np.nan_to_num(yb_s_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        th_idx = torch.from_numpy(h_idx).to(self.device)
        ts_idx = torch.from_numpy(s_idx).to(self.device)
        tW = torch.from_numpy(Wf[None, :, :]).to(self.device)
        tR = torch.from_numpy(r_hist_norm[None, :, :]).to(self.device)
        tYb = torch.from_numpy(yb_s_norm[None, :, :]).to(self.device)

        with torch.no_grad():
            r_hat_norm = self.model(th_idx, ts_idx, tW, tR, yb_s_norm=tYb)
            r_hat_norm = r_hat_norm.cpu().numpy()[0].astype(np.float32)

        r_hat = r_hat_norm * r_sigma[None, :] + r_mu[None, :]
        y_hat = yb + r_hat
        y_hat = np.clip(y_hat, 0.0, np.inf)
        return y_hat.astype(np.float32)


_WRAPPER: Optional[_ResidualModelWrapper] = None


def predict(
    sensor_history: np.ndarray,
    timestamp: str,
    weather_forecast: Optional[np.ndarray] = None,
    weather_history: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Leaderboard-compatible predict() that matches the API contract:

      Inputs:
        - sensor_history: (672,45) float
        - timestamp: ISO string for forecast start
        - weather_forecast: (N,11) optional, per-station rows, same column order as weather_forecasts.csv
        - weather_history:  (N,21) optional, per-station rows, same column order as weather_observations.csv

      Output:
        - (72,45) float predictions

    Implements:
      1) Weekly-blend baseline Y_base (same as pipeline.py guarantee)
      2) Residual history (normalized) consistent with training pipeline
      3) Parse weather_forecast into aligned (72, w_dim) features WITHOUT W_forecast.dat
      4) If weather missing/invalid, use zeros
      5) Model predicts normalized residuals
      6) Unnormalize -> Y_hat = Y_base + residual
      7) Return (72,45)
    """
    import numpy as np
    from pathlib import Path
    from datetime import datetime, timezone

    # -------------------------
    # Constants (must match training)
    # -------------------------
    HISTORY_LENGTH = 672
    HORIZON = 72
    N_SENSORS = 45
    WEEK_LAG = 168
    DAY_LAG = 72

    # Forecast CSV column indices (matches your pipeline.py)
    FC_STATION = 1
    FC_TEMP = 2
    FC_WIND = 3
    FC_CLOUD = 4
    FC_DEW = 8
    FC_RAINACC = 9
    FC_VALUE_DT = 10

    FORECAST_COLS = [FC_TEMP, FC_WIND, FC_CLOUD, FC_DEW, FC_RAINACC]  # numeric cols
    # We build main/other aggregates => w_dim = 2*len(FORECAST_COLS) = 10 (if data present)

    # -------------------------
    # Helpers
    # -------------------------
    def _parse_iso(ts: str) -> datetime:
        s = ts.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except Exception:
            dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _to_naive_hour(dt: datetime) -> np.datetime64:
        # UTC -> naive hour
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return np.datetime64(dt.replace(minute=0, second=0, microsecond=0), "h")

    def _weekly_blend_baseline_72(sensor_hist_672x45: np.ndarray, alpha_45: np.ndarray) -> np.ndarray:
        X = sensor_hist_672x45.astype(np.float32, copy=False)
        P = X[HISTORY_LENGTH - HORIZON : HISTORY_LENGTH, :]  # last 72h
        W = X[HISTORY_LENGTH - HORIZON - WEEK_LAG : HISTORY_LENGTH - WEEK_LAG, :]  # same hours last week
        a = alpha_45.astype(np.float32).reshape(1, -1)
        return a * P + (1.0 - a) * W  # (72,45)

    def _safe_float_col(col: np.ndarray) -> np.ndarray:
        """Try to coerce a 1D object col to float; NaN where impossible."""
        out = np.full((col.shape[0],), np.nan, dtype=np.float32)
        for i, v in enumerate(col):
            try:
                if v is None:
                    continue
                out[i] = float(v)
            except Exception:
                continue
        return out

    def _build_Wf_from_api_rows(
        wf_rows: Optional[np.ndarray],
        t0_hour: np.datetime64,
    ) -> np.ndarray:
        """
        Convert API weather_forecast rows (N,11) into (72, w_dim) float32.
        Policy (no leakage, consistent with pipeline):
          For each target hour (t0+k) and each station,
            pick row with latest issue_hour <= t0.
          Then aggregate numeric features:
            main station = most frequent station_id in provided rows
            other        = average of all other stations
          If no eligible rows for a target hour => zeros for that hour.
        """
        w_dim = 10  # 5 vars * (main, other)
        out = np.zeros((HORIZON, w_dim), dtype=np.float32)

        if wf_rows is None:
            return out

        a = np.asarray(wf_rows)
        if a.ndim != 2 or a.shape[1] < 11:
            return out

        # Parse target_hour (col 0) and issue_hour (col 10) -> datetime64[h]
        # NOTE: a is often dtype=object; convert via numpy string then datetime64
        try:
            target = a[:, 0].astype(str)
            issue = a[:, FC_VALUE_DT].astype(str)
            target_h = target.astype("datetime64[h]")
            issue_h = issue.astype("datetime64[h]")
        except Exception:
            # If parsing fails, return zeros (don’t crash leaderboard)
            return out

        # Station ids (col 1)
        st = _safe_float_col(a[:, FC_STATION]).astype(np.int64, copy=False)
        if st.size == 0:
            return out

        # Main station: most frequent nonzero id in this request
        # (This matches your pipeline "main_station inferred" idea, but per-request.)
        try:
            uniq, cnt = np.unique(st, return_counts=True)
            main_station = int(uniq[int(np.argmax(cnt))])
        except Exception:
            main_station = int(st[0])

        # Numeric forecast vars
        # Build matrix vars: (N,5)
        vars_mat = np.stack([_safe_float_col(a[:, j]) for j in FORECAST_COLS], axis=1)  # float32 with NaNs
        vars_mat = np.nan_to_num(vars_mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Precompute for each horizon hour k:
        #   eligible rows must satisfy: target_h == (t0+k) AND issue_h <= t0
        # Then among eligible rows, per-station choose max(issue_h), then aggregate
        t0h = t0_hour.astype("datetime64[h]")
        # mask for <= t0 once
        le_mask = issue_h <= t0h

        # We'll do hour-by-hour to keep code simple and robust.
        for k in range(HORIZON):
            tgt_h = (t0h + np.timedelta64(k, "h")).astype("datetime64[h]")
            m = (target_h == tgt_h) & le_mask
            if not np.any(m):
                continue

            # Eligible subset
            st_k = st[m]
            issue_k = issue_h[m]
            vars_k = vars_mat[m]  # (M,5)

            # For each station, pick the rows with max issue time (latest <= t0)
            # We'll do this by sorting by (station, issue) and taking last per station.
            order = np.lexsort((issue_k.astype("datetime64[h]").astype(np.int64), st_k))
            st_s = st_k[order]
            issue_s = issue_k[order]
            vars_s = vars_k[order]

            # indices of last occurrence per station
            # find boundaries
            if st_s.size == 0:
                continue
            change = np.r_[True, st_s[1:] != st_s[:-1]]
            starts = np.flatnonzero(change)
            ends = np.r_[starts[1:] - 1, st_s.size - 1]
            pick_idx = ends  # last in each station group = latest issue due to sort

            st_pick = st_s[pick_idx]
            vars_pick = vars_s[pick_idx]  # (n_station,5)

            # Aggregate main/other
            main_mask = st_pick == main_station
            if np.any(main_mask):
                main_vals = vars_pick[main_mask].mean(axis=0)
            else:
                main_vals = np.zeros((5,), dtype=np.float32)

            other_mask = ~main_mask
            if np.any(other_mask):
                other_vals = vars_pick[other_mask].mean(axis=0)
            else:
                other_vals = np.zeros((5,), dtype=np.float32)

            # Pack to (10,)
            out[k, 0:5] = main_vals
            out[k, 5:10] = other_vals

        return out

    # -------------------------
    # Validate & coerce sensor_history
    # -------------------------
    X = np.asarray(sensor_history, dtype=np.float32)
    if X.shape != (HISTORY_LENGTH, N_SENSORS):
        raise ValueError(f"sensor_history must be shape ({HISTORY_LENGTH},{N_SENSORS}), got {X.shape}")

    # Forecast start time t0 (hour)
    t0_dt = _parse_iso(timestamp)
    t0_hour = _to_naive_hour(t0_dt)

    # -------------------------
    # Load artifacts (alpha, residual stats, model weights)
    # (No memmap reliance, only small npy/pt artifacts)
    # -------------------------
    root = Path(__file__).resolve().parent

    # Alpha for weekly blend
    alpha_path = root / "memmaps" / "alpha.npy"
    if alpha_path.exists():
        alpha = np.load(alpha_path).astype(np.float32)
        if alpha.shape != (N_SENSORS,):
            alpha = np.full((N_SENSORS,), 0.5, dtype=np.float32)
    else:
        alpha = np.full((N_SENSORS,), 0.5, dtype=np.float32)

    # Residual normalization stats
    r_mean_path = root / "memmaps" / "r_mean.npy"
    r_std_path = root / "memmaps" / "r_std.npy"
    if r_mean_path.exists() and r_std_path.exists():
        r_mean = np.load(r_mean_path).astype(np.float32)
        r_std = np.load(r_std_path).astype(np.float32)
        if r_mean.shape != (N_SENSORS,) or r_std.shape != (N_SENSORS,):
            r_mean = np.zeros((N_SENSORS,), dtype=np.float32)
            r_std = np.ones((N_SENSORS,), dtype=np.float32)
    else:
        r_mean = np.zeros((N_SENSORS,), dtype=np.float32)
        r_std = np.ones((N_SENSORS,), dtype=np.float32)
    r_std = np.where(r_std > 0, r_std, 1.0).astype(np.float32)

    # -------------------------
    # 1) Weekly-blend baseline Y_base (72,45)
    # -------------------------
    Y_base = _weekly_blend_baseline_72(X, alpha)  # (72,45)

    # -------------------------
    # 2) Residual history R_hist_norm (consistent with training pipeline)
    # Training pipeline used:
    #   baseline_hist[t] = alpha*X[t-72] + (1-alpha)*X[t-168]
    #   residual_hist[t] = X[t] - baseline_hist[t]
    # and then normalized with r_mean/r_std per sensor.
    #
    # We'll feed the last 168 hours of residual history ending at t0 (i.e. up to X[599]).
    # -------------------------
    t_hist = 168  # keep aligned with your HistConvResidual training default
    end = HISTORY_LENGTH - HORIZON          # 600 (t0 is at index 600)
    start = end - t_hist                   # 432
    Yt = X[start:end, :]                   # (168,45) = X[432:600]
    P = X[start - DAY_LAG : end - DAY_LAG, :]   # X[360:528]
    W = X[start - WEEK_LAG : end - WEEK_LAG, :] # X[264:432]
    base_hist = alpha.reshape(1, -1) * P + (1.0 - alpha.reshape(1, -1)) * W
    R_hist = Yt - base_hist
    R_hist_norm = (R_hist - r_mean.reshape(1, -1)) / r_std.reshape(1, -1)
    R_hist_norm = np.nan_to_num(R_hist_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)  # (168,45)

    # -------------------------
    # 3) Parse weather_forecast into (72, w_dim) WITHOUT W_forecast.dat
    # -------------------------
    Wf = _build_Wf_from_api_rows(weather_forecast, t0_hour)  # (72,10)

    # -------------------------
    # 4) If weather missing/invalid -> zeros (already ensured)
    # -------------------------

    # -------------------------
    # 5) Run your learned model to predict normalized residuals
    # -------------------------
    # Fallback if torch/weights missing: return baseline only (safe)
    weights_path = root / "histconv.pt"
    try:
        import torch
        from model_good import HistConvResidual  # your provided model

        if not weights_path.exists():
            # no learned weights -> baseline
            return Y_base.astype(np.float32)

        # Load once (cached as function attributes)
        if not hasattr(predict, "_net"):
            ckpt = torch.load(weights_path, map_location="cpu")
            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

            # Infer dims from state dict
            w_dim = None
            h_dim = None
            c_dim = 1
            for k, v in state.items():
                if k.endswith("w_proj.weight"):
                    w_dim = int(v.shape[1])
                if k.endswith("h_emb.weight"):
                    h_dim = int(v.shape[1])
                if k.endswith("in_proj.weight"):
                    c_dim = int(v.shape[1])

            if w_dim is None:
                w_dim = Wf.shape[1]
            if h_dim is None:
                raise RuntimeError("Could not infer h_dim from checkpoint.")

            net = HistConvResidual(
                n_sensors=N_SENSORS,
                horizon=HORIZON,
                h_dim=h_dim,
                w_dim=w_dim,
                c_dim=c_dim,
            )
            net.load_state_dict(state, strict=False)
            net.eval()

            predict._net = net
            predict._w_dim = w_dim

        net = predict._net
        w_dim_need = int(getattr(predict, "_w_dim", Wf.shape[1]))

        # Match model w_dim
        if Wf.shape[1] < w_dim_need:
            Wf_use = np.pad(Wf, ((0, 0), (0, w_dim_need - Wf.shape[1])), mode="constant")
        else:
            Wf_use = Wf[:, :w_dim_need]
        Wf_use = Wf_use.astype(np.float32, copy=False)

        # Build required indices + tensors (model_good forward signature)
        # HistConvResidual expects:
        #   h_idx: (B,H)
        #   s_idx: (B,H,S)
        #   Wf:    (B,H,w_dim)
        #   R_hist:(B,t_hist,S)
        # plus optional yb_s_norm (we provide a simple normalized baseline injection)
        h_idx = np.arange(HORIZON, dtype=np.int64)[None, :]
        s_idx = np.arange(N_SENSORS, dtype=np.int64)[None, None, :].repeat(HORIZON, axis=1)

        # Baseline injection normalized (simple z-score over sensor_history)
        mu_x = np.nanmean(X, axis=0).astype(np.float32)
        sig_x = (np.nanstd(X, axis=0).astype(np.float32) + 1e-6)
        yb_s_norm = (Y_base - mu_x.reshape(1, -1)) / sig_x.reshape(1, -1)
        yb_s_norm = np.nan_to_num(yb_s_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        with torch.no_grad():
            th = torch.from_numpy(h_idx)
            ts = torch.from_numpy(s_idx)
            tW = torch.from_numpy(Wf_use[None, :, :])
            tR = torch.from_numpy(R_hist_norm[None, :, :])
            tYb = torch.from_numpy(yb_s_norm[None, :, :])

            r_hat_norm = net(th, ts, tW, tR, yb_s_norm=tYb)  # (1,72,45)
            r_hat_norm = r_hat_norm.cpu().numpy()[0].astype(np.float32)

    except Exception:
        # If anything about the learned model fails, keep it leaderboard-safe
        return Y_base.astype(np.float32)

    # -------------------------
    # 6) Unnormalize residuals & add baseline -> predictions
    # -------------------------
    r_hat = r_hat_norm * r_std.reshape(1, -1) + r_mean.reshape(1, -1)
    Y_hat = Y_base + r_hat
    Y_hat = np.nan_to_num(Y_hat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    Y_hat = np.clip(Y_hat, 0.0, np.inf)

    # -------------------------
    # 7) Return (72,45)
    # -------------------------
    if Y_hat.shape != (HORIZON, N_SENSORS):
        # ultimate safety net
        return Y_base.astype(np.float32)
    return Y_hat
