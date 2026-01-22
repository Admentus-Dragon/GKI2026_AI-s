# save_score_weights.py
import numpy as np
from pathlib import Path

OUT = Path("memmaps")
OUT.mkdir(exist_ok=True)

data = np.load("data/train.npz", allow_pickle=True)
Y = data["y_train"].astype(np.float32)  # (N,72,S)

mean_flow = np.nanmean(Y, axis=(0, 1))  # (S,)
w = np.sqrt(np.maximum(mean_flow, 0.0))
w_sum = np.nansum(w)

if not np.isfinite(w_sum) or w_sum <= 0:
    w = np.ones_like(w, dtype=np.float32) / float(len(w))
else:
    w = (w / w_sum).astype(np.float32)

np.save(OUT / "score_weights.npy", w)
print("[DONE] wrote memmaps/score_weights.npy", w.shape, "sum=", float(w.sum()))
