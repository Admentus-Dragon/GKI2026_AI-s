# build_train_npz.py
import pandas as pd
import numpy as np
from pathlib import Path

HIST = 672
HOR  = 72

def main():
    df = pd.read_csv("data/sensor_timeseries.csv", parse_dates=["CTime"])
    df = df.sort_values("CTime")

    sensors = [c for c in df.columns if c.startswith("M")] + ["FRAMRENNSLI_TOTAL"]
    arr = df[sensors].to_numpy(np.float32)

    X, Y, T = [], [], []

    for i in range(len(arr) - HIST - HOR):
        X.append(arr[i:i+HIST])
        Y.append(arr[i+HIST:i+HIST+HOR])
        T.append(df["CTime"].iloc[i+HIST])

    np.savez(
        "data/train.npz",
        X_train=np.stack(X),
        y_train=np.stack(Y),
        timestamps=np.array(T),
        sensor_names=np.array(sensors)
    )

    print("Saved train.npz", len(X))

if __name__ == "__main__":
    main()
