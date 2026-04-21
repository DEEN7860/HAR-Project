import time
import requests
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import deque

# -----------------------
# SETTINGS
# -----------------------
MODEL_PATH = Path("outputs/har_random_forest.joblib")
FEATURES_PATH = Path("outputs/feature_columns.txt")

URL = (
    "http://192.168.0.154/get?"
    "accX=full&accY=full&accZ=full&acc_time=full&"
    "gyroX=full&gyroY=full&gyroZ=full&gyro_time=full&"
    "lin_accX=full&lin_accY=full&lin_accZ=full&lin_acc_time=full&"
    "graX=full&graY=full&graZ=full&graT=full"
)

WINDOW_SEC = 2.0
FS = 50
WIN_LEN = int(WINDOW_SEC * FS)

# Stillness thresholds 
LIN_STD_THRESH = 0.25
GYRO_MEAN_THRESH = 0.25

SMOOTHING_WINDOW = 5

# -----------------------
# LOAD MODEL
# -----------------------
clf = joblib.load(MODEL_PATH)
feature_cols = FEATURES_PATH.read_text().splitlines()
pred_buffer = deque(maxlen=SMOOTHING_WINDOW)

# -----------------------
# HELPERS
# -----------------------
def clip_spikes_1d(x, z=5.0):
    mu = np.mean(x)
    sd = np.std(x) + 1e-9
    zscores = (x - mu) / sd
    y = x.copy()
    for i in range(len(y)):
        if abs(zscores[i]) > z:
            y[i] = y[i-1] if i > 0 else mu
    return y

def build_features_from_arrays(acc, gyro, lin, gra):
    feats = {}

    def add(prefix, arr):
        for i, axis in enumerate(["X", "Y", "Z"]):
            v = arr[:, i]
            feats[f"{prefix}{axis}_mean"] = float(np.mean(v))
            feats[f"{prefix}{axis}_std"]  = float(np.std(v))
            feats[f"{prefix}{axis}_min"]  = float(np.min(v))
            feats[f"{prefix}{axis}_max"]  = float(np.max(v))

        mag = np.linalg.norm(arr, axis=1)
        mag = clip_spikes_1d(mag, z=5.0)

        feats[f"{prefix}_mag_mean"] = float(np.mean(mag))
        feats[f"{prefix}_mag_std"]  = float(np.std(mag))
        feats[f"{prefix}_mag_min"]  = float(np.min(mag))
        feats[f"{prefix}_mag_max"]  = float(np.max(mag))

    add("acc", acc)
    add("gyro", gyro)
    add("lin", lin)
    add("gra", gra)

    return feats

def majority_vote(buffer):
    return max(set(buffer), key=buffer.count)

# -----------------------
# MAIN LOOP
# -----------------------
while True:
    try:
        r = requests.get(URL, timeout=2).json()
        b = r["buffer"]

        def lastN(name):
            arr = b[name]["buffer"]
            return np.array(arr[-WIN_LEN:], dtype=float)

        acc = np.stack([lastN("accX"), lastN("accY"), lastN("accZ")], axis=1)
        gyro = np.stack([lastN("gyroX"), lastN("gyroY"), lastN("gyroZ")], axis=1)
        lin = np.stack([lastN("lin_accX"), lastN("lin_accY"), lastN("lin_accZ")], axis=1)
        gra = np.stack([lastN("graX"), lastN("graY"), lastN("graZ")], axis=1)

        # -----------------------
        # STILLNESS GATE
        # -----------------------
        lin_mag = np.linalg.norm(lin, axis=1)
        gyro_mag = np.linalg.norm(gyro, axis=1)

        if np.std(lin_mag) < LIN_STD_THRESH and np.mean(gyro_mag) < GYRO_MEAN_THRESH:
            raw_pred = "standing"
        else:
            feats = build_features_from_arrays(acc, gyro, lin, gra)
            X = pd.DataFrame([feats])
            X = X.reindex(columns=feature_cols)
            raw_pred = clf.predict(X)[0]

        # -----------------------
        # SMOOTHING
        # -----------------------
        pred_buffer.append(raw_pred)
        smooth_pred = majority_vote(list(pred_buffer))

        print("Predicted activity:", smooth_pred)

    except Exception as e:
        print("Waiting / error:", e)

    time.sleep(0.25)
