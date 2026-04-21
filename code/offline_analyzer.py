import os
import zipfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from collections import Counter

# -----------------------
# CONFIGURATION
# -----------------------
# Put the name of the zip file for analysis
FILE_TO_ANALYZE = "Mix activity data for offline analysis.zip" 

MODEL_PATH = Path("outputs_svm/har_svm.joblib")
FEATURES_PATH = Path("outputs_svm/feature_columns.txt")

WINDOW_SEC = 2.0
OVERLAP = 0.5
SENSOR_FILES = [
    "Accelerometer.csv",
    "Gyroscope.csv",
    "Linear Accelerometer.csv",
    "Gravity.csv",
]

# -----------------------
# PREPROCESSING HELPERS (From the pipeline)
# -----------------------
def read_sensor_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    t = df.iloc[:, 0].astype(float).to_numpy()
    xyz = df.iloc[:, 1:4].astype(float).to_numpy()
    return t, xyz

def merge_sensors_on_time(sensor_data: dict):
    t_ref, acc = sensor_data["acc"]
    out = {"time": t_ref,
           "accX": acc[:, 0], "accY": acc[:, 1], "accZ": acc[:, 2]}

    for key, prefix in [("gyro", "gyro"), ("lin", "lin"), ("gra", "gra")]:
        t, xyz = sensor_data[key]
        out[f"{prefix}X"] = np.interp(t_ref, t, xyz[:, 0])
        out[f"{prefix}Y"] = np.interp(t_ref, t, xyz[:, 1])
        out[f"{prefix}Z"] = np.interp(t_ref, t, xyz[:, 2])

    return pd.DataFrame(out)

def estimate_fs(t):
    dt = np.diff(t)
    if len(dt) < 3: return None
    return 1.0 / np.median(dt)

def clip_spikes_1d(x, z=5.0):
    mu = np.mean(x)
    sd = np.std(x) + 1e-9
    zscores = (x - mu) / sd
    y = x.copy()
    for i in range(len(y)):
        if abs(zscores[i]) > z:
            y[i] = y[i-1] if i > 0 else mu
    return y

def build_features(window_df: pd.DataFrame):
    feats = {}
    def add_axis_feats(prefix):
        for axis in ["X", "Y", "Z"]:
            v = window_df[f"{prefix}{axis}"].to_numpy()
            feats[f"{prefix}{axis}_mean"] = float(np.mean(v))
            feats[f"{prefix}{axis}_std"]  = float(np.std(v))
            feats[f"{prefix}{axis}_min"]  = float(np.min(v))
            feats[f"{prefix}{axis}_max"]  = float(np.max(v))
        vX = window_df[f"{prefix}X"].to_numpy()
        vY = window_df[f"{prefix}Y"].to_numpy()
        vZ = window_df[f"{prefix}Z"].to_numpy()
        mag = np.sqrt(vX*vX + vY*vY + vZ*vZ)
        mag = clip_spikes_1d(mag, z=5.0)
        feats[f"{prefix}_mag_mean"] = float(np.mean(mag))
        feats[f"{prefix}_mag_std"]  = float(np.std(mag))
        feats[f"{prefix}_mag_min"]  = float(np.min(mag))
        feats[f"{prefix}_mag_max"]  = float(np.max(mag))
    add_axis_feats("acc")
    add_axis_feats("gyro")
    add_axis_feats("lin")
    add_axis_feats("gra")
    return feats

def load_zip_session(zip_path: str):
    tmp_dir = Path("tmp_analyzer")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp_dir)
        
    sensor_map = {}
    for f in SENSOR_FILES:
        p = tmp_dir / f
        if not p.exists():
            matches = [x for x in tmp_dir.glob("*") if x.name.lower() == f.lower()]
            if matches: p = matches[0]
            else: raise FileNotFoundError(f"Missing {f} in {zip_path}")
        t, xyz = read_sensor_csv(p)
        if f.lower().startswith("accelerometer"): sensor_map["acc"] = (t, xyz)
        elif f.lower().startswith("gyroscope"): sensor_map["gyro"] = (t, xyz)
        elif f.lower().startswith("linear"): sensor_map["lin"] = (t, xyz)
        elif f.lower().startswith("gravity"): sensor_map["gra"] = (t, xyz)
        
    return merge_sensors_on_time(sensor_map)

def windows_from_session(df: pd.DataFrame):
    t = df["time"].to_numpy()
    fs = estimate_fs(t)
    if fs is None: return []
    win_len = int(round(WINDOW_SEC * fs))
    step = int(round(win_len * (1.0 - OVERLAP)))
    
    rows = []
    for start in range(0, len(df) - win_len + 1, step):
        w = df.iloc[start:start + win_len]
        rows.append(build_features(w))
    return rows

# -----------------------
# MAIN ANALYSIS LOGIC
# -----------------------
def analyze_recording(zip_file_path):
    print(f"\n--- Analyzing Recording: {zip_file_path} ---")
    
    if not os.path.exists(zip_file_path):
        print(f"ERROR: Could not find file '{zip_file_path}'. Please check the path.")
        return

    # 1. Load Model and Feature Order
    try:
        clf = joblib.load(MODEL_PATH)
        feature_cols = FEATURES_PATH.read_text().splitlines()
    except Exception as e:
        print("ERROR: Could not load model or feature columns. Did you run train_svm.py first?")
        return

    # 2. Extract and Preprocess
    print("Extracting and aligning sensor data...")
    merged_df = load_zip_session(zip_file_path)
    
    print("Segmenting into 2-second windows...")
    feature_rows = windows_from_session(merged_df)
    
    if not feature_rows:
        print("ERROR: Not enough data in recording to create a 2-second window.")
        return

    # 3. Predict
    print(f"Extracted {len(feature_rows)} valid windows. Running predictions...")
    X = pd.DataFrame(feature_rows)
    X = X.reindex(columns=feature_cols) # Ensure columns match exactly
    
    predictions = clf.predict(X)
    
    # 4. Generate Report
    counts = Counter(predictions)
    total = len(predictions)
    
    print("\n==============================")
    print("      DIAGNOSTIC REPORT       ")
    print("==============================")
    print(f"Total Time Analyzed: ~{total} seconds (with 50% overlap)")
    print("Activity Breakdown:")
    
    for activity, count in counts.most_common():
        percentage = (count / total) * 100
        print(f"  - {activity.capitalize()}: {count} windows ({percentage:.1f}%)")
    print("==============================\n")


if __name__ == "__main__":
    # Change 'test_recording.zip' to the name of any zip file in your folder!
    analyze_recording(FILE_TO_ANALYZE)
