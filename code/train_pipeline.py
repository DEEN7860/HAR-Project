import os
import zipfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib


# -----------------------
# SETTINGS
# -----------------------
DATA_ROOT = Path("data")         # contains subfolders: walking, sitting, standing, running
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Windowing
FS_TARGET = 50                   # expected sampling rate
WINDOW_SEC = 2.0                 # 2 seconds (100 samples at 50 Hz)
OVERLAP = 0.5                    # 50% overlap

# Model
N_ESTIMATORS = 300
RANDOM_STATE = 42

# Expected sensor filenames inside each session zip
SENSOR_FILES = [
    "Accelerometer.csv",
    "Gyroscope.csv",
    "Linear Accelerometer.csv",
    "Gravity.csv",
]

# -----------------------
# HELPERS
# -----------------------

def read_sensor_csv(csv_path: Path):
    """Read phyphox-exported sensor CSV. Assumes first col is time, next 3 are X,Y,Z."""
    df = pd.read_csv(csv_path)
    t = df.iloc[:, 0].astype(float).to_numpy()
    xyz = df.iloc[:, 1:4].astype(float).to_numpy()
    return t, xyz

def merge_sensors_on_time(sensor_data: dict):
    """
    Merge sensors using the Accelerometer time basei.e. use acc time as the reference timeline,
    then interpolate other sensors onto that timeline.
    Returns a DataFrame with columns:
    time, accX/Y/Z, gyroX/Y/Z, linX/Y/Z, graX/Y/Z
    """
    # Use accelerometer timeline as reference
    t_ref, acc = sensor_data["acc"]
    out = {"time": t_ref,
           "accX": acc[:, 0], "accY": acc[:, 1], "accZ": acc[:, 2]}

    for key, prefix in [("gyro", "gyro"), ("lin", "lin"), ("gra", "gra")]:
        t, xyz = sensor_data[key]
        # Interpolate each axis onto t_ref
        out[f"{prefix}X"] = np.interp(t_ref, t, xyz[:, 0])
        out[f"{prefix}Y"] = np.interp(t_ref, t, xyz[:, 1])
        out[f"{prefix}Z"] = np.interp(t_ref, t, xyz[:, 2])

    return pd.DataFrame(out)

def estimate_fs(t):
    dt = np.diff(t)
    if len(dt) < 3:
        return None
    return 1.0 / np.median(dt)

def clip_spikes_1d(x, z=5.0):
    """Simple spike clipping on a 1D signal."""
    mu = np.mean(x)
    sd = np.std(x) + 1e-9
    zscores = (x - mu) / sd
    y = x.copy()
    for i in range(len(y)):
        if abs(zscores[i]) > z:
            y[i] = y[i-1] if i > 0 else mu
    return y

def build_features(window_df: pd.DataFrame):
    """
    Feature extraction for one window.
    Produces a compact, report-friendly feature set:
    mean/std/min/max per axis for key channels, plus magnitudes.
    """
    feats = {}

    def add_axis_feats(prefix):
        for axis in ["X", "Y", "Z"]:
            v = window_df[f"{prefix}{axis}"].to_numpy()
            feats[f"{prefix}{axis}_mean"] = float(np.mean(v))
            feats[f"{prefix}{axis}_std"]  = float(np.std(v))
            feats[f"{prefix}{axis}_min"]  = float(np.min(v))
            feats[f"{prefix}{axis}_max"]  = float(np.max(v))

        # magnitude features
        vX = window_df[f"{prefix}X"].to_numpy()
        vY = window_df[f"{prefix}Y"].to_numpy()
        vZ = window_df[f"{prefix}Z"].to_numpy()
        mag = np.sqrt(vX*vX + vY*vY + vZ*vZ)

        # optional spike clipping on magnitude (helps treadmill transitions)
        mag = clip_spikes_1d(mag, z=5.0)

        feats[f"{prefix}_mag_mean"] = float(np.mean(mag))
        feats[f"{prefix}_mag_std"]  = float(np.std(mag))
        feats[f"{prefix}_mag_min"]  = float(np.min(mag))
        feats[f"{prefix}_mag_max"]  = float(np.max(mag))

    # Add for each sensor vector
    add_axis_feats("acc")
    add_axis_feats("gyro")
    add_axis_feats("lin")
    add_axis_feats("gra")

    return feats

def windows_from_session(df: pd.DataFrame, label: str, group_id: str):
    """
    Convert merged time-series into sliding windows -> feature rows.
    group_id identifies the session for leakage-safe splitting.
    """
    t = df["time"].to_numpy()
    fs = estimate_fs(t)
    if fs is None:
        return []

    win_len = int(round(WINDOW_SEC * fs))
    step = int(round(win_len * (1.0 - OVERLAP)))
    if win_len < 10 or step < 1:
        return []

    rows = []
    for start in range(0, len(df) - win_len + 1, step):
        w = df.iloc[start:start + win_len]
        feats = build_features(w)
        feats["label"] = label
        feats["group"] = group_id
        rows.append(feats)

    return rows

def load_zip_session(zip_path: Path):
    """
    Extract a session zip to temp folder, load sensor files,
    return merged dataframe.
    """
    tmp_dir = Path("tmp_extract")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp_dir)

    # Map expected files to keys
    sensor_map = {}
    for f in SENSOR_FILES:
        p = tmp_dir / f
        if not p.exists():
            # Try case-insensitive match
            matches = [x for x in tmp_dir.glob("*") if x.name.lower() == f.lower()]
            if matches:
                p = matches[0]
            else:
                raise FileNotFoundError(f"Missing {f} in {zip_path.name}")

        t, xyz = read_sensor_csv(p)

        if f.lower().startswith("accelerometer"):
            sensor_map["acc"] = (t, xyz)
        elif f.lower().startswith("gyroscope"):
            sensor_map["gyro"] = (t, xyz)
        elif f.lower().startswith("linear"):
            sensor_map["lin"] = (t, xyz)
        elif f.lower().startswith("gravity"):
            sensor_map["gra"] = (t, xyz)

    merged = merge_sensors_on_time(sensor_map)
    return merged


# -----------------------
# MAIN: BUILD DATASET
# -----------------------
def build_dataset():
    all_rows = []
    for class_dir in sorted(DATA_ROOT.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name.lower()

        for zip_path in sorted(class_dir.glob("*.zip")):
            session_id = f"{label}__{zip_path.stem}"
            merged = load_zip_session(zip_path)

            rows = windows_from_session(merged, label=label, group_id=session_id)
            all_rows.extend(rows)
            print(f"Loaded {zip_path.name}: {len(rows)} windows")

    df_feat = pd.DataFrame(all_rows)
    if df_feat.empty:
        raise RuntimeError("No data loaded. Check your folder structure and zip contents.")

    return df_feat


def train_and_evaluate(df_feat: pd.DataFrame):
    # Split by session group to avoid leakage (windows from same recording stay together)
    X = df_feat.drop(columns=["label", "group"])
    y = df_feat["label"]
    groups = df_feat["group"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("\nAccuracy:", round(acc, 4))
    print("\nClassification report:\n", classification_report(y_test, preds))

    labels_sorted = sorted(y.unique())
    cm = confusion_matrix(y_test, preds, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels_sorted],
                         columns=[f"pred_{l}" for l in labels_sorted])
    print("\nConfusion matrix:\n", cm_df)

    # Save artifacts
    model_path = OUTPUT_DIR / "har_random_forest.joblib"
    joblib.dump(clf, model_path)

    features_path = OUTPUT_DIR / "feature_columns.txt"
    features_path.write_text("\n".join(X.columns))

    print(f"\nSaved model to: {model_path}")
    print(f"Saved feature column order to: {features_path}")

    return clf


if __name__ == "__main__":
    df_feat = build_dataset()
    print(f"\nTotal windows: {len(df_feat)}")
    print("Classes:", df_feat["label"].value_counts().to_dict())

    # Optional: Save the features CSV for your report / appendix
    df_feat.to_csv(OUTPUT_DIR / "features_dataset.csv", index=False)

    train_and_evaluate(df_feat)
