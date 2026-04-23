import os
import json
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# CONFIG
# ============================================================
LOCAL_FILE_PATH = "data/stock-data.json"
RUNS_BASE_DIR = "runs"
MODEL_NAME = "random_forest"

DATA_ROWS_TO_USE = 50000   # None = use all rows
WINDOW = 100

MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 20,
    "random_state": 42,
    "n_jobs": 16,
    "class_weight": "balanced"
}

FEATURES = [
    "return",
    "volume_change",
    "high_low",
    "open_close",
    "rolling_mean_5",
    "rolling_std_5",
    "rolling_mean_10",
    "rolling_std_10",
    "momentum_3",
    "momentum_5"
]

# ============================================================
# HELPERS
# ============================================================
def log_print(lines, text=""):
    print(text)
    lines.append(str(text))

def section(lines, title):
    line = "=" * 70
    log_print(lines, f"\n{line}")
    log_print(lines, title)
    log_print(lines, line)

# ============================================================
# START RUN
# ============================================================
run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_date = datetime.now().strftime("%Y-%m-%d")
run_dir = os.path.join(RUNS_BASE_DIR, MODEL_NAME, run_date)
os.makedirs(run_dir, exist_ok=True)

report_path = os.path.join(run_dir, f"run_{run_timestamp}.txt")

report_lines = []

section(report_lines, "RUN METADATA")
log_print(report_lines, f"Run timestamp: {run_timestamp}")
log_print(report_lines, f"Run directory: {run_dir}")
log_print(report_lines, "Data source: LOCAL FILE")
log_print(report_lines, f"File path: {LOCAL_FILE_PATH}")
log_print(report_lines, f"Rows to use: {DATA_ROWS_TO_USE}")
log_print(report_lines, f"Window size: {WINDOW}")
log_print(report_lines, f"Features: {FEATURES}")
log_print(report_lines, f"Model params: {MODEL_PARAMS}")

# ============================================================
# 1. LOAD DATA
# ============================================================
section(report_lines, "1. LOAD DATA")

if not os.path.exists(LOCAL_FILE_PATH):
    raise FileNotFoundError(f"File not found: {LOCAL_FILE_PATH}")

with open(LOCAL_FILE_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

if not isinstance(raw, dict) or "data" not in raw:
    raise ValueError("Invalid JSON format: expected {'data': [...]}")

df = pd.DataFrame(raw["data"])

log_print(report_lines, f"Rows in file: {len(df)}")
log_print(report_lines, f"Columns: {df.columns.tolist()}")

# ============================================================
# 2. CLEANING
# ============================================================
section(report_lines, "2. CLEANING")

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

if DATA_ROWS_TO_USE is not None:
    df = df.tail(DATA_ROWS_TO_USE).reset_index(drop=True)

log_print(report_lines, f"Rows after cleaning / row cap: {len(df)}")

# ============================================================
# 3. FEATURES
# ============================================================
section(report_lines, "3. FEATURES")

df["next_close"] = df["close"].shift(-1)
df["target"] = (df["next_close"] > df["close"]).astype(int)

df["return"] = df["close"].pct_change()
df["volume_change"] = df["volume"].pct_change()
df["high_low"] = df["high"] - df["low"]
df["open_close"] = df["close"] - df["open"]
df["rolling_mean_5"] = df["close"].rolling(5).mean()
df["rolling_std_5"] = df["close"].rolling(5).std()
df["rolling_mean_10"] = df["close"].rolling(10).mean()
df["rolling_std_10"] = df["close"].rolling(10).std()
df["momentum_3"] = df["close"] - df["close"].shift(3)
df["momentum_5"] = df["close"] - df["close"].shift(5)

df = df.dropna().reset_index(drop=True)

log_print(report_lines, f"Rows after feature engineering: {len(df)}")

# ============================================================
# 4. WINDOWS
# ============================================================
section(report_lines, "4. WINDOWS")

X, y = [], []

for i in range(WINDOW, len(df)):
    X.append(df[FEATURES].iloc[i - WINDOW:i].values)
    y.append(df["target"].iloc[i])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int8)

log_print(report_lines, f"X shape: {X.shape}")

X_flat = X.reshape(X.shape[0], -1)

# ============================================================
# 5. SPLIT
# ============================================================
section(report_lines, "5. SPLIT")

n = len(X_flat)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

X_train, y_train = X_flat[:train_end], y[:train_end]
X_val, y_val = X_flat[train_end:val_end], y[train_end:val_end]
X_test, y_test = X_flat[val_end:], y[val_end:]

log_print(report_lines, f"Train rows: {len(X_train)}")
log_print(report_lines, f"Val rows:   {len(X_val)}")
log_print(report_lines, f"Test rows:  {len(X_test)}")

# ============================================================
# 6. MODEL
# ============================================================
section(report_lines, "6. MODEL")

model = RandomForestClassifier(**MODEL_PARAMS)
model.fit(X_train, y_train)

# ============================================================
# 7. RESULTS
# ============================================================
section(report_lines, "7. RESULTS")

val_preds = model.predict(X_val)
test_preds = model.predict(X_test)

val_acc = accuracy_score(y_val, val_preds)
test_acc = accuracy_score(y_test, test_preds)

log_print(report_lines, f"Val Accuracy: {val_acc:.6f}")
log_print(report_lines, f"Test Accuracy: {test_acc:.6f}")
log_print(report_lines, classification_report(y_test, test_preds))
log_print(report_lines, str(confusion_matrix(y_test, test_preds)))

# ============================================================
# SAVE
# ============================================================
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"\nSaved: {report_path}")