import os
import json
import copy
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import TimeSeriesSplit

# ============================================================
# CONFIG
# ============================================================
LOCAL_FILE_PATH = "msft_data.json"
RUNS_DIR = "runs"

WINDOW = 120
PREDICTION_HORIZON = 60

FIXED_RETURN_THRESHOLD = None
TRAIN_MOVE_QUANTILE = 0.55

TEST_FRACTION = 0.15
CV_SPLITS = 3
CV_GAP = 20
RECENT_SUMMARY_WINDOW = 20

ENSEMBLE_SEEDS = [42, 1337, 2026]
MIN_SEED_OOF_AUC = 0.52

MAX_EPOCHS = 35
BATCH_SIZE = 512
MAX_LR = 5e-4
WEIGHT_DECAY = 5e-4
DROPOUT = 0.35
LABEL_SMOOTHING = 0.02
EARLY_STOPPING_PATIENCE = 6
NUM_WORKERS = 0
GRAD_CLIP_NORM = 1.0

THRESHOLD_MIN = 0.35
THRESHOLD_MAX = 0.65
THRESHOLD_STEPS = 121
MIN_POSITIVE_RATE = 0.35
MAX_POSITIVE_RATE = 0.65

CLIP_LOWER_Q = 0.005
CLIP_UPPER_Q = 0.995

HIDDEN_DIMS = [256, 128, 64]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_FEATURES = [
    "return_1",
    "return_5",
    "return_15",
    "return_30",
    "log_return_1",
    "volume_change_1",
    "volume_change_5",
    "high_low",
    "open_close",
    "body_to_range",
    "upper_wick",
    "lower_wick",
    "rolling_std_5",
    "rolling_std_20",
    "rolling_std_50",
    "ema_spread_5_20",
    "ema_spread_10_50",
    "return_std_5",
    "return_std_20",
    "range_std_5",
    "range_std_20",
    "volume_z_10",
    "volume_z_30",
    "momentum_5",
    "momentum_20",
    "momentum_30",
    "rsi_14",
    "session_progress",
    "minute_sin",
    "minute_cos",
    "dow_sin",
    "dow_cos",
]

SUMMARY_STATS = [
    "last",
    "mean",
    "std",
    "min",
    "max",
    "delta",
    "recent_minus_mean",
]

# ============================================================
# HELPERS
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_print(lines, text=""):
    print(text)
    lines.append(str(text))


def section(lines, title):
    line = "=" * 100
    log_print(lines, f"\n{line}")
    log_print(lines, title)
    log_print(lines, line)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_class_balance_text(name, y):
    y = np.asarray(y).astype(int)
    counts = pd.Series(y).value_counts().sort_index()
    ratios = pd.Series(y).value_counts(normalize=True).sort_index()
    return (
        f"{name} counts:\n{counts.to_string()}\n\n"
        f"{name} ratios:\n{ratios.to_string()}"
    )


def compute_metrics(y_true, y_pred, y_prob):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "report": classification_report(y_true, y_pred, digits=4, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["avg_precision"] = average_precision_score(y_true, y_prob)
    except Exception:
        out["avg_precision"] = float("nan")
    return out


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def build_row_features(df):
    df = df.copy()

    df["future_close"] = df["close"].shift(-PREDICTION_HORIZON)
    df["future_return"] = df["future_close"] / df["close"] - 1.0

    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)
    df["return_10"] = df["close"].pct_change(10)
    df["return_15"] = df["close"].pct_change(15)
    df["return_30"] = df["close"].pct_change(30)
    df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))

    df["volume_change_1"] = df["volume"].pct_change(1)
    df["volume_change_3"] = df["volume"].pct_change(3)
    df["volume_change_5"] = df["volume"].pct_change(5)

    df["high_low"] = df["high"] - df["low"]
    df["open_close"] = df["close"] - df["open"]
    df["body_abs"] = df["open_close"].abs()
    safe_range = df["high_low"].replace(0, np.nan)
    df["body_to_range"] = (df["body_abs"] / safe_range).clip(0, 2)
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    for w in [5, 10, 20, 50]:
        df[f"rolling_mean_{w}"] = df["close"].rolling(w).mean()
        df[f"rolling_std_{w}"] = df["close"].rolling(w).std()

    for span in [5, 10, 20, 50]:
        df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False).mean()
        df[f"price_vs_ema_{span}"] = df["close"] / df[f"ema_{span}"] - 1.0
        df[f"price_vs_mean_{span}"] = df["close"] / df[f"rolling_mean_{span}"] - 1.0

    df["ema_spread_5_20"] = df["ema_5"] / df["ema_20"] - 1.0
    df["ema_spread_10_50"] = df["ema_10"] / df["ema_50"] - 1.0

    for w in [5, 10, 20]:
        df[f"return_mean_{w}"] = df["return_1"].rolling(w).mean()
        df[f"return_std_{w}"] = df["return_1"].rolling(w).std()
        df[f"range_mean_{w}"] = df["high_low"].rolling(w).mean()
        df[f"range_std_{w}"] = df["high_low"].rolling(w).std()

    for w in [10, 30]:
        df[f"volume_mean_{w}"] = df["volume"].rolling(w).mean()
        df[f"volume_std_{w}"] = df["volume"].rolling(w).std()
        df[f"volume_z_{w}"] = (
            (df["volume"] - df[f"volume_mean_{w}"]) / df[f"volume_std_{w}"].replace(0, np.nan)
        )

    df["momentum_5"] = df["close"] - df["close"].shift(5)
    df["momentum_20"] = df["close"] - df["close"].shift(20)
    df["momentum_30"] = df["close"] - df["close"].shift(30)

    df["rsi_14"] = compute_rsi(df["close"], period=14)

    ts = pd.to_datetime(df["timestamp"], utc=True)
    try:
        ts_local = ts.dt.tz_convert("America/New_York")
    except Exception:
        ts_local = ts

    minutes_since_open = (ts_local.dt.hour * 60 + ts_local.dt.minute) - (9 * 60 + 30)
    session_len = 390.0
    df["session_progress"] = (minutes_since_open / session_len).clip(0, 1)

    minute_of_day = ts_local.dt.hour * 60 + ts_local.dt.minute
    df["minute_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    df["minute_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)

    dow = ts_local.dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    return df


def choose_move_threshold(trainval_future_returns):
    arr = np.asarray(trainval_future_returns, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    if FIXED_RETURN_THRESHOLD is not None:
        return float(FIXED_RETURN_THRESHOLD)
    return float(np.quantile(np.abs(arr), TRAIN_MOVE_QUANTILE))


def label_rows_from_threshold(df, move_threshold):
    out = df.copy()
    out["target"] = np.nan
    out.loc[out["future_return"] > move_threshold, "target"] = 1.0
    out.loc[out["future_return"] < -move_threshold, "target"] = 0.0
    out = out.dropna(subset=["target"]).copy()
    out["target"] = out["target"].astype(np.float32)
    return out


def build_sequence_windows(df_split, base_features, window, split_name, report_lines):
    X, y, times, future_ret = [], [], [], []

    needed_cols = list(base_features) + ["target", "timestamp", "future_return"]
    work = df_split.dropna(subset=needed_cols).reset_index(drop=True).copy()

    for end_idx in range(window - 1, len(work)):
        start_idx = end_idx - window + 1

        window_matrix = work.loc[start_idx:end_idx, base_features].values.astype(np.float32)

        label = np.float32(work.loc[end_idx, "target"])
        ts = work.loc[end_idx, "timestamp"]
        fr = np.float32(work.loc[end_idx, "future_return"])

        X.append(window_matrix)
        y.append(label)
        times.append(ts)
        future_ret.append(fr)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    future_ret = np.array(future_ret, dtype=np.float32)

    log_print(report_lines, f"{split_name} sequence windows built:")
    log_print(report_lines, f"  X shape: {X.shape}")
    log_print(report_lines, f"  y shape: {y.shape}")

    return X, y, times, future_ret


def train_only_clip_scale(X_train, X_other):
    clip_lower = np.nanquantile(X_train, CLIP_LOWER_Q, axis=0)
    clip_upper = np.nanquantile(X_train, CLIP_UPPER_Q, axis=0)

    X_train_clip = np.clip(X_train, clip_lower, clip_upper)
    X_other_clip = np.clip(X_other, clip_lower, clip_upper)

    train_mean = np.nanmean(X_train_clip, axis=0)
    train_std = np.nanstd(X_train_clip, axis=0)
    train_std = np.where(train_std < 1e-8, 1.0, train_std)

    X_train_scaled = (X_train_clip - train_mean) / train_std
    X_other_scaled = (X_other_clip - train_mean) / train_std

    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_other_scaled = np.nan_to_num(X_other_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    return X_train_scaled, X_other_scaled, clip_lower, clip_upper, train_mean, train_std


# ============================================================
# PLACEHOLDER MODEL SECTION
# ============================================================
# MLP model still present in original file, but LSTM model not implemented yet.
# This progress version stops before training on purpose.


# ============================================================
# START RUN
# ============================================================
os.makedirs(RUNS_DIR, exist_ok=True)

run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
report_path = os.path.join(RUNS_DIR, f"run_lstm_progress_{run_timestamp}.txt")

report_lines = []

section(report_lines, "RUN METADATA")
log_print(report_lines, f"Run timestamp: {run_timestamp}")
log_print(report_lines, f"Data source: LOCAL FILE")
log_print(report_lines, f"File path: {LOCAL_FILE_PATH}")
log_print(report_lines, f"Window size: {WINDOW}")
log_print(report_lines, f"Prediction horizon: {PREDICTION_HORIZON}")
log_print(report_lines, f"Base features: {BASE_FEATURES}")

# ============================================================
# 1. LOAD DATA
# ============================================================
section(report_lines, "1. LOAD DATA")

if not os.path.exists(LOCAL_FILE_PATH):
    raise FileNotFoundError(f"File not found: {LOCAL_FILE_PATH}")

with open(LOCAL_FILE_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

if isinstance(raw, dict) and "data" in raw:
    df = pd.DataFrame(raw["data"])
else:
    raise ValueError("Invalid JSON format")

log_print(report_lines, f"Loaded rows: {len(df)}")
log_print(report_lines, f"Columns: {df.columns.tolist()}")
log_print(report_lines, df.head().to_string())

# ============================================================
# 2. CLEANING
# ============================================================
section(report_lines, "2. CLEANING")

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

log_print(report_lines, f"Rows after cleaning: {len(df)}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
section(report_lines, "3. FEATURE ENGINEERING")

df_feat = build_row_features(df)

log_print(report_lines, f"Rows after feature engineering: {len(df_feat)}")
log_print(report_lines, f"Feature columns present: {sum(col in df_feat.columns for col in BASE_FEATURES)} / {len(BASE_FEATURES)}")

# ============================================================
# 4. SPLIT ROWS
# ============================================================
section(report_lines, "4. SPLIT ROWS")

split_idx = int(len(df_feat) * (1.0 - TEST_FRACTION))
df_trainval_raw = df_feat.iloc[:split_idx].copy()
df_test_raw = df_feat.iloc[split_idx:].copy()

move_threshold = choose_move_threshold(df_trainval_raw["future_return"])
log_print(report_lines, f"Move threshold: {move_threshold:.8f}")

df_trainval = label_rows_from_threshold(df_trainval_raw, move_threshold)
df_test = label_rows_from_threshold(df_test_raw, move_threshold)

log_print(report_lines, f"Trainval labeled rows: {len(df_trainval)}")
log_print(report_lines, f"Test labeled rows: {len(df_test)}")

log_print(report_lines, get_class_balance_text("trainval", df_trainval["target"].values))
log_print(report_lines, get_class_balance_text("test", df_test["target"].values))

# ============================================================
# 5. BUILD SEQUENCE WINDOWS
# ============================================================
section(report_lines, "5. BUILD SEQUENCE WINDOWS")

X_trainval, y_trainval, time_trainval, future_ret_trainval = build_sequence_windows(
    df_trainval,
    BASE_FEATURES,
    WINDOW,
    "trainval",
    report_lines,
)

X_test, y_test, time_test, future_ret_test = build_sequence_windows(
    df_test,
    BASE_FEATURES,
    WINDOW,
    "test",
    report_lines,
)

log_print(report_lines, f"\nSequence trainval shape: {X_trainval.shape}, {y_trainval.shape}")
log_print(report_lines, f"Sequence test shape:     {X_test.shape}, {y_test.shape}")

print("Sequence window build worked.")
print("This is a progress checkpoint before LSTM model integration.")
raise SystemExit(0)
