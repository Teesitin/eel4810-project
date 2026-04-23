import os
import json
import copy
import random
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# DEVICE SELECTION - MUST HAPPEN BEFORE IMPORTING TORCH
# ============================================================
MIN_SUPPORTED_COMPUTE_CAPABILITY = 7.5


def parse_compute_capability(cc_str):
    try:
        return float(str(cc_str).strip())
    except Exception:
        return -1.0


def detect_supported_gpus_with_nvidia_smi(min_cc=MIN_SUPPORTED_COMPUTE_CAPABILITY):
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,compute_cap,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        all_gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue

            idx, name, cc, mem = parts
            cc_float = parse_compute_capability(cc)

            all_gpus.append(
                {
                    "index": int(idx),
                    "name": name,
                    "compute_cap": cc,
                    "compute_cap_float": cc_float,
                    "memory_mb": int(mem),
                    "supported": cc_float >= min_cc,
                }
            )

        supported_gpus = [g for g in all_gpus if g["supported"]]
        return all_gpus, supported_gpus

    except Exception:
        return [], []


def pick_best_supported_gpu(supported_gpus):
    if not supported_gpus:
        return None

    candidates = sorted(
        supported_gpus,
        key=lambda g: (-g["compute_cap_float"], -g["memory_mb"], g["index"])
    )
    return candidates[0]


def choose_device_pre_torch():
    all_gpus, supported_gpus = detect_supported_gpus_with_nvidia_smi()
    best_gpu = pick_best_supported_gpu(supported_gpus)

    print("\n" + "=" * 100)
    print("DEVICE SELECTION")
    print("=" * 100)

    if all_gpus:
        print("Detected system GPUs:\n")
        for gpu in all_gpus:
            status = "SUPPORTED" if gpu["supported"] else "unsupported"
            print(
                f"System GPU {gpu['index']}: {gpu['name']} | "
                f"CC {gpu['compute_cap']} | "
                f"{gpu['memory_mb']} MB | "
                f"{status}"
            )
    else:
        print("No NVIDIA GPUs detected with nvidia-smi.")

    print("\nAvailable choices:\n")
    print("[cpu] CPU")

    for gpu in supported_gpus:
        print(
            f"[{gpu['index']}] {gpu['name']} | "
            f"CC {gpu['compute_cap']} | "
            f"{gpu['memory_mb']} MB"
        )

    if not supported_gpus:
        print("\nNo supported CUDA GPUs found for this PyTorch build.")
        print("Defaulting to CPU.\n")
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        return "cpu", None, all_gpus, supported_gpus

    print("\nNotes:")
    print(f"- Only GPUs with compute capability >= {MIN_SUPPORTED_COMPUTE_CAPABILITY:.1f} are listed.")
    print("- Unsupported GPUs are intentionally excluded so the script does not crash later.")
    print("- Press Enter to use the best supported GPU automatically.")

    while True:
        choice = input("\nSelection: ").strip().lower()

        if choice == "":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu["index"])
            print(f"\nUsing best supported GPU {best_gpu['index']}: {best_gpu['name']}")
            print("PyTorch will only see this GPU as cuda:0.\n")
            return "cuda", best_gpu, all_gpus, supported_gpus

        if choice == "cpu":
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            print("\nUsing CPU.\n")
            return "cpu", None, all_gpus, supported_gpus

        if choice.isdigit():
            selected_index = int(choice)
            gpu = next((g for g in supported_gpus if g["index"] == selected_index), None)

            if gpu is None:
                print("That is not one of the available supported choices. Try again.")
                continue

            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
            print(f"\nUsing GPU {gpu['index']}: {gpu['name']}")
            print("PyTorch will only see this GPU as cuda:0.\n")
            return "cuda", gpu, all_gpus, supported_gpus

        print("Invalid input. Choose 'cpu', a listed GPU index, or press Enter.")


SELECTED_DEVICE_TYPE, SELECTED_GPU_INFO, ALL_DETECTED_GPUS, SUPPORTED_GPUS = choose_device_pre_torch()


# ============================================================
# IMPORT TORCH AFTER DEVICE SELECTION
# ============================================================
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
LOCAL_FILE_PATH = "data/stock-data.json"
RUNS_BASE_DIR = "runs"
MODEL_NAME = "lstm"

DATA_ROWS_TO_USE = 50000   # None = use all rows

WINDOW = 120
PREDICTION_HORIZON = 60

FIXED_RETURN_THRESHOLD = None
TRAIN_MOVE_QUANTILE = 0.55

TEST_FRACTION = 0.15
CV_SPLITS = 3
CV_GAP = 20

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

HIDDEN_SIZE = 128
NUM_LAYERS = 2

DEVICE = "cuda" if (SELECTED_DEVICE_TYPE == "cuda" and torch.cuda.is_available()) else "cpu"

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
        df[f"volume_z_{w}"] = (df["volume"] - df[f"volume_mean_{w}"]) / df[f"volume_std_{w}"]

    for w in [5, 10, 20, 30]:
        df[f"momentum_{w}"] = df["close"] - df["close"].shift(w)

    df["rsi_14"] = compute_rsi(df["close"], period=14)

    ny_time = df["timestamp"].dt.tz_convert("America/New_York")
    minute_of_day = ny_time.dt.hour * 60 + ny_time.dt.minute
    day_of_week = ny_time.dt.dayofweek
    df["session_progress"] = minute_of_day / 1440.0
    df["minute_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    df["minute_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    return df


def choose_move_threshold(trainval_future_returns):
    if FIXED_RETURN_THRESHOLD is not None:
        return float(FIXED_RETURN_THRESHOLD)
    return float(np.quantile(np.abs(trainval_future_returns), TRAIN_MOVE_QUANTILE))


def label_rows_from_threshold(df, move_threshold):
    df = df.copy()
    df["target"] = np.nan
    df.loc[df["future_return"] > move_threshold, "target"] = 1.0
    df.loc[df["future_return"] < -move_threshold, "target"] = 0.0
    return df


def build_sequence_windows(df_split, base_features, window, split_name, report_lines):
    X, y, times, future_ret = [], [], [], []

    needed_cols = list(base_features) + ["timestamp", "future_return"]
    work = df_split.dropna(subset=needed_cols).reset_index(drop=True).copy()

    skipped_neutral = 0

    for end_idx in range(window - 1, len(work)):
        label = work.loc[end_idx, "target"]

        if pd.isna(label):
            skipped_neutral += 1
            continue

        start_idx = end_idx - window + 1
        window_matrix = work.loc[start_idx:end_idx, base_features].values.astype(np.float32)

        ts = work.loc[end_idx, "timestamp"]
        fr = np.float32(work.loc[end_idx, "future_return"])

        X.append(window_matrix)
        y.append(np.float32(label))
        times.append(ts)
        future_ret.append(fr)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    times = np.array(times)
    future_ret = np.array(future_ret, dtype=np.float32)

    log_print(
        report_lines,
        f"{split_name}: rows={len(work)}, built_windows={len(X)}, skipped_neutral={skipped_neutral}"
    )
    log_print(report_lines, f"{split_name} X shape: {X.shape}")
    log_print(report_lines, f"{split_name} y shape: {y.shape}")

    return X, y, times, future_ret


def train_only_clip_scale(X_train, X_other):
    clip_lower = np.quantile(X_train, CLIP_LOWER_Q, axis=0, keepdims=True)
    clip_upper = np.quantile(X_train, CLIP_UPPER_Q, axis=0, keepdims=True)

    X_train = np.clip(X_train, clip_lower, clip_upper)
    X_other = np.clip(X_other, clip_lower, clip_upper)

    train_mean = X_train.mean(axis=0, keepdims=True)
    train_std = X_train.std(axis=0, keepdims=True)
    train_std[train_std < 1e-8] = 1.0

    X_train = ((X_train - train_mean) / train_std).astype(np.float32)
    X_other = ((X_other - train_mean) / train_std).astype(np.float32)

    return X_train, X_other, clip_lower, clip_upper, train_mean, train_std


def transform_with_existing_stats(X, clip_lower, clip_upper, train_mean, train_std):
    X = np.clip(X, clip_lower, clip_upper)
    X = ((X - train_mean) / train_std).astype(np.float32)
    return X


def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS)
    rows = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        pos_rate = float(y_pred.mean())
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        rows.append({
            "threshold": float(t),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "f1": float(f1),
            "positive_rate": float(pos_rate),
        })

    df = pd.DataFrame(rows)
    constrained = df[
        (df["positive_rate"] >= MIN_POSITIVE_RATE) &
        (df["positive_rate"] <= MAX_POSITIVE_RATE)
    ].copy()

    search_df = constrained if len(constrained) > 0 else df.copy()
    search_df = search_df.sort_values(
        ["balanced_accuracy", "accuracy", "f1"],
        ascending=False
    )

    best = search_df.iloc[0]
    return float(best["threshold"]), search_df


# ============================================================
# MODEL
# ============================================================
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.35):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden).squeeze(-1)
        return logits


# ============================================================
# TRAIN / EVAL
# ============================================================
def predict_probs(model, X_np, batch_size, device):
    dataset = TensorDataset(torch.tensor(X_np, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    model.eval()
    probs_all = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            probs_all.extend(probs.cpu().numpy())

    return np.array(probs_all, dtype=np.float32)


def run_eval(model, X_np, y_np, criterion, batch_size, device):
    dataset = TensorDataset(
        torch.tensor(X_np, dtype=torch.float32),
        torch.tensor(y_np, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    model.eval()
    losses, probs_all, targets_all = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            probs = torch.sigmoid(logits)

            losses.append(loss.item())
            probs_all.extend(probs.cpu().numpy())
            targets_all.extend(yb.cpu().numpy())

    return (
        float(np.mean(losses)),
        np.array(targets_all, dtype=np.int32),
        np.array(probs_all, dtype=np.float32),
    )


def train_one_model(seed, X_train, y_train, X_stop, y_stop, device, tag, report_lines, history_rows, verbose=True):
    set_seed(seed)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    input_size = X_train.shape[2]

    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    best_state = None
    best_stop_auc = -1.0
    best_stop_bal_acc = -1.0
    best_epoch = None
    epochs_no_improve = 0

    if verbose:
        log_print(report_lines, f"\n--- Training {tag} | seed={seed} ---")
        log_print(report_lines, f"Trainable params: {count_parameters(model):,}")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            yb_smoothed = yb * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb_smoothed)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

            train_losses.append(loss.item())

        stop_loss, stop_targets, stop_probs = run_eval(
            model, X_stop, y_stop, criterion, BATCH_SIZE, device
        )
        stop_auc = roc_auc_score(stop_targets, stop_probs)
        stop_preds_05 = (stop_probs >= 0.5).astype(int)
        stop_bal_acc = balanced_accuracy_score(stop_targets, stop_preds_05)
        stop_acc_05 = accuracy_score(stop_targets, stop_preds_05)

        scheduler.step(stop_auc)

        history_rows.append({
            "tag": tag,
            "seed": seed,
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "stop_loss": float(stop_loss),
            "stop_auc": float(stop_auc),
            "stop_bal_acc_0.5": float(stop_bal_acc),
            "stop_acc_0.5": float(stop_acc_05),
            "lr": optimizer.param_groups[0]["lr"],
        })

        if verbose:
            log_print(
                report_lines,
                f"{tag} | seed={seed} | epoch={epoch:03d} | "
                f"train_loss={np.mean(train_losses):.6f} | "
                f"stop_loss={stop_loss:.6f} | "
                f"stop_auc={stop_auc:.6f} | "
                f"stop_bal_acc@0.5={stop_bal_acc:.6f} | "
                f"stop_acc@0.5={stop_acc_05:.6f} | "
                f"lr={optimizer.param_groups[0]['lr']:.6e}"
            )

        improved = (stop_auc > best_stop_auc + 1e-4) or (
            abs(stop_auc - best_stop_auc) <= 1e-4 and stop_bal_acc > best_stop_bal_acc + 1e-4
        )

        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_stop_auc = stop_auc
            best_stop_bal_acc = stop_bal_acc
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            if verbose:
                log_print(report_lines, f"Early stopping for {tag} | seed={seed} at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_epoch, best_stop_auc, best_stop_bal_acc


# ============================================================
# MAIN
# ============================================================
set_seed(42)

run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_date = datetime.now().strftime("%Y-%m-%d")
run_dir = os.path.join(RUNS_BASE_DIR, MODEL_NAME, run_date)
os.makedirs(run_dir, exist_ok=True)

if DEVICE == "cuda":
    print(f"PyTorch sees GPU: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch is running on CPU.")

run_name = f"lstm_timesplit_sequence_w{WINDOW}_h{PREDICTION_HORIZON}_{run_timestamp}"

report_path = os.path.join(run_dir, f"{run_name}.txt")
history_csv_path = os.path.join(run_dir, f"{run_name}_history.csv")
predictions_csv_path = os.path.join(run_dir, f"{run_name}_test_predictions.csv")

report_lines = []
history_rows = []

section(report_lines, "RUN METADATA")
log_print(report_lines, f"Run timestamp: {run_timestamp}")
log_print(report_lines, f"Run name: {run_name}")
log_print(report_lines, f"Run directory: {run_dir}")
log_print(report_lines, f"Data source: LOCAL FILE")
log_print(report_lines, f"File path: {LOCAL_FILE_PATH}")
log_print(report_lines, f"Rows to use: {DATA_ROWS_TO_USE}")
log_print(report_lines, f"Device: {DEVICE}")

log_print(report_lines, "\nSystem GPU scan results:")
if ALL_DETECTED_GPUS:
    for gpu in ALL_DETECTED_GPUS:
        status = "SUPPORTED" if gpu["supported"] else "unsupported"
        log_print(
            report_lines,
            f"GPU {gpu['index']}: {gpu['name']} | "
            f"CC {gpu['compute_cap']} | {gpu['memory_mb']} MB | {status}"
        )
else:
    log_print(report_lines, "No NVIDIA GPUs detected with nvidia-smi.")

if DEVICE == "cuda":
    log_print(report_lines, f"\nVisible CUDA device count: {torch.cuda.device_count()}")
    log_print(report_lines, f"CUDA device name: {torch.cuda.get_device_name(0)}")
    if SELECTED_GPU_INFO is not None:
        log_print(
            report_lines,
            f"Selected GPU (original system index): {SELECTED_GPU_INFO['index']} | "
            f"{SELECTED_GPU_INFO['name']} | CC {SELECTED_GPU_INFO['compute_cap']} | "
            f"{SELECTED_GPU_INFO['memory_mb']} MB"
        )
else:
    log_print(report_lines, "\nRunning on CPU")

log_print(report_lines, f"Window size: {WINDOW}")
log_print(report_lines, f"Prediction horizon: {PREDICTION_HORIZON}")
log_print(report_lines, f"Label style: meaningful move direction with neutral zone dropped")
log_print(report_lines, f"Fixed threshold override: {FIXED_RETURN_THRESHOLD}")
log_print(report_lines, f"Train move quantile: {TRAIN_MOVE_QUANTILE}")
log_print(report_lines, f"Base features ({len(BASE_FEATURES)}): {BASE_FEATURES}")
log_print(report_lines, f"CV splits: {CV_SPLITS}")
log_print(report_lines, f"CV gap: {CV_GAP}")
log_print(report_lines, f"Ensemble seeds: {ENSEMBLE_SEEDS}")
log_print(report_lines, f"Min seed OOF AUC: {MIN_SEED_OOF_AUC}")
log_print(report_lines, f"Max epochs: {MAX_EPOCHS}")
log_print(report_lines, f"Batch size: {BATCH_SIZE}")
log_print(report_lines, f"Max LR: {MAX_LR}")
log_print(report_lines, f"Weight decay: {WEIGHT_DECAY}")
log_print(report_lines, f"Dropout: {DROPOUT}")
log_print(report_lines, f"Hidden size: {HIDDEN_SIZE}")
log_print(report_lines, f"Num layers: {NUM_LAYERS}")
log_print(report_lines, f"Label smoothing: {LABEL_SMOOTHING}")
log_print(report_lines, f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
log_print(report_lines, f"Clip quantiles: [{CLIP_LOWER_Q}, {CLIP_UPPER_Q}]")
log_print(report_lines, f"Threshold search range: [{THRESHOLD_MIN}, {THRESHOLD_MAX}] with {THRESHOLD_STEPS} steps")

section(report_lines, "1. LOAD DATA")
if not os.path.exists(LOCAL_FILE_PATH):
    raise FileNotFoundError(f"File not found: {LOCAL_FILE_PATH}")

with open(LOCAL_FILE_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

if not isinstance(raw, dict) or "data" not in raw:
    raise ValueError("Invalid JSON format: expected {'data': [...]}")

df = pd.DataFrame(raw["data"])
rows_in_file = len(df)

log_print(report_lines, f"Rows in file: {rows_in_file}")
log_print(report_lines, f"Columns: {df.columns.tolist()}")
log_print(report_lines, "\nRaw head:")
log_print(report_lines, df.head().to_string())

section(report_lines, "2. CLEAN DATA")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

if DATA_ROWS_TO_USE is not None:
    df = df.tail(DATA_ROWS_TO_USE).reset_index(drop=True)

log_print(report_lines, f"Rows after cleaning / row cap: {len(df)}")
log_print(report_lines, f"Timestamp min: {df['timestamp'].min()}")
log_print(report_lines, f"Timestamp max: {df['timestamp'].max()}")

section(report_lines, "3. FEATURE ENGINEERING")
df_feat = build_row_features(df)
log_print(report_lines, f"Rows after feature engineering + dropna: {len(df_feat)}")
log_print(report_lines, "\nFuture return summary:")
log_print(
    report_lines,
    df_feat["future_return"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string()
)
log_print(report_lines, "\nFeature preview:")
preview_cols = ["timestamp", "close", "future_close", "future_return"] + BASE_FEATURES[:12]
log_print(report_lines, df_feat[preview_cols].head().to_string())

section(report_lines, "4. SPLIT ROWS")
n_rows = len(df_feat)
test_start = int(n_rows * (1.0 - TEST_FRACTION))

df_trainval_rows = df_feat.iloc[:test_start].copy()
df_test_rows = df_feat.iloc[test_start:].copy()

move_threshold = choose_move_threshold(df_trainval_rows["future_return"].values)
log_print(report_lines, f"Chosen move threshold (absolute future return): {move_threshold:.6f}")

df_trainval_rows = label_rows_from_threshold(df_trainval_rows, move_threshold)
df_test_rows = label_rows_from_threshold(df_test_rows, move_threshold)

for name, split_df in [("Trainval rows", df_trainval_rows), ("Test rows", df_test_rows)]:
    labeled = split_df[split_df["target"].notna()]["target"].values
    neutral_count = int(split_df["target"].isna().sum())
    log_print(report_lines, f"\n{name}: total={len(split_df)}, neutral_dropped={neutral_count}")
    if len(labeled) > 0:
        log_print(report_lines, get_class_balance_text(name + " labeled", labeled))

section(report_lines, "5. BUILD SEQUENCE WINDOWS")
X_trainval, y_trainval, time_trainval, future_ret_trainval = build_sequence_windows(
    df_trainval_rows,
    BASE_FEATURES,
    WINDOW,
    "Trainval",
    report_lines,
)
X_test, y_test, time_test, future_ret_test = build_sequence_windows(
    df_test_rows,
    BASE_FEATURES,
    WINDOW,
    "Test",
    report_lines,
)

log_print(report_lines, f"\nSequence trainval shape: {X_trainval.shape}, {y_trainval.shape}")
log_print(report_lines, f"Sequence test shape:     {X_test.shape}, {y_test.shape}")

section(report_lines, "6. WALK-FORWARD CV")
tscv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=CV_GAP)

oof_probs_by_seed = {seed: np.full(len(X_trainval), np.nan, dtype=np.float32) for seed in ENSEMBLE_SEEDS}
fold_results = []

for fold_idx, (cv_train_idx, cv_val_idx) in enumerate(tscv.split(X_trainval), start=1):
    if len(cv_train_idx) < 1000 or len(cv_val_idx) < 200:
        log_print(report_lines, f"Skipping fold {fold_idx}: too small")
        continue

    stop_size = max(256, int(len(cv_train_idx) * 0.15))
    inner_train_idx = cv_train_idx[:-stop_size]
    stop_idx = cv_train_idx[-stop_size:]

    X_inner_train = X_trainval[inner_train_idx]
    y_inner_train = y_trainval[inner_train_idx]
    X_stop = X_trainval[stop_idx]
    y_stop = y_trainval[stop_idx]
    X_val_fold = X_trainval[cv_val_idx]
    y_val_fold = y_trainval[cv_val_idx]

    X_inner_train_scaled, X_stop_scaled, clip_lower, clip_upper, train_mean, train_std = train_only_clip_scale(
        X_inner_train, X_stop
    )
    X_val_fold_scaled = transform_with_existing_stats(
        X_val_fold, clip_lower, clip_upper, train_mean, train_std
    )

    log_print(
        report_lines,
        f"\nFold {fold_idx}: train={len(inner_train_idx)}, stop={len(stop_idx)}, val={len(cv_val_idx)}"
    )

    for seed in ENSEMBLE_SEEDS:
        tag = f"cv_fold{fold_idx}"
        model, best_epoch, best_stop_auc, best_stop_bal_acc = train_one_model(
            seed=seed,
            X_train=X_inner_train_scaled,
            y_train=y_inner_train,
            X_stop=X_stop_scaled,
            y_stop=y_stop,
            device=DEVICE,
            tag=tag,
            report_lines=report_lines,
            history_rows=history_rows,
            verbose=True,
        )

        val_probs = predict_probs(model, X_val_fold_scaled, BATCH_SIZE, DEVICE)
        oof_probs_by_seed[seed][cv_val_idx] = val_probs

        fold_auc = roc_auc_score(y_val_fold, val_probs)
        fold_bal_acc = balanced_accuracy_score(y_val_fold, (val_probs >= 0.5).astype(int))
        fold_results.append({
            "fold": fold_idx,
            "seed": seed,
            "val_auc": float(fold_auc),
            "val_bal_acc_0.5": float(fold_bal_acc),
            "best_epoch": best_epoch,
            "best_stop_auc": float(best_stop_auc),
            "best_stop_bal_acc": float(best_stop_bal_acc),
        })

        log_print(
            report_lines,
            f"Fold {fold_idx} | seed={seed} | "
            f"val_auc={fold_auc:.6f} | val_bal_acc@0.5={fold_bal_acc:.6f}"
        )

section(report_lines, "7. OOF SEED SELECTION")
seed_scores = []
for seed, probs in oof_probs_by_seed.items():
    mask = ~np.isnan(probs)
    if mask.sum() == 0:
        continue
    seed_auc = roc_auc_score(y_trainval[mask], probs[mask])
    seed_bal_acc = balanced_accuracy_score(y_trainval[mask], (probs[mask] >= 0.5).astype(int))
    seed_scores.append({
        "seed": seed,
        "oof_auc": float(seed_auc),
        "oof_bal_acc_0.5": float(seed_bal_acc),
        "covered_samples": int(mask.sum()),
    })

seed_scores_df = pd.DataFrame(seed_scores).sort_values("oof_auc", ascending=False)
log_print(report_lines, seed_scores_df.to_string(index=False))

stable_seeds = seed_scores_df[seed_scores_df["oof_auc"] >= MIN_SEED_OOF_AUC]["seed"].tolist()
if len(stable_seeds) == 0:
    stable_seeds = seed_scores_df.head(1)["seed"].tolist()

log_print(report_lines, f"\nStable seeds selected: {stable_seeds}")

stable_oof_arrays = []
for seed in stable_seeds:
    stable_oof_arrays.append(oof_probs_by_seed[seed])

stable_oof_stack = np.stack(stable_oof_arrays, axis=0)
mask_all = ~np.isnan(stable_oof_stack).any(axis=0)
oof_probs_ens = stable_oof_stack[:, mask_all].mean(axis=0)
oof_targets = y_trainval[mask_all].astype(int)

best_threshold, threshold_df = find_best_threshold(oof_targets, oof_probs_ens)
log_print(report_lines, f"Best threshold from OOF validation: {best_threshold:.4f}")
log_print(report_lines, "\nTop threshold candidates:")
log_print(report_lines, threshold_df.head(20).to_string(index=False))

section(report_lines, "8. FINAL TRAINING ON TRAINVAL")
stop_size_final = max(512, int(len(X_trainval) * 0.15))
final_train_idx = np.arange(0, len(X_trainval) - stop_size_final)
final_stop_idx = np.arange(len(X_trainval) - stop_size_final, len(X_trainval))

X_final_train = X_trainval[final_train_idx]
y_final_train = y_trainval[final_train_idx]
X_final_stop = X_trainval[final_stop_idx]
y_final_stop = y_trainval[final_stop_idx]

X_final_train_scaled, X_final_stop_scaled, clip_lower, clip_upper, train_mean, train_std = train_only_clip_scale(
    X_final_train, X_final_stop
)
X_test_scaled = transform_with_existing_stats(X_test, clip_lower, clip_upper, train_mean, train_std)

final_checkpoints = []
test_prob_members = []

for seed in stable_seeds:
    tag = "final"
    model, best_epoch, best_stop_auc, best_stop_bal_acc = train_one_model(
        seed=seed,
        X_train=X_final_train_scaled,
        y_train=y_final_train,
        X_stop=X_final_stop_scaled,
        y_stop=y_final_stop,
        device=DEVICE,
        tag=tag,
        report_lines=report_lines,
        history_rows=history_rows,
        verbose=True,
    )

    checkpoint_path = os.path.join(run_dir, f"{run_name}_seed{seed}_best.pt")
    torch.save(model.state_dict(), checkpoint_path)
    final_checkpoints.append(checkpoint_path)

    test_probs = predict_probs(model, X_test_scaled, BATCH_SIZE, DEVICE)
    test_prob_members.append(test_probs)

    log_print(
        report_lines,
        f"Final seed={seed} | best_epoch={best_epoch} | best_stop_auc={best_stop_auc:.6f} | best_stop_bal_acc={best_stop_bal_acc:.6f}"
    )

history_df = pd.DataFrame(history_rows)
history_df.to_csv(history_csv_path, index=False)

section(report_lines, "9. FINAL TEST EVALUATION")
test_probs_ens = np.mean(np.stack(test_prob_members, axis=0), axis=0)
test_preds = (test_probs_ens >= best_threshold).astype(int)
test_targets = y_test.astype(int)

test_metrics = compute_metrics(test_targets, test_preds, test_probs_ens)

flipped_auc = roc_auc_score(test_targets, 1.0 - test_probs_ens)

log_print(report_lines, f"Test Accuracy:           {test_metrics['accuracy']:.6f}")
log_print(report_lines, f"Test Balanced Accuracy:  {test_metrics['balanced_accuracy']:.6f}")
log_print(report_lines, f"Test F1:                 {test_metrics['f1']:.6f}")
log_print(report_lines, f"Test Precision:          {test_metrics['precision']:.6f}")
log_print(report_lines, f"Test Recall:             {test_metrics['recall']:.6f}")
log_print(report_lines, f"Test ROC-AUC:            {test_metrics['roc_auc']:.6f}")
log_print(report_lines, f"Test AP:                 {test_metrics['avg_precision']:.6f}")
log_print(report_lines, f"Diagnostic flipped ROC-AUC (do not tune on this): {flipped_auc:.6f}")

log_print(report_lines, "\nTest Classification Report:")
log_print(report_lines, test_metrics["report"])
log_print(report_lines, "Test Confusion Matrix:")
log_print(report_lines, str(test_metrics["confusion_matrix"]))

log_print(report_lines, "\nProbability diagnostics:")
log_print(
    report_lines,
    f"Test probs mean={test_probs_ens.mean():.6f} std={test_probs_ens.std():.6f} "
    f"min={test_probs_ens.min():.6f} max={test_probs_ens.max():.6f}"
)

section(report_lines, "10. FEATURE IMPORTANCE APPROXIMATION")
log_print(report_lines, "Skipped for the current LSTM version.")
log_print(report_lines, "The previous feature-importance method was specific to the MLP architecture.")

section(report_lines, "11. SAMPLE PREDICTIONS")
results = pd.DataFrame({
    "timestamp": time_test,
    "future_return": future_ret_test,
    "actual": test_targets.astype(int),
    "predicted": test_preds.astype(int),
    "prob_up": test_probs_ens.astype(float),
})
results["correct"] = (results["actual"] == results["predicted"]).astype(int)
results.to_csv(predictions_csv_path, index=False)

log_print(report_lines, f"Saved test predictions CSV to: {predictions_csv_path}")
log_print(report_lines, "\nFirst 40 test predictions:")
log_print(report_lines, results.head(40).to_string(index=False))

wrong_results = results[results["correct"] == 0].copy()
wrong_results["confidence"] = np.where(
    wrong_results["predicted"] == 1,
    wrong_results["prob_up"],
    1.0 - wrong_results["prob_up"]
)

log_print(report_lines, "\nTop 40 most confident wrong predictions:")
log_print(
    report_lines,
    wrong_results.sort_values("confidence", ascending=False).head(40).to_string(index=False)
)

section(report_lines, "12. AUTO NOTES")
notes = []

if test_metrics["accuracy"] >= 0.52:
    notes.append("Test accuracy cleared 52%. This configuration is worth keeping as an LSTM candidate")
elif test_metrics["accuracy"] >= 0.50:
    notes.append("This is above coin-flip territory, but the edge is still modest.")
else:
    notes.append("Still below the target. Next levers: TRAIN_MOVE_QUANTILE=0.50, WINDOW=90, or PREDICTION_HORIZON=30.")

if test_metrics["roc_auc"] < 0.45 and flipped_auc > 0.55:
    notes.append("Warning: test ranking appears directionally inverted versus the learned mapping. Check regime shift and label stability.")

notes.append("This file now keeps full sequence windows instead of compressing them into summary vectors before modeling.")
notes.append("This file uses walk-forward validation and OOF threshold selection, which is safer than picking one contiguous validation slice.")
notes.append("If this still saturates, the next stronger comparison could be a GRU or a larger/tuned LSTM on the same target logic.")

for i, note in enumerate(notes, start=1):
    log_print(report_lines, f"{i}. {note}")

section(report_lines, "13. SAVE REPORT")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"\nSaved run report to: {report_path}")
print(f"Saved history CSV to: {history_csv_path}")
print(f"Saved predictions CSV to: {predictions_csv_path}")