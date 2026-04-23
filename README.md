# EEL4810 Project — Intraday Stock Direction Modeling

A research-style machine learning project for predicting short-horizon stock price direction using **1-minute OHLCV bars** and multiple model families (Random Forest, MLP, CNN, LSTM).

---

## Overview

This repository contains:

- A data ingestion pipeline for minute-level market data from Alpaca.
- Feature engineering for intraday price/volume behavior.
- Multiple model scripts to compare classical ML and deep learning approaches.
- Timestamped run artifacts (metrics logs, prediction CSVs, trained checkpoints).

The current workflow focuses on one symbol at a time (for example, MSFT), with time-aware train/validation/test splits.

> ⚠️ This project is for experimentation and education. It is **not** production trading advice.

---

## Repository Structure

```text
.
├── data/
│   └── stock-data.json             # Local minute-bar dataset used by training scripts
├── runs/
│   ├── random_forest/...           # Random Forest run logs
│   ├── mlp/...                     # MLP logs, predictions, and checkpoints
│   ├── cnn/...                     # CNN logs, predictions, and checkpoints
│   └── lstm/...                    # LSTM logs, predictions, and checkpoints
├── fetch_stock_data.py             # Pulls minute bars from Alpaca and writes JSON
├── random_forest_model.py          # Baseline tree-based model
├── mlp_model.py                    # Feedforward neural network model
├── cnn_model.py                    # 1D CNN model
├── lstm_model.py                   # Sequence model (LSTM)
└── README.md
```

---

## Data Pipeline

### 1) Configure credentials

Create a `.env` file in the repo root:

```bash
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here
```

### 2) Fetch market data

```bash
python fetch_stock_data.py <SYMBOL> <TARGET_ROWS>
```

Example:

```bash
python fetch_stock_data.py MSFT 50000
```

What the script does:

- Requests 1-minute bars in chunks while walking backward through time.
- Converts timestamps to `America/New_York`.
- Keeps regular market hours (9:30–16:00 ET weekdays).
- Drops incomplete trading days.
- Writes normalized output to `data/stock-data.json`.

---

## Model Training

Each model script is standalone and writes timestamped artifacts under `runs/<model>/<timestamp>/`.

### Random Forest (baseline)

```bash
python random_forest_model.py
```

Outputs include:

- Run report text file with config + metrics.

### MLP

```bash
python mlp_model.py
```

Outputs typically include:

- Training history CSV.
- Test predictions CSV.
- Best model checkpoints (`.pt`).
- Run summary text report.

### CNN

```bash
python cnn_model.py
```

Outputs are similar to MLP (history, predictions, checkpoints, summary).

### LSTM

```bash
python lstm_model.py
```

LSTM script adds:

- Interactive GPU/CPU selection (when supported GPUs are detected via `nvidia-smi`).
- Multi-seed ensembling workflow with seed filtering by validation quality.

---

## Environment Setup

Recommended Python version: **3.10+**.

Install dependencies (example):

```bash
pip install numpy pandas scikit-learn torch python-dotenv alpaca-py
```

If using GPU with PyTorch, install a CUDA-compatible torch build that matches your system.

---

## Notes on Experiment Design

- Time-aware splitting is used to avoid random leakage from future to past.
- Labels are based on future move direction over a configurable horizon.
- Thresholding and class-balance controls are included in neural scripts.
- Most configuration is currently set as constants near the top of each script.

---

## Suggested Next Improvements

- Add a shared `requirements.txt` (or `pyproject.toml`) for reproducible installs.
- Centralize common feature engineering to reduce duplication.
- Add CLI arguments for key hyperparameters.
- Add model comparison notebook/dashboard for metrics across runs.
- Add unit tests for data cleaning and feature generation.

---

## Quick Start

```bash
# 1) Fetch minute data
python fetch_stock_data.py MSFT 50000

# 2) Train baseline + neural models
python random_forest_model.py
python mlp_model.py
python cnn_model.py
python lstm_model.py
```

---

## Disclaimer

This repository is intended for academic and prototyping use only. Live trading requires robust risk controls, execution infrastructure, monitoring, and regulatory awareness that are outside this codebase.