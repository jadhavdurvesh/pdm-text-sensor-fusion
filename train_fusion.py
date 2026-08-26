"""
train_fusion.py
Trains the fusion (sensor + text) model on C-MAPSS FD001 + the synthetic
maintenance logs, and compares its test RMSE against your sensor-only
baseline.

Prerequisites (run these first, in order):
    python train_baseline.py       -> gives you the baseline RMSE
    python synthetic_logs.py       -> creates results/train_logs.csv, test_logs.csv

Usage:
    python train_fusion.py --baseline_rmse 13.121
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from data_utils import load_cmapss
from fusion_model import FusionModel

# ---- Config ----
DATA_DIR = "./data"
RESULTS_DIR = "./results"
WINDOW_SIZE = 30
MAX_RUL = 125
MAX_TEXT_LEN = 64
BATCH_SIZE = 32
EPOCHS = 15          # fewer than baseline: transformer forward passes are slower
LR = 1e-3
VAL_SPLIT = 0.15
SEED = 42
TEXT_MODEL_NAME = "distilbert-base-uncased"
FREEZE_TEXT_ENCODER = False   # True = fast (recommended first run); False = full fine-tune

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FusionDataset(Dataset):
    def __init__(self, X_sensor, texts, y, tokenizer, max_len=MAX_TEXT_LEN):
        self.X_sensor = X_sensor
        self.texts = texts
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "sensor_x": torch.tensor(self.X_sensor[idx], dtype=torch.float32),
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def run_epoch(model, loader, optimizer, loss_fn, train=True):
    model.train() if train else model.eval()
    losses, all_preds, all_true = [], [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            sensor_x = batch["sensor_x"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            y = batch["y"].to(DEVICE)

            if train:
                optimizer.zero_grad()
            pred = model(sensor_x, input_ids, attention_mask)
            loss = loss_fn(pred, y)
            if train:
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            all_preds.append(pred.detach().cpu().numpy())
            all_true.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    return np.mean(losses), rmse(all_preds, all_true)


def main(baseline_rmse=None):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading sensor windows (must match synthetic_logs.py windowing)...")
    X_train, y_train, X_test, y_test, feature_cols, _ = load_cmapss(
        data_dir=DATA_DIR, subset="FD001", window_size=WINDOW_SIZE, max_rul=MAX_RUL
    )

    print("Loading synthetic maintenance logs...")
    train_logs_df = pd.read_csv(f"{RESULTS_DIR}/train_logs.csv")
    test_logs_df = pd.read_csv(f"{RESULTS_DIR}/test_logs.csv")

    assert len(train_logs_df) == len(X_train), (
        "Mismatch between train_logs.csv and sensor windows -- did you run "
        "synthetic_logs.py with the same WINDOW_SIZE/MAX_RUL as this script?"
    )
    assert len(test_logs_df) == len(X_test), (
        "Mismatch between test_logs.csv and sensor windows -- re-run synthetic_logs.py."
    )

    train_texts = train_logs_df["log_text"].tolist()
    test_texts = test_logs_df["log_text"].tolist()

    print(f"Loading tokenizer/model: {TEXT_MODEL_NAME} (requires internet on first run)...")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    # Train/val split -- split indices together so sensor windows and their
    # matching log text stay aligned.
    idx = np.arange(len(y_train))
    idx_tr, idx_val = train_test_split(idx, test_size=VAL_SPLIT, random_state=SEED)

    train_ds = FusionDataset(X_train[idx_tr], [train_texts[i] for i in idx_tr], y_train[idx_tr], tokenizer)
    val_ds = FusionDataset(X_train[idx_val], [train_texts[i] for i in idx_val], y_train[idx_val], tokenizer)
    test_ds = FusionDataset(X_test, test_texts, y_test, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = FusionModel(
        num_sensor_features=len(feature_cols),
        text_model_name=TEXT_MODEL_NAME,
        freeze_text=FREEZE_TEXT_ENCODER,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    loss_fn = nn.MSELoss()

    print(f"\nTraining fusion model (text encoder frozen={FREEZE_TEXT_ENCODER})...")
    best_val_rmse = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_rmse = run_epoch(model, train_loader, optimizer, loss_fn, train=True)
        _, val_rmse = run_epoch(model, val_loader, optimizer, loss_fn, train=False)
        best_val_rmse = min(best_val_rmse, val_rmse)
        print(f"Epoch {epoch:2d} | train_loss={train_loss:.3f} | train_RMSE={train_rmse:.3f} | val_RMSE={val_rmse:.3f}")

    _, test_rmse = run_epoch(model, test_loader, optimizer, loss_fn, train=False)

    print("\n==== Fusion model results ====")
    print(f"Best validation RMSE: {best_val_rmse:.3f}")
    print(f"Final TEST RMSE:      {test_rmse:.3f}")

    torch.save(model.state_dict(), "fusion_model.pt")
    print("Model weights saved to fusion_model.pt")

    # ---- Comparison table (this is your core paper result) ----
    print("\n==== Baseline vs. Fusion comparison ====")
    if baseline_rmse is not None:
        improvement = baseline_rmse - test_rmse
        pct = 100 * improvement / baseline_rmse
        print(f"{'Model':<20}{'Test RMSE':>12}")
        print(f"{'Sensor-only (base)':<20}{baseline_rmse:>12.3f}")
        print(f"{'Sensor + text fusion':<20}{test_rmse:>12.3f}")
        print(f"\nChange: {improvement:+.3f} RMSE ({pct:+.1f}%)")
        if improvement > 0:
            print("Fusion model IMPROVED over the sensor-only baseline.")
        else:
            print("Fusion model did NOT improve over baseline -- still a valid, "
                  "reportable result (see note below).")

        pd.DataFrame({
            "model": ["sensor_only_baseline", "sensor_text_fusion"],
            "test_rmse": [baseline_rmse, test_rmse],
        }).to_csv(f"{RESULTS_DIR}/comparison.csv", index=False)
        print(f"\nComparison table saved to {RESULTS_DIR}/comparison.csv")
    else:
        print("No --baseline_rmse provided -- pass it in to get the comparison table, "
              f"e.g.: python train_fusion.py --baseline_rmse {test_rmse:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_rmse", type=float, default=None,
                         help="Your sensor-only baseline test RMSE, e.g. 13.121")
    args = parser.parse_args()
    main(baseline_rmse=args.baseline_rmse)
