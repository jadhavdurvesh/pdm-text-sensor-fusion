"""
train_baseline.py
Trains and evaluates the sensor-only LSTM baseline on C-MAPSS FD001.

Usage (after placing train_FD001.txt, test_FD001.txt, RUL_FD001.txt in ./data/):
    python train_baseline.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from data_utils import load_cmapss
from model import LSTMRegressor

# ---- Config ----
DATA_DIR = "./data"
WINDOW_SIZE = 30
MAX_RUL = 125
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3
VAL_SPLIT = 0.15
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading and preprocessing C-MAPSS FD001...")
    X_train, y_train, X_test, y_test, feature_cols, scaler = load_cmapss(
        data_dir=DATA_DIR, subset="FD001", window_size=WINDOW_SIZE, max_rul=MAX_RUL
    )
    print(f"Train windows: {X_train.shape}, Test engines: {X_test.shape}")
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    # Train/val split (validation used for early-stopping-style monitoring)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SPLIT, random_state=SEED
    )

    def to_loader(X, y, shuffle):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = to_loader(X_tr, y_tr, shuffle=True)
    val_loader = to_loader(X_val, y_val, shuffle=False)

    model = LSTMRegressor(num_features=len(feature_cols)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                pred = model(xb).cpu().numpy()
                val_preds.append(pred)
                val_true.append(yb.numpy())
        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)
        val_rmse = rmse(val_preds, val_true)
        best_val_rmse = min(best_val_rmse, val_rmse)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_loss={np.mean(train_losses):.3f} | val_RMSE={val_rmse:.3f}")

    # ---- Final test evaluation ----
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        test_preds = model(X_test_t).cpu().numpy()
    test_rmse = rmse(test_preds, y_test)

    print("\n==== Baseline (sensor-only) results ====")
    print(f"Best validation RMSE during training: {best_val_rmse:.3f}")
    print(f"Final TEST RMSE (C-MAPSS FD001):      {test_rmse:.3f}")
    print("Save this number -- it's the baseline your fusion model needs to beat.")

    torch.save(model.state_dict(), "baseline_lstm.pt")
    print("\nModel weights saved to baseline_lstm.pt")


if __name__ == "__main__":
    main()
