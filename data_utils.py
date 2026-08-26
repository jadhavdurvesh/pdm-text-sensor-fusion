"""
data_utils.py
Loads and preprocesses the NASA C-MAPSS turbofan degradation dataset (FD001)
for Remaining Useful Life (RUL) prediction.

Expected files (download from the NASA C-MAPSS dataset page and place in ./data/):
    train_FD001.txt
    test_FD001.txt
    RUL_FD001.txt
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Column names for the C-MAPSS dataset (26 columns total)
COLUMN_NAMES = (
    ["unit_number", "time_in_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors that are constant / near-constant in FD001 and add no signal.
# (Well known in C-MAPSS literature; drop them to reduce noise.)
DROP_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                 "sensor_16", "sensor_18", "sensor_19"]


def load_raw(path):
    """Load a raw C-MAPSS space-separated .txt file into a DataFrame."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df


def load_rul(path):
    """Load the RUL_FD001.txt file: one true RUL value per test engine, in order."""
    rul = pd.read_csv(path, sep=r"\s+", header=None, names=["RUL"])
    return rul


def add_train_rul(train_df, max_rul=125):
    """
    Compute the RUL label for every row of the training data.
    RUL = (max cycle for that engine) - (current cycle), capped at max_rul.
    Capping is standard practice: early-life degradation is ~flat, so capping
    prevents the model from being penalized for not predicting huge RUL values
    with no informative signal.
    """
    max_cycle = train_df.groupby("unit_number")["time_in_cycles"].transform("max")
    rul = max_cycle - train_df["time_in_cycles"]
    train_df = train_df.copy()
    train_df["RUL"] = rul.clip(upper=max_rul)
    return train_df


def get_feature_columns(df):
    """Return the list of sensor/op-setting columns to use as model input."""
    op_settings = [c for c in df.columns if c.startswith("op_setting")]
    sensors = [c for c in df.columns if c.startswith("sensor") and c not in DROP_SENSORS]
    return op_settings + sensors


def normalize(train_df, test_df, feature_cols):
    """
    Fit a MinMaxScaler on the TRAINING data only, apply to both train and test.
    (Never fit on test data -- that would leak information.)
    """
    scaler = MinMaxScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_df, test_df, scaler


def make_train_windows(train_df, feature_cols, window_size=30):
    """
    Slice each training engine's time-series into overlapping windows of
    length `window_size`. Label = RUL at the LAST cycle in the window.

    Engines with fewer than `window_size` cycles are skipped (rare in FD001,
    but possible depending on window_size).

    Returns:
        X: np.array of shape (num_windows, window_size, num_features)
        y: np.array of shape (num_windows,)
    """
    X, y = [], []
    for unit in train_df["unit_number"].unique():
        unit_df = train_df[train_df["unit_number"] == unit].sort_values("time_in_cycles")
        data = unit_df[feature_cols].values
        labels = unit_df["RUL"].values
        n_cycles = data.shape[0]

        if n_cycles < window_size:
            continue  # not enough history for a full window

        for start in range(n_cycles - window_size + 1):
            end = start + window_size
            X.append(data[start:end])
            y.append(labels[end - 1])

    return np.array(X), np.array(y)


def make_test_windows(test_df, feature_cols, window_size=30):
    """
    For the TEST set, C-MAPSS convention is: take only the LAST `window_size`
    cycles of each engine (since the true RUL label in RUL_FD001.txt applies
    to the final observed cycle). If an engine has fewer cycles than
    `window_size`, pad by repeating its first row.

    Returns:
        X: np.array of shape (num_engines, window_size, num_features)
    """
    X = []
    for unit in sorted(test_df["unit_number"].unique()):
        unit_df = test_df[test_df["unit_number"] == unit].sort_values("time_in_cycles")
        data = unit_df[feature_cols].values
        n_cycles = data.shape[0]

        if n_cycles >= window_size:
            window = data[-window_size:]
        else:
            pad_len = window_size - n_cycles
            padding = np.repeat(data[0:1], pad_len, axis=0)
            window = np.vstack([padding, data])

        X.append(window)

    return np.array(X)


def load_cmapss(data_dir="./data", subset="FD001", window_size=30, max_rul=125):
    """
    Convenience wrapper: loads, preprocesses, and windows the full FD001 dataset.

    Returns:
        X_train, y_train, X_test, y_test, feature_cols, scaler
    """
    train_df = load_raw(f"{data_dir}/train_{subset}.txt")
    test_df = load_raw(f"{data_dir}/test_{subset}.txt")
    rul_df = load_rul(f"{data_dir}/RUL_{subset}.txt")

    train_df = add_train_rul(train_df, max_rul=max_rul)
    feature_cols = get_feature_columns(train_df)
    train_df, test_df, scaler = normalize(train_df, test_df, feature_cols)

    X_train, y_train = make_train_windows(train_df, feature_cols, window_size)
    X_test = make_test_windows(test_df, feature_cols, window_size)
    y_test = rul_df["RUL"].clip(upper=max_rul).values

    return X_train, y_train, X_test, y_test, feature_cols, scaler


if __name__ == "__main__":
    # Quick smoke test with synthetic data so you can verify the pipeline
    # runs before you download the real dataset.
    print("Running smoke test with synthetic dummy data...")

    rng = np.random.default_rng(0)
    n_engines, cols = 5, COLUMN_NAMES
    rows = []
    for unit in range(1, n_engines + 1):
        n_cycles = rng.integers(40, 60)
        for t in range(1, n_cycles + 1):
            row = [unit, t] + list(rng.normal(size=3)) + list(rng.normal(size=21))
            rows.append(row)
    dummy_train = pd.DataFrame(rows, columns=cols)
    dummy_train = add_train_rul(dummy_train)

    feature_cols = get_feature_columns(dummy_train)
    X, y = make_train_windows(dummy_train, feature_cols, window_size=20)
    print(f"Synthetic train windows: X={X.shape}, y={y.shape}")
    assert X.shape[0] == y.shape[0]
    assert X.shape[2] == len(feature_cols)
    print("Smoke test passed.")
