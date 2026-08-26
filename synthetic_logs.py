"""
synthetic_logs.py
Generates synthetic technician-style maintenance-log text for each C-MAPSS
sensor window, based on which sensors show the largest degradation trend
within that window and how close the engine is to failure (RUL).

This is deterministic given a fixed random seed (for phrase variety only --
which sensors get flagged and their severity is fully data-driven, not random),
so results are fully reproducible -- important for your methodology section.

Design note for your paper: sensor->physical-parameter mapping below follows
the official C-MAPSS sensor documentation (Saxena et al., 2008), so the
generated text reflects real engine physics rather than arbitrary labels.
"""

import random
import numpy as np
import pandas as pd

from data_utils import load_cmapss, get_feature_columns

# Official C-MAPSS sensor descriptions (Saxena et al., 2008)
SENSOR_INFO = {
    "sensor_1":  ("T2 (fan inlet temperature)", "temperature"),
    "sensor_2":  ("T24 (LPC outlet temperature)", "temperature"),
    "sensor_3":  ("T30 (HPC outlet temperature)", "temperature"),
    "sensor_4":  ("T50 (LPT outlet temperature)", "temperature"),
    "sensor_5":  ("P2 (fan inlet pressure)", "pressure"),
    "sensor_6":  ("P15 (bypass-duct pressure)", "pressure"),
    "sensor_7":  ("P30 (HPC outlet pressure)", "pressure"),
    "sensor_8":  ("Nf (fan speed)", "speed"),
    "sensor_9":  ("Nc (core speed)", "speed"),
    "sensor_10": ("EPR (engine pressure ratio)", "pressure"),
    "sensor_11": ("Ps30 (HPC outlet static pressure)", "pressure"),
    "sensor_12": ("phi (fuel flow / Ps30 ratio)", "flow"),
    "sensor_13": ("NRf (corrected fan speed)", "speed"),
    "sensor_14": ("NRc (corrected core speed)", "speed"),
    "sensor_15": ("BPR (bypass ratio)", "flow"),
    "sensor_16": ("farB (burner fuel-air ratio)", "flow"),
    "sensor_17": ("htBleed (bleed enthalpy)", "flow"),
    "sensor_18": ("Nf_dmd (demanded fan speed)", "speed"),
    "sensor_19": ("PCNfR_dmd (demanded corrected fan speed)", "speed"),
    "sensor_20": ("W31 (LPT coolant bleed)", "flow"),
    "sensor_21": ("W32 (HPT coolant bleed)", "flow"),
}

CAUSE_BY_TYPE = {
    "temperature": ["consistent with compressor/turbine efficiency degradation",
                    "suggesting thermal drift in the gas path"],
    "pressure":    ["consistent with airflow restriction",
                    "suggesting a pressure-ratio anomaly across the gas path"],
    "speed":       ["consistent with rotor/shaft performance degradation",
                    "suggesting spool speed instability"],
    "flow":        ["consistent with a bleed or flow-path irregularity",
                     "suggesting fuel-air ratio drift"],
}

DIRECTION_WORD = {1: "rising", -1: "falling"}

SEVERITY_OPENERS = {
    "healthy":  ["Routine inspection, engine operating within normal parameters.",
                 "No significant anomalies noted during routine check."],
    "warning":  ["Minor deviations observed during inspection.",
                 "Technician notes early signs of parameter drift."],
    "critical": ["Significant anomaly detected -- recommend priority follow-up.",
                 "Multiple parameters trending sharply -- inspection flagged urgent."],
}

ACTION_BY_SEVERITY = {
    "healthy":  ["No action required.", "Continue standard monitoring schedule."],
    "warning":  ["Recommend increased monitoring frequency.",
                 "Schedule follow-up inspection within next maintenance cycle."],
    "critical": ["Recommend immediate detailed inspection.",
                 "Flagged for priority maintenance action."],
}


def classify_severity(rul, max_rul=125):
    if rul >= 0.75 * max_rul:
        return "healthy"
    elif rul >= 0.25 * max_rul:
        return "warning"
    else:
        return "critical"


def detect_anomalous_sensors(window, feature_cols, threshold=0.15, top_k=3):
    """
    Compare the last cycle to the first cycle of the window for each feature
    (data is already 0-1 normalized). Returns up to top_k (sensor, delta)
    pairs whose absolute change exceeds `threshold`, sorted by magnitude.
    """
    delta = window[-1] - window[0]  # shape: (num_features,)
    candidates = [
        (feature_cols[i], delta[i])
        for i in range(len(feature_cols))
        if abs(delta[i]) >= threshold and feature_cols[i] in SENSOR_INFO
    ]
    candidates.sort(key=lambda x: abs(x[1]), reverse=True)
    return candidates[:top_k]


def generate_log(window, feature_cols, rul, rng, max_rul=125):
    """Generate one synthetic maintenance-log string for a single window."""
    severity = classify_severity(rul, max_rul)
    anomalies = detect_anomalous_sensors(window, feature_cols)

    opener = rng.choice(SEVERITY_OPENERS[severity])
    action = rng.choice(ACTION_BY_SEVERITY[severity])

    if not anomalies or severity == "healthy":
        return f"{opener} {action}"

    clauses = []
    for sensor, delta in anomalies:
        name, sensor_type = SENSOR_INFO[sensor]
        direction = DIRECTION_WORD[1 if delta > 0 else -1]
        cause = rng.choice(CAUSE_BY_TYPE[sensor_type])
        clauses.append(f"{name} {direction} ({cause})")

    body = "Observed: " + "; ".join(clauses) + "."
    return f"{opener} {body} {action}"


def generate_logs_for_windows(X, y, feature_cols, max_rul=125, seed=42):
    """
    X: array (num_windows, window_size, num_features)
    y: array (num_windows,) of RUL labels
    Returns: list[str] of log texts, same order/length as X and y.
    """
    rng = random.Random(seed)
    logs = []
    for i in range(X.shape[0]):
        logs.append(generate_log(X[i], feature_cols, y[i], rng, max_rul))
    return logs


def main():
    DATA_DIR = "./data"
    MAX_RUL = 125
    WINDOW_SIZE = 30

    print("Loading C-MAPSS data and rebuilding windows...")
    X_train, y_train, X_test, y_test, feature_cols, _ = load_cmapss(
        data_dir=DATA_DIR, subset="FD001", window_size=WINDOW_SIZE, max_rul=MAX_RUL
    )

    print("Generating synthetic logs for training windows...")
    train_logs = generate_logs_for_windows(X_train, y_train, feature_cols, MAX_RUL, seed=42)

    print("Generating synthetic logs for test windows...")
    test_logs = generate_logs_for_windows(X_test, y_test, feature_cols, MAX_RUL, seed=123)

    pd.DataFrame({"RUL": y_train, "log_text": train_logs}).to_csv(
        "results/train_logs.csv", index=False
    )
    pd.DataFrame({"RUL": y_test, "log_text": test_logs}).to_csv(
        "results/test_logs.csv", index=False
    )
    print(f"Saved {len(train_logs)} train logs and {len(test_logs)} test logs to results/")
    print("\nSample logs:")
    for i in [0, len(train_logs) // 2, len(train_logs) - 1]:
        print(f"  RUL={y_train[i]:.0f} | {train_logs[i]}")


if __name__ == "__main__":
    main()
