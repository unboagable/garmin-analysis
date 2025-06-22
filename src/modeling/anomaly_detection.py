import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging
from src.utils import load_master_dataframe, standardize_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

FEATURE_COLS = [
    "steps", "activity_minutes", "training_effect",
    "total_sleep_min", "rem_sleep_min", "deep_sleep_min", "awake",
    "stress_avg", "stress_max", "stress_duration"
]

def detect_anomalies(X_scaled):
    model = IsolationForest(contamination=0.1, random_state=42)
    preds = model.fit_predict(X_scaled)
    return preds, model

def plot_anomalies(df, anomaly_labels):
    df["anomaly"] = anomaly_labels
    df["anomaly"] = df["anomaly"].map({1: "Normal", -1: "Anomaly"})

    if "score" not in df.columns:
        logging.warning("'score' column missing — skipping anomaly plot.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df["day"], df["score"], label="Score")
    plt.scatter(df[df["anomaly"] == "Anomaly"]["day"],
                df[df["anomaly"] == "Anomaly"]["score"],
                color='red', label="Anomalies")
    plt.xlabel("Date")
    plt.ylabel("Score")
    plt.title("Anomaly Detection in Sleep/Activity/Stress")
    plt.legend()
    plt.tight_layout()
    out_path = PLOTS_DIR / f"{timestamp_str}_anomaly_detection_plot.png"
    plt.savefig(out_path)
    logging.info(f"Saved anomaly plot to {out_path}")
    plt.close()

if __name__ == "__main__":
    df = load_master_dataframe()
    required_cols = FEATURE_COLS + ["score"]
    available_cols = [col for col in required_cols if col in df.columns]
    if len(available_cols) < len(required_cols):
        missing = set(required_cols) - set(available_cols)
        logging.warning(f"Skipping missing columns: {missing}")

    df = df.dropna(subset=available_cols)
    if df.empty:
        logging.warning("No usable data available after dropping rows with NaNs. Exiting.")
        exit()

    feature_cols_available = [col for col in FEATURE_COLS if col in df.columns]
    features = df[feature_cols_available].copy()
    X_scaled = standardize_features(df, feature_cols_available)

    anomaly_labels, model = detect_anomalies(X_scaled)
    plot_anomalies(df, anomaly_labels)
