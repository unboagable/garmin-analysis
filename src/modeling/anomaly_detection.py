import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils import load_master_dataframe, standardize_features
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

def detect_anomalies(X):
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    preds = model.fit_predict(X)
    return preds, model

def main():
    df = load_master_dataframe()

    features = [
        "steps", "activity_minutes", "training_effect",
        "total_sleep_min", "rem_sleep_min", "deep_sleep_min", "awake",
        "stress_avg", "stress_max", "stress_duration"
    ]

    X_scaled = standardize_features(df, features)
    if X_scaled.size == 0:
        logging.warning("No data left after dropping NaNs in columns: %s", features)
        sys.exit(0)

    anomaly_labels, model = detect_anomalies(X_scaled)

    # Plot anomalies in 2D PCA space for simplicity
    from sklearn.decomposition import PCA
    X_pca = PCA(n_components=2).fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=anomaly_labels, cmap="coolwarm", alpha=0.7)
    plt.title("Anomaly Detection via Isolation Forest")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()

    out_path = PLOTS_DIR / "20250621_233652_anomaly_detection_plot.png"
    plt.savefig(out_path)
    logging.info(f"Saved anomaly plot to {out_path}")
    plt.close()

if __name__ == "__main__":
    main()
