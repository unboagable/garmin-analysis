import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from src.utils import load_master_dataframe, standardize_features
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

def detect_anomalies(X):
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    preds = model.fit_predict(X)
    return preds, model

def run_anomaly_detection(df):
    features = [
        "steps", "activity_minutes", "training_effect",
        "total_sleep_min", "rem_sleep_min", "deep_sleep_min", "awake",
        "stress_avg", "stress_max", "stress_duration"
    ]

    X_scaled = standardize_features(df, features)
    if X_scaled.size == 0:
        logging.warning("No data left after dropping NaNs in columns: %s", features)
        return pd.DataFrame(), None

    anomaly_labels, model = detect_anomalies(X_scaled)

    # PCA plot
    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=anomaly_labels, cmap="coolwarm", alpha=0.7)
    plt.title("Anomaly Detection via Isolation Forest")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PLOTS_DIR / f"anomaly_detection_{timestamp_str}.png"
    plt.savefig(out_path)
    logging.info(f"Saved anomaly plot to {out_path}")
    plt.close()

    df_result = df.copy()
    df_result["anomaly_label"] = anomaly_labels
    anomalies_df = df_result[df_result["anomaly_label"] == -1]
    return anomalies_df, str(out_path)

def main():
    df = load_master_dataframe()
    anomalies_df, plot_path = run_anomaly_detection(df)
    if not anomalies_df.empty:
        logging.info(f"Detected {len(anomalies_df)} anomalies. Plot saved to {plot_path}")
    else:
        logging.info("No anomalies detected.")

if __name__ == "__main__":
    main()
