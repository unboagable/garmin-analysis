import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
from garmin_analysis.config import PLOTS_DIR
from garmin_analysis.utils.data_loading import load_master_dataframe
from garmin_analysis.utils.data_filtering import standardize_features
from garmin_analysis.utils_cleaning import clean_data


logger = logging.getLogger(__name__)
# Logging is configured at package level

def run_anomaly_detection(df):
    df = clean_data(df)

    features = [
        "steps", "activity_minutes", "training_effect",
        "total_sleep_min", "rem_sleep_min", "deep_sleep_min", "awake",
        "stress_avg", "stress_max", "stress_duration"
    ]

    df_clean = df.dropna(subset=features, how='any')
    if df_clean.empty or len(df_clean) < 10:
        logger.warning("Not enough complete rows for anomaly detection. Skipping.")
        return pd.DataFrame(), None

    X_scaled = standardize_features(df_clean, features)

    # Use IsolationForest for anomaly detection
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    anomaly_labels = model.fit_predict(X_scaled)

    # PCA plot
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=anomaly_labels, cmap="coolwarm", alpha=0.7)
    plt.title("Anomaly Detection via Isolation Forest")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()

    out_path = PLOTS_DIR / "anomaly_detection.png"
    plt.savefig(out_path)
    logger.info(f"Saved anomaly plot to {out_path}")
    plt.close()

    df_result = df_clean.copy()
    df_result["anomaly_label"] = anomaly_labels
    anomalies_df = df_result[df_result["anomaly_label"] == -1]
    return anomalies_df, str(out_path)

def main():
    df = load_master_dataframe()
    anomalies_df, plot_path = run_anomaly_detection(df)
    if not anomalies_df.empty:
        logger.info(f"Detected {len(anomalies_df)} anomalies. Plot saved to {plot_path}")
    else:
        logger.info("No anomalies detected.")

if __name__ == "__main__":
    main()
