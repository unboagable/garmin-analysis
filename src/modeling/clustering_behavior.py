import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
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
    "total_sleep", "rem_sleep", "deep_sleep", "awake",
    "stress_avg", "stress_max", "stress_duration"
]

def cluster_days(X_scaled, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X_scaled)
    return model, labels

def plot_clusters(X_scaled, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette="Set2")
    plt.title("Clustered Days by Behavior Archetype (PCA)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    out_path = PLOTS_DIR / f"{timestamp_str}_cluster_pca_plot.png"
    plt.savefig(out_path)
    logging.info(f"Saved cluster PCA plot to {out_path}")
    plt.close()

def summarize_clusters(features, labels):
    features["cluster"] = labels
    summary = features.groupby("cluster").mean()
    logging.info("Cluster Summary:\n%s", summary)
    return summary

if __name__ == "__main__":
    df = load_master_dataframe()
    available_cols = [col for col in FEATURE_COLS if col in df.columns]
    if len(available_cols) < len(FEATURE_COLS):
        missing = set(FEATURE_COLS) - set(available_cols)
        logging.warning(f"Skipping missing columns: {missing}")

    df = df.dropna(subset=available_cols)
    features = df[available_cols].copy()
    X_scaled = standardize_features(df, available_cols)

    model, labels = cluster_days(X_scaled, n_clusters=3)
    summary = summarize_clusters(features, labels)
    plot_clusters(X_scaled, labels)
