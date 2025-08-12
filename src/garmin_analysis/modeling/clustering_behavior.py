import logging
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
from garmin_analysis.utils import load_master_dataframe, standardize_features

# Logging is configured at package level

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

def main():
    df = load_master_dataframe()

    features = [
        "steps", "activity_minutes", "training_effect",
        "total_sleep_min", "rem_sleep_min", "deep_sleep_min", "awake",
        "stress_avg", "stress_max", "stress_duration"
    ]

    X_scaled = standardize_features(df, features)
    if X_scaled.size == 0:
        logging.warning("No usable data available for clustering after dropping NaNs. Exiting.")
        return

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plot clusters
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="viridis", alpha=0.7)
    plt.title("Behavioral Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()

    out_path = PLOTS_DIR / "behavioral_clusters.png"
    plt.savefig(out_path)
    logging.info(f"Saved cluster plot to {out_path}")
    plt.close()

if __name__ == "__main__":
    main()
