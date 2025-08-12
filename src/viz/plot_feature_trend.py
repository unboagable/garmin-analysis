import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime

# Logging is configured at package level

def plot_feature_trend(df: pd.DataFrame, feature: str,
                        date_col: str = None,
                        rolling_days: int = 7,
                        output_dir: str = "plots",
                        anomalies: pd.DataFrame = None):
    """
    Plots a single feature over time with an optional rolling average and anomaly highlights.

    Args:
        df (pd.DataFrame): Input DataFrame with date and feature columns.
        feature (str): Name of the feature column to plot.
        date_col (str or None): Name of the datetime column. If None, attempts auto-detection.
        rolling_days (int): Window for the rolling average.
        output_dir (str): Directory to save the plot.
        anomalies (pd.DataFrame or None): DataFrame with anomalies to highlight.
    """
    if feature not in df.columns:
        raise ValueError(f"Feature column '{feature}' not found in DataFrame.")

    # Auto-detect date column if not specified
    if date_col is None:
        candidate_cols = [col for col in df.columns if "date" in col.lower()]
        if not candidate_cols and "day" in df.columns:
            candidate_cols = ["day"]
        if not candidate_cols:
            raise ValueError("Could not auto-detect a date column. Please specify 'date_col'.")
        date_col = candidate_cols[0]  # choose the first candidate
        logging.info(f"Auto-detected date column: '{date_col}'")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    plt.figure(figsize=(16, 6))
    plt.plot(df[date_col], df[feature], label=feature, alpha=0.5)

    if rolling_days:
        df_rolling = df[[date_col, feature]].copy()
        df_rolling[feature] = df_rolling[feature].rolling(rolling_days).mean()
        plt.plot(df_rolling[date_col], df_rolling[feature], label=f"{feature} (Rolling {rolling_days}d)", linewidth=2)

    # Highlight anomalies
    if anomalies is not None and date_col in anomalies.columns:
        anomalies[date_col] = pd.to_datetime(anomalies[date_col])
        anomaly_dates = anomalies[date_col].tolist()
        anomaly_values = df[df[date_col].isin(anomaly_dates)][feature]
        plt.scatter(anomaly_dates, anomaly_values, color='red', label="Anomalies", zorder=5)

    plt.title(f"{feature} Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel(feature)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"{feature}_trend_{timestamp}.png")
    plt.savefig(out_path)
    plt.close()

    logging.info(f"Saved trend plot to {out_path}")

# Example usage:
if __name__ == "__main__":
    df = pd.read_csv("data/master_daily_summary.csv")
    anomalies = pd.read_csv("data/anomalies.csv") if os.path.exists("data/anomalies.csv") else None
    plot_feature_trend(df, feature="stress_avg", anomalies=anomalies)
