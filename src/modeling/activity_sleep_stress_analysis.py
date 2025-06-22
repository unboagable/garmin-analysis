import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MERGED_PATH = Path("data/master_daily_summary.csv")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

def load_data():
    if not MERGED_PATH.exists():
        raise FileNotFoundError(f"Merged dataset not found: {MERGED_PATH}")
    df = pd.read_csv(MERGED_PATH, parse_dates=["day"])
    return df

def plot_correlations(df, show=False):
    numeric = df.select_dtypes(include=["number"]).dropna()
    corr = numeric.corr(method="spearman")

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Spearman Correlation Matrix: Activity vs Sleep/Stress")
    plt.tight_layout()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PLOTS_DIR / f"{timestamp_str}_correlation_matrix.png"
    plt.savefig(out_path)
    logging.info(f"Saved heatmap to {out_path}")
    if show:
        plt.show()
    else:
        plt.close()

def scatter_plot(df, x_col, y_col, show=False):
    df_clean = df[[x_col, y_col]].dropna().copy()

    def flatten(val):
        if isinstance(val, (list, np.ndarray)):
            if np.ndim(val) == 1 and len(val) > 0:
                return val[0]
            else:
                return np.nan
        return val

    df_clean[x_col] = df_clean[x_col].apply(flatten)
    df_clean[y_col] = df_clean[y_col].apply(flatten)

    def is_numeric(val):
        return isinstance(val, (int, float, np.integer, np.floating)) and not pd.isna(val)

    df_clean = df_clean[df_clean.apply(lambda row: is_numeric(row[x_col]) and is_numeric(row[y_col]), axis=1)]

    if df_clean.empty:
        logging.warning(f"No valid data to plot for {x_col} vs {y_col}")
        return

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_clean, x=x_col, y=y_col)
    sns.regplot(data=df_clean, x=x_col, y=y_col, scatter=False, color="red")
    plt.title(f"{x_col} vs {y_col}")
    plt.tight_layout()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PLOTS_DIR / f"{timestamp_str}_scatter_{x_col}_vs_{y_col}.png"
    plt.savefig(out_path)
    logging.info(f"Saved scatter plot to {out_path}")
    if show:
        plt.show()
    else:
        plt.close()

def grouped_box_plot(df, activity_col, target_col, show=False):
    col_data = df[activity_col].dropna()
    if col_data.nunique() < 4:
        logging.warning(f"Not enough unique values to create quartiles for {activity_col}")
        return
    try:
        df["activity_level"] = pd.qcut(df[activity_col], q=4, labels=["Low", "Med-Low", "Med-High", "High"], duplicates='drop')
    except ValueError as e:
        logging.warning(f"Failed to compute quartiles for {activity_col}: {e}")
        return

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="activity_level", y=target_col)
    plt.title(f"{target_col} by {activity_col} Quartiles")
    plt.tight_layout()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PLOTS_DIR / f"{timestamp_str}_box_{target_col}_by_{activity_col}_quartiles.png"
    plt.savefig(out_path)
    logging.info(f"Saved box plot to {out_path}")
    if show:
        plt.show()
    else:
        plt.close()

def main():
    df = load_data()

    logging.info(f"Available columns: {df.columns.tolist()}")
    df = df.rename(columns={"stress_avg_y": "stress_avg"})

    # Warn early if all-NaN in key features
    for col in ["steps", "yesterday_activity_minutes", "yesterday_training_effect"]:
        if col in df.columns and df[col].isna().all():
            logging.warning(f"Column '{col}' contains only NaNs")

    # Heatmap
    plot_correlations(df)

    # Scatter plots
    scatter_pairs = [
        ("steps", "score"),
        ("yesterday_activity_minutes", "score"),
        ("yesterday_training_effect", "score"),
        ("yesterday_activity_minutes", "stress_avg"),
    ]
    for x_col, y_col in scatter_pairs:
        if x_col in df.columns and y_col in df.columns:
            logging.info(f"Plotting: {x_col} vs {y_col}")
            scatter_plot(df, x_col, y_col)
        else:
            logging.warning(f"Skipping scatter plot: '{x_col}' or '{y_col}' not in data")

    # Box plots
    activity_cols = ["steps", "yesterday_activity_minutes", "yesterday_training_effect"]
    for activity_col in activity_cols:
        if activity_col in df.columns and "score" in df.columns:
            logging.info(f"Box plot: {activity_col} by score quartiles")
            grouped_box_plot(df, activity_col, "score")
        else:
            logging.warning(f"Skipping box plot: '{activity_col}' or 'score' not in data")

if __name__ == "__main__":
    main()
