import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MERGED_PATH = Path("data/master_daily_summary.csv")


def load_data():
    if not MERGED_PATH.exists():
        raise FileNotFoundError(f"Merged dataset not found: {MERGED_PATH}")
    df = pd.read_csv(MERGED_PATH, parse_dates=["day"])
    return df


def plot_correlations(df):
    numeric = df.select_dtypes(include=["number"]).dropna()
    corr = numeric.corr(method="spearman")  # more robust than Pearson for nonlinear

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Spearman Correlation Matrix: Activity vs Sleep/Stress")
    plt.tight_layout()
    plt.show()


def scatter_plot(df, x_col, y_col):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col)
    sns.regplot(data=df, x=x_col, y=y_col, scatter=False, color="red")
    plt.title(f"{x_col} vs {y_col}")
    plt.tight_layout()
    plt.show()


def grouped_box_plot(df, activity_col, target_col):
    df["activity_level"] = pd.qcut(df[activity_col], q=4, labels=["Low", "Med-Low", "Med-High", "High"])
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="activity_level", y=target_col)
    plt.title(f"{target_col} by {activity_col} Quartiles")
    plt.tight_layout()
    plt.show()


def main():
    df = load_data()

    # Ensure column naming is consistent
    df = df.rename(columns={
        "stress_avg_y": "stress_avg"
    })

    # Correlation heatmap
    plot_correlations(df)

    # Activity → Sleep examples
    scatter_plot(df, "steps", "score")
    scatter_plot(df, "yesterday_activity_minutes", "score")
    scatter_plot(df, "yesterday_training_effect", "score")
    scatter_plot(df, "yesterday_activity_minutes", "stress_avg")

    # Box plots by quartile
    grouped_box_plot(df, "steps", "score")
    grouped_box_plot(df, "yesterday_activity_minutes", "score")
    grouped_box_plot(df, "yesterday_training_effect", "score")


if __name__ == "__main__":
    main()

