import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_trend_summary(df: pd.DataFrame, date_col='day', exclude_cols=None, output_dir='reports'):
    """
    Generate a trend summary report:
    - Top correlations
    - Most volatile features (std deviation)
    - Date ranges with missing data

    Saves a markdown and notebook-compatible output.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(output_dir, f"trend_summary_{timestamp}.md")

    if exclude_cols is None:
        exclude_cols = []

    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found.")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(date_col)

    logging.info("Calculating top correlated feature pairs...")
    corr_matrix = df.select_dtypes(include='number').drop(columns=exclude_cols, errors='ignore').corr()
    corr_pairs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
        .rename(columns={0: 'correlation', 'level_0': 'feature_1', 'level_1': 'feature_2'})
        .sort_values(by='correlation', key=abs, ascending=False)
    )

    logging.info("Computing most volatile features...")
    volatility = df.select_dtypes(include='number').std().sort_values(ascending=False)

    logging.info("Identifying dates with missing data...")
    missing_by_date = df.set_index(date_col).isna().sum(axis=1)
    missing_ranges = missing_by_date[missing_by_date > 0]

    logging.info("Writing trend summary markdown report...")
    with open(md_path, 'w') as f:
        f.write(f"# Trend Summary Report\n\nGenerated on {timestamp}\n\n")

        f.write("## Top Correlated Feature Pairs\n\n")
        f.write(corr_pairs.head(10).to_markdown(index=False))
        f.write("\n\n")

        f.write("## Most Volatile Features (Standard Deviation)\n\n")
        f.write(volatility.head(10).to_frame(name='std_dev').to_markdown())
        f.write("\n\n")

        f.write("## Dates with Missing Data\n\n")
        if missing_ranges.empty:
            f.write("No missing values by date.\n")
        else:
            f.write(missing_ranges.to_frame(name='missing_count').to_markdown())

    logging.info(f"Saved trend summary markdown to {md_path}")

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("data/master_daily_summary.csv")
    generate_trend_summary(df, date_col="day", exclude_cols=["timestamp"])
