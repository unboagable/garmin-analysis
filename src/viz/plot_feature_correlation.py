import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_feature_correlation(df: pd.DataFrame, output_path: str = None,
                              method: str = "pearson", figsize=(18, 14),
                              exclude_cols=None, cmap="coolwarm"):
    """
    Generate a correlation matrix heatmap for numerical features in the DataFrame.

    Args:
        df (pd.DataFrame): Input dataset.
        output_path (str): File path to save the output plot. If None, autogenerates with timestamp.
        method (str): Correlation method: 'pearson', 'spearman', or 'kendall'.
        figsize (tuple): Size of the plot.
        exclude_cols (list or None): Columns to exclude from correlation computation.
        cmap (str): Colormap for the heatmap.
    """
    if exclude_cols is None:
        exclude_cols = []

    # Filter numeric columns and drop excluded
    df_corr = df.select_dtypes(include=['number']).drop(columns=exclude_cols, errors='ignore')

    if df_corr.empty:
        raise ValueError("No numeric columns available for correlation matrix.")

    # Compute correlation matrix
    corr_matrix = df_corr.corr(method=method)

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, square=True, cbar_kws={"shrink": 0.75})
    plt.title(f"Feature Correlation Matrix ({method.title()})")
    plt.tight_layout()

    # Generate timestamped output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"plots/feature_correlation_{timestamp}.png"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print(f"Saved correlation matrix to {output_path}")

# Example usage:
if __name__ == "__main__":
    sample = pd.read_csv("data/master_daily_summary.csv")  # adjust if needed
    plot_feature_correlation(sample, exclude_cols=["timestamp", "date"])
