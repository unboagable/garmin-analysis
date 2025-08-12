import pandas as pd
import logging
from pathlib import Path
from garmin_analysis.utils import load_master_dataframe
from garmin_analysis.utils_cleaning import clean_data

# Logging is configured at package level

def log_top_correlations(corr_df, threshold=0.5, max_pairs=20):
    seen = set()
    top_corrs = []

    for col in corr_df.columns:
        for idx in corr_df.index:
            if col == idx:
                continue
            pair = tuple(sorted((col, idx)))
            if pair in seen:
                continue
            seen.add(pair)
            value = corr_df.loc[idx, col]
            if abs(value) >= threshold:
                top_corrs.append((pair[0], pair[1], value))

    top_corrs.sort(key=lambda x: abs(x[2]), reverse=True)

    logging.info(f"Top {min(max_pairs, len(top_corrs))} correlated pairs with |r| ≥ {threshold}:")
    for x, y, r in top_corrs[:max_pairs]:
        logging.info(f"  • {x} ↔ {y}: {r:.2f}")

def generate_trend_summary(df, date_col='day', output_dir='reports'):
    df = clean_data(df)
    numeric_df = df.select_dtypes(include="number").dropna(axis=1, how="all")
    corr_matrix = numeric_df.corr(method="pearson")

    # ✅ Log top correlation pairs
    log_top_correlations(corr_matrix, threshold=0.5)

    # Identify top volatile features
    std_dev = numeric_df.std().sort_values(ascending=False)
    top_volatile = std_dev.head(10)

    # Missing data summary
    missing_pct = df.isnull().mean().sort_values(ascending=False)
    top_missing = missing_pct[missing_pct > 0].head(10)

    output_path = Path(output_dir) / f"trend_summary_{pd.Timestamp.now():%Y%m%d_%H%M%S}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# 📈 Garmin Data Trend Summary\n\n")

        f.write("## 🔗 Top Volatile Features (Std Dev)\n")
        f.write(top_volatile.to_string())
        f.write("\n\n")

        f.write("## ❗ Features with Missing Data\n")
        f.write(top_missing.to_string())
        f.write("\n\n")

    logging.info(f"Saved trend summary markdown to {output_path}")

if __name__ == "__main__":
    df = load_master_dataframe()
    generate_trend_summary(df)
