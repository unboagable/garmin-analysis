import pandas as pd
import logging
from pathlib import Path
import argparse
from garmin_analysis.utils.data_loading import load_master_dataframe
from garmin_analysis.utils_cleaning import clean_data
from garmin_analysis.utils.cli_helpers import add_24h_coverage_args, apply_24h_coverage_filter_from_args

logger = logging.getLogger(__name__)

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

    logger.info(f"Top {min(max_pairs, len(top_corrs))} correlated pairs with |r| ≥ {threshold}:")
    for x, y, r in top_corrs[:max_pairs]:
        logger.info(f"  • {x} ↔ {y}: {r:.2f}")

def generate_trend_summary(df, date_col='day', output_dir='reports', filter_24h_coverage=False, max_gap_minutes=2, day_edge_tolerance_minutes=2, coverage_allowance_minutes=0, timestamp=None):
    # Apply 24-hour coverage filtering if requested (using internal logic for function API compatibility)
    if filter_24h_coverage:
        from garmin_analysis.features.coverage import filter_by_24h_coverage
        logger.info("Filtering to days with 24-hour continuous coverage...")
        max_gap = pd.Timedelta(minutes=max_gap_minutes)
        day_edge_tolerance = pd.Timedelta(minutes=day_edge_tolerance_minutes)
        total_missing_allowance = pd.Timedelta(minutes=max(0, min(coverage_allowance_minutes, 300)))
        df = filter_by_24h_coverage(df, max_gap=max_gap, day_edge_tolerance=day_edge_tolerance, total_missing_allowance=total_missing_allowance)
        logger.info(f"After 24h coverage filtering: {len(df)} days remaining")
    
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

    # Use provided timestamp or generate new one
    if timestamp is None:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    output_path = Path(output_dir) / f"trend_summary_{timestamp}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# 📈 Garmin Data Trend Summary\n\n")
        
        if filter_24h_coverage:
            f.write("**Note:** This analysis is filtered to days with 24-hour continuous coverage.\n\n")

        f.write("## 🔗 Top Volatile Features (Std Dev)\n")
        f.write(top_volatile.to_string())
        f.write("\n\n")

        f.write("## ❗ Features with Missing Data\n")
        f.write(top_missing.to_string())
        f.write("\n\n")

    logger.info(f"Saved trend summary markdown to {output_path}")
    return str(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate trend summary from Garmin data')
    add_24h_coverage_args(parser)
    
    args = parser.parse_args()
    
    df = load_master_dataframe()
    generate_trend_summary(df, filter_24h_coverage=args.filter_24h_coverage, 
                         max_gap_minutes=args.max_gap, 
                         day_edge_tolerance_minutes=args.day_edge_tolerance,
                         coverage_allowance_minutes=args.coverage_allowance_minutes)
