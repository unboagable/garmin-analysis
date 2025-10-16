import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import argparse
from garmin_analysis.config import PLOTS_DIR
from garmin_analysis.utils.data_filtering import filter_required_columns
from garmin_analysis.utils.data_loading import load_master_dataframe
from garmin_analysis.utils.cli_helpers import add_24h_coverage_args, apply_24h_coverage_filter_from_args

logger = logging.getLogger(__name__)

def plot_columns(df, columns, title):
    available = [col for col in columns if col in df.columns]
    missing = set(columns) - set(available)
    if missing:
        logger.warning(f"Skipping missing columns: {missing}")
    if available:
        ax = df.set_index("day")[available].plot(subplots=True, figsize=(12, 6), title=title)
        plt.tight_layout()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}_" + title.lower().replace(" ", "_") + ".png"
        out_path = PLOTS_DIR / filename
        plt.savefig(out_path)
        logger.info(f"Saved plot to {out_path}")
        plt.close()
    else:
        logger.warning(f"No valid columns available for: {title}")

def main():
    parser = argparse.ArgumentParser(description='Generate trend plots from Garmin data')
    add_24h_coverage_args(parser)
    
    args = parser.parse_args()
    
    # Load data using standardized loader
    logger.info("Loading master daily summary data...")
    df = load_master_dataframe()
    df = df.sort_values("day")

    # Rename columns for consistency if needed
    if "steps_y" in df.columns:
        df = df.rename(columns={"steps_y": "steps"})

    # Filter out rows missing key predictors
    df = filter_required_columns(df, ["yesterday_activity_minutes", "stress_avg"])
    
    # Apply 24-hour coverage filtering if requested
    df = apply_24h_coverage_filter_from_args(df, args)

    # --- Trend Plots ---
    logger.info("Generating trend plots...")
    plot_columns(df, ["steps", "calories_total", "hr_min", "hr_max", "distance"], "Daily Activity Trends")
    plot_columns(df, ["score", "total_sleep_min", "rem_sleep_min", "deep_sleep_min"], "Sleep Quality Trends")
    plot_columns(df, ["stress_avg", "stress_max", "stress_duration"], "Stress Trends")
    plot_columns(df, ["yesterday_activity_minutes", "yesterday_training_effect"], "Lagged Activity Effects")

if __name__ == "__main__":
    main()
