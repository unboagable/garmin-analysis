import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from src.utils import filter_required_columns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

def plot_columns(df, columns, title):
    available = [col for col in columns if col in df.columns]
    missing = set(columns) - set(available)
    if missing:
        logging.warning(f"Skipping missing columns: {missing}")
    if available:
        ax = df.set_index("day")[available].plot(subplots=True, figsize=(12, 6), title=title)
        plt.tight_layout()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}_" + title.lower().replace(" ", "_") + ".png"
        out_path = PLOTS_DIR / filename
        plt.savefig(out_path)
        logging.info(f"Saved plot to {out_path}")
        plt.close()
    else:
        logging.warning(f"No valid columns available for: {title}")

def main():
    merged_path = "data/master_daily_summary.csv"
    if not os.path.exists(merged_path):
        logging.error("Missing master_daily_summary.csv — please run load_all_garmin_dbs.py first.")
        return

    logging.info("Loading master daily summary data...")
    df = pd.read_csv(merged_path, parse_dates=["day"])
    df = df.sort_values("day")

    # Rename columns for consistency if needed
    if "steps_y" in df.columns:
        df = df.rename(columns={"steps_y": "steps"})

    # Filter out rows missing key predictors
    df = filter_required_columns(df, ["yesterday_activity_minutes", "stress_avg"])

    # --- Trend Plots ---
    logging.info("Generating trend plots...")
    plot_columns(df, ["steps", "calories_total", "hr_min", "hr_max", "distance"], "Daily Activity Trends")
    plot_columns(df, ["score", "total_sleep_min", "rem_sleep_min", "deep_sleep_min"], "Sleep Quality Trends")
    plot_columns(df, ["stress_avg", "stress_max", "stress_duration"], "Stress Trends")
    plot_columns(df, ["yesterday_activity_minutes", "yesterday_training_effect"], "Lagged Activity Effects")

if __name__ == "__main__":
    main()
