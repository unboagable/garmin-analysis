import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    merged_path = "data/master_daily_summary.csv"
    if not os.path.exists(merged_path):
        logging.error("Missing master_daily_summary.csv — please run load_all_garmin_dbs.py first.")
        return

    logging.info("Loading master daily summary data...")
    df = pd.read_csv(merged_path, parse_dates=["day"])
    df = df.sort_values("day")

    # --- Trend Plots ---
    logging.info("Generating trend plots...")
    df.set_index("day")[[
        "steps", "calories_total", "hr_min", "hr_max", "distance"
    ]].plot(subplots=True, figsize=(12, 10), title="Daily Activity Trends")
    plt.tight_layout()
    plt.show()

    df.set_index("day")[[
        "score", "total_sleep_min", "rem_sleep_min", "deep_sleep_min"
    ]].plot(subplots=True, figsize=(12, 8), title="Sleep Quality Trends")
    plt.tight_layout()
    plt.show()

    df.set_index("day")[[
        "stress_avg", "stress_max", "stress_duration"
    ]].plot(subplots=True, figsize=(12, 8), title="Stress Trends")
    plt.tight_layout()
    plt.show()

    df.set_index("day")[[
        "yesterday_activity_minutes", "yesterday_training_effect"
    ]].plot(subplots=True, figsize=(12, 6), title="Lagged Activity Effects")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
