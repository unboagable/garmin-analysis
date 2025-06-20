import os
import logging
import pandas as pd
from utils import load_garmin_tables, filter_by_date, normalize_dates

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    if not os.path.exists("garmin.db"):
        logging.error("Missing garmin.db — please run garmindb_cli.py or place the database in the root directory.")
        return

    logging.info("Loading Garmin data tables...")
    try:
        tables = load_garmin_tables()
    except Exception as e:
        logging.exception("Failed to load tables from database: %s", e)
        return

    daily = filter_by_date(tables["daily"], days_back=60)
    sleep = filter_by_date(tables["sleep"], days_back=60)
    stress = filter_by_date(tables["stress"], date_col="timestamp", days_back=60)
    rest_hr = filter_by_date(tables["rest_hr"], days_back=60)

    if daily.empty or sleep.empty or stress.empty or rest_hr.empty:
        logging.warning("One or more filtered tables are empty. Check your Garmin data range.")

    # --- High-Level Trend Plots ---
    logging.info("Generating high-level trend plots...")
    daily.sort_values("day").set_index("day")[["steps", "calories_total", "hr_min", "hr_max", "distance"]].plot(subplots=True, figsize=(12, 10), title="Daily Activity Trends")
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.show()

    sleep.sort_values("day").set_index("day")[["score", "total_sleep"]].plot(subplots=True, figsize=(12, 6), title="Sleep Score and Duration")
    plt.tight_layout()
    plt.show()

    # Aggregate stress by day
    stress["day"] = pd.to_datetime(stress["timestamp"]).dt.normalize()
    stress_daily = stress.groupby("day")["stress"].mean().reset_index()
    rest_hr["day"] = pd.to_datetime(rest_hr["day"]).dt.normalize()
    stress_daily["day"] = pd.to_datetime(stress_daily["day"])
    rest_stress = pd.merge(rest_hr, stress_daily, on="day")
    rest_stress.sort_values("day").plot(x="day", y=["resting_heart_rate", "stress"], figsize=(12, 6), title="Resting Heart Rate vs. Stress")
    plt.tight_layout()
    plt.show()

    # --- Correlation Exploration ---
    logging.info("Exploring correlations...")
    sleep["day"] = pd.to_datetime(sleep["day"]).dt.normalize()
    sleep_stress = pd.merge(sleep, stress_daily, on="day")
    sleep_stress["total_sleep"] = pd.to_timedelta(sleep_stress["total_sleep"]).dt.total_seconds() / 3600
    logging.info("\nCorrelation between total sleep and stress:\n%s", sleep_stress[["total_sleep", "stress"]].corr())

    daily["next_day_rhr"] = daily["rhr"].shift(-1)
    logging.info("\nCorrelation between today's steps and next day's RHR:\n%s", daily[["steps", "next_day_rhr"]].corr())

    # --- Custom Wellness Score ---
    logging.info("Computing wellness score...")
    sleep["score_norm"] = sleep["score"] / sleep["score"].max()
    rest_hr["rhr_norm"] = 1 - (rest_hr["resting_heart_rate"] - rest_hr["resting_heart_rate"].min()) / (rest_hr["resting_heart_rate"].max() - rest_hr["resting_heart_rate"].min())
    stress_daily["stress_norm"] = 1 - (stress_daily["stress"] - stress_daily["stress"].min()) / (stress_daily["stress"].max() - stress_daily["stress"].min())

    wellness = sleep[["day", "score_norm"]].merge(rest_hr[["day", "rhr_norm"]], on="day").merge(stress_daily[["day", "stress_norm"]], on="day")
    wellness["wellness_score"] = wellness[["score_norm", "rhr_norm", "stress_norm"]].mean(axis=1)

    wellness.sort_values("day").plot(x="day", y="wellness_score", figsize=(12, 6), title="Custom Wellness Score")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


