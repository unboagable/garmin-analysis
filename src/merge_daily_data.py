import pandas as pd
from pathlib import Path
import logging
from utils import load_garmin_tables, convert_time_to_minutes, aggregate_stress, normalize_dates

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === CONFIG ===
OUTPUT_PATH = Path("data/merged_daily.csv")

# === MAIN MERGE FUNCTION ===
def merge_garmin_data():
    tables = load_garmin_tables()
    if not tables:
        raise RuntimeError("Could not load Garmin tables from database.")

    daily = tables["daily"]
    sleep = tables["sleep"]
    stress = tables["stress"]
    rhr = tables["rest_hr"]

    logging.info("Cleaning and processing tables...")

    # Normalize date columns
    for df in [daily, sleep, rhr]:
        normalize_dates(df, col="day")

    # Clean sleep data
    sleep = sleep.copy()
    for col in ["total_sleep", "deep_sleep", "rem_sleep"]:
        sleep[col + "_min"] = sleep[col].apply(convert_time_to_minutes)
    sleep = sleep.drop(columns=["total_sleep", "deep_sleep", "rem_sleep"])
    sleep = sleep[pd.to_numeric(sleep["score"], errors="coerce") > 0]

    # Clean daily summary
    daily = daily.dropna(subset=["steps", "calories_total"])

    # Aggregate stress to daily level
    stress_daily = aggregate_stress(stress)

    # Merge everything
    logging.info("Merging all data on 'day'...")
    merged = daily.merge(sleep, on="day", how="inner")
    merged = merged.merge(stress_daily, on="day", how="left")
    merged = merged.merge(rhr, on="day", how="left")

    logging.info(f"Final merged shape: {merged.shape}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"Saved merged dataset to {OUTPUT_PATH}")

if __name__ == "__main__":
    merge_garmin_data()
