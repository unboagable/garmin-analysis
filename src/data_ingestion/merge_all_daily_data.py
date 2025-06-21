import sqlite3
import pandas as pd
import logging
from pathlib import Path
from src.utils import convert_time_to_minutes, aggregate_stress, normalize_dates

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DB_DIR = Path("db")
OUTPUT_PATH = Path("data") / "merged_daily.csv"

DB_PATHS = {
    "garmin": DB_DIR / "garmin.db",
    "activities": DB_DIR / "garmin_activities.db",
    "monitoring": DB_DIR / "garmin_monitoring.db",
    "summary": DB_DIR / "garmin_summary.db",
    "summary2": DB_DIR / "summary.db",
}

def load_table(db_path: Path, table_name: str, parse_dates=None):
    if not db_path.exists():
        logging.warning(f"Database not found: {db_path}")
        return pd.DataFrame()

    with sqlite3.connect(db_path) as conn:
        try:
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=parse_dates)
            logging.info(f"Loaded {table_name} from {db_path.name} ({len(df)} rows)")
            return df
        except Exception as e:
            logging.warning(f"Failed to load {table_name} from {db_path.name}: {e}")
            return pd.DataFrame()

def clean_and_merge(tables: dict) -> pd.DataFrame:
    daily = tables.get("daily_summary")
    sleep = tables.get("sleep")
    stress = tables.get("stress")
    rhr = tables.get("resting_hr")
    days_summary = tables.get("days_summary")
    activities = tables.get("activities")
    steps_activities = tables.get("steps_activities")

    if daily is None or daily.empty:
        raise RuntimeError("Missing daily_summary table")

    for df in [daily, sleep, rhr, days_summary]:
        if df is not None and "day" in df.columns:
            normalize_dates(df, col="day")

    if sleep is not None and not sleep.empty:
        for col in ["total_sleep", "deep_sleep", "rem_sleep"]:
            sleep[col + "_min"] = sleep[col].apply(convert_time_to_minutes)
        sleep = sleep.drop(columns=["total_sleep", "deep_sleep", "rem_sleep"], errors="ignore")
        sleep = sleep[pd.to_numeric(sleep["score"], errors="coerce") > 0]

    daily = daily.dropna(subset=["steps", "calories_total"])
    stress_daily = aggregate_stress(stress) if stress is not None else pd.DataFrame()

    merged = daily.copy()
    if sleep is not None:
        merged = merged.merge(sleep, on="day", how="left")
    if not stress_daily.empty:
        merged = merged.merge(stress_daily, on="day", how="left")
    if rhr is not None:
        merged = merged.merge(rhr, on="day", how="left")
    if days_summary is not None and not days_summary.empty:
        merged = merged.merge(days_summary, on="day", how="left", suffixes=("", "_days"))

    if activities is not None and "start_time" in activities.columns:
        activities["day"] = pd.to_datetime(activities["start_time"]).dt.normalize()
        activities_day = activities.groupby("day").agg({
            "training_effect": "mean",
            "anaerobic_training_effect": "mean",
            "distance": "sum",
            "calories": "sum"
        }).reset_index()
        merged = merged.merge(activities_day, on="day", how="left")

    if steps_activities is not None and "activity_id" in steps_activities.columns:
        steps_activities = steps_activities.merge(
            activities[["activity_id", "start_time"]],
            on="activity_id", how="left"
        )
        steps_activities["day"] = pd.to_datetime(steps_activities["start_time"]).dt.normalize()
        steps_day = steps_activities.drop(columns=["activity_id", "start_time"], errors="ignore")
        steps_day = steps_day.groupby("day").mean(numeric_only=True).reset_index()
        merged = merged.merge(steps_day, on="day", how="left")

    logging.info(f"Final merged shape: {merged.shape}")
    return merged.sort_values("day")

def main():
    tables = {
        "daily_summary": load_table(DB_PATHS["garmin"], "daily_summary"),
        "sleep": load_table(DB_PATHS["garmin"], "sleep"),
        "stress": load_table(DB_PATHS["garmin"], "stress"),
        "resting_hr": load_table(DB_PATHS["garmin"], "resting_hr"),
        "days_summary": load_table(DB_PATHS["summary"], "days_summary"),
        "activities": load_table(DB_PATHS["activities"], "activities"),
        "steps_activities": load_table(DB_PATHS["activities"], "steps_activities"),
    }

    merged_df = clean_and_merge(tables)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    merged_df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"Merged and cleaned daily data saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
