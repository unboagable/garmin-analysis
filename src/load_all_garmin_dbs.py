import sqlite3
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DB_PATHS = [
    "garmin.db",
    "garmin_activities.db",
    "garmin_monitoring.db",
    "garmin_summary.db",
    "summary.db"
]

OUTPUT_PATH = Path("data/master_daily_summary.csv")


def load_table(db_file, table_name, parse_dates=None):
    if not Path(db_file).exists():
        logging.warning(f"{db_file} not found. Skipping.")
        return pd.DataFrame()
    with sqlite3.connect(db_file) as conn:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn, parse_dates=parse_dates)
            logging.info(f"Loaded {table_name} from {db_file} ({len(df)} rows)")
            return df
        except Exception as e:
            logging.warning(f"Failed to load {table_name} from {db_file}: {e}")
            return pd.DataFrame()


def summarize_and_merge():
    # Load from garmin.db
    daily = load_table("garmin.db", "daily_summary", parse_dates=["day"])
    sleep = load_table("garmin.db", "sleep", parse_dates=["day"])
    stress = load_table("garmin.db", "stress", parse_dates=["timestamp"])
    rhr = load_table("garmin.db", "resting_hr", parse_dates=["day"])

    # Load activities
    activities = load_table("garmin_activities.db", "activities", parse_dates=["start_time"])
    if not activities.empty:
        activities["day"] = pd.to_datetime(activities["start_time"]).dt.normalize()
        activities["had_workout"] = 1
        activity_summary = activities.groupby("day").agg(
            had_workout=("had_workout", "max"),
            activity_count=("activity_id", "count"),
            activity_minutes=("elapsed_time", lambda x: pd.to_timedelta(x).dt.total_seconds().sum() / 60),
            activity_calories=("calories", "sum"),
            training_effect=("training_effect", "mean"),
            anaerobic_te=("anaerobic_training_effect", "mean")
        ).reset_index()
    else:
        activity_summary = pd.DataFrame()

    # Aggregate stress
    if not stress.empty:
        stress["day"] = pd.to_datetime(stress["timestamp"]).dt.normalize()
        stress_daily = stress.groupby("day").agg(
            stress_avg=("stress", "mean"),
            stress_max=("stress", "max"),
            stress_duration=("stress", lambda x: (x > 0).sum())
        ).reset_index()
    else:
        stress_daily = pd.DataFrame()

    # Load monitoring HR and pulse ox
    mon_hr = load_table("garmin_monitoring.db", "monitoring_hr", parse_dates=["timestamp"])
    mon_ox = load_table("garmin_monitoring.db", "monitoring_pulse_ox", parse_dates=["timestamp"])
    if not mon_hr.empty:
        mon_hr["day"] = pd.to_datetime(mon_hr["timestamp"]).dt.normalize()
        mon_hr_daily = mon_hr.groupby("day").agg(monitoring_hr_avg=("heart_rate", "mean")).reset_index()
    else:
        mon_hr_daily = pd.DataFrame()
    if not mon_ox.empty:
        mon_ox["day"] = pd.to_datetime(mon_ox["timestamp"]).dt.normalize()
        mon_ox_daily = mon_ox.groupby("day").agg(pulse_ox_avg=("pulse_ox", "mean")).reset_index()
    else:
        mon_ox_daily = pd.DataFrame()

    # Load weekly summary data
    weeks_summary = load_table("summary.db", "weeks_summary", parse_dates=["first_day"])
    if not weeks_summary.empty:
        weeks_summary = weeks_summary.rename(columns={"first_day": "day"})
        summary_cols = ["day", "steps", "calories_avg", "stress_avg", "sleep_avg", "rem_sleep_avg"]
        week_summary_reduced = weeks_summary[summary_cols].copy()
    else:
        week_summary_reduced = pd.DataFrame()

    # Merge all into daily
    df = daily.copy()
    for df_extra in [sleep, rhr, stress_daily, activity_summary, mon_hr_daily, mon_ox_daily, week_summary_reduced]:
        if not df_extra.empty:
            df = pd.merge(df, df_extra, on="day", how="left")

    # Add lag features: yesterday's activity → today
    lag_cols = ["had_workout", "activity_minutes", "activity_calories", "training_effect", "anaerobic_te"]
    for col in lag_cols:
        if col in df.columns:
            df[f"yesterday_{col}"] = df[col].shift(1)

    logging.info(f"Final merged shape: {df.shape}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"Saved master dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    summarize_and_merge()
