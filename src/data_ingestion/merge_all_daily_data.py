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

def load_table(db_path: Path, table_name: str, parse_dates=None) -> pd.DataFrame:
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

def ensure_datetime(df: pd.DataFrame, col: str) -> None:
    if df is not None and col in df.columns:
        df[col] = pd.to_datetime(df[col])

def normalize_day_column(df: pd.DataFrame, prefer_col: str = "day") -> pd.DataFrame:
    """
    Ensures a DataFrame has a 'day' column in datetime format.
    Falls back to 'calendarDate' if 'day' is missing.
    Logs and skips if neither exists.
    """
    if df is None or df.empty:
        return df

    if prefer_col in df.columns:
        df["day"] = pd.to_datetime(df[prefer_col])
    elif "calendarDate" in df.columns:
        df["day"] = pd.to_datetime(df["calendarDate"])
    elif "timestamp" in df.columns:
        df["day"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    else:
        logging.warning("Could not normalize 'day' column: no suitable source found.")
    return df

def preprocess_sleep(sleep: pd.DataFrame) -> pd.DataFrame:
    for col in ["total_sleep", "deep_sleep", "rem_sleep"]:
        sleep[col + "_min"] = sleep[col].apply(convert_time_to_minutes)
    sleep = sleep.drop(columns=["total_sleep", "deep_sleep", "rem_sleep"], errors="ignore")
    sleep = sleep[pd.to_numeric(sleep["score"], errors="coerce") > 0]
    return sleep

def merge_activity_stats(merged: pd.DataFrame, activities: pd.DataFrame) -> pd.DataFrame:
    if "start_time" not in activities.columns:
        return merged
    ensure_datetime(merged, "day")
    activities["day"] = pd.to_datetime(activities["start_time"]).dt.normalize()
    agg = activities.groupby("day").agg({
        "training_effect": "mean",
        "anaerobic_training_effect": "mean",
        "distance": "sum",
        "calories": "sum"
    }).reset_index()
    return merged.merge(agg, on="day", how="left")

def merge_step_stats(merged: pd.DataFrame, steps_activities: pd.DataFrame, activities: pd.DataFrame) -> pd.DataFrame:
    if "activity_id" not in steps_activities.columns:
        return merged
    ensure_datetime(merged, "day")
    steps = steps_activities.merge(
        activities[["activity_id", "start_time"]],
        on="activity_id", how="left"
    )
    steps["day"] = pd.to_datetime(steps["start_time"]).dt.normalize()
    steps_day = steps.drop(columns=["activity_id", "start_time"], errors="ignore")
    steps_day = steps_day.groupby("day").mean(numeric_only=True).reset_index()
    return merged.merge(steps_day, on="day", how="left")

def add_rolling_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("day")
    if "steps" in df.columns:
        df["steps_avg_7d"] = df["steps"].rolling(window=7, min_periods=1).mean()
    return df

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

    daily = normalize_day_column(daily)
    daily = daily.dropna(subset=["steps", "calories_total"])
    merged = daily.copy()

    for df in [sleep, rhr, days_summary, stress, activities, steps_activities]:
        normalize_day_column(df)

    ensure_datetime(merged, "day")

    if sleep is not None and not sleep.empty and "day" in sleep.columns:
        sleep = preprocess_sleep(sleep)
        ensure_datetime(sleep, "day")
        merged = merged.merge(sleep, on="day", how="left")

    stress_daily = aggregate_stress(stress) if stress is not None else pd.DataFrame()
    if not stress_daily.empty:
        ensure_datetime(stress_daily, "day")
        merged = merged.merge(stress_daily, on="day", how="left")

    if rhr is not None and not rhr.empty:
        ensure_datetime(rhr, "day")
        merged = merged.merge(rhr, on="day", how="left")

    if days_summary is not None and not days_summary.empty:
        ensure_datetime(days_summary, "day")
        merged = merged.merge(days_summary, on="day", how="left", suffixes=("", "_days"))

    if activities is not None and not activities.empty:
        merged = merge_activity_stats(merged, activities)

    if steps_activities is not None and not steps_activities.empty and activities is not None:
        merged = merge_step_stats(merged, steps_activities, activities)

    merged = add_rolling_metrics(merged)

    # Flag missing values
    if "score" in merged.columns:
        merged["missing_score"] = merged["score"].isna()
    if "training_effect" in merged.columns:
        merged["missing_training_effect"] = merged["training_effect"].isna()

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
