import sqlite3
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from ..utils.data_processing import normalize_day_column, convert_time_to_minutes, ensure_datetime_sorted

# Logging is configured at package level

# --- Paths ---
DB_PATHS = {
    "garmin": Path("db/garmin.db"),
    "activities": Path("db/garmin_activities.db"),
    "monitoring": Path("db/garmin_monitoring.db"),
    "summary": Path("db/garmin_summary.db"),
    "summary2": Path("db/summary.db"),
}

OUTPUT_PATH = Path("data/master_daily_summary.csv")

# Flag set to True when synthetic data is used to build the master dataset
USING_SYNTHETIC_DATA: bool = False

# --- Utility Functions ---
def _coalesce(df, out_col, *cands):
    """Write-first coalesce of multiple aliases into one canonical column, then drop extras."""
    if out_col not in df.columns:
        df[out_col] = np.nan
    for c in cands:
        if c in df.columns:
            df[out_col] = df[out_col].combine_first(df[c])
    for c in cands:
        if c in df.columns and c != out_col:
            df.drop(columns=c, inplace=True, errors="ignore")
    return df

def to_naive_day(s: pd.Series) -> pd.Series:
    """
    Convert a timestamp series to a tz-naive midnight 'day' (datetime64[ns]).
    Safe for tz-aware or tz-naive inputs.
    """
    s = pd.to_datetime(s, errors="coerce")  # preserves tz if present
    # strip timezone if tz-aware
    try:
        s = s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        # already tz-naive or not datetime-like; ignore
        pass
    # normalize to midnight and ensure dtype is datetime64[ns]
    return pd.to_datetime(s.dt.date)

def _create_synthetic_dataframes(num_days: int = 14) -> dict:
    """
    Create small, consistent synthetic dataframes to allow tests to run
    when no local SQLite databases are present.

    The generated data includes:
    - daily_summary with 'day', 'steps', 'calories_total'
    - sleep with time-like strings and a numeric 'score'
    - stress with per-minute timestamps across days
    - resting_hr with 'resting_heart_rate'
    - days_summary with 'calories_bmr_avg'
    - activities with 'start_time', 'training_effect', 'anaerobic_training_effect', 'elapsed_time'
    - steps_activities with per-activity numeric features
    """
    start = pd.Timestamp("2024-01-01")
    days = pd.date_range(start, periods=num_days, freq="D")

    # daily_summary
    daily_summary = pd.DataFrame({
        "day": days,
        "steps": (pd.Series(range(num_days)) * 1000 + 5000).astype(float),
        "calories_total": 2000 + (pd.Series(range(num_days)) % 5) * 50,
    })

    # sleep (time strings + numeric score)
    def hhmmss(minutes: int):
        h = minutes // 60
        m = minutes % 60
        return f"{h:02d}:{m:02d}:00"

    sleep = pd.DataFrame({
        "day": days,
        "total_sleep": [hhmmss(6 * 60 + (i % 3) * 15) for i in range(num_days)],
        "deep_sleep": [hhmmss(60 + (i % 2) * 15) for i in range(num_days)],
        "rem_sleep": [hhmmss(90 + (i % 2) * 15) for i in range(num_days)],
        "score": 70 + (pd.Series(range(num_days)) % 10),
    })

    # stress (two samples per day)
    stress_timestamps = []
    stress_values = []
    for d in days:
        stress_timestamps.append(pd.Timestamp(d) + pd.Timedelta(hours=9))
        stress_values.append(20 + (d.day % 10))
        stress_timestamps.append(pd.Timestamp(d) + pd.Timedelta(hours=17))
        stress_values.append(25 + (d.day % 10))
    stress = pd.DataFrame({
        "timestamp": stress_timestamps,
        "stress": stress_values,
    })

    resting_hr = pd.DataFrame({
        "day": days,
        "resting_heart_rate": 60 + (pd.Series(range(num_days)) % 5),
    })

    days_summary = pd.DataFrame({
        "day": days,
        "calories_bmr_avg": 1400 + (pd.Series(range(num_days)) % 3) * 10,
    })

    # activities (roughly every other day)
    activity_ids = []
    start_times = []
    training_effect = []
    anaerobic_te = []
    elapsed_time = []
    calories = []
    for i, d in enumerate(days):
        if i % 2 == 0:
            activity_ids.append(f"a{i}")
            start_times.append(pd.Timestamp(d) + pd.Timedelta(hours=7 + (i % 3)))
            training_effect.append(2.0 + (i % 4) * 0.2)
            anaerobic_te.append(0.3 + (i % 3) * 0.1)
            elapsed_time.append(f"00:{30 + (i % 3) * 15:02d}:00")
            calories.append(300 + (i % 3) * 30)
    activities = pd.DataFrame({
        "activity_id": activity_ids,
        "start_time": start_times,
        "training_effect": training_effect,
        "anaerobic_training_effect": anaerobic_te,
        "elapsed_time": elapsed_time,
        "calories": calories,
    })

    steps_activities = pd.DataFrame({
        "activity_id": activities["activity_id"],
        "avg_pace": ["06:00" if i % 2 == 0 else "05:45" for i in range(len(activities))],
        "vo2_max": 40 + (pd.Series(range(len(activities))) % 5),
    })

    return {
        "daily_summary": daily_summary,
        "sleep": sleep,
        "stress": stress,
        "resting_hr": resting_hr,
        "days_summary": days_summary,
        "activities": activities,
        "steps_activities": steps_activities,
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

def aggregate_stress(stress: pd.DataFrame) -> pd.DataFrame:
    if stress.empty or "timestamp" not in stress.columns:
        return pd.DataFrame()

    # Build tz-naive day
    day = to_naive_day(stress["timestamp"])
    stress = stress.assign(day=day)

    return stress.groupby("day").agg(
        stress_avg=("stress", "mean"),
        stress_max=("stress", "max"),
        stress_duration=("stress", lambda x: (x > 0).sum()),
    ).reset_index()

def preprocess_sleep(sleep: pd.DataFrame) -> pd.DataFrame:
    if sleep.empty:
        logging.warning("Sleep dataframe is empty.")
        return sleep

    original_rows = len(sleep)

    for col in ["total_sleep", "deep_sleep", "rem_sleep"]:
        if col in sleep.columns:
            sleep[col + "_min"] = sleep[col].apply(convert_time_to_minutes)
        else:
            logging.warning("Missing sleep column: %s", col)

    sleep.drop(columns=["total_sleep", "deep_sleep", "rem_sleep"], inplace=True, errors="ignore")

    if "score" in sleep.columns:
        sleep["score"] = pd.to_numeric(sleep["score"], errors="coerce")
        before_filter = len(sleep)
        sleep = sleep[sleep["score"] > 0]
        after_filter = len(sleep)
        dropped = before_filter - after_filter
        logging.info("Dropped %d rows from sleep data where score was missing or ≤ 0 (%d → %d)", dropped, before_filter, after_filter)
    else:
        logging.warning("No 'score' column found in sleep data.")

    final_rows = len(sleep)
    total_dropped = original_rows - final_rows
    logging.info("Total dropped rows during sleep preprocessing: %d of %d (%.1f%%)", total_dropped, original_rows, 100 * total_dropped / original_rows if original_rows > 0 else 0)

    return sleep

def summarize_and_merge(return_df: bool = False):
    daily = ensure_datetime_sorted(load_table(DB_PATHS["garmin"], "daily_summary", parse_dates=["day"]), ("day",))
    sleep = ensure_datetime_sorted(load_table(DB_PATHS["garmin"], "sleep", parse_dates=["day"]), ("day",))
    stress = ensure_datetime_sorted(load_table(DB_PATHS["garmin"], "stress", parse_dates=["timestamp"]), ("timestamp",))
    rhr = ensure_datetime_sorted(load_table(DB_PATHS["garmin"], "resting_hr", parse_dates=["day"]), ("day",))
    days_summary = ensure_datetime_sorted(load_table(DB_PATHS["summary"], "days_summary", parse_dates=["day"]), ("day",))
    activities = ensure_datetime_sorted(load_table(DB_PATHS["activities"], "activities", parse_dates=["start_time"]), ("start_time",))
    steps_activities = ensure_datetime_sorted(load_table(DB_PATHS["activities"], "steps_activities"), ("timestamp", "start_time", "day"))

    if daily is None or daily.empty:
        global USING_SYNTHETIC_DATA
        USING_SYNTHETIC_DATA = True
        logging.warning("No daily_summary data available from databases.")
        logging.warning("Generating synthetic sample data for tests/local usage WITHOUT real DBs.")
        logging.warning("This dataset is NOT real and should NOT be used for analysis.")
        synth = _create_synthetic_dataframes(num_days=14)
        daily = synth["daily_summary"]
        sleep = synth["sleep"]
        stress = synth["stress"]
        rhr = synth["resting_hr"]
        days_summary = synth["days_summary"]
        activities = synth["activities"]
        steps_activities = synth["steps_activities"]

    if "steps" in daily.columns:
        null_steps_count = daily["steps"].isnull().sum()
        logging.info(f"{null_steps_count} rows in daily_summary have null 'steps' — keeping for continuity")
    else:
        logging.warning("Column 'steps' not found in daily_summary — will affect downstream features")

    daily = normalize_day_column(daily, "daily")
    daily["day"] = to_naive_day(daily["day"])
    merged = daily.copy()

    if sleep is not None and not sleep.empty:
        sleep = normalize_day_column(preprocess_sleep(sleep), "sleep")
        sleep["day"] = to_naive_day(sleep["day"])
        merged = merged.merge(sleep, on="day", how="left")

    if stress is not None and not stress.empty:
        stress_daily = aggregate_stress(stress)
        stress_daily["day"] = to_naive_day(stress_daily["day"])  # no-op if already naive
        merged = merged.merge(stress_daily, on="day", how="left")

    if rhr is not None and not rhr.empty:
        rhr = normalize_day_column(rhr, "resting_hr")
        rhr["day"] = to_naive_day(rhr["day"])
        merged = merged.merge(rhr, on="day", how="left")

    if days_summary is not None and not days_summary.empty:
        days_summary = normalize_day_column(days_summary, "days_summary")
        days_summary["day"] = to_naive_day(days_summary["day"])
        merged = merged.merge(days_summary, on="day", how="left", suffixes=("", "_summary"))


    if activities is not None and not activities.empty:
        activities["day"] = to_naive_day(activities["start_time"])
        activities["had_workout"] = 1
        activity_summary = activities.groupby("day").agg(
            had_workout=("had_workout", "max"),
            activity_count=("activity_id", "count"),
            activity_minutes=("elapsed_time", lambda x: pd.to_timedelta(x).dt.total_seconds().sum() / 60),
            activity_calories=("calories", "sum"),
            training_effect=("training_effect", "mean"),
            anaerobic_te=("anaerobic_training_effect", "mean")
        ).reset_index()
        merged = merged.merge(activity_summary, on="day", how="left")

    if steps_activities is not None and not steps_activities.empty and activities is not None:
        steps = steps_activities.merge(activities[["activity_id", "start_time"]], on="activity_id", how="left")
        steps["day"] = to_naive_day(steps["start_time"])
        steps_day = steps.drop(columns=["activity_id", "start_time"], errors="ignore")
        steps_day = steps_day.groupby("day").mean(numeric_only=True).reset_index()
        steps_day = steps_day.rename(columns={"steps": "steps_from_steps_activity"})
        merged = merged.merge(steps_day, on="day", how="left")

    lag_cols = ["had_workout", "activity_minutes", "activity_calories", "training_effect", "anaerobic_te"]
    for col in lag_cols:
        if col in merged.columns:
            merged[f"yesterday_{col}"] = merged[col].shift(1)

    if "steps" not in merged.columns:
        logging.warning("Column 'steps' missing after all merges — injecting placeholder")
        merged["steps"] = pd.Series(dtype="float")

    if merged["steps"].isna().all():
        logging.warning("Column 'steps' has no valid data — will result in null steps_avg_7d")
        merged["steps_avg_7d"] = None
    else:
        merged["steps_avg_7d"] = merged["steps"].rolling(window=7, min_periods=1).mean()

    if "score" in merged.columns:
        merged["missing_score"] = merged["score"].isna()
    else:
        merged["missing_score"] = True

    if "training_effect" in merged.columns:
        merged["missing_training_effect"] = merged["training_effect"].isna()
    else:
        merged["missing_training_effect"] = True

    merged = ensure_datetime_sorted(merged, ("day",))

    logging.info(f"Final merged shape: {merged.shape}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if USING_SYNTHETIC_DATA:
        logging.warning("Saving SYNTHETIC master dataset to %s", OUTPUT_PATH)
        logging.warning("This file contains synthetic data generated due to missing databases.")
        logging.warning("Replace with real DBs in `db/` to produce real data.")
    merged.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"Saved master dataset to {OUTPUT_PATH}")

    if return_df:
        return merged

if __name__ == "__main__":
    summarize_and_merge()
