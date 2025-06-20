import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Connect to the Garmin DB
conn = sqlite3.connect("garmin.db")

# Load tables
print("Loading data...")
daily = pd.read_sql_query("SELECT * FROM daily_summary", conn, parse_dates=["day"])
sleep = pd.read_sql_query("SELECT * FROM sleep", conn, parse_dates=["day"])
stress = pd.read_sql_query("SELECT * FROM stress", conn, parse_dates=["timestamp"])
rest_hr = pd.read_sql_query("SELECT * FROM resting_hr", conn, parse_dates=["day"])

conn.close()

# --- High-Level Trend Plots ---
print("Generating high-level trend plots...")
daily.sort_values("day").set_index("day")[["steps", "calories_total", "hr_min", "hr_max", "distance"]].plot(subplots=True, figsize=(12, 10), title="Daily Activity Trends")
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
print("Exploring correlations...")
sleep["day"] = pd.to_datetime(sleep["day"]).dt.normalize()
sleep_stress = pd.merge(sleep, stress_daily, on="day")
# Ensure numeric columns only
sleep_stress["total_sleep"] = pd.to_timedelta(sleep_stress["total_sleep"]).dt.total_seconds() / 3600  # Convert to hours if timedelta
print("\nCorrelation between total sleep and stress:")
print(sleep_stress[["total_sleep", "stress"]].corr())

daily["next_day_rhr"] = daily["rhr"].shift(-1)
print("\nCorrelation between today's steps and next day's RHR:")
print(daily[["steps", "next_day_rhr"]].corr())

# --- Custom Wellness Score ---
print("Computing wellness score...")
sleep["score_norm"] = sleep["score"] / sleep["score"].max()
rest_hr["rhr_norm"] = 1 - (rest_hr["resting_heart_rate"] - rest_hr["resting_heart_rate"].min()) / (rest_hr["resting_heart_rate"].max() - rest_hr["resting_heart_rate"].min())
stress_daily["stress_norm"] = 1 - (stress_daily["stress"] - stress_daily["stress"].min()) / (stress_daily["stress"].max() - stress_daily["stress"].min())

wellness = sleep[["day", "score_norm"]].merge(rest_hr[["day", "rhr_norm"]], on="day").merge(stress_daily[["day", "stress_norm"]], on="day")
wellness["wellness_score"] = wellness[["score_norm", "rhr_norm", "stress_norm"]].mean(axis=1)

wellness.sort_values("day").plot(x="day", y="wellness_score", figsize=(12, 6), title="Custom Wellness Score")
plt.tight_layout()
plt.show()


