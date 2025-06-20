import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect("garmin.db")

# List all available tables
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Available tables:\n", tables["name"].tolist())

# Preview daily_summary with a few key metrics
daily = pd.read_sql_query("SELECT * FROM daily_summary", conn, parse_dates=["day"])
print("\nDaily Summary Sample:\n", daily[["day", "steps", "calories_total", "hr_min", "hr_max"]].head())

conn.close()
