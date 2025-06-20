import sqlite3
import pandas as pd

def load_and_preview_table(conn, table_name, parse_dates=None, cols_to_show=None):
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn, parse_dates=parse_dates)
    print(f"\n📌 {table_name} — {len(df)} rows")
    if cols_to_show:
        print(df[cols_to_show].head())
    else:
        print(df.head(3))
    return df

# Connect to DB
conn = sqlite3.connect("garmin.db")

# Daily Health Summary
daily = load_and_preview_table(conn, "daily_summary", parse_dates=["day"],
                                cols_to_show=["day", "steps", "hr_min", "hr_max", "rhr"])

# Sleep Patterns
sleep = load_and_preview_table(conn, "sleep", parse_dates=["day"],
                                cols_to_show=["day", "total_sleep", "deep_sleep", "rem_sleep", "score"])

# Stress
stress = load_and_preview_table(conn, "stress", parse_dates=["timestamp"],
                                 cols_to_show=["timestamp", "stress"])

# Resting HR
rest_hr = load_and_preview_table(conn, "resting_hr", parse_dates=["day"],
                                  cols_to_show=["day", "resting_heart_rate"])

# Weight (optional)
weight = load_and_preview_table(conn, "weight", parse_dates=["timestamp"],
                                 cols_to_show=["timestamp", "weight"])

# Attributes (quick key:value dump)
attrs = pd.read_sql_query("SELECT * FROM attributes", conn)
print("\n🧬 Attributes Sample:\n", attrs.head(3))

# File types
files = pd.read_sql_query("SELECT * FROM files", conn)
file_types = files["type"].value_counts()
print("\n📂 File types in DB:\n", file_types)

# Close connection
conn.close()
