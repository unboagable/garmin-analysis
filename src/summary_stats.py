import pandas as pd
from utils import load_garmin_tables

def preview_table(df, name, cols=None, max_rows=3):
    print(f"\n📌 {name} — {len(df)} rows")
    if cols:
        print(df[cols].head(max_rows))
    else:
        print(df.head(max_rows))

def main():
    tables = load_garmin_tables()

    print("\n📂 Available tables:")
    for name in tables:
        print(f"- {name} ({len(tables[name])} rows)")

    preview_table(tables["daily"], "daily_summary", ["day", "steps", "calories_total", "hr_min", "hr_max", "rhr"])
    preview_table(tables["sleep"], "sleep", ["day", "total_sleep", "deep_sleep", "rem_sleep", "score"])
    preview_table(tables["stress"], "stress", ["timestamp", "stress"])
    preview_table(tables["rest_hr"], "resting_hr", ["day", "resting_heart_rate"])

if __name__ == "__main__":
    main()
