import sqlite3
import pandas as pd

# Connect to Garmin DB
conn = sqlite3.connect("garmin.db")

# 1. File Types Present
files = pd.read_sql_query("SELECT * FROM files", conn)
file_types = files["type"].value_counts()
print("📂 File types in DB:\n", file_types)

# 2. Tables to Audit
tables_to_check = [
    "daily_summary", "sleep", "stress", "resting_hr", "weight",
    "attributes", "_attributes", "devices", "device_info"
]

# 3. Table Status Summary
print("\n📊 Table Status:")
for table in tables_to_check:
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5", conn)
        row_count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table}", conn)["count"][0]
        if df.isnull().all(axis=None):
            print(f"❗ {table:20} — {row_count} rows — all null/empty")
        else:
            print(f"✅ {table:20} — {row_count} rows — OK")
    except Exception as e:
        print(f"❌ {table:20} — not found")

# Close connection
conn.close()
