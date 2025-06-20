import sqlite3
import pandas as pd
import os

def audit_table_health(conn, tables_to_check):
    results = []
    for table in tables_to_check:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5", conn)
            row_count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table}", conn)["count"][0]
            status = "all null/empty" if df.isnull().all(axis=None) else "OK"
            results.append({"table": table, "rows": row_count, "status": status})
        except Exception:
            results.append({"table": table, "rows": 0, "status": "not found"})
    return pd.DataFrame(results)

def main(db_path="garmin.db", export_csv=True):
    if not os.path.exists(db_path):
        print(f"Database file '{db_path}' not found.")
        return

    conn = sqlite3.connect(db_path)

    # Check file types in DB
    try:
        files = pd.read_sql_query("SELECT * FROM files", conn)
        file_types = files["type"].value_counts()
        print("\n📂 File types in DB:\n", file_types)
    except Exception:
        print("\n⚠️  'files' table not found in the database.")

    # Table Audit
    print("\n📊 Table Status:")
    tables_to_check = [
        "daily_summary", "sleep", "stress", "resting_hr", "weight",
        "attributes", "_attributes", "devices", "device_info"
    ]
    report = audit_table_health(conn, tables_to_check)
    print(report.to_string(index=False))

    if export_csv:
        os.makedirs("data", exist_ok=True)
        report.to_csv("data/missing_report.csv", index=False)
        print("\n✅ Report saved to data/missing_report.csv")

    conn.close()

if __name__ == "__main__":
    main()
