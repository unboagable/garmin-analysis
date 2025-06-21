import sqlite3
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def inspect_sqlite_db(db_path):
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    logging.info(f"Inspecting database: {db_path}")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        logging.info(f"Found {len(tables)} tables.")

        for table in tables:
            print(f"\n📦 Table: {table}")
            try:
                cursor.execute(f"PRAGMA table_info({table});")
                columns = cursor.fetchall()
                if columns:
                    for col in columns:
                        cid, name, dtype, notnull, dflt, pk = col
                        print(f"  - {name} ({dtype})")
                else:
                    print("  (No columns found)")
            except Exception as e:
                print(f"  ⚠️ Error retrieving columns for {table}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect SQLite DB schema")
    parser.add_argument("db_path", help="Path to the .db SQLite file")
    args = parser.parse_args()

    inspect_sqlite_db(args.db_path)
