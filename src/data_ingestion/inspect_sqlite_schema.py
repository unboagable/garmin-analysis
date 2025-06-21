import sqlite3
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def inspect_sqlite_db(db_path):
    if not Path(db_path).exists():
        logging.warning(f"Database not found at {db_path}")
        return

    logging.info(f"Inspecting database: {db_path}")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

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

def inspect_all_dbs(directory="db"):
    db_paths = sorted(Path(directory).glob("*.db"))
    if not db_paths:
        logging.warning(f"No .db files found in '{directory}'")
    for db_path in db_paths:
        inspect_sqlite_db(db_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect all SQLite DB schemas in a folder")
    parser.add_argument("--dir", default="db", help="Directory containing .db SQLite files (default: db)")
    args = parser.parse_args()

    inspect_all_dbs(args.dir)

