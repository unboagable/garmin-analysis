import sqlite3
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Logging is configured at package level

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
            logging.info(f"Table: {table}")
            try:
                cursor.execute(f"PRAGMA table_info({table});")
                columns = cursor.fetchall()
                if columns:
                    for col in columns:
                        cid, name, dtype, notnull, dflt, pk = col
                        logging.info(f"  - {name} ({dtype})")
                else:
                    logging.warning("  (No columns found)")
            except Exception as e:
                logging.error(f"Error retrieving columns for {table}: {e}")

def extract_schema(db_path: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Return a mapping of table name -> list of (column_name, column_type)
    ordered by column position as defined in SQLite.
    """
    schema: Dict[str, List[Tuple[str, str]]] = {}
    if not Path(db_path).exists():
        return schema

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()
            # columns: (cid, name, type, notnull, dflt_value, pk)
            schema[table] = [(name, col_type) for _, name, col_type, *_ in columns]

    return schema

def detect_schema_drift(expected: Dict[str, List[Tuple[str, str]]], actual: Dict[str, List[Tuple[str, str]]]) -> Dict[str, dict]:
    """
    Compare expected vs actual schema and report drift per table.

    Returns a dict: {table: {"missing_columns": [...], "extra_columns": [...], "type_mismatches": [...]}}
    """
    report: Dict[str, dict] = {}
    all_tables = set(expected.keys()) | set(actual.keys())
    for table in sorted(all_tables):
        exp_cols = {c: t for c, t in expected.get(table, [])}
        act_cols = {c: t for c, t in actual.get(table, [])}

        missing = [c for c in exp_cols.keys() if c not in act_cols]
        extra = [c for c in act_cols.keys() if c not in exp_cols]
        mismatches = [c for c in exp_cols.keys() & act_cols.keys() if (exp_cols[c] or '').upper() != (act_cols[c] or '').upper()]

        if missing or extra or mismatches:
            report[table] = {
                "missing_columns": missing,
                "extra_columns": extra,
                "type_mismatches": [(c, exp_cols[c], act_cols[c]) for c in mismatches],
            }

    return report

def inspect_all_dbs(directory="db"):
    db_paths = sorted(Path(directory).glob("*.db"))
    if not db_paths:
        logging.warning(f"No .db files found in '{directory}'")
    for db_path in db_paths:
        inspect_sqlite_db(db_path)

if __name__ == "__main__":
    import argparse, json, sys
    parser = argparse.ArgumentParser(description="SQLite schema tools: inspect, export, and compare.")
    subparsers = parser.add_subparsers(dest="command", required=False)

    # inspect directory of dbs (default behaviour)
    parser.add_argument("--dir", default="db", help="Directory containing .db SQLite files (default: db)")

    # export schema
    export_p = subparsers.add_parser("export", help="Export a DB schema to JSON")
    export_p.add_argument("db", help="Path to SQLite database")
    export_p.add_argument("out", help="Output JSON file path")

    # compare schemas
    compare_p = subparsers.add_parser("compare", help="Compare a live DB against an expected schema JSON")
    compare_p.add_argument("db", help="Path to SQLite database")
    compare_p.add_argument("expected", help="Expected schema JSON path")
    compare_p.add_argument("--fail-on-drift", action="store_true", help="Exit with non-zero code if drift detected")

    args = parser.parse_args()

    if args.command == "export":
        schema = extract_schema(args.db)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(schema, f, indent=2)
        logging.info("Exported schema for %s to %s", args.db, args.out)
    elif args.command == "compare":
        with open(args.expected) as f:
            expected = json.load(f)
        actual = extract_schema(args.db)
        drift = detect_schema_drift(expected, actual)
        if not drift:
            logging.info("No schema drift detected.")
            sys.exit(0)
        # Pretty print drift
        logging.warning("Schema drift detected:")
        for table, info in drift.items():
            logging.warning("- %s", table)
            if info["missing_columns"]:
                logging.warning("  missing: %s", info["missing_columns"])
            if info["extra_columns"]:
                logging.warning("  extra: %s", info["extra_columns"])
            if info["type_mismatches"]:
                logging.warning("  type_mismatches: %s", info["type_mismatches"])
        if args.fail_on_drift:
            sys.exit(1)
    else:
        # default: inspect directory
        inspect_all_dbs(args.dir)

