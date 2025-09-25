import os
import sys
import sqlite3
from pathlib import Path
import pytest

# Ensure project root and src are on sys.path so `garmin_analysis` can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Use a non-interactive backend for matplotlib in tests
try:
    import matplotlib

    matplotlib.use("Agg")
except Exception:
    # If matplotlib is not installed for some environments, ignore
    pass

def _apply_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS daily_summary (
            day TEXT,
            steps REAL,
            calories_total REAL
        );
        CREATE TABLE IF NOT EXISTS sleep (
            day TEXT,
            total_sleep TEXT,
            deep_sleep TEXT,
            rem_sleep TEXT,
            score REAL
        );
        CREATE TABLE IF NOT EXISTS stress (
            timestamp TEXT,
            stress REAL
        );
        CREATE TABLE IF NOT EXISTS resting_hr (
            day TEXT,
            resting_heart_rate REAL
        );
        """
    )
    conn.commit()


def _seed_core(conn: sqlite3.Connection, num_days: int = 5):
    cur = conn.cursor()
    import datetime as _dt
    start = _dt.date(2024, 1, 1)
    daily_rows = []
    sleep_rows = []
    stress_rows = []
    rhr_rows = []
    for i in range(num_days):
        day = start + _dt.timedelta(days=i)
        day_str = day.isoformat()
        daily_rows.append((day_str, 5000.0 + i * 100.0, 2000.0 + (i % 5) * 20.0))
        # Sleep strings
        total_minutes = 6 * 60 + (i % 3) * 15
        deep_minutes = 60 + (i % 2) * 15
        rem_minutes = 90 + (i % 2) * 15
        def _hhmmss(m):
            return f"{m // 60:02d}:{m % 60:02d}:00"
        sleep_rows.append((day_str, _hhmmss(total_minutes), _hhmmss(deep_minutes), _hhmmss(rem_minutes), 70 + (i % 10)))
        # Stress 2 samples/day
        stress_rows.append((f"{day_str} 08:00:00", 20 + (i % 10)))
        stress_rows.append((f"{day_str} 17:00:00", 25 + (i % 10)))
        # Resting HR
        rhr_rows.append((day_str, 60 + (i % 5)))
    cur.executemany(
        "INSERT INTO daily_summary (day, steps, calories_total) VALUES (?, ?, ?)",
        daily_rows,
    )
    cur.executemany(
        "INSERT INTO sleep (day, total_sleep, deep_sleep, rem_sleep, score) VALUES (?, ?, ?, ?, ?)",
        sleep_rows,
    )
    cur.executemany(
        "INSERT INTO stress (timestamp, stress) VALUES (?, ?)",
        stress_rows,
    )
    cur.executemany(
        "INSERT INTO resting_hr (day, resting_heart_rate) VALUES (?, ?)",
        rhr_rows,
    )
    conn.commit()


@pytest.fixture
def mem_db():
    """In-memory SQLite DB for unit tests of pure SQL/transform functions."""
    conn = sqlite3.connect(":memory:")
    _apply_schema(conn)
    _seed_core(conn, num_days=5)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def tmp_db(tmp_path):
    """Temp file-backed SQLite DBs and patched paths for integration tests."""
    tmp_dir = Path(tmp_path)
    garmin_db = tmp_dir / "garmin.db"
    activities_db = tmp_dir / "garmin_activities.db"
    summary_db = tmp_dir / "garmin_summary.db"

    with sqlite3.connect(garmin_db) as conn:
        _apply_schema(conn)
        _seed_core(conn, num_days=60)
    with sqlite3.connect(summary_db) as conn:
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS days_summary (
                day TEXT,
                calories_bmr_avg REAL
            );
            """
        )
        import datetime as _dt
        start = _dt.date(2024, 1, 1)
        rows = [( (start + _dt.timedelta(days=i)).isoformat(), 1400 + (i % 3) * 10 ) for i in range(60)]
        cur.executemany("INSERT INTO days_summary (day, calories_bmr_avg) VALUES (?, ?)", rows)
        conn.commit()
    with sqlite3.connect(activities_db) as conn:
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS activities (
                activity_id TEXT,
                start_time TEXT,
                sport TEXT,
                name TEXT,
                description TEXT,
                training_effect REAL,
                anaerobic_training_effect REAL,
                elapsed_time TEXT,
                calories REAL
            );
            CREATE TABLE IF NOT EXISTS steps_activities (
                activity_id TEXT,
                avg_pace TEXT,
                vo2_max REAL
            );
            """
        )
        import datetime as _dt
        start = _dt.datetime(2024, 1, 1, 7, 0, 0)
        act_rows = []
        steps_rows = []
        sports = ['running', 'cycling', 'fitness_equipment', 'walking', 'swimming']
        for i in range(0, 60, 2):
            act_id = f"a{i}"
            st = start + _dt.timedelta(days=i, hours=(i % 3))
            sport = sports[i % len(sports)]
            name = f"Activity {i}"
            description = f"Description for activity {i}"
            act_rows.append((act_id, st.strftime("%Y-%m-%d %H:%M:%S"), sport, name, description, 2.0 + (i % 4) * 0.2, 0.3 + (i % 3) * 0.1, "00:30:00", 300 + (i % 3) * 25))
            steps_rows.append((act_id, "06:00" if i % 4 else "05:45", 40 + (i % 5)))
        cur.executemany(
            "INSERT INTO activities (activity_id, start_time, sport, name, description, training_effect, anaerobic_training_effect, elapsed_time, calories) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            act_rows,
        )
        cur.executemany(
            "INSERT INTO steps_activities (activity_id, avg_pace, vo2_max) VALUES (?, ?, ?)",
            steps_rows,
        )
        conn.commit()

    import garmin_analysis.data_ingestion.load_all_garmin_dbs as ladd
    original_paths = ladd.DB_PATHS.copy()
    ladd.DB_PATHS["garmin"] = garmin_db
    ladd.DB_PATHS["activities"] = activities_db
    ladd.DB_PATHS["summary"] = summary_db

    try:
        yield {
            "garmin": garmin_db,
            "activities": activities_db,
            "summary": summary_db,
        }
    finally:
        # Restore original DB paths
        ladd.DB_PATHS.update(original_paths)
        # Clean up any generated master CSV from integration runs
        master_csv = Path("data/master_daily_summary.csv")
        if master_csv.exists():
            try:
                master_csv.unlink()
            except Exception:
                pass

