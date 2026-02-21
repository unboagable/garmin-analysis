import sqlite3
import tempfile
import os
import logging
import pytest
from garmin_analysis.data_ingestion.inspect_sqlite_schema import inspect_sqlite_db


@pytest.fixture
def temp_db():
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT);")
    cursor.execute("CREATE TABLE logs (id INTEGER PRIMARY KEY, message TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP);")
    conn.commit()
    conn.close()

    yield db_path

    os.remove(db_path)


class TestInspectSchema:

    def test_prints_output(self, temp_db, caplog):
        caplog.set_level(logging.INFO)

        inspect_sqlite_db(temp_db)

        text = caplog.text
        assert "Table: users" in text
        assert "  - id (INTEGER)" in text
        assert "  - name (TEXT)" in text
        assert "  - email (TEXT)" in text
        assert "Table: logs" in text
        assert "  - message (TEXT)" in text
        assert "  - created_at (DATETIME)" in text
