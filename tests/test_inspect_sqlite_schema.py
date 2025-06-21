import sqlite3
import tempfile
import os
from io import StringIO
import sys
import pytest
from src.data_ingestion.inspect_sqlite_schema import inspect_sqlite_db

@pytest.fixture
def temp_db():
    # Create a temporary SQLite DB
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT);")
    cursor.execute("CREATE TABLE logs (id INTEGER PRIMARY KEY, message TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP);")
    conn.commit()
    conn.close()

    yield db_path

    # Clean up
    os.remove(db_path)

def test_inspect_sqlite_db_output(temp_db, capsys):
    # Run the inspection
    inspect_sqlite_db(temp_db)

    # Capture output
    captured = capsys.readouterr()
    out = captured.out

    # Assertions
    assert "📦 Table: users" in out
    assert "  - id (INTEGER)" in out
    assert "  - name (TEXT)" in out
    assert "  - email (TEXT)" in out
    assert "📦 Table: logs" in out
    assert "  - message (TEXT)" in out
    assert "  - created_at (DATETIME)" in out
