# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Garmin Analysis is a pure-Python health data analytics platform (Dash dashboard, ML pipeline, CLI tools). No Docker, no external services, no databases to run — everything is self-contained with SQLite.

### Running services

- **Dashboard**: `MPLBACKEND=Agg poetry run python run_dashboard.py` — serves at `http://localhost:8050`
  - Set `MPLBACKEND=Agg` to avoid display-server errors in headless environments.
  - The app auto-generates synthetic data when real Garmin DBs are absent (with warnings); this is expected.

### Key commands (see README for full list)

| Task | Command |
|------|---------|
| Lint | `poetry run ruff check .` |
| Format check | `poetry run ruff format --check .` |
| Unit tests | `MPLBACKEND=Agg poetry run pytest -q -m "not integration"` |
| Integration tests | `MPLBACKEND=Agg poetry run pytest -q -m integration --ignore=tests/test_notebooks.py` |
| All tests | `MPLBACKEND=Agg poetry run pytest` |

### Gotchas

- Always set `MPLBACKEND=Agg` when running tests or the dashboard in headless/CI environments to prevent matplotlib from trying to open a display.
- `poetry` is installed via pip to `~/.local/bin`; ensure `$HOME/.local/bin` is on `PATH`.
- One pre-existing test failure exists in `tests/test_adversarial.py::TestValidateFilePathAdversarial::test_path_traversal` — this is not caused by environment setup.
- The `ruff` config in `pyproject.toml` uses `include` to scope checks to specific files (CLI entry points and `run_dashboard.py`). Lint-only checks on `src/` or `tests/` dirs may need `--extend-include` or running from the project root.
