# Garmin Analysis

A comprehensive Garmin health data analysis platform with interactive dashboard, machine learning capabilities, automated reporting, and automated data-quality checks.

## Features

- **📊 Interactive Dashboard**: Metric trends and correlation analysis
- **🤖 Machine Learning**: Enhanced anomaly detection, clustering, predictive modeling, and an end-to-end pipeline
- **📈 Visualization**: Correlations and trend plots
- **📋 Reporting**: Automated summaries and reports
- **🔍 Data Quality**: Quick checks, comprehensive audits, and tests

## Getting Started

### Poetry
```bash
pipx install poetry
poetry install
```

### Garmin data (via GarminDB)
1) Download/import with GarminDB, producing a `garmin.db`.
2) Copy it here:
```bash
mkdir -p db
cp /path/to/GarminDB/garmin.db db/garmin.db
```

## Common Commands

### Generate unified dataset
```bash
poetry run python -m garmin_analysis.data_ingestion.load_all_garmin_dbs
```
Creates `data/master_daily_summary.csv`.

### Launch dashboard
```bash
poetry run python -m garmin_analysis.dashboard.app
```
Open `http://localhost:8050`.

### Visualization utilities
```bash
poetry run python -m garmin_analysis.viz.plot_trends_range
poetry run python -m garmin_analysis.features.summary_stats
```

### Modeling
- Full pipeline (recommended):
```bash
poetry run python -m garmin_analysis.modeling.comprehensive_modeling_pipeline
```
- Individual modules:
```bash
poetry run python -m garmin_analysis.modeling.enhanced_anomaly_detection
poetry run python -m garmin_analysis.modeling.enhanced_clustering
poetry run python -m garmin_analysis.modeling.predictive_modeling
```

### Reporting
```bash
poetry run python -m garmin_analysis.reporting.run_all_analytics
poetry run python -m garmin_analysis.reporting.generate_trend_summary
```

### Data quality tools
- Quick check (summary, completeness, feature suitability):
```bash
poetry run python -m garmin_analysis.features.quick_data_check            # full quick check
poetry run python -m garmin_analysis.features.quick_data_check --summary
poetry run python -m garmin_analysis.features.quick_data_check --completeness
poetry run python -m garmin_analysis.features.quick_data_check --features
```
- Comprehensive audit with reports (JSON + Markdown in `data_quality_reports/`):
```bash
poetry run python -m garmin_analysis.features.data_quality_analysis
```

### SQLite schema inspection & drift
```bash
# Inspect one DB
poetry run python -m garmin_analysis.data_ingestion.inspect_sqlite_schema db/garmin.db

# Inspect directory of DBs (default: db)
poetry run python -m garmin_analysis.data_ingestion.inspect_sqlite_schema --dir db

# Export expected schema
poetry run python -m garmin_analysis.data_ingestion.inspect_sqlite_schema export db/garmin.db reports/expected_schema.json

# Compare live DB vs expected (non‑zero exit on drift)
poetry run python -m garmin_analysis.data_ingestion.inspect_sqlite_schema compare db/garmin.db reports/expected_schema.json --fail-on-drift
```

### Testing
Run unit tests (no I/O, uses in-memory DB fixtures):
```bash
poetry run pytest -q -m "not integration"
```

Run integration tests (file-backed temp DBs; exercises real I/O paths):
```bash
poetry run pytest -q -m integration
```

Run everything:
```bash
poetry run pytest -q
```

Test fixtures:
- `mem_db` (unit): In-memory SQLite with minimal schema and seed data for pure SQL/transform functions.
- `tmp_db` (integration): Temp file-backed SQLite DBs with realistic seeds; test code patches `garmin_analysis.data_ingestion.load_all_garmin_dbs.DB_PATHS` to point to these files.

Notes on data sources:
- If real Garmin DBs are available, place them under `db/` (e.g., `db/garmin.db`, `db/garmin_summary.db`, `db/garmin_activities.db`).
- When DBs are missing outside of tests, some commands may generate a synthetic dataset for convenience and log clear WARNINGS. This synthetic data is only for smoke testing and should not be used for real analysis.

## Project Structure
```
src/
└── garmin_analysis/
    ├── dashboard/
    ├── data_ingestion/
    ├── features/
    ├── modeling/
    ├── reporting/
    ├── viz/
    ├── utils.py
    └── utils_cleaning.py
```

## Notes
- This repo uses the `src/garmin_analysis` package layout. Prefer running modules via `python -m garmin_analysis.<module>`.
- If `poetry check` warns about mixed `[project.*]` and `[tool.poetry.*]`, migrate fully to PEP 621 `[project]` fields.
