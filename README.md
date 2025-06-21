# Garmin Analysis

Garmin analysis and dashboard

## Getting Started

### Installing pipx

Install pipx using Homebrew on macOS:

```bash
brew install pipx
pipx ensurepath
sudo pipx ensurepath --global
```

Install pipx via Scoop on Windows:

```bash
scoop install pipx
pipx ensurepath
```

### Installing Poetry

Install Poetry using pipx:

```bash
pipx install poetry
```

### Installing Dependencies

Lock dependencies and install them with Poetry:

```bash
poetry lock
poetry install
```

### Getting the Garmin Data

This project works with data extracted via [GarminDB by @tcgoetz](https://github.com/tcgoetz/GarminDB), a Python tool to download, import, and analyze data from Garmin Connect.

#### 1. Install GarminDB

Clone and install dependencies:
```bash
git clone https://github.com/tcgoetz/GarminDB.git
cd GarminDB
pip install -r requirements.txt
```

Create a `.netrc` file with your Garmin credentials:

```bash
machine connect.garmin.com
login YOUR_USERNAME
password YOUR_PASSWORD
```

#### 2. Download and Import Data

Run the GarminDB CLI tool to pull and process your data:

```bash
python garmindb_cli.py --all --download --import --analyze --latest
```

This will create a `garmin.db` SQLite file in the working directory.

#### 3. Move the Database

Copy the generated database into your project db directory:

```bash
mkdir -p db
cp path/to/GarminDB/garmin.db ./db/garmin.db
```

You're now ready to run analysis and models using your scripts.

#### 4. Add Optional Garmin Databases

To enrich your insights, copy additional `.db` files into the `db/` directory:
- `garmin_activities.db`
- `garmin_monitoring.db`
- `garmin_summary.db` or `summary.db`

These will be used to generate an enhanced dataset.

## Running the Application

### Generate Unified Dataset

Use the master loader to combine all Garmin sources:

```bash
poetry run python src/data_ingestion/load_all_garmin_dbs.py
```

This will create `data/master_daily_summary.csv` with sleep, stress, workouts, lagged features, and monitoring data.

### Run Exploratory Analysis

```bash
poetry run python src/viz/plot_trends_range.py
poetry run python src/modeling/activity_sleep_stress_analysis.py
```

### Run Predictive Model

```bash
poetry run python src/modeling/sleep_predictor.py
```

### Run Utility Checks

```bash
poetry run python src/features/check_missing_data.py
poetry run python src/features/summary_stats.py
poetry run python src/data_ingestion/inspect_sqlite_schema.py db/garmin.db
```

## Running Using Notebooks

In the `notebooks` directory, create `.ipynb` files that can utilize `requirements.txt` to install dependencies.
