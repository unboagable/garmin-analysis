# Garmin Analysis

A comprehensive Garmin health data analysis platform with interactive dashboard, machine learning capabilities, and automated reporting.

## Features

- **📊 Interactive Dashboard**: Web-based dashboard with metric trends and correlation analysis
- **🤖 Machine Learning**: Anomaly detection, behavioral clustering, and predictive modeling
- **📈 Advanced Visualization**: Correlation matrices, trend analysis, and anomaly highlighting
- **📋 Automated Reporting**: Generate comprehensive health reports with trend summaries
- **🔍 Data Quality**: Missing data analysis and data health auditing
- **📱 Multi-Source Integration**: Combines data from multiple Garmin databases

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

### Launch Interactive Dashboard

Start the web-based dashboard for interactive exploration:

```bash
poetry run python src/dashboard/app.py
```

The dashboard will be available at `http://localhost:8050` and includes:
- Metric trends with date range selection
- Correlation heatmap visualization
- Interactive plots and filtering

### Run Exploratory Analysis

```bash
poetry run python src/viz/plot_trends_range.py
poetry run python src/modeling/activity_sleep_stress_analysis.py
```

### Run Machine Learning Models

**Anomaly Detection**: Identify unusual patterns in your health data
```bash
poetry run python src/modeling/anomaly_detection.py
```

**Behavioral Clustering**: Discover behavioral patterns and clusters
```bash
poetry run python src/modeling/clustering_behavior.py
```

### Generate Comprehensive Reports

**Full Analysis Report**: Complete health analysis with all insights
```bash
poetry run python src/reporting/run_all_analytics.py
```

**Monthly Report**: Focus on recent month's data
```bash
poetry run python src/reporting/run_all_analytics.py --monthly
```

**Trend Summary**: Generate trend analysis report
```bash
poetry run python src/reporting/generate_trend_summary.py
```

### Run Utility Checks

**Data Quality Audit**: Check for missing data and table health
```bash
poetry run python src/features/check_missing_data.py
```

**Summary Statistics**: Get overview of available data
```bash
poetry run python src/features/summary_stats.py
```

**Database Schema Inspection**: Examine database structure
```bash
poetry run python src/data_ingestion/inspect_sqlite_schema.py db/garmin.db
```

### Advanced Visualization

**Feature Correlation Analysis**: Generate correlation heatmaps
```bash
poetry run python src/viz/plot_feature_correlation.py
```

**Individual Feature Trends**: Plot specific metrics with rolling averages
```bash
poetry run python src/viz/plot_feature_trend.py
```

### Data Preparation

**Prepare Modeling Dataset**: Clean and prepare data for ML models
```bash
poetry run python src/data_ingestion/prepare_modeling_dataset.py
```

## Running Using Notebooks

In the `notebooks` directory, create `.ipynb` files that can utilize `requirements.txt` to install dependencies.

## Project Structure

```
src/
├── dashboard/          # Interactive web dashboard
├── data_ingestion/     # Data loading and preparation
├── features/           # Data quality and summary tools
├── modeling/           # Machine learning models
├── reporting/          # Automated report generation
├── viz/               # Visualization tools
└── utils.py           # Utility functions
```

## Dependencies

The project uses several key libraries:
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, tsfresh, statsmodels, prophet
- **Visualization**: matplotlib, seaborn, plotly
- **Dashboard**: Dash
- **Testing**: pytest

## Output Files

- **Plots**: Saved to `plots/` directory
- **Reports**: Generated in `reports/` directory
- **Data**: Processed datasets in `data/` directory
- **Logs**: All operations are logged with timestamps
