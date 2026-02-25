# Garmin Analysis

A comprehensive Garmin health data analysis platform with interactive dashboard, machine learning capabilities, automated reporting, and **24-hour coverage filtering** for high-quality data analysis.

## 📑 Table of Contents

- [Quick Start](#quick-start-in-5-minutes)
- [What's New](#-whats-new)
- [Features](#features)
- [Getting Started](#getting-started)
- [Common Commands](#common-commands)
- [24-Hour Coverage Filtering](#-quick-start-with-24-hour-coverage-filtering)
- [Dashboard](#launch-dashboard)
- [Day-of-Week Analysis](#-day-of-week-analysis)
- [Time-of-Day Stress Analysis](#-time-of-day-stress-analysis)
- [Activity Calendar](#-activity-calendar--type-mappings)
- [Machine Learning & Modeling](#machine-learning--modeling)
- [Reporting & Analytics](#reporting--analytics) (incl. Weekly Health Report)
- [Data Quality Tools](#data-quality-tools)
- [Use Cases](#use-cases)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Licensing](#licensing)

## Quick Start (in 5 minutes)

**Bootstrap (single command):**
```bash
poetry install
poetry run garmin init   # or: garmin-init — checks DBs, creates folders, validates schema, prints next commands
```

**Option A: Automated sync with Garmin Connect** (requires GarminDB from source):
```bash
# 1. Install dependencies
pipx install poetry
poetry install

# 2. Install GarminDB from source (required for automated sync)
git clone --recursive https://github.com/tcgoetz/GarminDB.git ~/GarminDB
cd ~/GarminDB && make setup

# 3. Configure Garmin Connect credentials
cd ~/Code/garmin-analysis
poetry run python -m garmin_analysis.cli_garmin_sync --setup \
  --username your@email.com \
  --password yourpassword \
  --start-date 01/01/2024

# 4. Download your Garmin data
poetry run python -m garmin_analysis.cli_garmin_sync --sync --all

# 5. Generate unified dataset
poetry run python -m garmin_analysis.data_ingestion.load_all_garmin_dbs

# 6. Launch the dashboard
poetry run python run_dashboard.py
# Open http://localhost:8050 in your browser
```

**Option B: Manual setup** (if you already have garmin.db):
```bash
# 1. Install dependencies
pipx install poetry
poetry install

# 2. Set up your Garmin data
mkdir -p db
cp /path/to/GarminDB/garmin.db db/garmin.db

# 3. Generate unified dataset
poetry run python -m garmin_analysis.data_ingestion.load_all_garmin_dbs

# 4. Launch the dashboard
poetry run python run_dashboard.py
# Open http://localhost:8050 in your browser
```

For detailed setup instructions, see [Getting Started](#getting-started).

## 🆕 What's New

### 🚀 **Single-Command Bootstrap & Export** (February 2026)

- **`garmin init`** — Checks databases, creates folders, validates schema, prints next commands
- **Daily data quality score** — Composite 0–100 score (coverage + completeness), persisted to CSV and dashboard tab
- **Export to Parquet** — `data/export/master_daily_summary.parquet` for faster analytics
- **Optional DuckDB** — SQL queries on master dataset: `poetry run python -m garmin_analysis.cli_export --duckdb`

### 📅 **Weekly Health Report** (February 2026)

Automated weekly Markdown reports that track three key health metrics over time:

- **Sleep score trend** — weekly average with week-over-week delta and direction
- **Resting HR delta** — weekly average with delta (lower is better)
- **Stress minutes delta** — weekly total with delta (lower is better)

```bash
# Generate a weekly report (last 4 weeks by default)
poetry run python -m garmin_analysis.cli_weekly_report

# Last 8 weeks, with 24h coverage filtering
poetry run python -m garmin_analysis.cli_weekly_report --weeks 8 --filter-24h-coverage
```

Reports are saved to `reports/weekly_report_<timestamp>.md` with a narrative summary, metric tables with trend indicators, and week-over-week deltas.

- 129 comprehensive tests covering helpers, section builders, aggregation, CLI, and edge cases

### 🌙 **HR & Activity Impact on Sleep Model** (October 2025)

**NEW!** Comprehensive model analyzing how heart rate metrics and physical activities affect sleep quality!

- ✅ **Sophisticated ML model** - 6 algorithms tested (ElasticNet best: R²=0.258)
- ✅ **28 features analyzed** - HR (min/max/resting), activities, and lag features
- ✅ **Configurable imputation** - 6 strategies for handling missing data
- ✅ **4 visualizations** - Performance, importance, predictions, correlations
- ✅ **Comprehensive testing** - 42 dedicated tests ensuring reliability
- ✅ **Extensive documentation** - Complete guides and examples

**Key Finding**: Body Battery is the strongest predictor of sleep quality, followed by heart rate metrics (23.4% importance) and activity metrics (20.7% importance).

```bash
# Run the sleep analysis model
poetry run python -m garmin_analysis.modeling.hr_activity_sleep_model

# Or use programmatically
from garmin_analysis.modeling.hr_activity_sleep_model import HRActivitySleepModel
model = HRActivitySleepModel()
results = model.run_analysis(imputation_strategy='median')
```

**See**: `docs/imputation_strategies.md` for complete guide

### 🔧 **Repository-Wide Imputation Standardization** (October 2025)

Standardized missing value handling across all core modeling files!

- ✅ **Shared imputation utility** - `utils/imputation.py` with 6 strategies
- ✅ **Applied to 4 core files** - Prevents 53% data loss
- ✅ **Improved performance** - 33% better R² with median vs drop
- ✅ **32 comprehensive tests** - Full coverage of all strategies
- ✅ **Backward compatible** - Existing code works unchanged

**Strategies**: `median` (default, recommended), `mean`, `drop`, `forward_fill`, `backward_fill`, `none`

```python
from garmin_analysis.utils.imputation import impute_missing_values

# Robust median imputation (recommended for health data)
df_clean = impute_missing_values(df, ['hr_min', 'steps'], strategy='median')
```

### 📅 **Day-of-Week Analysis**

Analyze sleep score, body battery, and water intake patterns by day of the week to identify weekly trends and optimize your health routines.

- ✅ **Interactive dashboard** with day-of-week analysis tab
- ✅ **CLI tool** for standalone day-of-week analysis
- ✅ **Comprehensive visualizations** with bar charts and trend comparisons
- ✅ **24-hour coverage filtering** support for reliable analysis
- ✅ **Automated summary reports** showing best/worst days

**24-Hour Coverage Filtering** is available across all analysis tools! This major feature enhancement allows you to filter your analysis to only include days with complete 24-hour continuous data coverage, ensuring more reliable and accurate results.

- ✅ **All visualization tools** now support `--filter-24h-coverage`
- ✅ **Interactive dashboard** has real-time filtering checkboxes
- ✅ **Modeling pipeline** can train on high-quality data only
- ✅ **Reporting tools** generate cleaner, more reliable reports
- ✅ **Configurable parameters** for gap tolerance and edge tolerance

## Features

- **🔄 Garmin Connect Integration**: **NEW!** CLI tools for [GarminDB](https://github.com/tcgoetz/GarminDB) to automate data download and configuration
- **🌙 HR & Activity → Sleep Model**: **NEW!** Analyze how heart rate and activities affect sleep quality with 6 ML algorithms
- **🔧 Flexible Imputation**: **NEW!** 6 strategies for handling missing data (median, mean, drop, forward/backward fill, none)
- **📅 Day-of-Week Analysis**: Analyze sleep score, body battery, and water intake patterns by day of the week
- **⏰ 24-Hour Coverage Filtering**: Filter analysis to only days with complete 24-hour continuous data coverage for more reliable results
- **📅 Activity Calendar**: Visualize activity patterns with color-coded calendar showing different activity types
- **🏷️ Activity Type Mappings**: Customize display names and colors for unknown or poorly named activity types
- **📊 Interactive Dashboard**: Real-time metric trends, correlation analysis, and day-of-week analysis with filtering options
- **🤖 Machine Learning**: Comprehensive ML pipeline with anomaly detection, clustering, and predictive modeling
- **📈 Visualization**: Multiple plotting tools for trends, correlations, and feature analysis
- **📋 Reporting**: Automated summaries, comprehensive analytics reports, and weekly health reports
- **🔍 Data Quality**: Advanced data quality analysis and coverage assessment tools
- **🗄️ Data Ingestion**: Unified data loading from multiple Garmin databases with schema validation
- **🧪 Testing**: Comprehensive test suite with 970+ tests (unit and integration)
- **📓 Notebooks**: Interactive Jupyter notebooks for exploratory analysis

## Getting Started

### Prerequisites

- Python 3.11 or 3.12 or 3.13 (required)
- [Poetry](https://python-poetry.org/) for dependency management
- **Garmin Connect account** (for automated data sync via GarminDB)
- OR a pre-existing `garmin.db` file from [GarminDB](https://github.com/tcgoetz/GarminDB)

### Installation

1. **Install Poetry** (if not already installed):
```bash
pipx install poetry
```

2. **Clone the repository** (if you haven't already):
```bash
git clone <repository-url>
cd garmin-analysis
```

3. **Install dependencies**:
```bash
poetry install
```

### Data Setup

**NEW!** Automated sync with Garmin Connect (requires GarminDB from source):

1. **Install GarminDB** (one-time setup):
```bash
git clone --recursive https://github.com/tcgoetz/GarminDB.git ~/GarminDB
cd ~/GarminDB && make setup
```

2. **Set up your Garmin Connect credentials**:
```bash
cd ~/Code/garmin-analysis
poetry run python -m garmin_analysis.cli_garmin_sync --setup \
  --username your@email.com \
  --password yourpassword \
  --start-date 01/01/2024
```

3. **Download all your Garmin data** (first time only):
```bash
poetry run python -m garmin_analysis.cli_garmin_sync --sync --all
```

4. **Generate the unified dataset**:
```bash
poetry run python -m garmin_analysis.data_ingestion.load_all_garmin_dbs
```

**Alternative:** Manual export (if you prefer):

1. **Export your Garmin data** using [GarminDB](https://github.com/tcgoetz/GarminDB) to produce a `garmin.db` file.

2. **Copy the database** to this project:
```bash
mkdir -p db
cp /path/to/GarminDB/garmin.db db/garmin.db
```

3. **Generate the unified dataset**:
```bash
poetry run python -m garmin_analysis.data_ingestion.load_all_garmin_dbs
```

This creates `data/master_daily_summary.csv` combining all your Garmin data.

### Quick Verification

Check your data quality:
```bash
poetry run python -m garmin_analysis.features.quick_data_check --summary
```

### Next Steps

- **Launch the dashboard**: `poetry run python run_dashboard.py`
- **Run your first analysis**: See [Common Commands](#common-commands) below
- **Explore visualizations**: Check the [Visualization utilities](#visualization-utilities) section

## Common Commands

### CLI Shortcuts (after `poetry install`)

```bash
garmin init              # Bootstrap: check DBs, create folders, validate schema
garmin-export            # Export master to Parquet
garmin-export --duckdb   # Also export to DuckDB
garmin-sync --sync --all # Sync from Garmin Connect
garmin-weekly-report     # Generate weekly health report
garmin-day-of-week       # Day-of-week analysis
garmin-stress-by-time    # Time-of-day stress analysis
garmin-activity-calendar # Activity calendar visualization
```

### Garmin Connect Sync (NEW!)

**First-time setup:**
```bash
# Install GarminDB from source (one-time)
git clone --recursive https://github.com/tcgoetz/GarminDB.git ~/GarminDB
cd ~/GarminDB && make setup

# Configure your Garmin Connect credentials
cd ~/Code/garmin-analysis
poetry run python -m garmin_analysis.cli_garmin_sync --setup \
  --username your@email.com \
  --password yourpassword \
  --start-date 01/01/2024

# Download all historical data (do this once)
poetry run python -m garmin_analysis.cli_garmin_sync --sync --all
```

**Daily updates:**
```bash
# Download only the latest data (fast, run daily)
poetry run python -m garmin_analysis.cli_garmin_sync --sync --latest

# Then regenerate your unified dataset
poetry run python -m garmin_analysis.data_ingestion.load_all_garmin_dbs
```

**Equivalent to running GarminDB directly:**
```bash
# Our wrapper runs this for you:
garmindb_cli.py --all --download --import --analyze --latest
```

**Setup options:**
```bash
# Full setup command with all options
poetry run python -m garmin_analysis.cli_garmin_sync --setup \
  --username your@email.com \
  --password yourpassword \
  --start-date 01/01/2024 \
  --download-latest-activities 50 \
  --download-all-activities 2000
```

**Other operations:**
```bash
# Backup your databases
poetry run python -m garmin_analysis.cli_garmin_sync --backup

# View statistics about your data
poetry run python -m garmin_analysis.cli_garmin_sync --stats
```

**Copy databases to project:**
```bash
# Find GarminDB databases (located in ~/HealthData/DBs/)
poetry run python -m garmin_analysis.cli_garmin_sync --find-dbs

# Copy databases to project db/ directory
poetry run python -m garmin_analysis.cli_garmin_sync --copy-dbs
```

**Automation script:**
```bash
# Use the provided script for daily updates
./examples/daily_update.sh

# Or with dashboard restart
./examples/daily_update.sh --restart

# Add to cron for automatic daily updates (6 AM)
crontab -e
# Add: 0 6 * * * /path/to/garmin-analysis/examples/daily_update.sh >> ~/garmin-update.log 2>&1
```

**Note:** See `docs/garmin_connect_integration.md` for complete GarminDB integration guide and troubleshooting.

### Data Ingestion & Preparation
- **Generate unified dataset:**
```bash
poetry run python -m garmin_analysis.data_ingestion.load_all_garmin_dbs
```
Creates `data/master_daily_summary.csv` and `data/daily_data_quality.csv`.

- **Prepare modeling-ready dataset:**
```bash
poetry run python -m garmin_analysis.data_ingestion.prepare_modeling_dataset
```
Creates `data/modeling_ready_dataset.csv` with cleaned data for ML.

- **Schema inspection and drift detection:**
```bash
# Inspect database schemas
poetry run python -m garmin_analysis.data_ingestion.inspect_sqlite_schema db/garmin.db

# Inspect all databases in a directory
poetry run python -m garmin_analysis.data_ingestion.inspect_sqlite_schema --dir db

# Export expected schema
poetry run python -m garmin_analysis.data_ingestion.inspect_sqlite_schema export db/garmin.db reports/expected_schema.json

# Compare live DB vs expected schema
poetry run python -m garmin_analysis.data_ingestion.inspect_sqlite_schema compare db/garmin.db reports/expected_schema.json --fail-on-drift
```

## 🚀 Quick Start with 24-Hour Coverage Filtering

**NEW FEATURE!** All analysis tools now support filtering to only days with complete 24-hour continuous data coverage. This ensures more reliable analysis by excluding days with data gaps.

### Try it now:

```bash
# Generate plots with high-quality data only
poetry run python -m garmin_analysis.viz.plot_trends_range --filter-24h-coverage

# Run comprehensive modeling with filtered data
poetry run python -m garmin_analysis.modeling.comprehensive_modeling_pipeline --filter-24h-coverage

# Generate reports with 24h coverage filtering
poetry run python -m garmin_analysis.reporting.run_all_analytics --filter-24h-coverage

# Launch dashboard with filtering options
poetry run python -m garmin_analysis.dashboard.app
# Then check the "Only days with 24-hour continuous coverage" checkbox
# and set "Max gap (minutes)" to your preferred tolerance (default: 2)
```

### Why use 24-hour coverage filtering?

- **🎯 More Reliable Analysis**: Ensures data completeness for time-series analysis
- **🧠 Better Model Training**: Reduces noise from incomplete data days
- **📊 Consistent Comparisons**: Enables fair comparison across different time periods
- **⚡ Configurable**: Customize gap tolerance and edge tolerance parameters

### When to use 24-hour coverage filtering

**Recommended for:**
- 📈 **Time-series analysis** - Ensures continuous data streams
- 🤖 **Machine learning** - Reduces noise and improves model accuracy
- 📊 **Trend analysis** - Provides consistent data points for comparison
- 🔬 **Research studies** - Ensures data quality for scientific analysis
- 📋 **Reporting** - Generates cleaner, more reliable reports

**Optional for:**
- 🔍 **Exploratory analysis** - When you want to see all available data
- 📱 **Quick checks** - When data completeness is less critical
- 🎯 **Specific day analysis** - When analyzing particular events or days

### Launch dashboard
```bash
# Using the convenient script
poetry run python run_dashboard.py

# Or run directly
poetry run python -m garmin_analysis.dashboard.app
```
Open `http://localhost:8050`.

The dashboard now includes:
- **📅 Day of Week Analysis**: Sleep score, body battery, and water intake by day of week
- **📊 30-Day Health Overview**: Variable 30-day window for stress, HR, body battery, and sleep
- **📈 Data Quality**: Daily data quality score timeline, distribution, and coverage vs completeness
- **📊 24-Hour Coverage Analysis**: Watch wear time and coverage metrics
- **😰 Stress by Time of Day**: Hourly stress patterns
- **📈 Metric Trends**: Time series plots with filtering

### Visualization Tools

#### Trend Analysis
```bash
# Generate comprehensive trend plots for all metrics
poetry run python -m garmin_analysis.viz.plot_trends_range

# With 24-hour coverage filtering
poetry run python -m garmin_analysis.viz.plot_trends_range --filter-24h-coverage
```

#### Correlation Analysis
```bash
# Generate feature correlation heatmaps
poetry run python -m garmin_analysis.viz.plot_feature_correlation

# Plot individual feature trends
poetry run python -m garmin_analysis.viz.plot_feature_trend
```

#### Activity Calendar (NEW!)
```bash
# Generate calendar for all available data
poetry run python -m garmin_analysis.viz.cli_activity_calendar

# Last 6 months
poetry run python -m garmin_analysis.viz.cli_activity_calendar --months 6

# Specific date range
poetry run python -m garmin_analysis.viz.cli_activity_calendar --start-date 2024-01-01 --end-date 2024-12-31
```

#### Summary Statistics
```bash
# Generate summary statistics for all metrics
poetry run python -m garmin_analysis.features.summary_stats
```

## 📅 Day-of-Week Analysis

**NEW FEATURE!** Analyze your sleep score, body battery, and water intake patterns by day of the week to identify weekly trends and optimize your health routines.

### Quick Start

```bash
# Run day-of-week analysis with visualizations
poetry run python -m garmin_analysis.cli_day_of_week

# Run with verbose output
poetry run python -m garmin_analysis.cli_day_of_week --verbose

# Show plots interactively (instead of saving)
poetry run python -m garmin_analysis.cli_day_of_week --show-plots

# Skip saving plots to files
poetry run python -m garmin_analysis.cli_day_of_week --no-save

# Use 24-hour coverage filtering (optional)
poetry run python -m garmin_analysis.cli_day_of_week --filter-24h-coverage

# Customize filtering parameters
poetry run python -m garmin_analysis.cli_day_of_week --filter-24h-coverage \
  --max-gap 5 --day-edge-tolerance 5 --coverage-allowance-minutes 60
```

### Dashboard Integration

The day-of-week analysis is also available in the interactive dashboard:

```bash
# Launch the dashboard
poetry run python run_dashboard.py
# Or: poetry run python -m garmin_analysis.dashboard.app
```

Then navigate to the **"📅 Day of Week Analysis"** tab to:
- Select which metrics to analyze (Sleep Score, Body Battery Max/Min, Water Intake)
- Apply 24-hour coverage filtering for reliable results
- View interactive bar charts and trend comparisons
- Explore patterns in real-time

### Key Features

- **📊 Comprehensive Analysis**: Sleep score, body battery max/min, and water intake
- **📈 Multiple Visualizations**: Bar charts with error bars and trend line comparisons
- **🎯 Interactive Controls**: Select metrics and apply filters in real-time
- **📋 Automated Summaries**: Best/worst days with statistical differences
- **⚡ 24-Hour Coverage Filtering**: Optional filtering for high-quality data only
- **🎨 Color-Coded Metrics**: Easy identification of different health metrics

### Understanding Your Results

The analysis shows:

- **Sleep Score**: Average sleep quality by day of week (0-100 scale)
- **Body Battery Max**: Peak energy level by day of week (0-100 scale)
- **Body Battery Min**: Lowest energy level by day of week (0-100 scale)
- **Water Intake**: Daily hydration by day of week (ml)

### Example Output

```
DAY-OF-WEEK AVERAGES SUMMARY
============================================================

Sleep Score:
----------------------------------------
      Monday:   61.8 ±  18.2 (n=62)
     Tuesday:   62.1 ±  15.8 (n=58)
   Wednesday:   61.3 ±  15.7 (n=61)
    Thursday:   61.1 ±  17.8 (n=50)
      Friday:   59.5 ±  20.1 (n=53)
    Saturday:   60.9 ±  19.0 (n=61)
      Sunday:   60.4 ±  20.5 (n=57)

Best day:  Tuesday (62.1)
Worst day: Friday (59.5)
Difference: 2.6
```

### Use Cases

- **🛌 Sleep Optimization**: Identify which days you sleep best and adjust your routine
- **⚡ Energy Management**: Find patterns in your body battery to optimize activity timing
- **💧 Hydration Tracking**: Monitor water intake patterns (if tracked by your device)
- **📊 Weekly Planning**: Use insights to plan your week for optimal health
- **🔍 Pattern Recognition**: Spot trends that might not be obvious in daily data

### Generated Files

The analysis creates several visualization files in the `plots/` directory:
- `*_day_of_week_sleep_score.png` - Sleep score by day of week
- `*_day_of_week_body_battery_max.png` - Peak body battery by day of week
- `*_day_of_week_body_battery_min.png` - Minimum body battery by day of week
- `*_day_of_week_water_intake.png` - Water intake by day of week
- `*_day_of_week_combined.png` - All metrics comparison chart

### Data Requirements

- **Sleep Score**: Requires data in the `sleep` table with `score` column
- **Body Battery**: Requires data in the `daily_summary` table with `bb_max` and `bb_min` columns
- **Water Intake**: Requires data in the `daily_summary` table with `hydration_intake` column

### Testing

```bash
# Run day-of-week analysis tests
poetry run pytest tests/test_day_of_week_analysis.py -v

# Run dashboard integration tests
poetry run pytest tests/test_dashboard_integration.py -v
```

## 😰 Time-of-Day Stress Analysis

**NEW FEATURE!** Analyze your stress patterns throughout the day to identify peak stress times, low-stress periods, and patterns by day of week.

### Quick Start

```bash
# Run full stress analysis with all visualizations
poetry run python -m garmin_analysis.cli_time_of_day_stress

# Run with verbose output
poetry run python -m garmin_analysis.cli_time_of_day_stress --verbose

# Show plots interactively (instead of saving)
poetry run python -m garmin_analysis.cli_time_of_day_stress --show-plots

# Skip weekday analysis (faster for large datasets)
poetry run python -m garmin_analysis.cli_time_of_day_stress --no-weekday-analysis

# Use custom database path
poetry run python -m garmin_analysis.cli_time_of_day_stress --db-path /path/to/garmin.db
```

### Dashboard Integration

The stress analysis is also available in the interactive dashboard:

```bash
# Launch the dashboard
poetry run python run_dashboard.py
```

Then navigate to the **"😰 Stress by Time of Day"** tab to:
- View hourly stress patterns with confidence intervals
- See color-coded stress distribution by hour
- Explore interactive heatmaps showing stress by day of week and hour
- Toggle weekday breakdown on/off

### Key Features

- **📈 Hourly Patterns**: Average stress levels for each hour of the day (0-23)
- **📊 Interactive Visualizations**: Line charts, bar charts, and heatmaps
- **🗓️ Day-of-Week Breakdown**: See how stress patterns vary across the week
- **📉 Confidence Intervals**: Statistical confidence bands (95% CI) on line charts
- **🎨 Color-Coded Insights**: Green (low), orange (medium), red (high stress)
- **🕐 Time Period Analysis**: Automatic grouping into morning, afternoon, evening, night

### Understanding Your Results

The analysis provides:

- **Hourly Averages**: Mean stress level for each hour with standard deviation
- **Peak Stress Times**: The 5 hours with highest average stress
- **Low Stress Times**: The 5 hours with lowest average stress
- **Time Period Breakdown**: 
  - Morning (06:00-11:59)
  - Afternoon (12:00-17:59)
  - Evening (18:00-22:59)
  - Night (23:00-05:59)
- **Weekday Patterns**: How stress varies by day of week at different times

### Example Output

```
======================================================================
STRESS ANALYSIS BY TIME OF DAY
======================================================================

📊 Overall Stress Statistics:
----------------------------------------------------------------------
  Total measurements: 1,003,864
  Overall mean stress: 42.3
  Overall std dev: 18.7

⬆️  Peak Stress Times:
----------------------------------------------------------------------
  14:00 - 15:00:  52.3 ± 17.2 (n=42,156)
  15:00 - 16:00:  51.8 ± 17.5 (n=42,089)
  13:00 - 14:00:  51.2 ± 17.1 (n=42,201)
  16:00 - 17:00:  50.9 ± 17.8 (n=41,987)
  12:00 - 13:00:  50.1 ± 17.3 (n=42,034)

⬇️  Low Stress Times:
----------------------------------------------------------------------
  03:00 - 04:00:  28.5 ± 12.3 (n=41,234)
  04:00 - 05:00:  28.9 ± 12.5 (n=41,156)
  02:00 - 03:00:  29.2 ± 12.7 (n=41,298)
  05:00 - 06:00:  30.1 ± 13.1 (n=41,087)
  01:00 - 02:00:  30.8 ± 13.4 (n=41,267)

🕐 Time Period Analysis:
----------------------------------------------------------------------
  Morning (06:00-11:59):   38.2 ± 15.4
  Afternoon (12:00-17:59): 51.5 ± 17.6
  Evening (18:00-22:59):   45.3 ± 16.8
  Night (23:00-05:59):     29.7 ± 12.9
```

### Generated Files

The analysis creates visualization files in the `plots/` directory:
- `*_stress_by_hour.png` - Hourly stress with confidence interval
- `*_stress_by_hour_bars.png` - Color-coded bar chart by hour
- `*_stress_heatmap_weekday_hour.png` - Heatmap of stress by day/hour
- `*_stress_by_weekday_hour.png` - Line chart comparison by day of week

### Use Cases

- **⏰ Schedule Optimization**: Plan important tasks during your low-stress periods
- **🧘 Stress Management**: Identify when to take breaks or practice relaxation
- **💤 Sleep Insights**: See how nighttime stress affects your rest
- **📊 Weekly Planning**: Find which days/times are most stressful
- **🔍 Pattern Discovery**: Uncover stress triggers you weren't aware of
- **⚖️ Work-Life Balance**: Compare weekday vs weekend stress patterns

### Data Requirements

- **Stress Data**: Requires minute-by-minute stress measurements in the `stress` table of `garmin.db`
- **Continuous Monitoring**: Best results with devices that track stress 24/7
- **Data Volume**: Analysis works with any amount of data but more data provides better insights

### Testing

```bash
# Run time-of-day stress analysis tests
poetry run pytest tests/test_time_of_day_stress_analysis.py -v

# Run integration test with real database
poetry run pytest tests/test_time_of_day_stress_analysis.py::test_real_database_integration -v
```

## 📅 Activity Calendar & Type Mappings

**NEW FEATURES!** Visualize your activity patterns with a beautiful calendar view and customize how unknown activity types are displayed.

### Activity Calendar

Create calendar-style visualizations showing your daily activities with different colors for each activity type:

```bash
# Create calendar for all available data
poetry run python -m garmin_analysis.viz.cli_activity_calendar

# Create calendar for last 6 months
poetry run python -m garmin_analysis.viz.cli_activity_calendar --months 6

# Create calendar for specific date range
poetry run python -m garmin_analysis.viz.cli_activity_calendar --start-date 2024-01-01 --end-date 2024-12-31

# Create calendar with custom figure size
poetry run python -m garmin_analysis.viz.cli_activity_calendar --figsize 20 15

# Create calendar without activity type mappings (raw names)
poetry run python -m garmin_analysis.viz.cli_activity_calendar --no-mappings
```

### Activity Type Mappings

Customize how unknown or poorly named activity types are displayed:

```bash
# Check for unmapped activity types
poetry run python -m garmin_analysis.viz.cli_activity_calendar --suggest-mappings

# Use custom mappings configuration file
poetry run python -m garmin_analysis.viz.cli_activity_calendar --mappings-config my_mappings.json
```

### Key Features

- **🎨 Color-coded activities**: Each activity type gets a distinct color
- **📅 Calendar grid layout**: Shows days in a proper weekly calendar format
- **🔍 Multiple activities handling**: Darker colors for days with multiple activities
- **📊 Activity statistics**: Summary of activity patterns and frequencies
- **🏷️ Custom mappings**: Map unknown activity types to meaningful names
- **⚙️ Configurable**: Customize colors, date ranges, and display options

### Activity Type Mapping System

The system automatically maps unknown activity types to more meaningful names. For example:
- `UnknownEnumValue_67` → `"Training Assessment"` (automatic fitness assessments)
- `generic` → `"General Activity"` (unspecified activities)

#### Managing Mappings

Edit `config/activity_type_mappings.json` to customize mappings:

```json
{
  "unknown_activity_mappings": {
    "UnknownEnumValue_67": {
      "display_name": "Training Assessment",
      "description": "Automatic fitness assessments and recovery measurements",
      "category": "assessment",
      "color": "#9B59B6"
    }
  }
}
```

#### Adding New Mappings Programmatically

```python
from garmin_analysis.utils.activity_mappings import add_activity_mapping

add_activity_mapping(
    activity_type="UnknownEnumValue_68",
    display_name="Recovery Check",
    description="Automatic recovery measurements",
    category="assessment",
    color="#3498DB"
)
```

### Example Output

The activity calendar generates:
- **Calendar grid** with days colored by activity type
- **Legend** showing all activity types with their colors
- **Summary statistics** in logs showing activity frequency
- **High-resolution PNG** saved to the `plots/` directory

### Use Cases

- **🏃‍♂️ Activity Pattern Analysis**: See when you're most active throughout the year
- **🎯 Goal Tracking**: Visualize consistency in your workout routines
- **📊 Trend Identification**: Spot seasonal patterns in your activities
- **🔍 Data Quality**: Identify gaps in your activity data
- **📈 Progress Monitoring**: Track improvement in activity consistency

### Machine Learning & Modeling

#### HR & Activity → Sleep Analysis (NEW!)

Analyze how heart rate metrics and physical activities affect sleep quality:

```bash
# Run the sleep analysis model
poetry run python -m garmin_analysis.modeling.hr_activity_sleep_model
```

**Programmatic usage:**
```python
from garmin_analysis.modeling.hr_activity_sleep_model import HRActivitySleepModel

model = HRActivitySleepModel()
results = model.run_analysis(
    use_lag_features=True,          # Include yesterday's metrics
    imputation_strategy='median'     # Robust to outliers (recommended)
)

# Results include:
# - Best model and performance metrics
# - Top features affecting sleep
# - Visualizations (4 plots)
# - Detailed text report
```

#### Comprehensive Modeling Pipeline (Recommended)

Run all modeling analyses in one command:

```bash
# Run full pipeline
poetry run python -m garmin_analysis.modeling.comprehensive_modeling_pipeline

# With 24-hour coverage filtering
poetry run python -m garmin_analysis.modeling.comprehensive_modeling_pipeline --filter-24h-coverage
```

#### Individual Modeling Modules

All modules support flexible imputation strategies:

```bash
# Enhanced anomaly detection
poetry run python -m garmin_analysis.modeling.enhanced_anomaly_detection

# Advanced clustering analysis
poetry run python -m garmin_analysis.modeling.enhanced_clustering

# Predictive modeling
poetry run python -m garmin_analysis.modeling.predictive_modeling

# Activity-sleep-stress correlation
poetry run python -m garmin_analysis.modeling.activity_sleep_stress_analysis

# Basic clustering
poetry run python -m garmin_analysis.modeling.clustering_behavior

# Basic anomaly detection
poetry run python -m garmin_analysis.modeling.anomaly_detection
```

#### Imputation Strategies

All modeling modules support 6 imputation strategies for handling missing data:

```python
# Median imputation (default, robust to outliers - RECOMMENDED)
predictor.prepare_features(df, imputation_strategy='median')

# Mean imputation
predictor.prepare_features(df, imputation_strategy='mean')

# Drop rows with missing values
predictor.prepare_features(df, imputation_strategy='drop')

# Forward fill
predictor.prepare_features(df, imputation_strategy='forward_fill')

# Backward fill
predictor.prepare_features(df, imputation_strategy='backward_fill')

# No imputation
predictor.prepare_features(df, imputation_strategy='none')
```

See `docs/imputation_strategies.md` for detailed guidance.

### Reporting & Analytics

#### Weekly Health Report (NEW!)
Generate automated weekly Markdown reports tracking sleep score, resting HR, and stress minutes:
```bash
# Default: last 4 weeks
poetry run python -m garmin_analysis.cli_weekly_report

# Last 8 weeks with 24h coverage filtering
poetry run python -m garmin_analysis.cli_weekly_report --weeks 8 --filter-24h-coverage

# Custom output directory
poetry run python -m garmin_analysis.cli_weekly_report --output-dir reports/weekly
```

#### Comprehensive Analytics Report
Run all analytics and generate comprehensive reports:
```bash
# Full analytics report
poetry run python -m garmin_analysis.reporting.run_all_analytics

# With 24-hour coverage filtering
poetry run python -m garmin_analysis.reporting.run_all_analytics --filter-24h-coverage
```

#### Trend Summary Report
Generate statistical trend summaries:
```bash
# Generate trend summary
poetry run python -m garmin_analysis.reporting.generate_trend_summary

# With 24-hour coverage filtering
poetry run python -m garmin_analysis.reporting.generate_trend_summary --filter-24h-coverage
```

Reports are saved to the `reports/` directory.

### Data quality tools
- **Quick check (summary, completeness, feature suitability):**
```bash
poetry run python -m garmin_analysis.features.quick_data_check            # full quick check
poetry run python -m garmin_analysis.features.quick_data_check --summary
poetry run python -m garmin_analysis.features.quick_data_check --completeness
poetry run python -m garmin_analysis.features.quick_data_check --features
poetry run python -m garmin_analysis.features.quick_data_check --continuous-24h
```

- **Comprehensive audit with reports (JSON + Markdown in `data_quality_reports/`):**
```bash
poetry run python -m garmin_analysis.features.data_quality_analysis
```

- **Additional data quality tools:**
```bash
# Check for missing data patterns
poetry run python -m garmin_analysis.features.check_missing_data

# Generate comprehensive coverage analysis
poetry run python -m garmin_analysis.features.coverage
```

## Use Cases

This platform is designed for comprehensive Garmin health data analysis:

### 🏃‍♂️ **Fitness Enthusiasts**
- Track daily activity trends and patterns
- Identify optimal workout timing and intensity
- Monitor sleep quality and recovery metrics
- Analyze stress levels and their impact on performance

### 🔬 **Health Researchers**
- Conduct longitudinal health studies
- Analyze correlations between different health metrics
- Detect anomalies in health patterns
- Generate comprehensive health reports

### 📊 **Data Scientists**
- Apply machine learning to health data
- Build predictive models for health outcomes
- Perform clustering analysis to identify health patterns
- Create custom visualizations and reports

### 🏥 **Healthcare Professionals**
- Monitor patient health trends over time
- Identify potential health issues through anomaly detection
- Generate patient health summaries
- Track treatment effectiveness

## Key Capabilities

- **🌙 Sleep Quality Analysis**: **NEW!** Analyze how HR and activities affect sleep with ML models
- **🔧 Flexible Data Imputation**: **NEW!** 6 strategies for handling missing values (prevents data loss)
- **📈 Time Series Analysis**: Comprehensive trend analysis with configurable time windows
- **🤖 Machine Learning**: Multiple algorithms for anomaly detection, clustering, and prediction
- **📊 Interactive Visualization**: Real-time dashboard with filtering capabilities
- **📅 Activity Calendar**: Calendar-style visualization of activity patterns with color coding
- **🏷️ Activity Type Mapping**: Customize display names and colors for unknown activity types
- **🔍 Data Quality Assurance**: Advanced tools for data validation and quality assessment
- **📋 Automated Reporting**: Generate comprehensive health reports automatically
- **⚡ Performance Optimization**: 24-hour coverage filtering for faster, more reliable analysis
- **🧪 Comprehensive Testing**: 970+ tests with full coverage (unit and integration)
- **📓 Interactive Analysis**: Jupyter notebooks for exploratory data analysis

## 24-Hour Coverage Filtering

Many analysis tools now support filtering to only days with complete 24-hour continuous data coverage. This is useful for:

- **More reliable analysis**: Ensures data completeness for time-series analysis
- **Better model training**: Reduces noise from incomplete data days
- **Consistent comparisons**: Enables fair comparison across different time periods

### How it works

The system analyzes the stress timeseries data to identify days where:
- Data coverage starts within 2 minutes of midnight
- Data coverage ends within 2 minutes of midnight  
- No gap between consecutive samples exceeds 2 minutes

### Available Tools with 24h Coverage Filtering

| Tool | Command | Description |
|------|---------|-------------|
| **Plot Generation** | `plot_trends_range --filter-24h-coverage` | Generate trend plots with filtered data |
| **Trend Summary** | `generate_trend_summary --filter-24h-coverage` | Create summary reports with filtered data |
| **Full Analytics** | `run_all_analytics --filter-24h-coverage` | Run comprehensive analytics with filtering |
| **Modeling Pipeline** | `comprehensive_modeling_pipeline --filter-24h-coverage` | Complete ML pipeline with filtered data |
| **Interactive Dashboard** | Checkbox in UI | Real-time filtering in web interface |

### Complete Tool Reference

| Category | Tool | Command | Description |
|----------|------|---------|-------------|
| **Data Ingestion** | Load All DBs | `load_all_garmin_dbs` | Merge all Garmin databases into unified dataset |
| | Prepare Dataset | `prepare_modeling_dataset` | Clean data for machine learning |
| | Schema Inspector | `inspect_sqlite_schema` | Inspect and validate database schemas |
| **Visualization** | Trend Plots | `plot_trends_range` | Generate comprehensive trend visualizations |
| | Correlation Matrix | `plot_feature_correlation` | Create feature correlation heatmaps |
| | Feature Trends | `plot_feature_trend` | Plot individual feature trends over time |
| | Activity Calendar | `cli_activity_calendar` | Create calendar view of activity patterns |
| | Day-of-Week Analysis | `cli_day_of_week` | **NEW!** Analyze sleep, body battery, water intake by day of week |
| | Summary Stats | `summary_stats` | Generate statistical summaries |
| **Modeling** | Full Pipeline | `comprehensive_modeling_pipeline` | Complete ML analysis pipeline |
| | Anomaly Detection | `enhanced_anomaly_detection` | Advanced anomaly detection algorithms |
| | Clustering | `enhanced_clustering` | Multiple clustering algorithms |
| | Predictive Modeling | `predictive_modeling` | Health outcome prediction models |
| | Activity Analysis | `activity_sleep_stress_analysis` | Correlation analysis between metrics |
| **Bootstrap** | Init | `garmin init` or `cli_init` | Check DBs, create folders, validate schema |
| **Export** | Parquet/DuckDB | `cli_export` | Export master to Parquet (and optionally DuckDB) |
| **Data Quality** | Quick Check | `quick_data_check` | Fast data quality assessment |
| | Daily Score | (auto in `load_all_garmin_dbs`) | Daily data quality score → CSV + dashboard |
| | Comprehensive Audit | `data_quality_analysis` | Detailed data quality reports |
| | Missing Data | `check_missing_data` | Analyze missing data patterns |
| | Coverage Analysis | `coverage` | 24-hour coverage assessment |
| **Reporting** | Weekly Report | `cli_weekly_report` | **NEW!** Weekly sleep, HR, stress Markdown report |
| | Full Analytics | `run_all_analytics` | Comprehensive analytics reports |
| | Trend Summary | `generate_trend_summary` | Statistical trend summaries |
| **Dashboard** | Web Interface | `dashboard.app` | Interactive web dashboard with day-of-week analysis |
| **Testing** | Unit Tests | `pytest -m "not integration"` | Fast unit tests |
| | Integration Tests | `pytest -m integration` | Full integration tests |
| | All Tests | `pytest` | Complete test suite |

### Usage Examples

```bash
# Basic usage - filter to high-quality data only
poetry run python -m garmin_analysis.viz.plot_trends_range --filter-24h-coverage
poetry run python -m garmin_analysis.reporting.generate_trend_summary --filter-24h-coverage

# Advanced usage - customize coverage parameters
poetry run python -m garmin_analysis.viz.plot_trends_range --filter-24h-coverage --max-gap 5 --day-edge-tolerance 5 --coverage-allowance-minutes 60

# Full pipeline with filtering
poetry run python -m garmin_analysis.modeling.comprehensive_modeling_pipeline --filter-24h-coverage --target-col score

# Monthly reports with filtering
poetry run python -m garmin_analysis.reporting.run_all_analytics --filter-24h-coverage --monthly --coverage-allowance-minutes 120
```

### Dashboard Usage

In the interactive dashboard, you can toggle the "Only days with 24-hour continuous coverage" checkbox to filter trend plots and analysis views. Use the adjacent "Max gap (minutes)" input to set the maximum allowed gap between samples (default 2). The filtering is applied in real-time and plot titles will indicate when filtering is active.

### Configuration Parameters

- `--filter-24h-coverage`: Enable 24-hour coverage filtering
- `--max-gap`: Maximum allowed gap between consecutive samples (default: 2 minutes)
- `--day-edge-tolerance`: Allowed tolerance at day start/end (default: 2 minutes)
- `--coverage-allowance-minutes`: Total allowed missing minutes within a day (0–300, default: 0). This allowance applies to the sum of: (a) all internal gaps that exceed `--max-gap` and (b) late starts/early ends beyond `--day-edge-tolerance` at the day's edges. If the cumulative missing time is within the allowance, the day qualifies even if individual gaps exceed `--max-gap`.

Dashboard-specific:
- "Max gap (minutes)": Same as `--max-gap`, adjustable per-tab in the UI
- "Coverage allowance (minutes)": Same as `--coverage-allowance-minutes`

### Data Quality Check

Check which days have 24-hour coverage:
```bash
poetry run python -m garmin_analysis.features.quick_data_check --continuous-24h
```

### Daily Data Quality Score (NEW!)

A composite daily data quality score (0–100) combining 24h coverage and metric completeness. Persisted to `data/daily_data_quality.csv` and shown in the dashboard **Data Quality** tab.

- **Computed automatically** when you run `load_all_garmin_dbs`
- **Dashboard tab**: 📈 Data Quality — timeline, distribution, coverage vs completeness scatter
- **CSV columns**: `day`, `data_quality_score`, `coverage_score`, `completeness_score`, `key_metrics_count`, `key_metrics_total`

### Export to Parquet & DuckDB (NEW!)

Export the master dataset for faster analytics and downstream use:

```bash
# Export to Parquet (includes daily data quality)
poetry run python -m garmin_analysis.cli_export

# Also export to DuckDB (requires: pip install duckdb or poetry add duckdb)
poetry run python -m garmin_analysis.cli_export --duckdb
```

Outputs:
- `data/export/master_daily_summary.parquet` — columnar format for pandas, DuckDB, Spark
- `data/export/master.duckdb` — SQL database (optional)

### Performance Benefits

Using 24-hour coverage filtering can also improve performance:

- **⚡ Faster Processing**: Fewer data points mean faster analysis
- **💾 Lower Memory Usage**: Reduced dataset size for large-scale analysis
- **🎯 Focused Results**: More relevant insights from high-quality data
- **📈 Better Visualizations**: Cleaner plots without gaps or missing data artifacts

### Example: Before vs After Filtering

```bash
# Check how many days you have total
poetry run python -m garmin_analysis.features.quick_data_check --summary

# Check how many days have 24h coverage
poetry run python -m garmin_analysis.features.quick_data_check --continuous-24h

# Compare results with and without filtering
poetry run python -m garmin_analysis.viz.plot_trends_range  # All data
poetry run python -m garmin_analysis.viz.plot_trends_range --filter-24h-coverage  # Filtered data
```

### Troubleshooting 24-Hour Coverage Filtering

**No qualifying days found?**
- Check if you have stress data: `poetry run python -m garmin_analysis.features.quick_data_check --continuous-24h`
- Try relaxing the parameters: `--max-gap 10 --day-edge-tolerance 10`
- Ensure your Garmin device was worn continuously during the day

**Filtering too strict?**
- Increase gap tolerance: `--max-gap 5` (default: 2 minutes)
- Increase edge tolerance: `--day-edge-tolerance 5` (default: 2 minutes)

**Want to see what's being filtered?**
- Run without filtering first to see all data
- Use `--continuous-24h` to see which specific days qualify
- Compare results side-by-side

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

## Testing

The project has a comprehensive test suite with **970+ tests** across 30+ test modules covering unit and integration scenarios.

### Run All Tests
```bash
# Run all tests (recommended for CI/CD)
poetry run pytest

# Run with verbose output
poetry run pytest -v

# Run with quiet mode
poetry run pytest -q
```

### Run Test Categories

**Unit Tests** (fast, in-memory DB fixtures):
```bash
poetry run pytest -m "not integration"
```

**Integration Tests** (file-backed temp DBs, tests real I/O):
```bash
poetry run pytest -m integration
```

### Run Specific Test Modules

```bash
# Weekly report tests (129 tests)
poetry run pytest tests/test_generate_weekly_report.py -v

# Coverage filtering tests
poetry run pytest tests/test_coverage_filtering.py -v

# Data quality tests
poetry run pytest tests/test_data_quality.py -v

# Dashboard tests
poetry run pytest tests/test_dashboard_dependencies.py -v
poetry run pytest tests/test_dashboard_integration.py -v

# Modeling tests
poetry run pytest tests/test_hr_activity_sleep_model.py -v
poetry run pytest tests/test_imputation.py -v

# Day-of-week analysis tests
poetry run pytest tests/test_day_of_week_analysis.py -v
```

### Test Coverage

- **970+ total tests** across 30+ test modules
- **42 tests** for HR & Activity Sleep Model
- **32 tests** for imputation strategies
- Full coverage of unit and integration scenarios
- Tests use in-memory SQLite for speed and file-backed DBs for integration testing

### Jupyter Notebooks
Interactive analysis notebooks are available in the `notebooks/` directory:
- `analysis.ipynb` - Comprehensive data analysis
- `hr_daily.ipynb` - Heart rate daily analysis

To use notebooks:
```bash
# Start Jupyter Lab
poetry run jupyter lab

# Or start Jupyter Notebook
poetry run jupyter notebook
```

## Code Organization

### Package Structure
```
src/garmin_analysis/
├── dashboard/          # Interactive Dash web application
├── data_ingestion/     # Database loading, export, CSV generation
├── features/           # Data quality and feature engineering
├── modeling/           # Machine learning models
├── reporting/          # Automated report generation
├── viz/                # Visualization tools
├── utils/              # Utility modules
│   ├── data_loading.py     # Load data from DB/CSV
│   ├── data_processing.py  # Transform and clean data
│   ├── data_filtering.py   # Date filters and feature prep
│   ├── cleaning.py        # Data cleaning (placeholders, outliers)
│   ├── imputation.py      # Missing value strategies
│   └── activity_mappings.py # Activity type customization
├── cli_*.py            # CLI entry points (init, export, sync, weekly, etc.)
└── ...
```
Configuration: `config/activity_type_mappings.json` at project root.

### Utility Modules

**When to use each**:

| Module | Use For | Example |
|--------|---------|---------|
| `utils.data_loading` | Loading master dataframe, Garmin tables | `load_master_dataframe()` |
| `utils.data_processing` | Date normalization, time conversions | `normalize_day_column()` |
| `utils.data_filtering` | Date ranges, feature filtering | `filter_by_date()` |
| `utils.imputation` | Handling missing values | `impute_missing_values()` |
| `utils.cleaning` | Data cleaning (placeholders, outliers) | `clean_data()` |

### Configuration Files
- `config/activity_type_mappings.json` - Customize activity names and colors
- `logging_config.py` - Centralized logging setup

## Project Structure

```
garmin-analysis/
├── src/garmin_analysis/           # Main package
│   ├── dashboard/                 # Interactive web dashboard
│   ├── data_ingestion/           # Data loading and preparation
│   ├── features/                 # Data quality and feature analysis
│   ├── modeling/                 # Machine learning algorithms
│   │   ├── hr_activity_sleep_model.py  # NEW! HR & Activity → Sleep analysis
│   │   ├── predictive_modeling.py      # General predictive models (with imputation)
│   │   ├── enhanced_clustering.py      # Clustering algorithms (with imputation)
│   │   └── enhanced_anomaly_detection.py  # Anomaly detection (with imputation)
│   ├── reporting/                # Automated report generation (incl. weekly health report)
│   │   └── generate_weekly_report.py  # NEW! Weekly sleep, HR, stress report
│   ├── utils/                    # Utility modules
│   │   ├── data_loading.py       # Database and file loading
│   │   ├── data_processing.py    # Data transformation
│   │   ├── data_filtering.py     # Filtering and feature preparation
│   │   ├── cleaning.py          # Data cleaning (placeholders, outliers, column names)
│   │   ├── imputation.py         # Missing value handling strategies
│   │   └── activity_mappings.py  # Activity type customization
│   ├── viz/                      # Visualization tools
│   ├── cli_*.py                  # CLI entry points (init, export, sync, weekly, etc.)
│   └── ...
├── config/                       # Configuration files
│   └── activity_type_mappings.json # Activity type mappings
├── docs/                         # Documentation
│   ├── activity_type_mappings.md # Activity mapping documentation
│   ├── garmin_connect_integration.md # GarminDB setup and troubleshooting
│   └── imputation_strategies.md  # Imputation guide
├── examples/                     # Example scripts
│   └── activity_calendar_example.py # Activity calendar example
├── run_dashboard.py              # Convenient dashboard launcher script
├── tests/                        # Comprehensive test suite
│   ├── test_generate_weekly_report.py   # NEW! Weekly report tests (129 tests)
│   ├── test_imputation.py              # Imputation utility tests (32 tests)
│   ├── test_hr_activity_sleep_model.py  # Sleep model tests (42 tests)
│   └── ...                             # Other test files (30+ modules)
├── notebooks/                    # Jupyter notebooks
├── data/                         # Generated datasets
│   ├── master_daily_summary.csv  # Unified daily data
│   ├── daily_data_quality.csv    # Daily data quality scores
│   └── export/                   # Parquet and DuckDB exports
├── plots/                        # Generated plots
├── reports/                      # Generated reports
├── modeling_results/             # ML model outputs
│   ├── plots/                    # Model visualizations
│   └── reports/                  # Model analysis reports
└── db/                          # Garmin database files
```

## Dependencies

Dependencies are managed via Poetry in `pyproject.toml`.

**Installation**:
```bash
poetry install
```

**For non-Poetry users**, generate requirements.txt:
```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
pip install -r requirements.txt
```

**Core Libraries**:
- **Data**: pandas, numpy, pyarrow (Parquet export)
- **ML**: scikit-learn, tsfresh, statsmodels, prophet  
- **Visualization**: matplotlib, seaborn, plotly, dash
- **Garmin Integration**: [GarminDB](https://github.com/tcgoetz/GarminDB) - For Garmin Connect data export (see [Credits](#credits--acknowledgments))
- **Development**: pytest, jupyter

See `pyproject.toml` for version constraints.

### Test Fixtures
- `mem_db` (unit): In-memory SQLite with minimal schema and seed data for pure SQL/transform functions.
- `tmp_db` (integration): Temp file-backed SQLite DBs with realistic seeds; test code patches `garmin_analysis.data_ingestion.load_all_garmin_dbs.DB_PATHS` to point to these files.

Notes on data sources:
- If real Garmin DBs are available, place them under `db/` (e.g., `db/garmin.db`, `db/garmin_summary.db`, `db/garmin_activities.db`).
- When DBs are missing outside of tests, some commands may generate a synthetic dataset for convenience and log clear WARNINGS. This synthetic data is only for smoke testing and should not be used for real analysis.

## Contributing

This project uses:
- **Python 3.11+**: Required for compatibility
- **Poetry**: Dependency management
- **pytest**: Testing framework with 970+ tests
- **Ruff**: Linting and formatting

## License

See the [LICENSE](LICENSE) file for details.

## Licensing

This project is licensed under the MIT License. For clarity on dependencies and outputs:

- **GarminDB**: GarminDB is an optional, external tool licensed under GPL-2.0. This repository does not vendor or bundle GarminDB; users install it separately (e.g., from source) when they want automated Garmin Connect sync. Our integration invokes GarminDB as a subprocess; we do not link or distribute GarminDB code.
- **Outputs**: All generated reports, plots, datasets, and analytics outputs are derived from the user's own data. They are not derivative works of GarminDB or any other dependency.

## Credits & Acknowledgments

This project builds upon and integrates with several excellent open-source projects:

- **[GarminDB](https://github.com/tcgoetz/GarminDB)** by Tom Goetz - Provides the core functionality for downloading and parsing Garmin Connect data. Licensed under GPL-2.0.
- **[Python Garmin Connect API](https://github.com/cyberjunky/python-garmin-connect)** - Used by GarminDB for Garmin Connect authentication.
- **scikit-learn**, **pandas**, **matplotlib** and other open-source libraries that make this analysis possible.

Special thanks to the Garmin developer community for their work on reverse-engineering and documenting Garmin's data formats.

## Notes

- This repo uses the `src/garmin_analysis` package layout. Always run modules via `python -m garmin_analysis.<module>`.
- Logging is used instead of print statements throughout the codebase.
- Test fixtures use in-memory SQLite (`mem_db`) for unit tests and file-backed DBs (`tmp_db`) for integration tests.
- If real Garmin databases are unavailable, some commands may generate synthetic data for smoke testing (with warnings).
