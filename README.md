# Garmin Analysis

A comprehensive Garmin health data analysis platform with interactive dashboard, machine learning capabilities, automated reporting, and **24-hour coverage filtering** for high-quality data analysis.

## 🆕 What's New

**24-Hour Coverage Filtering** is now available across all analysis tools! This major feature enhancement allows you to filter your analysis to only include days with complete 24-hour continuous data coverage, ensuring more reliable and accurate results.

- ✅ **All visualization tools** now support `--filter-24h-coverage`
- ✅ **Interactive dashboard** has real-time filtering checkboxes
- ✅ **Modeling pipeline** can train on high-quality data only
- ✅ **Reporting tools** generate cleaner, more reliable reports
- ✅ **Configurable parameters** for gap tolerance and edge tolerance

## Features

- **⏰ 24-Hour Coverage Filtering**: **NEW!** Filter analysis to only days with complete 24-hour continuous data coverage for more reliable results
- **📅 Activity Calendar**: **NEW!** Visualize activity patterns with color-coded calendar showing different activity types
- **🏷️ Activity Type Mappings**: **NEW!** Customize display names and colors for unknown or poorly named activity types
- **📊 Interactive Dashboard**: Real-time metric trends and correlation analysis with filtering options
- **🤖 Machine Learning**: Comprehensive ML pipeline with anomaly detection, clustering, and predictive modeling
- **📈 Visualization**: Multiple plotting tools for trends, correlations, and feature analysis
- **📋 Reporting**: Automated summaries and comprehensive analytics reports
- **🔍 Data Quality**: Advanced data quality analysis and coverage assessment tools
- **🗄️ Data Ingestion**: Unified data loading from multiple Garmin databases with schema validation
- **🧪 Testing**: Comprehensive test suite with unit and integration tests
- **📓 Notebooks**: Interactive Jupyter notebooks for exploratory analysis

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

### Data Ingestion & Preparation
- **Generate unified dataset:**
```bash
poetry run python -m garmin_analysis.data_ingestion.load_all_garmin_dbs
```
Creates `data/master_daily_summary.csv`.

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
poetry run python -m garmin_analysis.dashboard.app
```
Open `http://localhost:8050`.

### Visualization utilities
```bash
# Generate comprehensive trend plots
poetry run python -m garmin_analysis.viz.plot_trends_range
poetry run python -m garmin_analysis.viz.plot_trends_range --filter-24h-coverage

# Generate feature correlation heatmaps
poetry run python -m garmin_analysis.viz.plot_feature_correlation

# Generate individual feature trend plots
poetry run python -m garmin_analysis.viz.plot_feature_trend

# Generate activity calendar (NEW!)
poetry run python -m garmin_analysis.viz.cli_activity_calendar
poetry run python -m garmin_analysis.viz.cli_activity_calendar --months 6
poetry run python -m garmin_analysis.viz.cli_activity_calendar --start-date 2024-01-01 --end-date 2024-12-31

# Generate summary statistics
poetry run python -m garmin_analysis.features.summary_stats
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

### Modeling
- **Full pipeline (recommended):**
```bash
poetry run python -m garmin_analysis.modeling.comprehensive_modeling_pipeline
poetry run python -m garmin_analysis.modeling.comprehensive_modeling_pipeline --filter-24h-coverage
```

- **Individual modules:**
```bash
# Enhanced anomaly detection with multiple algorithms
poetry run python -m garmin_analysis.modeling.enhanced_anomaly_detection

# Advanced clustering analysis (K-means, DBSCAN, Gaussian Mixture, etc.)
poetry run python -m garmin_analysis.modeling.enhanced_clustering

# Predictive modeling (Random Forest, Gradient Boosting, Neural Networks, etc.)
poetry run python -m garmin_analysis.modeling.predictive_modeling

# Activity-sleep-stress correlation analysis
poetry run python -m garmin_analysis.modeling.activity_sleep_stress_analysis

# Basic clustering behavior analysis
poetry run python -m garmin_analysis.modeling.clustering_behavior

# Basic anomaly detection
poetry run python -m garmin_analysis.modeling.anomaly_detection
```

### Reporting
```bash
poetry run python -m garmin_analysis.reporting.run_all_analytics
poetry run python -m garmin_analysis.reporting.run_all_analytics --filter-24h-coverage
poetry run python -m garmin_analysis.reporting.generate_trend_summary
poetry run python -m garmin_analysis.reporting.generate_trend_summary --filter-24h-coverage
```

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

- **📈 Time Series Analysis**: Comprehensive trend analysis with configurable time windows
- **🤖 Machine Learning**: Multiple algorithms for anomaly detection, clustering, and prediction
- **📊 Interactive Visualization**: Real-time dashboard with filtering capabilities
- **📅 Activity Calendar**: Calendar-style visualization of activity patterns with color coding
- **🏷️ Activity Type Mapping**: Customize display names and colors for unknown activity types
- **🔍 Data Quality Assurance**: Advanced tools for data validation and quality assessment
- **📋 Automated Reporting**: Generate comprehensive health reports automatically
- **⚡ Performance Optimization**: 24-hour coverage filtering for faster, more reliable analysis
- **🧪 Comprehensive Testing**: Full test coverage with unit and integration tests
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
| | Activity Calendar | `cli_activity_calendar` | **NEW!** Create calendar view of activity patterns |
| | Summary Stats | `summary_stats` | Generate statistical summaries |
| **Modeling** | Full Pipeline | `comprehensive_modeling_pipeline` | Complete ML analysis pipeline |
| | Anomaly Detection | `enhanced_anomaly_detection` | Advanced anomaly detection algorithms |
| | Clustering | `enhanced_clustering` | Multiple clustering algorithms |
| | Predictive Modeling | `predictive_modeling` | Health outcome prediction models |
| | Activity Analysis | `activity_sleep_stress_analysis` | Correlation analysis between metrics |
| **Data Quality** | Quick Check | `quick_data_check` | Fast data quality assessment |
| | Comprehensive Audit | `data_quality_analysis` | Detailed data quality reports |
| | Missing Data | `check_missing_data` | Analyze missing data patterns |
| | Coverage Analysis | `coverage` | 24-hour coverage assessment |
| **Reporting** | Full Analytics | `run_all_analytics` | Comprehensive analytics reports |
| | Trend Summary | `generate_trend_summary` | Statistical trend summaries |
| **Dashboard** | Web Interface | `dashboard.app` | Interactive web dashboard |
| **Testing** | Unit Tests | `pytest -m "not integration"` | Fast unit tests |
| | Integration Tests | `pytest -m integration` | Full integration tests |
| | All Tests | `pytest` | Complete test suite |

### Usage Examples

```bash
# Basic usage - filter to high-quality data only
poetry run python -m garmin_analysis.viz.plot_trends_range --filter-24h-coverage
poetry run python -m garmin_analysis.reporting.generate_trend_summary --filter-24h-coverage

# Advanced usage - customize coverage parameters
poetry run python -m garmin_analysis.viz.plot_trends_range --filter-24h-coverage --max-gap 5 --day-edge-tolerance 5

# Full pipeline with filtering
poetry run python -m garmin_analysis.modeling.comprehensive_modeling_pipeline --filter-24h-coverage --target-col score

# Monthly reports with filtering
poetry run python -m garmin_analysis.reporting.run_all_analytics --filter-24h-coverage --monthly
```

### Dashboard Usage

In the interactive dashboard, you can toggle the "Only days with 24-hour continuous coverage" checkbox to filter both trend plots and correlation heatmaps. The filtering is applied in real-time and plot titles will indicate when filtering is active.

### Configuration Parameters

- `--max-gap`: Maximum allowed gap between consecutive samples (default: 2 minutes)
- `--day-edge-tolerance`: Allowed tolerance at day start/end (default: 2 minutes)
- `--filter-24h-coverage`: Enable 24-hour coverage filtering

### Data Quality Check

Check which days have 24-hour coverage:
```bash
poetry run python -m garmin_analysis.features.quick_data_check --continuous-24h
```

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

### Testing
- **Run unit tests (no I/O, uses in-memory DB fixtures):**
```bash
poetry run pytest -q -m "not integration"
```

- **Run integration tests (file-backed temp DBs; exercises real I/O paths):**
```bash
poetry run pytest -q -m integration
```

- **Run all tests:**
```bash
poetry run pytest -q
```

- **Run specific test modules:**
```bash
poetry run pytest tests/test_coverage_filtering.py -v
poetry run pytest tests/test_data_quality.py -v
poetry run pytest tests/test_dashboard_dependencies.py -v
```

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

## Project Structure

```
garmin-analysis/
├── src/garmin_analysis/           # Main package
│   ├── dashboard/                 # Interactive web dashboard
│   ├── data_ingestion/           # Data loading and preparation
│   ├── features/                 # Data quality and feature analysis
│   ├── modeling/                 # Machine learning algorithms
│   ├── reporting/                # Automated report generation
│   ├── utils/                    # Utility modules (activity mappings, etc.)
│   ├── viz/                      # Visualization tools
│   └── utils.py                  # Utility functions
├── config/                       # Configuration files
│   └── activity_type_mappings.json # Activity type mappings
├── docs/                         # Documentation
│   └── activity_type_mappings.md # Activity mapping documentation
├── examples/                     # Example scripts
│   └── activity_calendar_example.py # Activity calendar example
├── tests/                        # Test suite
├── notebooks/                    # Jupyter notebooks
├── data/                         # Generated datasets
├── plots/                        # Generated plots
├── reports/                      # Generated reports
├── modeling_results/             # ML model outputs
└── db/                          # Garmin database files
```

## Dependencies

The project uses modern Python data science libraries:
- **Core**: pandas, numpy, matplotlib, seaborn
- **ML**: scikit-learn, tsfresh, statsmodels, prophet
- **Visualization**: plotly, dash
- **Development**: pytest, jupyter

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
