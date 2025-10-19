#!/bin/bash
# Daily Garmin Data Update Script
# 
# This script automates the daily update process for your Garmin health data.
# It downloads the latest data from Garmin Connect, regenerates the unified dataset,
# and can optionally restart the dashboard.
#
# Usage:
#   ./daily_update.sh              # Update data only
#   ./daily_update.sh --restart    # Update data and restart dashboard
#
# Setup for daily automation (cron):
#   crontab -e
#   # Add: 0 6 * * * /path/to/garmin-analysis/examples/daily_update.sh >> ~/garmin-update.log 2>&1

set -e  # Exit on any error

# Configuration
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
RESTART_DASHBOARD=false

# Parse arguments
if [ "$1" = "--restart" ] || [ "$1" = "-r" ]; then
    RESTART_DASHBOARD=true
fi

echo "========================================"
echo "Garmin Data Daily Update"
echo "========================================"
echo "Started: $(date)"
echo "Project: $PROJECT_DIR"
echo ""

# Step 1: Download latest data from Garmin Connect
echo "📡 Step 1/3: Downloading latest data from Garmin Connect..."
cd "$PROJECT_DIR"
poetry run python -m garmin_analysis.cli_garmin_sync --sync --latest

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to download latest data from Garmin Connect"
    echo "Check your Garmin Connect credentials in ~/.GarminDb/GarminConnectConfig.json"
    exit 1
fi

echo "✅ Latest data downloaded successfully"
echo ""

# Step 2: Regenerate unified dataset
echo "🔨 Step 2/3: Regenerating unified dataset..."
poetry run python -m garmin_analysis.data_ingestion.load_all_garmin_dbs

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to regenerate unified dataset"
    exit 1
fi

echo "✅ Unified dataset regenerated"
echo ""

# Step 3: Optionally restart dashboard
if [ "$RESTART_DASHBOARD" = true ]; then
    echo "🔄 Step 3/3: Restarting dashboard..."
    
    # Kill existing dashboard process
    pkill -f "run_dashboard.py" || true
    sleep 2
    
    # Start dashboard in background
    nohup poetry run python run_dashboard.py > dashboard.log 2>&1 &
    
    echo "✅ Dashboard restarted (PID: $!)"
    echo "📊 Access at: http://localhost:8050"
else
    echo "ℹ️  Step 3/3: Skipped dashboard restart (use --restart to enable)"
fi

echo ""
echo "========================================"
echo "✅ Daily update complete!"
echo "Completed: $(date)"
echo "========================================"
echo ""
echo "📊 To view your data:"
echo "  poetry run python run_dashboard.py"
echo ""
echo "📈 To run analysis:"
echo "  poetry run python -m garmin_analysis.viz.plot_trends_range"

