#!/usr/bin/env python3
"""
Script to run the Garmin Health Dashboard with day-of-week analysis.
"""

import logging
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def main():
    """Run the dashboard."""
    try:
        from garmin_analysis.dashboard.app import app
        from garmin_analysis.logging_config import setup_logging
        from garmin_analysis.utils import load_master_dataframe

        # Configure logging
        setup_logging(level=logging.INFO)

        # Load data and start dashboard
        logging.info("Loading data for dashboard...")
        df = load_master_dataframe()
        logging.info(f"Loaded {len(df)} days of data")

        logging.info("Starting Garmin Health Dashboard...")
        logging.info("Dashboard will be available at: http://127.0.0.1:8050")
        logging.info("Press Ctrl+C to stop the dashboard")

        app.run(debug=True, host="127.0.0.1", port=8050)

    except KeyboardInterrupt:
        logging.info("Dashboard stopped by user")
    except Exception as e:
        logging.error(f"Failed to start dashboard: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
