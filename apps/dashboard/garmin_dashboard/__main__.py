"""Entry point: `poetry run python -m garmin_dashboard`."""

import logging


def main() -> int:
    """Run the dashboard."""
    try:
        from garmin_analysis.logging_config import setup_logging
        from garmin_analysis.utils import load_master_dataframe
        from garmin_dashboard.app import app

        setup_logging(level=logging.INFO)

        logging.info("Loading data for dashboard...")
        df = load_master_dataframe()
        logging.info("Loaded %s days of data", len(df))

        logging.info("Starting Garmin Health Dashboard...")
        logging.info("Dashboard will be available at: http://127.0.0.1:8050")
        logging.info("Press Ctrl+C to stop the dashboard")

        app.run(debug=True, host="127.0.0.1", port=8050)

    except KeyboardInterrupt:
        logging.info("Dashboard stopped by user")
    except Exception as e:
        logging.error("Failed to start dashboard: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
