import logging
import pandas as pd
from garmin_analysis.utils.data_loading import load_garmin_tables

# Logging is configured at package level


logger = logging.getLogger(__name__)
def preview_table(df, name, cols=None, max_rows=3):
    if df.empty:
        logger.warning("Table '%s' is empty.", name)
        return
    logger.info("\n📌 %s — %d rows", name, len(df))
    if cols:
        logger.info(f"\n{df[cols].head(max_rows)}")
    else:
        logger.info(f"\n{df.head(max_rows)}")

def main():
    tables = load_garmin_tables()

    if not tables:
        logger.error("No tables were loaded. Exiting.")
        return

    logger.info("\n📂 Available tables:")
    for name in tables:
        logger.info("- %s (%d rows)", name, len(tables[name]))

    preview_table(tables["daily"], "daily_summary", ["day", "steps", "calories_total", "hr_min", "hr_max", "rhr"])
    preview_table(tables["sleep"], "sleep", ["day", "total_sleep", "deep_sleep", "rem_sleep", "score"])
    preview_table(tables["stress"], "stress", ["timestamp", "stress"])
    preview_table(tables["rest_hr"], "resting_hr", ["day", "resting_heart_rate"])

if __name__ == "__main__":
    main()
