"""Tests for quick_data_check CLI coverage diagnostics."""

from datetime import timedelta
from unittest.mock import patch

import pandas as pd
import pytest


def _full_day_hr(day: pd.Timestamp) -> pd.DataFrame:
    timestamps = [day + timedelta(minutes=i) for i in range(1440)]
    return pd.DataFrame({"timestamp": timestamps, "heart_rate": [70] * len(timestamps)})


class TestQuickDataCheckContinuous24h:
    def test_lists_days_from_monitoring_hr(self, capsys):
        from garmin_analysis.features import quick_data_check

        hr_df = _full_day_hr(pd.Timestamp("2024-01-01"))
        master_df = pd.DataFrame({"day": pd.date_range("2024-01-01", periods=1), "steps": [1000]})

        with (
            patch("garmin_analysis.logging_config.setup_logging"),
            patch(
                "garmin_analysis.utils.data_loading.load_master_dataframe",
                return_value=master_df,
            ),
            patch(
                "garmin_analysis.features.coverage.load_monitoring_hr_for_coverage",
                return_value=hr_df,
            ),
            patch("sys.argv", ["quick_data_check", "--continuous-24h"]),
        ):
            quick_data_check.main()

        captured = capsys.readouterr().out
        assert "2024-01-01" in captured
        assert "Total: 1 days" in captured

    def test_exits_when_no_hr_data(self):
        from garmin_analysis.features import quick_data_check

        master_df = pd.DataFrame({"day": pd.date_range("2024-01-01", periods=1), "steps": [1000]})

        with (
            patch("garmin_analysis.logging_config.setup_logging"),
            patch(
                "garmin_analysis.utils.data_loading.load_master_dataframe",
                return_value=master_df,
            ),
            patch(
                "garmin_analysis.features.coverage.load_monitoring_hr_for_coverage",
                return_value=None,
            ),
            patch("sys.argv", ["quick_data_check", "--continuous-24h"]),
        ):
            with pytest.raises(SystemExit) as exc:
                quick_data_check.main()

        assert exc.value.code == 2
