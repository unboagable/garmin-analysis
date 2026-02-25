"""Tests for dashboard Data Quality tab."""

import pytest
import pandas as pd
from pathlib import Path

from garmin_analysis.dashboard.app import (
    create_layout,
    update_data_quality_charts,
)
from garmin_analysis.utils.data_loading import load_master_dataframe


class TestDataQualityTabLayout:
    """Tests that Data Quality tab exists in layout."""

    def test_layout_includes_data_quality_tab(self):
        df = load_master_dataframe()
        layout = create_layout(df)
        tabs = layout.children[1].children
        tab_labels = [tab.label for tab in tabs]
        assert "📈 Data Quality" in tab_labels


class TestUpdateDataQualityCharts:
    """Tests for update_data_quality_charts callback."""

    def test_empty_dq_returns_empty_figures_and_message(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "garmin_analysis.dashboard.app.DAILY_DATA_QUALITY_CSV",
            tmp_path / "nonexistent_dq.csv",
        )
        monkeypatch.setattr(
            "garmin_analysis.features.daily_data_quality.DAILY_DATA_QUALITY_CSV",
            tmp_path / "nonexistent_dq.csv",
        )
        timeline, dist, scatter, stats = update_data_quality_charts(
            None, None, 0
        )
        assert timeline is not None
        assert dist is not None
        assert scatter is not None
        assert stats is not None
        assert "No data quality" in str(stats) or "not found" in str(stats).lower()

    def test_with_dq_data_returns_figures(self, tmp_path, monkeypatch):
        dq_df = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=10, freq="D"),
            "data_quality_score": [75, 80, 85, 90, 88, 82, 78, 85, 90, 92],
            "coverage_score": [95, 98, 100, 99, 97, 96, 94, 98, 100, 99],
            "completeness_score": [55, 62, 70, 81, 79, 68, 62, 72, 80, 85],
            "key_metrics_count": [4, 5, 6, 7, 6, 5, 5, 6, 6, 7],
            "key_metrics_total": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        })
        dq_path = tmp_path / "daily_data_quality.csv"
        dq_df.to_csv(dq_path, index=False)

        monkeypatch.setattr(
            "garmin_analysis.dashboard.app.DAILY_DATA_QUALITY_CSV",
            dq_path,
        )
        monkeypatch.setattr(
            "garmin_analysis.features.daily_data_quality.DAILY_DATA_QUALITY_CSV",
            dq_path,
        )

        timeline, dist, scatter, stats = update_data_quality_charts(
            "2024-01-01", "2024-01-10", 0
        )
        assert timeline is not None
        assert len(timeline.data) >= 1
        assert dist is not None
        assert scatter is not None
        assert stats is not None
        assert "Average" in str(stats) or "Days" in str(stats)

    def test_date_filter_narrows_data(self, tmp_path, monkeypatch):
        dq_df = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=20, freq="D"),
            "data_quality_score": [80] * 20,
            "coverage_score": [95] * 20,
            "completeness_score": [65] * 20,
            "key_metrics_count": [5] * 20,
            "key_metrics_total": [8] * 20,
        })
        dq_path = tmp_path / "daily_data_quality.csv"
        dq_df.to_csv(dq_path, index=False)

        monkeypatch.setattr(
            "garmin_analysis.dashboard.app.DAILY_DATA_QUALITY_CSV",
            dq_path,
        )
        monkeypatch.setattr(
            "garmin_analysis.features.daily_data_quality.DAILY_DATA_QUALITY_CSV",
            dq_path,
        )

        timeline, dist, scatter, stats = update_data_quality_charts(
            "2024-01-05", "2024-01-10", 0
        )
        assert timeline is not None
        assert len(timeline.data[0].x) == 6
