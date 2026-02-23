"""
Comprehensive tests for the weekly health report feature.

Covers:
- Helper functions (_format_delta, _direction_word, _arrow, _resolve_rhr_column)
- Internal section builders (_metric_section, _stress_section, _latest_week_summary)
- Weekly aggregation logic (compute_weekly_aggregates)
- Full report generation (generate_weekly_report)
- CLI entry-point (cli_weekly_report.main)
- Edge cases (year boundaries, duplicates, unsorted data, all-NaN, boundary
  day-counts, negative values, extreme values)
"""

import argparse
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from garmin_analysis.reporting.generate_weekly_report import (
    SLEEP_COL,
    STRESS_COL,
    DATE_COL,
    MIN_DAYS_FOR_WEEK,
    _arrow,
    _direction_word,
    _format_delta,
    _latest_week_summary,
    _metric_section,
    _resolve_rhr_column,
    _stress_section,
    compute_weekly_aggregates,
    generate_weekly_report,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def four_weeks_data():
    """28 days of deterministic data spanning four complete ISO weeks (Mon-Sun)."""
    np.random.seed(0)
    n = 28
    dates = pd.date_range("2024-06-03", periods=n, freq="D")  # Mon W23 – Sun W26
    return pd.DataFrame(
        {
            "day": dates,
            "score": np.random.randint(55, 95, n).astype(float),
            "resting_heart_rate": np.random.uniform(55, 70, n),
            "stress_duration": np.random.randint(100, 500, n),
        }
    )


@pytest.fixture
def two_weeks_deterministic():
    """14 days with hand-picked values so delta math is verifiable."""
    dates = pd.date_range("2024-07-01", periods=14, freq="D")  # Mon W27 – Sun W28
    return pd.DataFrame(
        {
            "day": dates,
            "score": [70.0] * 7 + [80.0] * 7,           # avg 70 → 80, delta +10
            "resting_heart_rate": [60.0] * 7 + [55.0] * 7,  # avg 60 → 55, delta -5
            "stress_duration": [200] * 7 + [300] * 7,     # total 1400 → 2100, delta +700
        }
    )


@pytest.fixture
def sparse_data():
    """Data with gaps and NaN values across two weeks."""
    dates = pd.to_datetime(
        ["2024-06-03", "2024-06-05", "2024-06-10", "2024-06-11", "2024-06-12"]
    )
    return pd.DataFrame(
        {
            "day": dates,
            "score": [80, np.nan, 72, 65, 90],
            "resting_heart_rate": [60, 62, np.nan, 58, 59],
            "stress_duration": [200, 300, 150, np.nan, 250],
        }
    )


@pytest.fixture
def weekly_with_all_columns(two_weeks_deterministic):
    """Pre-computed weekly aggregate from the deterministic fixture."""
    return compute_weekly_aggregates(two_weeks_deterministic, num_weeks=10)


# ---------------------------------------------------------------------------
# Tests – helper utilities
# ---------------------------------------------------------------------------

class TestFormatDelta:
    def test_positive(self):
        assert _format_delta(3.5) == "+3.5"

    def test_negative(self):
        assert _format_delta(-2.1) == "-2.1"

    def test_zero(self):
        assert _format_delta(0.0) == "+0.0"

    def test_precision_zero(self):
        assert _format_delta(7.8, precision=0) == "+8"

    def test_precision_two(self):
        assert _format_delta(1.234, precision=2) == "+1.23"

    def test_precision_three_negative(self):
        assert _format_delta(-0.1267, precision=3) == "-0.127"

    def test_large_positive(self):
        assert _format_delta(1234.5) == "+1234.5"

    def test_tiny_negative(self):
        assert _format_delta(-0.04, precision=1) == "-0.0"


class TestDirectionWord:
    def test_up(self):
        assert _direction_word(1) == "up"
        assert _direction_word(0.001) == "up"

    def test_down(self):
        assert _direction_word(-1) == "down"
        assert _direction_word(-0.001) == "down"

    def test_flat(self):
        assert _direction_word(0) == "flat"


class TestArrow:
    def test_up(self):
        assert _arrow(5) == "^"

    def test_down(self):
        assert _arrow(-3) == "v"

    def test_equal(self):
        assert _arrow(0) == "="


class TestResolveRhrColumn:
    def test_prefers_primary(self):
        df = pd.DataFrame({"resting_heart_rate": [60.0], "rhr": [61]})
        assert _resolve_rhr_column(df) == "resting_heart_rate"

    def test_falls_back_to_rhr(self):
        df = pd.DataFrame({"rhr": [61]})
        assert _resolve_rhr_column(df) == "rhr"

    def test_none_when_absent(self):
        df = pd.DataFrame({"steps": [1000]})
        assert _resolve_rhr_column(df) is None

    def test_skips_all_nan_primary(self):
        df = pd.DataFrame({"resting_heart_rate": [np.nan, np.nan], "rhr": [61, 62]})
        assert _resolve_rhr_column(df) == "rhr"

    def test_both_all_nan(self):
        df = pd.DataFrame({"resting_heart_rate": [np.nan], "rhr": [np.nan]})
        assert _resolve_rhr_column(df) is None

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        assert _resolve_rhr_column(df) is None

    def test_partial_nan_still_picks_primary(self):
        df = pd.DataFrame({"resting_heart_rate": [np.nan, 60.0], "rhr": [61, 62]})
        assert _resolve_rhr_column(df) == "resting_heart_rate"


# ---------------------------------------------------------------------------
# Tests – _metric_section
# ---------------------------------------------------------------------------

class TestMetricSection:
    """Unit tests for the generic _metric_section table builder."""

    def _build_weekly(self, avgs, counts):
        """Convenience: build a minimal weekly DataFrame."""
        n = len(avgs)
        idx = pd.MultiIndex.from_tuples(
            [(2024, 27 + i) for i in range(n)], names=["iso_year", "iso_week"]
        )
        return pd.DataFrame(
            {"col_avg": avgs, "col_count": counts, "week_start": pd.NaT, "week_end": pd.NaT},
            index=idx,
        )

    def test_missing_column_returns_no_data(self):
        weekly = self._build_weekly([70], [7])
        result = _metric_section(weekly, "MISSING", "col_count", "Title", "u")
        assert "No data available" in result
        assert "### Title" in result

    def test_first_week_has_no_delta(self):
        weekly = self._build_weekly([75.0], [7])
        result = _metric_section(weekly, "col_avg", "col_count", "T", "")
        assert "75.0" in result
        assert "| -- |" in result  # delta placeholder

    def test_second_week_shows_delta(self):
        weekly = self._build_weekly([70.0, 80.0], [7, 7])
        result = _metric_section(weekly, "col_avg", "col_count", "T", "", precision=1)
        assert "+10.0" in result

    def test_negative_delta(self):
        weekly = self._build_weekly([80.0, 70.0], [7, 7])
        result = _metric_section(weekly, "col_avg", "col_count", "T", "", precision=1)
        assert "-10.0" in result

    def test_lower_is_better_positive_trend(self):
        """When lower_is_better=True, a decrease should show '+' (improvement)."""
        weekly = self._build_weekly([65.0, 60.0], [7, 7])
        result = _metric_section(
            weekly, "col_avg", "col_count", "RHR", "bpm", lower_is_better=True
        )
        lines = result.strip().split("\n")
        data_line = lines[-1]
        assert "| + |" in data_line

    def test_lower_is_better_negative_trend(self):
        """When lower_is_better=True, an increase should show '-' (worsening)."""
        weekly = self._build_weekly([60.0, 65.0], [7, 7])
        result = _metric_section(
            weekly, "col_avg", "col_count", "RHR", "bpm", lower_is_better=True
        )
        lines = result.strip().split("\n")
        data_line = lines[-1]
        assert "| - |" in data_line

    def test_higher_is_better_positive_trend(self):
        """Default: increase shows '+' (improvement)."""
        weekly = self._build_weekly([70.0, 80.0], [7, 7])
        result = _metric_section(
            weekly, "col_avg", "col_count", "Sleep", "", lower_is_better=False
        )
        lines = result.strip().split("\n")
        data_line = lines[-1]
        assert "| + |" in data_line

    def test_higher_is_better_negative_trend(self):
        """Default: decrease shows '-' (worsening)."""
        weekly = self._build_weekly([80.0, 70.0], [7, 7])
        result = _metric_section(
            weekly, "col_avg", "col_count", "Sleep", "", lower_is_better=False
        )
        lines = result.strip().split("\n")
        data_line = lines[-1]
        assert "| - |" in data_line

    def test_zero_delta_shows_equal(self):
        weekly = self._build_weekly([70.0, 70.0], [7, 7])
        result = _metric_section(weekly, "col_avg", "col_count", "T", "")
        lines = result.strip().split("\n")
        data_line = lines[-1]
        assert "| = |" in data_line

    def test_insufficient_days_shows_dashes(self):
        weekly = self._build_weekly([70.0, 80.0], [7, MIN_DAYS_FOR_WEEK - 1])
        result = _metric_section(weekly, "col_avg", "col_count", "T", "")
        lines = result.strip().split("\n")
        last_data = lines[-1]
        assert "| -- | -- | -- |" in last_data

    def test_insufficient_days_resets_prev_val(self):
        """After a week with too few days, the next valid week should have no delta."""
        weekly = self._build_weekly(
            [70.0, 80.0, 75.0],
            [7, MIN_DAYS_FOR_WEEK - 1, 7],
        )
        result = _metric_section(weekly, "col_avg", "col_count", "T", "")
        lines = [l for l in result.strip().split("\n") if l.startswith("|") and "Week" not in l and "---" not in l]
        assert len(lines) == 3
        # Third row (W29) should show "--" for delta because W28 was skipped
        assert "| -- |" in lines[2]

    def test_nan_average_treated_like_insufficient(self):
        weekly = self._build_weekly([70.0, np.nan], [7, 7])
        result = _metric_section(weekly, "col_avg", "col_count", "T", "")
        lines = result.strip().split("\n")
        last_data = lines[-1]
        assert "| -- | -- | -- |" in last_data

    def test_precision_applied(self):
        weekly = self._build_weekly([70.123], [7])
        result = _metric_section(weekly, "col_avg", "col_count", "T", "", precision=2)
        assert "70.12" in result

    def test_unit_string_appears(self):
        weekly = self._build_weekly([60.0], [7])
        result = _metric_section(weekly, "col_avg", "col_count", "RHR", "bpm")
        assert "bpm" in result

    def test_table_header_present(self):
        weekly = self._build_weekly([70.0], [7])
        result = _metric_section(weekly, "col_avg", "col_count", "Title", "")
        assert "| Week | Avg | Delta | Trend | Days |" in result
        assert "|------|-----|-------|-------|------|" in result

    def test_week_label_format(self):
        weekly = self._build_weekly([70.0], [7])
        result = _metric_section(weekly, "col_avg", "col_count", "T", "")
        assert "2024-W27" in result

    def test_multiple_consecutive_weeks(self):
        weekly = self._build_weekly([60.0, 65.0, 70.0, 75.0], [7, 7, 7, 7])
        result = _metric_section(weekly, "col_avg", "col_count", "T", "")
        assert "W27" in result
        assert "W28" in result
        assert "W29" in result
        assert "W30" in result


# ---------------------------------------------------------------------------
# Tests – _stress_section
# ---------------------------------------------------------------------------

class TestStressSection:
    """Unit tests for the stress-minutes section builder."""

    def _build_weekly(self, totals, counts):
        n = len(totals)
        idx = pd.MultiIndex.from_tuples(
            [(2024, 27 + i) for i in range(n)], names=["iso_year", "iso_week"]
        )
        return pd.DataFrame(
            {
                "stress_minutes_total": totals,
                "stress_minutes_count": counts,
                "week_start": pd.NaT,
                "week_end": pd.NaT,
            },
            index=idx,
        )

    def test_missing_column_returns_no_data(self):
        idx = pd.MultiIndex.from_tuples([(2024, 27)], names=["iso_year", "iso_week"])
        weekly = pd.DataFrame({"other": [1]}, index=idx)
        result = _stress_section(weekly)
        assert "No data available" in result

    def test_first_week_no_delta(self):
        weekly = self._build_weekly([1400], [7])
        result = _stress_section(weekly)
        assert "1400 min" in result
        assert "| -- |" in result  # no prior week

    def test_stress_decrease_is_improvement(self):
        """Lower stress total → '+' (improvement)."""
        weekly = self._build_weekly([2000, 1500], [7, 7])
        result = _stress_section(weekly)
        lines = [l for l in result.strip().split("\n") if l.startswith("|") and "Week" not in l and "---" not in l]
        assert "| + |" in lines[-1]

    def test_stress_increase_is_worsening(self):
        """Higher stress total → '-' (worsening)."""
        weekly = self._build_weekly([1500, 2000], [7, 7])
        result = _stress_section(weekly)
        lines = [l for l in result.strip().split("\n") if l.startswith("|") and "Week" not in l and "---" not in l]
        assert "| - |" in lines[-1]

    def test_stress_unchanged_is_equal(self):
        weekly = self._build_weekly([1500, 1500], [7, 7])
        result = _stress_section(weekly)
        lines = [l for l in result.strip().split("\n") if l.startswith("|") and "Week" not in l and "---" not in l]
        assert "| = |" in lines[-1]

    def test_delta_value_correct(self):
        weekly = self._build_weekly([1400, 2100], [7, 7])
        result = _stress_section(weekly)
        assert "+700" in result

    def test_negative_delta_value(self):
        weekly = self._build_weekly([2100, 1400], [7, 7])
        result = _stress_section(weekly)
        assert "-700" in result

    def test_insufficient_days_shows_dashes(self):
        weekly = self._build_weekly([1400, 2100], [7, MIN_DAYS_FOR_WEEK - 1])
        result = _stress_section(weekly)
        lines = result.strip().split("\n")
        assert "| -- | -- | -- |" in lines[-1]

    def test_insufficient_days_resets_tracking(self):
        weekly = self._build_weekly([1400, 2100, 1800], [7, MIN_DAYS_FOR_WEEK - 1, 7])
        result = _stress_section(weekly)
        lines = [l for l in result.strip().split("\n") if l.startswith("|") and "Week" not in l and "---" not in l]
        # W29 should have no delta since W28 was skipped
        assert "| -- |" in lines[2]

    def test_table_header(self):
        weekly = self._build_weekly([1400], [7])
        result = _stress_section(weekly)
        assert "| Week | Total min | Delta | Trend | Days |" in result

    def test_uses_integer_totals(self):
        weekly = self._build_weekly([1400.7], [7])
        result = _stress_section(weekly)
        assert "1401 min" in result  # rounded


# ---------------------------------------------------------------------------
# Tests – _latest_week_summary
# ---------------------------------------------------------------------------

class TestLatestWeekSummary:
    """Unit tests for the narrative summary builder."""

    def _build_weekly(self, rows, columns):
        n = len(rows)
        idx = pd.MultiIndex.from_tuples(
            [(2024, 27 + i) for i in range(n)], names=["iso_year", "iso_week"]
        )
        return pd.DataFrame(rows, index=idx, columns=columns)

    def _with_counts(self, rows, count=7):
        """Add matching count columns so the summary can pass MIN_DAYS_FOR_WEEK."""
        enriched = []
        for row in rows:
            r = dict(row)
            if "sleep_score_avg" in r:
                r.setdefault("sleep_score_count", count)
            if "rhr_avg" in r:
                r.setdefault("rhr_count", count)
            if "stress_minutes_total" in r:
                r.setdefault("stress_minutes_count", count)
            enriched.append(r)
        return enriched

    def test_returns_empty_for_single_week(self):
        rows = self._with_counts([{"sleep_score_avg": 75.0}])
        weekly = self._build_weekly(rows, list(rows[0].keys()))
        assert _latest_week_summary(weekly) == ""

    def test_returns_empty_for_empty_df(self):
        idx = pd.MultiIndex.from_tuples([], names=["iso_year", "iso_week"])
        weekly = pd.DataFrame(index=idx)
        assert _latest_week_summary(weekly) == ""

    def test_full_narrative(self, weekly_with_all_columns):
        result = _latest_week_summary(weekly_with_all_columns)
        assert "Week at a glance" in result
        assert "Sleep score" in result
        assert "Resting HR" in result
        assert "Stress" in result
        assert "bpm" in result
        assert "min" in result

    def test_narrative_delta_values(self):
        rows = self._with_counts([
            {"sleep_score_avg": 70.0, "rhr_avg": 60.0, "stress_minutes_total": 1400},
            {"sleep_score_avg": 80.0, "rhr_avg": 55.0, "stress_minutes_total": 2100},
        ])
        weekly = self._build_weekly(rows, list(rows[0].keys()))
        result = _latest_week_summary(weekly)
        assert "80.0" in result        # current sleep
        assert "+10.0" in result       # sleep delta
        assert "55.0 bpm" in result    # current rhr
        assert "-5.0 bpm" in result    # rhr delta
        assert "2100 min" in result    # current stress
        assert "+700" in result        # stress delta

    def test_missing_sleep_column(self):
        rows = self._with_counts([
            {"rhr_avg": 60.0},
            {"rhr_avg": 55.0},
        ])
        weekly = self._build_weekly(rows, list(rows[0].keys()))
        result = _latest_week_summary(weekly)
        assert "Sleep" not in result
        assert "Resting HR" in result

    def test_missing_rhr_column(self):
        rows = self._with_counts([
            {"sleep_score_avg": 70.0},
            {"sleep_score_avg": 80.0},
        ])
        weekly = self._build_weekly(rows, list(rows[0].keys()))
        result = _latest_week_summary(weekly)
        assert "Sleep" in result
        assert "Resting HR" not in result

    def test_missing_stress_column(self):
        rows = self._with_counts([
            {"sleep_score_avg": 70.0},
            {"sleep_score_avg": 80.0},
        ])
        weekly = self._build_weekly(rows, list(rows[0].keys()))
        result = _latest_week_summary(weekly)
        assert "Stress" not in result

    def test_nan_in_current_week_skips_metric(self):
        rows = self._with_counts([
            {"sleep_score_avg": 70.0, "rhr_avg": 60.0},
            {"sleep_score_avg": np.nan, "rhr_avg": 55.0},
        ])
        weekly = self._build_weekly(rows, list(rows[0].keys()))
        result = _latest_week_summary(weekly)
        assert "Sleep" not in result
        assert "Resting HR" in result

    def test_nan_in_previous_week_skips_metric(self):
        rows = self._with_counts([
            {"sleep_score_avg": np.nan, "rhr_avg": 60.0},
            {"sleep_score_avg": 80.0, "rhr_avg": 55.0},
        ])
        weekly = self._build_weekly(rows, list(rows[0].keys()))
        result = _latest_week_summary(weekly)
        assert "Sleep" not in result
        assert "Resting HR" in result

    def test_all_nan_returns_empty(self):
        rows = self._with_counts([
            {"sleep_score_avg": np.nan},
            {"sleep_score_avg": np.nan},
        ])
        weekly = self._build_weekly(rows, list(rows[0].keys()))
        assert _latest_week_summary(weekly) == ""

    def test_direction_word_in_narrative(self):
        rows = self._with_counts([
            {"sleep_score_avg": 80.0},
            {"sleep_score_avg": 70.0},
        ])
        weekly = self._build_weekly(rows, list(rows[0].keys()))
        result = _latest_week_summary(weekly)
        assert "down" in result

    def test_flat_direction_in_narrative(self):
        rows = self._with_counts([
            {"sleep_score_avg": 75.0},
            {"sleep_score_avg": 75.0},
        ])
        weekly = self._build_weekly(rows, list(rows[0].keys()))
        result = _latest_week_summary(weekly)
        assert "flat" in result

    def test_insufficient_count_skips_metric(self):
        """When count < MIN_DAYS_FOR_WEEK, the metric should not appear in summary."""
        rows = [
            {"sleep_score_avg": 70.0, "sleep_score_count": 1},
            {"sleep_score_avg": 80.0, "sleep_score_count": 1},
        ]
        weekly = self._build_weekly(rows, list(rows[0].keys()))
        assert _latest_week_summary(weekly) == ""

    def test_mixed_sufficient_insufficient(self):
        """Only metrics with sufficient counts should appear."""
        rows = [
            {"sleep_score_avg": 70.0, "sleep_score_count": 7,
             "rhr_avg": 60.0, "rhr_count": 1},
            {"sleep_score_avg": 80.0, "sleep_score_count": 7,
             "rhr_avg": 55.0, "rhr_count": 1},
        ]
        weekly = self._build_weekly(rows, list(rows[0].keys()))
        result = _latest_week_summary(weekly)
        assert "Sleep" in result
        assert "Resting HR" not in result


# ---------------------------------------------------------------------------
# Tests – compute_weekly_aggregates
# ---------------------------------------------------------------------------

class TestComputeWeeklyAggregates:
    def test_basic_shape(self, four_weeks_data):
        weekly = compute_weekly_aggregates(four_weeks_data, num_weeks=10)
        assert len(weekly) == 4
        assert "sleep_score_avg" in weekly.columns
        assert "rhr_avg" in weekly.columns
        assert "stress_minutes_total" in weekly.columns

    def test_num_weeks_limits(self, four_weeks_data):
        weekly = compute_weekly_aggregates(four_weeks_data, num_weeks=2)
        assert len(weekly) == 2

    def test_num_weeks_larger_than_data(self, four_weeks_data):
        weekly = compute_weekly_aggregates(four_weeks_data, num_weeks=100)
        assert len(weekly) == 4

    def test_missing_columns_graceful(self):
        df = pd.DataFrame({"day": pd.date_range("2024-06-03", periods=7), "steps": range(7)})
        weekly = compute_weekly_aggregates(df)
        assert "sleep_score_avg" not in weekly.columns
        assert "rhr_avg" not in weekly.columns
        assert "stress_minutes_total" not in weekly.columns
        assert "week_start" in weekly.columns
        assert "week_end" in weekly.columns

    def test_week_start_end_ordering(self, four_weeks_data):
        weekly = compute_weekly_aggregates(four_weeks_data, num_weeks=10)
        for _, row in weekly.iterrows():
            assert row["week_start"] <= row["week_end"]

    def test_deterministic_averages(self, two_weeks_deterministic):
        weekly = compute_weekly_aggregates(two_weeks_deterministic, num_weeks=10)
        assert len(weekly) == 2
        assert weekly["sleep_score_avg"].iloc[0] == pytest.approx(70.0)
        assert weekly["sleep_score_avg"].iloc[1] == pytest.approx(80.0)

    def test_deterministic_rhr(self, two_weeks_deterministic):
        weekly = compute_weekly_aggregates(two_weeks_deterministic, num_weeks=10)
        assert weekly["rhr_avg"].iloc[0] == pytest.approx(60.0)
        assert weekly["rhr_avg"].iloc[1] == pytest.approx(55.0)

    def test_deterministic_stress_total(self, two_weeks_deterministic):
        weekly = compute_weekly_aggregates(two_weeks_deterministic, num_weeks=10)
        assert weekly["stress_minutes_total"].iloc[0] == pytest.approx(1400)
        assert weekly["stress_minutes_total"].iloc[1] == pytest.approx(2100)

    def test_count_reflects_non_nan(self):
        dates = pd.date_range("2024-06-03", periods=7, freq="D")
        df = pd.DataFrame({
            "day": dates,
            "score": [70, np.nan, 75, np.nan, 80, 85, 90],
        })
        weekly = compute_weekly_aggregates(df)
        assert weekly["sleep_score_count"].iloc[0] == 5

    def test_string_dates_coerced(self):
        df = pd.DataFrame({
            "day": ["2024-06-03", "2024-06-04", "2024-06-05",
                    "2024-06-06", "2024-06-07", "2024-06-08", "2024-06-09"],
            "score": [75.0] * 7,
        })
        weekly = compute_weekly_aggregates(df)
        assert len(weekly) == 1
        assert weekly["sleep_score_avg"].iloc[0] == pytest.approx(75.0)

    def test_nan_dates_dropped(self):
        df = pd.DataFrame({
            "day": [pd.NaT, "2024-06-03", "2024-06-04", "2024-06-05",
                    "2024-06-06", "2024-06-07", "2024-06-08", "2024-06-09"],
            "score": [99.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0],
        })
        weekly = compute_weekly_aggregates(df)
        assert len(weekly) == 1
        assert weekly["sleep_score_avg"].iloc[0] == pytest.approx(75.0)

    def test_invalid_dates_coerced_to_nat(self):
        df = pd.DataFrame({
            "day": ["NOT_A_DATE", "2024-06-03", "2024-06-04", "2024-06-05",
                    "2024-06-06", "2024-06-07", "2024-06-08", "2024-06-09"],
            "score": [99.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0],
        })
        weekly = compute_weekly_aggregates(df)
        assert len(weekly) >= 1

    def test_single_week(self):
        dates = pd.date_range("2024-06-03", periods=7, freq="D")
        df = pd.DataFrame({"day": dates, "score": [80.0] * 7})
        weekly = compute_weekly_aggregates(df, num_weeks=10)
        assert len(weekly) == 1

    def test_all_nan_score_column(self):
        dates = pd.date_range("2024-06-03", periods=7, freq="D")
        df = pd.DataFrame({"day": dates, "score": [np.nan] * 7})
        weekly = compute_weekly_aggregates(df)
        assert weekly["sleep_score_count"].iloc[0] == 0

    def test_year_boundary(self):
        """Data spanning Dec 30 2024 → Jan 5 2025 crosses ISO week boundary."""
        dates = pd.date_range("2024-12-30", periods=7, freq="D")
        df = pd.DataFrame({"day": dates, "score": [80.0] * 7})
        weekly = compute_weekly_aggregates(df, num_weeks=10)
        assert len(weekly) >= 1

    def test_rhr_fallback_used_in_aggregation(self):
        dates = pd.date_range("2024-06-03", periods=7, freq="D")
        df = pd.DataFrame({"day": dates, "rhr": [62] * 7})
        weekly = compute_weekly_aggregates(df)
        assert "rhr_avg" in weekly.columns
        assert weekly["rhr_avg"].iloc[0] == pytest.approx(62.0)

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["day", "score"])
        weekly = compute_weekly_aggregates(df)
        assert weekly.empty

    def test_unsorted_input(self):
        dates = pd.date_range("2024-06-03", periods=7, freq="D")
        df = pd.DataFrame({"day": dates, "score": [70, 71, 72, 73, 74, 75, 76]})
        df_shuffled = df.sample(frac=1, random_state=42)
        weekly = compute_weekly_aggregates(df_shuffled)
        assert weekly["sleep_score_avg"].iloc[0] == pytest.approx(73.0)

    def test_duplicate_dates(self):
        dates = list(pd.date_range("2024-06-03", periods=7, freq="D")) + [
            pd.Timestamp("2024-06-03"),
            pd.Timestamp("2024-06-04"),
        ]
        df = pd.DataFrame({"day": dates, "score": [70.0] * 9})
        weekly = compute_weekly_aggregates(df)
        assert weekly["sleep_score_count"].iloc[0] == 9


# ---------------------------------------------------------------------------
# Tests – generate_weekly_report (full report output)
# ---------------------------------------------------------------------------

class TestGenerateWeeklyReport:
    def test_creates_file(self, four_weeks_data, tmp_path):
        path = generate_weekly_report(
            four_weeks_data, output_dir=str(tmp_path), timestamp="20240701_120000"
        )
        assert Path(path).exists()
        assert Path(path).name == "weekly_report_20240701_120000.md"

    def test_auto_timestamp(self, four_weeks_data, tmp_path):
        path = generate_weekly_report(four_weeks_data, output_dir=str(tmp_path))
        assert Path(path).exists()
        assert "weekly_report_" in Path(path).name

    def test_creates_nested_output_dir(self, four_weeks_data, tmp_path):
        nested = tmp_path / "a" / "b"
        path = generate_weekly_report(four_weeks_data, output_dir=str(nested), timestamp="t")
        assert nested.exists()
        assert Path(path).exists()

    def test_markdown_headers(self, four_weeks_data, tmp_path):
        path = generate_weekly_report(four_weeks_data, output_dir=str(tmp_path), timestamp="t")
        content = Path(path).read_text()
        assert "# Weekly Health Report" in content
        assert "## Summary" in content
        assert "## Metrics" in content
        assert "### Sleep Score Trend" in content
        assert "### Resting Heart Rate" in content
        assert "### Stress Minutes" in content

    def test_metadata_header(self, four_weeks_data, tmp_path):
        path = generate_weekly_report(
            four_weeks_data, output_dir=str(tmp_path), timestamp="20240701_120000"
        )
        content = Path(path).read_text()
        assert "Generated: 20240701_120000" in content
        assert "Covering:" in content
        assert "Weeks shown: 4" in content

    def test_week_range_in_header(self, two_weeks_deterministic, tmp_path):
        path = generate_weekly_report(
            two_weeks_deterministic, output_dir=str(tmp_path), timestamp="t"
        )
        content = Path(path).read_text()
        assert "2024-07-01" in content
        assert "2024-07-14" in content

    def test_horizontal_rule(self, four_weeks_data, tmp_path):
        path = generate_weekly_report(four_weeks_data, output_dir=str(tmp_path), timestamp="t")
        content = Path(path).read_text()
        assert "---" in content

    def test_table_rows_present(self, four_weeks_data, tmp_path):
        path = generate_weekly_report(
            four_weeks_data, num_weeks=4, output_dir=str(tmp_path), timestamp="t"
        )
        content = Path(path).read_text()
        assert "W23" in content or "W24" in content

    def test_summary_paragraph(self, four_weeks_data, tmp_path):
        path = generate_weekly_report(four_weeks_data, output_dir=str(tmp_path), timestamp="t")
        content = Path(path).read_text()
        assert "Week at a glance" in content

    def test_empty_dataframe_returns_empty(self, tmp_path):
        df = pd.DataFrame(columns=["day", "score"])
        result = generate_weekly_report(df, output_dir=str(tmp_path), timestamp="t")
        assert result == ""

    def test_only_sleep_column(self, tmp_path):
        n = 14
        df = pd.DataFrame({
            "day": pd.date_range("2024-06-03", periods=n),
            "score": np.random.randint(60, 90, n).astype(float),
        })
        path = generate_weekly_report(df, output_dir=str(tmp_path), timestamp="t")
        content = Path(path).read_text()
        assert "Sleep Score Trend" in content
        assert content.count("No data available") == 2  # RHR + stress

    def test_only_rhr_column(self, tmp_path):
        n = 14
        df = pd.DataFrame({
            "day": pd.date_range("2024-06-03", periods=n),
            "resting_heart_rate": np.random.uniform(55, 65, n),
        })
        path = generate_weekly_report(df, output_dir=str(tmp_path), timestamp="t")
        content = Path(path).read_text()
        assert "Resting Heart Rate" in content
        assert "bpm" in content
        assert content.count("No data available") == 2  # sleep + stress

    def test_only_stress_column(self, tmp_path):
        n = 14
        df = pd.DataFrame({
            "day": pd.date_range("2024-06-03", periods=n),
            "stress_duration": np.random.randint(100, 400, n),
        })
        path = generate_weekly_report(df, output_dir=str(tmp_path), timestamp="t")
        content = Path(path).read_text()
        assert "Stress Minutes" in content
        assert content.count("No data available") == 2  # sleep + RHR

    def test_rhr_fallback_column(self, tmp_path):
        n = 14
        df = pd.DataFrame({
            "day": pd.date_range("2024-06-03", periods=n),
            "rhr": np.random.randint(55, 70, n),
        })
        path = generate_weekly_report(df, output_dir=str(tmp_path), timestamp="t")
        content = Path(path).read_text()
        assert "Resting Heart Rate" in content
        assert "bpm" in content

    def test_sparse_data_handling(self, sparse_data, tmp_path):
        path = generate_weekly_report(sparse_data, output_dir=str(tmp_path), timestamp="t")
        assert Path(path).exists()

    def test_returns_string_path(self, four_weeks_data, tmp_path):
        result = generate_weekly_report(four_weeks_data, output_dir=str(tmp_path), timestamp="t")
        assert isinstance(result, str)
        assert result.endswith(".md")

    def test_num_weeks_passthrough(self, four_weeks_data, tmp_path):
        path = generate_weekly_report(
            four_weeks_data, num_weeks=2, output_dir=str(tmp_path), timestamp="t"
        )
        content = Path(path).read_text()
        assert "Weeks shown: 2" in content

    def test_single_week_no_summary(self, tmp_path):
        """A single week should produce a report but no narrative summary."""
        dates = pd.date_range("2024-06-03", periods=7, freq="D")
        df = pd.DataFrame({
            "day": dates,
            "score": [75.0] * 7,
            "resting_heart_rate": [60.0] * 7,
            "stress_duration": [200] * 7,
        })
        path = generate_weekly_report(df, output_dir=str(tmp_path), timestamp="t")
        content = Path(path).read_text()
        assert "Week at a glance" not in content
        assert "### Sleep Score Trend" in content

    def test_auto_load_data(self, tmp_path):
        """When df=None, generate_weekly_report should call load_master_dataframe."""
        mock_df = pd.DataFrame({
            "day": pd.date_range("2024-06-03", periods=14),
            "score": [75.0] * 14,
        })
        with patch(
            "garmin_analysis.reporting.generate_weekly_report.load_master_dataframe",
            return_value=mock_df,
        ) as mock_load:
            path = generate_weekly_report(
                df=None, output_dir=str(tmp_path), timestamp="t"
            )
            mock_load.assert_called_once()
            assert Path(path).exists()

    def test_default_output_dir(self):
        """When output_dir=None, should use REPORTS_DIR."""
        mock_df = pd.DataFrame({
            "day": pd.date_range("2024-06-03", periods=14),
            "score": [75.0] * 14,
        })
        with patch("builtins.open", create=True) as mock_open, \
             patch("garmin_analysis.reporting.generate_weekly_report.Path.mkdir"):
            mock_open.return_value.__enter__ = lambda s: MagicMock()
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            result = generate_weekly_report(
                df=mock_df, output_dir=None, timestamp="t"
            )
            assert "reports" in result.lower() or "report" in result.lower()

    def test_logging_on_success(self, four_weeks_data, tmp_path, caplog):
        with caplog.at_level(logging.INFO):
            generate_weekly_report(
                four_weeks_data, output_dir=str(tmp_path), timestamp="t"
            )
        assert "weekly report" in caplog.text.lower() or "saved" in caplog.text.lower()

    def test_logging_on_empty(self, tmp_path, caplog):
        df = pd.DataFrame(columns=["day", "score"])
        with caplog.at_level(logging.WARNING):
            generate_weekly_report(df, output_dir=str(tmp_path), timestamp="t")
        assert "no weekly data" in caplog.text.lower() or caplog.text == "" or "warning" in caplog.text.lower()

    def test_deterministic_deltas_in_output(self, two_weeks_deterministic, tmp_path):
        path = generate_weekly_report(
            two_weeks_deterministic, output_dir=str(tmp_path), timestamp="t"
        )
        content = Path(path).read_text()
        assert "+10.0" in content    # sleep score delta
        assert "-5.0" in content     # rhr delta
        assert "+700" in content     # stress delta


# ---------------------------------------------------------------------------
# Tests – CLI entry-point
# ---------------------------------------------------------------------------

class TestCLI:
    """Tests for cli_weekly_report.main()."""

    def _mock_df(self):
        return pd.DataFrame({
            "day": pd.date_range("2024-06-03", periods=28),
            "score": [75.0] * 28,
            "resting_heart_rate": [60.0] * 28,
            "stress_duration": [200] * 28,
        })

    @patch("garmin_analysis.cli_weekly_report.load_master_dataframe")
    @patch("garmin_analysis.cli_weekly_report.generate_weekly_report")
    @patch("garmin_analysis.cli_weekly_report.apply_24h_coverage_filter_from_args")
    def test_main_default_args(self, mock_filter, mock_gen, mock_load):
        mock_load.return_value = self._mock_df()
        mock_filter.side_effect = lambda df, args: df
        mock_gen.return_value = "/tmp/report.md"

        with patch("sys.argv", ["prog"]):
            from garmin_analysis.cli_weekly_report import main
            main()

        mock_load.assert_called_once()
        mock_filter.assert_called_once()
        mock_gen.assert_called_once()
        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs["num_weeks"] == 4
        assert call_kwargs["output_dir"] is None  # default "plots" → None

    @patch("garmin_analysis.cli_weekly_report.load_master_dataframe")
    @patch("garmin_analysis.cli_weekly_report.generate_weekly_report")
    @patch("garmin_analysis.cli_weekly_report.apply_24h_coverage_filter_from_args")
    def test_main_custom_weeks(self, mock_filter, mock_gen, mock_load):
        mock_load.return_value = self._mock_df()
        mock_filter.side_effect = lambda df, args: df
        mock_gen.return_value = "/tmp/report.md"

        with patch("sys.argv", ["prog", "--weeks", "8"]):
            from garmin_analysis.cli_weekly_report import main
            main()

        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs["num_weeks"] == 8

    @patch("garmin_analysis.cli_weekly_report.load_master_dataframe")
    @patch("garmin_analysis.cli_weekly_report.generate_weekly_report")
    @patch("garmin_analysis.cli_weekly_report.apply_24h_coverage_filter_from_args")
    def test_main_custom_output_dir(self, mock_filter, mock_gen, mock_load):
        mock_load.return_value = self._mock_df()
        mock_filter.side_effect = lambda df, args: df
        mock_gen.return_value = "/tmp/report.md"

        with patch("sys.argv", ["prog", "--output-dir", "/tmp/custom"]):
            from garmin_analysis.cli_weekly_report import main
            main()

        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs["output_dir"] == "/tmp/custom"

    @patch("garmin_analysis.cli_weekly_report.load_master_dataframe")
    @patch("garmin_analysis.cli_weekly_report.generate_weekly_report")
    @patch("garmin_analysis.cli_weekly_report.apply_24h_coverage_filter_from_args")
    def test_main_empty_report(self, mock_filter, mock_gen, mock_load, caplog):
        mock_load.return_value = self._mock_df()
        mock_filter.side_effect = lambda df, args: df
        mock_gen.return_value = ""

        with patch("sys.argv", ["prog"]), caplog.at_level(logging.WARNING):
            from garmin_analysis.cli_weekly_report import main
            main()

    @patch("garmin_analysis.cli_weekly_report.load_master_dataframe")
    @patch("garmin_analysis.cli_weekly_report.generate_weekly_report")
    @patch("garmin_analysis.cli_weekly_report.apply_24h_coverage_filter_from_args")
    def test_main_coverage_filter_called(self, mock_filter, mock_gen, mock_load):
        mock_load.return_value = self._mock_df()
        mock_filter.side_effect = lambda df, args: df
        mock_gen.return_value = "/tmp/report.md"

        with patch("sys.argv", ["prog", "--filter-24h-coverage"]):
            from garmin_analysis.cli_weekly_report import main
            main()

        args_passed = mock_filter.call_args[0][1]
        assert args_passed.filter_24h_coverage is True


# ---------------------------------------------------------------------------
# Tests – edge cases & boundary conditions
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_year_boundary_dec_to_jan(self, tmp_path):
        """Data crossing Dec → Jan should not crash and should group correctly."""
        dates = pd.date_range("2024-12-28", periods=14, freq="D")
        df = pd.DataFrame({
            "day": dates,
            "score": [75.0] * 14,
            "stress_duration": [200] * 14,
        })
        path = generate_weekly_report(df, output_dir=str(tmp_path), timestamp="t")
        content = Path(path).read_text()
        assert Path(path).exists()
        assert "Weekly Health Report" in content

    def test_all_zeros_stress(self, tmp_path):
        dates = pd.date_range("2024-06-03", periods=14, freq="D")
        df = pd.DataFrame({
            "day": dates,
            "stress_duration": [0] * 14,
        })
        path = generate_weekly_report(df, output_dir=str(tmp_path), timestamp="t")
        content = Path(path).read_text()
        assert "0 min" in content

    def test_negative_stress_values(self, tmp_path):
        """Negative stress_duration shouldn't crash (even if semantically invalid)."""
        dates = pd.date_range("2024-06-03", periods=7, freq="D")
        df = pd.DataFrame({"day": dates, "stress_duration": [-10] * 7})
        path = generate_weekly_report(df, output_dir=str(tmp_path), timestamp="t")
        assert Path(path).exists()

    def test_very_large_values(self, tmp_path):
        dates = pd.date_range("2024-06-03", periods=7, freq="D")
        df = pd.DataFrame({
            "day": dates,
            "score": [99999.9] * 7,
            "resting_heart_rate": [200.0] * 7,
            "stress_duration": [100000] * 7,
        })
        path = generate_weekly_report(df, output_dir=str(tmp_path), timestamp="t")
        assert Path(path).exists()

    def test_exactly_min_days_per_week(self, tmp_path):
        """A week with exactly MIN_DAYS_FOR_WEEK days should be included."""
        dates = pd.date_range("2024-06-03", periods=MIN_DAYS_FOR_WEEK, freq="D")
        df = pd.DataFrame({"day": dates, "score": [80.0] * MIN_DAYS_FOR_WEEK})
        weekly = compute_weekly_aggregates(df)
        assert weekly["sleep_score_count"].iloc[0] == MIN_DAYS_FOR_WEEK
        path = generate_weekly_report(df, output_dir=str(tmp_path), timestamp="t")
        content = Path(path).read_text()
        assert "80.0" in content

    def test_one_below_min_days_per_week(self, tmp_path):
        """A week with MIN_DAYS_FOR_WEEK - 1 days should show '--'."""
        count = MIN_DAYS_FOR_WEEK - 1
        dates = pd.date_range("2024-06-03", periods=count, freq="D")
        df = pd.DataFrame({"day": dates, "score": [80.0] * count})
        path = generate_weekly_report(df, output_dir=str(tmp_path), timestamp="t")
        content = Path(path).read_text()
        assert "| -- | -- | -- |" in content

    def test_duplicate_dates_in_input(self, tmp_path):
        dates = list(pd.date_range("2024-06-03", periods=7, freq="D")) * 2
        df = pd.DataFrame({"day": dates, "score": [75.0] * 14})
        path = generate_weekly_report(df, output_dir=str(tmp_path), timestamp="t")
        assert Path(path).exists()

    def test_unsorted_input_produces_valid_report(self, tmp_path):
        dates = pd.date_range("2024-06-03", periods=14, freq="D")
        df = pd.DataFrame({
            "day": dates,
            "score": list(range(60, 74)),
            "resting_heart_rate": list(range(55, 69)),
            "stress_duration": list(range(100, 114)),
        })
        df_shuffled = df.sample(frac=1, random_state=99)
        path = generate_weekly_report(df_shuffled, output_dir=str(tmp_path), timestamp="t")
        content = Path(path).read_text()
        assert "Weekly Health Report" in content
        assert "### Sleep Score Trend" in content

    def test_single_day_of_data(self, tmp_path):
        df = pd.DataFrame({
            "day": [pd.Timestamp("2024-06-05")],
            "score": [82.0],
        })
        path = generate_weekly_report(df, output_dir=str(tmp_path), timestamp="t")
        content = Path(path).read_text()
        assert "| -- | -- | -- |" in content  # 1 day < MIN_DAYS_FOR_WEEK

    def test_all_nan_in_every_column(self, tmp_path):
        dates = pd.date_range("2024-06-03", periods=14, freq="D")
        df = pd.DataFrame({
            "day": dates,
            "score": [np.nan] * 14,
            "resting_heart_rate": [np.nan] * 14,
            "stress_duration": [np.nan] * 14,
        })
        path = generate_weekly_report(df, output_dir=str(tmp_path), timestamp="t")
        content = Path(path).read_text()
        assert Path(path).exists()
        assert "Week at a glance" not in content

    def test_mixed_valid_and_nan_weeks(self, tmp_path):
        """Week 1 has all valid data, week 2 has all NaN scores."""
        dates = pd.date_range("2024-06-03", periods=14, freq="D")
        scores = [75.0] * 7 + [np.nan] * 7
        df = pd.DataFrame({"day": dates, "score": scores})
        path = generate_weekly_report(df, output_dir=str(tmp_path), timestamp="t")
        content = Path(path).read_text()
        assert "75.0" in content

    def test_many_weeks(self, tmp_path):
        """90 days (~13 weeks) with num_weeks=4 should only show last 4."""
        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        df = pd.DataFrame({
            "day": dates,
            "score": np.random.RandomState(0).randint(60, 90, 90).astype(float),
        })
        path = generate_weekly_report(
            df, num_weeks=4, output_dir=str(tmp_path), timestamp="t"
        )
        content = Path(path).read_text()
        assert "Weeks shown: 4" in content

    def test_no_metric_columns_at_all(self, tmp_path):
        """DataFrame with only 'day' and unrelated columns."""
        dates = pd.date_range("2024-06-03", periods=14, freq="D")
        df = pd.DataFrame({"day": dates, "steps": range(14)})
        path = generate_weekly_report(df, output_dir=str(tmp_path), timestamp="t")
        content = Path(path).read_text()
        assert content.count("No data available") == 3

    def test_report_file_is_valid_utf8(self, four_weeks_data, tmp_path):
        path = generate_weekly_report(
            four_weeks_data, output_dir=str(tmp_path), timestamp="t"
        )
        content = Path(path).read_text(encoding="utf-8")
        assert len(content) > 0

    def test_report_contains_no_html(self, four_weeks_data, tmp_path):
        """Markdown report should not contain raw HTML tags."""
        path = generate_weekly_report(
            four_weeks_data, output_dir=str(tmp_path), timestamp="t"
        )
        content = Path(path).read_text()
        assert "<table" not in content
        assert "<div" not in content

    def test_markdown_table_alignment_separators(self, four_weeks_data, tmp_path):
        """Every table should have proper markdown separator rows."""
        path = generate_weekly_report(
            four_weeks_data, output_dir=str(tmp_path), timestamp="t"
        )
        content = Path(path).read_text()
        # Three metric tables, each with a separator row
        assert content.count("|---") >= 3


# ---------------------------------------------------------------------------
# Tests – integration (full round-trip with deterministic assertions)
# ---------------------------------------------------------------------------

class TestIntegration:
    @pytest.mark.integration
    def test_full_round_trip(self, two_weeks_deterministic, tmp_path):
        """End-to-end: build weekly aggregates → write report → verify content."""
        path = generate_weekly_report(
            two_weeks_deterministic, output_dir=str(tmp_path), timestamp="int_test"
        )
        content = Path(path).read_text()

        # Header
        assert "# Weekly Health Report" in content
        assert "Generated: int_test" in content
        assert "Weeks shown: 2" in content
        assert "Covering: 2024-07-01 to 2024-07-14" in content

        # Summary paragraph
        assert "Week at a glance" in content
        assert "80.0" in content         # sleep current
        assert "+10.0" in content        # sleep delta
        assert "55.0 bpm" in content     # rhr current
        assert "-5.0 bpm" in content     # rhr delta
        assert "2100 min" in content     # stress current
        assert "+700" in content         # stress delta

        # Sleep section
        assert "### Sleep Score Trend" in content
        assert "70.0" in content
        assert "80.0" in content

        # RHR section
        assert "### Resting Heart Rate" in content
        assert "60.0" in content
        assert "55.0" in content

        # Stress section
        assert "### Stress Minutes" in content
        assert "1400 min" in content
        assert "2100 min" in content

    @pytest.mark.integration
    def test_trend_directions_correct(self, two_weeks_deterministic, tmp_path):
        """Verify that trend signs are semantically correct for each metric."""
        path = generate_weekly_report(
            two_weeks_deterministic, output_dir=str(tmp_path), timestamp="t"
        )
        content = Path(path).read_text()

        sleep_section = content.split("### Sleep Score Trend")[1].split("###")[0]
        rhr_section = content.split("### Resting Heart Rate")[1].split("###")[0]
        stress_section = content.split("### Stress Minutes")[1]

        # Sleep went up (70→80): improvement for sleep
        assert "| + |" in sleep_section

        # RHR went down (60→55): improvement for RHR (lower_is_better)
        assert "| + |" in rhr_section

        # Stress went up (1400→2100): worsening for stress
        assert "| - |" in stress_section

    @pytest.mark.integration
    def test_three_week_delta_chain(self, tmp_path):
        """Three weeks with known values — verify cascading deltas."""
        dates = pd.date_range("2024-06-03", periods=21, freq="D")
        scores = [60.0] * 7 + [70.0] * 7 + [65.0] * 7
        df = pd.DataFrame({"day": dates, "score": scores})

        path = generate_weekly_report(df, output_dir=str(tmp_path), timestamp="t")
        content = Path(path).read_text()
        section = content.split("### Sleep Score Trend")[1].split("###")[0]

        assert "+10.0" in section   # W24: 70 - 60
        assert "-5.0" in section    # W25: 65 - 70


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
