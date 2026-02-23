"""
Weekly Health Report Generator.

Produces a Markdown report comparing the current week to the prior week
across three key metrics:
  - Sleep score trend (weekly average + direction)
  - Resting heart rate delta
  - Stress minutes delta
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from garmin_analysis.config import REPORTS_DIR
from garmin_analysis.utils.data_loading import load_master_dataframe

logger = logging.getLogger(__name__)

SLEEP_COL = "score"
RHR_PRIMARY = "resting_heart_rate"
RHR_FALLBACK = "rhr"
STRESS_COL = "stress_duration"
DATE_COL = "day"

MIN_DAYS_FOR_WEEK = 3


def _resolve_rhr_column(df: pd.DataFrame) -> Optional[str]:
    """Pick the best available resting-HR column."""
    for col in (RHR_PRIMARY, RHR_FALLBACK):
        if col in df.columns and df[col].notna().any():
            return col
    return None


def _arrow(delta: float) -> str:
    if delta > 0:
        return "^"
    if delta < 0:
        return "v"
    return "="


def _direction_word(delta: float) -> str:
    if delta > 0:
        return "up"
    if delta < 0:
        return "down"
    return "flat"


def _format_delta(value: float, precision: int = 1) -> str:
    """Format a numeric delta with a sign prefix."""
    if value >= 0:
        return f"+{value:.{precision}f}"
    return f"{value:.{precision}f}"


def compute_weekly_aggregates(
    df: pd.DataFrame,
    num_weeks: int = 4,
) -> pd.DataFrame:
    """
    Aggregate daily data into ISO-week buckets.

    Returns a DataFrame indexed by ``(iso_year, iso_week)`` with columns:
      sleep_score_avg, sleep_score_count,
      rhr_avg, rhr_count,
      stress_minutes_total, stress_minutes_count,
      week_start, week_end
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)

    df["iso_year"] = df[DATE_COL].dt.isocalendar().year.astype(int)
    df["iso_week"] = df[DATE_COL].dt.isocalendar().week.astype(int)

    rhr_col = _resolve_rhr_column(df)

    agg_dict: dict = {
        "week_start": (DATE_COL, "min"),
        "week_end": (DATE_COL, "max"),
    }

    if SLEEP_COL in df.columns:
        agg_dict["sleep_score_avg"] = (SLEEP_COL, "mean")
        agg_dict["sleep_score_count"] = (SLEEP_COL, "count")
    if rhr_col:
        agg_dict["rhr_avg"] = (rhr_col, "mean")
        agg_dict["rhr_count"] = (rhr_col, "count")
    if STRESS_COL in df.columns:
        agg_dict["stress_minutes_total"] = (STRESS_COL, "sum")
        agg_dict["stress_minutes_count"] = (STRESS_COL, "count")

    weekly = (
        df.groupby(["iso_year", "iso_week"])
        .agg(**agg_dict)
        .sort_index()
    )

    if num_weeks and len(weekly) > num_weeks:
        weekly = weekly.iloc[-num_weeks:]

    return weekly


def _metric_section(
    weekly: pd.DataFrame,
    col_avg: str,
    col_count: str,
    title: str,
    unit: str,
    precision: int = 1,
    lower_is_better: bool = False,
) -> str:
    """Build a markdown section for one metric comparing successive weeks."""
    if col_avg not in weekly.columns:
        return f"### {title}\n\nNo data available.\n\n"

    lines = [f"### {title}\n"]
    lines.append(f"| Week | Avg | Delta | Trend | Days |")
    lines.append(f"|------|-----|-------|-------|------|")

    prev_val: Optional[float] = None
    for (iso_year, iso_week), row in weekly.iterrows():
        avg = row[col_avg]
        count = int(row[col_count])

        if pd.isna(avg) or count < MIN_DAYS_FOR_WEEK:
            lines.append(f"| {iso_year}-W{iso_week:02d} | -- | -- | -- | {count} |")
            prev_val = None
            continue

        if prev_val is not None:
            delta = avg - prev_val
            delta_str = _format_delta(delta, precision)
            improved = (delta < 0) if lower_is_better else (delta > 0)
            trend = "+" if improved else ("-" if not improved and delta != 0 else "=")
        else:
            delta_str = "--"
            trend = "--"

        lines.append(
            f"| {iso_year}-W{iso_week:02d} "
            f"| {avg:.{precision}f} {unit} "
            f"| {delta_str} "
            f"| {trend} "
            f"| {count} |"
        )
        prev_val = avg

    lines.append("")
    return "\n".join(lines) + "\n"


def _stress_section(weekly: pd.DataFrame) -> str:
    """Build the stress-minutes section (uses totals, not averages)."""
    col_total = "stress_minutes_total"
    col_count = "stress_minutes_count"

    if col_total not in weekly.columns:
        return "### Stress Minutes\n\nNo data available.\n\n"

    lines = ["### Stress Minutes\n"]
    lines.append("| Week | Total min | Delta | Trend | Days |")
    lines.append("|------|-----------|-------|-------|------|")

    prev_val: Optional[float] = None
    for (iso_year, iso_week), row in weekly.iterrows():
        total = row[col_total]
        count = int(row[col_count])

        if pd.isna(total) or count < MIN_DAYS_FOR_WEEK:
            lines.append(f"| {iso_year}-W{iso_week:02d} | -- | -- | -- | {count} |")
            prev_val = None
            continue

        total_int = int(round(total))
        if prev_val is not None:
            delta = total_int - prev_val
            delta_str = _format_delta(delta, 0)
            trend = "+" if delta < 0 else ("-" if delta > 0 else "=")
        else:
            delta_str = "--"
            trend = "--"

        lines.append(
            f"| {iso_year}-W{iso_week:02d} "
            f"| {total_int} min "
            f"| {delta_str} "
            f"| {trend} "
            f"| {count} |"
        )
        prev_val = total_int

    lines.append("")
    return "\n".join(lines) + "\n"


def _latest_week_summary(weekly: pd.DataFrame) -> str:
    """One-paragraph narrative of the most recent complete week vs prior."""
    if len(weekly) < 2:
        return ""

    curr = weekly.iloc[-1]
    prev = weekly.iloc[-2]
    parts: list[str] = []

    if "sleep_score_avg" in weekly.columns and "sleep_score_count" in weekly.columns:
        s_curr, s_prev = curr.get("sleep_score_avg"), prev.get("sleep_score_avg")
        c_curr, c_prev = curr.get("sleep_score_count", 0), prev.get("sleep_score_count", 0)
        if pd.notna(s_curr) and pd.notna(s_prev) and c_curr >= MIN_DAYS_FOR_WEEK and c_prev >= MIN_DAYS_FOR_WEEK:
            d = s_curr - s_prev
            parts.append(
                f"Sleep score averaged **{s_curr:.1f}** "
                f"({_format_delta(d, 1)} vs prior week, {_direction_word(d)})"
            )

    if "rhr_avg" in weekly.columns and "rhr_count" in weekly.columns:
        r_curr, r_prev = curr.get("rhr_avg"), prev.get("rhr_avg")
        c_curr, c_prev = curr.get("rhr_count", 0), prev.get("rhr_count", 0)
        if pd.notna(r_curr) and pd.notna(r_prev) and c_curr >= MIN_DAYS_FOR_WEEK and c_prev >= MIN_DAYS_FOR_WEEK:
            d = r_curr - r_prev
            parts.append(
                f"Resting HR averaged **{r_curr:.1f} bpm** "
                f"({_format_delta(d, 1)} bpm)"
            )

    if "stress_minutes_total" in weekly.columns and "stress_minutes_count" in weekly.columns:
        st_curr, st_prev = curr.get("stress_minutes_total"), prev.get("stress_minutes_total")
        c_curr, c_prev = curr.get("stress_minutes_count", 0), prev.get("stress_minutes_count", 0)
        if pd.notna(st_curr) and pd.notna(st_prev) and c_curr >= MIN_DAYS_FOR_WEEK and c_prev >= MIN_DAYS_FOR_WEEK:
            d = int(round(st_curr - st_prev))
            parts.append(
                f"Stress logged **{int(round(st_curr))} min** total "
                f"({_format_delta(d, 0)} min)"
            )

    if not parts:
        return ""
    return "**Week at a glance:** " + ". ".join(parts) + ".\n\n"


def generate_weekly_report(
    df: Optional[pd.DataFrame] = None,
    num_weeks: int = 4,
    output_dir: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    """
    Build and write a weekly health report as Markdown.

    Args:
        df: Master daily DataFrame. Loaded automatically if None.
        num_weeks: Number of recent weeks to include (default 4).
        output_dir: Directory for the output file. Defaults to REPORTS_DIR.
        timestamp: Optional timestamp string for the filename.

    Returns:
        Absolute path to the written report file.
    """
    if df is None:
        df = load_master_dataframe()

    if output_dir is None:
        output_dir = str(REPORTS_DIR)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    weekly = compute_weekly_aggregates(df, num_weeks=num_weeks)

    if weekly.empty:
        logger.warning("No weekly data to report.")
        return ""

    report_file = out_path / f"weekly_report_{timestamp}.md"

    week_range = (
        f"{weekly['week_start'].iloc[0].strftime('%Y-%m-%d')} to "
        f"{weekly['week_end'].iloc[-1].strftime('%Y-%m-%d')}"
    )

    with open(report_file, "w") as f:
        f.write("# Weekly Health Report\n\n")
        f.write(f"Generated: {timestamp}  \n")
        f.write(f"Covering: {week_range}  \n")
        f.write(f"Weeks shown: {len(weekly)}\n\n")
        f.write("---\n\n")

        f.write("## Summary\n\n")
        f.write(_latest_week_summary(weekly))

        f.write("## Metrics\n\n")

        f.write(
            _metric_section(
                weekly,
                col_avg="sleep_score_avg",
                col_count="sleep_score_count",
                title="Sleep Score Trend",
                unit="",
                precision=1,
                lower_is_better=False,
            )
        )

        f.write(
            _metric_section(
                weekly,
                col_avg="rhr_avg",
                col_count="rhr_count",
                title="Resting Heart Rate",
                unit="bpm",
                precision=1,
                lower_is_better=True,
            )
        )

        f.write(_stress_section(weekly))

    logger.info("Weekly report saved to %s", report_file)
    return str(report_file)
