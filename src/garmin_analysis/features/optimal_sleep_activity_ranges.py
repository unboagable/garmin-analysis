"""
Optimal Sleep Activity Ranges Analysis

Identifies the steps and intensity minutes ranges associated with your best sleep.
Output: "Your best sleep occurs when steps are between A–B or intensity minutes are between A–B"
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from garmin_analysis.config import PLOTS_DIR

logger = logging.getLogger(__name__)


def _to_minutes(series: pd.Series) -> pd.Series:
    """Convert timedelta or time string to minutes."""
    if series.empty:
        return series
    try:
        if series.dtype == "object" or "timedelta" in str(series.dtype):
            return pd.to_timedelta(series, errors="coerce").dt.total_seconds() / 60
        return series
    except Exception:
        return series


def _get_intensity_column(df: pd.DataFrame) -> Optional[str]:
    """Resolve intensity minutes from available columns."""
    if "intensity_time" in df.columns:
        return "intensity_time"
    if "moderate_activity_time" in df.columns and "vigorous_activity_time" in df.columns:
        return "_computed_intensity"
    if "yesterday_activity_minutes" in df.columns:
        return "yesterday_activity_minutes"
    return None


def _ensure_intensity_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Add _computed_intensity if needed (moderate + vigorous)."""
    if "_computed_intensity" in df.columns:
        return df
    if "moderate_activity_time" in df.columns and "vigorous_activity_time" in df.columns:
        mod = _to_minutes(df["moderate_activity_time"])
        vig = _to_minutes(df["vigorous_activity_time"])
        df = df.copy()
        df["_computed_intensity"] = mod.fillna(0) + vig.fillna(0)
    return df


def _bin_and_aggregate(
    df: pd.DataFrame,
    value_col: str,
    score_col: str,
    n_bins: int = 5,
    min_samples: int = 3,
) -> pd.DataFrame:
    """Bin values and compute mean sleep score per bin."""
    valid = df[[value_col, score_col]].dropna()
    if len(valid) < min_samples:
        return pd.DataFrame()

    vals = valid[value_col].values
    try:
        edges = np.percentile(vals, np.linspace(0, 100, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    valid = valid.copy()
    valid["bin"] = pd.cut(
        valid[value_col],
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    )
    agg = valid.groupby("bin", observed=True)[score_col].agg(["mean", "count", "std"]).reset_index()
    agg["bin_low"] = agg["bin"].apply(lambda x: x.left)
    agg["bin_high"] = agg["bin"].apply(lambda x: x.right)
    return agg


def compute_optimal_sleep_ranges(
    df: pd.DataFrame,
    score_col: str = "score",
    steps_col: str = "steps",
    n_bins: int = 5,
    min_samples: int = 3,
) -> dict:
    """
    Compute the steps and intensity minutes ranges associated with best sleep.

    Args:
        df: Master DataFrame with day, score (sleep), steps, and intensity columns.
        score_col: Column name for sleep score.
        steps_col: Column name for daily steps.
        n_bins: Number of bins for discretization.
        min_samples: Minimum samples per bin to consider.

    Returns:
        Dictionary with:
          - steps_range: (low, high) for best sleep by steps, or None
          - intensity_range: (low, high) for best sleep by intensity, or None
          - steps_summary: per-bin stats for steps
          - intensity_summary: per-bin stats for intensity
          - message: Human-readable summary
    """
    result = {
        "steps_range": None,
        "intensity_range": None,
        "steps_summary": None,
        "intensity_summary": None,
        "message": "",
    }

    if df.empty or score_col not in df.columns:
        result["message"] = "No sleep score data available."
        return result

    messages = []

    # Steps analysis
    if steps_col in df.columns:
        steps_agg = _bin_and_aggregate(df, steps_col, score_col, n_bins, min_samples)
        if not steps_agg.empty and steps_agg["count"].max() >= min_samples:
            best_idx = steps_agg["mean"].idxmax()
            best = steps_agg.loc[best_idx]
            low, high = best["bin_low"], best["bin_high"]
            result["steps_range"] = (float(low), float(high))
            result["steps_summary"] = steps_agg
            a, b = int(round(low)), int(round(high))
            messages.append(f"Your best sleep occurs when steps are between {a:,}–{b:,}")
        else:
            messages.append("Insufficient steps data to determine optimal range.")
    else:
        messages.append("Steps column not found in dataset.")

    # Intensity minutes analysis
    df_int = _ensure_intensity_minutes(df)
    intensity_col = _get_intensity_column(df_int)
    if intensity_col:
        int_agg = _bin_and_aggregate(df_int, intensity_col, score_col, n_bins, min_samples)
        if not int_agg.empty and int_agg["count"].max() >= min_samples:
            best_idx = int_agg["mean"].idxmax()
            best = int_agg.loc[best_idx]
            low, high = best["bin_low"], best["bin_high"]
            result["intensity_range"] = (float(low), float(high))
            result["intensity_summary"] = int_agg
            a, b = int(round(low)), int(round(high))
            messages.append(f"Your best sleep occurs when intensity minutes are between {a}–{b}")
        else:
            messages.append("Insufficient intensity minutes data to determine optimal range.")
    else:
        messages.append("Intensity minutes column not found in dataset.")

    result["message"] = " ".join(messages)
    return result


def plot_optimal_sleep_ranges(
    df: pd.DataFrame,
    save_plots: bool = True,
    show_plots: bool = False,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Create visualizations of sleep score by steps and intensity minutes.

    Returns:
        Dictionary mapping plot names to file paths.
    """
    if output_dir is None:
        output_dir = PLOTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = compute_optimal_sleep_ranges(df)
    plot_files = {}
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.style.use("default")
    sns.set_palette("husl")

    # Steps vs sleep score
    if result["steps_summary"] is not None and not result["steps_summary"].empty:
        agg = result["steps_summary"]
        fig, ax = plt.subplots(figsize=(10, 6))
        x_labels = [f"{int(lo):,}–{int(hi):,}" for lo, hi in zip(agg["bin_low"], agg["bin_high"])]
        bars = ax.bar(range(len(agg)), agg["mean"], color=sns.color_palette("husl", len(agg)))
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels(x_labels, rotation=30, ha="right")
        ax.set_xlabel("Steps (daily)")
        ax.set_ylabel("Mean Sleep Score")
        ax.set_title("Sleep Score by Steps Range")
        for bar, n in zip(bars, agg["count"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"n={int(n)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        plt.tight_layout()
        if save_plots:
            path = output_dir / f"{timestamp_str}_optimal_sleep_steps.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plot_files["steps"] = str(path)
        if show_plots:
            plt.show()
        else:
            plt.close()

    # Intensity vs sleep score
    if result["intensity_summary"] is not None and not result["intensity_summary"].empty:
        agg = result["intensity_summary"]
        fig, ax = plt.subplots(figsize=(10, 6))
        x_labels = [f"{int(lo)}–{int(hi)} min" for lo, hi in zip(agg["bin_low"], agg["bin_high"])]
        bars = ax.bar(range(len(agg)), agg["mean"], color=sns.color_palette("husl", len(agg)))
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels(x_labels, rotation=30, ha="right")
        ax.set_xlabel("Intensity Minutes")
        ax.set_ylabel("Mean Sleep Score")
        ax.set_title("Sleep Score by Intensity Minutes Range")
        for bar, n in zip(bars, agg["count"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"n={int(n)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        plt.tight_layout()
        if save_plots:
            path = output_dir / f"{timestamp_str}_optimal_sleep_intensity.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plot_files["intensity"] = str(path)
        if show_plots:
            plt.show()
        else:
            plt.close()

    return plot_files


def print_optimal_sleep_summary(df: pd.DataFrame) -> None:
    """Print human-readable summary of optimal sleep activity ranges."""
    result = compute_optimal_sleep_ranges(df)
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMAL SLEEP ACTIVITY RANGES")
    logger.info("=" * 60)
    logger.info(result["message"])
    logger.info("=" * 60)


def get_optimal_sleep_plotly_figures(df: pd.DataFrame) -> Tuple[dict, Optional[object], Optional[object]]:
    """
    Return optimal sleep analysis results for dashboard use.

    Returns:
        Tuple of (result dict, plotly steps figure, plotly intensity figure).
        Figures are None if no data.
    """
    import plotly.graph_objects as go

    result = compute_optimal_sleep_ranges(df)
    steps_fig = None
    intensity_fig = None

    if result["steps_summary"] is not None and not result["steps_summary"].empty:
        agg = result["steps_summary"]
        x_labels = [f"{int(lo):,}–{int(hi):,}" for lo, hi in zip(agg["bin_low"], agg["bin_high"])]
        steps_fig = go.Figure(
            data=[go.Bar(x=x_labels, y=agg["mean"], text=agg["count"].astype(int), textposition="outside")],
            layout=go.Layout(
                title="Sleep Score by Steps Range",
                xaxis_title="Steps (daily)",
                yaxis_title="Mean Sleep Score",
                template="plotly_white",
            ),
        )

    if result["intensity_summary"] is not None and not result["intensity_summary"].empty:
        agg = result["intensity_summary"]
        x_labels = [f"{int(lo)}–{int(hi)} min" for lo, hi in zip(agg["bin_low"], agg["bin_high"])]
        intensity_fig = go.Figure(
            data=[go.Bar(x=x_labels, y=agg["mean"], text=agg["count"].astype(int), textposition="outside")],
            layout=go.Layout(
                title="Sleep Score by Intensity Minutes Range",
                xaxis_title="Intensity Minutes",
                yaxis_title="Mean Sleep Score",
                template="plotly_white",
            ),
        )

    return result, steps_fig, intensity_fig


def main() -> None:
    """Main function to run optimal sleep activity ranges analysis."""
    from garmin_analysis.utils.data_loading import load_master_dataframe

    try:
        logger.info("Loading master daily summary data...")
        df = load_master_dataframe()
        if df.empty:
            logger.error("No data loaded")
            return
        logger.info("Loaded %d days of data", len(df))
        print_optimal_sleep_summary(df)
        logger.info("\nGenerating visualizations...")
        plot_files = plot_optimal_sleep_ranges(df, save_plots=True, show_plots=False)
        if plot_files:
            logger.info("Generated %d plots", len(plot_files))
            for name, path in plot_files.items():
                logger.info("  %s: %s", name, path)
    except Exception as e:
        logger.exception("Error in optimal sleep activity ranges analysis: %s", e)
        raise


if __name__ == "__main__":
    from garmin_analysis.logging_config import setup_logging

    setup_logging(level=logging.INFO)
    main()
