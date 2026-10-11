# -*- coding: utf-8 -*-
"""
@Time    : 2026/7/5 12:23
@Author  : HuangZhiyong
@email   : 1524338616@qq.com
@File    : 01_cluster_over_time.py
@Function:
    Compare cluster cumulative evolution curves from the original 24 h DEC
    result and the newly added per-event different-duration DEC experiments.

    One figure is generated for each event. Each row corresponds to one
    duration window, and each line is the cumulative count or cumulative
    fraction of one cluster inside that duration.

    Important:
        Different duration windows are clustered independently, so cluster
        labels are comparable within a row only. A label such as "Cluster 2"
        in 1 h is not assumed to be the same physical state as "Cluster 2"
        in 24 h.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# 1. Path and plotting configuration
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben")
CLUSTERING_ROOT = PROJECT_ROOT / "uilt_oneday_all" / "02clustering"

# Original full-day DEC result.
ORIGINAL_24H_NPZ_DIR = (
    CLUSTERING_ROOT
    / "result_all_event_DEC_K6"
    / "DEC_result"
    / "all_parement"
)

# Newly added per-event DEC result for different duration windows.
DIFFERENT_DURATION_ROOT = CLUSTERING_ROOT / "result_different_duration_DEC_per_event"

OUTPUT_ROOT = SCRIPT_DIR / "cluster_over_time_comparison"

# None means all events found from the original 24 h folder.
# Example: EVENT_NAMES = ["event1", "event7", "event20"]
EVENT_NAMES: list[str] | None = None

# None means all available duration_*h folders plus the original 24 h result.
# Example: DURATION_HOURS = [1, 2, 4, 8, 12, 18, 24]
DURATION_HOURS: list[int] | None = None

# Use absolute cumulative counts to compare the original number of classified
# windows across duration experiments.
NORMALIZE_CUMULATIVE = False

# Use the same x-axis for every duration window to compare whether short-window
# precursor identification overlaps with the original 24 h result.
COMMON_X_WINDOW_HOURS = 24

# Save a compact CSV for each event-duration-cluster curve.
SAVE_CURVE_CSV = True

FIG_DPI = 300
MAX_LEGEND_COLUMNS = 8

plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "Arial",
        "axes.unicode_minus": False,
    }
)
plt.ioff()


# =========================================================
# 2. Data helpers
# =========================================================
@dataclass
class ClusterTimeSeries:
    event: str
    duration_hours: int
    npz_path: Path
    time: np.ndarray
    hours_to_end: np.ndarray
    labels: np.ndarray
    n_clusters: int


def natural_event_key(value: str) -> int:
    match = re.search(r"event(\d+)", str(value), flags=re.IGNORECASE)
    return int(match.group(1)) if match else 10**9


def duration_key(duration_hours: int) -> tuple[int, int]:
    # Put the original 24 h baseline after the shorter supplementary windows.
    return (1 if int(duration_hours) == 24 else 0, int(duration_hours))


def parse_duration_from_dir_name(path: Path) -> int | None:
    match = re.search(r"duration_(\d+)h", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_time_from_wave_file(wave_file: object) -> datetime | None:
    match = re.search(r"(\d{8}T\d{6})", str(wave_file))
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")


def infer_event_name_from_npz(path: Path) -> str:
    match = re.search(r"event\d+", path.name, flags=re.IGNORECASE)
    if match:
        return match.group(0).lower()
    return path.stem.replace("_all_parement", "").lower()


def discover_event_names() -> list[str]:
    if EVENT_NAMES is not None:
        return sorted([item.lower() for item in EVENT_NAMES], key=natural_event_key)

    event_files = sorted(
        ORIGINAL_24H_NPZ_DIR.glob("event*_all_parement.npz"),
        key=lambda item: natural_event_key(item.name),
    )
    if not event_files:
        raise FileNotFoundError(f"No original 24 h event npz files found: {ORIGINAL_24H_NPZ_DIR}")

    return [infer_event_name_from_npz(path) for path in event_files]


def discover_duration_hours() -> list[int]:
    if DURATION_HOURS is not None:
        durations = [int(item) for item in DURATION_HOURS]
    else:
        durations = []
        for path in DIFFERENT_DURATION_ROOT.glob("duration_*h"):
            duration = parse_duration_from_dir_name(path)
            if duration is not None:
                durations.append(duration)
        durations.append(24)

    durations = sorted(set(durations), key=duration_key)
    return durations


def get_npz_path(event_name: str, duration_hours: int) -> Path:
    if int(duration_hours) == 24:
        return ORIGINAL_24H_NPZ_DIR / f"{event_name}_all_parement.npz"

    return (
        DIFFERENT_DURATION_ROOT
        / f"duration_{int(duration_hours):02d}h"
        / "DEC_result"
        / "all_parement"
        / f"{event_name}_all_parement.npz"
    )


def load_cluster_time_series(event_name: str, duration_hours: int) -> ClusterTimeSeries | None:
    npz_path = get_npz_path(event_name, duration_hours)
    if not npz_path.exists():
        print(f"[WARN] missing npz: {npz_path}")
        return None

    data = np.load(npz_path, allow_pickle=True)
    if "x_file" not in data.files or "y_pred_last" not in data.files:
        raise KeyError(f"'x_file' or 'y_pred_last' not found in {npz_path}")

    x_file = data["x_file"]
    labels = np.asarray(data["y_pred_last"]).astype(int)
    times = np.asarray([parse_time_from_wave_file(item) for item in x_file], dtype=object)
    valid = np.asarray([item is not None for item in times], dtype=bool)

    if valid.sum() == 0:
        raise ValueError(f"No valid timestamps parsed from {npz_path}")

    times = times[valid]
    labels = labels[valid]
    order = np.argsort(times)
    times = times[order]
    labels = labels[order]

    end_time = max(times)
    hours_to_end = np.asarray(
        [(item - end_time).total_seconds() / 3600.0 for item in times],
        dtype=float,
    )

    if "n_clusters" in data.files:
        n_clusters = int(np.asarray(data["n_clusters"]).item())
    else:
        n_clusters = int(labels.max() + 1)

    return ClusterTimeSeries(
        event=event_name,
        duration_hours=int(duration_hours),
        npz_path=npz_path,
        time=times,
        hours_to_end=hours_to_end,
        labels=labels,
        n_clusters=n_clusters,
    )


def build_cluster_curves(series: ClusterTimeSeries) -> pd.DataFrame:
    rows = []
    unique_labels = sorted(np.unique(series.labels).astype(int).tolist())

    for cluster_label in unique_labels:
        counts = np.cumsum(series.labels == cluster_label).astype(float)
        if NORMALIZE_CUMULATIVE and counts[-1] > 0:
            y_value = counts / counts[-1]
            y_name = "cumulative_fraction"
        else:
            y_value = counts
            y_name = "cumulative_count"

        for x, y, raw_count in zip(series.hours_to_end, y_value, counts):
            rows.append(
                {
                    "event": series.event,
                    "duration_hours": series.duration_hours,
                    "cluster": int(cluster_label),
                    "hours_to_end": float(x),
                    y_name: float(y),
                    "cumulative_count": int(raw_count),
                }
            )

    return pd.DataFrame(rows)


# =========================================================
# 3. Plotting
# =========================================================
def plot_event_duration_comparison(
    event_name: str,
    duration_series: list[ClusterTimeSeries],
    output_dir: Path,
) -> Path:
    duration_series = sorted(duration_series, key=lambda item: duration_key(item.duration_hours))
    n_rows = len(duration_series)
    fig_height = max(2.3 * n_rows, 5.0)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(8, fig_height),
        sharex=False,
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab20")
    y_label = "Cumulative fraction" if NORMALIZE_CUMULATIVE else "Cumulative count"

    for ax, series in zip(axes, duration_series):
        unique_labels = sorted(np.unique(series.labels).astype(int).tolist())
        for idx, cluster_label in enumerate(unique_labels):
            mask = series.labels == cluster_label
            y = np.cumsum(mask).astype(float)
            if NORMALIZE_CUMULATIVE and y[-1] > 0:
                y = y / y[-1]
            ax.plot(
                series.hours_to_end,
                y,
                linewidth=1.7,
                color=cmap(idx % cmap.N),
                label=f"C{cluster_label} (n={int(mask.sum())})",
            )

        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
        ax.grid(alpha=0.25)
        ax.set_ylabel(y_label)
        ax.set_xlim(-COMMON_X_WINDOW_HOURS, 0.05)
        if NORMALIZE_CUMULATIVE:
            ax.set_ylim(-0.03, 1.03)
        ax.set_title(
            f"{series.duration_hours} h window, K={series.n_clusters}, "
            f"samples={len(series.labels)}",
            loc="left",
            fontsize=11,
        )
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.005, 1.02),
            fontsize=8,
            ncol=1 if len(unique_labels) <= MAX_LEGEND_COLUMNS else 2,
            frameon=True,
        )

    axes[-1].set_xlabel("Time before event end (h)")
    fig.suptitle(
        f"{event_name}: cluster cumulative evolution across duration windows\n"
        "Cluster labels are duration-specific because each event-duration pair was clustered independently.",
        fontsize=14,
        fontweight="bold",
    )

    suffix = "fraction" if NORMALIZE_CUMULATIVE else "count"
    output_path = output_dir / (
        f"{event_name}_duration_cluster_cumulative_{suffix}_x{COMMON_X_WINDOW_HOURS}h.png"
    )
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def save_event_curves_csv(
    event_name: str,
    duration_series: list[ClusterTimeSeries],
    output_dir: Path,
) -> Path:
    frames = [build_cluster_curves(series) for series in duration_series]
    curves = pd.concat(frames, ignore_index=True)
    csv_path = output_dir / f"{event_name}_duration_cluster_cumulative_curves.csv"
    curves.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


# =========================================================
# 4. Main
# =========================================================
def main() -> None:
    output_dir = OUTPUT_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    event_names = discover_event_names()
    durations = discover_duration_hours()

    print("Events:", len(event_names))
    print("Durations:", durations)
    print("Original 24 h npz dir:", ORIGINAL_24H_NPZ_DIR)
    print("Different-duration root:", DIFFERENT_DURATION_ROOT)
    print("Output:", output_dir)

    summary_rows = []
    for event_name in event_names:
        duration_series = []
        for duration_hours in durations:
            series = load_cluster_time_series(event_name, duration_hours)
            if series is not None:
                duration_series.append(series)

        if not duration_series:
            print(f"[WARN] {event_name}: no available duration results, skipped.")
            continue

        figure_path = plot_event_duration_comparison(event_name, duration_series, output_dir)
        csv_path = save_event_curves_csv(event_name, duration_series, output_dir) if SAVE_CURVE_CSV else ""

        for series in duration_series:
            summary_rows.append(
                {
                    "event": event_name,
                    "duration_hours": series.duration_hours,
                    "n_clusters": series.n_clusters,
                    "sample_count": int(len(series.labels)),
                    "npz_path": str(series.npz_path),
                    "figure_path": str(figure_path),
                    "curve_csv_path": str(csv_path),
                }
            )

        print(f"[OK] {event_name}: {figure_path}")

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(
            by=["event", "duration_hours"],
            key=lambda s: s.map(natural_event_key) if s.name == "event" else s,
        )
    summary_path = output_dir / "duration_cluster_cumulative_comparison_summary.xlsx"
    summary.to_excel(summary_path, index=False)

    print("\nDone.")
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
