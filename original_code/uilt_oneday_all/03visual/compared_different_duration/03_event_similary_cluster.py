# -*- coding: utf-8 -*-
"""
Compare PCS-selected precursor clusters from different duration windows with
the original 24 h PCS-selected precursor cluster for the same event.

Different DEC runs have different cluster-label spaces, so Cluster 2 in a 3 h
run is not automatically the same physical state as Cluster 2 in the 24 h run.
This script therefore compares clusters by temporal overlap in the common time
interval.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# 1. Paths and configuration
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PCS_OUTPUT_ROOT = SCRIPT_DIR / "pcs_different_duration"
PCS_WORKBOOK = PCS_OUTPUT_ROOT / "pcs_selection_by_duration.xlsx"
REFERENCE_DURATION_HOURS = 24

REFERENCE_HELPER_SCRIPT = SCRIPT_DIR / "02_PCS_different_dura.py"
OUTPUT_DIR = PCS_OUTPUT_ROOT / "event_similarity_to_24h"

# "Mostly same" means that the selected short-window precursor cluster overlaps
# strongly with the 24 h precursor cluster in their shared timestamps.
PRECISION_THRESHOLD = 0.70
RECALL_THRESHOLD = 0.50
JACCARD_THRESHOLD = 0.50

FIG_DPI = 300

plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "Arial",
        "axes.unicode_minus": False,
    }
)
plt.ioff()


# =========================================================
# 2. Helpers
# =========================================================
def load_duration_helper_module():
    """Load helper functions used by the duration PCS script."""
    module_name = "pcs_different_duration_helpers"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, REFERENCE_HELPER_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helper script: {REFERENCE_HELPER_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def natural_event_key(value: str) -> int:
    match = re.search(r"event(\d+)", str(value), flags=re.IGNORECASE)
    return int(match.group(1)) if match else 10**9


def load_pcs_tables(workbook: Path = PCS_WORKBOOK) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not workbook.exists():
        raise FileNotFoundError(f"PCS workbook not found: {workbook}")

    selected_df = pd.read_excel(workbook, sheet_name="pcs_selected")
    missing_df = pd.read_excel(workbook, sheet_name="pcs_not_selected")

    for df in [selected_df, missing_df]:
        if not df.empty:
            df["event"] = df["event"].astype(str).str.lower()
            df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce")
            df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")

    selected_df = selected_df[
        (selected_df["selection_status"] == "selected")
        & selected_df["duration_hours"].notna()
        & selected_df["cluster"].notna()
        & (selected_df["cluster"] >= 0)
    ].copy()
    selected_df["duration_hours"] = selected_df["duration_hours"].astype(int)
    selected_df["cluster"] = selected_df["cluster"].astype(int)

    missing_df = missing_df[missing_df["duration_hours"].notna()].copy()
    missing_df["duration_hours"] = missing_df["duration_hours"].astype(int)
    return selected_df, missing_df


def get_selected_lookup(selected_df: pd.DataFrame) -> dict[tuple[str, int], object]:
    lookup: dict[tuple[str, int], object] = {}
    for row in selected_df.itertuples(index=False):
        lookup[(str(row.event), int(row.duration_hours))] = row
    return lookup


def event_duration_universe(selected_df: pd.DataFrame, missing_df: pd.DataFrame) -> tuple[list[str], list[int]]:
    events = sorted(
        set(selected_df["event"].astype(str)) | set(missing_df["event"].astype(str)),
        key=natural_event_key,
    )
    durations = sorted(
        set(selected_df["duration_hours"].astype(int))
        | set(missing_df["duration_hours"].astype(int))
    )
    return events, durations


def load_event_data_from_row(row: object):
    helper = load_duration_helper_module()
    return helper.load_event_duration_data(Path(row.npz_path), int(row.duration_hours))


def cluster_membership_by_time(row: object) -> dict[object, bool]:
    event_data = load_event_data_from_row(row)
    cluster_id = int(row.cluster)
    return {
        time_value: bool(label == cluster_id)
        for time_value, label in zip(event_data.times, event_data.labels)
    }


def compare_cluster_overlap(short_row: object, reference_row: object) -> dict[str, float | int | str | bool]:
    short_membership = cluster_membership_by_time(short_row)
    reference_membership = cluster_membership_by_time(reference_row)
    common_times = sorted(set(short_membership) & set(reference_membership))

    if not common_times:
        return {
            "common_samples": 0,
            "short_selected_samples": 0,
            "reference_selected_samples": 0,
            "intersection_samples": 0,
            "union_samples": 0,
            "precision_to_24h": np.nan,
            "recall_to_24h": np.nan,
            "jaccard_to_24h": np.nan,
            "f1_to_24h": np.nan,
            "mostly_same_as_24h": False,
            "match_level": "no_common_time",
        }

    short_mask = np.asarray([short_membership[item] for item in common_times], dtype=bool)
    reference_mask = np.asarray([reference_membership[item] for item in common_times], dtype=bool)
    intersection = int(np.logical_and(short_mask, reference_mask).sum())
    union = int(np.logical_or(short_mask, reference_mask).sum())
    short_count = int(short_mask.sum())
    reference_count = int(reference_mask.sum())

    precision = intersection / short_count if short_count else np.nan
    recall = intersection / reference_count if reference_count else np.nan
    jaccard = intersection / union if union else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else np.nan
    )

    mostly_same = bool(
        (np.isfinite(jaccard) and jaccard >= JACCARD_THRESHOLD)
        or (
            np.isfinite(precision)
            and np.isfinite(recall)
            and precision >= PRECISION_THRESHOLD
            and recall >= RECALL_THRESHOLD
        )
    )
    if mostly_same:
        match_level = "strong_match"
    elif np.isfinite(precision) and precision >= PRECISION_THRESHOLD:
        match_level = "short_cluster_subset_of_24h"
    elif np.isfinite(recall) and recall >= RECALL_THRESHOLD:
        match_level = "captures_part_of_24h"
    elif union > 0:
        match_level = "weak_match"
    else:
        match_level = "no_selected_overlap"

    return {
        "common_samples": len(common_times),
        "short_selected_samples": short_count,
        "reference_selected_samples": reference_count,
        "intersection_samples": intersection,
        "union_samples": union,
        "precision_to_24h": precision,
        "recall_to_24h": recall,
        "jaccard_to_24h": jaccard,
        "f1_to_24h": f1,
        "mostly_same_as_24h": mostly_same,
        "match_level": match_level,
    }


# =========================================================
# 3. Analysis tables
# =========================================================
def build_event_similarity_table(selected_df: pd.DataFrame, missing_df: pd.DataFrame) -> pd.DataFrame:
    events, durations = event_duration_universe(selected_df, missing_df)
    selected_lookup = get_selected_lookup(selected_df)
    rows = []

    for event in events:
        reference_row = selected_lookup.get((event, REFERENCE_DURATION_HOURS))
        reference_selected = reference_row is not None

        for duration in durations:
            if duration == REFERENCE_DURATION_HOURS:
                continue

            short_row = selected_lookup.get((event, duration))
            duration_selected = short_row is not None
            row = {
                "event": event,
                "duration_hours": duration,
                "duration_selected": duration_selected,
                "reference_24h_selected": reference_selected,
                "event_level_same_as_24h": bool(duration_selected and reference_selected),
                "duration_cluster": int(short_row.cluster) if duration_selected else -1,
                "reference_24h_cluster": int(reference_row.cluster) if reference_selected else -1,
                "duration_pcs": float(short_row.pcs) if duration_selected and hasattr(short_row, "pcs") else np.nan,
                "reference_24h_pcs": (
                    float(reference_row.pcs) if reference_selected and hasattr(reference_row, "pcs") else np.nan
                ),
            }

            if duration_selected and reference_selected:
                row.update(compare_cluster_overlap(short_row, reference_row))
            else:
                reason = (
                    "duration_not_selected"
                    if reference_selected and not duration_selected
                    else "reference_24h_not_selected"
                    if duration_selected and not reference_selected
                    else "both_not_selected"
                )
                row.update(
                    {
                        "common_samples": np.nan,
                        "short_selected_samples": np.nan,
                        "reference_selected_samples": np.nan,
                        "intersection_samples": np.nan,
                        "union_samples": np.nan,
                        "precision_to_24h": np.nan,
                        "recall_to_24h": np.nan,
                        "jaccard_to_24h": np.nan,
                        "f1_to_24h": np.nan,
                        "mostly_same_as_24h": False,
                        "match_level": reason,
                    }
                )
            rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df["_event_order"] = result_df["event"].map(natural_event_key)
    return result_df.sort_values(["duration_hours", "_event_order"]).drop(columns="_event_order")


def summarize_by_duration(comparison_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for duration, group in comparison_df.groupby("duration_hours", sort=True):
        both_selected = group[group["event_level_same_as_24h"]]
        rows.append(
            {
                "duration_hours": int(duration),
                "total_events": int(group["event"].nunique()),
                "selected_events": int(group["duration_selected"].sum()),
                "reference_24h_selected_events": int(group["reference_24h_selected"].sum()),
                "selected_also_selected_by_24h": int(group["event_level_same_as_24h"].sum()),
                "event_overlap_rate_vs_24h_selected": (
                    group["event_level_same_as_24h"].sum()
                    / group["reference_24h_selected"].sum()
                    if group["reference_24h_selected"].sum() > 0
                    else np.nan
                ),
                "duration_selected_precision_vs_24h": (
                    group["event_level_same_as_24h"].sum()
                    / group["duration_selected"].sum()
                    if group["duration_selected"].sum() > 0
                    else np.nan
                ),
                "mostly_same_cluster_events": int(group["mostly_same_as_24h"].sum()),
                "mostly_same_rate_among_both_selected": (
                    group["mostly_same_as_24h"].sum() / len(both_selected)
                    if len(both_selected) > 0
                    else np.nan
                ),
                "median_jaccard_among_both_selected": both_selected["jaccard_to_24h"].median(),
                "median_precision_among_both_selected": both_selected["precision_to_24h"].median(),
                "median_recall_among_both_selected": both_selected["recall_to_24h"].median(),
            }
        )
    return pd.DataFrame(rows)


# =========================================================
# 4. Figures
# =========================================================
def plot_duration_summary(summary_df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(11, 6), constrained_layout=True)
    x = np.arange(len(summary_df))
    labels = [f"{int(item)}h" for item in summary_df["duration_hours"]]

    ax1.bar(
        x - 0.18,
        summary_df["duration_selected_precision_vs_24h"],
        width=0.36,
        color="#4C78A8",
        alpha=0.9,
        label="Selected events also selected by 24h",
    )
    ax1.bar(
        x + 0.18,
        summary_df["mostly_same_rate_among_both_selected"],
        width=0.36,
        color="#F58518",
        alpha=0.9,
        label="Mostly same cluster among both-selected events",
    )
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Observation-window length")
    ax1.set_ylabel("Rate")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        summary_df["median_jaccard_among_both_selected"],
        color="#54A24B",
        marker="o",
        linewidth=2.0,
        label="Median temporal Jaccard",
    )
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Median temporal Jaccard")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=True)
    ax1.set_title(
        "Consistency of PCS-selected precursor clusters relative to 24 h",
        fontsize=14,
        fontweight="bold",
    )

    output_path = OUTPUT_DIR / "duration_similarity_summary_to_24h.png"
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_event_duration_heatmap(comparison_df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pivot = comparison_df.pivot(index="event", columns="duration_hours", values="jaccard_to_24h")
    events = sorted(pivot.index.astype(str), key=natural_event_key)
    durations = sorted(pivot.columns.astype(int))
    pivot = pivot.loc[events, durations]

    fig, ax = plt.subplots(
        figsize=(max(10, len(durations) * 0.9), max(10, len(events) * 0.25)),
        constrained_layout=True,
    )
    matrix = pivot.to_numpy(dtype=float)
    masked_matrix = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#E5E5E5")
    im = ax.imshow(masked_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(durations)))
    ax.set_xticklabels([f"{duration}h" for duration in durations])
    ax.set_yticks(np.arange(len(events)))
    ax.set_yticklabels(events)
    ax.set_xlabel("Observation-window length")
    ax.set_ylabel("Event")
    ax.set_title(
        "Temporal overlap between duration-specific PCS clusters and 24 h PCS clusters",
        fontsize=14,
        fontweight="bold",
    )

    for row_idx, event in enumerate(events):
        for col_idx, duration in enumerate(durations):
            row = comparison_df[
                (comparison_df["event"] == event) & (comparison_df["duration_hours"] == duration)
            ]
            if row.empty:
                continue
            item = row.iloc[0]
            if bool(item["mostly_same_as_24h"]):
                ax.text(col_idx, row_idx, "*", ha="center", va="center", color="black", fontsize=10)
            elif bool(item["event_level_same_as_24h"]) and pd.isna(item["jaccard_to_24h"]):
                ax.text(col_idx, row_idx, "?", ha="center", va="center", color="black", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temporal Jaccard with 24 h PCS cluster")

    output_path = OUTPUT_DIR / "event_duration_jaccard_to_24h_heatmap.png"
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_match_level_counts(comparison_df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    count_df = (
        comparison_df.groupby(["duration_hours", "match_level"])
        .size()
        .reset_index(name="count")
    )
    durations = sorted(count_df["duration_hours"].unique())
    levels = [
        "strong_match",
        "short_cluster_subset_of_24h",
        "captures_part_of_24h",
        "weak_match",
        "duration_not_selected",
        "reference_24h_not_selected",
        "both_not_selected",
        "no_common_time",
        "no_selected_overlap",
    ]
    colors = {
        "strong_match": "#54A24B",
        "short_cluster_subset_of_24h": "#72B7B2",
        "captures_part_of_24h": "#B279A2",
        "weak_match": "#F58518",
        "duration_not_selected": "#BAB0AC",
        "reference_24h_not_selected": "#E45756",
        "both_not_selected": "#D6D6D6",
        "no_common_time": "#9D755D",
        "no_selected_overlap": "#79706E",
    }

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    bottom = np.zeros(len(durations), dtype=float)
    x = np.arange(len(durations))
    for level in levels:
        values = []
        for duration in durations:
            subset = count_df[
                (count_df["duration_hours"] == duration) & (count_df["match_level"] == level)
            ]
            values.append(int(subset["count"].iloc[0]) if not subset.empty else 0)
        if sum(values) == 0:
            continue
        ax.bar(x, values, bottom=bottom, color=colors.get(level, "#CCCCCC"), label=level)
        bottom += np.asarray(values, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(duration)}h" for duration in durations])
    ax.set_xlabel("Observation-window length")
    ax.set_ylabel("Number of events")
    ax.set_title("Event match categories relative to 24 h PCS selection", fontsize=14, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    output_path = OUTPUT_DIR / "match_level_counts_to_24h.png"
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


# =========================================================
# 5. Export and main
# =========================================================
def export_results(comparison_df: pd.DataFrame, summary_df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "event_cluster_similarity_to_24h.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="duration_summary", index=False)
        comparison_df.to_excel(writer, sheet_name="event_duration_comparison", index=False)
    return output_path


def main() -> None:
    selected_df, missing_df = load_pcs_tables()
    comparison_df = build_event_similarity_table(selected_df, missing_df)
    summary_df = summarize_by_duration(comparison_df)
    excel_path = export_results(comparison_df, summary_df)
    figure_paths = [
        plot_duration_summary(summary_df),
        plot_event_duration_heatmap(comparison_df),
        plot_match_level_counts(comparison_df),
    ]

    print("Saved Excel:", excel_path)
    print("Saved figures:")
    for path in figure_paths:
        print("  ", path)
    print("\nDuration summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
