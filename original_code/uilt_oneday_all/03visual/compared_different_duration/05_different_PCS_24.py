# -*- coding: utf-8 -*-
"""
Compare PCS-selected precursor clusters from different time windows with the
24 h PCS-selected precursor cluster.

Main index:
    overlap_ratio_of_own_cluster =
        N(samples in duration-specific PCS cluster and also in 24 h PCS cluster)
        / N(samples in duration-specific PCS cluster) * 100%

This answers:
    "不同时间长度选取出的前兆簇中，与 24 h 前兆簇相同的样本占自己簇的百分之多少？"

Notes:
    Cluster labels from different DEC runs are not directly comparable, so this
    script compares sample membership at the same timestamps in the overlapping
    time interval.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PCS_OUTPUT_ROOT = SCRIPT_DIR / "pcs_different_duration"
PCS_WORKBOOK = PCS_OUTPUT_ROOT / "pcs_selection_by_duration.xlsx"
HELPER_SCRIPT = SCRIPT_DIR / "02_PCS_different_dura.py"

REFERENCE_DURATION_HOURS = 24
OUTPUT_DIR = PCS_OUTPUT_ROOT / "pcs_overlap_with_24h"
FIG_DPI = 300

plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "Arial",
        "axes.unicode_minus": False,
    }
)
plt.ioff()


def natural_event_key(value: object) -> int:
    match = re.search(r"event(\d+)", str(value), flags=re.IGNORECASE)
    return int(match.group(1)) if match else 10**9


def load_duration_helper_module():
    """Reuse NPZ loading and timestamp parsing from the PCS duration script."""
    module_name = "pcs_different_duration_helpers_for_overlap"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, HELPER_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helper script: {HELPER_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_pcs_selected_table(workbook: Path = PCS_WORKBOOK) -> pd.DataFrame:
    if not workbook.exists():
        raise FileNotFoundError(f"PCS selection workbook not found: {workbook}")

    df = pd.read_excel(workbook, sheet_name="pcs_selected")
    required = {"event", "duration_hours", "cluster", "selection_status", "npz_path"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"pcs_selected sheet is missing required columns: {missing}")

    df = df.copy()
    df["event"] = df["event"].astype(str).str.lower()
    df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce")
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")

    df = df[
        (df["selection_status"].astype(str).str.lower() == "selected")
        & df["duration_hours"].notna()
        & df["cluster"].notna()
        & (df["cluster"] >= 0)
    ].copy()
    df["duration_hours"] = df["duration_hours"].astype(int)
    df["cluster"] = df["cluster"].astype(int)
    df["_event_order"] = df["event"].map(natural_event_key)
    df = df.sort_values(["duration_hours", "_event_order"]).drop(columns="_event_order")
    return df


def selected_lookup(selected_df: pd.DataFrame) -> dict[tuple[str, int], object]:
    return {
        (str(row.event), int(row.duration_hours)): row
        for row in selected_df.itertuples(index=False)
    }


def load_cluster_membership(row: object) -> dict[object, bool]:
    """Return timestamp -> whether sample belongs to the selected cluster."""
    helper = load_duration_helper_module()
    event_data = helper.load_event_duration_data(Path(row.npz_path), int(row.duration_hours))
    cluster_id = int(row.cluster)
    return {
        time_value: bool(label == cluster_id)
        for time_value, label in zip(event_data.times, event_data.labels)
    }


def compare_to_24h(duration_row: object, reference_row: object) -> dict[str, float | int | str]:
    duration_membership = load_cluster_membership(duration_row)
    reference_membership = load_cluster_membership(reference_row)
    common_times = sorted(set(duration_membership) & set(reference_membership))

    if not common_times:
        return {
            "common_samples": 0,
            "duration_cluster_samples_in_common_time": 0,
            "reference_24h_cluster_samples_in_common_time": 0,
            "same_samples_with_24h": 0,
            "union_samples": 0,
            "overlap_ratio_of_own_cluster": np.nan,
            "overlap_percent_of_own_cluster": np.nan,
            "reference_coverage_percent": np.nan,
            "jaccard_percent": np.nan,
            "match_level": "no_common_time",
        }

    duration_mask = np.asarray([duration_membership[item] for item in common_times], dtype=bool)
    reference_mask = np.asarray([reference_membership[item] for item in common_times], dtype=bool)

    same_count = int(np.logical_and(duration_mask, reference_mask).sum())
    union_count = int(np.logical_or(duration_mask, reference_mask).sum())
    duration_count = int(duration_mask.sum())
    reference_count = int(reference_mask.sum())

    own_ratio = same_count / duration_count if duration_count else np.nan
    reference_coverage = same_count / reference_count if reference_count else np.nan
    jaccard = same_count / union_count if union_count else np.nan

    if np.isfinite(own_ratio) and own_ratio >= 0.8:
        match_level = "high_own_overlap"
    elif np.isfinite(own_ratio) and own_ratio >= 0.5:
        match_level = "moderate_own_overlap"
    elif np.isfinite(own_ratio):
        match_level = "low_own_overlap"
    else:
        match_level = "no_duration_cluster_samples"

    return {
        "common_samples": len(common_times),
        "duration_cluster_samples_in_common_time": duration_count,
        "reference_24h_cluster_samples_in_common_time": reference_count,
        "same_samples_with_24h": same_count,
        "union_samples": union_count,
        "overlap_ratio_of_own_cluster": own_ratio,
        "overlap_percent_of_own_cluster": own_ratio * 100 if np.isfinite(own_ratio) else np.nan,
        "reference_coverage_percent": (
            reference_coverage * 100 if np.isfinite(reference_coverage) else np.nan
        ),
        "jaccard_percent": jaccard * 100 if np.isfinite(jaccard) else np.nan,
        "match_level": match_level,
    }


def build_overlap_table(selected_df: pd.DataFrame) -> pd.DataFrame:
    lookup = selected_lookup(selected_df)
    events = sorted(selected_df["event"].unique(), key=natural_event_key)
    durations = sorted(
        duration
        for duration in selected_df["duration_hours"].unique()
        if int(duration) != REFERENCE_DURATION_HOURS
    )
    rows = []

    for event in events:
        reference_row = lookup.get((event, REFERENCE_DURATION_HOURS))
        for duration in durations:
            duration_row = lookup.get((event, int(duration)))
            base_row = {
                "event": event,
                "duration_hours": int(duration),
                "duration_selected": duration_row is not None,
                "reference_24h_selected": reference_row is not None,
                "duration_cluster": int(duration_row.cluster) if duration_row is not None else -1,
                "reference_24h_cluster": int(reference_row.cluster) if reference_row is not None else -1,
                "duration_pcs": float(duration_row.pcs) if duration_row is not None and hasattr(duration_row, "pcs") else np.nan,
                "reference_24h_pcs": (
                    float(reference_row.pcs) if reference_row is not None and hasattr(reference_row, "pcs") else np.nan
                ),
                "duration_npz_path": str(duration_row.npz_path) if duration_row is not None else "",
                "reference_24h_npz_path": str(reference_row.npz_path) if reference_row is not None else "",
            }

            if duration_row is None or reference_row is None:
                reason = (
                    "duration_not_selected"
                    if duration_row is None and reference_row is not None
                    else "reference_24h_not_selected"
                    if duration_row is not None and reference_row is None
                    else "both_not_selected"
                )
                base_row.update(
                    {
                        "common_samples": np.nan,
                        "duration_cluster_samples_in_common_time": np.nan,
                        "reference_24h_cluster_samples_in_common_time": np.nan,
                        "same_samples_with_24h": np.nan,
                        "union_samples": np.nan,
                        "overlap_ratio_of_own_cluster": np.nan,
                        "overlap_percent_of_own_cluster": np.nan,
                        "reference_coverage_percent": np.nan,
                        "jaccard_percent": np.nan,
                        "match_level": reason,
                    }
                )
            else:
                base_row.update(compare_to_24h(duration_row, reference_row))

            rows.append(base_row)

    result_df = pd.DataFrame(rows)
    result_df["_event_order"] = result_df["event"].map(natural_event_key)
    result_df = result_df.sort_values(["duration_hours", "_event_order"]).drop(columns="_event_order")
    return result_df


def summarize_by_duration(overlap_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for duration, group in overlap_df.groupby("duration_hours", sort=True):
        valid = group[group["overlap_percent_of_own_cluster"].notna()]
        rows.append(
            {
                "duration_hours": int(duration),
                "event_rows": int(len(group)),
                "both_selected_events": int(len(valid)),
                "duration_selected_events": int(group["duration_selected"].sum()),
                "reference_24h_selected_events": int(group["reference_24h_selected"].sum()),
                "mean_overlap_percent_of_own_cluster": valid["overlap_percent_of_own_cluster"].mean(),
                "median_overlap_percent_of_own_cluster": valid["overlap_percent_of_own_cluster"].median(),
                "q25_overlap_percent_of_own_cluster": valid["overlap_percent_of_own_cluster"].quantile(0.25),
                "q75_overlap_percent_of_own_cluster": valid["overlap_percent_of_own_cluster"].quantile(0.75),
                "events_overlap_ge_80pct": int((valid["overlap_percent_of_own_cluster"] >= 80).sum()),
                "events_overlap_ge_50pct": int((valid["overlap_percent_of_own_cluster"] >= 50).sum()),
                "median_reference_coverage_percent": valid["reference_coverage_percent"].median(),
                "median_jaccard_percent": valid["jaccard_percent"].median(),
            }
        )
    return pd.DataFrame(rows)


def plot_overlap_boxplot(overlap_df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    valid = overlap_df[overlap_df["overlap_percent_of_own_cluster"].notna()].copy()
    durations = sorted(valid["duration_hours"].unique())
    data = [
        valid.loc[valid["duration_hours"] == duration, "overlap_percent_of_own_cluster"].to_numpy()
        for duration in durations
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(durations) * 0.8), 5.2), constrained_layout=True)
    box = ax.boxplot(
        data,
        tick_labels=[f"{int(duration)}h" for duration in durations],
        showfliers=False,
        patch_artist=True,
        widths=0.55,
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#4C78A8")
        patch.set_alpha(0.35)
        patch.set_edgecolor("#2F5F8F")
    for median in box["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.4)

    rng = np.random.default_rng(20260711)
    for idx, duration in enumerate(durations, start=1):
        values = valid.loc[
            valid["duration_hours"] == duration,
            "overlap_percent_of_own_cluster",
        ].to_numpy(dtype=float)
        ax.scatter(
            np.full(len(values), idx) + rng.normal(0, 0.04, size=len(values)),
            values,
            s=22,
            color="#4C78A8",
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
        )

    ax.axhline(50, color="#F58518", linestyle="--", linewidth=1.2, label="50%")
    ax.axhline(80, color="#54A24B", linestyle="--", linewidth=1.2, label="80%")
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Observation-window length")
    ax.set_ylabel("Overlap with 24 h PCS cluster (% of own selected cluster)")
    ax.set_title("Sample-level consistency of duration-specific PCS clusters with 24 h")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)

    out = OUTPUT_DIR / "overlap_percent_of_own_cluster_boxplot.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_overlap_heatmap(overlap_df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pivot = overlap_df.pivot(
        index="event",
        columns="duration_hours",
        values="overlap_percent_of_own_cluster",
    )
    events = sorted(pivot.index.astype(str), key=natural_event_key)
    durations = sorted(pivot.columns.astype(int))
    pivot = pivot.loc[events, durations]

    fig, ax = plt.subplots(
        figsize=(max(9, len(durations) * 0.8), max(9, len(events) * 0.24)),
        constrained_layout=True,
    )
    matrix = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#E5E5E5")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(durations)))
    ax.set_xticklabels([f"{int(duration)}h" for duration in durations])
    ax.set_yticks(np.arange(len(events)))
    ax.set_yticklabels(events)
    ax.set_xlabel("Observation-window length")
    ax.set_ylabel("Event")
    ax.set_title("Overlap ratio: duration-specific PCS cluster vs 24 h PCS cluster")

    for row_idx, event in enumerate(events):
        for col_idx, duration in enumerate(durations):
            row = overlap_df[
                (overlap_df["event"] == event) & (overlap_df["duration_hours"] == duration)
            ]
            if row.empty:
                continue
            value = row.iloc[0]["overlap_percent_of_own_cluster"]
            if pd.notna(value):
                ax.text(col_idx, row_idx, f"{value:.0f}", ha="center", va="center", fontsize=6)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Overlap with 24 h PCS cluster (% of own cluster)")

    out = OUTPUT_DIR / "overlap_percent_of_own_cluster_heatmap.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_duration_summary(summary_df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    x = np.arange(len(summary_df))
    labels = [f"{int(item)}h" for item in summary_df["duration_hours"]]

    ax.bar(
        x,
        summary_df["median_overlap_percent_of_own_cluster"],
        color="#4C78A8",
        alpha=0.85,
        label="Median overlap",
    )
    ax.plot(
        x,
        summary_df["mean_overlap_percent_of_own_cluster"],
        color="#E76F51",
        marker="o",
        linewidth=2.0,
        label="Mean overlap",
    )
    ax.axhline(50, color="#F58518", linestyle="--", linewidth=1.2)
    ax.axhline(80, color="#54A24B", linestyle="--", linewidth=1.2)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Observation-window length")
    ax.set_ylabel("Overlap with 24 h PCS cluster (%)")
    ax.set_title("Duration-level overlap summary")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)

    out = OUTPUT_DIR / "overlap_percent_duration_summary.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def export_results(overlap_df: pd.DataFrame, summary_df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "pcs_overlap_with_24h.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="duration_summary", index=False)
        overlap_df.to_excel(writer, sheet_name="event_duration_overlap", index=False)
    return out


def main() -> None:
    selected_df = load_pcs_selected_table()
    overlap_df = build_overlap_table(selected_df)
    summary_df = summarize_by_duration(overlap_df)

    excel_path = export_results(overlap_df, summary_df)
    figures = [
        plot_overlap_boxplot(overlap_df),
        plot_overlap_heatmap(overlap_df),
        plot_duration_summary(summary_df),
    ]

    print("Saved Excel:", excel_path)
    print("Saved figures:")
    for figure in figures:
        print("  ", figure)
    print("\nDuration summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
