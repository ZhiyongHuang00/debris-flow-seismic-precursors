# -*- coding: utf-8 -*-
"""
Use the 24 h PCS-selected precursor cluster as a reference and find, for each
event and each other duration, the cluster with the largest sample-level overlap
with the 24 h precursor cluster.

Main selection rule:
    For each event-duration-cluster candidate:

        overlap_percent_of_own_cluster =
            N(candidate cluster samples also belonging to the 24 h precursor cluster)
            / N(candidate cluster samples in the common time interval) * 100%

    The new precursor cluster for that duration is the cluster with the highest
    overlap_percent_of_own_cluster. Ties are broken by intersection count,
    Jaccard overlap, and PCS score.

Outputs:
    1. Excel table with all candidate overlaps and selected matched clusters.
    2. Cumulative curves of the 24 h-matched clusters by duration.
    3. Event-duration heatmap of overlap percentage.
    4. Duration-level summary figure.
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
COMPARE_DIR = SCRIPT_DIR
PCS_OUTPUT_ROOT = COMPARE_DIR / "pcs_different_duration"
PCS_WORKBOOK = PCS_OUTPUT_ROOT / "pcs_selection_by_duration.xlsx"
HELPER_SCRIPT = COMPARE_DIR / "02_PCS_different_dura.py"

REFERENCE_DURATION_HOURS = 24
OUTPUT_DIR = PCS_OUTPUT_ROOT / "match_24h_selected_clusters"
CURVE_DIR = OUTPUT_DIR / "matched_cumulative_curves_by_duration"
FIG_DPI = 300

# Set to >1 if tiny clusters should not be allowed to win by a trivial 100% overlap.
MIN_CANDIDATE_SAMPLES_IN_COMMON = 1

plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "Arial",
        "axes.unicode_minus": False,
    }
)
plt.ioff()


# =========================================================
# 2. Basic helpers
# =========================================================
def natural_event_key(value: object) -> int:
    match = re.search(r"event(\d+)", str(value), flags=re.IGNORECASE)
    return int(match.group(1)) if match else 10**9


def load_duration_helper_module():
    """Reuse timestamp parsing and NPZ loading from the duration PCS script."""
    module_name = "pcs_duration_helpers_for_match_24h"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, HELPER_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helper script: {HELPER_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sort_event_duration(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_event_order"] = out["event"].map(natural_event_key)
    sort_cols = ["duration_hours", "_event_order"]
    if "matched_cluster" in out.columns:
        sort_cols.append("matched_cluster")
    elif "cluster" in out.columns:
        sort_cols.append("cluster")
    out = out.sort_values(sort_cols)
    return out.drop(columns="_event_order")


def load_pcs_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PCS_WORKBOOK.exists():
        raise FileNotFoundError(f"PCS selection workbook not found: {PCS_WORKBOOK}")

    all_metrics = pd.read_excel(PCS_WORKBOOK, sheet_name="all_cluster_metrics")
    pcs_selected = pd.read_excel(PCS_WORKBOOK, sheet_name="pcs_selected")

    for df in [all_metrics, pcs_selected]:
        df["event"] = df["event"].astype(str).str.lower()
        df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce").astype("Int64")
        df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")

    pcs_selected = pcs_selected[
        (pcs_selected["selection_status"].astype(str).str.lower() == "selected")
        & pcs_selected["duration_hours"].notna()
        & pcs_selected["cluster"].notna()
        & (pcs_selected["cluster"] >= 0)
    ].copy()
    pcs_selected["duration_hours"] = pcs_selected["duration_hours"].astype(int)
    pcs_selected["cluster"] = pcs_selected["cluster"].astype(int)

    all_metrics = all_metrics[
        all_metrics["duration_hours"].notna()
        & all_metrics["cluster"].notna()
        & (all_metrics["cluster"] >= 0)
    ].copy()
    all_metrics["duration_hours"] = all_metrics["duration_hours"].astype(int)
    all_metrics["cluster"] = all_metrics["cluster"].astype(int)
    return all_metrics, pcs_selected


def selected_lookup(pcs_selected: pd.DataFrame) -> dict[tuple[str, int], object]:
    return {
        (str(row.event), int(row.duration_hours)): row
        for row in pcs_selected.itertuples(index=False)
    }


EVENT_DATA_CACHE: dict[tuple[str, int], object] = {}


def load_event_data(npz_path: str | Path, duration_hours: int):
    key = (str(npz_path), int(duration_hours))
    if key not in EVENT_DATA_CACHE:
        helper = load_duration_helper_module()
        EVENT_DATA_CACHE[key] = helper.load_event_duration_data(Path(npz_path), int(duration_hours))
    return EVENT_DATA_CACHE[key]


def label_by_time(row: object) -> dict[object, int]:
    event_data = load_event_data(row.npz_path, int(row.duration_hours))
    return {
        time_value: int(label)
        for time_value, label in zip(event_data.times, event_data.labels)
    }


def compare_candidate_to_reference(candidate_row: object, reference_row: object) -> dict[str, float | int | str]:
    candidate_labels = label_by_time(candidate_row)
    reference_labels = label_by_time(reference_row)
    common_times = sorted(set(candidate_labels) & set(reference_labels))

    if not common_times:
        return {
            "common_samples": 0,
            "candidate_cluster_samples_in_common_time": 0,
            "reference_24h_cluster_samples_in_common_time": 0,
            "same_samples_with_24h": 0,
            "union_samples": 0,
            "overlap_ratio_of_own_cluster": np.nan,
            "overlap_percent_of_own_cluster": np.nan,
            "reference_coverage_percent": np.nan,
            "jaccard_percent": np.nan,
            "candidate_valid_for_selection": False,
            "match_status": "no_common_time",
        }

    candidate_cluster = int(candidate_row.cluster)
    reference_cluster = int(reference_row.cluster)
    candidate_mask = np.asarray(
        [candidate_labels[item] == candidate_cluster for item in common_times],
        dtype=bool,
    )
    reference_mask = np.asarray(
        [reference_labels[item] == reference_cluster for item in common_times],
        dtype=bool,
    )

    same_count = int(np.logical_and(candidate_mask, reference_mask).sum())
    union_count = int(np.logical_or(candidate_mask, reference_mask).sum())
    candidate_count = int(candidate_mask.sum())
    reference_count = int(reference_mask.sum())

    own_ratio = same_count / candidate_count if candidate_count else np.nan
    reference_coverage = same_count / reference_count if reference_count else np.nan
    jaccard = same_count / union_count if union_count else np.nan
    valid = bool(candidate_count >= MIN_CANDIDATE_SAMPLES_IN_COMMON)

    return {
        "common_samples": len(common_times),
        "candidate_cluster_samples_in_common_time": candidate_count,
        "reference_24h_cluster_samples_in_common_time": reference_count,
        "same_samples_with_24h": same_count,
        "union_samples": union_count,
        "overlap_ratio_of_own_cluster": own_ratio,
        "overlap_percent_of_own_cluster": own_ratio * 100 if np.isfinite(own_ratio) else np.nan,
        "reference_coverage_percent": (
            reference_coverage * 100 if np.isfinite(reference_coverage) else np.nan
        ),
        "jaccard_percent": jaccard * 100 if np.isfinite(jaccard) else np.nan,
        "candidate_valid_for_selection": valid,
        "match_status": "valid" if valid else "too_few_candidate_samples",
    }


# =========================================================
# 3. Build matching tables
# =========================================================
def build_candidate_overlap_table(
    all_metrics: pd.DataFrame,
    pcs_selected: pd.DataFrame,
) -> pd.DataFrame:
    lookup = selected_lookup(pcs_selected)
    reference_rows = {
        event: row
        for (event, duration), row in lookup.items()
        if int(duration) == REFERENCE_DURATION_HOURS
    }
    rows = []

    candidate_metrics = all_metrics[
        all_metrics["duration_hours"].astype(int) != REFERENCE_DURATION_HOURS
    ].copy()

    for candidate in candidate_metrics.itertuples(index=False):
        event = str(candidate.event)
        duration = int(candidate.duration_hours)
        reference = reference_rows.get(event)

        base = {
            "event": event,
            "duration_hours": duration,
            "candidate_cluster": int(candidate.cluster),
            "reference_24h_selected": reference is not None,
            "reference_24h_cluster": int(reference.cluster) if reference is not None else -1,
            "candidate_pcs": float(candidate.pcs) if hasattr(candidate, "pcs") else np.nan,
            "candidate_sample_count_metric": (
                int(candidate.sample_count) if hasattr(candidate, "sample_count") else np.nan
            ),
            "candidate_npz_path": str(candidate.npz_path),
            "reference_24h_npz_path": str(reference.npz_path) if reference is not None else "",
        }

        if reference is None:
            base.update(
                {
                    "common_samples": np.nan,
                    "candidate_cluster_samples_in_common_time": np.nan,
                    "reference_24h_cluster_samples_in_common_time": np.nan,
                    "same_samples_with_24h": np.nan,
                    "union_samples": np.nan,
                    "overlap_ratio_of_own_cluster": np.nan,
                    "overlap_percent_of_own_cluster": np.nan,
                    "reference_coverage_percent": np.nan,
                    "jaccard_percent": np.nan,
                    "candidate_valid_for_selection": False,
                    "match_status": "reference_24h_not_selected",
                }
            )
        else:
            base.update(compare_candidate_to_reference(candidate, reference))

        rows.append(base)

    return sort_event_duration(pd.DataFrame(rows))


def select_best_matched_clusters(candidate_df: pd.DataFrame) -> pd.DataFrame:
    valid = candidate_df[
        candidate_df["reference_24h_selected"]
        & candidate_df["candidate_valid_for_selection"]
        & candidate_df["overlap_percent_of_own_cluster"].notna()
    ].copy()

    if valid.empty:
        return pd.DataFrame()

    valid = valid.sort_values(
        [
            "event",
            "duration_hours",
            "overlap_percent_of_own_cluster",
            "same_samples_with_24h",
            "jaccard_percent",
            "candidate_pcs",
        ],
        ascending=[True, True, False, False, False, False],
    )
    selected = valid.groupby(["event", "duration_hours"], as_index=False).head(1).copy()
    selected = selected.rename(columns={"candidate_cluster": "matched_cluster"})
    selected["selection_rule"] = (
        "max overlap_percent_of_own_cluster; tie by intersection, Jaccard, PCS"
    )
    return sort_event_duration(selected)


def summarize_by_duration(selected_df: pd.DataFrame, candidate_df: pd.DataFrame) -> pd.DataFrame:
    durations = sorted(candidate_df["duration_hours"].dropna().astype(int).unique())
    rows = []
    for duration in durations:
        duration_candidates = candidate_df[candidate_df["duration_hours"] == duration]
        duration_selected = selected_df[selected_df["duration_hours"] == duration]
        rows.append(
            {
                "duration_hours": int(duration),
                "events_with_candidates": int(duration_candidates["event"].nunique()),
                "events_with_24h_reference": int(
                    duration_candidates.loc[
                        duration_candidates["reference_24h_selected"], "event"
                    ].nunique()
                ),
                "matched_events": int(duration_selected["event"].nunique()),
                "mean_overlap_percent_of_own_cluster": duration_selected[
                    "overlap_percent_of_own_cluster"
                ].mean(),
                "median_overlap_percent_of_own_cluster": duration_selected[
                    "overlap_percent_of_own_cluster"
                ].median(),
                "q25_overlap_percent_of_own_cluster": duration_selected[
                    "overlap_percent_of_own_cluster"
                ].quantile(0.25),
                "q75_overlap_percent_of_own_cluster": duration_selected[
                    "overlap_percent_of_own_cluster"
                ].quantile(0.75),
                "matched_events_ge_80pct": int(
                    (duration_selected["overlap_percent_of_own_cluster"] >= 80).sum()
                ),
                "matched_events_ge_50pct": int(
                    (duration_selected["overlap_percent_of_own_cluster"] >= 50).sum()
                ),
                "median_reference_coverage_percent": duration_selected[
                    "reference_coverage_percent"
                ].median(),
                "median_jaccard_percent": duration_selected["jaccard_percent"].median(),
            }
        )
    return pd.DataFrame(rows)


# =========================================================
# 4. Figures
# =========================================================
def plot_matched_curves_by_duration(selected_df: pd.DataFrame) -> list[Path]:
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    for duration, group in selected_df.groupby("duration_hours", sort=True):
        duration = int(duration)
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

        for row in group.sort_values("event", key=lambda s: s.map(natural_event_key)).itertuples(index=False):
            event_data = load_event_data(row.candidate_npz_path, duration)
            mask = event_data.labels.astype(int) == int(row.matched_cluster)
            if mask.sum() == 0:
                continue
            order = np.argsort(event_data.hours_to_end)
            x = event_data.hours_to_end[order]
            y = np.cumsum(mask[order])
            ax.plot(
                x,
                y,
                linewidth=1.0,
                alpha=0.75,
                label=str(row.event),
            )

        ax.set_xlim(-duration, 0.05)
        ax.set_xlabel("Time before event end (h)")
        ax.set_ylabel("Cumulative count")
        ax.set_title(
            f"24 h-matched precursor clusters, duration={duration} h\n"
            f"cluster selected by maximum sample overlap with the 24 h PCS cluster",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(alpha=0.25)
        if len(group) <= 16:
            ax.legend(ncol=2, fontsize=8, frameon=True)

        out = CURVE_DIR / f"duration_{duration:02d}h_matched_to_24h_cumulative_curves.png"
        fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(out)

    return paths


def plot_overlap_heatmap(selected_df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pivot = selected_df.pivot(
        index="event",
        columns="duration_hours",
        values="overlap_percent_of_own_cluster",
    )
    events = sorted(pivot.index.astype(str), key=natural_event_key)
    durations = sorted(pivot.columns.astype(int))
    pivot = pivot.loc[events, durations]

    fig, ax = plt.subplots(
        figsize=(max(9, len(durations) * 0.85), max(9, len(events) * 0.24)),
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
    ax.set_title(
        "Best cluster overlap with the 24 h PCS precursor cluster",
        fontsize=13,
        fontweight="bold",
    )

    cluster_lookup = {
        (str(row.event), int(row.duration_hours)): int(row.matched_cluster)
        for row in selected_df.itertuples(index=False)
    }
    for row_idx, event in enumerate(events):
        for col_idx, duration in enumerate(durations):
            value = pivot.loc[event, duration]
            if pd.notna(value):
                cluster = cluster_lookup.get((event, int(duration)), -1)
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.0f}\nC{cluster}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Overlap with 24 h PCS cluster (% of matched cluster)")

    out = OUTPUT_DIR / "best_cluster_overlap_with_24h_heatmap.png"
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
        label="Median best-cluster overlap",
    )
    ax.plot(
        x,
        summary_df["mean_overlap_percent_of_own_cluster"],
        color="#E76F51",
        marker="o",
        linewidth=2.0,
        label="Mean best-cluster overlap",
    )
    ax.axhline(50, color="#F58518", linestyle="--", linewidth=1.2, label="50%")
    ax.axhline(80, color="#54A24B", linestyle="--", linewidth=1.2, label="80%")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Observation-window length")
    ax.set_ylabel("Overlap with 24 h PCS cluster (%)")
    ax.set_title("Best-matched cluster consistency relative to 24 h")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)

    out = OUTPUT_DIR / "best_cluster_overlap_duration_summary.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_selected_cluster_heatmap(selected_df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pivot = selected_df.pivot(index="event", columns="duration_hours", values="matched_cluster")
    events = sorted(pivot.index.astype(str), key=natural_event_key)
    durations = sorted(pivot.columns.astype(int))
    pivot = pivot.loc[events, durations]

    fig, ax = plt.subplots(
        figsize=(max(9, len(durations) * 0.85), max(9, len(events) * 0.24)),
        constrained_layout=True,
    )
    matrix = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    cmap = plt.get_cmap("tab20").copy()
    cmap.set_bad("#E5E5E5")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(durations)))
    ax.set_xticklabels([f"{int(duration)}h" for duration in durations])
    ax.set_yticks(np.arange(len(events)))
    ax.set_yticklabels(events)
    ax.set_xlabel("Observation-window length")
    ax.set_ylabel("Event")
    ax.set_title("Cluster selected by maximum overlap with 24 h PCS cluster")

    for row_idx, event in enumerate(events):
        for col_idx, duration in enumerate(durations):
            value = pivot.loc[event, duration]
            if pd.notna(value):
                ax.text(col_idx, row_idx, f"C{int(value)}", ha="center", va="center", fontsize=6)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Matched cluster id")

    out = OUTPUT_DIR / "selected_cluster_id_heatmap.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# =========================================================
# 5. Export and main
# =========================================================
def export_results(
    candidate_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "match_24h_selected_clusters.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="duration_summary", index=False)
        selected_df.to_excel(writer, sheet_name="best_matched_clusters", index=False)
        candidate_df.to_excel(writer, sheet_name="all_candidate_overlaps", index=False)
    return out


def main() -> None:
    all_metrics, pcs_selected = load_pcs_tables()
    candidate_df = build_candidate_overlap_table(all_metrics, pcs_selected)
    selected_df = select_best_matched_clusters(candidate_df)
    summary_df = summarize_by_duration(selected_df, candidate_df)

    excel_path = export_results(candidate_df, selected_df, summary_df)
    curve_paths = plot_matched_curves_by_duration(selected_df)
    figure_paths = [
        plot_overlap_heatmap(selected_df),
        plot_selected_cluster_heatmap(selected_df),
        plot_duration_summary(summary_df),
    ] + curve_paths

    print("Saved Excel:", excel_path)
    print("Saved figures:")
    for path in figure_paths:
        print("  ", path)
    print("\nDuration summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
