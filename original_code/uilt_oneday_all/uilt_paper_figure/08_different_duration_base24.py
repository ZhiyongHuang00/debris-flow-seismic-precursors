# -*- coding: utf-8 -*-
"""Track the 24 h reference state across independently clustered shorter windows.

For each event and each 1--18 h observation window, the script selects the
short-window cluster that captures the largest fraction of samples assigned to
the 24 h PCS-selected reference state within the common time interval:

    24 h-state capture = intersection / 24 h reference-state samples

Ties are resolved by matched-cluster purity, Jaccard similarity, and then the
numeric cluster identifier. PCS is not used for state alignment, allowing it
to be evaluated as a separate property of the aligned cluster.
The selected state is called a *24 h-aligned dominant cluster*; it is a
post-hoc state-tracking result, not an independently detected precursor.

The figure tests three linked questions:
  a. Does the dominant cluster capture more of the 24 h state than expected
     after preserving cluster sizes and temporal autocorrelation?
  b. Does that dominant cluster retain a high precursor-cluster score (PCS)
     after temporal truncation and independent reclustering?
  c. Does its event-balanced occupancy evolve similarly to the 24 h reference
     state as initiation is approached?

Circular time shifts provide an event-specific null for the maximum capture
obtained after searching across all eligible short-window clusters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# -----------------------------------------------------------------------------
# Paths and analysis contract
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MATCH_ROOT = (
    PROJECT_ROOT
    / "03visual"
    / "compared_different_duration"
    / "pcs_different_duration"
)
MATCH_WORKBOOK = (
    MATCH_ROOT
    / "match_24h_selected_clusters"
    / "match_24h_selected_clusters.xlsx"
)
# The 24 h reference must be identical to the event-specific DEC/PCS result
# used by the main 24 h analysis (09_export_precursor_curve_metrics.py).
# MATCH_WORKBOOK remains the source of independently reclustered 1--18 h
# candidates only; its legacy 24 h reference and overlap columns are ignored.
REFERENCE_PCS_WORKBOOK = (
    PROJECT_ROOT
    / "02clustering"
    / "result"
    / "DEC_result"
    / "precursor_curve_metrics"
    / "precursor_by_metric"
    / "pcs_precursor_clusters.xlsx"
)
REFERENCE_PCS_SHEET = "selected_precursor"
REFERENCE_NPZ_DIR = (
    PROJECT_ROOT / "02clustering" / "result" / "DEC_result" / "all_parement"
)
EXPECTED_REFERENCE_EVENTS = 35
OUTPUT_DIR = SCRIPT_DIR / "different_duration_base24"
OUTPUT_STEM = OUTPUT_DIR / "base24_aligned_dominant_cluster_recovery"
SOURCE_DATA_FILE = OUTPUT_DIR / "base24_aligned_dominant_cluster_source_data.xlsx"

DURATION_ORDER = [1, 2, 3, 4, 6, 8, 12, 18]
REFERENCE_DURATION_HOURS = 24
TIME_BINS = 10
MIN_MATCHED_CLUSTER_SAMPLES = 5
MAX_CIRCULAR_SHIFTS = 999
BOOTSTRAP_ITERATIONS = 5_000
SIGN_FLIP_ITERATIONS = 100_000
RANDOM_SEED = 20260802

# Double-column journal figure.
FIG_WIDTH_MM = 183
FIG_HEIGHT_MM = 178
PNG_DPI = 400
TIFF_DPI = 600


# -----------------------------------------------------------------------------
# Visual language
# -----------------------------------------------------------------------------
BLUE = "#2F6F9F"
BLUE_LIGHT = "#C9DDEB"
ORANGE = "#D87943"
ORANGE_LIGHT = "#F0D5C5"
TEAL = "#347F78"
GREY = "#777777"
LIGHT_GREY = "#D8D8D8"
PALE_GREY = "#EEEEEE"
DARK = "#252525"
GRID_GREY = "#E7E7E7"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.labelsize": 7.4,
        "axes.titlesize": 7.4,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.70,
        "ytick.major.width": 0.70,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }
)
plt.ioff()


# -----------------------------------------------------------------------------
# Data structures and validation
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class LabelSeries:
    times: np.ndarray
    labels: np.ndarray


LABEL_CACHE: dict[str, LabelSeries] = {}


def require_columns(df: pd.DataFrame, columns: set[str], table_name: str) -> None:
    missing = sorted(columns.difference(df.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def as_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0)
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
        "nan": False,
        "none": False,
        "": False,
    }
    normalized = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized.unique()).difference(mapping))
    if unknown:
        raise ValueError(f"Unrecognized boolean values: {unknown}")
    return normalized.map(mapping).astype(bool)


def natural_event_key(value: object) -> tuple[int, str]:
    text = str(value).strip().lower()
    match = re.search(r"event\s*(\d+)", text, flags=re.IGNORECASE)
    return (int(match.group(1)), text) if match else (10**9, text)


def parse_timestamp(value: object) -> datetime | None:
    match = re.search(r"(\d{8}T\d{6})", str(value))
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")


def load_label_series(path_value: object) -> LabelSeries:
    path = Path(str(path_value))
    cache_key = str(path.resolve())
    if cache_key in LABEL_CACHE:
        return LABEL_CACHE[cache_key]
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")

    data = np.load(path, allow_pickle=True)
    required = {"x_file", "y_pred_last"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"{path} is missing NPZ arrays: {missing}")

    raw_times = np.asarray([parse_timestamp(item) for item in data["x_file"]], dtype=object)
    labels = np.asarray(data["y_pred_last"], dtype=int)
    valid = np.asarray([item is not None for item in raw_times], dtype=bool)
    if valid.sum() == 0:
        raise ValueError(f"No valid timestamps parsed from {path}")

    times = raw_times[valid]
    labels = labels[valid]
    order = np.argsort(times)
    times = times[order]
    labels = labels[order]
    if len(set(times.tolist())) != len(times):
        raise ValueError(f"Duplicate timestamps found in {path}")

    result = LabelSeries(times=times, labels=labels)
    LABEL_CACHE[cache_key] = result
    return result


def load_main_24h_reference_table() -> pd.DataFrame:
    """Load the exact 24 h PCS selections used by the main-paper analysis."""
    if not REFERENCE_PCS_WORKBOOK.exists():
        raise FileNotFoundError(
            f"Main 24 h PCS workbook not found: {REFERENCE_PCS_WORKBOOK}"
        )
    references = pd.read_excel(
        REFERENCE_PCS_WORKBOOK,
        sheet_name=REFERENCE_PCS_SHEET,
    )
    require_columns(
        references,
        {"event", "cluster", "selection_status"},
        f"{REFERENCE_PCS_WORKBOOK.name}:{REFERENCE_PCS_SHEET}",
    )
    references["event"] = (
        references["event"].astype(str).str.strip().str.lower()
    )
    references["cluster"] = pd.to_numeric(
        references["cluster"], errors="coerce"
    )
    selected = references[
        references["selection_status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("selected")
        & references["cluster"].ge(0)
    ][["event", "cluster", "selection_status"]].copy()
    if selected["event"].duplicated().any():
        duplicates = sorted(selected.loc[selected["event"].duplicated(), "event"])
        raise ValueError(f"Duplicate main 24 h PCS selections: {duplicates}")

    selected["reference_24h_cluster"] = selected["cluster"].astype(int)
    selected["reference_24h_selected"] = True
    selected["reference_24h_npz_path"] = selected["event"].map(
        lambda event: str(REFERENCE_NPZ_DIR / f"{event}_all_parement.npz")
    )
    missing_npz = [
        path
        for path in selected["reference_24h_npz_path"]
        if not Path(path).exists()
    ]
    if missing_npz:
        raise FileNotFoundError(
            "Main 24 h reference NPZ files are missing: " + ", ".join(missing_npz)
        )
    if len(selected) != EXPECTED_REFERENCE_EVENTS:
        raise ValueError(
            f"Expected {EXPECTED_REFERENCE_EVENTS} main 24 h PCS-selected events, "
            f"found {len(selected)}."
        )
    return selected[
        [
            "event",
            "reference_24h_selected",
            "reference_24h_cluster",
            "reference_24h_npz_path",
            "selection_status",
        ]
    ].rename(columns={"selection_status": "reference_24h_selection_status"})


def align_label_series(
    candidate_path: object,
    reference_path: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align candidate and reference cluster labels by exact timestamp."""
    candidate = load_label_series(candidate_path)
    reference = load_label_series(reference_path)
    candidate_by_time = {
        time_value: int(label)
        for time_value, label in zip(candidate.times, candidate.labels)
    }
    reference_by_time = {
        time_value: int(label)
        for time_value, label in zip(reference.times, reference.labels)
    }
    common_times = sorted(set(candidate_by_time).intersection(reference_by_time))
    if not common_times:
        raise ValueError(
            f"No common timestamps between {candidate_path} and {reference_path}"
        )
    candidate_labels = np.asarray(
        [candidate_by_time[item] for item in common_times], dtype=int
    )
    reference_labels = np.asarray(
        [reference_by_time[item] for item in common_times], dtype=int
    )
    return np.asarray(common_times, dtype=object), candidate_labels, reference_labels


def recompute_main_reference_overlaps(
    candidates: pd.DataFrame,
    references: pd.DataFrame,
) -> pd.DataFrame:
    """Replace legacy K=7 overlaps with overlaps to the main 24 h state."""
    stale_reference_columns = {
        "reference_24h_selected",
        "reference_24h_cluster",
        "reference_24h_npz_path",
        "reference_24h_selection_status",
        "common_samples",
        "candidate_cluster_samples_in_common_time",
        "reference_24h_cluster_samples_in_common_time",
        "same_samples_with_24h",
        "union_samples",
        "overlap_ratio_of_own_cluster",
        "overlap_percent_of_own_cluster",
        "reference_coverage_percent",
        "jaccard_percent",
        "candidate_valid_for_selection",
        "match_status",
    }
    candidates = candidates.drop(
        columns=[
            column for column in stale_reference_columns if column in candidates.columns
        ]
    )
    candidates = candidates.merge(
        references,
        on="event",
        how="inner",
        validate="many_to_one",
    )
    if candidates.empty:
        raise ValueError("No short-window candidates match the main 24 h event list.")

    overlap_rows: list[pd.DataFrame] = []
    group_columns = [
        "event",
        "duration_hours",
        "candidate_npz_path",
        "reference_24h_npz_path",
        "reference_24h_cluster",
    ]
    for _, group in candidates.groupby(group_columns, sort=False, dropna=False):
        first = group.iloc[0]
        common_times, candidate_labels, reference_labels = align_label_series(
            first["candidate_npz_path"],
            first["reference_24h_npz_path"],
        )
        reference_mask = reference_labels == int(first["reference_24h_cluster"])
        reference_count = int(reference_mask.sum())
        updated = group.copy()
        updated["common_samples"] = len(common_times)
        updated["reference_24h_cluster_samples_in_common_time"] = reference_count

        candidate_counts: list[int] = []
        intersections: list[int] = []
        unions: list[int] = []
        for cluster in updated["candidate_cluster"].astype(int):
            candidate_mask = candidate_labels == cluster
            candidate_count = int(candidate_mask.sum())
            intersection = int(np.logical_and(candidate_mask, reference_mask).sum())
            union = candidate_count + reference_count - intersection
            candidate_counts.append(candidate_count)
            intersections.append(intersection)
            unions.append(union)

        updated["candidate_cluster_samples_in_common_time"] = candidate_counts
        updated["same_samples_with_24h"] = intersections
        updated["union_samples"] = unions
        updated["overlap_ratio_of_own_cluster"] = np.divide(
            intersections,
            candidate_counts,
            out=np.full(len(updated), np.nan, dtype=float),
            where=np.asarray(candidate_counts) > 0,
        )
        updated["overlap_percent_of_own_cluster"] = (
            100.0 * updated["overlap_ratio_of_own_cluster"]
        )
        updated["reference_coverage_percent"] = (
            100.0 * np.asarray(intersections, dtype=float) / reference_count
            if reference_count > 0
            else np.nan
        )
        updated["jaccard_percent"] = np.divide(
            100.0 * np.asarray(intersections, dtype=float),
            unions,
            out=np.full(len(updated), np.nan, dtype=float),
            where=np.asarray(unions) > 0,
        )
        updated["candidate_valid_for_selection"] = (
            np.asarray(candidate_counts) >= MIN_MATCHED_CLUSTER_SAMPLES
        ) & (reference_count > 0)
        updated["match_status"] = "recomputed_against_main_24h_reference"
        overlap_rows.append(updated)

    result = pd.concat(overlap_rows, ignore_index=True)
    event_duration_counts = result.groupby("duration_hours")["event"].nunique()
    incomplete = {
        int(duration): int(count)
        for duration, count in event_duration_counts.items()
        if int(count) != EXPECTED_REFERENCE_EVENTS
    }
    if incomplete or sorted(event_duration_counts.index.astype(int)) != DURATION_ORDER:
        raise ValueError(
            "Main-reference candidate coverage is incomplete; event counts by "
            f"duration: {event_duration_counts.to_dict()}"
        )
    return result


def load_and_select_dominant_clusters() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Select the cluster capturing the largest share of the 24 h state."""
    if not MATCH_WORKBOOK.exists():
        raise FileNotFoundError(f"Matching workbook not found: {MATCH_WORKBOOK}")
    candidates = pd.read_excel(MATCH_WORKBOOK, sheet_name="all_candidate_overlaps")

    require_columns(
        candidates,
        {
            "event",
            "duration_hours",
            "candidate_cluster",
            "candidate_pcs",
            "candidate_npz_path",
        },
        "all_candidate_overlaps",
    )
    candidates["event"] = candidates["event"].astype(str).str.strip().str.lower()
    candidates["duration_hours"] = pd.to_numeric(
        candidates["duration_hours"], errors="coerce"
    )
    candidates["candidate_cluster"] = pd.to_numeric(
        candidates["candidate_cluster"], errors="coerce"
    )
    numeric_columns = ["candidate_pcs"]
    for column in numeric_columns:
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")

    candidates = candidates[
        candidates["duration_hours"].isin(DURATION_ORDER)
        & candidates["candidate_cluster"].notna()
    ].copy()
    candidates["duration_hours"] = candidates["duration_hours"].astype(int)
    candidates["candidate_cluster"] = candidates["candidate_cluster"].astype(int)

    references = load_main_24h_reference_table()
    candidates = recompute_main_reference_overlaps(candidates, references)

    reference_present = (
        candidates["reference_24h_selected"]
        & candidates["candidate_valid_for_selection"]
        & candidates["reference_coverage_percent"].notna()
        & candidates["reference_24h_cluster_samples_in_common_time"].gt(0)
    )
    eligible = candidates[
        reference_present
        & candidates["candidate_cluster_samples_in_common_time"].ge(
            MIN_MATCHED_CLUSTER_SAMPLES
        )
    ].copy()
    if eligible.empty:
        raise ValueError("No eligible short-window clusters were found.")

    eligible = eligible.sort_values(
        [
            "event",
            "duration_hours",
            "reference_coverage_percent",
            "overlap_percent_of_own_cluster",
            "jaccard_percent",
            "candidate_cluster",
        ],
        ascending=[True, True, False, False, False, True],
    )
    selected = eligible.groupby(["event", "duration_hours"], as_index=False).head(1)
    selected = selected.rename(columns={"candidate_cluster": "aligned_cluster"})
    selected["selection_rule"] = (
        "maximum 24 h-state capture; ties by aligned-cluster purity, "
        "Jaccard similarity, and numeric cluster identifier"
    )
    selected["reference_prevalence_percent"] = (
        100.0
        * selected["reference_24h_cluster_samples_in_common_time"]
        / selected["common_samples"]
    )
    selected["aligned_cluster_prevalence_percent"] = (
        100.0
        * selected["candidate_cluster_samples_in_common_time"]
        / selected["common_samples"]
    )
    selected["capture_lift_over_random_occupancy"] = (
        selected["reference_coverage_percent"]
        / selected["aligned_cluster_prevalence_percent"].replace(0, np.nan)
    )

    # Fragmentation of the 24 h state across all short-window clusters.
    fragmentation_rows: list[dict[str, float | int | str]] = []
    fragmentation_source = candidates[reference_present].copy()
    for (event, duration), group in fragmentation_source.groupby(
        ["event", "duration_hours"], sort=False
    ):
        shares = (
            group["reference_coverage_percent"]
            .dropna()
            .clip(lower=0)
            .sort_values(ascending=False)
            .to_numpy(dtype=float)
        )
        if shares.size == 0:
            continue
        top1 = float(shares[0])
        top2 = float(shares[1]) if shares.size > 1 else 0.0
        total = float(shares.sum())
        probabilities = shares / total if total > 0 else np.zeros_like(shares)
        if shares.size > 1 and np.any(probabilities > 0):
            entropy = float(
                -np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0]))
                / np.log(shares.size)
            )
        else:
            entropy = 0.0
        fragmentation_rows.append(
            {
                "event": event,
                "duration_hours": int(duration),
                "n_short_window_clusters": int(shares.size),
                "top1_reference_share_percent": top1,
                "top2_reference_share_percent": top2,
                "top1_plus_top2_percent": min(100.0, top1 + top2),
                "remaining_reference_share_percent": max(0.0, 100.0 - top1 - top2),
                "normalized_fragmentation_entropy": entropy,
                "coverage_sum_percent": total,
            }
        )
    fragmentation = pd.DataFrame(fragmentation_rows)

    selected["_event_order"] = selected["event"].map(natural_event_key)
    selected["_duration_order"] = selected["duration_hours"].map(
        {duration: index for index, duration in enumerate(DURATION_ORDER)}
    )
    selected = selected.sort_values(["_duration_order", "_event_order"]).drop(
        columns=["_event_order", "_duration_order"]
    )

    observed_durations = sorted(selected["duration_hours"].unique())
    if observed_durations != DURATION_ORDER:
        raise ValueError(
            f"Expected matched durations {DURATION_ORDER}, found {observed_durations}."
        )
    event_counts = selected.groupby("duration_hours")["event"].nunique()
    if not event_counts.eq(EXPECTED_REFERENCE_EVENTS).all():
        raise ValueError(
            "Expected one aligned cluster for each of the "
            f"{EXPECTED_REFERENCE_EVENTS} reference events per duration; found "
            f"{event_counts.to_dict()}."
        )
    return candidates, selected.reset_index(drop=True), fragmentation


# -----------------------------------------------------------------------------
# Event-level matching null and temporal occupancy
# -----------------------------------------------------------------------------
def common_label_arrays(row: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return align_label_series(row.candidate_npz_path, row.reference_24h_npz_path)


def circular_shift_max_capture(
    candidate_labels: np.ndarray,
    reference_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Maximum reference capture after non-zero circular shifts."""
    n_samples = len(reference_mask)
    if n_samples < 2 or int(reference_mask.sum()) == 0:
        return np.asarray([], dtype=float)
    eligible_masks = [
        candidate_labels == cluster
        for cluster in np.unique(candidate_labels)
        if int((candidate_labels == cluster).sum()) >= MIN_MATCHED_CLUSTER_SAMPLES
    ]
    if not eligible_masks:
        return np.asarray([], dtype=float)

    available_shifts = np.arange(1, n_samples, dtype=int)
    if len(available_shifts) <= MAX_CIRCULAR_SHIFTS:
        shifts = available_shifts
    else:
        shifts = rng.choice(
            available_shifts,
            size=MAX_CIRCULAR_SHIFTS,
            replace=False,
        )

    reference_count = int(reference_mask.sum())
    null_values = np.empty(len(shifts), dtype=float)
    for index, shift in enumerate(shifts):
        shifted = np.roll(reference_mask, int(shift))
        null_values[index] = max(
            float(np.logical_and(shifted, mask).sum()) / reference_count * 100.0
            for mask in eligible_masks
        )
    return null_values


def analyse_event_states(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    null_rows: list[dict[str, float | int | str]] = []
    occupancy_rows: list[dict[str, float | int | str]] = []
    temporal_feature_rows: list[dict[str, float | int | str]] = []

    for row in selected.itertuples(index=False):
        duration = int(row.duration_hours)
        times, candidate_labels, reference_labels = common_label_arrays(row)
        if len(times) != int(row.common_samples):
            raise ValueError(
                f"Common-sample mismatch for {row.event}, {duration} h: "
                f"{len(times)} vs {int(row.common_samples)}"
            )

        aligned_mask = candidate_labels == int(row.aligned_cluster)
        reference_mask = reference_labels == int(row.reference_24h_cluster)
        intersection = int(np.logical_and(aligned_mask, reference_mask).sum())
        observed_capture = intersection / int(reference_mask.sum()) * 100.0
        if not np.isclose(observed_capture, float(row.reference_coverage_percent)):
            raise ValueError(
                f"Capture mismatch for {row.event}, {duration} h: "
                f"{observed_capture} vs {row.reference_coverage_percent}"
            )

        null = circular_shift_max_capture(candidate_labels, reference_mask, rng)
        if null.size == 0:
            null_median = np.nan
            null_low = np.nan
            null_high = np.nan
            empirical_p = np.nan
        else:
            null_median = float(np.median(null))
            null_low, null_high = np.percentile(null, [2.5, 97.5])
            empirical_p = float((1 + np.sum(null >= observed_capture)) / (1 + null.size))
        null_rows.append(
            {
                "event": row.event,
                "duration_hours": duration,
                "observed_capture_percent": observed_capture,
                "n_circular_shifts": int(null.size),
                "null_median_max_capture_percent": null_median,
                "null_2.5_percent": float(null_low),
                "null_97.5_percent": float(null_high),
                "capture_excess_over_null_percent": observed_capture - null_median,
                "capture_ratio_to_null": (
                    observed_capture / null_median if null_median > 0 else np.nan
                ),
                "empirical_p_value": empirical_p,
                "above_null_p_lt_0.05": bool(empirical_p < 0.05)
                if np.isfinite(empirical_p)
                else False,
            }
        )

        end_time = max(times)
        hours_to_end = np.asarray(
            [(item - end_time).total_seconds() / 3600.0 for item in times],
            dtype=float,
        )
        edges = np.linspace(-duration, 0.0, TIME_BINS + 1)
        bin_ids = np.digitize(hours_to_end, edges[1:-1], right=False)
        event_occupancy: dict[str, list[float]] = {
            "24 h-aligned dominant cluster": [],
            "24 h reference state": [],
        }
        for bin_index in range(TIME_BINS):
            in_bin = bin_ids == bin_index
            if not np.any(in_bin):
                aligned_occupancy = np.nan
                reference_occupancy = np.nan
            else:
                aligned_occupancy = float(aligned_mask[in_bin].mean() * 100.0)
                reference_occupancy = float(reference_mask[in_bin].mean() * 100.0)
            bin_center = float((edges[bin_index] + edges[bin_index + 1]) / 2.0)
            relative_position = float((bin_index + 0.5) / TIME_BINS)
            for state_name, occupancy in [
                ("24 h-aligned dominant cluster", aligned_occupancy),
                ("24 h reference state", reference_occupancy),
            ]:
                event_occupancy[state_name].append(occupancy)
                occupancy_rows.append(
                    {
                        "event": row.event,
                        "duration_hours": duration,
                        "time_bin": bin_index + 1,
                        "time_before_window_end_hours": bin_center,
                        "relative_window_position": relative_position,
                        "state": state_name,
                        "state_occupancy_percent": occupancy,
                    }
                )

        aligned_values = np.asarray(
            event_occupancy["24 h-aligned dominant cluster"], dtype=float
        )
        reference_values = np.asarray(
            event_occupancy["24 h reference state"], dtype=float
        )
        temporal_feature_rows.append(
            {
                "event": row.event,
                "duration_hours": duration,
                "aligned_cluster_samples": int(aligned_mask.sum()),
                "reference_state_samples": int(reference_mask.sum()),
                "aligned_first_20pct_occupancy": float(np.nanmean(aligned_values[:2])),
                "aligned_last_20pct_occupancy": float(np.nanmean(aligned_values[-2:])),
                "aligned_terminal_change_percent": float(
                    np.nanmean(aligned_values[-2:]) - np.nanmean(aligned_values[:2])
                ),
                "reference_first_20pct_occupancy": float(np.nanmean(reference_values[:2])),
                "reference_last_20pct_occupancy": float(np.nanmean(reference_values[-2:])),
                "reference_terminal_change_percent": float(
                    np.nanmean(reference_values[-2:]) - np.nanmean(reference_values[:2])
                ),
            }
        )

    return (
        pd.DataFrame(null_rows),
        pd.DataFrame(occupancy_rows),
        pd.DataFrame(temporal_feature_rows),
    )


# -----------------------------------------------------------------------------
# Statistical summaries
# -----------------------------------------------------------------------------
def bootstrap_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    statistic: str,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    if values.size == 1:
        value = float(values[0])
        return value, value, value
    samples = rng.choice(
        values,
        size=(BOOTSTRAP_ITERATIONS, values.size),
        replace=True,
    )
    if statistic == "median":
        observed = float(np.median(values))
        bootstrap_values = np.median(samples, axis=1)
    elif statistic == "mean":
        observed = float(np.mean(values))
        bootstrap_values = np.mean(samples, axis=1)
    else:
        raise ValueError(f"Unsupported statistic: {statistic}")
    low, high = np.percentile(bootstrap_values, [2.5, 97.5])
    return observed, float(low), float(high)


METRIC_SPECS = {
    "reference_coverage_percent": {
        "label": "24 h-state capture",
        "color": BLUE,
        "marker": "o",
        "linestyle": "-",
    },
    "overlap_percent_of_own_cluster": {
        "label": "Aligned-cluster purity",
        "color": ORANGE,
        "marker": "s",
        "linestyle": "--",
    },
    "jaccard_percent": {
        "label": "Jaccard similarity",
        "color": TEAL,
        "marker": "D",
        "linestyle": "-.",
    },
    "null_median_max_capture_percent": {
        "label": "Circular-shift null",
        "color": GREY,
        "marker": "^",
        "linestyle": ":",
    },
}


def build_metric_summary(event_metrics: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 1)
    rows: list[dict[str, float | int | str]] = []
    for duration in DURATION_ORDER:
        duration_data = event_metrics[event_metrics["duration_hours"] == duration]
        for metric, spec in METRIC_SPECS.items():
            values = duration_data[metric].to_numpy(dtype=float)
            centre, low, high = bootstrap_interval(values, rng, statistic="median")
            rows.append(
                {
                    "duration_hours": duration,
                    "metric": metric,
                    "metric_label": spec["label"],
                    "n_events": int(np.isfinite(values).sum()),
                    "median_percent": centre,
                    "bootstrap_95_low_percent": low,
                    "bootstrap_95_high_percent": high,
                }
            )
    return pd.DataFrame(rows)


def build_aligned_pcs_summary(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize the PCS of the 24 h-aligned cluster without reclassifying it."""
    pcs_events = selected[
        ["event", "duration_hours", "aligned_cluster", "candidate_pcs"]
    ].copy()
    pcs_events = pcs_events.rename(columns={"candidate_pcs": "pcs"})
    pcs_events["pcs_available"] = pcs_events["pcs"].notna()

    rows: list[dict[str, float | int]] = []
    for duration in DURATION_ORDER:
        subset = pcs_events[pcs_events["duration_hours"] == duration]
        values = subset["pcs"].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "duration_hours": duration,
                "n_events_total": int(len(subset)),
                "n_events_with_pcs": int(values.size),
                "median_pcs": float(np.median(values)) if values.size else np.nan,
                "q1_pcs": float(np.percentile(values, 25)) if values.size else np.nan,
                "q3_pcs": float(np.percentile(values, 75)) if values.size else np.nan,
                "minimum_pcs": float(np.min(values)) if values.size else np.nan,
                "maximum_pcs": float(np.max(values)) if values.size else np.nan,
                "events_pcs_above_zero": int(np.sum(values > 0)),
                "events_pcs_above_zero_percent": (
                    float(100.0 * np.mean(values > 0)) if values.size else np.nan
                ),
            }
        )
    return pcs_events, pd.DataFrame(rows)


def build_occupancy_summary(occupancy: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 2)
    rows: list[dict[str, float | int | str]] = []
    for duration in DURATION_ORDER:
        for state in ["24 h-aligned dominant cluster", "24 h reference state"]:
            subset = occupancy[
                (occupancy["duration_hours"] == duration)
                & (occupancy["state"] == state)
            ]
            for bin_index in range(1, TIME_BINS + 1):
                bin_data = subset[subset["time_bin"] == bin_index]
                values = bin_data["state_occupancy_percent"].to_numpy(dtype=float)
                centre, low, high = bootstrap_interval(values, rng, statistic="mean")
                rows.append(
                    {
                        "duration_hours": duration,
                        "state": state,
                        "time_bin": bin_index,
                        "time_before_window_end_hours": float(
                            bin_data["time_before_window_end_hours"].iloc[0]
                        ),
                        "relative_window_position": float(
                            bin_data["relative_window_position"].iloc[0]
                        ),
                        "n_events": int(np.isfinite(values).sum()),
                        "event_mean_occupancy_percent": centre,
                        "bootstrap_95_low_percent": low,
                        "bootstrap_95_high_percent": high,
                    }
                )
    return pd.DataFrame(rows)


def build_fragmentation_summary(fragmentation: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "top1_reference_share_percent",
        "top2_reference_share_percent",
        "top1_plus_top2_percent",
        "remaining_reference_share_percent",
        "normalized_fragmentation_entropy",
    ]
    rows: list[dict[str, float | int]] = []
    for duration in DURATION_ORDER:
        subset = fragmentation[fragmentation["duration_hours"] == duration]
        row: dict[str, float | int] = {
            "duration_hours": duration,
            "n_events": len(subset),
            "median_n_short_window_clusters": float(
                subset["n_short_window_clusters"].median()
            ),
        }
        for column in columns:
            row[f"median_{column}"] = float(subset[column].median())
            row[f"mean_{column}"] = float(subset[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm family-wise correction for a one-dimensional p-value array."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)
    valid_indices = np.flatnonzero(np.isfinite(p_values))
    if valid_indices.size == 0:
        return adjusted
    ordered = valid_indices[np.argsort(p_values[valid_indices])]
    running_max = 0.0
    n_tests = len(ordered)
    for rank, original_index in enumerate(ordered):
        candidate = min(1.0, (n_tests - rank) * p_values[original_index])
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def build_terminal_change_summary(temporal_features: pd.DataFrame) -> pd.DataFrame:
    """Paired event-level test of last-20% minus first-20% occupancy."""
    rows: list[dict[str, float | int]] = []
    for duration in DURATION_ORDER:
        bootstrap_rng = np.random.default_rng(RANDOM_SEED + 300 + duration)
        sign_flip_rng = np.random.default_rng(RANDOM_SEED + 400 + duration)
        values = temporal_features.loc[
            temporal_features["duration_hours"] == duration,
            "aligned_terminal_change_percent",
        ].dropna().to_numpy(dtype=float)
        centre, low, high = bootstrap_interval(
            values,
            bootstrap_rng,
            statistic="mean",
        )
        if values.size:
            signs = sign_flip_rng.choice(
                np.asarray([-1.0, 1.0]),
                size=(SIGN_FLIP_ITERATIONS, values.size),
                replace=True,
            )
            null_means = np.mean(signs * values, axis=1)
            p_value = float(
                (1 + np.sum(np.abs(null_means) >= abs(centre)))
                / (1 + len(null_means))
            )
        else:
            p_value = np.nan
        rows.append(
            {
                "duration_hours": duration,
                "n_events": int(values.size),
                "mean_last20_minus_first20_percent": centre,
                "bootstrap_95_low_percent": low,
                "bootstrap_95_high_percent": high,
                "positive_change_events": int((values > 0).sum()),
                "two_sided_sign_flip_p_value": p_value,
            }
        )
    summary = pd.DataFrame(rows)
    summary["holm_adjusted_p_value"] = holm_adjust(
        summary["two_sided_sign_flip_p_value"].to_numpy(dtype=float)
    )
    return summary


def build_discovery_summary(
    event_metrics: pd.DataFrame,
    null_results: pd.DataFrame,
    pcs_summary: pd.DataFrame,
    temporal_features: pd.DataFrame,
    fragmentation_summary: pd.DataFrame,
    terminal_change_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for duration in DURATION_ORDER:
        metrics = event_metrics[event_metrics["duration_hours"] == duration]
        null = null_results[null_results["duration_hours"] == duration]
        pcs = pcs_summary[pcs_summary["duration_hours"] == duration].iloc[0]
        temporal = temporal_features[temporal_features["duration_hours"] == duration]
        fragment = fragmentation_summary[
            fragmentation_summary["duration_hours"] == duration
        ].iloc[0]
        terminal_test = terminal_change_summary[
            terminal_change_summary["duration_hours"] == duration
        ].iloc[0]
        rows.append(
            {
                "duration_hours": duration,
                "n_reference_events": len(metrics),
                "median_24h_state_capture_percent": float(
                    metrics["reference_coverage_percent"].median()
                ),
                "median_aligned_cluster_purity_percent": float(
                    metrics["overlap_percent_of_own_cluster"].median()
                ),
                "median_jaccard_percent": float(metrics["jaccard_percent"].median()),
                "median_null_max_capture_percent": float(
                    null["null_median_max_capture_percent"].median()
                ),
                "events_above_null_p_lt_0.05": int(
                    null["above_null_p_lt_0.05"].sum()
                ),
                "events_capture_ge_50pct": int(
                    (metrics["reference_coverage_percent"] >= 50).sum()
                ),
                "events_capture_ge_80pct": int(
                    (metrics["reference_coverage_percent"] >= 80).sum()
                ),
                "aligned_cluster_pcs_n": int(pcs["n_events_with_pcs"]),
                "aligned_cluster_median_pcs": float(pcs["median_pcs"]),
                "aligned_cluster_q1_pcs": float(pcs["q1_pcs"]),
                "aligned_cluster_q3_pcs": float(pcs["q3_pcs"]),
                "aligned_cluster_events_pcs_above_zero": int(
                    pcs["events_pcs_above_zero"]
                ),
                "aligned_cluster_events_pcs_above_zero_percent": float(
                    pcs["events_pcs_above_zero_percent"]
                ),
                "mean_aligned_first_20pct_occupancy": float(
                    temporal["aligned_first_20pct_occupancy"].mean()
                ),
                "mean_aligned_last_20pct_occupancy": float(
                    temporal["aligned_last_20pct_occupancy"].mean()
                ),
                "mean_aligned_terminal_change_percent": float(
                    temporal["aligned_terminal_change_percent"].mean()
                ),
                "terminal_change_bootstrap_95_low_percent": float(
                    terminal_test["bootstrap_95_low_percent"]
                ),
                "terminal_change_bootstrap_95_high_percent": float(
                    terminal_test["bootstrap_95_high_percent"]
                ),
                "terminal_change_holm_adjusted_p_value": float(
                    terminal_test["holm_adjusted_p_value"]
                ),
                "events_with_positive_terminal_change": int(
                    terminal_test["positive_change_events"]
                ),
                "median_top1_plus_top2_reference_share_percent": float(
                    fragment["median_top1_plus_top2_percent"]
                ),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def style_axis(ax: plt.Axes, y_grid: bool = True) -> None:
    ax.spines["left"].set_color(DARK)
    ax.spines["bottom"].set_color(DARK)
    ax.tick_params(colors=DARK)
    if y_grid:
        ax.grid(axis="y", color=GRID_GREY, linewidth=0.55, zorder=0)


def add_panel_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.12,
    y: float = 1.08,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="black",
    )


def plot_metric_panel(
    ax: plt.Axes,
    metric_summary: pd.DataFrame,
    null_results: pd.DataFrame,
) -> None:
    x = np.arange(len(DURATION_ORDER), dtype=float)
    for metric, spec in METRIC_SPECS.items():
        subset = (
            metric_summary[metric_summary["metric"] == metric]
            .set_index("duration_hours")
            .reindex(DURATION_ORDER)
        )
        median = subset["median_percent"].to_numpy(dtype=float)
        low = subset["bootstrap_95_low_percent"].to_numpy(dtype=float)
        high = subset["bootstrap_95_high_percent"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            median,
            yerr=np.vstack([median - low, high - median]),
            color=spec["color"],
            marker=spec["marker"],
            linestyle=spec["linestyle"],
            markerfacecolor="white",
            markeredgewidth=0.75,
            markersize=3.5,
            linewidth=1.1,
            elinewidth=0.65,
            capsize=1.8,
            label=spec["label"],
            zorder=3,
        )

    significant_counts = (
        null_results.groupby("duration_hours")["above_null_p_lt_0.05"]
        .sum()
        .reindex(DURATION_ORDER)
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    event_counts = (
        null_results.groupby("duration_hours")["event"]
        .nunique()
        .reindex(DURATION_ORDER)
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    for position, significant, total in zip(x, significant_counts, event_counts):
        ax.text(
            position,
            103.0,
            f"{significant}/{total}",
            ha="center",
            va="bottom",
            fontsize=5.2,
            color=GREY,
        )

    ax.text(
        0.01,
        0.02,
        "top labels: events above circular-shift null (p<0.05)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.0,
        color=GREY,
    )
    ax.set_xlim(-0.45, len(DURATION_ORDER) - 0.55)
    ax.set_ylim(0, 111)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{duration} h" for duration in DURATION_ORDER])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Observation-window duration")
    ax.set_ylabel("State correspondence (%)")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.015),
        ncol=2,
        fontsize=5.35,
        handlelength=2.1,
        columnspacing=0.9,
        borderaxespad=0,
    )
    style_axis(ax)
    add_panel_label(ax, "a")


def plot_aligned_pcs_panel(
    ax: plt.Axes,
    pcs_events: pd.DataFrame,
    pcs_summary: pd.DataFrame,
) -> None:
    """Show event-level PCS distributions for the 24 h-aligned cluster."""
    x = np.arange(len(DURATION_ORDER), dtype=float)
    grouped_values: list[np.ndarray] = []
    for duration in DURATION_ORDER:
        values = pcs_events.loc[
            pcs_events["duration_hours"] == duration, "pcs"
        ].dropna().to_numpy(dtype=float)
        grouped_values.append(values)

    boxplot = ax.boxplot(
        grouped_values,
        positions=x,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": DARK, "linewidth": 1.25},
        boxprops={"facecolor": BLUE_LIGHT, "edgecolor": BLUE, "linewidth": 0.9},
        whiskerprops={"color": BLUE, "linewidth": 0.8},
        capprops={"color": BLUE, "linewidth": 0.8},
        zorder=2,
    )
    for box in boxplot["boxes"]:
        box.set_alpha(0.72)

    rng = np.random.default_rng(RANDOM_SEED + 4)
    for position, duration, values in zip(x, DURATION_ORDER, grouped_values):
        if values.size:
            jitter = rng.uniform(-0.14, 0.14, size=values.size)
            ax.scatter(
                np.full(values.size, position) + jitter,
                values,
                s=9,
                facecolor=BLUE,
                edgecolor="white",
                linewidth=0.3,
                alpha=0.66,
                zorder=3,
            )
        n_value = int(
            pcs_summary.loc[
                pcs_summary["duration_hours"] == duration, "n_events_with_pcs"
            ].iloc[0]
        )
        ax.text(
            position,
            1.015,
            f"n={n_value}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=5.1,
            color=GREY,
        )

    finite_values = np.concatenate(
        [values for values in grouped_values if values.size]
    )
    lower = min(-8.0, float(np.floor(finite_values.min() - 0.75)))
    upper = max(9.0, float(np.ceil(finite_values.max() + 0.75)))
    ax.axhline(0, color=GREY, linewidth=0.75, linestyle=(0, (3, 2)), zorder=1)
    ax.text(
        0.01,
        0.49,
        "PCS = 0",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.0,
        color=GREY,
    )
    ax.set_xlim(-0.5, len(DURATION_ORDER) - 0.5)
    ax.set_ylim(lower, upper)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{duration} h" for duration in DURATION_ORDER])
    ax.set_xlabel("Observation-window duration")
    ax.set_ylabel("Precursor-cluster score (PCS)")
    style_axis(ax)
    add_panel_label(ax, "b")


def plot_occupancy_small_multiples(
    axes: list[plt.Axes],
    occupancy_summary: pd.DataFrame,
    terminal_change_summary: pd.DataFrame,
) -> list[Line2D]:
    max_ci = float(occupancy_summary["bootstrap_95_high_percent"].max())
    temporal_ymax = min(100.0, max(60.0, np.ceil((max_ci + 5.0) / 10.0) * 10.0))

    for index, (ax, duration) in enumerate(zip(axes, DURATION_ORDER)):
        duration_data = occupancy_summary[
            occupancy_summary["duration_hours"] == duration
        ]
        for state, color, linestyle, alpha in [
            ("24 h-aligned dominant cluster", BLUE, "-", 0.15),
            ("24 h reference state", ORANGE, "--", 0.10),
        ]:
            subset = duration_data[duration_data["state"] == state].sort_values(
                "time_bin"
            )
            x = subset["time_before_window_end_hours"].to_numpy(dtype=float)
            mean = subset["event_mean_occupancy_percent"].to_numpy(dtype=float)
            low = subset["bootstrap_95_low_percent"].to_numpy(dtype=float)
            high = subset["bootstrap_95_high_percent"].to_numpy(dtype=float)
            ax.fill_between(x, low, high, color=color, alpha=alpha, linewidth=0)
            ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=1.1)

        n_events = int(duration_data["n_events"].max())
        ax.text(
            0.04,
            0.92,
            f"{duration} h  (n={n_events})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            fontweight="bold",
            color=DARK,
        )
        adjusted_p = float(
            terminal_change_summary.loc[
                terminal_change_summary["duration_hours"] == duration,
                "holm_adjusted_p_value",
            ].iloc[0]
        )
        if np.isfinite(adjusted_p):
            if adjusted_p < 0.001:
                p_label = r"$P_{\mathrm{Holm}}<0.001$"
            else:
                p_label = rf"$P_{{\mathrm{{Holm}}}}={adjusted_p:.3f}$"
        else:
            p_label = r"$P_{\mathrm{Holm}}=\mathrm{NA}$"
        ax.text(
            0.96,
            0.92,
            p_label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.2,
            color=BLUE,
        )
        ax.set_xlim(-duration, 0)
        ax.set_ylim(0, temporal_ymax)
        ax.set_xticks([-duration, -duration / 2.0, 0])
        ax.set_xticklabels(
            [f"{-duration:g}", f"{-duration / 2.0:g}", "0"]
        )
        ax.set_yticks([0, temporal_ymax / 2.0, temporal_ymax])
        if index % 4 == 0:
            ax.set_ylabel("State occupancy\nper time bin (%)")
        else:
            ax.tick_params(labelleft=False)
        if index >= 4:
            ax.set_xlabel("Time before window end (h)")
        style_axis(ax)

    add_panel_label(axes[0], "c", x=-0.30, y=1.23)
    return [
        Line2D([0], [0], color=BLUE, linewidth=1.2, label="24 h-aligned dominant cluster"),
        Line2D(
            [0],
            [0],
            color=ORANGE,
            linewidth=1.2,
            linestyle="--",
            label="24 h reference state",
        ),
    ]


def build_figure(
    metric_summary: pd.DataFrame,
    null_results: pd.DataFrame,
    pcs_events: pd.DataFrame,
    pcs_summary: pd.DataFrame,
    occupancy_summary: pd.DataFrame,
    terminal_change_summary: pd.DataFrame,
) -> plt.Figure:
    fig = plt.figure(
        figsize=(FIG_WIDTH_MM / 25.4, FIG_HEIGHT_MM / 25.4),
        facecolor="white",
    )
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.86, 1.42],
        hspace=0.48,
        left=0.085,
        right=0.985,
        top=0.935,
        bottom=0.085,
    )
    top = outer[0].subgridspec(1, 2, wspace=0.32)
    bottom = outer[1].subgridspec(2, 4, hspace=0.42, wspace=0.25)

    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    occupancy_axes: list[plt.Axes] = []
    first_axis: plt.Axes | None = None
    for row in range(2):
        for column in range(4):
            if first_axis is None:
                axis = fig.add_subplot(bottom[row, column])
                first_axis = axis
            else:
                axis = fig.add_subplot(bottom[row, column], sharey=first_axis)
            occupancy_axes.append(axis)

    plot_metric_panel(ax_a, metric_summary, null_results)
    plot_aligned_pcs_panel(ax_b, pcs_events, pcs_summary)
    occupancy_handles = plot_occupancy_small_multiples(
        occupancy_axes,
        occupancy_summary,
        terminal_change_summary,
    )
    fig.legend(
        handles=occupancy_handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.585),
        ncol=2,
        fontsize=5.7,
        handlelength=2.5,
        columnspacing=1.6,
    )
    return fig


# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------
def sanitize_export_table(df: pd.DataFrame) -> pd.DataFrame:
    """Replace absolute local paths with project-relative provenance strings."""
    result = df.copy()
    project_root_resolved = PROJECT_ROOT.resolve()
    for column in [item for item in result.columns if "path" in item.lower()]:
        def make_relative(value: object) -> object:
            if pd.isna(value) or str(value).strip() == "":
                return value
            path = Path(str(value))
            try:
                return path.resolve().relative_to(project_root_resolved).as_posix()
            except (OSError, ValueError):
                return path.name

        result[column] = result[column].map(make_relative)
    return result


def export_source_data(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    null_results: pd.DataFrame,
    metric_summary: pd.DataFrame,
    pcs_events: pd.DataFrame,
    pcs_summary: pd.DataFrame,
    occupancy: pd.DataFrame,
    occupancy_summary: pd.DataFrame,
    temporal_features: pd.DataFrame,
    terminal_change_summary: pd.DataFrame,
    fragmentation: pd.DataFrame,
    fragmentation_summary: pd.DataFrame,
    discovery_summary: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pd.DataFrame(
        {
            "item": [
                "reference state",
                "reference PCS table",
                "reference DEC labels",
                "aligned-cluster selection",
                "capture definition",
                "purity definition",
                "null model",
                "minimum candidate size",
                "panel a centre and interval",
                "panel b PCS definition",
                "panel b distribution",
                "temporal aggregation",
                "terminal-change test",
                "interpretation boundary",
            ],
            "definition": [
                (
                    "24 h PCS-selected state from the main event-specific DEC/PCS "
                    "analysis, restricted to the common time interval"
                ),
                (
                    "02clustering/result/DEC_result/precursor_curve_metrics/"
                    "precursor_by_metric/pcs_precursor_clusters.xlsx; sheet "
                    "selected_precursor"
                ),
                "02clustering/result/DEC_result/all_parement/{event}_all_parement.npz",
                (
                    "maximum 24 h-state capture; ties by purity, Jaccard, and "
                    "numeric cluster identifier; PCS is not used for alignment"
                ),
                "intersection / 24 h reference-state samples",
                "intersection / aligned short-window cluster samples",
                (
                    "non-zero circular shifts of the 24 h reference mask; maximum "
                    "capture is reselected after every shift"
                ),
                str(MIN_MATCHED_CLUSTER_SAMPLES),
                "event median and event-bootstrap 95% confidence interval",
                (
                    "event-wise PCS = Z(final concentration) + "
                    "Z(rate acceleration) - Z(c value) + "
                    "Z(terminal growth ratio)"
                ),
                (
                    "event points with median, interquartile range, and "
                    "1.5-IQR whiskers; PCS=0 is the event-wise standardized reference"
                ),
                (
                    f"{TIME_BINS} equal-duration bins; event-balanced mean with "
                    "event bootstrap 95% CI"
                ),
                (
                    "paired event-level last-20% minus first-20% occupancy; "
                    f"two-sided Monte Carlo sign-flip test ({SIGN_FLIP_ITERATIONS:,} "
                    "randomizations) with Holm correction across durations; "
                    "panel c labels report Holm-adjusted P values"
                ),
                "post-hoc state tracking, not independent precursor validation",
            ],
        }
    )
    selected_export = sanitize_export_table(selected)
    pcs_events_export = sanitize_export_table(pcs_events)
    candidates_export = sanitize_export_table(candidates)
    with pd.ExcelWriter(SOURCE_DATA_FILE, engine="openpyxl") as writer:
        metadata.to_excel(writer, sheet_name="metadata", index=False)
        discovery_summary.to_excel(writer, sheet_name="findings_summary", index=False)
        selected_export.to_excel(
            writer, sheet_name="aligned_cluster_events", index=False
        )
        null_results.to_excel(writer, sheet_name="panel_a_null_events", index=False)
        metric_summary.to_excel(writer, sheet_name="panel_a_metric_summary", index=False)
        pcs_events_export.to_excel(writer, sheet_name="panel_b_pcs_events", index=False)
        pcs_summary.to_excel(writer, sheet_name="panel_b_pcs_summary", index=False)
        occupancy.to_excel(writer, sheet_name="panel_c_occupancy_events", index=False)
        occupancy_summary.to_excel(
            writer, sheet_name="panel_c_occupancy_summary", index=False
        )
        temporal_features.to_excel(writer, sheet_name="temporal_features", index=False)
        terminal_change_summary.to_excel(
            writer, sheet_name="terminal_change_tests", index=False
        )
        fragmentation.to_excel(writer, sheet_name="fragmentation_events", index=False)
        fragmentation_summary.to_excel(
            writer, sheet_name="fragmentation_summary", index=False
        )
        candidates_export.to_excel(
            writer, sheet_name="all_candidate_overlaps", index=False
        )


def save_figure(fig: plt.Figure) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        OUTPUT_STEM.with_suffix(".svg"),
        OUTPUT_STEM.with_suffix(".pdf"),
        OUTPUT_STEM.with_suffix(".png"),
        OUTPUT_STEM.with_suffix(".tiff"),
    ]
    for path in paths:
        kwargs: dict[str, object] = {
            "bbox_inches": "tight",
            "facecolor": "white",
        }
        if path.suffix.lower() == ".png":
            kwargs["dpi"] = PNG_DPI
        elif path.suffix.lower() == ".tiff":
            kwargs["dpi"] = TIFF_DPI
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
    return paths


def main() -> None:
    candidates, selected, fragmentation = load_and_select_dominant_clusters()
    null_results, occupancy, temporal_features = analyse_event_states(selected)
    event_metrics = selected.merge(
        null_results,
        on=["event", "duration_hours"],
        how="left",
        validate="one_to_one",
    )
    metric_summary = build_metric_summary(event_metrics)
    pcs_events, pcs_summary = build_aligned_pcs_summary(selected)
    occupancy_summary = build_occupancy_summary(occupancy)
    fragmentation_summary = build_fragmentation_summary(fragmentation)
    terminal_change_summary = build_terminal_change_summary(temporal_features)
    discovery_summary = build_discovery_summary(
        event_metrics,
        null_results,
        pcs_summary,
        temporal_features,
        fragmentation_summary,
        terminal_change_summary,
    )

    figure = build_figure(
        metric_summary,
        null_results,
        pcs_events,
        pcs_summary,
        occupancy_summary,
        terminal_change_summary,
    )
    export_source_data(
        candidates,
        event_metrics,
        null_results,
        metric_summary,
        pcs_events,
        pcs_summary,
        occupancy,
        occupancy_summary,
        temporal_features,
        terminal_change_summary,
        fragmentation,
        fragmentation_summary,
        discovery_summary,
    )
    paths = save_figure(figure)
    plt.close(figure)

    print(f"Source data: {SOURCE_DATA_FILE}")
    print("Figures:")
    for path in paths:
        print(f"  {path}")
    print("\nExploratory findings:")
    print(discovery_summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
