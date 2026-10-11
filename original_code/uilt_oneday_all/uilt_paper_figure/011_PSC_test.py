# -*- coding: utf-8 -*-
"""Temporal-alignment null test for the precursor-cluster score (PCS).

The DEC labels are kept unchanged and circularly shifted relative to the
catalogued debris-flow initiation time.  For every pseudo-endpoint, the script
recomputes all cluster metrics and reruns the complete PCS rule implemented in
``uilt_oneday_all/03visual/09_export_precursor_curve_metrics.py``:

    hard criteria:
        z_final > 2
        c_value < 1
        terminal_growth_flag == 1

    ranking:
        PCS = Z(z_final) + Z(rate_acceleration) - Z(c_value)
              + Z(terminal_growth_ratio)

The test therefore answers whether the observed event alignment produces
more PCS-selected events and stronger terminal enrichment than arbitrary
alignments of the same clustered records.  It does not test specificity
against non-event or non-triggering rainfall periods.

Main outputs
------------
``Figure_PCS_time_shift_null.(png|pdf|svg)``
    Four-panel publication figure.
``pcs_time_shift_null_results.xlsx``
    Summary, observed results, event-level null summaries, and global null.
``pcs_time_shift_null_*.csv``
    Plain-text source data for reproducibility.

Examples
--------
Publication run (default):
    python 011_PSC_test.py

Quick smoke test:
    python 011_PSC_test.py --n-shifts 8 --global-replicates 500 --workers 2

Exhaustive minute-resolution event-level shifts (slow):
    python 011_PSC_test.py --n-shifts 1439 --workers 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
# This is the script that generated the authoritative 35/44 workbook.  A later
# plotting copy (09_psc.py) uses a 2 h terminal window and therefore must not be
# used for inference against the published 1 h PCS cohort.
PCS_REFERENCE_SCRIPT = (
    PROJECT_ROOT
    / "uilt_oneday_all"
    / "03visual"
    / "09_export_precursor_curve_metrics.py"
)
DEC_LABEL_DIR = (
    PROJECT_ROOT
    / "uilt_oneday_all"
    / "02clustering"
    / "result"
    / "DEC_result"
    / "all_parement"
)
AUTHORITATIVE_PCS_WORKBOOK = (
    PROJECT_ROOT
    / "uilt_oneday_all"
    / "02clustering"
    / "result"
    / "DEC_result"
    / "precursor_curve_metrics"
    / "precursor_clusters_all_metrics.xlsx"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "Fig_PCS_time_shift_null"

DEFAULT_N_SHIFTS = 120
DEFAULT_GLOBAL_REPLICATES = 20000
DEFAULT_SEED = 20260812
DEFAULT_WORKERS = min(4, max((os.cpu_count() or 2) - 1, 1))
N_HOURLY_BINS = 24

BLUE = "#0F4D92"
BLUE_LIGHT = "#8CB9D9"
RED = "#B64342"
ORANGE = "#D97732"
GREY = "#CFCECE"
DARK_GREY = "#4D4D4D"
LIGHT_GREY = "#E9ECEF"

PUBLICATION_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 1.0,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

_PCS_MODULE = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-shifts",
        type=int,
        default=DEFAULT_N_SHIFTS,
        help=(
            "Pseudo-endpoints per event. Values >= n_samples-1 use every "
            "non-zero minute shift (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--global-replicates",
        type=int,
        default=DEFAULT_GLOBAL_REPLICATES,
        help="Monte Carlo draws combining one pseudo-endpoint per event (default: %(default)s).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for figure and source-data outputs.",
    )
    parser.add_argument(
        "--skip-authoritative-check",
        action="store_true",
        help="Do not compare recomputed observed selections with the existing PCS workbook.",
    )
    return parser.parse_args()


def event_number(event_name: str) -> int:
    match = re.search(r"(\d+)$", str(event_name))
    return int(match.group(1)) if match else 10**9


def load_pcs_module():
    """Load the manuscript's authoritative PCS implementation once per process."""
    global _PCS_MODULE
    if _PCS_MODULE is not None:
        return _PCS_MODULE
    if not PCS_REFERENCE_SCRIPT.exists():
        raise FileNotFoundError(f"PCS reference script not found: {PCS_REFERENCE_SCRIPT}")
    spec = importlib.util.spec_from_file_location("pcs_reference_09", PCS_REFERENCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import PCS reference script: {PCS_REFERENCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _PCS_MODULE = module
    return module


def stratified_circular_shifts(n_samples: int, n_shifts: int, seed: int) -> np.ndarray:
    """Sample non-zero circular shifts with near-uniform coverage of the 24 h cycle."""
    if n_samples < 2:
        raise ValueError("At least two labels are required for a circular-shift test.")
    possible = n_samples - 1
    if n_shifts <= 0:
        raise ValueError("n_shifts must be positive.")
    if n_shifts >= possible:
        return np.arange(1, n_samples, dtype=int)

    rng = np.random.default_rng(seed)
    edges = np.linspace(1, n_samples, n_shifts + 1)
    shifts = []
    for left, right in zip(edges[:-1], edges[1:]):
        low = int(math.ceil(left))
        high = int(math.ceil(right))
        high = max(high, low + 1)
        shifts.append(int(rng.integers(low, min(high, n_samples))))
    shifts = np.unique(np.asarray(shifts, dtype=int))

    # Rare rounding collisions are filled from the unsampled offsets.
    if len(shifts) < n_shifts:
        remaining = np.setdiff1d(np.arange(1, n_samples, dtype=int), shifts)
        fill = rng.choice(remaining, size=n_shifts - len(shifts), replace=False)
        shifts = np.concatenate([shifts, fill])
    return np.sort(shifts)


def hourly_occupancy(mask: np.ndarray, n_bins: int = N_HOURLY_BINS) -> np.ndarray:
    """Return equal-duration bin occupancy; for 1440 samples these are 1 h bins."""
    return np.asarray([float(part.mean()) for part in np.array_split(mask, n_bins)])


def selected_state_result(
    labels: np.ndarray,
    event_name: str,
    pcs_module,
) -> Dict[str, Any]:
    """Recompute all PCS inputs, rerun selection, and summarize the selected state."""
    n_samples = len(labels)
    time_points = np.arange(n_samples, dtype=float) * float(pcs_module.DT_SECONDS)
    metrics = pcs_module.compute_cluster_metrics(
        time_points=time_points,
        time_labels=labels,
        unique_labels=np.unique(labels),
        dt=pcs_module.DT_SECONDS,
        bg_ratio=pcs_module.BACKGROUND_RATIO,
    )
    metrics.insert(0, "event", event_name)
    # Reproduce select_precursor_clusters_by_pcs exactly, while avoiding a
    # formatting-only bug in the reference function when no cluster passes
    # the hard rule (its output column ``pcs_rank_within_event`` is absent).
    valid = metrics.dropna(
        subset=["z_final", "rate_acceleration", "c_value", "terminal_growth_ratio"]
    ).copy()
    if valid.empty:
        row = pd.Series(dtype=float)
        is_selected = False
    else:
        valid["z_z_final"] = pcs_module.within_event_zscore(valid["z_final"])
        valid["z_rate_acceleration"] = pcs_module.within_event_zscore(
            valid["rate_acceleration"]
        )
        valid["z_c_value"] = pcs_module.within_event_zscore(valid["c_value"])
        valid["z_terminal_growth_ratio"] = pcs_module.within_event_zscore(
            valid["terminal_growth_ratio"]
        )
        valid["pcs"] = (
            valid["z_z_final"]
            + valid["z_rate_acceleration"]
            - valid["z_c_value"]
            + valid["z_terminal_growth_ratio"]
        )
        candidates = valid[
            (valid["z_final"] > pcs_module.Z_FINAL_THRESHOLD)
            & (valid["c_value"] < pcs_module.C_VALUE_THRESHOLD)
            & (valid["terminal_growth_flag"] == 1)
        ].copy()
        is_selected = not candidates.empty
        if is_selected:
            candidates = candidates.sort_values(
                by=["pcs", "z_final", "terminal_growth_ratio", "sample_count"],
                ascending=[False, False, False, False],
            )
            row = candidates.iloc[0]
        else:
            row = pd.Series(dtype=float)

    result = {
        "selected": is_selected,
        "cluster": -1,
        "pcs": np.nan,
        "z_final": np.nan,
        "c_value": np.nan,
        "terminal_growth_ratio": np.nan,
        "terminal_enrichment": np.nan,
        "hourly_occupancy": np.full(N_HOURLY_BINS, np.nan, dtype=float),
    }
    if not is_selected:
        return result

    cluster_id = int(row["cluster"])
    mask = np.asarray(labels == cluster_id, dtype=float)
    terminal_points = max(
        int(round(pcs_module.TERMINAL_WINDOW_SECONDS / pcs_module.DT_SECONDS)), 1
    )
    overall_rate = float(mask.mean())
    terminal_rate = float(mask[-terminal_points:].mean())
    terminal_enrichment = terminal_rate / overall_rate if overall_rate > 0 else np.nan

    result.update(
        {
            "cluster": cluster_id,
            "pcs": float(row["pcs"]),
            "z_final": float(row["z_final"]),
            "c_value": float(row["c_value"]),
            "terminal_growth_ratio": float(row["terminal_growth_ratio"]),
            "terminal_enrichment": float(terminal_enrichment),
            "hourly_occupancy": hourly_occupancy(mask),
        }
    )
    return result


def analyze_event_worker(task: Tuple[str, str, int, int]) -> Dict[str, Any]:
    """Worker: observed endpoint plus a stratified set of circular shifts for one event."""
    event_name, npz_path_string, n_shifts, seed = task
    pcs_module = load_pcs_module()
    with np.load(npz_path_string, allow_pickle=True) as archive:
        labels = np.asarray(archive["y_pred_last"], dtype=int)

    observed = selected_state_result(labels, event_name, pcs_module)
    shifts = stratified_circular_shifts(len(labels), n_shifts, seed)
    n_null = len(shifts)

    selected = np.zeros(n_null, dtype=bool)
    clusters = np.full(n_null, -1, dtype=int)
    pcs_values = np.full(n_null, np.nan, dtype=float)
    terminal_enrichment = np.full(n_null, np.nan, dtype=float)
    hourly = np.full((n_null, N_HOURLY_BINS), np.nan, dtype=float)

    for index, shift in enumerate(shifts):
        shifted_labels = np.roll(labels, int(shift))
        result = selected_state_result(shifted_labels, event_name, pcs_module)
        selected[index] = result["selected"]
        clusters[index] = result["cluster"]
        pcs_values[index] = result["pcs"]
        terminal_enrichment[index] = result["terminal_enrichment"]
        hourly[index] = result["hourly_occupancy"]

    return {
        "event": event_name,
        "n_samples": len(labels),
        "observed": observed,
        "shifts": shifts,
        "selected": selected,
        "clusters": clusters,
        "pcs": pcs_values,
        "terminal_enrichment": terminal_enrichment,
        "hourly": hourly,
    }


def discover_event_files() -> List[Tuple[str, Path]]:
    files = []
    for path in DEC_LABEL_DIR.glob("event*_all_parement.npz"):
        event_name = path.name.split("_")[0]
        files.append((event_name, path))
    files.sort(key=lambda item: event_number(item[0]))
    if not files:
        raise FileNotFoundError(f"No event label files found in {DEC_LABEL_DIR}")
    return files


def run_event_analyses(
    event_files: List[Tuple[str, Path]],
    n_shifts: int,
    seed: int,
    workers: int,
) -> List[Dict[str, Any]]:
    tasks = [
        (event_name, str(path), n_shifts, seed + event_number(event_name) * 10007)
        for event_name, path in event_files
    ]
    results = []
    workers = max(int(workers), 1)
    if workers == 1:
        for index, task in enumerate(tasks, 1):
            result = analyze_event_worker(task)
            results.append(result)
            print(f"[{index:02d}/{len(tasks):02d}] {result['event']} complete", flush=True)
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(analyze_event_worker, task): task[0] for task in tasks}
                for index, future in enumerate(as_completed(futures), 1):
                    result = future.result()
                    results.append(result)
                    print(
                        f"[{index:02d}/{len(tasks):02d}] {result['event']} complete",
                        flush=True,
                    )
        except PermissionError:
            # Sandboxed Windows sessions can disallow multiprocessing pipes.
            print(
                "[WARN] Multiprocessing is not permitted; restarting in serial mode.",
                flush=True,
            )
            results = []
            for index, task in enumerate(tasks, 1):
                result = analyze_event_worker(task)
                results.append(result)
                print(f"[{index:02d}/{len(tasks):02d}] {result['event']} complete", flush=True)
    return sorted(results, key=lambda item: event_number(item["event"]))


def validate_observed_against_workbook(results: List[Dict[str, Any]]) -> None:
    """Confirm that the null-test implementation reproduces the manuscript cohort."""
    if not AUTHORITATIVE_PCS_WORKBOOK.exists():
        print(f"[WARN] Authoritative PCS workbook not found: {AUTHORITATIVE_PCS_WORKBOOK}")
        return
    reference = pd.read_excel(AUTHORITATIVE_PCS_WORKBOOK, sheet_name="pcs")
    reference["cluster"] = pd.to_numeric(reference["cluster"], errors="coerce").fillna(-1).astype(int)
    reference_map = dict(zip(reference["event"].astype(str), reference["cluster"]))
    mismatches = []
    for result in results:
        observed_cluster = int(result["observed"]["cluster"])
        expected_cluster = int(reference_map.get(result["event"], -999))
        if observed_cluster != expected_cluster:
            mismatches.append((result["event"], expected_cluster, observed_cluster))
    if mismatches:
        raise RuntimeError(
            "Recomputed observed PCS selections do not match the authoritative workbook: "
            + ", ".join(f"{e}: expected {a}, got {b}" for e, a, b in mismatches)
        )


def assemble_event_arrays(results: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    n_shift_values = {len(result["shifts"]) for result in results}
    if len(n_shift_values) != 1:
        raise ValueError(
            "Events have different numbers of sampled shifts. Use n_shifts below every "
            "event's n_samples-1 so global resampling has a rectangular array."
        )
    return {
        "selected": np.stack([result["selected"] for result in results]),
        "terminal_enrichment": np.stack(
            [result["terminal_enrichment"] for result in results]
        ),
        "hourly": np.stack([result["hourly"] for result in results]),
    }


def global_null_resampling(
    results: List[Dict[str, Any]],
    n_replicates: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Draw one pseudo-endpoint per event and form study-level null distributions."""
    arrays = assemble_event_arrays(results)
    selected = arrays["selected"]
    enrichment = arrays["terminal_enrichment"]
    hourly = arrays["hourly"]
    n_events, n_shifts = selected.shape
    rng = np.random.default_rng(seed + 7919)

    counts = np.empty(n_replicates, dtype=int)
    median_enrichment = np.full(n_replicates, np.nan, dtype=float)
    mean_hourly = np.full((n_replicates, N_HOURLY_BINS), np.nan, dtype=float)
    event_indices = np.arange(n_events)[None, :]

    # Chunking controls temporary memory while retaining every replicate curve.
    chunk_size = 1000
    for start in range(0, n_replicates, chunk_size):
        stop = min(start + chunk_size, n_replicates)
        indices = rng.integers(0, n_shifts, size=(stop - start, n_events))
        chosen_selected = selected[event_indices, indices]
        chosen_enrichment = enrichment[event_indices, indices]
        chosen_hourly = hourly[event_indices, indices]

        counts[start:stop] = chosen_selected.sum(axis=1)
        with np.errstate(all="ignore"):
            median_enrichment[start:stop] = np.nanmedian(chosen_enrichment, axis=1)
            mean_hourly[start:stop] = np.nanmean(chosen_hourly, axis=1)

    return {
        "selected_count": counts,
        "median_terminal_enrichment": median_enrichment,
        "mean_hourly_occupancy": mean_hourly,
    }


def observed_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    selected_results = [result["observed"] for result in results if result["observed"]["selected"]]
    enrichments = np.asarray(
        [result["terminal_enrichment"] for result in selected_results], dtype=float
    )
    hourly = np.stack([result["hourly_occupancy"] for result in selected_results])
    return {
        "selected_count": len(selected_results),
        "selection_rate": len(selected_results) / len(results),
        "median_terminal_enrichment": float(np.nanmedian(enrichments)),
        "mean_hourly_occupancy": np.nanmean(hourly, axis=0),
    }


def monte_carlo_pvalue(null_values: np.ndarray, observed_value: float) -> float:
    valid = np.asarray(null_values, dtype=float)
    valid = valid[np.isfinite(valid)]
    return float((1 + np.sum(valid >= observed_value)) / (len(valid) + 1))


def build_output_tables(
    results: List[Dict[str, Any]],
    null: Dict[str, np.ndarray],
    observed: Dict[str, Any],
    n_requested_shifts: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p_count = monte_carlo_pvalue(null["selected_count"], observed["selected_count"])
    p_enrichment = monte_carlo_pvalue(
        null["median_terminal_enrichment"], observed["median_terminal_enrichment"]
    )
    count_ci = np.percentile(null["selected_count"], [2.5, 50, 97.5])
    enrich_ci = np.nanpercentile(null["median_terminal_enrichment"], [2.5, 50, 97.5])

    summary = pd.DataFrame(
        [
            {
                "test": "PCS-selected event count",
                "observed": observed["selected_count"],
                "null_median": count_ci[1],
                "null_2.5%": count_ci[0],
                "null_97.5%": count_ci[2],
                "one_sided_monte_carlo_p": p_count,
            },
            {
                "test": "Median terminal enrichment of PCS-selected states",
                "observed": observed["median_terminal_enrichment"],
                "null_median": enrich_ci[1],
                "null_2.5%": enrich_ci[0],
                "null_97.5%": enrich_ci[2],
                "one_sided_monte_carlo_p": p_enrichment,
            },
        ]
    )
    summary["n_events"] = len(results)
    summary["requested_shifts_per_event"] = n_requested_shifts
    summary["global_null_replicates"] = len(null["selected_count"])
    summary["random_seed"] = seed

    observed_rows = []
    event_null_rows = []
    shift_rows = []
    for result in results:
        obs = result["observed"]
        observed_rows.append(
            {
                "event": result["event"],
                "selected": obs["selected"],
                "cluster": obs["cluster"],
                "pcs": obs["pcs"],
                "z_final": obs["z_final"],
                "c_value": obs["c_value"],
                "terminal_growth_ratio": obs["terminal_growth_ratio"],
                "terminal_enrichment": obs["terminal_enrichment"],
            }
        )
        null_enrichment = result["terminal_enrichment"]
        event_null_rows.append(
            {
                "event": result["event"],
                "observed_selected": obs["selected"],
                "observed_cluster": obs["cluster"],
                "null_selection_probability": float(np.mean(result["selected"])),
                "n_shifts": len(result["shifts"]),
                "null_terminal_enrichment_median": (
                    float(np.nanmedian(null_enrichment))
                    if np.isfinite(null_enrichment).any()
                    else np.nan
                ),
                "null_terminal_enrichment_97.5%": (
                    float(np.nanpercentile(null_enrichment, 97.5))
                    if np.isfinite(null_enrichment).any()
                    else np.nan
                ),
            }
        )
        for i, shift in enumerate(result["shifts"]):
            row = {
                "event": result["event"],
                "shift_samples": int(shift),
                "shift_hours": float(shift / 60.0),
                "selected": bool(result["selected"][i]),
                "selected_cluster": int(result["clusters"][i]),
                "pcs": float(result["pcs"][i]),
                "terminal_enrichment": float(result["terminal_enrichment"][i]),
            }
            for hour_index, value in enumerate(result["hourly"][i]):
                row[f"occupancy_hour_{hour_index - 24}_to_{hour_index - 23}"] = float(value)
            shift_rows.append(row)

    observed_df = pd.DataFrame(observed_rows)
    event_null_df = pd.DataFrame(event_null_rows)
    shift_df = pd.DataFrame(shift_rows)
    global_null_df = pd.DataFrame(
        {
            "replicate": np.arange(1, len(null["selected_count"]) + 1),
            "selected_event_count": null["selected_count"],
            "median_terminal_enrichment": null["median_terminal_enrichment"],
        }
    )
    return summary, observed_df, event_null_df, shift_df, global_null_df


def save_source_data(
    output_dir: Path,
    tables: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    results: List[Dict[str, Any]],
    null: Dict[str, np.ndarray],
    observed: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, observed_df, event_null_df, shift_df, global_null_df = tables

    summary.to_csv(output_dir / "pcs_time_shift_null_summary.csv", index=False)
    observed_df.to_csv(output_dir / "pcs_time_shift_observed_events.csv", index=False)
    event_null_df.to_csv(output_dir / "pcs_time_shift_event_null_summary.csv", index=False)
    shift_df.to_csv(output_dir / "pcs_time_shift_all_pseudo_endpoints.csv", index=False)
    global_null_df.to_csv(output_dir / "pcs_time_shift_global_null.csv", index=False)

    hourly_df = pd.DataFrame(
        {
            "time_before_initiation_h": np.arange(-23.5, 0.5, 1.0),
            "observed_mean_occupancy": observed["mean_hourly_occupancy"],
            "null_mean_occupancy_median": np.nanmedian(null["mean_hourly_occupancy"], axis=0),
            "null_mean_occupancy_2.5%": np.nanpercentile(
                null["mean_hourly_occupancy"], 2.5, axis=0
            ),
            "null_mean_occupancy_97.5%": np.nanpercentile(
                null["mean_hourly_occupancy"], 97.5, axis=0
            ),
        }
    )
    hourly_df.to_csv(output_dir / "pcs_time_shift_hourly_curve.csv", index=False)

    workbook = output_dir / "pcs_time_shift_null_results.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        observed_df.to_excel(writer, sheet_name="observed_events", index=False)
        event_null_df.to_excel(writer, sheet_name="event_null_summary", index=False)
        global_null_df.to_excel(writer, sheet_name="global_null", index=False)
        hourly_df.to_excel(writer, sheet_name="hourly_curve", index=False)
        shift_df.to_excel(writer, sheet_name="pseudo_endpoints", index=False)

    metadata = {
        "description": "Circular-shift temporal-alignment null for the complete PCS pipeline.",
        "pcs_reference_script": str(PCS_REFERENCE_SCRIPT),
        "label_directory": str(DEC_LABEL_DIR),
        "n_events": len(results),
        "n_shifts_requested_per_event": args.n_shifts,
        "n_global_replicates": args.global_replicates,
        "random_seed": args.seed,
        "workers": args.workers,
        "terminal_window_seconds": int(load_pcs_module().TERMINAL_WINDOW_SECONDS),
        "interpretation_boundary": (
            "Tests alignment to the catalogued initiation time; does not estimate "
            "specificity or false alarms in non-event periods."
        ),
    }
    (output_dir / "pcs_time_shift_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=11)


def format_pvalue(p_value: float) -> str:
    """Avoid displaying a finite Monte Carlo probability as p = 0.0000."""
    if p_value < 0.0001:
        return "< 0.0001"
    return f"= {p_value:.4f}"


def plot_publication_figure(
    output_dir: Path,
    results: List[Dict[str, Any]],
    null: Dict[str, np.ndarray],
    observed: Dict[str, Any],
) -> None:
    plt.rcParams.update(PUBLICATION_RCPARAMS)
    p_count = monte_carlo_pvalue(null["selected_count"], observed["selected_count"])
    p_enrichment = monte_carlo_pvalue(
        null["median_terminal_enrichment"], observed["median_terminal_enrichment"]
    )

    fig = plt.figure(figsize=(7.2, 6.5))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    # a. Study-level count null.
    counts = null["selected_count"]
    bins = np.arange(counts.min() - 0.5, counts.max() + 1.5, 1)
    ax_a.hist(counts, bins=bins, density=True, color=GREY, edgecolor="white", linewidth=0.5)
    ax_a.axvline(observed["selected_count"], color=RED, linewidth=2.0)
    count_ci = np.percentile(counts, [2.5, 50, 97.5])
    ax_a.axvspan(count_ci[0], count_ci[2], color=DARK_GREY, alpha=0.08, linewidth=0)
    ax_a.set_xlabel("PCS-selected events per null replicate")
    ax_a.set_ylabel("Probability density")
    ax_a.set_title("Event-level PCS selection", loc="left", fontsize=10)
    ax_a.text(
        0.03,
        0.96,
        f"Observed: {observed['selected_count']}/{len(results)}\n"
        f"Null median [95%]: {count_ci[1]:.0f} [{count_ci[0]:.0f}, {count_ci[2]:.0f}]\n"
        f"Monte Carlo p {format_pvalue(p_count)}",
        transform=ax_a.transAxes,
        va="top",
        ha="left",
        fontsize=8,
    )
    panel_label(ax_a, "a")

    # b. Study-level terminal-enrichment null.
    enrich = null["median_terminal_enrichment"]
    enrich = enrich[np.isfinite(enrich)]
    ax_b.hist(enrich, bins=35, density=True, color=BLUE_LIGHT, edgecolor="white", linewidth=0.5)
    ax_b.axvline(observed["median_terminal_enrichment"], color=RED, linewidth=2.0)
    enrich_ci = np.percentile(enrich, [2.5, 50, 97.5])
    ax_b.axvspan(enrich_ci[0], enrich_ci[2], color=BLUE, alpha=0.07, linewidth=0)
    ax_b.set_xlabel("Median terminal enrichment")
    ax_b.set_ylabel("Probability density")
    ax_b.set_title("Terminal concentration of selected states", loc="left", fontsize=10)
    ax_b.text(
        0.03,
        0.96,
        f"Observed: {observed['median_terminal_enrichment']:.2f}x\n"
        f"Null median [95%]: {enrich_ci[1]:.2f} "
        f"[{enrich_ci[0]:.2f}, {enrich_ci[2]:.2f}]\n"
        f"Monte Carlo p {format_pvalue(p_enrichment)}",
        transform=ax_b.transAxes,
        va="top",
        ha="left",
        fontsize=8,
    )
    panel_label(ax_b, "b")

    # c. Event-aligned occupancy curve and global-null envelope.
    hours = np.arange(-23.5, 0.5, 1.0)
    null_curve = null["mean_hourly_occupancy"]
    low = np.nanpercentile(null_curve, 2.5, axis=0)
    median = np.nanmedian(null_curve, axis=0)
    high = np.nanpercentile(null_curve, 97.5, axis=0)
    ax_c.fill_between(hours, low, high, color=GREY, alpha=0.55, linewidth=0, label="Null 95% interval")
    ax_c.plot(hours, median, color=DARK_GREY, linewidth=1.2, linestyle="--", label="Null median")
    ax_c.plot(
        hours,
        observed["mean_hourly_occupancy"],
        color=BLUE,
        linewidth=2.0,
        marker="o",
        markersize=2.5,
        label="Observed alignment",
    )
    ax_c.axvspan(-2, 0, color=ORANGE, alpha=0.10, linewidth=0)
    ax_c.axvline(0, color="black", linewidth=0.8)
    ax_c.set_xlim(-24, 0)
    ax_c.set_xticks(np.arange(-24, 1, 6))
    ax_c.set_xlabel("Time before debris-flow initiation (h)")
    ax_c.set_ylabel("Mean state occupancy")
    ax_c.set_title("Temporal evolution after complete PCS selection", loc="left", fontsize=10)
    ax_c.legend(loc="upper left", fontsize=7)
    panel_label(ax_c, "c")

    # d. Event-wise probability that an arbitrary alignment passes PCS.
    names = [result["event"] for result in results]
    probabilities = np.asarray([np.mean(result["selected"]) for result in results])
    observed_selected = np.asarray([result["observed"]["selected"] for result in results])
    x = np.arange(len(names))
    colors = np.where(observed_selected, ORANGE, GREY)
    ax_d.bar(x, probabilities, color=colors, width=0.82, edgecolor="none")
    ax_d.axhline(probabilities.mean(), color=DARK_GREY, linestyle="--", linewidth=1.0)
    ax_d.set_ylim(0, 1)
    tick_indices = np.arange(0, len(names), 5)
    ax_d.set_xticks(tick_indices)
    ax_d.set_xticklabels([str(event_number(names[i])) for i in tick_indices])
    ax_d.set_xlabel("Event number")
    ax_d.set_ylabel("Null PCS-selection probability")
    ax_d.set_title("Sensitivity to endpoint placement", loc="left", fontsize=10)
    ax_d.text(
        0.97,
        0.96,
        "Orange: selected at true endpoint",
        transform=ax_d.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color=ORANGE,
    )
    panel_label(ax_d, "d")

    fig.suptitle("PCS temporal-alignment null test", fontsize=12, fontweight="bold", y=0.995)
    output_stem = output_dir / "Figure_PCS_time_shift_null"
    fig.savefig(str(output_stem) + ".png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(str(output_stem) + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(str(output_stem) + ".svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    event_files = discover_event_files()
    print(
        f"Running complete PCS circular-shift test for {len(event_files)} events; "
        f"requested shifts/event={args.n_shifts}, workers={args.workers}",
        flush=True,
    )
    results = run_event_analyses(
        event_files=event_files,
        n_shifts=args.n_shifts,
        seed=args.seed,
        workers=args.workers,
    )
    if not args.skip_authoritative_check:
        validate_observed_against_workbook(results)

    observed = observed_summary(results)
    null = global_null_resampling(results, args.global_replicates, args.seed)
    tables = build_output_tables(results, null, observed, args.n_shifts, args.seed)
    save_source_data(args.output_dir, tables, results, null, observed, args)
    plot_publication_figure(args.output_dir, results, null, observed)

    summary = tables[0]
    print("\n" + summary.to_string(index=False), flush=True)
    print(f"\nOutputs saved to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
