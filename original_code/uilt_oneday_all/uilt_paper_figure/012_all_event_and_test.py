# -*- coding: utf-8 -*-
"""Combine all-event PCS evolution with the temporal-alignment null test.

Panels
------
a. Cumulative occurrence curves of the 35 PCS-selected states.
b. Hourly occurrence heatmap for the same event-state pairs.
c. Circular-shift null distribution of median final-hour enrichment.
d. Observed event-aligned occupancy curve and pointwise null envelope.

This script only reads existing PCS selections, DEC labels, and the outputs of
``011_PSC_test.py``. It does not recompute PCS or the circular-shift test.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEC_RESULT_DIR = PROJECT_ROOT / "uilt_oneday_all" / "02clustering" / "result" / "DEC_result"
PARAMETER_DIR = DEC_RESULT_DIR / "all_parement"
PCS_WORKBOOK = (
    DEC_RESULT_DIR
    / "precursor_curve_metrics"
    / "precursor_clusters_all_metrics.xlsx"
)
DEFAULT_NULL_DIR = SCRIPT_DIR / "Fig_PCS_time_shift_null"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "Fig_all_event_and_null_test"

EXPECTED_EVENTS = 44
EXPECTED_SELECTED = 35
N_HOURS = 24

BLUE = "#0F4D92"
BLUE_LIGHT = "#8CB9D9"
ORANGE = "#D97732"
RED = "#B64342"
GREY = "#CFCECE"
DARK_GREY = "#4D4D4D"

PUBLICATION_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 7.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--null-dir", type=Path, default=DEFAULT_NULL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def event_number(event_name: str) -> int:
    match = re.search(r"(\d+)$", str(event_name))
    return int(match.group(1)) if match else 10**9


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")


def load_selected_states() -> pd.DataFrame:
    """Load the authoritative 1 h-terminal-window PCS cohort."""
    require_file(PCS_WORKBOOK)
    selected = pd.read_excel(PCS_WORKBOOK, sheet_name="pcs")
    required = {"event", "cluster", "selection_status"}
    missing = required.difference(selected.columns)
    if missing:
        raise KeyError(f"Missing columns in PCS workbook: {sorted(missing)}")

    selected["event"] = selected["event"].astype(str)
    selected["cluster"] = pd.to_numeric(selected["cluster"], errors="coerce").fillna(-1).astype(int)
    selected = selected.loc[
        selected["selection_status"].eq("selected") & selected["cluster"].ge(0)
    ].copy()
    selected["event_number"] = selected["event"].map(event_number)
    selected = selected.sort_values("event_number").reset_index(drop=True)

    total_rows = pd.read_excel(PCS_WORKBOOK, sheet_name="pcs", usecols=["event"])
    if total_rows["event"].nunique() != EXPECTED_EVENTS:
        raise ValueError(
            f"Expected {EXPECTED_EVENTS} events in the PCS table, "
            f"found {total_rows['event'].nunique()}."
        )
    if len(selected) != EXPECTED_SELECTED:
        raise ValueError(
            f"Expected {EXPECTED_SELECTED} PCS-selected events, found {len(selected)}."
        )
    return selected


def load_event_state(event_name: str, cluster_id: int) -> Tuple[np.ndarray, np.ndarray]:
    path = PARAMETER_DIR / f"{event_name}_all_parement.npz"
    require_file(path)
    with np.load(path, allow_pickle=True) as archive:
        labels = np.asarray(archive["y_pred_last"], dtype=int)
    relative_hours = np.arange(len(labels), dtype=float) / 60.0 - 24.0
    return relative_hours, labels == int(cluster_id)


def build_all_event_data(
    selected: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Return minute cumulative curves and hourly counts for all selected states."""
    cumulative_rows: List[np.ndarray] = []
    hourly_rows: List[np.ndarray] = []
    event_labels: List[str] = []
    minute_axis = np.arange(1441, dtype=float) / 60.0 - 24.0
    hour_edges = np.arange(-24, 1, 1, dtype=float)

    for row in selected.itertuples(index=False):
        relative_hours, state_mask = load_event_state(row.event, int(row.cluster))
        valid = (relative_hours >= -24.0) & (relative_hours < 0.0)
        relative_hours = relative_hours[valid]
        state_mask = state_mask[valid]

        minute_index = np.floor((relative_hours + 24.0) * 60.0).astype(int)
        minute_index = np.clip(minute_index, 0, 1439)
        minute_counts = np.bincount(minute_index[state_mask], minlength=1440)
        cumulative_rows.append(np.r_[0.0, np.cumsum(minute_counts, dtype=float)])

        hourly_counts, _ = np.histogram(relative_hours[state_mask], bins=hour_edges)
        hourly_rows.append(hourly_counts.astype(float))
        event_labels.append(f"E{event_number(row.event)}")

    cumulative = np.stack(cumulative_rows)
    hourly = np.stack(hourly_rows)

    # Retain event-number order, matching the original all-event figure.
    return cumulative, hourly, event_labels, minute_axis


def load_null_outputs(null_dir: Path) -> Dict[str, np.ndarray]:
    global_path = null_dir / "pcs_time_shift_global_null.csv"
    hourly_path = null_dir / "pcs_time_shift_hourly_curve.csv"
    summary_path = null_dir / "pcs_time_shift_null_summary.csv"
    for path in (global_path, hourly_path, summary_path):
        require_file(path)

    global_null = pd.read_csv(global_path)
    hourly = pd.read_csv(hourly_path)
    summary = pd.read_csv(summary_path)
    enrichment_row = summary.loc[
        summary["test"].eq("Median terminal enrichment of PCS-selected states")
    ]
    if len(enrichment_row) != 1:
        raise ValueError("Could not identify the terminal-enrichment summary row.")
    enrichment_row = enrichment_row.iloc[0]

    return {
        "null_enrichment": global_null["median_terminal_enrichment"].to_numpy(float),
        "observed_enrichment": float(enrichment_row["observed"]),
        "null_enrichment_median": float(enrichment_row["null_median"]),
        "null_enrichment_low": float(enrichment_row["null_2.5%"]),
        "null_enrichment_high": float(enrichment_row["null_97.5%"]),
        "p_enrichment": float(enrichment_row["one_sided_monte_carlo_p"]),
        "time_hours": hourly["time_before_initiation_h"].to_numpy(float),
        "observed_occupancy": hourly["observed_mean_occupancy"].to_numpy(float),
        "null_occupancy_median": hourly["null_mean_occupancy_median"].to_numpy(float),
        "null_occupancy_low": hourly["null_mean_occupancy_2.5%"].to_numpy(float),
        "null_occupancy_high": hourly["null_mean_occupancy_97.5%"].to_numpy(float),
    }


def format_pvalue(value: float) -> str:
    return "<0.0001" if value < 0.0001 else f"={value:.4f}"


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.09, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_combined_figure(
    selected: pd.DataFrame,
    cumulative: np.ndarray,
    hourly: np.ndarray,
    heatmap_labels: List[str],
    minute_axis: np.ndarray,
    null_data: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    plt.rcParams.update(PUBLICATION_STYLE)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.2, 6.0))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.35],
        width_ratios=[1.12, 1.0],
        hspace=0.44,
        wspace=0.36,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[0, 1])
    ax_d = fig.add_subplot(grid[1, 1])

    # a. All-event cumulative curves.
    line_colors = plt.get_cmap("tab20", len(cumulative))
    for index, curve in enumerate(cumulative):
        ax_a.plot(
            minute_axis,
            curve,
            color=line_colors(index),
            linewidth=0.8,
            alpha=0.90,
        )
    ax_a.set_xlim(-24, 0)
    ax_a.set_xticks(np.arange(-24, 1, 6))
    ax_a.set_xlabel("Time before debris-flow initiation (h)")
    ax_a.set_ylabel("Cumulative count")
    ax_a.set_title("Cumulative occurrence of PCS-selected states", loc="left")
    ax_a.grid(color="0.88", linewidth=0.45, alpha=0.8)
    add_panel_label(ax_a, "a", x=-0.14)

    # b. Hourly all-event heatmap.
    image = ax_b.imshow(
        hourly,
        aspect="auto",
        interpolation="nearest",
        cmap="YlOrRd",
        extent=[-24, 0, len(heatmap_labels) - 0.5, -0.5],
        vmin=0,
    )
    ax_b.set_xlim(-24, 0)
    ax_b.set_xticks(np.arange(-24, 1, 6))
    ax_b.set_xlabel("Time before debris-flow initiation (h)")
    ax_b.set_ylabel("Event")
    y_ticks = np.arange(len(heatmap_labels))
    ax_b.set_yticks(y_ticks)
    ax_b.set_yticklabels([heatmap_labels[i] for i in y_ticks], fontsize=4.5)
    ax_b.set_title("Hourly occurrence of PCS-selected states", loc="left")
    colorbar = fig.colorbar(
        image,
        ax=ax_b,
        orientation="horizontal",
        pad=0.20,
        fraction=0.075,
        aspect=28,
    )
    colorbar.set_label("Hourly count", labelpad=2)
    add_panel_label(ax_b, "b", x=-0.14, y=1.06)

    # c. Terminal-enrichment null distribution (panel b of Fig. 011).
    enrichment = null_data["null_enrichment"]
    enrichment = enrichment[np.isfinite(enrichment)]
    ax_c.hist(
        enrichment,
        bins=35,
        density=True,
        color=BLUE_LIGHT,
        edgecolor="white",
        linewidth=0.45,
    )
    ax_c.axvspan(
        null_data["null_enrichment_low"],
        null_data["null_enrichment_high"],
        color=BLUE,
        alpha=0.07,
        linewidth=0,
    )
    ax_c.axvline(null_data["observed_enrichment"], color=RED, linewidth=2.0)
    ax_c.set_xlabel("Median final-hour enrichment")
    ax_c.set_ylabel("Probability density")
    ax_c.set_title("Terminal enrichment relative to shifted endpoints", loc="left")
    ax_c.text(
        0.03,
        0.96,
        f"Observed: {null_data['observed_enrichment']:.2f}x\n"
        f"Null median [95%]: {null_data['null_enrichment_median']:.2f} "
        f"[{null_data['null_enrichment_low']:.2f}, "
        f"{null_data['null_enrichment_high']:.2f}]\n"
        f"Monte Carlo p {format_pvalue(null_data['p_enrichment'])}",
        transform=ax_c.transAxes,
        va="top",
        ha="left",
        fontsize=7.1,
    )
    add_panel_label(ax_c, "c", x=-0.16)

    # d. Event-aligned occupancy curve (panel c of Fig. 011).
    hours = null_data["time_hours"]
    ax_d.fill_between(
        hours,
        null_data["null_occupancy_low"],
        null_data["null_occupancy_high"],
        color=GREY,
        alpha=0.60,
        linewidth=0,
        label="Null 95% interval",
    )
    ax_d.plot(
        hours,
        null_data["null_occupancy_median"],
        color=DARK_GREY,
        linestyle="--",
        linewidth=1.2,
        label="Null median",
    )
    ax_d.plot(
        hours,
        null_data["observed_occupancy"],
        color=BLUE,
        linewidth=1.9,
        marker="o",
        markersize=2.3,
        label="Observed alignment",
    )
    ax_d.axvspan(-1, 0, color=ORANGE, alpha=0.10, linewidth=0)
    ax_d.axvline(0, color="black", linewidth=0.75)
    ax_d.set_xlim(-24, 0)
    ax_d.set_xticks(np.arange(-24, 1, 6))
    ax_d.set_xlabel("Time before debris-flow initiation (h)")
    ax_d.set_ylabel("Mean state occupancy")
    ax_d.set_title("Temporal alignment after complete PCS selection", loc="left")
    handles, labels = ax_d.get_legend_handles_labels()
    order = [2, 1, 0]
    ax_d.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left")
    add_panel_label(ax_d, "d", x=-0.16)

    output_stem = output_dir / "Figure_all_event_and_terminal_enrichment_null"
    fig.savefig(str(output_stem) + ".png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(str(output_stem) + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(str(output_stem) + ".svg", bbox_inches="tight", facecolor="white")
    fig.savefig(
        str(output_stem) + ".tiff",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    # Figure source data for panels a and b; c and d already have source CSVs
    # exported by 011_PSC_test.py.
    curve_data = pd.DataFrame({"time_before_initiation_h": minute_axis})
    for row, curve in zip(selected.itertuples(index=False), cumulative):
        curve_data[f"{row.event}_C{int(row.cluster)}"] = curve
    curve_data.to_csv(output_dir / "panel_a_cumulative_curves.csv", index=False)

    heatmap_data = pd.DataFrame(hourly, index=heatmap_labels)
    heatmap_data.columns = [f"{hour}_to_{hour + 1}_h" for hour in range(-24, 0)]
    heatmap_data.index.name = "event"
    heatmap_data.to_csv(output_dir / "panel_b_hourly_heatmap.csv")


def main() -> None:
    args = parse_args()
    selected = load_selected_states()
    cumulative, hourly, heatmap_labels, minute_axis = build_all_event_data(selected)
    null_data = load_null_outputs(args.null_dir.resolve())
    plot_combined_figure(
        selected=selected,
        cumulative=cumulative,
        hourly=hourly,
        heatmap_labels=heatmap_labels,
        minute_axis=minute_axis,
        null_data=null_data,
        output_dir=args.output_dir.resolve(),
    )
    print(f"PCS-selected events: {len(selected)}/{EXPECTED_EVENTS}")
    print(f"Combined figure saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
