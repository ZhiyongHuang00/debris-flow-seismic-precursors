# -*- coding: utf-8 -*-
"""Publication figure: rainfall context of PCS-selected seismic states.

The figure uses events as the statistical unit and addresses two questions:

1. Did the 35 events with a PCS-selected state occur under different rainfall
   conditions from the nine events without such a state?
2. Did Pattern I and Pattern II occur under different rainfall conditions?

The forest plots evaluate cumulative rainfall over 12 nested windows from 2 to
24 h at 2 h increments. Event distributions for the 6 h window are shown
separately as a representative near-initiation scale with comparatively large
group contrasts.

Cumulative 7 d rainfall is retained only as an exploratory antecedent-wetness
analysis in the source-data workbook; it is not displayed in the main figure.

For every comparison, the plot shows all event-level observations, a boxplot,
Cliff's delta with a bootstrap 95% confidence interval, and a two-sided
permutation P value corrected across the 12 rainfall windows using Holm's
method. Positive Cliff's delta means that the first displayed group is higher.

The authoritative 35-event PCS cohort is read from the same event-specific
24 h DEC/PCS table used by the main analysis. Pattern labels are accepted only
when their event set matches that cohort exactly. Metric-specific complete-case
analyses exclude rainfall windows containing missing 30 min records; an
all-numeric-record sensitivity analysis is exported separately.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Paths and analysis settings
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
BASE = HERE.parent

RAIN_WORKBOOK = (
    BASE
    / "03visual"
    / "rain_coupling"
    / "gpm_rainfall_analysis"
    / "GPM_IMERG_Illgraben_rainfall_analysis.xlsx"
)
RAIN_CONTINUOUS_CSV = (
    BASE
    / "03visual"
    / "rain_coupling"
    / "gpm_rainfall_analysis"
    / "GPM_IMERG_Illgraben_30min_continuous_roll.csv"
)
MODE_WORKBOOK = (
    BASE
    / "03visual"
    / "compared_different_duration"
    / "mode_rainfall_relationship"
    / "mode_rainfall_relationship.xlsx"
)
PCS_WORKBOOK = (
    BASE
    / "02clustering"
    / "result"
    / "DEC_result"
    / "precursor_curve_metrics"
    / "precursor_by_metric"
    / "pcs_precursor_clusters.xlsx"
)
PCS_SHEET = "selected_precursor"

OUT_DIR = HERE / "Fig7_rainfall_context"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_BASE = OUT_DIR / "figure7_rainfall_context_estimation"

RANDOM_SEED = 20260803
N_PERMUTATIONS = 20_000
N_BOOTSTRAPS = 10_000

FOREST_HOURS = list(range(2, 25, 2))
METRICS = [
    {
        "column": f"event_cum_{hours}h_mm",
        "title": f"Cumulative {hours} h rainfall",
        "ylabel": "Rainfall (mm)",
        "timescale": f"last {hours} h",
        "missing_column": f"event_missing_{hours}h_count",
        "hours": hours,
    }
    for hours in FOREST_HOURS
]

ANTECEDENT_METRICS = [
    {
        "column": "event_cum_7d_mm",
        "title": "Cumulative 7 d rainfall",
        "ylabel": "Rainfall (mm)",
        "timescale": "last 7 d",
        "missing_column": "event_missing_168h_count",
    }
]
ALL_METRICS = METRICS + ANTECEDENT_METRICS

PCS_GROUPS = ["PCS-selected", "No PCS state"]
PCS_COLORS = ["#D97732", "#9AA0A6"]
PATTERN_GROUPS = ["Pattern I", "Pattern II"]
PATTERN_COLORS = ["#2F80B7", "#E76F51"]


# -----------------------------------------------------------------------------
# Journal-style defaults
# -----------------------------------------------------------------------------
MM_TO_INCH = 1 / 25.4
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------
def normalize_event(value: object) -> str:
    """Return a stable lower-case event label such as ``event12``."""
    match = re.search(r"(\d+)", str(value))
    if not match:
        raise ValueError(f"Cannot parse event identifier: {value!r}")
    return f"event{int(match.group(1))}"


def event_number(value: object) -> int:
    return int(re.search(r"(\d+)", str(value)).group(1))


def require_columns(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns in {source.name}: {missing}")


def add_even_hour_rainfall_windows(rainfall: pd.DataFrame) -> pd.DataFrame:
    """Calculate complete 2--24 h rainfall windows from the 30 min record."""
    require_columns(
        rainfall,
        ["rain_record_time_utc"],
        RAIN_WORKBOOK,
    )
    continuous = pd.read_csv(
        RAIN_CONTINUOUS_CSV,
        usecols=["time_utc", "rain_mm_30min", "is_missing_record"],
    )
    continuous["time_utc"] = pd.to_datetime(continuous["time_utc"], utc=True)
    continuous = continuous.sort_values("time_utc").reset_index(drop=True)
    if continuous["time_utc"].duplicated().any():
        raise ValueError("Duplicate timestamps in the continuous rainfall record.")
    time_steps = continuous["time_utc"].diff().dropna()
    if not time_steps.eq(pd.Timedelta(minutes=30)).all():
        raise ValueError(
            "The continuous rainfall table is not a complete 30 min time grid."
        )

    rainfall = rainfall.copy()
    rainfall["rain_record_time_utc"] = pd.to_datetime(
        rainfall["rain_record_time_utc"], utc=True
    )
    record_times = rainfall["rain_record_time_utc"]
    rainfall_values = continuous["rain_mm_30min"].fillna(0.0).astype(float)
    missing_values = (
        continuous["is_missing_record"].fillna(True).astype(bool)
        | continuous["rain_mm_30min"].isna()
    ).astype(int)

    computed = pd.DataFrame({"time_utc": continuous["time_utc"]})
    for hours in FOREST_HOURS:
        records = hours * 2
        value_column = f"event_cum_{hours}h_mm"
        missing_column = f"event_missing_{hours}h_count"
        computed[value_column] = rainfall_values.rolling(
            records,
            min_periods=records,
        ).sum()
        computed[missing_column] = missing_values.rolling(
            records,
            min_periods=records,
        ).sum()

    computed = computed.set_index("time_utc")
    missing_matches = sorted(set(record_times).difference(computed.index))
    if missing_matches:
        raise ValueError(
            "Event rainfall timestamps are absent from the continuous record: "
            f"{missing_matches}"
        )
    aligned = computed.loc[record_times].reset_index(drop=True)
    for hours in FOREST_HOURS:
        value_column = f"event_cum_{hours}h_mm"
        missing_column = f"event_missing_{hours}h_count"
        rainfall[value_column] = aligned[value_column].to_numpy(float)
        rainfall[missing_column] = aligned[missing_column].to_numpy(float)

    # Confirm that the new rolling implementation reproduces every rainfall
    # window already available in the event summary table.
    reported_value_columns = {
        2: "event_cum_2h_mm",
        4: "event_cum_4h_mm",
        6: "event_cum_6h_mm",
        8: "event_cum_8h_mm",
        12: "event_cum_12h_mm",
        24: "event_cum_1d_mm",
    }
    for hours, reported_column in reported_value_columns.items():
        original_column = f"_reported_{reported_column}"
        if original_column not in rainfall.columns:
            continue
        calculated_column = f"event_cum_{hours}h_mm"
        if not np.allclose(
            rainfall[original_column],
            rainfall[calculated_column],
            equal_nan=True,
        ):
            raise ValueError(
                f"Recalculated {hours} h rainfall does not match {reported_column}."
            )
    return rainfall


def load_event_table() -> pd.DataFrame:
    """Build the 44-event cohort from the authoritative main-paper PCS table."""
    rainfall = pd.read_excel(RAIN_WORKBOOK, sheet_name="event_rainfall")
    modes = pd.read_excel(MODE_WORKBOOK, sheet_name="mode_rainfall_event_table")
    pcs = pd.read_excel(PCS_WORKBOOK, sheet_name=PCS_SHEET)

    reported_windows = {
        2: "event_cum_2h_mm",
        4: "event_cum_4h_mm",
        6: "event_cum_6h_mm",
        8: "event_cum_8h_mm",
        12: "event_cum_12h_mm",
        24: "event_cum_1d_mm",
    }
    required_rain = [
        "event",
        "time_offset_min",
        "rain_record_time_utc",
        *reported_windows.values(),
        "event_cum_7d_mm",
        "event_missing_168h_count",
    ]
    require_columns(rainfall, required_rain, RAIN_WORKBOOK)
    require_columns(modes, ["event", "mode_short_name"], MODE_WORKBOOK)
    require_columns(pcs, ["event", "cluster", "selection_status"], PCS_WORKBOOK)

    rainfall = rainfall[required_rain].copy()
    for reported_column in reported_windows.values():
        rainfall[f"_reported_{reported_column}"] = rainfall[reported_column]
    rainfall = add_even_hour_rainfall_windows(rainfall)
    modes = modes[["event", "mode_short_name"]].copy()
    rainfall["event"] = rainfall["event"].map(normalize_event)
    modes["event"] = modes["event"].map(normalize_event)
    pcs["event"] = pcs["event"].map(normalize_event)
    pcs["cluster"] = pd.to_numeric(pcs["cluster"], errors="coerce")
    pcs_selected = pcs[
        pcs["selection_status"].astype(str).str.strip().str.lower().eq("selected")
        & pcs["cluster"].ge(0)
    ][["event", "cluster"]].copy()

    if rainfall["event"].duplicated().any():
        duplicated = rainfall.loc[rainfall["event"].duplicated(), "event"].tolist()
        raise ValueError(f"Duplicated events in rainfall table: {duplicated}")
    if modes["event"].duplicated().any():
        duplicated = modes.loc[modes["event"].duplicated(), "event"].tolist()
        raise ValueError(f"Duplicated events in mode table: {duplicated}")
    if pcs_selected["event"].duplicated().any():
        duplicated = pcs_selected.loc[
            pcs_selected["event"].duplicated(), "event"
        ].tolist()
        raise ValueError(f"Duplicated events in main PCS table: {duplicated}")
    if len(rainfall) != 44:
        raise ValueError(f"Expected 44 rainfall events, found {len(rainfall)}")
    if len(pcs_selected) != 35:
        raise ValueError(
            f"Expected 35 main-paper PCS-selected events, found {len(pcs_selected)}"
        )

    pcs_events = set(pcs_selected["event"])
    mode_events = set(modes["event"])
    if mode_events != pcs_events:
        only_mode = sorted(mode_events - pcs_events, key=event_number)
        only_pcs = sorted(pcs_events - mode_events, key=event_number)
        raise ValueError(
            "Pattern labels are not aligned with the authoritative main-paper "
            f"PCS cohort. Only in pattern table: {only_mode}; only in PCS table: "
            f"{only_pcs}. Regenerate the pattern table before plotting."
        )
    invalid_modes = sorted(
        set(modes["mode_short_name"].dropna()).difference(PATTERN_GROUPS)
    )
    if invalid_modes:
        raise ValueError(f"Unexpected pattern labels: {invalid_modes}")

    data = rainfall.merge(modes, on="event", how="left", validate="one_to_one")
    data["pcs_group"] = np.where(
        data["event"].isin(pcs_events), "PCS-selected", "No PCS state"
    )
    data["main_24h_pcs_cluster"] = data["event"].map(
        pcs_selected.set_index("event")["cluster"].astype(int)
    )
    if data["time_offset_min"].abs().max() > 30:
        raise ValueError(
            "At least one debris-flow time is more than 30 min from the matched "
            "GPM record. Review event-to-rainfall alignment."
        )
    data = data.sort_values("event", key=lambda x: x.map(event_number)).reset_index(drop=True)

    print(
        "Rainfall cohort: "
        f"total={len(data)}, PCS-selected={len(pcs_events)}, "
        f"no PCS state={(data['pcs_group'] == 'No PCS state').sum()}"
    )
    print("Pattern counts:", modes["mode_short_name"].value_counts().to_dict())
    return data


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta; positive values mean the first group is larger."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.sign(x[:, None] - y[None, :]).mean())


def bootstrap_cliffs_delta(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_bootstraps: int = N_BOOTSTRAPS,
) -> tuple[float, float, float]:
    observed = cliffs_delta(x, y)
    estimates = np.empty(n_bootstraps, dtype=float)
    for index in range(n_bootstraps):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        estimates[index] = cliffs_delta(xb, yb)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return observed, float(low), float(high)


def permutation_test_cliffs_delta(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int = N_PERMUTATIONS,
) -> tuple[float, float]:
    """Two-sided label-permutation test for Cliff's delta."""
    observed = cliffs_delta(x, y)
    pooled = np.concatenate([x, y])
    n_x = len(x)
    exceedances = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(pooled)
        statistic = cliffs_delta(permuted[:n_x], permuted[n_x:])
        exceedances += abs(statistic) >= abs(observed)
    p_value = (exceedances + 1) / (n_permutations + 1)
    return observed, float(p_value)


def holm_adjust(p_values: list[float]) -> np.ndarray:
    """Holm family-wise error correction without external dependencies."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running_max = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        candidate = min((m - rank) * p[index], 1.0)
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def comparison_statistics(
    data: pd.DataFrame,
    group_column: str,
    groups: list[str],
    comparison_name: str,
    seed_offset: int,
    complete_case: bool,
    metrics: list[dict[str, object]] = METRICS,
) -> pd.DataFrame:
    """Calculate event-level effect sizes and corrected P values."""
    records: list[dict[str, object]] = []
    raw_p_values: list[float] = []

    for metric_index, metric in enumerate(metrics):
        column = metric["column"]
        valid = data[column].notna()
        if complete_case:
            valid &= data[metric["missing_column"]].fillna(np.inf).eq(0)
        x = data.loc[valid & data[group_column].eq(groups[0]), column].to_numpy(float)
        y = data.loc[valid & data[group_column].eq(groups[1]), column].to_numpy(float)
        if len(x) == 0 or len(y) == 0:
            raise ValueError(f"No data for {comparison_name}, metric={column}")

        rng_boot = np.random.default_rng(RANDOM_SEED + seed_offset + metric_index)
        rng_perm = np.random.default_rng(RANDOM_SEED + seed_offset + 100 + metric_index)
        delta, delta_low, delta_high = bootstrap_cliffs_delta(x, y, rng_boot)
        _, p_raw = permutation_test_cliffs_delta(x, y, rng_perm)
        median_difference = float(np.median(x) - np.median(y))
        raw_p_values.append(p_raw)
        records.append(
            {
                "comparison": comparison_name,
                "analysis_set": (
                    "complete rainfall windows" if complete_case else "all numeric records"
                ),
                "metric": column,
                "timescale": metric["timescale"],
                "group_1": groups[0],
                "group_2": groups[1],
                "n_1": len(x),
                "n_2": len(y),
                "median_1_mm": float(np.median(x)),
                "median_2_mm": float(np.median(y)),
                "median_difference_mm": median_difference,
                "cliffs_delta": delta,
                "cliffs_delta_ci_low": delta_low,
                "cliffs_delta_ci_high": delta_high,
                "p_permutation_cliffs_delta_raw": p_raw,
                "missing_window_records_excluded": bool(complete_case),
            }
        )

    adjusted = holm_adjust(raw_p_values)
    for record, p_adjusted in zip(records, adjusted):
        record["p_holm"] = float(p_adjusted)
    return pd.DataFrame.from_records(records)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def format_p(p_value: float) -> str:
    if p_value < 0.001:
        return "< 0.001"
    return f"= {p_value:.3f}"


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.20,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def draw_event_boxplot(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: dict[str, object],
    group_column: str,
    groups: list[str],
    colors: list[str],
    stats_row: pd.Series,
    panel_label: str,
    jitter_seed: int,
) -> None:
    valid = data[metric["column"]].notna() & data[
        metric["missing_column"]
    ].fillna(np.inf).eq(0)
    arrays = [
        data.loc[
            valid & data[group_column].eq(group), metric["column"]
        ].to_numpy(float)
        for group in groups
    ]

    box = ax.boxplot(
        arrays,
        positions=[0, 1],
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops={"color": "#222222", "linewidth": 1.15},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
        boxprops={"color": "#555555", "linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.24)
        patch.set_edgecolor(color)

    rng = np.random.default_rng(jitter_seed)
    for position, (values, color) in enumerate(zip(arrays, colors)):
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=16,
            color=color,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )

    labels = [f"{group}\n(n={len(values)})" for group, values in zip(groups, arrays)]
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel(metric["ylabel"])
    ax.set_title(metric["title"], loc="left", pad=4, fontweight="bold")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.55, 1.55)
    data_maximum = max(float(np.nanmax(values)) for values in arrays)
    ax.set_ylim(0, max(data_maximum * 1.32, 1.0))

    annotation = (
        f"$\\delta$ = {stats_row['cliffs_delta']:.2f} "
        f"(95% CI {stats_row['cliffs_delta_ci_low']:.2f}, "
        f"{stats_row['cliffs_delta_ci_high']:.2f})\n"
        f"$P_{{Holm}}$ {format_p(float(stats_row['p_holm']))}"
    )
    ax.text(
        0.98,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color="#303030",
    )
    add_panel_label(ax, panel_label)


def draw_effect_forest(
    ax: plt.Axes,
    statistics: pd.DataFrame,
    color: str,
    title: str,
    panel_label: str,
) -> None:
    """Show effect sizes for cumulative rainfall at 2 h increments."""
    rows = (
        statistics.set_index("metric")
        .loc[[metric["column"] for metric in METRICS]]
        .reset_index()
    )
    y_positions = np.arange(len(rows), dtype=float)
    ax.axvspan(-0.1, 0.1, color="#F2F2F2", zorder=0)
    ax.axvline(0, color="#777777", linestyle="--", linewidth=0.8, zorder=1)
    ax.plot(
        rows["cliffs_delta"],
        y_positions,
        color="#B8B8B8",
        linewidth=0.75,
        zorder=1,
    )

    for y_position, row in zip(y_positions, rows.itertuples(index=False)):
        highlight = row.metric == "event_cum_6h_mm"
        point_color = color if highlight else "#777777"
        line_width = 1.35 if highlight else 0.95
        marker_size = 5.4 if highlight else 4.4
        ax.errorbar(
            row.cliffs_delta,
            y_position,
            xerr=np.asarray(
                [[row.cliffs_delta - row.cliffs_delta_ci_low],
                 [row.cliffs_delta_ci_high - row.cliffs_delta]]
            ),
            fmt="o",
            color=point_color,
            ecolor=point_color,
            elinewidth=line_width,
            capsize=2.8,
            markersize=marker_size,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
        )
        if float(row.p_holm) < 0.05:
            ax.text(
                row.cliffs_delta,
                y_position - 0.28,
                "*",
                ha="center",
                va="bottom",
                fontsize=7.0,
                color=point_color,
            )

    ax.set_yticks(
        y_positions,
        [f"{hours} h" for hours in FOREST_HOURS],
    )
    ax.invert_yaxis()
    ax.set_xlim(-0.72, 1.02)
    ax.set_xticks([-0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel("Cliff's $\\delta$ (first group higher →)")
    ax.set_title(title, loc="left", pad=5, fontweight="bold")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    add_panel_label(ax, panel_label)


def make_figure(data: pd.DataFrame, statistics: pd.DataFrame) -> plt.Figure:
    """Combine the strongest raw contrast with a non-cherry-picked effect summary."""
    width = 183 * MM_TO_INCH
    height = 170 * MM_TO_INCH
    fig = plt.figure(figsize=(width, height), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.82, 1.40],
        left=0.085,
        right=0.985,
        bottom=0.15,
        top=0.93,
        wspace=0.34,
        hspace=0.36,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]

    pcs_stats = statistics.loc[statistics["comparison"].eq("PCS presence")]
    pattern_stats = statistics.loc[
        statistics["comparison"].eq("Pattern I vs Pattern II")
    ]
    six_hour_metric = next(
        metric for metric in METRICS if metric["column"] == "event_cum_6h_mm"
    )
    pcs_six_hour = pcs_stats.loc[
        pcs_stats["metric"].eq("event_cum_6h_mm")
    ].iloc[0]
    pattern_six_hour = pattern_stats.loc[
        pattern_stats["metric"].eq("event_cum_6h_mm")
    ].iloc[0]

    draw_event_boxplot(
        axes[0],
        data,
        six_hour_metric,
        "pcs_group",
        PCS_GROUPS,
        PCS_COLORS,
        pcs_six_hour,
        panel_label="a",
        jitter_seed=RANDOM_SEED,
    )
    draw_event_boxplot(
        axes[1],
        data.loc[data["mode_short_name"].notna()].copy(),
        six_hour_metric,
        "mode_short_name",
        PATTERN_GROUPS,
        PATTERN_COLORS,
        pattern_six_hour,
        panel_label="b",
        jitter_seed=RANDOM_SEED + 10,
    )
    draw_effect_forest(
        axes[2],
        pcs_stats,
        PCS_COLORS[0],
        "PCS-selected versus no PCS state",
        panel_label="c",
    )
    draw_effect_forest(
        axes[3],
        pattern_stats,
        PATTERN_COLORS[0],
        "Pattern I versus Pattern II",
        panel_label="d",
    )

    fig.text(
        0.535,
        0.025,
        (
            "Points represent debris-flow events; boxes show median and IQR; "
            "forest plots show Cliff's δ with bootstrap 95% CI for cumulative "
            "rainfall at 2 h increments. Complete rainfall windows only; P values "
            "are Holm-adjusted across 12 windows."
        ),
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="#555555",
    )
    return fig


def save_outputs(
    fig: plt.Figure,
    data: pd.DataFrame,
    statistics: pd.DataFrame,
    sensitivity_statistics: pd.DataFrame,
    antecedent_statistics: pd.DataFrame,
) -> None:
    source_columns = [
        "event",
        "pcs_group",
        "mode_short_name",
        "main_24h_pcs_cluster",
        "time_offset_min",
        *[metric["column"] for metric in ALL_METRICS],
        *sorted({metric["missing_column"] for metric in ALL_METRICS}),
    ]
    source_data = data[source_columns].copy()
    source_data.to_csv(
        OUT_DIR / "figure7_rainfall_context_source_data.csv", index=False
    )
    statistics.to_csv(OUT_DIR / "figure7_rainfall_context_statistics.csv", index=False)
    sensitivity_statistics.to_csv(
        OUT_DIR / "figure7_rainfall_context_sensitivity_all_records.csv",
        index=False,
    )
    antecedent_statistics.to_csv(
        OUT_DIR / "figure7_rainfall_context_exploratory_7d.csv",
        index=False,
    )
    metadata = pd.DataFrame(
        {
            "item": [
                "statistical unit",
                "authoritative PCS cohort",
                "pattern cohort",
                "primary completeness rule",
                "effect size",
                "confidence interval",
                "hypothesis test",
                "multiple testing",
                "exploratory antecedent context",
                "interpretation boundary",
            ],
            "definition": [
                "one debris-flow event",
                (
                    "35 events selected in the main event-specific 24 h DEC/PCS "
                    "analysis"
                ),
                "Pattern I/II labels validated against the main PCS cohort",
                (
                    "exclude an event for a metric when its corresponding rainfall "
                    "window contains one or more missing 30 min records"
                ),
                "Cliff's delta; positive values indicate the first displayed group is higher",
                f"event bootstrap percentile 95% CI ({N_BOOTSTRAPS:,} resamples)",
                (
                    "two-sided label-permutation test of Cliff's delta "
                    f"({N_PERMUTATIONS:,} permutations)"
                ),
                "Holm correction across 12 cumulative-rainfall windows (2--24 h at 2 h increments) per comparison",
                "7 d cumulative rainfall is exported separately and is not shown in the main figure",
                "rainfall association/context, not a deterministic triggering mechanism",
            ],
        }
    )
    with pd.ExcelWriter(
        OUT_DIR / "figure7_rainfall_context_source_data.xlsx",
        engine="openpyxl",
    ) as writer:
        metadata.to_excel(writer, sheet_name="metadata", index=False)
        source_data.to_excel(writer, sheet_name="event_data", index=False)
        statistics.to_excel(writer, sheet_name="figure_statistics", index=False)
        sensitivity_statistics.to_excel(
            writer,
            sheet_name="sensitivity_all_records",
            index=False,
        )
        antecedent_statistics.to_excel(
            writer,
            sheet_name="exploratory_7d_context",
            index=False,
        )

    fig.savefig(OUT_BASE.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_BASE.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_BASE.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        OUT_BASE.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def main() -> None:
    data = load_event_table()

    pcs_statistics = comparison_statistics(
        data=data,
        group_column="pcs_group",
        groups=PCS_GROUPS,
        comparison_name="PCS presence",
        seed_offset=0,
        complete_case=True,
    )
    pattern_statistics = comparison_statistics(
        data=data.loc[data["mode_short_name"].notna()].copy(),
        group_column="mode_short_name",
        groups=PATTERN_GROUPS,
        comparison_name="Pattern I vs Pattern II",
        seed_offset=1_000,
        complete_case=True,
    )
    statistics = pd.concat([pcs_statistics, pattern_statistics], ignore_index=True)

    pcs_sensitivity = comparison_statistics(
        data=data,
        group_column="pcs_group",
        groups=PCS_GROUPS,
        comparison_name="PCS presence",
        seed_offset=2_000,
        complete_case=False,
    )
    pattern_sensitivity = comparison_statistics(
        data=data.loc[data["mode_short_name"].notna()].copy(),
        group_column="mode_short_name",
        groups=PATTERN_GROUPS,
        comparison_name="Pattern I vs Pattern II",
        seed_offset=3_000,
        complete_case=False,
    )
    sensitivity_statistics = pd.concat(
        [pcs_sensitivity, pattern_sensitivity], ignore_index=True
    )

    pcs_antecedent = comparison_statistics(
        data=data,
        group_column="pcs_group",
        groups=PCS_GROUPS,
        comparison_name="PCS presence",
        seed_offset=4_000,
        complete_case=True,
        metrics=ANTECEDENT_METRICS,
    )
    pattern_antecedent = comparison_statistics(
        data=data.loc[data["mode_short_name"].notna()].copy(),
        group_column="mode_short_name",
        groups=PATTERN_GROUPS,
        comparison_name="Pattern I vs Pattern II",
        seed_offset=5_000,
        complete_case=True,
        metrics=ANTECEDENT_METRICS,
    )
    antecedent_statistics = pd.concat(
        [pcs_antecedent, pattern_antecedent], ignore_index=True
    )
    figure=make_figure(data,statistics)
    for ax in figure.axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.75)
        ax.tick_params(axis="both",which="both",direction="out",bottom=True,left=True,top=False,right=False,labeltop=False,labelright=False)
        ax.tick_params(axis="both",which="major",length=3,width=0.65)

    save_outputs(figure,data,statistics,sensitivity_statistics,antecedent_statistics)
    save_root=Path(r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben\uilt_oneday_all\uilt_paper_figure\Fig_all")
    (save_root/"png").mkdir(parents=True,exist_ok=True)
    (save_root/"tiff").mkdir(parents=True,exist_ok=True)
    png_path=save_root/"png"/"Fig 8.png"
    tiff_path=save_root/"tiff"/"Fig 8.tiff"
    figure.savefig(png_path,format="png",dpi=600,bbox_inches="tight",facecolor="white")
    figure.savefig(tiff_path,format="tiff",dpi=600,bbox_inches="tight",facecolor="white",pil_kwargs={"compression":"tiff_lzw"})
    print(png_path)
    print(tiff_path)
    # plt.show()
    # plt.close(figure)
    # figure = make_figure(data, statistics)
    # save_outputs(
    #     figure,
    #     data,
    #     statistics,
    #     sensitivity_statistics,
    #     antecedent_statistics,
    # )
    plt.show()
    plt.close(figure)

    print("\nEvent-level rainfall statistics:")
    print(
        statistics[
            [
                "comparison",
                "metric",
                "n_1",
                "n_2",
                "median_1_mm",
                "median_2_mm",
                "cliffs_delta",
                "cliffs_delta_ci_low",
                "cliffs_delta_ci_high",
                "p_holm",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print(f"\nSaved figure and source data to: {OUT_DIR}")


if __name__ == "__main__":
    main()
