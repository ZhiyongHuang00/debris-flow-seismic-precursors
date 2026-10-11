# -*- coding: utf-8 -*-
"""
Analyze rainfall controls for different precursor modes.

The precursor modes are read from the secondary clustering result used by the
paper figure workflow. Rainfall metrics are read from the GPM IMERG event
rainfall analysis workbook. The script exports an event-level table, mode-level
statistics, non-parametric tests, and several figures for manuscript discussion.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu, spearmanr
except ImportError:  # pragma: no cover - allows plotting even if scipy is absent
    mannwhitneyu = None
    spearmanr = None


PROJECT_ROOT = Path(r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben")
SCRIPT_DIR = Path(__file__).resolve().parent

RAIN_ANALYSIS_XLSX = (
    PROJECT_ROOT
    / "uilt_oneday_all"
    / "03visual"
    / "rain_coupling"
    / "gpm_rainfall_analysis"
    / "GPM_IMERG_Illgraben_rainfall_analysis.xlsx"
)

OUTPUT_DIR = SCRIPT_DIR / "mode_rainfall_relationship"
FIGURE_DIR = OUTPUT_DIR / "figures"
PREFERRED_MODE_RESULT_DIR_NAME = "kmeans_clustering_from_matched_precursor_features"

MODE_NAME_MAP = {
    0: "Pattern I: high-frequency / terminal",
    1: "Pattern II: lower-frequency / broader",
}
MODE_SHORT_NAME_MAP = {
    0: "Pattern I",
    1: "Pattern II",
}
MODE_COLOR_MAP = {
    0: "#2F80B7",  # consistent with 06_discussion_two_pattern_mechanism.py
    1: "#E76F51",
}

RAIN_CUMULATIVE_COLS = [
    "event_cum_1h_mm",
    "event_cum_6h_mm",
    "event_cum_1d_mm",
    "event_cum_3d_mm",
    "event_cum_7d_mm",
]

RAIN_INTENSITY_COLS = [
    "rain_30min_at_event_mm",
    "rain_rate_at_event_mm_h",
    "max_30min_mm_last_24h",
    "max_1h_mm_last_24h",
    "max_3h_mm_last_24h",
    "max_6h_mm_last_24h",
]

DERIVED_RAIN_COLS = [
    "short_term_fraction_1d_over_7d",
    "antecedent_fraction_pre1d_in_7d",
    "rain_3d_minus_1d_mm",
    "rain_7d_minus_1d_mm",
]

FEATURE_COLS_FOR_CORRELATION = [
    "RappMaxMean",
    "FmaxFFT",
    "FCentroid",
    "RMS",
    "Relative_Energy",
    "duration",
    "npz_sample_count",
    "matched_parameter_count",
    "PC1",
    "PC2",
]


def natural_event_number(value: object) -> int:
    """Return the numeric part of an event label for natural event ordering."""
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else 10**9


def normalize_event(value: object) -> str:
    """Normalize event ids to the form event1, event2, ..."""
    number = natural_event_number(value)
    if number == 10**9:
        return str(value).strip().lower()
    return f"event{number}"


def find_mode_result_file() -> Path:
    """Find the PCS-matched secondary clustering workbook."""
    candidates = list(PROJECT_ROOT.rglob("kmeans_clustering_all_results.xlsx"))
    preferred = [
        p
        for p in candidates
        if PREFERRED_MODE_RESULT_DIR_NAME in str(p).replace("\\", "/").lower()
    ]
    if preferred:
        return max(preferred, key=lambda p: p.stat().st_mtime)
    if candidates:
        raise FileNotFoundError(
            "Could not find the PCS-matched secondary clustering workbook "
            f"under a directory named '{PREFERRED_MODE_RESULT_DIR_NAME}'. "
            f"Found other workbooks: {[str(p) for p in candidates]}"
        )

    csv_candidates = [
        p
        for p in PROJECT_ROOT.rglob("kmeans_clustering_results.csv")
        if PREFERRED_MODE_RESULT_DIR_NAME in str(p).replace("\\", "/").lower()
    ]
    if csv_candidates:
        return max(csv_candidates, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(
        "Could not find PCS-matched secondary clustering result file: "
        f"{PREFERRED_MODE_RESULT_DIR_NAME}/kmeans_clustering_all_results.xlsx"
    )


def load_mode_results(path: Path | None = None) -> pd.DataFrame:
    """Load event-level precursor mode assignment from secondary clustering."""
    mode_path = path or find_mode_result_file()
    if mode_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(mode_path, sheet_name="kmeans_results")
    else:
        df = pd.read_csv(mode_path)

    required = {"event", "kmeans_cluster"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Mode result file is missing required columns: {missing}")

    df = df.copy()
    df["event"] = df["event"].map(normalize_event)
    df["kmeans_cluster"] = pd.to_numeric(df["kmeans_cluster"], errors="coerce").astype("Int64")
    df["mode_short_name"] = df["kmeans_cluster"].map(MODE_SHORT_NAME_MAP)
    df["mode_name"] = df["kmeans_cluster"].map(MODE_NAME_MAP)
    df["mode_color"] = df["kmeans_cluster"].map(MODE_COLOR_MAP)
    df["mode_result_file"] = str(mode_path)
    return df


def load_event_rainfall(path: Path = RAIN_ANALYSIS_XLSX) -> pd.DataFrame:
    """Load event rainfall metrics from GPM IMERG rainfall analysis workbook."""
    if not path.exists():
        raise FileNotFoundError(f"Rainfall analysis workbook not found: {path}")

    df = pd.read_excel(path, sheet_name="event_rainfall")
    if "event" not in df.columns:
        raise ValueError("Rainfall workbook sheet 'event_rainfall' lacks an event column.")

    df = df.copy()
    df["event"] = df["event"].map(normalize_event)
    for col in RAIN_CUMULATIVE_COLS + RAIN_INTENSITY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_merged_table(mode_df: pd.DataFrame, rain_df: pd.DataFrame) -> pd.DataFrame:
    """Merge mode assignment and rainfall metrics at event level."""
    mode_cols = [
        "event",
        "cluster_id",
        "raw_precursor_label",
        "kmeans_cluster",
        "kmeans_cluster_name",
        "mode_short_name",
        "mode_name",
        "mode_color",
        "hierarchical_cluster",
        "npz_sample_count",
        "matched_parameter_count",
        "PC1",
        "PC2",
        "TSNE1",
        "TSNE2",
        "mode_result_file",
    ]
    mode_cols += [c for c in FEATURE_COLS_FOR_CORRELATION if c in mode_df.columns]
    mode_cols = list(dict.fromkeys([c for c in mode_cols if c in mode_df.columns]))

    rain_cols = [
        "event",
        "event_time_utc",
        "rain_record_time_utc",
        "time_offset_min",
    ]
    rain_cols += [c for c in RAIN_CUMULATIVE_COLS + RAIN_INTENSITY_COLS if c in rain_df.columns]
    rain_cols = list(dict.fromkeys([c for c in rain_cols if c in rain_df.columns]))

    merged = mode_df[mode_cols].merge(rain_df[rain_cols], on="event", how="left", validate="one_to_one")

    if {"event_cum_1d_mm", "event_cum_7d_mm"}.issubset(merged.columns):
        denominator = merged["event_cum_7d_mm"].replace(0, np.nan)
        merged["short_term_fraction_1d_over_7d"] = merged["event_cum_1d_mm"] / denominator
        merged["antecedent_fraction_pre1d_in_7d"] = (
            merged["event_cum_7d_mm"] - merged["event_cum_1d_mm"]
        ) / denominator
        merged["rain_7d_minus_1d_mm"] = merged["event_cum_7d_mm"] - merged["event_cum_1d_mm"]

    if {"event_cum_3d_mm", "event_cum_1d_mm"}.issubset(merged.columns):
        merged["rain_3d_minus_1d_mm"] = merged["event_cum_3d_mm"] - merged["event_cum_1d_mm"]

    merged["_event_order"] = merged["event"].map(natural_event_number)
    merged = merged.sort_values(["kmeans_cluster", "_event_order"]).drop(columns="_event_order")
    return merged


def summarize_by_mode(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute robust rainfall statistics for each precursor mode."""
    metrics = [c for c in RAIN_CUMULATIVE_COLS + RAIN_INTENSITY_COLS + DERIVED_RAIN_COLS if c in merged.columns]
    rows = []
    for mode_id, group in merged.groupby("kmeans_cluster", dropna=False):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    "kmeans_cluster": mode_id,
                    "mode_short_name": MODE_SHORT_NAME_MAP.get(int(mode_id), str(mode_id))
                    if pd.notna(mode_id)
                    else "missing",
                    "mode_name": MODE_NAME_MAP.get(int(mode_id), str(mode_id))
                    if pd.notna(mode_id)
                    else "missing",
                    "metric": metric,
                    "n": int(values.shape[0]),
                    "mean": values.mean(),
                    "median": values.median(),
                    "q25": values.quantile(0.25),
                    "q75": values.quantile(0.75),
                    "std": values.std(ddof=1),
                }
            )
    return pd.DataFrame(rows)


def compare_modes(merged: pd.DataFrame) -> pd.DataFrame:
    """Compare two precursor modes using Mann-Whitney U tests."""
    metrics = [c for c in RAIN_CUMULATIVE_COLS + RAIN_INTENSITY_COLS + DERIVED_RAIN_COLS if c in merged.columns]
    modes = sorted([int(m) for m in merged["kmeans_cluster"].dropna().unique()])
    if len(modes) != 2 or mannwhitneyu is None:
        return pd.DataFrame(
            columns=[
                "metric",
                "mode_a",
                "mode_b",
                "n_a",
                "n_b",
                "median_a",
                "median_b",
                "median_difference_a_minus_b",
                "u_statistic",
                "p_value",
            ]
        )

    rows = []
    mode_a, mode_b = modes
    for metric in metrics:
        a = pd.to_numeric(merged.loc[merged["kmeans_cluster"] == mode_a, metric], errors="coerce").dropna()
        b = pd.to_numeric(merged.loc[merged["kmeans_cluster"] == mode_b, metric], errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            p_value = np.nan
            u_stat = np.nan
        else:
            test = mannwhitneyu(a, b, alternative="two-sided")
            u_stat = test.statistic
            p_value = test.pvalue
        rows.append(
            {
                "metric": metric,
                "mode_a": MODE_SHORT_NAME_MAP.get(mode_a, str(mode_a)),
                "mode_b": MODE_SHORT_NAME_MAP.get(mode_b, str(mode_b)),
                "n_a": len(a),
                "n_b": len(b),
                "median_a": a.median(),
                "median_b": b.median(),
                "median_difference_a_minus_b": a.median() - b.median(),
                "u_statistic": u_stat,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def mode_rainfall_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    """Correlate selected precursor features with rainfall metrics."""
    if spearmanr is None:
        return pd.DataFrame(columns=["feature", "rain_metric", "n", "spearman_r", "p_value"])

    feature_cols = [c for c in FEATURE_COLS_FOR_CORRELATION if c in merged.columns]
    rain_cols = [c for c in RAIN_CUMULATIVE_COLS + RAIN_INTENSITY_COLS + DERIVED_RAIN_COLS if c in merged.columns]
    rows = []
    for feature in feature_cols:
        for rain_col in rain_cols:
            pair = merged[[feature, rain_col]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 5 or pair[feature].nunique() < 2 or pair[rain_col].nunique() < 2:
                rho = np.nan
                p_value = np.nan
            else:
                rho, p_value = spearmanr(pair[feature], pair[rain_col])
            rows.append(
                {
                    "feature": feature,
                    "rain_metric": rain_col,
                    "n": len(pair),
                    "spearman_r": rho,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def make_excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove timezone information because Excel cannot store tz-aware datetimes."""
    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[col]):
            safe[col] = pd.to_datetime(safe[col], errors="coerce").dt.tz_localize(None)
        elif safe[col].dtype == object and "time" in col.lower():
            safe[col] = safe[col].astype(str)
    return safe


def setup_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.linewidth": 1.0,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def plot_rainfall_boxplots_by_mode(merged: pd.DataFrame) -> Path:
    """Plot cumulative rainfall distributions for each precursor mode."""
    metrics = [c for c in RAIN_CUMULATIVE_COLS if c in merged.columns]
    labels = ["1 h", "6 h", "1 d", "3 d", "7 d"][: len(metrics)]
    modes = sorted([int(m) for m in merged["kmeans_cluster"].dropna().unique()])

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 3.4), sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    rng = np.random.default_rng(20260708)
    for ax, metric, label in zip(axes, metrics, labels):
        data = [
            pd.to_numeric(merged.loc[merged["kmeans_cluster"] == mode, metric], errors="coerce").dropna()
            for mode in modes
        ]
        box = ax.boxplot(
            data,
            tick_labels=[MODE_SHORT_NAME_MAP.get(mode, str(mode)) for mode in modes],
            showfliers=False,
            patch_artist=True,
            widths=0.55,
        )
        for patch, mode in zip(box["boxes"], modes):
            patch.set_facecolor(MODE_COLOR_MAP.get(mode, "#999999"))
            patch.set_alpha(0.42)
            patch.set_edgecolor(MODE_COLOR_MAP.get(mode, "#999999"))
        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(1.4)
        for i, (mode, values) in enumerate(zip(modes, data), start=1):
            jitter = rng.normal(0, 0.045, size=len(values))
            ax.scatter(
                np.full(len(values), i) + jitter,
                values,
                s=22,
                color=MODE_COLOR_MAP.get(mode, "#999999"),
                alpha=0.72,
                edgecolor="white",
                linewidth=0.35,
            )
        ax.set_title(label)
        ax.set_ylabel("Rainfall (mm)" if metric == metrics[0] else "")
        ax.grid(axis="y", color="#d0d0d0", alpha=0.5, linewidth=0.7)
        ax.tick_params(axis="x", rotation=20)

    add_panel_label(axes[0], "a")
    fig.suptitle("Antecedent rainfall by precursor mode", y=1.02, fontweight="bold")
    fig.tight_layout()

    out = FIGURE_DIR / "mode_cumulative_rainfall_boxplots.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_short_vs_antecedent_scatter(merged: pd.DataFrame) -> Path | None:
    """Plot short-term rainfall against antecedent storage proxy."""
    required = {"event_cum_1d_mm", "rain_7d_minus_1d_mm", "kmeans_cluster"}
    if not required.issubset(merged.columns):
        return None

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for mode, group in merged.groupby("kmeans_cluster"):
        mode = int(mode)
        ax.scatter(
            group["event_cum_1d_mm"],
            group["rain_7d_minus_1d_mm"],
            s=58,
            color=MODE_COLOR_MAP.get(mode, "#999999"),
            label=MODE_SHORT_NAME_MAP.get(mode, str(mode)),
            alpha=0.82,
            edgecolor="white",
            linewidth=0.6,
        )
        for _, row in group.iterrows():
            ax.text(
                row["event_cum_1d_mm"],
                row["rain_7d_minus_1d_mm"],
                f"E{natural_event_number(row['event'])}",
                fontsize=7,
                ha="left",
                va="bottom",
                alpha=0.75,
            )

    ax.set_xlabel("Rainfall during last 1 day (mm)")
    ax.set_ylabel("Rainfall from 7 days to 1 day before event (mm)")
    ax.set_title("Short-term trigger versus antecedent rainfall")
    ax.grid(color="#d0d0d0", alpha=0.55, linewidth=0.7)
    ax.legend(frameon=True)
    add_panel_label(ax, "b")
    fig.tight_layout()

    out = FIGURE_DIR / "mode_short_vs_antecedent_rainfall.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_mode_rainfall_median_profile(summary: pd.DataFrame) -> Path:
    """Plot median rainfall accumulation profile for each precursor mode."""
    metric_order = [c for c in RAIN_CUMULATIVE_COLS if c in summary["metric"].unique()]
    x_labels = ["1 h", "6 h", "1 d", "3 d", "7 d"][: len(metric_order)]

    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    for mode, group in summary.groupby("kmeans_cluster"):
        mode = int(mode)
        values = []
        q25 = []
        q75 = []
        for metric in metric_order:
            row = group.loc[group["metric"] == metric].iloc[0]
            values.append(row["median"])
            q25.append(row["q25"])
            q75.append(row["q75"])
        x = np.arange(len(metric_order))
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2.0,
            color=MODE_COLOR_MAP.get(mode, "#999999"),
            label=MODE_SHORT_NAME_MAP.get(mode, str(mode)),
        )
        ax.fill_between(
            x,
            q25,
            q75,
            color=MODE_COLOR_MAP.get(mode, "#999999"),
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Cumulative rainfall (mm)")
    ax.set_title("Median antecedent rainfall profile")
    ax.grid(color="#d0d0d0", alpha=0.55, linewidth=0.7)
    ax.legend(frameon=True)
    add_panel_label(ax, "c")
    fig.tight_layout()

    out = FIGURE_DIR / "mode_median_rainfall_profile.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_event_mode_rainfall_heatmap(merged: pd.DataFrame) -> Path:
    """Plot event-by-window rainfall heatmap sorted by precursor mode."""
    metrics = [c for c in RAIN_CUMULATIVE_COLS if c in merged.columns]
    plot_df = merged.copy()
    plot_df["_event_order"] = plot_df["event"].map(natural_event_number)
    plot_df = plot_df.sort_values(["kmeans_cluster", "_event_order"])

    values = plot_df[metrics].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, max(5.0, 0.22 * len(plot_df))))
    image = ax.imshow(values, aspect="auto", cmap="YlGnBu")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Rainfall (mm)")

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(["1 h", "6 h", "1 d", "3 d", "7 d"][: len(metrics)])
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(
        [
            f"E{natural_event_number(row.event)} | {row.mode_short_name}"
            for row in plot_df.itertuples()
        ]
    )
    ax.set_title("Event rainfall matrix sorted by precursor mode")
    ax.set_xlabel("Antecedent window")
    ax.set_ylabel("Event and mode")

    boundaries = []
    previous = None
    for idx, mode in enumerate(plot_df["kmeans_cluster"].tolist()):
        if previous is not None and mode != previous:
            boundaries.append(idx - 0.5)
        previous = mode
    for boundary in boundaries:
        ax.axhline(boundary, color="white", linewidth=2.0)

    add_panel_label(ax, "d")
    fig.tight_layout()

    out = FIGURE_DIR / "mode_event_rainfall_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def export_outputs(
    merged: pd.DataFrame,
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    correlations: pd.DataFrame,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "mode_rainfall_relationship.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        make_excel_safe(merged).to_excel(writer, sheet_name="mode_rainfall_event_table", index=False)
        make_excel_safe(summary).to_excel(writer, sheet_name="mode_rainfall_summary", index=False)
        make_excel_safe(tests).to_excel(writer, sheet_name="mode_rainfall_tests", index=False)
        make_excel_safe(correlations).to_excel(writer, sheet_name="rain_feature_correlations", index=False)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    setup_figure_style()

    mode_path = find_mode_result_file()
    mode_df = load_mode_results(mode_path)
    rain_df = load_event_rainfall()
    merged = build_merged_table(mode_df, rain_df)
    summary = summarize_by_mode(merged)
    tests = compare_modes(merged)
    correlations = mode_rainfall_correlations(merged)

    workbook = export_outputs(merged, summary, tests, correlations)
    figures = [
        plot_rainfall_boxplots_by_mode(merged),
        plot_short_vs_antecedent_scatter(merged),
        plot_mode_rainfall_median_profile(summary),
        plot_event_mode_rainfall_heatmap(merged),
    ]
    figures = [fig for fig in figures if fig is not None]

    print(f"Mode result file: {mode_path}")
    print(f"Output workbook: {workbook}")
    print("Figures:")
    for figure in figures:
        print(f"  - {figure}")
    print("\nMode rainfall summary preview:")
    preview = summary.loc[summary["metric"].isin(RAIN_CUMULATIVE_COLS)]
    print(preview[["mode_short_name", "metric", "n", "median", "q25", "q75"]].to_string(index=False))


if __name__ == "__main__":
    main()
