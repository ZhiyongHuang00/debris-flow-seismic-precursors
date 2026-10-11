# -*- coding: utf-8 -*-
"""
Export precursor-curve metrics for all events and all clusters.

Outputs:
1. One Excel per metric for the selected precursor cluster of each event.
2. One combined Excel with one sheet per metric for all event-cluster values.
3. One combined Excel with one sheet per metric for selected precursor clusters.

The metrics combine two groups:
- AMR / DAMR inspired metrics from precursor-curve literature
- background-vs-precursor anomaly metrics already used in this project
"""

from __future__ import annotations

import glob
import math
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from tqdm import tqdm


PROJECT_ROOT = Path(r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben")
DEC_RESULT_DIR = PROJECT_ROOT / r"uilt_oneday_all\02clustering\result\DEC_result"
DEC_PARAMETER_GLOB = str(DEC_RESULT_DIR / "all_parement" / "*.npz")
TIME_FILE = PROJECT_ROOT / r"data\新版完整数据集时间.xlsx"
OUTPUT_DIR = DEC_RESULT_DIR / "precursor_curve_metrics"
PRECURSOR_DIR = OUTPUT_DIR / "precursor_by_metric"
FIGURE_DIR = OUTPUT_DIR / "metric_figures"
BOXPLOT_DIR = OUTPUT_DIR / "metric_boxplots"
PCS_FIGURE_DIR = OUTPUT_DIR / "pcs_figures"

DT_SECONDS = 60
BACKGROUND_RATIO = 0.5
MIN_VALID_COUNT = 6
LOG_AMR_MIN_COUNT = 5
LOG_AMR_MIN_POINTS = 20
EPS = 1e-6
Z_FINAL_THRESHOLD = 2.0
C_VALUE_THRESHOLD = 1.0
TERMINAL_WINDOW_SECONDS = 3600
PREVIOUS_WINDOW_SECONDS = 5 * 3600
TERMINAL_RATIO_THRESHOLD = 0.5
INVERSE_RATE_BIN_SECONDS = 3600

plt.rcParams.update({"font.size": 14, "font.family": "Arial"})


METRIC_SPECS = [
    {
        "name": "exp_k",
        "ascending": False,
        "description": "Exponential growth rate k fitted from cumulative cluster occurrence; larger suggests faster exponential escalation.",
    },
    {
        "name": "amr_m",
        "ascending": True,
        "description": "AMR power-law exponent m; smaller is treated as more precursory.",
    },
    {
        "name": "c_value",
        "ascending": True,
        "description": "Curvature indicator c = RMS_power / RMS_linear; smaller suggests stronger acceleration.",
    },
    {
        "name": "qt_flag",
        "ascending": False,
        "description": "DAMR-inspired acceleration flag; 1 means later-stage rate exceeds early-stage rate.",
    },
    {
        "name": "onset_time_s",
        "ascending": True,
        "description": "Earliest anomaly onset time estimated from cumulative Z-score threshold crossing.",
    },
    {
        "name": "acc",
        "ascending": False,
        "description": "Accumulated count anomaly above background expectation.",
    },
    {
        "name": "car",
        "ascending": False,
        "description": "Cumulative anomaly ratio relative to the background expectation.",
    },
    {
        "name": "z_final",
        "ascending": False,
        "description": "Final cumulative anomaly Z-score.",
    },
    {
        "name": "rate_acceleration",
        "ascending": False,
        "description": "Difference between late-stage and early-stage cumulative slopes.",
    },
    {
        "name": "terminal_growth_ratio",
        "ascending": False,
        "description": "Last-1h occurrence rate divided by the preceding 5h rate; larger means stronger terminal persistence.",
    },
    {
        "name": "terminal_growth_flag",
        "ascending": False,
        "description": "Terminal growth criterion; 1 means the last-1h rate is at least the configured fraction of the previous 5h rate.",
    },
    {
        "name": "inverse_rate_slope",
        "ascending": True,
        "description": "Slope of inverse occurrence rate over hourly bins; more negative suggests failure-forecast-style acceleration.",
    },
]


def safe_sheet_name(name: str) -> str:
    """Excel sheet names must be <= 31 chars and cannot contain some symbols."""
    for old, new in {"/": "_", "\\": "_", "*": "_", "?": "_", ":": "_", "[": "(", "]": ")"}.items():
        name = name.replace(old, new)
    return name[:31]


def get_writable_output_path(path: Path) -> Path:
    """Return a fallback path if the requested Excel file is currently locked."""
    if not path.exists():
        return path

    try:
        with path.open("a+b"):
            return path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        print(f"Warning: {path} is locked; writing to {fallback_path}")
        return fallback_path


def extract_event_order(event_name: str) -> int:
    """Extract the numeric suffix from names like event12 for natural sorting."""
    match = re.search(r"event(\d+)", str(event_name))
    return int(match.group(1)) if match else 10**9


def sort_by_event_order(df: pd.DataFrame, extra_sort_columns: list[str] | None = None, ascending: list[bool] | None = None) -> pd.DataFrame:
    """Sort a DataFrame by natural event order plus optional extra columns."""
    sorted_df = df.copy()
    sorted_df["_event_order"] = sorted_df["event"].map(extract_event_order)

    sort_columns = ["_event_order", "event"]
    sort_ascending = [True, True]

    if extra_sort_columns:
        sort_columns.extend(extra_sort_columns)
        if ascending is None:
            sort_ascending.extend([True] * len(extra_sort_columns))
        else:
            sort_ascending.extend(ascending)

    sorted_df = sorted_df.sort_values(by=sort_columns, ascending=sort_ascending)
    return sorted_df.drop(columns="_event_order")


def build_event_color_map(event_names: list[str]) -> dict[str, tuple]:
    """Assign a stable color to each event."""
    sorted_events = sorted(event_names, key=extract_event_order)
    cmap = plt.get_cmap("tab20", max(len(sorted_events), 1))
    return {event_name: cmap(i) for i, event_name in enumerate(sorted_events)}


def fit_best_power_law(t_rel_s: np.ndarray, cumulative_counts: np.ndarray) -> dict:
    """
    Fit a simple time-to-failure power-law curve by grid-searching m.

    Model:
        N(t) = c0 - a * (tc - t + 1)^m

    For a fixed m, the model is linear in c0 and a, so we solve it with least
    squares and choose the m with the smallest RMS residual.
    """
    tc = float(t_rel_s[-1])
    m_grid = np.linspace(0.05, 0.95, 91)

    best = {
        "amr_m": np.nan,
        "power_rms": np.nan,
        "power_r2": np.nan,
        "power_c0": np.nan,
        "power_a": np.nan,
    }

    if len(t_rel_s) < 3:
        return best

    y = cumulative_counts.astype(float)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    for m_value in m_grid:
        phi = np.power(np.maximum(tc - t_rel_s + 1.0, 1.0), m_value)
        design = np.column_stack([np.ones_like(phi), -phi])
        coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        c0, a_value = coeffs

        # Require a non-negative amplitude so the curve increases toward failure.
        if a_value < 0:
            continue

        y_pred = design @ coeffs
        rms = math.sqrt(np.mean((y - y_pred) ** 2))
        r2 = 1.0 - np.sum((y - y_pred) ** 2) / ss_tot if ss_tot > 0 else 1.0

        if np.isnan(best["power_rms"]) or rms < best["power_rms"]:
            best = {
                "amr_m": float(m_value),
                "power_rms": float(rms),
                "power_r2": float(r2),
                "power_c0": float(c0),
                "power_a": float(a_value),
            }

    return best


def fit_exponential_growth(t_rel_s: np.ndarray, cumulative_counts: np.ndarray) -> dict:
    """
    Fit a simple exponential model to cumulative counts.

    Model:
        N(t) + 1 = A * exp(k * t)

    Taking logarithm:
        log(N(t) + 1) = log(A) + k t
    """
    result = {
        "exp_k": np.nan,
        "exp_intercept": np.nan,
        "exp_r2": np.nan,
        "exp_rms": np.nan,
    }

    if len(t_rel_s) < 3:
        return result

    y_log = np.log(cumulative_counts.astype(float) + 1.0)
    coeffs = np.polyfit(t_rel_s, y_log, 1)
    k_value, intercept = coeffs
    y_log_pred = np.polyval(coeffs, t_rel_s)
    y_pred = np.exp(y_log_pred) - 1.0

    ss_tot = np.sum((cumulative_counts - np.mean(cumulative_counts)) ** 2)
    ss_res = np.sum((cumulative_counts - y_pred) ** 2)
    exp_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    exp_rms = float(math.sqrt(np.mean((cumulative_counts - y_pred) ** 2)))

    result["exp_k"] = float(k_value)
    result["exp_intercept"] = float(intercept)
    result["exp_r2"] = float(exp_r2)
    result["exp_rms"] = exp_rms
    return result


def compute_terminal_growth_metrics(
    mask: np.ndarray,
    dt: int = DT_SECONDS,
    terminal_window_s: int = TERMINAL_WINDOW_SECONDS,
    previous_window_s: int = PREVIOUS_WINDOW_SECONDS,
    ratio_threshold: float = TERMINAL_RATIO_THRESHOLD,
) -> dict:
    """Measure whether a cluster keeps growing during the terminal stage."""
    n_points = len(mask)
    terminal_points = max(int(round(terminal_window_s / dt)), 1)
    previous_points = max(int(round(previous_window_s / dt)), 1)

    terminal_start = max(n_points - terminal_points, 0)
    previous_start = max(terminal_start - previous_points, 0)

    terminal_count = int(mask[terminal_start:].sum())
    previous_count = int(mask[previous_start:terminal_start].sum())

    terminal_duration_s = max((n_points - terminal_start) * dt, dt)
    previous_duration_s = max((terminal_start - previous_start) * dt, dt)

    terminal_rate = float(terminal_count / terminal_duration_s)
    previous_rate = float(previous_count / previous_duration_s)
    terminal_ratio = float(terminal_rate / (previous_rate + EPS))
    terminal_delta = float(terminal_rate - previous_rate)

    if previous_count == 0:
        terminal_flag = int(terminal_count > 0)
    else:
        terminal_flag = int(terminal_rate >= ratio_threshold * previous_rate)

    return {
        "terminal_count_1h": terminal_count,
        "previous_count_5h": previous_count,
        "terminal_rate_1h": terminal_rate,
        "previous_rate_5h": previous_rate,
        "terminal_growth_ratio": terminal_ratio,
        "terminal_growth_delta": terminal_delta,
        "terminal_growth_flag": terminal_flag,
    }


def compute_inverse_rate_metrics(
    mask: np.ndarray,
    dt: int = DT_SECONDS,
    bin_seconds: int = INVERSE_RATE_BIN_SECONDS,
) -> dict:
    """
    Fit inverse occurrence rate over time using fixed bins.

    A negative slope means the inverse rate decreases toward failure, consistent
    with a failure-forecast-style acceleration of the occurrence rate.
    """
    result = {
        "inverse_rate_slope": np.nan,
        "inverse_rate_intercept": np.nan,
        "inverse_rate_r2": np.nan,
        "inverse_rate_valid_bins": 0,
    }

    points_per_bin = max(int(round(bin_seconds / dt)), 1)
    n_bins = len(mask) // points_per_bin
    if n_bins < 4:
        return result

    trimmed_mask = mask[-n_bins * points_per_bin:]
    bin_counts = trimmed_mask.reshape(n_bins, points_per_bin).sum(axis=1).astype(float)
    bin_rates = bin_counts / bin_seconds
    valid = bin_rates > 0
    result["inverse_rate_valid_bins"] = int(valid.sum())

    if valid.sum() < 4:
        return result

    time_hours = (np.arange(n_bins, dtype=float) + 0.5) * (bin_seconds / 3600.0)
    inverse_rates = 1.0 / (bin_rates[valid] + EPS)
    fit_time = time_hours[valid]

    coeffs = np.polyfit(fit_time, inverse_rates, 1)
    predicted = np.polyval(coeffs, fit_time)
    ss_tot = np.sum((inverse_rates - np.mean(inverse_rates)) ** 2)
    ss_res = np.sum((inverse_rates - predicted) ** 2)
    inverse_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    result["inverse_rate_slope"] = float(coeffs[0])
    result["inverse_rate_intercept"] = float(coeffs[1])
    result["inverse_rate_r2"] = float(inverse_r2)
    return result


def compute_cluster_metrics(
    time_points: np.ndarray,
    time_labels: np.ndarray,
    unique_labels: np.ndarray,
    dt: int = DT_SECONDS,
    bg_ratio: float = BACKGROUND_RATIO,
) -> pd.DataFrame:
    """Compute all event-cluster precursor-curve metrics."""
    t_rel_s = np.arange(len(time_points), dtype=float) * dt
    total_duration_s = len(time_points) * dt
    bg_len = int(len(time_points) * bg_ratio)
    bg_len = max(bg_len, 1)

    rows = []
    for cluster_id in unique_labels:
        mask = (time_labels == cluster_id).astype(int)
        cumulative_counts = np.cumsum(mask)
        total_count = int(mask.sum())

        lambda_bg = float(mask[:bg_len].sum() / (bg_len * dt))
        acc = float(total_count - lambda_bg * total_duration_s)
        car = float(total_count / (lambda_bg * total_duration_s + EPS))
        z_final = float(
            (total_count - lambda_bg * total_duration_s)
            / math.sqrt(lambda_bg * total_duration_s + EPS)
        )
        terminal_metrics = compute_terminal_growth_metrics(mask=mask, dt=dt)
        inverse_rate_metrics = compute_inverse_rate_metrics(mask=mask, dt=dt)

        if len(cumulative_counts) > 10:
            mid = len(cumulative_counts) // 2
            early_slope = float(np.polyfit(t_rel_s[:mid], cumulative_counts[:mid], 1)[0])
            late_slope = float(np.polyfit(t_rel_s[mid:], cumulative_counts[mid:], 1)[0])
            rate_acceleration = late_slope - early_slope
        else:
            early_slope = np.nan
            late_slope = np.nan
            rate_acceleration = np.nan

        z_curve = (
            cumulative_counts - lambda_bg * t_rel_s
        ) / np.sqrt(lambda_bg * t_rel_s + EPS)
        onset_idx = np.where(z_curve > 2.0)[0]
        onset_time_s = float(t_rel_s[onset_idx[0]]) if len(onset_idx) > 0 else np.nan

        qt_flag = 0
        if len(cumulative_counts) > 20:
            mid = len(cumulative_counts) // 2
            r1 = float(np.mean(np.diff(cumulative_counts[:mid])))
            r2 = float(np.mean(np.diff(cumulative_counts[mid:])))
            qt_flag = int(r2 > r1)

        linear_r2 = np.nan
        linear_rms = np.nan
        linear_slope = np.nan
        linear_intercept = np.nan
        quadratic_a = np.nan
        quadratic_r2 = np.nan
        r2_gain = np.nan

        if total_count > 1:
            linear_coeffs = np.polyfit(t_rel_s, cumulative_counts, 1)
            linear_pred = np.polyval(linear_coeffs, t_rel_s)
            linear_rms = float(math.sqrt(np.mean((cumulative_counts - linear_pred) ** 2)))
            ss_tot = np.sum((cumulative_counts - np.mean(cumulative_counts)) ** 2)
            linear_r2 = (
                float(1.0 - np.sum((cumulative_counts - linear_pred) ** 2) / ss_tot)
                if ss_tot > 0
                else 1.0
            )
            linear_slope = float(linear_coeffs[0])
            linear_intercept = float(linear_coeffs[1])

        if total_count > 2:
            quad_coeffs = np.polyfit(t_rel_s, cumulative_counts, 2)
            quad_pred = np.polyval(quad_coeffs, t_rel_s)
            ss_tot = np.sum((cumulative_counts - np.mean(cumulative_counts)) ** 2)
            quadratic_r2 = (
                float(1.0 - np.sum((cumulative_counts - quad_pred) ** 2) / ss_tot)
                if ss_tot > 0
                else 1.0
            )
            quadratic_a = float(quad_coeffs[0])
            if not np.isnan(linear_r2):
                r2_gain = quadratic_r2 - linear_r2

        amr_m = np.nan
        c_value = np.nan
        power_r2 = np.nan
        power_rms = np.nan
        power_c0 = np.nan
        power_a = np.nan
        exp_k = np.nan
        exp_intercept = np.nan
        exp_r2 = np.nan
        exp_rms = np.nan

        if total_count > 2:
            exp_fit = fit_exponential_growth(t_rel_s, cumulative_counts)
            exp_k = exp_fit["exp_k"]
            exp_intercept = exp_fit["exp_intercept"]
            exp_r2 = exp_fit["exp_r2"]
            exp_rms = exp_fit["exp_rms"]

        valid = cumulative_counts > LOG_AMR_MIN_COUNT
        if valid.sum() >= LOG_AMR_MIN_POINTS:
            power_fit = fit_best_power_law(t_rel_s[valid], cumulative_counts[valid])
            amr_m = power_fit["amr_m"]
            power_r2 = power_fit["power_r2"]
            power_rms = power_fit["power_rms"]
            power_c0 = power_fit["power_c0"]
            power_a = power_fit["power_a"]
            if not np.isnan(power_rms) and not np.isnan(linear_rms):
                c_value = power_rms / (linear_rms + EPS)

        rows.append(
            {
                "cluster": int(cluster_id),
                "sample_count": total_count,
                "background_rate_hz": lambda_bg,
                "acc": acc,
                "car": car,
                "z_final": z_final,
                "early_slope": early_slope,
                "late_slope": late_slope,
                "rate_acceleration": rate_acceleration,
                "onset_time_s": onset_time_s,
                "qt_flag": qt_flag,
                "terminal_count_1h": terminal_metrics["terminal_count_1h"],
                "previous_count_5h": terminal_metrics["previous_count_5h"],
                "terminal_rate_1h": terminal_metrics["terminal_rate_1h"],
                "previous_rate_5h": terminal_metrics["previous_rate_5h"],
                "terminal_growth_ratio": terminal_metrics["terminal_growth_ratio"],
                "terminal_growth_delta": terminal_metrics["terminal_growth_delta"],
                "terminal_growth_flag": terminal_metrics["terminal_growth_flag"],
                "inverse_rate_slope": inverse_rate_metrics["inverse_rate_slope"],
                "inverse_rate_intercept": inverse_rate_metrics["inverse_rate_intercept"],
                "inverse_rate_r2": inverse_rate_metrics["inverse_rate_r2"],
                "inverse_rate_valid_bins": inverse_rate_metrics["inverse_rate_valid_bins"],
                "linear_slope": linear_slope,
                "linear_intercept": linear_intercept,
                "linear_r2": linear_r2,
                "linear_rms": linear_rms,
                "quadratic_a": quadratic_a,
                "quadratic_r2": quadratic_r2,
                "r2_gain_quadratic_minus_linear": r2_gain,
                "exp_k": exp_k,
                "exp_intercept": exp_intercept,
                "exp_r2": exp_r2,
                "exp_rms": exp_rms,
                "amr_m": amr_m,
                "c_value": c_value,
                "power_r2": power_r2,
                "power_rms": power_rms,
                "power_c0": power_c0,
                "power_a": power_a,
            }
        )

    return pd.DataFrame(rows)


def prepare_event_times(time_file: pd.DataFrame, event_name: str) -> tuple[UTCDateTime, UTCDateTime]:
    """Get the 24 h analysis window for one event."""
    matching_row = time_file[time_file["Event"] == event_name]
    if matching_row.empty:
        raise KeyError(f"Event {event_name} not found in time file.")

    end_time = UTCDateTime(matching_row.iloc[0]["Manual_Label_start"])
    start_time = end_time - 24 * 3600
    return start_time, end_time


def build_all_metrics_table() -> tuple[pd.DataFrame, dict]:
    """Compute metrics for every event-cluster pair and cache event curves."""
    parameter_files = sorted(glob.glob(DEC_PARAMETER_GLOB))
    if not parameter_files:
        raise FileNotFoundError(f"No DEC parameter files found: {DEC_PARAMETER_GLOB}")

    time_file = pd.read_excel(TIME_FILE)
    all_rows = []
    event_curve_cache = {}

    for parameter_file in tqdm(parameter_files, desc="Computing precursor metrics"):
        event_name = os.path.basename(parameter_file).split("_")[0]
        start_time, end_time = prepare_event_times(time_file, event_name)

        data = np.load(parameter_file, allow_pickle=True)
        labels = np.asarray(data["y_pred_last"])
        probabilities = np.asarray(data["probalities"])

        time_points = np.arange(start_time.timestamp, end_time.timestamp, DT_SECONDS)
        min_len = min(len(time_points), len(labels), len(probabilities))
        time_points = time_points[:min_len]
        labels = labels[:min_len]
        event_curve_cache[event_name] = {
            "time_points": time_points,
            "labels": labels,
            "relative_hours": (time_points - time_points[-1]) / 3600.0,
        }

        metrics_df = compute_cluster_metrics(
            time_points=time_points,
            time_labels=labels,
            unique_labels=np.unique(labels),
            dt=DT_SECONDS,
            bg_ratio=BACKGROUND_RATIO,
        )
        metrics_df.insert(0, "event", event_name)
        metrics_df.insert(1, "window_start", start_time.isoformat())
        metrics_df.insert(2, "window_end", end_time.isoformat())
        all_rows.append(metrics_df)

    return pd.concat(all_rows, ignore_index=True), event_curve_cache


def select_precursor_clusters_by_metric(
    all_metrics_df: pd.DataFrame,
    metric_name: str,
    ascending: bool,
) -> pd.DataFrame:
    """Select one precursor cluster per event according to one metric."""
    selected_rows = []

    for event_name, event_df in all_metrics_df.groupby("event", sort=True):
        candidate_df = event_df.dropna(subset=[metric_name]).copy()
        if candidate_df.empty:
            continue

        if metric_name == "qt_flag":
            candidate_df = candidate_df.sort_values(
                by=["qt_flag", "rate_acceleration", "z_final", "sample_count"],
                ascending=[False, False, False, False],
            )
        elif metric_name == "exp_k":
            candidate_df = candidate_df.sort_values(
                by=["exp_k", "exp_r2", "z_final", "sample_count"],
                ascending=[False, False, False, False],
            )
        elif metric_name == "onset_time_s":
            candidate_df = candidate_df.sort_values(
                by=["onset_time_s", "z_final", "sample_count"],
                ascending=[True, False, False],
            )
        elif metric_name == "c_value":
            candidate_df = candidate_df.sort_values(
                by=["c_value", "amr_m", "z_final", "sample_count"],
                ascending=[True, True, False, False],
            )
        elif metric_name == "amr_m":
            candidate_df = candidate_df.sort_values(
                by=["amr_m", "c_value", "z_final", "sample_count"],
                ascending=[True, True, False, False],
            )
        else:
            candidate_df = candidate_df.sort_values(
                by=[metric_name, "z_final", "sample_count"],
                ascending=[ascending, False, False],
            )

        candidate_df["metric_rank_within_event"] = np.arange(1, len(candidate_df) + 1)
        best_row = candidate_df.iloc[0].copy()
        best_row["selected_metric"] = metric_name
        best_row["selected_metric_value"] = best_row[metric_name]
        selected_rows.append(best_row)

    if not selected_rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(selected_rows)
    front_cols = [
        "event",
        "cluster",
        "selected_metric",
        "selected_metric_value",
        "metric_rank_within_event",
        "sample_count",
    ]
    remaining_cols = [col for col in result_df.columns if col not in front_cols]
    result_df = result_df[front_cols + remaining_cols]
    return sort_by_event_order(result_df, extra_sort_columns=["metric_rank_within_event"], ascending=[True])


def select_precursor_clusters_by_z_final_c_value(
    all_metrics_df: pd.DataFrame,
    z_threshold: float = Z_FINAL_THRESHOLD,
    c_threshold: float = C_VALUE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Select precursor clusters using the combined z_final + c_value rule.

    Rule:
        1. Keep clusters with z_final > z_threshold and c_value < c_threshold.
        2. Within each event, select the candidate with the smallest c_value.
        3. Use z_final and sample_count as tie-breakers.
    """
    selected_rows = []
    candidate_rows = []
    missing_rows = []

    required_columns = ["event", "cluster", "z_final", "c_value"]
    missing_columns = [column for column in required_columns if column not in all_metrics_df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns for combined precursor selection: {missing_columns}")

    for event_name, event_df in all_metrics_df.groupby("event", sort=True):
        valid_df = event_df.dropna(subset=["z_final", "c_value"]).copy()
        if valid_df.empty:
            missing_row = {
                "event": event_name,
                "cluster": -1,
                "selected_metric": "z_final_c_value",
                "selection_rule": f"z_final>{z_threshold:g} and c_value<{c_threshold:g}; min c_value",
                "combined_rule_rank": np.nan,
                "z_final": np.nan,
                "c_value": np.nan,
                "sample_count": 0,
                "selection_status": "not_selected",
                "reason": "no_valid_z_final_or_c_value",
                "z_threshold": z_threshold,
                "c_threshold": c_threshold,
                "best_z_final": np.nan,
                "best_c_value": np.nan,
            }
            missing_rows.append(missing_row)
            selected_rows.append(pd.Series(missing_row))
            continue

        candidate_df = valid_df[
            (valid_df["z_final"] > z_threshold)
            & (valid_df["c_value"] < c_threshold)
        ].copy()

        if candidate_df.empty:
            best_z_row = valid_df.sort_values(
                by=["z_final", "sample_count"],
                ascending=[False, False],
            ).iloc[0]
            best_c_row = valid_df.sort_values(
                by=["c_value", "z_final", "sample_count"],
                ascending=[True, False, False],
            ).iloc[0]
            missing_row = {
                "event": event_name,
                "cluster": -1,
                "selected_metric": "z_final_c_value",
                "selection_rule": f"z_final>{z_threshold:g} and c_value<{c_threshold:g}; min c_value",
                "combined_rule_rank": np.nan,
                "z_final": np.nan,
                "c_value": np.nan,
                "sample_count": 0,
                "selection_status": "not_selected",
                "reason": "no_cluster_passed_strict_rule",
                "z_threshold": z_threshold,
                "c_threshold": c_threshold,
                "best_z_final": best_z_row["z_final"],
                "best_z_final_cluster": best_z_row["cluster"],
                "best_c_value": best_c_row["c_value"],
                "best_c_value_cluster": best_c_row["cluster"],
            }
            missing_rows.append(missing_row)
            selected_rows.append(pd.Series(missing_row))
            continue

        candidate_df = candidate_df.sort_values(
            by=["c_value", "z_final", "sample_count"],
            ascending=[True, False, False],
        )
        candidate_df["combined_rule_rank"] = np.arange(1, len(candidate_df) + 1)
        candidate_df["selected_metric"] = "z_final_c_value"
        candidate_df["selection_rule"] = f"z_final>{z_threshold:g} and c_value<{c_threshold:g}; min c_value"
        candidate_df["selection_status"] = "selected"

        candidate_rows.append(candidate_df)
        selected_rows.append(candidate_df.iloc[0].copy())

    selected_df = pd.DataFrame(selected_rows)
    candidates_df = pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame()
    missing_df = pd.DataFrame(missing_rows)

    front_cols = [
        "event",
        "cluster",
        "selected_metric",
        "selection_rule",
        "combined_rule_rank",
        "z_final",
        "c_value",
        "sample_count",
    ]

    if not selected_df.empty:
        remaining_cols = [column for column in selected_df.columns if column not in front_cols]
        selected_df = selected_df[front_cols + remaining_cols]
        selected_df = sort_by_event_order(
            selected_df,
            extra_sort_columns=["combined_rule_rank"],
            ascending=[True],
        )

    if not candidates_df.empty:
        remaining_cols = [column for column in candidates_df.columns if column not in front_cols]
        candidates_df = candidates_df[front_cols + remaining_cols]
        candidates_df = sort_by_event_order(
            candidates_df,
            extra_sort_columns=["combined_rule_rank"],
            ascending=[True],
        )

    if not missing_df.empty:
        missing_df = sort_by_event_order(missing_df)

    return selected_df, candidates_df, missing_df


def within_event_zscore(values: pd.Series) -> pd.Series:
    """Compute a stable within-event z-score for ranking cluster metrics."""
    numeric_values = pd.to_numeric(values, errors="coerce")
    std_value = numeric_values.std(ddof=0)
    if pd.isna(std_value) or std_value <= 0:
        return pd.Series(np.zeros(len(numeric_values)), index=values.index, dtype=float)
    return (numeric_values - numeric_values.mean()) / std_value


def select_precursor_clusters_by_composite_score(all_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select one precursor cluster per event using:

        Score = Z(z_final) + Z(rate_acceleration) - Z(c_value)

    Z(.) is computed within each event across all clusters.
    """
    selected_rows = []
    required_columns = ["event", "cluster", "z_final", "rate_acceleration", "c_value"]
    missing_columns = [column for column in required_columns if column not in all_metrics_df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns for composite-score selection: {missing_columns}")

    for event_name, event_df in all_metrics_df.groupby("event", sort=True):
        valid_df = event_df.dropna(subset=["z_final", "rate_acceleration", "c_value"]).copy()
        if valid_df.empty:
            selected_rows.append(
                pd.Series(
                    {
                        "event": event_name,
                        "cluster": -1,
                        "selected_metric": "composite_score",
                        "selection_rule": "Score = Z(z_final) + Z(rate_acceleration) - Z(c_value)",
                        "selection_status": "not_selected",
                        "reason": "no_valid_z_final_rate_acceleration_or_c_value",
                        "composite_score": np.nan,
                        "z_z_final": np.nan,
                        "z_rate_acceleration": np.nan,
                        "z_c_value": np.nan,
                        "z_final": np.nan,
                        "rate_acceleration": np.nan,
                        "c_value": np.nan,
                        "sample_count": 0,
                    }
                )
            )
            continue

        valid_df["z_z_final"] = within_event_zscore(valid_df["z_final"])
        valid_df["z_rate_acceleration"] = within_event_zscore(valid_df["rate_acceleration"])
        valid_df["z_c_value"] = within_event_zscore(valid_df["c_value"])
        valid_df["composite_score"] = (
            valid_df["z_z_final"]
            + valid_df["z_rate_acceleration"]
            - valid_df["z_c_value"]
        )
        valid_df = valid_df.sort_values(
            by=["composite_score", "z_final", "rate_acceleration", "sample_count"],
            ascending=[False, False, False, False],
        )
        valid_df["composite_rank_within_event"] = np.arange(1, len(valid_df) + 1)
        best_row = valid_df.iloc[0].copy()
        best_row["selected_metric"] = "composite_score"
        best_row["selection_rule"] = "Score = Z(z_final) + Z(rate_acceleration) - Z(c_value)"
        best_row["selection_status"] = "selected"
        selected_rows.append(best_row)

    selected_df = pd.DataFrame(selected_rows)
    front_cols = [
        "event",
        "cluster",
        "selected_metric",
        "selection_rule",
        "selection_status",
        "composite_rank_within_event",
        "composite_score",
        "z_z_final",
        "z_rate_acceleration",
        "z_c_value",
        "z_final",
        "rate_acceleration",
        "c_value",
        "sample_count",
    ]
    remaining_cols = [column for column in selected_df.columns if column not in front_cols]
    selected_df = selected_df[front_cols + remaining_cols]
    return sort_by_event_order(
        selected_df,
        extra_sort_columns=["composite_rank_within_event"],
        ascending=[True],
    )


def select_precursor_clusters_by_pcs(
    all_metrics_df: pd.DataFrame,
    z_threshold: float = Z_FINAL_THRESHOLD,
    c_threshold: float = C_VALUE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Select high-confidence precursor clusters using the PCS rule.

    Candidate rule:
        z_final > z_threshold
        c_value < c_threshold
        terminal_growth_flag == 1

    Ranking rule:
        PCS = Z(z_final) + Z(rate_acceleration) - Z(c_value) + Z(terminal_growth_ratio)
    """
    selected_rows = []
    candidate_rows = []
    missing_rows = []
    required_columns = [
        "event",
        "cluster",
        "z_final",
        "rate_acceleration",
        "c_value",
        "terminal_growth_ratio",
        "terminal_growth_flag",
    ]
    missing_columns = [column for column in required_columns if column not in all_metrics_df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns for PCS selection: {missing_columns}")

    selection_rule = (
        f"z_final>{z_threshold:g}, c_value<{c_threshold:g}, terminal_growth_flag=1; "
        "max PCS"
    )

    for event_name, event_df in all_metrics_df.groupby("event", sort=True):
        valid_df = event_df.dropna(
            subset=["z_final", "rate_acceleration", "c_value", "terminal_growth_ratio"]
        ).copy()

        if valid_df.empty:
            missing_row = {
                "event": event_name,
                "cluster": -1,
                "selected_metric": "pcs",
                "selection_rule": selection_rule,
                "selection_status": "not_selected",
                "reason": "no_valid_pcs_metrics",
                "pcs": np.nan,
                "z_z_final": np.nan,
                "z_rate_acceleration": np.nan,
                "z_c_value": np.nan,
                "z_terminal_growth_ratio": np.nan,
                "z_final": np.nan,
                "rate_acceleration": np.nan,
                "c_value": np.nan,
                "terminal_growth_ratio": np.nan,
                "terminal_growth_flag": np.nan,
                "sample_count": 0,
            }
            missing_rows.append(missing_row)
            selected_rows.append(pd.Series(missing_row))
            continue

        valid_df["z_z_final"] = within_event_zscore(valid_df["z_final"])
        valid_df["z_rate_acceleration"] = within_event_zscore(valid_df["rate_acceleration"])
        valid_df["z_c_value"] = within_event_zscore(valid_df["c_value"])
        valid_df["z_terminal_growth_ratio"] = within_event_zscore(valid_df["terminal_growth_ratio"])
        valid_df["pcs"] = (
            valid_df["z_z_final"]
            + valid_df["z_rate_acceleration"]
            - valid_df["z_c_value"]
            + valid_df["z_terminal_growth_ratio"]
        )

        candidate_df = valid_df[
            (valid_df["z_final"] > z_threshold)
            & (valid_df["c_value"] < c_threshold)
            & (valid_df["terminal_growth_flag"] == 1)
        ].copy()

        if candidate_df.empty:
            best_soft_row = valid_df.sort_values(
                by=["pcs", "z_final", "terminal_growth_ratio", "sample_count"],
                ascending=[False, False, False, False],
            ).iloc[0]
            missing_row = {
                "event": event_name,
                "cluster": -1,
                "selected_metric": "pcs",
                "selection_rule": selection_rule,
                "selection_status": "not_selected",
                "reason": "no_cluster_passed_pcs_strict_rule",
                "pcs": np.nan,
                "z_z_final": np.nan,
                "z_rate_acceleration": np.nan,
                "z_c_value": np.nan,
                "z_terminal_growth_ratio": np.nan,
                "z_final": np.nan,
                "rate_acceleration": np.nan,
                "c_value": np.nan,
                "terminal_growth_ratio": np.nan,
                "terminal_growth_flag": np.nan,
                "sample_count": 0,
                "best_soft_cluster": best_soft_row["cluster"],
                "best_soft_pcs": best_soft_row["pcs"],
                "best_soft_z_final": best_soft_row["z_final"],
                "best_soft_c_value": best_soft_row["c_value"],
                "best_soft_terminal_growth_flag": best_soft_row["terminal_growth_flag"],
                "best_soft_terminal_growth_ratio": best_soft_row["terminal_growth_ratio"],
            }
            missing_rows.append(missing_row)
            selected_rows.append(pd.Series(missing_row))
            continue

        candidate_df = candidate_df.sort_values(
            by=["pcs", "z_final", "terminal_growth_ratio", "sample_count"],
            ascending=[False, False, False, False],
        )
        candidate_df["pcs_rank_within_event"] = np.arange(1, len(candidate_df) + 1)
        candidate_df["selected_metric"] = "pcs"
        candidate_df["selection_rule"] = selection_rule
        candidate_df["selection_status"] = "selected"

        candidate_rows.append(candidate_df)
        selected_rows.append(candidate_df.iloc[0].copy())

    selected_df = pd.DataFrame(selected_rows)
    candidates_df = pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame()
    missing_df = pd.DataFrame(missing_rows)

    front_cols = [
        "event",
        "cluster",
        "selected_metric",
        "selection_rule",
        "selection_status",
        "pcs_rank_within_event",
        "pcs",
        "z_z_final",
        "z_rate_acceleration",
        "z_c_value",
        "z_terminal_growth_ratio",
        "z_final",
        "rate_acceleration",
        "c_value",
        "terminal_growth_ratio",
        "terminal_growth_flag",
        "sample_count",
    ]

    if not selected_df.empty:
        remaining_cols = [column for column in selected_df.columns if column not in front_cols]
        selected_df = selected_df[front_cols + remaining_cols]
        selected_df = sort_by_event_order(
            selected_df,
            extra_sort_columns=["pcs_rank_within_event"],
            ascending=[True],
        )

    if not candidates_df.empty:
        remaining_cols = [column for column in candidates_df.columns if column not in front_cols]
        candidates_df = candidates_df[front_cols + remaining_cols]
        candidates_df = sort_by_event_order(
            candidates_df,
            extra_sort_columns=["pcs_rank_within_event"],
            ascending=[True],
        )

    if not missing_df.empty:
        missing_df = sort_by_event_order(missing_df)

    return selected_df, candidates_df, missing_df


def add_pcs_score_columns(all_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Add event-wise standardized PCS components for every event-cluster row."""
    scored_df = all_metrics_df.copy()
    required_columns = ["z_final", "rate_acceleration", "c_value", "terminal_growth_ratio"]
    missing_columns = [column for column in required_columns if column not in scored_df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns for PCS scoring: {missing_columns}")

    scored_df["z_z_final"] = scored_df.groupby("event")["z_final"].transform(within_event_zscore)
    scored_df["z_rate_acceleration"] = scored_df.groupby("event")["rate_acceleration"].transform(within_event_zscore)
    scored_df["z_c_value"] = scored_df.groupby("event")["c_value"].transform(within_event_zscore)
    scored_df["z_terminal_growth_ratio"] = scored_df.groupby("event")["terminal_growth_ratio"].transform(within_event_zscore)
    scored_df["pcs"] = (
        scored_df["z_z_final"]
        + scored_df["z_rate_acceleration"]
        - scored_df["z_c_value"]
        + scored_df["z_terminal_growth_ratio"]
    )
    return scored_df


def export_metric_workbooks(all_metrics_df: pd.DataFrame):
    """Export all required Excel files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PRECURSOR_DIR.mkdir(parents=True, exist_ok=True)

    precursor_tables = {}
    summary_tail_columns = [
        "sample_count",
        "z_final",
        "rate_acceleration",
        "exp_k",
        "amr_m",
        "c_value",
        "terminal_growth_ratio",
        "terminal_growth_flag",
        "inverse_rate_slope",
        "inverse_rate_r2",
        "onset_time_s",
        "qt_flag",
    ]

    for metric_spec in METRIC_SPECS:
        metric_name = metric_spec["name"]
        ascending = metric_spec["ascending"]

        metric_columns = ["event", "cluster", metric_name]
        metric_columns.extend(
            [column for column in summary_tail_columns if column != metric_name]
        )
        metric_df = all_metrics_df[metric_columns].copy()
        metric_df = sort_by_event_order(metric_df, extra_sort_columns=[metric_name], ascending=[ascending])

        selected_df = select_precursor_clusters_by_metric(
            all_metrics_df=all_metrics_df,
            metric_name=metric_name,
            ascending=ascending,
        )
        precursor_tables[metric_name] = selected_df

        metric_output = get_writable_output_path(PRECURSOR_DIR / f"{metric_name}_precursor_clusters.xlsx")
        with pd.ExcelWriter(metric_output) as writer:
            selected_df.to_excel(writer, sheet_name="precursor_clusters", index=False)
            pd.DataFrame(
                [
                    {
                        "metric": metric_name,
                        "ascending": ascending,
                        "description": metric_spec["description"],
                    }
                ]
            ).to_excel(writer, sheet_name="metric_info", index=False)

    combined_selected_df, combined_candidates_df, combined_missing_df = select_precursor_clusters_by_z_final_c_value(
        all_metrics_df=all_metrics_df,
        z_threshold=Z_FINAL_THRESHOLD,
        c_threshold=C_VALUE_THRESHOLD,
    )
    precursor_tables["z_final_c_value"] = combined_selected_df

    combined_output = get_writable_output_path(PRECURSOR_DIR / "z_final_c_value_precursor_clusters.xlsx")
    with pd.ExcelWriter(combined_output) as writer:
        combined_selected_df.to_excel(writer, sheet_name="selected_precursor", index=False)
        combined_candidates_df.to_excel(writer, sheet_name="all_candidates", index=False)
        combined_missing_df.to_excel(writer, sheet_name="events_without_candidate", index=False)
        pd.DataFrame(
            [
                {
                    "rule_name": "z_final_c_value",
                    "z_final_threshold": Z_FINAL_THRESHOLD,
                    "c_value_threshold": C_VALUE_THRESHOLD,
                    "selection_rule": (
                        "Keep clusters with z_final > threshold and c_value < threshold; "
                        "select the smallest c_value within each event."
                    ),
                    "selected_event_count": len(combined_selected_df),
                    "missing_event_count": len(combined_missing_df),
                }
            ]
        ).to_excel(writer, sheet_name="rule_info", index=False)

    score_selected_df = select_precursor_clusters_by_composite_score(all_metrics_df)
    precursor_tables["composite_score"] = score_selected_df

    score_output = get_writable_output_path(PRECURSOR_DIR / "composite_score_precursor_clusters.xlsx")
    with pd.ExcelWriter(score_output) as writer:
        score_selected_df.to_excel(writer, sheet_name="selected_precursor", index=False)
        pd.DataFrame(
            [
                {
                    "rule_name": "composite_score",
                    "selection_rule": "Score = Z(z_final) + Z(rate_acceleration) - Z(c_value)",
                    "zscore_scope": "Within each event across all clusters",
                    "selected_event_count": len(score_selected_df),
                    "not_selected_event_count": int((score_selected_df["cluster"] == -1).sum()),
                }
            ]
        ).to_excel(writer, sheet_name="rule_info", index=False)

    pcs_selected_df, pcs_candidates_df, pcs_missing_df = select_precursor_clusters_by_pcs(
        all_metrics_df=all_metrics_df,
        z_threshold=Z_FINAL_THRESHOLD,
        c_threshold=C_VALUE_THRESHOLD,
    )
    precursor_tables["pcs"] = pcs_selected_df

    pcs_output = get_writable_output_path(PRECURSOR_DIR / "pcs_precursor_clusters.xlsx")
    with pd.ExcelWriter(pcs_output) as writer:
        pcs_selected_df.to_excel(writer, sheet_name="selected_precursor", index=False)
        pcs_candidates_df.to_excel(writer, sheet_name="all_candidates", index=False)
        pcs_missing_df.to_excel(writer, sheet_name="events_without_candidate", index=False)
        pd.DataFrame(
            [
                {
                    "rule_name": "pcs",
                    "selection_rule": (
                        "Candidate: z_final > threshold, c_value < threshold, "
                        "terminal_growth_flag = 1; rank by PCS."
                    ),
                    "pcs_formula": "PCS = Z(z_final) + Z(rate_acceleration) - Z(c_value) + Z(terminal_growth_ratio)",
                    "zscore_scope": "Within each event across all clusters",
                    "z_final_threshold": Z_FINAL_THRESHOLD,
                    "c_value_threshold": C_VALUE_THRESHOLD,
                    "selected_event_count": int((pcs_selected_df["cluster"] != -1).sum()),
                    "not_selected_event_count": int((pcs_selected_df["cluster"] == -1).sum()),
                }
            ]
        ).to_excel(writer, sheet_name="rule_info", index=False)

    all_metrics_output = get_writable_output_path(OUTPUT_DIR / "all_cluster_metrics_by_sheet.xlsx")
    with pd.ExcelWriter(all_metrics_output) as writer:
        sort_by_event_order(all_metrics_df).to_excel(writer, sheet_name="all_metrics_wide", index=False)
        for metric_spec in METRIC_SPECS:
            metric_name = metric_spec["name"]
            metric_columns = ["event", "cluster", metric_name]
            metric_columns.extend(
                [column for column in summary_tail_columns if column != metric_name]
            )
            sheet_df = all_metrics_df[metric_columns].copy()
            sheet_df = sort_by_event_order(
                sheet_df,
                extra_sort_columns=[metric_name],
                ascending=[metric_spec["ascending"]],
            )
            sheet_df.to_excel(writer, sheet_name=safe_sheet_name(metric_name), index=False)

    precursor_summary_output = get_writable_output_path(OUTPUT_DIR / "precursor_clusters_all_metrics.xlsx")
    with pd.ExcelWriter(precursor_summary_output) as writer:
        for metric_name, selected_df in precursor_tables.items():
            selected_df.to_excel(writer, sheet_name=safe_sheet_name(metric_name), index=False)


def plot_metric_precursor_vs_nonprecursor(
    metric_name: str,
    selected_df: pd.DataFrame,
    event_curve_cache: dict,
    save_dir: Path,
):
    """
    Plot one figure per metric.

    Row 1: selected precursor cluster curve for each event.
    Row 2: all non-selected cluster curves for all events.
    """
    if selected_df.empty:
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    ordered_events = sorted(selected_df["event"].unique(), key=extract_event_order)
    event_color_map = build_event_color_map(ordered_events)
    selected_cluster_map = dict(zip(selected_df["event"], selected_df["cluster"]))

    fig, axes = plt.subplots(2, 1, figsize=(6, 10), sharex=True)
    ax_top, ax_bottom = axes

    precursor_line_count = 0
    nonprecursor_line_count = 0

    for event_name in ordered_events:
        if event_name not in event_curve_cache:
            continue

        cache = event_curve_cache[event_name]
        relative_hours = cache["relative_hours"]
        labels = cache["labels"]
        selected_cluster = selected_cluster_map[event_name]
        event_color = event_color_map[event_name]

        precursor_curve = np.cumsum((labels == selected_cluster).astype(int))
        ax_top.plot(
            relative_hours,
            precursor_curve,
            color=event_color,
            alpha=0.85,
            linewidth=1.8,
        )
        precursor_line_count += 1

        for cluster_id in np.unique(labels):
            if int(cluster_id) == int(selected_cluster):
                continue
            nonprecursor_curve = np.cumsum((labels == cluster_id).astype(int))
            ax_bottom.plot(
                relative_hours,
                nonprecursor_curve,
                color=event_color,
                alpha=0.18,
                linewidth=1.0,
            )
            nonprecursor_line_count += 1

    ax_top.axvline(0, color="black", linestyle="--", linewidth=1)
    ax_bottom.axvline(0, color="black", linestyle="--", linewidth=1)

    ax_top.set_title(
        f"{metric_name}: selected precursor cluster curves",
        fontsize=16,
    )
    ax_bottom.set_title(
        f"{metric_name}: all non-precursor cluster curves",
        fontsize=16,
    )

    ax_top.set_ylabel("Cumulative count")
    ax_bottom.set_ylabel("Cumulative count")
    ax_bottom.set_xlabel("Time before debris-flow initiation (hours)")

    ax_top.grid(alpha=0.3)
    ax_bottom.grid(alpha=0.3)

    ax_top.text(
        0.01,
        0.95,
        f"Lines: {precursor_line_count}",
        transform=ax_top.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    ax_bottom.text(
        0.01,
        0.95,
        f"Lines: {nonprecursor_line_count}",
        transform=ax_bottom.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    plt.tight_layout()
    fig.savefig(save_dir / f"{metric_name}_precursor_vs_nonprecursor.png", dpi=300)
    plt.close(fig)


def export_metric_figures(all_metrics_df: pd.DataFrame, event_curve_cache: dict):
    """Export one two-row figure per metric."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for metric_spec in METRIC_SPECS:
        metric_name = metric_spec["name"]
        selected_df = select_precursor_clusters_by_metric(
            all_metrics_df=all_metrics_df,
            metric_name=metric_name,
            ascending=metric_spec["ascending"],
        )
        plot_metric_precursor_vs_nonprecursor(
            metric_name=metric_name,
            selected_df=selected_df,
            event_curve_cache=event_curve_cache,
            save_dir=FIGURE_DIR,
        )


def remove_iqr_outliers(values: np.ndarray, iqr_multiplier: float = 1.5) -> tuple[np.ndarray, int]:
    """Remove outliers using the IQR rule and return kept values plus removed count."""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]

    if len(finite_values) < 4:
        return finite_values, 0

    q1, q3 = np.percentile(finite_values, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return finite_values, 0

    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    keep_mask = (finite_values >= lower_bound) & (finite_values <= upper_bound)
    return finite_values[keep_mask], int((~keep_mask).sum())


def plot_metric_value_boxplots(all_metrics_df: pd.DataFrame):
    """Plot violin plots with overlaid boxplots after removing IQR outliers."""
    BOXPLOT_DIR.mkdir(parents=True, exist_ok=True)

    for metric_spec in METRIC_SPECS:
        metric_name = metric_spec["name"]
        selected_df = select_precursor_clusters_by_metric(
            all_metrics_df=all_metrics_df,
            metric_name=metric_name,
            ascending=metric_spec["ascending"],
        )
        if selected_df.empty:
            continue

        selected_pairs = {
            (row.event, int(row.cluster))
            for row in selected_df.itertuples(index=False)
        }

        plot_df = all_metrics_df[["event", "cluster", metric_name]].copy()
        plot_df = plot_df.dropna(subset=[metric_name]).copy()
        plot_df["group"] = plot_df.apply(
            lambda row: "Precursor"
            if (row["event"], int(row["cluster"])) in selected_pairs
            else "Non-precursor",
            axis=1,
        )

        precursor_values_raw = plot_df.loc[plot_df["group"] == "Precursor", metric_name].values
        nonprecursor_values_raw = plot_df.loc[plot_df["group"] == "Non-precursor", metric_name].values

        precursor_values, precursor_removed = remove_iqr_outliers(precursor_values_raw)
        nonprecursor_values, nonprecursor_removed = remove_iqr_outliers(nonprecursor_values_raw)

        if len(precursor_values) == 0 or len(nonprecursor_values) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        positions = [1, 2]
        labels = ["Precursor", "Non-precursor"]
        colors = ["#d95f02", "#1b9e77"]

        violin = ax.violinplot(
            [precursor_values, nonprecursor_values],
            positions=positions,
            widths=0.75,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, color in zip(violin["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.25)

        box = ax.boxplot(
            [precursor_values, nonprecursor_values],
            positions=positions,
            patch_artist=True,
            widths=0.28,
            showfliers=False,
        )

        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.0)
        for whisker in box["whiskers"]:
            whisker.set_color("black")
            whisker.set_linewidth(1.0)
        for cap in box["caps"]:
            cap.set_color("black")
            cap.set_linewidth(1.0)
        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(1.8)

        precursor_median = np.median(precursor_values)
        nonprecursor_median = np.median(nonprecursor_values)

        ax.set_title(f"{metric_name}: precursor vs non-precursor", fontsize=15)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric_name)
        ax.grid(axis="y", alpha=0.3)
        ax.text(
            0.02,
            0.98,
            (
                f"Precursor n={len(precursor_values)}, median={precursor_median:.4g}, "
                f"removed={precursor_removed}\n"
                f"Non-precursor n={len(nonprecursor_values)}, median={nonprecursor_median:.4g}, "
                f"removed={nonprecursor_removed}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

        plt.tight_layout()
        fig.savefig(BOXPLOT_DIR / f"{metric_name}_precursor_vs_nonprecursor_violin_boxplot.png", dpi=300)
        plt.close(fig)


def plot_two_group_violin_boxplot(
    precursor_values_raw: np.ndarray,
    nonprecursor_values_raw: np.ndarray,
    metric_label: str,
    title: str,
    save_path: Path,
):
    """Plot a two-group violin plot with an overlaid boxplot."""
    precursor_values, precursor_removed = remove_iqr_outliers(precursor_values_raw)
    nonprecursor_values, nonprecursor_removed = remove_iqr_outliers(nonprecursor_values_raw)

    if len(precursor_values) == 0 or len(nonprecursor_values) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    positions = [1, 2]
    labels = ["PCS precursor", "Non-precursor"]
    colors = ["#d95f02", "#1b9e77"]

    violin = ax.violinplot(
        [precursor_values, nonprecursor_values],
        positions=positions,
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.25)

    box = ax.boxplot(
        [precursor_values, nonprecursor_values],
        positions=positions,
        patch_artist=True,
        widths=0.28,
        showfliers=False,
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)
    for line_group in ["whiskers", "caps", "medians"]:
        for line in box[line_group]:
            line.set_color("black")
            line.set_linewidth(1.5 if line_group == "medians" else 1.0)

    precursor_median = np.median(precursor_values)
    nonprecursor_median = np.median(nonprecursor_values)

    ax.set_title(title, fontsize=15)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric_label)
    ax.grid(axis="y", alpha=0.3)
    ax.text(
        0.02,
        0.98,
        (
            f"PCS precursor n={len(precursor_values)}, median={precursor_median:.4g}, "
            f"removed={precursor_removed}\n"
            f"Non-precursor n={len(nonprecursor_values)}, median={nonprecursor_median:.4g}, "
            f"removed={nonprecursor_removed}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def export_pcs_distribution_figures(all_metrics_df: pd.DataFrame):
    """Export PCS and component-metric distributions for PCS-selected clusters."""
    PCS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pcs_selected_df, _, _ = select_precursor_clusters_by_pcs(
        all_metrics_df=all_metrics_df,
        z_threshold=Z_FINAL_THRESHOLD,
        c_threshold=C_VALUE_THRESHOLD,
    )
    selected_pairs = {
        (row.event, int(row.cluster))
        for row in pcs_selected_df.itertuples(index=False)
        if int(row.cluster) != -1
    }
    if not selected_pairs:
        return

    scored_df = add_pcs_score_columns(all_metrics_df)
    scored_df["group"] = scored_df.apply(
        lambda row: "PCS precursor"
        if (row["event"], int(row["cluster"])) in selected_pairs
        else "Non-precursor",
        axis=1,
    )

    metrics_to_plot = [
        ("pcs", "PCS"),
        ("z_final", r"$S_c$ / z_final"),
        ("rate_acceleration", r"$A_c$ / rate acceleration"),
        ("c_value", r"$C_c$ / c-value"),
        ("terminal_growth_ratio", r"$G_c$ / terminal growth ratio"),
    ]

    for metric_name, metric_label in metrics_to_plot:
        metric_df = scored_df.dropna(subset=[metric_name]).copy()
        precursor_values = metric_df.loc[metric_df["group"] == "PCS precursor", metric_name].values
        nonprecursor_values = metric_df.loc[metric_df["group"] == "Non-precursor", metric_name].values
        plot_two_group_violin_boxplot(
            precursor_values_raw=precursor_values,
            nonprecursor_values_raw=nonprecursor_values,
            metric_label=metric_label,
            title=f"PCS-selected precursor vs non-precursor: {metric_name}",
            save_path=PCS_FIGURE_DIR / f"pcs_selected_{metric_name}_violin_boxplot.png",
        )


def export_pcs_temporal_evolution_figure_11(all_metrics_df: pd.DataFrame, event_curve_cache: dict):
    """Plot PCS-selected precursor cumulative curves and hourly-count heatmap."""
    PCS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pcs_selected_df, _, _ = select_precursor_clusters_by_pcs(
        all_metrics_df=all_metrics_df,
        z_threshold=Z_FINAL_THRESHOLD,
        c_threshold=C_VALUE_THRESHOLD,
    )
    pcs_selected_df = pcs_selected_df[pcs_selected_df["cluster"] != -1].copy()
    if pcs_selected_df.empty:
        return

    ordered_events = [
        event_name
        for event_name in sorted(pcs_selected_df["event"].unique(), key=extract_event_order)
        if event_name in event_curve_cache
    ]
    if not ordered_events:
        return

    selected_cluster_map = dict(zip(pcs_selected_df["event"], pcs_selected_df["cluster"]))
    event_color_map = build_event_color_map(ordered_events)
    hour_edges = np.arange(-24, 1, 1)
    heatmap_rows = []
    heatmap_labels = []

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7, 12),
        gridspec_kw={"height_ratios": [1.0, 1.0]},
        sharex=False,
    )
    ax_curve, ax_heatmap = axes

    for event_name in ordered_events:
        cache = event_curve_cache[event_name]
        relative_hours = cache["relative_hours"]
        labels = cache["labels"]
        selected_cluster = int(selected_cluster_map[event_name])
        mask = labels == selected_cluster
        cumulative_curve = np.cumsum(mask.astype(int))

        ax_curve.plot(
            relative_hours,
            cumulative_curve,
            color=event_color_map[event_name],
            alpha=0.78,
            linewidth=1.5,
        )

        hourly_counts, _ = np.histogram(relative_hours[mask], bins=hour_edges)
        heatmap_rows.append(hourly_counts)
        heatmap_labels.append(event_name)

    heatmap = np.asarray(heatmap_rows, dtype=float)
    im = ax_heatmap.imshow(
        heatmap,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=[hour_edges[0], hour_edges[-1], len(heatmap_labels) - 0.5, -0.5],
        cmap="YlOrRd",
    )

    ax_curve.axvline(0, color="black", linestyle="--", linewidth=1)
    ax_curve.set_title("PCS-selected precursor cluster cumulative evolution", fontsize=16)
    ax_curve.set_ylabel("Cumulative count")
    ax_curve.set_xlim(-24, 0)
    ax_curve.grid(alpha=0.25)
    ax_curve.text(
        0.01,
        0.95,
        f"Selected events: {len(ordered_events)}",
        transform=ax_curve.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    ax_heatmap.set_title("Hourly precursor-cluster occurrence counts", fontsize=16)
    ax_heatmap.set_xlabel("Time before debris-flow initiation (hours)")
    ax_heatmap.set_ylabel("Event")
    ax_heatmap.set_xticks(np.arange(-24, 1, 1))
    ax_heatmap.set_xticklabels([str(hour) for hour in np.arange(-24, 1, 1)], rotation=0)
    ax_heatmap.set_yticks(np.arange(len(heatmap_labels)))
    ax_heatmap.set_yticklabels(heatmap_labels)

    cbar = fig.colorbar(im, ax=ax_heatmap, pad=0.015)
    cbar.set_label("Hourly count")

    plt.tight_layout()
    fig.savefig(PCS_FIGURE_DIR / "pcs_precursor_temporal_evolution_heatmap.png", dpi=300)
    plt.close(fig)


def export_pcs_temporal_evolution_figure(
        all_metrics_df: pd.DataFrame,
        event_curve_cache: dict
):
    """Plot PCS-selected precursor cumulative curves and hourly-count heatmap.

    图1：每分钟累计前兆簇出现次数曲线
    图2：每小时前兆簇出现次数热力图
    """
    import seaborn as sns

    PCS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    pcs_selected_df, _, _ = select_precursor_clusters_by_pcs(
        all_metrics_df=all_metrics_df,
        z_threshold=Z_FINAL_THRESHOLD,
        c_threshold=C_VALUE_THRESHOLD,
    )

    pcs_selected_df = pcs_selected_df[
        pcs_selected_df["cluster"] != -1
    ].copy()

    if pcs_selected_df.empty:
        return

    ordered_events = [
        event_name
        for event_name in sorted(
            pcs_selected_df["event"].unique(),
            key=extract_event_order
        )
        if event_name in event_curve_cache
    ]

    if not ordered_events:
        return

    selected_cluster_map = dict(
        zip(
            pcs_selected_df["event"],
            pcs_selected_df["cluster"]
        )
    )

    # ========================
    # 图像设置
    # ========================
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(8, 13),
        gridspec_kw={"height_ratios": [1, 1.2]}
    )

    # ========================
    # 颜色设置
    # ========================
    tab20 = sns.color_palette("tab20", 20)
    tab20b = sns.color_palette("tab20b", 20)
    tab20c = sns.color_palette("tab20c", 20)

    colors = tab20 + tab20b + tab20c

    heatmap_data = []
    event_labels = []

    # ========================
    # 遍历每个事件
    # ========================
    for i, event_name in enumerate(ordered_events):

        cache = event_curve_cache[event_name]

        relative_hours = np.asarray(cache["relative_hours"])
        labels = np.asarray(cache["labels"])

        selected_cluster = int(selected_cluster_map[event_name])

        # 只保留 -24 h 到 0 h
        valid_mask = (
            (relative_hours >= -24) &
            (relative_hours <= 0)
        )

        relative_hours = relative_hours[valid_mask]
        labels = labels[valid_mask]

        if len(relative_hours) == 0:
            continue

        # 排序，防止曲线乱连
        sort_idx = np.argsort(relative_hours)
        relative_hours = relative_hours[sort_idx]
        labels = labels[sort_idx]

        precursor_mask = labels == selected_cluster

        # ================================
        # 图1：每分钟累计前兆数曲线
        # ================================
        relative_minutes = ((relative_hours + 24) * 60).astype(int)

        minute_df = pd.DataFrame({
            "Minute": relative_minutes,
            "New_Label": precursor_mask.astype(int)
        })

        minute_df = minute_df[
            (minute_df["Minute"] >= 0) &
            (minute_df["Minute"] <= 1440)
        ]

        minutely_precursor = (
            minute_df
            .groupby("Minute")["New_Label"]
            .sum()
            .reindex(np.arange(0, 1441), fill_value=0)
            .cumsum()
        )

        ax[0].plot(
            range(len(minutely_precursor)),
            minutely_precursor.values,
            color=colors[i % len(colors)],
            linewidth=1.5,
            alpha=0.85,
            label=f"{event_name}"
        )

        # ================================
        # 图2：每小时热力图
        # ================================
        hour_edges = np.arange(-24, 1, 1)

        hourly_counts, _ = np.histogram(
            relative_hours[precursor_mask],
            bins=hour_edges
        )

        if len(hourly_counts) < 24:
            hourly_counts = np.pad(
                hourly_counts,
                (0, 24 - len(hourly_counts)),
                constant_values=0
            )

        heatmap_data.append(hourly_counts[:24])
        event_labels.append(event_name)

    # ========================
    # 图1：累计曲线设置
    # ========================
    ax[0].set_xlim(0, 1440)
    ax[0].set_title(
        "Cumulative Precursor Signal Count per Minute (Last 1 Days)"
    )
    ax[0].set_ylabel("Cumulative Count")
    ax[0].set_xlabel("Time before debris-flow initiation (min)")

    ax[0].grid(alpha=0.25)

    ax[0].legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fontsize=7,
        frameon=False
    )

    # ========================
    # 图2：热力图
    # ========================
    heatmap_array = np.array(heatmap_data)

    cbar_ax = fig.add_axes(
        [0.25, 0.08, 0.5, 0.02]
    )

    sns.heatmap(
        heatmap_array,
        ax=ax[1],
        cmap="YlOrRd",
        cbar=True,
        cbar_ax=cbar_ax,
        xticklabels=[f"-{23 - i}h" for i in range(24)],
        yticklabels=event_labels
    )

    ax[1].tick_params(axis="y", labelsize=10)
    ax[1].tick_params(axis="x", labelsize=10, rotation=0)

    ax[1].set_title(
        "Hourly Precursor Signal Count (Heatmap)"
    )
    ax[1].set_xlabel(
        "Time (hour before debris flow)"
    )
    ax[1].set_ylabel("Event")

    cbar_ax.set_xlabel("Hourly Count")

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    fig.savefig(
        PCS_FIGURE_DIR / "pcs_precursor_cumulative_and_hourly_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)


def export_pcs_precursor_vs_nonprecursor_evolution_figure(
    all_metrics_df: pd.DataFrame,
    event_curve_cache: dict,
):
    """Plot PCS-selected precursor curves and all non-precursor cluster curves.

    Top panel:
        One cumulative curve per PCS-selected precursor cluster.

    Bottom panel:
        Cumulative curves for every cluster that was not selected as the
        PCS precursor. If an event has no PCS-selected precursor cluster,
        all clusters in that event are plotted in the bottom panel.
    """
    PCS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    pcs_selected_df, _, _ = select_precursor_clusters_by_pcs(
        all_metrics_df=all_metrics_df,
        z_threshold=Z_FINAL_THRESHOLD,
        c_threshold=C_VALUE_THRESHOLD,
    )

    if pcs_selected_df.empty:
        return

    ordered_events = [
        event_name
        for event_name in sorted(
            pcs_selected_df["event"].unique(),
            key=extract_event_order,
        )
        if event_name in event_curve_cache
    ]

    if not ordered_events:
        return

    selected_cluster_map = {
        row.event: int(row.cluster)
        for row in pcs_selected_df.itertuples(index=False)
    }
    event_color_map = build_event_color_map(ordered_events)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15]},
    )
    ax_top, ax_bottom = axes

    precursor_line_count = 0
    nonprecursor_line_count = 0

    for event_name in ordered_events:
        cache = event_curve_cache[event_name]
        relative_hours = np.asarray(cache["relative_hours"])
        labels = np.asarray(cache["labels"])

        valid_mask = (relative_hours >= -24) & (relative_hours <= 0)
        relative_hours = relative_hours[valid_mask]
        labels = labels[valid_mask]

        if len(relative_hours) == 0:
            continue

        sort_idx = np.argsort(relative_hours)
        relative_hours = relative_hours[sort_idx]
        labels = labels[sort_idx]

        selected_cluster = selected_cluster_map.get(event_name, -1)
        event_color = event_color_map[event_name]

        if selected_cluster != -1:
            precursor_curve = np.cumsum((labels == selected_cluster).astype(int))
            ax_top.plot(
                relative_hours,
                precursor_curve,
                color=event_color,
                alpha=0.82,
                linewidth=1.6,
            )
            precursor_line_count += 1

        for cluster_id in np.unique(labels):
            if selected_cluster != -1 and int(cluster_id) == selected_cluster:
                continue

            nonprecursor_curve = np.cumsum((labels == cluster_id).astype(int))
            ax_bottom.plot(
                relative_hours,
                nonprecursor_curve,
                color=event_color,
                alpha=0.16,
                linewidth=0.9,
            )
            nonprecursor_line_count += 1

    for ax in axes:
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(-24, 0)
        ax.set_ylabel("Cumulative count")
        ax.grid(alpha=0.25)

    ax_top.set_title("PCS-selected precursor cluster cumulative curves", fontsize=16)
    ax_bottom.set_title("All non-precursor cluster cumulative curves under PCS selection", fontsize=16)
    ax_bottom.set_xlabel("Time before debris-flow initiation (hours)")

    ax_top.text(
        0.01,
        0.95,
        f"Selected precursor curves: {precursor_line_count}",
        transform=ax_top.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax_bottom.text(
        0.01,
        0.95,
        f"Non-precursor curves: {nonprecursor_line_count}",
        transform=ax_bottom.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    plt.tight_layout()
    fig.savefig(
        PCS_FIGURE_DIR / "pcs_precursor_vs_nonprecursor_evolution_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    all_metrics_df, event_curve_cache = build_all_metrics_table()
    export_metric_workbooks(all_metrics_df)
    export_metric_figures(all_metrics_df, event_curve_cache)
    plot_metric_value_boxplots(all_metrics_df)
    export_pcs_distribution_figures(all_metrics_df)
    export_pcs_temporal_evolution_figure(all_metrics_df, event_curve_cache)
    export_pcs_precursor_vs_nonprecursor_evolution_figure(all_metrics_df, event_curve_cache)
    print(f"Saved precursor metric outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
