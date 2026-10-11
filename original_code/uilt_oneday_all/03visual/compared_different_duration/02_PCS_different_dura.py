# -*- coding: utf-8 -*-
"""
@Time    : 2026/7/5 14:23
@Author  : HuangZhiyong
@email   : 1524338616@qq.com
@File    : 02_PCS_different_dura.py
@Function:
    Run PCS precursor-cluster selection on each event-duration DEC result.

    Outputs:
    1. PCS metrics for all event-duration-cluster rows.
    2. Strict PCS selection result for each event and duration.
    3. Cumulative-count curves of PCS-selected clusters, grouped by duration.
    4. Boxplots for PCS and its component metrics.

    PCS rule:
        hard candidate rule:
            z_final > 2.0
            c_value < 1.0
            terminal_growth_flag == 1

        ranking score:
            PCS = Z(z_final) + Z(rate_acceleration)
                  - Z(c_value) + Z(terminal_growth_ratio)
"""

from __future__ import annotations

import math
import os
import re
import importlib.util
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


# =========================================================
# 1. Configuration
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben")
CLUSTERING_ROOT = PROJECT_ROOT / "uilt_oneday_all" / "02clustering"
REFERENCE_PCS_SCRIPT = SCRIPT_DIR.parent / "09_export_precursor_curve_metrics.py"

DIFFERENT_DURATION_ROOT = CLUSTERING_ROOT / "result_different_duration_DEC_per_event"
ORIGINAL_24H_NPZ_DIR = (
    CLUSTERING_ROOT
    / "result_all_event_DEC_K6"
    / "DEC_result"
    / "all_parement"
)
OUTPUT_ROOT = SCRIPT_DIR / "pcs_different_duration"
CURVE_DIR = OUTPUT_ROOT / "pcs_cumulative_curves_by_duration"
BOXPLOT_DIR = OUTPUT_ROOT / "pcs_boxplots"
ANALYSIS_FIGURE_DIR = OUTPUT_ROOT / "pcs_analysis_figures"

# None means all duration_*h folders under DIFFERENT_DURATION_ROOT.
# Example: DURATION_HOURS = [1, 2, 4, 8, 12, 18]
DURATION_HOURS: list[int] | None = None
INCLUDE_ORIGINAL_24H = True

# None means all event*_all_parement.npz files found in each duration folder.
# Example: EVENT_NAMES = ["event1", "event7", "event20"]
EVENT_NAMES: list[str] | None = None

DT_SECONDS = 60
BACKGROUND_RATIO = 0.5
EPS = 1e-6

LOG_AMR_MIN_COUNT = 5
LOG_AMR_MIN_POINTS = 20
TERMINAL_WINDOW_SECONDS = 3600
PREVIOUS_WINDOW_SECONDS = 5 * 3600
TERMINAL_RATIO_THRESHOLD = 0.5
INVERSE_RATE_BIN_SECONDS = 3600

Z_FINAL_THRESHOLD = 2.0
C_VALUE_THRESHOLD = 1.0

FIG_DPI = 300
PLOT_X_AS_DURATION = True
FOCUS_DURATIONS = [3, 4, 24]

plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "Arial",
        "axes.unicode_minus": False,
    }
)
plt.ioff()


def load_reference_pcs_module():
    """
    Load the original PCS implementation so this duration-comparison script
    uses exactly the same metric and selection algorithm as 09_export.
    """
    module_name = "reference_09_export_precursor_curve_metrics"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, REFERENCE_PCS_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reference PCS script: {REFERENCE_PCS_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# =========================================================
# 2. Basic helpers
# =========================================================
@dataclass
class EventDurationData:
    event: str
    duration_hours: int
    npz_path: Path
    times: np.ndarray
    hours_to_end: np.ndarray
    labels: np.ndarray
    n_clusters: int


def natural_event_key(value: str) -> int:
    match = re.search(r"event(\d+)", str(value), flags=re.IGNORECASE)
    return int(match.group(1)) if match else 10**9


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


def discover_duration_hours() -> list[int]:
    if DURATION_HOURS is not None:
        durations = {int(item) for item in DURATION_HOURS}
        if INCLUDE_ORIGINAL_24H:
            durations.add(24)
        return sorted(durations)

    durations = []
    for duration_dir in DIFFERENT_DURATION_ROOT.glob("duration_*h"):
        duration = parse_duration_from_dir_name(duration_dir)
        if duration is not None:
            durations.append(duration)
    if INCLUDE_ORIGINAL_24H:
        durations.append(24)
    return sorted(set(durations))


def discover_npz_files(duration_hours: int) -> list[Path]:
    if int(duration_hours) == 24:
        npz_dir = ORIGINAL_24H_NPZ_DIR
    else:
        npz_dir = (
            DIFFERENT_DURATION_ROOT
            / f"duration_{int(duration_hours):02d}h"
            / "DEC_result"
            / "all_parement"
        )
    files = sorted(
        npz_dir.glob("event*_all_parement.npz"),
        key=lambda item: natural_event_key(item.name),
    )
    if EVENT_NAMES is not None:
        wanted = {item.lower() for item in EVENT_NAMES}
        files = [item for item in files if infer_event_name_from_npz(item) in wanted]
    return files


def sort_by_event_duration(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    sorted_df = df.copy()
    if "event" in sorted_df.columns:
        sorted_df["_event_order"] = sorted_df["event"].map(natural_event_key)
    else:
        sorted_df["_event_order"] = 0

    sort_cols = ["duration_hours", "_event_order"]
    if "cluster" in sorted_df.columns:
        sort_cols.append("cluster")

    sorted_df = sorted_df.sort_values(sort_cols)
    return sorted_df.drop(columns="_event_order")


def within_group_zscore(values: pd.Series) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce")
    std_value = numeric_values.std(ddof=0)
    if pd.isna(std_value) or std_value <= 0:
        return pd.Series(np.zeros(len(numeric_values)), index=values.index, dtype=float)
    return (numeric_values - numeric_values.mean()) / std_value


def load_event_duration_data(npz_path: Path, duration_hours: int) -> EventDurationData:
    data = np.load(npz_path, allow_pickle=True)
    required_keys = ["x_file", "y_pred_last"]
    missing = [key for key in required_keys if key not in data.files]
    if missing:
        raise KeyError(f"Missing keys {missing} in {npz_path}")

    event_name = infer_event_name_from_npz(npz_path)
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

    return EventDurationData(
        event=event_name,
        duration_hours=int(duration_hours),
        npz_path=npz_path,
        times=times,
        hours_to_end=hours_to_end,
        labels=labels,
        n_clusters=n_clusters,
    )


# =========================================================
# 3. Metric computation
# =========================================================
def fit_best_power_law(t_rel_s: np.ndarray, cumulative_counts: np.ndarray) -> dict[str, float]:
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
        pred = c0 - a_value * phi
        rms = math.sqrt(np.mean((y - pred) ** 2))
        r2 = 1.0 - np.sum((y - pred) ** 2) / ss_tot if ss_tot > 0 else 1.0

        if np.isnan(best["power_rms"]) or rms < best["power_rms"]:
            best = {
                "amr_m": float(m_value),
                "power_rms": float(rms),
                "power_r2": float(r2),
                "power_c0": float(c0),
                "power_a": float(a_value),
            }
    return best


def fit_exponential_growth(t_rel_s: np.ndarray, cumulative_counts: np.ndarray) -> dict[str, float]:
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
    y_log_pred = np.polyval(coeffs, t_rel_s)
    y_pred = np.exp(y_log_pred) - 1.0

    ss_tot = np.sum((cumulative_counts - np.mean(cumulative_counts)) ** 2)
    ss_res = np.sum((cumulative_counts - y_pred) ** 2)
    result["exp_k"] = float(coeffs[0])
    result["exp_intercept"] = float(coeffs[1])
    result["exp_r2"] = float(1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0)
    result["exp_rms"] = float(math.sqrt(np.mean((cumulative_counts - y_pred) ** 2)))
    return result


def compute_terminal_growth_metrics(mask: np.ndarray, dt: int = DT_SECONDS) -> dict[str, float]:
    n_points = len(mask)
    terminal_points = max(int(round(TERMINAL_WINDOW_SECONDS / dt)), 1)
    previous_points = max(int(round(PREVIOUS_WINDOW_SECONDS / dt)), 1)
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
        terminal_flag = int(terminal_rate >= TERMINAL_RATIO_THRESHOLD * previous_rate)

    return {
        "terminal_count_1h": terminal_count,
        "previous_count_5h": previous_count,
        "terminal_rate_1h": terminal_rate,
        "previous_rate_5h": previous_rate,
        "terminal_growth_ratio": terminal_ratio,
        "terminal_growth_delta": terminal_delta,
        "terminal_growth_flag": terminal_flag,
    }


def compute_inverse_rate_metrics(mask: np.ndarray, dt: int = DT_SECONDS) -> dict[str, float]:
    result = {
        "inverse_rate_slope": np.nan,
        "inverse_rate_intercept": np.nan,
        "inverse_rate_r2": np.nan,
        "inverse_rate_valid_bins": 0,
    }
    points_per_bin = max(int(round(INVERSE_RATE_BIN_SECONDS / dt)), 1)
    n_bins = len(mask) // points_per_bin
    if n_bins < 4:
        return result

    trimmed_mask = mask[-n_bins * points_per_bin:]
    bin_counts = trimmed_mask.reshape(n_bins, points_per_bin).sum(axis=1).astype(float)
    bin_rates = bin_counts / INVERSE_RATE_BIN_SECONDS
    valid = bin_rates > 0
    result["inverse_rate_valid_bins"] = int(valid.sum())
    if valid.sum() < 4:
        return result

    time_hours = (np.arange(n_bins, dtype=float) + 0.5) * (INVERSE_RATE_BIN_SECONDS / 3600.0)
    inverse_rates = 1.0 / (bin_rates[valid] + EPS)
    fit_time = time_hours[valid]
    coeffs = np.polyfit(fit_time, inverse_rates, 1)
    predicted = np.polyval(coeffs, fit_time)
    ss_tot = np.sum((inverse_rates - np.mean(inverse_rates)) ** 2)
    ss_res = np.sum((inverse_rates - predicted) ** 2)

    result["inverse_rate_slope"] = float(coeffs[0])
    result["inverse_rate_intercept"] = float(coeffs[1])
    result["inverse_rate_r2"] = float(1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0)
    return result


def compute_cluster_metrics(event_data: EventDurationData, dt: int = DT_SECONDS) -> pd.DataFrame:
    labels = event_data.labels
    unique_labels = sorted(np.unique(labels).astype(int).tolist())
    t_rel_s = np.arange(len(labels), dtype=float) * dt
    total_duration_s = len(labels) * dt
    bg_len = max(int(len(labels) * BACKGROUND_RATIO), 1)

    rows = []
    for cluster_id in unique_labels:
        mask = (labels == cluster_id).astype(int)
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
        onset_idx = np.where(z_curve > Z_FINAL_THRESHOLD)[0]
        onset_time_s = float(t_rel_s[onset_idx[0]]) if len(onset_idx) else np.nan

        linear_rms = np.nan
        linear_r2 = np.nan
        linear_slope = np.nan
        quadratic_a = np.nan
        quadratic_r2 = np.nan
        r2_gain = np.nan

        if total_count > 1:
            linear_coeffs = np.polyfit(t_rel_s, cumulative_counts, 1)
            linear_pred = np.polyval(linear_coeffs, t_rel_s)
            ss_tot = np.sum((cumulative_counts - np.mean(cumulative_counts)) ** 2)
            linear_rms = float(math.sqrt(np.mean((cumulative_counts - linear_pred) ** 2)))
            linear_r2 = float(1.0 - np.sum((cumulative_counts - linear_pred) ** 2) / ss_tot) if ss_tot > 0 else 1.0
            linear_slope = float(linear_coeffs[0])

        if total_count > 2:
            quad_coeffs = np.polyfit(t_rel_s, cumulative_counts, 2)
            quad_pred = np.polyval(quad_coeffs, t_rel_s)
            ss_tot = np.sum((cumulative_counts - np.mean(cumulative_counts)) ** 2)
            quadratic_r2 = float(1.0 - np.sum((cumulative_counts - quad_pred) ** 2) / ss_tot) if ss_tot > 0 else 1.0
            quadratic_a = float(quad_coeffs[0])
            if not np.isnan(linear_r2):
                r2_gain = quadratic_r2 - linear_r2

        exp_fit = fit_exponential_growth(t_rel_s, cumulative_counts) if total_count > 2 else {}
        valid = cumulative_counts > LOG_AMR_MIN_COUNT
        power_fit = (
            fit_best_power_law(t_rel_s[valid], cumulative_counts[valid])
            if valid.sum() >= LOG_AMR_MIN_POINTS
            else {}
        )
        power_rms = power_fit.get("power_rms", np.nan)
        c_value = (
            float(power_rms / (linear_rms + EPS))
            if not np.isnan(power_rms) and not np.isnan(linear_rms)
            else np.nan
        )

        rows.append(
            {
                "event": event_data.event,
                "duration_hours": int(event_data.duration_hours),
                "cluster": int(cluster_id),
                "sample_count": total_count,
                "n_clusters": int(event_data.n_clusters),
                "npz_path": str(event_data.npz_path),
                "background_rate_hz": lambda_bg,
                "acc": acc,
                "car": car,
                "z_final": z_final,
                "early_slope": early_slope,
                "late_slope": late_slope,
                "rate_acceleration": rate_acceleration,
                "onset_time_s": onset_time_s,
                "terminal_count_1h": terminal_metrics["terminal_count_1h"],
                "previous_count_5h": terminal_metrics["previous_count_5h"],
                "terminal_rate_1h": terminal_metrics["terminal_rate_1h"],
                "previous_rate_5h": terminal_metrics["previous_rate_5h"],
                "terminal_growth_ratio": terminal_metrics["terminal_growth_ratio"],
                "terminal_growth_delta": terminal_metrics["terminal_growth_delta"],
                "terminal_growth_flag": terminal_metrics["terminal_growth_flag"],
                "inverse_rate_slope": inverse_rate_metrics["inverse_rate_slope"],
                "inverse_rate_r2": inverse_rate_metrics["inverse_rate_r2"],
                "inverse_rate_valid_bins": inverse_rate_metrics["inverse_rate_valid_bins"],
                "linear_slope": linear_slope,
                "linear_r2": linear_r2,
                "linear_rms": linear_rms,
                "quadratic_a": quadratic_a,
                "quadratic_r2": quadratic_r2,
                "r2_gain_quadratic_minus_linear": r2_gain,
                "exp_k": exp_fit.get("exp_k", np.nan),
                "exp_r2": exp_fit.get("exp_r2", np.nan),
                "exp_rms": exp_fit.get("exp_rms", np.nan),
                "amr_m": power_fit.get("amr_m", np.nan),
                "c_value": c_value,
                "power_r2": power_fit.get("power_r2", np.nan),
                "power_rms": power_rms,
            }
        )

    return pd.DataFrame(rows)


def add_pcs_score_columns(metrics_df: pd.DataFrame) -> pd.DataFrame:
    reference_pcs = load_reference_pcs_module()
    scored_parts = []
    for duration_hours, duration_df in metrics_df.groupby("duration_hours", sort=True):
        scored_df = reference_pcs.add_pcs_score_columns(duration_df.copy())
        if "duration_hours" not in scored_df.columns:
            scored_df.insert(0, "duration_hours", int(duration_hours))
        scored_parts.append(scored_df)

    return pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()


def compute_cluster_metrics_with_reference(event_data: EventDurationData) -> pd.DataFrame:
    """Compute metrics using exactly the original 09_export implementation."""
    reference_pcs = load_reference_pcs_module()
    time_points = np.arange(len(event_data.labels), dtype=float)
    unique_labels = np.unique(event_data.labels)
    metrics_df = reference_pcs.compute_cluster_metrics(
        time_points=time_points,
        time_labels=event_data.labels,
        unique_labels=unique_labels,
        dt=reference_pcs.DT_SECONDS,
        bg_ratio=reference_pcs.BACKGROUND_RATIO,
    )
    metrics_df.insert(0, "event", event_data.event)
    metrics_df.insert(1, "duration_hours", int(event_data.duration_hours))
    metrics_df.insert(3, "n_clusters", int(event_data.n_clusters))
    metrics_df.insert(4, "npz_path", str(event_data.npz_path))
    return metrics_df


def select_pcs_by_duration(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply the original PCS selector independently to each duration subset.

    The reference selector groups by event internally, so passing one duration
    at a time preserves the original algorithm while avoiding cross-duration
    mixing in the within-event z-score.
    """
    reference_pcs = load_reference_pcs_module()
    selected_parts = []
    candidate_parts = []
    missing_parts = []

    for duration_hours, duration_df in metrics_df.groupby("duration_hours", sort=True):
        try:
            selected_df, candidates_df, missing_df = reference_pcs.select_precursor_clusters_by_pcs(
                duration_df.copy(),
                z_threshold=reference_pcs.Z_FINAL_THRESHOLD,
                c_threshold=reference_pcs.C_VALUE_THRESHOLD,
            )
        except KeyError as exc:
            if "pcs_rank_within_event" not in str(exc):
                raise
            selected_df, candidates_df, missing_df = select_pcs_reference_compatible(
                duration_df.copy(),
                reference_pcs=reference_pcs,
            )

        for df in [selected_df, candidates_df, missing_df]:
            if not df.empty and "duration_hours" not in df.columns:
                df.insert(0, "duration_hours", int(duration_hours))

        selected_parts.append(selected_df)
        candidate_parts.append(candidates_df)
        missing_parts.append(missing_df)

    selected_all = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    candidates_all = pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame()
    missing_all = pd.concat(missing_parts, ignore_index=True) if missing_parts else pd.DataFrame()

    return (
        sort_by_event_duration(selected_all),
        sort_by_event_duration(candidates_all),
        sort_by_event_duration(missing_all),
    )


def select_pcs_reference_compatible(
    duration_df: pd.DataFrame,
    reference_pcs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Same PCS algorithm as 09_export, with one guard for duration subsets where
    no event has a strict PCS candidate.
    """
    selected_rows = []
    candidate_rows = []
    missing_rows = []
    z_threshold = reference_pcs.Z_FINAL_THRESHOLD
    c_threshold = reference_pcs.C_VALUE_THRESHOLD
    selection_rule = (
        f"z_final>{z_threshold:g}, c_value<{c_threshold:g}, terminal_growth_flag=1; "
        "max PCS"
    )

    for event_name, event_df in duration_df.groupby("event", sort=True):
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

        valid_df["z_z_final"] = reference_pcs.within_event_zscore(valid_df["z_final"])
        valid_df["z_rate_acceleration"] = reference_pcs.within_event_zscore(valid_df["rate_acceleration"])
        valid_df["z_c_value"] = reference_pcs.within_event_zscore(valid_df["c_value"])
        valid_df["z_terminal_growth_ratio"] = reference_pcs.within_event_zscore(valid_df["terminal_growth_ratio"])
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
    for df in [selected_df, candidates_df, missing_df]:
        for column in front_cols:
            if column not in df.columns:
                df[column] = np.nan

    selected_df = selected_df[front_cols + [col for col in selected_df.columns if col not in front_cols]]
    candidates_df = candidates_df[front_cols + [col for col in candidates_df.columns if col not in front_cols]]
    missing_df = missing_df[front_cols + [col for col in missing_df.columns if col not in front_cols]]
    return selected_df, candidates_df, missing_df


# =========================================================
# 4. Plotting
# =========================================================
def plot_pcs_curves_for_duration(
    duration_hours: int,
    selected_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    event_cache: dict[tuple[int, str], EventDurationData],
) -> Path | None:
    duration_selected = selected_df[
        (selected_df["duration_hours"] == duration_hours)
        & (selected_df["selection_status"] == "selected")
        & (selected_df["cluster"] >= 0)
    ].copy()

    if duration_selected.empty:
        print(f"[WARN] duration {duration_hours} h: no PCS-selected events for curve plot.")
        return None

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    cmap = plt.get_cmap("tab20", max(len(duration_selected), 1))

    for idx, row in enumerate(duration_selected.itertuples(index=False)):
        event_data = event_cache[(int(row.duration_hours), str(row.event))]
        cluster_id = int(row.cluster)
        mask = event_data.labels == cluster_id
        y = np.cumsum(mask).astype(float)
        label = f"{row.event}-C{cluster_id} (PCS={row.pcs:.2f}, n={int(mask.sum())})"
        ax.plot(
            event_data.hours_to_end,
            y,
            linewidth=1.6,
            alpha=0.85,
            color=cmap(idx % cmap.N),
            label=label,
        )

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    if PLOT_X_AS_DURATION:
        ax.set_xlim(-duration_hours, 0.05)
    ax.set_xlabel("Time before event end (h)")
    ax.set_ylabel("Cumulative count of PCS-selected cluster")
    total_event_count = (
        selected_df.loc[selected_df["duration_hours"] == duration_hours, "event"].nunique()
        + missing_df.loc[missing_df["duration_hours"] == duration_hours, "event"].nunique()
    )
    ax.set_title(
        f"PCS-selected precursor cumulative curves across events, duration={duration_hours} h\n"
        f"selected events: {len(duration_selected)} / {total_event_count}",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=True)

    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CURVE_DIR / f"duration_{duration_hours:02d}h_pcs_selected_cumulative_curves.png"
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def build_selection_summary(selected_df: pd.DataFrame, missing_df: pd.DataFrame) -> pd.DataFrame:
    selected_count = (
        selected_df.groupby("duration_hours")["event"]
        .nunique()
        .reset_index(name="selected_events")
        if not selected_df.empty
        else pd.DataFrame(columns=["duration_hours", "selected_events"])
    )
    missing_count = (
        missing_df.groupby("duration_hours")["event"]
        .nunique()
        .reset_index(name="not_selected_events")
        if not missing_df.empty
        else pd.DataFrame(columns=["duration_hours", "not_selected_events"])
    )
    summary = selected_count.merge(missing_count, on="duration_hours", how="outer").fillna(0)
    summary["selected_events"] = summary["selected_events"].astype(int)
    summary["not_selected_events"] = summary["not_selected_events"].astype(int)
    summary["total_events"] = summary["selected_events"] + summary["not_selected_events"]
    summary["selection_rate"] = summary["selected_events"] / summary["total_events"].replace(0, np.nan)
    return summary.sort_values("duration_hours")


def plot_selection_count_summary(selected_df: pd.DataFrame, missing_df: pd.DataFrame) -> Path:
    summary = build_selection_summary(selected_df, missing_df)

    fig, ax1 = plt.subplots(figsize=(10, 5), constrained_layout=True)
    x = np.arange(len(summary))
    ax1.bar(x, summary["selected_events"], color="#4C78A8", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(v)}h" for v in summary["duration_hours"]])
    ax1.set_ylabel("Number of PCS-selected events")
    ax1.set_xlabel("Duration")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, summary["selection_rate"], color="#E45756", marker="o", linewidth=2)
    ax2.set_ylabel("Selection rate")
    ax2.set_ylim(0, 1.05)

    ax1.set_title("PCS strict-rule selection count by duration", fontsize=14, fontweight="bold")

    BOXPLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BOXPLOT_DIR / "pcs_selection_count_by_duration.png"
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_metric_boxplots(metrics_df: pd.DataFrame, selected_df: pd.DataFrame) -> list[Path]:
    plot_df = metrics_df.copy()
    selected_keys = set(
        zip(
            selected_df.loc[selected_df["selection_status"] == "selected", "duration_hours"].astype(int),
            selected_df.loc[selected_df["selection_status"] == "selected", "event"].astype(str),
            selected_df.loc[selected_df["selection_status"] == "selected", "cluster"].astype(int),
        )
    )
    plot_df["group"] = [
        "PCS selected" if (int(row.duration_hours), str(row.event), int(row.cluster)) in selected_keys else "Other clusters"
        for row in plot_df.itertuples(index=False)
    ]

    metric_specs = [
        ("pcs", "PCS"),
        ("z_final", "z_final"),
        ("rate_acceleration", "rate acceleration"),
        ("c_value", "c-value"),
        ("terminal_growth_ratio", "terminal growth ratio"),
    ]

    BOXPLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for metric_name, metric_label in metric_specs:
        metric_df = plot_df.dropna(subset=[metric_name]).copy()
        if metric_df.empty:
            continue

        durations = sorted(metric_df["duration_hours"].unique().astype(int).tolist())
        fig, ax = plt.subplots(figsize=(max(10, len(durations) * 1.4), 6), constrained_layout=True)

        positions = []
        data = []
        colors = []
        labels = []
        for i, duration in enumerate(durations):
            base = i * 3
            selected_values = metric_df[
                (metric_df["duration_hours"] == duration)
                & (metric_df["group"] == "PCS selected")
            ][metric_name].to_numpy(dtype=float)
            other_values = metric_df[
                (metric_df["duration_hours"] == duration)
                & (metric_df["group"] == "Other clusters")
            ][metric_name].to_numpy(dtype=float)

            for offset, values, color, label in [
                (0, selected_values, "#F58518", "PCS selected"),
                (1, other_values, "#72B7B2", "Other clusters"),
            ]:
                if len(values) == 0:
                    values = np.array([np.nan])
                positions.append(base + offset)
                data.append(values)
                colors.append(color)
                labels.append(label)

        box = ax.boxplot(
            data,
            positions=positions,
            widths=0.7,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        tick_positions = [i * 3 + 0.5 for i in range(len(durations))]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{duration}h" for duration in durations])
        ax.set_xlabel("Duration")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label}: PCS-selected clusters vs other clusters", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)

        handles = [
            plt.Line2D([0], [0], color="#F58518", linewidth=8, alpha=0.75, label="PCS selected"),
            plt.Line2D([0], [0], color="#72B7B2", linewidth=8, alpha=0.75, label="Other clusters"),
        ]
        ax.legend(handles=handles, loc="best")

        output_path = BOXPLOT_DIR / f"{metric_name}_boxplot_by_duration.png"
        fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


# =========================================================
# 5. Export
# =========================================================
def export_excel(
    metrics_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    missing_df: pd.DataFrame,
) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary = build_selection_summary(selected_df, missing_df)

    output_path = OUTPUT_ROOT / "pcs_selection_by_duration.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        sort_by_event_duration(metrics_df).to_excel(writer, sheet_name="all_cluster_metrics", index=False)
        sort_by_event_duration(selected_df).to_excel(writer, sheet_name="pcs_selected", index=False)
        sort_by_event_duration(candidates_df).to_excel(writer, sheet_name="pcs_candidates", index=False)
        sort_by_event_duration(missing_df).to_excel(writer, sheet_name="pcs_not_selected", index=False)
        summary.to_excel(writer, sheet_name="selection_summary", index=False)

    metrics_df.to_csv(OUTPUT_ROOT / "all_cluster_metrics.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(OUTPUT_ROOT / "pcs_selected_by_duration.csv", index=False, encoding="utf-8-sig")
    return output_path


# =========================================================
# 6. Post-hoc analysis figures
# =========================================================
def load_exported_pcs_tables(
    excel_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if excel_path is None:
        excel_path = OUTPUT_ROOT / "pcs_selection_by_duration.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(f"PCS result workbook not found: {excel_path}")

    selected_df = pd.read_excel(excel_path, sheet_name="pcs_selected")
    missing_df = pd.read_excel(excel_path, sheet_name="pcs_not_selected")
    summary_df = pd.read_excel(excel_path, sheet_name="selection_summary")
    return selected_df, missing_df, summary_df


def selected_event_sets_by_duration(selected_df: pd.DataFrame) -> dict[int, set[str]]:
    selected_only = selected_df[
        (selected_df["selection_status"] == "selected")
        & (pd.to_numeric(selected_df["cluster"], errors="coerce") >= 0)
    ].copy()
    return {
        int(duration): set(group["event"].astype(str))
        for duration, group in selected_only.groupby("duration_hours")
    }


def plot_duration_selection_overlap(
    selected_df: pd.DataFrame,
    missing_df: pd.DataFrame,
) -> Path:
    ANALYSIS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = build_selection_summary(selected_df, missing_df)
    event_sets = selected_event_sets_by_duration(selected_df)
    baseline_events = event_sets.get(24, set())

    overlap_rows = []
    for duration in summary_df["duration_hours"].astype(int):
        current_events = event_sets.get(duration, set())
        overlap_count = len(current_events & baseline_events)
        overlap_rate = overlap_count / len(baseline_events) if baseline_events else np.nan
        overlap_rows.append(
            {
                "duration_hours": duration,
                "overlap_with_24h": overlap_count,
                "overlap_rate_with_24h": overlap_rate,
                "only_current": len(current_events - baseline_events),
                "missed_24h": len(baseline_events - current_events),
            }
        )
    overlap_df = pd.DataFrame(overlap_rows)
    plot_df = summary_df.merge(overlap_df, on="duration_hours", how="left")

    fig, ax1 = plt.subplots(figsize=(11, 6), constrained_layout=True)
    x = np.arange(len(plot_df))
    duration_labels = [f"{int(v)}h" for v in plot_df["duration_hours"]]

    ax1.bar(
        x - 0.18,
        plot_df["selection_rate"],
        width=0.36,
        color="#4C78A8",
        alpha=0.85,
        label="PCS selection rate",
    )
    ax1.bar(
        x + 0.18,
        plot_df["overlap_rate_with_24h"],
        width=0.36,
        color="#F58518",
        alpha=0.85,
        label="Overlap with 24h selected events",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(duration_labels)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Duration")
    ax1.set_ylabel("Rate")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper right")

    for idx, row in plot_df.iterrows():
        ax1.text(
            idx - 0.18,
            row["selection_rate"] + 0.025,
            f"{int(row['selected_events'])}/{int(row['total_events'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        if not pd.isna(row["overlap_rate_with_24h"]):
            ax1.text(
                idx + 0.18,
                row["overlap_rate_with_24h"] + 0.025,
                f"{int(row['overlap_with_24h'])}/{len(baseline_events)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax1.set_title(
        "PCS duration sensitivity and consistency with 24h baseline",
        fontsize=14,
        fontweight="bold",
    )

    output_path = ANALYSIS_FIGURE_DIR / "duration_selection_rate_and_24h_overlap.png"
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    plot_df.to_csv(
        ANALYSIS_FIGURE_DIR / "duration_selection_overlap_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return output_path


def build_event_duration_matrix(selected_df: pd.DataFrame, missing_df: pd.DataFrame) -> pd.DataFrame:
    all_events = sorted(
        set(selected_df["event"].dropna().astype(str)) | set(missing_df["event"].dropna().astype(str)),
        key=natural_event_key,
    )
    all_durations = sorted(
        set(selected_df["duration_hours"].dropna().astype(int))
        | set(missing_df["duration_hours"].dropna().astype(int))
    )
    matrix = pd.DataFrame(0, index=all_events, columns=all_durations, dtype=int)
    for row in selected_df.itertuples(index=False):
        if getattr(row, "selection_status", "") == "selected" and int(getattr(row, "cluster")) >= 0:
            matrix.loc[str(row.event), int(row.duration_hours)] = 1
    return matrix


def plot_event_duration_heatmap(
    selected_df: pd.DataFrame,
    missing_df: pd.DataFrame,
) -> tuple[Path, Path]:
    ANALYSIS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    matrix = build_event_duration_matrix(selected_df, missing_df)
    stability = matrix.sum(axis=1).sort_values(ascending=False)
    ordered_matrix = matrix.loc[stability.index]

    fig, ax = plt.subplots(figsize=(10, 13), constrained_layout=True)
    im = ax.imshow(ordered_matrix.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(ordered_matrix.columns)))
    ax.set_xticklabels([f"{int(v)}h" for v in ordered_matrix.columns])
    ax.set_yticks(np.arange(len(ordered_matrix.index)))
    ax.set_yticklabels(ordered_matrix.index, fontsize=8)
    ax.set_xlabel("Duration")
    ax.set_ylabel("Event, sorted by PCS stability")
    ax.set_title(
        "PCS selection stability across duration windows",
        fontsize=14,
        fontweight="bold",
    )

    for y in range(ordered_matrix.shape[0]):
        for x in range(ordered_matrix.shape[1]):
            value = ordered_matrix.iat[y, x]
            ax.text(x, y, "1" if value else "", ha="center", va="center", fontsize=7, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["not selected", "selected"])

    heatmap_path = ANALYSIS_FIGURE_DIR / "event_duration_pcs_selection_heatmap.png"
    fig.savefig(heatmap_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 11), constrained_layout=True)
    y_pos = np.arange(len(stability))
    colors = ["#2F855A" if value >= 7 else "#D69E2E" if value >= 5 else "#C53030" for value in stability]
    ax.barh(y_pos, stability.to_numpy(), color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stability.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of durations selected by PCS")
    ax.set_ylabel("Event")
    ax.set_xlim(0, len(matrix.columns) + 0.5)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Event-level PCS stability ranking", fontsize=14, fontweight="bold")
    for y, value in enumerate(stability):
        ax.text(value + 0.1, y, f"{int(value)}/{len(matrix.columns)}", va="center", fontsize=8)

    ranking_path = ANALYSIS_FIGURE_DIR / "event_pcs_stability_ranking.png"
    fig.savefig(ranking_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stability_df = stability.reset_index()
    stability_df.columns = ["event", "selected_duration_count"]
    stability_df["selected_duration_ratio"] = stability_df["selected_duration_count"] / len(matrix.columns)
    stability_df.to_csv(
        ANALYSIS_FIGURE_DIR / "event_pcs_stability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return heatmap_path, ranking_path


def plot_focus_duration_metric_summary(selected_df: pd.DataFrame) -> Path:
    ANALYSIS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    selected_only = selected_df[
        (selected_df["selection_status"] == "selected")
        & (pd.to_numeric(selected_df["cluster"], errors="coerce") >= 0)
    ].copy()
    metrics = ["pcs", "z_final", "rate_acceleration", "c_value", "terminal_growth_ratio"]
    available_focus = [duration for duration in FOCUS_DURATIONS if duration in set(selected_only["duration_hours"].astype(int))]
    if not available_focus:
        raise RuntimeError("No focus durations found in PCS selected table.")

    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(10, 3.0 * len(metrics)),
        constrained_layout=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        data = []
        labels = []
        for duration in available_focus:
            values = selected_only.loc[
                selected_only["duration_hours"].astype(int) == duration,
                metric,
            ].dropna().to_numpy(dtype=float)
            data.append(values)
            labels.append(f"{duration}h\nn={len(values)}")

        box = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
        )
        for patch, color in zip(box["boxes"], ["#54A24B", "#E45756", "#4C78A8"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_title(
        "PCS-selected metric distributions in recommended windows",
        fontsize=14,
        fontweight="bold",
    )
    axes[-1].set_xlabel("Duration")

    output_path = ANALYSIS_FIGURE_DIR / "focus_duration_metric_boxplots.png"
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_stable_event_curves_for_focus_durations(
    selected_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    top_n: int = 10,
) -> Path:
    ANALYSIS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    matrix = build_event_duration_matrix(selected_df, missing_df)
    stability = matrix.sum(axis=1).sort_values(ascending=False)
    stable_events = stability.head(top_n).index.tolist()

    selected_df = selected_df.copy()
    selected_df["duration_numeric"] = pd.to_numeric(selected_df["duration_hours"], errors="coerce")
    selected_df["cluster_numeric"] = pd.to_numeric(selected_df["cluster"], errors="coerce")
    selected_only = selected_df[
        (selected_df["selection_status"] == "selected")
        & selected_df["duration_numeric"].notna()
        & selected_df["cluster_numeric"].notna()
        & (selected_df["cluster_numeric"] >= 0)
        & (selected_df["event"].isin(stable_events))
        & (selected_df["duration_numeric"].isin(FOCUS_DURATIONS))
    ].copy()

    available_focus = sorted(selected_only["duration_numeric"].astype(int).unique().tolist())
    if not available_focus:
        raise RuntimeError("No selected stable events found for focus duration curve plot.")

    fig, axes = plt.subplots(
        len(available_focus),
        1,
        figsize=(13, 4.2 * len(available_focus)),
        sharex=False,
        constrained_layout=True,
    )
    if len(available_focus) == 1:
        axes = [axes]

    event_color_map = {
        event: plt.get_cmap("tab20", max(len(stable_events), 1))(idx)
        for idx, event in enumerate(stable_events)
    }

    for ax, duration in zip(axes, available_focus):
        duration_df = selected_only[selected_only["duration_numeric"].astype(int) == duration].copy()
        for row in duration_df.itertuples(index=False):
            event_data = load_event_duration_data(Path(row.npz_path), duration)
            cluster_id = int(row.cluster_numeric)
            mask = event_data.labels == cluster_id
            ax.plot(
                event_data.hours_to_end,
                np.cumsum(mask),
                color=event_color_map[str(row.event)],
                linewidth=1.8,
                alpha=0.85,
                label=f"{row.event}-C{cluster_id}",
            )

        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
        ax.set_xlim(-duration, 0.05)
        ax.set_ylabel("Cumulative count")
        ax.set_title(f"{duration}h, top stable PCS-selected events", loc="left", fontsize=12)
        ax.grid(alpha=0.25)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=True)

    axes[-1].set_xlabel("Time before event end (h)")
    fig.suptitle(
        "Cumulative curves of stable PCS events in focus duration windows",
        fontsize=14,
        fontweight="bold",
    )

    output_path = ANALYSIS_FIGURE_DIR / "stable_events_focus_duration_cumulative_curves.png"
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_posthoc_analysis_from_excel(excel_path: Path | None = None) -> list[Path]:
    selected_df, missing_df, _ = load_exported_pcs_tables(excel_path)
    output_paths: list[Path] = []
    output_paths.append(plot_duration_selection_overlap(selected_df, missing_df))
    output_paths.extend(plot_event_duration_heatmap(selected_df, missing_df))
    output_paths.append(plot_focus_duration_metric_summary(selected_df))
    output_paths.append(plot_stable_event_curves_for_focus_durations(selected_df, missing_df))
    return output_paths


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    BOXPLOT_DIR.mkdir(parents=True, exist_ok=True)

    durations = discover_duration_hours()
    if not durations:
        raise FileNotFoundError(f"No duration folders found under {DIFFERENT_DURATION_ROOT}")

    print("Durations:", durations)
    print("Input root:", DIFFERENT_DURATION_ROOT)
    print("Output root:", OUTPUT_ROOT)

    metric_frames = []
    event_cache: dict[tuple[int, str], EventDurationData] = {}

    for duration_hours in durations:
        npz_files = discover_npz_files(duration_hours)
        if not npz_files:
            print(f"[WARN] duration={duration_hours}h: no npz files found.")
            continue

        for npz_path in tqdm(npz_files, desc=f"Computing PCS metrics {duration_hours}h"):
            event_data = load_event_duration_data(npz_path, duration_hours)
            event_cache[(duration_hours, event_data.event)] = event_data
            metric_frames.append(compute_cluster_metrics_with_reference(event_data))

    if not metric_frames:
        raise RuntimeError("No metrics were computed.")

    metrics_df = pd.concat(metric_frames, ignore_index=True)
    metrics_df = add_pcs_score_columns(metrics_df)
    selected_df, candidates_df, missing_df = select_pcs_by_duration(metrics_df)

    excel_path = export_excel(metrics_df, selected_df, candidates_df, missing_df)
    count_summary_path = plot_selection_count_summary(selected_df, missing_df)
    boxplot_paths = plot_metric_boxplots(metrics_df, selected_df)

    curve_paths = []
    for duration_hours in durations:
        curve_path = plot_pcs_curves_for_duration(duration_hours, selected_df, missing_df, event_cache)
        if curve_path is not None:
            curve_paths.append(curve_path)
    analysis_paths = plot_posthoc_analysis_from_excel(excel_path)

    print("\nDone.")
    print("Excel:", excel_path)
    print("Selection count figure:", count_summary_path)
    print("Boxplots:", len(boxplot_paths))
    print("Duration curve figures:", len(curve_paths))
    print("Post-hoc analysis figures:", len(analysis_paths))


if __name__ == "__main__":
    main()
