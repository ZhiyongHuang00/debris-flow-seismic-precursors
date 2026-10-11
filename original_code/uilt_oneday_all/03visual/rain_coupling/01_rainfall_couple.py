# -*- coding: utf-8 -*-
"""
Post-process and analyse downloaded GPM IMERG rainfall data for Illgraben.

This script uses the monthly CSV files produced by:
    data/降雨数据下载/01_download_process_gpm_imerg_2017_2023.py

Main outputs:
    1. continuous 30 min rainfall time series with missing-data flags
    2. rolling antecedent rainfall metrics
    3. monthly / yearly rainfall statistics
    4. event-level rainfall metrics before debris-flow initiation
    5. figures for rainfall quality check and event coupling

The script does not re-download data and does not modify the raw HDF5 files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# 1. Paths and configuration
# =========================================================
PROJECT_ROOT = Path(r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben")
SCRIPT_DIR = Path(__file__).resolve().parent

PRIMARY_GPM_DIR = Path(r"H:\Illgraben\GPM_IMERG_Illgraben")
FALLBACK_GPM_DIR = PROJECT_ROOT / r"data\降雨数据下载\GPM_IMERG_Illgraben"
EVENT_NPZ_DIR = PROJECT_ROOT / r"uilt_oneday_all\02clustering\result\DEC_result\all_parement"

OUTPUT_DIR = SCRIPT_DIR / "gpm_rainfall_analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"

TIME_FREQ = "30min"
TIME_ZONE = "UTC"

ROLLING_WINDOWS_STEPS = {
    "cum_1h_mm": 2,
    "cum_2h_mm": 4,
    "cum_3h_mm": 6,
    "cum_4h_mm": 8,
    "cum_6h_mm": 12,
    "cum_8h_mm": 16,
    "cum_12h_mm": 24,
    "cum_24h_mm": 48,
    "cum_48h_mm": 96,
    "cum_72h_mm": 144,
    "cum_7d_mm": 336,
}

EVENT_WINDOWS_HOURS = [1, 2, 3, 4, 6, 8, 12, 24, 48, 72, 168]
FIG_DPI = 300

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 11,
        "axes.unicode_minus": False,
    }
)
plt.ioff()


@dataclass
class RainfallData:
    raw: pd.DataFrame
    continuous: pd.DataFrame
    monthly_summary: pd.DataFrame
    yearly_summary: pd.DataFrame
    completeness_summary: pd.DataFrame
    event_summary: pd.DataFrame


# =========================================================
# 2. Basic helpers
# =========================================================
def resolve_gpm_dir() -> Path:
    if PRIMARY_GPM_DIR.exists():
        return PRIMARY_GPM_DIR
    if FALLBACK_GPM_DIR.exists():
        return FALLBACK_GPM_DIR
    raise FileNotFoundError(
        f"Cannot find GPM output directory. Checked: {PRIMARY_GPM_DIR} and {FALLBACK_GPM_DIR}"
    )


def natural_event_key(value: str) -> int:
    match = re.search(r"event(\d+)", str(value), flags=re.IGNORECASE)
    return int(match.group(1)) if match else 10**9


def parse_time_from_wave_file(wave_file: object) -> pd.Timestamp | pd.NaT:
    match = re.search(r"(\d{8}T\d{6})", str(wave_file))
    if match is None:
        return pd.NaT
    return pd.to_datetime(match.group(1), format="%Y%m%dT%H%M%S", utc=True)


def infer_event_name_from_npz(path: Path) -> str:
    match = re.search(r"event\d+", path.name, flags=re.IGNORECASE)
    if match:
        return match.group(0).lower()
    return path.stem.replace("_all_parement", "").lower()


def find_monthly_gpm_csv_files(gpm_dir: Path) -> list[Path]:
    files = []
    for path in gpm_dir.glob("GPM_IMERG_Illgraben_??????.csv"):
        if re.search(r"GPM_IMERG_Illgraben_\d{6}\.csv$", path.name):
            files.append(path)
    return sorted(files)


# =========================================================
# 3. Rainfall loading and processing
# =========================================================
def load_monthly_gpm_csvs(gpm_dir: Path) -> pd.DataFrame:
    files = find_monthly_gpm_csv_files(gpm_dir)
    if not files:
        raise FileNotFoundError(f"No monthly GPM CSV files found in {gpm_dir}")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        required = {"time_utc", "rain_rate_mm_h", "rain_mm_30min"}
        missing = required - set(df.columns)
        if missing:
            print(f"[WARN] skip {path.name}: missing columns {sorted(missing)}")
            continue
        df["source_file_month"] = path.name
        frames.append(df)

    if not frames:
        raise RuntimeError("No valid monthly GPM CSV files were loaded.")

    rainfall = pd.concat(frames, ignore_index=True)
    rainfall["time_utc"] = pd.to_datetime(rainfall["time_utc"], utc=True, errors="coerce")
    rainfall = rainfall.dropna(subset=["time_utc"])
    rainfall = rainfall.drop_duplicates(subset=["time_utc"], keep="last")
    rainfall = rainfall.sort_values("time_utc")

    for col in ["rain_rate_mm_h", "rain_mm_30min", "grid_lon", "grid_lat"]:
        if col in rainfall.columns:
            rainfall[col] = pd.to_numeric(rainfall[col], errors="coerce")

    rainfall["rain_rate_mm_h"] = rainfall["rain_rate_mm_h"].clip(lower=0)
    rainfall["rain_mm_30min"] = rainfall["rain_mm_30min"].clip(lower=0)
    return rainfall


def build_continuous_30min_series(rainfall: pd.DataFrame) -> pd.DataFrame:
    df = rainfall.copy()
    df = df.set_index("time_utc").sort_index()

    full_index = pd.date_range(
        start=df.index.min().floor(TIME_FREQ),
        end=df.index.max().ceil(TIME_FREQ),
        freq=TIME_FREQ,
        tz=TIME_ZONE,
    )
    continuous = df.reindex(full_index)
    continuous.index.name = "time_utc"

    continuous["is_original_record"] = continuous["rain_mm_30min"].notna()
    continuous["is_missing_record"] = ~continuous["is_original_record"]
    continuous["rain_rate_mm_h"] = continuous["rain_rate_mm_h"].fillna(0.0)
    continuous["rain_mm_30min"] = continuous["rain_mm_30min"].fillna(0.0)
    continuous["grid_lon"] = continuous["grid_lon"].ffill().bfill()
    continuous["grid_lat"] = continuous["grid_lat"].ffill().bfill()

    for col, steps in ROLLING_WINDOWS_STEPS.items():
        continuous[col] = continuous["rain_mm_30min"].rolling(steps, min_periods=1).sum()

    continuous["max_30min_mm_last_24h"] = (
        continuous["rain_mm_30min"].rolling(48, min_periods=1).max()
    )
    continuous["max_1h_mm_last_24h"] = (
        continuous["rain_mm_30min"]
        .rolling(2, min_periods=1)
        .sum()
        .rolling(48, min_periods=1)
        .max()
    )
    continuous["max_3h_mm_last_24h"] = (
        continuous["rain_mm_30min"]
        .rolling(6, min_periods=1)
        .sum()
        .rolling(48, min_periods=1)
        .max()
    )
    continuous["max_6h_mm_last_24h"] = (
        continuous["rain_mm_30min"]
        .rolling(12, min_periods=1)
        .sum()
        .rolling(48, min_periods=1)
        .max()
    )

    continuous["date"] = continuous.index.date
    continuous["year"] = continuous.index.year
    continuous["month"] = continuous.index.month
    continuous["year_month"] = continuous.index.tz_localize(None).to_period("M").astype(str)
    return continuous.reset_index()


def summarize_completeness(continuous: pd.DataFrame) -> pd.DataFrame:
    total = len(continuous)
    missing = int(continuous["is_missing_record"].sum())
    summary_rows = [
        {
            "scope": "all",
            "start_time_utc": continuous["time_utc"].min(),
            "end_time_utc": continuous["time_utc"].max(),
            "expected_30min_records": total,
            "missing_30min_records": missing,
            "missing_fraction": missing / total if total else np.nan,
            "rain_total_mm": continuous["rain_mm_30min"].sum(),
        }
    ]

    by_month = (
        continuous.groupby("year_month")
        .agg(
            start_time_utc=("time_utc", "min"),
            end_time_utc=("time_utc", "max"),
            expected_30min_records=("time_utc", "size"),
            missing_30min_records=("is_missing_record", "sum"),
            rain_total_mm=("rain_mm_30min", "sum"),
        )
        .reset_index()
    )
    by_month["missing_fraction"] = (
        by_month["missing_30min_records"] / by_month["expected_30min_records"]
    )
    by_month = by_month.rename(columns={"year_month": "scope"})
    return pd.concat([pd.DataFrame(summary_rows), by_month], ignore_index=True)


def summarize_monthly_yearly(continuous: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = (
        continuous.groupby("year_month")
        .agg(
            record_count=("time_utc", "size"),
            total_rain_mm=("rain_mm_30min", "sum"),
            wet_30min_count=("rain_mm_30min", lambda x: int((x > 0).sum())),
            max_30min_mm=("rain_mm_30min", "max"),
            max_1h_mm=("rain_mm_30min", lambda x: x.rolling(2, min_periods=1).sum().max()),
            max_3h_mm=("rain_mm_30min", lambda x: x.rolling(6, min_periods=1).sum().max()),
            max_6h_mm=("rain_mm_30min", lambda x: x.rolling(12, min_periods=1).sum().max()),
            max_24h_mm=("rain_mm_30min", lambda x: x.rolling(48, min_periods=1).sum().max()),
            missing_30min_count=("is_missing_record", "sum"),
        )
        .reset_index()
    )
    monthly["wet_fraction"] = monthly["wet_30min_count"] / monthly["record_count"].replace(0, np.nan)

    yearly = (
        continuous.groupby("year")
        .agg(
            record_count=("time_utc", "size"),
            total_rain_mm=("rain_mm_30min", "sum"),
            wet_30min_count=("rain_mm_30min", lambda x: int((x > 0).sum())),
            max_30min_mm=("rain_mm_30min", "max"),
            max_1h_mm=("rain_mm_30min", lambda x: x.rolling(2, min_periods=1).sum().max()),
            max_3h_mm=("rain_mm_30min", lambda x: x.rolling(6, min_periods=1).sum().max()),
            max_6h_mm=("rain_mm_30min", lambda x: x.rolling(12, min_periods=1).sum().max()),
            max_24h_mm=("rain_mm_30min", lambda x: x.rolling(48, min_periods=1).sum().max()),
            missing_30min_count=("is_missing_record", "sum"),
        )
        .reset_index()
    )
    yearly["wet_fraction"] = yearly["wet_30min_count"] / yearly["record_count"].replace(0, np.nan)
    return monthly, yearly


# =========================================================
# 4. Event rainfall coupling
# =========================================================
def build_event_catalog_from_npz(npz_dir: Path = EVENT_NPZ_DIR) -> pd.DataFrame:
    if not npz_dir.exists():
        print(f"[WARN] event npz directory not found: {npz_dir}")
        return pd.DataFrame(columns=["event", "event_time_utc", "npz_path", "n_samples"])

    records = []
    for npz_path in sorted(npz_dir.glob("event*_all_parement.npz"), key=lambda x: natural_event_key(x.name)):
        try:
            data = np.load(npz_path, allow_pickle=True)
            if "x_file" not in data.files:
                continue
            times = pd.Series([parse_time_from_wave_file(item) for item in data["x_file"]]).dropna()
            if times.empty:
                continue
            records.append(
                {
                    "event": infer_event_name_from_npz(npz_path),
                    "event_time_utc": times.max(),
                    "npz_path": str(npz_path),
                    "n_samples": len(times),
                }
            )
        except Exception as exc:
            print(f"[WARN] skip event catalog from {npz_path.name}: {exc}")

    catalog = pd.DataFrame(records)
    if catalog.empty:
        return catalog
    catalog["_event_order"] = catalog["event"].map(natural_event_key)
    return catalog.sort_values("_event_order").drop(columns="_event_order")


def add_event_rainfall_metrics(continuous: pd.DataFrame, event_catalog: pd.DataFrame) -> pd.DataFrame:
    if event_catalog.empty:
        return pd.DataFrame()

    rain = continuous.copy()
    rain["time_utc"] = pd.to_datetime(rain["time_utc"], utc=True)
    rain = rain.sort_values("time_utc").set_index("time_utc")

    rows = []
    for row in event_catalog.itertuples(index=False):
        event_time = pd.to_datetime(row.event_time_utc, utc=True)
        if event_time < rain.index.min() or event_time > rain.index.max():
            print(f"[WARN] {row.event}: event time outside rainfall range: {event_time}")
            continue

        nearest_indexer = rain.index.get_indexer([event_time], method="pad")[0]
        if nearest_indexer < 0:
            continue
        nearest_time = rain.index[nearest_indexer]
        event_row = rain.loc[nearest_time]
        record = {
            "event": row.event,
            "event_time_utc": event_time,
            "rain_record_time_utc": nearest_time,
            "time_offset_min": (event_time - nearest_time).total_seconds() / 60.0,
            "rain_30min_at_event_mm": float(event_row["rain_mm_30min"]),
            "rain_rate_at_event_mm_h": float(event_row["rain_rate_mm_h"]),
            "max_30min_mm_last_24h": float(event_row["max_30min_mm_last_24h"]),
            "max_1h_mm_last_24h": float(event_row["max_1h_mm_last_24h"]),
            "max_3h_mm_last_24h": float(event_row["max_3h_mm_last_24h"]),
            "max_6h_mm_last_24h": float(event_row["max_6h_mm_last_24h"]),
        }

        for hours in EVENT_WINDOWS_HOURS:
            col_name = f"event_cum_{hours}h_mm" if hours < 24 else f"event_cum_{hours // 24}d_mm"
            start = event_time - pd.Timedelta(hours=hours)
            mask = (rain.index > start) & (rain.index <= event_time)
            record[col_name] = float(rain.loc[mask, "rain_mm_30min"].sum())
            record[f"event_missing_{hours}h_count"] = int(rain.loc[mask, "is_missing_record"].sum())

        rows.append(record)

    event_summary = pd.DataFrame(rows)
    if event_summary.empty:
        return event_summary
    event_summary["_event_order"] = event_summary["event"].map(natural_event_key)
    return event_summary.sort_values("_event_order").drop(columns="_event_order")


# =========================================================
# 5. Figures
# =========================================================
def plot_monthly_yearly_summary(monthly: pd.DataFrame, yearly: pd.DataFrame) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = []

    fig, ax = plt.subplots(figsize=(16, 5), constrained_layout=True)
    ax.bar(monthly["year_month"], monthly["total_rain_mm"], color="#4C78A8", alpha=0.85)
    ax.set_ylabel("Monthly rainfall (mm)")
    ax.set_xlabel("Month")
    ax.set_title("GPM IMERG monthly rainfall at Illgraben", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="y", alpha=0.25)
    path = FIGURE_DIR / "gpm_monthly_rainfall_total.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    output_paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(yearly["year"].astype(str), yearly["total_rain_mm"], color="#54A24B", alpha=0.85)
    ax.set_ylabel("Yearly rainfall (mm)")
    ax.set_xlabel("Year")
    ax.set_title("GPM IMERG yearly rainfall at Illgraben", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    path = FIGURE_DIR / "gpm_yearly_rainfall_total.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    output_paths.append(path)

    return output_paths


def _window_hours_from_col(col: str) -> int:
    match_h = re.search(r"event_cum_(\d+)h_mm", col)
    if match_h:
        return int(match_h.group(1))
    match_d = re.search(r"event_cum_(\d+)d_mm", col)
    if match_d:
        return int(match_d.group(1)) * 24
    return 999


def plot_event_rainfall_summary(event_summary: pd.DataFrame) -> list[Path]:
    if event_summary.empty:
        return []

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = []
    plot_df = event_summary.copy()
    plot_df["_event_order"] = plot_df["event"].map(natural_event_key)
    plot_df = plot_df.sort_values("_event_order")

    columns = ["event_cum_1h_mm", "event_cum_6h_mm", "event_cum_1d_mm", "event_cum_3d_mm", "event_cum_7d_mm"]
    available_columns = [col for col in columns if col in plot_df.columns]

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    x = np.arange(len(plot_df))
    width = 0.8 / max(len(available_columns), 1)
    for i, col in enumerate(available_columns):
        ax.bar(x + (i - (len(available_columns) - 1) / 2) * width, plot_df[col], width=width, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["event"], rotation=90)
    ax.set_ylabel("Antecedent rainfall (mm)")
    ax.set_xlabel("Event")
    ax.set_title("Antecedent rainfall before debris-flow events", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, ncol=min(len(available_columns), 5))
    ax.grid(axis="y", alpha=0.25)
    path = FIGURE_DIR / "event_antecedent_rainfall_bars.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    output_paths.append(path)

    heat_cols = [col for col in plot_df.columns if col.startswith("event_cum_") and col.endswith("_mm")]
    heat_cols = sorted(heat_cols, key=_window_hours_from_col)
    matrix = plot_df[heat_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(10, len(heat_cols) * 1.1), max(9, len(plot_df) * 0.25)), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(heat_cols)))
    ax.set_xticklabels(heat_cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["event"])
    ax.set_title("Event antecedent rainfall heatmap", fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Rainfall (mm)")
    path = FIGURE_DIR / "event_antecedent_rainfall_heatmap.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    output_paths.append(path)

    return output_paths


# =========================================================
# 6. Export and main
# =========================================================
def make_excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timezone-aware datetime columns to Excel-compatible UTC-naive values."""
    safe_df = df.copy()
    for col in safe_df.columns:
        if pd.api.types.is_datetime64_any_dtype(safe_df[col]):
            safe_df[col] = pd.to_datetime(safe_df[col], utc=True, errors="coerce").dt.tz_localize(None)
            continue
        if safe_df[col].dtype == "object" and ("time" in col.lower() or col.endswith("_utc")):
            converted = pd.to_datetime(safe_df[col], utc=True, errors="coerce")
            if converted.notna().any():
                safe_df[col] = converted.dt.tz_localize(None)
    return safe_df


def export_outputs(data: RainfallData) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    continuous_csv = OUTPUT_DIR / "GPM_IMERG_Illgraben_30min_continuous_roll.csv"
    excel_path = OUTPUT_DIR / "GPM_IMERG_Illgraben_rainfall_analysis.xlsx"

    data.continuous.to_csv(continuous_csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        make_excel_safe(data.completeness_summary).to_excel(writer, sheet_name="data_completeness", index=False)
        make_excel_safe(data.monthly_summary).to_excel(writer, sheet_name="monthly_summary", index=False)
        make_excel_safe(data.yearly_summary).to_excel(writer, sheet_name="yearly_summary", index=False)
        make_excel_safe(data.event_summary).to_excel(writer, sheet_name="event_rainfall", index=False)
        make_excel_safe(data.continuous.head(200000)).to_excel(writer, sheet_name="continuous_30min_roll", index=False)

    return continuous_csv, excel_path


def run_analysis() -> RainfallData:
    gpm_dir = resolve_gpm_dir()
    print(f"Loading GPM monthly CSV files from: {gpm_dir}")

    raw = load_monthly_gpm_csvs(gpm_dir)
    continuous = build_continuous_30min_series(raw)
    completeness = summarize_completeness(continuous)
    monthly, yearly = summarize_monthly_yearly(continuous)
    event_catalog = build_event_catalog_from_npz()
    event_summary = add_event_rainfall_metrics(continuous, event_catalog)

    return RainfallData(
        raw=raw,
        continuous=continuous,
        monthly_summary=monthly,
        yearly_summary=yearly,
        completeness_summary=completeness,
        event_summary=event_summary,
    )


def main() -> None:
    data = run_analysis()
    continuous_csv, excel_path = export_outputs(data)
    figure_paths = []
    figure_paths.extend(plot_monthly_yearly_summary(data.monthly_summary, data.yearly_summary))
    figure_paths.extend(plot_event_rainfall_summary(data.event_summary))

    print("\nSaved processed rainfall CSV:", continuous_csv)
    print("Saved rainfall analysis Excel:", excel_path)
    print("Saved figures:")
    for path in figure_paths:
        print("  ", path)

    print("\nData completeness:")
    print(data.completeness_summary.head(12).to_string(index=False))

    if not data.event_summary.empty:
        print("\nTop events by 24 h antecedent rainfall:")
        top_events = data.event_summary.sort_values("event_cum_1d_mm", ascending=False).head(10)
        print(top_events[["event", "event_time_utc", "event_cum_1d_mm", "event_cum_3d_mm", "event_cum_7d_mm"]].to_string(index=False))


if __name__ == "__main__":
    main()
