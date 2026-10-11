# -*- coding: utf-8 -*-
"""Event-scale coupling between hourly rainfall and precursor-cluster activity."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
MODE_DIR = BASE / "03visual/二次聚类/kmeans_clustering_from_matched_precursor_features"
SAMPLES = MODE_DIR / "matched_precursor_samples.csv"
MODE_RAIN = (
    BASE / "03visual/compared_different_duration/mode_rainfall_relationship"
    / "mode_rainfall_relationship.xlsx"
)
RAIN = (
    BASE / "03visual/rain_coupling/gpm_rainfall_analysis"
    / "GPM_IMERG_Illgraben_30min_continuous_roll.csv"
)
OUTPUT = HERE / "figure_rainfall_precursor_event_coupling.png"

HOURS = np.arange(-24, 0)  # bins: [-24,-23), ..., [-1,0) h
LAGS = np.arange(0, 13)   # positive lag: rainfall precedes precursor activity
MODES = ["Pattern I", "Pattern II"]
COLORS = ["#2878B5", "#E76F51"]

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
})


def event_number(value):
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else 10**9


def load_data():
    modes = pd.read_excel(MODE_RAIN, sheet_name="mode_rainfall_event_table")
    modes = modes[["event", "mode_short_name", "event_time_utc"]].copy()
    modes["event"] = modes["event"].str.lower()
    modes["event_time_utc"] = pd.to_datetime(modes["event_time_utc"], utc=True)

    samples = pd.read_csv(SAMPLES, usecols=["event", "timestamp"])
    samples["event"] = samples["event"].str.lower()
    samples["timestamp"] = pd.to_datetime(
        samples["timestamp"], format="%Y%m%dT%H%M%S", utc=True
    )

    rain = pd.read_csv(RAIN, usecols=["time_utc", "rain_mm_30min"])
    rain["time_utc"] = pd.to_datetime(rain["time_utc"], utc=True)
    return modes, samples, rain


def build_hourly_matrices(modes, samples, rain):
    modes = modes.sort_values(
        ["mode_short_name", "event"],
        key=lambda col: col.map(event_number) if col.name == "event" else col,
    )
    rainfall, precursor, labels, mode_list = [], [], [], []

    for row in modes.itertuples(index=False):
        start, end = row.event_time_utc - pd.Timedelta(hours=24), row.event_time_utc
        event_rain = rain[(rain.time_utc >= start) & (rain.time_utc < end)].copy()
        event_rain["hour"] = np.floor(
            (event_rain.time_utc - row.event_time_utc).dt.total_seconds() / 3600
        ).astype(int)
        rain_hourly = event_rain.groupby("hour").rain_mm_30min.sum().reindex(HOURS, fill_value=0)

        event_samples = samples[
            (samples.event == row.event)
            & (samples.timestamp >= start)
            & (samples.timestamp < end)
        ].copy()
        event_samples["hour"] = np.floor(
            (event_samples.timestamp - row.event_time_utc).dt.total_seconds() / 3600
        ).astype(int)
        precursor_hourly = event_samples.groupby("hour").size().reindex(HOURS, fill_value=0)

        rainfall.append(rain_hourly.to_numpy(float))
        precursor.append(precursor_hourly.to_numpy(float))
        labels.append(f"E{event_number(row.event)}")
        mode_list.append(row.mode_short_name)

    return np.asarray(rainfall), np.asarray(precursor), labels, np.asarray(mode_list)


def lag_correlations(rainfall, precursor):
    result = np.full((len(rainfall), len(LAGS)), np.nan)
    for i, (rain, count) in enumerate(zip(rainfall, precursor)):
        for j, lag in enumerate(LAGS):
            x = rain[:-lag or None]
            y = count[lag:]
            if len(x) >= 6 and np.std(x) > 0 and np.std(y) > 0:
                result[i, j] = spearmanr(x, y).statistic
    return result


def add_mode_separator(ax, mode_list):
    boundary = np.flatnonzero(mode_list[1:] != mode_list[:-1])
    if len(boundary):
        ax.axhline(boundary[0] + 0.5, color="white", linewidth=2.0)


def main():
    modes, samples, rain = load_data()
    rainfall, precursor, labels, mode_list = build_hourly_matrices(modes, samples, rain)
    precursor_fraction = precursor / np.maximum(precursor.sum(axis=1, keepdims=True), 1)
    correlations = lag_correlations(rainfall, precursor)

    fig = plt.figure(figsize=(8.2, 7.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.55, 1], hspace=0.33, wspace=0.32)
    ax1, ax2 = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, :])

    vmax = max(np.nanpercentile(rainfall, 97), 0.1)
    im1 = ax1.imshow(rainfall, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
    im2 = ax2.imshow(precursor_fraction, aspect="auto", cmap="Oranges", vmin=0)
    for ax in (ax1, ax2):
        ax.set_xticks([0, 6, 12, 18, 23], ["-24", "-18", "-12", "-6", "0"])
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Time before debris-flow initiation (h)")
        ax.tick_params(length=0, labelsize=6)
        add_mode_separator(ax, mode_list)
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax1.set_ylabel("Event (Pattern I / Pattern II)")
    ax2.tick_params(labelleft=False)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03).set_label("Hourly rainfall (mm)")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03).set_label("Hourly precursor fraction")
    ax1.text(-0.15, 1.03, "a", transform=ax1.transAxes, fontsize=11, fontweight="bold")
    ax2.text(-0.10, 1.03, "b", transform=ax2.transAxes, fontsize=11, fontweight="bold")

    for mode, color in zip(MODES, COLORS):
        values = correlations[mode_list == mode]
        median = np.nanmedian(values, axis=0)
        q25, q75 = np.nanpercentile(values, [25, 75], axis=0)
        ax3.plot(LAGS, median, "o-", color=color, label=f"{mode} (n={len(values)})")
        ax3.fill_between(LAGS, q25, q75, color=color, alpha=0.18, linewidth=0)
    ax3.axhline(0, color="#777777", linestyle="--", linewidth=0.8)
    ax3.set_xticks(LAGS)
    ax3.set_xlabel("Rainfall lead time relative to precursor activity (h)")
    ax3.set_ylabel("Event-wise Spearman correlation")
    ax3.set_ylim(-0.65, 0.65)
    ax3.grid(color="#E6E6E6", linewidth=0.6)
    ax3.legend(frameon=False, ncol=2, loc="upper right")
    ax3.text(-0.06, 1.04, "c", transform=ax3.transAxes, fontsize=11, fontweight="bold")

    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
