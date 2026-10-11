# -*- coding: utf-8 -*-
"""
Plot pre-event seismic evidence for Figure 1.

The script reads debris-flow initiation times and SAC file paths from the
event Excel file, loops through all retained events, and exports a compact
right-column figure for each event containing:
    (a) retained debris-flow event inventory,
    (b) 24 h pre-event RMS amplitude envelope,
    (c) 10 min PSD time evolution before initiation.

Run with:
    D:\Anaconda\python.exe 01_precursor_wave.py
"""

from __future__ import annotations

import math
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime, read
from scipy.signal import welch


PROJECT_ROOT = Path(r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben")
EXCEL_PATH = PROJECT_ROOT / r"data\新版完整数据集时间.xlsx"
OUTPUT_DIR = PROJECT_ROOT / r"uilt_oneday_all\uilt_paper_figure\Fig1"

# Set to e.g. ["event26"] to draw selected events only. If None, all events
# in the Excel table are drawn.
EVENTS_TO_PLOT: list[str] | None = None

PRE_EVENT_HOURS = 24
HIGHPASS_HZ = 1.0
RMS_WINDOW_SECONDS = 60
PSD_WINDOW_SECONDS = 10 * 60
PSD_OVERLAP = 0.5
PSD_FREQ_MIN = 1.0
PSD_FREQ_MAX = 50.0
EPS = 1e-20


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)


def excel_column_to_index(cell_ref: str) -> int:
    """Convert an Excel cell reference such as 'AB12' to a zero-based column."""
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch.upper()) - ord("A") + 1)
    return value - 1


def read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    """Read shared strings from an xlsx archive without openpyxl/pandas."""
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []

    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings: list[str] = []
    for si in root.findall("x:si", ns):
        parts = [node.text or "" for node in si.findall(".//x:t", ns)]
        strings.append("".join(parts))
    return strings


def read_xlsx_first_sheet(path: Path) -> list[dict[str, str]]:
    """Read the first worksheet into a list of dictionaries using stdlib only."""
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows: list[list[str]] = []

    with zipfile.ZipFile(path) as zf:
        shared_strings = read_shared_strings(zf)
        sheet_xml = zf.read("xl/worksheets/sheet1.xml")
        root = ET.fromstring(sheet_xml)

        for row in root.findall(".//x:sheetData/x:row", ns):
            values: list[str] = []
            for cell in row.findall("x:c", ns):
                col_idx = excel_column_to_index(cell.attrib.get("r", "A1"))
                while len(values) <= col_idx:
                    values.append("")

                cell_type = cell.attrib.get("t")
                if cell_type == "s":
                    raw = cell.findtext("x:v", default="", namespaces=ns)
                    values[col_idx] = shared_strings[int(raw)] if raw else ""
                elif cell_type == "inlineStr":
                    values[col_idx] = "".join(
                        t.text or "" for t in cell.findall(".//x:t", ns)
                    )
                else:
                    values[col_idx] = cell.findtext("x:v", default="", namespaces=ns)
            rows.append(values)

    if not rows:
        return []

    headers = [str(item).strip() for item in rows[0]]
    records: list[dict[str, str]] = []
    for row in rows[1:]:
        record = {headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))}
        if any(str(v).strip() for v in record.values()):
            records.append(record)
    return records


def parse_event_time(value: str) -> UTCDateTime:
    """Parse ISO strings or Excel serial-date strings into UTCDateTime."""
    text = str(value).strip()
    if re.fullmatch(r"\d+(\.\d+)?", text):
        # Excel serial date. Excel's day 1 is 1900-01-01, with the 1900 leap-year bug.
        serial = float(text)
        unix_days = serial - 25569.0
        return UTCDateTime(unix_days * 86400.0)
    return UTCDateTime(text)


def to_float(value: str, default: float = math.nan) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def get_event_name(record: dict[str, str]) -> str:
    """Return a stable event name from either Event or Event1 columns."""
    for key in ("Event", "Event1", "Event.1"):
        value = str(record.get(key, "")).strip()
        if value:
            return value

    path_text = str(record.get("SAC_File_Path", ""))
    match = re.search(r"event\d+", path_text, flags=re.IGNORECASE)
    return match.group(0) if match else "event"


def get_records_to_plot(records: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return records selected for plotting."""
    if EVENTS_TO_PLOT is None:
        return records

    selected = []
    requested = set(EVENTS_TO_PLOT)
    for record in records:
        if get_event_name(record) in requested:
            selected.append(record)

    found = {get_event_name(record) for record in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"Requested events were not found: {missing}")
    return selected


def load_pre_event_trace(record: dict[str, str]):
    """Read, trim and preprocess one 24 h pre-event trace."""
    event_time = parse_event_time(record["Manual_Label_start"])
    start_time = event_time - PRE_EVENT_HOURS * 3600
    sac_path = Path(str(record["SAC_File_Path"]).strip())

    stream = read(str(sac_path), starttime=start_time, endtime=event_time)
    stream.merge(method=1, fill_value="interpolate")
    trace = stream[0]
    trace.trim(start_time, event_time, pad=True, fill_value=0.0)

    trace.detrend("demean")
    trace.detrend("linear")
    trace.taper(max_percentage=0.01, type="hann")
    trace.filter("highpass", freq=HIGHPASS_HZ, corners=4, zerophase=True)
    return trace, event_time


def minute_rms(data: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Return time before initiation and RMS amplitude in 1 min windows."""
    samples_per_window = int(round(RMS_WINDOW_SECONDS * sampling_rate))
    n_windows = len(data) // samples_per_window
    trimmed = data[: n_windows * samples_per_window]
    if n_windows == 0:
        raise ValueError("Trace is shorter than one RMS window.")

    windows = trimmed.reshape(n_windows, samples_per_window)
    rms = np.sqrt(np.mean(windows**2, axis=1))
    hours_before = -PRE_EVENT_HOURS + (np.arange(n_windows) + 0.5) * RMS_WINDOW_SECONDS / 3600.0
    return hours_before, rms


def compute_psd_evolution(
    data: np.ndarray, sampling_rate: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Welch PSD in sliding windows across the 24 h pre-event trace."""
    samples_per_window = int(round(PSD_WINDOW_SECONDS * sampling_rate))
    step = int(round(samples_per_window * (1.0 - PSD_OVERLAP)))
    starts = np.arange(0, len(data) - samples_per_window + 1, step, dtype=int)
    if len(starts) == 0:
        raise ValueError("Trace is shorter than one PSD window.")

    psd_columns = []
    centers_hours = []
    freqs = None
    nperseg = min(4096, samples_per_window)

    for start in starts:
        segment = data[start : start + samples_per_window]
        f, pxx = welch(
            segment,
            fs=sampling_rate,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend="constant",
            scaling="density",
        )
        if freqs is None:
            freqs = f
        psd_columns.append(10.0 * np.log10(pxx + EPS))
        center = start + samples_per_window / 2.0
        centers_hours.append(-PRE_EVENT_HOURS + center / sampling_rate / 3600.0)

    psd = np.column_stack(psd_columns)
    freq_mask = (freqs >= PSD_FREQ_MIN) & (freqs <= PSD_FREQ_MAX)
    return np.asarray(centers_hours), freqs[freq_mask], psd[freq_mask, :]


def plot_figure(records: list[dict[str, str]], record: dict[str, str]) -> Path:
    """Create and save the Figure 1 right-column evidence panel."""
    trace, event_time = load_pre_event_trace(record)
    data = trace.data.astype(float)
    sampling_rate = float(trace.stats.sampling_rate)

    rms_hours, rms = minute_rms(data, sampling_rate)
    psd_hours, psd_freqs, psd = compute_psd_evolution(data, sampling_rate)

    event_points = []
    for row in records:
        try:
            row_time = parse_event_time(row["Manual_Label_start"]).datetime
        except Exception:
            continue
        event_points.append(
            {
                "time": row_time,
                "year": row_time.year,
                "month": row_time.month,
                "day": row_time.day,
                "event": get_event_name(row),
                "amplitude": to_float(row.get("Max_Amplitude", ""), default=math.nan),
            }
        )

    fig = plt.figure(figsize=(4.25, 6.25), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.15, 2.2])
    ax_events = fig.add_subplot(gs[0, 0])
    ax_wave = fig.add_subplot(gs[1, 0])
    ax_psd = fig.add_subplot(gs[2, 0], sharex=ax_wave)

    # Panel a: event inventory with seasonality and representative event.
    amplitudes = np.asarray([p["amplitude"] for p in event_points], dtype=float)
    valid_amp = amplitudes[np.isfinite(amplitudes) & (amplitudes > 0)]
    if len(valid_amp) > 0:
        amp_scale = np.nanpercentile(valid_amp, 90)
    else:
        amp_scale = 1.0

    years = np.asarray([p["year"] for p in event_points], dtype=float)
    months = np.asarray([p["month"] + (p["day"] - 15) / 45.0 for p in event_points], dtype=float)
    sizes = []
    for p in event_points:
        amp = p["amplitude"]
        if np.isfinite(amp) and amp > 0:
            sizes.append(13 + 55 * min(amp / amp_scale, 1.6))
        else:
            sizes.append(18)
    sizes = np.asarray(sizes)

    ax_events.scatter(
        years,
        months,
        s=sizes,
        color="#555555",
        alpha=0.72,
        edgecolor="white",
        linewidth=0.35,
    )
    selected_name = get_event_name(record)
    selected_points = [p for p in event_points if p["event"] == selected_name]
    if selected_points:
        p = selected_points[0]
        selected_y = p["month"] + (p["day"] - 15) / 45.0
        ax_events.scatter(
            [p["year"]],
            [selected_y],
            s=115,
            facecolor="none",
            edgecolor="#B83A2E",
            linewidth=1.35,
            zorder=5,
        )
        ax_events.text(
            p["year"] + 0.08,
            selected_y + 0.08,
            selected_name,
            color="#B83A2E",
            fontsize=6.5,
            va="bottom",
        )

    for month in range(5, 11):
        ax_events.axhline(month, color="0.92", lw=0.45, zorder=0)
    ax_events.set_ylim(4.55, 10.45)
    ax_events.set_yticks([5, 6, 7, 8, 9, 10])
    ax_events.set_yticklabels(["May", "Jun", "Jul", "Aug", "Sep", "Oct"])
    year_min = int(np.nanmin(years)) if len(years) else 2017
    year_max = int(np.nanmax(years)) if len(years) else 2023
    ax_events.set_xlim(year_min - 0.35, year_max + 0.35)
    ax_events.set_xticks(np.arange(year_min, year_max + 1, 1))
    ax_events.tick_params(axis="x", labelsize=6)
    ax_events.set_title(
        "Retained events by year and season (n = 44)",
        loc="left",
        fontsize=8,
    )
    ax_events.set_xlabel("Year")
    ax_events.set_ylabel("Month")

    # Panel b: minutely RMS envelope.
    rms_norm = rms / np.nanpercentile(rms, 99) if np.nanpercentile(rms, 99) > 0 else rms
    ax_wave.plot(rms_hours, rms_norm, color="#2B5C8A", lw=0.75)
    ax_wave.fill_between(rms_hours, 0, rms_norm, color="#2B5C8A", alpha=0.18, lw=0)
    ax_wave.axvline(0, color="#B83A2E", lw=1.0)
    ax_wave.set_xlim(-PRE_EVENT_HOURS, 0)
    ax_wave.set_ylabel("RMS\n(norm.)")
    ax_wave.set_title(
        f"{get_event_name(record)} pre-event seismic envelope",
        loc="left",
        fontsize=8,
    )
    ax_wave.grid(axis="y", color="0.9", lw=0.5)

    # Panel c: PSD time evolution.
    vmin, vmax = np.nanpercentile(psd, [5, 98])
    image = ax_psd.pcolormesh(
        psd_hours,
        psd_freqs,
        psd,
        shading="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    ax_psd.axvline(0, color="#B83A2E", lw=1.0)
    ax_psd.set_ylim(PSD_FREQ_MIN, PSD_FREQ_MAX)
    ax_psd.set_ylabel("Frequency (Hz)")
    ax_psd.set_xlabel("Time before debris-flow initiation (h)")
    ax_psd.set_title("PSD evolution", loc="left", fontsize=8)

    cbar = fig.colorbar(image, ax=ax_psd, pad=0.015, fraction=0.055)
    cbar.set_label("PSD (dB)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    for label, ax in zip(["a", "b", "c"], [ax_events, ax_wave, ax_psd]):
        ax.text(
            -0.11,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"figure1_right_precursor_wave_psd_{get_event_name(record)}"
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight", dpi=600)
    plt.close(fig)
    return OUTPUT_DIR / f"{stem}.png"


def main() -> None:
    records = read_xlsx_first_sheet(EXCEL_PATH)
    selected_records = get_records_to_plot(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failed = []
    for idx, record in enumerate(selected_records, start=1):
        event_name = get_event_name(record)
        sac_path = str(record.get("SAC_File_Path", "")).strip()
        print(f"[{idx}/{len(selected_records)}] Plotting {event_name}: {sac_path}")
        try:
            out_path = plot_figure(records, record)
            success_count += 1
            print(f"  saved: {out_path}")
        except Exception as exc:
            failed.append((event_name, str(exc)))
            print(f"  failed: {exc}")

    print(f"Finished. Saved {success_count} PNG figures to: {OUTPUT_DIR}")
    if failed:
        print("Failed events:")
        for event_name, reason in failed:
            print(f"  {event_name}: {reason}")


if __name__ == "__main__":
    main()
