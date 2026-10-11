# -*- coding: utf-8 -*-
from pathlib import Path
import csv
import textwrap
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from obspy import read
from scipy.signal import welch

from figure_rainfall_couple_event import (
    LAGS,
    build_hourly_matrices,
    lag_correlations,
    load_data as load_coupling_data,
)


PROJECT_ROOT = Path(r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben")
PAPER_DIR = PROJECT_ROOT / "uilt_oneday_all" / "uilt_paper_figure"
OUT_DIR = PAPER_DIR / "Fig5_record_section_two_precursor_patterns"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_DIR = next(
    PROJECT_ROOT.glob("uilt_oneday_all/03visual/**/kmeans_clustering_from_matched_precursor_features")
)
RESULT_CSV = RESULT_DIR / "kmeans_clustering_results.csv"
EVIDENCE_XLSX = RESULT_DIR / "kmeans2_mode_waveform_process_evidence" / "kmeans2_mode_waveform_process_evidence.xlsx"
NPZ_DIR = PROJECT_ROOT / "uilt_oneday_all" / "02clustering" / "result" / "DEC_result" / "all_parement"
SEGMENT_ROOT = PROJECT_ROOT / "uilt_oneday_all" / "01data_preprocess"

MODE_COLORS = {0: "#1F77B4", 1: "#E76F51"}
MODE_NAMES = {0: "Pattern I", 1: "Pattern II"}
NYQUIST_HZ = 50.0

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.linewidth": 0.75,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8.2,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def read_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(row, key, default=np.nan):
    try:
        return float(row[key])
    except Exception:
        return default


def to_int(row, key, default=-1):
    try:
        return int(float(row[key]))
    except Exception:
        return default


def values(rows, key):
    return np.array([to_float(r, key) for r in rows], dtype=float)


def xlsx_col_index(ref):
    letters = "".join(ch for ch in ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + ord(ch.upper()) - 64
    return idx - 1


def xlsx_cell_value(cell, shared, ns):
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.findall(".//a:t", ns))
    value = cell.find("a:v", ns)
    text = "" if value is None else value.text
    if cell_type == "s" and text != "":
        return shared[int(text)]
    return text


def read_xlsx_sheet(path, sheet_name):
    ns = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    with ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                shared.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))

        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        rid = None
        for sheet in workbook.find("a:sheets", ns):
            if sheet.attrib["name"] == sheet_name:
                rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
                break

        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels:
            if rel.attrib["Id"] == rid:
                target = rel.attrib["Target"].lstrip("/")
                break

        sheet_xml = ET.fromstring(z.read(target))
        raw_rows = []
        for row in sheet_xml.findall(".//a:sheetData/a:row", ns):
            values_by_col = {}
            for cell in row.findall("a:c", ns):
                values_by_col[xlsx_col_index(cell.attrib["r"])] = xlsx_cell_value(cell, shared, ns)
            if values_by_col:
                raw_rows.append([values_by_col.get(i, "") for i in range(max(values_by_col) + 1)])

    header = [str(v) for v in raw_rows[0]]
    return [{header[i]: raw[i] if i < len(raw) else "" for i in range(len(header))} for raw in raw_rows[1:]]


def npz_path(event):
    for name in [f"{event}_all_parement.npz", f"{event}_all_parameter.npz"]:
        path = NPZ_DIR / name
        if path.exists():
            return path
    return next(NPZ_DIR.glob(f"{event}*.npz"))


def resolve_sac_path(x_file):
    path = Path(str(x_file).replace("\\", "/"))
    if path.is_absolute() and path.exists():
        return path
    return SEGMENT_ROOT / str(x_file).replace(".\\", "").replace("\\", "/")


def normalize_waveform(x):
    x = np.asarray(x, dtype=float)
    x = x - np.nanmedian(x)
    scale = np.nanpercentile(np.abs(x), 99)
    return x / scale if np.isfinite(scale) and scale > 0 else x


def center_waveform(row):
    event = row["event"]
    cluster_id = to_int(row, "cluster_id")
    z = np.load(npz_path(event), allow_pickle=True)
    labels = np.asarray(z["y_pred_last"]).astype(int)
    distances = np.asarray(z["distances"], dtype=float)
    indices = np.where(labels == cluster_id)[0]
    center_index = indices[np.argmin(distances[indices, cluster_id])]
    tr = read(str(resolve_sac_path(z["x_file"][center_index])))[0]
    raw = np.asarray(tr.data, dtype=float)
    raw = raw - np.nanmedian(raw)
    return {
        "event": event.replace("event", "E"),
        "cluster": f"C{cluster_id}",
        "time_before_h": -24 + center_index / 60,
        "time": np.arange(tr.stats.npts) / float(tr.stats.sampling_rate),
        "waveform": normalize_waveform(raw),
        "raw": raw,
        "sampling_rate": float(tr.stats.sampling_rate),
    }


def build_hourly_mode_curves(rows):
    sheet = read_xlsx_sheet(EVIDENCE_XLSX, "hourly_counts")
    hour_cols = [f"hour_{h}_to_{h + 1}" for h in range(-24, 0)]
    curves = {}
    for mode in sorted({to_int(r, "kmeans_cluster") for r in rows}):
        event_ids = {r["event"] for r in rows if to_int(r, "kmeans_cluster") == mode}
        matrix = []
        for r in sheet:
            if r.get("event") in event_ids:
                matrix.append([float(r.get(col, 0) or 0) for col in hour_cols])
        matrix = np.asarray(matrix, dtype=float)
        curves[mode] = {
            "x": np.array([h + 0.5 for h in range(-24, 0)], dtype=float),
            "mean": np.nanmean(matrix, axis=0),
            "q25": np.nanpercentile(matrix, 25, axis=0),
            "q75": np.nanpercentile(matrix, 75, axis=0),
            "n": matrix.shape[0],
        }
    return curves


def build_psd_profiles(center_waveforms):
    profiles = {}
    for mode, items in center_waveforms.items():
        spectra = []
        freq_out = None
        for item in items:
            x = np.asarray(item["raw"], dtype=float)
            fs = float(item["sampling_rate"])
            nperseg = min(x.size, int(fs * 8))
            freq, psd = welch(
                x, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2,
                detrend="constant", scaling="density"
            )
            keep = (freq >= 0) & (freq <= NYQUIST_HZ)
            freq_out = freq[keep]
            spectra.append(10 * np.log10(psd[keep] + np.finfo(float).eps))
        matrix = np.asarray(spectra, dtype=float)
        profiles[mode] = {
            "freq": freq_out,
            "median": np.nanmedian(matrix, axis=0),
            "q25": np.nanpercentile(matrix, 25, axis=0),
            "q75": np.nanpercentile(matrix, 75, axis=0),
            "n": matrix.shape[0],
        }
    return profiles


def selected_shap_features():
    features = [r.get("selected_feature", "") for r in read_xlsx_sheet(EVIDENCE_XLSX, "selected_features")]
    return [f for f in features if f in rows[0]]


def pretty_feature_name(feature):
    names = {
        "NBRPEAKFREQMAX": "Peak-frequency\nmaximum",
        "DISTQ2Q1": "Q2-Q1\nfrequency distance",
        "NBRPEAKFREQCENTER": "Peak-frequency\ncenter",
        "NBRPEAKMEAN": "Peak-count\nmean",
    }
    return names.get(feature, "\n".join(textwrap.wrap(feature, width=14)))


def save_all(fig, base, dpi=400):
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(base.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight", pad_inches=0.04)


def panel_label(ax, label, x=-0.10, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


rows = read_rows(RESULT_CSV)
modes = sorted({to_int(row, "kmeans_cluster") for row in rows})
mode_counts = {mode: sum(to_int(row, "kmeans_cluster") == mode for row in rows) for mode in modes}
hourly_curves = build_hourly_mode_curves(rows)
waveforms = {
    mode: [center_waveform(r) for r in sorted(
        [row for row in rows if to_int(row, "kmeans_cluster") == mode],
        key=lambda row: int(row["event"].replace("event", ""))
    )]
    for mode in modes
}
psd_profiles = build_psd_profiles(waveforms)
top_features = selected_shap_features()[:4]


def plot_record_section(ax, mode):
    items = sorted(waveforms[mode], key=lambda item: item["time_before_h"])
    n = len(items)

    # ax.add_patch(Rectangle((0, -0.55), 60, n + 0.1, facecolor=MODE_COLORS[mode], alpha=0.045, lw=0))
    for i, item in enumerate(items):
        mask = item["time"] <= 60
        ax.plot(item["time"][mask], item["waveform"][mask] * 0.28 + i, color="black", lw=0.30)
        ax.axhline(i, color="0.87", lw=0.24, zorder=0)

    ax.plot([-0.045, -0.045], [0, 1], transform=ax.transAxes, color=MODE_COLORS[mode], lw=3.8, clip_on=False)
    ax.set_title(f"{MODE_NAMES[mode]} (n={mode_counts[mode]})", color=MODE_COLORS[mode], pad=2)
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel("")
    ax.set_ylabel("Trace index" if mode == 0 else "")
    ax.set_yticks([0, n - 1])
    ax.set_yticklabels([1, n])
    ax.tick_params(axis="y", length=2.2, width=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_feature_box(ax, feature, show_xlabel=False):
    rng = np.random.default_rng(20260701)
    for mode, pos in zip(modes, [0, 1]):
        vals = values([r for r in rows if to_int(r, "kmeans_cluster") == mode], feature)
        ax.boxplot(
            vals, positions=[pos], widths=0.48, patch_artist=True, showfliers=False,
            boxprops={"facecolor": MODE_COLORS[mode], "alpha": 0.16, "edgecolor": MODE_COLORS[mode], "lw": 0.75},
            medianprops={"color": MODE_COLORS[mode], "lw": 1.1},
            whiskerprops={"color": MODE_COLORS[mode], "lw": 0.75},
            capprops={"color": MODE_COLORS[mode], "lw": 0.75},
        )
        ax.scatter(
            np.full(vals.size, pos) + rng.normal(0, 0.035, vals.size), vals,
            s=12, color=MODE_COLORS[mode], edgecolor="white", linewidth=0.25, alpha=0.78,
        )
    ax.set_title(pretty_feature_name(feature), pad=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["I", "II"] if show_xlabel else [])
    ax.set_xlim(-0.55, 1.55)
    ax.grid(axis="y", color="0.90", lw=0.45)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_xlabel:
        ax.set_xlabel("Pattern")


def plot_hourly(ax):
    for mode in modes:
        c = hourly_curves[mode]
        color = MODE_COLORS[mode]
        ax.plot(c["x"], c["mean"], color=color, marker="o", ms=2.6, lw=1.6,
                label=f"{MODE_NAMES[mode]} (n={c['n']})")
        ax.fill_between(c["x"], c["q25"], c["q75"], color=color, alpha=0.15, lw=0)
    ax.set_xlim(-24, 0)
    ax.set_xticks([-24, -18, -12, -6, 0])
    ax.set_xticks(np.arange(-23.5, 0, 1), minor=True)
    ax.tick_params(axis="x", which="minor", length=2.0, width=0.55)
    ax.set_xlabel("Time before debris-flow initiation (h)")
    ax.set_ylabel("Hourly occurrence count")
    ax.set_title("Hourly evolution before debris-flow initiation", loc="left", pad=3)
    ax.grid(True, color="0.88", lw=0.45)
    ax.legend(loc="upper left", handlelength=1.2)


def plot_psd(ax):
    for mode in modes:
        p = psd_profiles[mode]
        color = MODE_COLORS[mode]
        ax.plot(p["freq"], p["median"], color=color, lw=1.7, label=f"{MODE_NAMES[mode]} (n={p['n']})")
        ax.fill_between(p["freq"], p["q25"], p["q75"], color=color, alpha=0.15, lw=0)
    ax.set_xlim(0, NYQUIST_HZ)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title("Median raw PSD profiles", loc="left", pad=3)
    ax.grid(True, color="0.88", lw=0.45)
    ax.legend(loc="upper left", handlelength=1.2)


def plot_rainfall_lag(ax):
    """Plot event-wise rainfall lead correlations using the same 35 precursor clusters."""
    mode_table, matched_samples, rain = load_coupling_data()
    rainfall, precursor, _, mode_labels = build_hourly_matrices(
        mode_table, matched_samples, rain
    )
    correlations = lag_correlations(rainfall, precursor)

    for mode, mode_name in zip(modes, ["Pattern I", "Pattern II"]):
        matrix = correlations[mode_labels == mode_name]
        median = np.nanmedian(matrix, axis=0)
        q25, q75 = np.nanpercentile(matrix, [25, 75], axis=0)
        color = MODE_COLORS[mode]
        ax.plot(
            LAGS, median, color=color, marker="o", ms=3.2, lw=1.6,
            label=f"{MODE_NAMES[mode]} (n={len(matrix)})",
        )
        ax.fill_between(LAGS, q25, q75, color=color, alpha=0.15, lw=0)

    ax.axhline(0, color="0.45", linestyle="--", lw=0.8)
    ax.set_xlim(-0.3, 12.3)
    ax.set_ylim(-0.65, 0.65)
    ax.set_xticks(LAGS)
    ax.set_xlabel("Rainfall lead time relative to precursor activity (h)")
    ax.set_ylabel("Event-wise Spearman correlation")
    ax.set_title("Event-scale rainfall–precursor coupling", loc="left", pad=3)
    ax.grid(True, color="0.88", lw=0.45)
    ax.legend(loc="upper right", ncol=2, handlelength=1.4)


fig = plt.figure(figsize=(10, 8.1))
outer = fig.add_gridspec(
    6, 5,
    width_ratios=[0.95, 0.95, 0.5, 0.5, 0.5],
    height_ratios=[1, 1, 1, 1, 0.82, 0.82],
    left=0.065, right=0.985, bottom=0.065, top=0.965,
    wspace=0.36, hspace=0.92,
)

ax_wave_i = fig.add_subplot(outer[0:2, 0])
ax_wave_ii = fig.add_subplot(outer[0:2, 1])
box_grid = outer[2:4, 0:2].subgridspec(2, 2, hspace=0.55, wspace=0.45)
feature_axes = [fig.add_subplot(box_grid[i // 2, i % 2]) for i in range(4)]
ax_hourly = fig.add_subplot(outer[0:2, 2:5])
ax_psd = fig.add_subplot(outer[2:4, 2:5])
ax_rainfall_lag = fig.add_subplot(outer[4:6, 0:5])

plot_record_section(ax_wave_i, 0)
plot_record_section(ax_wave_ii, 1)
for i, (ax, feature) in enumerate(zip(feature_axes, top_features)):
    plot_feature_box(ax, feature, show_xlabel=i == len(top_features) - 1)
plot_hourly(ax_hourly)
plot_psd(ax_psd)
plot_rainfall_lag(ax_rainfall_lag)

panel_label(ax_wave_i, "a", x=-0.13)
panel_label(feature_axes[0], "b", x=-0.25)
panel_label(ax_hourly, "c")
panel_label(ax_psd, "d")
panel_label(ax_rainfall_lag, "e", x=-0.055)

base = OUT_DIR / "figure5_record_section_two_precursor_patterns"
save_all(fig, base)
plt.show()
plt.close(fig)

print(base.with_suffix(".png"))
