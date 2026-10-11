# -*- coding: utf-8 -*-
"""Build the main-text and supplementary DEC model-result figures.

Figure contract
---------------
Core conclusion
    Event-specific DEC models converge to separable, spectrally interpretable
    pre-initiation states, while the preferred number of states varies among
    the 44 debris-flow events.
Archetype
    Asymmetric mixed-modality figure (quantitative validation + spectrograms).
Main-text evidence
    Across-event AE convergence, event-specific K, across-event clustering
    quality, a representative t-SNE embedding, and cluster-medoid spectra.
Supplementary evidence
    Individual-event convergence, sensitivity of K to the selection criterion,
    across-event DEC convergence with correctly weighted loss components, and a
    complete reconstruction audit containing original spectra, reconstructions,
    residuals, and quality metrics.

The script reads existing model outputs only; it does not retrain any model.
Run with the Python environment used for the DEC analysis, for example::

    conda run -n py39 python 02_model_result.py --event event5
"""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


# Nature-style publication defaults. SVG text remains editable.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9.3,
        "ytick.labelsize": 9.3,
        "legend.fontsize": 9.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "legend.frameon": False,
        "savefig.facecolor": "white",
    }
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "uilt_oneday_all" / "02clustering" / "result"
AE_DIR = RESULT_ROOT / "AE"
DEC_DIR = RESULT_ROOT / "DEC_result"
K_DIR = RESULT_ROOT / "Figure" / "optimal_cluster"
BEST_K_FILE = RESULT_ROOT / "Figure" / "summary_best_k.xlsx"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "Fig2_cluster_results" / "submission"

GAMMA = 0.1
TIME_RANGE_S = (0.0, 60.0)
FREQUENCY_RANGE_HZ = (0.0, 50.0)

COLORS = {
    "blue": "#0F4D92",
    "blue_mid": "#5B7FCA",
    "blue_soft": "#B4C0E4",
    "orange": "#D97706",
    "lilac": "#9A7CC1",
    "neutral_light": "#D8D8D8",
    "neutral_mid": "#8F8F8F",
    "neutral_dark": "#4D4D4D",
    "black": "#272727",
}

# Okabe-Ito-derived, colour-vision-deficiency-friendly cluster colours.
CLUSTER_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermilion
    "#F0E442",  # yellow
    "#000000",  # black
]

K_METRIC_COLUMNS = {"WCSS": "WCSS"}


def event_number(value: str | Path) -> int:
    """Return the numeric part of an event name for natural sorting."""
    match = re.search(r"event(\d+)", str(value))
    if match is None:
        raise ValueError(f"Cannot extract event number from {value!r}")
    return int(match.group(1))


def read_csv_numeric(path: Path) -> List[Dict[str, float | str]]:
    """Read a CSV and convert numeric fields to floats."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows: List[Dict[str, float | str]] = []
        for row in csv.DictReader(handle):
            clean: Dict[str, float | str] = {}
            for key, value in row.items():
                try:
                    clean[key] = float(value)
                except (TypeError, ValueError):
                    clean[key] = value
            rows.append(clean)
    return rows


def _column_index(cell_ref: str) -> int:
    letters = re.match(r"([A-Z]+)", cell_ref)
    if letters is None:
        raise ValueError(f"Invalid XLSX cell reference: {cell_ref}")
    value = 0
    for char in letters.group(1):
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def _shared_text(node: ET.Element, ns: Mapping[str, str]) -> str:
    return "".join(item.text or "" for item in node.findall(".//m:t", ns))


def read_xlsx_sheet(path: Path, sheet_name: str = "Sheet1") -> List[Dict[str, object]]:
    """Read a small XLSX worksheet without requiring pandas/openpyxl."""
    ns = {
        "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with zipfile.ZipFile(path) as archive:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [_shared_text(node, ns) for node in root.findall("m:si", ns)]

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in relationships.findall("rel:Relationship", ns)
        }
        target = None
        for sheet in workbook.findall(".//m:sheet", ns):
            if sheet.attrib["name"] == sheet_name:
                target = targets[sheet.attrib[f"{{{ns['r']}}}id"]]
                break
        if target is None:
            raise ValueError(f"Worksheet {sheet_name!r} not found in {path}")

        target = target.lstrip("/")
        sheet_xml = target if target.startswith("xl/") else f"xl/{target}"
        worksheet = ET.fromstring(archive.read(sheet_xml))

    table: List[List[object]] = []
    for row in worksheet.findall(".//m:sheetData/m:row", ns):
        values: List[object] = []
        for cell in row.findall("m:c", ns):
            index = _column_index(cell.attrib["r"])
            while len(values) <= index:
                values.append("")
            cell_type = cell.attrib.get("t")
            if cell_type == "inlineStr":
                values[index] = _shared_text(cell, ns)
                continue
            value_node = cell.find("m:v", ns)
            if value_node is None:
                values[index] = ""
                continue
            raw = value_node.text or ""
            if cell_type == "s":
                values[index] = shared[int(raw)]
            else:
                try:
                    values[index] = float(raw)
                except ValueError:
                    values[index] = raw
        table.append(values)

    if not table:
        return []
    header = [str(value) for value in table[0]]
    records: List[Dict[str, object]] = []
    for values in table[1:]:
        padded = (values + [""] * len(header))[: len(header)]
        records.append(dict(zip(header, padded)))
    return records


def stack_curves(curves: Sequence[np.ndarray]) -> np.ndarray:
    """Pad unequal-length curves with NaN so event summaries remain aligned."""
    if not curves:
        raise ValueError("No curves were provided")
    max_length = max(len(curve) for curve in curves)
    array = np.full((len(curves), max_length), np.nan, dtype=float)
    for row, curve in enumerate(curves):
        array[row, : len(curve)] = curve
    return array


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding (no artificial end collapse)."""
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < window:
        return values.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.16, y: float = 1.10) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def cluster_color(cluster_id: int) -> str:
    return CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]


def load_ae_curves() -> Tuple[List[str], np.ndarray, np.ndarray]:
    paths = sorted(
        AE_DIR.glob("event*_stft_maxmin_Z_pretrain_log.csv"),
        key=event_number,
    )
    if not paths:
        raise FileNotFoundError(f"No AE logs found in {AE_DIR}")
    events: List[str] = []
    training: List[np.ndarray] = []
    validation: List[np.ndarray] = []
    for path in paths:
        rows = read_csv_numeric(path)
        events.append(re.search(r"event\d+", path.name).group(0))
        training.append(np.asarray([row["loss"] for row in rows], dtype=float))
        validation.append(np.asarray([row["val_loss"] for row in rows], dtype=float))
    return events, stack_curves(training), stack_curves(validation)


def load_k_data() -> Tuple[List[Tuple[str, np.ndarray, np.ndarray]], Dict[str, Dict[str, int]]]:
    paths = sorted(K_DIR.glob("event*.xlsx"), key=event_number)
    if not paths:
        raise FileNotFoundError(f"No WCSS files found in {K_DIR}")

    curves: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for path in paths:
        rows = read_xlsx_sheet(path)
        k_values = np.asarray([row["n_clusters"] for row in rows], dtype=float)
        wcss = np.asarray([row["WCSS"] for row in rows], dtype=float)
        curves.append((path.stem, k_values, wcss))

    summary: Dict[str, Dict[str, int]] = {}
    for row in read_xlsx_sheet(BEST_K_FILE):
        event = str(row["Event"])
        summary[event] = {
            short_name: int(float(row[column]))
            for short_name, column in K_METRIC_COLUMNS.items()
        }
    return curves, summary


def compute_cluster_quality(npz_paths: Sequence[Path]) -> List[Dict[str, float | int | str]]:
    """Compute cluster quality in the final 32-dimensional DEC embedding."""
    metrics: List[Dict[str, float | int | str]] = []
    for path in sorted(npz_paths, key=event_number):
        with np.load(path, allow_pickle=True) as archive:
            features = np.asarray(archive["feature"], dtype=np.float64)
            labels = np.asarray(archive["y_pred_last"], dtype=int)
        unique = np.unique(labels)
        if len(unique) < 2 or len(unique) >= len(labels):
            silhouette = np.nan
            calinski = np.nan
            davies = np.nan
        else:
            silhouette = float(silhouette_score(features, labels, metric="euclidean"))
            calinski = float(calinski_harabasz_score(features, labels))
            davies = float(davies_bouldin_score(features, labels))
        metrics.append(
            {
                "event": re.search(r"event\d+", path.name).group(0),
                "n_samples": int(len(labels)),
                "n_clusters": int(len(unique)),
                "silhouette": silhouette,
                "calinski_harabasz": calinski,
                "davies_bouldin": davies,
            }
        )
    return metrics


def representative_indices(labels: np.ndarray, distances: np.ndarray) -> Dict[int, int]:
    """Select the observed sample nearest each DEC centroid (cluster medoid proxy)."""
    representatives: Dict[int, int] = {}
    for cluster_id in sorted(int(value) for value in np.unique(labels)):
        members = np.where(labels == cluster_id)[0]
        if distances.ndim == 2 and cluster_id < distances.shape[1]:
            chosen = members[np.argmin(distances[members, cluster_id])]
        else:
            chosen = members[len(members) // 2]
        representatives[cluster_id] = int(chosen)
    return representatives


def load_representative_event(event: str) -> Dict[str, object]:
    npz_path = DEC_DIR / "all_parement" / f"{event}_all_parement.npz"
    loss_path = DEC_DIR / "loss_csv" / f"{event}training_loss.csv"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not loss_path.exists():
        raise FileNotFoundError(loss_path)
    with np.load(npz_path, allow_pickle=True) as archive:
        data: Dict[str, object] = {
            "x_file": np.asarray(archive["x_file"]),
            "spectral": np.asarray(archive["spectral"]),
            "reconstructed": np.asarray(archive["respectral"]),
            "features": np.asarray(archive["feature"]),
            "tsne": np.asarray(archive["redu"]),
            "labels": np.asarray(archive["y_pred_last"], dtype=int),
            "n_clusters": int(np.asarray(archive["n_clusters"])),
            "distances": np.asarray(archive["distances"]),
        }
    data["loss_rows"] = read_csv_numeric(loss_path)
    data["representatives"] = representative_indices(data["labels"], data["distances"])
    return data


def load_dec_loss_curves() -> Dict[str, List[Dict[str, float | str]]]:
    """Load DEC optimization logs for all events in natural event order."""
    paths = sorted((DEC_DIR / "loss_csv").glob("event*training_loss.csv"), key=event_number)
    if not paths:
        raise FileNotFoundError(f"No DEC loss logs found in {DEC_DIR / 'loss_csv'}")
    return {
        re.search(r"event\d+", path.name).group(0): read_csv_numeric(path)
        for path in paths
    }


def summarize_curves(array: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "median": np.nanmedian(array, axis=0),
        "q25": np.nanpercentile(array, 25, axis=0),
        "q75": np.nanpercentile(array, 75, axis=0),
        "n": np.sum(np.isfinite(array), axis=0),
    }


def positive_for_log(values: np.ndarray) -> np.ndarray:
    finite_positive = values[np.isfinite(values) & (values > 0)]
    floor = np.nanmin(finite_positive) * 0.5 if len(finite_positive) else 1e-8
    return np.where(values > 0, values, floor)


def draw_ae_summary(ax: plt.Axes, train: np.ndarray, validation: np.ndarray) -> None:
    epoch = np.arange(train.shape[1])
    train_summary = summarize_curves(train)
    val_summary = summarize_curves(validation)
    for summary, color, label in (
        (train_summary, COLORS["blue"], "Training"),
        (val_summary, COLORS["orange"], "Validation"),
    ):
        median = positive_for_log(summary["median"])
        lower = positive_for_log(summary["q25"])
        upper = positive_for_log(summary["q75"])
        ax.fill_between(epoch, lower, upper, color=color, alpha=0.18, linewidth=0)
        ax.plot(epoch, median, color=color, lw=1.35, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("AE convergence across events")
    ax.legend(loc="upper right", handlelength=1.3)
    ax.text(0.02, 0.05, f"median and IQR, n={train.shape[0]} events", transform=ax.transAxes, color=COLORS["neutral_dark"])


def draw_k_summary(
    ax_curve: plt.Axes,
    ax_hist: plt.Axes,
    curves: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    best_k: Mapping[str, Mapping[str, int]],
    event: str,
) -> None:
    for current_event, k_values, wcss in curves:
        normalized = wcss / wcss[0]
        color = COLORS["blue"] if current_event == event else COLORS["neutral_mid"]
        alpha = 1.0 if current_event == event else 0.28
        width = 1.4 if current_event == event else 0.6
        ax_curve.plot(k_values, normalized, color=color, alpha=alpha, lw=width)
        chosen = best_k[current_event]["WCSS"]
        selected = np.where(k_values == chosen)[0]
        if len(selected):
            ax_curve.scatter(
                chosen,
                normalized[selected[0]],
                s=58 if current_event == event else 12,
                marker="*",
                color=COLORS["orange"],
                edgecolor=COLORS["black"] if current_event == event else "none",
                linewidth=0.35,
                alpha=1.0 if current_event == event else 0.65,
                zorder=4,
            )
    ax_curve.set_xlabel("Number of clusters, K")
    ax_curve.set_ylabel("Normalized WCSS")
    ax_curve.set_title("Event-specific elbow selection")
    ax_curve.text(
        0.04,
        0.08,
        f"{event}: K={best_k[event]['WCSS']}",
        transform=ax_curve.transAxes,
        color=COLORS["blue"],
        fontweight="bold",
    )

    counts = Counter(values["WCSS"] for values in best_k.values())
    k_all = np.arange(min(counts), max(counts) + 1)
    heights = np.asarray([counts.get(int(k), 0) for k in k_all])
    bars = ax_hist.bar(k_all, heights, width=0.72, color=COLORS["blue_soft"], edgecolor=COLORS["blue"], linewidth=0.6)
    for bar, value in zip(bars, heights):
        if value:
            ax_hist.text(bar.get_x() + bar.get_width() / 2, value + 0.4, str(value), ha="center", va="bottom", fontsize=6)
    ax_hist.set_xlabel("Selected K")
    ax_hist.set_ylabel("Events")
    ax_hist.set_title("K distribution")
    ax_hist.set_xticks(k_all)
    ax_hist.set_ylim(0, max(heights) * 1.18)


def draw_quality_summary(ax: plt.Axes, quality: Sequence[Mapping[str, object]], event: str) -> None:
    values = np.asarray([row["silhouette"] for row in quality], dtype=float)
    finite = np.isfinite(values)
    values = values[finite]
    rng = np.random.default_rng(20260812)
    x = rng.normal(1.0, 0.035, size=len(values))
    ax.boxplot(
        values,
        positions=[1],
        widths=0.34,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": COLORS["blue_soft"], "edgecolor": COLORS["blue"], "linewidth": 0.8},
        medianprops={"color": COLORS["black"], "linewidth": 1.0},
        whiskerprops={"color": COLORS["neutral_dark"], "linewidth": 0.8},
        capprops={"color": COLORS["neutral_dark"], "linewidth": 0.8},
    )
    ax.scatter(x, values, s=10, color=COLORS["neutral_dark"], alpha=0.55, edgecolor="none", rasterized=True)
    representative = next(row for row in quality if row["event"] == event)
    representative_value = float(representative["silhouette"])
    ax.scatter(1, representative_value, marker="*", s=72, color=COLORS["orange"], edgecolor=COLORS["black"], linewidth=0.4, zorder=5)
    ax.text(
        1.09,
        representative_value,
        f"{event}: {representative_value:.3f}",
        va="center",
        color=COLORS["orange"],
        fontweight="bold",
    )
    ax.set_xlim(0.72, 1.34)
    ax.set_xticks([1])
    ax.set_xticklabels(["44 events"])
    ax.set_ylabel("Silhouette coefficient")
    ax.set_title("Cluster separation in final DEC space")
    ax.text(
        0.03,
        0.96,
        "32-dimensional embedding",
        transform=ax.transAxes,
        color=COLORS["neutral_dark"],
        ha="left",
        va="top",
    )


def draw_tsne(ax: plt.Axes, data: Mapping[str, object], event: str) -> None:
    embedding = np.asarray(data["tsne"])
    labels = np.asarray(data["labels"])
    representatives = data["representatives"]
    for cluster_id in sorted(int(value) for value in np.unique(labels)):
        mask = labels == cluster_id
        color = cluster_color(cluster_id)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=5,
            alpha=0.48,
            color=color,
            edgecolor="none",
            rasterized=True,
        )
        center = np.median(embedding[mask], axis=0)
        label = ax.text(center[0], center[1], f"C{cluster_id}", ha="center", va="center", color="white", fontsize=6.5, fontweight="bold")
        label.set_path_effects([path_effects.withStroke(linewidth=1.6, foreground=color)])
    for cluster_id, index in representatives.items():
        ax.scatter(
            embedding[index, 0],
            embedding[index, 1],
            s=44,
            marker="*",
            color=cluster_color(cluster_id),
            edgecolor="white",
            linewidth=0.55,
            zorder=5,
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"Representative event ({event})")
    ax.text(0.02, 0.04, "stars: cluster medoids", transform=ax.transAxes, color=COLORS["neutral_dark"])


def _squeezed(array: np.ndarray) -> np.ndarray:
    result = np.squeeze(np.asarray(array)).astype(float)
    if result.ndim != 2:
        raise ValueError(f"Expected a 2-D spectrogram, got shape {result.shape}")
    return result


def selected_spectrograms(data: Mapping[str, object]) -> Tuple[List[int], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    representatives = data["representatives"]
    spectral = np.asarray(data["spectral"])
    reconstructed = np.asarray(data["reconstructed"])
    clusters = sorted(representatives)
    original = [_squeezed(spectral[representatives[cluster]]) for cluster in clusters]
    recon = [_squeezed(reconstructed[representatives[cluster]]) for cluster in clusters]
    residual = [np.abs(source - predicted) for source, predicted in zip(original, recon)]
    return clusters, original, recon, residual


def draw_spectrogram(
    ax: plt.Axes,
    spectrogram: np.ndarray,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    show_x: bool,
    show_y: bool,
) -> matplotlib.image.AxesImage:
    image = ax.imshow(
        spectrogram,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[*TIME_RANGE_S, *FREQUENCY_RANGE_HZ],
        rasterized=True,
    )
    ax.set_xticks([0, 30, 60] if show_x else [])
    ax.set_yticks([0, 25, 50] if show_y else [])
    if show_x:
        ax.set_xlabel("Time (s)", labelpad=1.5)
    if show_y:
        ax.set_ylabel("Frequency (Hz)", labelpad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.45)
        spine.set_color("0.35")
    return image


def cluster_frequency_profiles(
    data: Mapping[str, object],
) -> Tuple[np.ndarray, Dict[int, Dict[str, np.ndarray]]]:
    """Summarize time-averaged spectra within each DEC cluster."""
    spectral = np.asarray(data["spectral"])
    labels = np.asarray(data["labels"])
    frequencies = np.linspace(
        FREQUENCY_RANGE_HZ[0],
        FREQUENCY_RANGE_HZ[1],
        spectral.shape[1],
    )
    summaries: Dict[int, Dict[str, np.ndarray]] = {}
    for cluster_id in sorted(int(value) for value in np.unique(labels)):
        mask = labels == cluster_id
        # Each row is one one-minute window; average over time before the
        # across-window median/IQR so short, high-amplitude spikes do not
        # dominate the state-level profile.
        profiles = np.mean(
            spectral[mask, :, :, 0],
            axis=2,
            dtype=np.float32,
        )
        summaries[cluster_id] = {
            "median": np.median(profiles, axis=0),
            "q25": np.percentile(profiles, 25, axis=0),
            "q75": np.percentile(profiles, 75, axis=0),
        }
    return frequencies, summaries


def draw_frequency_profiles(ax: plt.Axes, data: Mapping[str, object]) -> None:
    frequencies, summaries = cluster_frequency_profiles(data)
    for cluster_id, summary in summaries.items():
        color = cluster_color(cluster_id)
        ax.fill_between(
            frequencies,
            summary["q25"],
            summary["q75"],
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        ax.plot(
            frequencies,
            summary["median"],
            color=color,
            lw=1.25,
            label=f"C{cluster_id}",
        )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean normalized amplitude")
    ax.set_title("Cluster frequency profiles (median and IQR)")
    ax.set_xlim(FREQUENCY_RANGE_HZ)
    ax.legend(ncol=min(4, len(summaries)), loc="upper right", columnspacing=0.9, handlelength=1.3)


def draw_main_state_plate(
    spec_axes: Sequence[plt.Axes],
    data: Mapping[str, object],
) -> None:
    clusters, original, _, _ = selected_spectrograms(data)
    labels = np.asarray(data["labels"])
    pooled = np.concatenate([item.ravel() for item in original])
    vmin = max(0.0, float(np.nanpercentile(pooled, 1)))
    vmax = float(np.nanpercentile(pooled, 99))
    if vmax <= vmin:
        vmax = vmin + 1.0
    image = None
    for column, cluster_id in enumerate(clusters):
        ax = spec_axes[column]
        count = int(np.sum(labels == cluster_id))
        image = draw_spectrogram(
            ax,
            original[column],
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            show_x=True,
            show_y=column == 0,
        )
        ax.set_title(
            f"C{cluster_id}  (n={count})",
            color=cluster_color(cluster_id),
            fontweight="bold",
            pad=2,
        )
        if column == 0:
            ax.text(
                -0.37,
                0.5,
                "Representative window",
                rotation=90,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontweight="bold",
            )
    add_panel_label(spec_axes[0], "e", x=-0.34, y=1.22)
    spec_axes[0].text(
        0,
        1.22,
        "Representative spectrograms within clusters",
        transform=spec_axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
    )


def build_main_figure(
    event: str,
    train: np.ndarray,
    validation: np.ndarray,
    k_curves: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    best_k: Mapping[str, Mapping[str, int]],
    quality: Sequence[Mapping[str, object]],
    event_data: Mapping[str, object],
) -> plt.Figure:
    """Build the double-column main-text figure (183 mm wide)."""
    clusters = sorted(event_data["representatives"])
    fig = plt.figure(figsize=(7.2, 6.65))
    outer = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.92, 1.0, 0.76],
        hspace=0.52,
        wspace=0.38,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    b_grid = outer[0, 1].subgridspec(1, 2, width_ratios=[1.45, 0.75], wspace=0.42)
    ax_b_curve = fig.add_subplot(b_grid[0, 0])
    ax_b_hist = fig.add_subplot(b_grid[0, 1])
    ax_c = fig.add_subplot(outer[1, 0])
    ax_d = fig.add_subplot(outer[1, 1])
    spec_grid = outer[2, :].subgridspec(1, len(clusters), wspace=0.08)
    spec_axes = [fig.add_subplot(spec_grid[0, column]) for column in range(len(clusters))]

    draw_ae_summary(ax_a, train, validation)
    add_panel_label(ax_a, "a")
    draw_k_summary(ax_b_curve, ax_b_hist, k_curves, best_k, event)
    add_panel_label(ax_b_curve, "b", x=-0.22)
    draw_quality_summary(ax_c, quality, event)
    add_panel_label(ax_c, "c")
    draw_tsne(ax_d, event_data, event)
    add_panel_label(ax_d, "d")
    draw_main_state_plate(spec_axes, event_data)

    fig.subplots_adjust(left=0.08, right=0.985, top=0.97, bottom=0.08)
    return fig


def draw_individual_ae(ax: plt.Axes, array: np.ndarray, *, color: str, title: str) -> None:
    epoch = np.arange(array.shape[1])
    for curve in array:
        valid = np.isfinite(curve)
        ax.plot(epoch[valid], positive_for_log(curve[valid]), color=COLORS["neutral_mid"], alpha=0.22, lw=0.45)
    summary = summarize_curves(array)
    ax.plot(epoch, positive_for_log(summary["median"]), color=color, lw=1.4, label="Median")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title(title)
    ax.legend(loc="upper right")


def draw_wcss_k_by_event(
    ax: plt.Axes,
    best_k: Mapping[str, Mapping[str, int]],
    representative_event: str,
) -> None:
    ordered_events = sorted(best_k, key=event_number)
    event_ids = np.asarray([event_number(event) for event in ordered_events])
    selected_k = np.asarray([best_k[event]["WCSS"] for event in ordered_events])

    ax.scatter(
        event_ids,
        selected_k,
        s=18,
        color=COLORS["blue_mid"],
        edgecolor="white",
        linewidth=0.35,
        alpha=0.85,
    )
    representative_id = event_number(representative_event)
    representative_k = best_k[representative_event]["WCSS"]
    ax.scatter(
        representative_id,
        representative_k,
        marker="*",
        s=82,
        color=COLORS["orange"],
        edgecolor=COLORS["black"],
        linewidth=0.4,
        zorder=5,
    )
    ax.text(
        representative_id + 1.2,
        representative_k,
        representative_event,
        color=COLORS["orange"],
        fontweight="bold",
        va="center",
    )
    unique_k = np.unique(selected_k)
    ax.set_xlabel("Event number")
    ax.set_ylabel("WCSS-selected K")
    ax.set_xticks([1, 10, 20, 30, 40, 44])
    ax.set_yticks(unique_k)
    ax.set_ylim(unique_k.min() - 0.45, unique_k.max() + 0.45)
    ax.set_title("WCSS-selected K across events")
    ax.text(0.97, 0.07, f"n={len(ordered_events)} events", transform=ax.transAxes, ha="right", color=COLORS["neutral_dark"])


def prepare_dec_loss(loss_rows: Sequence[Mapping[str, object]]) -> Dict[str, np.ndarray]:
    total = np.asarray([row["total_loss"] for row in loss_rows], dtype=float)
    cluster_raw = np.asarray([row["cluster_loss"] for row in loss_rows], dtype=float)
    reconstruction = np.asarray([row["reconstruction_loss"] for row in loss_rows], dtype=float)
    cluster_weighted = GAMMA * cluster_raw
    return {
        "step": np.arange(len(total)),
        "total": total,
        "cluster_raw": cluster_raw,
        "cluster_weighted": cluster_weighted,
        "reconstruction": reconstruction,
    }


def summarize_dec_losses(
    loss_rows_by_event: Mapping[str, Sequence[Mapping[str, object]]],
    n_progress_points: int = 101,
) -> Dict[str, object]:
    """Align unequal DEC logs by relative progress and summarize across events.

    The first CSV record is an initialization diagnostic recorded before the
    first optimization update. It is omitted so that 0% denotes the first actual
    DEC update for every event.
    """
    progress = np.linspace(0.0, 100.0, n_progress_points)
    event_names = sorted(loss_rows_by_event, key=event_number)
    components = ("total", "cluster_weighted", "reconstruction")
    aligned = {
        component: np.full((len(event_names), n_progress_points), np.nan, dtype=float)
        for component in components
    }
    original_steps = np.zeros(len(event_names), dtype=int)

    for row_index, event in enumerate(event_names):
        prepared = prepare_dec_loss(loss_rows_by_event[event])
        original_steps[row_index] = len(prepared["step"])
        start_index = 1 if len(prepared["step"]) > 1 else 0
        source_progress = np.linspace(0.0, 100.0, len(prepared["step"]) - start_index)
        for component in components:
            values = np.asarray(prepared[component][start_index:], dtype=float)
            aligned[component][row_index] = np.interp(progress, source_progress, values)

    summaries = {component: summarize_curves(values) for component, values in aligned.items()}
    return {
        "progress": progress,
        "event_names": event_names,
        "original_steps": original_steps,
        "aligned": aligned,
        "summaries": summaries,
    }


def draw_dec_loss(ax: plt.Axes, loss_summary: Mapping[str, object]) -> None:
    progress = np.asarray(loss_summary["progress"], dtype=float)
    aligned = loss_summary["aligned"]
    summaries = loss_summary["summaries"]

    # Light individual total-loss trajectories expose event-level heterogeneity.
    for values in aligned["total"]:
        ax.plot(progress, values, color=COLORS["neutral_mid"], alpha=0.10, lw=0.38, zorder=1)

    series = [
        ("total", COLORS["blue"], "Total"),
        ("cluster_weighted", COLORS["orange"], rf"{GAMMA:g} $\times$ KL"),
        ("reconstruction", COLORS["lilac"], "Reconstruction"),
    ]
    for component, color, label in series:
        summary = summaries[component]
        ax.fill_between(
            progress,
            summary["q25"],
            summary["q75"],
            color=color,
            alpha=0.14,
            linewidth=0,
            zorder=2,
        )
        ax.plot(progress, summary["median"], color=color, lw=1.35, label=label, zorder=3)

    ax.set_xlabel("Normalized training progress (%)")
    ax.set_ylabel("Weighted loss")
    ax.set_title("DEC convergence across events")
    ax.legend(loc="upper right", handlelength=1.2)
    ax.text(
        0.03,
        0.94,
        r"$L_{total}=0.1L_{KL}+L_{recon}$",
        transform=ax.transAxes,
        color=COLORS["neutral_dark"],
        ha="left",
        va="top",
    )
    ax.text(
        0.97,
        0.52,
        f"median and IQR, n={len(loss_summary['event_names'])} events",
        transform=ax.transAxes,
        color=COLORS["neutral_dark"],
        ha="right",
        va="bottom",
        fontsize=8.2,
    )


def pearson_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float).ravel()
    second = np.asarray(second, dtype=float).ravel()
    first = first - np.mean(first)
    second = second - np.mean(second)
    denominator = np.sqrt(np.sum(first * first) * np.sum(second * second))
    if denominator <= 0:
        return np.nan
    return float(np.sum(first * second) / denominator)


def reconstruction_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    residual = original - reconstructed
    original_variance = float(np.var(original))
    return {
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "pixel_correlation": pearson_correlation(original, reconstructed),
        "frequency_profile_correlation": pearson_correlation(
            np.mean(original, axis=1),
            np.mean(reconstructed, axis=1),
        ),
        "time_profile_correlation": pearson_correlation(
            np.mean(original, axis=0),
            np.mean(reconstructed, axis=0),
        ),
        "variance_retention": (
            float(np.var(reconstructed)) / original_variance
            if original_variance > 0
            else np.nan
        ),
    }


def draw_reconstruction_audit_plate(
    fig: plt.Figure,
    original_axes: Sequence[plt.Axes],
    reconstruction_axes: Sequence[plt.Axes],
    residual_axes: Sequence[plt.Axes],
    amplitude_colorbar_ax: plt.Axes,
    residual_colorbar_ax: plt.Axes,
    data: Mapping[str, object],
    panel_label: str = "e",
) -> None:
    clusters, original, reconstructed, residual = selected_spectrograms(data)
    labels = np.asarray(data["labels"])
    amplitude_pool = np.concatenate(
        [values.ravel() for values in original + reconstructed]
    )
    amplitude_vmin = max(0.0, float(np.nanpercentile(amplitude_pool, 1)))
    amplitude_vmax = float(np.nanpercentile(amplitude_pool, 99))
    residual_vmax = float(
        np.nanpercentile(
            np.concatenate([values.ravel() for values in residual]),
            99,
        )
    )

    amplitude_image = None
    residual_image = None
    for column, cluster_id in enumerate(clusters):
        metrics = reconstruction_metrics(original[column], reconstructed[column])
        count = int(np.sum(labels == cluster_id))
        amplitude_image = draw_spectrogram(
            original_axes[column],
            original[column],
            cmap="magma",
            vmin=amplitude_vmin,
            vmax=amplitude_vmax,
            show_x=False,
            show_y=column == 0,
        )
        draw_spectrogram(
            reconstruction_axes[column],
            reconstructed[column],
            cmap="magma",
            vmin=amplitude_vmin,
            vmax=amplitude_vmax,
            show_x=False,
            show_y=column == 0,
        )
        residual_image = draw_spectrogram(
            residual_axes[column],
            residual[column],
            cmap="viridis",
            vmin=0.0,
            vmax=residual_vmax,
            show_x=True,
            show_y=column == 0,
        )
        residual_axes[column].set_xlabel("")
        original_axes[column].set_title(
            f"C{cluster_id}  (n={count})",
            color=cluster_color(cluster_id),
            fontweight="bold",
            pad=2,
        )
        reconstruction_axes[column].text(
            0.03,
            0.95,
            rf"$r_{{freq}}$={metrics['frequency_profile_correlation']:.2f}",
            transform=reconstruction_axes[column].transAxes,
            ha="left",
            va="top",
            fontsize=5.7,
            color="white",
            bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.45, "pad": 1.0},
        )
        residual_axes[column].set_title(
            f"RMSE={metrics['rmse']:.3f}",
            fontsize=6.2,
            pad=2,
        )

    residual_axes[-1].text(
        0.97,
        0.04,
        "Time (s)",
        transform=residual_axes[-1].transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="white",
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.35, "pad": 0.8},
    )

    row_specs = (
        (original_axes[0], "Original"),
        (reconstruction_axes[0], "Reconstruction"),
        (residual_axes[0], "|Residual|"),
    )
    for ax, label in row_specs:
        ax.text(
            -0.34,
            0.5,
            label,
            rotation=90,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontweight="bold",
        )
    add_panel_label(original_axes[0], panel_label, x=-0.34, y=1.25)
    original_axes[0].text(
        0,
        1.25,
        "Spectrogram reconstruction audit",
        transform=original_axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
    )
    amplitude_colorbar = fig.colorbar(
        amplitude_image,
        cax=amplitude_colorbar_ax,
        orientation="horizontal",
    )
    amplitude_colorbar.set_label("Normalized amplitude", labelpad=2)
    residual_colorbar = fig.colorbar(
        residual_image,
        cax=residual_colorbar_ax,
        orientation="horizontal",
    )
    residual_colorbar.set_label("|Residual|", labelpad=2)


def build_supplementary_figure(
    event: str,
    train: np.ndarray,
    validation: np.ndarray,
    best_k: Mapping[str, Mapping[str, int]],
    event_data: Mapping[str, object],
    dec_loss_summary: Mapping[str, object],
) -> plt.Figure:
    """Build the supplementary model-diagnostic figure."""
    clusters = sorted(event_data["representatives"])
    fig = plt.figure(figsize=(7.2, 9.0))
    outer = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.0, 1.78],
        hspace=0.48,
        wspace=0.38,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[1, 0])
    ax_d = fig.add_subplot(outer[1, 1])
    audit_grid = outer[2, :].subgridspec(
        4,
        len(clusters),
        height_ratios=[1.0, 1.0, 1.0, 0.09],
        hspace=0.46,
        wspace=0.08,
    )
    original_axes = [fig.add_subplot(audit_grid[0, column]) for column in range(len(clusters))]
    reconstruction_axes = [fig.add_subplot(audit_grid[1, column]) for column in range(len(clusters))]
    residual_axes = [fig.add_subplot(audit_grid[2, column]) for column in range(len(clusters))]
    split = max(1, len(clusters) // 2)
    amplitude_colorbar_ax = fig.add_subplot(audit_grid[3, :split])
    residual_colorbar_ax = fig.add_subplot(audit_grid[3, split:])

    draw_individual_ae(ax_a, train, color=COLORS["blue"], title="AE training loss by event")
    add_panel_label(ax_a, "a")
    draw_individual_ae(ax_b, validation, color=COLORS["orange"], title="AE validation loss by event")
    add_panel_label(ax_b, "b")
    draw_wcss_k_by_event(ax_c, best_k, event)
    add_panel_label(ax_c, "c")
    draw_dec_loss(ax_d, dec_loss_summary)
    add_panel_label(ax_d, "d")
    draw_reconstruction_audit_plate(
        fig,
        original_axes,
        reconstruction_axes,
        residual_axes,
        amplitude_colorbar_ax,
        residual_colorbar_ax,
        event_data,
        panel_label="e",
    )

    fig.subplots_adjust(left=0.08, right=0.985, top=0.975, bottom=0.065)
    return fig


def save_submission_figure(
    fig: plt.Figure,
    output_base: Path,
    formats: Iterable[str],
    tiff_dpi: int,
) -> List[Path]:
    """Save editable vector outputs plus high-resolution submission rasters."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for requested in formats:
        extension = requested.lower().lstrip(".")
        if extension == "tif":
            extension = "tiff"
        path = output_base.with_suffix(f".{extension}")
        kwargs: Dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if extension == "png":
            kwargs["dpi"] = 300
        elif extension == "tiff":
            kwargs["dpi"] = tiff_dpi
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_source_data(
    output_dir: Path,
    events: Sequence[str],
    train: np.ndarray,
    validation: np.ndarray,
    best_k: Mapping[str, Mapping[str, int]],
    quality: Sequence[Mapping[str, object]],
    event_data: Mapping[str, object],
    dec_loss_summary: Mapping[str, object],
) -> List[Path]:
    source_dir = output_dir / "source_data"
    train_summary = summarize_curves(train)
    val_summary = summarize_curves(validation)
    ae_rows = []
    for epoch in range(train.shape[1]):
        ae_rows.append(
            {
                "epoch": epoch,
                "training_median": train_summary["median"][epoch],
                "training_q25": train_summary["q25"][epoch],
                "training_q75": train_summary["q75"][epoch],
                "validation_median": val_summary["median"][epoch],
                "validation_q25": val_summary["q25"][epoch],
                "validation_q75": val_summary["q75"][epoch],
                "n_events": int(min(train_summary["n"][epoch], val_summary["n"][epoch])),
            }
        )
    ae_path = source_dir / "Fig2a_AE_summary.csv"
    write_csv(ae_path, ae_rows, list(ae_rows[0]))

    k_rows = [{"event": event, **best_k[event]} for event in sorted(best_k, key=event_number)]
    k_path = source_dir / "Fig2b_selected_K.csv"
    write_csv(k_path, k_rows, ["event", *K_METRIC_COLUMNS.keys()])

    quality_path = source_dir / "Fig2c_cluster_quality.csv"
    write_csv(
        quality_path,
        quality,
        ["event", "n_samples", "n_clusters", "silhouette", "calinski_harabasz", "davies_bouldin"],
    )

    labels = np.asarray(event_data["labels"])
    representatives = event_data["representatives"]
    x_files = np.asarray(event_data["x_file"])
    clusters, original, reconstructed, _ = selected_spectrograms(event_data)
    medoid_metrics = {
        cluster_id: reconstruction_metrics(source, predicted)
        for cluster_id, source, predicted in zip(clusters, original, reconstructed)
    }
    cluster_rows = []
    for cluster_id in sorted(representatives):
        index = representatives[cluster_id]
        cluster_rows.append(
            {
                "cluster": cluster_id,
                "n_windows": int(np.sum(labels == cluster_id)),
                "medoid_index": index,
                "medoid_source_id": Path(str(x_files[index])).name,
                **medoid_metrics[cluster_id],
            }
        )
    cluster_path = source_dir / "Fig2de_representative_event_clusters.csv"
    write_csv(cluster_path, cluster_rows, list(cluster_rows[0]))

    frequencies, profile_summaries = cluster_frequency_profiles(event_data)
    profile_rows = []
    for frequency_index, frequency in enumerate(frequencies):
        row: Dict[str, object] = {"frequency_hz": float(frequency)}
        for cluster_id, summary in profile_summaries.items():
            row[f"C{cluster_id}_median"] = float(summary["median"][frequency_index])
            row[f"C{cluster_id}_q25"] = float(summary["q25"][frequency_index])
            row[f"C{cluster_id}_q75"] = float(summary["q75"][frequency_index])
        profile_rows.append(row)
    profile_path = source_dir / "Fig2e_cluster_frequency_profiles.csv"
    write_csv(profile_path, profile_rows, list(profile_rows[0]))

    reconstruction_rows = [
        {"cluster": cluster_id, **medoid_metrics[cluster_id]}
        for cluster_id in clusters
    ]
    reconstruction_path = source_dir / "FigS2e_reconstruction_metrics.csv"
    write_csv(
        reconstruction_path,
        reconstruction_rows,
        list(reconstruction_rows[0]),
    )

    loss_rows = []
    progress = np.asarray(dec_loss_summary["progress"])
    summaries = dec_loss_summary["summaries"]
    for index, progress_value in enumerate(progress):
        row: Dict[str, object] = {"training_progress_percent": progress_value}
        for component in ("total", "cluster_weighted", "reconstruction"):
            summary = summaries[component]
            row[f"{component}_median"] = summary["median"][index]
            row[f"{component}_q25"] = summary["q25"][index]
            row[f"{component}_q75"] = summary["q75"][index]
        row["n_events"] = len(dec_loss_summary["event_names"])
        loss_rows.append(row)
    loss_path = source_dir / "FigS2d_DEC_loss_summary.csv"
    write_csv(loss_path, loss_rows, list(loss_rows[0]))

    individual_loss_rows = []
    event_names = dec_loss_summary["event_names"]
    original_steps = np.asarray(dec_loss_summary["original_steps"])
    aligned = dec_loss_summary["aligned"]
    for event_index, event_name in enumerate(event_names):
        for progress_index, progress_value in enumerate(progress):
            individual_loss_rows.append(
                {
                    "event": event_name,
                    "original_training_steps": int(original_steps[event_index]),
                    "training_progress_percent": progress_value,
                    "total": aligned["total"][event_index, progress_index],
                    "cluster_weighted": aligned["cluster_weighted"][event_index, progress_index],
                    "reconstruction": aligned["reconstruction"][event_index, progress_index],
                }
            )
    individual_loss_path = source_dir / "FigS2d_DEC_loss_by_event.csv"
    write_csv(individual_loss_path, individual_loss_rows, list(individual_loss_rows[0]))
    return [
        ae_path,
        k_path,
        quality_path,
        cluster_path,
        profile_path,
        loss_path,
        individual_loss_path,
        reconstruction_path,
    ]


def validate_inputs(event: str) -> None:
    if not re.fullmatch(r"event\d+", event):
        raise ValueError("--event must have the form event5")
    required = [AE_DIR, DEC_DIR, K_DIR, BEST_K_FILE]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(str(path) for path in missing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default="event5", help="Representative event (default: event5)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for figures and source-data tables")
    parser.add_argument("--only", choices=["both", "main", "supplement"], default="both", help="Which figure set to build")
    parser.add_argument("--formats", nargs="+", default=["svg", "pdf", "tiff", "png"], choices=["svg", "pdf", "tiff", "png"], help="Output formats")
    parser.add_argument("--tiff-dpi", type=int, default=600, help="TIFF resolution (default: 600 dpi)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_inputs(args.event)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    events, train, validation = load_ae_curves()
    k_curves, best_k = load_k_data()
    npz_paths = sorted((DEC_DIR / "all_parement").glob("event*_all_parement.npz"), key=event_number)
    quality = compute_cluster_quality(npz_paths)
    event_data = load_representative_event(args.event)
    dec_loss_rows = load_dec_loss_curves()
    dec_loss_summary = summarize_dec_losses(dec_loss_rows)

    outputs: List[Path] = []
    loss_data: Dict[str, np.ndarray]
    if args.only in {"both", "main"}:
        main_figure = build_main_figure(
            args.event,
            train,
            validation,
            k_curves,
            best_k,
            quality,
            event_data,
        )
        outputs.extend(save_submission_figure(main_figure, args.output_dir / "Fig2_model_result_main", args.formats, args.tiff_dpi))

    if args.only in {"both", "supplement"}:
        supplementary_figure = build_supplementary_figure(
            args.event,
            train,
            validation,
            best_k,
            event_data,
            dec_loss_summary,
        )
        outputs.extend(save_submission_figure(supplementary_figure, args.output_dir / "FigS2_model_diagnostics", args.formats, args.tiff_dpi))

    source_files = export_source_data(
        args.output_dir,
        events,
        train,
        validation,
        best_k,
        quality,
        event_data,
        dec_loss_summary,
    )

    print(
        f"Loaded {len(events)} AE logs, {len(k_curves)} WCSS curves, "
        f"{len(quality)} DEC result files, and {len(dec_loss_rows)} DEC loss logs."
    )
    print(f"Representative event: {args.event}; K={event_data['n_clusters']}; windows={len(event_data['labels'])}.")
    print("Saved figures:")
    for path in outputs:
        print(path)
    print("Saved source data:")
    for path in source_files:
        print(path)


if __name__ == "__main__":
    main()
