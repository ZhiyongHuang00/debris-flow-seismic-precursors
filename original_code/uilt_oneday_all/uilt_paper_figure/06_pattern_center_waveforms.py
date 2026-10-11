from pathlib import Path
import csv

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from obspy import read


PROJECT_ROOT = Path(r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben")
RESULT_DIR = next(
    PROJECT_ROOT.glob("uilt_oneday_all/03visual/**/kmeans_clustering_from_matched_precursor_features")
)
RESULT_CSV = RESULT_DIR / "kmeans_clustering_results.csv"
NPZ_DIR = PROJECT_ROOT / "uilt_oneday_all" / "02clustering" / "result" / "DEC_result" / "all_parement"
SEGMENT_ROOT = PROJECT_ROOT / "uilt_oneday_all" / "01data_preprocess"
OUT_DIR = PROJECT_ROOT / "uilt_oneday_all" / "uilt_paper_figure" / "Fig6_pattern_center_waveforms"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PER_PATTERN = None
MODE_NAMES = {0: "Pattern I", 1: "Pattern II"}
MODE_COLORS = {0: "#1F77B4", 1: "#E76F51"}

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.linewidth": 0.8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def read_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_int(row, key):
    return int(float(row[key]))


def to_float(row, key):
    return float(row[key])


def npz_path(event):
    for name in [f"{event}_all_parement.npz", f"{event}_all_parameter.npz"]:
        p = NPZ_DIR / name
        if p.exists():
            return p
    return next(NPZ_DIR.glob(f"{event}*.npz"))


def resolve_sac_path(x_file):
    p = Path(str(x_file).replace("\\", "/"))
    if p.is_absolute() and p.exists():
        return p
    return SEGMENT_ROOT / str(x_file).replace(".\\", "").replace("\\", "/")


def robust_normalize(x):
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
    sac_file = resolve_sac_path(z["x_file"][center_index])

    tr = read(str(sac_file))[0]
    raw = np.asarray(tr.data, dtype=float)
    raw = raw - np.nanmedian(raw)
    waveform = robust_normalize(raw)
    time = np.arange(waveform.size) / float(tr.stats.sampling_rate)
    return {
        "event": event.replace("event", "E"),
        "cluster": f"C{cluster_id}",
        "time_before_h": -24 + center_index / 60,
        "time": time,
        "waveform": waveform,
        "rms": float(np.sqrt(np.nanmean(raw ** 2))),
        "peak": float(np.nanmax(np.abs(raw))),
    }


def select_rows(rows, mode):
    mode_rows = [r for r in rows if to_int(r, "kmeans_cluster") == mode]
    mode_rows = sorted(mode_rows, key=lambda r: int(r["event"].replace("event", "")))
    if N_PER_PATTERN is None:
        return mode_rows
    if len(mode_rows) <= N_PER_PATTERN:
        return mode_rows
    picks = np.linspace(0, len(mode_rows) - 1, N_PER_PATTERN).round().astype(int)
    return [mode_rows[i] for i in picks]


rows = read_rows(RESULT_CSV)
examples = {
    mode: [center_waveform(row) for row in select_rows(rows, mode)]
    for mode in [0, 1]
}

max_rows = max(len(items) for items in examples.values())
fig_height = max(5.6, 0.28 * max_rows + 1.2)
fig = plt.figure(figsize=(8.6, fig_height))
gs = fig.add_gridspec(
    2, 3,
    width_ratios=[1.0, 1.0, 0.43],
    height_ratios=[1.0, 1.0],
    wspace=0.28,
    hspace=0.35,
)
axes = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])]
metric_axes = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])]
fig.suptitle("Representative center waveforms of two precursor patterns", x=0.5, y=0.99)

for ax, mode in zip(axes[:2], [0, 1]):
    color = MODE_COLORS[mode]
    items = examples[mode]
    spacing = 1.32
    offsets = np.arange(len(items))[::-1] * spacing

    for item, offset in zip(items, offsets):
        mask = item["time"] <= 60
        ax.plot(item["time"][mask], item["waveform"][mask] * 0.42 + offset, color=color, lw=0.55)
        ax.axhline(offset, color="0.86", lw=0.45, zorder=0)

    ax.set_title(f"{MODE_NAMES[mode]} (n={sum(to_int(r, 'kmeans_cluster') == mode for r in rows)})")
    ax.set_xlim(0, 64 if mode == 1 else 60)
    ax.set_ylim(-spacing, offsets[0] + spacing)
    ax.set_xlabel("Time (s)")
    ax.set_yticks(offsets)
    labels = [f"{item['event']} {item['cluster']} ({item['time_before_h']:.1f} h)" for item in items]
    ax.tick_params(axis="y", length=0, pad=2)
    if mode == 1:
        ax.set_yticklabels([])
        for label, offset in zip(labels, offsets):
            ax.text(63.4, offset, label, ha="right", va="center", fontsize=5.0, color="0.20")
    else:
        ax.set_yticklabels(labels)
        ax.tick_params(axis="y", labelsize=5.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_metric_box(ax, key, title, show_legend=False):
    rng = np.random.default_rng(20260701)
    for mode, pos in zip([0, 1], [0, 1]):
        vals = np.array([item[key] for item in examples[mode]], dtype=float)
        ax.boxplot(
            vals, positions=[pos], widths=0.48, patch_artist=True, showfliers=False,
            boxprops={"facecolor": MODE_COLORS[mode], "alpha": 0.18, "edgecolor": MODE_COLORS[mode], "lw": 0.8},
            medianprops={"color": MODE_COLORS[mode], "lw": 1.2},
            whiskerprops={"color": MODE_COLORS[mode], "lw": 0.8},
            capprops={"color": MODE_COLORS[mode], "lw": 0.8},
        )
        ax.scatter(
            np.full(vals.size, pos) + rng.normal(0, 0.035, vals.size), vals,
            s=18, color=MODE_COLORS[mode], edgecolor="white", linewidth=0.35, alpha=0.85,
            label=MODE_NAMES[mode],
        )

    ax.set_title(title, pad=2)
    ax.set_ylabel("Amplitude")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["I", "II"])
    ax.set_xlim(-0.55, 1.55)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(axis="y", color="0.90", lw=0.5)
    if show_legend:
        ax.legend(loc="upper left", fontsize=5.8, handlelength=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


plot_metric_box(metric_axes[0], "rms", "RMS", show_legend=True)
plot_metric_box(metric_axes[1], "peak", "Peak amplitude")
metric_axes[1].set_xlabel("Pattern")

for suffix in ["png", "pdf", "svg"]:
    fig.savefig(OUT_DIR / f"figure6_pattern_center_waveforms.{suffix}", dpi=400, bbox_inches="tight", pad_inches=0.04)

plt.close(fig)
print(OUT_DIR / "figure6_pattern_center_waveforms.png")
