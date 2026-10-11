# -*- coding: utf-8 -*-
'''
@Time    : 2026/5/20 16:53
@Author  : HuangZhiyong
@email   : 1524338616@qq.com
@File    : 06_all_kneed_kmeans.py
@Function: {}:
'''
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


# =========================================================
# 1. Path configuration
# =========================================================
INPUT_NPZ = Path(r".\result_all_event\all_parameters\all_events_ACE_model_light.npz")

OUTPUT_DIR = Path(r".\result_all_event\kmeans_elbow")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 2. Clustering configuration
# =========================================================
K_MIN = 2
K_MAX = 20

RANDOM_STATE = 42
N_INIT = 20

# 是否对 features 标准化
STANDARDIZE_FEATURES = True

# 是否先 PCA 降维后再聚类
# 如果 features 维度很高，建议 True
USE_PCA_FOR_CLUSTERING = False

# 如果 USE_PCA_FOR_CLUSTERING=True，则使用这个 PCA 方差保留比例
PCA_VARIANCE_RATIO = 0.95

# 最终是否自动用 knee 方法估计最佳 K
AUTO_DETECT_ELBOW = True

# 如果自动肘部法失败，则默认使用这个 K
DEFAULT_K = 3

# 计算 silhouette 时最多抽样多少点，避免太慢
MAX_SILHOUETTE_SAMPLES = 20000


# =========================================================
# 3. Helper functions
# =========================================================
def load_features(npz_path: Path):
    """Load latent features and metadata."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Input file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    if "features" not in data:
        raise KeyError(f"'features' not found in {npz_path}")

    features = data["features"]

    wave_file = data["wave_file"] if "wave_file" in data else np.array([])
    event = data["event"] if "event" in data else np.array([])

    features = np.asarray(features, dtype=np.float32)

    print("Loaded features:")
    print("features shape:", features.shape)
    print("features dtype:", features.dtype)
    print("NaN count:", np.isnan(features).sum())
    print("Inf count:", np.isinf(features).sum())

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features, wave_file, event


def prepare_features(features: np.ndarray):
    """
    Standardize features and optionally apply PCA.
    """
    x = features.copy()

    scaler = None
    pca = None

    if STANDARDIZE_FEATURES:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    if USE_PCA_FOR_CLUSTERING:
        pca = PCA(n_components=PCA_VARIANCE_RATIO, random_state=RANDOM_STATE)
        x = pca.fit_transform(x)

        print("\nPCA applied.")
        print("PCA output shape:", x.shape)
        print("Cumulative explained variance:", np.sum(pca.explained_variance_ratio_))

    return x, scaler, pca


def detect_elbow_by_max_distance(k_values, wcss_values):
    """
    Detect elbow point by maximum distance to the straight line
    connecting the first and last WCSS points.

    This is a simple knee-detection method.
    """
    k_values = np.asarray(k_values, dtype=float)
    wcss_values = np.asarray(wcss_values, dtype=float)

    if len(k_values) < 3:
        return int(k_values[0])

    # Normalize x and y to [0, 1]
    x = (k_values - k_values.min()) / (k_values.max() - k_values.min())
    y = (wcss_values - wcss_values.min()) / (wcss_values.max() - wcss_values.min())

    # line from first to last point
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    distances = []

    for xi, yi in zip(x, y):
        p = np.array([xi, yi])
        distance = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distances.append(distance)

    distances = np.asarray(distances)

    elbow_idx = int(np.argmax(distances))
    best_k = int(k_values[elbow_idx])

    return best_k


def compute_clustering_scores(x, labels):
    """
    Compute internal clustering scores.
    """
    scores = {
        "silhouette": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
    }

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return scores

    x_eval = x
    labels_eval = labels

    if len(x) > MAX_SILHOUETTE_SAMPLES:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(x), size=MAX_SILHOUETTE_SAMPLES, replace=False)
        x_eval = x[idx]
        labels_eval = labels[idx]

    try:
        scores["silhouette"] = float(silhouette_score(x_eval, labels_eval))
    except Exception as exc:
        print(f"[WARN] silhouette_score failed: {exc}")

    try:
        scores["calinski_harabasz"] = float(calinski_harabasz_score(x_eval, labels_eval))
    except Exception as exc:
        print(f"[WARN] calinski_harabasz_score failed: {exc}")

    try:
        scores["davies_bouldin"] = float(davies_bouldin_score(x_eval, labels_eval))
    except Exception as exc:
        print(f"[WARN] davies_bouldin_score failed: {exc}")

    return scores


def run_elbow_analysis(x):
    """
    Run KMeans for different K and compute WCSS.
    """
    records = []

    for k in range(K_MIN, K_MAX + 1):
        print(f"Running KMeans for K={k}...")

        kmeans = KMeans(
            n_clusters=k,
            n_init=N_INIT,
            random_state=RANDOM_STATE,
        )

        labels = kmeans.fit_predict(x)

        wcss = float(kmeans.inertia_)
        scores = compute_clustering_scores(x, labels)

        records.append(
            {
                "K": k,
                "WCSS": wcss,
                "silhouette": scores["silhouette"],
                "calinski_harabasz": scores["calinski_harabasz"],
                "davies_bouldin": scores["davies_bouldin"],
            }
        )

        print(
            f"K={k}, WCSS={wcss:.4g}, "
            f"silhouette={scores['silhouette']:.4g}, "
            f"CH={scores['calinski_harabasz']:.4g}, "
            f"DB={scores['davies_bouldin']:.4g}"
        )

    result_df = pd.DataFrame(records)

    if AUTO_DETECT_ELBOW:
        best_k = detect_elbow_by_max_distance(
            result_df["K"].values,
            result_df["WCSS"].values,
        )
    else:
        best_k = DEFAULT_K

    print("\nEstimated best K by elbow method:", best_k)

    return result_df, best_k


def plot_elbow_curve(result_df: pd.DataFrame, best_k: int):
    """
    Plot WCSS elbow curve.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(
        result_df["K"],
        result_df["WCSS"],
        marker="o",
        linewidth=2,
    )

    ax.axvline(
        best_k,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Estimated elbow K={best_k}",
    )

    ax.set_xlabel("Number of clusters K")
    ax.set_ylabel("WCSS / Inertia")
    ax.set_title("Elbow method for KMeans clustering")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    png_path = OUTPUT_DIR / "elbow_wcss_curve.png"
    pdf_path = OUTPUT_DIR / "elbow_wcss_curve.pdf"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved elbow curve to: {png_path}")
    print(f"Saved elbow curve to: {pdf_path}")


def run_final_kmeans(x, best_k: int):
    """
    Run final KMeans with selected best K.
    """
    kmeans = KMeans(
        n_clusters=best_k,
        n_init=N_INIT,
        random_state=RANDOM_STATE,
    )

    labels = kmeans.fit_predict(x)
    distances = kmeans.transform(x)

    return kmeans, labels, distances


def plot_pca_scatter(x, labels, best_k: int):
    """
    Plot PCA 2D scatter of final clustering result.
    """
    pca_vis = PCA(n_components=2, random_state=RANDOM_STATE)
    x_2d = pca_vis.fit_transform(x)

    fig, ax = plt.subplots(figsize=(8, 6.5))

    scatter = ax.scatter(
        x_2d[:, 0],
        x_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=8,
        alpha=0.65,
    )

    plt.colorbar(scatter, ax=ax, label="Cluster label")

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"KMeans clustering result, K={best_k}")
    ax.grid(alpha=0.25)

    fig.tight_layout()

    save_path = OUTPUT_DIR / f"pca_kmeans_scatter_K{best_k}.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Saved PCA scatter to: {save_path}")


def save_final_outputs(
    features,
    x_used,
    wave_file,
    event,
    labels,
    distances,
    best_k,
    elbow_df,
):
    """
    Save final clustering outputs.
    """
    # Save elbow table
    elbow_path = OUTPUT_DIR / "elbow_wcss_results.xlsx"
    elbow_df.to_excel(elbow_path, index=False)

    print(f"Saved elbow WCSS results to: {elbow_path}")

    # Save final clustering npz
    final_npz_path = OUTPUT_DIR / f"kmeans_global_clustering_K{best_k}.npz"

    np.savez(
        final_npz_path,
        features=features,
        features_used=x_used,
        wave_file=wave_file,
        event=event,
        labels=labels,
        distances=distances,
        best_k=best_k,
        elbow_table=elbow_df.to_dict(orient="list"),
    )

    print(f"Saved final clustering npz to: {final_npz_path}")

    # Save summary table
    summary_rows = []

    for cluster_id in np.unique(labels):
        count = int(np.sum(labels == cluster_id))
        ratio = count / len(labels)

        summary_rows.append(
            {
                "cluster": int(cluster_id),
                "sample_count": count,
                "sample_ratio": ratio,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("cluster")

    summary_path = OUTPUT_DIR / "kmeans_global_cluster_summary.xlsx"
    summary_df.to_excel(summary_path, index=False)

    print(f"Saved cluster summary to: {summary_path}")

    # Save event-cluster count table
    if len(event) == len(labels):
        event_cluster_df = pd.DataFrame(
            {
                "event": event,
                "cluster": labels,
            }
        )

        event_cluster_count = (
            event_cluster_df.groupby(["event", "cluster"])
            .size()
            .reset_index(name="count")
        )

        event_cluster_count_path = OUTPUT_DIR / "event_cluster_count_summary.xlsx"
        event_cluster_count.to_excel(event_cluster_count_path, index=False)

        print(f"Saved event-cluster count summary to: {event_cluster_count_path}")


# =========================================================
# 4. Main
# =========================================================
def main():
    features, wave_file, event = load_features(INPUT_NPZ)

    x_used, scaler, pca = prepare_features(features)

    elbow_df, best_k = run_elbow_analysis(x_used)

    plot_elbow_curve(elbow_df, best_k)

    final_kmeans, labels, distances = run_final_kmeans(x_used, best_k)

    plot_pca_scatter(x_used, labels, best_k)

    save_final_outputs(
        features=features,
        x_used=x_used,
        wave_file=wave_file,
        event=event,
        labels=labels,
        distances=distances,
        best_k=best_k,
        elbow_df=elbow_df,
    )

    print("\nDone.")
    print(f"Best K estimated by elbow method: {best_k}")
    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()