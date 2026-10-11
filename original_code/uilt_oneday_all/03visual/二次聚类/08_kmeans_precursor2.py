# -*- coding: utf-8 -*-
'''
@Time    : 2026/6/13 21:49
@Author  : HuangZhiyong
@email   : 1524338616@qq.com
@File    : 08_kmeans_precursor2.py
@Function: {}:
'''
# -*- coding: utf-8 -*-
# """
# KMeans 二次聚类：从 matched_precursor_features_with_distance.xlsx 中提取前兆簇中心特征后聚类
#
# 输入：
#     D:\Pycharm_Project\Find_precursor_signs\ILLgarben\uilt_oneday_all\03visual\二次聚类\matched_precursor_features\matched_precursor_features_with_distance.xlsx
#
# 流程：
#     1. 读取前兆簇样本特征表
#     2. 按 event + cluster_id 计算每个事件前兆簇中心特征
#     3. 标准化、PCA 降维
#     4. 肘部法确定 KMeans 最优 K
#     5. KMeans 聚类
#     6. PCA / t-SNE 可视化
#     7. 簇中心热力图
#     8. 随机森林代理解释特征重要性
#     9. SHAP 解释不同模式主导参数
#     10. 关键特征箱型图
#     11. 保存 Excel、CSV、图片
# """

from pathlib import Path
import re
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import seaborn as sns
except Exception:
    sns = None

try:
    from kneed import KneeLocator
except Exception:
    KneeLocator = None

try:
    import shap
except Exception:
    shap = None


warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 1. 参数设置
# ============================================================

PROJECT_ROOT = Path(r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben")

SECONDARY_DIR = PROJECT_ROOT / r"uilt_oneday_all\03visual\二次聚类"

INPUT_FILE = SECONDARY_DIR / r"matched_precursor_features\matched_precursor_features_with_distance.xlsx"

OUT_DIR = SECONDARY_DIR / "kmeans_clustering_from_matched_precursor_features"

OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
K_RANGE = range(1, 7)
PCA_VARIANCE = 0.95
KMEANS_N_INIT = 100

REMOVE_HIGH_CORR = True
CORR_THRESHOLD = 0.95

TOP_N_FEATURES_FOR_BOXPLOT = 20
SHOW_OUTLIERS = False

TOP_N_FEATURES_FOR_SHAP = 20
TOP_N_FEATURES_FOR_SHAP_DEPENDENCE = 6

print("INPUT_FILE =", INPUT_FILE)
print("OUT_DIR =", OUT_DIR)


# ============================================================
# 2. 工具函数
# ============================================================

def natural_event_key(value):
    text = str(value)
    match = re.search(r"(\d+)$", text)
    return int(match.group(1)) if match else 10**9


def save_fig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def write_excel(path, sheets):
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=str(sheet_name)[:31], index=False)
        return path

    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")

        with pd.ExcelWriter(fallback, engine="openpyxl") as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=str(sheet_name)[:31], index=False)

        return fallback


def feature_group(name):
    key = name.lower()

    if any(t in key for t in ["rectilinearity", "azimuth", "dip", "planarity", "polar"]):
        return "polarization"

    if any(t in key for t in ["fft", "freq", "fmax", "fcentroid", "fquart", "spect", "gamma"]):
        return "frequency"

    if any(t in key for t in ["energy", "rms", "peak_amplitude", "amp", "density"]):
        return "energy"

    if any(t in key for t in ["kurto", "skew", "iqr", "env", "duration", "corpeak", "asdec"]):
        return "time"

    if any(t in key for t in ["nbrpeak", "ratioenv", "distq", "distmax", "int_ratio"]):
        return "time_frequency"

    return "other"


def remove_correlated_features(df, threshold):
    corr = df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=drop_cols), drop_cols


def choose_elbow_k(score_df):
    k_values = score_df["k"].to_numpy(dtype=float)
    inertias = score_df["inertia"].to_numpy(dtype=float)

    if KneeLocator is not None and len(k_values) >= 3:
        locator = KneeLocator(
            k_values,
            inertias,
            curve="convex",
            direction="decreasing",
        )

        if locator.knee is not None:
            return int(locator.knee), "KneeLocator"

    if len(k_values) < 3:
        return int(k_values[-1]), "fallback_last_k"

    points = np.column_stack([k_values, inertias])

    p1 = points[0]
    p2 = points[-1]

    line = p2 - p1
    line_norm = np.linalg.norm(line)

    distances = np.abs(np.cross(line, p1 - points)) / line_norm
    distances[0] = -np.inf

    best_idx = int(np.nanargmax(distances))

    return int(k_values[best_idx]), "max_distance_to_elbow_line"


def run_kmeans(X, k):
    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=KMEANS_N_INIT,
        init="k-means++",
    )
    model.fit(X)
    return model


def get_tree_shap_by_class(model, X):
    if shap is None:
        return {}

    explainer = shap.TreeExplainer(model)

    try:
        values = explainer.shap_values(X)
    except Exception:
        explanation = explainer(X)
        values = explanation.values

    classes = [int(v) for v in model.classes_]

    if isinstance(values, list):
        return {
            classes[idx]: np.asarray(values[idx])
            for idx in range(len(classes))
        }

    values = np.asarray(values)

    if values.ndim == 3:
        if values.shape[2] == len(classes):
            return {
                classes[idx]: values[:, :, idx]
                for idx in range(len(classes))
            }

        if values.shape[0] == len(classes):
            return {
                classes[idx]: values[idx, :, :]
                for idx in range(len(classes))
            }

    if values.ndim == 2 and len(classes) == 2:
        return {classes[-1]: values}

    raise ValueError(f"Unsupported SHAP value shape: {values.shape}")


# ============================================================
# 3. 读取匹配好的前兆簇样本特征表
# ============================================================

matched_df = pd.read_excel(INPUT_FILE)

matched_df["event"] = matched_df["event"].astype(str)
matched_df["timestamp"] = matched_df["timestamp"].astype(str)

if "selected_cluster_id" in matched_df.columns:
    matched_df = matched_df.rename(columns={"selected_cluster_id": "cluster_id"})

matched_df["cluster_id"] = matched_df["cluster_id"].astype(int)

matched_df = matched_df.sort_values(
    ["event", "cluster_id", "Distance_to_Center"],
    key=lambda s: s.map(natural_event_key) if s.name == "event" else s,
).reset_index(drop=True)

print("matched rows =", len(matched_df))
print("matched columns =", len(matched_df.columns))
print(matched_df.head())


# ============================================================
# 4. 从样本特征中计算事件级前兆簇中心特征
# ============================================================

metadata_cols_for_center = {
    "event",
    "timestamp",
    "cluster_id",
    "raw_precursor_label",
    "Distance_to_Center",
    "x_file",
    "duration",
}

feature_cols_for_center = [
    col
    for col in matched_df.columns
    if col not in metadata_cols_for_center
    and not str(col).startswith("PC")
    and pd.api.types.is_numeric_dtype(matched_df[col])
]

center_df = (
    matched_df
    .groupby(["event", "cluster_id"], as_index=False)
    .agg(
        sample_count=("timestamp", "count"),
        mean_distance_to_center=("Distance_to_Center", "mean"),
        min_distance_to_center=("Distance_to_Center", "min"),
        max_distance_to_center=("Distance_to_Center", "max"),
        **{col: (col, "mean") for col in feature_cols_for_center}
    )
)

center_df = center_df.sort_values(
    "event",
    key=lambda s: s.map(natural_event_key)
).reset_index(drop=True)

print("event-cluster centers =", len(center_df))
print("feature count for center =", len(feature_cols_for_center))
print(center_df.head())


# ============================================================
# 5. 构建 KMeans 特征矩阵
# ============================================================

metadata_cols = {
    "event",
    "cluster_id",
    "sample_count",
    "mean_distance_to_center",
    "min_distance_to_center",
    "max_distance_to_center",
}

feature_cols = [
    col
    for col in center_df.columns
    if col not in metadata_cols
    and not str(col).startswith("PC")
    and pd.api.types.is_numeric_dtype(center_df[col])
]

X_raw = center_df[feature_cols].replace([np.inf, -np.inf], np.nan)

missing_rate_df = (
    X_raw.isna()
    .mean()
    .sort_values(ascending=False)
    .rename("missing_rate")
    .reset_index()
    .rename(columns={"index": "feature"})
)

keep_cols = missing_rate_df.loc[
    missing_rate_df["missing_rate"] <= 0.5,
    "feature"
].tolist()

X_raw = X_raw[keep_cols]

imputer = SimpleImputer(strategy="median")

X_imputed = pd.DataFrame(
    imputer.fit_transform(X_raw),
    columns=X_raw.columns,
    index=X_raw.index,
)

std_values = X_imputed.std(axis=0)

non_constant_cols = std_values[std_values > 1e-12].index.tolist()

X_filtered = X_imputed[non_constant_cols].copy()

dropped_corr_cols = []

if REMOVE_HIGH_CORR:
    X_filtered, dropped_corr_cols = remove_correlated_features(
        X_filtered,
        CORR_THRESHOLD,
    )

scaler = StandardScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(X_filtered),
    columns=X_filtered.columns,
    index=X_filtered.index,
)

feature_info_df = pd.DataFrame({
    "feature": X_scaled.columns,
    "feature_group": [feature_group(col) for col in X_scaled.columns],
})

print("original numeric features:", len(feature_cols))
print("after missing filter:", len(keep_cols))
print("after constant filter:", len(non_constant_cols))
print("after correlation filter:", X_scaled.shape[1])
print("dropped high-correlation features:", len(dropped_corr_cols))


# ============================================================
# 6. PCA 降维
# ============================================================

pca_full = PCA(random_state=RANDOM_STATE)
pca_full.fit(X_scaled)

cum_var = np.cumsum(pca_full.explained_variance_ratio_)

n_components = int(np.searchsorted(cum_var, PCA_VARIANCE) + 1)

n_components = max(
    2,
    min(
        n_components,
        X_scaled.shape[0] - 1,
        X_scaled.shape[1],
    )
)

pca = PCA(n_components=n_components, random_state=RANDOM_STATE)

X_pca = pca.fit_transform(X_scaled)

pca_cols = [f"PC{i + 1}" for i in range(n_components)]

pca_df = pd.DataFrame(
    X_pca,
    columns=pca_cols,
    index=center_df.index,
)

pca_explained_df = pd.DataFrame({
    "component": [f"PC{i + 1}" for i in range(len(pca_full.explained_variance_ratio_))],
    "explained_variance_ratio": pca_full.explained_variance_ratio_,
    "cumulative_variance_ratio": cum_var,
})

plt.figure(figsize=(7, 4))
plt.plot(np.arange(1, len(cum_var) + 1), cum_var, marker="o", linewidth=2)
plt.axhline(PCA_VARIANCE, color="red", linestyle="--", label=f"{PCA_VARIANCE:.0%}")
plt.axvline(n_components, color="black", linestyle=":", label=f"n={n_components}")
plt.xlabel("PCA component number")
plt.ylabel("Cumulative explained variance")
plt.title("PCA cumulative explained variance before KMeans")
plt.grid(alpha=0.25)
plt.legend()
save_fig(OUT_DIR / "kmeans_pca_cumulative_variance.png")

print("PCA retained components:", n_components)
print("PCA cumulative variance:", pca.explained_variance_ratio_.sum())


# ============================================================
# 7. 肘部法确定最优簇数
# ============================================================

score_rows = []

valid_k_values = [
    k for k in K_RANGE
    if 1 <= k <= len(center_df) - 1
]

for k in valid_k_values:

    model = run_kmeans(X_pca, k)

    labels = model.labels_

    row = {
        "k": k,
        "inertia": float(model.inertia_),
        "min_cluster_size": int(pd.Series(labels).value_counts().min()),
        "cluster_sizes": dict(pd.Series(labels).value_counts().sort_index()),
    }

    if k >= 2 and len(np.unique(labels)) >= 2:
        row["silhouette"] = silhouette_score(X_pca, labels)
        row["calinski_harabasz"] = calinski_harabasz_score(X_pca, labels)
        row["davies_bouldin"] = davies_bouldin_score(X_pca, labels)
    else:
        row["silhouette"] = np.nan
        row["calinski_harabasz"] = np.nan
        row["davies_bouldin"] = np.nan

    score_rows.append(row)

score_df = pd.DataFrame(score_rows)

BEST_K, elbow_method = choose_elbow_k(score_df)

print("BEST_K =", BEST_K)
print("elbow_method =", elbow_method)
print(score_df)


plt.figure(figsize=(8, 5))

fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(
    score_df["k"],
    score_df["inertia"],
    marker="o",
    linewidth=2,
    color="#1f77b4",
    label="Inertia",
)

ax1.axvline(
    BEST_K,
    color="red",
    linestyle="--",
    label=f"elbow K={BEST_K}",
)

ax1.set_xlabel("Number of clusters K")
ax1.set_ylabel("Inertia / WCSS")
ax1.set_title(f"KMeans elbow method, selected K={BEST_K} ({elbow_method})")
ax1.grid(alpha=0.25)

ax2 = ax1.twinx()

if score_df["silhouette"].notna().any():
    ax2.plot(
        score_df["k"],
        score_df["silhouette"],
        marker="s",
        linewidth=1.5,
        color="#ff7f0e",
        label="Silhouette",
    )
    ax2.set_ylabel("Silhouette score")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

save_fig(OUT_DIR / "kmeans_elbow_selected_k.png")


# ============================================================
# 8. 使用最优 K 进行 KMeans 聚类
# ============================================================

best_model = run_kmeans(X_pca, BEST_K)

kmeans_labels = best_model.labels_

result_df = center_df.copy()

result_df["kmeans_cluster"] = kmeans_labels

result_df["kmeans_cluster_name"] = result_df["kmeans_cluster"].map(
    lambda x: f"type_{int(x)}"
)

for col in pca_cols:
    result_df[col] = pca_df[col]

result_df = result_df.sort_values(
    "event",
    key=lambda s: s.map(natural_event_key)
).reset_index(drop=True)

cluster_counts_df = (
    result_df
    .groupby("kmeans_cluster")
    .agg(
        event_count=("event", "count"),
        events=("event", lambda x: ", ".join(x.astype(str))),
        selected_precursor_clusters=("cluster_id", lambda x: ", ".join([str(int(v)) for v in x])),
        mean_sample_count=("sample_count", "mean"),
        mean_distance_to_center=("mean_distance_to_center", "mean"),
    )
    .reset_index()
)

print("KMeans cluster counts:")
print(cluster_counts_df)


# ============================================================
# 9. PCA 可视化
# ============================================================

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    result_df["PC1"],
    result_df["PC2"],
    c=result_df["kmeans_cluster"],
    cmap="tab10",
    s=90,
    edgecolor="black",
    alpha=0.85,
)

for row in result_df.itertuples(index=False):
    plt.text(
        getattr(row, "PC1"),
        getattr(row, "PC2"),
        str(row.event).replace("event", "E"),
        fontsize=8,
        ha="left",
        va="bottom",
    )

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.title("KMeans clustering of selected precursor centers")
plt.colorbar(scatter, label="KMeans cluster")
plt.grid(alpha=0.25)

save_fig(OUT_DIR / "kmeans_pca_scatter.png")


# ============================================================
# 10. t-SNE 可视化
# ============================================================

perplexity = max(2, min(15, len(result_df) // 3, len(result_df) - 1))

tsne = TSNE(
    n_components=2,
    random_state=RANDOM_STATE,
    perplexity=perplexity,
    init="pca",
    learning_rate="auto",
)

tsne_xy = tsne.fit_transform(X_pca)

result_df["TSNE1"] = tsne_xy[:, 0]
result_df["TSNE2"] = tsne_xy[:, 1]

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    result_df["TSNE1"],
    result_df["TSNE2"],
    c=result_df["kmeans_cluster"],
    cmap="tab10",
    s=90,
    edgecolor="black",
    alpha=0.85,
)

for row in result_df.itertuples(index=False):
    plt.text(
        row.TSNE1,
        row.TSNE2,
        str(row.event).replace("event", "E"),
        fontsize=8,
    )

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE visualization of KMeans clusters")
plt.colorbar(scatter, label="KMeans cluster")
plt.grid(alpha=0.25)

save_fig(OUT_DIR / "kmeans_tsne_scatter.png")


# ============================================================
# 11. KMeans 簇中心热力图
# ============================================================

scaled_with_label = X_scaled.copy()

scaled_with_label["kmeans_cluster"] = kmeans_labels

center_scaled_df = (
    scaled_with_label
    .groupby("kmeans_cluster")
    .mean()
    .sort_index()
)

top_heatmap_features = (
    center_scaled_df
    .var(axis=0)
    .sort_values(ascending=False)
    .head(40)
    .index
    .tolist()
)

plt.figure(figsize=(14, max(4, BEST_K * 1.3)))

if sns is not None:
    sns.heatmap(
        center_scaled_df[top_heatmap_features],
        cmap="RdBu_r",
        center=0,
        linewidths=0.4,
        cbar_kws={"label": "standardized cluster mean"},
    )
else:
    plt.imshow(
        center_scaled_df[top_heatmap_features],
        aspect="auto",
        cmap="RdBu_r",
    )
    plt.colorbar(label="standardized cluster mean")
    plt.xticks(
        range(len(top_heatmap_features)),
        top_heatmap_features,
        rotation=90,
    )
    plt.yticks(range(BEST_K), center_scaled_df.index)

plt.title("KMeans cluster centers: standardized feature means")
plt.xlabel("Feature")
plt.ylabel("KMeans cluster")

save_fig(OUT_DIR / "kmeans_feature_mean_heatmap.png")


# ============================================================
# 12. 随机森林代理解释 KMeans 聚类结果
# ============================================================

rf = RandomForestClassifier(
    n_estimators=500,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    max_features="sqrt",
)

rf.fit(X_scaled, kmeans_labels)

importance_df = pd.DataFrame({
    "feature": X_scaled.columns,
    "feature_group": [feature_group(col) for col in X_scaled.columns],
    "rf_importance": rf.feature_importances_,
}).sort_values("rf_importance", ascending=False)

top_importance = importance_df.head(25).iloc[::-1]

plt.figure(figsize=(9, 7))
plt.barh(
    top_importance["feature"],
    top_importance["rf_importance"],
    color="#4477AA",
)
plt.xlabel("RandomForest surrogate importance")
plt.title("Top features explaining KMeans clusters")
plt.grid(axis="x", alpha=0.25)

save_fig(OUT_DIR / "kmeans_rf_feature_importance.png")

print("Top RF importance:")
print(importance_df.head(20))


# ============================================================
# 13. SHAP 解释
# ============================================================

shap_status_df = pd.DataFrame()
shap_feature_importance_df = pd.DataFrame()
shap_global_importance_df = pd.DataFrame()
shap_mode_direction_df = pd.DataFrame()
shap_by_class = {}

if shap is None:

    shap_status_df = pd.DataFrame([{
        "status": "skipped",
        "reason": "The shap package is not installed in the current Python environment.",
    }])

    print("SHAP skipped: package not installed.")

else:

    try:
        shap_by_class = get_tree_shap_by_class(rf, X_scaled)

        class_labels = sorted([int(v) for v in np.unique(kmeans_labels)])

        shap_rows = []

        for class_label, class_values in shap_by_class.items():

            mean_abs = np.abs(class_values).mean(axis=0)

            for feature, value in zip(X_scaled.columns, mean_abs):

                shap_rows.append({
                    "class_label": class_label,
                    "feature": feature,
                    "feature_group": feature_group(feature),
                    "mean_abs_shap": float(value),
                })

        shap_feature_importance_df = (
            pd.DataFrame(shap_rows)
            .sort_values(["class_label", "mean_abs_shap"], ascending=[True, False])
            .reset_index(drop=True)
        )

        shap_global_importance_df = (
            shap_feature_importance_df
            .groupby(["feature", "feature_group"], as_index=False)["mean_abs_shap"]
            .mean()
            .rename(columns={"mean_abs_shap": "global_mean_abs_shap"})
            .sort_values("global_mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        top_shap_features = (
            shap_global_importance_df["feature"]
            .head(TOP_N_FEATURES_FOR_SHAP)
            .tolist()
        )

        mode_rows = []

        for feature in top_shap_features:

            row = {
                "feature": feature,
                "feature_group": feature_group(feature),
                "global_mean_abs_shap": float(
                    shap_global_importance_df.loc[
                        shap_global_importance_df["feature"] == feature,
                        "global_mean_abs_shap"
                    ].iloc[0]
                ),
            }

            means = center_scaled_df[feature].to_dict()

            for cluster_id, value in means.items():
                row[f"kmeans_cluster_{int(cluster_id)}_mean_z"] = float(value)

            row["mode_with_higher_feature_mean"] = int(max(means, key=means.get))
            row["mode_with_lower_feature_mean"] = int(min(means, key=means.get))

            mode_rows.append(row)

        shap_mode_direction_df = pd.DataFrame(mode_rows)

        top_plot_df = shap_global_importance_df.head(TOP_N_FEATURES_FOR_SHAP).iloc[::-1]

        plt.figure(figsize=(9, 7))
        plt.barh(
            top_plot_df["feature"],
            top_plot_df["global_mean_abs_shap"],
            color="#AA3377",
        )
        plt.xlabel("Mean |SHAP value|")
        plt.title("SHAP global importance for KMeans pattern separation")
        plt.grid(axis="x", alpha=0.25)

        save_fig(OUT_DIR / "kmeans_shap_global_importance.png")

        for class_label in class_labels:

            if class_label not in shap_by_class:
                continue

            shap.summary_plot(
                shap_by_class[class_label],
                X_scaled,
                feature_names=X_scaled.columns.tolist(),
                max_display=TOP_N_FEATURES_FOR_SHAP,
                show=False,
            )
            plt.title(f"SHAP beeswarm: features pushing toward KMeans mode {class_label}")
            save_fig(OUT_DIR / f"kmeans_shap_beeswarm_mode_{class_label}.png")

            shap.summary_plot(
                shap_by_class[class_label],
                X_scaled,
                feature_names=X_scaled.columns.tolist(),
                max_display=TOP_N_FEATURES_FOR_SHAP,
                plot_type="bar",
                show=False,
            )
            plt.title(f"SHAP bar: KMeans mode {class_label}")
            save_fig(OUT_DIR / f"kmeans_shap_bar_mode_{class_label}.png")

        if top_shap_features:

            plt.figure(figsize=(12, max(4, BEST_K * 1.4)))

            if sns is not None:
                sns.heatmap(
                    center_scaled_df[top_shap_features],
                    cmap="RdBu_r",
                    center=0,
                    linewidths=0.4,
                    cbar_kws={"label": "standardized cluster mean"},
                )
            else:
                plt.imshow(
                    center_scaled_df[top_shap_features],
                    aspect="auto",
                    cmap="RdBu_r",
                )
                plt.colorbar(label="standardized cluster mean")
                plt.xticks(
                    range(len(top_shap_features)),
                    top_shap_features,
                    rotation=90,
                )
                plt.yticks(range(BEST_K), center_scaled_df.index)

            plt.title("KMeans modes: standardized means of SHAP-dominant features")
            plt.xlabel("SHAP-dominant feature")
            plt.ylabel("KMeans cluster")

            save_fig(OUT_DIR / "kmeans_shap_mode_feature_direction_heatmap.png")

        target_class = class_labels[-1]

        if target_class in shap_by_class:

            dep_dir = OUT_DIR / "shap_dependence_plots"
            dep_dir.mkdir(parents=True, exist_ok=True)

            for feature in top_shap_features[:TOP_N_FEATURES_FOR_SHAP_DEPENDENCE]:

                shap.dependence_plot(
                    feature,
                    shap_by_class[target_class],
                    X_scaled,
                    feature_names=X_scaled.columns.tolist(),
                    interaction_index=None,
                    show=False,
                )

                plt.title(f"SHAP dependence: {feature} -> mode {target_class}")

                safe_name = str(feature).replace("/", "_").replace("\\", "_").replace(":", "_")

                save_fig(dep_dir / f"shap_dependence_{safe_name}.png")

        shap_status_df = pd.DataFrame([{
            "status": "success",
            "class_labels": ", ".join(map(str, class_labels)),
            "note": (
                "SHAP explains the RandomForest surrogate model that approximates "
                "KMeans labels. It explains pattern separation, not causal proof."
            ),
        }])

        print("SHAP analysis finished.")
        print(shap_global_importance_df.head(20))
        print(shap_mode_direction_df.head(20))

    except Exception as exc:

        shap_status_df = pd.DataFrame([{
            "status": "failed",
            "reason": repr(exc),
        }])

        print("SHAP failed:", repr(exc))


# ============================================================
# 14. 关键特征箱型图
# ============================================================

boxplot_features = importance_df["feature"].head(TOP_N_FEATURES_FOR_BOXPLOT).tolist()

n_cols = 4
n_rows = math.ceil(len(boxplot_features) / n_cols)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(4.2 * n_cols, 3.2 * n_rows),
    squeeze=False,
)

plot_df = result_df[["kmeans_cluster"] + boxplot_features].copy()

for idx, feature in enumerate(boxplot_features):

    ax = axes[idx // n_cols, idx % n_cols]

    local_df = (
        plot_df[["kmeans_cluster", feature]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if sns is not None:

        sns.boxplot(
            data=local_df,
            x="kmeans_cluster",
            y=feature,
            showfliers=SHOW_OUTLIERS,
            color="#9ecae1",
            ax=ax,
        )

        sns.stripplot(
            data=local_df,
            x="kmeans_cluster",
            y=feature,
            color="black",
            alpha=0.55,
            size=3,
            jitter=0.18,
            ax=ax,
        )

    else:

        clusters = sorted(local_df["kmeans_cluster"].unique())

        data = [
            local_df.loc[
                local_df["kmeans_cluster"] == c,
                feature
            ].values
            for c in clusters
        ]

        ax.boxplot(
            data,
            labels=clusters,
            showfliers=SHOW_OUTLIERS,
        )

    ax.set_title(feature, fontsize=10)
    ax.set_xlabel("KMeans cluster")
    ax.set_ylabel("")
    ax.grid(axis="y", alpha=0.25)

for idx in range(len(boxplot_features), n_rows * n_cols):
    axes[idx // n_cols, idx % n_cols].axis("off")

fig.suptitle("Top feature boxplots by KMeans cluster", fontsize=15)

save_fig(OUT_DIR / "kmeans_top_feature_boxplots.png")


# ============================================================
# 15. 保存结果
# ============================================================

run_summary_df = pd.DataFrame([
    {"item": "input_file", "value": str(INPUT_FILE)},
    {"item": "matched_sample_rows", "value": len(matched_df)},
    {"item": "valid_precursor_centers", "value": len(result_df)},
    {"item": "original_numeric_features", "value": len(feature_cols)},
    {"item": "final_feature_count", "value": X_scaled.shape[1]},
    {"item": "pca_retained_components", "value": n_components},
    {"item": "pca_cumulative_variance", "value": pca.explained_variance_ratio_.sum()},
    {"item": "k_range", "value": f"{min(valid_k_values)}-{max(valid_k_values)}"},
    {"item": "selected_k_by_elbow", "value": BEST_K},
    {"item": "elbow_method", "value": elbow_method},
])

kmeans_centers_original_scale_df = pd.DataFrame(
    scaler.inverse_transform(center_scaled_df[X_scaled.columns]),
    columns=X_scaled.columns,
)

kmeans_centers_original_scale_df.insert(
    0,
    "kmeans_cluster",
    center_scaled_df.index,
)

output_excel = write_excel(
    OUT_DIR / "kmeans_clustering_all_results.xlsx",
    {
        "matched_precursor_samples": matched_df,
        "precursor_cluster_centers": center_df,
        "kmeans_results": result_df,
        "cluster_summary": cluster_counts_df,
        "run_summary": run_summary_df,
        "elbow_scores": score_df,
        "feature_info": feature_info_df,
        "missing_rate": missing_rate_df,
        "dropped_corr_features": pd.DataFrame({"dropped_feature": dropped_corr_cols}),
        "pca_explained_variance": pca_explained_df,
        "kmeans_centers_standardized": center_scaled_df.reset_index(),
        "kmeans_centers_original": kmeans_centers_original_scale_df,
        "rf_feature_importance": importance_df,
        "shap_status": shap_status_df,
        "shap_global_importance": shap_global_importance_df,
        "shap_by_class_importance": shap_feature_importance_df,
        "shap_mode_direction": shap_mode_direction_df,
        "boxplot_features": pd.DataFrame({"feature": boxplot_features}),
    },
)

result_df.to_csv(
    OUT_DIR / "kmeans_clustering_results.csv",
    index=False,
    encoding="utf-8-sig",
)

score_df.to_csv(
    OUT_DIR / "kmeans_elbow_scores.csv",
    index=False,
    encoding="utf-8-sig",
)

center_df.to_csv(
    OUT_DIR / "precursor_cluster_centers.csv",
    index=False,
    encoding="utf-8-sig",
)

matched_df.to_csv(
    OUT_DIR / "matched_precursor_samples.csv",
    index=False,
    encoding="utf-8-sig",
)

print("全部完成！")
print("Saved Excel:", output_excel)
print("Saved CSV:", OUT_DIR / "kmeans_clustering_results.csv")
print("Saved figures in:", OUT_DIR)