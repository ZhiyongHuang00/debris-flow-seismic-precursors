# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_ROOT = Path(r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben")

PRECURSOR_FILE = PROJECT_ROOT / "data" / "前兆标签选择.xlsx"

PARAMETER_CSV = PROJECT_ROOT / "uilt_oneday_all" / "calculate_paramater" / "result" / "all_parameter.csv"

NPZ_DIR = PROJECT_ROOT / "uilt_oneday_all" / "02clustering" / "result" / "DEC_result" / "all_parement"

OUT_DIR = PROJECT_ROOT / "uilt_oneday_all" / "03visual" / "二次聚类" / "matched_precursor_features"

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_EXCEL = OUT_DIR / "matched_precursor_features_with_distance.xlsx"


# ============================================================
# 2. 读取前兆标签表
# ============================================================

label_df = pd.read_excel(PRECURSOR_FILE)

label_df = label_df[["Event", "Precursor_label"]].copy()
label_df.columns = ["event", "cluster_id"]

label_df["event"] = label_df["event"].astype(str)
label_df["cluster_id"] = label_df["cluster_id"].astype(int)


# ============================================================
# 3. 读取全部特征参数表
# ============================================================

parameter_df = pd.read_csv(PARAMETER_CSV)

parameter_df["event"] = parameter_df["event"].astype(str)
parameter_df["timestamp"] = parameter_df["timestamp"].astype(str)


# ============================================================
# 4. 遍历每个事件，只处理前兆标签不为 -1 的事件
# ============================================================

all_matched_df = []

for _, row in label_df.iterrows():

    event = row["event"]
    cluster_id = int(row["cluster_id"])

    # -1 表示前兆簇不存在，直接跳过
    if cluster_id == -1:
        print(f"跳过 {event}：前兆簇不存在，Precursor_label = -1")
        continue

    print(f"处理 {event}, 前兆簇 {cluster_id}")

    # 读取该事件对应的 npz 文件
    npz_path = NPZ_DIR / f"{event}_all_parement.npz"

    npz = np.load(npz_path, allow_pickle=True)

    y_pred = npz["y_pred_last"].astype(int)
    x_file = npz["x_file"]
    distances = npz["distances"]

    npz.close()

    # 找到该事件中属于前兆簇 cluster_id 的样本
    cluster_mask = y_pred == cluster_id

    precursor_files = x_file[cluster_mask]

    # 提取该前兆簇样本到簇中心的距离
    dist_c = distances[cluster_mask]

    # 如果 distances 是二维矩阵 N × n_clusters
    # 取当前前兆簇 cluster_id 对应的距离列
    if np.ndim(dist_c) > 1:
        dist_c = dist_c[:, cluster_id]

    dist_c = np.ravel(dist_c)

    # 根据 x_file 提取 timestamp，并建立距离表
    distance_records = []

    for file, distance in zip(precursor_files, dist_c):

        name = str(file).replace("\\", "/").split("/")[-1]
        stem = Path(name).stem

        # 默认格式：event1_20240701120000_xxx
        timestamp = stem.split("_")[1]

        distance_records.append({
            "event": event,
            "timestamp": str(timestamp),
            "selected_cluster_id": cluster_id,
            "Distance_to_Center": float(distance),
            "x_file": str(file)
        })

    distance_df = pd.DataFrame(distance_records)

    # 同一个 timestamp 如果重复，只保留距离中心最近的一条
    distance_df = (
        distance_df
        .sort_values("Distance_to_Center")
        .drop_duplicates(subset=["event", "timestamp"], keep="first")
    )

    # 在 all_parameter.csv 中匹配对应特征行
    matched_df = parameter_df[
        (parameter_df["event"] == event) &
        (parameter_df["timestamp"].isin(distance_df["timestamp"]))
    ].copy()

    # 合并距离信息
    matched_df = matched_df.merge(
        distance_df,
        on=["event", "timestamp"],
        how="left"
    )

    # 当前事件内部按距离从小到大排序
    matched_df = matched_df.sort_values("Distance_to_Center").reset_index(drop=True)

    all_matched_df.append(matched_df)


# ============================================================
# 5. 合并并保存
# ============================================================

result_df = pd.concat(all_matched_df, ignore_index=True)

result_df = result_df.sort_values(
    ["event", "selected_cluster_id", "Distance_to_Center"]
).reset_index(drop=True)

result_df.to_excel(OUT_EXCEL, index=False)

print("完成！")
print("匹配到的特征行数:", len(result_df))
print("保存路径:", OUT_EXCEL)