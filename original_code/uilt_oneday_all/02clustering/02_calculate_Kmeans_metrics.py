from obspy import UTCDateTime
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score

plt.rcParams.update({'font.size': 18, 'font.family': 'Arial'})

# 特征文件路径
parament_files = glob.glob(
    r".\result\all_parameters\*.npz"
)


# 输出路径
save_dir = r'.\result\Figure\optimal_cluster'
os.makedirs(save_dir, exist_ok=True)

# 遍历每个事件特征文件
for all_parameter_filename in parament_files:
    base_name = os.path.basename(all_parameter_filename)
    match = re.match(r'^(.*?)(?=_stft_maxmin)', base_name)
    extracted_name = match.group(1)

    all_parameter = np.load(all_parameter_filename)
    features = all_parameter['features']
    wavefore_file = all_parameter['wave_file']

    # 聚类数范围
    cluster_range = range(2, 11)

    # 初始化指标记录
    ch_scores = []
    silhouette_scores = []
    dbi_scores = []
    wcss_scores = []

    for n in cluster_range:
        kmeans = KMeans(n_clusters=n, n_init=10, random_state=42)
        labels = kmeans.fit_predict(features)

        ch_scores.append(calinski_harabasz_score(features, labels))
        silhouette_scores.append(silhouette_score(features, labels))
        dbi_scores.append(davies_bouldin_score(features, labels))
        wcss_scores.append(kmeans.inertia_)

    # 保存为 Excel 表格
    results_df = pd.DataFrame({
        'n_clusters': list(cluster_range),
        'CH_score': ch_scores,
        'Silhouette_score': silhouette_scores,
        'Davies_Bouldin_Index': dbi_scores,
        'WCSS': wcss_scores
    })

    excel_path = os.path.join(save_dir, f'{extracted_name}.xlsx')
    results_df.to_excel(excel_path, index=False)
    print(f"已保存：{excel_path}")

    # 绘图保存
    plt.figure(figsize=(14, 10))
    titles = [
        ('Calinski-Harabasz Score', ch_scores, 'blue'),
        ('Silhouette Score', silhouette_scores, 'green'),
        ('Davies-Bouldin Index (Lower Better)', dbi_scores, 'orange'),
        ('WCSS (Inertia, Lower Better)', wcss_scores, 'purple')
    ]

    for idx, (title, scores, color) in enumerate(titles):
        plt.subplot(2, 2, idx + 1)
        plt.plot(cluster_range, scores, marker='o', color=color)
        plt.title(title)
        plt.xlabel('Number of Clusters')
        plt.ylabel(title.split()[0])
        plt.grid(True)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, f'{extracted_name}.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
