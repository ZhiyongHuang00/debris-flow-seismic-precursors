from kneed import KneeLocator
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

file_path = glob.glob(r'.\result\Figure\optimal_cluster\*.xlsx')
metric_list = ['CH_score', 'Silhouette_score', 'Davies_Bouldin_Index', 'WCSS']
summary_list = []

for file in file_path:
    metrics = pd.read_excel(file)
    cluster_counts = metrics['n_clusters']
    results = {}
    file_base = os.path.basename(file).replace('.xlsx', '')
    save_dir = os.path.dirname(file)
    plt.figure(figsize=(14, 10))
    for i, metric in enumerate(metric_list, 1):
        if metric not in metrics.columns:
            continue
        y = metrics[metric].values
        if pd.isnull(y).any():
            continue

        # 决定趋势
        if metric in ['AIC', 'BIC', 'Davies_Bouldin_Index', 'WCSS']:
            curve_type, direction = 'convex', 'decreasing'
        else:
            curve_type, direction = 'concave', 'increasing'

        try:
            knee = KneeLocator(cluster_counts, y, curve=curve_type, direction=direction)
            best_k = knee.knee
        except Exception as e:
            best_k = None
        results[metric] = best_k

        # 绘图
        plt.subplot(2, 3, i)
        plt.plot(cluster_counts, y, marker='o', label=metric)
        if best_k is not None:
            plt.axvline(best_k, color='red', linestyle='--', label=f"Knee: {best_k}")
        else:
            plt.text(cluster_counts.iloc[-1], y[-1], "No knee", fontsize=10, color='gray')
        plt.xlabel("n_clusters")
        plt.ylabel(metric)
        plt.title(metric)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    save_dirs = r'.\result\Figure\optimal_cluster_result'
    if not os.path.exists(save_dirs):
        os.makedirs(save_dirs)
    # save_fig = os.path.join(save_dir, f"{file_base}_knee.png")
    save_fig = rf"./result\Figure\optimal_cluster_result\{file_base}_knee.png"

    plt.savefig(save_fig, dpi=300)
    plt.close()

    # 打印和追加到汇总表
    print(f"\n📊 {file_base} 最优簇数建议：")
    row = {'Event': file_base}
    for k, v in results.items():
        print(f"  {k}: {v}")
        row[k] = v
    summary_list.append(row)

# 保存总表
summary_df = pd.DataFrame(summary_list)
summary_path = rf'./result\Figure\summary_best_k.xlsx'
summary_df.to_excel(summary_path, index=False)
print(f"\n✅ 所有事件最佳K值已汇总保存到：{summary_path}")
