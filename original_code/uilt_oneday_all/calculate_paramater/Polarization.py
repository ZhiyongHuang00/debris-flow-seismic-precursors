# -*- coding: utf-8 -*-
'''
@Time    : 2025/7/3 22:40
@Author  : HuangZhiyong
@email   : 1524338616@qq.com
@File    : Polarization.py
@Function: {}:
'''
import glob
import os
from tqdm import tqdm

import numpy as np
from obspy import read
from scipy.signal import hilbert, lfilter
import pandas as pd
from collections import defaultdict


# 极化函数（与你提供的一致）
def get_polarization_stuff(st, env):
    sps = st[0].stats.sampling_rate
    strong_filter = np.ones(int(sps)) / float(sps)
    smooth_env = lfilter(strong_filter, 1, env[0])
    imax = np.argmax(smooth_env)
    end_window = int(np.round(imax / 3.))

    xP = st[2].data[0:end_window]
    yP = st[1].data[0:end_window]
    zP = st[0].data[0:end_window]

    if len(xP) < 10:
        return np.nan, np.nan, np.nan, np.nan

    MP = np.cov(np.array([xP, yP, zP]))
    w, v = np.linalg.eig(MP)

    indexes = np.argsort(w)
    DP = w[indexes]
    pP = v[:, indexes]

    rectilinP = 1 - ((DP[0] + DP[1]) / (2 * DP[2]))
    azimuthP = np.arctan2(pP[1, 2], pP[0, 2]) * 180. / np.pi
    dipP = np.arctan(pP[2, 2] / np.sqrt(pP[1, 2] ** 2 + pP[0, 2] ** 2)) * 180. / np.pi
    Plani = 1 - (2 * DP[0]) / (DP[1] + DP[2])

    return rectilinP, azimuthP, dipP, Plani


# 搜索所有 SAC 文件
base_dir = r'D:\Pycharm_Project\Find_precursor_signs\ILLgarben\uilt\01data_preprocess\output_segments'
sac_files = glob.glob(os.path.join(base_dir, 'event*', '*', '*.sac'))

# 分组结构：{event_name: {timestamp: {'E': path, 'N': path, 'Z': path}}}
event_groups = defaultdict(lambda: defaultdict(dict))

# 按事件、时间戳、分量分组
for filepath in sac_files:
    parts = filepath.split(os.sep)
    event_name = parts[-3]  # event1, event2, etc.
    filename = os.path.basename(filepath)

    # 提取时间戳和分量（你命名为 event1_20170527T165000_E.sac）

    prefix, ts, component_with_ext = filename.split('_')
    component = component_with_ext.split('.')[0][-1]  # E, N, Z
    timestamp = ts  # 20170527T165000
    event_groups[event_name][timestamp][component] = filepath


results = []

# 遍历每个事件和时间戳组合
# for event_name, timestamps in event_groups.items():
for event_name, timestamps in tqdm(event_groups.items(), desc="Processing Events"):
    for ts, comps in tqdm(timestamps.items(), desc=f"{event_name}", leave=False):
    # for ts, comps in timestamps.items():
        if all(k in comps for k in ['E', 'N', 'Z']):
            st_e = read(comps['E'])[0]
            st_n = read(comps['N'])[0]
            st_z = read(comps['Z'])[0]
            st = [st_z, st_n, st_e]

            # 包络（只用 Z 分量）
            envelope = np.abs(hilbert(st_z.data))

            rect, az, dip, plan = get_polarization_stuff(st, env=[envelope])

            results.append({
                'event': event_name,
                'timestamp': ts,
                'rectilinearity': rect,
                'azimuth': az,
                'dip': dip,
                'planarity': plan
            })


# 保存为 CSV
df = pd.DataFrame(results)
df.to_csv('polarization_by_event.csv', index=False)
print("极化分析结果已保存为 polarization_by_event.csv")
# 保存所有结果为一个总表（可选）
df = pd.DataFrame(results)
df.to_csv(r'.\result\all_parameter.csv', index=False)

# 将不同事件保存为不同 Excel 文件
grouped = df.groupby("event")

output_dir = r".\result\parameter_by_event_excels"
os.makedirs(output_dir, exist_ok=True)

for event_name, group_df in grouped:
    save_path = os.path.join(output_dir, f"{event_name}.xlsx")
    group_df.to_excel(save_path, index=False)

print(f"极化分析结果已按事件分别保存至文件夹：{output_dir}")
