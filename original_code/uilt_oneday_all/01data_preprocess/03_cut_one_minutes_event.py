# -*- coding: utf-8 -*-
'''
@Time    : 2025/6/15 15:51
@Author  : HuangZhiyong
@email   : 1524338616@qq.com
@File    : 03_cut_one_minutes_event.py
@Function: {}:
'''
from obspy import UTCDateTime, read, read_inventory
import pandas as pd
import os
import matplotlib.pyplot as plt
from obspy import read_inventory
from obspy.signal.invsim import corn_freq_2_paz

import obspy
from tqdm import trange
xml_file = r"D:\Pycharm_Project\Open_Source_Data_Download\Down_Debris_flow\XML\origin\combine\base_ILL18.xml"
inventory = read_inventory(xml_file)

file_csv = pd.read_excel(r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben\data\新版完整数据集时间.xlsx")[34:35]
sac_paths = file_csv["SAC_File_Path"].tolist()
events = file_csv['Event'].tolist()
paz = {
    'poles': [(-4.444+4.444j), (-4.444-4.444j), (-1.083+0j)],
    'zeros': [0j, 0j, 0j],
    'sensitivity':1 ,  # 灵敏度
    # 'gain': 160000000,  # 增益
    'gain': 160000000,  # 增益
    'normalization_frequency': 5  # 归一化频率
}

paz_1hz = corn_freq_2_paz(1, damp=0.707)
paz_1hz['sensitivity'] = 1
for i in trange(len(file_csv)):
# for i in trange(2):
    end_time = UTCDateTime(file_csv["Manual_Label_start"].iloc[i])
    # end_time = UTCDateTime(file_csv["Manual_Label_start"][i])
    start_time = end_time -  24 * 60 * 60

    z_path = sac_paths[i]
    # 替换路径中 Z 文件夹为 E 和 N，文件名中 EHZ 改为 EHE / EHN
    # 改为 E 和 N：同时改路径中的文件夹名和文件名的通道标识
    e_path = z_path.replace('\\Z\\', '\\E\\').replace('.EHZ.', '.EHE.')
    n_path = z_path.replace('\\Z\\', '\\N\\').replace('.EHZ.', '.EHN.')
    for sacfile in [ z_path, e_path, n_path]:
        st = obspy.read(sacfile)
        st_trimmed = st.copy().trim(starttime=start_time-60*60*2, endtime=end_time+60*60*2)
        st_trimmed.simulate(paz_remove=paz, paz_simulate=paz_1hz)

        # st_trimmed.remove_response(inventory=inventory, output='VEL', pre_filt=(0.01, 0.02, 45, 50), zero_mean=True, taper=False)
        st_trimmed.filter('bandpass', freqmin=1.0, freqmax=49.0, corners=4, zerophase=True)

        # 遍历每个1分钟窗口
        t = start_time
        while t + 60 <= end_time:
            st_cut = st_trimmed.copy().trim(starttime=t, endtime=t + 60-0.01)
            if len(st_cut) == 0:
                t += 60
                continue
            channel = st_cut[0].stats.channel[-1]  # 如 EHE、EHN、EHZ

            # 保存每一段分割数据
            save_dir = f"output_segments/{events[i]}/{channel}"
            os.makedirs(save_dir, exist_ok=True)
            segment_path = os.path.join(save_dir, f"{events[i]}_{t.strftime('%Y%m%dT%H%M%S')}_{channel}.sac")
            st_cut.write(segment_path, format="SAC")
            t += 60  # 跳到下一分钟
    # st.plot()
    print(f"Finish: {z_path}")

    debug = 0