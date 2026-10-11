import glob
import numpy as np
import scipy.signal
from obspy import read
from copy import deepcopy
import os
from tqdm import tqdm
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

def compute_STFT_maxmin(data):
    tmp_data = deepcopy(data)
    tmp_data -= np.mean(tmp_data)
    f, t, tmp_FT = scipy.signal.stft(tmp_data, fs=100, nperseg=19,
                                     nfft=127, boundary=None)
    amp_zxx = abs(tmp_FT)
    resized_amp = zoom(amp_zxx, (1, 512 / amp_zxx.shape[1]), order=3)

    # fs = 100
    # duration_sec = 60
    # # 正确的时间轴（横轴）：等分 duration_sec 秒
    # time_axis = np.linspace(0, duration_sec, resized_amp.shape[1])
    #
    # # 正确的频率轴（纵轴）：与 resize 前一致
    # freq_axis = np.linspace(0, fs / 2, resized_amp.shape[0])  # fs/2 为 Nyquist 频率
    #
    # plt.figure(figsize=(10, 5))
    # plt.imshow(resized_amp, aspect='auto', origin='lower',
    #            extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    #            cmap='viridis')
    # plt.title(f'STFT Resized (64×512), duration: {duration_sec}s')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')
    # plt.colorbar(label='Amplitude')
    # plt.tight_layout()
    # plt.show()


    resized_amp -= np.min(resized_amp)
    resized_amp /= np.max(resized_amp)
    return resized_amp

# 1. 获取所有 event 文件夹路径（按事件分组）
# event_folders = glob.glob(r'D:\Pycharm_Project\Find_precursor_signs\ILLgarben\uilt\01data_preprocess\output_segments\event*')
event_folders = glob.glob(r'.\output_segments\event*\*')
# base_path = r'D:\Pycharm_Project\Find_precursor_signs\ILLgarben\uilt\01data_preprocess\output_segments\*'
import re

def event_sort_key(path):
    m = re.search(r'event(\d+)', path)
    return int(m.group(1)) if m else 999999

event_folders = sorted(event_folders, key=event_sort_key)  #[104:105]
# print(event_folders)
# 2. STFT 保存路径
save_path = r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben\data\STFT_oneday_all"
os.makedirs(save_path, exist_ok=True)

# 3. 遍历每个事件文件夹，读取所有 SAC 文件
for event_folder in tqdm(event_folders, desc="Processing events"):
    sac_files = glob.glob(os.path.join(event_folder, '*.sac'))
    if not sac_files:
        continue  # 如果没有sac文件，跳过

    stft_data_list = []
    files = []

    for sac_file in sac_files:
        st = read(sac_file)
        stft = compute_STFT_maxmin(st[0].data)
        stft_data_list.append(stft)
        files.append(sac_file)

    # 保存 npz 文件，命名为 event 文件夹名
    # 获取最后一级（如 E）
    channel_name = os.path.basename(event_folder)

    # 获取倒数第二级（如 event1）
    event_name = os.path.basename(os.path.dirname(event_folder))
    # 构建保存目录路径，并确保存在
    save_dir = os.path.join(save_path, channel_name)
    os.makedirs(save_dir, exist_ok=True)
    npz_filename = os.path.join(save_dir, f"{event_name}_stft_maxmin_{channel_name}.npz")
    np.savez(npz_filename, stft=stft_data_list, files=files)
    a = 1
    # np.savez(npz_filename, stft=stft_data_list, files=files)
