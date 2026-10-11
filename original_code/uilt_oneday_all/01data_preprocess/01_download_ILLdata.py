# -*- coding: utf-8 -*-
'''
@Time    : 2025/6/12 21:06
@Author  : HuangZhiyong
@email   : 1524338616@qq.com
@File    : 01_download_ILLdata.py
@Function: {}:
'''
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import matplotlib.pyplot as plt
from tqdm import trange
from obspy import read, UTCDateTime
from obspy.clients.fdsn import Client
from obspy import read_inventory
from obspy.core.inventory import Inventory, Network, Station
from obspy.clients.fdsn.header import URL_MAPPINGS
from obspy.clients.fdsn.mass_downloader import CircularDomain, Restrictions, MassDownloader
from obspy import read_inventory
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
"下载论文Machine Learning Improves Debris Flow Warning中的数据"
client = Client("ETH")

data = pd.read_excel(r"E:\Data_Open_Source\Debers_Flow\Combine.xlsx")
data["Start_Time"] = pd.to_datetime(data["Date"].astype(str) + 'T' + data["Arrival time"].astype(str))
starttimes = data["Start_Time"]
stations = data['station']
volume = data['Volume']

# data = pd.read_excel(r"E:\Data_Open_Source\Debers_Flow\Train_Data.xlsx")
# data["Start_Time"] = pd.to_datetime(data["Date"].astype(str) + 'T' + data["Arrival time"].astype(str))
# starttimes = data["Start_Time"]

# station_efficiently = pd.read_excel(r"E:\Data_Open_Source\Debers_Flow\Train\Train_Time.xlsx")
# stations = ["ILL18"]
events = ["event"]
records = []

for j in trange(0,len(data["Start_Time"])):
# for j in trange(0,5):
    event_start_str = data["Start_Time"][j]
    event_utc = UTCDateTime(event_start_str)

    start_time = UTCDateTime(data["Start_Time"][j]) - 3600 * 24 * 10  #24*7
    end_time = UTCDateTime(data["Start_Time"][j]) + 3600 * 9
    records.append({
                "Station": stations[j],
                "Event_Start (str)": event_start_str,
                "UTC_Time": event_utc.isoformat(),
                "Start_Time (-7day)": start_time.isoformat(),
                "End_Time (+9h)": end_time.isoformat(),
                'Volume': volume[j]
            })
    try:
        new_station_codes = ['ILL18', 'ILL08', 'ILL12', 'ILL14', 'ILL16', 'ILL17', 'ILL19', 'ILL06']

        waveforms = client.get_waveforms(network="9S", station = stations[j], location="*", channel="*", starttime=start_time, endtime=end_time)
        waveforms = waveforms.merge(fill_value='interpolate')

        # inventory = client.get_stations(network="9S", station=stations[i], location="*", channel="*",
        #                                 starttime=start_time, endtime=end_time, level="response")
        inventory = client.get_stations(network="9S", station=stations[j], location="*", channel="*",
                                        level="response")
        inventory.write(fr"D:\Pycharm_Project\Open_Source_Data_Download\Down_Debris_flow\XML\{stations[j]}.xml", format="STATIONXML")

        station_info = client.get_stations(network="9S", station=stations[j], starttime=start_time, endtime=end_time)

        # waveforms.plot()
        network_code = "9S"
        station_code = stations[j]
        starttime_str = start_time.strftime("%Y%m%dT%H%M%S")
        """筛选高频信号"""
        high_freq_waveforms = []  # 用于存储高频信号的列表
        for trace in waveforms:
            # 计算采样率
            sampling_rate = trace.stats.sampling_rate
            # 设置采样率阈值，例如100 Hz
            high_freq_threshold = 100

            # 如果采样率大于阈值，则认为是高频信号
            if sampling_rate == high_freq_threshold:
                high_freq_waveforms.append(trace)
        # 如果高频信号的数量大于3，则只保留高频信号
        if len(high_freq_waveforms) == 3:
            waveforms = high_freq_waveforms
        for trace in waveforms:
            tr = trace.copy()
            # tr.remove_response(inventory=inventory)
            # tr.filter("highpass",  freq=5, corners=4, zerophase=True)
            # plt.plot(tr.times("matplotlib"), tr.data, label=trace.stats.channel)
            # plt.show()

            trace.stats.latitude = station_info[0][0].latitude
            trace.stats.longitude = station_info[0][0].longitude
            trace.stats.elevation = station_info[0][0].elevation
            channel_code = trace.stats.channel[-3:]  # Extract last three characters (e.g., EHE, EHN, EHZ)
            filename = f"{network_code}.{station_code}.{starttime_str}.{channel_code}.sac"
            print(filename)
            save_path = rf"E:\Data_Open_Source\Debers_Flow\ILL18\sac\{channel_code[-1]}\event{str(j)}"
            print(os.path.join(save_path, filename))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print(f"Saved waveform data as {filename}")
            trace.write(os.path.join(save_path, filename), format='SAC')

    except Exception as e:
        print(f"Error: {e}")
        continue
# 转换成 DataFrame
df_out = pd.DataFrame(records)

# 保存为 Excel 文件
df_out.to_excel(r"E:\Data_Open_Source\Debers_Flow\ILL18\station_time_windows.xlsx", index=False)