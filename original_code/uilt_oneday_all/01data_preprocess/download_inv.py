# -*- coding: utf-8 -*-
'''
@Time    : 2025/6/17 11:01
@Author  : HuangZhiyong
@email   : 1524338616@qq.com
@File    : download_inv.py
@Function: {}:
'''
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from tqdm import tqdm

client = Client("ETH")

stations_list = ['ILL18', 'ILL11', 'ILL01', 'ILL14', 'ILL12', 'ILL17', 'ILL13', 'ILL16']
start_time = UTCDateTime("2019-07-01")
end_time = UTCDateTime("2019-07-02")

for station in tqdm(stations_list):
    try:
        inv = client.get_stations(network="9S", station=station, location="*", channel="*",
                                  starttime=start_time, endtime=end_time, level="response")
        inv.write(fr"D:\Pycharm_Project\Open_Source_Data_Download\Down_Debris_flow\XML\origin\{station}.xml", format="STATIONXML")
    except Exception as e:
        print(f"Failed to download for station {station}: {e}")
