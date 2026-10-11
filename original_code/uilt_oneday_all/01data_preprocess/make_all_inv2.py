# -*- coding: utf-8 -*-
'''
@Time    : 2025/6/17 11:39
@Author  : HuangZhiyong
@email   : 1524338616@qq.com
@File    : make_all_inv2.py
@Function: 复制并添加标准通道
'''

from obspy.core.inventory import read_inventory, Inventory, Network, Station, Channel, Site
from obspy.core import UTCDateTime
import glob
import os

response_paths = glob.glob(r"D:\Pycharm_Project\Open_Source_Data_Download\Down_Debris_flow\XML\origin\*.xml")
network_code = '9S'
new_station_codes = ['ILL18', 'ILL08', 'ILL12', 'ILL14', 'ILL16', 'ILL17', 'ILL19', 'ILL06']

for path in response_paths:
    filename = os.path.splitext(os.path.basename(path))[0]  # 'ILL11'
    station_code = filename  # 原 station 代码不包含 '.xml'

    inventory = read_inventory(path)
    modified_stations = []

    for new_station_code in new_station_codes:
        inventory = read_inventory(path)  # 每次重新加载，防止污染原数据

        for net in inventory:
            if net.code == network_code:
                for sta in net:
                    if sta.code == station_code:
                        # 创建新站
                        new_station = Station(
                            code=new_station_code,
                            latitude=sta.latitude,
                            longitude=sta.longitude,
                            elevation=sta.elevation,
                            creation_date=sta.creation_date,
                            site=Site(name=new_station_code),
                            start_date=sta.start_date,
                            end_date=sta.end_date,
                            total_number_of_channels=len(sta.channels),
                            channels=[],
                            description=sta.description
                        )

                        # 添加原通道
                        for channel in sta:
                            new_channel = Channel(
                                code=channel.code,
                                location_code=channel.location_code,
                                latitude=channel.latitude,
                                longitude=channel.longitude,
                                elevation=channel.elevation,
                                depth=channel.depth,
                                azimuth=channel.azimuth,
                                dip=channel.dip,
                                sample_rate=channel.sample_rate,
                                start_date=UTCDateTime(2000, 1, 1),
                                sensor=channel.sensor,
                                response=channel.response
                            )
                            new_station.channels.append(new_channel)

                        # 添加新通道 EHZ, EHE, EHN（以第一个原始通道为模板）
                        if sta.channels:
                            base_channel = sta.channels[0]
                            azimuth_map = {'EHZ': 0.0, 'EHE': 90.0, 'EHN': 0.0}
                            dip_map = {'EHZ': -90.0, 'EHE': 0.0, 'EHN': 0.0}

                            for code in ['EHZ', 'EHE', 'EHN']:
                                extra_channel = Channel(
                                    code=code,
                                    location_code=base_channel.location_code,
                                    latitude=base_channel.latitude,
                                    longitude=base_channel.longitude,
                                    elevation=base_channel.elevation,
                                    depth=base_channel.depth,
                                    azimuth=azimuth_map[code],
                                    dip=dip_map[code],
                                    sample_rate=base_channel.sample_rate,
                                    start_date=UTCDateTime(2000, 1, 1),
                                    sensor=base_channel.sensor,
                                    response=base_channel.response
                                )
                                new_station.channels.append(extra_channel)

                        # 添加进列表
                        if new_station.channels:
                            modified_stations.append(new_station)
                        break

    # 写入新的 StationXML 文件
    if modified_stations:
        network = Network(
            code=network_code,
            stations=modified_stations,
            description=f"Modified stations based on {station_code} template"
        )
        new_inventory = Inventory(
            networks=[network],
            source="HuangZhiyong_Custom"
        )

        output_dir = r"D:\Pycharm_Project\Open_Source_Data_Download\Down_Debris_flow\XML\origin\combine"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"base_{station_code}.xml")
        new_inventory.write(output_path, format="STATIONXML")
        print(f"统一 Inventory 保存到：{output_path}")
    else:
        print(f"{station_code}：没有生成任何 station。")
