# -*- coding: utf-8 -*-
'''
@Time    : 2026/8/27 19:49
@Author  : HuangZhiyong
@email   : 1524338616@qq.com
@File    : map.py
@Function: {}:
'''
import os

os.environ["GDAL_DRIVER_PATH"] = (
    r"D:\Anaconda\envs\pygmt\Library\lib\gdalplugins"
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LightSource, LinearSegmentedColormap
from scipy.ndimage import distance_transform_edt
from shapely import contains_xy

import pygmt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader


# ============================================================
# 1. 参数
# ============================================================

# 绘图范围：[西, 东, 南, 北]
region = [5.65, 10.65, 45.65, 47.90]

# 图中十字线所代表的位置
site_lon = 7 + 35 / 60       # 7°35′ E
site_lat = 46 + 25 / 60      # 46°25′ N

cities = {
    "Geneva":   (6.1432, 46.2044),
    "Lausanne": (6.6323, 46.5197),
    "Bern":     (7.4474, 46.9480),
    "Zürich":   (8.5417, 47.3769),
    "Chur":     (9.5300, 46.8508),
}

# 标签位置可根据需要微调
city_labels = {
    "Geneva":   (5.88, 46.02),
    "Lausanne": (6.30, 46.68),
    "Bern":     (7.10, 47.08),
    "Zürich":   (8.52, 47.58),
    "Chur":     (9.35, 47.02),
}


# ============================================================
# 2. 获取瑞士国界
# ============================================================

shp_path = shpreader.natural_earth(
    resolution="10m",
    category="cultural",
    name="admin_0_countries",
)

switzerland = None

for record in shpreader.Reader(shp_path).records():
    attrs = record.attributes

    if (
        attrs.get("ADM0_A3") == "CHE"
        or attrs.get("SOV_A3") == "CHE"
        or attrs.get("ISO_A3") == "CHE"
    ):
        switzerland = record.geometry
        break

if switzerland is None:
    raise RuntimeError("没有在 Natural Earth 数据中找到瑞士国界。")


# ============================================================
# 3. 下载 DEM
# ============================================================

# 15s 约为 450 m 分辨率，绘制全国图通常已经足够。
# 若需要更精细，可改为 resolution="03s"，但下载量更大。
dem = pygmt.datasets.load_earth_relief(
    resolution="15s",
    region=region,
)

lon = np.asarray(dem.coords["lon"].values)
lat = np.asarray(dem.coords["lat"].values)
elevation = np.asarray(dem.values, dtype=float)

# 保证数组方向与 imshow(origin="lower") 一致
if lat[0] > lat[-1]:
    lat = lat[::-1]
    elevation = elevation[::-1, :]

if lon[0] > lon[-1]:
    lon = lon[::-1]
    elevation = elevation[:, ::-1]


# ============================================================
# 4. 根据瑞士国界创建掩膜
# ============================================================

xx, yy = np.meshgrid(lon, lat)
inside = contains_xy(switzerland, xx, yy)

# 用距离最近的境内像元填充境外区域。
# 这样计算 hillshade 时不会在国界处产生黑边。
nearest_indices = distance_transform_edt(
    ~inside,
    return_distances=False,
    return_indices=True,
)
elevation_filled = elevation[tuple(nearest_indices)]


# ============================================================
# 5. 计算阴影地形
# ============================================================

mean_lat = np.mean(lat)

# 将经纬度分辨率近似换算成米
dx = abs(np.mean(np.diff(lon))) * 111_320 * np.cos(np.deg2rad(mean_lat))
dy = abs(np.mean(np.diff(lat))) * 110_540

grey_cmap = LinearSegmentedColormap.from_list(
    "relief_grey",
    ["#a9a9a9", "#eeeeee"],
)

light = LightSource(
    azdeg=315,   # 光源来自西北
    altdeg=42,
)

valid_values = elevation[inside]

relief = light.shade(
    elevation_filled,
    cmap=grey_cmap,
    vert_exag=2.0,
    dx=dx,
    dy=dy,
    vmin=np.percentile(valid_values, 2),
    vmax=np.percentile(valid_values, 98),
    blend_mode="soft",
)

relief = np.array(relief, copy=True)
relief[..., 3] = inside.astype(float)  # 瑞士境外完全透明


# ============================================================
# 6. 绘图
# ============================================================

data_crs = ccrs.PlateCarree()
map_crs = ccrs.UTM(zone=32)

fig = plt.figure(figsize=(7.2, 4.3))
ax = fig.add_subplot(1, 1, 1, projection=map_crs)

ax.set_extent(region, crs=data_crs)
ax.set_facecolor("white")

# 阴影地形
ax.imshow(
    relief,
    extent=[lon.min(), lon.max(), lat.min(), lat.max()],
    origin="lower",
    transform=data_crs,
    interpolation="bilinear",
    zorder=1,
)

# 国界
ax.add_geometries(
    [switzerland],
    crs=data_crs,
    facecolor="none",
    edgecolor="#777777",
    linewidth=0.55,
    zorder=2,
)

# 十字经纬线
ax.plot(
    [site_lon, site_lon],
    [region[2] + 0.08, region[3] - 0.05],
    color="#333333",
    linewidth=1.1,
    transform=data_crs,
    zorder=3,
)

ax.plot(
    [region[0] + 0.08, region[1] - 0.08],
    [site_lat, site_lat],
    color="#333333",
    linewidth=1.1,
    transform=data_crs,
    zorder=3,
)

# 城市点
for name, (x, y) in cities.items():
    ax.scatter(
        x, y,
        s=54,
        color="black",
        edgecolor="black",
        transform=data_crs,
        zorder=5,
    )

# 标签白色描边，避免被地形纹理遮挡
text_effect = [
    pe.withStroke(linewidth=2.8, foreground="white", alpha=0.9)
]

for name, (tx, ty) in city_labels.items():
    ax.text(
        tx, ty, name,
        transform=data_crs,
        ha="center",
        va="center",
        fontsize=13,
        fontfamily="DejaVu Serif",
        fontweight="bold",
        color="black",
        path_effects=text_effect,
        zorder=6,
    )

# 研究点
ax.scatter(
    site_lon,
    site_lat,
    s=72,
    color="red",
    edgecolor="red",
    transform=data_crs,
    zorder=7,
)

ax.text(
    site_lon + 0.30,
    site_lat + 0.15,
    "ILLgraben",
    transform=data_crs,
    ha="center",
    va="center",
    fontsize=13,
    fontfamily="DejaVu Serif",
    fontweight="bold",
    path_effects=text_effect,
    zorder=7,
)

# 经纬度文字
ax.text(
    site_lon,
    region[2] + 0.02,
    r"$7^\circ35^\prime\ \mathrm{E}$",
    transform=data_crs,
    ha="center",
    va="bottom",
    fontsize=13,
    fontfamily="DejaVu Serif",
    fontweight="bold",
    clip_on=False,
    zorder=7,
)

ax.text(
    region[1] - 0.08,
    site_lat - 0.12,
    r"$46^\circ25^\prime\ \mathrm{N}$",
    transform=data_crs,
    ha="right",
    va="top",
    fontsize=13,
    fontfamily="DejaVu Serif",
    fontweight="bold",
    clip_on=False,
    zorder=7,
)

ax.set_axis_off()

plt.savefig(
    r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben\uilt_oneday_all\uilt_paper_figure\map\switzerland_relief_map.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.03,
    facecolor="white",
)
plt.savefig(
    r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben\uilt_oneday_all\uilt_paper_figure\map\switzerland_relief_map.pdf",
    bbox_inches="tight",
    pad_inches=0.03,
    facecolor="white",
)
plt.show()