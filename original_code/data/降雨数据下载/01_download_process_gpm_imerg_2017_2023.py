# -*- coding: utf-8 -*-
"""
GPM IMERG Final V07 local download + point extraction.

解决问题：
1. 避免 earthaccess.open() 远程流式读取导致 403 Forbidden
2. 先下载 HDF5 到本地，再从本地读取
3. 修正月末时间，避免 2017-01 搜到 1536 个文件
4. 仅下载 2017-05 到 2023-01
5. 下载失败时按缺失文件重试，减少超时/502 导致的漏档
"""

from pathlib import Path
from urllib.parse import urlparse
import warnings
import calendar
import time

import earthaccess
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


# ============================================================
# 1. 参数设置
# ============================================================

START_YEAR = 2019
START_MONTH = 3

END_YEAR = 2019
END_MONTH = 3

TARGET_LON = 7.63
TARGET_LAT = 46.28

# west, south, east, north
BBOX = (7.3, 46.0, 7.9, 46.6)

OUT_DIR = Path(r"H:\Illgraben\GPM_IMERG_Illgraben")
RAW_DIR = OUT_DIR / "raw_hdf5"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

SHORT_NAME = "GPM_3IMERGHH"
VERSION = "07"
PRECIP_VAR = "precipitation"

DOWNLOAD_RETRY_ROUNDS = 4
RETRY_WAIT_SECONDS = 8


# ============================================================
# 2. 时间函数
# ============================================================

def month_start_end(year, month):
    """
    返回某个月精确起止时间。
    注意：end 设为本月最后一天 23:59:59，
    避免把下个月 1 日也搜进来。
    """
    last_day = calendar.monthrange(year, month)[1]

    start = f"{year}-{month:02d}-01T00:00:00"
    end = f"{year}-{month:02d}-{last_day:02d}T23:59:59"

    return start, end


def iter_target_months():
    """按起止年月迭代目标月份。"""
    year = START_YEAR
    month = START_MONTH

    while (year, month) <= (END_YEAR, END_MONTH):
        yield year, month
        month += 1
        if month > 12:
            month = 1
            year += 1


def parse_time_from_filename(filename):
    """
    从 IMERG 文件名解析时间。
    例：
    3B-HHR.MS.MRG.3IMERG.20170101-S183000-E185959.1110.V07B.HDF5
    """
    name = Path(str(filename)).name

    try:
        parts = name.split(".")
        date_part = parts[4]
        ymd = date_part.split("-")[0]
        start_part = date_part.split("-")[1]
        hms = start_part[1:]

        return pd.to_datetime(ymd + hms, format="%Y%m%d%H%M%S", utc=True)
    except Exception:
        return pd.NaT


def get_time_from_dataset(ds, filename):
    try:
        if "time" in ds.coords:
            t = np.asarray(ds["time"].values).ravel()[0]
            return pd.to_datetime(t, utc=True)
    except Exception:
        pass

    return parse_time_from_filename(filename)


def extract_filename_from_granule(granule):
    """
    尽量从 earthaccess 返回的 granule 中提取文件名。
    优先使用下载 URL 的 basename。
    """
    link_candidates = []

    if hasattr(granule, "data_links"):
        try:
            link_candidates.extend(granule.data_links())
        except Exception:
            pass

    if hasattr(granule, "umm"):
        try:
            related_urls = granule.umm.get("RelatedUrls", [])
            link_candidates.extend(item.get("URL") for item in related_urls if item.get("URL"))
        except Exception:
            pass

    if isinstance(granule, dict):
        try:
            for item in granule.get("umm", {}).get("RelatedUrls", []):
                if item.get("URL"):
                    link_candidates.append(item["URL"])
        except Exception:
            pass

    for link in link_candidates:
        try:
            name = Path(urlparse(link).path).name
            if name.upper().endswith(".HDF5"):
                return name
        except Exception:
            continue

    for candidate in ("meta",):
        meta = getattr(granule, candidate, None)
        if isinstance(meta, dict):
            native_id = meta.get("native-id") or meta.get("native_id")
            if isinstance(native_id, str) and native_id.upper().endswith(".HDF5"):
                return native_id

    if isinstance(granule, dict):
        native_id = granule.get("meta", {}).get("native-id") or granule.get("meta", {}).get("native_id")
        if isinstance(native_id, str) and native_id.upper().endswith(".HDF5"):
            return native_id

    return None


def get_expected_filenames(results):
    """为 granule 列表建立 预期文件名 -> granule 映射。"""
    mapping = {}

    for granule in results:
        filename = extract_filename_from_granule(granule)
        if filename:
            mapping[filename] = granule

    return mapping


# ============================================================
# 3. 降雨提取函数
# ============================================================

def extract_point_rainfall_from_file(hdf5_file):
    """
    从本地 HDF5 文件中提取最近格点降雨。
    IMERG precipitation 单位通常为 mm/h。
    半小时降雨量 = mm/h * 0.5。
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        ds = xr.open_dataset(
            hdf5_file,
            group="Grid",
            engine="h5netcdf"
        )

        if PRECIP_VAR not in ds.data_vars:
            raise KeyError(f"找不到变量 {PRECIP_VAR}，当前变量为：{list(ds.data_vars)}")

        da = ds[PRECIP_VAR]

        p = da.sel(
            lon=TARGET_LON,
            lat=TARGET_LAT,
            method="nearest"
        ).squeeze()

        rain_rate_mm_h = float(np.asarray(p.values).ravel()[0])

        if not np.isfinite(rain_rate_mm_h):
            rain_rate_mm_h = 0.0

        rain_mm_30min = rain_rate_mm_h * 0.5

        grid_lon = float(np.asarray(p["lon"].values).ravel()[0]) if "lon" in p.coords else np.nan
        grid_lat = float(np.asarray(p["lat"].values).ravel()[0]) if "lat" in p.coords else np.nan

        time_utc = get_time_from_dataset(ds, hdf5_file)

        ds.close()

    return {
        "time_utc": time_utc,
        "rain_rate_mm_h": rain_rate_mm_h,
        "rain_mm_30min": rain_mm_30min,
        "grid_lon": grid_lon,
        "grid_lat": grid_lat,
        "source": "GPM_IMERG_Final_V07",
        "file": Path(hdf5_file).name,
    }


# ============================================================
# 4. 下载函数
# ============================================================

def download_missing_granules(raw_month_dir, missing_map):
    """仅下载当前仍缺失的 granule，并进行多轮重试。"""
    if not missing_map:
        return

    remaining_map = dict(missing_map)

    for attempt in range(1, DOWNLOAD_RETRY_ROUNDS + 1):
        missing_names = sorted(remaining_map)
        print(f"开始下载到本地。第 {attempt}/{DOWNLOAD_RETRY_ROUNDS} 轮，待下载 {len(missing_names)} 个文件。")

        earthaccess.download(
            list(remaining_map.values()),
            local_path=str(raw_month_dir)
        )

        local_name_set = {p.name for p in raw_month_dir.glob("*.HDF5")}
        remaining_map = {
            name: granule
            for name, granule in remaining_map.items()
            if name not in local_name_set
        }

        if not remaining_map:
            print("本轮缺失文件已补齐。")
            return

        print(f"本轮后仍缺失 {len(remaining_map)} 个文件。")
        if attempt < DOWNLOAD_RETRY_ROUNDS:
            print(f"等待 {RETRY_WAIT_SECONDS} 秒后重试。")
            time.sleep(RETRY_WAIT_SECONDS)

    missing_names = sorted(remaining_map)
    preview = missing_names[:10]
    print(f"警告：重试后仍缺失 {len(missing_names)} 个文件。示例：{preview}")


def download_one_month(year, month):
    """
    下载某个月的 GPM IMERG HDF5 文件到本地。
    通过预期文件名校验缺失项，避免因超时/502 留下漏档。
    """
    start, end = month_start_end(year, month)

    raw_month_dir = RAW_DIR / f"{year}{month:02d}"
    raw_month_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n搜索 GPM IMERG: {start} 到 {end}")

    results = earthaccess.search_data(
        short_name=SHORT_NAME,
        version=VERSION,
        temporal=(start, end),
        bounding_box=BBOX,
    )

    print(f"找到 granule 数量：{len(results)}")

    if len(results) == 0:
        return []

    expected_map = get_expected_filenames(results)
    if len(expected_map) == 0:
        print("警告：未能从 granule 中解析预期文件名，将直接返回当前本地文件。")
        return sorted(raw_month_dir.glob("*.HDF5"))

    local_name_set = {p.name for p in raw_month_dir.glob("*.HDF5")}
    missing_map = {
        name: granule
        for name, granule in expected_map.items()
        if name not in local_name_set
    }

    print(f"预期文件数：{len(expected_map)}，本地已存在：{len(local_name_set & set(expected_map))}，缺失：{len(missing_map)}")

    if missing_map:
        download_missing_granules(raw_month_dir, missing_map)
    else:
        print("本地文件已完整，跳过下载。")

    local_files = sorted(
        p for p in raw_month_dir.glob("*.HDF5")
        if p.name in expected_map
    )

    print(f"本月本地 HDF5 文件数：{len(local_files)} / {len(expected_map)}")
    return local_files


# ============================================================
# 5. 月处理函数
# ============================================================

def process_one_month(year, month):
    """
    下载并处理一个月。
    """
    out_month_csv = OUT_DIR / f"GPM_IMERG_Illgraben_{year}{month:02d}.csv"

    if out_month_csv.exists() and out_month_csv.stat().st_size > 0:
        print(f"已有月结果，跳过：{out_month_csv}")
        df = pd.read_csv(out_month_csv)
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
        return df

    hdf5_files = download_one_month(year, month)

    if len(hdf5_files) == 0:
        print(f"{year}-{month:02d} 没有下载到 HDF5 文件。")
        return pd.DataFrame()

    records = []

    for f in tqdm(hdf5_files, desc=f"处理 GPM {year}-{month:02d}"):
        try:
            rec = extract_point_rainfall_from_file(f)
            records.append(rec)

        except Exception as e:
            print(f"跳过文件：{f.name}，原因：{e}")

    if len(records) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.dropna(subset=["time_utc"])
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df.drop_duplicates(subset=["time_utc"])
    df = df.sort_values("time_utc")

    df.to_csv(out_month_csv, index=False, encoding="utf-8-sig")
    print(f"保存月结果：{out_month_csv}")

    return df


# ============================================================
# 6. 滚动降雨统计
# ============================================================

def add_rolling_features_gpm(df):
    df = df.copy()

    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df.sort_values("time_utc")
    df = df.drop_duplicates(subset=["time_utc"])

    df = df.set_index("time_utc")

    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="30min",
        tz="UTC"
    )

    df = df.reindex(full_index)
    df.index.name = "time_utc"

    df["rain_rate_mm_h"] = df["rain_rate_mm_h"].fillna(0.0)
    df["rain_mm_30min"] = df["rain_mm_30min"].fillna(0.0)

    windows = {
        "cum_1h_mm": 2,
        "cum_4h_mm": 8,
        "cum_6h_mm": 12,
        "cum_8h_mm": 16,
        "cum_12h_mm": 24,
        "cum_24h_mm": 48,
        "cum_48h_mm": 96,
        "cum_72h_mm": 144,
    }

    for col, n in windows.items():
        df[col] = df["rain_mm_30min"].rolling(n, min_periods=1).sum()

    df["max_30min_mm_last_24h"] = (
        df["rain_mm_30min"]
        .rolling(48, min_periods=1)
        .max()
    )

    df["max_1h_mm_last_24h"] = (
        df["rain_mm_30min"]
        .rolling(2, min_periods=1)
        .sum()
        .rolling(48, min_periods=1)
        .max()
    )

    return df.reset_index()


# ============================================================
# 7. 主程序
# ============================================================

def main():
    # 重新交互式登录，并保存认证信息
    auth = earthaccess.login(strategy="interactive", persist=True)

    if not auth.authenticated:
        raise RuntimeError("Earthdata 登录失败。请检查用户名、密码和 GES DISC 授权。")

    all_dfs = []

    for year, month in iter_target_months():
        df_month = process_one_month(year, month)

        if len(df_month) > 0:
            all_dfs.append(df_month)

    if len(all_dfs) == 0:
        raise RuntimeError("没有成功处理任何 GPM 数据。")

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all["time_utc"] = pd.to_datetime(df_all["time_utc"], utc=True)
    df_all = df_all.drop_duplicates(subset=["time_utc"])
    df_all = df_all.sort_values("time_utc")

    out_all_csv = OUT_DIR / f"GPM_IMERG_Illgraben_{START_YEAR}{START_MONTH:02d}_{END_YEAR}{END_MONTH:02d}_30min.csv"
    df_all.to_csv(out_all_csv, index=False, encoding="utf-8-sig")
    print(f"\n保存总降雨序列：{out_all_csv}")

    df_roll = add_rolling_features_gpm(df_all)

    out_roll_csv = OUT_DIR / f"GPM_IMERG_Illgraben_{START_YEAR}{START_MONTH:02d}_{END_YEAR}{END_MONTH:02d}_30min_roll.csv"
    out_roll_xlsx = OUT_DIR / f"GPM_IMERG_Illgraben_{START_YEAR}{START_MONTH:02d}_{END_YEAR}{END_MONTH:02d}_30min_roll.xlsx"

    df_roll.to_csv(out_roll_csv, index=False, encoding="utf-8-sig")
    df_roll.to_excel(out_roll_xlsx, index=False)

    print(f"保存滚动统计 CSV：{out_roll_csv}")
    print(f"保存滚动统计 Excel：{out_roll_xlsx}")


if __name__ == "__main__":
    main()
