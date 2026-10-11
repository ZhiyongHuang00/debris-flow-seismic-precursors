# -*- coding: utf-8 -*-
"""
Plot publication-ready waveforms and spectrograms for three debris-flow events.

This script produces panel b of Figure 1. Internal event rows are deliberately
not lettered, avoiding conflicts with the main-panel label.

Main improvements:
    1. Automatically display the complete duration of the longest event.
    2. Read adjacent SAC files in the same directory when necessary.
    3. Merge multiple SAC files before preprocessing.
    4. Display the complete waveform amplitude without clipping.
    5. Use a common PSD color scale for all three events.
    6. Save both high-resolution PNG and vector PDF.

Author: Huang Zhiyong
"""

from __future__ import annotations

import glob
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

from obspy import Stream, UTCDateTime, read
from scipy.signal import spectrogram


# ==========================================================
# 1. PATHS
# ==========================================================

ROOT = Path(
    r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben"
)

EXCEL = ROOT / r"data\新版完整数据集时间.xlsx"

OUT = (
    ROOT
    / r"uilt_oneday_all"
    / r"uilt_paper_figure"
    / r"Fig1_three_events"
)


# ==========================================================
# 2. SELECTED EVENTS
# ==========================================================

# 必须正好填写3个事件
SELECTED_EVENTS = [
    "event2",
    "event28",
    "event1",
]


# ==========================================================
# 3. TIME PARAMETERS
# ==========================================================

# 启动前显示时长
PRE_H = 24.0

# 最长泥石流事件结束后继续显示的时间
POST_EVENT_BUFFER_H = 0.5

# 是否限制启动后的最大显示时长
#
# None:
#   完整显示3个事件中持续时间最长的事件。
#
# 例如设为3.0:
#   最多显示到启动后3 h，适合正文图，但可能截断长事件。
MAX_POST_H = 3.0


# ==========================================================
# 4. PREPROCESSING PARAMETERS
# ==========================================================

# 高通滤波截止频率
HP_FREQ = 1.0

# Butterworth滤波器阶数
HP_CORNERS = 4

# 是否自动读取SAC文件所在目录中的相邻文件
#
# 当事件跨越多个日文件时，需要设置为True。
READ_ADJACENT_FILES = True

# 是否优先选择Z分量
USE_Z_COMPONENT = True

# SAC文件扩展名
SUPPORTED_SUFFIXES = {
    ".sac",
    ".mseed",
    ".miniseed",
    ".seed",
    ".msd",
}


# ==========================================================
# 5. WAVEFORM DISPLAY PARAMETERS
# ==========================================================

# 每多少秒计算一个最小值—最大值包络
#
# 该参数只影响绘图，不影响PSD计算。
WAVEFORM_BIN_SEC = 5.0

# 波形归一化采用的绝对振幅百分位
#
# 99.99表示以99.99百分位振幅作为归一化尺度。
WAVEFORM_NORMALIZE_PERCENTILE = 99.99

# 波形纵轴模式：
#
# "common":
#   所有事件采用相同的归一化纵轴，便于跨事件比较。
#
# "full":
#   使用包络绝对最大值确定纵轴，保证波形完全显示。
#
# "percentile":
#   使用WAVEFORM_DISPLAY_PERCENTILE确定纵轴，
#   可以避免极少数异常尖峰压缩整个波形。
WAVEFORM_Y_MODE = "common"

WAVEFORM_DISPLAY_PERCENTILE = 99.95

# 波形纵轴上下留白比例
WAVEFORM_Y_MARGIN = 1.05

# Robustly normalized waveforms share one display scale across all events.
WAVEFORM_CLIP_LIMIT = 1.0
COMMON_WAVEFORM_LIMIT = 1.05


# ==========================================================
# 6. SPECTROGRAM PARAMETERS
# ==========================================================

# scipy.signal.spectrogram的窗长
#
# 对100 Hz数据，1024点约为10.24 s。
SPEC_NPERSEG = 1024

# 相邻短时频谱的重叠比例
SPEC_OVERLAP = 0.50

# 将短时频谱平均为多少秒一个时间格
#
# 60 s表示每分钟一个PSD时间格。
SPEC_TIME_BIN_SEC = 60.0

# 显示频率范围
PSD_FREQ = (0.0, 50.0)

# 频谱配色
PSD_CMAP = "magma"

# 公共色标范围
#
# None:
#   根据3个事件自动确定公共色标。
#
# 固定范围示例：
# POWER_DB_RANGE = (-180, -100)
POWER_DB_RANGE = None

# 自动色标采用的百分位
COLOR_PERCENTILES = (2.0, 99.0)

# Overlay the 1-min, frequency-band-mean PSD on each spectrogram.
# The mean is calculated in linear PSD space and converted to dB
# afterwards. Set to False to recover the spectrogram-only layout.
SHOW_MEAN_PSD_CURVE = True
MEAN_PSD_CURVE_COLOR = "#19D3C5"
MEAN_PSD_CURVE_LINE_WIDTH = 1.15
MEAN_PSD_CURVE_ALPHA = 0.95
MEAN_PSD_HALO_LINE_WIDTH = 2.4
MEAN_PSD_HALO_ALPHA = 0.55
MEAN_PSD_LIMIT_PERCENTILES = (1.0, 99.5)

EPS = 1e-20


# ==========================================================
# 7. FIGURE PARAMETERS
# ==========================================================

FIG_SIZE = (8, 10)

# Centralized typography controls for the complete Figure 1b panel.
# Values are specified in points for the current 5.4-inch-wide panel.
FONT_SIZE_BASE = 15
FONT_SIZE_LABEL = 15
FONT_SIZE_TICK = 14
FONT_SIZE_EVENT_TITLE = 14
FONT_SIZE_COLORBAR_LABEL = 15
FONT_SIZE_COLORBAR_TICK = 14
FONT_SIZE_MEAN_PSD_LABEL = 12
FONT_SIZE_MEAN_PSD_TICK = 11
FONT_SIZE_PANEL = 18.0
FONT_WEIGHT_EVENT_TITLE = "semibold"

# Tick and label spacing.
TICK_LENGTH = 2.5
TICK_WIDTH = 0.7
TICK_LABEL_PAD = 1.5
COLORBAR_LABEL_PAD = 3.0

# Line widths.
AXES_LINE_WIDTH = 0.8
WAVEFORM_BASELINE_LINE_WIDTH = 0.45
INITIATION_WAVEFORM_LINE_WIDTH = 0.9
INITIATION_SPECTROGRAM_HALO_LINE_WIDTH = 1.4
INITIATION_SPECTROGRAM_LINE_WIDTH = 0.65
EVENT_END_LINE_WIDTH = 0.9

# Panel spacing and page margins.
MAIN_GRID_HSPACE = 0.18
EVENT_GRID_HSPACE = 0.12
FIGURE_LEFT = 0.125
FIGURE_RIGHT = 0.992
FIGURE_BOTTOM = 0.070
FIGURE_TOP = 0.935

# 波形与频谱的相对高度
EVENT_HEIGHT_RATIOS = (0.48, 1.0)

# 保存分辨率
SAVE_DPI = 600


# ==========================================================
# 8. MATPLOTLIB STYLE
# ==========================================================

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT_SIZE_BASE,
        "axes.labelsize": FONT_SIZE_LABEL,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "axes.linewidth": AXES_LINE_WIDTH,
        "xtick.major.width": TICK_WIDTH,
        "ytick.major.width": TICK_WIDTH,
        "xtick.major.size": TICK_LENGTH,
        "ytick.major.size": TICK_LENGTH,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "legend.frameon": False,
    }
)


# ==========================================================
# 9. READ XLSX WITHOUT PANDAS
# ==========================================================

def col_index(cell_ref: str) -> int:
    """
    Convert an Excel column reference to a zero-based index.

    Examples
    --------
    A  -> 0
    B  -> 1
    AA -> 26
    """

    letters = "".join(
        character
        for character in cell_ref
        if character.isalpha()
    )

    number = 0

    for character in letters:
        number = (
            number * 26
            + ord(character.upper())
            - 64
        )

    return number - 1


def read_xlsx(path: Path) -> list[dict]:
    """
    Read the first worksheet of an xlsx file without pandas.

    Parameters
    ----------
    path
        Excel file path.

    Returns
    -------
    list of dict
        Each Excel row is converted to a dictionary.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Excel文件不存在：{path}"
        )

    namespace = {
        "x":
        "http://schemas.openxmlformats.org/"
        "spreadsheetml/2006/main"
    }

    with zipfile.ZipFile(path) as zip_file:

        # --------------------------------------------------
        # Shared strings
        # --------------------------------------------------

        try:
            shared_strings_xml = ET.fromstring(
                zip_file.read(
                    "xl/sharedStrings.xml"
                )
            )

            shared_strings = [
                "".join(
                    text_node.text or ""
                    for text_node in item.findall(
                        ".//x:t",
                        namespace
                    )
                )
                for item in shared_strings_xml
            ]

        except KeyError:
            shared_strings = []

        # --------------------------------------------------
        # First worksheet
        # --------------------------------------------------

        worksheet_xml = ET.fromstring(
            zip_file.read(
                "xl/worksheets/sheet1.xml"
            )
        )

        raw_rows = []

        for row in worksheet_xml.findall(
            ".//x:sheetData/x:row",
            namespace
        ):
            values = []

            for cell in row.findall(
                "x:c",
                namespace
            ):
                index = col_index(
                    cell.attrib["r"]
                )

                while len(values) <= index:
                    values.append("")

                cell_type = cell.attrib.get("t")

                if cell_type == "s":

                    value = cell.findtext(
                        "x:v",
                        default="",
                        namespaces=namespace
                    )

                    values[index] = (
                        shared_strings[int(value)]
                        if value else ""
                    )

                elif cell_type == "inlineStr":

                    values[index] = "".join(
                        node.text or ""
                        for node in cell.findall(
                            ".//x:t",
                            namespace
                        )
                    )

                else:

                    values[index] = cell.findtext(
                        "x:v",
                        default="",
                        namespaces=namespace
                    )

            raw_rows.append(values)

    if not raw_rows:
        raise ValueError(
            f"Excel文件中没有有效数据：{path}"
        )

    header = [
        str(value).strip()
        for value in raw_rows[0]
    ]

    rows = []

    for raw_row in raw_rows[1:]:

        row_dict = {
            header[index]:
            (
                raw_row[index]
                if index < len(raw_row)
                else ""
            )
            for index in range(
                len(header)
            )
        }

        rows.append(row_dict)

    return rows


# ==========================================================
# 10. EVENT AND TIME UTILITIES
# ==========================================================

def utc_time(value) -> UTCDateTime:
    """
    Convert an Excel serial date or a datetime string
    to ObsPy UTCDateTime.
    """

    text = str(value).strip()

    if not text:
        raise ValueError(
            "时间字段为空。"
        )

    # Excel serial date
    if re.fullmatch(
        r"\d+(\.\d+)?",
        text
    ):
        return UTCDateTime(
            (float(text) - 25569.0)
            * 86400.0
        )

    return UTCDateTime(text)


def event_name(row: dict) -> str:
    """
    Read the event name from possible Excel columns.
    """

    value = (
        row.get("Event1")
        or row.get("Event")
        or row.get("event")
        or ""
    )

    return str(value).strip()


def event_station(row: dict) -> str:
    """
    Read station name.
    """

    value = (
        row.get("Station")
        or row.get("station")
        or ""
    )

    return str(value).strip()


def event_volume(row: dict) -> float | None:
    """
    Read event volume from possible Excel columns.
    """

    value = (
        row.get("Volume")
        or row.get("volume")
        or ""
    )

    text = str(value).strip()

    if not text:
        return None

    try:
        return float(text)

    except ValueError:
        return None


def get_sac_path(row: dict) -> str:
    """
    Read the SAC path from possible Excel columns.
    """

    possible_columns = [
        "SAC_File_Path",
        "SAC_file_path",
        "Sac_File_Path",
        "File_Path",
        "file_path",
    ]

    for column in possible_columns:

        value = str(
            row.get(column, "")
        ).strip()

        if value:
            return value

    raise ValueError(
        f"{event_name(row)}缺少SAC文件路径列。"
    )


def select_event_rows(
        rows: list[dict],
        selected_events: list[str]
) -> list[dict]:
    """
    Select exactly three events from the Excel table.
    """

    if len(selected_events) != 3:
        raise ValueError(
            "SELECTED_EVENTS必须正好包含3个事件。"
        )

    lookup = {
        event_name(row).lower(): row
        for row in rows
        if event_name(row)
    }

    selected_rows = []
    missing_events = []

    for event in selected_events:

        key = str(event).strip().lower()

        if key not in lookup:
            missing_events.append(event)
        else:
            selected_rows.append(
                lookup[key]
            )

    if missing_events:

        available_events = sorted(
            lookup.keys()
        )

        raise ValueError(
            "以下事件在Excel中未找到："
            f"{missing_events}\n\n"
            "Excel中的事件包括：\n"
            f"{available_events}"
        )

    return selected_rows


def determine_display_post_hours(
        selected_rows: list[dict]
) -> float:
    """
    Determine the common post-initiation display duration.

    The default display range extends to:
        longest event duration + POST_EVENT_BUFFER_H
    """

    durations = []

    for row in selected_rows:

        start = utc_time(
            row["Manual_Label_start"]
        )

        end = utc_time(
            row["Manual_Label_end"]
        )

        duration_h = (
            end - start
        ) / 3600.0

        if duration_h <= 0:

            raise ValueError(
                f"{event_name(row)}持续时间异常："
                f"{duration_h:.3f} h"
            )

        durations.append(duration_h)

    longest_duration = max(durations)

    display_post_h = (
        longest_duration
        + POST_EVENT_BUFFER_H
    )

    if MAX_POST_H is not None:

        if MAX_POST_H <= 0:
            raise ValueError(
                "MAX_POST_H必须大于0或设置为None。"
            )

        display_post_h = min(
            display_post_h,
            float(MAX_POST_H)
        )

    print(
        f"Longest event duration: "
        f"{longest_duration:.3f} h"
    )

    print(
        f"Common display range: "
        f"-{PRE_H:.1f} h to "
        f"+{display_post_h:.3f} h"
    )

    return display_post_h


# ==========================================================
# 11. DISCOVER WAVEFORM FILES
# ==========================================================

def contains_wildcard(path_text: str) -> bool:
    """
    Check whether a path contains glob wildcard characters.
    """

    return any(
        character in path_text
        for character in ["*", "?", "[", "]"]
    )


def discover_waveform_files(
        path_text: str
) -> list[Path]:
    """
    Find waveform files based on an Excel path.

    Supported cases
    ---------------
    1. Exact file path.
    2. Directory path.
    3. Wildcard expression.
    4. Exact file plus adjacent files in the same directory.
    """

    path_text = str(path_text).strip()

    if not path_text:
        raise ValueError(
            "SAC文件路径为空。"
        )

    files = []

    # ------------------------------------------------------
    # Wildcard path
    # ------------------------------------------------------

    if contains_wildcard(path_text):

        files = [
            Path(item)
            for item in glob.glob(
                path_text
            )
            if Path(item).is_file()
        ]

    else:

        path = Path(path_text)

        # --------------------------------------------------
        # Directory
        # --------------------------------------------------

        if path.is_dir():

            files = [
                item
                for item in path.iterdir()
                if (
                    item.is_file()
                    and (
                        not item.suffix
                        or item.suffix.lower()
                        in SUPPORTED_SUFFIXES
                    )
                )
            ]

        # --------------------------------------------------
        # Exact file
        # --------------------------------------------------

        elif path.is_file():

            if READ_ADJACENT_FILES:

                suffix = path.suffix.lower()

                files = [
                    item
                    for item in path.parent.iterdir()
                    if (
                        item.is_file()
                        and (
                            not suffix
                            or item.suffix.lower()
                            == suffix
                        )
                    )
                ]

            else:
                files = [path]

        else:

            matched = glob.glob(
                path_text
            )

            files = [
                Path(item)
                for item in matched
                if Path(item).is_file()
            ]

    files = sorted(
        set(files),
        key=lambda item: str(item)
    )

    if not files:
        raise FileNotFoundError(
            f"没有找到任何地震文件：{path_text}"
        )

    return files


# ==========================================================
# 12. READ AND MERGE EVENT DATA
# ==========================================================

def trace_matches_event(
        trace,
        station_hint: str,
        starttime: UTCDateTime,
        endtime: UTCDateTime
) -> bool:
    """
    Determine whether a trace overlaps the requested interval
    and matches the station/component requirements.
    """

    # No temporal overlap
    if trace.stats.endtime < starttime:
        return False

    if trace.stats.starttime > endtime:
        return False

    # Station filter
    trace_station = str(
        getattr(
            trace.stats,
            "station",
            ""
        )
    ).strip()

    if station_hint and trace_station:

        if (
            trace_station.upper()
            != station_hint.upper()
        ):
            return False

    # Z-component filter
    if USE_Z_COMPONENT:

        channel = str(
            getattr(
                trace.stats,
                "channel",
                ""
            )
        ).strip()

        if channel and not channel.upper().endswith("Z"):
            return False

    return True


def read_event_stream(
        row: dict,
        plot_start: UTCDateTime,
        plot_end: UTCDateTime
) -> Stream:
    """
    Read all waveform files overlapping the requested window.

    The function first checks file headers, then reads only
    files that overlap the requested time range.
    """

    path_text = get_sac_path(row)

    station_hint = event_station(row)

    waveform_files = discover_waveform_files(
        path_text
    )

    stream = Stream()

    print(
        f"  Candidate waveform files: "
        f"{len(waveform_files)}"
    )

    matched_file_count = 0

    for file_path in waveform_files:

        try:

            header_stream = read(
                str(file_path),
                headonly=True
            )

        except Exception:
            continue

        file_matches = any(
            trace_matches_event(
                trace,
                station_hint,
                plot_start,
                plot_end
            )
            for trace in header_stream
        )

        if not file_matches:
            continue

        try:

            file_stream = read(
                str(file_path),
                starttime=plot_start,
                endtime=plot_end
            )

        except Exception as error:

            print(
                f"  Skip unreadable file: "
                f"{file_path.name}; {error}"
            )

            continue

        matched_traces = Stream()

        for trace in file_stream:

            if trace_matches_event(
                trace,
                station_hint,
                plot_start,
                plot_end
            ):
                matched_traces += trace

        if len(matched_traces) > 0:

            stream += matched_traces
            matched_file_count += 1

    print(
        f"  Matched waveform files: "
        f"{matched_file_count}"
    )

    if len(stream) == 0:

        raise RuntimeError(
            f"{event_name(row)}没有读取到符合条件的地震数据。\n"
            f"原始路径：{path_text}\n"
            f"时间范围：{plot_start} 至 {plot_end}"
        )

    return stream


def read_event_trace(
        row: dict,
        display_post_h: float
):
    """
    Read, merge and preprocess one debris-flow event.
    """

    event = event_name(row)

    manual_start = utc_time(
        row["Manual_Label_start"]
    )

    manual_end = utc_time(
        row["Manual_Label_end"]
    )

    if manual_end <= manual_start:

        raise ValueError(
            f"{event}结束时间早于或等于开始时间：\n"
            f"{manual_start} -> {manual_end}"
        )

    plot_start = (
        manual_start
        - PRE_H * 3600.0
    )

    plot_end = (
        manual_start
        + display_post_h * 3600.0
    )

    stream = read_event_stream(
        row,
        plot_start,
        plot_end
    )

    stream.sort(
        keys=[
            "network",
            "station",
            "location",
            "channel",
            "starttime",
        ]
    )

    # ------------------------------------------------------
    # Prefer Z component
    # ------------------------------------------------------

    if USE_Z_COMPONENT:

        z_stream = Stream(
            traces=[
                trace
                for trace in stream
                if (
                    not str(
                        getattr(
                            trace.stats,
                            "channel",
                            ""
                        )
                    ).strip()
                    or str(
                        getattr(
                            trace.stats,
                            "channel",
                            ""
                        )
                    ).upper().endswith("Z")
                )
            ]
        )

        if len(z_stream) > 0:
            stream = z_stream

    # ------------------------------------------------------
    # Merge multiple SAC segments
    # ------------------------------------------------------

    stream.merge(
        method=1,
        fill_value=0.0
    )

    if len(stream) > 1:

        print(
            f"  Warning: {event} contains "
            f"{len(stream)} merged traces; "
            "the longest trace is selected."
        )

    trace = max(
        stream,
        key=lambda item: item.stats.npts
    ).copy()

    # ------------------------------------------------------
    # Trim to common range
    # ------------------------------------------------------

    trace.trim(
        starttime=plot_start,
        endtime=plot_end,
        pad=True,
        fill_value=0.0
    )

    if np.ma.isMaskedArray(
        trace.data
    ):

        trace.data = np.ma.filled(
            trace.data,
            fill_value=0.0
        )

    trace.data = np.asarray(
        trace.data,
        dtype=np.float64
    )

    # Replace invalid values
    invalid_mask = ~np.isfinite(
        trace.data
    )

    if np.any(invalid_mask):
        trace.data[
            invalid_mask
        ] = 0.0

    # ------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------

    trace.detrend("demean")
    trace.detrend("linear")

    trace.taper(
        max_percentage=0.005,
        type="hann"
    )

    trace.filter(
        "highpass",
        freq=HP_FREQ,
        corners=HP_CORNERS,
        zerophase=True
    )

    return (
        trace,
        manual_start,
        manual_end,
    )


# ==========================================================
# 13. WAVEFORM ENVELOPE
# ==========================================================

def waveform_minmax_envelope(
        data: np.ndarray,
        sampling_rate: float
):
    """
    Convert the original waveform into a min-max envelope.

    The function does not clip the waveform to ±1.
    Therefore, large-amplitude signals remain fully visible.
    """

    samples_per_bin = max(
        1,
        int(
            round(
                WAVEFORM_BIN_SEC
                * sampling_rate
            )
        )
    )

    number_of_bins = (
        len(data)
        // samples_per_bin
    )

    if number_of_bins < 1:

        raise ValueError(
            "波形长度不足，无法计算最小值—最大值包络。"
        )

    usable_data = data[
        :number_of_bins
        * samples_per_bin
    ]

    reshaped = usable_data.reshape(
        number_of_bins,
        samples_per_bin
    )

    waveform_min = np.min(
        reshaped,
        axis=1
    )

    waveform_max = np.max(
        reshaped,
        axis=1
    )

    # ------------------------------------------------------
    # Normalize waveform
    # ------------------------------------------------------

    absolute_data = np.abs(
        usable_data
    )

    valid_absolute_data = absolute_data[
        np.isfinite(absolute_data)
    ]

    if valid_absolute_data.size == 0:

        normalization_scale = 1.0

    else:

        normalization_scale = np.percentile(
            valid_absolute_data,
            WAVEFORM_NORMALIZE_PERCENTILE
        )

    if (
        not np.isfinite(
            normalization_scale
        )
        or normalization_scale <= 0
    ):

        normalization_scale = np.max(
            valid_absolute_data
        )

    if (
        not np.isfinite(
            normalization_scale
        )
        or normalization_scale <= 0
    ):
        normalization_scale = 1.0

    waveform_min = (
        waveform_min
        / normalization_scale
    )

    waveform_max = (
        waveform_max
        / normalization_scale
    )

    waveform_min = np.clip(
        waveform_min,
        -WAVEFORM_CLIP_LIMIT,
        WAVEFORM_CLIP_LIMIT,
    )

    waveform_max = np.clip(
        waveform_max,
        -WAVEFORM_CLIP_LIMIT,
        WAVEFORM_CLIP_LIMIT,
    )

    # 不使用np.clip，保留完整振幅
    time_hours = (
        -PRE_H
        + (
            np.arange(
                number_of_bins
            )
            + 0.5
        )
        * samples_per_bin
        / sampling_rate
        / 3600.0
    )

    return (
        time_hours,
        waveform_min,
        waveform_max,
    )


def determine_waveform_limit(
        waveform_min: np.ndarray,
        waveform_max: np.ndarray
) -> float:
    """
    Determine the symmetric waveform y-axis limit.
    """

    envelope_absolute = np.maximum(
        np.abs(waveform_min),
        np.abs(waveform_max)
    )

    valid_values = envelope_absolute[
        np.isfinite(
            envelope_absolute
        )
    ]

    if valid_values.size == 0:
        return COMMON_WAVEFORM_LIMIT

    if WAVEFORM_Y_MODE.lower() == "common":
        return float(COMMON_WAVEFORM_LIMIT)

    if WAVEFORM_Y_MODE.lower() == "full":

        waveform_limit = np.max(
            valid_values
        )

    elif WAVEFORM_Y_MODE.lower() == "percentile":

        waveform_limit = np.percentile(
            valid_values,
            WAVEFORM_DISPLAY_PERCENTILE
        )

    else:

        raise ValueError(
            "WAVEFORM_Y_MODE must be 'common', 'full', or 'percentile'."
        )

    if (
        not np.isfinite(
            waveform_limit
        )
        or waveform_limit <= 0
    ):
        waveform_limit = 1.0

    waveform_limit *= (
        WAVEFORM_Y_MARGIN
    )

    return float(
        waveform_limit
    )


# ==========================================================
# 14. SPECTROGRAM
# ==========================================================

def calculate_spectrogram(
        data: np.ndarray,
        sampling_rate: float
):
    """
    Calculate PSD spectrogram and average it into
    fixed-duration time bins.
    """

    number_of_samples = len(data)

    nperseg = min(
        SPEC_NPERSEG,
        number_of_samples
    )

    if nperseg < 16:

        raise ValueError(
            "数据长度不足，无法计算时频谱。"
        )

    noverlap = int(
        round(
            nperseg
            * SPEC_OVERLAP
        )
    )

    noverlap = min(
        noverlap,
        nperseg - 1
    )

    frequency, time_seconds, psd = spectrogram(
        data,
        fs=sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        mode="psd"
    )

    frequency_mask = (
        (frequency >= PSD_FREQ[0])
        & (frequency <= PSD_FREQ[1])
    )

    frequency = frequency[
        frequency_mask
    ]

    psd = psd[
        frequency_mask,
        :
    ]

    if psd.size == 0:

        raise ValueError(
            "指定的频率范围内没有有效频谱数据。"
        )

    # ------------------------------------------------------
    # Average spectra into fixed time bins
    # ------------------------------------------------------

    time_bin_index = np.floor(
        time_seconds
        / SPEC_TIME_BIN_SEC
    ).astype(int)

    number_of_time_bins = (
        int(
            time_bin_index.max()
        )
        + 1
    )

    binned_psd = np.full(
        (
            len(frequency),
            number_of_time_bins
        ),
        np.nan,
        dtype=np.float64
    )

    binned_time = np.full(
        number_of_time_bins,
        np.nan,
        dtype=np.float64
    )

    for index in range(
        number_of_time_bins
    ):

        column_mask = (
            time_bin_index == index
        )

        if not np.any(column_mask):
            continue

        # 在线性PSD空间平均
        binned_psd[:, index] = np.mean(
            psd[:, column_mask],
            axis=1
        )

        binned_time[index] = np.mean(
            time_seconds[column_mask]
        )

    valid_time_mask = np.isfinite(
        binned_time
    )

    binned_time = binned_time[
        valid_time_mask
    ]

    binned_psd = binned_psd[
        :,
        valid_time_mask
    ]

    psd_db = (
        10.0
        * np.log10(
            np.maximum(
                binned_psd,
                EPS
            )
        )
    )

    # Frequency-band mean for each 1-min time bin. Averaging is
    # performed before the logarithmic conversion so that the curve
    # represents a physically meaningful arithmetic mean PSD.
    mean_psd_linear = np.nanmean(
        binned_psd,
        axis=0
    )

    mean_psd_db = (
        10.0
        * np.log10(
            np.maximum(
                mean_psd_linear,
                EPS
            )
        )
    )

    relative_time_hours = (
        -PRE_H
        + binned_time
        / 3600.0
    )

    return (
        relative_time_hours,
        frequency,
        psd_db,
        mean_psd_db,
    )


# ==========================================================
# 15. PREPARE EVENTS
# ==========================================================

def prepare_events(
        selected_rows: list[dict],
        display_post_h: float
) -> list[dict]:
    """
    Read and process the three selected events.
    """

    results = []

    for index, row in enumerate(
        selected_rows,
        start=1
    ):

        event = event_name(row)

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"[{index}/3] Processing {event}"
        )

        trace, start, end = read_event_trace(
            row,
            display_post_h
        )

        sampling_rate = float(
            trace.stats.sampling_rate
        )

        data = np.asarray(
            trace.data,
            dtype=np.float64
        )

        (
            waveform_time,
            waveform_min,
            waveform_max,
        ) = waveform_minmax_envelope(
            data,
            sampling_rate
        )

        waveform_limit = (
            determine_waveform_limit(
                waveform_min,
                waveform_max
            )
        )

        (
            spectrogram_time,
            frequency,
            psd_db,
            mean_psd_db,
        ) = calculate_spectrogram(
            data,
            sampling_rate
        )

        duration_hours = (
            end - start
        ) / 3600.0

        print(
            f"  Sampling rate: "
            f"{sampling_rate:.2f} Hz"
        )

        print(
            f"  Event duration: "
            f"{duration_hours:.3f} h"
        )

        print(
            f"  Waveform y-limit: "
            f"±{waveform_limit:.3f}"
        )

        results.append(
            {
                "row": row,
                "event": event,
                "station": event_station(row),
                "start": start,
                "end": end,
                "duration_hours": duration_hours,
                "volume": event_volume(row),

                "waveform_time": waveform_time,
                "waveform_min": waveform_min,
                "waveform_max": waveform_max,
                "waveform_limit": waveform_limit,

                "spectrogram_time": spectrogram_time,
                "frequency": frequency,
                "psd_db": psd_db,
                "mean_psd_db": mean_psd_db,
            }
        )

    return results


# ==========================================================
# 16. COMMON PSD COLOR LIMITS
# ==========================================================

def determine_color_limits(
        results: list[dict]
) -> tuple[float, float]:
    """
    Determine one common PSD color range for all events.
    """

    if POWER_DB_RANGE is not None:

        vmin, vmax = POWER_DB_RANGE

        if vmax <= vmin:

            raise ValueError(
                "POWER_DB_RANGE必须满足vmax > vmin。"
            )

        return (
            float(vmin),
            float(vmax),
        )

    all_psd_values = np.concatenate(
        [
            item["psd_db"].ravel()
            for item in results
        ]
    )

    valid_psd_values = all_psd_values[
        np.isfinite(
            all_psd_values
        )
    ]

    if valid_psd_values.size == 0:

        raise ValueError(
            "没有有效PSD值，无法确定公共色标。"
        )

    vmin, vmax = np.percentile(
        valid_psd_values,
        COLOR_PERCENTILES
    )

    if not np.isfinite(vmin):
        vmin = np.min(
            valid_psd_values
        )

    if not np.isfinite(vmax):
        vmax = np.max(
            valid_psd_values
        )

    if vmax <= vmin:
        vmax = vmin + 1.0

    # Clean, reproducible colorbar limits for publication display.
    vmin = 5.0 * np.floor(vmin / 5.0)
    vmax = 5.0 * np.ceil(vmax / 5.0)

    print(
        "\nCommon PSD color range: "
        f"{vmin:.2f} to {vmax:.2f} dB"
    )

    return (
        float(vmin),
        float(vmax),
    )


def determine_mean_psd_limits(
        results: list[dict]
) -> tuple[float, float]:
    """Determine a common robust y-range for all mean-PSD curves."""

    values = np.concatenate(
        [
            item["mean_psd_db"].ravel()
            for item in results
        ]
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        raise ValueError(
            "No finite band-mean PSD values are available."
        )

    lower, upper = np.percentile(
        values,
        MEAN_PSD_LIMIT_PERCENTILES
    )

    if not np.isfinite(lower):
        lower = np.min(values)

    if not np.isfinite(upper):
        upper = np.max(values)

    if upper <= lower:
        upper = lower + 1.0

    padding = 0.05 * (upper - lower)
    lower = 5.0 * np.floor((lower - padding) / 5.0)
    upper = 5.0 * np.ceil((upper + padding) / 5.0)

    print(
        "Common 1-min mean-PSD range: "
        f"{lower:.2f} to {upper:.2f} dB"
    )

    return float(lower), float(upper)


# ==========================================================
# 17. LABELS
# ==========================================================

def build_event_title(
        item: dict
) -> str:
    """
    Build the information label for one event.
    """

    event_text = str(item["event"]).strip()
    event_match = re.fullmatch(
        r"event\s*(\d+)",
        event_text,
        flags=re.IGNORECASE,
    )

    if event_match:
        event_text = f"Event {int(event_match.group(1))}"

    station = item["station"]

    start_text = item[
        "start"
    ].strftime(
        "%d %b %Y, %H:%M UTC"
    )

    title_parts = [
        station,
        start_text,
    ]

    volume = item["volume"]

    if volume is not None:

        title_parts.append(
            "Volume: "
            rf"${volume / 10 ** np.floor(np.log10(volume)):.2f}"
            rf"\times10^{{{int(np.floor(np.log10(volume)))}}}$ m$^3$"
            if volume > 0
            else "Estimated volume: 0 m$^3$"
        )

    return " | ".join(
        part
        for part in title_parts
        if part
    )


# ==========================================================
# 18. PLOT THREE EVENTS
# ==========================================================

def plot_three_events(
        results: list[dict],
        display_post_h: float
):
    """
    Plot the three event panels.

    Each event includes:
        upper panel: normalized waveform
        lower panel: PSD spectrogram
    """

    vmin, vmax = determine_color_limits(
        results
    )

    mean_psd_limits = (
        determine_mean_psd_limits(results)
        if SHOW_MEAN_PSD_CURVE
        else None
    )

    color_normalization = (
        mpl.colors.Normalize(
            vmin=vmin,
            vmax=vmax
        )
    )

    figure = plt.figure(
        figsize=FIG_SIZE
    )

    # One colorbar row + three event rows
    main_grid = figure.add_gridspec(
        nrows=4,
        ncols=1,
        height_ratios=[
            0.10,
            1.0,
            1.0,
            1.0,
        ],
        hspace=MAIN_GRID_HSPACE
    )

    # ------------------------------------------------------
    # Common horizontal colorbar
    # ------------------------------------------------------

    colorbar_axis = figure.add_subplot(
        main_grid[0]
    )

    scalar_mappable = (
        mpl.cm.ScalarMappable(
            norm=color_normalization,
            cmap=PSD_CMAP
        )
    )

    scalar_mappable.set_array([])

    colorbar = figure.colorbar(
        scalar_mappable,
        cax=colorbar_axis,
        orientation="horizontal"
    )

    colorbar.set_label(
        "Power spectral density (dB)",
        fontsize=FONT_SIZE_COLORBAR_LABEL,
        labelpad=COLORBAR_LABEL_PAD
    )

    colorbar.ax.xaxis.set_ticks_position(
        "top"
    )

    colorbar.ax.xaxis.set_label_position(
        "top"
    )

    colorbar.ax.tick_params(
        labelsize=FONT_SIZE_COLORBAR_TICK,
        pad=TICK_LABEL_PAD,
        length=TICK_LENGTH,
        width=TICK_WIDTH,
    )

    # Main-panel label. Event rows are intentionally not lettered.
    # figure.text(
    #     0.012,
    #     0.988,
    #     "b",
    #     ha="left",
    #     va="top",
    #     fontsize=FONT_SIZE_PANEL,
    #     fontweight="bold",
    # )

    # ------------------------------------------------------
    # Three events
    # ------------------------------------------------------

    for index, item in enumerate(
        results
    ):

        event_grid = main_grid[
            index + 1
        ].subgridspec(
            nrows=2,
            ncols=1,
            height_ratios=EVENT_HEIGHT_RATIOS,
            hspace=EVENT_GRID_HSPACE
        )

        waveform_axis = figure.add_subplot(
            event_grid[0]
        )

        spectrogram_axis = figure.add_subplot(
            event_grid[1],
            sharex=waveform_axis
        )

        duration_hours = item[
            "duration_hours"
        ]

        shaded_end = min(
            duration_hours,
            display_post_h
        )

        # --------------------------------------------------
        # Manual debris-flow interval
        # --------------------------------------------------

        if shaded_end > 0:

            waveform_axis.axvspan(
                0,
                shaded_end,
                facecolor="#D55E00",
                alpha=0.13,
                edgecolor="none",
                zorder=0
            )

            # Do not shade the spectrogram: an overlay would modify the
            # apparent PSD colors. Initiation and end times are shown by lines.

        # --------------------------------------------------
        # Waveform
        # --------------------------------------------------

        waveform_axis.fill_between(
            item["waveform_time"],
            item["waveform_min"],
            item["waveform_max"],
            facecolor="black",
            edgecolor="none",
            linewidth=0,
            alpha=0.92,
            zorder=2
        )

        waveform_axis.axhline(
            0,
            color="0.65",
            linewidth=WAVEFORM_BASELINE_LINE_WIDTH,
            zorder=1
        )

        waveform_axis.axvline(
            0,
            color="#D55E00",
            linestyle="--",
            linewidth=INITIATION_WAVEFORM_LINE_WIDTH,
            zorder=5
        )

        waveform_limit = item[
            "waveform_limit"
        ]

        waveform_axis.set_ylim(
            -waveform_limit,
            waveform_limit
        )

        waveform_axis.set_xlim(
            -PRE_H,
            display_post_h
        )

        waveform_axis.set_ylabel(
            "Normalized\namplitude",
            fontsize=FONT_SIZE_LABEL
        )

        waveform_axis.set_yticks(
            [-1, 0, 1]
        )

        waveform_axis.tick_params(
            axis="x",
            which="both",
            bottom=False,
            labelbottom=False
        )

        waveform_axis.tick_params(
            axis="y",
            labelsize=FONT_SIZE_TICK,
            length=TICK_LENGTH,
            width=TICK_WIDTH,
            pad=TICK_LABEL_PAD,
        )

        waveform_axis.spines[
            "top"
        ].set_visible(False)

        waveform_axis.spines[
            "right"
        ].set_visible(False)

        waveform_axis.text(
            0.008,
            1.16,
            build_event_title(item),
            transform=waveform_axis.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_SIZE_EVENT_TITLE,
            fontweight=FONT_WEIGHT_EVENT_TITLE,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
                "pad": 1.0,
            },
            zorder=10
        )

        # --------------------------------------------------
        # Spectrogram
        # --------------------------------------------------

        spectrogram_axis.pcolormesh(
            item["spectrogram_time"],
            item["frequency"],
            item["psd_db"],
            shading="auto",
            cmap=PSD_CMAP,
            norm=color_normalization,
            rasterized=True
        )

        # --------------------------------------------------
        # One-minute frequency-band-mean PSD
        # --------------------------------------------------

        if SHOW_MEAN_PSD_CURVE:

            mean_psd_axis = spectrogram_axis.twinx()

            mean_psd_line, = mean_psd_axis.plot(
                item["spectrogram_time"],
                item["mean_psd_db"],
                color=MEAN_PSD_CURVE_COLOR,
                linewidth=MEAN_PSD_CURVE_LINE_WIDTH,
                alpha=MEAN_PSD_CURVE_ALPHA,
                zorder=9,
            )

            mean_psd_line.set_path_effects(
                [
                    path_effects.Stroke(
                        linewidth=MEAN_PSD_HALO_LINE_WIDTH,
                        foreground="black",
                        alpha=MEAN_PSD_HALO_ALPHA,
                    ),
                    path_effects.Normal(),
                ]
            )

            mean_psd_axis.set_ylim(
                mean_psd_limits
            )

            mean_psd_axis.locator_params(
                axis="y",
                nbins=3
            )

            mean_psd_axis.tick_params(
                axis="y",
                colors=MEAN_PSD_CURVE_COLOR,
                labelsize=FONT_SIZE_MEAN_PSD_TICK,
                length=TICK_LENGTH,
                width=TICK_WIDTH,
                pad=TICK_LABEL_PAD,
            )

            mean_psd_axis.spines["right"].set_color(
                MEAN_PSD_CURVE_COLOR
            )
            mean_psd_axis.spines["right"].set_linewidth(
                AXES_LINE_WIDTH
            )
            mean_psd_axis.spines["top"].set_visible(False)
            mean_psd_axis.spines["bottom"].set_visible(False)
            mean_psd_axis.grid(False)

            if index == 1:
                mean_psd_axis.set_ylabel(
                    "1-min mean PSD (dB)",
                    color=MEAN_PSD_CURVE_COLOR,
                    fontsize=FONT_SIZE_MEAN_PSD_LABEL,
                )

        # Initiation time
        spectrogram_axis.axvline(
            0,
            color="white",
            linestyle="--",
            linewidth=INITIATION_SPECTROGRAM_HALO_LINE_WIDTH,
            zorder=6
        )

        spectrogram_axis.axvline(
            0,
            color="#D55E00",
            linestyle="--",
            linewidth=INITIATION_SPECTROGRAM_LINE_WIDTH,
            zorder=7
        )

        # Manual end time
        if duration_hours <= display_post_h:

            spectrogram_axis.axvline(
                duration_hours,
                color="white",
                linestyle=":",
                linewidth=EVENT_END_LINE_WIDTH,
                alpha=0.9,
                zorder=6
            )

        spectrogram_axis.set_xlim(
            -PRE_H,
            display_post_h
        )

        spectrogram_axis.set_ylim(
            PSD_FREQ
        )

        spectrogram_axis.set_ylabel(
            "Frequency (Hz)",
            fontsize=FONT_SIZE_LABEL
        )

        spectrogram_axis.set_yticks(
            [0, 25, 50]
        )

        spectrogram_axis.tick_params(
            axis="both",
            labelsize=FONT_SIZE_TICK,
            length=TICK_LENGTH,
            width=TICK_WIDTH,
            pad=TICK_LABEL_PAD,
        )

        spectrogram_axis.set_xticks(
            [-24, -18, -12, -6, 0, display_post_h]
        )

        for spine in (
            spectrogram_axis.spines.values()
        ):
            spine.set_visible(True)
            spine.set_linewidth(AXES_LINE_WIDTH)

        if index == len(results) - 1:

            spectrogram_axis.set_xlabel(
                "Time relative to catalogued "
                "debris-flow initiation (h)",
                fontsize=FONT_SIZE_LABEL
            )

        else:

            spectrogram_axis.tick_params(
                axis="x",
                labelbottom=False
            )

    figure.subplots_adjust(
        left=FIGURE_LEFT,
        right=FIGURE_RIGHT,
        bottom=FIGURE_BOTTOM,
        top=FIGURE_TOP
    )

    return figure


# ==========================================================
# 19. MAIN
# ==========================================================

def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "=" * 60
    )

    print(
        "Reading Excel:"
    )

    print(
        EXCEL
    )

    rows = read_xlsx(
        EXCEL
    )

    selected_rows = select_event_rows(
        rows,
        SELECTED_EVENTS
    )

    display_post_h = (
        determine_display_post_hours(
            selected_rows
        )
    )

    results = prepare_events(
        selected_rows,
        display_post_h
    )

    figure = plot_three_events(
        results,
        display_post_h
    )

    event_text = "_".join(
        SELECTED_EVENTS
    )

    png_file = (
        OUT
        / (
            "Fig1b_three_events_revised_"
            f"{event_text}.png"
        )
    )

    pdf_file = (
        OUT
        / (
            "Fig1b_three_events_revised_"
            f"{event_text}.pdf"
        )
    )
    svg_file = (
            OUT
            / (
                "Fig1b_three_events_revised_"
                f"{event_text}.svg"
            )
    )
    tiff_file = (
        OUT
        / (
            "Fig1b_three_events_revised_"
            f"{event_text}.tiff"
        )
    )
    figure.savefig(
        png_file,
        dpi=SAVE_DPI,
        bbox_inches="tight"
    )
    print(png_file)
    plt.show()
    plt.close(
        figure
    )

if __name__ == "__main__":
    main()
