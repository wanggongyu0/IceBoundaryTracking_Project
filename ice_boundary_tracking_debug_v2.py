# -*- coding: utf-8 -*-
"""
ice_boundary_tracking_h5py_fix.py

说明：
    这是加入 MATLAB v7.3 MAT 文件读取能力后的版本。
    如果你的 thermal_data.mat 是 MATLAB 用 save(...,'-v7.3') 保存的，
    scipy.io.loadmat 不能直接读取，需要 h5py 读取。

依赖：
    pip install numpy opencv-python pandas matplotlib scipy h5py
"""

import os
import glob
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


@dataclass
class Config:
    data_path: str = r"./data/exp_001/thermal_data.mat"
    out_dir: str = r"./result/exp_001"

    raw_fps: float = 100.0
    sample_fps: float = 1.0
    heat_on_time_in_file_s: float = 0.0

    roi: Tuple[int, int, int, int] = (180, 120, 280, 260)
    ref_roi: Optional[Tuple[int, int, int, int]] = (20, 20, 20, 20)

    # ROI 使用方式：
    # True：第一次运行时用鼠标框选 ROI，并保存到 roi_config.json
    # False：直接使用上面手动写的 roi/ref_roi
    select_roi_interactively: bool = True

    # True：即使已经有 roi_config.json，也重新框选
    # False：优先读取已有 roi_config.json
    force_reselect_roi: bool = False

    # None：默认保存到 thermal_data.mat 同文件夹下的 roi_config.json
    # 也可以手动指定，例如 r"./data/exp_001/roi_config.json"
    roi_config_path: Optional[str] = None
    initial_ice_diameter_mm: float = 60.0
    completion_area_ratio: float = 0.03
    continuous_seconds: float = 3.0
    min_blob_ratio: float = 0.0005
    rate_lag_s: float = 5.0
    # 调试阶段建议先关闭升温速率筛选，避免第5秒附近误删冰区
    use_heating_rate: bool = False

    # 是否启用时间连续性约束；调试阶段建议先关闭，确认单帧分割正确后再开启
    use_temporal_constraint: bool = False

    # 冰区在归一化图像中是低值还是高值：'low'、'high'
    # 先分别跑 low/high，对比 overlays 哪个边界更贴近真实冰边界
    ice_polarity: str = 'low'

    morph_kernel_size: int = 3
    allowed_dilate_iter: int = 2
    area_smooth_window_s: float = 3.0
    save_overlay_every_s: float = 1.0
    visible_complete_interval_s: Optional[Tuple[float, float]] = (140.0, 150.0)
def make_run_dir(cfg: Config) -> str:
    """
    根据当前参数自动生成本次运行的结果文件夹。
    不改变算法，只改变保存路径。
    """
    base_dir = Path(cfg.out_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    x, y, w, h = cfg.roi

    run_name = (
        f"{timestamp}"
        f"_pol-{cfg.ice_polarity}"
        f"_fps-{cfg.raw_fps:g}"
        f"_sfps-{cfg.sample_fps:g}"
        f"_roi-{x}-{y}-{w}-{h}"
        f"_rate-{int(cfg.use_heating_rate)}"
        f"_temp-{int(cfg.use_temporal_constraint)}"
        f"_k-{cfg.morph_kernel_size}"
    )

    run_dir = base_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    return str(run_dir)

def save_config_used(cfg: Config):
    """
    保存本次运行实际使用的参数，方便后续对比。
    """
    config_path = Path(cfg.out_dir) / "config_used.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

def ensure_frames_nhw(arr: np.ndarray) -> np.ndarray:
    """
    将红外数据统一为 [N, H, W]。
    允许输入 [N,H,W]、[H,W,N]、[W,H,N] 等常见格式。
    """
    arr = np.asarray(arr).squeeze()

    if arr.ndim != 3:
        raise ValueError(f"温度数据必须是三维数组，但当前维度为 {arr.shape}")

    # 如果已经是 [N,512,640] 或 [N,640,512]
    if arr.shape[1:] == (512, 640):
        return arr.astype(np.float32)
    if arr.shape[1:] == (640, 512):
        return np.transpose(arr, (0, 2, 1)).astype(np.float32)

    # 如果是 [512,640,N]
    if arr.shape[0:2] == (512, 640):
        return np.moveaxis(arr, -1, 0).astype(np.float32)

    # 如果是 [640,512,N]
    if arr.shape[0:2] == (640, 512):
        arr = np.transpose(arr, (1, 0, 2))
        return np.moveaxis(arr, -1, 0).astype(np.float32)

    # h5py 读取 MATLAB v7.3 时，有时可能变成 [640,512,N] 或 [N,640,512]
    # 如果最后一维像帧数，前两维像图像尺寸，则转成 [N,H,W]
    if arr.shape[0] in (512, 640) and arr.shape[1] in (512, 640) and arr.shape[2] > 10:
        if arr.shape[0] == 512 and arr.shape[1] == 640:
            return np.moveaxis(arr, -1, 0).astype(np.float32)
        if arr.shape[0] == 640 and arr.shape[1] == 512:
            arr = np.transpose(arr, (1, 0, 2))
            return np.moveaxis(arr, -1, 0).astype(np.float32)

    raise ValueError(f"无法自动判断 frames 维度方向，当前 shape={arr.shape}。请截图发我。")


def load_mat_v73_with_h5py(path: Path) -> np.ndarray:
    """
    读取 MATLAB -v7.3 保存的 HDF5 MAT 文件。
    优先寻找 frames 变量。
    """
    with h5py.File(path, "r") as f:
        print("[INFO] 检测到 MATLAB v7.3 MAT 文件，使用 h5py 读取。")
        print("[INFO] MAT 文件顶层变量：", list(f.keys()))

        if "frames" in f:
            dset = f["frames"]
            arr = dset[()]
            print(f"[INFO] 读取变量 frames, 原始 shape={arr.shape}, dtype={arr.dtype}")
            return ensure_frames_nhw(arr)

        # 如果没有 frames，则自动寻找第一个三维数值数据集
        candidates = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and len(obj.shape) == 3:
                candidates.append(name)

        f.visititems(visitor)

        if not candidates:
            raise ValueError("v7.3 MAT 文件中没有找到三维矩阵变量，请确认 MATLAB 保存了 frames。")

        name = candidates[0]
        arr = f[name][()]
        print(f"[INFO] 未找到 frames，改读第一个三维变量 {name}, 原始 shape={arr.shape}, dtype={arr.dtype}")
        return ensure_frames_nhw(arr)


def load_thermal_data(data_path: str) -> np.ndarray:
    """
    支持读取：
    1. .npy
    2. .npz
    3. .mat，包括普通MAT和MATLAB v7.3 MAT
    4. 文件夹内 csv/txt 帧序列
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"找不到数据路径：{data_path}")

    if path.is_file():
        suffix = path.suffix.lower()

        if suffix == ".npy":
            return ensure_frames_nhw(np.load(path))

        if suffix == ".npz":
            data = np.load(path)
            if "frames" in data:
                return ensure_frames_nhw(data["frames"])
            for key in data.files:
                if data[key].ndim == 3:
                    return ensure_frames_nhw(data[key])
            raise ValueError(".npz 文件中没有找到三维温度矩阵")

        if suffix == ".mat":
            try:
                data = loadmat(path)
                candidates = []
                for key, value in data.items():
                    if key.startswith("__"):
                        continue
                    if isinstance(value, np.ndarray) and value.ndim == 3:
                        candidates.append((key, value))

                if not candidates:
                    raise ValueError(".mat 文件中没有找到三维温度矩阵")

                key, arr = candidates[0]
                print(f"[INFO] 用 scipy.loadmat 读取变量：{key}, shape={arr.shape}")
                return ensure_frames_nhw(arr)

            except NotImplementedError:
                return load_mat_v73_with_h5py(path)
            except ValueError as e:
                # scipy 对 v7.3 经常抛 ValueError: Unknown mat file type
                msg = str(e).lower()
                if "unknown mat file type" in msg or "please use hdf reader" in msg:
                    return load_mat_v73_with_h5py(path)
                raise
            except Exception as e:
                # 对 v7.3 的常见兜底
                msg = str(e).lower()
                if "please use hdf reader" in msg or "hdf5" in msg or "unknown mat file type" in msg:
                    return load_mat_v73_with_h5py(path)
                raise

        raise ValueError(
            f"暂不支持直接读取 {suffix} 文件。请先从 ThermPulse/MATLAB 导出为 .mat/.npy/csv。"
        )

    if path.is_dir():
        files = sorted(glob.glob(str(path / "*.csv")) + glob.glob(str(path / "*.txt")))
        if not files:
            raise ValueError("文件夹中没有找到 csv 或 txt 温度矩阵文件")

        frames = []
        for f in files:
            try:
                mat = np.loadtxt(f, delimiter=",")
            except Exception:
                mat = np.loadtxt(f)
            frames.append(mat)

        return ensure_frames_nhw(np.stack(frames, axis=0))

    raise ValueError(f"无法识别的数据路径：{data_path}")


# ===== 以下为与你原主程序相同的处理函数 =====

def crop_roi(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return frame[y:y+h, x:x+w]


def clamp_roi(roi: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x, y, w, h = roi
    H, W = frame_shape
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h
roi: Tuple[int, int, int, int] = (180, 120, 280, 260)
ref_roi: Optional[Tuple[int, int, int, int]] = (20, 20, 20, 20)

# ROI 使用方式：
# True：第一次运行时用鼠标框选 ROI，并保存到 roi_config.json
# False：直接使用上面手动写的 roi/ref_roi
select_roi_interactively: bool = True

# True：即使已经有 roi_config.json，也重新框选
# False：优先读取已有 roi_config.json
force_reselect_roi: bool = False

# None：默认保存到 thermal_data.mat 同文件夹下的 roi_config.json
# 也可以手动指定，例如 r"./data/exp_001/roi_config.json"
roi_config_path: Optional[str] = None

def get_roi_config_path(cfg: Config) -> Path:
    """
    获取当前实验的 ROI 配置文件路径。
    默认放在 thermal_data.mat 所在文件夹下。
    """
    if cfg.roi_config_path is not None:
        return Path(cfg.roi_config_path)

    data_path = Path(cfg.data_path)
    if data_path.is_file():
        return data_path.parent / "roi_config.json"
    else:
        return data_path / "roi_config.json"


def thermal_frame_to_bgr(frame: np.ndarray) -> np.ndarray:
    """
    将热像矩阵转换为可显示的伪彩图，只用于鼠标框选 ROI。
    """
    p1, p99 = np.nanpercentile(frame, [1, 99])

    if abs(p99 - p1) < 1e-6:
        u8 = np.zeros_like(frame, dtype=np.uint8)
    else:
        u8 = (np.clip((frame - p1) / (p99 - p1), 0, 1) * 255).astype(np.uint8)

    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def select_rect_from_frame(frame: np.ndarray, window_name: str, allow_cancel: bool = False):
    """
    用 OpenCV 鼠标框选矩形 ROI。
    返回格式为 (x, y, w, h)。
    """
    img = thermal_frame_to_bgr(frame)

    print(f"[INFO] 请在弹出的窗口中框选：{window_name}")
    print("[INFO] 鼠标拖拽矩形，按 Enter 或 Space 确认，按 Esc 取消。")

    rect = cv2.selectROI(window_name, img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)

    x, y, w, h = [int(v) for v in rect]

    if w <= 0 or h <= 0:
        if allow_cancel:
            print(f"[INFO] 未选择 {window_name}，返回 None。")
            return None
        raise RuntimeError(f"没有选择有效的 {window_name}。")

    return (x, y, w, h)


def load_roi_config(path: Path):
    """
    从 roi_config.json 读取 ROI。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    roi = tuple(data["roi"])

    ref_roi = data.get("ref_roi", None)
    if ref_roi is not None:
        ref_roi = tuple(ref_roi)

    return roi, ref_roi


def save_roi_config(path: Path, cfg: Config):
    """
    保存当前实验使用的 ROI。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "roi": list(cfg.roi),
        "ref_roi": list(cfg.ref_roi) if cfg.ref_roi is not None else None,
        "note": "roi/ref_roi format is (x, y, w, h)"
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] ROI 配置已保存：{path}")


def prepare_rois(cfg: Config, first_frame: np.ndarray, frame_shape: Tuple[int, int]):
    """
    准备本次实验使用的 ROI。
    优先级：
    1. 如果已有 roi_config.json 且不强制重选，则读取；
    2. 否则弹窗手动框选；
    3. 如果关闭交互式选择，则使用 Config 里写死的 roi/ref_roi。
    """
    roi_config_path = get_roi_config_path(cfg)

    if roi_config_path.exists() and not cfg.force_reselect_roi:
        cfg.roi, cfg.ref_roi = load_roi_config(roi_config_path)
        print(f"[INFO] 已读取 ROI 配置：{roi_config_path}")

    elif cfg.select_roi_interactively:
        cfg.roi = select_rect_from_frame(
            first_frame,
            "Select ICE ROI",
            allow_cancel=False
        )

        cfg.ref_roi = select_rect_from_frame(
            first_frame,
            "Select REF ROI - press Esc to skip",
            allow_cancel=True
        )

        save_roi_config(roi_config_path, cfg)

    else:
        print("[INFO] 未启用交互式 ROI 选择，使用 Config 中的 roi/ref_roi。")

    cfg.roi = clamp_roi(cfg.roi, frame_shape)

    if cfg.ref_roi is not None:
        cfg.ref_roi = clamp_roi(cfg.ref_roi, frame_shape)

    print(f"[INFO] 本次使用 ROI: {cfg.roi}")
    print(f"[INFO] 本次使用 Ref ROI: {cfg.ref_roi}")
def save_roi_preview(frame: np.ndarray, cfg, save_path: str):
    p1, p99 = np.nanpercentile(frame, [1, 99])
    if abs(p99 - p1) < 1e-6:
        u8 = np.zeros_like(frame, dtype=np.uint8)
    else:
        u8 = (np.clip((frame - p1) / (p99 - p1), 0, 1) * 255).astype(np.uint8)

    color = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)

    x, y, w, h = cfg.roi
    cv2.rectangle(color, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(color, "ROI", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if cfg.ref_roi is not None:
        rx, ry, rw, rh = cfg.ref_roi
        cv2.rectangle(color, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), 2)
        cv2.putText(color, "REF", (rx, max(20, ry - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(save_path, color)


def get_ref_temperature(frame: np.ndarray, ref_roi: Optional[Tuple[int, int, int, int]]) -> float:
    if ref_roi is None:
        return 0.0
    x, y, w, h = ref_roi
    return float(np.nanmedian(frame[y:y+h, x:x+w]))


def robust_normalize_to_uint8(img: np.ndarray):
    p5, p95 = np.nanpercentile(img, [5, 95])
    if abs(p95 - p5) < 1e-6:
        return np.zeros_like(img, dtype=np.uint8), p5, p95
    norm = (img - p5) / (p95 - p5)
    return (np.clip(norm, 0, 1) * 255).astype(np.uint8), float(p5), float(p95)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8) * 255
    h, w = mask_u8.shape
    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    return (mask_u8 | cv2.bitwise_not(flood)) > 0


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            out[labels == label] = 1
    return out.astype(bool)


def morphology_clean(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    return fill_holes(mask_u8 > 0)


def segment_ice_frame(frame, cfg, prev_mask=None, lag_frame=None, dt_lag=None):
    roi_frame = crop_roi(frame, cfg.roi)
    ref_temp = get_ref_temperature(frame, cfg.ref_roi)
    roi_rel = roi_frame - ref_temp
    roi_u8, p5, p95 = robust_normalize_to_uint8(roi_rel)

    # Otsu 自适应阈值。
    # ice_polarity='low'：低温/低灰度区域视作冰区；
    # ice_polarity='high'：高温/高灰度区域视作冰区。
    if cfg.ice_polarity.lower() == "low":
        otsu_thr, mask_u8 = cv2.threshold(
            roi_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    elif cfg.ice_polarity.lower() == "high":
        otsu_thr, mask_u8 = cv2.threshold(
            roi_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        raise ValueError("cfg.ice_polarity 只能设置为 'low' 或 'high'。")

    mask = mask_u8 > 0

    if cfg.use_heating_rate and lag_frame is not None and dt_lag is not None and dt_lag > 0:
        lag_roi = crop_roi(lag_frame, cfg.roi)
        heating_rate = (roi_frame - lag_roi) / dt_lag
        if np.any(mask):
            rate_threshold = np.nanpercentile(heating_rate, 80)
            mask = mask & (heating_rate < rate_threshold)

    if cfg.use_temporal_constraint and prev_mask is not None:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_kernel_size, cfg.morph_kernel_size)
        )
        allowed = cv2.dilate(
            prev_mask.astype(np.uint8), kernel, iterations=cfg.allowed_dilate_iter
        ) > 0
        mask = mask & allowed

    mask = morphology_clean(mask, cfg.morph_kernel_size)

    info = {
        "ref_temp": ref_temp,
        "p5": p5,
        "p95": p95,
        "otsu_thr": float(otsu_thr),
        "roi_u8": roi_u8,
    }
    return mask, info


def save_overlay_image(roi_u8, mask, save_path, time_s, alpha=None):
    color = cv2.applyColorMap(roi_u8, cv2.COLORMAP_INFERNO)
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color, contours, -1, (0, 255, 255), 2)

    text = f"t = {time_s:.1f} s"
    if alpha is not None:
        text += f", alpha = {alpha:.3f}"
    cv2.putText(color, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(save_path, color)


def find_completion_time(times, alpha, eps, continuous_s):
    if len(times) < 2:
        return None
    dt = np.median(np.diff(times))
    window = max(1, int(round(continuous_s / dt)))
    for i in range(0, len(alpha) - window + 1):
        if np.all(alpha[i:i+window] <= eps):
            return float(times[i])
    return None


def run_tracking(cfg: Config):
    cfg.out_dir = make_run_dir(cfg)

    overlay_dir = os.path.join(cfg.out_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    print(f"[INFO] 本次运行结果目录: {cfg.out_dir}")

    print("[INFO] 正在读取红外数据...")
    print(f"[INFO] 调试参数: use_heating_rate={cfg.use_heating_rate}, "
          f"use_temporal_constraint={cfg.use_temporal_constraint}, "
          f"ice_polarity={cfg.ice_polarity}")
    frames = load_thermal_data(cfg.data_path)
    total_frames, H, W = frames.shape
    print(f"[INFO] 数据尺寸: N={total_frames}, H={H}, W={W}")

    prepare_rois(cfg, frames[0], (H, W))

    roi_preview_path = os.path.join(cfg.out_dir, "roi_preview.png")
    save_roi_preview(frames[0], cfg, roi_preview_path)
    print(f"[INFO] ROI preview: {roi_preview_path}")
    print(f"[INFO] ROI after prepare: {cfg.roi}, Ref ROI: {cfg.ref_roi}")

    save_config_used(cfg)
    step = max(1, int(round(cfg.raw_fps / cfg.sample_fps)))
    frames_s = frames[::step]
    times = np.arange(len(frames_s)) * step / cfg.raw_fps - cfg.heat_on_time_in_file_s

    start_idx = int(np.searchsorted(times, 0.0))
    frames_s = frames_s[start_idx:]
    times = times[start_idx:]

    print(f"[INFO] 抽帧后用于处理的帧数: {len(frames_s)}")
    print(f"[INFO] 时间范围: {times[0]:.2f} s ~ {times[-1]:.2f} s")

    prev_mask = None
    records = []
    rate_lag_frames = max(1, int(round(cfg.rate_lag_s * cfg.sample_fps)))
    save_every_frames = max(1, int(round(cfg.save_overlay_every_s * cfg.sample_fps)))

    A0_px = None
    pixel_area_mm2 = None

    for i, (frame, t) in enumerate(zip(frames_s, times)):
        if i - rate_lag_frames >= 0:
            lag_frame = frames_s[i - rate_lag_frames]
            dt_lag = times[i] - times[i - rate_lag_frames]
        else:
            lag_frame = None
            dt_lag = None

        mask, info = segment_ice_frame(frame, cfg, prev_mask, lag_frame, dt_lag)

        if A0_px is None:
            A0_px = int(np.sum(mask))
            if A0_px <= 0:
                raise RuntimeError("初始帧没有识别到冰区，请检查ROI或阈值处理结果。")
            initial_area_mm2 = np.pi * (cfg.initial_ice_diameter_mm / 2.0) ** 2
            pixel_area_mm2 = initial_area_mm2 / A0_px
            min_blob_area = max(5, int(A0_px * cfg.min_blob_ratio))
            print(f"[INFO] 初始冰区像素面积 A0 = {A0_px} px")
            print(f"[INFO] pixel_area = {pixel_area_mm2:.4f} mm^2/px")
            print(f"[INFO] 最小连通域面积阈值 = {min_blob_area} px")
        else:
            min_blob_area = max(5, int(A0_px * cfg.min_blob_ratio))

        mask = remove_small_components(mask, min_blob_area)

        area_px = int(np.sum(mask))
        area_mm2 = area_px * pixel_area_mm2
        alpha_raw = area_px / A0_px

        records.append({
            "time_s": float(t),
            "area_px": area_px,
            "area_mm2": area_mm2,
            "alpha_raw": alpha_raw,
            "ref_temp": info["ref_temp"],
            "p5": info["p5"],
            "p95": info["p95"],
            "otsu_thr": info["otsu_thr"],
        })

        prev_mask = mask.copy()

        if i % save_every_frames == 0:
            save_path = os.path.join(overlay_dir, f"overlay_t_{t:07.2f}s.png")
            save_overlay_image(info["roi_u8"], mask, save_path, t, alpha_raw)

    df = pd.DataFrame(records)

    smooth_window = max(1, int(round(cfg.area_smooth_window_s * cfg.sample_fps)))
    if smooth_window % 2 == 0:
        smooth_window += 1

    df["alpha_smooth"] = (
        df["alpha_raw"].rolling(window=smooth_window, center=True, min_periods=1).median()
    )
    df["alpha_physical"] = np.minimum.accumulate(df["alpha_smooth"].values)

    t_complete = find_completion_time(
        df["time_s"].values,
        df["alpha_physical"].values,
        cfg.completion_area_ratio,
        cfg.continuous_seconds,
    )

    visible_result = "未设置可见光验证区间"
    if cfg.visible_complete_interval_s is not None and t_complete is not None:
        t1, t2 = cfg.visible_complete_interval_s
        if t1 <= t_complete <= t2:
            visible_result = f"通过：算法融冰时间 {t_complete:.2f}s 落入可见光区间 [{t1}, {t2}]s"
        else:
            center = 0.5 * (t1 + t2)
            err = abs(t_complete - center)
            visible_result = (
                f"未落入区间：算法 {t_complete:.2f}s，可见光区间 [{t1}, {t2}]s，"
                f"相对区间中心误差 {err:.2f}s"
            )
    elif t_complete is None:
        visible_result = "未检测到完全融冰时间。"

    csv_path = os.path.join(cfg.out_dir, "result_table.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    plt.plot(df["time_s"], df["alpha_raw"], label="raw area ratio", alpha=0.5)
    plt.plot(df["time_s"], df["alpha_smooth"], label="smoothed area ratio", linewidth=2)
    plt.plot(df["time_s"], df["alpha_physical"], label="physical constrained", linewidth=2)
    plt.axhline(cfg.completion_area_ratio, linestyle="--", label=f"threshold={cfg.completion_area_ratio}")
    if t_complete is not None:
        plt.axvline(t_complete, linestyle="--", label=f"IR complete={t_complete:.1f}s")
    if cfg.visible_complete_interval_s is not None:
        t1, t2 = cfg.visible_complete_interval_s
        plt.axvspan(t1, t2, alpha=0.2, label="visible interval")
    plt.xlabel("Time after heating starts / s")
    plt.ylabel("Remaining ice area ratio")
    plt.title("Ice melting area tracking")
    plt.legend()
    plt.grid(True)

    curve_path = os.path.join(cfg.out_dir, "result_curve.png")
    plt.tight_layout()
    plt.savefig(curve_path, dpi=300)
    plt.close()

    summary_path = os.path.join(cfg.out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("红外冰边界识别结果总结\n")
        f.write("=" * 40 + "\n")
        f.write(f"数据路径: {cfg.data_path}\n")
        f.write(f"原始/有效帧率: {cfg.raw_fps} fps\n")
        f.write(f"处理帧率: {cfg.sample_fps} fps\n")
        f.write(f"ROI: {cfg.roi}\n")
        f.write(f"Ref ROI: {cfg.ref_roi}\n")
        f.write(f"初始冰区面积: {A0_px} px\n")
        f.write(f"像素面积换算: {pixel_area_mm2:.6f} mm^2/px\n")
        f.write(f"完全融冰面积阈值: {cfg.completion_area_ratio}\n")
        f.write(f"连续判定时间: {cfg.continuous_seconds} s\n")
        f.write(f"算法判定完全融冰时间: {t_complete}\n")
        f.write(f"可见光验证结果: {visible_result}\n")

    print("[INFO] 处理完成")
    print(f"[INFO] 结果表: {csv_path}")
    print(f"[INFO] 面积曲线: {curve_path}")
    print(f"[INFO] 总结文件: {summary_path}")
    print(f"[INFO] {visible_result}")

    return df, t_complete


if __name__ == "__main__":
    cfg = Config(
        data_path=r"./data/exp_001/thermal_data_fixed.mat",
        out_dir=r"./result/exp_001",
        raw_fps=100.0,
        sample_fps=1.0,

        heat_on_time_in_file_s=0.0,

        # 必须根据第一帧红外图实际修改
        # First-frame ice disk candidate: x=325..453, y=399..512.
        # Keep a small margin around it, but avoid the lamp and most of the cold stage.
        roi=(303, 388, 173, 124),
        ref_roi=(105, 40, 146, 173),
        select_roi_interactively=True,
        force_reselect_roi=False,
        roi_config_path=None,

        initial_ice_diameter_mm=60.0,
        #初始冰区面积 为了换算起初冰元素像素点
        completion_area_ratio=0.03,
        continuous_seconds=3.0,
        # 第一轮调试：关闭升温速率筛选和时间连续约束，只看单帧分割是否正确
        use_heating_rate=False,
        use_temporal_constraint=True ,
        ice_polarity='low',
        rate_lag_s=5.0,#与使用温度速率判别有关
        visible_complete_interval_s=(110.0, 120.0),
    )

    run_tracking(cfg)
