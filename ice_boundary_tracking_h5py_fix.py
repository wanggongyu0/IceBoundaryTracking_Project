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
from pathlib import Path
from dataclasses import dataclass
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

    raw_fps: float = 1938 / 240
    sample_fps: float = 1.0
    heat_on_time_in_file_s: float = 0.0

    roi: Tuple[int, int, int, int] = (180, 120, 280, 260)
    ref_roi: Optional[Tuple[int, int, int, int]] = (20, 20, 20, 20)

    initial_ice_diameter_mm: float = 60.0
    completion_area_ratio: float = 0.03
    continuous_seconds: float = 3.0
    min_blob_ratio: float = 0.005
    rate_lag_s: float = 5.0
    use_heating_rate: bool = True
    morph_kernel_size: int = 5
    allowed_dilate_iter: int = 2
    area_smooth_window_s: float = 3.0
    save_overlay_every_s: float = 10.0
    visible_complete_interval_s: Optional[Tuple[float, float]] = (140.0, 150.0)


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

    otsu_thr, mask_u8 = cv2.threshold(
        roi_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    mask = mask_u8 > 0

    if cfg.use_heating_rate and lag_frame is not None and dt_lag is not None and dt_lag > 0:
        lag_roi = crop_roi(lag_frame, cfg.roi)
        heating_rate = (roi_frame - lag_roi) / dt_lag
        if np.any(mask):
            rate_threshold = np.nanpercentile(heating_rate, 80)
            mask = mask & (heating_rate < rate_threshold)

    if prev_mask is not None:
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
    os.makedirs(cfg.out_dir, exist_ok=True)
    overlay_dir = os.path.join(cfg.out_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    print("[INFO] 正在读取红外数据...")
    frames = load_thermal_data(cfg.data_path)
    total_frames, H, W = frames.shape
    print(f"[INFO] 数据尺寸: N={total_frames}, H={H}, W={W}")

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
        data_path=r"./data/exp_001/thermal_data.mat",
        out_dir=r"./result/exp_001",

        # 如果软件确认1938帧对应240s，则用 1938/240。
        # 如果软件确认1938帧对应193.8s，则改为 10.0。
        raw_fps=1938 / 240,
        sample_fps=1.0,

        heat_on_time_in_file_s=0.0,

        # 必须根据第一帧红外图实际修改
        roi=(180, 120, 280, 260),
        ref_roi=(20, 20, 20, 20),

        initial_ice_diameter_mm=60.0,

        completion_area_ratio=0.03,
        continuous_seconds=3.0,
        use_heating_rate=True,
        rate_lag_s=5.0,
        visible_complete_interval_s=(140.0, 150.0),
    )

    run_tracking(cfg)
