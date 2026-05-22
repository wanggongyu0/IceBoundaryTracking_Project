# -*- coding: utf-8 -*-
"""
ice_boundary_tracking.py

用途：
    基于红外温度矩阵的冰边界动态识别、融冰面积计算与完全融冰时间判定。

适用课题：
    电热膜加热1060铝板模拟机翼前缘结冰/融冰全过程的红外热像追踪研究。

使用前准备：
    1. 将 ThermPulse S12 导出的 .novel 数据先转换/导出为 .mat、.npy、.npz 或 csv/txt帧序列。
    2. 在 Config 中修改 data_path、out_dir、roi、ref_roi、visible_complete_interval_s 等参数。
    3. 建议先用 sample_fps=1.0 跑通流程，再在融冰完成附近提升到 5 或 10 fps 精细分析。

依赖安装：
    pip install numpy opencv-python pandas matplotlib scipy
"""

import os
import glob
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


# ============================================================
# 1. 参数配置区
# ============================================================

@dataclass
class Config:
    # 数据路径：支持 .npy / .npz / .mat / csv文件夹
    data_path: str = r"./thermal_data.mat"

    # 输出目录
    out_dir: str = r"./result"

    # 原始红外帧率，ThermPulse S12 当前实验为100 fps
    raw_fps: float = 100.0

    # 实际处理帧率
    # 建议先用 1 fps 快速分析；后期接近融冰完成阶段可改成 5 或 10 fps
    sample_fps: float = 1.0

    # 红外文件中，电热膜开始通电的时间点
    # 如果红外采集一开始就同时插电，则设为 0
    # 如果先采集了10秒才插电，则设为 10
    heat_on_time_in_file_s: float = 0.0

    # ROI：冰区所在图像区域，格式为 x, y, w, h
    # 你需要根据第一帧红外图像手动修改
    roi: Tuple[int, int, int, int] = (180, 120, 280, 260)

    # 参考温度点ROI：远离冰层和加热区的小区域，用作背景/室温参考
    # 格式同样是 x, y, w, h
    # 建议选一个固定点附近 10x10 或 20x20 像素区域
    ref_roi: Optional[Tuple[int, int, int, int]] = (20, 20, 20, 20)

    # 初始圆形冰层直径，单位 mm
    initial_ice_diameter_mm: float = 60.0

    # 面积阈值：剩余冰面积低于初始面积的多少，认为接近融冰完成
    # 建议 0.02~0.03
    completion_area_ratio: float = 0.03

    # 连续多少秒保持在阈值以下，才判定为完全融冰
    continuous_seconds: float = 3.0

    # 最小连通域面积比例，小于初始面积的该比例则删除
    min_blob_ratio: float = 0.005

    # 升温速率辅助判断的时间间隔，单位 s
    rate_lag_s: float = 5.0

    # 是否使用升温速率辅助筛选
    use_heating_rate: bool = True

    # 形态学核尺寸，建议 3、5、7
    morph_kernel_size: int = 5

    # 时间连续约束中，允许当前冰区在上一帧冰区外扩多少次膨胀
    allowed_dilate_iter: int = 2

    # 面积曲线平滑窗口，单位 s
    area_smooth_window_s: float = 3.0

    # 每隔多少秒保存一张边界叠加图
    save_overlay_every_s: float = 10.0

    # 手机可见光人工判断的完全融冰时间区间
    # 例如肉眼判断145s左右完成、误差±5s，则填 (140, 150)
    visible_complete_interval_s: Optional[Tuple[float, float]] = (140.0, 150.0)


# ============================================================
# 2. 数据读取函数
# ============================================================

def ensure_frames_nhw(arr: np.ndarray) -> np.ndarray:
    """
    将红外数据统一为 [N, H, W] 格式。
    N: 帧数
    H: 图像高度
    W: 图像宽度
    """
    arr = np.asarray(arr).squeeze()

    if arr.ndim != 3:
        raise ValueError(f"温度数据必须是三维数组，但当前维度为 {arr.shape}")

    # 常见情况1：[N, H, W]
    # 常见情况2：[H, W, N]
    # 如果前两维像图像尺寸，最后一维像帧数，则转置
    if arr.shape[0] in [512, 640] and arr.shape[1] in [512, 640] and arr.shape[2] > 100:
        arr = np.moveaxis(arr, -1, 0)

    return arr.astype(np.float32)


def load_thermal_data(data_path: str) -> np.ndarray:
    """
    支持读取：
    1. .npy: 三维数组 [N,H,W] 或 [H,W,N]
    2. .npz: 内含 frames 或第一个三维数组
    3. .mat: 自动寻找第一个三维矩阵
    4. 文件夹：读取其中 csv/txt 文件，每个文件一帧
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"找不到数据路径：{data_path}")

    if path.is_file():
        suffix = path.suffix.lower()

        if suffix == ".npy":
            arr = np.load(path)
            return ensure_frames_nhw(arr)

        elif suffix == ".npz":
            data = np.load(path)
            if "frames" in data:
                return ensure_frames_nhw(data["frames"])
            for key in data.files:
                if data[key].ndim == 3:
                    return ensure_frames_nhw(data[key])
            raise ValueError(".npz 文件中没有找到三维温度矩阵")

        elif suffix == ".mat":
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
            print(f"[INFO] 从 .mat 文件中读取变量：{key}, shape={arr.shape}")
            return ensure_frames_nhw(arr)

        else:
            raise ValueError(
                f"暂不支持直接读取 {suffix} 文件。\n"
                f"如果是 .novel 文件，请先从 ThermPulse 软件导出为 .mat/.npy/csv，"
                f"或者后续把一个样例文件给我，我再帮你适配读取函数。"
            )

    elif path.is_dir():
        files = sorted(
            glob.glob(str(path / "*.csv")) +
            glob.glob(str(path / "*.txt"))
        )
        if not files:
            raise ValueError("文件夹中没有找到 csv 或 txt 温度矩阵文件")

        frames = []
        for f in files:
            try:
                mat = np.loadtxt(f, delimiter=",")
            except Exception:
                mat = np.loadtxt(f)
            frames.append(mat)

        arr = np.stack(frames, axis=0)
        return ensure_frames_nhw(arr)

    else:
        raise ValueError(f"无法识别的数据路径：{data_path}")


# ============================================================
# 3. 图像处理基础函数
# ============================================================

def crop_roi(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return frame[y:y+h, x:x+w]


def get_ref_temperature(frame: np.ndarray, ref_roi: Optional[Tuple[int, int, int, int]]) -> float:
    """
    取参考点小区域的中位数温度。
    不建议取单像素，容易受噪声影响。
    """
    if ref_roi is None:
        return 0.0

    x, y, w, h = ref_roi
    ref_area = frame[y:y+h, x:x+w]
    return float(np.nanmedian(ref_area))


def robust_normalize_to_uint8(img: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    将温度场稳健归一化到 0~255。
    使用5%和95%分位数，避免异常点影响。
    """
    p5, p95 = np.nanpercentile(img, [5, 95])
    if abs(p95 - p5) < 1e-6:
        norm = np.zeros_like(img, dtype=np.uint8)
        return norm, p5, p95

    norm = (img - p5) / (p95 - p5)
    norm = np.clip(norm, 0, 1)
    norm_u8 = (norm * 255).astype(np.uint8)

    return norm_u8, float(p5), float(p95)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    填充二值图中的内部孔洞。
    """
    mask_u8 = (mask.astype(np.uint8) * 255)
    h, w = mask_u8.shape

    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    # 从左上角开始填充背景
    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    flood_inv = cv2.bitwise_not(flood)
    filled = mask_u8 | flood_inv

    return filled > 0


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    删除小连通域。
    注意：不只保留最大连通域，因为融冰后期可能存在多个残余冰块。
    """
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    out = np.zeros_like(mask_u8)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == label] = 1

    return out.astype(bool)


def morphology_clean(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    形态学开闭运算，平滑边界、去除噪声。
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_u8 = mask.astype(np.uint8) * 255

    # 开运算：去掉小噪点
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    # 闭运算：连接断裂边界
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)

    mask_bool = mask_u8 > 0
    mask_bool = fill_holes(mask_bool)

    return mask_bool


# ============================================================
# 4. 单帧冰区识别函数
# ============================================================

def segment_ice_frame(
    frame: np.ndarray,
    cfg: Config,
    prev_mask: Optional[np.ndarray] = None,
    lag_frame: Optional[np.ndarray] = None,
    dt_lag: Optional[float] = None
):
    """
    对单帧红外温度场进行冰区识别。
    """
    roi_frame = crop_roi(frame, cfg.roi)
    ref_temp = get_ref_temperature(frame, cfg.ref_roi)

    # 相对温度场：减去参考点温度，降低实验间环境温度漂移影响
    roi_rel = roi_frame - ref_temp

    # 稳健归一化
    roi_u8, p5, p95 = robust_normalize_to_uint8(roi_rel)

    # Otsu自适应阈值
    # THRESH_BINARY_INV 表示低温区域为前景，即候选冰区
    otsu_thr, mask_u8 = cv2.threshold(
        roi_u8,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    mask = mask_u8 > 0

    # 升温速率辅助筛选
    if cfg.use_heating_rate and lag_frame is not None and dt_lag is not None and dt_lag > 0:
        lag_roi = crop_roi(lag_frame, cfg.roi)
        heating_rate = (roi_frame - lag_roi) / dt_lag

        # 裸露铝板或水膜区域升温通常更快
        # 删除升温速率明显偏高的区域
        if np.any(mask):
            rate_threshold = np.nanpercentile(heating_rate, 80)
            mask = mask & (heating_rate < rate_threshold)

    # 时间连续性约束：当前冰区不应凭空出现在上一帧冰区很远以外
    if prev_mask is not None:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg.morph_kernel_size, cfg.morph_kernel_size)
        )
        allowed = cv2.dilate(
            prev_mask.astype(np.uint8),
            kernel,
            iterations=cfg.allowed_dilate_iter
        ) > 0

        mask = mask & allowed

    # 形态学清理
    mask = morphology_clean(mask, cfg.morph_kernel_size)

    info = {
        "ref_temp": ref_temp,
        "p5": p5,
        "p95": p95,
        "otsu_thr": float(otsu_thr),
        "roi_u8": roi_u8
    }

    return mask, info


# ============================================================
# 5. 保存边界叠加图
# ============================================================

def save_overlay_image(
    roi_u8: np.ndarray,
    mask: np.ndarray,
    save_path: str,
    time_s: float,
    alpha: Optional[float] = None
):
    """
    保存红外伪彩图 + 冰边界轮廓叠加图。
    """
    color = cv2.applyColorMap(roi_u8, cv2.COLORMAP_INFERNO)

    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 黄色边界
    cv2.drawContours(color, contours, -1, (0, 255, 255), 2)

    text = f"t = {time_s:.1f} s"
    if alpha is not None:
        text += f", alpha = {alpha:.3f}"

    cv2.putText(
        color,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imwrite(save_path, color)


# ============================================================
# 6. 完全融冰时间判断
# ============================================================

def find_completion_time(times: np.ndarray, alpha: np.ndarray, eps: float, continuous_s: float):
    """
    判断完全融冰时间：
    剩余面积占比 alpha 连续 continuous_s 秒低于 eps。
    """
    if len(times) < 2:
        return None

    dt = np.median(np.diff(times))
    window = max(1, int(round(continuous_s / dt)))

    for i in range(0, len(alpha) - window + 1):
        if np.all(alpha[i:i+window] <= eps):
            return float(times[i])

    return None


# ============================================================
# 7. 主处理函数
# ============================================================

def run_tracking(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)
    overlay_dir = os.path.join(cfg.out_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    print("[INFO] 正在读取红外数据...")
    frames = load_thermal_data(cfg.data_path)
    total_frames, H, W = frames.shape
    print(f"[INFO] 数据尺寸: N={total_frames}, H={H}, W={W}")

    # 抽帧
    step = max(1, int(round(cfg.raw_fps / cfg.sample_fps)))
    frames_s = frames[::step]
    times = np.arange(len(frames_s)) * step / cfg.raw_fps - cfg.heat_on_time_in_file_s

    # 只分析通电后的部分
    start_idx = int(np.searchsorted(times, 0.0))
    frames_s = frames_s[start_idx:]
    times = times[start_idx:]

    print(f"[INFO] 抽帧后用于处理的帧数: {len(frames_s)}")
    print(f"[INFO] 时间范围: {times[0]:.2f} s ~ {times[-1]:.2f} s")

    prev_mask = None
    records = []
    masks = []

    rate_lag_frames = max(1, int(round(cfg.rate_lag_s * cfg.sample_fps)))
    save_every_frames = max(1, int(round(cfg.save_overlay_every_s * cfg.sample_fps)))

    A0_px = None
    pixel_area_mm2 = None

    for i, (frame, t) in enumerate(zip(frames_s, times)):
        # 找到用于计算升温速率的滞后帧
        if i - rate_lag_frames >= 0:
            lag_frame = frames_s[i - rate_lag_frames]
            dt_lag = times[i] - times[i - rate_lag_frames]
        else:
            lag_frame = None
            dt_lag = None

        mask, info = segment_ice_frame(
            frame=frame,
            cfg=cfg,
            prev_mask=prev_mask,
            lag_frame=lag_frame,
            dt_lag=dt_lag
        )

        # 第一帧作为初始冰区面积
        if A0_px is None:
            A0_px = int(np.sum(mask))
            if A0_px <= 0:
                raise RuntimeError("初始帧没有识别到冰区，请检查ROI或阈值处理结果。")

            initial_area_mm2 = np.pi * (cfg.initial_ice_diameter_mm / 2.0) ** 2
            pixel_area_mm2 = initial_area_mm2 / A0_px

            min_blob_area = max(5, int(A0_px * cfg.min_blob_ratio))
            print(f"[INFO] 初始冰区像素面积 A0 = {A0_px} px")
            print(f"[INFO] 由60mm直径换算得到 pixel_area = {pixel_area_mm2:.4f} mm^2/px")
            print(f"[INFO] 最小连通域面积阈值 = {min_blob_area} px")
        else:
            min_blob_area = max(5, int(A0_px * cfg.min_blob_ratio))

        # 初步分割后，再删除小连通域
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
            "otsu_thr": info["otsu_thr"]
        })

        masks.append(mask)
        prev_mask = mask.copy()

        # 保存边界叠加图
        if i % save_every_frames == 0:
            save_path = os.path.join(overlay_dir, f"overlay_t_{t:07.2f}s.png")
            save_overlay_image(
                roi_u8=info["roi_u8"],
                mask=mask,
                save_path=save_path,
                time_s=t,
                alpha=alpha_raw
            )

    df = pd.DataFrame(records)

    # 面积曲线平滑
    smooth_window = max(1, int(round(cfg.area_smooth_window_s * cfg.sample_fps)))
    if smooth_window % 2 == 0:
        smooth_window += 1

    df["alpha_smooth"] = (
        df["alpha_raw"]
        .rolling(window=smooth_window, center=True, min_periods=1)
        .median()
    )

    # 物理约束后的面积曲线：融冰过程中剩余冰面积总体应下降
    df["alpha_physical"] = np.minimum.accumulate(df["alpha_smooth"].values)

    # 判断完全融冰时间
    t_complete = find_completion_time(
        times=df["time_s"].values,
        alpha=df["alpha_physical"].values,
        eps=cfg.completion_area_ratio,
        continuous_s=cfg.continuous_seconds
    )

    # 与可见光验证区间对比
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
        visible_result = "未检测到完全融冰时间：可能是面积阈值过低、ROI不准确，或红外数据未覆盖融冰完成时刻。"

    # 保存结果表
    csv_path = os.path.join(cfg.out_dir, "result_table.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 绘制面积曲线
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

    # 保存总结
    summary_path = os.path.join(cfg.out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("红外冰边界识别结果总结\n")
        f.write("=" * 40 + "\n")
        f.write(f"数据路径: {cfg.data_path}\n")
        f.write(f"原始帧率: {cfg.raw_fps} fps\n")
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


# ============================================================
# 8. 程序入口
# ============================================================

if __name__ == "__main__":
    cfg = Config(
        data_path=r"data/exp_001/thermal_data.mat",
        out_dir=r"./result/exp_001",

        raw_fps=100.0,
        sample_fps=1.0,

        # 如果红外采集开始时立刻插电，就写0
        heat_on_time_in_file_s=0.0,

        # 下面两个ROI需要你根据实际红外图像修改
        roi=(180, 120, 280, 260),
        ref_roi=(20, 20, 20, 20),

        initial_ice_diameter_mm=60.0,

        completion_area_ratio=0.03,
        continuous_seconds=3.0,

        use_heating_rate=True,
        rate_lag_s=5.0,

        visible_complete_interval_s=(140.0, 150.0)
    )

    run_tracking(cfg)
