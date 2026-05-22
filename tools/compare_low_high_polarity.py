# -*- coding: utf-8 -*-
"""
Compare low/high ice polarity on selected key frames.

Edit ROI and REF_ROI below before running:
    .\.venv\Scripts\python.exe tools\compare_low_high_polarity.py
"""

from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ice_boundary_tracking_debug_v2 import (  # noqa: E402
    Config,
    load_thermal_data,
    remove_small_components,
    save_overlay_image,
    segment_ice_frame,
)


DATA_PATH = PROJECT_ROOT / "data" / "exp_001" / "thermal_data.mat"
RESULT_ROOT = PROJECT_ROOT / "result"

# Replace these with the values printed by tools/preview_first_frame.py.
ROI = (300, 370, 190, 142)
REF_ROI = (20, 20, 20, 20)

RAW_FPS = 1938 / 240
TARGET_TIMES_S = [0, 30, 60, 90, 120, 140]


def make_config(polarity: str, out_dir: Path) -> Config:
    return Config(
        data_path=str(DATA_PATH),
        out_dir=str(out_dir),
        raw_fps=RAW_FPS,
        sample_fps=1.0,
        heat_on_time_in_file_s=0.0,
        roi=ROI,
        ref_roi=REF_ROI,
        initial_ice_diameter_mm=60.0,
        completion_area_ratio=0.03,
        continuous_seconds=3.0,
        min_blob_ratio=0.005,
        rate_lag_s=5.0,
        use_heating_rate=False,
        use_temporal_constraint=False,
        ice_polarity=polarity,
        morph_kernel_size=5,
        allowed_dilate_iter=2,
        area_smooth_window_s=3.0,
        save_overlay_every_s=1.0,
        visible_complete_interval_s=(140.0, 150.0),
    )


def frame_index_for_time(time_s: float, total_frames: int) -> int:
    idx = int(round(time_s * RAW_FPS))
    return max(0, min(idx, total_frames - 1))


def add_panel_title(image: np.ndarray, title: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return out


def run_polarity(frames: np.ndarray, polarity: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_paths = []

    cfg = make_config(polarity, out_dir)
    first_mask, _ = segment_ice_frame(frames[0], cfg)
    a0_px = int(np.sum(first_mask))
    if a0_px <= 0:
        raise RuntimeError(f"Initial frame found no ice pixels for polarity={polarity}.")
    min_blob_area = max(5, int(a0_px * cfg.min_blob_ratio))

    for target_t in TARGET_TIMES_S:
        idx = frame_index_for_time(target_t, len(frames))
        actual_t = idx / RAW_FPS - cfg.heat_on_time_in_file_s
        mask, info = segment_ice_frame(frames[idx], cfg)
        mask = remove_small_components(mask, min_blob_area)
        alpha = float(np.sum(mask) / a0_px)

        save_path = out_dir / f"overlay_t_{target_t:03d}s_{polarity}.png"
        save_overlay_image(info["roi_u8"], mask, str(save_path), actual_t, alpha)
        overlay_paths.append(save_path)
        print(
            f"[{polarity}] target={target_t:>3}s, frame={idx:>4}, "
            f"actual_t={actual_t:7.2f}s, alpha={alpha:.4f}, saved={save_path}"
        )

    return overlay_paths


def build_summary(low_paths, high_paths, save_path: Path):
    rows = []
    for label, paths in [("low", low_paths), ("high", high_paths)]:
        panels = []
        for target_t, path in zip(TARGET_TIMES_S, paths):
            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(f"Could not read overlay image: {path}")
            panels.append(add_panel_title(img, f"{label}, t={target_t}s"))
        rows.append(cv2.hconcat(panels))

    summary = cv2.vconcat(rows)
    cv2.imwrite(str(save_path), summary)
    print(f"[INFO] Saved summary: {save_path}")


def main():
    print(f"[INFO] Loading thermal data: {DATA_PATH}")
    frames = load_thermal_data(str(DATA_PATH))
    print(f"[INFO] frames shape: {frames.shape}")
    print(f"[INFO] ROI={ROI}, REF_ROI={REF_ROI}")

    low_paths = run_polarity(frames, "low", RESULT_ROOT / "compare_low")
    high_paths = run_polarity(frames, "high", RESULT_ROOT / "compare_high")
    build_summary(low_paths, high_paths, RESULT_ROOT / "compare_low_high_summary.png")


if __name__ == "__main__":
    main()
