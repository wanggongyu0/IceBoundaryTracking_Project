# -*- coding: utf-8 -*-
"""
Preview the first thermal frame and interactively select ice/ref ROIs.

Controls:
  1. Drag a rectangle for the ice ROI, then press Enter or Space.
  2. Drag a rectangle for the reference ROI, then press Enter or Space.
  3. Press c to cancel the current selection.
"""

from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ice_boundary_tracking_debug_v2 import load_thermal_data  # noqa: E402


DATA_PATH = PROJECT_ROOT / "data" / "exp_001" / "thermal_data.mat"
OUT_PATH = PROJECT_ROOT / "tools" / "first_frame_preview.png"


def normalize_to_u8(frame: np.ndarray) -> np.ndarray:
    p1, p99 = np.nanpercentile(frame, [1, 99])
    if abs(p99 - p1) < 1e-6:
        return np.zeros_like(frame, dtype=np.uint8)
    norm = (frame - p1) / (p99 - p1)
    return (np.clip(norm, 0, 1) * 255).astype(np.uint8)


def select_roi(window_name: str, image: np.ndarray, label: str):
    cv2.imshow(window_name, image)
    print(f"[INFO] Please select {label}, then press Enter or Space.")
    x, y, w, h = cv2.selectROI(window_name, image, showCrosshair=True, fromCenter=False)
    if w <= 0 or h <= 0:
        raise RuntimeError(f"{label} selection was cancelled or empty.")
    return int(x), int(y), int(w), int(h)


def draw_roi(image: np.ndarray, roi, label: str, color):
    x, y, w, h = roi
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    text_y = max(22, y - 8)
    cv2.putText(image, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():
    frames = load_thermal_data(str(DATA_PATH))
    first_frame = frames[0]

    frame_u8 = normalize_to_u8(first_frame)
    preview = cv2.applyColorMap(frame_u8, cv2.COLORMAP_INFERNO)

    window_name = "First frame ROI selector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    ice_roi = select_roi(window_name, preview.copy(), "ice ROI")

    preview_with_ice = preview.copy()
    draw_roi(preview_with_ice, ice_roi, "ice ROI", (0, 255, 255))
    ref_roi = select_roi(window_name, preview_with_ice, "reference ROI")

    final_preview = preview.copy()
    draw_roi(final_preview, ice_roi, "ice ROI", (0, 255, 255))
    draw_roi(final_preview, ref_roi, "ref ROI", (255, 255, 255))

    cv2.imwrite(str(OUT_PATH), final_preview)
    cv2.imshow(window_name, final_preview)

    print(f"roi={ice_roi}")
    print(f"ref_roi={ref_roi}")
    print(f"[INFO] Saved preview: {OUT_PATH}")
    print("[INFO] Press any key in the image window to close.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
