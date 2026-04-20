from __future__ import annotations

import cv2
import numpy as np


def apply_color_mode(frame: np.ndarray, view_mode: int) -> tuple[np.ndarray, str]:
    if view_mode == 0:
        output = frame.copy()
        mode_name = "normal"
    elif view_mode == 1:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        mode_name = "b/w"
    else:
        x = frame.astype(np.float32)
        gray = np.mean(x, axis=2, keepdims=True)
        coeff = 2
        y = (x - gray) * coeff + gray
        output = np.clip(y, 0.0, 255.0).astype(np.uint8)
        mode_name = "contrast"
    return output, mode_name
