from __future__ import annotations

import cv2
import numpy as np


def apply_edge(frame: np.ndarray, edge_mode: int) -> tuple[np.ndarray, str]:
    if edge_mode == 0:
        return frame, "off"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if edge_mode == 1:
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), "Canny"
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR), "Sobel"
