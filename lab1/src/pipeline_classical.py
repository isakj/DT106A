"""pipeline + YOLO / Faster R-CNN overlays"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from pipeline_detection import apply_detection
from pipeline_color_mode import apply_color_mode
from pipeline_edge import apply_edge
from pipeline_motion_blur import apply_motion_blur


def _det_banner(out: np.ndarray, text: str) -> None:
    cv2.putText(out, text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)


def process_frame(
    frame: np.ndarray,
    view_mode: int,
    blur_mode: int,
    blur_state: dict[str, Any],  # start with empty dict
    edge_mode: int,
    det_mode: int,
    det_state: dict[str, Any],  # start with empty dict
) -> np.ndarray:
    """process one frame and return the processed frame"""

    status = []
    frame, mode_name = apply_color_mode(frame, view_mode)
    status += [mode_name]
    frame, mode_name = apply_motion_blur(blur_state, frame, blur_mode)
    status += [mode_name]
    frame, mode_name = apply_edge(frame, edge_mode)
    status += [mode_name]
    frame, mode_name = apply_detection(det_state, frame, det_mode)
    status += [mode_name]

    _det_banner(frame, ", ".join(status))

    return frame
