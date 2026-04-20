from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


def apply_motion_blur(
    blur_state: dict[str, Any], frame: np.ndarray, blur_mode: int
) -> tuple[np.ndarray, str]:
    if "buf" not in blur_state:
        blur_state["buf"] = deque(maxlen=10)
    buf = blur_state["buf"]
    buf.append(frame.copy())

    if blur_mode != 0:
        stack = np.stack(list(buf), axis=0).astype(np.float32)
        m = np.mean(stack, axis=0)
        if blur_mode == 1:
            frame = m.astype(np.uint8)
            mode_name = "mean"
        elif blur_mode == 2:
            frame[:, :, 2] = m[:, :, 2]
            mode_name = "R"
        elif blur_mode == 3:
            frame[:, :, 1] = m[:, :, 1]
            mode_name = "G"
        elif blur_mode == 4:
            frame[:, :, 0] = m[:, :, 0]
            mode_name = "B"
    else:
        mode_name = "normal"
    return frame, mode_name
