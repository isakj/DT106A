"""Classical image-processing pipeline for the student scaffold.

This version is intentionally simple. It demonstrates the function boundary for
Lab 1 without giving away a full solution.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def process_frame(frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Process one frame and return both output data and lightweight status.

    TODO:
    - Add at least one meaningful processing step beyond grayscale conversion.
    - Experiment with thresholding, edges, contours, or another classical method.
    - Return information that can be shown in the UI overlay.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Minimal example only:
    # convert grayscale back to BGR so the display pipeline stays consistent.
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    status = {
        "mode": "grayscale_demo",
        "note": "TODO: extend the classical pipeline in pipeline_classical.py",
    }
    return output, status
