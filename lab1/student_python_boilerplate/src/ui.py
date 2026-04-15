"""User-interface helpers for simple overlays.

This module keeps display annotations separate from the main processing logic.
Students can extend the overlay as the pipeline becomes more informative.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def draw_status_overlay(frame: np.ndarray, status: dict[str, Any]) -> np.ndarray:
    """Draw a small status block on top of the frame.

    TODO:
    - Add more helpful overlay text for your own pipeline.
    - Consider showing thresholds, mode names, or detected object counts.
    """
    output = frame.copy()

    mode_text = f"Mode: {status.get('mode', 'unknown')}"
    note_text = str(status.get("note", "TODO: add overlay details"))

    cv2.putText(output, mode_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(output, note_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return output
