"""User-interface helpers for simple overlays.

This module keeps display annotations separate from the main processing logic.
Students can extend the overlay as the pipeline becomes more informative.
"""

from __future__ import annotations

import sys
from typing import Any, Callable

import cv2
import numpy as np

BUTTON_LABELS = ("View mode", "Motion blur", "Edge", "Detector", "Quit")
BAR_HEIGHT = 52
_BTN_MARGIN_X = 10
_BTN_MARGIN_Y = 8
_BTN_GAP = 10
_FILL = (228, 228, 228)
_BORDER = (90, 90, 90)
_TEXT = (25, 25, 25)


def _button_rects(w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Return the rectangle coordinates for the buttons"""
    y0 = h - BAR_HEIGHT + _BTN_MARGIN_Y
    bh = BAR_HEIGHT - 2 * _BTN_MARGIN_Y
    inner_w = w - 2 * _BTN_MARGIN_X - 2 * _BTN_GAP
    bw = inner_w // len(BUTTON_LABELS)
    rects = []
    for i in range(len(BUTTON_LABELS)):
        x = _BTN_MARGIN_X + i * (bw + _BTN_GAP)
        rects.append((x, y0, bw, bh))
    return rects


def draw_status_overlay(frame: np.ndarray) -> np.ndarray:
    """Draw status text at the top and pushbuttons along the bottom."""
    output = frame.copy()
    h, w = output.shape[:2]

    cv2.rectangle(output, (0, h - BAR_HEIGHT), (w, h), (200, 200, 200), -1)
    cv2.line(output, (0, h - BAR_HEIGHT), (w, h - BAR_HEIGHT), _BORDER, 1)

    for i, (x, y, bw, bh) in enumerate(_button_rects(w, h)):
        cv2.rectangle(output, (x, y), (x + bw, y + bh), _FILL, -1)
        cv2.rectangle(output, (x, y), (x + bw, y + bh), _BORDER, 1)
        label = BUTTON_LABELS[i]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tx = x + (bw - tw) // 2
        ty = y + (bh + th) // 2
        cv2.putText(output, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _TEXT, 1)

    return output


def make_button_mouse_callback(
    ui_state: dict[str, Any],
) -> Callable[[int, int, int, int, Any], None]:
    """Return a cv2 mouse callback that updates ui_state on button hits."""

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ww = int(ui_state.get("w") or 0)
        hh = int(ui_state.get("h") or 0)
        if ww <= 0 or hh <= 0:
            return
        for i, (bx, by, bw, bh) in enumerate(_button_rects(ww, hh)):
            if bx <= x < bx + bw and by <= y < by + bh:
                if i == 0:
                    ui_state["view_mode"] = (int(ui_state.get("view_mode", 0)) + 1) % 3
                elif i == 1:
                    ui_state["blur_mode"] = (int(ui_state.get("blur_mode", 0)) + 1) % 5
                elif i == 2:
                    ui_state["edge_mode"] = (int(ui_state.get("edge_mode", 0)) + 1) % 3
                elif i == 3:
                    ui_state["det_mode"] = (int(ui_state.get("det_mode", 0)) + 1) % 3
                elif i == 4:
                    sys.exit(0)
                break

    return on_mouse
