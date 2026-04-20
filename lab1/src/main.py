"""Main application entry point for the student Lab 1 scaffold.

This file shows the orchestration pattern for a small vision system:
input -> processing -> output.

Students should read this file first to understand how the modules connect.
"""

from __future__ import annotations

import argparse

import cv2

from camera import FrameSource
from pipeline_classical import process_frame
from settings import APP_TITLE, DEFAULT_PATH, DEFAULT_SOURCE
from ui import draw_status_overlay, make_button_mouse_callback


def _side_by_side(input_bgr, processed_bgr):
    """Stack input (left) and processed (right); resize processed if shapes differ."""
    left = input_bgr.copy()
    right = processed_bgr.copy()
    th, tw = left.shape[:2]
    if right.shape[:2] != (th, tw):
        right = cv2.resize(right, (tw, th), interpolation=cv2.INTER_AREA)
    cv2.putText(left, "Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(right, "Processed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    return cv2.hconcat([left, right])


def parse_args() -> argparse.Namespace:
    """Parse command-line options for selecting the input source."""
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument(
        "--source",
        choices=["webcam", "image", "video"],
        default=DEFAULT_SOURCE,
        help="Choose the input source.",
    )
    parser.add_argument(
        "--path",
        default=DEFAULT_PATH,
        help="Optional path for image or video input.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        metavar="N",
        help="Webcam index when --source webcam (default 0).",
    )
    return parser.parse_args()


def main() -> int:
    """Run the Lab 1 application loop."""
    args = parse_args()
    source = FrameSource(args.source, args.path, args.camera)

    try:
        source.open()
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 1

    ui_state: dict = {"w": 0, "h": 0, "view_mode": 0, "blur_mode": 0, "edge_mode": 0, "det_mode": 0}
    blur_state: dict = {}
    det_state: dict = {}
    cv2.namedWindow(APP_TITLE, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(APP_TITLE, make_button_mouse_callback(ui_state))

    while True:
        ok, frame = source.read()
        if not ok or frame is None:
            print("[info] No more frames available. Exiting.")
            break

        frame = cv2.flip(frame, 1)

        processed_frame = process_frame(
            frame,
            ui_state["view_mode"],
            ui_state["blur_mode"],
            blur_state,
            ui_state["edge_mode"],
            ui_state["det_mode"],
            det_state,
        )
        combined = _side_by_side(frame, processed_frame)
        ui_state["w"] = int(combined.shape[1])
        ui_state["h"] = int(combined.shape[0])
        annotated_frame = draw_status_overlay(combined)

        cv2.imshow(APP_TITLE, annotated_frame)
        key = cv2.waitKey(1 if source.is_streaming else 0) & 0xFF

        if key in (27, ord("q")):
            break

        if not source.is_streaming:
            break

    source.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
