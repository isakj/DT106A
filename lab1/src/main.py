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
from ui import draw_status_overlay


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

    while True:
        ok, frame = source.read()
        if not ok or frame is None:
            print("[info] No more frames available. Exiting.")
            break

        processed_frame, status = process_frame(frame)
        annotated_frame = draw_status_overlay(processed_frame, status)

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
