"""Input source abstraction for webcam, image, and video.

The goal of this file is to hide source-specific details behind one small API.
Students can extend this class if they need extra controls later in the lab.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class FrameSource:
    """Present webcam, image, and video input through one small interface."""

    def __init__(self, source_type: str, path: str = "", camera_index: int = 0) -> None:
        self.source_type = source_type
        self.path = path
        self.camera_index = camera_index
        self.capture: cv2.VideoCapture | None = None
        self.single_frame: np.ndarray | None = None
        self.is_streaming = source_type in {"webcam", "video"}

    def open(self) -> None:
        """Open the selected source or raise a helpful error."""
        if self.source_type == "webcam":
            self.capture = cv2.VideoCapture(self.camera_index)
            if not self.capture.isOpened():
                raise RuntimeError(
                    "Could not open webcam. Try another --camera index, "
                    "or '--source image' / '--source video'."
                )
            return

        if self.source_type == "image":
            if not self.path:
                raise RuntimeError("A file path is required for image and video sources.")
            input_path = Path(self.path)
            if not input_path.exists():
                raise RuntimeError(f"Input file not found: {input_path}")

            self.single_frame = cv2.imread(str(input_path))
            if self.single_frame is None:
                raise RuntimeError(f"Could not read image file: {input_path}")
            return

        if self.source_type == "video":
            if not self.path:
                raise RuntimeError("A file path is required for image and video sources.")
            input_path = Path(self.path)
            if not input_path.exists():
                raise RuntimeError(f"Input file not found: {input_path}")

            self.capture = cv2.VideoCapture(str(input_path))
            if not self.capture.isOpened():
                raise RuntimeError(f"Could not open video file: {input_path}")
            return

        raise RuntimeError(f"Unsupported source type: {self.source_type}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read the next frame from the selected source."""
        if self.source_type == "image":
            if self.single_frame is None:
                return False, None
            frame = self.single_frame.copy()
            self.single_frame = None
            return True, frame

        if self.capture is None:
            return False, None

        return self.capture.read()

    def release(self) -> None:
        """Release any OpenCV capture object."""
        if self.capture is not None:
            self.capture.release()
