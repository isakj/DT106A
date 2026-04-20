from __future__ import annotations

import sys
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)


def apply_detection(
    det_state: dict[str, Any], frame: np.ndarray, det_mode: int
) -> tuple[np.ndarray, str]:
    ds = det_state if det_state is not None else {}

    try:
        if det_mode == 0:
            mode_name = "off"
            out = frame
        elif det_mode == 1:
            mode_name = "YOLO"
            out = frame.copy()
            if "yolo" not in det_state:
                from ultralytics import YOLO

                det_state["yolo"] = YOLO("yolov8n.pt")
            model = det_state["yolo"]

            res = model(out, verbose=False)[0]
            plotted = res.plot()
            if plotted is not None:
                out = plotted
        else:
            mode_name = "R-CNN"
            out = frame.copy()

            if "rcnn" not in det_state:
                w = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                m = fasterrcnn_resnet50_fpn(weights=w).eval()
                if torch.cuda.is_available():
                    m = m.cuda()
                det_state["rcnn"] = m
                det_state["rcnn_device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = det_state["rcnn"]
            dev = ds["rcnn_device"]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = torch.from_numpy(rgb).permute(2, 0, 1).to(dev)
            with torch.no_grad():
                pred = model([t])[0]
            boxes = pred["boxes"].detach().cpu().numpy()
            scores = pred["scores"].detach().cpu().numpy()
            thr = 0.7
            for box, sc in zip(boxes, scores):
                if sc < thr:
                    break
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(
                    out,
                    f"{sc:.2f}",
                    (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 200, 0),
                    1,
                )
    except Exception as exc:  # noqa: BLE001
        print(f"[{mode_name} error] {exc}", file=sys.stderr)
        raise SystemExit(1)

    return out, mode_name
