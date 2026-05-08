import cv2
import os
from ultralytics import YOLO
from settings import MODEL_NAME, MODEL_DIR, PRED_THRESHOLD


class LivePredictor:
    def __init__(self, model_name=MODEL_NAME, camera_index=0, conf_threshold=PRED_THRESHOLD):
        self.camera_index = camera_index
        self.conf_threshold = conf_threshold
        self.cap = None

        # Resolve model path relative to this script (mirrors train.py layout)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(MODEL_DIR, MODEL_NAME, "weights", "best.pt"
        )

        print(f"[INFO] Loading model from: {model_path}")
        self.model = YOLO(model_path)
        print("[INFO] Model loaded successfully")

    def open_camera(self):
        print("[INFO] Opening camera...")
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera (index={self.camera_index}). Try index 1."
            )

        print("[INFO] Camera opened successfully")

    def draw_overlay(self, frame, result):
        """Draw bounding boxes, labels, and HUD onto the frame."""
        boxes = result.boxes

        for box in boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue

            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label background
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(
                frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
            )

        # HUD
        n_det = sum(1 for b in boxes if float(b.conf[0]) >= self.conf_threshold)
        cv2.putText(
            frame, f"Detections: {n_det}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.putText(
            frame, f"Conf threshold: {self.conf_threshold:.2f}  (+/- to adjust)  |  q to quit",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
        )

        return frame

    def run(self):
        try:
            self.open_camera()
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            return

        print("[INFO] Running live inference. Press 'q' / ESC to quit.")
        print("[INFO] Press '+' to raise confidence threshold, '-' to lower it.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Failed to read frame")
                break

            # Run inference (verbose=False keeps the console clean)
            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                verbose=False,
            )

            annotated = self.draw_overlay(frame, results[0])
            cv2.imshow("YOLOv8 Live Prediction", annotated)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):           # q or ESC
                print("[INFO] Exiting...")
                break
            elif key == ord("+") or key == ord("="):
                self.conf_threshold = min(0.99, round(self.conf_threshold + 0.05, 2))
                print(f"[INFO] Confidence threshold: {self.conf_threshold}")
            elif key == ord("-"):
                self.conf_threshold = max(0.05, round(self.conf_threshold - 0.05, 2))
                print(f"[INFO] Confidence threshold: {self.conf_threshold}")

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    predictor = LivePredictor()
    predictor.run()