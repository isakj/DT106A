import cv2
import math
import os
import random
import time

import numpy as np
from ultralytics import YOLO
from settings import MODEL_NAME, MODEL_DIR, PRED_THRESHOLD

ROBOT_PANEL_W = 346
ROBOT_SPRITE_RECTS = (
    (80, 172, 854, 1182),
    (980, 172, 854, 1182),
    (1880, 172, 854, 1182),
)
RPS_NAMES = ("rock", "scissors", "paper")
YOLO_NAMES = ("paper", "rock", "scissors")
CLS_TO_MOVE = (2, 0, 1)

SCORE_PULSE_SEC = 1.8
COUNTDOWN_SEC = 3.0
SAMPLE_MS = 600
SAMPLE_MIN_MS = 150
SAMPLE_OTHER_MAX_MS = 50
RETRY_FEEDBACK_SEC = 5.0
ROUND_RESULT_SEC = 5.0
GAME_OVER_SEC = 10.0

RED = (0, 0, 255)
ORANGE = (0, 140, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)


def _conf_step(conf: float, floor: float) -> int:
    span = max(1e-6, 1.0 - floor)
    t = (conf - floor) / span
    if t >= 1.0:
        return 5
    return min(5, max(1, int(t * 5) + 1))


def _draw_conf_rect(frame, x1, y1, x2, y2, step: int):
    if step == 1:
        cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)
    elif step == 2:
        cv2.rectangle(frame, (x1, y1), (x2, y2), ORANGE, 2)
    elif step == 3:
        cv2.rectangle(frame, (x1, y1), (x2, y2), YELLOW, 2)
    elif step == 4:
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 4)


def _rps_beats(a: int, b: int) -> int:
    """0 tie, 1 if a wins, 2 if b wins. Order: rock -> scissors -> paper -> rock."""
    if a == b:
        return 0
    return 1 if (a + 1) % 3 == b else 2


def _sample_totals(log):
    totals = {0: 0.0, 1: 0.0, 2: 0.0}
    none_ms = 0.0
    for dt, cls in log:
        if cls is not None:
            totals[cls] += dt
        else:
            none_ms += dt
    return totals, none_ms


def _evaluate_sample(log):
    """Returns (move|None, totals, none_ms, fail_reason|None). log entries are move ids."""
    totals, none_ms = _sample_totals(log)
    best = max(totals, key=totals.get)
    if totals[best] < SAMPLE_MIN_MS:
        return None, totals, none_ms, (
            f"best {RPS_NAMES[best]} only {totals[best]:.0f}ms "
            f"(need >={SAMPLE_MIN_MS}ms)"
        )
    for cls, ms in totals.items():
        if cls != best and ms > SAMPLE_OTHER_MAX_MS:
            return None, totals, none_ms, (
                f"{RPS_NAMES[cls]} held {ms:.0f}ms "
                f"(max other {SAMPLE_OTHER_MAX_MS}ms)"
            )
    return best, totals, none_ms, None


def _format_sample_stats(totals, none_ms, log):
    elapsed = sum(dt for dt, _ in log)
    parts = [f"{RPS_NAMES[k]}={totals[k]:.0f}ms" for k in range(3)]
    parts.append(f"none={none_ms:.0f}ms")
    parts.append(f"elapsed={elapsed:.0f}ms")
    return ", ".join(parts)


class LivePredictor:
    def __init__(self, model_name=MODEL_NAME, camera_index=1, conf_threshold=PRED_THRESHOLD):
        self.camera_index = camera_index
        self.conf_threshold = conf_threshold
        self.cap = None
        self._mouse_x = -1
        self._mouse_y = -1
        self._mouse_click = False

        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(MODEL_DIR, MODEL_NAME, "weights", "best.pt")

        print(f"[INFO] Loading model from: {model_path}")
        self.model = YOLO(model_path)
        print("[INFO] Model loaded successfully")

        self._robot_sprites_raw = self._load_robot_sprites()
        self._robot_cache_h = None
        self._robot_cache = []
        self._reset_game()

    def _reset_game(self):
        self.phase = "intro"
        self.best_of = None
        self.user_score = 0
        self.robot_score = 0
        self.phase_t0 = time.time()
        self.countdown_elapsed = 0.0
        self.sample_log = []
        self.sample_t0 = 0.0
        self.last_frame_t = time.time()
        self.user_pick = None
        self.robot_pick = None
        self.round_msg = ""
        self.retry_stats = ""
        self.robot_sprite_idx = None
        self.show_detections = True

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_x, self._mouse_y = x, y
            self._mouse_click = True

    def _load_robot_sprites(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(
            script_dir, "..", "Gemini_Generated_Image_o3szmio3szmio3sz.png"
        )
        sheet = cv2.imread(path)
        if sheet is None:
            raise RuntimeError(f"Could not load robot sheet: {path}")
        return [sheet[y : y + h, x : x + w].copy() for x, y, w, h in ROBOT_SPRITE_RECTS]

    def _resize_robot_sprite(self, sprite, panel_h):
        sw, sh = sprite.shape[1], sprite.shape[0]
        scale = min(ROBOT_PANEL_W / sw, panel_h / sh)
        nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
        resized = cv2.resize(sprite, (nw, nh), interpolation=cv2.INTER_AREA)
        panel = np.full((panel_h, ROBOT_PANEL_W, 3), 32, dtype=np.uint8)
        ox, oy = (ROBOT_PANEL_W - nw) // 2, (panel_h - nh) // 2
        panel[oy : oy + nh, ox : ox + nw] = resized
        return panel

    def _robot_cache_for_height(self, panel_h):
        if panel_h != self._robot_cache_h:
            self._robot_cache = [
                self._resize_robot_sprite(s, panel_h) for s in self._robot_sprites_raw
            ]
            self._robot_cache_h = panel_h
        return self._robot_cache

    def _robot_panel(self, panel_h, visible=True, sprite_idx=None):
        if not visible:
            return np.full((panel_h, ROBOT_PANEL_W, 3), 32, dtype=np.uint8)
        cache = self._robot_cache_for_height(panel_h)
        idx = self.robot_sprite_idx if sprite_idx is None else sprite_idx
        if idx is None:
            return np.full((panel_h, ROBOT_PANEL_W, 3), 32, dtype=np.uint8)
        return cache[idx]

    def _compose_display(self, camera_frame, robot_visible=True, robot_idx=None):
        h, w = camera_frame.shape[:2]
        robot = self._robot_panel(h, visible=robot_visible, sprite_idx=robot_idx)
        out = np.zeros((h, w + ROBOT_PANEL_W, 3), dtype=np.uint8)
        out[:, :w] = camera_frame
        out[:, w:] = robot
        return out

    def _dominant_class(self, result):
        best_cls, best_conf = None, 0.0
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
            if conf > best_conf:
                best_conf = conf
                best_cls = int(box.cls[0])
        return best_cls

    def _intro_pick(self, key):
        if key in (ord("3"), ord("5"), ord("7")):
            return int(chr(key))
        if self._mouse_click:
            self._mouse_click = False
            cam_w = self._last_cam_w
            if self._mouse_x < cam_w:
                third = cam_w // 3
                if self._mouse_x < third:
                    return 3
                if self._mouse_x < 2 * third:
                    return 5
                return 7
        return None

    def _start_match(self, best_of):
        self.best_of = best_of
        self.user_score = 0
        self.robot_score = 0
        self._begin_round()
        print(f"[INFO] Best of {best_of} — first to {best_of // 2 + 1} wins")

    def _begin_round(self):
        self.phase = "score"
        self.phase_t0 = time.time()
        self.user_pick = None
        self.robot_pick = None
        self.robot_sprite_idx = None
        self.round_msg = ""

    def _begin_countdown(self):
        self.phase = "countdown"
        self.phase_t0 = time.time()
        self.countdown_elapsed = 0.0

    def _begin_sample(self):
        self.phase = "sample"
        self.sample_t0 = time.time()
        self.sample_log = []
        self.last_frame_t = self.sample_t0

    def _finish_sample(self):
        pick, totals, none_ms, fail_reason = _evaluate_sample(self.sample_log)
        stats = _format_sample_stats(totals, none_ms, self.sample_log)
        if pick is None:
            print(
                f"[INFO] No stable gesture — retrying round "
                f"({SAMPLE_MS}ms sample): {stats} — {fail_reason}"
            )
            self.round_msg = "Try again!"
            self.retry_stats = stats
            self.phase = "retry"
            self.phase_t0 = time.time()
            return
        self.user_pick = pick
        self.robot_pick = random.randint(0, 2)
        self.robot_sprite_idx = self.robot_pick
        outcome = _rps_beats(self.user_pick, self.robot_pick)
        if outcome == 1:
            self.user_score += 1
            self.round_msg = f"You win!"
        elif outcome == 2:
            self.robot_score += 1
            self.round_msg = f"Robot wins!"
        else:
            self.round_msg = "Tie"
        print(
            f"[INFO] You={RPS_NAMES[self.user_pick]} "
            f"Robot={RPS_NAMES[self.robot_pick]} — {self.round_msg}"
        )
        need = self.best_of // 2 + 1
        if self.user_score >= need or self.robot_score >= need:
            self.phase = "game_over"
            self.phase_t0 = time.time()
        else:
            self.phase = "round_result"
            self.phase_t0 = time.time()

    def _update_phase(self, result, frame_dt_ms):
        now = time.time()

        if self.phase == "score":
            if now - self.phase_t0 >= SCORE_PULSE_SEC:
                self._begin_countdown()

        elif self.phase == "countdown":
            self.countdown_elapsed = now - self.phase_t0
            if self.countdown_elapsed >= COUNTDOWN_SEC:
                self._begin_sample()

        elif self.phase == "sample":
            cls = self._dominant_class(result)
            move = CLS_TO_MOVE[cls] if cls is not None else None
            dt = (now - self.last_frame_t) * 1000
            self.last_frame_t = now
            self.sample_log.append((dt, move))
            if (now - self.sample_t0) * 1000 >= SAMPLE_MS:
                self._finish_sample()

        elif self.phase == "retry":
            if now - self.phase_t0 >= RETRY_FEEDBACK_SEC:
                self._begin_countdown()

        elif self.phase == "round_result":
            if now - self.phase_t0 >= ROUND_RESULT_SEC:
                self._begin_round()

        elif self.phase == "game_over":
            if now - self.phase_t0 >= GAME_OVER_SEC:
                self._reset_game()

    def _draw_intro(self, display):
        h, w = display.shape[:2]
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)
        cam_w = w - ROBOT_PANEL_W
        choices = ("3", "5", "7")
        for i, num in enumerate(choices):
            cx = int(cam_w * (i + 0.5) / 3)
            cy = h // 2
            scale = 12
            thick = 24
            (tw, th), _ = cv2.getTextSize(
                num, cv2.FONT_HERSHEY_DUPLEX, scale, thick
            )
            cv2.putText(
                display, num, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), thick,
            )
        #cv2.putText(
        #    display, "Best of 3 / 5 / 7  —  click or press 3, 5, 7",
        #    (cam_w // 2 - 420, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2,
        #)

    def _draw_score_pulse(self, frame):
        t = time.time() - self.phase_t0
        pulse = 0.75 + 0.25 * math.sin(t * 6.0)
        h, w = frame.shape[:2]
        text = f"{self.user_score}  -  {self.robot_score}"
        scale = 3.5 * pulse
        thick = max(4, int(8 * pulse))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
        color = (int(180 + 75 * pulse), int(30 + 75 * pulse), 255)
        cv2.putText(
            frame, text, (w // 2 - tw // 2, h // 2 + th // 2),
            cv2.FONT_HERSHEY_DUPLEX, scale, color, thick,
        )

    def _draw_countdown(self, frame):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        progress = min(1.0, self.countdown_elapsed / COUNTDOWN_SEC)
        remaining = max(0, 3 - int(self.countdown_elapsed))
        if self.countdown_elapsed >= COUNTDOWN_SEC:
            remaining = 0

        radius = min(w, h) // 6
        cv2.circle(frame, (cx, cy), radius, (70, 70, 70), 6)
        if progress > 0:
            end_angle = -90 + int(360 * progress)
            cv2.ellipse(
                frame, (cx, cy), (radius, radius), 0, -90, end_angle,
                (0, 220, 255), 6,
            )
        num = str(remaining)
        scale = 5.0
        thick = 12
        (tw, th), _ = cv2.getTextSize(num, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
        cv2.putText(
            frame, num, (cx - tw // 2, cy + th // 2),
            cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), thick,
        )

    def _draw_retry(self, frame):
        h, w = frame.shape[:2]
        cv2.putText(
            frame, "Try again!", (w // 2 - 140, h // 2 - 40),
            cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 200, 255), 3,
        )
        #stats = getattr(self, "retry_stats", "")
        #if stats:
        #    y = h // 2 + 20
        #    for i, line in enumerate((stats, f"need one >={SAMPLE_MIN_MS}ms, others <={SAMPLE_OTHER_MAX_MS}ms")):
        #        cv2.putText(
        #            frame, line, (20, y + i * 32),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1,
        #        )

    def _draw_round_result(self, frame):
        h, w = frame.shape[:2]
        if self.user_pick is not None:
            cv2.putText(
                frame, f"{RPS_NAMES[self.user_pick]}",
                (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
            )
        if self.round_msg:
            (tw, th), _ = cv2.getTextSize(
                self.round_msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2
            )
            cv2.putText(
                frame, self.round_msg, (w // 2 - tw // 2, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 180), 2,
            )

    def _draw_game_over(self, frame):
        need = self.best_of // 2 + 1
        if self.user_score >= need:
            msg = "You win the match!"
        else:
            msg = "Robot wins the match!"
        h, w = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.8, 4)
        cv2.putText(
            frame, msg, (w // 2 - tw // 2, h // 2),
            cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 255, 255), 4,
        )
        cv2.putText(
            frame, f"Final {self.user_score} - {self.robot_score}",
            (w // 2 - 120, h // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2,
        )

    def _draw_game_hud(self, frame):
        if self.best_of is None:
            return
        cv2.putText(
            frame, f"Best of {self.best_of}  |  {self.user_score}-{self.robot_score}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3
        )
        cv2.putText(
            frame, f"Best of {self.best_of}  |  {self.user_score}-{self.robot_score}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1
        )
        if self.phase == "sample":
            cv2.putText(
                frame, "Sampling...",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )
        elif self.round_msg and self.phase == "score":
            cv2.putText(
                frame, self.round_msg, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1,
            )

    def open_camera(self):
        print("[INFO] Opening camera...")
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera (index={self.camera_index}). Try index 1."
            )
        print("[INFO] Camera opened successfully")

    def draw_overlay(self, frame, result, show_boxes=True):
        boxes = result.boxes

        if show_boxes:
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue
                cls_id = int(box.cls[0])
                label = YOLO_NAMES[cls_id] if cls_id < 3 else self.model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                step = _conf_step(conf, self.conf_threshold)
                _draw_conf_rect(frame, x1, y1, x2, y2, step)
                label_colors = (
                    (RED, (255, 255, 255)),
                    (ORANGE, (0, 0, 0)),
                    (YELLOW, (0, 0, 0)),
                    (GREEN, (0, 0, 0)),
                    (GREEN, (0, 0, 0)),
                )
                bar_bgr, txt_bgr = label_colors[step - 1]
                text = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), bar_bgr, -1)
                cv2.putText(
                    frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_bgr, 1,
                )

        self._draw_game_hud(frame)
        if self.phase == "intro":
            cv2.putText(
                frame, f"Conf: {self.conf_threshold:.2f}  (+/-)  |  q quit",
                (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1,
            )
        return frame

    def run(self):
        try:
            self.open_camera()
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            return

        win = "YOLOv8 Live Prediction"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self._on_mouse)

        print("[INFO] Rock-paper-scissors. Pick 3, 5, or 7 to start. q/ESC quit.")

        prev_t = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Failed to read frame")
                break

            now = time.time()
            frame_dt_ms = (now - prev_t) * 1000
            prev_t = now
            self._last_cam_w = frame.shape[1]

            results = self.model.predict(
                frame, conf=self.conf_threshold, verbose=False,
            )
            show_boxes = self.phase != "retry"
            annotated = self.draw_overlay(frame.copy(), results[0], show_boxes=show_boxes)

            if self.phase != "intro":
                self._update_phase(results[0], frame_dt_ms)

            if self.phase == "score":
                self._draw_score_pulse(annotated)
            elif self.phase == "retry":
                self._draw_retry(annotated)
            elif self.phase == "countdown":
                self._draw_countdown(annotated)
            elif self.phase == "round_result":
                self._draw_round_result(annotated)
            elif self.phase == "game_over":
                self._draw_game_over(annotated)

            robot_visible = self.phase in ("intro", "round_result", "game_over")
            display = self._compose_display(
                annotated, robot_visible=robot_visible, robot_idx=self.robot_sprite_idx,
            )
            if self.phase == "intro":
                self._draw_intro(display)

            cv2.imshow(win, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("[INFO] Exiting...")
                break
            if self.phase == "intro":
                pick = self._intro_pick(key)
                if pick in (3, 5, 7):
                    self._start_match(pick)
            elif key in (ord("+"), ord("=")):
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
