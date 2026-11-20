"""Automated, objective benchmarking script for See2Speak.

Features:
  1. Auto-screenshots: Automatically saves highest-confidence annotated detection images to docs/screenshots/.
  2. Manual snapshot: Press [S] or click [SNAPSHOT] at any time to capture the current screen.
  3. Smooth 30+ FPS camera feed: Asynchronous background YOLO inference prevents UI stutter/lag.
  4. Enlarged, high-visibility 1280x720 display canvas with large text and buttons.
  5. Manual 'START' gate: Doesn't rush into capture until you click [START] or press Space/Enter.
  6. Bounding boxes are live during setup so you can position the object with confidence.
  7. Automatic capture: Once started, captures trials automatically every 2.0s without human bias.
  8. Working PAUSE / RESUME button (clickable or via Spacebar / P).
  9. Dedicated 'NEXT CLASS' button (clickable or via N) so you can prepare the next object at your own pace.
  10. Visual freeze confirmation (0.8s) displaying the exact recorded bounding box and HIT/MISS outcome.
  11. Automated calculation of Precision, Recall, F1-Score, Accuracy, and Latency.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config
from backend.detection import Detector

CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
SCREENSHOTS_DIR = ROOT / "docs" / "screenshots"


def print_banner(text: str):
    print("\n" + "=" * 72)
    print(f" {text}")
    print("=" * 72)


class AsyncInferenceWorker:
    """Runs YOLO-World preview detections in a background thread.
    This decouples camera capture (30+ FPS) from model inference (5-8 FPS)
    so the live video never lags or stutters.
    """

    def __init__(self, detector: Detector):
        self.detector = detector
        self._lock = threading.Lock()
        self._latest_frame = None
        self._current_dets: list[dict] = []
        self._current_best_rej: dict | None = None
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def update_frame(self, frame):
        with self._lock:
            self._latest_frame = frame

    def get_results(self) -> tuple[list[dict], dict | None]:
        with self._lock:
            return list(self._current_dets), self._current_best_rej

    def _worker_loop(self):
        while self._running:
            target_frame = None
            with self._lock:
                if self._latest_frame is not None:
                    target_frame = self._latest_frame
                    self._latest_frame = None

            if target_frame is not None:
                h, w = target_frame.shape[:2]
                if w > 640:
                    scale = 640.0 / w
                    small = cv2.resize(target_frame, (640, int(h * scale)), interpolation=cv2.INTER_LINEAR)
                else:
                    small = target_frame

                ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    res = self.detector.detect(enc.tobytes())
                    with self._lock:
                        self._current_dets = res.get("detections", [])
                        self._current_best_rej = res.get("best_rejected")
            else:
                time.sleep(0.015)

    def stop(self):
        self._running = False


class GuiButton:
    """Clickable and renderable on-screen UI button for OpenCV windows."""

    def __init__(self, key_id: str, rect: tuple[int, int, int, int], text: str, bg_color: tuple[int, int, int]):
        self.key_id = key_id
        self.rect = rect  # (x1, y1, x2, y2)
        self.text = text
        self.bg_color = bg_color

    def contains(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def draw(self, img, is_active: bool = True):
        x1, y1, x2, y2 = self.rect
        color = self.bg_color if is_active else (60, 60, 60)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

        font_scale = 0.62
        thickness = 2
        (tw, th), _ = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx = x1 + (x2 - x1 - tw) // 2
        ty = y1 + (y2 - y1 + th) // 2
        cv2.putText(img, self.text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)


def draw_bounding_boxes(frame, detections, target_key: str | None = None):
    h, w = frame.shape[:2]
    for box_info in detections:
        b = box_info["box"]
        x1 = int(b["x"] * w)
        y1 = int(b["y"] * h)
        x2 = int((b["x"] + b["w"]) * w)
        y2 = int((b["y"] + b["h"]) * h)

        if target_key is None:
            color = (0, 0, 255)  # Negative control unexpected
            tag = f"UNEXPECTED: {box_info['label']} {int(box_info['confidence'] * 100)}%"
        else:
            is_match = (box_info["key"] == target_key)
            color = (0, 255, 0) if is_match else (0, 140, 255)
            tag = f"{box_info['label']} {int(box_info['confidence'] * 100)}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        font_scale = 0.70
        thickness = 2
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 12)), (x1 + tw + 10, y1), color, -1)
        cv2.putText(
            frame,
            tag,
            (x1 + 5, max(th + 5, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0) if color == (0, 255, 0) else (255, 255, 255),
            thickness,
        )


def run_benchmark(trials_per_class: int = 10, interval_sec: float = 2.0):
    window_name = "See2Speak Benchmark (High-Speed View)"
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    print_banner("See2Speak Objective Benchmark Tool")
    print(f"Active vocabulary: {[w.label for w in config.VOCABULARY]}")
    print(f"Configured detection threshold: {config.DETECT_MIN_CONF:.2f}")
    print(f"Trials per object: {trials_per_class} | Capture interval: {interval_sec:.1f}s")
    print(f"Screenshots saved to: {SCREENSHOTS_DIR.relative_to(ROOT)}")
    print("\nControls:")
    print("  - Click [START] button or press [SPACE] / [ENTER] to start auto trials when ready.")
    print("  - Click [SNAPSHOT] button or press [S] to save an instant screenshot for your report.")
    print("  - Click [PAUSE] button or press [SPACE] / [P] to pause/resume anytime.")
    print("  - Click [NEXT CLASS] button or press [N] to move to the next object.")
    print("  - Click [QUIT] button or press [Q] to finish early and show summary table.\n")

    print("[..] Initializing YOLO-World detector...")
    detector = Detector()
    t0 = time.perf_counter()
    detector.load()
    print(f"[ok] Detector ready in {time.perf_counter() - t0:.2f}s")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Error: Could not open camera (index 0).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CANVAS_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CANVAS_HEIGHT)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CANVAS_WIDTH, CANVAS_HEIGHT)

    worker = AsyncInferenceWorker(detector)

    clicked_action: str | None = None

    def on_mouse(event, x, y, flags, param):
        nonlocal clicked_action
        if event == cv2.EVENT_LBUTTONDOWN:
            for btn in active_buttons:
                if btn.contains(x, y):
                    clicked_action = btn.key_id
                    break

    cv2.setMouseCallback(window_name, on_mouse)

    stats = {
        w.key: {
            "label": w.label,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "conf_sum": 0.0,
            "latencies": [],
            "trials": 0,
        }
        for w in config.VOCABULARY
    }
    negatives = {"trials": 0, "tn": 0, "fp": 0, "latencies": []}
    saved_screenshots: dict[str, str] = {}

    quit_all = False
    toast_msg = ""
    toast_until = 0.0
    active_buttons: list[GuiButton] = []

    phases: list[dict] = []
    for w in config.VOCABULARY:
        phases.append({"type": "target", "key": w.key, "label": w.label, "target_trials": trials_per_class})
    phases.append({"type": "negative", "key": "negative_control", "label": "Negative Control", "target_trials": trials_per_class})

    phase_idx = 0
    while phase_idx < len(phases) and not quit_all:
        phase = phases[phase_idx]
        is_negative = (phase["type"] == "negative")
        target_key = phase["key"]
        target_label = phase["label"]
        needed_trials = phase["target_trials"]
        best_conf_for_phase = 0.0

        print_banner(f"PHASE: {target_label.upper()} ({needed_trials} trials)")
        if is_negative:
            print("  -> Hold up unsupported items (mug, wallet, keys, empty hand).")
        else:
            print(f"  -> Position the {target_label} in view. Click [START] when ready.")

        state = "WAITING"
        trial_idx = 0
        timer_start = 0.0
        remaining_in_interval = interval_sec
        flash_start = 0.0
        flash_frame = None
        flash_border_color = (0, 255, 0)
        flash_headline = ""
        flash_subline = ""

        while not quit_all:
            ret, raw_frame = cap.read()
            if not ret:
                break

            if raw_frame.shape[1] != CANVAS_WIDTH or raw_frame.shape[0] != CANVAS_HEIGHT:
                frame = cv2.resize(raw_frame, (CANVAS_WIDTH, CANVAS_HEIGHT), interpolation=cv2.INTER_LINEAR)
            else:
                frame = raw_frame

            h, w = frame.shape[:2]
            now = time.perf_counter()

            # 4-button responsive dock layout
            btn_h = 54
            btn_y1 = h - 72
            btn_y2 = btn_y1 + btn_h

            btn_width = 220
            gap = 20
            total_btn_span = (btn_width * 4) + (gap * 3)
            start_x = (w - total_btn_span) // 2

            b1_x1 = start_x
            b1_x2 = b1_x1 + btn_width

            b2_x1 = b1_x2 + gap
            b2_x2 = b2_x1 + btn_width

            b3_x1 = b2_x2 + gap
            b3_x2 = b3_x1 + btn_width

            b4_x1 = b3_x2 + gap
            b4_x2 = b4_x1 + btn_width

            # Configure buttons
            if state == "WAITING":
                btn1 = GuiButton("TOGGLE_RUN", (b1_x1, btn_y1, b1_x2, btn_y2), "START (Space)", (40, 160, 40))
            elif state == "RUNNING":
                btn1 = GuiButton("TOGGLE_RUN", (b1_x1, btn_y1, b1_x2, btn_y2), "PAUSE (Space)", (0, 140, 255))
            elif state == "PAUSED":
                btn1 = GuiButton("TOGGLE_RUN", (b1_x1, btn_y1, b1_x2, btn_y2), "RESUME (Space)", (40, 160, 40))
            elif state == "DONE":
                btn1 = GuiButton("NEXT_CLASS", (b1_x1, btn_y1, b1_x2, btn_y2), "NEXT CLASS (Enter)", (200, 120, 20))
            else:
                btn1 = GuiButton("TOGGLE_RUN", (b1_x1, btn_y1, b1_x2, btn_y2), "LOGGED...", (70, 70, 70))

            btn2 = GuiButton("SNAPSHOT", (b2_x1, btn_y1, b2_x2, btn_y2), "SNAPSHOT (S)", (140, 70, 20))
            btn3 = GuiButton("NEXT_CLASS", (b3_x1, btn_y1, b3_x2, btn_y2), "NEXT CLASS (N)", (160, 90, 40))
            btn4 = GuiButton("QUIT", (b4_x1, btn_y1, b4_x2, btn_y2), "QUIT (Q)", (40, 40, 180))

            active_buttons = [btn1, btn2, btn3, btn4]

            if state != "FLASH":
                worker.update_frame(frame)
            current_dets, best_rej = worker.get_results()

            if state == "FLASH":
                display = flash_frame.copy() if flash_frame is not None else frame.copy()
                cv2.rectangle(display, (0, 0), (w, h), flash_border_color, 10)
                cv2.rectangle(display, (0, 0), (w, 105), (15, 15, 15), -1)
                cv2.putText(display, flash_headline, (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, flash_border_color, 3)
                cv2.putText(display, flash_subline, (30, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                if now - flash_start >= 0.8:
                    trial_idx += 1
                    if trial_idx >= needed_trials:
                        state = "DONE"
                        print(f"  [✓] All {needed_trials} trials completed for {target_label}!")
                    else:
                        state = "RUNNING"
                        timer_start = time.perf_counter()
                        remaining_in_interval = interval_sec
            else:
                display = frame.copy()
                draw_bounding_boxes(display, current_dets, None if is_negative else target_key)

                cv2.rectangle(display, (0, 0), (w, 105), (20, 20, 20), -1)

                if state == "WAITING":
                    title_text = f"PREPARE: {target_label.upper()}"
                    sub_text = "Position object in view. Click [START] or press [SPACE] when ready!"
                    title_color = (0, 215, 255)
                    sub_color = (255, 255, 255)
                elif state == "RUNNING":
                    time_in_interval = now - timer_start
                    remaining_in_interval = max(0.0, interval_sec - time_in_interval)
                    title_text = f"TARGET: {target_label.upper()} | Trial {trial_idx + 1}/{needed_trials}"
                    sub_text = f"Auto-capturing in {remaining_in_interval:.1f}s... (Click [PAUSE] or press Space)"
                    title_color = (0, 255, 0) if not is_negative else (0, 165, 255)
                    sub_color = (0, 255, 255)

                    pct = 1.0 - (remaining_in_interval / interval_sec) if interval_sec > 0 else 1.0
                    cv2.rectangle(display, (0, 105), (int(w * pct), 112), (0, 255, 0), -1)

                elif state == "PAUSED":
                    title_text = f"TARGET: {target_label.upper()} | Trial {trial_idx + 1}/{needed_trials} [PAUSED]"
                    sub_text = f"PAUSED at {remaining_in_interval:.1f}s — Click [RESUME] or press [SPACE]"
                    title_color = (0, 165, 255)
                    sub_color = (255, 255, 255)
                elif state == "DONE":
                    title_text = f"COMPLETED: {target_label.upper()} ({needed_trials}/{needed_trials} recorded)"
                    sub_text = "Prepare next object! Click [NEXT CLASS] or press [ENTER/SPACE]"
                    title_color = (0, 255, 0)
                    sub_color = (255, 255, 255)

                cv2.putText(display, title_text, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, title_color, 3)
                cv2.putText(display, sub_text, (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, sub_color, 2)

            # On-screen toast banner if snapshot taken
            if now < toast_until:
                cv2.rectangle(display, (w // 2 - 250, 120), (w // 2 + 250, 165), (0, 140, 0), -1)
                cv2.rectangle(display, (w // 2 - 250, 120), (w // 2 + 250, 165), (255, 255, 255), 2)
                cv2.putText(display, toast_msg, (w // 2 - 230, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.rectangle(display, (0, h - 90), (w, h), (20, 20, 20), -1)
            for btn in active_buttons:
                btn.draw(display)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(2) & 0xFF
            action = clicked_action
            clicked_action = None

            if key == ord("q"):
                action = "QUIT"
            elif key == ord("n"):
                action = "NEXT_CLASS"
            elif key in (ord("s"), ord("S")):
                action = "SNAPSHOT"
            elif key in (32, 13, ord("p")):
                if state == "DONE":
                    action = "NEXT_CLASS"
                else:
                    action = "TOGGLE_RUN"

            if action == "QUIT":
                quit_all = True
                break

            if action == "NEXT_CLASS":
                print(f"  -> Moving to next phase from {target_label}.")
                break

            if action == "SNAPSHOT":
                snap_name = f"manual_{target_key}_{int(time.time())}.png"
                snap_path = SCREENSHOTS_DIR / snap_name
                cv2.imwrite(str(snap_path), display)
                toast_msg = f"[SNAPSHOT SAVED] {snap_name}"
                toast_until = now + 1.5
                print(f"  [📸] Saved manual screenshot to: {snap_path.relative_to(ROOT)}")
                continue

            if action == "TOGGLE_RUN":
                if state == "WAITING":
                    state = "RUNNING"
                    timer_start = time.perf_counter()
                    remaining_in_interval = interval_sec
                    print(f"  [*] Started automated testing for {target_label}!")
                elif state == "RUNNING":
                    state = "PAUSED"
                    time_in_interval = now - timer_start
                    remaining_in_interval = max(0.0, interval_sec - time_in_interval)
                    print(f"  [||] Paused ({remaining_in_interval:.1f}s remaining).")
                elif state == "PAUSED":
                    state = "RUNNING"
                    timer_start = time.perf_counter() - (interval_sec - remaining_in_interval)
                    print("  [>] Resumed automated testing.")

            if state == "RUNNING" and remaining_in_interval <= 0.0:
                eval_frame = frame.copy()
                ok, enc = cv2.imencode(".jpg", eval_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    timer_start = time.perf_counter()
                    continue

                t_inf = time.perf_counter()
                res = detector.detect(enc.tobytes())
                elapsed_ms = (time.perf_counter() - t_inf) * 1000

                dets = res.get("detections", [])
                detected_keys = [d["key"] for d in dets]
                eval_best_rej = res.get("best_rejected")

                flash_frame = eval_frame.copy()

                if is_negative:
                    negatives["trials"] += 1
                    negatives["latencies"].append(elapsed_ms)
                    draw_bounding_boxes(flash_frame, dets, None)
                    if len(dets) == 0:
                        negatives["tn"] += 1
                        flash_border_color = (0, 255, 0)
                        flash_headline = f"NEGATIVE TRIAL {trial_idx + 1}: PASSED"
                        flash_subline = f"True Negative (0 boxes) | Latency: {elapsed_ms:.1f}ms"
                        print(f"  [Negative Trial {trial_idx + 1}] PASSED (True Negative - 0 boxes)")

                        # Auto-save negative control screenshot
                        if "Negative Control" not in saved_screenshots:
                            nc_path = SCREENSHOTS_DIR / "detection_negative_control.png"
                            cv2.imwrite(str(nc_path), flash_frame)
                            saved_screenshots["Negative Control"] = str(nc_path.relative_to(ROOT))
                    else:
                        negatives["fp"] += 1
                        flash_border_color = (0, 0, 255)
                        flash_headline = f"NEGATIVE TRIAL {trial_idx + 1}: FALSE POSITIVE"
                        flash_subline = f"Spurious: {[d['label'] for d in dets]} | Latency: {elapsed_ms:.1f}ms"
                        print(f"  [Negative Trial {trial_idx + 1}] FALSE POSITIVE -> {[d['label'] for d in dets]}")
                else:
                    stats[target_key]["trials"] += 1
                    stats[target_key]["latencies"].append(elapsed_ms)
                    draw_bounding_boxes(flash_frame, dets, target_key)

                    is_hit = (target_key in detected_keys)
                    if is_hit:
                        stats[target_key]["tp"] += 1
                        conf = next(d["confidence"] for d in dets if d["key"] == target_key)
                        stats[target_key]["conf_sum"] += conf
                        flash_border_color = (0, 255, 0)
                        flash_headline = f"TRIAL {trial_idx + 1}: HIT! ({target_label} {int(conf * 100)}%)"
                        flash_subline = f"Latency: {elapsed_ms:.1f}ms | Confidence: {conf:.2f}"
                        print(f"  [{target_label} Trial {trial_idx + 1}] HIT  (Confidence: {conf:.2f}, Latency: {elapsed_ms:.1f}ms)")

                        # Auto-save best hit screenshot for this class
                        if conf > best_conf_for_phase:
                            best_conf_for_phase = conf
                            shot_file = SCREENSHOTS_DIR / f"detection_{target_key}.png"
                            cv2.imwrite(str(shot_file), flash_frame)
                            saved_screenshots[target_label] = str(shot_file.relative_to(ROOT))
                    else:
                        stats[target_key]["fn"] += 1
                        diag = f"best rejected: {eval_best_rej['label']} ({eval_best_rej['confidence']:.2f})" if eval_best_rej else "no boxes"
                        flash_border_color = (0, 0, 255)
                        flash_headline = f"TRIAL {trial_idx + 1}: MISS"
                        flash_subline = f"Below {config.DETECT_MIN_CONF:.2f} threshold ({diag})"
                        print(f"  [{target_label} Trial {trial_idx + 1}] MISS ({diag})")

                    for d in dets:
                        if d["key"] != target_key and d["key"] in stats:
                            stats[d["key"]]["fp"] += 1
                            print(f"    -> Spurious detection: {d['label']} ({d['confidence']:.2f})")

                state = "FLASH"
                flash_start = time.perf_counter()

        phase_idx += 1

    worker.stop()
    cap.release()
    cv2.destroyAllWindows()
    generate_summary_table(stats, negatives, saved_screenshots)


def generate_summary_table(stats: dict, negatives: dict, saved_screenshots: dict[str, str]):
    print_banner("OBJECTIVE BENCHMARK RESULTS (AUTOMATED CAPTURE)")
    print(f"{'Target Object':<16} | {'Trials':<6} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'Accuracy':<8} | {'Avg Conf':<8} | {'Latency':<8}")
    print("-" * 92)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_trials = 0
    all_latencies = []

    for key, data in stats.items():
        tp = data["tp"]
        fp = data["fp"]
        fn = data["fn"]
        trials = data["trials"]

        prec = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        acc = (tp / trials) * 100 if trials > 0 else 0.0
        avg_conf = (data["conf_sum"] / tp) if tp > 0 else 0.0
        avg_lat = (sum(data["latencies"]) / len(data["latencies"])) if data["latencies"] else 0.0

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_trials += trials
        all_latencies.extend(data["latencies"])

        print(f"{data['label']:<16} | {trials:<6} | {prec:>8.1f}% | {rec:>7.1f}% | {f1:>7.1f}% | {acc:>7.1f}% | {avg_conf:>8.2f} | {avg_lat:>6.1f}ms")

    # Negative Control row directly in the table
    neg_trials = negatives["trials"]
    if neg_trials > 0:
        neg_acc = (negatives["tn"] / neg_trials) * 100
        neg_lat = (sum(negatives["latencies"]) / len(negatives["latencies"])) if negatives["latencies"] else 0.0
        all_latencies.extend(negatives["latencies"])
        print(f"{'Negative Control':<16} | {neg_trials:<6} | {'--':>9} | {'--':>8} | {'--':>8} | {neg_acc:>7.1f}% | {'--':>8} | {neg_lat:>6.1f}ms")

    print("-" * 92)
    overall_prec = (total_tp / (total_tp + total_fp)) * 100 if (total_tp + total_fp) > 0 else 0.0
    overall_rec = (total_tp / (total_tp + total_fn)) * 100 if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (2 * overall_prec * overall_rec / (overall_prec + overall_rec)) if (overall_prec + overall_rec) > 0 else 0.0

    grand_trials = total_trials + neg_trials
    grand_acc = ((total_tp + negatives["tn"]) / grand_trials) * 100 if grand_trials > 0 else 0.0
    overall_lat = (sum(all_latencies) / len(all_latencies)) if all_latencies else 0.0

    print(f"{'OVERALL MEAN':<16} | {grand_trials:<6} | {overall_prec:>8.1f}% | {overall_rec:>7.1f}% | {overall_f1:>7.1f}% | {grand_acc:>7.1f}% | {'--':>8} | {overall_lat:>6.1f}ms")
    print(f"\nNegative Control Specificity: {negatives['tn']}/{neg_trials} correctly rejected ({neg_acc if neg_trials > 0 else 0.0:.1f}%).")
    print("Benchmark complete.\n")

    if saved_screenshots:
        print("=" * 92)
        print(" SAVED BENCHMARK SCREENSHOTS (READY FOR REPORT)")
        print("=" * 92)
        for label, rel_path in saved_screenshots.items():
            print(f"  ✓ {label:<18} -> {rel_path}")
        print("=" * 92 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="See2Speak Objective Benchmark Tool")
    parser.add_argument("--trials", "--samples", "-n", dest="trials", type=int, default=10, help="Trials per class (default: 10)")
    parser.add_argument("--interval", "-i", dest="interval", type=float, default=2.0, help="Interval in seconds between automatic captures (default: 2.0s)")
    args = parser.parse_args()

    run_benchmark(trials_per_class=args.trials, interval_sec=args.interval)
