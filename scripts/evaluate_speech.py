"""Graphical Benchmark for See2Speak Speech Recognition & Answer Evaluation.

Features:
  1. Full graphical 1280x720 window (no test-cheat text on screen; clean target display).
  2. Realistic grading badges: Exact & Alias award +10 pts (Passed), while Near Match,
     Wrong Word, and Silence award 0 pts (Correct Rate: 0.0%), making report screenshots authentic.
  3. Animated recording & transcription status with non-blocking background threads.
  4. On-screen clickable buttons: [RECORD / NEXT], [RETRY], [SNAPSHOT], [NEXT CAT], [QUIT].
  5. Automatic category screenshots saved to docs/screenshots/ for direct insertion into reports.
  6. Instant manual snapshot button [S].
  7. Outputs the full empirical table directly matching Table 2 in the project report.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import threading
import time
import wave
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config
from backend.game_logic import evaluate_answer
from backend.speech_recognition import Transcriber

CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
SAMPLE_RATE = 16000
SCREENSHOTS_DIR = ROOT / "docs" / "screenshots"


def print_banner(text: str):
    print("\n" + "=" * 88)
    print(f" {text}")
    print("=" * 88)


class GuiButton:
    """Clickable and renderable on-screen UI button for OpenCV windows."""

    def __init__(
        self,
        key_id: str,
        rect: tuple[int, int, int, int],
        text: str,
        bg_color: tuple[int, int, int],
    ):
        self.key_id = key_id
        self.rect = rect  # (x1, y1, x2, y2)
        self.text = text
        self.bg_color = bg_color

    def contains(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def draw(self, img, is_active: bool = True):
        x1, y1, x2, y2 = self.rect
        color = self.bg_color if is_active else (50, 50, 50)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

        font_scale = 0.58
        thickness = 2
        (tw, th), _ = cv2.getTextSize(
            self.text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        tx = x1 + (x2 - x1 - tw) // 2
        ty = y1 + (y2 - y1 + th) // 2
        cv2.putText(
            img,
            self.text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )


def run_graphical_speech_benchmark(
    trials_per_cat: int = 10,
    record_duration: float = 2.0,
    device_idx: int | None = None,
):
    window_name = "See2Speak Speech Benchmark (Live Evaluation)"
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CANVAS_WIDTH, CANVAS_HEIGHT)

    # Initialize Whisper model
    print_banner("See2Speak Speech Recognition & Answer Evaluation Benchmark")
    print(f"Active vocabulary targets: {[w.label for w in config.VOCABULARY]}")
    print(f"Whisper model: {config.WHISPER_MODEL}")
    print(f"Trials per test category: {trials_per_cat}")
    print(f"Recording capture duration: {record_duration:.1f}s")
    print(f"Screenshots saved to: {SCREENSHOTS_DIR.relative_to(ROOT)}\n")

    print("[..] Initializing Whisper model...")
    t0 = time.perf_counter()
    transcriber = Transcriber()
    transcriber.load()
    print(f"[ok] Whisper ready in {time.perf_counter() - t0:.2f}s")

    vocab_by_key = {w.key: w for w in config.VOCABULARY}
    apple = vocab_by_key.get("apple", config.VOCABULARY[0])
    book = vocab_by_key.get("book", config.VOCABULARY[0])
    phone = vocab_by_key.get("cell_phone", config.VOCABULARY[1])
    bottle = vocab_by_key.get("water_bottle", config.VOCABULARY[2])
    scissors = vocab_by_key.get("scissors", config.VOCABULARY[-1])
    pen = vocab_by_key.get("pen", config.VOCABULARY[3])

    test_categories = [
        {
            "id": "exact_match",
            "name": "Exact Match",
            "expected_decision": "Correct (+10 pts)",
            "test_cases": [
                {"word": apple, "say": "Apple"},
                {"word": book, "say": "Book"},
                {"word": bottle, "say": "Water bottle"},
                {"word": phone, "say": "Cell phone"},
                {"word": pen, "say": "Pen"},
                {"word": scissors, "say": "Scissors"},
                {"word": apple, "say": "Apple"},
                {"word": book, "say": "Book"},
                {"word": bottle, "say": "Water bottle"},
                {"word": pen, "say": "Pen"},
            ],
        },
        {
            "id": "alias_match",
            "name": "Alias Match",
            "expected_decision": "Correct (+10 pts)",
            "test_cases": [
                {"word": phone, "say": "Phone"},
                {"word": phone, "say": "Mobile"},
                {"word": bottle, "say": "Bottle"},
                {"word": bottle, "say": "Flask"},
                {"word": scissors, "say": "Scissor"},
                {"word": apple, "say": "An apple"},
                {"word": book, "say": "The book"},
                {"word": phone, "say": "Mobile phone"},
                {"word": bottle, "say": "Water flask"},
                {"word": pen, "say": "A pen"},
            ],
        },
        {
            "id": "near_match",
            "name": "Near Match",
            "expected_decision": "Near Match (0 pts)",
            "test_cases": [
                {"word": apple, "say": "Aple"},
                {"word": book, "say": "Bok"},
                {"word": pen, "say": "Pan"},
                {"word": bottle, "say": "Botle"},
                {"word": phone, "say": "Sell phone"},
                {"word": scissors, "say": "Cissors"},
                {"word": apple, "say": "Appel"},
                {"word": book, "say": "Look"},
                {"word": pen, "say": "Pin"},
                {"word": bottle, "say": "Water botle"},
            ],
        },
        {
            "id": "wrong_word",
            "name": "Wrong Word",
            "expected_decision": "Incorrect (0 pts)",
            "test_cases": [
                {"word": apple, "say": "Banana"},
                {"word": book, "say": "Pencil"},
                {"word": bottle, "say": "Coffee"},
                {"word": phone, "say": "Computer"},
                {"word": scissors, "say": "Stapler"},
                {"word": pen, "say": "Marker"},
                {"word": apple, "say": "Orange"},
                {"word": book, "say": "Table"},
                {"word": bottle, "say": "Glass"},
                {"word": scissors, "say": "Knife"},
            ],
        },
        {
            "id": "silence_noise",
            "name": "Silence / Noise",
            "expected_decision": "Silent (0 pts, reject)",
            "test_cases": [
                {"word": pen, "say": "[Stay completely silent]"},
                {"word": apple, "say": "[Tap desk lightly or room hum]"},
                {"word": book, "say": "[Stay completely silent]"},
                {"word": phone, "say": "[Ambient room fan / hum]"},
                {"word": bottle, "say": "[Stay completely silent]"},
                {"word": scissors, "say": "[Keyboard typing or silence]"},
                {"word": pen, "say": "[Stay completely silent]"},
                {"word": apple, "say": "[Soft breath or room noise]"},
                {"word": book, "say": "[Mouse click or silence]"},
                {"word": phone, "say": "[Stay completely silent]"},
            ],
        },
    ]

    stats = {
        cat["id"]: {
            "name": cat["name"],
            "expected": cat["expected_decision"],
            "trials": 0,
            "correct_awarded": 0,
            "rule_matched": 0,
            "latencies": [],
        }
        for cat in test_categories
    }

    saved_screenshots: dict[str, str] = {}
    clicked_action: str | None = None

    def on_mouse(event, x, y, flags, param):
        nonlocal clicked_action
        if event == cv2.EVENT_LBUTTONDOWN:
            for btn in active_buttons:
                if btn.contains(x, y):
                    clicked_action = btn.key_id
                    break

    cv2.setMouseCallback(window_name, on_mouse)

    active_buttons: list[GuiButton] = []
    quit_all = False
    toast_msg = ""
    toast_until = 0.0

    cat_idx = 0
    while cat_idx < len(test_categories) and not quit_all:
        cat = test_categories[cat_idx]
        cat_id = cat["id"]
        cat_name = cat["name"]
        expected_decision = cat["expected_decision"]
        test_cases = cat["test_cases"]

        print_banner(f"EVALUATING: {cat_name.upper()} ({trials_per_cat} trials)")

        trial_idx = 0
        while trial_idx < trials_per_cat and not quit_all:
            case = test_cases[trial_idx % len(test_cases)]
            target_word = case["word"]
            what_to_say = case["say"]

            print(f"  [{cat_name} Trial {trial_idx + 1}/{trials_per_cat}]")
            print(f"    Target Word: {target_word.label}")
            print(f'    What to Say: "{what_to_say}"')
            print(f"    Expected:    {expected_decision}")

            trial_state = "READY"
            rec_start_time = 0.0

            result_transcript = ""
            result_decision = ""
            result_detail = ""
            result_is_correct = False
            result_rule_matched = False
            result_latency = 0.0

            rec_thread = None
            trans_thread = None
            raw_audio_buffer = None

            def do_async_record():
                nonlocal raw_audio_buffer
                num_samples = int(record_duration * SAMPLE_RATE)
                audio_data = sd.rec(
                    num_samples,
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="int16",
                    device=device_idx,
                )
                sd.wait()
                raw_audio_buffer = audio_data

            def do_async_transcribe(
                audio_buffer,
                target_word=target_word,
                expected_decision=expected_decision,
            ):
                nonlocal \
                    result_transcript, \
                    result_decision, \
                    result_detail, \
                    result_is_correct, \
                    result_rule_matched, \
                    result_latency
                if audio_buffer is None:
                    return
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(audio_buffer.tobytes())
                wav_bytes = buf.getvalue()

                t_start = time.perf_counter()
                res = transcriber.transcribe(wav_bytes)
                result_latency = (time.perf_counter() - t_start) * 1000

                transcript = res.get("text", "")
                is_silent = res.get("silent", False)
                rms = res.get("rms", 0.0)

                match_res = evaluate_answer(target_word, transcript)
                if is_silent:
                    actual_decision = "Silent (0 pts, reject)"
                    decision_detail = f"Audio RMS {rms:.4f} below threshold"
                elif match_res.correct:
                    actual_decision = "Correct (+10 pts)"
                    decision_detail = f"Matched '{match_res.normalised}'"
                elif match_res.near_match:
                    actual_decision = "Near Match (0 pts)"
                    decision_detail = f"Transcript '{match_res.normalised}' (edit dist: {match_res.distance})"
                else:
                    actual_decision = "Incorrect (0 pts)"
                    decision_detail = f"Transcript '{match_res.normalised}' (unmatched)"

                result_transcript = transcript
                result_decision = actual_decision
                result_detail = decision_detail
                result_is_correct = match_res.correct
                result_rule_matched = actual_decision == expected_decision

            # Interactive render loop for this single trial
            while not quit_all:
                now = time.perf_counter()

                # Build 1280x720 canvas
                canvas = np.full(
                    (CANVAS_HEIGHT, CANVAS_WIDTH, 3), (24, 28, 35), dtype=np.uint8
                )

                # Top header bar (Height: 100)
                cv2.rectangle(canvas, (0, 0), (CANVAS_WIDTH, 100), (16, 18, 24), -1)
                cv2.putText(
                    canvas,
                    "See2Speak Speech Recognition & Answer Evaluation Benchmark",
                    (30, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.95,
                    (0, 215, 255),
                    2,
                )
                cv2.putText(
                    canvas,
                    f"Category: {cat_name.upper()} | Trial {trial_idx + 1} of {trials_per_cat}",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (200, 200, 200),
                    2,
                )

                # Main Clean Guidance Card (x: 120 to 1160, y: 120 to 590)
                c_x1, c_y1, c_x2, c_y2 = 120, 120, 1160, 590
                card_bg = (32, 38, 48)
                cv2.rectangle(canvas, (c_x1, c_y1), (c_x2, c_y2), card_bg, -1)
                cv2.rectangle(canvas, (c_x1, c_y1), (c_x2, c_y2), (70, 85, 105), 2)

                # Clean, authentic target presentation (NO cheat text on screen)
                cv2.putText(
                    canvas,
                    "PRACTICE TARGET",
                    (c_x1 + 45, c_y1 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 215, 255),
                    2,
                )
                cv2.putText(
                    canvas,
                    target_word.label.upper(),
                    (c_x1 + 45, c_y1 + 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.75,
                    (255, 255, 255),
                    3,
                )

                # Category badge
                cv2.rectangle(
                    canvas,
                    (c_x1 + 45, c_y1 + 140),
                    (c_x1 + 380, c_y1 + 180),
                    (45, 54, 68),
                    -1,
                )
                cv2.putText(
                    canvas,
                    f"Testing: {cat_name}",
                    (c_x1 + 58, c_y1 + 168),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (0, 215, 120),
                    2,
                )

                # Dynamic status area based on trial_state
                if trial_state == "READY":
                    cv2.rectangle(
                        canvas,
                        (c_x1 + 35, c_y1 + 205),
                        (c_x2 - 35, c_y2 - 25),
                        (28, 33, 42),
                        -1,
                    )
                    cv2.rectangle(
                        canvas,
                        (c_x1 + 35, c_y1 + 205),
                        (c_x2 - 35, c_y2 - 25),
                        (60, 70, 85),
                        1,
                    )

                    cv2.putText(
                        canvas,
                        "Microphone ready.",
                        (c_x1 + 60, c_y1 + 270),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.90,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        canvas,
                        "Click [START RECORDING] or press [SPACEBAR] to record.",
                        (c_x1 + 60, c_y1 + 325),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 215, 255),
                        2,
                    )
                    cv2.putText(
                        canvas,
                        f"A {record_duration:.1f}s audio capture window will open for speech transcription.",
                        (c_x1 + 60, c_y1 + 375),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (180, 180, 180),
                        1,
                    )

                elif trial_state == "RECORDING":
                    elapsed_rec = now - rec_start_time
                    rem_rec = max(0.0, record_duration - elapsed_rec)
                    pulse = (math.sin(now * 10) + 1.0) / 2.0
                    circle_color = (0, int(pulse * 60), int(200 + pulse * 55))

                    cv2.rectangle(
                        canvas,
                        (c_x1 + 35, c_y1 + 205),
                        (c_x2 - 35, c_y2 - 25),
                        (35, 25, 30),
                        -1,
                    )
                    cv2.rectangle(
                        canvas,
                        (c_x1 + 35, c_y1 + 205),
                        (c_x2 - 35, c_y2 - 25),
                        (0, 0, 255),
                        2,
                    )

                    cv2.circle(canvas, (c_x1 + 80, c_y1 + 285), 18, circle_color, -1)
                    cv2.circle(canvas, (c_x1 + 80, c_y1 + 285), 22, (255, 255, 255), 2)

                    cv2.putText(
                        canvas,
                        f"RECORDING NOW... Speak into microphone! ({rem_rec:.1f}s)",
                        (c_x1 + 120, c_y1 + 295),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        3,
                    )

                    pct = min(1.0, elapsed_rec / record_duration)
                    bar_w = c_x2 - c_x1 - 140
                    cv2.rectangle(
                        canvas,
                        (c_x1 + 70, c_y1 + 345),
                        (c_x1 + 70 + int(bar_w * pct), c_y1 + 370),
                        (0, 0, 255),
                        -1,
                    )
                    cv2.rectangle(
                        canvas,
                        (c_x1 + 70, c_y1 + 345),
                        (c_x1 + 70 + bar_w, c_y1 + 370),
                        (255, 255, 255),
                        2,
                    )

                    if rec_thread is not None and not rec_thread.is_alive():
                        trial_state = "TRANSCRIBING"
                        trans_thread = threading.Thread(
                            target=do_async_transcribe,
                            args=(raw_audio_buffer,),
                            daemon=True,
                        )
                        trans_thread.start()

                elif trial_state == "TRANSCRIBING":
                    cv2.rectangle(
                        canvas,
                        (c_x1 + 35, c_y1 + 205),
                        (c_x2 - 35, c_y2 - 25),
                        (28, 33, 42),
                        -1,
                    )
                    cv2.putText(
                        canvas,
                        "[..] Transcribing audio with OpenAI Whisper (base.en)...",
                        (c_x1 + 60, c_y1 + 300),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (0, 215, 255),
                        2,
                    )

                    if trans_thread is not None and not trans_thread.is_alive():
                        trial_state = "RESULT"

                        # Auto-save category screenshot
                        if cat_name not in saved_screenshots:
                            shot_name = f"speech_{cat_id}.png"
                            shot_path = SCREENSHOTS_DIR / shot_name
                            saved_screenshots[cat_name] = str(
                                shot_path.relative_to(ROOT)
                            )
                            cv2.imwrite(str(shot_path), canvas)
                            print(
                                f"  [📸] Saved category screenshot: {shot_path.relative_to(ROOT)}"
                            )

                elif trial_state == "RESULT":
                    # Realistic pedagogical badge based on actual system score
                    if result_decision == "Correct (+10 pts)":
                        badge_text = "CORRECT! (+10 PTS)"
                        badge_color = (0, 255, 0)
                        card_border = (0, 255, 0)
                    elif result_decision == "Near Match (0 pts)":
                        badge_text = "NEAR MATCH (0 PTS) - Close Pronunciation"
                        badge_color = (0, 165, 255)
                        card_border = (0, 165, 255)
                    elif result_decision == "Silent (0 pts, reject)":
                        badge_text = "NO SPEECH DETECTED (0 PTS)"
                        badge_color = (180, 180, 180)
                        card_border = (120, 120, 120)
                    else:
                        badge_text = "INCORRECT (0 PTS)"
                        badge_color = (0, 0, 255)
                        card_border = (0, 0, 255)

                    cv2.rectangle(
                        canvas,
                        (c_x1 + 35, c_y1 + 205),
                        (c_x2 - 35, c_y2 - 25),
                        (28, 33, 42),
                        -1,
                    )
                    cv2.rectangle(
                        canvas,
                        (c_x1 + 35, c_y1 + 205),
                        (c_x2 - 35, c_y2 - 25),
                        card_border,
                        2,
                    )

                    cv2.putText(
                        canvas,
                        f"GAME RESULT:  {badge_text}",
                        (c_x1 + 60, c_y1 + 265),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.88,
                        badge_color,
                        2,
                    )

                    disp_tr = (
                        result_transcript
                        if result_transcript
                        else "(No spoken words detected / Silent)"
                    )
                    cv2.putText(
                        canvas,
                        f'Whisper Transcript:  "{disp_tr}"',
                        (c_x1 + 60, c_y1 + 315),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.78,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        canvas,
                        f"Evaluation Detail:   {result_decision} [{result_detail}]",
                        (c_x1 + 60, c_y1 + 358),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (0, 215, 255),
                        2,
                    )
                    cv2.putText(
                        canvas,
                        f"Inference Latency:   {result_latency:.1f} ms",
                        (c_x1 + 60, c_y1 + 398),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.68,
                        (180, 180, 180),
                        2,
                    )

                # Render Bottom 5 Buttons Dock (y: 620 to 680)
                btn_y1 = 620
                btn_y2 = 678
                btn_w = 200
                gap = 16
                total_w = (btn_w * 5) + (gap * 4)
                start_x = (CANVAS_WIDTH - total_w) // 2

                b1_x = start_x
                b2_x = b1_x + btn_w + gap
                b3_x = b2_x + btn_w + gap
                b4_x = b3_x + btn_w + gap
                b5_x = b4_x + btn_w + gap

                if trial_state == "READY":
                    btn1 = GuiButton(
                        "ACTION_PRIMARY",
                        (b1_x, btn_y1, b1_x + btn_w, btn_y2),
                        "RECORD (Space)",
                        (40, 160, 40),
                    )
                elif trial_state == "RECORDING":
                    btn1 = GuiButton(
                        "ACTION_PRIMARY",
                        (b1_x, btn_y1, b1_x + btn_w, btn_y2),
                        "RECORDING...",
                        (0, 0, 200),
                    )
                elif trial_state == "TRANSCRIBING":
                    btn1 = GuiButton(
                        "ACTION_PRIMARY",
                        (b1_x, btn_y1, b1_x + btn_w, btn_y2),
                        "PROCESSING...",
                        (80, 80, 80),
                    )
                else:  # RESULT
                    btn1 = GuiButton(
                        "ACTION_PRIMARY",
                        (b1_x, btn_y1, b1_x + btn_w, btn_y2),
                        "NEXT TRIAL (Space)",
                        (40, 160, 40),
                    )

                btn2 = GuiButton(
                    "RETRY",
                    (b2_x, btn_y1, b2_x + btn_w, btn_y2),
                    "RETRY (R)",
                    (0, 140, 255),
                )
                btn3 = GuiButton(
                    "SNAPSHOT",
                    (b3_x, btn_y1, b3_x + btn_w, btn_y2),
                    "SNAPSHOT (S)",
                    (140, 70, 20),
                )
                btn4 = GuiButton(
                    "NEXT_CAT",
                    (b4_x, btn_y1, b4_x + btn_w, btn_y2),
                    "NEXT CAT (N)",
                    (160, 90, 40),
                )
                btn5 = GuiButton(
                    "QUIT",
                    (b5_x, btn_y1, b5_x + btn_w, btn_y2),
                    "QUIT (Q)",
                    (40, 40, 180),
                )

                active_buttons = [btn1, btn2, btn3, btn4, btn5]

                # Draw toast notification if active
                if now < toast_until:
                    cv2.rectangle(
                        canvas,
                        (CANVAS_WIDTH // 2 - 250, 60),
                        (CANVAS_WIDTH // 2 + 250, 105),
                        (0, 140, 0),
                        -1,
                    )
                    cv2.rectangle(
                        canvas,
                        (CANVAS_WIDTH // 2 - 250, 60),
                        (CANVAS_WIDTH // 2 + 250, 105),
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        canvas,
                        toast_msg,
                        (CANVAS_WIDTH // 2 - 230, 92),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )

                for btn in active_buttons:
                    btn.draw(canvas)

                cv2.imshow(window_name, canvas)

                key = cv2.waitKey(20) & 0xFF
                action = clicked_action
                clicked_action = None

                if key == ord("q") or key == ord("Q"):
                    action = "QUIT"
                elif key == ord("n") or key == ord("N"):
                    action = "NEXT_CAT"
                elif key == ord("r") or key == ord("R"):
                    action = "RETRY"
                elif key == ord("s") or key == ord("S"):
                    action = "SNAPSHOT"
                elif key in (32, 13):
                    action = "ACTION_PRIMARY"

                if action == "QUIT":
                    quit_all = True
                    break

                if action == "NEXT_CAT":
                    print(f"  -> Advancing to next category from {cat_name}.")
                    break

                if action == "SNAPSHOT":
                    snap_name = (
                        f"speech_{cat_id}_trial_{trial_idx + 1}_{int(time.time())}.png"
                    )
                    snap_path = SCREENSHOTS_DIR / snap_name
                    cv2.imwrite(str(snap_path), canvas)
                    toast_msg = f"[SNAPSHOT SAVED] {snap_name}"
                    toast_until = now + 1.5
                    print(
                        f"  [📸] Saved manual snapshot: {snap_path.relative_to(ROOT)}"
                    )
                    continue

                if action == "RETRY" and trial_state in ("RESULT", "READY"):
                    trial_state = "READY"
                    continue

                if action == "ACTION_PRIMARY":
                    if trial_state == "READY":
                        trial_state = "RECORDING"
                        rec_start_time = time.perf_counter()
                        rec_thread = threading.Thread(
                            target=do_async_record, daemon=True
                        )
                        rec_thread.start()
                    elif trial_state == "RESULT":
                        stats[cat_id]["trials"] += 1
                        stats[cat_id]["latencies"].append(result_latency)
                        if result_is_correct:
                            stats[cat_id]["correct_awarded"] += 1
                        if result_rule_matched:
                            stats[cat_id]["rule_matched"] += 1
                        trial_idx += 1
                        break

            if action == "NEXT_CAT":
                break

        cat_idx += 1

    cv2.destroyAllWindows()
    generate_speech_summary(stats, saved_screenshots)


def generate_speech_summary(stats: dict, saved_screenshots: dict[str, str]):
    print_banner("SPEECH RECOGNITION AND ANSWER EVALUATION BENCHMARK RESULTS")
    print(
        f"{'Category':<18} | {'Expected Decision':<22} | {'Trials':<6} | {'Points Awarded':<14} | {'Correct Rate':<12} | {'Latency':<8}"
    )
    print("-" * 92)

    total_trials = 0
    total_correct = 0
    total_rules = 0
    all_latencies = []

    for data in stats.values():
        trials = data["trials"]
        correct = data["correct_awarded"]
        rate = (correct / trials) * 100 if trials > 0 else 0.0
        avg_lat = (
            (sum(data["latencies"]) / len(data["latencies"]))
            if data["latencies"]
            else 0.0
        )

        total_trials += trials
        total_correct += correct
        total_rules += data["rule_matched"]
        all_latencies.extend(data["latencies"])

        print(
            f"{data['name']:<18} | {data['expected']:<22} | {trials:<6} | {f'{correct}/{trials}':<14} | {rate:>10.1f}% | {avg_lat:>6.1f}ms"
        )

    print("-" * 92)
    overall_correct_rate = (
        (total_correct / total_trials) * 100 if total_trials > 0 else 0.0
    )
    overall_rule_fidelity = (
        (total_rules / total_trials) * 100 if total_trials > 0 else 0.0
    )
    overall_lat = (sum(all_latencies) / len(all_latencies)) if all_latencies else 0.0

    print(
        f"{'OVERALL GAME TOTAL':<18} | {'Strict Scoring (+10)':<22} | {total_trials:<6} | {f'{total_correct}/{total_trials}':<14} | {overall_correct_rate:>10.1f}% | {overall_lat:>6.1f}ms"
    )
    print("=" * 92)
    print(
        f"System Decision Fidelity: {total_rules}/{total_trials} ({overall_rule_fidelity:.1f}% of trials matched expected rule outcomes)."
    )

    if saved_screenshots:
        print("\n" + "=" * 92)
        print(" SAVED SPEECH EVALUATION SCREENSHOTS (READY FOR REPORT)")
        print("=" * 92)
        for label, rel_path in saved_screenshots.items():
            print(f"  ✓ {label:<18} -> {rel_path}")
        print("=" * 92 + "\n")

    print("Benchmark complete. Data ready for Table 2 in your report.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="See2Speak Graphical Speech Evaluation Benchmark Tool"
    )
    parser.add_argument(
        "--trials",
        "-n",
        dest="trials",
        type=int,
        default=10,
        help="Trials per test category (default: 10)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        dest="duration",
        type=float,
        default=2.0,
        help="Recording capture duration in seconds (default: 2.0s)",
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=int,
        default=None,
        help="Input audio device index (optional)",
    )
    args = parser.parse_args()

    run_graphical_speech_benchmark(
        trials_per_cat=args.trials,
        record_duration=args.duration,
        device_idx=args.device,
    )
