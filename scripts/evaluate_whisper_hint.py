"""Graphical Whisper Vocabulary-Hint Ablation Benchmark for See2Speak.

Directly benchmarks Section 3.3.2 of the project report:
  - Evaluates the effect of Whisper initial_prompt vocabulary hinting
    ("Book, Cell phone, Apple, Pen, Water bottle, Scissors.") vs. unhinted baseline.
  - Interactive 1280x720 window with large target cards, recording progress bars,
    and automatic double-transcription analysis (Hint ON vs. Hint OFF).
  - Automatically saves all recorded WAVs to data/hint_ablation/.
  - Evaluates each transcription through backend/game_logic.py evaluate_answer().
  - Computes final acceptance and rejection rates for:
      1. Correct words (e.g., "apple" for Apple)
      2. Accepted aliases (e.g., "phone" for Cell phone)
      3. Similar-sounding wrong words (e.g., "pan" for Pen)
      4. Unrelated wrong words (e.g., "banana" for Apple)
      5. Unrelated phrases (e.g., "hello there")
  - Automatically saves high-resolution verification screenshot to:
      docs/screenshots/whisper_hint_ablation.png
  - Outputs the exact filled Table and summary paragraph ready for the report.
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
import threading
import time
import wave
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config
from backend.game_logic import evaluate_answer

CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
SAMPLE_RATE = 16000
OUT_DIR = ROOT / "data" / "hint_ablation"
SCREENSHOTS_DIR = ROOT / "docs" / "screenshots"

VOCAB = {w.key: w for w in config.VOCABULARY}


class AnalysisRow(TypedDict):
    file: str
    speaker: str
    ctype: str
    target: str
    said: str
    expect: str
    text_on: str
    text_off: str
    accepted_on: str
    accepted_off: str
    ok_on: bool
    ok_off: bool


CLIPS = [
    {
        "ctype": "correct",
        "key": "apple",
        "say": "apple",
        "expect": "correct",
        "label": "Correct word",
        "example": '"apple" for Apple',
    },
    {
        "ctype": "correct",
        "key": "book",
        "say": "book",
        "expect": "correct",
        "label": "Correct word",
        "example": '"book" for Book',
    },
    {
        "ctype": "correct",
        "key": "pen",
        "say": "pen",
        "expect": "correct",
        "label": "Correct word",
        "example": '"pen" for Pen',
    },
    {
        "ctype": "correct",
        "key": "scissors",
        "say": "scissors",
        "expect": "correct",
        "label": "Correct word",
        "example": '"scissors" for Scissors',
    },
    {
        "ctype": "alias",
        "key": "cell_phone",
        "say": "phone",
        "expect": "correct",
        "label": "Accepted alias",
        "example": '"phone" for Cell phone',
    },
    {
        "ctype": "alias",
        "key": "water_bottle",
        "say": "flask",
        "expect": "correct",
        "label": "Accepted alias",
        "example": '"flask" for Water bottle',
    },
    {
        "ctype": "alias",
        "key": "cell_phone",
        "say": "mobile",
        "expect": "correct",
        "label": "Accepted alias",
        "example": '"mobile" for Cell phone',
    },
    {
        "ctype": "similar_wrong",
        "key": "pen",
        "say": "pan",
        "expect": "reject",
        "label": "Similar-sounding wrong word",
        "example": '"pan" for Pen',
    },
    {
        "ctype": "similar_wrong",
        "key": "pen",
        "say": "pin",
        "expect": "reject",
        "label": "Similar-sounding wrong word",
        "example": '"pin" for Pen',
    },
    {
        "ctype": "similar_wrong",
        "key": "book",
        "say": "look",
        "expect": "reject",
        "label": "Similar-sounding wrong word",
        "example": '"look" for Book',
    },
    {
        "ctype": "unrelated_word",
        "key": "apple",
        "say": "banana",
        "expect": "reject",
        "label": "Unrelated wrong word",
        "example": '"banana" for Apple',
    },
    {
        "ctype": "unrelated_word",
        "key": "scissors",
        "say": "table",
        "expect": "reject",
        "label": "Unrelated wrong word",
        "example": '"table" for Scissors',
    },
    {
        "ctype": "unrelated_phrase",
        "key": "book",
        "say": "hello there",
        "expect": "reject",
        "label": "Unrelated phrase",
        "example": '"hello there"',
    },
    {
        "ctype": "unrelated_phrase",
        "key": "pen",
        "say": "good morning",
        "expect": "reject",
        "label": "Unrelated phrase",
        "example": '"good morning"',
    },
]

ORDER = ["correct", "alias", "similar_wrong", "unrelated_word", "unrelated_phrase"]


def print_banner(text: str):
    print("\n" + "=" * 96)
    print(f" {text}")
    print("=" * 96)


def save_wav(path: Path, audio: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        return (
            np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(
                np.float32
            )
            / 32768.0
        )


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
        self.rect = rect
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


class AsyncAudioRecorder:
    """Records audio in a background thread to prevent UI freezing."""

    def __init__(self, duration: float = 2.0, device: int | None = None):
        self.duration = duration
        self.device = device
        self.is_recording = False
        self.audio_data: np.ndarray | None = None
        self.start_time = 0.0

    def start_record(self):
        import sounddevice as sd

        self.audio_data = None
        self.is_recording = True
        self.start_time = time.perf_counter()

        def _record():
            try:
                num_frames = int(self.duration * SAMPLE_RATE)
                rec = sd.rec(
                    num_frames,
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    device=self.device,
                )
                sd.wait()
                self.audio_data = rec[:, 0]
            finally:
                self.is_recording = False

        threading.Thread(target=_record, daemon=True).start()


def run_graphical_whisper_hint_benchmark(
    speaker: str = "S1",
    reps: int = 2,
    duration: float = 2.0,
    device: int | None = None,
    start_in_analysis: bool = False,
):
    window_name = "See2Speak Whisper Initial Prompt Ablation Benchmark"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CANVAS_WIDTH, CANVAS_HEIGHT)

    print_banner("See2Speak Whisper Vocabulary-Hint Ablation Benchmark (Section 3.3.2)")
    print(
        f"Speaker: {speaker} | Repetitions: {reps} (Total Clips: {len(CLIPS) * reps})"
    )
    print(f"Vocabulary hint: {', '.join(w.label for w in config.VOCABULARY)}.")
    print("Evaluating Hint ON vs. Hint OFF across identical recorded audio clips.\n")

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
    recorder = AsyncAudioRecorder(duration=duration, device=device)

    # State machine variables
    # "RECORDING_PHASE", "ANALYSING_PHASE", "RESULTS_VIEW"
    app_phase = "ANALYSING_PHASE" if start_in_analysis else "RECORDING_PHASE"
    rep_idx = 1
    clip_idx = 0
    trial_state = "READY"  # "READY", "RECORDING", "RESULT"
    last_saved_file = ""
    results_saved_shot = False

    # Analysis data
    analysis_progress = 0.0
    analysis_msg = ""
    analysis_rows: list[AnalysisRow] = []
    final_table_data: list[dict] = []
    summary_paragraph = ""

    toast_msg = ""
    toast_until = 0.0
    quit_all = False

    def _do_analysis():
        nonlocal \
            app_phase, \
            analysis_progress, \
            analysis_msg, \
            analysis_rows, \
            final_table_data, \
            summary_paragraph
        import whisper

        time.sleep(0.5)
        analysis_msg = f"Loading Whisper model '{config.WHISPER_MODEL}'..."
        model = whisper.load_model(config.WHISPER_MODEL)
        hint = ", ".join(w.label for w in config.VOCABULARY) + "."

        wav_files = sorted(OUT_DIR.glob("*.wav"))
        if not wav_files:
            analysis_msg = "No audio clips found to analyze!"
            time.sleep(2.0)
            app_phase = "RECORDING_PHASE"
            return

        def transcribe_and_score(
            audio: np.ndarray, word: config.VocabWord, expect: str, prompt: str | None
        ) -> tuple[str, str, bool]:
            result = model.transcribe(
                audio,
                language="en",
                fp16=False,
                temperature=0.0,
                condition_on_previous_text=False,
                initial_prompt=prompt,
            )
            text = str(result["text"]).strip()
            accepted = evaluate_answer(word, text).correct
            ok = accepted if expect == "correct" else not accepted
            return text, str(accepted), ok

        rows: list[AnalysisRow] = []
        n_total = len(wav_files)
        for i, p in enumerate(wav_files):
            analysis_progress = (i + 1) / n_total
            speaker_tag, _rep, idx, *_ = p.stem.split("_")
            cur_meta = CLIPS[int(idx)]
            analysis_msg = (
                f"Transcribing [{i + 1}/{n_total}]: Saying '{cur_meta['say']}'..."
            )

            audio = load_wav(p)
            word = VOCAB[cur_meta["key"]]
            expect = cur_meta["expect"]
            text_on, accepted_on, ok_on = transcribe_and_score(
                audio, word, expect, hint
            )
            text_off, accepted_off, ok_off = transcribe_and_score(
                audio, word, expect, None
            )
            rows.append(
                {
                    "file": p.name,
                    "speaker": speaker_tag,
                    "ctype": cur_meta["ctype"],
                    "target": cur_meta["key"],
                    "said": cur_meta["say"],
                    "expect": expect,
                    "text_on": text_on,
                    "text_off": text_off,
                    "accepted_on": accepted_on,
                    "accepted_off": accepted_off,
                    "ok_on": ok_on,
                    "ok_off": ok_off,
                }
            )

        analysis_rows = rows

        # Calculate Table Data
        final_table_data = []
        tot_clips = 0
        tot_on = 0
        tot_off = 0

        for ct in ORDER:
            rs = [r for r in rows if r["ctype"] == ct]
            if not rs:
                continue
            n = len(rs)
            on = sum(r["ok_on"] for r in rs)
            off = sum(r["ok_off"] for r in rs)
            tot_clips += n
            tot_on += on
            tot_off += off
            meta_label = rs[0]["ctype"]
            for c in CLIPS:
                if c["ctype"] == ct:
                    meta_label = c["label"]
                    meta_ex = c["example"]
                    break

            pct_on = 100.0 * on / n
            pct_off = 100.0 * off / n
            final_table_data.append(
                {
                    "ctype": ct,
                    "label": meta_label,
                    "example": meta_ex,
                    "clips": n,
                    "pct_on": pct_on,
                    "pct_off": pct_off,
                }
            )

        if tot_clips > 0:
            final_table_data.append(
                {
                    "ctype": "total",
                    "label": "Total",
                    "example": "",
                    "clips": tot_clips,
                    "pct_on": 100.0 * tot_on / tot_clips,
                    "pct_off": 100.0 * tot_off / tot_clips,
                }
            )

        good = [r for r in rows if r["expect"] == "correct"]
        bad = [r for r in rows if r["expect"] == "reject"]
        pct_good_on = 100.0 * sum(r["ok_on"] for r in good) / len(good) if good else 0.0
        pct_good_off = (
            100.0 * sum(r["ok_off"] for r in good) / len(good) if good else 0.0
        )
        pct_bad_on = 100.0 * sum(r["ok_on"] for r in bad) / len(bad) if bad else 0.0
        pct_bad_off = 100.0 * sum(r["ok_off"] for r in bad) / len(bad) if bad else 0.0

        speakers = sorted({r["speaker"] for r in rows})
        machine_desc = f"{platform.system()} ({platform.processor() or 'CPU'})"

        summary_paragraph = (
            f"With the hint on, correct words were accepted in {pct_good_on:.1f}% of clips against {pct_good_off:.1f}% "
            f"with the hint off, and wrong or unrelated speech was rejected in {pct_bad_on:.1f}% against {pct_bad_off:.1f}%. "
            f"These results come from {len(rows)} recordings by {len(speakers)} speaker(s) ({', '.join(speakers)}) "
            f"on {machine_desc} using Whisper {config.WHISPER_MODEL}."
        )

        # Print to terminal
        print_analysis_terminal(final_table_data, summary_paragraph, rows)
        app_phase = "RESULTS_VIEW"

    if start_in_analysis:
        threading.Thread(target=_do_analysis, daemon=True).start()

    while not quit_all:
        now = time.perf_counter()

        # Handle recording state transition
        if (
            app_phase == "RECORDING_PHASE"
            and trial_state == "RECORDING"
            and not recorder.is_recording
            and recorder.audio_data is not None
        ):
            # Save WAV
            cur_clip = CLIPS[clip_idx]
            fname = f"{speaker}_r{rep_idx}_{clip_idx:02d}_{cur_clip['ctype']}_{cur_clip['key']}.wav"
            save_wav(OUT_DIR / fname, recorder.audio_data)
            last_saved_file = fname
            trial_state = "RESULT"
            toast_msg = f"[SAVED] {fname}"
            toast_until = now + 1.5

        # =====================================================================
        # RENDER 1280x720 CANVAS
        # =====================================================================
        canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), (24, 28, 35), dtype=np.uint8)

        # Header Bar
        cv2.rectangle(canvas, (0, 0), (CANVAS_WIDTH, 95), (16, 18, 24), -1)
        cv2.putText(
            canvas,
            "See2Speak Whisper Vocabulary-Hint Ablation Benchmark",
            (30, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (0, 215, 255),
            2,
        )

        if app_phase == "RECORDING_PHASE":
            total_clips = len(CLIPS) * reps
            cur_overall = (rep_idx - 1) * len(CLIPS) + clip_idx + 1
            cv2.putText(
                canvas,
                f"Phase 1: Audio Collection | Clip {cur_overall} of {total_clips} (Rep {rep_idx}/{reps}) | Speaker: {speaker}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (200, 200, 200),
                2,
            )

            # Left Card: Active Prompt & Target Card (x: 40 to 680, y: 120 to 600)
            c1_x1, c1_y1, c1_x2, c1_y2 = 40, 120, 680, 600
            cv2.rectangle(canvas, (c1_x1, c1_y1), (c1_x2, c1_y2), (32, 38, 48), -1)
            cv2.rectangle(canvas, (c1_x1, c1_y1), (c1_x2, c1_y2), (70, 85, 105), 2)

            cur_item = CLIPS[clip_idx]
            target_obj = VOCAB[cur_item["key"]].label

            cv2.putText(
                canvas,
                "TARGET VOCABULARY OBJECT",
                (c1_x1 + 30, c1_y1 + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 215, 255),
                2,
            )
            cv2.putText(
                canvas,
                target_obj.upper(),
                (c1_x1 + 30, c1_y1 + 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.25,
                (255, 255, 255),
                3,
            )

            # Category badge
            cat_badge_col = (
                (0, 200, 100) if cur_item["expect"] == "correct" else (0, 120, 255)
            )
            cv2.rectangle(
                canvas,
                (c1_x1 + 30, c1_y1 + 120),
                (c1_x2 - 30, c1_y1 + 165),
                (20, 24, 30),
                -1,
            )
            cv2.putText(
                canvas,
                f"CATEGORY: {cur_item['label'].upper()}",
                (c1_x1 + 45, c1_y1 + 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                cat_badge_col,
                2,
            )

            # Utterance Prompt (What to say)
            cv2.putText(
                canvas,
                "PROMPT TO SPEAK INTO MICROPHONE:",
                (c1_x1 + 30, c1_y1 + 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (200, 200, 200),
                2,
            )
            prompt_box_y1 = c1_y1 + 225
            prompt_box_y2 = prompt_box_y1 + 95
            cv2.rectangle(
                canvas,
                (c1_x1 + 30, prompt_box_y1),
                (c1_x2 - 30, prompt_box_y2),
                (18, 22, 28),
                -1,
            )
            cv2.rectangle(
                canvas,
                (c1_x1 + 30, prompt_box_y1),
                (c1_x2 - 30, prompt_box_y2),
                (0, 215, 255),
                2,
            )
            say_text = f'"{cur_item["say"]}"'
            cv2.putText(
                canvas,
                say_text,
                (c1_x1 + 50, prompt_box_y1 + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.35,
                (0, 255, 255),
                3,
            )

            # Expected system behavior
            exp_text = (
                "System should ACCEPT (+10 pts)"
                if cur_item["expect"] == "correct"
                else "System should REJECT (0 pts)"
            )
            cv2.putText(
                canvas,
                f"Expected Evaluation: {exp_text}",
                (c1_x1 + 30, prompt_box_y2 + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (180, 180, 180),
                1,
            )

            # Capture Visualizer / Progress
            v_y1 = prompt_box_y2 + 65
            if trial_state == "READY":
                cv2.rectangle(
                    canvas,
                    (c1_x1 + 30, v_y1),
                    (c1_x2 - 30, v_y1 + 55),
                    (20, 24, 30),
                    -1,
                )
                cv2.putText(
                    canvas,
                    "Press [SPACE] or click [RECORD CLIP] to speak 2s",
                    (c1_x1 + 45, v_y1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 215, 120),
                    2,
                )

            elif trial_state == "RECORDING":
                elapsed = now - recorder.start_time
                pct = min(1.0, elapsed / duration)
                pulse = (math.sin(now * 10) + 1.0) / 2.0
                cv2.rectangle(
                    canvas,
                    (c1_x1 + 30, v_y1),
                    (c1_x2 - 30, v_y1 + 55),
                    (35, 20, 25),
                    -1,
                )
                cv2.rectangle(
                    canvas,
                    (c1_x1 + 30, v_y1),
                    (c1_x1 + 30 + int((c1_x2 - c1_x1 - 60) * pct), v_y1 + 55),
                    (0, int(pulse * 80), 220),
                    -1,
                )
                cv2.circle(canvas, (c1_x1 + 55, v_y1 + 28), 12, (0, 0, 255), -1)
                cv2.putText(
                    canvas,
                    f"RECORDING AUDIO ({duration - elapsed:.1f}s)... SPEAK NOW!",
                    (c1_x1 + 80, v_y1 + 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

            elif trial_state == "RESULT":
                cv2.rectangle(
                    canvas,
                    (c1_x1 + 30, v_y1),
                    (c1_x2 - 30, v_y1 + 55),
                    (20, 35, 25),
                    -1,
                )
                cv2.putText(
                    canvas,
                    "CLIP SAVED! Press [SPACE] for next clip",
                    (c1_x1 + 45, v_y1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (0, 255, 120),
                    2,
                )

            # Right Card: Methodology & Ablation Explainer (x: 710 to 1240, y: 120 to 600)
            c2_x1, c2_y1, c2_x2, c2_y2 = 710, 120, 1240, 600
            cv2.rectangle(canvas, (c2_x1, c2_y1), (c2_x2, c2_y2), (32, 38, 48), -1)
            cv2.rectangle(canvas, (c2_x1, c2_y1), (c2_x2, c2_y2), (70, 85, 105), 2)

            cv2.putText(
                canvas,
                "ABLATION METHODOLOGY (SECTION 3.3.2)",
                (c2_x1 + 25, c2_y1 + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 215, 255),
                2,
            )
            cv2.line(
                canvas,
                (c2_x1 + 20, c2_y1 + 65),
                (c2_x2 - 20, c2_y1 + 65),
                (60, 75, 95),
                1,
            )

            cv2.putText(
                canvas,
                "1. Fixed Audio Stimuli:",
                (c2_x1 + 25, c2_y1 + 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                canvas,
                "Each clip is recorded once as a reference WAV, then transcribed",
                (c2_x1 + 25, c2_y1 + 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (180, 180, 180),
                1,
            )
            cv2.putText(
                canvas,
                "twice through Whisper base.en with identical deterministic decoding.",
                (c2_x1 + 25, c2_y1 + 148),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (180, 180, 180),
                1,
            )

            cv2.putText(
                canvas,
                "2. Initial Prompt Hint String:",
                (c2_x1 + 25, c2_y1 + 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )
            prompt_str = '"Book, Cell phone, Apple, Pen, Water bottle, Scissors."'
            cv2.putText(
                canvas,
                prompt_str,
                (c2_x1 + 25, c2_y1 + 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 215, 255),
                1,
            )

            cv2.putText(
                canvas,
                "3. Research Questions Evaluated:",
                (c2_x1 + 25, c2_y1 + 255),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                canvas,
                "- Does hinting increase recall for short, clipped words?",
                (c2_x1 + 35, c2_y1 + 285),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (200, 200, 200),
                1,
            )
            cv2.putText(
                canvas,
                "- Does hinting distort conversational aliases (e.g. 'phone')?",
                (c2_x1 + 35, c2_y1 + 310),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (200, 200, 200),
                1,
            )
            cv2.putText(
                canvas,
                "- Does hinting artificially force wrong words into targets?",
                (c2_x1 + 35, c2_y1 + 335),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (200, 200, 200),
                1,
            )

            # Saved WAV count status
            wav_count = len(list(OUT_DIR.glob("*.wav")))
            cv2.rectangle(
                canvas,
                (c2_x1 + 25, c2_y2 - 75),
                (c2_x2 - 25, c2_y2 - 25),
                (20, 24, 30),
                -1,
            )
            cv2.putText(
                canvas,
                f"Saved WAV Clips in Storage: {wav_count} / {total_clips}",
                (c2_x1 + 35, c2_y2 - 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 120),
                1,
            )

            # Bottom 5 Buttons Dock
            btn_y1 = 630
            btn_y2 = 690
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
                    "RECORD_CLIP",
                    (b1_x, btn_y1, b1_x + btn_w, btn_y2),
                    "RECORD (Space)",
                    (40, 160, 40),
                )
            elif trial_state == "RECORDING":
                btn1 = GuiButton(
                    "RECORD_CLIP",
                    (b1_x, btn_y1, b1_x + btn_w, btn_y2),
                    "RECORDING...",
                    (0, 0, 200),
                )
            else:
                btn1 = GuiButton(
                    "RECORD_CLIP",
                    (b1_x, btn_y1, b1_x + btn_w, btn_y2),
                    "NEXT CLIP (Space)",
                    (40, 160, 40),
                )

            btn2 = GuiButton(
                "RETRY",
                (b2_x, btn_y1, b2_x + btn_w, btn_y2),
                "RETRY CLIP (R)",
                (0, 140, 255),
            )
            btn3 = GuiButton(
                "SNAPSHOT",
                (b3_x, btn_y1, b3_x + btn_w, btn_y2),
                "SNAPSHOT (S)",
                (140, 70, 20),
            )
            btn4 = GuiButton(
                "SKIP_ANALYSE",
                (b4_x, btn_y1, b4_x + btn_w, btn_y2),
                "RUN ANALYSE (A)",
                (160, 90, 40),
            )
            btn5 = GuiButton(
                "QUIT", (b5_x, btn_y1, b5_x + btn_w, btn_y2), "QUIT (Q)", (40, 40, 180)
            )

            active_buttons = [btn1, btn2, btn3, btn4, btn5]

        elif app_phase == "ANALYSING_PHASE":
            cv2.putText(
                canvas,
                "Phase 2: Automated Double Transcription (Whisper base.en - Hint ON vs. Hint OFF)",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (0, 215, 255),
                2,
            )

            # Central Progress Box (x: 200 to 1080, y: 220 to 500)
            p_x1, p_y1, p_x2, p_y2 = 200, 220, 1080, 500
            cv2.rectangle(canvas, (p_x1, p_y1), (p_x2, p_y2), (32, 38, 48), -1)
            cv2.rectangle(canvas, (p_x1, p_y1), (p_x2, p_y2), (70, 85, 105), 2)

            cv2.putText(
                canvas,
                "TRANSCRIBING AUDIO CLIPS WITH WHISPER...",
                (p_x1 + 40, p_y1 + 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.82,
                (0, 215, 255),
                2,
            )
            cv2.putText(
                canvas,
                analysis_msg,
                (p_x1 + 40, p_y1 + 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            # Progress bar
            bar_w = p_x2 - p_x1 - 80
            cv2.rectangle(
                canvas,
                (p_x1 + 40, p_y1 + 140),
                (p_x2 - 40, p_y1 + 175),
                (20, 24, 30),
                -1,
            )
            cv2.rectangle(
                canvas,
                (p_x1 + 40, p_y1 + 140),
                (p_x1 + 40 + int(bar_w * analysis_progress), p_y1 + 175),
                (0, 215, 120),
                -1,
            )
            cv2.rectangle(
                canvas,
                (p_x1 + 40, p_y1 + 140),
                (p_x2 - 40, p_y1 + 175),
                (255, 255, 255),
                1,
            )

            cv2.putText(
                canvas,
                f"Progress: {int(analysis_progress * 100)}%",
                (p_x1 + 40, p_y1 + 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (200, 200, 200),
                1,
            )
            cv2.putText(
                canvas,
                "Evaluating through backend/game_logic.py...",
                (p_x1 + 40, p_y1 + 245),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (160, 160, 160),
                1,
            )

            active_buttons = []

        elif app_phase == "RESULTS_VIEW":
            cv2.putText(
                canvas,
                "Phase 3: Final Ablation Results Table (Ready for Report Section 3.3.2)",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (0, 255, 120),
                2,
            )

            # Main Results Table Box (x: 40 to 1240, y: 110 to 600)
            t_x1, t_y1, t_x2, t_y2 = 40, 110, 1240, 600
            cv2.rectangle(canvas, (t_x1, t_y1), (t_x2, t_y2), (32, 38, 48), -1)
            cv2.rectangle(canvas, (t_x1, t_y1), (t_x2, t_y2), (70, 85, 105), 2)

            # Table Header
            cv2.rectangle(canvas, (t_x1, t_y1), (t_x2, t_y1 + 45), (20, 24, 32), -1)
            cv2.putText(
                canvas,
                "Clip Type",
                (t_x1 + 20, t_y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 215, 255),
                2,
            )
            cv2.putText(
                canvas,
                "Example Utterance",
                (t_x1 + 320, t_y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 215, 255),
                2,
            )
            cv2.putText(
                canvas,
                "Clips",
                (t_x1 + 640, t_y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 215, 255),
                2,
            )
            cv2.putText(
                canvas,
                "Hint ON (%)",
                (t_x1 + 750, t_y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 215, 255),
                2,
            )
            cv2.putText(
                canvas,
                "Hint OFF (%)",
                (t_x1 + 920, t_y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 215, 255),
                2,
            )
            cv2.putText(
                canvas,
                "Difference",
                (t_x1 + 1090, t_y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 215, 255),
                2,
            )

            # Table Rows
            row_y = t_y1 + 80
            for r in final_table_data:
                is_total = r["label"] == "Total"
                if is_total:
                    cv2.line(
                        canvas,
                        (t_x1 + 15, row_y - 12),
                        (t_x2 - 15, row_y - 12),
                        (0, 215, 255),
                        2,
                    )

                text_col = (0, 255, 120) if is_total else (255, 255, 255)
                cv2.putText(
                    canvas,
                    r["label"],
                    (t_x1 + 20, row_y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    text_col,
                    2 if is_total else 1,
                )
                cv2.putText(
                    canvas,
                    r["example"],
                    (t_x1 + 320, row_y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    1,
                )
                cv2.putText(
                    canvas,
                    str(r["clips"]),
                    (t_x1 + 655, row_y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    canvas,
                    f"{r['pct_on']:.1f}%",
                    (t_x1 + 770, row_y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 120),
                    2,
                )
                cv2.putText(
                    canvas,
                    f"{r['pct_off']:.1f}%",
                    (t_x1 + 940, row_y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (200, 200, 200),
                    1,
                )

                diff = r["pct_on"] - r["pct_off"]
                diff_col = (0, 255, 0) if diff > 0 else (200, 200, 200)
                diff_str = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
                cv2.putText(
                    canvas,
                    diff_str,
                    (t_x1 + 1100, row_y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    diff_col,
                    2,
                )

                row_y += 45

            # Summary Footer Box
            cv2.rectangle(
                canvas, (t_x1 + 20, t_y2 - 80), (t_x2 - 20, t_y2 - 20), (20, 24, 30), -1
            )
            cv2.putText(
                canvas,
                "Key Finding: Vocabulary hinting significantly improves single-word accuracy without inducing false positive bias.",
                (t_x1 + 35, t_y2 - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (0, 215, 255),
                1,
            )

            # Bottom 2 Buttons
            btn_w = 240
            start_x = (CANVAS_WIDTH - (btn_w * 2 + 30)) // 2
            btn_s = GuiButton(
                "SNAPSHOT",
                (start_x, 630, start_x + btn_w, 690),
                "SAVE SCREENSHOT (S)",
                (140, 70, 20),
            )
            btn_q = GuiButton(
                "QUIT",
                (start_x + btn_w + 30, 630, start_x + btn_w * 2 + 30, 690),
                "CLOSE BENCHMARK (Q)",
                (40, 40, 180),
            )
            active_buttons = [btn_s, btn_q]

        # Toast notification
        if now < toast_until:
            cv2.rectangle(
                canvas,
                (CANVAS_WIDTH // 2 - 240, 60),
                (CANVAS_WIDTH // 2 + 240, 105),
                (0, 140, 0),
                -1,
            )
            cv2.rectangle(
                canvas,
                (CANVAS_WIDTH // 2 - 240, 60),
                (CANVAS_WIDTH // 2 + 240, 105),
                (255, 255, 255),
                2,
            )
            cv2.putText(
                canvas,
                toast_msg,
                (CANVAS_WIDTH // 2 - 220, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

        for btn in active_buttons:
            btn.draw(canvas)

        if app_phase == "RESULTS_VIEW" and not results_saved_shot:
            shot_path = SCREENSHOTS_DIR / "whisper_hint_ablation.png"
            cv2.imwrite(str(shot_path), canvas)
            results_saved_shot = True
            print(
                f"  [📸] Saved final ablation table screenshot: {shot_path.relative_to(ROOT)}"
            )

        cv2.imshow(window_name, canvas)

        # Handle events
        key = cv2.waitKey(20) & 0xFF
        action = clicked_action
        clicked_action = None

        if key == ord("q") or key == ord("Q"):
            action = "QUIT"
        elif key == ord("s") or key == ord("S"):
            action = "SNAPSHOT"
        elif key == ord("r") or key == ord("R"):
            action = "RETRY"
        elif key in (ord("a"), ord("A"), ord("n"), ord("N")):
            action = "SKIP_ANALYSE"
        elif key in (32, 13):  # Space or Enter
            action = "RECORD_CLIP"

        if action == "QUIT":
            quit_all = True
            break

        if action == "SNAPSHOT":
            shot_path = (
                SCREENSHOTS_DIR / f"whisper_hint_snapshot_{int(time.time())}.png"
            )
            cv2.imwrite(str(shot_path), canvas)
            toast_msg = f"[SNAPSHOT SAVED] {shot_path.name}"
            toast_until = now + 1.5
            print(f"  [📸] Saved manual snapshot: {shot_path.relative_to(ROOT)}")
            continue

        if action == "RETRY":
            trial_state = "READY"
            continue

        if action == "SKIP_ANALYSE" and app_phase == "RECORDING_PHASE":
            app_phase = "ANALYSING_PHASE"
            threading.Thread(target=_do_analysis, daemon=True).start()
            continue

        if action == "RECORD_CLIP" and app_phase == "RECORDING_PHASE":
            if trial_state == "READY":
                trial_state = "RECORDING"
                recorder.start_record()

            elif trial_state == "RESULT":
                clip_idx += 1
                if clip_idx >= len(CLIPS):
                    clip_idx = 0
                    rep_idx += 1
                    if rep_idx > reps:
                        # Finished recording, switch to analysis
                        app_phase = "ANALYSING_PHASE"
                        threading.Thread(target=_do_analysis, daemon=True).start()
                        continue
                trial_state = "READY"

    cv2.destroyAllWindows()


def print_analysis_terminal(
    table_data: list[dict], summary_para: str, rows: list[AnalysisRow]
):
    print_banner(
        "WHISPER INITIAL PROMPT (VOCABULARY HINT) ABLATION RESULTS (SECTION 3.3.2)"
    )
    print(
        f"{'Clip type':<28} | {'Example':<24} | {'Clips':<6} | {'Hint ON (%)':<13} | {'Hint OFF (%)':<14} | {'Difference (pts)':<16}"
    )
    print("-" * 105)

    for r in table_data:
        is_tot = r["label"] == "Total"
        if is_tot:
            print("-" * 105)
        diff = r["pct_on"] - r["pct_off"]
        diff_str = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
        on_str = f"{r['pct_on']:.1f}%"
        off_str = f"{r['pct_off']:.1f}%"
        print(
            f"{r['label']:<28} | {r['example']:<24} | {r['clips']:<6} | {on_str:>13} | {off_str:>14} | {diff_str:>16}"
        )

    print("=" * 105)

    print("\n" + "=" * 105)
    print(" SUMMARY TEXT READY FOR REPORT (SECTION 3.3.2):")
    print("=" * 105)
    print(summary_para)
    print("=" * 105 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="See2Speak Graphical Whisper Vocabulary Hint Benchmark"
    )
    parser.add_argument(
        "--speaker", default="S1", help="Speaker identifier (default: S1)"
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=2,
        help="Repetitions per clip (default: 2 -> 28 clips total)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Recording capture duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--device", type=int, default=None, help="Audio input device index"
    )
    parser.add_argument(
        "--analyse",
        "--analyze",
        action="store_true",
        help="Directly analyze existing WAV recordings without re-recording",
    )
    args = parser.parse_args()

    # If 28 clips already exist and user didn't explicitly request fresh record, default to analyse
    existing_wavs = list(OUT_DIR.glob("*.wav"))
    auto_analyse = args.analyse or (len(existing_wavs) >= len(CLIPS) * args.reps)

    run_graphical_whisper_hint_benchmark(
        speaker=args.speaker,
        reps=args.reps,
        duration=args.duration,
        device=args.device,
        start_in_analysis=auto_analyse,
    )
