"""Central configuration for See2Speak.

Everything a student might want to tweak (words, thresholds, points, model
names) lives here so the rest of the code never needs editing for that.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
FRONTEND_DIR = ROOT / "frontend"

YOLO_WEIGHTS = MODELS_DIR / "yolov8s-worldv2.pt"
FACE_LANDMARKER_TASK = MODELS_DIR / "face_landmarker.task"
# Whisper model: "base.en" per the proposal. Set SEE2SPEAK_WHISPER=tiny.en
# if base.en is too slow on your laptop (the proposal names tiny.en as fallback).
WHISPER_MODEL = os.environ.get("SEE2SPEAK_WHISPER", "base.en")
DB_PATH = DATA_DIR / "see2speak.sqlite3"


@dataclass(frozen=True)
class VocabWord:
    key: str                 # stable id, also the audio file name (assets/audio/<key>.mp3)
    label: str               # word shown to the child and used as the target
    yolo_prompt: str         # text prompt given to YOLO-World
    accepted: tuple[str, ...]  # normalised spoken forms that count as correct
    min_conf: float = 0.25   # per-word detection threshold


# To add a word: add a line here. Nothing else needs to change.
# Prompts are plain noun phrases (YOLO-World matches them against image regions with CLIP text
# embeddings). Thin / small / look-alike objects get a lower threshold so they can appear at all,
# which also means more false positives for them.
VOCABULARY: tuple[VocabWord, ...] = (
    VocabWord("book", "Book", "book", ("book",)),
    VocabWord("cell_phone", "Cell phone", "cell phone",
              ("cell phone", "cellphone", "phone", "mobile phone", "mobile")),
    VocabWord("apple", "Apple", "apple", ("apple",)),
    # Pens, pencils and markers are thin and look alike: low thresholds, expect confusion.
    VocabWord("pen", "Pen", "pen", ("pen",), min_conf=0.12),
    VocabWord("pencil", "Pencil", "pencil", ("pencil",), min_conf=0.12),
    VocabWord("marker", "Marker", "marker pen", ("marker", "marker pen"), min_conf=0.12),
    VocabWord("ruler", "Ruler", "ruler", ("ruler", "scale"), min_conf=0.15),
    VocabWord("water_bottle", "Water bottle", "water bottle", ("water bottle", "bottle"), min_conf=0.20),
    # Paper clips are tiny: hold one close to the camera against a plain background.
    VocabWord("paper_clip", "Paper clip", "paper clip", ("paper clip", "paperclip", "clip"), min_conf=0.10),
    VocabWord("tape_dispenser", "Tape dispenser", "tape dispenser",
              ("tape dispenser", "tape", "dispenser"), min_conf=0.15),
    VocabWord("eraser", "Eraser", "eraser", ("eraser",), min_conf=0.15),
    VocabWord("notebook", "Notebook", "notebook", ("notebook", "note book"), min_conf=0.20),
    VocabWord("stapler", "Stapler", "stapler", ("stapler",), min_conf=0.20),
    VocabWord("scissors", "Scissors", "scissors", ("scissors", "scissor"), min_conf=0.20),
    VocabWord("glue_stick", "Glue stick", "glue stick", ("glue stick", "glue"), min_conf=0.15),
)
VOCAB_BY_KEY = {w.key: w for w in VOCABULARY}


# ---- Game rules -----------------------------------------------------------
POINTS_CORRECT = 10
ROUND_TARGET = len(VOCABULARY)      # progress bar: distinct words answered correctly
# True: a word pays out once per session (stops point-farming by re-recording the
# same object). False: every correct attempt pays POINTS_CORRECT.
AWARD_ONCE_PER_WORD = True
NEAR_MATCH_MAX_DISTANCE = 1         # feedback only; never awards points

# ---- Detection -------------------------------------------------------------
DETECT_IMGSZ = 640
DETECT_MAX_BOXES = 6

# ---- Mouth movement check ---------------------------------------------------
# All distances are normalised by the inter-ocular distance so they do not
# depend on face size / camera distance. These are STARTING values: the
# proposal says thresholds must be tuned on separate tuning recordings.
MOUTH_CALIB_FRAMES = 15           # valid neutral-mouth samples needed to finish calibration
MOUTH_CALIB_MAX_OPEN = 0.10        # a sample only counts if the mouth is at most this open (relaxed/closed)
MOUTH_MIN_FRAMES = 4               # fewer valid face frames => "not enough data"
MOUTH_OPEN_DELTA = 0.06            # max opening above baseline that counts as movement
MOUTH_WIDTH_DELTA = 0.08           # max |width change| that counts as movement
MOUTH_CLOSURE_MAX = 0.03           # opening at/below this (+baseline) = lips closed
LIP_CLOSURE_LETTERS = ("b", "p", "m")
