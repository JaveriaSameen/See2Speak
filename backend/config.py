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
YOLO_BAKED_WEIGHTS = MODELS_DIR / "yolov8s-see2speak.pt"
FACE_LANDMARKER_TASK = MODELS_DIR / "face_landmarker.task"
# Whisper model: "base.en" per the proposal. Set SEE2SPEAK_WHISPER=tiny.en
# if base.en is too slow on your laptop (the proposal names tiny.en as fallback).
WHISPER_MODEL = os.environ.get("SEE2SPEAK_WHISPER", "base.en")
DB_PATH = DATA_DIR / "see2speak.sqlite3"


@dataclass(frozen=True)
class VocabWord:
    key: str                 # stable id, also the audio file name (assets/audio/<key>.mp3)
    label: str               # word shown to the child and used as the target
    yolo_prompt: str         # main text prompt given to YOLO-World
    accepted: tuple[str, ...]  # normalised spoken forms that count as correct
    min_conf: float = 0.0    # optional EXTRA floor for this word; the global DETECT_MIN_CONF always applies
    extra_prompts: tuple[str, ...] = ()   # more phrasings that all map to this same word

    @property
    def prompts(self) -> tuple[str, ...]:
        return (self.yolo_prompt,) + self.extra_prompts


# To add a word: add a line here. Nothing else needs to change.
# Prompts are plain noun phrases (YOLO-World matches them against image regions with CLIP text
# embeddings). `extra_prompts` are alternative phrasings that map to the SAME word.
VOCABULARY: tuple[VocabWord, ...] = (
    VocabWord("book", "Book", "book", ("book",),
              extra_prompts=("hardcover book", "paperback book", "reading book", "closed book")),
    VocabWord("cell_phone", "Cell phone", "smartphone with glass screen",
              ("cell phone", "cellphone", "phone", "mobile phone", "mobile"),
              min_conf=0.88,
              extra_prompts=("touchscreen smartphone", "black glass smartphone", "illuminated phone display")),
    VocabWord("apple", "Apple", "apple", ("apple",),
              extra_prompts=("red apple", "green apple", "fresh apple")),
    VocabWord("pen", "Pen", "pen", ("pen",),
              extra_prompts=(
                  "ballpoint pen", "writing pen", "plastic ballpoint pen", "pen with cap",
                  "holding a pen", "pen in hand", "hand holding a pen", "blue pen", "black pen",
                  "pointer pen", "fineliner pen", "felt tip pen", "fine point pen",
                  "black fineliner", "sketch pen", "slim marker pen", "dollar pointer pen"
              )),
    # Ordinary plastic bottles AND reusable / insulated / metal flasks all count as "Water bottle".
    VocabWord("water_bottle", "Water bottle", "water bottle",
              ("water bottle", "bottle", "flask", "water flask"),
              extra_prompts=("plastic water bottle", "reusable water bottle", "insulated metal flask", "drinking bottle")),
    VocabWord("scissors", "Scissors", "scissors", ("scissors", "scissor"),
              extra_prompts=(
                  "pair of scissors", "desk scissors", "craft scissors",
                  "closed scissors", "pair of closed scissors", "cutting scissors",
                  "office scissors", "stationery scissors", "scissors with plastic handles",
                  "holding scissors"
              )),
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
TESTING = False
# A box + label is shown ONLY when the model's confidence is at least this.
DETECT_MIN_CONF = float(os.environ.get("SEE2SPEAK_MIN_CONF", "0.80"))
# Calibration: open-vocabulary zero-shot cosine similarities naturally range 0.35-0.65.
# Set SEE2SPEAK_CALIBRATE=1 (default) to scale real matches into standard 0.80-0.95 range.
CALIBRATE_CONFIDENCE = os.environ.get("SEE2SPEAK_CALIBRATE", "1") == "1"
# Inference size (multiple of 32). 640 is YOLO's native resolution, giving 50% more detail
# for small objects like pens and glue sticks. Override: SEE2SPEAK_IMGSZ=512
DETECT_IMGSZ = int(os.environ.get("SEE2SPEAK_IMGSZ", "640"))
DETECT_MAX_BOXES = 6
DETECT_MAX_DET = 20
DETECT_CANDIDATE_FLOOR = 0.25      # lower scores are ignored even for the "best rejected" hint
MERGE_IOU = 0.5                    # same-word boxes (from different prompts) overlapping this much are merged
CROSS_MERGE_IOU = 0.60              # different-word boxes overlapping this much are suppressed (higher conf wins)

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
