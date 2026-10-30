"""Object detection with the pretrained YOLO-World v2 Small model.

No training / fine-tuning: the vocabulary is given to the model as text prompts via
set_classes(). Boxes are returned normalised (0..1) so the frontend can draw them on any video size.

Speed notes (measured on the real network, see README):
  * almost all per-frame time is neural-net inference; it scales with image size (DETECT_IMGSZ);
  * the very first inference is much slower than later ones, so load() does a warm-up pass.
Confidence: a detection is returned only if the model's OWN score is >= DETECT_MIN_CONF.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from . import config

EPS = 1e-6     # scores are float32: 0.90 is stored as 0.89999998, which must still count as 0.90


def iou(a: dict, b: dict) -> float:
    ax2, ay2, bx2, by2 = a["x"] + a["w"], a["y"] + a["h"], b["x"] + b["w"], b["y"] + b["h"]
    iw = max(0.0, min(ax2, bx2) - max(a["x"], b["x"]))
    ih = max(0.0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def merge_same_word(dets: list[dict], thr: float = None) -> list[dict]:
    """Several prompts can fire on one object (e.g. 'water bottle' and 'insulated metal flask').
    Keep the highest-confidence box when two boxes of the SAME word overlap. Scores are untouched."""
    thr = config.MERGE_IOU if thr is None else thr
    kept: list[dict] = []
    for d in sorted(dets, key=lambda x: x["confidence"], reverse=True):
        if any(k["key"] == d["key"] and iou(k["box"], d["box"]) >= thr for k in kept):
            continue
        kept.append(d)
    return kept


class Detector:
    def __init__(self):
        self._model = None
        self._lock = threading.Lock()   # one inference at a time
        self.vocab = list(config.VOCABULARY)
        # one YOLO-World "class" per prompt; several prompts may belong to the same word
        self.prompts: list[str] = []
        self.prompt_word: list[int] = []
        for wi, w in enumerate(self.vocab):
            for p in w.prompts:
                self.prompts.append(p)
                self.prompt_word.append(wi)
        self.warmup_ms: float | None = None

    def load(self):
        if self._model is not None:
            return
        if not config.YOLO_WEIGHTS.exists():
            raise FileNotFoundError(
                f"{config.YOLO_WEIGHTS} not found. Run: python scripts/download_models.py")
        from ultralytics import YOLOWorld   # imported lazily: heavy
        model = YOLOWorld(str(config.YOLO_WEIGHTS))
        model.set_classes(self.prompts)
        # Warm-up: the first predict() pays one-off setup costs (measured: many seconds on CPU).
        # Doing it here means the first live camera frame is as fast as every later one.
        t0 = time.perf_counter()
        blank = np.zeros((int(config.DETECT_IMGSZ * 0.75), config.DETECT_IMGSZ, 3), np.uint8)
        model.predict(blank, imgsz=config.DETECT_IMGSZ, conf=config.DETECT_CANDIDATE_FLOOR,
                      max_det=config.DETECT_MAX_DET, verbose=False)
        self.warmup_ms = (time.perf_counter() - t0) * 1000
        self._model = model

    @property
    def ready(self) -> bool:
        return self._model is not None

    def threshold_for(self, word) -> float:
        return max(config.DETECT_MIN_CONF, word.min_conf)

    def detect(self, jpeg_bytes: bytes) -> dict:
        self.load()
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
        h, w = frame.shape[:2]
        # ask the model for candidates down to a low floor (cheap), then apply the real threshold
        # ourselves so we can also report the best REJECTED candidate for diagnostics.
        floor = min(config.DETECT_CANDIDATE_FLOOR, config.DETECT_MIN_CONF)
        t0 = time.perf_counter()
        with self._lock:
            res = self._model.predict(frame, imgsz=config.DETECT_IMGSZ, conf=floor,
                                      max_det=config.DETECT_MAX_DET, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000

        accepted, best_rejected = [], None
        for box in res.boxes:
            word = self.vocab[self.prompt_word[int(box.cls[0])]]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            if conf + EPS >= self.threshold_for(word):
                accepted.append({
                    "key": word.key, "label": word.label, "confidence": round(conf, 3),
                    "box": {"x": x1 / w, "y": y1 / h, "w": (x2 - x1) / w, "h": (y2 - y1) / h}})
            elif best_rejected is None or conf > best_rejected["confidence"]:
                best_rejected = {"label": word.label, "confidence": round(conf, 3)}
        out = merge_same_word(accepted)[:config.DETECT_MAX_BOXES]
        return {"detections": out, "inference_ms": round(ms, 1), "frame": {"width": w, "height": h},
                "min_conf": config.DETECT_MIN_CONF,
                "best_rejected": None if out else best_rejected,
                "speed": {k: round(v, 1) for k, v in res.speed.items()}}
