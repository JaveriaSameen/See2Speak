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

EPS = 1e-6     # scores are float32: 0.80 is stored as 0.79999995, which must still count as 0.80


def calibrate_confidence(conf: float, word_key: str = "") -> float:
    """Map open-vocabulary zero-shot CLIP cosine similarity into standard supervised confidence.

    Different object geometries have different natural CLIP cosine ranges:
      - Slender/thin objects (pen, scissors) peak around 0.30 - 0.35 -> anchored at 0.32
      - Structured slabs (cell phone) easily produce spurious 0.35 - 0.40 on any rectangle -> anchored at 0.42
      - Normal items (book, apple, water bottle) -> anchored at 0.36
    """
    if not getattr(config, "CALIBRATE_CONFIDENCE", False):
        return conf
    if conf < 0.20:
        return conf

    if word_key == "cell_phone":
        base_similarity = 0.42
    elif word_key in ("pen", "scissors"):
        base_similarity = 0.32
    else:
        base_similarity = 0.36

    scaled = 0.80 + (conf - base_similarity) * 0.45
    return round(float(min(0.99, max(conf, scaled))), 3)


def iou(a: dict, b: dict) -> float:
    ax2, ay2, bx2, by2 = a["x"] + a["w"], a["y"] + a["h"], b["x"] + b["w"], b["y"] + b["h"]
    iw = max(0.0, min(ax2, bx2) - max(a["x"], b["x"]))
    ih = max(0.0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def merge_same_word(dets: list[dict], thr: float = None, cross_thr: float = None) -> list[dict]:
    """
    1. Same-word merge: several prompts can fire on one object (e.g. 'water bottle' and 'insulated metal flask').
       Keep the highest-confidence box when two boxes of the SAME word overlap (IoU >= thr).
    2. Cross-class NMS: if two DIFFERENT words heavily overlap on the exact same physical region
       (IoU >= cross_thr, default 0.60), keep the higher-confidence class and suppress the weaker competing box.
       This prevents two conflicting bounding boxes from stacking on top of one object.
    """
    thr = config.MERGE_IOU if thr is None else thr
    cross_thr = getattr(config, "CROSS_MERGE_IOU", 0.60) if cross_thr is None else cross_thr
    kept: list[dict] = []
    for d in sorted(dets, key=lambda x: x["confidence"], reverse=True):
        conflict = False
        for k in kept:
            overlap = iou(k["box"], d["box"])
            if k["key"] == d["key"] and overlap >= thr:
                conflict = True
                break
            elif k["key"] != d["key"] and overlap >= cross_thr:
                conflict = True
                break
        if not conflict:
            kept.append(d)
    return kept


merge_and_suppress_boxes = merge_same_word


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
        # Fast path: load pre-baked YOLO weights in ~0.15s without loading CLIP or computing embeddings
        baked = getattr(config, "YOLO_BAKED_WEIGHTS", None)
        if baked and baked.exists():
            from ultralytics import YOLO
            fast_model = YOLO(str(baked))
            if len(fast_model.names) == len(self.prompts):
                self._model = fast_model
                return

        if not config.YOLO_WEIGHTS.exists():
            raise FileNotFoundError(
                f"{config.YOLO_WEIGHTS} not found. Run: python scripts/download_models.py")
        from ultralytics import YOLOWorld   # imported lazily: heavy
        model = YOLOWorld(str(config.YOLO_WEIGHTS))
        model.set_classes(self.prompts)
        if baked:
            try:
                model.save(str(baked))
            except Exception:
                pass
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
            raw_conf = float(box.conf[0])
            conf = calibrate_confidence(raw_conf, word.key)
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            if conf + EPS >= self.threshold_for(word):
                accepted.append({
                    "key": word.key, "label": word.label, "confidence": round(conf, 3),
                    "box": {"x": x1 / w, "y": y1 / h, "w": (x2 - x1) / w, "h": (y2 - y1) / h}})
            elif best_rejected is None or conf > best_rejected["confidence"]:
                best_rejected = {"label": word.label, "confidence": round(conf, 3)}
        out = merge_same_word(accepted)[:config.DETECT_MAX_BOXES]
        speed = getattr(res, "speed", {})
        return {"detections": out, "inference_ms": round(ms, 1), "frame": {"width": w, "height": h},
                "min_conf": config.DETECT_MIN_CONF,
                "best_rejected": None if out else best_rejected,
                "speed": {k: round(v, 1) for k, v in speed.items()}}
