"""Object detection with the pretrained YOLO-World v2 Small model.

No training / fine-tuning: the vocabulary is given to the model as text
prompts via set_classes(). Boxes are returned normalised (0..1) so the
frontend can draw them on any video size.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from . import config


class Detector:
    def __init__(self):
        self._model = None
        self._lock = threading.Lock()   # one inference at a time
        self.vocab = list(config.VOCABULARY)

    def load(self):
        if self._model is not None:
            return
        if not config.YOLO_WEIGHTS.exists():
            raise FileNotFoundError(
                f"{config.YOLO_WEIGHTS} not found. Run: python scripts/download_models.py")
        from ultralytics import YOLOWorld   # imported lazily: heavy
        model = YOLOWorld(str(config.YOLO_WEIGHTS))
        model.set_classes([w.yolo_prompt for w in self.vocab])
        self._model = model

    @property
    def ready(self) -> bool:
        return self._model is not None

    def detect(self, jpeg_bytes: bytes) -> dict:
        self.load()
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
        h, w = frame.shape[:2]
        floor = min(v.min_conf for v in self.vocab)
        t0 = time.perf_counter()
        with self._lock:
            res = self._model.predict(frame, imgsz=config.DETECT_IMGSZ, conf=floor, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000

        out = []
        for box in res.boxes:
            idx = int(box.cls[0])
            conf = float(box.conf[0])
            word = self.vocab[idx]
            if conf < word.min_conf:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            out.append({
                "key": word.key, "label": word.label, "confidence": round(conf, 3),
                "box": {"x": x1 / w, "y": y1 / h, "w": (x2 - x1) / w, "h": (y2 - y1) / h},
            })
        out.sort(key=lambda d: d["confidence"], reverse=True)
        return {"detections": out[:config.DETECT_MAX_BOXES], "inference_ms": round(ms, 1),
                "frame": {"width": w, "height": h}}
