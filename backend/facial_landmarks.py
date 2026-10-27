"""Visible mouth-movement check using MediaPipe Face Landmarker.

IMPORTANT LIMITS (from the proposal): this is NOT lip reading. It only asks
"did the lips visibly open / change width compared with a neutral mouth?".
It cannot tell live speech from played-back audio or a lip-sync attempt, and
it never decides whether the answer is correct. It does not affect points.

Measurements (all divided by inter-ocular distance => scale invariant):
  opening = |inner upper lip (13) - inner lower lip (14)|
  width   = |left mouth corner (78) - right mouth corner (308)|
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import config

UPPER, LOWER, LEFT, RIGHT = 13, 14, 78, 308
EYE_L, EYE_R = 33, 263          # outer eye corners


@dataclass
class MouthSample:
    opening: float
    width: float


def metrics_from_points(pts: np.ndarray) -> Optional[MouthSample]:
    """pts: (N,2) landmark array in any consistent unit. Pure function (testable)."""
    iod = float(np.linalg.norm(pts[EYE_L] - pts[EYE_R]))
    if iod < 1e-6:
        return None
    return MouthSample(
        opening=float(np.linalg.norm(pts[UPPER] - pts[LOWER])) / iod,
        width=float(np.linalg.norm(pts[LEFT] - pts[RIGHT])) / iod)


def analyse_samples(attempt: list[Optional[MouthSample]], baseline: Optional[MouthSample],
                    target_label: str) -> dict:
    """Decide 'visible movement' from per-frame samples. Pure function (testable)."""
    total = len(attempt)
    valid = [s for s in attempt if s is not None]
    if len(valid) < config.MOUTH_MIN_FRAMES:
        return {"status": "no_face", "frames": total, "valid_frames": len(valid),
                "message": "I couldn't see your face clearly, so the mouth check was skipped. "
                           "Face the camera in good light."}
    op = np.array([s.opening for s in valid])
    wd = np.array([s.width for s in valid])
    if baseline is not None:
        base_op, base_wd, base_src = baseline.opening, baseline.width, "calibrated"
    else:                        # fallback: smallest opening seen in the attempt
        base_op, base_wd, base_src = float(op.min()), float(np.median(wd)), "estimated"

    open_delta = float(op.max() - base_op)
    width_delta = float(np.max(np.abs(wd - base_wd)))
    moved = open_delta >= config.MOUTH_OPEN_DELTA or width_delta >= config.MOUTH_WIDTH_DELTA

    first = target_label.strip().lower()[:1]
    closure = None
    if first in config.LIP_CLOSURE_LETTERS:
        # lips should meet around the start of a b/p/m word: look at the first third
        head = op[: max(2, len(op) // 3)]
        closure = bool(head.min() <= base_op + config.MOUTH_CLOSURE_MAX)

    if moved:
        msg = "I saw your mouth move while you spoke."
        status = "movement"
    else:
        msg = ("I didn't see much mouth movement. Try saying the word clearly "
               "while facing the camera.")
        status = "no_movement"
    if closure is not None:
        msg += (f" Your lips {'did' if closure else 'did not'} close at the start "
                f"(words starting with '{first}' need that).")
    return {"status": status, "moved": moved, "lip_closure": closure,
            "open_delta": round(open_delta, 4), "width_delta": round(width_delta, 4),
            "baseline": base_src, "frames": total, "valid_frames": len(valid),
            "message": msg + " (This is only a visual check, not a pronunciation score.)"}


class MouthAnalyser:
    def __init__(self):
        self._landmarker = None
        self._lock = threading.Lock()

    def load(self):
        if self._landmarker is not None:
            return
        if not config.FACE_LANDMARKER_TASK.exists():
            raise FileNotFoundError(
                f"{config.FACE_LANDMARKER_TASK} not found. Run: python scripts/download_models.py")
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(config.FACE_LANDMARKER_TASK)),
            running_mode=vision.RunningMode.IMAGE, num_faces=1)
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)

    @property
    def ready(self) -> bool:
        return self._landmarker is not None

    def sample(self, jpeg_bytes: bytes) -> Optional[MouthSample]:
        import mediapipe as mp
        self.load()
        bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        with self._lock:
            res = self._landmarker.detect(img)
        if not res.face_landmarks:
            return None
        lm = res.face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)  # pixel units: keeps aspect ratio
        return metrics_from_points(pts)

    def baseline(self, frames: list[bytes]) -> Optional[MouthSample]:
        """Neutral-mouth baseline = median over calibration frames."""
        samples = [s for s in (self.sample(f) for f in frames) if s]
        if len(samples) < config.MOUTH_MIN_FRAMES:
            return None
        return MouthSample(float(np.median([s.opening for s in samples])),
                           float(np.median([s.width for s in samples])))

    def analyse(self, frames: list[bytes], baseline: Optional[MouthSample], target_label: str) -> dict:
        return analyse_samples([self.sample(f) for f in frames], baseline, target_label)
