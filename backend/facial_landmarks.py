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

# The two closed lip loops (verified against MediaPipe's FACE_LANDMARKS_LIPS table). Only these
# 40 landmarks are ever sent to the browser, so the overlay stays focused on the lips.
LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]


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


def lip_outline(norm_pts: np.ndarray) -> dict:
    """norm_pts: (N,2) landmarks in 0..1 image coordinates -> the lip loops to draw."""
    return {"outer": [[round(float(norm_pts[i][0]), 4), round(float(norm_pts[i][1]), 4)] for i in LIP_OUTER],
            "inner": [[round(float(norm_pts[i][0]), 4), round(float(norm_pts[i][1]), 4)] for i in LIP_INNER]}


class CalibrationTracker:
    """Neutral-mouth calibration driven ONLY by real landmark samples.

    A sample counts when a face was found AND the mouth is relaxed/closed
    (opening <= MOUTH_CALIB_MAX_OPEN). Progress = counted / needed, so it
    only moves while the landmarks are actually detected. The baseline is the
    median of the counted samples.
    """

    def __init__(self, needed: int = None, max_open: float = None):
        self.needed = needed or config.MOUTH_CALIB_FRAMES
        self.max_open = max_open if max_open is not None else config.MOUTH_CALIB_MAX_OPEN
        self.samples: list[MouthSample] = []
        self.baseline: Optional[MouthSample] = None

    def add(self, sample: Optional[MouthSample]) -> dict:
        if self.baseline is not None:
            return self.status("done", "Calibration complete")
        if sample is None:
            return self.status("no_face", "Keep your face visible")
        if sample.opening > self.max_open:
            return self.status("mouth_open", "Relax your mouth and keep it gently closed")
        self.samples.append(sample)
        if len(self.samples) >= self.needed:
            self.baseline = MouthSample(float(np.median([x.opening for x in self.samples])),
                                        float(np.median([x.width for x in self.samples])))
            return self.status("done", "Calibration complete")
        return self.status("measuring", "Calibrating...")

    def status(self, state: str, message: str) -> dict:
        n = len(self.samples)
        return {"state": state, "message": message, "collected": n, "needed": self.needed,
                "progress": round(min(1.0, n / self.needed), 3), "complete": self.baseline is not None,
                "baseline_opening": None if self.baseline is None else round(self.baseline.opening, 4),
                "baseline_width": None if self.baseline is None else round(self.baseline.width, 4)}


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

    def _landmarks(self, jpeg_bytes: bytes):
        """-> (pixel_pts, normalised_pts) for the first face, or None."""
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
        norm = np.array([[p.x, p.y] for p in res.face_landmarks[0]], dtype=np.float32)
        return norm * np.array([w, h], dtype=np.float32), norm   # pixel units keep the aspect ratio

    def sample(self, jpeg_bytes: bytes) -> Optional[MouthSample]:
        lm = self._landmarks(jpeg_bytes)
        return None if lm is None else metrics_from_points(lm[0])

    def track(self, jpeg_bytes: bytes) -> tuple[dict, Optional[MouthSample]]:
        """One live frame: lip outline to draw + the measured opening/width."""
        lm = self._landmarks(jpeg_bytes)
        if lm is None:
            return {"face": False}, None
        sample = metrics_from_points(lm[0])
        if sample is None:
            return {"face": False}, None
        return {"face": True, "lips": lip_outline(lm[1]),
                "opening": round(sample.opening, 4), "width": round(sample.width, 4)}, sample

    def baseline(self, frames: list[bytes]) -> Optional[MouthSample]:
        """Neutral-mouth baseline = median over calibration frames."""
        samples = [s for s in (self.sample(f) for f in frames) if s]
        if len(samples) < config.MOUTH_MIN_FRAMES:
            return None
        return MouthSample(float(np.median([s.opening for s in samples])),
                           float(np.median([s.width for s in samples])))

    def analyse(self, frames: list[bytes], baseline: Optional[MouthSample], target_label: str) -> dict:
        return analyse_samples([self.sample(f) for f in frames], baseline, target_label)
