"""See2Speak backend package.

Provides FastAPI application endpoints, object detection with YOLO-World,
speech recognition with OpenAI Whisper, facial landmark mouth analysis
with MediaPipe, and game scoring logic backed by SQLite.
"""
from __future__ import annotations

from .main import app

__version__ = "0.2.0"
__all__ = ["app", "__version__"]

