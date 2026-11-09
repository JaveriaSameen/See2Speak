"""One-click launcher for See2Speak.

Usage:
    python run.py
    python run.py --port 8000 --whisper tiny.en
    python run.py --min-conf 0.80

Checks prerequisites, model files, and starts the FastAPI server while
automatically opening the browser to http://localhost:8000.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


def check_python():
    version = sys.version_info
    if version.major == 3 and version.minor >= 14:
        print("[!] Warning: You are running Python 3.14+.")
        print("    PyTorch, Whisper, and MediaPipe prebuilt wheels are most stable on Python 3.10 - 3.12.")
        print("    If you encounter import errors, create a venv using Python 3.11: uv venv --python 3.11 .venv\n")


def check_models():
    required = ["yolov8s-worldv2.pt", "face_landmarker.task"]
    missing = [f for f in required if not (MODELS_DIR / f).exists()]
    if missing:
        print(f"[!] Missing model files in {MODELS_DIR}: {', '.join(missing)}")
        print("    Attempting to download them automatically...")
        try:
            import scripts.download_models as dl
            dl.main()
        except Exception as e:
            print(f"[!!] Automated download failed: {e}")
            print("     Please run: python scripts/download_models.py")
            sys.exit(1)


def open_browser(url: str, delay: float = 1.5):
    def _open():
        time.sleep(delay)
        print(f"[->] Opening browser at {url}")
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description="Run See2Speak server and game.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--whisper", default="", help="Whisper model override (e.g. tiny.en or base.en)")
    parser.add_argument("--min-conf", default="", help="Detection threshold override (default: 0.80)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser")
    args = parser.parse_args()

    check_python()
    check_models()

    if args.whisper:
        os.environ["SEE2SPEAK_WHISPER"] = args.whisper
    if args.min_conf:
        os.environ["SEE2SPEAK_MIN_CONF"] = args.min_conf

    url = f"http://localhost:{args.port}"
    print("=" * 60)
    print(" See2Speak – Computer-Vision Vocabulary Game")
    print(f" URL: {url}")
    print(" Press Ctrl+C in this terminal to stop.")
    print("=" * 60)

    if not args.no_browser:
        open_browser(url)

    try:
        import uvicorn
        uvicorn.run("backend.main:app", host=args.host, port=args.port, reload=False)
    except KeyboardInterrupt:
        print("\n[ok] Server stopped.")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()

