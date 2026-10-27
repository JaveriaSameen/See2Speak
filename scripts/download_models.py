"""Download the pretrained weights See2Speak needs (no training involved).

    python scripts/download_models.py

Places files in ./models :
  yolov8s-worldv2.pt     YOLO-World v2 Small            (Ultralytics GitHub release)
  face_landmarker.task   MediaPipe Face Landmarker      (Google storage)
Whisper base.en is fetched by the whisper package itself (cached in ~/.cache/whisper);
this script triggers that download too so the first game doesn't stall.
"""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
FILES = {
    "yolov8s-worldv2.pt":
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt",
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}


def fetch(name: str, url: str):
    dest = MODELS / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[ok] {name} already present")
        return
    print(f"[..] downloading {name}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
        print(f"[ok] {name} ({dest.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"[!!] could not download {name}: {e}\n     Download it manually from:\n     {url}\n     and save it as {dest}")


def main():
    MODELS.mkdir(exist_ok=True)
    for name, url in FILES.items():
        fetch(name, url)
    try:
        import whisper
        from backend import config
        print(f"[..] fetching Whisper {config.WHISPER_MODEL} (cached by whisper)")
        whisper.load_model(config.WHISPER_MODEL)
        print("[ok] Whisper ready")
    except Exception as e:
        print(f"[!!] Whisper download skipped: {e}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
