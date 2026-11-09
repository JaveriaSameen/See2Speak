"""Download the pretrained weights See2Speak needs (no training involved).

    python scripts/download_models.py

Places files in ./models :
  yolov8s-worldv2.pt     YOLO-World v2 Small            (Ultralytics GitHub release)
  face_landmarker.task   MediaPipe Face Landmarker      (Google storage)
Whisper base.en is fetched by the whisper package itself (cached in ~/.cache/whisper);
this script triggers that download and pre-warms YOLO-World + CLIP too so the first
game doesn't stall.
"""
from __future__ import annotations

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
        print(f"[ok] {name} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"[..] downloading {name}...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) See2Speak/0.2.0"}
    )
    try:
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 256  # 256 KB
            last_pct = -1

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = int(downloaded * 100 / total_size)
                    if pct != last_pct and pct % 10 == 0:
                        sys.stdout.write(f"\r[..] {name}: {pct}% ({downloaded / 1e6:.1f}/{total_size / 1e6:.1f} MB)")
                        sys.stdout.flush()
                        last_pct = pct

            if total_size > 0:
                sys.stdout.write("\n")

        tmp.rename(dest)
        print(f"[ok] {name} ready ({dest.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"\n[!!] could not download {name}: {e}\n     Download it manually from:\n     {url}\n     and save it as {dest}")


def warmup_yolo():
    try:
        from ultralytics import YOLOWorld
        from backend import config
        yolo_path = MODELS / "yolov8s-worldv2.pt"
        if not yolo_path.exists():
            return
        baked = MODELS / "yolov8s-see2speak.pt"
        print("[..] validating YOLO-World & pre-caching CLIP text encoder...")
        model = YOLOWorld(str(yolo_path))
        prompts = [p for w in config.VOCABULARY for p in w.prompts]
        model.set_classes(prompts)
        model.save(str(baked))
        print(f"[ok] YOLO-World baked weights saved ({baked.stat().st_size / 1e6:.1f} MB, {len(prompts)} prompts)")
    except Exception as e:
        print(f"[!!] YOLO-World warmup skipped: {e}")


def warmup_whisper():
    try:
        import whisper
        from backend import config
        print(f"[..] fetching Whisper {config.WHISPER_MODEL} (cached by whisper)...")
        whisper.load_model(config.WHISPER_MODEL)
        print(f"[ok] Whisper {config.WHISPER_MODEL} ready")
    except Exception as e:
        print(f"[!!] Whisper download skipped: {e}")


def main():
    MODELS.mkdir(exist_ok=True)
    for name, url in FILES.items():
        fetch(name, url)
    warmup_whisper()
    warmup_yolo()


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
