# See2Speak – a computer-vision vocabulary game

Hold a **book, chair, cell phone, apple, pen or pencil** in front your laptop webcam. See2Speak
detects it, draws a labelled box, lets you hear the word, records you saying it, transcribes your
speech locally, and awards points when the transcript matches. Your slime buddy
(the original animated character) wanders the screen and reacts to what happens.

| Part | Technology (all pretrained – no training or fine-tuning) |
|---|---|
| Object detection | YOLO-World v2 Small (`yolov8s-worldv2.pt`) via Ultralytics |
| Speech → text | OpenAI Whisper `base.en`, local on CPU (`tiny.en` fallback) |
| Mouth movement check | MediaPipe Face Landmarker |
| Backend | Python, FastAPI, SQLite (scores) |
| Frontend | the original slime HTML, extended (HTML/CSS/JS, no build step) |

## Project layout

```
See2Speak/
├── frontend/            index.html, style.css, script.js, assets/audio/ (optional recordings)
├── backend/
│   ├── main.py              FastAPI app + endpoints, serves the frontend
│   ├── config.py            vocabulary, thresholds, points  ← edit here
│   ├── detection.py         YOLO-World
│   ├── speech_recognition.py Whisper
│   ├── facial_landmarks.py  MediaPipe mouth opening / width
│   └── game_logic.py        matching, scoring, SQLite
├── scripts/download_models.py
├── models/              yolov8s-worldv2.pt, face_landmarker.task go here
├── tests/               pytest (matching, scoring, mouth logic, API)
└── requirements.txt
```

## Setup

1. **Python 3.10–3.12** recommended (PyTorch/Whisper wheels are most reliable there).
2. Create an environment and install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Download the pretrained weights (needs internet once):
   ```bash
   python scripts/download_models.py
   ```
   This puts `yolov8s-worldv2.pt` and `face_landmarker.task` in `models/` and caches Whisper
   `base.en`. If you already have `yolov8s-worldv2.pt`, just copy it into `models/`.
   First use of YOLO-World also installs OpenAI's CLIP text encoder automatically (Ultralytics
   does this; it needs `git` and internet the first time).
4. **Start the server** (from the `See2Speak/` folder):
   ```bash
   uvicorn backend.main:app --port 8000
   ```
5. Open **http://localhost:8000** in Chrome, Edge or Firefox.
   Do *not* open `index.html` by double-clicking: browsers block camera/microphone on `file://`.
6. Press **Start game**, allow the **camera**, relax your mouth for the 2-second calibration, then
   hold up an object. Press **Record** the first time and allow the **microphone**.
7. Stop cleanly with **Stop camera & mic** (releases both devices), then `Ctrl+C` in the terminal.

The System panel (right side) shows when each model has finished loading. Whisper `base.en` is
slow on weak CPUs; run `SEE2SPEAK_WHISPER=tiny.en uvicorn backend.main:app --port 8000`
(Windows: `set SEE2SPEAK_WHISPER=tiny.en` first) to use the faster fallback from the proposal.

## How a round works

1. Frames from the webcam are sent (~3–6 per second, one at a time) to `/api/detect`.
2. An object must appear in **2 consecutive detections** before it is "locked", so single-frame
   flickers never trigger anything. The target changes only when the locked object changes, and
   it is frozen while you record/process. Click a box to pick one when several are visible.
3. **Listen to Pronunciation** plays `frontend/assets/audio/<word>.mp3` if you provide one,
   otherwise the browser's built-in English text-to-speech. It can't overlap with recording.
4. **Record** captures microphone audio (auto-stops after 4 s) and, at the same time, small
   webcam frames for the mouth check. Audio is converted to 16 kHz WAV in the browser and
   sent to `/api/attempt`; nothing is stored on disk (audio is processed in memory).
5. Whisper transcribes → text is lowercased, punctuation removed, a leading *a/an/the* dropped →
   compared **exactly** with the word (or its accepted forms, e.g. *phone* for "Cell phone").
6. Correct → **+10 points**. Wrong, silent or failed → **0**. Edit distance ≤ 1 only changes the
   feedback ("Close!") and never counts as correct.

### Scoring rules (all in `backend/config.py`)

- `POINTS_CORRECT = 10`.
- Each attempt has a unique id; re-sending it can't score twice (server-side).
- `AWARD_ONCE_PER_WORD = True`: a word pays out once per session so re-recording the same
  object can't farm points. Later correct answers still count as correct attempts but give 0.
  Set it to `False` if you want every correct attempt to pay.
- Silent/unintelligible recordings and errors are **not** counted as attempts.
- Progress = distinct words collected (of 6). Finishing all six, or **Finish**, shows the summary
  (with past sessions from SQLite in `data/see2speak.sqlite3`). **Play again** starts a new session at 0.

### Mouth-movement check (supplementary)

MediaPipe Face Landmarker gives lip landmarks. Inner-lip **opening** and mouth **width** are
divided by the inter-ocular distance (so face size/camera distance matter less) and compared
with a neutral-mouth baseline taken in the calibration step (**Recalibrate mouth** repeats it).
For words starting with b/p/m it also reports whether the lips visibly closed at the start.
It is shown as a hint only: **it never changes the score**, it is not lip-reading, and it cannot
tell live speech from played-back audio or lip-syncing.

## Adding or changing words

Edit `VOCABULARY` in `backend/config.py` (label, YOLO text prompt, accepted spoken forms,
detection threshold). The frontend reads the list from the server, so nothing else changes.

## Known limitations

- A matching transcript means Whisper *heard* the word. It does not prove each sound was
  pronounced correctly.
- Detection depends on lighting, distance, background and laptop speed. **Pens and pencils are
  thin and the hardest to detect** (their threshold is lowered, so expect more mistakes).
- Background noise and accents affect Whisper. Short single words are harder than sentences.
- The mouth thresholds in `config.py` are starting values. The proposal calls for tuning them on
  separate recordings; this has not been done.
- If the face isn't visible, the mouth check is skipped with a message; the answer is still scored.
- The browser's text-to-speech voice depends on the operating system.

## Tests

```bash
pytest -q
```
Covers answer matching, scoring/idempotency, session summary, mouth-metric logic on synthetic
landmarks, and API plumbing with stand-in models. **These tests do not exercise the real
models, webcam or microphone** – see the test report you were given for what was and wasn't run.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Can't reach the See2Speak server" | Start `uvicorn backend.main:app --port 8000` and open http://localhost:8000 |
| Camera/mic doesn't start | Use `localhost` (not a file or LAN IP), allow the permission in the address bar, close other apps using the camera |
| System panel shows `error: … not found` | Run `python scripts/download_models.py` or place the file in `models/` |
| YOLO-World fails mentioning `clip` | `pip install git+https://github.com/ultralytics/CLIP.git` (needs git) |
| Whisper slow | `SEE2SPEAK_WHISPER=tiny.en` |
| Nothing detected | More light, plain background, hold the object 30–60 cm away, fill ~⅓ of the frame |
| "I couldn't hear anything" | Pick the right microphone in the browser permission menu; speak closer |
| No pronunciation sound | Check volume; add `frontend/assets/audio/<word>.mp3`; try Chrome/Edge |
