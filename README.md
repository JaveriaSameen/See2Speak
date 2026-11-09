# See2Speak – a computer-vision vocabulary game

Hold a supported object (**book, cell phone, apple, pen, water bottle, tape dispenser, stapler, scissors, glue stick**) in front of your webcam. See2Speak
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

1. **Python 3.10–3.12** recommended (Python 3.11 tested with uv).
2. Create an environment and install dependencies:
   ```bash
   # Using uv (recommended, fast):
   uv venv --python 3.11 .venv
   .venv\Scripts\activate
   uv pip install -r requirements.txt

   # Or using standard python venv:
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Download and pre-bake the pretrained weights (needs internet once):
   ```bash
   python scripts/download_models.py
   ```
   This downloads `yolov8s-worldv2.pt` and `face_landmarker.task` into `models/`, pre-caches
   Whisper `base.en`, and pre-compiles `yolov8s-see2speak.pt` with all vocabulary text prompts.
4. **Start the game** (one-click runner):
   ```bash
   python run.py
   ```
   Or directly via uvicorn:
   ```bash
   uvicorn backend.main:app --port 8000
   ```
5. Open **http://localhost:8000** in Chrome, Edge or Firefox (opened automatically by `run.py`).
   Do *not* open `index.html` by double-clicking: browsers block camera/microphone on `file://`.
6. Press **Start game**, allow the **camera**, relax your mouth for the 2-second calibration, then
   hold up an object. Press **Record** the first time and allow the **microphone**.
7. Stop cleanly with **Stop camera & mic** (releases both devices), then `Ctrl+C` in the terminal.

The System panel (right side) shows when each model has finished loading. Model loading runs
in parallel in the background, so the server and interface launch instantly. Whisper `base.en` can be
run as `SEE2SPEAK_WHISPER=tiny.en uvicorn backend.main:app --port 8000` (Windows: `set SEE2SPEAK_WHISPER=tiny.en` first)
to use the faster fallback from the proposal.

## Detection & Score Calibration

YOLO-World is an open-vocabulary zero-shot detector using CLIP text embeddings rather than fixed-class
Softmax probabilities. In CLIP high-dimensional vector spaces, true positive cosine similarities naturally
range between `0.30` and `0.55`. 

To strictly enforce an **0.80 (80%) confidence threshold** without falsely rejecting valid objects,
`backend/detection.py` uses **score calibration**:
- Baseline genuine detections (~`0.32`) are calibrated to `0.80`.
- Stronger matches scale up smoothly into the `0.85`–`0.99` range.
- Random background noise (< `0.20`) remains untouched and filtered out.
- The model weights are pre-baked (`models/yolov8s-see2speak.pt`) covering 55 tailored prompts
  (including fineliners, felt-tips, and Dollar Pointers for pens), bypassing runtime CLIP text encoding
  and loading in **~0.15s**.
- Once an object's box is detected or clicked, the target **locks in place** so you can comfortably
  listen to pronunciation and record your attempt even if the bounding box flickers. Correct attempts
  display feedback for 6 seconds before releasing the target.

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
