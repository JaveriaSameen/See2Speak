# See2Speak

See2Speak is a webcam-based vocabulary game. Show an object to the camera, hear its name, say the word, and earn points for a matching answer. The app combines object detection, local speech recognition, and optional mouth-movement feedback in a small browser game.

## Vocabulary

The game currently recognizes these six targets:

**Book · Cell phone · Apple · Pen · Water bottle · Scissors**

The vocabulary, detection prompts, accepted spoken forms, and thresholds are configured in [`backend/config.py`](backend/config.py).

## How a round works

1. The browser sends webcam frames to the local server for object detection.
2. A detected object is shown with a labelled box. If several objects appear, select the one you want to practice.
3. Choose **Listen to Pronunciation** to hear the target. The game uses an optional MP3 recording when present, otherwise it uses the browser's English text-to-speech voice.
4. Choose **Record** and say the word. The browser records a short audio clip and sends it to the server.
5. Whisper transcribes the clip locally. The app normalizes the transcript and compares it with the target's accepted forms.
6. A correct answer earns 10 points. Each word pays out once per session by default. A close answer can receive “Close!” feedback but does not score as correct.

MediaPipe can also estimate mouth movement from webcam frames. The player calibrates a relaxed-mouth baseline at the start. This result is supplementary feedback: it does not change scoring, identify what was said, or verify that a person spoke live.

## Requirements

- Python 3.10–3.12 recommended (the repository's `.python-version` is 3.11).
- A webcam and microphone.
- A modern browser such as Chrome, Edge, or Firefox.
- Internet access for the initial model download.
- Enough disk space and memory for PyTorch, YOLO-World, Whisper, and MediaPipe.

All inference runs locally after setup; the application does not send audio to a hosted transcription service.

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
uv venv --python 3.11 .venv
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Or use Python's built-in environment manager and pip:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

Download the required model assets:

```powershell
python scripts/download_models.py
```

The model files are stored in `models/` and are intentionally excluded from Git. The setup script also prepares the local Whisper model and detector assets as supported by the installed dependencies.

## Run

Start the app with the launcher:

```powershell
python run.py
```

The launcher checks for model files, starts the server at `http://127.0.0.1:8000`, and opens the page in a browser. Use `Ctrl+C` in the terminal to stop it.

Useful options:

```powershell
python run.py --no-browser
python run.py --port 8080
python run.py --whisper tiny.en
python run.py --min-conf 0.80
```

Alternatively, start the server directly:

```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open the page through `http://localhost` or `http://127.0.0.1`. Do not open `frontend/index.html` as a file: browsers restrict camera and microphone access on `file://` pages. Allow camera access when prompted; allow microphone access when you first record.

## Models and configuration

- **Object detection:** Ultralytics YOLO-World, using the project's configured pretrained weights and vocabulary prompts.
- **Speech recognition:** OpenAI Whisper, defaulting to `base.en`. Set `SEE2SPEAK_WHISPER=tiny.en` or pass `--whisper tiny.en` for a faster, less accurate option.
- **Mouth feedback:** MediaPipe Face Landmarker.
- **Persistence:** SQLite session and score history under `data/`.

Models load in the background after server startup. The interface's system panel reports their status. If a model is missing, run `python scripts/download_models.py`.

To change the words or scoring behavior, edit `backend/config.py`. Detection prompts and accepted spoken aliases are defined per vocabulary entry. The default correct-answer award is 10 points and `AWARD_ONCE_PER_WORD` limits each word to one payout per session. The mouth movement thresholds are also configurable there.

## Optional pronunciation recordings

Place MP3 files in `frontend/assets/audio/`, named after the vocabulary key in `backend/config.py`, for example:

```text
book.mp3  cell_phone.mp3  apple.mp3  pen.mp3  water_bottle.mp3  scissors.mp3
```

If a file is missing, the browser's built-in speech synthesis is used.

## Evaluation and System Benchmark

To evaluate live object detection performance and compute formal computer vision metrics (Precision, Recall, F1-Score, Accuracy, and Latency), run the evaluation tool:

```powershell
python scripts/evaluate_system.py --trials 10
```

### Benchmark Features
- **Widescreen 1280×720 UI**: High-visibility camera canvas with large text, prominent status banners, and live bounding box overlays.
- **Asynchronous Threading (30+ FPS)**: Runs camera capture and UI rendering independently from YOLO-World inference, ensuring a smooth webcam feed with zero motion lag.
- **Interactive Session Control**:
  - **`[START]` (Space/Enter)**: Opens each object in preparation mode so you can position the object and verify detection before testing starts.
  - **`[PAUSE]` / `[RESUME]` (Space/P)**: Pauses the automated timer at any moment to adjust positioning.
  - **`[NEXT CLASS]` (N)**: Moves to the next vocabulary class at your own pace.
  - **`[QUIT]` (Q)**: Exits early and immediately calculates the summary table.
- **Automated Paced Capture**: Records trials at configurable intervals (default: 2.0s) with an 0.8-second visual freeze confirmation displaying the logged box and HIT/MISS status.
- **Automated Report Screenshots**: Automatically saves high-resolution, annotated bounding-box images of your best detections into `docs/screenshots/` (e.g. `detection_book.png`, `detection_apple.png`, `detection_negative_control.png`). You can also press **`[S]`** or click **`[SNAPSHOT]`** anytime to capture an instant image.
- **Negative Control Testing**: Evaluates background and unsupported everyday objects (e.g., mugs, keys, empty hands) to measure rejection specificity.
- **Automated Metrics Table**: Generates a complete computer vision evaluation table ready for project reports:
  - Per-class Precision, Recall, F1-Score, and Accuracy (at $\ge 0.80$ confidence)
  - True Negative Specificity for Negative Control
  - Average detection confidence and inference latency in milliseconds
  - Overall Mean across all evaluated trials

### Command Line Options
```powershell
# Custom trial count per object (e.g., 15 trials per class and 15 negative control trials)
python scripts/evaluate_system.py --trials 15

# Custom capture interval (e.g., 2.5 seconds between automatic captures)
python scripts/evaluate_system.py --trials 10 --interval 2.5
```

### Speech Recognition Benchmark

To evaluate Whisper `base.en` speech transcription and the game's answer matching logic across the 5 evaluation categories (Exact Match, Alias Match, Near Match, Wrong Word, Silence/Noise) in a dedicated **1280×720 graphical window**, run:

```powershell
python scripts/evaluate_speech.py --trials 10
```

- **Graphical Guidance Window (1280×720)**: Displays target words, instructions on what to say, and the expected system decision on a sleek presentation canvas.
- **Interactive UI Buttons**: Click **`[RECORD / NEXT]`** (Space), **`[RETRY]`** (R), **`[SNAPSHOT]`** (S), **`[NEXT CAT]`** (N), and **`[QUIT]`** (Q).
- **Automated Category Screenshots**: Automatically saves high-resolution verification screenshots to `docs/screenshots/` (e.g. `speech_exact_match.png`, `speech_alias_match.png`) for direct insertion into your report.
- **Metrics Table**: Outputs the complete empirical decision match table matching **Table 2** of the project report.

### Facial Landmark & Mouth Movement Benchmark

To evaluate MediaPipe Face Landmarker, neutral-mouth baseline calibration, movement sensitivity ($\Delta \ge 0.05$), and bilabial lip closure ('B' and 'P') in a **1280×720 split-screen interactive window**, run:

```powershell
python scripts/evaluate_face.py
```

- **Interactive Split-Screen Window (1280×720)**: Shows your live camera feed overlaid with the real-time **cyan/magenta lip mesh** and **yellow eye tracking line** alongside numerical metrics and gauge bars.
- **Three Progressive Evaluation Phases**:
  1. *Neutral-Mouth Calibration*: Verifies baseline convergence across 20 stable frames.
  2. *Movement Detection Sensitivity*: Tests speech movement vs. resting mouth controls.
  3. *Bilabial Lip Closure*: Tests initial pre-phonation closure for 'B' & 'P' vocabulary words.
- **Automated Screenshots**: Saves high-resolution verification images into `docs/screenshots/` (`face_calibration_mesh.png`, `face_speech_movement.png`, `face_bilabial_closure.png`).
- **Metrics Table**: Outputs the empirical evaluation table matching **Table 3** of the project report.

## Troubleshooting

| Issue | Try this |
|---|---|
| The page cannot reach the server | Start `python run.py` or `uvicorn backend.main:app --port 8000`; open the matching localhost address. |
| Camera or microphone is unavailable | Use localhost, allow the browser permission, and close other apps using the device. |
| A model is missing | Run `python scripts/download_models.py`. |
| YOLO-World reports a CLIP dependency error | Follow the Ultralytics/CLIP installation instructions for your environment. |
| Whisper is too slow | Select `tiny.en` with `python run.py --whisper tiny.en`. |
| No object is detected | Improve lighting, use a plain background, and hold the object closer to the camera. |
| Speech is not recognized | Speak clearly near the selected microphone and reduce background noise. |
| No pronunciation audio plays | Check system volume or add the word's MP3 file under `frontend/assets/audio/`. |
