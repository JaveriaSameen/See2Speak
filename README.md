# See2Speak

See2Speak is a webcam-based vocabulary game. Show an object to the camera, hear its name, say the word, and earn points for a matching answer. The app combines object detection, local speech recognition, and optional mouth-movement feedback in a small browser game.

## Vocabulary

The game currently recognizes these nine targets:

**Book · Cell phone · Apple · Pen · Water bottle · Tape dispenser · Stapler · Scissors · Glue stick**

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
book.mp3  cell_phone.mp3  apple.mp3  pen.mp3  water_bottle.mp3
tape_dispenser.mp3  stapler.mp3  scissors.mp3  glue_stick.mp3
```

If a file is missing, the browser's built-in speech synthesis is used.

## Tests and current limitations

Run the Python test suite with:

```powershell
pytest -q
```

The tests cover answer matching, scoring and duplicate-attempt handling, session history, detector prompt wiring, mouth metric calculations, calibration logic, and API behavior. See [`TEST_REPORT.md`](TEST_REPORT.md) for the recorded results and coverage details.

Automated tests use stand-in detector and face models. They verify integration and application logic; they do **not** establish real-world recognition accuracy. Try the complete flow on the target laptop with its webcam, microphone, and downloaded models. Detection can vary with lighting, camera distance, background, and hardware. Thin objects such as pens may be harder to detect. Whisper results can vary with accent, microphone quality, and background noise. Mouth movement feedback depends on face visibility and lighting and is not used to decide whether an answer is correct.

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
