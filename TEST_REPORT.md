# Test report

Environment: sandbox with **no webcam, no microphone, and no access to the CLIP, Whisper or MediaPipe
weight hosts**. Passed items are separated from items you must still try on a laptop.

## Passed (actually run)
| Area | How | Result |
|---|---|---|
| 15-class vocabulary: exact list, Chair gone, one distinct prompt each, round target 15 | pytest | pass |
| Detector passes all 15 prompts to YOLO-World in order; index -> word mapping; per-class thresholds; box normalisation (fake model object) | pytest `test_detection_classes.py` | pass |
| New words accept expected speech ("bottle", "paperclip", "scale", "tape", "marker pen"...) and look-alikes are not confused (pen/marker/pencil, book/notebook) | pytest | pass |
| Scoring: +10 / 0, retry, duplicate id, once-per-word, sessions, round complete | pytest | pass |
| Lip loops = the 40 MediaPipe lip landmarks (checked against MediaPipe's own `FACE_LANDMARKS_LIPS` table: 40/40 segments) | Python | pass |
| Calibration tracker: no face / open mouth give no progress; relaxed frames do; baseline = median; retry resets; "Play again" inherits baseline | pytest | pass |
| `/api/face`, `/api/calibration/start` plumbing | pytest | pass |
| Browser (Chromium, fake camera/mic): calibration panel texts, progress rises only with valid samples (7% -> 100%), "Calibration Complete", panel hides, baseline tick | Playwright | pass |
| Browser: lip dots/lines drawn at the mirrored landmark positions (20/20 outer, 20/20 inner), move when the mouth opens, stay aligned after window resize, cleared when the face is lost, live opening % shown | Playwright (canvas pixel checks) | pass |
| Browser: new classes through the UI: box + target word, Listen asks for the right word and tries `assets/audio/<key>.mp3`, correct speech +10, wrong 0, slime happy/gentle reactions | Playwright | pass |
| Layout at 1366x768: no scrolling, no horizontal overflow, slime stays inside the camera area and off the side panel | Playwright | pass |
| Earlier checks (matching, scoring, API, full browser flow, audio edge cases) | pytest + Playwright | pass |

Total: 54 pytest tests + 26 browser checks.

## IMPORTANT: what those tests did NOT use
- The lip-overlay and calibration browser tests used a **scripted stand-in for MediaPipe** that
  returns synthetic lip points and openings. They prove the browser/backend wiring, calibration logic
  and alignment maths, **not** that the real Face Landmarker finds a real face. `face_landmarker.task`
  could not be downloaded here.
- The detection tests used a **fake YOLO object**. They prove how the 15 classes are wired, **not**
  that YOLO-World can recognise them.

## NOT tested (please try on your laptop)
- **Real YOLO-World detection of any of the 15 classes**, so no class can be called reliable.
  (CLIP text-encoder weights are blocked here.) Likely weak spots, as expectations only: Paper clip,
  Pen/Pencil/Marker mix-ups, Book vs Notebook, Eraser, Glue stick.
- **Real MediaPipe lips on a real face**: dot/line placement, the 10 % relaxed-mouth limit, the
  7 s retry prompt, behaviour in poor light.
- Whisper transcription of real speech; browser text-to-speech voices; real microphone capture.
- The proposal's accuracy and speed targets.
