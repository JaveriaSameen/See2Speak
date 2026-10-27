# Test report

Run in a sandbox with **no webcam, no microphone, and no access to the Whisper, MediaPipe or
CLIP weight hosts**. So what passed is listed separately from what must still be tried on a laptop.

## Passed
| Area | How it was tested | Result |
|---|---|---|
| Answer matching (normalising, aliases, near-miss, pen≠pencil) | `pytest tests/` (26 unit tests) | pass |
| Scoring: +10 correct, 0 wrong, retry allowed, duplicate attempt id never scores twice, once-per-word rule, new session resets, round complete | `pytest` | pass |
| Mouth metrics on synthetic landmarks: scale invariance, no-face handling, still vs. open mouth, b/p/m lip-closure rule | `pytest` | pass |
| API plumbing (sessions, attempts, silence not counted, bad word/session rejected, history, frontend served) with stand-in models | `pytest tests/test_api.py` (7 tests) | pass |
| Whole browser flow in Chromium with Chrome's fake camera/mic and a stub backend: Start → boxes + labels → target locks → click other box switches target → Listen disabled while recording → Record/Stop → transcript, result, points → wrong answer gives +0 → repeat of a paid word gives +0 → Finish → summary → camera released | Playwright script | pass |
| Original slime still wanders while idle; after start it parks beside the target; happy / gentle-oops faces and bubble react to results | Playwright + screenshots | pass |
| Detection result parsing → normalised boxes (checked with standard YOLOv8n on a sample photo) | Python | pass |
| Audio edge cases: silence, too-short clip, non-WAV data, 48 kHz input | Python | pass |
| Model-download script reports blocked downloads with manual instructions | run | pass |

## NOT tested here (please try on your laptop)
- **YOLO-World v2 Small inference on real objects.** The weights file downloaded, but the model
  needs OpenAI CLIP's text-encoder weights, which this sandbox couldn't fetch. Detection accuracy
  for book/chair/phone/apple/pen/pencil, and frame rate, are therefore unmeasured.
- **Whisper `base.en` transcription** of real speech (weights not downloadable here).
- **MediaPipe Face Landmarker** on a real face, and therefore the real-world behaviour of the
  mouth thresholds in `backend/config.py` (starting values only, not tuned).
- **Browser text-to-speech** (the headless browser has no voices; the app showed its error
  message correctly instead of freezing). Real microphone capture quality.
- The proposal's evaluation targets (≥80 % labels, ≥80 % speech acceptance, ≥85 % mouth check,
  <3 s to show the word) are **not measured**. They need your collected evaluation recordings.
