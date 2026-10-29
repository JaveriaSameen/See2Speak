"""See2Speak FastAPI backend.   Run:  uvicorn backend.main:app --port 8000

Serves the frontend too, so the page is loaded from http://localhost:8000
(browsers only allow webcam/microphone on https or localhost, not file://).
"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .detection import Detector
from .facial_landmarks import CalibrationTracker, MouthAnalyser, MouthSample
from .game_logic import GameStore, evaluate_answer
from .speech_recognition import AudioError, Transcriber

log = logging.getLogger("see2speak")


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_load_models, daemon=True).start()   # server starts instantly
    yield


app = FastAPI(title="See2Speak", lifespan=lifespan)
# Lets the page be opened from another local dev server (index.html?api=http://localhost:8000).
app.add_middleware(CORSMiddleware, allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
                   allow_methods=["*"], allow_headers=["*"])

store = GameStore(config.DB_PATH)
detector = Detector()
transcriber = Transcriber()
mouth = MouthAnalyser()
baselines: dict[str, MouthSample] = {}     # neutral-mouth baseline per session
calibrators: dict[str, CalibrationTracker] = {}
model_status: dict[str, str] = {"detector": "loading", "speech": "loading", "mouth": "loading"}


def _load_models():
    """Load models in the background so the server starts instantly."""
    for name, obj in (("detector", detector), ("speech", transcriber), ("mouth", mouth)):
        try:
            obj.load()
            model_status[name] = "ready"
        except Exception as e:     # keep serving; report the reason to the UI
            log.exception("Failed to load %s", name)
            model_status[name] = f"error: {e}"


# ------------------------------------------------------------------ meta
@app.get("/api/health")
def health():
    return {"models": model_status, "whisper_model": config.WHISPER_MODEL}


@app.get("/api/config")
def get_config():
    return {
        "points_correct": config.POINTS_CORRECT,
        "round_target": config.ROUND_TARGET,
        "vocabulary": [{"key": w.key, "label": w.label} for w in config.VOCABULARY],
    }


# --------------------------------------------------------------- sessions
@app.post("/api/session")
def new_session(inherit_from: str = Form("")):
    sid = store.start_session()
    if inherit_from in baselines:            # "Play again" keeps the calibrated neutral mouth
        baselines[sid] = baselines[inherit_from]
    return {"session_id": sid, **store.stats(sid)}


def _need_session(sid: str):
    if not store.session_exists(sid):
        raise HTTPException(404, "Unknown session. Start a new game.")


@app.get("/api/session/{sid}")
def session_stats(sid: str):
    _need_session(sid)
    return store.stats(sid)


@app.post("/api/session/{sid}/end")
def end_session(sid: str):
    _need_session(sid)
    calibrators.pop(sid, None)          # (baseline is kept so Play again can inherit it)
    return store.end_session(sid)


@app.get("/api/history")
def history():
    return {"sessions": store.history()}


# -------------------------------------------------------------- detection
@app.post("/api/detect")
def detect(image: UploadFile = File(...)):
    if model_status["detector"] != "ready":
        raise HTTPException(503, f"Object detector not ready ({model_status['detector']})")
    try:
        return detector.detect(image.file.read())
    except ValueError as e:
        raise HTTPException(400, str(e))


# ---------------------------------------------------------- mouth baseline
@app.post("/api/calibrate")
def calibrate(session_id: str = Form(...), frames: List[UploadFile] = File(...)):
    """Neutral-mouth calibration frames (user keeps mouth relaxed/closed)."""
    _need_session(session_id)
    if model_status["mouth"] != "ready":
        return {"ok": False, "message": f"Mouth check unavailable ({model_status['mouth']})."}
    base = mouth.baseline([f.file.read() for f in frames])
    if base is None:
        baselines.pop(session_id, None)
        return {"ok": False, "message": "I couldn't see your face. Face the camera in good light and try again."}
    baselines[session_id] = base
    return {"ok": True, "message": "Neutral mouth saved."}


# ------------------------------------------------- live lips + calibration
@app.post("/api/calibration/start")
def calibration_start(session_id: str = Form(...)):
    """(Re)start neutral-mouth calibration. Progress then advances only via real /api/face samples."""
    _need_session(session_id)
    baselines.pop(session_id, None)
    calibrators[session_id] = CalibrationTracker()
    return calibrators[session_id].status("measuring", "Mouth Calibration")


@app.post("/api/face")
def face(image: UploadFile = File(...), session_id: str = Form(""), calibrating: str = Form("0")):
    """One live frame -> lip outline (normalised 0..1) + mouth opening/width from MediaPipe.

    While calibrating, the same sample also feeds the calibration tracker."""
    if model_status["mouth"] != "ready":
        raise HTTPException(503, f"Mouth analysis not ready ({model_status['mouth']})")
    try:
        out, sample = mouth.track(image.file.read())
    except ValueError as e:
        raise HTTPException(400, str(e))
    tracker = calibrators.get(session_id)
    if calibrating == "1" and tracker is not None:
        out["calibration"] = tracker.add(sample)
        if tracker.baseline is not None:
            baselines[session_id] = tracker.baseline
    base = baselines.get(session_id)
    out["baseline_opening"] = None if base is None else round(base.opening, 4)
    return out


# ---------------------------------------------------------------- attempts
@app.post("/api/attempt")
def attempt(session_id: str = Form(...), attempt_id: str = Form(...), word_key: str = Form(...),
            audio: UploadFile = File(...), frames: Optional[List[UploadFile]] = File(None)):
    """One spoken attempt. Points come ONLY from the transcript matching the target."""
    _need_session(session_id)
    word = config.VOCAB_BY_KEY.get(word_key)
    if word is None:
        raise HTTPException(400, "Unknown word.")

    # Same attempt submitted twice -> return the stored result, no extra points.
    prior = store.get_attempt(attempt_id)
    if prior:
        return JSONResponse({**_result_from_row(prior, word), "duplicate": True,
                             "stats": store.stats(session_id)})

    if model_status["speech"] != "ready":
        raise HTTPException(503, f"Speech recognition not ready ({model_status['speech']})")
    try:
        stt = transcriber.transcribe(audio.file.read())
    except AudioError as e:
        raise HTTPException(400, str(e))

    # Mouth check is supplementary: failures never block the answer.
    if model_status["mouth"] == "ready":
        try:
            mouth_result = mouth.analyse([f.file.read() for f in (frames or [])],
                                         baselines.get(session_id), word.label)
        except Exception as e:
            log.exception("mouth analysis failed")
            mouth_result = {"status": "error", "message": f"Mouth check failed: {e}"}
    else:
        mouth_result = {"status": "unavailable", "message": "Mouth check unavailable."}

    if stt["silent"]:
        # Nothing was said: not a completed attempt, no points, not counted.
        return {"status": "silent", "transcript": "", "correct": False, "points": 0,
                "message": "I couldn't hear anything. Check your microphone and try again.",
                "mouth": mouth_result, "stats": store.stats(session_id), "duplicate": False}

    match = evaluate_answer(word, stt["text"])
    if not match.normalised:   # speech detected but Whisper returned no words
        return {"status": "no_words", "transcript": "", "correct": False, "points": 0,
                "message": "I couldn't make out a word. Please try again.",
                "mouth": mouth_result, "stats": store.stats(session_id), "duplicate": False}

    saved = store.record_attempt(session_id, attempt_id, word, stt["text"], match,
                                 mouth_result.get("status", ""))
    if match.correct and saved.get("already_paid"):
        message = f"Correct again! Points for “{word.label}” were already collected. Try another object."
    elif match.correct:
        message = f"Correct! You said “{word.label}”."
    elif match.near_match:
        message = f"Close! I heard “{match.normalised}”. Listen again and try “{word.label}”."
    else:
        message = f"I heard “{match.normalised}”. The word is “{word.label}”. Try again!"
    return {"status": "correct" if match.correct else "incorrect", "transcript": stt["text"],
            "normalised": match.normalised, "target": word.label, "correct": match.correct,
            "near_match": match.near_match, "edit_distance": match.distance,
            "points": saved["points"], "message": message, "mouth": mouth_result,
            "stats": store.stats(session_id), "duplicate": saved["duplicate"]}


def _result_from_row(row: dict, word) -> dict:
    return {"status": "correct" if row["correct"] else "incorrect", "transcript": row["transcript"],
            "target": word.label, "correct": bool(row["correct"]), "points": row["points"],
            "message": "This attempt was already scored.", "mouth": {"status": row["mouth_status"]}}


# Frontend last, so /api/* routes win.
app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")
