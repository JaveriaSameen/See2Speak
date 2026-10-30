"""API plumbing tests. The three ML models are replaced by tiny stand-ins HERE ONLY,
so the HTTP/idempotency/scoring flow can be tested without webcam, mic or weights.
The shipped app never uses these stand-ins."""
import io
import wave

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.facial_landmarks import MouthSample
from backend.game_logic import GameStore


def wav_bytes(seconds=1.0):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return b.getvalue()


class FakeSTT:
    text, silent = "apple", False
    def transcribe(self, wav):
        return {"text": self.text, "silent": self.silent, "seconds": 1.0, "rms": 0.1}


class FakeMouth:
    """Stand-in for MediaPipe: returns scripted samples so calibration can be exercised."""
    next_sample = None          # MouthSample or None (= no face)

    def analyse(self, frames, baseline, label):
        return {"status": "movement", "message": "stub"}

    def track(self, jpeg):
        s = FakeMouth.next_sample
        if s is None:
            return {"face": False}, None
        return {"face": True, "lips": {"outer": [[0.4, 0.6]] * 20, "inner": [[0.5, 0.6]] * 20},
                "opening": s.opening, "width": s.width}, s


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "store", GameStore(":memory:"))
    monkeypatch.setattr(main, "transcriber", FakeSTT())
    monkeypatch.setattr(main, "mouth", FakeMouth())
    monkeypatch.setitem(main.model_status, "speech", "ready")
    monkeypatch.setitem(main.model_status, "mouth", "ready")
    monkeypatch.setitem(main.model_status, "detector", "ready")
    return TestClient(main.app)


def post_attempt(c, sid, aid, key="apple"):
    return c.post("/api/attempt", data={"session_id": sid, "attempt_id": aid, "word_key": key},
                  files=[("audio", ("s.wav", wav_bytes(), "audio/wav"))]).json()


def test_config_lists_vocabulary(client):
    labels = [w["label"] for w in client.get("/api/config").json()["vocabulary"]]
    assert labels == ["Book", "Cell phone", "Apple", "Pen", "Water bottle", "Tape dispenser", "Stapler", "Scissors", "Glue stick"]
    removed = {"Chair", "Paper clip", "Pencil", "Marker", "Notebook", "Ruler", "Eraser"}
    assert not removed & set(labels)


def test_correct_attempt_scores_and_duplicate_does_not(client):
    sid = client.post("/api/session").json()["session_id"]
    r = post_attempt(client, sid, "x1")
    assert r["correct"] and r["points"] == 10 and r["stats"]["score"] == 10
    dup = post_attempt(client, sid, "x1")
    assert dup["duplicate"] and dup["stats"]["score"] == 10


def test_wrong_word_scores_zero(client):
    sid = client.post("/api/session").json()["session_id"]
    r = post_attempt(client, sid, "x1", key="book")      # said "apple", target book
    assert not r["correct"] and r["points"] == 0 and r["stats"]["attempts"] == 1


def test_silence_is_not_an_attempt(client):
    main.transcriber.silent = True
    sid = client.post("/api/session").json()["session_id"]
    r = post_attempt(client, sid, "x1")
    assert r["status"] == "silent" and r["points"] == 0 and r["stats"]["attempts"] == 0


def test_unknown_word_and_session_rejected(client):
    sid = client.post("/api/session").json()["session_id"]
    assert client.post("/api/attempt", data={"session_id": sid, "attempt_id": "a", "word_key": "zebra"},
                       files=[("audio", ("s.wav", wav_bytes(), "audio/wav"))]).status_code == 400
    assert client.get("/api/session/nope").status_code == 404


def test_end_session_summary_and_history(client):
    sid = client.post("/api/session").json()["session_id"]
    post_attempt(client, sid, "x1")
    summary = client.post(f"/api/session/{sid}/end").json()
    assert summary["score"] == 10 and summary["mastered"] == ["apple"]
    assert client.get("/api/history").json()["sessions"][0]["score"] == 10


def test_frontend_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "See2Speak" in r.text


# ------------------------------------------------ live lips + calibration
def face(c, sid, calibrating=False):
    return c.post("/api/face", data={"session_id": sid, "calibrating": "1" if calibrating else "0"},
                  files=[("image", ("f.jpg", b"x", "image/jpeg"))]).json()


def test_face_endpoint_returns_lips_and_opening(client):
    sid = client.post("/api/session").json()["session_id"]
    FakeMouth.next_sample = MouthSample(0.03, 0.9)
    r = face(client, sid)
    assert r["face"] and len(r["lips"]["outer"]) == 20 and len(r["lips"]["inner"]) == 20
    assert r["opening"] == 0.03 and r["baseline_opening"] is None
    FakeMouth.next_sample = None
    assert face(client, sid) == {"face": False, "baseline_opening": None}


def test_calibration_progress_comes_only_from_real_samples(client):
    sid = client.post("/api/session").json()["session_id"]
    assert client.post("/api/calibration/start", data={"session_id": sid}).json()["progress"] == 0
    FakeMouth.next_sample = None                       # no face: no progress
    assert face(client, sid, True)["calibration"]["progress"] == 0
    FakeMouth.next_sample = MouthSample(0.5, 0.9)      # mouth wide open: no progress
    r = face(client, sid, True)["calibration"]
    assert r["progress"] == 0 and r["state"] == "mouth_open"
    FakeMouth.next_sample = MouthSample(0.02, 0.9)     # relaxed: progress
    r = face(client, sid, True)["calibration"]
    assert 0 < r["progress"] < 1 and r["state"] == "measuring"
    for _ in range(30):
        r = face(client, sid, True)
    assert r["calibration"]["complete"] and r["baseline_opening"] == pytest.approx(0.02)
    # a later (non-calibrating) frame still reports the stored baseline
    assert face(client, sid)["baseline_opening"] == pytest.approx(0.02)


def test_calibration_can_be_retried(client):
    sid = client.post("/api/session").json()["session_id"]
    FakeMouth.next_sample = MouthSample(0.02, 0.9)
    client.post("/api/calibration/start", data={"session_id": sid})
    for _ in range(20):
        face(client, sid, True)
    client.post("/api/calibration/start", data={"session_id": sid})
    r = face(client, sid, True)["calibration"]
    assert not r["complete"] and r["collected"] == 1


def test_play_again_inherits_calibrated_baseline(client):
    s1 = client.post("/api/session").json()["session_id"]
    FakeMouth.next_sample = MouthSample(0.02, 0.9)
    client.post("/api/calibration/start", data={"session_id": s1})
    for _ in range(20):
        face(client, s1, True)
    client.post(f"/api/session/{s1}/end")
    s2 = client.post("/api/session", data={"inherit_from": s1}).json()["session_id"]
    assert face(client, s2)["baseline_opening"] == pytest.approx(0.02)
    s3 = client.post("/api/session").json()["session_id"]           # no inherit -> no baseline
    assert face(client, s3)["baseline_opening"] is None
