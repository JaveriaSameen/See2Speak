"""API plumbing tests. The three ML models are replaced by tiny stand-ins HERE ONLY,
so the HTTP/idempotency/scoring flow can be tested without webcam, mic or weights.
The shipped app never uses these stand-ins."""
import io
import wave

import pytest
from fastapi.testclient import TestClient

from backend import main
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
    def analyse(self, frames, baseline, label):
        return {"status": "movement", "message": "stub"}


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
    assert labels == ["Book", "Chair", "Cell phone", "Apple", "Pen", "Pencil"]


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
