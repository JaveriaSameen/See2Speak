"""Answer matching, scoring and session storage (SQLite).

No model code here, so this file is fully unit-testable.
Points are awarded ONLY when the normalised transcript matches the target
(exact match against the word's accepted forms). Edit distance only produces
feedback text for near misses.
"""
from __future__ import annotations

import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from . import config
from .config import VocabWord

_ARTICLES = {"a", "an", "the"}


# ---------------------------------------------------------------- matching
def normalise(text: str) -> str:
    """lowercase, strip punctuation + surrounding spaces, drop a leading article."""
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    words = text.split()
    if len(words) > 1 and words[0] in _ARTICLES:   # "an apple" -> "apple"
        words = words[1:]
    return " ".join(words)


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class MatchResult:
    correct: bool
    normalised: str
    distance: int            # to the closest accepted form
    near_match: bool         # incorrect but within NEAR_MATCH_MAX_DISTANCE
    reason: str


def evaluate_answer(word: VocabWord, transcript: str) -> MatchResult:
    norm = normalise(transcript)
    if not norm:
        return MatchResult(False, "", 99, False, "empty")
    accepted = {normalise(a) for a in word.accepted} | {normalise(word.label)}
    if norm in accepted:
        return MatchResult(True, norm, 0, False, "exact")
    dist = min(edit_distance(norm, a) for a in accepted)
    near = dist <= config.NEAR_MATCH_MAX_DISTANCE
    return MatchResult(False, norm, dist, near, "near" if near else "different")


# ------------------------------------------------------------------ storage
_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
  id TEXT PRIMARY KEY, started REAL, ended REAL);
CREATE TABLE IF NOT EXISTS attempts(
  attempt_id TEXT PRIMARY KEY, session_id TEXT, ts REAL,
  word_key TEXT, transcript TEXT, correct INTEGER, points INTEGER,
  near_match INTEGER, mouth_status TEXT);
"""


class GameStore:
    def __init__(self, path):
        path = str(path)
        if path != ":memory:":
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def start_session(self) -> str:
        sid = uuid.uuid4().hex
        with self._lock, self._db:
            self._db.execute("INSERT INTO sessions(id, started) VALUES(?,?)", (sid, time.time()))
        return sid

    def session_exists(self, sid: str) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM sessions WHERE id=?", (sid,)).fetchone() is not None

    def get_attempt(self, attempt_id: str) -> Optional[dict]:
        with self._lock:
            row = self._db.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
        return dict(row) if row else None

    def record_attempt(self, session_id: str, attempt_id: str, word: VocabWord,
                       transcript: str, match: MatchResult, mouth_status: str) -> dict:
        """Store an attempt once. Re-sending the same attempt_id never adds points."""
        with self._lock, self._db:
            existing = self._db.execute(
                "SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if existing:
                return {**dict(existing), "duplicate": True}
            points = config.POINTS_CORRECT if match.correct else 0
            already_paid = False
            if points and config.AWARD_ONCE_PER_WORD:
                already_paid = self._db.execute(
                    "SELECT 1 FROM attempts WHERE session_id=? AND word_key=? AND points>0",
                    (session_id, word.key)).fetchone() is not None
                if already_paid:
                    points = 0
            self._db.execute(
                "INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?)",
                (attempt_id, session_id, time.time(), word.key, transcript,
                 int(match.correct), points, int(match.near_match), mouth_status))
        return {"attempt_id": attempt_id, "points": points, "correct": int(match.correct),
                "already_paid": already_paid, "duplicate": False}

    def stats(self, session_id: str) -> dict:
        with self._lock:
            rows = self._db.execute(
                "SELECT word_key, correct, points FROM attempts WHERE session_id=?",
                (session_id,)).fetchall()
        mastered = sorted({r["word_key"] for r in rows if r["correct"]})
        return {
            "score": sum(r["points"] for r in rows),
            "attempts": len(rows),
            "correct": sum(r["correct"] for r in rows),
            "mastered": mastered,
            "mastered_count": len(mastered),
            "round_target": config.ROUND_TARGET,
            "complete": len(mastered) >= config.ROUND_TARGET,
        }

    def end_session(self, session_id: str) -> dict:
        with self._lock, self._db:
            self._db.execute("UPDATE sessions SET ended=? WHERE id=? AND ended IS NULL",
                             (time.time(), session_id))
        return self.stats(session_id)

    def history(self, limit: int = 10) -> list[dict]:
        """Past sessions, for the 'recorded scores' requirement."""
        with self._lock:
            rows = self._db.execute(
                """SELECT s.id, s.started, COALESCE(SUM(a.points),0) AS score,
                          COUNT(a.attempt_id) AS attempts, COALESCE(SUM(a.correct),0) AS correct
                   FROM sessions s LEFT JOIN attempts a ON a.session_id=s.id
                   GROUP BY s.id ORDER BY s.started DESC LIMIT ?""", (limit,)).fetchall()
        return [dict(r) for r in rows]
