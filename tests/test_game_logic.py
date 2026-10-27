import numpy as np
import pytest

from backend import config
from backend.facial_landmarks import (MouthSample, analyse_samples, metrics_from_points,
                                      EYE_L, EYE_R, UPPER, LOWER, LEFT, RIGHT)
from backend.game_logic import GameStore, edit_distance, evaluate_answer, normalise

W = config.VOCAB_BY_KEY


# ---------------------------------------------------------------- matching
@pytest.mark.parametrize("raw,norm", [
    ("Apple.", "apple"), ("  APPLE!! ", "apple"), ("An apple", "apple"),
    ("the book", "book"), ("Cell-phone", "cell phone"), ("", ""), ("...", ""),
])
def test_normalise(raw, norm):
    assert normalise(raw) == norm


def test_edit_distance():
    assert edit_distance("pen", "pen") == 0
    assert edit_distance("pen", "pin") == 1
    assert edit_distance("pen", "pencil") == 3


def test_exact_match_correct():
    assert evaluate_answer(W["apple"], "Apple.").correct


def test_wrong_word_incorrect():
    r = evaluate_answer(W["book"], "Chair")
    assert not r.correct and not r.near_match


def test_cell_phone_aliases():
    for said in ("phone", "Cell phone.", "cellphone", "mobile phone"):
        assert evaluate_answer(W["cell_phone"], said).correct, said


def test_near_match_never_correct():
    r = evaluate_answer(W["pen"], "pan")     # edit distance 1
    assert not r.correct and r.near_match


def test_pen_vs_pencil_not_confused():
    assert not evaluate_answer(W["pen"], "pencil").correct
    assert not evaluate_answer(W["pencil"], "pen").correct


def test_empty_is_incorrect():
    assert not evaluate_answer(W["apple"], "").correct


# ------------------------------------------------------------------ scoring
@pytest.fixture
def store():
    return GameStore(":memory:")


def attempt(store, sid, aid, key, said):
    m = evaluate_answer(W[key], said)
    return store.record_attempt(sid, aid, W[key], said, m, "movement")


def test_correct_awards_points_incorrect_zero(store):
    sid = store.start_session()
    assert attempt(store, sid, "a1", "apple", "apple")["points"] == config.POINTS_CORRECT
    assert attempt(store, sid, "a2", "book", "chair")["points"] == 0
    st = store.stats(sid)
    assert (st["score"], st["attempts"], st["correct"]) == (config.POINTS_CORRECT, 2, 1)


def test_same_attempt_id_never_scores_twice(store):
    sid = store.start_session()
    attempt(store, sid, "a1", "apple", "apple")
    again = attempt(store, sid, "a1", "apple", "apple")
    assert again["duplicate"]
    st = store.stats(sid)
    assert st["score"] == config.POINTS_CORRECT and st["attempts"] == 1


def test_word_pays_once_per_session(store, monkeypatch):
    sid = store.start_session()
    attempt(store, sid, "a1", "apple", "apple")
    second = attempt(store, sid, "a2", "apple", "apple")
    assert second["points"] == 0 and second["already_paid"]
    st = store.stats(sid)
    assert st["score"] == config.POINTS_CORRECT and st["correct"] == 2 and st["mastered_count"] == 1


def test_pay_every_time_when_disabled(store, monkeypatch):
    monkeypatch.setattr(config, "AWARD_ONCE_PER_WORD", False)
    sid = store.start_session()
    attempt(store, sid, "a1", "apple", "apple")
    attempt(store, sid, "a2", "apple", "apple")
    assert store.stats(sid)["score"] == 2 * config.POINTS_CORRECT


def test_retry_after_wrong_answer_allowed(store):
    sid = store.start_session()
    attempt(store, sid, "a1", "apple", "ball")
    assert attempt(store, sid, "a2", "apple", "apple")["points"] == config.POINTS_CORRECT


def test_new_session_starts_at_zero(store):
    s1 = store.start_session(); attempt(store, s1, "a1", "apple", "apple")
    s2 = store.start_session()
    assert store.stats(s2)["score"] == 0
    assert store.stats(s1)["score"] == config.POINTS_CORRECT   # history preserved


def test_round_complete(store):
    sid = store.start_session()
    for i, w in enumerate(config.VOCABULARY):
        attempt(store, sid, f"a{i}", w.key, w.label)
    st = store.stats(sid)
    assert st["complete"] and st["score"] == config.POINTS_CORRECT * len(config.VOCABULARY)


# ------------------------------------------------------------ mouth analysis
def face(opening, width=0.9):
    """Synthetic landmark array with inter-ocular distance 1.0."""
    pts = np.zeros((478, 2), dtype=np.float32)
    pts[EYE_L], pts[EYE_R] = (0, 0), (1, 0)
    pts[LEFT], pts[RIGHT] = (0.05, 1), (0.05 + width, 1)
    pts[UPPER], pts[LOWER] = (0.5, 1), (0.5, 1 + opening)
    return pts


def test_metrics_scale_invariant():
    a = metrics_from_points(face(0.2))
    b = metrics_from_points(face(0.2) * 3.0)      # same face, closer to camera
    assert a.opening == pytest.approx(b.opening) and a.width == pytest.approx(b.width)


BASE = MouthSample(0.01, 0.9)


def test_no_face_is_reported_not_crashed():
    r = analyse_samples([None] * 10, BASE, "Apple")
    assert r["status"] == "no_face"


def test_still_mouth_is_no_movement():
    s = [MouthSample(0.012, 0.9)] * 10
    assert analyse_samples(s, BASE, "Apple")["status"] == "no_movement"


def test_open_mouth_is_movement():
    s = [MouthSample(0.01, 0.9)] * 3 + [MouthSample(0.25, 0.95)] * 4 + [MouthSample(0.02, 0.9)] * 3
    r = analyse_samples(s, BASE, "Apple")
    assert r["status"] == "movement" and r["lip_closure"] is None


def test_lip_closure_for_b_p_m_words():
    closed_then_open = [MouthSample(0.01, 0.9)] * 3 + [MouthSample(0.3, 0.9)] * 7
    assert analyse_samples(closed_then_open, BASE, "Book")["lip_closure"] is True
    never_closed = [MouthSample(0.2, 0.9)] * 10
    assert analyse_samples(never_closed, BASE, "Pen")["lip_closure"] is False
    assert analyse_samples(closed_then_open, BASE, "Chair")["lip_closure"] is None   # not b/p/m
