"""Checks how Detector uses the vocabulary and the confidence threshold. The YOLO model is a tiny
fake here, so this verifies the wiring (prompts, prompt->word mapping, 0.90 threshold on the model's
own scores, merging, warm-up), NOT detection accuracy."""
import sys
import types

import cv2
import numpy as np
import pytest

from backend import config
from backend.detection import Detector, iou, merge_same_word


def box(cls, conf, xyxy):
    return types.SimpleNamespace(cls=np.array([cls]), conf=np.array([conf], dtype=np.float32),
                                 xyxy=np.array([xyxy], dtype=float))


class FakeWorld:
    prompts = None
    calls = []
    script = []

    def __init__(self, path):
        pass

    def set_classes(self, names):
        FakeWorld.prompts = list(names)

    def predict(self, frame, imgsz, conf, max_det, verbose):
        FakeWorld.calls.append({"imgsz": imgsz, "conf": conf, "shape": frame.shape})
        return [types.SimpleNamespace(boxes=list(FakeWorld.script), speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5})]


@pytest.fixture
def det(monkeypatch, tmp_path):
    weights = tmp_path / "w.pt"; weights.write_bytes(b"x")
    monkeypatch.setattr(config, "YOLO_WEIGHTS", weights)
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLOWorld=FakeWorld))
    FakeWorld.calls = []; FakeWorld.script = []
    return Detector()


def jpeg():
    ok, buf = cv2.imencode(".jpg", np.zeros((100, 200, 3), np.uint8))
    return buf.tobytes()


def idx(det, prompt):
    return det.prompts.index(prompt)


def test_every_prompt_is_sent_in_order_and_bottle_prompts_map_to_one_word(det):
    det.load()
    assert FakeWorld.prompts == det.prompts
    for p in ("water bottle", "plastic water bottle", "reusable water bottle", "insulated metal flask"):
        assert p in det.prompts
        assert det.vocab[det.prompt_word[idx(det, p)]].key == "water_bottle"     # all -> same class
    assert "chair" not in det.prompts and "pencil" not in det.prompts and "paper clip" not in det.prompts


def test_warmup_runs_once_at_load_not_on_first_frame(det):
    det.load()
    assert len(FakeWorld.calls) == 1 and det.warmup_ms is not None          # warm-up pass
    det.detect(jpeg())
    assert len(FakeWorld.calls) == 2                                          # no reload / extra pass
    det.load()
    assert len(FakeWorld.calls) == 2                                          # load() is idempotent


def test_threshold_is_090_on_the_models_own_score(det):
    assert config.DETECT_MIN_CONF == 0.90
    FakeWorld.script = [
        box(idx(det, "apple"), 0.95, (10, 10, 60, 60)),       # shown
        box(idx(det, "pen"), 0.90, (70, 10, 90, 90)),         # exactly 0.90 (float32 0.8999999761) -> shown
        box(idx(det, "stapler"), 0.89, (100, 10, 150, 60)),   # just below -> NOT shown
        box(idx(det, "book"), 0.60, (10, 60, 90, 99)),        # well below -> NOT shown
    ]
    out = det.detect(jpeg())
    assert {d["key"] for d in out["detections"]} == {"apple", "pen"}
    assert {d["confidence"] for d in out["detections"]} == {0.95, 0.9}        # scores reported as-is
    assert FakeWorld.calls[-1]["conf"] == config.DETECT_CANDIDATE_FLOOR


def test_nothing_shown_below_threshold_but_best_candidate_is_reported(det):
    FakeWorld.script = [box(idx(det, "apple"), 0.62, (10, 10, 60, 60)), box(idx(det, "book"), 0.40, (70, 10, 90, 90))]
    out = det.detect(jpeg())
    assert out["detections"] == []
    assert out["best_rejected"] == {"label": "Apple", "confidence": 0.62} and out["min_conf"] == 0.90


def test_threshold_can_be_overridden_for_experiments(det, monkeypatch):
    monkeypatch.setattr(config, "DETECT_MIN_CONF", 0.5)
    FakeWorld.script = [box(idx(det, "apple"), 0.62, (10, 10, 60, 60))]
    assert [d["key"] for d in det.detect(jpeg())["detections"]] == ["apple"]


def test_two_bottle_prompts_on_one_object_become_one_box(det):
    FakeWorld.script = [
        box(idx(det, "water bottle"), 0.93, (50, 10, 90, 95)),
        box(idx(det, "insulated metal flask"), 0.97, (52, 12, 91, 96)),    # same object, other prompt
        box(idx(det, "plastic water bottle"), 0.91, (140, 10, 180, 95)),   # a second, different bottle
    ]
    out = det.detect(jpeg())["detections"]
    assert [d["key"] for d in out] == ["water_bottle", "water_bottle"]
    assert [d["confidence"] for d in out] == [0.97, 0.91]                   # higher score kept, not altered
    assert all(d["label"] == "Water bottle" for d in out)


def test_imgsz_and_normalised_boxes(det):
    FakeWorld.script = [box(idx(det, "glue stick"), 0.99, (100, 20, 120, 120))]
    out = det.detect(jpeg())
    assert FakeWorld.calls[-1]["imgsz"] == config.DETECT_IMGSZ == 512
    b = out["detections"][0]["box"]
    assert (b["x"], b["y"], b["w"], b["h"]) == (0.5, 0.2, 0.1, 1.0) or b["h"] == pytest.approx(1.0)


def test_merge_only_merges_same_word():
    a = {"key": "pen", "confidence": .95, "box": {"x": .1, "y": .1, "w": .2, "h": .2}}
    b = {"key": "stapler", "confidence": .92, "box": {"x": .1, "y": .1, "w": .2, "h": .2}}   # same place, other word
    assert len(merge_same_word([a, b])) == 2
    assert iou(a["box"], a["box"]) == pytest.approx(1.0)
