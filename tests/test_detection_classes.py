"""Checks how Detector uses the vocabulary. The YOLO model itself is a tiny fake here, so this
verifies the wiring (prompts, index->word mapping, per-class thresholds), NOT detection accuracy."""
import sys
import types

import numpy as np
import cv2

from backend import config
from backend.detection import Detector


class FakeBoxes(list):
    pass


def box(cls, conf, xyxy):
    t = lambda v: np.array([v])
    return types.SimpleNamespace(cls=t(cls), conf=t(conf), xyxy=np.array([xyxy], dtype=float))


class FakeWorld:
    prompts = None

    def __init__(self, path):
        self.path = path

    def set_classes(self, names):
        FakeWorld.prompts = list(names)

    def predict(self, frame, imgsz, conf, verbose):
        FakeWorld.conf_floor = conf
        idx = {w.key: i for i, w in enumerate(config.VOCABULARY)}
        return [types.SimpleNamespace(boxes=FakeBoxes([
            box(idx["paper_clip"], 0.11, (10, 10, 30, 30)),     # above its 0.10 threshold -> kept
            box(idx["stapler"], 0.15, (50, 50, 90, 90)),        # below its 0.20 threshold -> dropped
            box(idx["glue_stick"], 0.40, (100, 20, 120, 120)),  # kept
        ]))]


def jpeg():
    ok, buf = cv2.imencode(".jpg", np.zeros((100, 200, 3), np.uint8))
    return buf.tobytes()


def test_all_fifteen_prompts_go_to_yolo_world_in_order(monkeypatch, tmp_path):
    weights = tmp_path / "w.pt"; weights.write_bytes(b"x")
    monkeypatch.setattr(config, "YOLO_WEIGHTS", weights)
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLOWorld=FakeWorld))
    d = Detector(); d.load()
    assert FakeWorld.prompts == [w.yolo_prompt for w in config.VOCABULARY]
    assert len(FakeWorld.prompts) == 15 and "chair" not in FakeWorld.prompts
    assert "paper clip" in FakeWorld.prompts and "tape dispenser" in FakeWorld.prompts


def test_index_mapping_and_per_class_thresholds(monkeypatch, tmp_path):
    weights = tmp_path / "w.pt"; weights.write_bytes(b"x")
    monkeypatch.setattr(config, "YOLO_WEIGHTS", weights)
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLOWorld=FakeWorld))
    d = Detector()
    out = d.detect(jpeg())
    got = {x["key"]: x for x in out["detections"]}
    assert set(got) == {"paper_clip", "glue_stick"}                    # stapler (0.15 < 0.20) dropped
    assert got["glue_stick"]["label"] == "Glue stick"
    assert FakeWorld.conf_floor == min(w.min_conf for w in config.VOCABULARY)
    b = got["glue_stick"]["box"]                                         # normalised to the 200x100 frame
    assert (b["x"], b["y"], b["w"], b["h"]) == (0.5, 0.2, 0.1, 1.0)
