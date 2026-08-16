"""WS1: capture-time per-box uncertainty (margin + optional flip pass)."""
import json

from app import uncertainty
from app.uncertainty import (attach_uncertainty, box_iou, flip_delta,
                             margin_score)


def test_margin_curve_endpoints():
    # 1.0 exactly on the gate, linear to 0 at gate +- span, clamped beyond.
    assert margin_score(0.35, 0.35) == 1.0
    assert margin_score(0.35 + 0.25, 0.35) == 0.0
    assert margin_score(0.35 - 0.25, 0.35) == 0.0
    assert abs(margin_score(0.475, 0.35) - 0.5) < 1e-9
    assert margin_score(0.95, 0.35) == 0.0
    assert margin_score("bad", 0.35) == 0.0


def test_attach_blend_and_effective_gates():
    boxes = [{"cls": "person", "conf": 0.35},   # on the boosted gate
             {"cls": "car",    "conf": 0.80}]   # far above its gate
    attach_uncertainty(boxes, {"person": 0.35, "car": 0.35})
    assert boxes[0]["uncertainty"] == 0.6        # 0.6*1.0 + 0.4*0
    assert boxes[1]["uncertainty"] == 0.0
    # flip term folds in at 0.4 weight
    attach_uncertainty(boxes, {"person": 0.35, "car": 0.35}, flip={0: 1.0})
    assert boxes[0]["uncertainty"] == 1.0        # 0.6*1.0 + 0.4*1.0


def test_flip_delta_matches_mirrored_boxes(monkeypatch):
    import numpy as np
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    boxes = [{"cls": "person", "conf": 0.50,
              "x1": 10, "y1": 10, "x2": 30, "y2": 60},
             {"cls": "car", "conf": 0.40,
              "x1": 120, "y1": 20, "x2": 180, "y2": 70}]
    # The flipped pass "sees" the person mirrored with a moved conf, and
    # does NOT see the car at all.
    mirrored_person = {"cls": "person", "conf": 0.30,
                       "x1": 200 - 30, "y1": 10, "x2": 200 - 10, "y2": 60}
    monkeypatch.setattr(uncertainty, "_predict_boxes",
                        lambda model, fr, imgsz: [mirrored_person])
    d = flip_delta(object(), frame, boxes)
    assert abs(d[0] - 0.20) < 1e-9    # |0.50 - 0.30|
    assert d[1] == 1.0                # vanished under flip -> max instability


def test_box_iou_basics():
    a = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
    assert box_iou(a, a) == 1.0
    assert box_iou(a, {"x1": 20, "y1": 20, "x2": 30, "y2": 30}) == 0.0


def test_metadata_round_trip(tmp_path):
    """Boxes with uncertainty survive save_frame -> load_metadata, and
    frame_uncertainty prefers the persisted value over its own margin."""
    import numpy as np
    from app.labels import frame_uncertainty
    from app.review_frames import load_metadata, save_frame

    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    boxes = [{"cls": "person", "conf": 0.90,   # margin fallback would be ~0
              "uncertainty": 0.77,
              "x1": 1, "y1": 1, "x2": 20, "y2": 50}]
    rel = save_frame("cam_x", frame, boxes, snapshots_root=tmp_path)
    assert rel
    meta = load_metadata(rel, tmp_path)
    assert meta["boxes"][0]["uncertainty"] == 0.77
    assert frame_uncertainty(meta) == 0.77


def test_crop_filename_suffix(tmp_path):
    import numpy as np
    from app.badge import crop_uncertainty
    from app.live_samples import save_crop

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    box = {"cls": "person", "conf": 0.42, "uncertainty": 0.58,
           "x1": 5, "y1": 5, "x2": 60, "y2": 90}
    rel = save_crop("cam_y", frame, [box], snapshots_root=tmp_path)
    assert rel and "_u58.jpg" in rel
    assert crop_uncertainty(rel) == 0.58
    # pre-WS1 names carry no suffix -> None
    assert crop_uncertainty("live_samples/cam/123_person_42.jpg") is None
