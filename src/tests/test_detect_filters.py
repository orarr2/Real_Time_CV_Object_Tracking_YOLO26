"""Class-aware confidence, person plausibility filter, rider co-detection.

Uses a MockYOLO that just replays a canned list of (cls_id, conf, xyxy) tuples
so the tests never touch a real network / weights file.
"""
import numpy as np
import pytest

from app.detect_core import (
    DEFAULT_PER_CLASS_CONF,
    NAME_BY_ID,
    detect_with_boxes,
)


class _MockBoxes:
    def __init__(self, dets):
        self.xyxy = _Arr(np.array([d[2] for d in dets], dtype=float)) if dets \
                    else _Arr(np.zeros((0, 4)))
        self.cls  = _Arr(np.array([d[0] for d in dets], dtype=float)) if dets \
                    else _Arr(np.zeros(0))
        self.conf = _Arr(np.array([d[1] for d in dets], dtype=float)) if dets \
                    else _Arr(np.zeros(0))


class _Arr:
    def __init__(self, a): self._a = a
    def cpu(self): return self
    def numpy(self): return self._a
    def astype(self, t): return self._a.astype(t)


class _MockResult:
    def __init__(self, dets): self.boxes = _MockBoxes(dets)


class MockYOLO:
    """Returns the predetermined detections regardless of the frame, respecting
    the `conf=` filter the model receives (which is what YOLO's own gate does)."""

    def __init__(self, dets):
        self._all = dets

    def predict(self, frame, conf, classes, verbose=False, imgsz=None):
        keep = [d for d in self._all
                if d[1] >= conf and int(d[0]) in classes]
        return [_MockResult(keep)]


def _id_of(cls_name):
    for cid, n in NAME_BY_ID.items():
        if n == cls_name:
            return cid
    raise KeyError(cls_name)


PERSON = _id_of("person")
BICYCLE = _id_of("bicycle")
MOTOR   = _id_of("motorcycle")
CAR     = _id_of("car")

TALL_PERSON = [10, 10, 50, 120]                  # h=110, w=40  -> ar 2.75, OK
WIDE_FAUX_PERSON = [200, 200, 260, 220]          # h=20,  w=60  -> ar 0.33, stroller
RIDER_PERSON = [300, 100, 340, 200]              # h=100, w=40
RIDER_BIKE   = [295, 150, 350, 210]              # overlaps rider heavily


def test_legacy_single_conf_unchanged():
    """per_class_conf=None keeps the OLD behavior of a single global conf."""
    m = MockYOLO([(PERSON, 0.40, TALL_PERSON),
                  (CAR,    0.28, [400, 400, 500, 460])])
    counts, boxes = detect_with_boxes(m, np.zeros((720, 1280, 3)), conf=0.35,
                                      person_min_aspect=None, rider_iou=None)
    assert counts["person"] == 1 and counts["car"] == 0
    assert [b["cls"] for b in boxes] == ["person"]


def test_per_class_conf_lets_small_person_through():
    """A person at 0.25 makes it in under an explicit person=0.22 gate but
    not under legacy single-conf 0.35. Uses an explicit per-class map (not
    DEFAULT_PER_CLASS_CONF) so re-tuning the shipped production thresholds
    cannot silently break the mechanism this test covers."""
    m = MockYOLO([(PERSON, 0.25, TALL_PERSON)])
    _, legacy = detect_with_boxes(m, np.zeros((720, 1280, 3)), conf=0.35)
    _, new    = detect_with_boxes(m, np.zeros((720, 1280, 3)), conf=0.35,
                                  per_class_conf={"person": 0.22})
    assert legacy == []                             # OLD: 0.25 < 0.35, dropped
    assert len(new) == 1 and new[0]["cls"] == "person"


def test_per_class_conf_keeps_strict_car_gate():
    """A car at 0.30 must still fail under per-class (car=0.35)."""
    m = MockYOLO([(CAR, 0.30, [400, 400, 500, 460])])
    _, new = detect_with_boxes(m, np.zeros((720, 1280, 3)), conf=0.35,
                               per_class_conf=DEFAULT_PER_CLASS_CONF)
    assert new == []                                # avoids stroller-as-car type FPs


def test_stroller_shape_rejected_from_person():
    """A wide-and-short 'person' box is dropped by the aspect filter."""
    m = MockYOLO([(PERSON, 0.60, WIDE_FAUX_PERSON),
                  (PERSON, 0.60, TALL_PERSON)])
    counts, boxes = detect_with_boxes(m, np.zeros((720, 1280, 3)), conf=0.35)
    assert counts["person"] == 1
    kept = [b for b in boxes if b["cls"] == "person"]
    assert len(kept) == 1
    kh = kept[0]["y2"] - kept[0]["y1"]
    kw = kept[0]["x2"] - kept[0]["x1"]
    assert kh > kw


def test_stroller_filter_disabled_by_none():
    """person_min_aspect=None keeps every person box regardless of shape.
    The min-height gate (added later) must be disabled too - this test is
    about the ASPECT filter specifically, and the faux box is only 20px
    tall so the height gate would otherwise mask the behavior under test."""
    m = MockYOLO([(PERSON, 0.60, WIDE_FAUX_PERSON)])
    counts, _ = detect_with_boxes(m, np.zeros((720, 1280, 3)), conf=0.35,
                                  person_min_aspect=None,
                                  person_min_height_px=None)
    assert counts["person"] == 1


def test_rider_bike_rescued_below_gate():
    """A bike at 0.19 overlaps a person at 0.90 -> the bike is rescued."""
    m = MockYOLO([(PERSON,  0.90, RIDER_PERSON),
                  (BICYCLE, 0.19, RIDER_BIKE)])
    counts, boxes = detect_with_boxes(m, np.zeros((720, 1280, 3)), conf=0.15,
                                      per_class_conf=DEFAULT_PER_CLASS_CONF)
    assert counts["person"] == 1 and counts["bicycle"] == 1
    bike = next(b for b in boxes if b["cls"] == "bicycle")
    assert bike["conf"] == pytest.approx(0.19, abs=1e-4)


def test_isolated_bike_below_gate_still_dropped():
    """No person nearby -> the below-gate bike stays dropped (no false rescue)."""
    m = MockYOLO([(BICYCLE, 0.19, [500, 500, 540, 560])])
    counts, _ = detect_with_boxes(m, np.zeros((720, 1280, 3)), conf=0.15,
                                  per_class_conf=DEFAULT_PER_CLASS_CONF)
    assert counts["bicycle"] == 0


def test_motorcycle_rider_pair_kept_together():
    """The user's motorcycle case: rider AND motorcycle both survive."""
    m = MockYOLO([(PERSON, 0.75, RIDER_PERSON),
                  (MOTOR,  0.20, RIDER_BIKE)])
    counts, _ = detect_with_boxes(m, np.zeros((720, 1280, 3)), conf=0.15,
                                  per_class_conf=DEFAULT_PER_CLASS_CONF)
    assert counts["person"] == 1
    assert counts["motorcycle"] == 1
    assert counts["vehicles"] == 1


def test_rider_iou_disabled_by_none():
    """rider_iou=None disables the rescue path entirely."""
    m = MockYOLO([(PERSON, 0.90, RIDER_PERSON),
                  (MOTOR,  0.20, RIDER_BIKE)])
    counts, _ = detect_with_boxes(m, np.zeros((720, 1280, 3)), conf=0.15,
                                  per_class_conf=DEFAULT_PER_CLASS_CONF,
                                  rider_iou=None)
    assert counts["motorcycle"] == 0
