"""EXTRA_CLASSES opt-in: detection-only class extension.

Run from src/:  python -m pytest tests -q
"""
from app import detect_core as dc


def _cleanup(names):
    for n in names:
        cid = dc.CLASSES_OF_INTEREST.pop(n, None)
        if cid is not None:
            dc.NAME_BY_ID.pop(cid, None)
        dc.DEFAULT_PER_CLASS_CONF.pop(n, None)


def test_default_is_the_seven_class_set():
    # Guard: without the env opt-in the tables must be exactly the
    # shipped business set - byte-for-byte legacy behavior.
    assert set(dc.CLASSES_OF_INTEREST) >= {
        "person", "bicycle", "car", "motorcycle", "bus", "train", "truck"}
    assert "bird" not in dc.CLASSES_OF_INTEREST


def test_add_bird_with_custom_gate():
    added = dc._apply_extra_classes("bird:0.30")
    try:
        assert added == ["bird"]
        assert dc.CLASSES_OF_INTEREST["bird"] == 14
        assert dc.NAME_BY_ID[14] == "bird"
        assert dc.DEFAULT_PER_CLASS_CONF["bird"] == 0.30
        # The road-vehicle aggregate must NOT change.
        assert "bird" not in dc.VEHICLE_NAMES
        # counts_from_boxes picks the new class up automatically.
        counts = dc.counts_from_boxes(
            [{"x1": 0, "y1": 0, "x2": 5, "y2": 5, "cls": "bird"}])
        assert counts["bird"] == 1
        assert counts["vehicles"] == 0
    finally:
        _cleanup(added)


def test_default_gate_and_multiple():
    added = dc._apply_extra_classes("bird, dog")
    try:
        assert added == ["bird", "dog"]
        assert dc.DEFAULT_PER_CLASS_CONF["bird"] == 0.30
        assert dc.CLASSES_OF_INTEREST["dog"] == 16
    finally:
        _cleanup(added)


def test_unknown_and_duplicate_are_ignored():
    assert dc._apply_extra_classes("dragon") == []
    assert dc._apply_extra_classes("person") == []     # already core
    added = dc._apply_extra_classes("bird")
    try:
        assert dc._apply_extra_classes("bird") == []   # second add: no-op
    finally:
        _cleanup(added)


def test_bad_gate_falls_back():
    added = dc._apply_extra_classes("bird:not_a_number")
    try:
        assert dc.DEFAULT_PER_CLASS_CONF["bird"] == 0.30
    finally:
        _cleanup(added)


def test_empty_spec_is_a_noop():
    assert dc._apply_extra_classes("") == []
    assert dc._apply_extra_classes(" , ,") == []
