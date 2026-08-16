"""User-drawn counting-line override + crossing event log.

Round-trip: save_line writes a JSON the collector's resolve_line picks up
(beating the CAMERAS entry). Malformed input is rejected. The event log
grows bounded and returns newest-first."""
import json
from pathlib import Path

import pytest

from app import cameras, live_analysis


@pytest.fixture(autouse=True)
def _isolate_lines_dir(tmp_path, monkeypatch):
    """Send every write for this test to a fresh tmp dir - no state leaks
    between tests and no touching the real data/lines/."""
    d = tmp_path / "lines"
    monkeypatch.setattr(cameras, "_lines_dir", lambda: d)
    # crossings log lives elsewhere but pin it too so tests are hermetic.
    c = tmp_path / "crossings"
    monkeypatch.setattr(live_analysis, "_crossings_dir", lambda: c)
    return d


def test_resolve_line_falls_back_to_cameras_catalog(monkeypatch):
    """No override file -> whatever CAMERAS[cam]['line'] is."""
    monkeypatch.setitem(cameras.CAMERAS, "camA",
                        {"line": [[0.1, 0.5], [0.9, 0.5]], "country": "turkey"})
    assert cameras.resolve_line("camA") == [[0.1, 0.5], [0.9, 0.5]]
    assert cameras.resolve_line("camB_unknown") is None


def test_save_line_persists_and_overrides(monkeypatch):
    monkeypatch.setitem(cameras.CAMERAS, "camA",
                        {"line": [[0.1, 0.5], [0.9, 0.5]], "country": "turkey"})
    cameras.save_line("camA", [[0.2, 0.4], [0.8, 0.6]])
    assert cameras.resolve_line("camA") == [[0.2, 0.4], [0.8, 0.6]]


def test_save_line_rejects_bad_shape():
    """Every failure mode we can express in the spec must 400 out cleanly
    - one point, three points, out-of-range coords, non-numeric."""
    for bad in [
        None,
        [],
        [[0.5, 0.5]],                            # one point
        [[0.1, 0.5], [0.9, 0.5], [0.5, 0.5]],   # three points
        [[1.5, 0.5], [0.9, 0.5]],                # x > 1
        [[-0.1, 0.5], [0.9, 0.5]],               # x < 0
        [["a", "b"], [0.9, 0.5]],                # non-numeric
    ]:
        with pytest.raises(ValueError):
            cameras.save_line("camX", bad)


def test_clear_line_returns_to_catalog(monkeypatch):
    monkeypatch.setitem(cameras.CAMERAS, "camA",
                        {"line": [[0.1, 0.5], [0.9, 0.5]], "country": "turkey"})
    cameras.save_line("camA", [[0.2, 0.4], [0.8, 0.6]])
    assert cameras.resolve_line("camA") == [[0.2, 0.4], [0.8, 0.6]]
    assert cameras.clear_line("camA") is True
    assert cameras.resolve_line("camA") == [[0.1, 0.5], [0.9, 0.5]]
    # Idempotent: clearing again just says False.
    assert cameras.clear_line("camA") is False


def test_malformed_override_falls_back_silently(monkeypatch):
    """A garbage override file must degrade to the CAMERAS default rather
    than crashing the collector on the next round."""
    monkeypatch.setitem(cameras.CAMERAS, "camA",
                        {"line": [[0.1, 0.5], [0.9, 0.5]], "country": "turkey"})
    d = cameras._lines_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "camA.json").write_text('{"line": "not-a-list"}')
    assert cameras.resolve_line("camA") == [[0.1, 0.5], [0.9, 0.5]]


def test_crossing_event_log_bounded_and_newest_first():
    class _T:
        def __init__(self, tid, cls="person"):
            self.tid = tid; self.cls = cls
            self.boxes = [{"x1": 10, "y1": 20, "x2": 40, "y2": 80}]
    # Write more than CROSSING_LOG_KEEP events; verify bound + order.
    for i in range(live_analysis.CROSSING_LOG_KEEP + 30):
        live_analysis.log_crossing_event(
            "camA", "in" if i % 2 == 0 else "out", _T(tid=i), frame=None)
    events = live_analysis.read_crossing_events("camA", limit=200)
    assert len(events) == live_analysis.CROSSING_LOG_KEEP
    # Newest-first: the very last logged is at index 0.
    assert events[0]["tid"] == live_analysis.CROSSING_LOG_KEEP + 30 - 1


def test_read_events_missing_returns_empty():
    assert live_analysis.read_crossing_events("nobody", limit=10) == []


# ---- class filter --------------------------------------------------------

def test_save_line_persists_class_filter():
    cameras.save_line("camA", [[0.1, 0.5], [0.9, 0.5]],
                      classes=["person", "car"])
    assert cameras.resolve_line("camA") == [[0.1, 0.5], [0.9, 0.5]]
    assert cameras.resolve_line_classes("camA") == ["person", "car"]


def test_save_line_rejects_unknown_class_or_empty_list():
    for bad in [[], ["dog"], ["person", "spaceship"], [1], "person"]:
        with pytest.raises(ValueError):
            cameras.save_line("camA", [[0.1, 0.5], [0.9, 0.5]], classes=bad)


def test_resolve_line_classes_none_when_absent():
    cameras.save_line("camA", [[0.1, 0.5], [0.9, 0.5]])
    assert cameras.resolve_line_classes("camA") is None
    assert cameras.resolve_line_classes("nobody") is None


def test_resolve_line_classes_falls_back_on_bad_shape():
    """A hand-edited override with a bogus classes field must degrade to
    None (permissive) rather than crashing or silently counting nothing."""
    d = cameras._lines_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "camA.json").write_text(
        '{"line": [[0.1, 0.5], [0.9, 0.5]], "classes": "person"}')
    assert cameras.resolve_line_classes("camA") is None


# ---- update_crossings: class filter + cooldown ---------------------------


class _T:
    """Minimal track stand-in that `update_crossings` and
    `log_crossing_event` both accept."""

    def __init__(self, tid, cls="person"):
        self.tid = tid
        self.cls = cls
        self.misses = 0
        self.boxes = []


def _feed(track, x, y, w=30, h=60):
    """Push a foot point at (x, y) with the standard box footprint."""
    track.boxes = [{"x1": x - w / 2, "y1": y - h, "x2": x + w / 2, "y2": y}]


def test_update_crossings_filters_by_class():
    """Vehicles cross the sidewalk line; with classes={person} nothing counts,
    with classes=None they count."""
    SHAPE = (360, 640)
    line = live_analysis.DEFAULT_LINE           # y=0.62
    car = _T(tid=1, cls="car")
    _feed(car, 320, 180)                        # y = 180/360 = 0.50 (above)
    sides, cross = {}, {}
    live_analysis.update_crossings(sides, [car], SHAPE, line, cross,
                                   classes=["person"], now=0.0)
    _feed(car, 320, 252)                        # y = 252/360 = 0.70 (below)
    live_analysis.update_crossings(sides, [car], SHAPE, line, cross,
                                   classes=["person"], now=1.0)
    assert cross == {}, "car ignored because classes=[person]"

    # Same movements without a class filter DO count.
    sides2, cross2 = {}, {}
    car2 = _T(tid=2, cls="car")
    _feed(car2, 320, 180)
    live_analysis.update_crossings(sides2, [car2], SHAPE, line, cross2,
                                   classes=None, now=0.0)
    _feed(car2, 320, 252)
    live_analysis.update_crossings(sides2, [car2], SHAPE, line, cross2,
                                   classes=None, now=1.0)
    assert cross2 == {"in": 1}


def test_update_crossings_cooldown_swallows_jitter():
    """A foot point that jitters neg->pos->neg->pos within the cooldown
    window counts ONCE, not three times."""
    SHAPE = (360, 640)
    line = live_analysis.DEFAULT_LINE           # y=0.62 -> foot_y = 223.2
    tr = _T(tid=7, cls="person")
    sides, cross = {}, {}
    ts = {}
    # Establish a signed side well above the line first.
    _feed(tr, 320, 100)
    live_analysis.update_crossings(sides, [tr], SHAPE, line, cross,
                                   last_cross_ts=ts, cooldown_s=2.0, now=0.0)
    # Real crossing #1: goes below.
    _feed(tr, 320, 260)
    live_analysis.update_crossings(sides, [tr], SHAPE, line, cross,
                                   last_cross_ts=ts, cooldown_s=2.0, now=0.1)
    # Jitter back within the cooldown - dropped.
    _feed(tr, 320, 100)
    live_analysis.update_crossings(sides, [tr], SHAPE, line, cross,
                                   last_cross_ts=ts, cooldown_s=2.0, now=0.4)
    _feed(tr, 320, 260)
    live_analysis.update_crossings(sides, [tr], SHAPE, line, cross,
                                   last_cross_ts=ts, cooldown_s=2.0, now=0.7)
    assert cross == {"in": 1}, cross
    # After the cooldown expires, a real cross counts again.
    _feed(tr, 320, 100)
    live_analysis.update_crossings(sides, [tr], SHAPE, line, cross,
                                   last_cross_ts=ts, cooldown_s=2.0, now=5.0)
    assert cross == {"in": 1, "out": 1}


def test_update_crossings_without_cooldown_is_backward_compat():
    """Passing last_cross_ts=None reproduces the old behavior: every
    signed sign flip counts, jitter included. Kept so callers that don't
    care about cooldown (tests, one-shot analyses) don't have to opt in."""
    SHAPE = (360, 640)
    line = live_analysis.DEFAULT_LINE
    tr = _T(tid=9, cls="person")
    sides, cross = {}, {"in": 0, "out": 0}
    _feed(tr, 320, 100)
    live_analysis.update_crossings(sides, [tr], SHAPE, line, cross, now=0.0)
    _feed(tr, 320, 260)
    live_analysis.update_crossings(sides, [tr], SHAPE, line, cross, now=0.1)
    _feed(tr, 320, 100)
    live_analysis.update_crossings(sides, [tr], SHAPE, line, cross, now=0.2)
    _feed(tr, 320, 260)
    live_analysis.update_crossings(sides, [tr], SHAPE, line, cross, now=0.3)
    assert cross["in"] + cross["out"] == 3


# ---- hot-reload ---------------------------------------------------------


def test_live_session_hot_reloads_line(tmp_path, monkeypatch):
    """Bumping the JSON mtime while a session object exists must pick up
    the new line + classes on the next _maybe_reload_line call, and drop
    the stale side/cooldown state."""
    monkeypatch.setitem(cameras.CAMERAS, "hotcam",
                        {"line": [[0.1, 0.5], [0.9, 0.5]], "country": "turkey"})
    cameras.save_line("hotcam", [[0.2, 0.4], [0.8, 0.4]])

    sess = live_analysis.LiveSession.__new__(live_analysis.LiveSession)
    sess.cam = cameras.CAMERAS["hotcam"]
    sess.cam_id = "hotcam"
    sess.line = cameras.resolve_line("hotcam")
    sess.line_classes = cameras.resolve_line_classes("hotcam")
    sess._line_sides = {42: -1.0}
    sess._last_cross_ts = {42: 100.0}
    sess._line_mtime = sess._line_json_mtime()
    sess._next_line_check = 0.0

    # No change on disk -> nothing happens.
    sess._maybe_reload_line(now=10.0)
    assert sess.line == [[0.2, 0.4], [0.8, 0.4]]
    assert sess._line_sides == {42: -1.0}

    # Save a fresh line + class filter; force the next check.
    cameras.save_line("hotcam", [[0.15, 0.7], [0.85, 0.7]], classes=["person"])
    sess._next_line_check = 0.0
    sess._maybe_reload_line(now=20.0)
    assert sess.line == [[0.15, 0.7], [0.85, 0.7]]
    assert sess.line_classes == ["person"]
    assert sess._line_sides == {}
    assert sess._last_cross_ts == {}

    # Clearing the override drops back to the CAMERAS catalog.
    cameras.clear_line("hotcam")
    sess._next_line_check = 0.0
    sess._maybe_reload_line(now=30.0)
    assert sess.line == [[0.1, 0.5], [0.9, 0.5]]
    assert sess.line_classes is None
