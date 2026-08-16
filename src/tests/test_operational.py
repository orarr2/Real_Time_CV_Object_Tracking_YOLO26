"""Operational-analysis features: ROI, line crossings, loitering, alert sink.

Run from src/:  python -m pytest tests -q
"""
import numpy as np
import pytest

from app.alerts import AlertSink
from app.detect_core import (
    count_line_crossings,
    counts_from_boxes,
    filter_boxes_roi,
    point_in_polygon,
    track_burst,
)
from app.presence import PresenceTracker

FRAME_SHAPE = (100, 200, 3)          # H=100, W=200


def person(x1, y1, x2, y2, cls="person"):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cls": cls, "conf": 0.9}


# ---- ROI ---------------------------------------------------------------------

SQUARE = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]


def test_point_in_polygon():
    assert point_in_polygon(0.5, 0.5, SQUARE)
    assert not point_in_polygon(0.1, 0.5, SQUARE)
    assert not point_in_polygon(0.5, 0.9, SQUARE)
    assert not point_in_polygon(0.5, 0.5, [[0, 0], [1, 1]])   # degenerate


def test_roi_filters_by_foot_point():
    inside  = person(90, 20, 110, 60)    # foot (100, 60) -> (0.5, 0.6) in ROI
    outside = person(90, 20, 110, 95)    # foot (100, 95) -> (0.5, 0.95) below ROI
    kept = filter_boxes_roi([inside, outside], FRAME_SHAPE, SQUARE)
    assert kept == [inside]


def test_roi_exclude_carves_out():
    b = person(90, 20, 110, 60)
    hole = [[0.4, 0.5], [0.6, 0.5], [0.6, 0.7], [0.4, 0.7]]
    assert filter_boxes_roi([b], FRAME_SHAPE, SQUARE, [hole]) == []


def test_no_roi_passes_everything():
    b = person(0, 0, 5, 5)
    assert filter_boxes_roi([b], FRAME_SHAPE, None) == [b]


def test_counts_from_boxes():
    boxes = [person(0, 0, 5, 5), person(10, 0, 15, 5),
             person(20, 0, 25, 5, cls="car"), person(30, 0, 35, 5, cls="bus")]
    c = counts_from_boxes(boxes)
    assert c["person"] == 2 and c["car"] == 1 and c["vehicles"] == 2


# ---- burst tracking + line crossings ------------------------------------------

def test_track_burst_follows_moving_object():
    f0 = [person(10, 40, 30, 80)]
    f1 = [person(24, 40, 44, 80)]
    f2 = [person(38, 40, 58, 80)]
    tracks = track_burst([f0, f1, f2], FRAME_SHAPE)
    assert len(tracks) == 1
    assert len(tracks[0]) == 3


def test_track_burst_separates_distant_objects():
    f0 = [person(10, 40, 30, 80), person(160, 40, 180, 80)]
    f1 = [person(20, 40, 40, 80), person(150, 40, 170, 80)]
    tracks = track_burst([f0, f1], FRAME_SHAPE)
    assert sorted(len(t) for t in tracks) == [2, 2]


def test_track_burst_class_mismatch_never_matches():
    f0 = [person(10, 40, 30, 80)]
    f1 = [person(12, 40, 32, 80, cls="car")]
    tracks = track_burst([f0, f1], FRAME_SHAPE)
    assert sorted(len(t) for t in tracks) == [1, 1]


def test_line_crossing_in_and_out():
    # Horizontal line across the middle: A=(0,0.5) B=(1,0.5). Crossing from
    # ABOVE (negative side) to BELOW is "in" with this point order.
    line = [[0.0, 0.5], [1.0, 0.5]]
    going_down = [[person(90, 10, 110, 40)],   # foot y=40 (0.4, above)
                  [person(90, 30, 110, 70)]]   # foot y=70 (0.7, below)
    tracks = track_burst(going_down, FRAME_SHAPE)
    res = count_line_crossings(tracks, FRAME_SHAPE, line)
    assert res["in"] == 1 and res["out"] == 0 and res["person_in"] == 1

    going_up = [[person(90, 30, 110, 70)], [person(90, 10, 110, 40)]]
    res = count_line_crossings(track_burst(going_up, FRAME_SHAPE), FRAME_SHAPE, line)
    assert res["out"] == 1 and res["in"] == 0


def test_line_no_crossing_when_object_stays_on_one_side():
    line = [[0.0, 0.5], [1.0, 0.5]]
    frames = [[person(90, 10, 110, 30)], [person(100, 10, 120, 32)]]
    res = count_line_crossings(track_burst(frames, FRAME_SHAPE), FRAME_SHAPE, line)
    assert res["in"] == 0 and res["out"] == 0


def test_vehicle_crossing_counted_in_vehicle_bucket():
    line = [[0.0, 0.5], [1.0, 0.5]]
    frames = [[person(90, 10, 110, 40, cls="car")],
              [person(90, 30, 110, 70, cls="car")]]
    res = count_line_crossings(track_burst(frames, FRAME_SHAPE), FRAME_SHAPE, line)
    assert res["vehicles_in"] == 1 and res["person_in"] == 0


def test_line_crossing_counts_each_flip_not_just_the_last():
    # A track that walked in, then back out, then in again must register
    # ALL THREE crossings - the old overwrite pattern (`crossed = "in"` /
    # `crossed = "out"`) kept only the last direction, so a back-and-forth
    # track was counted once and busy pedestrians were undercounted.
    line = [[0.0, 0.5], [1.0, 0.5]]
    frames = [
        [person(90, 10, 110, 40)],   # above (side < 0)
        [person(90, 30, 110, 70)],   # below (side > 0)   -> in
        [person(90, 10, 110, 40)],   # above              -> out
        [person(90, 30, 110, 70)],   # below              -> in
    ]
    res = count_line_crossings(track_burst(frames, FRAME_SHAPE),
                               FRAME_SHAPE, line)
    assert res["in"] == 2 and res["out"] == 1
    assert res["person_in"] == 2 and res["person_out"] == 1


def test_line_crossing_ignores_touching_exactly():
    # A foot point that lands EXACTLY on the counting line (side == 0)
    # is ambiguous; the previous convention counted 0 as "positive side"
    # and turned a neg -> 0 -> neg jitter into one false crossing. Now
    # touching frames are neither counted nor allowed to reset the last
    # known side.
    line = [[0.0, 0.5], [1.0, 0.5]]           # normalized y = 0.5
    # FRAME_SHAPE is (H, W); pick a foot that lands exactly on 0.5*H.
    H = FRAME_SHAPE[0]
    on = int(0.5 * H)
    frames = [
        [person(90, on - 30, 110, on - 10)],  # foot above (neg)
        [person(90, on - 20, 110, on)],       # foot exactly on line
        [person(90, on - 30, 110, on - 10)],  # foot above again (neg)
    ]
    res = count_line_crossings(track_burst(frames, FRAME_SHAPE),
                               FRAME_SHAPE, line)
    assert res["in"] == 0 and res["out"] == 0


# ---- loitering ----------------------------------------------------------------

BOX = {"x1": 50, "y1": 30, "x2": 90, "y2": 90}
BOX_NUDGE = {"x1": 52, "y1": 31, "x2": 92, "y2": 91}
BOX_FAR   = {"x1": 150, "y1": 30, "x2": 190, "y2": 90}


def test_loiter_fires_after_threshold():
    p = PresenceTracker(person_sec=300)
    t = 1000.0
    assert p.observe("cam", 7, "person", BOX, FRAME_SHAPE, now=t) is None
    for i in range(1, 9):                      # 8 more samples, 40s apart
        ev = p.observe("cam", 7, "person", BOX_NUDGE, FRAME_SHAPE, now=t + i * 40)
    assert ev is not None and ev["kind"] == "loiter"
    assert ev["duration_sec"] >= 300


def test_loiter_movement_resets_stay():
    p = PresenceTracker(person_sec=300)
    t = 1000.0
    p.observe("cam", 7, "person", BOX, FRAME_SHAPE, now=t)
    for i in range(1, 5):
        p.observe("cam", 7, "person", BOX_NUDGE, FRAME_SHAPE, now=t + i * 40)
    # walks across the scene -> stay resets
    p.observe("cam", 7, "person", BOX_FAR, FRAME_SHAPE, now=t + 5 * 40)
    ev = p.observe("cam", 7, "person", BOX_FAR, FRAME_SHAPE, now=t + 9 * 40)
    assert ev is None                          # only ~160s at the new spot


def test_loiter_continuity_gap_resets_stay():
    p = PresenceTracker(person_sec=300, continuity_gap_sec=180)
    t = 1000.0
    p.observe("cam", 7, "person", BOX, FRAME_SHAPE, now=t)
    for i in range(1, 5):
        p.observe("cam", 7, "person", BOX, FRAME_SHAPE, now=t + i * 40)
    # unmatched for 10 minutes -> chain broken
    ev = p.observe("cam", 7, "person", BOX, FRAME_SHAPE, now=t + 800)
    assert ev is None
    ev = p.observe("cam", 7, "person", BOX, FRAME_SHAPE, now=t + 840)
    assert ev is None                          # stay restarted at t+800


def test_loiter_realert_cooldown():
    p = PresenceTracker(person_sec=100, realert_sec=1800)
    t = 1000.0
    p.observe("cam", 7, "person", BOX, FRAME_SHAPE, now=t)
    ev = p.observe("cam", 7, "person", BOX_NUDGE, FRAME_SHAPE, now=t + 120)
    assert ev is not None
    ev2 = p.observe("cam", 7, "person", BOX_NUDGE, FRAME_SHAPE, now=t + 160)
    assert ev2 is None                         # within realert window


def test_loiter_roi_gate():
    cam = {"loiter_roi": [[0.6, 0.0], [1.0, 0.0], [1.0, 1.0], [0.6, 1.0]]}
    p = PresenceTracker(person_sec=100)
    t = 1000.0
    # BOX foot point is (70/200=0.35, 0.9) -> outside loiter_roi -> silent
    p.observe("cam", 7, "person", BOX, FRAME_SHAPE, cam=cam, now=t)
    assert p.observe("cam", 7, "person", BOX_NUDGE, FRAME_SHAPE, cam=cam, now=t + 120) is None
    # BOX_FAR foot point (170/200=0.85, 0.9) -> inside -> fires
    p.observe("cam", 8, "person", BOX_FAR, FRAME_SHAPE, cam=cam, now=t)
    ev = p.observe("cam", 8, "person",
                   {"x1": 152, "y1": 31, "x2": 192, "y2": 91},
                   FRAME_SHAPE, cam=cam, now=t + 120)
    assert ev is not None


def test_loiter_per_camera_threshold_override():
    cam = {"loiter_person_sec": 60}
    p = PresenceTracker(person_sec=300)
    t = 1000.0
    p.observe("cam", 7, "person", BOX, FRAME_SHAPE, cam=cam, now=t)
    ev = p.observe("cam", 7, "person", BOX_NUDGE, FRAME_SHAPE, cam=cam, now=t + 80)
    assert ev is not None


# ---- static-object gate (2026-07-22 fix for FP kiosk/awning loiters) --------

BOX_STATIC = {"x1": 50, "y1": 30, "x2": 90, "y2": 90}     # same as BOX


def test_static_object_never_fires_loiter():
    """The 2026-07-22 midday+evening reports flagged loitering on a kiosk
    at Taksim (class 'car' for 1000s) and an awning at Eyup Sultan (class
    'bus' for 920s). Both had IoU ~1.0 between first-vs-current box - the
    hallmark of a static structure YOLO keeps re-classifying. The gate
    must refuse to alert regardless of duration."""
    p = PresenceTracker(vehicle_sec=100)          # match the report's "car"/"bus"
    t = 1000.0
    p.observe("cam", 7, "car", BOX_STATIC, FRAME_SHAPE, now=t)
    # 8 samples over 320s, box never moves - IoU stays at 1.0
    for i in range(1, 9):
        ev = p.observe("cam", 7, "car", BOX_STATIC, FRAME_SHAPE,
                       now=t + i * 40)
    assert ev is None                # duration exceeds threshold but IoU=1.0
    # Long tail - still no alert
    ev2 = p.observe("cam", 7, "car", BOX_STATIC, FRAME_SHAPE, now=t + 2000)
    assert ev2 is None


def test_static_object_starts_moving_and_fires():
    """Symmetric: a car that stood still for 5 minutes then rolled a bit
    IS a real loiter case (parked car repositioning). Once the box moves
    below the static-IoU threshold, the alert must fire."""
    p = PresenceTracker(vehicle_sec=100)
    t = 1000.0
    p.observe("cam", 7, "car", BOX, FRAME_SHAPE, now=t)
    # Continuity samples every 40s so the stay does not reset (default gap
    # is 180s); box is static so duration accumulates without firing.
    for i in range(1, 6):
        ev = p.observe("cam", 7, "car", BOX, FRAME_SHAPE, now=t + i * 40)
        assert ev is None                # static-object gate suppresses each
    # After ~200s and one small real drift (IoU ~0.66) - alert fires.
    BOX_DRIFT = {"x1": 56, "y1": 34, "x2": 96, "y2": 94}
    ev = p.observe("cam", 7, "car", BOX_DRIFT, FRAME_SHAPE, now=t + 240)
    assert ev is not None


# ---- alert sink ----------------------------------------------------------------

class CapturingSink(AlertSink):
    def __init__(self, **kw):
        super().__init__(telegram_token="T", telegram_chat_id="C",
                         webhook_url="https://example.invalid/hook", **kw)
        self.posts = []

    def _post(self, url, data, content_type):
        self.posts.append((url, data, content_type))


def test_alert_sends_to_both_backends():
    s = CapturingSink()
    ok = s.send("loiter", "cam", "slot", "2026-07-05T12:00:00Z",
                "title", "body", image_jpeg=b"\xff\xd8jpeg")
    assert ok
    urls = [u for u, _, _ in s.posts]
    assert any("api.telegram.org" in u and "sendPhoto" in u for u in urls)
    assert any("example.invalid" in u for u in urls)


def test_alert_per_key_cooldown():
    s = CapturingSink()
    assert s.send("loiter", "cam", "slot", "ts", "t")
    assert not s.send("loiter", "cam", "slot", "ts", "t")   # same kind+cam
    assert s.send("loiter", "other_cam", "slot", "ts", "t") # different cam ok


def test_alert_global_hourly_cap():
    s = CapturingSink(global_hourly_cap=2, per_key_cooldown_s=0)
    assert s.send("a", "c1", "s", "ts", "t")
    assert s.send("a", "c2", "s", "ts", "t")
    assert not s.send("a", "c3", "s", "ts", "t")


def test_alert_disabled_without_backends():
    s = AlertSink(telegram_token=None, telegram_chat_id=None, webhook_url=None)
    assert not s.enabled
    assert not s.send("a", "c", "s", "ts", "t")


def test_alert_backend_failure_never_raises():
    class FailingSink(CapturingSink):
        def _post(self, url, data, content_type):
            raise OSError("network down")
    s = FailingSink()
    assert s.send("a", "c", "s", "ts", "t") is False
