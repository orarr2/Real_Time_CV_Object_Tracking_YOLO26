"""Deep-window behavior profiles: per-individual stats + rendering.

Run from src/:  python -m pytest tests -q
"""
import pytest

from app.behavior import (
    _boxes_of_last_frame,
    attach_neighbor_stats,
    render_window,
    track_stats,
)
from app.tracker import Track

SHAPE = (360, 640)


def _box(x, y, w=30, h=60, cls="person", conf=0.9):
    return {"x1": x, "y1": y, "x2": x + w, "y2": y + h,
            "cls": cls, "conf": conf}


def _track(tid, positions, cls="person", dt=0.5, w=30, h=60):
    boxes = [_box(x, y, w=w, h=h, cls=cls) for x, y in positions]
    tr = Track(tid, boxes[0], 0.0)
    for i, b in enumerate(boxes[1:], start=1):
        tr.add(b, i * dt)
    return tr


def test_straight_walker_profile():
    tr = _track(1, [(100 + 40 * i, 100) for i in range(5)])
    s = track_stats(tr.cls, tr.boxes, tr.times, SHAPE)
    assert s["sightings"] == 5
    assert s["path_len_px"] == pytest.approx(160.0)
    assert s["net_disp_px"] == pytest.approx(160.0)
    assert s["moving_frac"] == 1.0
    assert s["stationary"] is False
    assert s["direction"] == "right"
    assert s["mean_speed_px_s"] == pytest.approx(80.0)   # 40px / 0.5s
    assert s["kmh_est"] is None                          # people get no ruler
    assert len(s["path"]) == 5


def test_jitterer_reads_as_stationary():
    tr = _track(1, [(100, 100), (101, 100), (100, 101), (101, 101)])
    s = track_stats(tr.cls, tr.boxes, tr.times, SHAPE)
    assert s["stationary"] is True
    assert s["direction"] is None
    assert s["moving_frac"] == 0.0


def test_vehicle_kmh_ruler():
    # Car 100px long moving 50px per 0.5s step -> 100 px/s.
    # m/px = 4.5/100; kmh = 100 * 0.045 * 3.6 = 16.2.
    tr = _track(2, [(100 + 50 * i, 200) for i in range(4)],
                cls="car", w=100, h=40)
    s = track_stats(tr.cls, tr.boxes, tr.times, SHAPE)
    assert s["kmh_est"] == pytest.approx(16.2, abs=0.1)


def test_direction_octants():
    down = track_stats("person",
                       [_box(100, 50 + 40 * i) for i in range(3)],
                       [0.0, 0.5, 1.0], SHAPE)
    assert down["direction"] == "down"
    diag = track_stats("person",
                       [_box(100 + 40 * i, 50 + 40 * i) for i in range(3)],
                       [0.0, 0.5, 1.0], SHAPE)
    assert diag["direction"] == "down-right"


def test_zones_follow_the_path():
    # 300px of travel crosses several 20px-wide grid columns.
    s = track_stats("person",
                    [_box(20 + 100 * i, 100) for i in range(4)],
                    [0.0, 0.5, 1.0, 1.5], SHAPE)
    assert len(s["zones"]) >= 3
    assert all("," in z for z in s["zones"])


def test_neighbor_stats_same_class_only():
    a = _track(1, [(100, 100), (140, 100)])
    b = _track(2, [(200, 100), (240, 100)])           # 100px to the right
    c = _track(3, [(100, 300), (140, 300)])
    c.cls = "car"
    for tr in (a, b):
        assert tr.cls == "person"
    stats = [track_stats(t.cls, t.boxes, t.times, SHAPE) for t in (a, b, c)]
    attach_neighbor_stats([a, b, c], stats)
    assert stats[0]["nn_min_px"] == pytest.approx(100.0)
    assert stats[1]["nn_min_px"] == pytest.approx(100.0)
    assert stats[2]["nn_min_px"] is None              # only car in view


def test_boxes_of_last_frame_picks_survivors():
    a = _track(1, [(0, 0), (10, 0), (20, 0)])         # lives to t=1.0
    b = _track(2, [(300, 300)])                       # died at t=0
    picked = _boxes_of_last_frame([a, b])
    assert picked == [a.boxes[-1]]


def test_render_window_draws_something():
    import numpy as np
    frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(3)]
    tr = _track(1, [(100 + 60 * i, 100) for i in range(3)])
    tr.boxes[-1]["track_id"] = tr.tid
    out = render_window(frames, [tr])
    assert out.shape == frames[0].shape
    assert out.sum() > 0
