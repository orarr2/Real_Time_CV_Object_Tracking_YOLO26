"""Deep-window behavior profiles: per-individual stats + rendering.

Run from src/:  python -m pytest tests -q
"""
import pytest

from app.behavior import track_stats
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






