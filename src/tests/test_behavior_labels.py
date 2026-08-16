"""Behavior labels: kinematic vocabulary + posture flags, all synthetic.

Run from src/:  python -m pytest tests -q
"""
from app.behavior import track_stats
from app.behavior_labels import (heading_turns, label_track, pose_flags_of)

SHAPE = (360, 640)          # diag ~ 734 px


def _box(x, y, w=30, h=60, cls="person", conf=0.9):
    return {"x1": x, "y1": y, "x2": x + w, "y2": y + h,
            "cls": cls, "conf": conf}


def _row(positions, cls="person", dt=0.5, w=30, h=60):
    boxes = [_box(x, y, w=w, h=h, cls=cls) for x, y in positions]
    times = [i * dt for i in range(len(boxes))]
    return track_stats(cls, boxes, times, SHAPE)


def _kps(overrides: dict) -> list:
    """A 17-keypoint frame; unspecified joints get conf 0 (ignored)."""
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    for idx, (x, y) in overrides.items():
        kps[idx] = [float(x), float(y), 0.9]
    return kps


# Upright anatomy: shoulders y=100, hips y=140, knees y=175, ankles y=210.
def _upright(wrist_left_y=130, wrist_right_y=130):
    return _kps({0: (100, 80),
                 5: (90, 100), 6: (110, 100),
                 7: (85, 120), 8: (115, 120),
                 9: (85, wrist_left_y), 10: (115, wrist_right_y),
                 11: (92, 140), 12: (108, 140),
                 13: (92, 175), 14: (108, 175),
                 15: (92, 210), 16: (108, 210)})


def test_walker_labels_walking():
    out = label_track(_row([(100 + 40 * i, 100) for i in range(5)]), SHAPE)
    assert out["label"] == "walking"
    assert out["alert"] is False
    assert out["label_reasons"]


def test_stationary_person_stands_vehicle_parks():
    jitter = [(100, 100), (101, 100), (100, 101), (101, 101)]
    assert label_track(_row(jitter), SHAPE)["label"] == "standing"
    assert label_track(_row(jitter, cls="car", w=100, h=40),
                       SHAPE)["label"] == "parked"


def test_moving_vehicle_drives_not_runs():
    # 100 px per 0.5s step = 200 px/s - "running" speed, but a car.
    out = label_track(_row([(50 + 100 * i, 200) for i in range(4)],
                           cls="car", w=100, h=40), SHAPE)
    assert out["label"] == "driving"


def test_fast_person_runs():
    # 60 px per 0.5s = 120 px/s >= 0.12 * 734 ~ 88.
    out = label_track(_row([(50 + 60 * i, 200) for i in range(5)]), SHAPE)
    assert out["label"] == "running"
    assert out["alert"] is False


def test_zigzag_is_erratic_and_alerts():
    # Four full reversals along x, each step 60 px.
    xs = [100, 160, 100, 160, 100, 160]
    out = label_track(_row([(x, 200) for x in xs]), SHAPE)
    assert out["label"] == "erratic"
    assert out["alert"] is True


def test_heading_turns_ignores_jitter_steps():
    # 1-px wiggles carry no heading -> no turns counted.
    path = [[i * 0.5, (100 + (i % 2)) / 640, 0.5] for i in range(6)]
    assert heading_turns(path, SHAPE) == 0


def test_pose_hand_raised_flag():
    flags = pose_flags_of(_upright(wrist_left_y=90))   # above shoulder
    assert "hand_raised" in flags
    assert "both_hands_up" not in flags
    both = pose_flags_of(_upright(wrist_left_y=90, wrist_right_y=90))
    assert "both_hands_up" in both


def test_pose_fall_suspect_from_horizontal_torso():
    # Shoulders and hips on one horizontal line = torso 90deg from vertical.
    kps = _kps({5: (100, 150), 6: (110, 150),
                11: (160, 152), 12: (170, 152)})
    assert "fall_suspect" in pose_flags_of(kps)


def test_pose_crouch_from_folded_legs():
    # Torso 40 px, hip-to-ankle only 20 px -> deep crouch.
    kps = _kps({5: (90, 100), 6: (110, 100),
                11: (92, 140), 12: (108, 140),
                13: (92, 150), 14: (108, 150),
                15: (92, 160), 16: (108, 160)})
    assert "crouching" in pose_flags_of(kps)


def test_fall_flag_needs_two_frames_and_wins_priority():
    row = _row([(100 + 40 * i, 100) for i in range(5)])   # walking pace
    fallen = _kps({5: (100, 150), 6: (110, 150),
                   11: (160, 152), 12: (170, 152)})
    one_frame = [None, fallen, None, None, None]
    assert label_track(row, SHAPE, one_frame)["label"] == "walking"
    two_frames = [None, fallen, fallen, None, None]
    out = label_track(row, SHAPE, two_frames)
    assert out["label"] == "fall_suspect"
    assert out["alert"] is True
    assert "fall_suspect" in out["pose_flags"]


def test_upright_walker_has_no_pose_flags():
    row = _row([(100 + 40 * i, 100) for i in range(5)])
    seq = [_upright() for _ in range(5)]
    out = label_track(row, SHAPE, seq)
    assert out["pose_flags"] == []
    assert out["label"] == "walking"
