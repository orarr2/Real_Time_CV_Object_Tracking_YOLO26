"""Burst tracker: stable per-individual ids from position + motion.

Run from src/:  python -m pytest tests -q
"""
from app.tracker import BurstTracker, assign_burst_ids

SHAPE = (360, 640)   # H, W -> diagonal ~734


def _box(x, y, w=30, h=60, cls="person", conf=0.9):
    return {"x1": x, "y1": y, "x2": x + w, "y2": y + h,
            "cls": cls, "conf": conf}


def test_single_object_keeps_one_id():
    frames = [[_box(100 + 20 * i, 100)] for i in range(4)]
    tracks = assign_burst_ids(frames, SHAPE)
    assert len(tracks) == 1
    assert all(b["track_id"] == tracks[0].tid
               for boxes in frames for b in boxes)


def test_crossing_objects_keep_their_ids():
    """Two same-class objects pass through each other. Velocity prediction
    keeps each id on its own trajectory; plain nearest-centroid matching
    would swap them at the crossing point (B lands exactly where A WAS)."""
    a = [_box(0, 100), _box(100, 100), _box(200, 100), _box(300, 100)]
    b = [_box(300, 100), _box(200, 100), _box(100, 100), _box(0, 100)]
    frames = [[a[i], b[i]] for i in range(4)]
    tracks = assign_burst_ids(frames, SHAPE)
    assert len(tracks) == 2
    ida = a[0]["track_id"]
    idb = b[0]["track_id"]
    assert ida != idb
    assert [x["track_id"] for x in a] == [ida] * 4
    assert [x["track_id"] for x in b] == [idb] * 4


def test_fifty_lookalikes_get_fifty_ids():
    """The flock case: many identical objects, distinguished by position
    alone. Every one keeps its own id across the window."""
    n = 50
    frames = []
    for t in range(3):
        frames.append([_box(12 * i, 100 + 4 * t, w=8, h=8, cls="bird",
                            conf=0.8) for i in range(n)])
    tracks = assign_burst_ids(frames, SHAPE)
    assert len(tracks) == n
    for i in range(n):
        ids = {frames[t][i]["track_id"] for t in range(3)}
        assert len(ids) == 1        # each individual kept exactly one id


def test_low_conf_cannot_start_a_track():
    frames = [[_box(100, 100, conf=0.3)], [_box(120, 100, conf=0.3)]]
    tracks = assign_burst_ids(frames, SHAPE)
    assert tracks == []
    assert "track_id" not in frames[0][0]


def test_low_conf_extends_an_existing_track():
    """A confident birth, then a gate-hugging sighting near the predicted
    spot: the id survives (BYTE's second association stage)."""
    frames = [[_box(100, 100, conf=0.9)],
              [_box(120, 100, conf=0.9)],
              [_box(140, 100, conf=0.28)]]
    tracks = assign_burst_ids(frames, SHAPE)
    assert len(tracks) == 1
    assert frames[2][0]["track_id"] == tracks[0].tid
    assert tracks[0].hits == 3


def test_miss_coasting_reclaims_the_object():
    """One dropped frame (occlusion) does not retire the id."""
    frames = [[_box(100, 100)], [_box(130, 100)], [], [_box(190, 100)]]
    tracks = assign_burst_ids(frames, SHAPE)
    assert len(tracks) == 1
    assert frames[3][0]["track_id"] == frames[0][0]["track_id"]


def test_class_consistency():
    frames = [[_box(100, 100, cls="person")],
              [_box(105, 100, cls="car")]]
    tracks = assign_burst_ids(frames, SHAPE)
    assert len(tracks) == 2       # a car never continues a person track


def test_teleport_beyond_budget_is_a_new_individual():
    frames = [[_box(0, 0)], [_box(600, 300)]]
    tracks = assign_burst_ids(frames, SHAPE)
    assert len(tracks) == 2


def test_update_streaming_api_matches_batch():
    trk = BurstTracker(SHAPE)
    b0, b1 = _box(50, 50), _box(70, 50)
    trk.update([b0], 0.0)
    trk.update([b1], 1.0)
    tracks = trk.close()
    assert len(tracks) == 1
    assert b0["track_id"] == b1["track_id"] == tracks[0].tid


def test_empty_burst():
    assert assign_burst_ids([], SHAPE) == []
    assert assign_burst_ids([[]], SHAPE) == []
