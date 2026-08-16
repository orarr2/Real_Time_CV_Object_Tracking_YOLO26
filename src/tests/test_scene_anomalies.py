"""Operator-defined anomalies, night gates, entity gallery, plain header."""
import numpy as np
import pytest

import app.collector as collector
from app.collector import check_scene_anomalies, weighted_vehicle_load
from app.detect_core import night_adjusted_conf, DEFAULT_PER_CLASS_CONF
from app.model_metrics import header_line


@pytest.fixture(autouse=True)
def _fresh_cooldowns():
    collector._SCENE_ANOMALY_LAST.clear()
    collector._LAST_LUMA.clear()
    yield


def test_is_night_clock_beats_brightness():
    """Lit city streets measure mean-gray 105-120 at 2 AM - brightness alone
    never fired. The local clock must declare night regardless of luma."""
    import datetime as dt
    from app.collector import is_night
    # 23:30 Turkey local (20:30 UTC), bright street -> night anyway
    late = dt.datetime(2026, 7, 10, 20, 30, tzinfo=dt.timezone.utc)
    assert is_night(115.0, late) is True
    # 12:00 local, bright -> day
    noon = dt.datetime(2026, 7, 10, 9, 0, tzinfo=dt.timezone.utc)
    assert is_night(115.0, noon) is False
    # 12:00 local but genuinely dark frame (storm / lens fault) -> night gates
    assert is_night(30.0, noon) is True
    # 05:00 local (02:00 UTC) still night; 07:00 local is day
    assert is_night(115.0, dt.datetime(2026, 7, 10, 2, 0, tzinfo=dt.timezone.utc)) is True
    assert is_night(115.0, dt.datetime(2026, 7, 10, 4, 0, tzinfo=dt.timezone.utc)) is False


def test_night_bump_raises_every_gate_clamped():
    g = night_adjusted_conf(DEFAULT_PER_CLASS_CONF)
    for cls, v in DEFAULT_PER_CLASS_CONF.items():
        assert g[cls] == pytest.approx(min(0.8, v + 0.08))
    assert night_adjusted_conf({"car": 0.79})["car"] == 0.8


def test_weighted_vehicle_load():
    assert weighted_vehicle_load({"car": 2, "bus": 1}) == pytest.approx(4.5)
    assert weighted_vehicle_load({}) == 0.0


def test_extreme_load_fires_with_cooldown():
    counts = {"person": 60}
    v1 = check_scene_anomalies("camA", counts, [], (720, 1280), 120.0, now=1000.0)
    assert v1 and v1[0]["kind"] == "extreme_load" and v1[0]["metric"] == "person"
    # inside cooldown -> silent
    assert check_scene_anomalies("camA", counts, [], (720, 1280), 120.0,
                                 now=1100.0) == []
    # past cooldown -> fires again
    assert check_scene_anomalies("camA", counts, [], (720, 1280), 120.0,
                                 now=1000.0 + 1801)[0]["kind"] == "extreme_load"


def test_vehicle_load_extreme():
    counts = {"person": 2, "bus": 10, "truck": 6}   # load 40 >= 38
    v = check_scene_anomalies("camB", counts, [], (720, 1280), 120.0, now=5.0)
    assert v and v[0]["metric"] == "vehicles"


def test_obstruction_giant_box():
    big = [{"x1": 0, "y1": 0, "x2": 1000, "y2": 500, "cls": "bus",
            "conf": 0.8}]                                    # 54% of frame
    v = check_scene_anomalies("camC", {"person": 0}, big, (720, 1280), 120.0,
                              now=7.0)
    assert v and v[0]["kind"] == "camera_obstructed"
    small = [{"x1": 0, "y1": 0, "x2": 100, "y2": 100, "cls": "car",
              "conf": 0.9}]
    assert check_scene_anomalies("camD", {"person": 0}, small, (720, 1280),
                                 120.0, now=9.0) == []


def test_obstruction_needs_confidence():
    """The 0.25-gate `train` class hallucinated half-frame boxes five times
    in one morning - a giant box only counts as blockage when the model is
    actually sure of it."""
    flaky = [{"x1": 0, "y1": 0, "x2": 1000, "y2": 600, "cls": "train",
              "conf": 0.27}]
    assert check_scene_anomalies("camF", {}, flaky, (720, 1280), 120.0,
                                 now=11.0) == []
    sure = [{"x1": 0, "y1": 0, "x2": 1000, "y2": 600, "cls": "train",
             "conf": 0.55}]
    v = check_scene_anomalies("camF", {}, sure, (720, 1280), 120.0, now=12.0)
    assert v and v[0]["kind"] == "camera_obstructed"


def test_camera_dark_transition_only():
    # first sample just records the baseline
    assert check_scene_anomalies("camE", {}, [], (720, 1280), 120.0, now=1.0) == []
    # bright -> near-black = alarm
    v = check_scene_anomalies("camE", {}, [], (720, 1280), 10.0, now=2.0)
    assert v and v[0]["kind"] == "camera_dark"
    # night staying dark is NOT an alarm
    assert check_scene_anomalies("camE", {}, [], (720, 1280), 8.0, now=3.0) == []


def test_entity_gallery_caps(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    from app import entity_gallery as eg
    monkeypatch.setattr(eg, "PER_ENTITY_MIN_GAP_S", 0.0)
    monkeypatch.setattr(eg, "PER_ENTITY_CAP", 3)
    frame = np.full((200, 200, 3), 128, dtype=np.uint8)
    box = {"x1": 10, "y1": 10, "x2": 100, "y2": 100}
    for _ in range(5):
        assert eg.save_sighting("camA", 7, frame, box, tmp_path)
    crops = list((tmp_path / "entities" / "camA" / "7").glob("*.jpg"))
    assert len(crops) == 3                    # per-entity cap enforced
    items = eg.list_sightings("camA", 7, tmp_path)
    assert len(items) == 3 and items[0]["url"].startswith("/snapshots/entities/")


def test_learning_curve_batches_and_trend(tmp_path):
    from app.labels import ReviewStore
    from app.model_metrics import learning_curve
    store = ReviewStore(tmp_path / "reviews.json")
    # 10 frames reviewed in order: first 5 are heavy-mistake, last 5 clean -
    # the curve must show two points with a falling error rate.
    for i in range(10):
        early = i < 5
        store.submit_frame(
            f"review_frames/camA/{1000 + i}.jpg", "camA",
            {"0": "wrong" if early else "correct",
             "1": "relabel:truck" if early else "correct"},
            [{"cls": "person", "box": [1, 1, 30, 60]}] if early else [])
    pts = learning_curve(store, batch_size=5)
    assert len(pts) == 2
    assert pts[0]["frames"] == 5 and pts[1]["frames"] == 5
    assert pts[0]["error_rate"] == 1.0          # 3 mistakes / 3 signals per frame
    assert pts[1]["error_rate"] == 0.0
    assert pts[0]["batch"] == 1 and pts[1]["batch"] == 2
    # a frame with zero signals contributes nothing
    store.submit_frame("review_frames/camA/2000.jpg", "camA", {}, [])
    assert len(learning_curve(store, batch_size=5)) == 2
    # empty store -> empty curve
    empty = ReviewStore(tmp_path / "r2.json")
    assert learning_curve(empty) == []


def test_header_line_plain_language():
    m = {"tp": 3, "fp": 0, "fn": 21, "n_precision": 3, "n_recall": 24,
         "accuracy": 1.0, "recall": 0.125}
    line = header_line(m, {"adjusted_cls": 6, "updated_at": ""})
    assert "right on 3 of 3 boxes you checked" in line
    assert "100% so far - small sample" in line     # number ALWAYS shown
    assert "21 objects it missed" in line
    assert "learning is ON" in line and "6 detection thresholds" in line
    assert "precision pending" not in line and "recall" not in line
    empty = header_line({"tp": 0, "fp": 0, "fn": 0,
                         "n_precision": 0, "n_recall": 0})
    assert "no feedback yet" in empty
