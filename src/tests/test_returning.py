"""Returning-visitor authenticity gates + re-ID registry behaviors.

Run from src/:  python -m pytest tests -q
"""
import numpy as np
import pytest

from app.collector import (
    CamObservationLog,
    RETURNING_GAP_SEC,
    _passes_returning_gates,
)
from app.detect_core import box_iou
from app.reid import ReidResult, ReidStore


def result(**kw):
    base = dict(entity_id=7, cls="person", is_new=False, sightings=5,
                similarity=0.99, gap_seconds=600.0)
    base.update(kw)
    return ReidResult(**base)


BOX_A = {"x1": 10, "y1": 10, "x2": 60, "y2": 110}     # entity's old spot
BOX_A_NUDGED = {"x1": 12, "y1": 11, "x2": 62, "y2": 112}
BOX_FAR = {"x1": 300, "y1": 200, "x2": 350, "y2": 300}


def gates(r, **kw):
    base = dict(gap_min_sec=RETURNING_GAP_SEC, sim_min=0.96, min_prior=2,
                cooldown_sec=1800, last_save_for_eid={},
                unobserved_sec=0.0, prev_box=BOX_A, new_box=BOX_FAR)
    base.update(kw)
    return _passes_returning_gates(r, **base)


def test_genuine_return_saves():
    passes, why = gates(result())
    assert passes and why == "save"


def test_short_gap_rejected():
    passes, why = gates(result(gap_seconds=RETURNING_GAP_SEC - 1))
    assert not passes and why == "short_gap"


def test_unobserved_gap_rejected():
    """Entity 'gone' for 10 min but the camera itself wasn't sampled for 6 of
    them (outage / fallback episode) - nothing returned, we looked away."""
    passes, why = gates(result(gap_seconds=600), unobserved_sec=360.0)
    assert not passes and why == "unobserved_gap"


def test_no_observation_history_rejected():
    passes, why = gates(result(), unobserved_sec=None)
    assert not passes and why == "no_observation_history"


def test_static_object_rejected():
    """Re-appearing in (almost) the same box = parked car / banner, not a
    return."""
    passes, why = gates(result(), prev_box=BOX_A, new_box=BOX_A_NUDGED)
    assert not passes and why == "static_object"


def test_first_sighting_has_no_prev_box_and_can_save():
    passes, why = gates(result(), prev_box=None, new_box=BOX_FAR)
    assert passes and why == "save"


def test_same_spot_flicker_rejected():
    """A fixed structure's detection flickers (narrow <-> wide around the
    same portal) so consecutive boxes overlap UNDER the IoU bar - but the
    center never leaves the spot. The Beyazit university-gate arch rode
    exactly this into three digests as a 260px returning 'person'
    (entity #166134): IoU 0.17 here slips the static_object gate, the
    center test still kills it."""
    arch_now = {"x1": 30, "y1": 30, "x2": 42, "y2": 150}
    assert box_iou(BOX_A, arch_now) < 0.35        # slips the IoU gate...
    passes, why = gates(result(), prev_box=BOX_A, new_box=arch_now)
    assert not passes and why == "same_spot"      # ...caught by the center


def test_box_iou_sanity():
    assert box_iou(BOX_A, BOX_A) == pytest.approx(1.0)
    assert box_iou(BOX_A, BOX_FAR) == 0.0
    assert box_iou(None, BOX_A) == 0.0
    assert box_iou(BOX_A, BOX_A_NUDGED) > 0.85


# ---- CamObservationLog -------------------------------------------------------

def test_observation_log_detects_sampling_hole(monkeypatch):
    import app.collector as collector
    now = [1000.0]
    monkeypatch.setattr(collector.time, "time", lambda: now[0])

    log = CamObservationLog(hole_threshold_sec=180.0)
    for t in range(1000, 2001, 40):           # sampled every 40s for ~17 min
        now[0] = float(t)
        log.record_success("cam")
    # continuous coverage -> essentially nothing unobserved
    now[0] = 2040.0
    assert log.unobserved_during("cam", 900) < 180

    # 600s outage, then sampling resumes
    now[0] = 2600.0
    log.record_success("cam")
    now[0] = 2640.0
    log.record_success("cam")
    unobs = log.unobserved_during("cam", 700)  # entity gap spans the outage
    assert unobs >= 500


def test_observation_log_pre_start_is_blind(monkeypatch):
    import app.collector as collector
    now = [5000.0]
    monkeypatch.setattr(collector.time, "time", lambda: now[0])
    log = CamObservationLog()
    log.record_success("cam")
    # a 40-min gap that mostly predates the process start counts as unobserved
    assert log.unobserved_during("cam", 2400) >= 2300


def test_observation_log_seed_extends_coverage(monkeypatch):
    """Restart recovery: replayed history timestamps count as observed time,
    so a long-gap return right after a restart isn't blanket-suppressed."""
    import app.collector as collector
    now = [10_000.0]
    monkeypatch.setattr(collector.time, "time", lambda: now[0])
    log = CamObservationLog(hole_threshold_sec=180.0)
    log.seed("cam", [10_000.0 - 3600 + i * 40 for i in range(90)])  # last hour
    log.record_success("cam")
    assert log.unobserved_during("cam", 3000) < 200


def test_observation_log_adapts_to_slow_rounds(monkeypatch):
    """On an undersized VM every round takes ~200s (> the fixed 180s hole
    threshold). Normal sampling must NOT be classified as one long hole -
    that would silently disable returning-visitor detection entirely."""
    import app.collector as collector
    now = [1000.0]
    monkeypatch.setattr(collector.time, "time", lambda: now[0])
    log = CamObservationLog(hole_threshold_sec=180.0)
    for _ in range(20):                    # healthy-but-slow cadence
        now[0] += 200.0
        log.record_success("cam")
    now[0] += 200.0
    assert log.unobserved_during("cam", 1800) < 400
    # but a REAL outage (5x the cadence) still counts
    now[0] += 1000.0
    log.record_success("cam")
    assert log.unobserved_during("cam", 1200) >= 900


# ---- re-ID registry ----------------------------------------------------------

def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_reid_ema_drifts_stored_embedding(tmp_path):
    store = ReidStore(tmp_path / "reid.db", threshold=0.92)
    e1 = _unit([1.0, 0.0, 0.0, 0.2])
    r1 = store.query("cam", "car", e1)
    assert r1.is_new

    e2 = _unit([1.0, 0.25, 0.0, 0.2])         # same car, slightly different light
    assert float(np.dot(e1, e2)) >= 0.92
    r2 = store.query("cam", "car", e2)
    assert not r2.is_new and r2.entity_id == r1.entity_id
    assert r2.sightings == 2

    blob = store.conn.execute(
        "SELECT embedding FROM entities WHERE entity_id=?",
        (r1.entity_id,)).fetchone()[0]
    stored = np.frombuffer(blob, dtype=np.float32)
    assert np.linalg.norm(stored) == pytest.approx(1.0, abs=1e-5)
    # drifted toward e2: closer to e2 than the original e1 was
    assert float(np.dot(stored, e2)) > float(np.dot(e1, e2))
    store.close()


def test_same_frame_boxes_cannot_match_one_entity(tmp_path):
    """Two similar objects visible at once (two white cars) must become two
    entities - not double-count sightings on one entity and drag its EMA
    embedding toward a blend of different physical objects."""
    store = ReidStore(tmp_path / "reid.db", threshold=0.92)
    e = _unit([1.0, 0.1, 0.0, 0.2])
    frame = np.random.randint(0, 255, (200, 200, 3), np.uint8)
    frame[:] = 200                                    # uniform bright frame ->
    boxes = [                                         # near-identical crops
        {"x1": 10,  "y1": 10, "x2": 60,  "y2": 110, "cls": "car", "conf": .9},
        {"x1": 120, "y1": 10, "x2": 170, "y2": 110, "cls": "car", "conf": .9},
    ]
    results = store.update_from_frame("cam", frame, boxes)
    assert len(results) == 2
    assert results[0].entity_id != results[1].entity_id
    assert results[1].is_new
    store.close()


def test_update_from_frame_box_index_alignment(tmp_path):
    """Skipped (degenerate/tiny) boxes must not shift which box a result maps
    to - the old positional zip saved the WRONG crop for every result after a
    skip."""
    store = ReidStore(tmp_path / "reid.db")
    frame = np.random.randint(0, 255, (200, 200, 3), np.uint8)
    boxes = [
        {"x1": 50, "y1": 50, "x2": 40, "y2": 90, "cls": "person", "conf": .9},  # degenerate
        {"x1": 2,  "y1": 2,  "x2": 4,  "y2": 5,  "cls": "person", "conf": .9},  # too tiny
        {"x1": 100, "y1": 80, "x2": 150, "y2": 190, "cls": "person", "conf": .9},
    ]
    results = store.update_from_frame("cam", frame, boxes)
    assert len(results) == 1
    assert results[0].box_index == 2
    store.close()
