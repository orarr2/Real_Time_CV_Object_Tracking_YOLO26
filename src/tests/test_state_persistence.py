"""Restart persistence: presence stays, static anchors, reid last_box.

The failure mode these lock down: the collector runs under Restart=always,
and every bounce used to wipe the loiter clocks, the static anchors and the
returning-gate's previous-box memory together - which re-armed "just past
threshold" loiter alerts for permanent structures and slid the returning
static-object gate open (IoU vs None = 0.0 always passes).
"""
from __future__ import annotations

import numpy as np

from app.presence import PresenceTracker
from app.static_watch import StaticWatch

BOX = {"x1": 100.0, "y1": 100.0, "x2": 180.0, "y2": 260.0}
SHAPE = (720, 1280, 3)


def test_presence_state_round_trip():
    a = PresenceTracker(person_sec=300)
    a.observe("cam1", 7, "person", dict(BOX), SHAPE, now=1000.0)
    a.observe("cam1", 7, "person", dict(BOX, x1=104.0), SHAPE, now=1100.0)

    b = PresenceTracker(person_sec=300)
    kept = b.load_state(a.to_state(), now=1150.0)
    assert kept == 1
    # the restored stay continues counting from its ORIGINAL start: crossing
    # the 300s threshold must alert even though `b` was "born" at load time.
    # Observations stay inside the 180s continuity gap (a Restart=always
    # bounce is seconds, not minutes) and the box drifts enough to clear
    # the static-IoU refusal but not enough to break continuity.
    drift1 = dict(BOX, x1=124.0, x2=204.0)
    assert b.observe("cam1", 7, "person", drift1, SHAPE, now=1250.0) is None
    drift2 = dict(BOX, x1=144.0, x2=224.0)
    ev = b.observe("cam1", 7, "person", drift2, SHAPE, now=1400.0)
    assert ev is not None and ev["kind"] == "loiter"
    assert ev["duration_sec"] >= 300


def test_presence_load_drops_expired_continuity():
    a = PresenceTracker()
    a.observe("cam1", 7, "person", dict(BOX), SHAPE, now=1000.0)
    b = PresenceTracker()
    # 500s after last_seen > continuity_gap (180s): the stay is dead weight.
    assert b.load_state(a.to_state(), now=1500.0) == 0


def test_static_watch_state_round_trip_keeps_settled_age_and_crop():
    a = StaticWatch(min_stay_sec=10, min_hits=2, evidence_gates=None)
    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    t = 1000.0
    for i in range(4):
        a.observe("cam1", [dict(BOX, cls="bus", conf=0.8)], SHAPE,
                  luma=120.0, frame=frame, now=t + i * 20)
    assert a.counts("cam1")["settled"] == 1

    b = StaticWatch(min_stay_sec=10, min_hits=2, evidence_gates=None)
    kept = b.load_state(a.to_state(), now=t + 100)
    assert kept >= 1
    # furniture question: the anchor's AGE must survive the restart
    age = b.settled_spot_age("cam1", dict(BOX), "bus", now=t + 100)
    assert age is not None and age >= 60
    # the settle-time crop must ride along for post-restart depart evidence
    anchor = b._anchors["cam1"][0]
    assert anchor["settled"] and anchor["crop_jpeg"]
    # id counter must not restart (fresh anchors keep unique ids)
    assert b._next_id >= a._next_id


def test_reid_last_box_survives_reopen(tmp_path):
    from app.reid import ReidStore

    db = tmp_path / "reid_t.db"
    rng = np.random.default_rng(0)
    emb = rng.random(514).astype(np.float32)
    emb /= np.linalg.norm(emb)

    s1 = ReidStore(db)
    r1 = s1.query("cam1", "car", emb, box=dict(BOX))
    assert r1.is_new and r1.prev_box is None
    s1.close()

    # "restart": a fresh store over the same file; the match must expose the
    # previous sighting's box so the static-object gate has evidence.
    s2 = ReidStore(db)
    r2 = s2.query("cam1", "car", emb, box=dict(BOX, x1=104.0))
    assert not r2.is_new and r2.entity_id == r1.entity_id
    assert r2.prev_box is not None
    assert abs(r2.prev_box["x1"] - BOX["x1"]) < 0.6
    s2.close()


def test_reid_migration_adds_last_box_to_legacy_db(tmp_path):
    import sqlite3

    from app.reid import SCHEMA, ReidStore

    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    legacy = SCHEMA.replace(",\n    last_box     TEXT", "")  # no-op if absent
    conn.executescript(legacy)
    conn.execute("PRAGMA table_info(entities)")
    conn.commit()
    conn.close()

    store = ReidStore(db)   # must not raise; must add the column
    cols = {r[1] for r in
            store.conn.execute("PRAGMA table_info(entities)").fetchall()}
    assert "last_box" in cols
    store.close()
