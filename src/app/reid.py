"""Appearance-based re-identification ("have I seen this person/car before?").

Approach - deliberately dependency-free so the notebook + collector + dashboard can all
use it without a torch detour:

  1. Per detection, crop the bounding box and normalize the crop size.
  2. Build a *masked* HSV color histogram (8x8x8 bins) - pixels with V<30 are dropped
     so the night-time sodium-light dominant background doesn't swamp the signature.
  3. Append (aspect_ratio, normalized_area) so persons of similar color but different
     build don't collide.
  4. L2-normalize -> 514-dim unit vector.
  5. Store in a SQLite table; on new detection, cosine-similarity-match against the
     same-class entities; if best match >= THRESH update its (last_seen, sightings)
     and return its id, otherwise insert a fresh entity.

This is a *demo-grade* signature. It will work well in daylight where each person has
distinct clothing colour; it will produce false matches at night (the saved frame from
the Konya camera shows the whole scene is yellow-tinted). For production-grade re-ID
swap embed_crop() for an OSNet/torchreid forward pass - the rest of the registry stays.
"""
from __future__ import annotations

import io
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# How aggressive about declaring a match with the DEFAULT histogram embedder.
# 0.92 is conservative (more new IDs); drop to 0.85 to merge more aggressively.
# An OsnetEmbedder brings its own default (see reid_embed.py) - pass
# threshold=None to ReidStore to use the embedder's default.
DEFAULT_THRESHOLD = 0.92
# On every match the stored embedding drifts toward the fresh observation:
# stored = normalize((1-EMA_ALPHA)*stored + EMA_ALPHA*fresh). Without this the
# FIRST crop is the signature forever, and gradual lighting change (afternoon
# -> dusk) splits the same entity into a chain of new IDs.
EMA_ALPHA = 0.25
# Match against the K nearest stored entities of the same class (top-1 only kept).
TOPK = 25
# Upper bound on candidates scored per query (most-recently-seen first).
MAX_SCAN = 400
PERSON_CROP = (64, 128)   # w x h (kept for import compat; source of truth
VEHICLE_CROP = (96, 96)   # lives in reid_embed.py)

SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    entity_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    cam_id       TEXT NOT NULL,
    cls          TEXT NOT NULL,              -- 'person' | 'car' | 'truck' | ...
    first_seen   TEXT NOT NULL,
    last_seen    TEXT NOT NULL,
    sightings    INTEGER NOT NULL DEFAULT 1,
    embedding    BLOB NOT NULL                -- float32 L2-normalized vector
);
CREATE INDEX IF NOT EXISTS idx_entities_cam_cls ON entities(cam_id, cls);

CREATE TABLE IF NOT EXISTS sightings (
    sighting_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id    INTEGER NOT NULL,
    ts           TEXT NOT NULL,
    similarity   REAL,                        -- match score; NULL if new entity
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id)
);
CREATE INDEX IF NOT EXISTS idx_sightings_entity ON sightings(entity_id);
CREATE INDEX IF NOT EXISTS idx_sightings_ts ON sightings(ts);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,                  -- e.g. 'embedder_id'
    value TEXT NOT NULL
);
"""


@dataclass
class ReidResult:
    entity_id: int
    cls: str
    is_new: bool
    sightings: int
    similarity: float            # 1.0 for brand-new entities (self-similarity)
    # Seconds since the *previous* sighting at this camera, or None for new
    # entities. Lets the collector save a "returning visitor" image only when
    # the gap clears its configured threshold (RETURNING_GAP_SEC, default
    # 5 min) instead of every consecutive sample.
    gap_seconds: float | None = None
    # Index into the `boxes` list passed to update_from_frame() that produced
    # this result. update_from_frame SKIPS degenerate/tiny crops, so results
    # are NOT positionally aligned with the input - always use this index to
    # get back to the detection box.
    box_index: int | None = None
    # Box of the PREVIOUS sighting (from the DB, so it survives collector
    # restarts). The returning-visitor "static object" gate compares it to
    # the current box; before this field the gate read a process-memory dict
    # that every restart wiped, so the first post-restart match of any
    # long-lived static entity passed the gate unchecked (IoU vs None = 0).
    prev_box: dict | None = None


from app.reid_embed import HistogramEmbedder, _l2norm  # noqa: E402

_DEFAULT_EMBEDDER = HistogramEmbedder()


def embed_crop(crop_bgr: np.ndarray, cls: str) -> np.ndarray | None:
    """Legacy helper: the default histogram embedding (see reid_embed.py for
    the pluggable embedders, including the OSNet upgrade)."""
    return _DEFAULT_EMBEDDER.embed(crop_bgr, cls)


def _blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _gap_seconds(prev_iso: str | None, now_iso: str) -> float | None:
    """Seconds between two stored timestamp strings (UTC), None if unparseable."""
    if not prev_iso:
        return None
    try:
        from datetime import datetime
        # stored strings use either '%Y-%m-%dT%H:%M:%SZ' (this module) or the
        # full ISO including microseconds (the collector record). Both parse.
        def _parse(s: str):
            s2 = s.replace("Z", "+00:00")
            return datetime.fromisoformat(s2)
        return max(0.0, (_parse(now_iso) - _parse(prev_iso)).total_seconds())
    except Exception:
        return None


class ReidStore:
    """SQLite-backed appearance memory. Safe to share across processes via the file.

    `embedder` produces the appearance vectors (default: HSV histogram;
    pass an OsnetEmbedder for real cross-lighting re-ID). The registry
    remembers which embedder produced its vectors and RESETS itself when the
    embedder changes - vectors from different embedders have different
    dimensions and metric scales and must never be compared.
    `threshold=None` uses the embedder's own default.
    """

    def __init__(self, db_path: str | Path = "data/reid.db",
                 threshold: float | None = DEFAULT_THRESHOLD,
                 embedder=None):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder if embedder is not None else _DEFAULT_EMBEDDER
        self.threshold = (threshold if threshold is not None
                          else self.embedder.default_threshold)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._migrate_last_box()
        self._check_embedder_version()

    def _migrate_last_box(self) -> None:
        """Add entities.last_box (JSON [x1,y1,x2,y2]) on registries created
        before it existed. ALTER ADD COLUMN is metadata-only in SQLite -
        instant even on a multi-hundred-MB registry."""
        cols = {r[1] for r in
                self.conn.execute("PRAGMA table_info(entities)").fetchall()}
        if "last_box" not in cols:
            self.conn.execute("ALTER TABLE entities ADD COLUMN last_box TEXT")
            self.conn.commit()

    def _check_embedder_version(self) -> None:
        eid = getattr(self.embedder, "embedder_id", "unknown")
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='embedder_id'").fetchone()
        stored = row[0] if row else None
        if stored is not None and stored != eid:
            n = self.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            print(f"reid: embedder changed ({stored} -> {eid}); resetting "
                  f"registry ({n} entities dropped - old vectors are not "
                  f"comparable to the new embedder's).")
            self.conn.execute("DELETE FROM sightings")
            self.conn.execute("DELETE FROM entities")
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('embedder_id', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (eid,))
        self.conn.commit()

    # ---- write path ------------------------------------------------------

    @staticmethod
    def _box_to_json(box: dict | None) -> str | None:
        if not box:
            return None
        try:
            return json.dumps([round(float(box[k]), 1)
                               for k in ("x1", "y1", "x2", "y2")])
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _box_from_json(raw: str | None) -> dict | None:
        if not raw:
            return None
        try:
            x1, y1, x2, y2 = json.loads(raw)
            return {"x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2)}
        except (ValueError, TypeError):
            return None

    def query(self, cam_id: str, cls: str, embedding: np.ndarray,
              exclude: set[int] | None = None,
              commit: bool = True, box: dict | None = None) -> ReidResult:
        """Match `embedding` to the stored entities (same cam + class). Either
        update an existing entity or insert a new one. Always returns a result.

        `exclude` skips entity ids already matched in the CURRENT frame -
        without it, two similar objects visible at once (two white cars) both
        match the same entity, double-counting sightings and dragging the EMA
        embedding toward a blend of different physical objects.
        `commit=False` lets update_from_frame batch one transaction per frame
        instead of one fsync per detection box.
        """
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # Pull the same-cam same-class candidates and score them. Bounded to
        # the most recently seen MAX_SCAN entities so a busy day can't turn
        # every query into a full-table scan (prune() keeps the table small,
        # this is the backstop). We pull `last_seen` too so we can report the
        # gap to the caller, which uses it to decide whether to save a
        # "returning visitor" image.
        cur = self.conn.execute(
            "SELECT entity_id, sightings, last_seen, embedding, last_box "
            "FROM entities "
            "WHERE cam_id=? AND cls=? ORDER BY last_seen DESC LIMIT ?",
            (cam_id, cls, MAX_SCAN))
        best_id, best_sim, best_sight, best_last, best_vec = None, -1.0, 0, None, None
        best_box_raw = None
        for eid, n_sight, last_seen, blob, box_raw in cur.fetchall():
            if exclude and eid in exclude:
                continue
            vec = _blob_to_vec(blob)
            sim = float(np.dot(vec, embedding))  # both L2-normalized -> cosine
            if sim > best_sim:
                best_sim, best_id, best_sight, best_last = sim, eid, n_sight, last_seen
                best_vec = vec
                best_box_raw = box_raw

        box_json = self._box_to_json(box)
        if best_id is not None and best_sim >= self.threshold:
            # match: compute the gap from the prior last_seen *before* we
            # overwrite it, then bump sightings + last_seen and drift the
            # stored embedding toward the fresh observation (EMA) so gradual
            # lighting change doesn't split the entity. Near-identical
            # observations (sim >= 0.995) skip the blob rewrite - the blend
            # would be a no-op costing a full-page write per detection.
            gap = _gap_seconds(best_last, ts)
            new_sight = best_sight + 1
            blended = (_l2norm((1.0 - EMA_ALPHA) * best_vec + EMA_ALPHA * embedding)
                       if best_sim < 0.995 else None)
            if blended is not None:
                self.conn.execute(
                    "UPDATE entities SET last_seen=?, sightings=?, embedding=?, "
                    "last_box=COALESCE(?, last_box) WHERE entity_id=?",
                    (ts, new_sight, blended.tobytes(), box_json, best_id))
            else:
                self.conn.execute(
                    "UPDATE entities SET last_seen=?, sightings=?, "
                    "last_box=COALESCE(?, last_box) WHERE entity_id=?",
                    (ts, new_sight, box_json, best_id))
            self.conn.execute(
                "INSERT INTO sightings (entity_id, ts, similarity) VALUES (?, ?, ?)",
                (best_id, ts, best_sim))
            if commit:
                self.conn.commit()
            return ReidResult(best_id, cls, is_new=False, sightings=new_sight,
                              similarity=best_sim, gap_seconds=gap,
                              prev_box=self._box_from_json(best_box_raw))

        # new entity
        cur = self.conn.execute(
            "INSERT INTO entities (cam_id, cls, first_seen, last_seen, sightings, "
            "embedding, last_box) VALUES (?, ?, ?, ?, 1, ?, ?)",
            (cam_id, cls, ts, ts, embedding.astype(np.float32).tobytes(),
             box_json))
        eid = cur.lastrowid
        self.conn.execute(
            "INSERT INTO sightings (entity_id, ts, similarity) VALUES (?, ?, NULL)",
            (eid, ts))
        if commit:
            self.conn.commit()
        return ReidResult(eid, cls, is_new=True, sightings=1, similarity=1.0,
                          gap_seconds=None)

    def update_from_frame(self, cam_id: str, frame_bgr: np.ndarray,
                          boxes: list[dict]) -> list[ReidResult]:
        """Convenience: embed every box's crop and query the registry.

        boxes is a list of {x1,y1,x2,y2,cls,conf} dicts.
        """
        results = []
        matched: set[int] = set()   # entities already claimed by this frame
        H, W = frame_bgr.shape[:2]
        for i, b in enumerate(boxes):
            x1 = max(0, int(b["x1"])); y1 = max(0, int(b["y1"]))
            x2 = min(W, int(b["x2"])); y2 = min(H, int(b["y2"]))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            emb = self.embedder.embed(crop, b["cls"])
            if emb is None:
                continue
            r = self.query(cam_id, b["cls"], emb, exclude=matched, commit=False,
                           box=b)
            matched.add(r.entity_id)
            r.box_index = i
            results.append(r)
        self.conn.commit()   # one transaction per frame, not per box
        return results

    def prune(self, max_age_hours: float = 48.0) -> int:
        """Delete entities (and their sightings) not seen for `max_age_hours`.

        A public street cam sees thousands of one-off passers-by a day; without
        pruning, the registry grows unbounded and every query() slows down while
        the "unique entities" stat inflates forever. 48h keeps the
        "came back the next day" case the dashboard reports, drops the rest.
        `last_seen` is stored as a fixed-width UTC ISO string, so plain string
        comparison against the cutoff is correct.
        """
        cutoff = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                               time.gmtime(time.time() - max_age_hours * 3600))
        self.conn.execute(
            "DELETE FROM sightings WHERE entity_id IN "
            "(SELECT entity_id FROM entities WHERE last_seen < ?)", (cutoff,))
        cur = self.conn.execute(
            "DELETE FROM entities WHERE last_seen < ?", (cutoff,))
        removed = cur.rowcount or 0
        self.conn.commit()
        return removed

    # ---- read path -------------------------------------------------------

    def stats(self, cam_id: str | None = None) -> dict:
        """Roll-up for the dashboard: unique entities, regulars, sightings distribution."""
        where = "WHERE cam_id=?" if cam_id else ""
        args = (cam_id,) if cam_id else ()
        cur = self.conn.execute(
            f"SELECT cls, COUNT(*) AS uniq, SUM(sightings) AS total, "
            f"       SUM(CASE WHEN sightings >= 3 THEN 1 ELSE 0 END) AS regulars "
            f"FROM entities {where} GROUP BY cls",
            args,
        )
        per_class = {row[0]: {"unique": row[1], "total_sightings": row[2],
                              "regulars": row[3]} for row in cur.fetchall()}
        cur = self.conn.execute(
            f"SELECT COUNT(*), SUM(sightings) FROM entities {where}", args)
        total_uniq, total_sight = cur.fetchone()
        return {
            "total_unique":    total_uniq or 0,
            "total_sightings": total_sight or 0,
            "per_class":       per_class,
        }

    def top_regulars(self, cam_id: str | None = None, n: int = 10) -> list[dict]:
        """Return the n entities with the most sightings."""
        where = "WHERE cam_id=?" if cam_id else ""
        args = (cam_id,) if cam_id else ()
        cur = self.conn.execute(
            f"SELECT entity_id, cls, sightings, first_seen, last_seen "
            f"FROM entities {where} ORDER BY sightings DESC LIMIT ?",
            args + (n,))
        return [{"entity_id": r[0], "cls": r[1], "sightings": r[2],
                 "first_seen": r[3], "last_seen": r[4]} for r in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()
