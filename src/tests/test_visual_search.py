"""Search-by-example behaviors that must hold without YOLO or a network.

Run from src/:  python -m pytest tests/test_visual_search.py -q

Uses synthetic colored 'objects' + the dependency-free histogram embedder;
the end-to-end YOLO path is exercised by tools/search_by_image.py --seed-images.
"""
import json

import cv2
import numpy as np
import pytest

from app.reid import ReidStore
from app.reid_embed import HistogramEmbedder
from app.visual_search import (
    QueryObject,
    SnapshotIndex,
    extract_query_objects,
    search_registry,
)


def blob(color_bgr, size=(60, 120), noise=6, seed=0):
    """A synthetic 'person crop': solid color + a little texture."""
    rng = np.random.default_rng(seed)
    img = np.full((size[1], size[0], 3), color_bgr, dtype=np.uint8)
    n = rng.integers(-noise, noise + 1, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)


RED   = (40, 40, 200)
GREEN = (60, 180, 60)
BLUE  = (200, 80, 40)


@pytest.fixture()
def snapshots(tmp_path):
    """A fake collector snapshot tree: 3 distinct 'returning' crops."""
    d = tmp_path / "snapshots" / "returning" / "slot_a"
    d.mkdir(parents=True)
    for name, color in [("red", RED), ("green", GREEN), ("blue", BLUE)]:
        cv2.imwrite(str(d / f"eid_{name}.jpg"), blob(color))
        # full frames must be ignored by the index
        cv2.imwrite(str(d / f"eid_{name}_full.jpg"), blob(color, size=(320, 180)))
    return tmp_path / "snapshots"


def query_for(color, embedder, seed=99):
    emb = embedder.embed(blob(color, seed=seed), "person")
    return QueryObject(cls="person", embedding=emb)


def test_index_skips_full_frames(snapshots):
    idx = SnapshotIndex(snapshots, embedder=HistogramEmbedder())
    idx.refresh()
    assert len(idx) == 3


def test_search_ranks_same_color_first(snapshots):
    e = HistogramEmbedder()
    idx = SnapshotIndex(snapshots, embedder=e)
    idx.refresh()
    hits = idx.search(query_for(RED, e), top_n=3, min_sim=0.0)
    assert hits and hits[0].extra["path"].endswith("eid_red.jpg")
    assert hits[0].strong                      # same object -> clears threshold
    assert hits[0].similarity > hits[-1].similarity


def test_cache_reused_and_invalidated_on_embedder_change(snapshots):
    e = HistogramEmbedder()
    idx = SnapshotIndex(snapshots, embedder=e)
    assert idx.refresh() == 3                  # first pass embeds everything
    idx2 = SnapshotIndex(snapshots, embedder=e)
    assert idx2.refresh() == 0                 # warm cache -> nothing re-embedded
    cache = json.loads((snapshots / SnapshotIndex.CACHE_NAME).read_text())
    assert cache["embedder_id"] == e.embedder_id

    class OtherEmbedder(HistogramEmbedder):
        embedder_id = "other_v1"

    idx3 = SnapshotIndex(snapshots, embedder=OtherEmbedder())
    assert idx3.refresh() == 3                 # different signature -> full rebuild


def test_whole_image_fallback_when_no_model():
    qs = extract_query_objects(blob(RED), model=None)
    assert len(qs) == 1 and qs[0].cls == "image"


def test_registry_search_read_only(tmp_path):
    e = HistogramEmbedder()
    db = tmp_path / "reid.db"
    store = ReidStore(db_path=db, embedder=e)
    emb_red = e.embed(blob(RED), "person")
    store.query("cam1", "person", emb_red)
    store.query("cam1", "person", e.embed(blob(BLUE), "person"))
    store.close()

    q = query_for(RED, e)
    hits = search_registry(q, db_path=db, embedder=e, min_sim=0.0)
    assert hits and hits[0].extra["cam_id"] == "cam1"
    assert hits[0].strong

    # the search must not have inserted/updated anything
    import sqlite3
    conn = sqlite3.connect(str(db))
    assert conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM sightings").fetchone()[0] == 2
    conn.close()


def test_registry_search_refuses_other_embedders_db(tmp_path):
    e = HistogramEmbedder()
    db = tmp_path / "reid.db"
    store = ReidStore(db_path=db, embedder=e)
    store.query("cam1", "person", e.embed(blob(RED), "person"))
    store.close()

    class OtherEmbedder(HistogramEmbedder):
        embedder_id = "other_v1"

    hits = search_registry(query_for(RED, OtherEmbedder()), db_path=db,
                           embedder=OtherEmbedder(), min_sim=0.0)
    assert hits == []                          # incomparable vector spaces
