"""Paced, uncertainty-first review queue + bounded local mirror."""
import json

import pytest

from app.labels import ReviewStore, frame_uncertainty, sample_frame
from app import pool_sync


# ---- uncertainty scoring --------------------------------------------------------

def test_frame_uncertainty_peaks_at_the_gate():
    at_gate = {"boxes": [{"cls": "car", "conf": 0.35}]}      # == car gate
    assert frame_uncertainty(at_gate) == 1.0
    sure = {"boxes": [{"cls": "car", "conf": 0.95}]}         # far above
    assert frame_uncertainty(sure) == 0.0
    empty = {"boxes": []}
    assert frame_uncertainty(empty) == 0.0
    # max over boxes: one on-the-fence box dominates a confident one
    mixed = {"boxes": [{"cls": "car", "conf": 0.95},
                       {"cls": "person", "conf": 0.36}]}
    assert frame_uncertainty(mixed) > 0.9


def _write_frame(root, cam, ts, confs):
    d = root / "review_frames" / cam
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{ts}.jpg").write_bytes(b"jpg")
    (d / f"{ts}.json").write_text(json.dumps({
        "cam_id": cam, "frame_w": 100, "frame_h": 100,
        "boxes": [{"id": i, "cls": "car", "conf": c,
                   "box": [1, 1, 50, 50]} for i, c in enumerate(confs)],
    }))
    return f"review_frames/{cam}/{ts}.jpg"


def test_sample_frame_serves_most_uncertain_first(tmp_path):
    store = ReviewStore(tmp_path / "reviews.json")
    _write_frame(tmp_path, "camA", "1111", [0.95])          # confident
    fence = _write_frame(tmp_path, "camA", "2222", [0.36])  # on the fence
    _write_frame(tmp_path, "camA", "3333", [0.80])          # fairly sure
    s = sample_frame(store, tmp_path)
    assert s["frame_path"] == fence
    assert s["uncertainty"] > 0.9
    # once reviewed, the queue moves to the next most uncertain
    store.submit_frame(fence, "camA", {"0": "correct"}, [])
    s2 = sample_frame(store, tmp_path)
    assert s2["frame_path"].endswith("3333.jpg")


# ---- bounded local mirror -------------------------------------------------------

def test_pull_never_evicts_reviewed_frames(tmp_path, monkeypatch):
    """Reviewed frames are the training bank: the bounded mirror must keep
    them locally forever even after they age out of the newest-N window -
    otherwise export_labels silently loses the images behind the verdicts."""
    monkeypatch.setattr(pool_sync, "LOCAL_MIRROR_FRAMES", 1)
    snap = tmp_path / "src" / "web" / "snapshots"
    snap.mkdir(parents=True)
    files, bodies = {}, {}

    def add_frame(ts, mtime):
        for ext in ("jpg", "json"):
            rel = f"review_frames/camA/{ts}.{ext}"
            files[rel] = {"mtime": mtime, "url": f"https://s.example/{rel}"}
            bodies[rel] = f"{ts}".encode()

    def fake_get(url, timeout=30):
        path = url.split("?")[0]
        if path.endswith("/" + pool_sync.MANIFEST_NAME):
            return json.dumps({"files": files}).encode()
        for rel, body in bodies.items():
            if path.endswith(rel):
                return body
        raise OSError(f"404 {url}")

    add_frame(1000, 1.0)
    monkeypatch.setattr(pool_sync, "_http_get", fake_get)
    assert pool_sync.pull_once(snap, bucket="b")["downloaded"] == 2

    # the operator reviews the (currently newest) frame
    data_dir = tmp_path / "src" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "reviews.json").write_text(json.dumps({
        "frame_reviews": [{"frame_path": "review_frames/camA/1000.jpg",
                           "cam_id": "camA", "box_verdicts": {"0": "correct"},
                           "missed_detections": [], "reviewed_at": "2026-07-11T00:00:00Z"}]}))

    # two newer frames arrive; window=1 would normally evict 1000
    add_frame(2000, 2.0)
    add_frame(3000, 3.0)
    pool_sync.pull_once(snap, bucket="b")
    assert (snap / "review_frames/camA/3000.jpg").is_file()      # newest mirrored
    assert not (snap / "review_frames/camA/2000.jpg").exists()   # outside window
    assert (snap / "review_frames/camA/1000.jpg").is_file()      # REVIEWED: kept
    assert (snap / "review_frames/camA/1000.json").is_file()


def test_pull_mirrors_only_newest_slice(tmp_path, monkeypatch):
    monkeypatch.setattr(pool_sync, "LOCAL_MIRROR_FRAMES", 1)
    monkeypatch.setattr(pool_sync, "LOCAL_MIRROR_CROPS", 2)
    files = {}
    bodies = {}
    for i in range(3):   # 3 frames (jpg+json), mtimes ascending
        for ext in ("jpg", "json"):
            rel = f"review_frames/camA/{1000+i}.{ext}"
            files[rel] = {"mtime": float(i),
                          "url": f"https://s.example/{rel}"}
            bodies[rel] = f"frame{i}".encode()
    for i in range(4):   # 4 crops
        rel = f"live_samples/camA/{2000+i}_car_50.jpg"
        files[rel] = {"mtime": float(i), "url": f"https://s.example/{rel}"}
        bodies[rel] = f"crop{i}".encode()

    def fake_get(url, timeout=30):
        path = url.split("?")[0]
        if path.endswith("/" + pool_sync.MANIFEST_NAME):
            return json.dumps({"files": files}).encode()
        for rel, body in bodies.items():
            if path.endswith(rel):
                return body
        raise OSError(f"404 {url}")
    monkeypatch.setattr(pool_sync, "_http_get", fake_get)

    stats = pool_sync.pull_once(tmp_path, bucket="b")
    # newest frame (jpg+json) + 2 newest crops = 4 downloads, nothing more
    assert stats["downloaded"] == 4 and stats["errors"] == 0
    assert (tmp_path / "review_frames/camA/1002.jpg").is_file()
    assert (tmp_path / "review_frames/camA/1002.json").is_file()
    assert not (tmp_path / "review_frames/camA/1000.jpg").exists()
    assert (tmp_path / "live_samples/camA/2003_car_50.jpg").is_file()
    assert not (tmp_path / "live_samples/camA/2000_car_50.jpg").exists()

    # the window slides: a newer frame arrives -> old mirrored one is evicted
    for ext in ("jpg", "json"):
        rel = f"review_frames/camA/{1003}.{ext}"
        files[rel] = {"mtime": 9.0, "url": f"https://s.example/{rel}"}
        bodies[rel] = b"newest"
    stats2 = pool_sync.pull_once(tmp_path, bucket="b")
    assert stats2["downloaded"] == 2
    assert stats2["removed"] == 2          # 1002 jpg+json aged out locally
    assert (tmp_path / "review_frames/camA/1003.jpg").is_file()
    assert not (tmp_path / "review_frames/camA/1002.jpg").exists()
