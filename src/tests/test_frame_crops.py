"""frame_crops: metadata-driven extraction from review frames (no YOLO)."""
import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app import frame_crops


class FakeEmbedder:
    """Embeds a crop as its normalized mean-color vector - deterministic and
    close for visually identical crops, so the dedup path is exercised."""
    embedder_id = "fake"

    def embed(self, img, cls):
        v = img.reshape(-1, 3).mean(axis=0).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n else None


def _frame_with_boxes(tmp_path, cam="camA", ts="1111", boxes=None, color=90):
    d = tmp_path / "review_frames" / cam
    d.mkdir(parents=True, exist_ok=True)
    img = np.full((240, 320, 3), color, dtype=np.uint8)
    # paint each box region a distinct shade so crops differ
    for i, b in enumerate(boxes or []):
        x1, y1, x2, y2 = [int(v) for v in b["box"]]
        img[y1:y2, x1:x2] = (40 + 50 * i) % 255
    cv2.imwrite(str(d / f"{ts}.jpg"), img)
    (d / f"{ts}.json").write_text(json.dumps({
        "cam_id": cam, "frame_w": 320, "frame_h": 240,
        "boxes": boxes or [],
    }))
    return d / f"{ts}.jpg"


BOXES = [
    {"id": 0, "cls": "car",    "conf": 0.7, "box": [10, 10, 120, 100]},
    {"id": 1, "cls": "person", "conf": 0.5, "box": [150, 40, 220, 200]},
    {"id": 2, "cls": "car",    "conf": 0.6, "box": [4, 4, 20, 20]},  # < MIN_CROP_SIDE
]


def test_refresh_extracts_boxes(tmp_path):
    _frame_with_boxes(tmp_path, boxes=BOXES)
    stats = frame_crops.refresh(FakeEmbedder(), tmp_path)
    assert stats["frames_touched"] == 1
    assert stats["crops_added"] == 2          # tiny box skipped
    crops = list((tmp_path / "review_crops").rglob("*.jpg"))
    assert len(crops) == 2
    m = frame_crops._load_manifest(tmp_path)
    assert len(m["crops"]) == 2
    rec = next(iter(m["crops"].values()))
    assert rec["cam_id"] == "camA" and rec["source_frame"].startswith("review_frames/")


def test_refresh_idempotent(tmp_path):
    _frame_with_boxes(tmp_path, boxes=BOXES)
    frame_crops.refresh(FakeEmbedder(), tmp_path)
    stats = frame_crops.refresh(FakeEmbedder(), tmp_path)
    assert stats == {"frames_touched": 0, "crops_added": 0,
                     "crops_skipped_dup": 0, "crops_evicted": 0}


def test_refresh_dedups_identical_objects(tmp_path):
    # Same box content in two frames -> second is a near-duplicate.
    one_box = [{"id": 0, "cls": "car", "conf": 0.7, "box": [10, 10, 120, 100]}]
    _frame_with_boxes(tmp_path, ts="1111", boxes=one_box)
    frame_crops.refresh(FakeEmbedder(), tmp_path)
    _frame_with_boxes(tmp_path, ts="2222", boxes=one_box)
    stats = frame_crops.refresh(FakeEmbedder(), tmp_path)
    assert stats["crops_added"] == 0
    assert stats["crops_skipped_dup"] == 1


def test_crops_survive_source_frame_deletion(tmp_path):
    """Deliberate: crops extend searchable history past the frames LRU."""
    frame = _frame_with_boxes(tmp_path, boxes=BOXES[:1])
    frame_crops.refresh(FakeEmbedder(), tmp_path)
    frame.unlink()
    frame.with_suffix(".json").unlink()
    stats = frame_crops.refresh(FakeEmbedder(), tmp_path)
    assert stats["frames_touched"] == 0
    assert len(list((tmp_path / "review_crops").rglob("*.jpg"))) == 1


def test_size_cap_evicts_oldest(tmp_path):
    _frame_with_boxes(tmp_path, boxes=BOXES[:2])
    frame_crops.refresh(FakeEmbedder(), tmp_path, cap_bytes=10**9)
    deleted, freed = frame_crops.enforce_size_cap(tmp_path, cap_bytes=1)
    assert deleted == 2 and freed > 0
    assert frame_crops._load_manifest(tmp_path)["crops"] == {}


def test_clear_all_resets(tmp_path):
    _frame_with_boxes(tmp_path, boxes=BOXES[:2])
    frame_crops.refresh(FakeEmbedder(), tmp_path)
    out = frame_crops.clear_all(tmp_path)
    assert out["deleted"] == 2
    # source frames untouched -> refresh re-extracts
    stats = frame_crops.refresh(FakeEmbedder(), tmp_path)
    assert stats["crops_added"] == 2
