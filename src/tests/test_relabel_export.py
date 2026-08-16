"""Relabel verdicts end-to-end: store validation -> metrics -> YOLO export."""
import json
from pathlib import Path

import numpy as np
import pytest

from app.labels import ReviewStore, valid_box_verdict, list_frames, load_frame
from tools.export_labels import collect_examples, export


def test_valid_box_verdict():
    assert valid_box_verdict("correct")
    assert valid_box_verdict("wrong")
    assert valid_box_verdict("relabel:bus")
    assert not valid_box_verdict("relabel:zeppelin")
    assert not valid_box_verdict("maybe")
    assert not valid_box_verdict("relabel:")


def _seed_frame(snap_root: Path, cam="camA", ts="1000"):
    d = snap_root / "review_frames" / cam
    d.mkdir(parents=True, exist_ok=True)
    cv2 = pytest.importorskip("cv2")
    img = np.full((240, 320, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(d / f"{ts}.jpg"), img)
    (d / f"{ts}.json").write_text(json.dumps({
        "cam_id": cam, "frame_w": 320, "frame_h": 240,
        "boxes": [
            {"id": 0, "cls": "car", "conf": 0.7, "box": [10, 10, 120, 100]},
            {"id": 1, "cls": "bus", "conf": 0.6, "box": [150, 40, 300, 200]},
            {"id": 2, "cls": "person", "conf": 0.5, "box": [40, 120, 90, 230]},
        ],
    }))
    return f"review_frames/{cam}/{ts}.jpg"


def test_store_keeps_relabel_and_reload(tmp_path):
    snap = tmp_path / "snaps"
    rel = _seed_frame(snap)
    store = ReviewStore(tmp_path / "reviews.json")
    store.submit_frame(rel, "camA",
                       {"0": "correct", "1": "relabel:truck", "2": "wrong",
                        "3": "banana"},
                       [{"cls": "person", "box": [1, 2, 30, 60]}])
    # invalid verdict filtered; relabel survives the round-trip
    again = ReviewStore(tmp_path / "reviews.json")
    fr = again._frames_by_path[rel]
    assert fr.box_verdicts == {"0": "correct", "1": "relabel:truck",
                               "2": "wrong"}
    s = again.summary()
    assert s["frame_tp"] == 1 and s["frame_fp"] == 2 and s["frame_fn"] == 1


def test_list_and_load_frame(tmp_path):
    snap = tmp_path / "snaps"
    rel = _seed_frame(snap)
    store = ReviewStore(tmp_path / "reviews.json")
    frames = list_frames(store, snap)
    assert len(frames) == 1 and frames[0]["reviewed"] is False
    assert frames[0]["n_boxes"] == 3

    store.submit_frame(rel, "camA", {"0": "correct"}, [], note="hm")
    frames = list_frames(store, snap)
    assert frames[0]["reviewed"] is True

    loaded = load_frame(store, rel, snap)
    assert loaded["existing"]["box_verdicts"] == {"0": "correct"}
    assert loaded["existing"]["note"] == "hm"
    assert load_frame(store, "review_frames/camA/nope.jpg", snap) is None


def test_export_applies_corrections(tmp_path):
    snap = tmp_path / "snaps"
    rel = _seed_frame(snap)
    store = ReviewStore(tmp_path / "reviews.json")
    store.submit_frame(rel, "camA",
                       {"0": "correct",          # keep car
                        "1": "relabel:truck",    # bus -> truck
                        "2": "wrong"},           # drop person
                       [{"cls": "bicycle", "box": [5, 5, 60, 60]}])
    ex = collect_examples(tmp_path / "reviews.json", snap)
    assert len(ex) == 1
    rows = ex[0]["rows"]
    # car kept + relabeled truck + added bicycle = 3; wrong person dropped
    assert len(rows) == 3
    ids = sorted(int(r.split()[0]) for r in rows)
    # NATIVE COCO ids (keeps the trained head base-shape-compatible):
    # person=0 bicycle=1 car=2 motorcycle=3 bus=5 train=6 truck=7
    assert ids == [1, 2, 7]
    assert ex[0]["stats"] == {"kept": 1, "dropped": 1, "relabeled": 1,
                              "weak": 0, "added_fn": 1}

    out = tmp_path / "ds"
    totals = export(out, ex, val_frac=0.1)
    assert totals["frames"] == 1 and totals["labels"] == 3
    yaml = (out / "dataset.yaml").read_text()
    # the yaml must name ALL 80 base-model classes, ours at COCO positions
    assert "7: truck" in yaml and "5: bus" in yaml and "0: person" in yaml
    assert "79: toothbrush" in yaml
    txts = list((out / "labels").rglob("*.txt"))
    imgs = list((out / "images").rglob("*.jpg"))
    assert len(txts) == 1 and len(imgs) == 1
    # normalized coords in range
    for line in txts[0].read_text().splitlines():
        vals = [float(v) for v in line.split()[1:]]
        assert all(0.0 <= v <= 1.0 for v in vals)


def test_export_weak_labels_toggle(tmp_path):
    snap = tmp_path / "snaps"
    rel = _seed_frame(snap)
    store = ReviewStore(tmp_path / "reviews.json")
    store.submit_frame(rel, "camA", {"0": "correct"}, [])   # boxes 1,2 untouched
    keep_weak = collect_examples(tmp_path / "reviews.json", snap)
    assert len(keep_weak[0]["rows"]) == 3          # weak labels kept
    strict = collect_examples(tmp_path / "reviews.json", snap,
                              reviewed_boxes_only=True)
    assert len(strict[0]["rows"]) == 1             # only the verified box
