"""WS4 (per-camera calibration) + WS5 (al-curve payload)."""
import json

import numpy as np

from app import adapters
from app.cameras import CAMERAS, _merge_per_camera_conf
from app.labels import ReviewStore
from tools.calibrate_conf import calibrate, conf_star


def _verdicts(spec):
    """[(conf, n_tp, n_fp), ...] -> flat [(conf, is_tp), ...]."""
    out = []
    for conf, n_tp, n_fp in spec:
        out += [(conf, True)] * n_tp + [(conf, False)] * n_fp
    return out


def test_conf_star_picks_lowest_qualifying_gate():
    # Below 0.40 the pool is noisy; from 0.40 up precision is 19/20 = 0.95.
    rows = _verdicts([(0.30, 2, 8), (0.40, 9, 1), (0.50, 10, 0)])
    assert conf_star(rows, target_precision=0.90) == 0.40
    # Impossible target -> None (never guess)
    assert conf_star(_verdicts([(0.30, 1, 9)]), target_precision=0.90) is None
    # Gates above max_conf don't count even if they'd qualify
    assert conf_star(_verdicts([(0.70, 30, 0)]), max_conf=0.60) is None


def test_calibrate_end_to_end(tmp_path):
    """Frame verdicts + crop verdicts feed one payload; small pairs are
    skipped; the merge overrides the boosted gate for calibrated pairs."""
    import cv2
    from app.review_frames import save_frame

    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    store = ReviewStore(tmp_path / "reviews.json")

    # 30 frame-review verdicts on cam_a/person: 10 FP at 0.36, 20 TP at 0.52
    boxes = ([{"cls": "person", "conf": 0.36,
               "x1": 1, "y1": 1, "x2": 10, "y2": 20}] * 10
             + [{"cls": "person", "conf": 0.52,
                 "x1": 1, "y1": 1, "x2": 10, "y2": 20}] * 20)
    rel = save_frame("cam_a", frame, boxes, snapshots_root=tmp_path)
    meta = json.loads((tmp_path / rel).with_suffix(".json").read_text())
    verdicts = {str(b["id"]): ("correct" if b["conf"] > 0.4 else "wrong")
                for b in meta["boxes"]}
    store.submit_frame(rel, "cam_a", verdicts, [])

    # 3 crop verdicts on cam_b/person - under min_reviews, must be skipped
    d = tmp_path / "live_samples" / "cam_b"
    d.mkdir(parents=True)
    for i, name in enumerate(["10_person_40.jpg", "11_person_50_u30.jpg",
                              "12_person_60.jpg"]):
        cv2.imwrite(str(d / name), np.zeros((30, 20, 3), dtype=np.uint8))
        store.submit(f"live_samples/cam_b/{name}", "correct",
                     original_cls="person")

    payload = calibrate(store, tmp_path, target_precision=0.90,
                        min_reviews=30)
    assert "cam_b" not in payload["cameras"]
    entry = payload["cameras"]["cam_a"]["person"]
    assert entry["conf"] == 0.52 and entry["n_reviews"] == 30

    # merge precedence: calibration overrides whatever gate is in place
    cam = CAMERAS["taksim_yeni"]
    before = dict(cam.get("per_class_conf") or {})
    try:
        _merge_per_camera_conf({"cameras": {"taksim_yeni":
                                            {"person": {"conf": 0.52}}}})
        assert cam["per_class_conf"]["person"] == 0.52
    finally:
        if before:
            cam["per_class_conf"] = before
        else:
            cam.pop("per_class_conf", None)


def test_al_curve_payload(tmp_path):
    adapters.append_history({"event": "gate", "candidate": "head_a.pt",
                             "promoted": False, "labels_total": 120,
                             "baseline": {"map50": 0.40},
                             "metrics": {"map50": 0.39}}, tmp_path)
    adapters.append_history({"event": "fetched", "file": "x"}, tmp_path)
    adapters.append_history({"event": "gate", "candidate": "head_b.pt",
                             "promoted": True, "labels_total": 240,
                             "baseline": {"map50": 0.41},
                             "metrics": {"map50": 0.45}}, tmp_path)
    (tmp_path / "head_b.pt").write_bytes(b"b")
    adapters.promote("head_b.pt", {"map50": 0.45}, tmp_path, base="yolov8n.pt")
    p = adapters.al_curve_payload(tmp_path)
    assert p["baseline_map50"] == 0.41 and p["current"] == "head_b.pt"
    assert [pt["promoted"] for pt in p["points"]] == [False, True]
    assert p["points"][1]["labels_total"] == 240
