"""Full-frame review pool - the frame-based counterpart to live_samples.

Where live_samples writes ONE cropped object per burst, this module writes
the WHOLE frame plus a JSON of every box the detector produced on it.
That lets the review UI show a scene with all detections overlaid, so a
user gives one verdict per BOX (correct / wrong) and can add "missing"
boxes for objects the model failed to see. That last part is what
finally makes recall computable end-to-end.

Layout on disk (each pair kept in sync):

    web/snapshots/review_frames/<cam_id>/<ts_us>.jpg     # the frame
    web/snapshots/review_frames/<cam_id>/<ts_us>.json    # its boxes

Metadata JSON schema:

    {
      "cam_id":  "konya_hukumet",
      "saved_at": "2026-07-07T14:00:00Z",
      "frame_w": 1280,
      "frame_h": 720,
      "boxes": [
        {"id": 0, "cls": "person", "conf": 0.65, "box": [x1, y1, x2, y2]},
        {"id": 1, "cls": "car",    "conf": 0.55, "box": [x1, y1, x2, y2]},
        ...
      ]
    }

Storage envelope (Free-Tier safe): 100 frames * ~200 KB per frame is
about 20 MB - a rounding error against the e2-micro's 30 GB PD quota.
LRU eviction runs after each write. Cadence is controlled by
REVIEW_FRAME_EVERY_N env var (default 5 bursts, same as live_samples).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from app.visual_search import SNAPSHOTS_ROOT

FRAMES_SUBDIR = "review_frames"

REVIEW_FRAME_EVERY_N = int(os.environ.get("REVIEW_FRAME_EVERY_N") or 5)
# 500-frame CLOUD bank (the VM's 30 GB disk is nearly empty and the operator
# wants history accumulating there). The operator's PERSONAL machine never
# mirrors all of it - pool_sync bounds the local mirror to a small working
# set - and the review UI serves a paced, uncertainty-sampled queue, so the
# bank's size is invisible to the person tagging.
REVIEW_FRAME_MAX_FILES = int(os.environ.get("REVIEW_FRAME_MAX_FILES") or 500)

# Bootstrap: seed the pool from shipped camera fixture frames the moment the
# dashboard server starts, so the review UI never opens on an empty pool
# even before the collector has produced its first sample. A marker file in
# the tree prevents re-seeding on subsequent boots.
BOOTSTRAP_MARKER = ".bootstrapped"


def _dir(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> Path:
    return Path(snapshots_root) / FRAMES_SUBDIR


def _counter_path(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> Path:
    return _dir(snapshots_root) / ".burst_counts.txt"


def _read_counts(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in path.read_text().splitlines():
            k, _, v = line.strip().partition(" ")
            if k and v.isdigit():
                out[k] = int(v)
    except OSError:
        pass
    return out


def _write_counts(path: Path, counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(f"{k} {v}" for k, v in sorted(counts.items())))
    tmp.replace(path)


def should_save(cam_id: str,
                every_n: int = REVIEW_FRAME_EVERY_N,
                snapshots_root: str | Path = SNAPSHOTS_ROOT) -> bool:
    if every_n <= 1:
        return True
    p = _counter_path(snapshots_root)
    counts = _read_counts(p)
    counts[cam_id] = counts.get(cam_id, 0) + 1
    _write_counts(p, counts)
    return counts[cam_id] % every_n == 0


def save_frame(cam_id: str, frame, boxes: list[dict],
               snapshots_root: str | Path = SNAPSHOTS_ROOT,
               cap_files: int | None = None) -> str | None:
    """Save the frame + its boxes metadata. Returns the rel path of the
    saved image, or None when the write failed."""
    if frame is None:
        return None
    import cv2
    H, W = frame.shape[:2]
    out_dir = _dir(snapshots_root) / cam_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_us = int(time.time() * 1_000_000)
    img_path = out_dir / f"{ts_us}.jpg"
    meta_path = out_dir / f"{ts_us}.json"
    if not cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 82]):
        return None
    meta = {
        "cam_id":   cam_id,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frame_w":  W,
        "frame_h":  H,
        "boxes": [
            {
                "id":   i,
                "cls":  b.get("cls", "?"),
                "conf": round(float(b.get("conf") or 0.0), 3),
                # WS1: capture-time uncertainty (against the burst's
                # EFFECTIVE gates); absent on pre-WS1 callers.
                **({"uncertainty": round(float(b["uncertainty"]), 4)}
                   if b.get("uncertainty") is not None else {}),
                "box":  [round(float(b["x1"]), 1), round(float(b["y1"]), 1),
                          round(float(b["x2"]), 1), round(float(b["y2"]), 1)],
            }
            for i, b in enumerate(boxes)
        ],
    }
    try:
        meta_path.write_text(json.dumps(meta))
    except OSError:
        try: img_path.unlink()
        except OSError: pass
        return None
    rel = str(img_path.relative_to(snapshots_root)).replace("\\", "/")
    enforce_cap(snapshots_root, cap_files)
    return rel


def bootstrap_from_fixtures(model,
                            fixtures_dir: str | Path,
                            snapshots_root: str | Path = SNAPSHOTS_ROOT,
                            imgsz: int | None = 640) -> int:
    """Seed the review-frames pool from shipped fixture frames.

    Runs once per install (guarded by a marker file in the pool tree). The
    fixture files ship as ``model_view_<cam_id>.jpg`` under ``src/docs/images``,
    so the cam_id embedded in the review UI matches the real production
    camera name - the user's first review-frame click looks exactly like a
    real live frame from that camera, boxes and all. Returns the number of
    frames written (0 when there was nothing to do).
    """
    if model is None or not fixtures_dir:
        return 0
    fixtures = Path(fixtures_dir)
    if not fixtures.is_dir():
        return 0
    root = _dir(snapshots_root)
    marker = root / BOOTSTRAP_MARKER
    if marker.exists():
        return 0

    import cv2
    from app.detect_core import detect_with_boxes, DEFAULT_PER_CLASS_CONF

    seeded = 0
    for p in sorted(fixtures.glob("model_view_*.jpg")):
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        try:
            _, boxes = detect_with_boxes(model, frame, conf=0.30, imgsz=imgsz,
                                         per_class_conf=DEFAULT_PER_CLASS_CONF)
        except Exception:
            continue
        # cam_id: strip the "model_view_" prefix so the review UI shows the
        # real production camera name (konya_hukumet etc.), not "_demo".
        cam_id = p.stem.replace("model_view_", "", 1) or "_demo"
        rel = save_frame(cam_id, frame, boxes, snapshots_root=snapshots_root)
        if rel:
            seeded += 1

    root.mkdir(parents=True, exist_ok=True)
    try:
        marker.write_text("")
    except OSError:
        pass
    return seeded


def load_metadata(frame_rel_path: str,
                  snapshots_root: str | Path = SNAPSHOTS_ROOT) -> dict | None:
    """Given a rel path to a saved frame .jpg, return the sibling .json
    metadata dict. Returns None when the metadata is missing or unreadable."""
    p = Path(snapshots_root) / frame_rel_path
    meta_path = p.with_suffix(".json")
    try:
        return json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return None


def _reviewed_rels(snapshots_root: str | Path) -> set[str]:
    """Frame rels the operator has REVIEWED (jpg + json sibling). These are
    the training set: evicting one deletes the image behind a verdict, and
    tools/export_labels.py then silently loses the example - the July batch
    shrank from 36 reviewed frames to a 3+1 usable export exactly this way.
    Repo #2 has no cloud mirror - always returns empty set (local-only mode)."""
    return set()


def enforce_cap(snapshots_root: str | Path = SNAPSHOTS_ROOT,
                cap_files: int | None = None) -> tuple[int, int]:
    """Delete oldest (frame, metadata) pairs until file count <= cap.
    Cap counts JPEGs, not JSON siblings. Reviewed frames are never
    deleted - they are labeled training data, not cache."""
    if cap_files is None:
        cap_files = REVIEW_FRAME_MAX_FILES
    root = _dir(snapshots_root)
    if not root.is_dir():
        return 0, 0
    files = [p for p in root.rglob("*.jpg")]
    if len(files) <= cap_files:
        return 0, 0
    protected = _reviewed_rels(snapshots_root)
    if protected:
        files = [p for p in files
                 if f"review_frames/{p.relative_to(root).as_posix()}"
                 not in protected]
    files.sort(key=lambda p: p.stat().st_mtime)
    to_delete = files[: max(0, len(files) - cap_files)]
    freed = 0; n = 0
    for p in to_delete:
        try:
            freed += p.stat().st_size
            p.unlink()
            n += 1
            js = p.with_suffix(".json")
            if js.is_file():
                freed += js.stat().st_size
                js.unlink()
        except OSError:
            continue
    return n, freed


def usage_stats(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> dict:
    root = _dir(snapshots_root)
    if not root.is_dir():
        return {"count": 0, "bytes": 0,
                "cap_files": REVIEW_FRAME_MAX_FILES,
                "every_n": REVIEW_FRAME_EVERY_N}
    files = list(root.rglob("*.jpg"))
    total = sum(p.stat().st_size for p in files if p.is_file())
    # include metadata size in the budget so users see the true footprint
    total += sum(p.stat().st_size for p in root.rglob("*.json") if p.is_file())
    return {
        "count":     len(files),
        "bytes":     total,
        "cap_files": REVIEW_FRAME_MAX_FILES,
        "every_n":   REVIEW_FRAME_EVERY_N,
        "path":      str(root),
    }


def clear_all(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> dict:
    """Wipe the UNREVIEWED pool. Reviewed frames survive the button: their
    images are the dataset behind data/reviews.json verdicts, and deleting
    them is how a 36-frame labeling effort once shrank to a 3-image export."""
    root = _dir(snapshots_root)
    if not root.is_dir():
        return {"deleted": 0, "bytes_freed": 0, "protected": 0}
    protected = _reviewed_rels(snapshots_root)
    freed = 0; n = 0; kept = 0
    for p in list(root.rglob("*")):
        if not p.is_file():
            continue
        rel = f"review_frames/{p.relative_to(root).as_posix()}"
        if rel in protected:
            kept += 1
            continue
        try:
            freed += p.stat().st_size
            p.unlink()
            if p.suffix == ".jpg":
                n += 1
        except OSError:
            continue
    for d in sorted(root.rglob("*"), reverse=True):
        if d.is_dir():
            try: d.rmdir()
            except OSError: pass
    return {"deleted": n, "bytes_freed": freed, "protected": kept}


def list_all_frames(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> list[str]:
    """Return rel paths of every saved frame image."""
    root = _dir(snapshots_root)
    if not root.is_dir():
        return []
    return [str(p.relative_to(snapshots_root)).replace("\\", "/")
            for p in sorted(root.rglob("*.jpg"))]
