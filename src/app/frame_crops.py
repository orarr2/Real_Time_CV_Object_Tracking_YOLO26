"""Extract per-object crops from review frames so search can see them.

``review_frames/`` holds full scenes + a JSON of every box the detector
produced - exactly what the operator meant by "frames captured in the
past". The search index, however, matches OBJECT crops; full frames were
invisible to it, which is why searching for a car the camera had seen an
hour earlier returned nothing.

This module walks ``review_frames/<cam>/<ts>.json`` and cuts each listed
box out of its sibling JPEG into ``review_crops/<cam>/<ts>__<id>_<cls>.jpg``.
No YOLO pass is needed (unlike ``anomaly_crops`` - anomaly frames carry no
box metadata); extraction is just a crop, so it is cheap enough to run on
every search request.

Deliberate retention asymmetry: crops OUTLIVE their source frame. The
frames pool is a 100-file LRU (and the VM->local sync mirrors those
evictions), but the crops tree keeps its content until its own size cap
evicts - so the searchable history extends well past the frames window.

Same manifest contract as ``anomaly_crops`` (``{"processed_frames": ...,
"crops": {rel: {cam_id, cls, conf, source_frame, ...}}}``) so the search
index's class-manifest reader handles both trees with one code path.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from app.visual_search import SNAPSHOTS_ROOT

FRAMES_SUBDIR = "review_frames"
CROPS_SUBDIR = "review_crops"
MANIFEST_NAME = ".review_crops.json"

_DEFAULT_CAP_MB = 150
CAP_MB = int(os.environ.get("REVIEW_CROPS_MAX_MB") or _DEFAULT_CAP_MB)
CAP_BYTES = CAP_MB * 1024 * 1024

# Same duplicate bar as anomaly_crops: a parked car present in every 5th
# burst would otherwise flood the tree with near-identical crops.
DEDUP_THRESHOLD = 0.95
MIN_CROP_SIDE = 24


def _dir(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> Path:
    return Path(snapshots_root) / CROPS_SUBDIR


def _manifest_path(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> Path:
    return _dir(snapshots_root) / MANIFEST_NAME


def _load_manifest(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> dict:
    try:
        return json.loads(_manifest_path(snapshots_root).read_text())
    except (OSError, ValueError):
        return {"processed_frames": {}, "crops": {}}


def _save_manifest(m: dict, snapshots_root: str | Path = SNAPSHOTS_ROOT) -> None:
    p = _manifest_path(snapshots_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(m))
    tmp.replace(p)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b))


def refresh(embedder,
            snapshots_root: str | Path = SNAPSHOTS_ROOT,
            cap_bytes: int | None = None) -> dict:
    """Bring review_crops/ up to date with review_frames/. Idempotent and
    cheap: frames already in the manifest are skipped by mtime, so the
    steady-state cost of calling this per search request is one directory
    listing."""
    if cap_bytes is None:
        cap_bytes = CAP_BYTES
    import cv2  # gated: callers without OpenCV can still import the module

    root = Path(snapshots_root)
    frames_root = root / FRAMES_SUBDIR
    crops_root = _dir(root)
    manifest = _load_manifest(root)
    processed = manifest.setdefault("processed_frames", {})
    crops_map = manifest.setdefault("crops", {})

    # Fast path: nothing new. Avoid embedding warmup entirely.
    todo: list[tuple[Path, str, float]] = []
    if frames_root.is_dir():
        for jp in sorted(frames_root.rglob("*.json")):
            if jp.name.startswith("."):
                continue
            frame = jp.with_suffix(".jpg")
            if not frame.is_file():
                continue
            rel_frame = str(frame.relative_to(root)).replace("\\", "/")
            try:
                mtime = frame.stat().st_mtime
            except OSError:
                continue
            if processed.get(rel_frame) == mtime:
                continue
            todo.append((jp, rel_frame, mtime))
    if not todo:
        return {"frames_touched": 0, "crops_added": 0,
                "crops_skipped_dup": 0, "crops_evicted": 0}

    # Dedup memory: embed the crops we already store, per (cam, cls).
    seen: dict[tuple[str, str], list[np.ndarray]] = {}
    if embedder is not None:
        for rel, meta in crops_map.items():
            p = root / rel
            if not p.is_file():
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            emb_cls = "person" if meta.get("cls") == "person" else "vehicle"
            vec = embedder.embed(img, emb_cls)
            if vec is None:
                continue
            seen.setdefault((meta.get("cam_id", "?"), meta.get("cls", "?")),
                            []).append(vec)

    added = skipped_dup = 0
    for jp, rel_frame, mtime in todo:
        try:
            meta = json.loads(jp.read_text())
        except (OSError, ValueError):
            processed[rel_frame] = mtime
            continue
        image = cv2.imread(str(root / rel_frame))
        if image is None:
            processed[rel_frame] = mtime
            continue
        H, W = image.shape[:2]
        cam_id = str(meta.get("cam_id") or "unknown")
        for b in meta.get("boxes") or []:
            box = b.get("box") or []
            cls = b.get("cls") or "?"
            if len(box) != 4:
                continue
            x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
            x2 = min(W, int(box[2])); y2 = min(H, int(box[3]))
            if x2 - x1 < MIN_CROP_SIDE or y2 - y1 < MIN_CROP_SIDE:
                continue
            crop = image[y1:y2, x1:x2]
            vec = None
            if embedder is not None:
                emb_cls = "person" if cls == "person" else "vehicle"
                vec = embedder.embed(crop, emb_cls)
                if vec is not None:
                    key = (cam_id, cls)
                    if any(_cosine(prev, vec) >= DEDUP_THRESHOLD
                           for prev in seen.get(key, ())):
                        skipped_dup += 1
                        continue
                    seen.setdefault(key, []).append(vec)
            out_dir = crops_root / cam_id
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(rel_frame).stem
            out_path = out_dir / f"{stem}__{int(b.get('id', 0)):02d}_{cls}.jpg"
            if not cv2.imwrite(str(out_path), crop,
                               [cv2.IMWRITE_JPEG_QUALITY, 85]):
                continue
            rel_crop = str(out_path.relative_to(root)).replace("\\", "/")
            crops_map[rel_crop] = {
                "cam_id":       cam_id,
                "cls":          cls,
                "conf":         round(float(b.get("conf") or 0.0), 3),
                "source_frame": rel_frame,
                "saved_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                              time.gmtime()),
                "foot_norm":    [round((x1 + x2) / 2.0 / W, 4),
                                 round(y2 / H, 4)],
            }
            added += 1
        processed[rel_frame] = mtime

    _save_manifest(manifest, root)
    evicted, _freed = enforce_size_cap(root, cap_bytes)
    return {"frames_touched": len(todo), "crops_added": added,
            "crops_skipped_dup": skipped_dup, "crops_evicted": evicted}


def enforce_size_cap(snapshots_root: str | Path = SNAPSHOTS_ROOT,
                     cap_bytes: int | None = None) -> tuple[int, int]:
    """Oldest-first eviction until the tree fits the cap; manifest entries
    for deleted crops are pruned in the same pass."""
    if cap_bytes is None:
        cap_bytes = CAP_BYTES
    root = _dir(snapshots_root)
    if not root.is_dir():
        return 0, 0
    files = sorted(root.rglob("*.jpg"), key=lambda p: p.stat().st_mtime)
    used = sum(p.stat().st_size for p in files)
    if used <= cap_bytes:
        return 0, 0
    freed = 0
    deleted: list[str] = []
    for p in files:
        if used - freed <= cap_bytes:
            break
        try:
            sz = p.stat().st_size
            p.unlink()
            freed += sz
            deleted.append(str(p.relative_to(Path(snapshots_root))).replace("\\", "/"))
        except OSError:
            continue
    if deleted:
        m = _load_manifest(snapshots_root)
        for rel in deleted:
            m["crops"].pop(rel, None)
        _save_manifest(m, snapshots_root)
    return len(deleted), freed


def usage_stats(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> dict:
    root = _dir(snapshots_root)
    if not root.is_dir():
        return {"count": 0, "bytes": 0, "cap_mb": CAP_MB}
    files = list(root.rglob("*.jpg"))
    return {"count": len(files),
            "bytes": sum(p.stat().st_size for p in files if p.is_file()),
            "cap_mb": CAP_MB,
            "path":  str(root)}


def clear_all(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> dict:
    """Wipe extracted crops + manifest. Source frames stay untouched;
    the next refresh() re-extracts whatever frames still exist."""
    root = _dir(snapshots_root)
    if not root.is_dir():
        return {"deleted": 0, "bytes_freed": 0}
    freed = 0; n = 0
    for p in list(root.rglob("*.jpg")):
        try:
            freed += p.stat().st_size
            p.unlink()
            n += 1
        except OSError:
            continue
    _save_manifest({"processed_frames": {}, "crops": {}}, snapshots_root)
    for d in sorted(root.rglob("*"), reverse=True):
        if d.is_dir():
            try: d.rmdir()
            except OSError: pass
    return {"deleted": n, "bytes_freed": freed}
