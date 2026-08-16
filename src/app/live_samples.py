"""Continuous per-object crop pool for the review UI.

Motivation: the ``returning/`` / ``events/`` / ``anomalies/`` subtrees only
populate when specific things happen (a re-ID rematch, a loitering event,
a z-score anomaly). Cameras that don't produce those events keep the
review pool empty forever, so the user sees "every stored crop has been
reviewed" the moment the dashboard loads and has nothing to teach the
system with.

This module gives the collector a very cheap, bounded, per-camera crop
pool. On every ``LIVE_SAMPLE_EVERY_N``-th sample burst it saves ONE
randomly chosen detection crop from the burst's representative frame to
``web/snapshots/live_samples/<cam_id>/<mtime>_<cls>_<conf>.jpg`` and
enforces an LRU cap of ``LIVE_SAMPLE_MAX_FILES`` files across the tree.
No Firestore writes, no Storage uploads, no extra egress - the crops
just sit on the VM disk, ready for the review UI to sample and for the
search UI to browse.

Size budget: at the shipped defaults (200 files × ~50 KB) the tree stays
around ~10 MB - well inside the e2-micro's 30 GB Always-Free disk quota.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

from app.visual_search import SNAPSHOTS_ROOT

LIVE_SAMPLES_SUBDIR = "live_samples"

# Save one crop every N bursts, per camera. Balances "steady supply" (so
# the review UI has fresh material within minutes of a restart) against
# "not overwhelming the disk" (LRU eviction is cheap but each write costs
# CPU and adds to the pool of crops the user has to grind through).
LIVE_SAMPLE_EVERY_N = int(os.environ.get("LIVE_SAMPLE_EVERY_N") or 5)

# Hard cap on the total file count in the pool. Oldest files get evicted
# first once the pool grows past this. 1000-crop CLOUD bank (2026-07): the
# VM disk holds days of per-object history; the operator's machine only
# mirrors a bounded recent slice (see pool_sync local-mirror caps).
LIVE_SAMPLE_MAX_FILES = int(os.environ.get("LIVE_SAMPLE_MAX_FILES") or 1000)

MIN_CROP_SIDE = 24     # matches anomaly_crops - anything smaller is noise

# Bootstrap: seed the pool from shipped camera fixture frames the moment the
# dashboard server starts, so the review UI never opens on an empty pool
# even before the collector has produced its first sample. A marker file
# in the tree prevents re-seeding on subsequent boots.
BOOTSTRAP_MARKER = ".bootstrapped"
BOOTSTRAP_CAM_ID = "_demo"
BOOTSTRAP_TOP_K = 2   # crops per fixture (4 fixtures × 2 = ~8 total)


def _dir(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> Path:
    return Path(snapshots_root) / LIVE_SAMPLES_SUBDIR


def _round_counter_path(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> Path:
    """Per-camera burst counter file so ``should_sample`` doesn't need
    global state - the collector process may restart mid-day and we want
    the sampling cadence to survive. One tiny int per line, keyed by cam."""
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


def should_sample(cam_id: str,
                  every_n: int = LIVE_SAMPLE_EVERY_N,
                  snapshots_root: str | Path = SNAPSHOTS_ROOT) -> bool:
    """Called once per burst per camera. Returns True on every Nth burst.

    Uses a persisted counter file so restarts don't skew the phase and
    every camera keeps its own cadence independently.
    """
    if every_n <= 1:
        return True
    p = _round_counter_path(snapshots_root)
    counts = _read_counts(p)
    counts[cam_id] = counts.get(cam_id, 0) + 1
    _write_counts(p, counts)
    return counts[cam_id] % every_n == 0


def save_crop(cam_id: str, frame, boxes: list[dict],
              snapshots_root: str | Path = SNAPSHOTS_ROOT,
              cap_files: int | None = None,
              rng=None) -> str | None:
    """Save one random detection crop from ``boxes`` to the pool.

    Returns the relative-to-``snapshots_root`` path of the saved file, or
    None when there was nothing worth saving. Enforces the LRU cap after
    writing so the pool stays bounded.
    """
    if not boxes:
        return None
    import cv2   # heavy import gated to the function
    rng = rng or random
    # Prefer classes users care about (person + vehicles + train), not
    # every-random-bicycle. Weight person 3x so pedestrian pools stay dense
    # even on car-heavy cameras.
    weighted = []
    for b in boxes:
        w = 3 if b.get("cls") == "person" else 1
        weighted.extend([b] * w)
    b = rng.choice(weighted)
    H, W = frame.shape[:2]
    x1 = max(0, int(b["x1"])); y1 = max(0, int(b["y1"]))
    x2 = min(W, int(b["x2"])); y2 = min(H, int(b["y2"]))
    if x2 - x1 < MIN_CROP_SIDE or y2 - y1 < MIN_CROP_SIDE:
        return None
    crop = frame[y1:y2, x1:x2]
    out_dir = _dir(snapshots_root) / cam_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Microsecond-precision timestamp keeps filenames unique even when
    # bootstrap saves several crops in the same millisecond.
    ts = int(time.time() * 1_000_000)
    conf_pct = int(round((b.get("conf") or 0) * 100))
    # WS1: capture-time uncertainty travels in the filename (crops have no
    # sidecar json) so the BADGE sampler can weight the pool without
    # re-running the model.
    u = b.get("uncertainty")
    u_sfx = f"_u{int(round(float(u) * 100)):02d}" if u is not None else ""
    name = f"{ts}_{b.get('cls','?')}_{conf_pct:02d}{u_sfx}.jpg"
    out_path = out_dir / name
    if not cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        return None
    rel = str(out_path.relative_to(snapshots_root)).replace("\\", "/")
    enforce_cap(snapshots_root, cap_files)
    return rel


def enforce_cap(snapshots_root: str | Path = SNAPSHOTS_ROOT,
                cap_files: int | None = None) -> tuple[int, int]:
    """Delete oldest crops until file count <= cap. Returns
    ``(deleted_count, freed_bytes)``."""
    if cap_files is None:
        cap_files = LIVE_SAMPLE_MAX_FILES
    root = _dir(snapshots_root)
    if not root.is_dir():
        return 0, 0
    files = [p for p in root.rglob("*.jpg")]
    if len(files) <= cap_files:
        return 0, 0
    files.sort(key=lambda p: p.stat().st_mtime)
    to_delete = files[: len(files) - cap_files]
    freed = 0; n = 0
    for p in to_delete:
        try:
            freed += p.stat().st_size
            p.unlink()
            n += 1
        except OSError:
            continue
    return n, freed


def usage_stats(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> dict:
    root = _dir(snapshots_root)
    if not root.is_dir():
        return {"count": 0, "bytes": 0,
                "cap_files": LIVE_SAMPLE_MAX_FILES,
                "every_n": LIVE_SAMPLE_EVERY_N}
    files = list(root.rglob("*.jpg"))
    total = sum(p.stat().st_size for p in files if p.is_file())
    return {
        "count":     len(files),
        "bytes":     total,
        "cap_files": LIVE_SAMPLE_MAX_FILES,
        "every_n":   LIVE_SAMPLE_EVERY_N,
        "path":      str(root),
    }


def bootstrap_from_fixtures(model,
                            fixtures_dir: str | Path,
                            snapshots_root: str | Path = SNAPSHOTS_ROOT,
                            top_k: int = BOOTSTRAP_TOP_K,
                            imgsz: int | None = 640) -> int:
    """Seed the live-samples pool from shipped fixture frames.

    Runs once per install (guarded by a marker file inside the pool). The
    fixture frames in ``src/docs/images/`` are real captures from the four
    production cameras, so the crops the user reviews first look exactly
    like the ones the collector will produce a few minutes later.

    Returns the number of crops written (0 when nothing to do).
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
    for p in sorted(fixtures.glob("*.jpg")):
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        try:
            _, boxes = detect_with_boxes(model, frame, conf=0.30, imgsz=imgsz,
                                         per_class_conf=DEFAULT_PER_CLASS_CONF)
        except Exception:
            continue
        # Largest boxes first - most visible / most educational to the user
        # on the first-look review pass.
        boxes.sort(key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]),
                   reverse=True)
        for b in boxes[:top_k]:
            saved = save_crop(BOOTSTRAP_CAM_ID, frame, [b],
                              snapshots_root=snapshots_root)
            if saved:
                seeded += 1

    root.mkdir(parents=True, exist_ok=True)
    try:
        marker.touch()
    except OSError:
        pass
    return seeded


def clear_all(snapshots_root: str | Path = SNAPSHOTS_ROOT) -> dict:
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
    # Also drop the bootstrap marker so the next server start re-seeds the
    # pool from fixtures. Without this a user who clicked "clear all" would
    # be left with an empty review UI until the collector catches up.
    marker = root / BOOTSTRAP_MARKER
    try:
        if marker.is_file(): marker.unlink()
    except OSError:
        pass
    # Prune per-cam sub-dirs
    for d in sorted(root.rglob("*"), reverse=True):
        if d.is_dir():
            try: d.rmdir()
            except OSError: pass
    # Reset burst counter so cadence starts fresh from a known point
    p = _round_counter_path(snapshots_root)
    try:
        if p.is_file(): p.unlink()
    except OSError:
        pass
    return {"deleted": n, "bytes_freed": freed}
