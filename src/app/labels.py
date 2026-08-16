"""Human-in-the-loop review of stored detections.

YOLOv8 is inference-only at runtime and does not learn from the stream, so
the only way to actually improve what the system shows the user is to let
them tell us when a label is wrong and to remember what they said. This
module is the persistence layer for that feedback loop:

  1. sample_crop()   - pick a saved crop the user has not reviewed yet
  2. submit_review() - persist the user's verdict (correct / wrong-label /
                       not-an-object) with an optional corrected class
  3. summary()       - count how many crops the user has reviewed so far,
                       broken down by verdict, so the UI can show progress

The store is a plain JSON file under ``data/reviews.json`` - append-only in
practice, keyed by ``crop_path``. That keeps the store trivially inspectable
and avoids a new DB dependency for a feature that produces at most a few
hundred rows per week of active use.

Downstream uses of the collected reviews:
  * flag known-bad crop paths so the collector's static-blacklist helper
    (see visual_search / cameras.py ``roi_exclude_class``) can be updated
    manually or via a small offline script;
  * later, once enough labels accumulate, export them as a COCO-format
    dataset for a real fine-tuning pass.
"""
from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from app.visual_search import CROP_SUBDIRS, SNAPSHOTS_ROOT

_SRC_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REVIEWS_PATH = _SRC_ROOT / "data" / "reviews.json"

# Verdict values the UI is allowed to POST. Anything else is rejected.
VERDICTS = ("correct", "wrong_label", "not_an_object")

# Frame-level box verdicts: same semantic space as the crop-level VERDICTS
# but the wire values coming from the canvas UI are more compact.
# "relabel:<cls>" is the third form: the box IS a real object, the model
# just called it the wrong class - the payload carries the user's fix, so
# the training exporter can emit a corrected label instead of dropping the
# box the way a plain "wrong" does.
# "object" (2026-07-18, restoring a crop-reviewer nuance the canvas UI
# lost): the detection is a STATIC THING - a building, a pole, a road
# divider - not a mis-classified participant. Trains as a drop exactly
# like "wrong", but the explicit signal feeds the static-position
# auto-blacklist faster and reads honestly in the review history.
BOX_VERDICTS = ("correct", "wrong", "object")

# Classes a relabel verdict may target. Mirrors detect_core's
# CLASSES_OF_INTEREST without importing it here: labels.py must stay
# importable in minimal test environments where cv2/ultralytics aren't
# installed, and detect_core pulls cv2 at module import.
RELABEL_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus",
                   "train", "truck")


def valid_box_verdict(v: str) -> bool:
    """True for 'correct', 'wrong', or 'relabel:<known class>'."""
    if v in BOX_VERDICTS:
        return True
    if isinstance(v, str) and v.startswith("relabel:"):
        return v.split(":", 1)[1] in RELABEL_CLASSES
    return False


@dataclass
class Review:
    crop_path: str             # relative to SNAPSHOTS_ROOT, forward-slash form
    verdict: str               # one of VERDICTS (label opinion)
    original_cls: str          # what the detector said
    corrected_cls: str | None  # what the user says it actually is (wrong_label)
    anomaly_verdict: str | None  # "yes" / "no" - was this really an anomaly?
    note: str | None
    reviewed_at: str           # ISO-8601 UTC
    # Spec 9.1: which sampler served the crop and how uncertain the model
    # was at selection time - the raw material for the naive-vs-BADGE
    # label-efficiency comparison (D8 replay).
    sampler: str | None = None
    uncertainty_at_selection: float | None = None

    def to_public(self) -> dict:
        d = {"crop_path": self.crop_path, "verdict": self.verdict,
             "original_cls": self.original_cls, "reviewed_at": self.reviewed_at}
        if self.corrected_cls:
            d["corrected_cls"] = self.corrected_cls
        if self.anomaly_verdict:
            d["anomaly_verdict"] = self.anomaly_verdict
        if self.note:
            d["note"] = self.note
        if self.sampler:
            d["sampler"] = self.sampler
        if self.uncertainty_at_selection is not None:
            d["uncertainty_at_selection"] = self.uncertainty_at_selection
        return d


@dataclass
class FrameReview:
    """Multi-verdict review of a full frame from the new canvas UX.

    Recall becomes computable at this level: box_verdicts["3"] = "wrong"
    is an FP, and missed_detections lists FN (boxes the user drew because
    the model failed to see them).
    """
    frame_path:  str
    cam_id:      str
    box_verdicts: dict[str, str]  # {"<box_id>": "correct" | "wrong"}
    missed_detections: list[dict]  # [{"cls": str, "box": [x1,y1,x2,y2]}]
    note:        str | None
    reviewed_at: str
    # Who produced the verdicts: "operator" (dashboard clicks) or "auto"
    # (an approved model-assisted batch). The digest counts them separately -
    # the 08.08 midday report credited the operator with 60 frames that a
    # script bulk-wrote in 3 minutes, and nothing in the payload could tell
    # the difference.
    source:      str = "operator"

    def to_public(self) -> dict:
        d = {"frame_path":         self.frame_path,
             "cam_id":             self.cam_id,
             "box_verdicts":       self.box_verdicts,
             "missed_detections":  self.missed_detections,
             "reviewed_at":        self.reviewed_at}
        if self.note:
            d["note"] = self.note
        if self.source and self.source != "operator":
            d["source"] = self.source
        return d


class ReviewStore:
    """Thread-safe on-disk store for both crop-level and frame-level reviews.

    Crop reviews are keyed by ``crop_path`` (legacy from the single-crop
    review UI). Frame reviews are keyed by ``frame_path`` (new canvas UI).
    Both live in the same JSON file so a single fsync writes both sides;
    the metrics endpoint aggregates verdicts from both.
    """

    def __init__(self, path: str | Path = DEFAULT_REVIEWS_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._by_path: dict[str, Review] = {}
        self._frames_by_path: dict[str, FrameReview] = {}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        for row in data.get("reviews", []):
            try:
                u_sel = row.get("uncertainty_at_selection")
                r = Review(
                    crop_path=str(row["crop_path"]),
                    verdict=str(row["verdict"]),
                    original_cls=str(row.get("original_cls", "?")),
                    corrected_cls=row.get("corrected_cls") or None,
                    anomaly_verdict=row.get("anomaly_verdict") or None,
                    note=row.get("note") or None,
                    reviewed_at=str(row.get("reviewed_at", "")),
                    sampler=row.get("sampler") or None,
                    uncertainty_at_selection=(float(u_sel)
                                              if u_sel is not None else None))
                self._by_path[r.crop_path] = r
            except (KeyError, TypeError):
                continue
        for row in data.get("frame_reviews", []):
            try:
                bv = row.get("box_verdicts") or {}
                if not isinstance(bv, dict):
                    continue
                miss = row.get("missed_detections") or []
                if not isinstance(miss, list):
                    miss = []
                fr = FrameReview(
                    frame_path=str(row["frame_path"]),
                    cam_id=str(row.get("cam_id", "?")),
                    box_verdicts={str(k): str(v) for k, v in bv.items()
                                  if valid_box_verdict(str(v))},
                    missed_detections=[m for m in miss
                                       if isinstance(m, dict)
                                       and m.get("cls") and m.get("box")],
                    note=row.get("note") or None,
                    reviewed_at=str(row.get("reviewed_at", "")),
                    source=str(row.get("source") or "operator"))
                self._frames_by_path[fr.frame_path] = fr
            except (KeyError, TypeError):
                continue

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reviews": [r.to_public() for r in self._by_path.values()],
            "frame_reviews": [r.to_public() for r in self._frames_by_path.values()],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path)

    def is_reviewed(self, crop_path: str) -> bool:
        return crop_path in self._by_path

    def submit(self, crop_path: str, verdict: str, *,
               original_cls: str = "?",
               corrected_cls: str | None = None,
               anomaly_verdict: str | None = None,
               note: str | None = None,
               sampler: str | None = None,
               uncertainty_at_selection: float | None = None) -> Review:
        if verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {verdict!r}; expected one of "
                             f"{VERDICTS}")
        if anomaly_verdict is not None and anomaly_verdict not in ("yes", "no"):
            raise ValueError(f"invalid anomaly_verdict {anomaly_verdict!r}; "
                             f"expected 'yes' | 'no' | None")
        r = Review(
            crop_path=str(crop_path), verdict=verdict,
            original_cls=str(original_cls),
            corrected_cls=(str(corrected_cls) if corrected_cls else None),
            anomaly_verdict=(str(anomaly_verdict) if anomaly_verdict else None),
            note=(str(note) if note else None),
            reviewed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            sampler=(str(sampler) if sampler else None),
            uncertainty_at_selection=(float(uncertainty_at_selection)
                                      if uncertainty_at_selection is not None
                                      else None))
        with self._lock:
            self._by_path[r.crop_path] = r
            self._save_locked()
        return r

    def is_frame_reviewed(self, frame_path: str) -> bool:
        return frame_path in self._frames_by_path

    def submit_frame(self, frame_path: str, cam_id: str,
                     box_verdicts: dict[str, str],
                     missed_detections: list[dict],
                     note: str | None = None,
                     source: str = "operator") -> FrameReview:
        clean_bv: dict[str, str] = {}
        for k, v in (box_verdicts or {}).items():
            if valid_box_verdict(str(v)):
                clean_bv[str(k)] = str(v)
        clean_missed: list[dict] = []
        for m in (missed_detections or []):
            cls = m.get("cls") if isinstance(m, dict) else None
            box = m.get("box") if isinstance(m, dict) else None
            if not (cls and isinstance(box, (list, tuple)) and len(box) == 4):
                continue
            try:
                clean_missed.append({
                    "cls": str(cls),
                    "box": [float(x) for x in box],
                })
            except (TypeError, ValueError):
                continue
        r = FrameReview(
            frame_path=str(frame_path),
            cam_id=str(cam_id or "?"),
            box_verdicts=clean_bv,
            missed_detections=clean_missed,
            note=(str(note) if note else None),
            reviewed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            source=("auto" if source == "auto" else "operator"))
        with self._lock:
            self._frames_by_path[r.frame_path] = r
            self._save_locked()
        return r

    def summary(self) -> dict:
        counts = {v: 0 for v in VERDICTS}
        for r in self._by_path.values():
            counts[r.verdict] = counts.get(r.verdict, 0) + 1
        # Frame-level aggregates: TP, FP, FN counts across every submitted frame.
        # A relabel is a precision miss for the class the model gave (the box
        # was real but the label wrong), so it lands in the FP bucket here.
        tp = fp = fn = 0
        for fr in self._frames_by_path.values():
            for v in fr.box_verdicts.values():
                if v == "correct": tp += 1
                elif (v in ("wrong", "object")
                      or v.startswith("relabel:")): fp += 1
            fn += len(fr.missed_detections or ())
        return {
            "total_reviewed":    len(self._by_path),
            "by_verdict":        counts,
            "frames_reviewed":   len(self._frames_by_path),
            "frame_tp":          tp,
            "frame_fp":          fp,
            "frame_fn":          fn,
        }

    def rejects_for_cls(self, cls: str) -> list[str]:
        """Crop paths the user rejected as `wrong_label` or `not_an_object`
        for a given class. Useful when auditing where false positives cluster."""
        out = []
        for r in self._by_path.values():
            if r.original_cls != cls:
                continue
            if r.verdict in ("wrong_label", "not_an_object"):
                out.append(r.crop_path)
        return out


# Per-class confidence gates for the uncertainty score below. Mirrors
# detect_core.DEFAULT_PER_CLASS_CONF without importing it (cv2-free module).
_GATE_BY_CLS = {"person": 0.35, "bicycle": 0.22, "motorcycle": 0.22,
                "car": 0.35, "bus": 0.35, "train": 0.25, "truck": 0.35}
_GATE_SPAN = 0.25
# Review queue recency scope: frames older than this only surface once the
# fresh pool is fully reviewed (or explicitly, via the strip / list API).
FRESH_WINDOW_H = 48


def frame_uncertainty(meta: dict) -> float:
    """How much would a human verdict on this frame teach the system?

    Score in [0, 1]: the max over boxes of how close each detection's
    confidence sits to its class gate. A box at exactly its threshold
    (conf == gate) scores 1.0 - the model was genuinely on the fence; a
    box far above the gate scores ~0 - the model is sure and a "correct"
    click confirms nothing new. This is what lets the review queue serve
    ONLY frames worth the operator's time instead of a random pile.
    """
    best = 0.0
    for b in meta.get("boxes") or []:
        # Prefer the capture-time value (WS1: scored on the burst's
        # EFFECTIVE boosted/night gates by the collector). The margin
        # below is the fallback for pre-WS1 frames, scored on defaults.
        u = b.get("uncertainty")
        if u is not None:
            try:
                best = max(best, float(u))
                continue
            except (TypeError, ValueError):
                pass
        try:
            conf = float(b.get("conf") or 0.0)
        except (TypeError, ValueError):
            continue
        gate = _GATE_BY_CLS.get(b.get("cls"), 0.35)
        margin = 1.0 - min(1.0, abs(conf - gate) / _GATE_SPAN)
        if margin > best:
            best = margin
    return round(best, 4)


def sample_frame(store: ReviewStore,
                 snapshots_root: str | Path = SNAPSHOTS_ROOT,
                 seed: int | None = None,
                 allowed_cams: set | None = None) -> dict | None:
    """Pick the un-reviewed frame the model is LEAST sure about and return
    its metadata packaged for the canvas UI:

        {
          "frame_path":  "review_frames/cam/1000000.jpg",
          "url":         "/snapshots/review_frames/cam/1000000.jpg",
          "cam_id":      "cam",
          "frame_w":     1280,
          "frame_h":     720,
          "boxes":       [{id, cls, conf, box:[x1,y1,x2,y2]}, ...],
          "uncertainty": 0.83,
          "remaining":   17
        }

    Selection is uncertainty-first (see frame_uncertainty), newest-first on
    ties - the operator's clicks go where they teach the most. ``seed`` is
    kept for API compatibility; selection is deterministic.

    Scope (2026-07-18): the queue serves frames captured in the last
    FRESH_WINDOW_H hours when any exist - the pool keeps frames from
    every camera the collector ever visited, and surfacing week-old
    Turkey frames ahead of the live session's cameras confused the
    operator. Older frames stay reachable through the strip / list.

    Returns None when every stored frame has been reviewed.
    """
    from app.review_frames import list_all_frames, load_metadata

    rels = list_all_frames(snapshots_root)
    pool: list[tuple[float, str, dict]] = []
    cutoff_us = (time.time() - FRESH_WINDOW_H * 3600) * 1_000_000
    fresh: list[tuple[float, str, dict]] = []
    for rel in rels:
        if store.is_frame_reviewed(rel):
            continue
        # Optional camera-universe filter (the MAIN edition passes its
        # picked slots + their catalog ids so the sampler can never serve
        # a frame pool-synced from the VM's Turkey grid).
        if allowed_cams is not None:
            parts = rel.replace("\\", "/").split("/")
            if len(parts) < 2 or parts[1] not in allowed_cams:
                continue
        meta = load_metadata(rel, snapshots_root)
        if not meta:
            continue
        item = (frame_uncertainty(meta), rel, meta)
        pool.append(item)
        try:
            if int(Path(rel).stem) >= cutoff_us:
                fresh.append(item)
        except ValueError:
            pass
    pool = fresh or pool
    if not pool:
        return None
    pool.sort(key=lambda t: t[1], reverse=True)   # newest timestamp-name first
    pool.sort(key=lambda t: -t[0])                # stable: most uncertain first
    unc, rel, meta = pool[0]
    return {
        "frame_path": rel,
        "url":        f"/snapshots/{rel}",
        "cam_id":     meta.get("cam_id", "?"),
        "frame_w":    meta.get("frame_w"),
        "frame_h":    meta.get("frame_h"),
        "boxes":      meta.get("boxes", []),
        "uncertainty": unc,
        "remaining":  len(pool),
    }


def list_frames(store: ReviewStore,
                snapshots_root: str | Path = SNAPSHOTS_ROOT) -> list[dict]:
    """Every stored frame, newest first, with its review status - powers the
    frame strip that lets the user RE-OPEN a reviewed frame instead of being
    locked out by the un-reviewed-only sampler."""
    from app.review_frames import list_all_frames, load_metadata

    out: list[dict] = []
    for rel in list_all_frames(snapshots_root):
        meta = load_metadata(rel, snapshots_root) or {}
        out.append({
            "frame_path":  rel,
            "url":         f"/snapshots/{rel}",
            "cam_id":      meta.get("cam_id", "?"),
            "saved_at":    meta.get("saved_at", ""),
            "n_boxes":     len(meta.get("boxes") or []),
            "uncertainty": frame_uncertainty(meta),
            "reviewed":    store.is_frame_reviewed(rel),
        })
    out.sort(key=lambda f: f["frame_path"], reverse=True)
    return out


def load_frame(store: ReviewStore, frame_path: str,
               snapshots_root: str | Path = SNAPSHOTS_ROOT) -> dict | None:
    """Package ONE specific frame for the canvas UI - reviewed or not.

    Same shape as sample_frame() plus an ``existing`` block carrying the
    prior verdicts when the frame was already reviewed, so the UI can
    prefill the boxes and let the user fix past mistakes (re-labeling).
    Returns None when the frame or its metadata are gone.
    """
    from app.review_frames import load_metadata

    meta = load_metadata(frame_path, snapshots_root)
    if not meta:
        return None
    out = {
        "frame_path": frame_path,
        "url":        f"/snapshots/{frame_path}",
        "cam_id":     meta.get("cam_id", "?"),
        "frame_w":    meta.get("frame_w"),
        "frame_h":    meta.get("frame_h"),
        "boxes":      meta.get("boxes", []),
        "remaining":  None,
    }
    prior = store._frames_by_path.get(frame_path)  # noqa: SLF001 - same module family
    if prior is not None:
        out["existing"] = {
            "box_verdicts":      dict(prior.box_verdicts),
            "missed_detections": list(prior.missed_detections or []),
            "note":              prior.note,
            "reviewed_at":       prior.reviewed_at,
        }
    return out


def sample_crop(store: ReviewStore,
                snapshots_root: str | Path = SNAPSHOTS_ROOT,
                seed: int | None = None) -> dict | None:
    """Pick one crop the user has not reviewed yet.

    Returns {"path", "url", "cls", "from_anomaly", "remaining"} or None
    when every stored crop has been reviewed.

    Sampling favors crops the user has NOT already reviewed AND that came
    from anomaly frames (they are the ones most likely to teach the system
    something new). Anomaly candidates are picked with 70% probability
    when they exist; otherwise fall back to routine returning/events crops.
    """
    from app.visual_search import SnapshotIndex, _is_from_anomaly

    root = Path(snapshots_root)
    idx = SnapshotIndex(root)
    manifest_cls = idx._manifest_cls()  # noqa: SLF001 - deliberate reuse

    anomaly_pool: list[tuple[str, str]] = []
    routine_pool: list[tuple[str, str]] = []
    for sub in CROP_SUBDIRS:
        base = root / sub
        if not base.is_dir():
            continue
        for p in base.rglob("*.jpg"):
            if p.name.endswith("_full.jpg"):
                continue
            rel = str(p.relative_to(root)).replace("\\", "/")
            if store.is_reviewed(rel):
                continue
            cls = manifest_cls.get(rel)
            if not cls:
                import cv2 as _cv2
                img = _cv2.imread(str(p))
                cls = idx._guess_cls(p, img.shape) if img is not None else "?"  # noqa: SLF001
            (anomaly_pool if _is_from_anomaly(rel) else routine_pool).append((rel, cls))

    total = len(anomaly_pool) + len(routine_pool)
    if total == 0:
        return None
    rng = random.Random(seed) if seed is not None else random
    prefer_anomaly = anomaly_pool and rng.random() < 0.7
    pool = anomaly_pool if prefer_anomaly else (routine_pool or anomaly_pool)
    rel, cls = rng.choice(pool)
    return {
        "path": rel,
        "url": f"/snapshots/{rel}",
        "cls": cls,
        "from_anomaly": _is_from_anomaly(rel),
        "remaining": total,
    }
