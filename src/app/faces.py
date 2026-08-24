"""Face DETECTION (and privacy blurring) - never face identification.

Scope, stated up front because it matters: this module finds face
RECTANGLES. It computes no face embeddings, keeps no face database and
matches nobody against anything - the project's "have I seen this
before?" question stays answered by body-appearance re-ID (app/reid.py),
which is the honest tool at street-camera distance where a face is a
dozen pixels. What face boxes ARE good for here:

  * `--blur-faces` (collector) - privacy mode: every snapshot the
    pipeline publishes (anomaly frames, model view, event crops + full
    frames, heatmap base) gets its faces gaussian-blurred BEFORE the
    bytes leave the process. Detection, counting and re-ID all run on the
    unblurred in-memory frame first, so enabling the flag changes nothing
    about the numbers - only about what viewers of the dashboard see;
  * deep-window annotation - face boxes drawn on the analysis frame
    (close-range cameras), the classic "face 0.87" overlay.

Detector: OpenCV's bundled YuNet (cv2.FaceDetectorYN) driving a ~230 KB
ONNX file - no new python dependency. The model file is NOT committed;
point `FACE_MODEL` at a downloaded copy (see README). Everything here
degrades to a silent no-op when the file or the cv2 API is absent:
`detect_faces()` returns []. A missing optional model must never cost a
sample.
"""
from __future__ import annotations

import os
from pathlib import Path

FACE_MODEL_ENV = "FACE_MODEL"
# Default drop path for the YuNet ONNX (see tools/setup_reid.sh precedent):
# used when FACE_MODEL is unset. Committing the ~230 KB model makes face
# DETECTION actually runnable - until 2026-08-08 no machine had the file,
# so every face feature was silently dead.
FACE_MODEL_DEFAULT = (Path(__file__).resolve().parent.parent / "data"
                      / "face_detection_yunet_2023mar.onnx")
# YuNet score threshold - below this a candidate is background texture.
FACE_SCORE = 0.60
FACE_NMS = 0.30
# Gaussian kernel is sized from the face box itself (an odd fraction of
# its width) so near faces get a heavy blur and far ones are not wasted on.

# Flipped by the collector's --blur-faces flag; module-level so every
# save-path helper sees one switch.

_detector = None
_failed = False


def _model_path() -> str | None:
    p = os.environ.get(FACE_MODEL_ENV, "").strip()
    if p and os.path.isfile(p):
        return p
    if not p and FACE_MODEL_DEFAULT.is_file():
        return str(FACE_MODEL_DEFAULT)
    return None


def _get_detector():
    """Build (once) the YuNet detector, or None when unavailable."""
    global _detector, _failed
    if _detector is not None:
        return _detector
    if _failed:
        return None
    path = _model_path()
    if path is None:
        _failed = True
        return None
    try:
        import cv2
        try:
            _detector = cv2.FaceDetectorYN.create(path, "", (320, 320),
                                                  FACE_SCORE, FACE_NMS, 5000)
        except cv2.error:
            # OpenCV's C++ loader cannot open non-ASCII ABSOLUTE paths on
            # Windows (the operator's repo lives under a Hebrew-named
            # folder). The path relative to the working directory (src/)
            # is plain ASCII - retry with it before giving up.
            _detector = cv2.FaceDetectorYN.create(
                os.path.relpath(path), "", (320, 320),
                FACE_SCORE, FACE_NMS, 5000)
    except Exception:
        _failed = True
        return None
    return _detector


def available() -> bool:
    """True when a face model is configured, loadable and ready."""
    return _get_detector() is not None


# Second-pass upscale factor for the far-field rescue (see detect_faces).
# 1.5x turns a 12 px street-distance face into ~18 px - back inside
# YuNet's practical floor - while keeping the extra pass cheap.
FACE_RESCUE_SCALE = 1.5
FACE_RESCUE_MAX_W = 1920   # never upscale beyond this width (CPU guard)


def _detect_once(det, frame, w, h) -> list[dict]:
    det.setInputSize((w, h))
    _rc, faces = det.detect(frame)
    out: list[dict] = []
    for f in (faces if faces is not None else []):
        x, y, fw, fh = (float(f[0]), float(f[1]),
                        float(f[2]), float(f[3]))
        out.append({
            "x1": max(0.0, x), "y1": max(0.0, y),
            "x2": min(float(w), x + fw), "y2": min(float(h), y + fh),
            "conf": round(float(f[-1]), 3),
        })
    return out


def detect_faces(frame) -> list[dict]:
    """Face rectangles on a BGR frame:
    [{"x1","y1","x2","y2","conf"}, ...]. Empty when unavailable.

    Two-scale pass (operator upgrade decision 2026-08-23): YuNet runs on
    the native frame first; when that returns nothing - the common case
    on far-field street cams where every face is under ~15 px - a second
    pass runs on a FACE_RESCUE_SCALE upscale and the hits are mapped
    back to native coordinates. Costs nothing when the first pass
    already found faces.
    """
    det = _get_detector()
    if det is None:
        return []
    try:
        h, w = frame.shape[:2]
        out = _detect_once(det, frame, w, h)
        if not out and w * FACE_RESCUE_SCALE <= FACE_RESCUE_MAX_W:
            import cv2
            up = cv2.resize(frame, (int(w * FACE_RESCUE_SCALE),
                                    int(h * FACE_RESCUE_SCALE)),
                            interpolation=cv2.INTER_CUBIC)
            uh, uw = up.shape[:2]
            for r in _detect_once(det, up, uw, uh):
                out.append({
                    "x1": r["x1"] / FACE_RESCUE_SCALE,
                    "y1": r["y1"] / FACE_RESCUE_SCALE,
                    "x2": r["x2"] / FACE_RESCUE_SCALE,
                    "y2": r["y2"] / FACE_RESCUE_SCALE,
                    "conf": r["conf"],
                })
    except Exception:
        return []
    return out






FACE_DRAW_COLOR = (0, 165, 255)   # uniform orange (operator 2026-08-24)


def draw_faces(img, faces: list[dict]):
    """Draw face boxes onto `img` in place (returns it).

    Operator direction 2026-08-24: ONE uniform orange for every face
    (per-face colors could not be matched between frame and side cards)
    plus a unique face NUMBER on the frame itself - the number is the
    pairing key to the side card, and the confidence lives on the card
    rather than cluttering the frame."""
    import cv2

    for i, f in enumerate(faces, 1):
        x1, y1 = int(f["x1"]), int(f["y1"])
        x2, y2 = int(f["x2"]), int(f["y2"])
        f.setdefault("n", i)
        cv2.rectangle(img, (x1, y1), (x2, y2), FACE_DRAW_COLOR, 2)
        cv2.putText(img, str(f["n"]), (x1, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, FACE_DRAW_COLOR, 2,
                    cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Face-crop audit trail (operator request 2026-08-24): same idea as the
# plate-crop store - each saved face crop lands at
# src/data/face_crops/<cam>/<ts>_<n>_<conf>.jpg so the Investigation
# flow can pull every angle a face was captured from and offer SAVE.
# A per-cell cooldown stops the same standing person from flooding the
# store every tick, and a rolling cap prunes the oldest files.
# ---------------------------------------------------------------------------
FACE_CROPS_ROOT = Path(__file__).resolve().parent.parent / "data" / "face_crops"
FACE_CROP_COOLDOWN_S = 30.0
FACE_CROPS_KEEP = int(os.environ.get("FACE_CROPS_KEEP") or 300)
_face_crop_last: dict = {}


def save_face_crops(cam_id: str, frame, faces: list[dict],
                    now: float | None = None, pad: float = 0.25) -> int:
    """Persist padded crops for this tick's faces; returns saves made.
    Silent-on-failure by design - an audit trail must never break the
    live tick."""
    import time as _t
    import cv2
    if not faces:
        return 0
    now = now or _t.time()
    H, W = frame.shape[:2]
    out_dir = FACE_CROPS_ROOT / (cam_id or "unknown")
    saved = 0
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in faces:
            x1, y1 = int(f["x1"]), int(f["y1"])
            x2, y2 = int(f["x2"]), int(f["y2"])
            # cooldown key: coarse grid cell, so a static face saves at
            # most once per FACE_CROP_COOLDOWN_S.
            key = (cam_id, x1 // 64, y1 // 64)
            if now - _face_crop_last.get(key, 0.0) < FACE_CROP_COOLDOWN_S:
                continue
            px = int((x2 - x1) * pad)
            py = int((y2 - y1) * pad)
            cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
            cx2, cy2 = min(W, x2 + px), min(H, y2 + py)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            name = f"{int(now * 1000)}_{f.get('n', 0)}_" \
                   f"{int(round(float(f.get('conf', 0)) * 100)):02d}.jpg"
            cv2.imwrite(str(out_dir / name), crop)
            _face_crop_last[key] = now
            saved += 1
        # rolling cap: prune oldest beyond FACE_CROPS_KEEP
        files = sorted(out_dir.glob("*.jpg"))
        for old in files[:-FACE_CROPS_KEEP]:
            try:
                old.unlink()
            except OSError:
                pass
    except Exception:
        pass
    return saved
