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


def detect_faces(frame) -> list[dict]:
    """Face rectangles on a BGR frame:
    [{"x1","y1","x2","y2","conf"}, ...]. Empty when unavailable."""
    det = _get_detector()
    if det is None:
        return []
    try:
        h, w = frame.shape[:2]
        det.setInputSize((w, h))
        _rc, faces = det.detect(frame)
    except Exception:
        return []
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






def draw_faces(img, faces: list[dict]):
    """Draw face boxes + confidence onto `img` in place (returns it)."""
    import cv2

    for f in faces:
        x1, y1 = int(f["x1"]), int(f["y1"])
        x2, y2 = int(f["x2"]), int(f["y2"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 255), 2)
        label = f'face {f.get("conf", 0):.2f}'
        cv2.putText(img, label, (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1,
                    cv2.LINE_AA)
    return img
