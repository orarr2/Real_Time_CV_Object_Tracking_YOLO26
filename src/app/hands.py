"""Hand-gesture recognition on person crops (operator upgrade
2026-08-24): OPEN PALM (offered handshake), FIST, and POINTING with a
coarse direction. Runs MediaPipe HandLandmarker (Tasks API, the only
surface in mediapipe 1.x) on a padded crop around each visible wrist
keypoint, so the 21-landmark model only ever sees hand-sized regions
instead of the whole street frame.

Model file: src/models/hand_landmarker.task (float16, downloaded once
from the official mediapipe-models bucket). Loading is lazy and cached;
when the file or the mediapipe package is missing the layer degrades to
the pose-only gestures it always had - never an exception on the tick.
"""
from __future__ import annotations

import math
import os
import threading
from pathlib import Path

HANDS_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / \
    "hand_landmarker.task"
# Per-tick budget: hand inference is ~10-20 ms per crop on this CPU;
# 4 crops keeps the gestures tick well under the pose pass itself.
HANDS_MAX_CROPS = int(os.environ.get("HANDS_MAX_CROPS") or 4)
# A wrist keypoint below this confidence is not worth cropping around.
HANDS_MIN_KP_CONF = 0.4
# Crop half-size as a fraction of the person box height - hands are
# roughly head-sized; 0.18 of body height with padding covers fingers.
_CROP_FRAC = 0.18

_landmarker = None
_load_err: str | None = None
_LOCK = threading.Lock()

# COCO-17 wrist/elbow indices (match app.pose).
_L_WRIST, _R_WRIST = 9, 10


def _load():
    global _landmarker, _load_err
    if _landmarker is not None or _load_err is not None:
        return _landmarker
    with _LOCK:
        if _landmarker is not None or _load_err is not None:
            return _landmarker
        try:
            if not HANDS_MODEL_PATH.is_file():
                _load_err = f"model missing: {HANDS_MODEL_PATH.name}"
                return None
            from mediapipe.tasks.python import BaseOptions, vision
            # model_asset_buffer, not model_asset_path: mediapipe's C++
            # loader cannot open paths with non-ASCII characters (this
            # repo lives under a Hebrew directory name). Python reads
            # the bytes fine; the buffer path sidesteps the issue.
            opts = vision.HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_buffer=HANDS_MODEL_PATH.read_bytes()),
                num_hands=1,
                min_hand_detection_confidence=0.4,
            )
            _landmarker = vision.HandLandmarker.create_from_options(opts)
        except Exception as e:  # noqa: BLE001 - degrade, never crash a tick
            _load_err = f"{type(e).__name__}: {e}"
    return _landmarker


def load_error() -> str | None:
    """The reason hands are unavailable, or None when loaded/untried."""
    return _load_err


def _finger_states(lm) -> list[bool]:
    """Extended-or-curled per finger [thumb, index, middle, ring, pinky].

    A finger counts as extended when its TIP is farther from the wrist
    than its PIP joint by a margin - rotation-invariant, no assumption
    about which way the hand points."""
    wrist = lm[0]

    def d(i):
        return math.hypot(lm[i].x - wrist.x, lm[i].y - wrist.y)

    out = []
    for tip, pip in ((4, 3), (8, 6), (12, 10), (16, 14), (20, 18)):
        out.append(d(tip) > d(pip) * 1.08)
    return out


def _classify(lm) -> tuple[str | None, str | None]:
    """(gesture, direction). Gesture in {open_palm, fist, pointing}."""
    st = _finger_states(lm)
    extended = sum(st)
    if st[1] and not any(st[2:]):
        # index only (thumb free) -> pointing; direction from the index
        # vector in image space, quantized to 8 compass words.
        dx = lm[8].x - lm[5].x
        dy = lm[8].y - lm[5].y
        ang = math.degrees(math.atan2(-dy, dx)) % 360
        words = ("right", "up-right", "up", "up-left",
                 "left", "down-left", "down", "down-right")
        return "pointing", words[int((ang + 22.5) // 45) % 8]
    if extended >= 4:
        return "open_palm", None
    if extended <= 1:
        return "fist", None
    return None, None


def analyze_hands(frame, boxes: list[dict],
                  max_crops: int = HANDS_MAX_CROPS) -> int:
    """Stamp `hand_gesture` / `hand_dir` onto person boxes, in place.

    Only persons that already carry pose keypoints are considered - the
    wrist keypoint is both the crop anchor and the visibility gate.
    Returns the number of hands analyzed."""
    lmk = _load()
    if lmk is None:
        return 0
    import cv2
    import mediapipe as mp
    import numpy as np

    H, W = frame.shape[:2]
    candidates = []
    for b in boxes:
        if b.get("cls") != "person" or not b.get("kps"):
            continue
        bh = float(b.get("y2", 0)) - float(b.get("y1", 0))
        if bh < 96:
            continue
        for wi in (_L_WRIST, _R_WRIST):
            x, y, c = b["kps"][wi]
            if c >= HANDS_MIN_KP_CONF:
                candidates.append((bh, b, x, y))
    # largest people first - their hands have readable resolution
    candidates.sort(key=lambda t: -t[0])
    done = 0
    for bh, b, x, y in candidates:
        if done >= max_crops:
            break
        r = max(28, int(bh * _CROP_FRAC))
        x1, y1 = max(0, int(x - r)), max(0, int(y - r))
        x2, y2 = min(W, int(x + r)), min(H, int(y + r))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            res = lmk.detect(mp_img)
        except Exception:
            continue
        done += 1
        if not res.hand_landmarks:
            continue
        gesture, direction = _classify(res.hand_landmarks[0])
        if gesture and not b.get("hand_gesture"):
            b["hand_gesture"] = gesture
            if direction:
                b["hand_dir"] = direction
    return done
