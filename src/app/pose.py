"""Skeleton keypoints for person detections (COCO-17, opt-in).

The counting pipeline only needs boxes; this module adds WHAT THE BODY IS
DOING on top of them - the input for the behavior labels ("fell over",
"crouching") and the gesture pass ("hand raised", "waving") that boxes
alone cannot answer.

Design constraints, in order:

  * the detection model stays the single source of truth for WHO exists.
    The pose model (yolov8n-pose by default) runs as a SECOND pass and its
    skeletons are matched onto the detector's `person` boxes by IoU - a
    pose-person with no matching detection is discarded, never counted.
    Counts can therefore never change because pose ran;
  * additive, in place - a matched box dict gains a `"kps"` key
    ([[x, y, conf] x 17]) exactly like the tracker's `track_id` and the
    speed pass's `kmh`. Every existing consumer (counts, ROI filters,
    draw_boxes, re-ID) reads only the keys it knows, so nothing downstream
    needs changing;
  * lazy + optional - the model loads on first use only. The 24/7 VM
    round never loads it unless the operator passes `--pose` (two YOLO
    models + the OSNet embedder do not fit the 1 GB e2-micro; the flag is
    for >=2 GB hosts). The natural home is the on-demand deep window
    (`behavior.analyze_window(pose=True)`), which already accepts the cost
    of extra inference per click.

This module imports nothing heavy at module level so the pure matching
logic stays unit-testable on any machine.
"""
from __future__ import annotations

import os
import threading

# Weights for the pose pass. Any ultralytics *-pose checkpoint works; nano
# matches the collector's detection tier and auto-downloads on first use.
POSE_WEIGHTS_DEFAULT = "yolov8n-pose.pt"
# Pose-pass confidence floor. Deliberately permissive: a pose-person that
# fails to match any detection is dropped anyway, so a loose gate here
# only costs a few wasted matches, while a tight one loses skeletons for
# people the detector already vouched for.
POSE_CONF = 0.25
# A keypoint below this confidence is not drawn and not used by the
# gesture/label rules - street-cam limbs are often occluded and a
# hallucinated wrist would fire "hand raised" on noise.
KP_MIN_CONF = 0.30

# COCO-17 keypoint order (what every ultralytics pose checkpoint emits).
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

# Indices used by gestures.py / behavior_labels.py - named here once so
# the rule modules never carry magic numbers.
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# Bones + keypoints, grouped so a person carries four clearly separated
# colors at once (2026-08-16): head/face cluster yellow (nose-eyes-ears),
# arms cyan (shoulders-elbows-wrists), torso trunk green (shoulder-hip
# rectangle), legs magenta (hips-knees-ankles). Colors are BGR (cv2
# convention). Head cluster includes the intra-face edges (nose-eyes,
# eyes-ears) so a well-lit face draws a small polygon rather than one
# green stem down to the shoulders.
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
_HEAD_KPS = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)
_ARM_KPS = (L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST)
_TORSO_KPS: tuple = ()   # torso shares its keypoints with arms/legs anchors
_LEG_KPS = (L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)

_YELLOW = (0, 255, 255)
_CYAN = (255, 255, 0)
_GREEN = (0, 255, 0)
_MAGENTA = (255, 0, 255)

_BONE_GROUPS = (
    # Head/face cluster (yellow): nose-eyes-ears geometry.
    (((NOSE, L_EYE), (NOSE, R_EYE), (L_EYE, L_EAR), (R_EYE, R_EAR)),
     _YELLOW),
    # Arms (cyan): shoulder-elbow-wrist chains, both sides.
    (((L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
      (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST)),
     _CYAN),
    # Torso trunk (green): shoulder-shoulder, shoulder-hip, hip-hip.
    # The neck bones (nose->shoulders) also live here so the head cluster
    # stays isolated to the face rather than dangling down the chest.
    (((NOSE, L_SHOULDER), (NOSE, R_SHOULDER),
      (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_HIP),
      (R_SHOULDER, R_HIP), (L_HIP, R_HIP)),
     _GREEN),
    # Legs (magenta): hip-knee-ankle chains, both sides.
    (((L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
      (R_HIP, R_KNEE), (R_KNEE, R_ANKLE)),
     _MAGENTA),
)
SKELETON = tuple(edge for edges, _color in _BONE_GROUPS for edge in edges)

# Per-keypoint color map (index -> BGR) so keypoint dots also inherit the
# region color instead of one flat white.
_KP_COLORS: dict = {i: _YELLOW for i in _HEAD_KPS}
_KP_COLORS.update({i: _CYAN for i in _ARM_KPS})
_KP_COLORS.update({i: _MAGENTA for i in _LEG_KPS})

_model = None
# Lazy-load guard: two live sessions hitting the first pose tick at the
# same time must not both construct the model (the pose pass no longer
# runs under the batcher lock, so the init race is real).
_LOAD_LOCK = threading.Lock()


def load_pose_model(weights: str | None = None):
    """Load (once) and return the pose model. `POSE_WEIGHTS` env overrides
    the default; ultralytics downloads the checkpoint on first use.

    A sibling `<stem>_openvino_model` export is preferred when present -
    same ~2-3x CPU gain as the detection engine. Export once with:
        YOLO("yolov8n-pose.pt").export(format="openvino", imgsz=256)
    (256 matches the per-crop inference size below; the static engine's
    input shape must equal the predict imgsz.)"""
    global _model
    if _model is not None:
        return _model
    with _LOAD_LOCK:
        if _model is None:
            from ultralytics import YOLO
            w = weights or os.environ.get("POSE_WEIGHTS",
                                          POSE_WEIGHTS_DEFAULT)
            if str(w).endswith(".pt"):
                ov_dir = str(w)[:-3] + "_openvino_model"
                if os.path.isdir(ov_dir):
                    w = ov_dir
                    print(f"pose: OpenVINO engine loaded ({ov_dir})")
            _model = YOLO(w)
    return _model


def attach_keypoints_crops(model, frame, boxes: list[dict],
                           imgsz: int = 256, pad_frac: float = 0.25,
                           min_box_h: int = 40,
                           max_crops: int | None = None,
                           conf: float = POSE_CONF) -> int:
    """Top-down pose: one pass PER PERSON CROP instead of one full-frame
    pass. Attaches `"kps"` (frame coordinates) in place; returns matches.

    Why: on a street cam a pedestrian is 30-120 px tall; a full-frame pose
    pass at 640 hands the model ~15 px of person and finds nothing, so
    gestures/fall/crouch got no input at all on exactly the scenes they
    exist for. Cropping each detector box (padded 25%) and letting the
    pose model see it at `imgsz` multiplies the effective resolution per
    person by 5-15x. The detector stays the source of truth for WHO
    exists: only its `person` boxes are cropped, and the best pose-person
    inside each crop claims that box - there is nothing else it could be.
    Boxes shorter than `min_box_h` px are skipped - below that even a
    dedicated crop holds no limbs, only upscaling artifacts. `max_crops`
    bounds the per-call cost on crowded frames: the TALLEST boxes win
    (closest people = the ones whose skeletons are actually legible).

    Cost: one small inference per person, run crop-by-crop - a static
    batch-1 OpenVINO engine cannot take a list, and per-crop calls keep
    torch and OpenVINO checkpoints interchangeable.
    """
    persons = [b for b in boxes
               if b.get("cls") == "person"
               and (b["y2"] - b["y1"]) >= min_box_h]
    if max_crops is not None and len(persons) > max_crops:
        persons = sorted(persons, key=lambda b: b["y2"] - b["y1"],
                         reverse=True)[:max_crops]
    if not persons:
        return 0
    H, W = frame.shape[:2]
    crops, offsets = [], []
    for b in persons:
        bw, bh = b["x2"] - b["x1"], b["y2"] - b["y1"]
        px, py = bw * pad_frac, bh * pad_frac
        x1 = max(0, int(b["x1"] - px)); y1 = max(0, int(b["y1"] - py))
        x2 = min(W, int(b["x2"] + px)); y2 = min(H, int(b["y2"] + py))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        crops.append(frame[y1:y2, x1:x2])
        offsets.append((b, x1, y1))
    if not crops:
        return 0
    # Serialized behind the process-wide predict lock: concurrent
    # predict() on a shared model instance permanently wedges OpenVINO's
    # InferRequest (see detect_core._PREDICT_LOCK).
    from app.detect_core import _PREDICT_LOCK
    results = []
    with _PREDICT_LOCK:
        for crop in crops:
            results.append(model.predict(crop, imgsz=imgsz, conf=conf,
                                         verbose=False)[0])
    matched = 0
    for (b, ox, oy), res in zip(offsets, results):
        if res.boxes is None or res.keypoints is None or len(res.boxes) == 0:
            continue
        # The crop is one person's neighborhood; take the pose-person with
        # the highest box confidence (bystanders clipped at the crop edge
        # score lower and lose).
        confs = [float(c) for c in res.boxes.conf.tolist()]
        qi = max(range(len(confs)), key=confs.__getitem__)
        kps = res.keypoints.data.tolist()[qi]
        b["kps"] = [[round(x + ox, 1), round(y + oy, 1), round(c, 3)]
                    for x, y, c in kps]
        matched += 1
    return matched


def draw_skeleton(img, boxes: list[dict], min_conf: float = KP_MIN_CONF):
    """Draw the skeleton of every box carrying `"kps"` onto `img` (in
    place, returns it). Call AFTER detect_core.draw_boxes - the bones sit
    on top of the box annotation, same layering as the trail overlay."""
    import cv2

    for b in boxes:
        kps = b.get("kps")
        if not kps:
            continue
        for edges, color in _BONE_GROUPS:
            for i, j in edges:
                xi, yi, ci = kps[i]
                xj, yj, cj = kps[j]
                if ci < min_conf or cj < min_conf:
                    continue
                cv2.line(img, (int(xi), int(yi)), (int(xj), int(yj)),
                         color, 2, cv2.LINE_AA)
        # Keypoint dots: colored by body region (2026-08-16). White dots
        # on top of colored bones gave one flat "confetti" look that lost
        # the region separation; per-region dots let each of the four
        # colors read on both bones AND joints.
        for idx, (x, y, c) in enumerate(kps):
            if c >= min_conf:
                col = _KP_COLORS.get(idx, (255, 255, 255))
                cv2.circle(img, (int(x), int(y)), 3, col, -1, cv2.LINE_AA)
    return img
