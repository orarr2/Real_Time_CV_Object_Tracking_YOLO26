"""Deterministic behavior labels on top of the deep-window kinematics.

behavior.track_stats() already measures everything a label needs - speed,
moving fraction, net displacement, path - but publishes raw numbers only.
This module is the missing translation layer: every individual gets ONE
readable label ("walking", "running", "erratic"...), each with the
numbers that fired it (`label_reasons`), in the same show-your-work style
as the anomaly verdicts' observed-vs-expected.

Two evidence tiers, both deterministic (no model, no training data):

  * kinematic rules - always available, computed from the stats row the
    track already carries. Vocabulary is role-aware: a stationary person
    is "standing", a stationary vehicle is "parked";
  * pose rules - only when the pose pass ran (app/pose.py) and the track's
    boxes carry `"kps"`. They add flags a trajectory cannot see: a raised
    hand, a crouch, a body gone horizontal. A pose flag must hold in >= 2
    frames (or the only frame that has keypoints) before it counts -
    single-frame keypoint jitter is noise, not posture.

`alert` marks the labels worth walking to the screen for (a fall, erratic
zigzagging) - the same operational bar the scene anomalies use. Running
is notable but common on a street, so it labels without alerting.

Pure python, no imports beyond the keypoint index names - unit-testable
exactly like the tracker and track_stats.
"""
from __future__ import annotations

import math

from app.pose import (KP_MIN_CONF, L_ANKLE, L_HIP, L_KNEE, L_SHOULDER,
                      R_ANKLE, R_HIP, R_KNEE, R_SHOULDER, L_WRIST, R_WRIST)

# ---- kinematic thresholds ----------------------------------------------------
# Mean speed above this fraction of the frame diagonal per second reads as
# running for a person. At 640x360 that is ~88 px/s - roughly 3x the
# walking pace observed on the wide street cams, and comfortably above
# detection jitter.
RUN_SPEED_DIAG_FRAC = 0.12
# A walk is a walk when the individual actually moved most of its steps.
WALK_MOVING_FRAC = 0.60
# Dwelling = moved around but went nowhere: net displacement under this
# fraction of the diagonal while NOT reading as plain standing.
DWELL_NET_FRAC = 0.02
DWELL_MIN_SIGHTINGS = 4
# Erratic = at least this many sharp course reversals within one window.
ERRATIC_MIN_TURNS = 3
# ...where "sharp" means the heading rotated by more than this between
# consecutive significant steps.
ERRATIC_TURN_DEG = 100.0
# Steps shorter than this fraction of the diagonal carry no reliable
# heading (box jitter) and are skipped by the turn counter.
TURN_MIN_STEP_FRAC = 0.008
# ...and regardless of the frame, a significant step must cover at least
# this fraction of the OBJECT's own diagonal - box jitter scales with box
# size, and a close-range person's wobble is many frame-pixels.
TURN_MIN_BODY_FRAC = 0.35
# Erratic additionally requires the track to actually MOVE: a near-still
# individual cannot zigzag, whatever its jitter says.
ERRATIC_MIN_MOVING_FRAC = 0.30

# ---- pose thresholds ---------------------------------------------------------
# Legs shorter than this multiple of the torso = folded legs = crouching.
# Standing anatomy runs ~1.3-1.8 torso lengths hip-to-ankle; a deep crouch
# collapses it well under one.
CROUCH_LEG_TORSO_RATIO = 0.90
# Torso axis (shoulder-mid -> hip-mid) tilted further than this from the
# vertical = the body is horizontal(ish) = possible fall.
FALL_TORSO_DEG = 60.0
# A pose flag must hold in this many frames of the window to count.
POSE_FLAG_MIN_FRAMES = 2

# Labels that should pull an operator's eyes. Kept tight on purpose - the
# scene-anomaly layer already taught us that a chatty badge gets ignored.
ALERT_LABELS = frozenset({"fall_suspect", "erratic"})

VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle",
                             "bicycle", "train"})


def _kp(kps, idx):
    """Keypoint (x, y) or None when below the shared confidence gate."""
    x, y, c = kps[idx]
    return (x, y) if c >= KP_MIN_CONF else None


def _mid(a, b):
    if a is None or b is None:
        return a or b
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def pose_flags_of(kps: list) -> list[str]:
    """Posture flags from ONE frame's keypoints: hand_raised /
    both_hands_up / crouching / fall_suspect. Empty when the needed
    joints are below confidence."""
    flags: list[str] = []
    l_sh, r_sh = _kp(kps, L_SHOULDER), _kp(kps, R_SHOULDER)
    l_wr, r_wr = _kp(kps, L_WRIST), _kp(kps, R_WRIST)

    # Raised hand: wrist ABOVE its shoulder (screen y grows downward).
    left_up = l_wr is not None and l_sh is not None and l_wr[1] < l_sh[1]
    right_up = r_wr is not None and r_sh is not None and r_wr[1] < r_sh[1]
    if left_up and right_up:
        flags.append("both_hands_up")
    elif left_up or right_up:
        flags.append("hand_raised")

    sh_mid = _mid(l_sh, r_sh)
    hip_mid = _mid(_kp(kps, L_HIP), _kp(kps, R_HIP))
    if sh_mid is not None and hip_mid is not None:
        dx, dy = hip_mid[0] - sh_mid[0], hip_mid[1] - sh_mid[1]
        torso_len = math.hypot(dx, dy)
        # A torso under 8px is not anatomy - it is a skeleton hallucinated
        # onto a distant blob, and its "tilt" fired fall_suspect on noise.
        # Real crops that matter (>= 40px boxes via the top-down pose pass)
        # always carry a torso well above this.
        if torso_len >= 8.0:
            tilt = math.degrees(math.atan2(abs(dx), abs(dy)))
            if tilt > FALL_TORSO_DEG:
                flags.append("fall_suspect")
            else:
                ankle_mid = _mid(_kp(kps, L_ANKLE), _kp(kps, R_ANKLE))
                knee_mid = _mid(_kp(kps, L_KNEE), _kp(kps, R_KNEE))
                lower = ankle_mid or knee_mid
                if lower is not None and ankle_mid is not None:
                    legs_len = abs(lower[1] - hip_mid[1])
                    if legs_len < CROUCH_LEG_TORSO_RATIO * torso_len:
                        flags.append("crouching")
    return flags


def heading_turns(path: list, frame_shape,
                  min_step_frac: float = TURN_MIN_STEP_FRAC,
                  turn_deg: float = ERRATIC_TURN_DEG,
                  body_diag_px: float | None = None) -> int:
    """Count sharp course reversals along a track_stats `path`
    ([[t, nx, ny], ...] normalized). Jitter-sized steps carry no heading
    and are skipped rather than counted as turns.

    `body_diag_px` scales the jitter floor to the OBJECT: detection
    jitter grows with box size, so a frame-relative floor alone let a
    close-range seated person's box wobble read as "course reversals"
    (audit 2026-08-14: 4-6 false ERRATIC alerts per tick on a seated
    group). A significant step must now also cover >= 35% of the
    object's own diagonal."""
    H, W = frame_shape[:2]
    diag = (H * H + W * W) ** 0.5 or 1.0
    pts = [(nx * W, ny * H) for _t, nx, ny in path]
    min_step = min_step_frac * diag
    if body_diag_px:
        min_step = max(min_step, TURN_MIN_BODY_FRAC * float(body_diag_px))
    headings: list[float] = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        dx, dy = x1 - x0, y1 - y0
        if math.hypot(dx, dy) >= min_step:
            headings.append(math.atan2(dy, dx))
    turns = 0
    for h0, h1 in zip(headings, headings[1:]):
        delta = abs(h1 - h0)
        if delta > math.pi:
            delta = 2 * math.pi - delta
        if math.degrees(delta) > turn_deg:
            turns += 1
    return turns


def _sustained_pose_flags(kps_seq: list) -> list[str]:
    """Flags that held in >= POSE_FLAG_MIN_FRAMES frames. The bar relaxes
    only for tracks SHORTER than that (a 1-frame track can never show two
    frames of anything) - a long track where pose matched a single frame
    stays flagless, because one skeleton out of twelve is jitter, not
    posture. `kps_seq` is one entry per track box, None where the pose
    pass had nothing for that frame."""
    counts: dict[str, int] = {}
    for kps in kps_seq:
        if not kps:
            continue
        for f in pose_flags_of(kps):
            counts[f] = counts.get(f, 0) + 1
    need = min(POSE_FLAG_MIN_FRAMES, max(1, len(kps_seq)))
    return sorted(f for f, n in counts.items() if n >= need)


def label_track(row: dict, frame_shape, kps_seq: list | None = None) -> dict:
    """One label + its evidence for one track_stats row.

    Returns {"label", "alert", "label_reasons", "pose_flags"} - the caller
    merges it into the row. Never mutates its inputs.
    """
    H, W = frame_shape[:2]
    diag = (H * H + W * W) ** 0.5 or 1.0
    cls = row.get("cls")
    is_vehicle = cls in VEHICLE_CLASSES
    reasons: list[str] = []

    pose_flags = _sustained_pose_flags(kps_seq) if kps_seq else []

    turns = heading_turns(row.get("path") or [], frame_shape,
                          body_diag_px=row.get("bbox_diag_px"))
    mean_speed = float(row.get("mean_speed_px_s") or 0.0)
    moving_frac = float(row.get("moving_frac") or 0.0)
    net = float(row.get("net_disp_px") or 0.0)

    if "fall_suspect" in pose_flags:
        label = "fall_suspect"
        reasons.append(f"torso past {FALL_TORSO_DEG:.0f}deg from vertical "
                       f"in >={POSE_FLAG_MIN_FRAMES} frames")
    elif (turns >= ERRATIC_MIN_TURNS
          and moving_frac >= ERRATIC_MIN_MOVING_FRAC
          and not row.get("stationary")):
        label = "erratic"
        reasons.append(f"{turns} course reversals > "
                       f"{ERRATIC_TURN_DEG:.0f}deg (>= {ERRATIC_MIN_TURNS}), "
                       f"moving {moving_frac:.0%}")
    elif (not is_vehicle
          and mean_speed >= RUN_SPEED_DIAG_FRAC * diag):
        label = "running"
        reasons.append(f"mean {mean_speed:.0f} px/s >= "
                       f"{RUN_SPEED_DIAG_FRAC * diag:.0f} "
                       f"({RUN_SPEED_DIAG_FRAC:.2f} x diag)")
    elif row.get("stationary"):
        label = "parked" if is_vehicle else "standing"
        reasons.append("stationary (low movement, no net displacement)")
    elif (net < DWELL_NET_FRAC * diag
          and int(row.get("sightings") or 0) >= DWELL_MIN_SIGHTINGS):
        label = "dwelling"
        reasons.append(f"net {net:.0f}px < {DWELL_NET_FRAC * diag:.0f}px "
                       f"over {row.get('sightings')} sightings")
    elif moving_frac >= WALK_MOVING_FRAC:
        label = "driving" if is_vehicle else "walking"
        reasons.append(f"moving {moving_frac:.0%} of steps "
                       f">= {WALK_MOVING_FRAC:.0%}")
    else:
        label = "normal"
        reasons.append("no rule fired (slow drift)")

    return {
        "label": label,
        "alert": label in ALERT_LABELS,
        "label_reasons": reasons,
        "pose_flags": pose_flags,
    }
