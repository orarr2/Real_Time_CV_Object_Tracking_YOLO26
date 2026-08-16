"""Coarse hand-gesture recognition from pose keypoints over a window.

Arm-level gestures only, by design. A street camera sees a whole square;
a hand is a handful of pixels and individual fingers simply are not in
the signal - so this module recognizes what the skeleton CAN carry at
that distance, and does it temporally (a gesture is a movement pattern
over the window, not one frame's silhouette):

  * hand_raised    - a wrist held above its shoulder for a sustained run
                     of frames (one frame is noise, three is a posture);
  * both_hands_up  - both wrists above both shoulders simultaneously,
                     same sustain rule;
  * wave           - a RAISED wrist swinging side to side across its
                     elbow: >= WAVE_MIN_SWINGS sign changes of
                     (wrist_x - elbow_x) while up. The elbow anchor makes
                     the test translation-invariant - a raised hand on a
                     walking person does not "wave" just because the body
                     moves.

Finger-level vocabulary (peace signs, counts) needs a close-range camera
and a hand-keypoint model; it is deliberately out of scope here and noted
in the README as a notebook-only experiment.

Input convention: `kps_seq` is one entry per track box - a 17x[x, y, conf]
keypoint list where the pose pass matched this individual, None where it
did not. Pure python, unit-testable with synthetic skeletons.
"""
from __future__ import annotations

from app.pose import (KP_MIN_CONF, L_ELBOW, L_SHOULDER, L_WRIST,
                      R_ELBOW, R_SHOULDER, R_WRIST)

# A raised wrist must hold this many CONSECUTIVE pose frames. With the
# deep window's default ~0.5s stride that is ~1.5s of held-up hand.
RAISE_MIN_FRAMES = 3
# ...unless the window only produced this few pose frames total, where the
# sustain requirement relaxes to all-of-them (short windows still answer).
RAISE_MIN_VALID = 2
# A wave is at least this many side-of-elbow changes while raised.
WAVE_MIN_SWINGS = 2
# Deadband floor: the wrist must clear the elbow x by at least this many
# px for a side to count. The EFFECTIVE deadband scales with the skeleton
# (WAVE_DEADBAND_SHOULDER_FRAC of the shoulder width, floored here) - a
# fixed 3px was simultaneously too twitchy for a close 300px person
# (8px of honest jitter read as swings) and near the noise floor for a
# distant 40px one.
WAVE_DEADBAND_PX = 2.0
WAVE_DEADBAND_SHOULDER_FRAC = 0.08

_SIDES = ((L_WRIST, L_ELBOW, L_SHOULDER), (R_WRIST, R_ELBOW, R_SHOULDER))


def _pt(kps, idx):
    x, y, c = kps[idx]
    return (x, y) if c >= KP_MIN_CONF else None


def _side_frames(kps_seq, wrist_i, elbow_i, shoulder_i):
    """Per pose-frame (raised?, wrist_x - elbow_x, deadband_px) for one
    arm; None for frames where any needed joint is missing or below
    confidence. The deadband is sized from THAT frame's shoulder width so
    the same rule works on a 40px person and a 300px one."""
    out = []
    for kps in kps_seq:
        if not kps:
            out.append(None)
            continue
        wrist, elbow, shoulder = (_pt(kps, wrist_i), _pt(kps, elbow_i),
                                  _pt(kps, shoulder_i))
        if wrist is None or shoulder is None:
            out.append(None)
            continue
        raised = wrist[1] < shoulder[1]          # screen y grows downward
        dx = (wrist[0] - elbow[0]) if elbow is not None else None
        l_sh, r_sh = _pt(kps, L_SHOULDER), _pt(kps, R_SHOULDER)
        sh_w = (abs(r_sh[0] - l_sh[0])
                if l_sh is not None and r_sh is not None else 0.0)
        dead = max(WAVE_DEADBAND_PX, WAVE_DEADBAND_SHOULDER_FRAC * sh_w)
        out.append((raised, dx, dead))
    return out


def _longest_run(flags: list[bool]) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def _sustain_needed(n_valid: int) -> int:
    return min(RAISE_MIN_FRAMES, max(RAISE_MIN_VALID, n_valid))


def detect_gestures(kps_seq: list) -> list[str]:
    """Gestures observed across one individual's window. Returns a sorted
    subset of {"hand_raised", "both_hands_up", "wave"} (raised is implied
    by the other two and therefore dropped when they fire)."""
    if not kps_seq or not any(kps_seq):
        return []

    sides = [_side_frames(kps_seq, w, e, s) for w, e, s in _SIDES]
    gestures: set[str] = set()

    # Raised, per arm (sustained consecutive run).
    raised_any = False
    for frames in sides:
        flags = [bool(f and f[0]) for f in frames]
        n_valid = sum(1 for f in frames if f is not None)
        if n_valid and _longest_run(flags) >= _sustain_needed(n_valid):
            raised_any = True
    if raised_any:
        gestures.add("hand_raised")

    # Both up simultaneously (sustained).
    both_flags = [bool(l and l[0] and r and r[0])
                  for l, r in zip(sides[0], sides[1])]
    n_both_valid = sum(1 for l, r in zip(sides[0], sides[1])
                       if l is not None and r is not None)
    if n_both_valid and _longest_run(both_flags) >= _sustain_needed(n_both_valid):
        gestures.add("both_hands_up")

    # Wave: swings of the raised wrist across the elbow.
    for frames in sides:
        swings, prev_sign = 0, 0
        for f in frames:
            if not f or not f[0] or f[1] is None:
                continue
            raised, dx, dead = f
            if abs(dx) < dead:
                continue
            sign = 1 if dx > 0 else -1
            if prev_sign and sign != prev_sign:
                swings += 1
            prev_sign = sign
        if swings >= WAVE_MIN_SWINGS:
            gestures.add("wave")
            break

    if gestures & {"both_hands_up", "wave"}:
        gestures.discard("hand_raised")
    return sorted(gestures)
