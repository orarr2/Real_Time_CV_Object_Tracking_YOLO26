"""Per-track kinematics for the live Body layer.

track_stats() turns one track's box history into a compact profile
(path, speed, moving fraction, direction, dwell cells) that
live_analysis consumes every tick to derive body-anomaly verdicts.
_TRAIL_COLORS gives each track id a stable trail color on the canvas.

The deep-window analyze_window() pipeline that used to live here
(operator-triggered long-window profiling with pose/faces/lock extras)
was removed in the 2026-08-23 cleanup - no dashboard surface called it.
"""
from __future__ import annotations


from app.heatmap import GRID_H, GRID_W

# Class-length ruler for the km/h estimate (moved here from detect_core
# in the 2026-08-23 cleanup - track_stats is its only consumer).
VEHICLE_LENGTH_M = {
    "car": 4.5, "truck": 10.0, "bus": 12.0,
    "motorcycle": 2.2, "bicycle": 1.8, "train": 22.0,
}
from app.tracker import _centroid


# A step slower than this fraction of the frame diagonal per second is
# "standing" (detection jitter moves a box a few px between frames).
MOVING_EPS_DIAG_FRAC = 0.005

# Net displacement below this fraction of the diagonal = the individual
# ended the window where it started.
STATIONARY_NET_FRAC = 0.02

_DIRECTIONS = ("right", "down-right", "down", "down-left",
               "left", "up-left", "up", "up-right")

# Layer rendering lives in app/live_analysis.py since fix 2 (the live
# per-tile engine); fix 3 removed this module's one-shot layers branch -
# it had no UI caller left. This API's scope is the per-individual
# window profile + the annotated trails view below.


def _foot(b: dict) -> tuple[float, float]:
    return (b["x1"] + b["x2"]) / 2.0, b["y2"]


def _direction_of(dx: float, dy: float) -> str:
    """Dominant screen direction (y grows downward)."""
    import math
    octant = int(round(math.atan2(dy, dx) / (math.pi / 4))) % 8
    return _DIRECTIONS[octant]


def track_stats(cls: str | None, boxes: list[dict], times: list[float],
                frame_shape) -> dict:
    """Behavior profile of one track. Pure math - unit-testable without
    cv2, streams, or a model."""
    H, W = frame_shape[:2]
    diag = (H * H + W * W) ** 0.5 or 1.0
    feet = [_foot(b) for b in boxes]
    cents = [_centroid(b) for b in boxes]

    path_len = 0.0
    speeds: list[float] = []
    moving_steps = 0
    for (x0, y0), (x1, y1), t0, t1 in zip(cents, cents[1:],
                                          times, times[1:]):
        d = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        path_len += d
        dt = t1 - t0
        if dt > 0:
            v = d / dt
            speeds.append(v)
            if v >= MOVING_EPS_DIAG_FRAC * diag:
                moving_steps += 1

    net_dx = cents[-1][0] - cents[0][0]
    net_dy = cents[-1][1] - cents[0][1]
    net = (net_dx ** 2 + net_dy ** 2) ** 0.5
    n_steps = max(1, len(boxes) - 1)
    moving_frac = moving_steps / n_steps if speeds else 0.0
    stationary = (moving_frac < 0.2 and net < STATIONARY_NET_FRAC * diag)

    kmh = None
    real_len = VEHICLE_LENGTH_M.get(cls or "")
    if real_len and speeds:
        exts = [max(b["x2"] - b["x1"], b["y2"] - b["y1"]) for b in boxes]
        exts = [e for e in exts if e > 0]
        if exts:
            m_per_px = real_len / (sum(exts) / len(exts))
            kmh = round(sum(speeds) / len(speeds) * m_per_px * 3.6, 1)

    zones = sorted({
        f"{min(GRID_W - 1, int(fx / W * GRID_W))},"
        f"{min(GRID_H - 1, int(fy / H * GRID_H))}"
        for fx, fy in feet if 0 <= fx <= W and 0 <= fy <= H})

    diags = [((b["x2"] - b["x1"]) ** 2 + (b["y2"] - b["y1"]) ** 2) ** 0.5
             for b in boxes]
    return {
        "cls": cls,
        "sightings": len(boxes),
        # Mean box diagonal - the object's own size on screen. Consumers
        # (behavior_labels.heading_turns) use it to scale jitter floors to
        # the OBJECT instead of the frame.
        "bbox_diag_px": round(sum(diags) / len(diags), 1) if diags else 0.0,
        "t_first": round(times[0], 2),
        "t_last": round(times[-1], 2),
        "path": [[round(t, 2), round(fx / W, 3), round(fy / H, 3)]
                 for t, (fx, fy) in zip(times, feet)],
        "path_len_px": round(path_len, 1),
        "net_disp_px": round(net, 1),
        "mean_speed_px_s": round(sum(speeds) / len(speeds), 1) if speeds else 0.0,
        "max_speed_px_s": round(max(speeds), 1) if speeds else 0.0,
        "moving_frac": round(moving_frac, 2),
        "stationary": stationary,
        "direction": (_direction_of(net_dx, net_dy)
                      if net >= STATIONARY_NET_FRAC * diag else None),
        "kmh_est": kmh,
        "zones": zones,
    }




# Trail palette (BGR) - cycled by track id so neighboring ids differ.
_TRAIL_COLORS = ((80, 175, 76), (60, 130, 246), (0, 200, 255),
                 (200, 60, 200), (0, 90, 230), (255, 160, 0),
                 (180, 220, 40), (140, 100, 255))














