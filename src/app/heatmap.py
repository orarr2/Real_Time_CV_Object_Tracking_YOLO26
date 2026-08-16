"""Long-horizon presence heatmap: WHERE activity stands, weighted by time.

Every successful sample already computes gated + ROI-filtered boxes; this
module banks each box's FOOT POINT (bottom-center, the same convention the
ROI filter uses) into a coarse normalized grid, weighted by the seconds
the observation "covers" (the gap since this camera's previous sample). A
person planted in one spot for ten minutes therefore accumulates ~15x the
weight of a passer-by crossing one sample - the map reads as dwell, not
just traffic.

Three layers per camera - `person`, `vehicles` (the road set, same
VEHICLE_NAMES convention as the counts), `other` (train + any
EXTRA_CLASSES additions) - each split into four local-time dayparts so
"where do people stand at night" and "where does the morning rush stack
up" are separate answers. Weights decay a few percent per day, so the map
follows the street's CURRENT layout instead of fossilizing a construction
detour forever.

Costs nothing measurable: pure accumulation from already-computed boxes,
one small JSON per camera (data/heatmap_<cam>.json), zero Firestore
writes. Rendering (cv2 colormap over the live frame) happens every
RENDER_EVERY_SAMPLES samples per camera and on demand from the dashboard.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _SRC_ROOT / "data"

# Grid resolution. 48x27 (16:9) - one cell is ~40px at 1080p, tight enough
# to separate two adjacent shop fronts that the previous 32x18 (60px cells)
# blurred into one hot blob, while a full state JSON stays ~200 KB. The
# _load shape guard restarts any 32x18 file cleanly (decay half-life is
# ~3 weeks anyway, so the map re-forms quickly).
GRID_W, GRID_H = 48, 27

# Local-time dayparts (camera timezone - a Bangkok evening is not an
# Istanbul evening).
DAYPARTS = ("night", "morning", "afternoon", "evening")

# Per-day multiplicative decay. 0.97/day ~ half-life of three weeks: long
# enough to keep a stable picture, short enough to forget a re-routed
# street within a month.
DAILY_DECAY = 0.97

# Accumulation weight = seconds since this camera's previous sample,
# clamped: a first-ever sample gets the default, a stream that was down
# for an hour must not credit its comeback frame with the whole hour.
WEIGHT_DEFAULT_S = 40.0
WEIGHT_MIN_S = 5.0
WEIGHT_MAX_S = 180.0

# Persist at most every N accumulations per camera (plus an age backstop)
# - the VM's disk sees a few writes per hour per camera, not per sample.
SAVE_EVERY = 15
SAVE_MAX_AGE_S = 600.0

# Re-render the published overlay JPEG every N samples per camera.
RENDER_EVERY_SAMPLES = 30

# The road-vehicle set. Mirrors detect_core.VEHICLE_NAMES without the
# import: this module must stay importable in minimal test environments
# (labels.py precedent), and detect_core pulls cv2 at module import.
_VEHICLE_NAMES = ("bicycle", "car", "motorcycle", "bus", "truck")

# cam_id -> state dict (lazy-loaded from disk). State shape:
#   {"layers": {layer: {daypart: [GRID_H rows][GRID_W cols] floats}},
#    "samples": int, "updated": epoch, "decay_day": "YYYY-MM-DD"}
_STATE: dict[str, dict] = {}
_LAST_TS: dict[str, float] = {}
_DIRTY: dict[str, int] = {}
_LAST_SAVE: dict[str, float] = {}


def _path(cam_id: str, root: Path | None = None) -> Path:
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_"
                   for ch in cam_id)
    return (root or DATA_DIR) / f"heatmap_{safe}.json"


def _blank_state() -> dict:
    return {
        "layers": {layer: {dp: [[0.0] * GRID_W for _ in range(GRID_H)]
                           for dp in DAYPARTS}
                   for layer in ("person", "vehicles", "other")},
        "samples": 0,
        "updated": 0.0,
        "decay_day": "",
    }


def _load(cam_id: str, root: Path | None = None) -> dict:
    st = _STATE.get(cam_id)
    if st is not None:
        return st
    p = _path(cam_id, root)
    if p.exists():
        try:
            st = json.loads(p.read_text(encoding="utf-8"))
            # Shape guard: a grid-size change between versions restarts the
            # map instead of crashing every accumulate.
            grid = st["layers"]["person"][DAYPARTS[0]]
            if len(grid) != GRID_H or len(grid[0]) != GRID_W:
                raise ValueError("grid shape changed")
        except Exception:
            st = _blank_state()
    else:
        st = _blank_state()
    _STATE[cam_id] = st
    return st


def layer_for_class(cls: str | None) -> str:
    if cls == "person":
        return "person"
    if cls in _VEHICLE_NAMES:
        return "vehicles"
    return "other"


def daypart_for_hour(hour: int) -> str:
    if 6 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 17:
        return "afternoon"
    if 18 <= hour <= 21:
        return "evening"
    return "night"


def _foot_cell(box: dict, frame_w: float, frame_h: float
               ) -> tuple[int, int] | None:
    fx = (box["x1"] + box["x2"]) / 2.0 / frame_w
    fy = box["y2"] / frame_h
    if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
        return None
    gx = min(GRID_W - 1, int(fx * GRID_W))
    gy = min(GRID_H - 1, int(fy * GRID_H))
    return gx, gy


def _maybe_decay(st: dict, local_day: str) -> None:
    prev = st.get("decay_day") or ""
    if prev == local_day:
        return
    if prev:
        try:
            import datetime as _dt
            d0 = _dt.date.fromisoformat(prev)
            d1 = _dt.date.fromisoformat(local_day)
            days = max(0, (d1 - d0).days)
        except ValueError:
            days = 1
        if days:
            f = DAILY_DECAY ** days
            for layer in st["layers"].values():
                for grid in layer.values():
                    for row in grid:
                        for i, v in enumerate(row):
                            if v:
                                row[i] = v * f
    st["decay_day"] = local_day


def accumulate(cam_id: str, boxes: list[dict], frame_shape,
               now: float | None = None, tz=None,
               root: Path | None = None) -> None:
    """Bank one sample's boxes. Weight = observed interval, see module doc."""
    if not boxes:
        return
    now = time.time() if now is None else now
    import datetime as _dt
    local = _dt.datetime.fromtimestamp(now, tz or _dt.timezone.utc)
    st = _load(cam_id, root)
    _maybe_decay(st, local.date().isoformat())
    dp = daypart_for_hour(local.hour)
    last = _LAST_TS.get(cam_id)
    weight = (WEIGHT_DEFAULT_S if last is None
              else min(WEIGHT_MAX_S, max(WEIGHT_MIN_S, now - last)))
    _LAST_TS[cam_id] = now
    H, W = frame_shape[:2]
    if not (H and W):
        return
    for b in boxes:
        cell = _foot_cell(b, float(W), float(H))
        if cell is None:
            continue
        gx, gy = cell
        st["layers"][layer_for_class(b.get("cls"))][dp][gy][gx] += weight
    st["samples"] += 1
    st["updated"] = now
    _DIRTY[cam_id] = _DIRTY.get(cam_id, 0) + 1
    # First accumulate arms the age clock instead of firing it - otherwise
    # every process start would write immediately for no reason.
    _LAST_SAVE.setdefault(cam_id, now)
    if (_DIRTY[cam_id] >= SAVE_EVERY
            or now - _LAST_SAVE[cam_id] >= SAVE_MAX_AGE_S):
        save(cam_id, root)


def export_state(cam_id: str) -> dict | None:
    """Slim copy of a camera's accumulated grids (same rounding save()
    uses on disk). The collector publishes this next to the overlay JPEG
    (snapshots/heatmaps/<cam>.json) so the operator dashboard can render
    ANY layer x daypart combination on demand - fix 3: the stored depth
    (person/vehicles/other x four dayparts) finally has a consumer."""
    st = _STATE.get(cam_id)
    if st is None:
        return None
    return {
        "layers": {ln: {dp: [[round(v, 2) for v in row] for row in grid]
                        for dp, grid in layer.items()}
                   for ln, layer in st["layers"].items()},
        "samples": st["samples"],
        "updated": st["updated"],
    }


def save(cam_id: str, root: Path | None = None) -> None:
    st = _STATE.get(cam_id)
    if st is None:
        return
    p = _path(cam_id, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Round on the way out: full float reprs double the file for noise.
    slim = {
        "layers": {ln: {dp: [[round(v, 2) for v in row] for row in grid]
                        for dp, grid in layer.items()}
                   for ln, layer in st["layers"].items()},
        "samples": st["samples"],
        "updated": st["updated"],
        "decay_day": st["decay_day"],
    }
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(slim, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, p)
    _DIRTY[cam_id] = 0
    _LAST_SAVE[cam_id] = time.time()


def grid_for(cam_id: str, layer: str = "person",
             daypart: str | None = None,
             root: Path | None = None) -> list[list[float]]:
    """The accumulated grid; dayparts summed when `daypart` is None."""
    st = _load(cam_id, root)
    layer_grids = st["layers"].get(layer) or {}
    if daypart:
        return [row[:] for row in
                (layer_grids.get(daypart) or _blank_state()["layers"]["person"]["night"])]
    out = [[0.0] * GRID_W for _ in range(GRID_H)]
    for grid in layer_grids.values():
        for y, row in enumerate(grid):
            for x, v in enumerate(row):
                out[y][x] += v
    return out


def stats(cam_id: str, root: Path | None = None) -> dict:
    st = _load(cam_id, root)
    covered = total = 0.0
    peak = 0.0
    for layer in st["layers"].values():
        for grid in layer.values():
            for row in grid:
                for v in row:
                    total += v
                    if v > peak:
                        peak = v
    for y in range(GRID_H):
        for x in range(GRID_W):
            if any(layer[dp][y][x] > 0
                   for layer in st["layers"].values() for dp in layer):
                covered += 1
    return {
        "cam_id": cam_id,
        "samples": st["samples"],
        "updated": st["updated"],
        "total_weight_s": round(total, 1),
        "peak_cell_s": round(peak, 1),
        "coverage_frac": round(covered / (GRID_W * GRID_H), 3),
    }


def render_due(cam_id: str) -> bool:
    """True every RENDER_EVERY_SAMPLES-th accumulated sample (and on the
    very first), so the collector refreshes the published overlay without
    tracking its own cadence."""
    st = _STATE.get(cam_id)
    if st is None:
        return False
    n = st["samples"]
    return n == 1 or (n % RENDER_EVERY_SAMPLES == 0)


def render(cam_id: str, base_frame=None, layer: str = "person",
           daypart: str | None = None, alpha: float = 0.45,
           size: tuple[int, int] = (640, 360),
           root: Path | None = None):
    """Colormap overlay of a camera's accumulated map (BGR ndarray).

    Thin wrapper over overlay(): loads this camera's persisted grid and
    hands it to the shared renderer.
    """
    return overlay(grid_for(cam_id, layer=layer, daypart=daypart),
                   base_frame=base_frame, alpha=alpha, size=size)


def overlay(grid, base_frame=None, alpha: float = 0.45,
            size: tuple[int, int] = (640, 360)):
    """Colormap overlay of ANY dwell grid (GRID_H rows x GRID_W cols of
    floats) - the renderer shared by render() (persisted per-camera maps)
    and the live-analysis heat layer (session-local accumulation).

    `base_frame` (BGR) gives the overlay its scene context; without one
    the map renders on a dark canvas at `size`. cv2/numpy import lives
    here so the accumulation path stays dependency-free.
    """
    import cv2
    import numpy as np

    grid = np.asarray(grid, dtype=np.float32)
    if base_frame is not None:
        H, W = base_frame.shape[:2]
        canvas = base_frame.copy()
    else:
        W, H = size
        canvas = np.full((H, W, 3), 24, dtype=np.uint8)
    peak = float(grid.max())
    if peak <= 0:
        return canvas
    # sqrt tone-curve: without it one bus stop's dwell peak crushes every
    # walking route into invisibility.
    norm = np.sqrt(grid / peak)
    heat = cv2.resize(norm, (W, H), interpolation=cv2.INTER_LINEAR)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=max(2.0, W / 96.0))
    m = float(heat.max())
    if m > 0:
        heat /= m
    colored = cv2.applyColorMap((heat * 255).astype(np.uint8),
                                cv2.COLORMAP_TURBO)
    # Blend only where there is signal - the empty street stays a photo.
    mask = (heat[..., None] * alpha)
    out = (canvas.astype(np.float32) * (1 - mask)
           + colored.astype(np.float32) * mask)
    return out.astype(np.uint8)


def save_all(root: Path | None = None) -> None:
    """Flush every camera's in-memory grid (collector shutdown path)."""
    for cam_id in list(_STATE):
        try:
            save(cam_id, root)
        except Exception:
            pass


def reset(cam_id: str | None = None) -> None:
    """Test/maintenance helper: forget in-memory state (all cams when None)."""
    if cam_id is None:
        _STATE.clear()
        _LAST_TS.clear()
        _DIRTY.clear()
        _LAST_SAVE.clear()
        return
    _STATE.pop(cam_id, None)
    _LAST_TS.pop(cam_id, None)
    _DIRTY.pop(cam_id, None)
    _LAST_SAVE.pop(cam_id, None)
