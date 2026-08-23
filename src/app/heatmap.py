"""Thermal-style overlay helper for the live Heat layer.

overlay_thermal() recolors a frame with a heat colormap driven by the
live session's in-memory activity grid; GRID_W/GRID_H define that grid's
resolution (shared with live_analysis and behavior). The long-horizon
persistent heatmap subsystem that used to live here (per-camera JSON
state, dayparts, decay, disk renders) was removed in the 2026-08-23
cleanup - the dashboard never called it.
"""
from __future__ import annotations



# Grid resolution. 48x27 (16:9) - one cell is ~40px at 1080p, tight enough
# to separate two adjacent shop fronts that the previous 32x18 (60px cells)
# blurred into one hot blob, while a full state JSON stays ~200 KB. The
# _load shape guard restarts any 32x18 file cleanly (decay half-life is
# ~3 weeks anyway, so the map re-forms quickly).
GRID_W, GRID_H = 48, 27


# cam_id -> state dict (lazy-loaded from disk). State shape:
#   {"layers": {layer: {daypart: [GRID_H rows][GRID_W cols] floats}},
#    "samples": int, "updated": epoch, "decay_day": "YYYY-MM-DD"}
_STATE: dict[str, dict] = {}
_LAST_TS: dict[str, float] = {}
_DIRTY: dict[str, int] = {}
_LAST_SAVE: dict[str, float] = {}


def overlay_thermal(grid, base_frame, alpha_heat: float = 0.75):
    """FULL-FRAME thermal recolor of `base_frame`, with the dwell grid
    driving the "hot" spots on top.

    This is the operator-requested look (2026-08-17): the entire frame
    is recoloured through the INFERNO colormap so the whole scene reads
    as a thermal image (dark blue where nothing / cold, warm cyan where
    the frame's own luminance is bright). Where the dwell grid has
    signal, the pixels are overwritten with the TURBO hot palette
    (yellow -> orange -> red) so activity pops as glowing tiles / blobs
    exactly like the reference thermal-camera images.

    Unlike overlay(), this recolours the ENTIRE frame - even empty
    areas turn thermal-blue instead of staying a plain photo.
    """
    import cv2
    import numpy as np

    if base_frame is None:
        return None
    H, W = base_frame.shape[:2]
    # 1. Luminance base -> INFERNO colormap for the whole scene.
    #    INFERNO's cold end is deep purple / near-black; warm end goes
    #    to bright yellow. Feels like a proper thermal-camera background.
    gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
    # Gentle contrast lift so a low-light night frame does not collapse
    # into one dark blue smear.
    gray = cv2.equalizeHist(gray)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    # 2. Dwell grid -> TURBO hot palette on the same frame size.
    grid = np.asarray(grid, dtype=np.float32)
    peak = float(grid.max())
    if peak <= 0:
        # No activity yet - return the pure thermal recolour so the
        # operator sees IMMEDIATELY the layer is running (was: bare
        # photo, indistinguishable from "not working").
        return thermal
    norm = np.sqrt(grid / peak)
    heat = cv2.resize(norm, (W, H), interpolation=cv2.INTER_LINEAR)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=max(2.0, W / 96.0))
    m = float(heat.max())
    if m > 0:
        heat /= m
    hot = cv2.applyColorMap((heat * 255).astype(np.uint8),
                            cv2.COLORMAP_TURBO)
    # 3. Composite: TURBO hotspots override INFERNO base where signal
    #    is present. alpha_heat controls how strongly hot pixels win;
    #    at 0.75 they are dominant but the INFERNO underneath still
    #    leaks through for a smooth transition at blob edges.
    mask = (heat[..., None] * alpha_heat)
    out = (thermal.astype(np.float32) * (1 - mask)
           + hot.astype(np.float32) * mask)
    return out.astype(np.uint8)


