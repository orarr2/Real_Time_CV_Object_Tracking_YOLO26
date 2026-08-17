"""Auto-detect parking spots from live camera (no operator polygons).

When the parking layer starts on a camera that has no operator-drawn
parking polygons, this module bootstraps a set of spots by watching
where vehicles actually park. Emitted spots feed the same
draw_zones_layer + _zone_stats path the manual polygons use, so once
a spot is inferred every downstream feature (occupancy flip events,
occupied/free caption, dashboard toast) works with zero code changes.

Algorithm - deliberately simple, works on CPU with the detector we
already run:

  1. Every parking tick, iterate the tracker's open tracks. A track
     is "stationary" when its foot-point moved < STATIONARY_PX_PER_S
     pixels/second averaged over its recent history AND it's a
     vehicle class (car / motorcycle / bus / truck / bicycle).
  2. Bin each stationary vehicle's foot-point into a normalized 24x14
     grid over the frame. Increment the cell's hit count.
  3. Track TIME as well: the bootstrap runs for BOOTSTRAP_S seconds
     (default 180 = 3 minutes).
  4. When the bootstrap ends, promote every cell with >= MIN_HITS
     stationary observations to a spot. A spot is a single-cell
     rectangle centered on the cell (normalized to 0..1).
  5. Persist the inferred spots to data/auto_parking_<cam>.json so a
     restart reuses them.

Bootstrap progress is exposed via the caption ("auto-detect 45/180 s,
2 candidate cells") so the operator sees progress even before spots
land. Manual polygons always take priority - if the operator draws a
parking polygon the auto-inferred spots are ignored.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable


BOOTSTRAP_S = float(60 * 3)         # 3 minutes of observation
GRID_W = 24
GRID_H = 14
STATIONARY_PX_PER_S = 8.0           # tighter than "moving" - actually parked
MIN_HITS = 6                        # tick observations needed to promote a cell
MIN_STATIONARY_TICKS = 4            # per-track: needs 4+ consecutive stationary
                                    # ticks before it feeds the grid
CELL_MARGIN_FRAC = 0.03             # shrink cell rects a hair so adjacent
                                    # inferred spots don't touch


def _cam_state_path(cam_id: str) -> Path:
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_"
                   for ch in cam_id)
    return (Path(__file__).resolve().parent.parent
            / "data" / f"auto_parking_{safe}.json")


class AutoParkingBootstrap:
    """One instance per camera. Cheap to construct."""

    def __init__(self, cam_id: str):
        self.cam_id = cam_id
        self.started_at: float | None = None
        self.grid = [[0 for _ in range(GRID_W)] for _ in range(GRID_H)]
        # tid -> deque-like list of recent foot points + ticks-stationary
        self._per_track: dict[int, dict] = {}
        # Loaded from disk if a previous session already bootstrapped.
        self._loaded = self._load_from_disk()

    # ------------------------------------------------------------------
    # Public API called from live_analysis.py
    # ------------------------------------------------------------------

    def has_persisted_spots(self) -> bool:
        """True when data/auto_parking_<cam>.json already holds spots
        from a previous session - the bootstrap is skipped and the
        stored spots are used immediately."""
        return bool(self._loaded)

    def sample(self, tracks: Iterable, frame_shape: tuple, now: float) -> None:
        """Feed one tick's tracker.open into the grid. Cheap: pure
        Python, no numpy, ~microseconds per call."""
        if self._loaded:
            return  # nothing to sample - we already have spots
        if self.started_at is None:
            self.started_at = now
        H, W = frame_shape[:2]
        if not (H and W):
            return
        for tr in tracks:
            cls = getattr(tr, "cls", "")
            if cls not in ("car", "motorcycle", "bus", "truck", "bicycle"):
                continue
            if getattr(tr, "misses", 0):
                continue
            b = tr.boxes[-1]
            fx = (b["x1"] + b["x2"]) / 2.0
            fy = b["y2"]
            state = self._per_track.setdefault(
                tr.tid, {"last_pt": None, "last_t": None, "stationary": 0})
            prev_pt = state["last_pt"]
            prev_t = state["last_t"]
            if prev_pt is not None and prev_t is not None and now > prev_t:
                dx = fx - prev_pt[0]
                dy = fy - prev_pt[1]
                dist = (dx * dx + dy * dy) ** 0.5
                px_per_s = dist / max(1e-3, now - prev_t)
                if px_per_s < STATIONARY_PX_PER_S:
                    state["stationary"] += 1
                else:
                    state["stationary"] = 0
            state["last_pt"] = (fx, fy)
            state["last_t"] = now
            # Only feed the grid AFTER a track has been stationary
            # enough ticks - filters out slow-moving traffic.
            if state["stationary"] >= MIN_STATIONARY_TICKS:
                if 0 <= fx <= W and 0 <= fy <= H:
                    gx = min(GRID_W - 1, int(fx / W * GRID_W))
                    gy = min(GRID_H - 1, int(fy / H * GRID_H))
                    self.grid[gy][gx] += 1

    def status(self, now: float) -> dict:
        """Return {elapsed, remaining, candidate_cells, done, spot_count}
        so the caller can render a progress caption."""
        if self._loaded:
            return {
                "elapsed": 0.0, "remaining": 0.0,
                "candidate_cells": 0, "done": True,
                "spot_count": len(self._loaded),
            }
        elapsed = 0.0 if self.started_at is None else (now - self.started_at)
        remaining = max(0.0, BOOTSTRAP_S - elapsed)
        cand = sum(1 for row in self.grid for v in row if v >= MIN_HITS)
        return {
            "elapsed": elapsed, "remaining": remaining,
            "candidate_cells": cand, "done": elapsed >= BOOTSTRAP_S,
            "spot_count": 0,
        }

    def emerge(self, now: float) -> list[dict]:
        """After bootstrap window, promote hot cells to spots and
        persist. Returns the spots (zone-format dicts). Idempotent -
        second call returns the persisted list."""
        if self._loaded:
            return list(self._loaded)
        if self.started_at is None or now - self.started_at < BOOTSTRAP_S:
            return []
        spots = self._grid_to_spots()
        if spots:
            self._save_to_disk(spots)
            self._loaded = spots
        return list(spots)

    def to_zones(self, now: float) -> list[dict]:
        """Zone-format list (kind=parking) ready to merge into
        LiveSession.zones. Empty until bootstrap finishes."""
        return self.emerge(now)

    def reset(self) -> None:
        """Wipe both the in-memory bootstrap AND the on-disk cache -
        so the next tick starts a fresh 3-minute observation. Used by
        the operator's 'reset auto-parking' control (if wired)."""
        self.started_at = None
        self.grid = [[0 for _ in range(GRID_W)] for _ in range(GRID_H)]
        self._per_track.clear()
        self._loaded = []
        p = _cam_state_path(self.cam_id)
        try:
            p.unlink()
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _grid_to_spots(self) -> list[dict]:
        """Every cell with >= MIN_HITS becomes a single-cell rectangle.
        A future pass could merge adjacent hot cells into one bigger
        polygon - for a first cut, single-cell spots read cleanly on
        the overlay and match the tightness of an operator-drawn
        square spot."""
        spots = []
        cw = 1.0 / GRID_W
        ch = 1.0 / GRID_H
        m = CELL_MARGIN_FRAC
        idx = 0
        for gy in range(GRID_H):
            for gx in range(GRID_W):
                if self.grid[gy][gx] < MIN_HITS:
                    continue
                x0 = gx * cw + cw * m
                x1 = (gx + 1) * cw - cw * m
                y0 = gy * ch + ch * m
                y1 = (gy + 1) * ch - ch * m
                idx += 1
                spots.append({
                    "kind": "parking",
                    "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                    "name": f"auto-{idx}",
                    # marker so downstream code / caption can tell
                    # auto-inferred spots from operator-drawn ones
                    "auto": True,
                })
        return spots

    def _save_to_disk(self, spots: list[dict]) -> None:
        p = _cam_state_path(self.cam_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cam_id": self.cam_id,
            "bootstrap_s": BOOTSTRAP_S,
            "grid_w": GRID_W, "grid_h": GRID_H,
            "min_hits": MIN_HITS,
            "created_at": time.time(),
            "spots": spots,
        }
        p.write_text(json.dumps(payload, indent=1))

    def _load_from_disk(self) -> list[dict]:
        p = _cam_state_path(self.cam_id)
        if not p.exists():
            return []
        try:
            data = json.loads(p.read_text())
        except (OSError, ValueError):
            return []
        return list(data.get("spots") or [])
