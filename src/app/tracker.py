"""Short-window multi-object tracker: a stable id per INDIVIDUAL in a burst.

Why position and motion, not appearance: fifty pigeons - or fifty
pedestrians in near-identical traditional dress - defeat appearance
embeddings BY CONSTRUCTION. There is no visual signature separating
individual 17 from individual 23, so reid.py (which answers "have I seen
this look before?") cannot tell them apart and was never meant to. What
does separate them is WHERE each body is and HOW it moves: two objects
cannot occupy the same spot, and each one's next position is predictable
from its last few. This tracker therefore matches purely on predicted
position:

  * constant-velocity prediction - each track carries an EMA-smoothed
    per-axis velocity; its centroid is extrapolated to the new frame's
    timestamp BEFORE matching. Two neighbors walking toward each other
    stay themselves instead of swapping ids the way plain nearest-centroid
    matching (track_burst) lets them;
  * two-stage association - confident detections claim tracks first; the
    leftovers (detections that barely cleared their class gate) may only
    EXTEND existing tracks, never START one. A half-occluded pedestrian
    flickering around the gate keeps one id instead of spawning a fresh
    id every frame, while gate-hugging noise cannot mint phantom
    individuals;
  * coasting - an unmatched track survives `max_misses` frames on its
    predicted path (a bus passing in front) and re-claims the object when
    it reappears; only then does the id retire.

Scope is ONE observation window (a 3-16 frame burst). Ids are stable
within the window; the collector's bursts are separated by tens of
seconds of unobserved time, and carrying identity across such gaps is an
appearance problem (reid.py's job), impossible for identical-looking
individuals - no honest system claims pigeon #17 is back after leaving
the frame. Pure python, no numpy - the whole thing is a few hundred
comparisons per frame.
"""
from __future__ import annotations

# Detections at or above this confidence may START a track (BYTE's "high"
# band). Below it they can only extend one. 0.45 sits above every default
# class gate (0.22-0.35) so the band is meaningful for all classes.
TRACK_HIGH_CONF = 0.45
# Stage-1 matching budget as a fraction of the frame diagonal - how far a
# confident detection may sit from a track's PREDICTED centroid. Matches
# the speed pass's 0.30 (vehicles at speed move a lot between ~1s frames).
TRACK_MATCH_FRAC = 0.30
# Stage-2 budget for gate-hugging detections: half the stage-1 radius.
# A low-confidence box is noisy evidence - only accept it where the track
# already expects to be.
TRACK_LOW_MATCH_FRAC = 0.15
# How many consecutive unmatched frames a track coasts before retiring.
TRACK_MAX_MISSES = 2
# EMA weight of the NEWEST velocity sample. High = reactive (short windows
# have few samples to average); the remainder keeps enough memory to damp
# a single bad match.
TRACK_VEL_SMOOTH = 0.6


def _centroid(b: dict) -> tuple[float, float]:
    return (b["x1"] + b["x2"]) / 2.0, (b["y1"] + b["y2"]) / 2.0


class Track:
    """One individual: its box history and motion state."""

    __slots__ = ("tid", "cls", "boxes", "times", "vx", "vy", "misses", "hits")

    def __init__(self, tid: int, box: dict, t: float):
        self.tid = tid
        self.cls = box.get("cls")
        self.boxes: list[dict] = [box]
        self.times: list[float] = [t]
        self.vx = 0.0
        self.vy = 0.0
        self.misses = 0
        self.hits = 1

    def predicted_centroid(self, t: float) -> tuple[float, float]:
        cx, cy = _centroid(self.boxes[-1])
        dt = t - self.times[-1]
        return cx + self.vx * dt, cy + self.vy * dt

    def add(self, box: dict, t: float) -> None:
        px, py = _centroid(self.boxes[-1])
        cx, cy = _centroid(box)
        dt = t - self.times[-1]
        if dt > 0:
            nvx, nvy = (cx - px) / dt, (cy - py) / dt
            if self.hits == 1:
                self.vx, self.vy = nvx, nvy
            else:
                s = TRACK_VEL_SMOOTH
                self.vx = s * nvx + (1 - s) * self.vx
                self.vy = s * nvy + (1 - s) * self.vy
        self.boxes.append(box)
        self.times.append(t)
        self.misses = 0
        self.hits += 1


class BurstTracker:
    """Feed one frame's (already gated + ROI-filtered) boxes at a time.

    Every matched or newly-tracked box dict gains a `track_id` key IN
    PLACE - the same shared-dict convention the speed estimator uses for
    `kmh`, so whichever frame the collector annotates shows the id.
    """

    def __init__(self, frame_shape,
                 high_conf: float = TRACK_HIGH_CONF,
                 match_frac: float = TRACK_MATCH_FRAC,
                 low_match_frac: float = TRACK_LOW_MATCH_FRAC,
                 max_misses: int = TRACK_MAX_MISSES):
        H, W = frame_shape[:2]
        self._diag = (H * H + W * W) ** 0.5
        self.high_conf = high_conf
        self.budget_high = match_frac * self._diag
        self.budget_low = low_match_frac * self._diag
        self.max_misses = max_misses
        self.open: list[Track] = []
        self.done: list[Track] = []
        self._next_id = 1

    # -- association ------------------------------------------------------

    def _greedy_match(self, dets: list[dict], t: float, budget: float,
                      taken_tracks: set[int], taken_dets: set[int]) -> None:
        """Greedily pair detections with open tracks by distance from each
        track's PREDICTED centroid. Same-class pairs only. Mutates the
        `taken_*` sets and extends matched tracks."""
        cands: list[tuple[float, int, int]] = []
        for ti, tr in enumerate(self.open):
            if ti in taken_tracks:
                continue
            pcx, pcy = tr.predicted_centroid(t)
            for di, b in enumerate(dets):
                if di in taken_dets or b.get("cls") != tr.cls:
                    continue
                cx, cy = _centroid(b)
                d = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
                if d <= budget:
                    cands.append((d, ti, di))
        cands.sort()
        for d, ti, di in cands:
            if ti in taken_tracks or di in taken_dets:
                continue
            taken_tracks.add(ti)
            taken_dets.add(di)
            tr = self.open[ti]
            tr.add(dets[di], t)
            dets[di]["track_id"] = tr.tid

    def update(self, boxes: list[dict], t: float) -> None:
        high = [b for b in boxes
                if float(b.get("conf") or 0.0) >= self.high_conf]
        low = [b for b in boxes
               if float(b.get("conf") or 0.0) < self.high_conf]

        taken_tracks: set[int] = set()
        # Stage 1: confident detections claim tracks (wide budget).
        used_high: set[int] = set()
        self._greedy_match(high, t, self.budget_high, taken_tracks, used_high)
        # Stage 2: gate-hugging detections may extend LEFTOVER tracks only,
        # inside the tighter radius. They never mint a new id.
        used_low: set[int] = set()
        self._greedy_match(low, t, self.budget_low, taken_tracks, used_low)

        # Unmatched confident detections become new individuals.
        for di, b in enumerate(high):
            if di in used_high:
                continue
            tr = Track(self._next_id, b, t)
            self._next_id += 1
            b["track_id"] = tr.tid
            self.open.append(tr)
            taken_tracks.add(len(self.open) - 1)  # freshly-born: not a miss

        # Unmatched open tracks coast; too many misses retires them.
        still_open: list[Track] = []
        for ti, tr in enumerate(self.open):
            if ti not in taken_tracks:
                tr.misses += 1
                if tr.misses > self.max_misses:
                    self.done.append(tr)
                    continue
            still_open.append(tr)
        self.open = still_open

    def close(self) -> list[Track]:
        """End the window: retire every open track, return ALL tracks in
        id order (id order == birth order)."""
        self.done.extend(self.open)
        self.open = []
        self.done.sort(key=lambda tr: tr.tid)
        return self.done


def assign_burst_ids(frames_boxes: list[list[dict]], frame_shape,
                     dt: float = 1.0,
                     times: list[float] | None = None) -> list[Track]:
    """Run a fresh tracker over a whole burst's per-frame box lists.

    Mutates the box dicts in place (adds `track_id`); returns the track
    list. `times` overrides the uniform `dt` spacing when the caller knows
    real timestamps (behavior.py's deep window does).
    """
    if not frames_boxes:
        return []
    if times is None:
        times = [i * dt for i in range(len(frames_boxes))]
    trk = BurstTracker(frame_shape)
    for t, boxes in zip(times, frames_boxes):
        trk.update(boxes, t)
    return trk.close()
