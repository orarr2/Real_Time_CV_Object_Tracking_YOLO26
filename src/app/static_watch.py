"""Static-object watch: things that SETTLED in place, then vanished.

The mirror image of presence.py. The loiter path deliberately REFUSES
perfectly static stays (LOITER_MAX_STATIC_IOU - a box that never moved is
a kiosk, not a loiterer), so the information "this object has been parked
here for 20 minutes" was being accumulated and then thrown away. This
module keeps it, and answers the question the loiter path can't: **when
did the static thing LEAVE?** A parked car pulling out, a market stall
packing up, a bag that sat by a wall and is suddenly gone - each becomes a
`static_departed` event carrying how long the object stayed and the crop
captured while it was still there.

Life cycle of an ANCHOR:
  candidate  - a detection with no matching anchor starts one;
  (grows)    - a same-class detection overlapping >= `match_iou` continues
               it: hits += 1, box drifts slowly (EMA) so slight jitter or
               a lighting shift doesn't shed the anchor;
  settled    - stay >= `min_stay_sec` AND >= `min_hits` sightings AND
               median confidence clears the class's UN-boosted default
               gate (same evidence floor as loiter alerts - a conf-0.23
               "car" that only exists on a loosened gate is a shadow, not
               a vehicle). At settle time a crop is captured - the LAST
               look at the object, available later when it's gone;
  departed   - a SETTLED anchor unmatched for `depart_misses` consecutive
               successful samples emits the event and retires. Candidates
               that fizzle just evaporate.

Honesty guards (each one maps to a real failure mode):
  * misses only count on samples that actually RAN - the collector calls
    observe() on successful grabs only, so a stream outage can never fake
    a departure (the anchor just waits; `stale_sec` eventually clears
    anchors whose camera left the grid for hours);
  * a DARK frame (luma < dark_luma) skips miss-counting - losing sight of
    everything at night is lighting, not departure;
  * a SCENE WIPE - most settled anchors unmatched at once - skips
    miss-counting too: the camera moved, refocused, or switched source;
    one object leaving is an event, all of them "leaving" is a cut.
"""
from __future__ import annotations

import base64
import time

from app.detect_core import box_iou

STATIC_MIN_STAY_SEC = 300.0   # the operator's "more than five minutes"
STATIC_MATCH_IOU = 0.55       # continuity: static boxes jitter a few px
STATIC_MIN_HITS = 4           # sightings before a stay can settle
STATIC_DEPART_MISSES = 2      # consecutive observed samples without it
STATIC_EMA = 0.9              # old-box weight when drifting the anchor
STATIC_STALE_SEC = 2 * 3600.0 # drop anchors idle this long (camera left)
STATIC_MAX_ANCHORS = 60       # per camera - a packed lot stays bounded
SCENE_WIPE_MIN = 2            # a wipe needs at least this many settled...
SCENE_WIPE_FRAC = 0.5         # ...and >= this fraction gone at once
DARK_LUMA = 40.0              # below this the frame can't testify
# Unattended-object gate (fix1-A1): a bag-class anchor that settles emits
# an event ONLY when no person stands near it - a bag at its owner's feet
# is luggage, not a threat. "Near" = a person foot point within this
# multiple of the bag's larger dimension from the bag's center.
UNATTENDED_PERSON_DIST_SCALE = 2.0
# Scene frames captured at settle time (fix1-A6) are memory-only evidence
# (settled anchors are bounded by STATIC_MAX_ANCHORS); downscale to keep
# the worst case a few MB per camera.
BEFORE_FRAME_MAX_W = 960


class StaticWatch:
    """Per-camera anchor registry. Feed every successful sample's boxes."""

    def __init__(self,
                 min_stay_sec: float = STATIC_MIN_STAY_SEC,
                 match_iou: float = STATIC_MATCH_IOU,
                 min_hits: int = STATIC_MIN_HITS,
                 depart_misses: int = STATIC_DEPART_MISSES,
                 evidence_gates: dict | None = None,
                 dark_luma: float = DARK_LUMA,
                 max_anchors: int = STATIC_MAX_ANCHORS,
                 skip_classes: tuple = ("person",),
                 unattended_classes: tuple = ()):
        self.min_stay_sec = min_stay_sec
        self.match_iou = match_iou
        self.min_hits = min_hits
        self.depart_misses = depart_misses
        # cls -> conf floor a stay must clear (median) to ever settle.
        # None skips the check (tests); the collector passes detect_core's
        # DEFAULT_PER_CLASS_CONF - deliberately the UN-boosted defaults.
        self.evidence_gates = evidence_gates
        self.dark_luma = dark_luma
        self.max_anchors = max_anchors
        # Classes that never become anchors. "Static object LEFT" means an
        # unattended OBJECT (bag, stall, parked car) - a human sitting on a
        # bench for six minutes and then walking away is not one, yet the
        # 07.08 Bulancak digest carried 15 such person "departures" in one
        # afternoon. People-dwelling is the loiter path's job.
        self.skip_classes = tuple(skip_classes or ())
        # Classes whose SETTLING (not departing) is itself the event - an
        # unattended bag. Empty tuple = the pre-fix1 behavior, byte for byte.
        self.unattended_classes = tuple(unattended_classes or ())
        self._anchors: dict[str, list[dict]] = {}
        self._next_id = 1

    # -- internals ---------------------------------------------------------

    def _settle_ok(self, a: dict, now: float) -> bool:
        if a["settled"] or now - a["first_ts"] < self.min_stay_sec \
                or a["hits"] < self.min_hits:
            return False
        if self.evidence_gates is not None:
            confs = sorted(a["confs"])
            med = confs[len(confs) // 2] if confs else 0.0
            if med < float(self.evidence_gates.get(a["cls"], 0.35)):
                return False
        return True

    @staticmethod
    def _full_jpeg(frame, max_w: int = BEFORE_FRAME_MAX_W) -> bytes | None:
        """Downscaled full-scene JPEG captured at settle time - the
        "before" evidence a departure/unattended card shows (fix1-A6).
        Memory-only: not persisted to the analysis-state snapshot (a
        restart degrades the card to the old crop-only layout)."""
        if frame is None:
            return None
        try:
            import cv2
            H, W = frame.shape[:2]
            img = frame
            if W > max_w:
                img = cv2.resize(frame, (max_w, int(H * max_w / W)))
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
            return buf.tobytes() if ok else None
        except Exception:
            return None

    @staticmethod
    def _person_nearby(box: dict, person_boxes: list[dict] | None,
                       scale: float = UNATTENDED_PERSON_DIST_SCALE) -> bool:
        """Is any person's foot point within `scale` x the object's larger
        dimension of the object's center? Owner-nearby = not unattended."""
        if not person_boxes:
            return False
        cx = (box["x1"] + box["x2"]) / 2.0
        cy = (box["y1"] + box["y2"]) / 2.0
        reach = scale * max(box["x2"] - box["x1"], box["y2"] - box["y1"], 1.0)
        for p in person_boxes:
            fx = (float(p["x1"]) + float(p["x2"])) / 2.0
            fy = float(p["y2"])
            if ((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5 <= reach:
                return True
        return False

    @staticmethod
    def _crop_jpeg(frame, box: dict) -> bytes | None:
        """Encode the anchor's crop - the evidence shown when it departs.
        Best-effort: None on any failure (headless test envs pass frame=None)."""
        if frame is None:
            return None
        try:
            import cv2
            H, W = frame.shape[:2]
            x1 = max(0, int(box["x1"])); y1 = max(0, int(box["y1"]))
            x2 = min(W, int(box["x2"])); y2 = min(H, int(box["y2"]))
            if not (x2 > x1 and y2 > y1):
                return None
            ok, buf = cv2.imencode(".jpg", frame[y1:y2, x1:x2],
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buf.tobytes() if ok else None
        except Exception:
            return None

    # -- main entry --------------------------------------------------------

    def observe(self, cam_id: str, boxes: list[dict], frame_shape,
                luma: float | None = None, frame=None,
                now: float | None = None,
                person_boxes: list[dict] | None = None) -> list[dict]:
        """One successful sample. Returns event dicts - `static_departed`
        for settled anchors that vanished, `unattended_object` for
        bag-class anchors that settled with no person nearby
        (`person_boxes` = the sample's person detections, which observe
        itself never anchors)."""
        now = time.time() if now is None else now
        if self.skip_classes:
            boxes = [b for b in boxes
                     if b.get("cls") not in self.skip_classes]
        anchors = self._anchors.setdefault(cam_id, [])

        # Greedy same-class IoU matching, best overlap first - one
        # detection continues at most one anchor and vice versa.
        cands: list[tuple[float, int, int]] = []
        for ai, a in enumerate(anchors):
            for bi, b in enumerate(boxes):
                if b.get("cls") != a["cls"]:
                    continue
                iou = box_iou(a["box"], b)
                if iou >= self.match_iou:
                    cands.append((-iou, ai, bi))
        cands.sort()
        used_a: set[int] = set()
        used_b: set[int] = set()
        for _niou, ai, bi in cands:
            if ai in used_a or bi in used_b:
                continue
            used_a.add(ai)
            used_b.add(bi)
            a, b = anchors[ai], boxes[bi]
            w = STATIC_EMA
            a["box"] = {k: w * a["box"][k] + (1 - w) * float(b[k])
                        for k in ("x1", "y1", "x2", "y2")}
            a["last_ts"] = now
            a["hits"] += 1
            a["misses"] = 0
            confs = a["confs"]
            confs.append(float(b.get("conf") or 0.0))
            if len(confs) > 20:
                del confs[0]
            if self._settle_ok(a, now):
                a["settled"] = True
                a["settle_ts"] = now
                a["crop_jpeg"] = self._crop_jpeg(frame, a["box"])
                a["before_jpeg"] = self._full_jpeg(frame)
                if a["cls"] in self.unattended_classes:
                    a["unattended_pending"] = True

        # New candidates from unmatched detections. Anchors born this very
        # sample sit at index >= n_before and are exempt from this sample's
        # miss accounting - being new is not being missed.
        n_before = len(anchors)
        for bi, b in enumerate(boxes):
            if bi in used_b:
                continue
            anchors.append({
                "id": self._next_id, "cls": b.get("cls"),
                "box": {k: float(b[k]) for k in ("x1", "y1", "x2", "y2")},
                "first_ts": now, "last_ts": now,
                "hits": 1, "misses": 0,
                "confs": [float(b.get("conf") or 0.0)],
                "settled": False, "settle_ts": None, "crop_jpeg": None,
            })
            self._next_id += 1

        events: list[dict] = []
        # Unattended objects (fix1-A1): a settled bag-class anchor fires
        # once, the first sample where NO person stands near it. While the
        # owner lingers the anchor stays pending; if the bag departs first,
        # the pending flag simply dies with it.
        for a in anchors:
            if not a.get("unattended_pending"):
                continue
            if self._person_nearby(a["box"], person_boxes):
                continue
            a["unattended_pending"] = False
            confs = sorted(a["confs"])
            events.append({
                "kind": "unattended_object",
                "cls": a["cls"],
                "anchor_id": a["id"],
                "box": dict(a["box"]),
                "dwell_sec": round(now - a["first_ts"], 1),
                "first_ts": a["first_ts"],
                "settle_ts": a["settle_ts"],
                "hits": a["hits"],
                "conf_median": confs[len(confs) // 2] if confs else 0.0,
                "crop_jpeg": a["crop_jpeg"],
                "before_jpeg": a.get("before_jpeg"),
            })

        # Miss accounting. A dark frame can't testify about absence.
        dark = luma is not None and luma < self.dark_luma
        settled_ids = [a["id"] for a in anchors if a["settled"]]
        settled_missed = [a["id"] for ai, a in enumerate(anchors[:n_before])
                          if a["settled"] and ai not in used_a]
        wipe = (len(settled_missed) >= SCENE_WIPE_MIN
                and len(settled_ids) > 0
                and len(settled_missed) / len(settled_ids) >= SCENE_WIPE_FRAC)
        if not dark and not wipe:
            survivors: list[dict] = []
            for ai, a in enumerate(anchors):
                if ai in used_a or ai >= n_before:
                    survivors.append(a)
                    continue
                a["misses"] += 1
                if a["settled"] and a["misses"] >= self.depart_misses:
                    confs = sorted(a["confs"])
                    events.append({
                        "kind": "static_departed",
                        "cls": a["cls"],
                        "anchor_id": a["id"],
                        "box": dict(a["box"]),
                        "dwell_sec": round(a["last_ts"] - a["first_ts"], 1),
                        "first_ts": a["first_ts"],
                        "last_ts": a["last_ts"],
                        "settle_ts": a["settle_ts"],
                        "hits": a["hits"],
                        "conf_median": (confs[len(confs) // 2]
                                        if confs else 0.0),
                        "crop_jpeg": a["crop_jpeg"],
                        "before_jpeg": a.get("before_jpeg"),
                    })
                    continue                       # departed: retire
                if not a["settled"] and a["misses"] >= self.depart_misses:
                    continue                       # fizzled candidate
                survivors.append(a)
            anchors[:] = survivors

        # Bound the registry: shed the weakest unsettled candidates first.
        if len(anchors) > self.max_anchors:
            anchors.sort(key=lambda a: (a["settled"], a["hits"]))
            del anchors[: len(anchors) - self.max_anchors]

        return events

    def prune(self, max_age_sec: float = STATIC_STALE_SEC,
              now: float | None = None) -> int:
        """Drop anchors not refreshed for `max_age_sec` (camera off-grid).
        Silent by design: an unobserved anchor proves nothing either way."""
        now = time.time() if now is None else now
        dropped = 0
        for cam_id, anchors in self._anchors.items():
            keep = [a for a in anchors if now - a["last_ts"] <= max_age_sec]
            dropped += len(anchors) - len(keep)
            self._anchors[cam_id] = keep
        return dropped

    # -- restart persistence -----------------------------------------------
    # _anchors and _next_id live in process memory; under Restart=always a
    # bounce wiped every anchor's age, which broke the loiter path's
    # furniture cross-check (settled_spot_age needs anchors OLDER than the
    # stay - both restarted from zero together). The collector snapshots
    # this each round and restores on boot. Crops ride along base64-encoded
    # (small ones only) so a post-restart departure still carries evidence.

    _STATE_CROP_MAX_B = 60_000

    def to_state(self) -> dict:
        cams: dict[str, list[dict]] = {}
        for cam_id, anchors in self._anchors.items():
            rows = []
            for a in anchors:
                d = dict(a)
                crop = d.pop("crop_jpeg", None)
                d.pop("before_jpeg", None)   # scene frames are memory-only
                if crop and len(crop) <= self._STATE_CROP_MAX_B:
                    d["crop_b64"] = base64.b64encode(crop).decode("ascii")
                rows.append(d)
            cams[cam_id] = rows
        return {"next_id": self._next_id, "anchors": cams}

    def load_state(self, state: dict, now: float | None = None,
                   max_age_sec: float = STATIC_STALE_SEC) -> int:
        """Restore anchors from to_state(). Returns how many were kept."""
        now = time.time() if now is None else now
        kept = 0
        for cam_id, rows in (state.get("anchors") or {}).items():
            keep: list[dict] = []
            for d in rows:
                try:
                    if now - float(d["last_ts"]) > max_age_sec:
                        continue
                    if d.get("cls") in self.skip_classes:
                        # An anchor class we no longer track (e.g. person):
                        # loading it would only produce a farewell-departure
                        # event burst once its detections stop matching.
                        continue
                    a = dict(d)
                    b64 = a.pop("crop_b64", None)
                    a["crop_jpeg"] = base64.b64decode(b64) if b64 else None
                    a["box"] = {k: float(a["box"][k])
                                for k in ("x1", "y1", "x2", "y2")}
                    keep.append(a)
                    kept += 1
                except (ValueError, TypeError, KeyError):
                    continue
            if keep:
                self._anchors[cam_id] = keep
        try:
            self._next_id = max(int(state.get("next_id") or 1), self._next_id)
        except (ValueError, TypeError):
            pass
        return kept

    def counts(self, cam_id: str | None = None) -> dict:
        """Observability: {"anchors": n, "settled": n} (one cam or all)."""
        pools = ([self._anchors.get(cam_id, [])] if cam_id is not None
                 else list(self._anchors.values()))
        n = sum(len(p) for p in pools)
        s = sum(1 for p in pools for a in p if a["settled"])
        return {"anchors": n, "settled": s}

    def settled_spot_age(self, cam_id: str, box: dict, cls: str | None,
                         min_iou: float = 0.40,
                         now: float | None = None) -> float | None:
        """Age (seconds) of the oldest SETTLED same-class anchor overlapping
        `box`, or None when no such anchor exists.

        This is the furniture question the loiter path needs answered
        (2026-07-24): a scene structure that YOLO keeps calling "car"/"bus"
        (the Taksim kiosk, the Eyup Sultan awning) survives the first-vs-
        current static-IoU gate because its box WANDERS slowly along the
        structure - continuity holds sample to sample while the stay's
        first box drifts out of view. What the wander cannot fake is
        HISTORY: the static watch's anchor (EMA box, so it follows the same
        slow drift) has been settled here since long before the loiter stay
        began. A genuinely parking car's anchor is born WITH its stay, so
        its age roughly equals the stay duration - the caller uses that
        margin to tell furniture from a fresh arrival."""
        now = time.time() if now is None else now
        best = None
        for a in self._anchors.get(cam_id, []):
            if not a["settled"] or a["cls"] != cls:
                continue
            if box_iou(a["box"], box) < min_iou:
                continue
            age = now - a["first_ts"]
            if best is None or age > best:
                best = age
        return best
