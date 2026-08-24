"""Live advanced-analysis engine for the private dashboard (fix 2).

The fix1-B picker returned ONE static analyzed frame (and aimed it at the
CLOUD camera paired with the tile, not the camera the operator was
watching). The fix 2 requirement replaces that completely: analysis is
LIVE and CONTINUOUS on the exact camera whose tile was clicked - the tile
morphs in place into a stream of analyzed frames while the VM collector
runs untouched. This module owns that:

  * one LiveSession per camera (max MAX_SESSIONS = the four grid tiles),
    holding the SAME stream the tile plays (registry camera or a
    local-picker slot resolved from web/local_grid.json), pacing one
    detection tick roughly every TICK_TARGET_S;
  * ONE analysis layer per session - the fix 2 semantics: a single layer
    per camera, up to four live analyses across the grid, duplicates
    fine. Switching the layer MUTATES the running session: the stream,
    the tracker and every accumulator survive, so heat -> gestures ->
    heat resumes the accumulated map instead of restarting;
  * per-layer rendering that draws ONLY that layer's semantics (pose =
    skeletons on close-enough people, never detection boxes) and says
    so honestly when a layer finds nothing ("none detected right now");
  * a latest-JPEG buffer the dashboard polls (~1/s). The client never
    touches the model directly and the VM is never involved.

Compute reality on an operator PC (CPU): one active session runs about
1-2 fps; four concurrent sessions about 0.3-0.5 fps each - INFER_LOCK
serializes model access so four sessions degrade gracefully instead of
thrashing the same weights from four threads.

The draw_* functions are pure (frame + data in, frame out) - since
fix 3 removed the one-shot layers branch, this module is the ONLY place
a layer's look is defined.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from app.heatmap import GRID_H, GRID_W

_SRC_ROOT = Path(__file__).resolve().parent.parent

# The analysis layers an operator can run live. "line" is the threshold-
# crossing layer added in fix 2. "fire" (2026-08-16) replaces the removed
# "loiter" layer: the notebook's anomaly-detection already covers dwell-
# in-a-zone (Turkey heritage: >5 min presence -> cropped save), so the
# live-dashboard loiter card was redundant. Fire detection uses a
# dedicated single-class YOLO placed at src/yolo_fire.pt (see
# load_fire_model below) and gracefully reports "model not loaded" when
# the weights file is absent, so a running session is never killed by a
# missing optional model.
LIVE_LAYERS = ("paths", "pose", "gestures", "body", "faces", "heat",
               "line", "fire", "parking", "plates")
DEFAULT_LOITER_DWELL_S = 30.0
_VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle", "bicycle")
LAYER_TITLES = {
    "paths":    "Paths & speeds",
    "pose":     "Pose & skeleton",
    "gestures": "Hand gestures",
    "body":     "Body anomalies",
    "faces":    "Face detection",
    "heat":     "Heat signature",
    "line":     "Line crossing",
    "fire":     "Fire detection",
    "plates":   "License plates (LPR)",
}

# Fire-detection layer (single-class YOLO on the current frame). The
# operator drops any compatible fire/smoke YOLO weights at FIRE_MODEL_PATH
# and the layer picks them up on the next start; if the file is missing
# the layer publishes a friendly "model not loaded" caption instead of
# crashing the session. Two consecutive positive ticks are required to
# raise the "FIRE DETECTED" banner (single-tick hits often flicker on
# bright headlights or window reflections).
FIRE_MODEL_PATH = _SRC_ROOT / "yolo_fire.pt"
FIRE_CONF = 0.35
FIRE_IMGSZ = 512
_FIRE_MODEL_CACHE: dict = {"loaded": False, "model": None, "err": None}
_FIRE_MODEL_LOCK = threading.Lock()

MAX_SESSIONS = 1          # single-camera design: only one analysis at a time
IDLE_STOP_S = 60.0        # no client poll this long -> session shuts down
TICK_TARGET_S = 0.8       # pacing floor between inference ticks
LIVE_IMGSZ = 512          # was 640; smaller shortens tick ~30-40% on CPU
                          # so box extrapolation stays within EXTRAP window

# ---- overlay display filters (2026-08 accuracy pass) ----------------------
# Raw single-frame detections flicker: one-tick ghosts, low-conf floaters,
# and COCO classes that make no sense on a street cam ("train" on a fence).
# The overlay publishes tracker-CONFIRMED objects instead - seen on at
# least DISPLAY_MIN_HITS ticks, recent conf at or above DISPLAY_MIN_CONF,
# class not blacklisted. The analytics accumulators (heat, crossings,
# counts) still consume every raw detection - display strictness must not
# starve the statistics.
DISPLAY_MIN_HITS = 1       # was 2; at 12-15s per tick a walker crossing
                           # the frame in ~10s never got a second hit and
                           # was invisible - operator saw "people walked by,
                           # no boxes at all" (audit 2026-08-15). One high-
                           # conf hit is enough to draw; the tracker still
                           # graduates it to full status on the next match.
DISPLAY_MIN_CONF = 0.32    # was 0.40; night street scenes carry a wide
                           # confidence range and the 0.40 floor cost the
                           # tail (visible pedestrians at 0.33-0.39). The
                           # tracker's two-stage association still cleans
                           # false positives before the display gate.
DISPLAY_MAX_MISSES = 1     # allow 1-tick coasting through brief occlusion
# 2026-08-17: train removed from the display blacklist - the operator
# now opts trains INTO the live detector via LIVE_CLASSES (see below),
# and the line layer counts them as a first-class channel. boat and
# airplane stay out because the street cams looking at rail/road never
# actually see them, so any hit is a class-confused hallucination.
DISPLAY_CLASS_BLACKLIST = {"boat", "airplane"}
# Below this person-box height (px) skeletons are guesswork, so kps are
# neither PUBLISHED nor COMPUTED: the pose pass crops only boxes at least
# this tall. One constant keeps compute aligned with the display gate -
# pose on a person whose skeleton would be hidden anyway is pure waste
# (measured: a full 5-min window on a far-field cam ran the pose model
# every tick with zero displayable skeletons).
KPS_MIN_BOX_H = 96
# Crowded-frame bound for the per-crop pose pass (tallest boxes win).
POSE_MAX_CROPS = 6

# Live-analysis detector envelope (2026-08 industry pass):
# - LIVE_CLASSES excludes COCO boat/airplane at the DETECTOR so
#   a wall can never become a boat pre-NMS (street scenes only; the
#   collector's counting path keeps its own class set untouched).
# - Animal COCO classes (14 bird, 15 cat, 16 dog, 17 horse, 18 sheep,
#   19 cow) added 2026-08-17: line-crossing layer must count animals
#   too (operator report). All six surface under the single "animal"
#   label via detect_core.NAME_BY_ID - one bucket rather than six
#   sparse ones. Extra classes are harmless to every other layer
#   (pose/gestures/body already person-only; plates already
#   vehicle-only; heat/paths/loiter/parking are class-agnostic).
# - train (6) added 2026-08-17: rail-facing cameras were losing tram
#   crossings entirely; the class ships under the "train" label and
#   is out of the display blacklist above.
# - The model floor drops to 0.12 and the per-class gates are scaled by
#   LIVE_GATE_SCALE so gate-hugging blurred pedestrians survive into the
#   tracker, whose ByteTrack-style second stage may extend existing
#   tracks with them (never mint new ones); DISPLAY_MIN_CONF still rules
#   what the operator sees.
# - agnostic NMS collapses car/truck double-boxes on one vehicle.
LIVE_CLASSES = [0, 1, 2, 3, 5, 6, 7, 14, 15, 16, 17, 18, 19]
LIVE_CONF_FLOOR = 0.12
LIVE_GATE_SCALE = 0.7
# Night profile: mean-gray below NIGHT_LUMA turns on CLAHE (the classical
# enhancer with the most consistent night-detection gains) ahead of
# inference. Checked with hysteresis so a passing headlight doesn't
# flip the profile every tick.
NIGHT_LUMA_ON = 65.0
NIGHT_LUMA_OFF = 80.0
HEAT_HALF_LIFE_S = 180.0   # dwell-heat half-life (recent-activity view)

# cam_id -> seconds between wall clock and the stream's PROGRAM-DATE-TIME
# live edge. Decision 21 (2026-08-23): populated by _measure_pdt_offset,
# fired in a background thread whenever get_shared_reader builds a fresh
# reader - the one place the resolved manifest URL is actually known.
# _publish_data subtracts it from cap_ts so the overlay timestamp lands
# in the video's own clock instead of the backend's. Empty entries keep
# the read-side default of 0.0 (no correction).
STREAM_PDT_OFFSET: dict[str, float] = {}


def _measure_pdt_offset(url: str, key: str) -> None:
    """Fill STREAM_PDT_OFFSET[key] = now - newest EXT-X-PROGRAM-DATE-TIME
    (the wall-clock age of the stream's live edge). Follows one variant
    hop when handed a master playlist. Every failure leaves the map
    untouched - the read side then applies no correction, exactly the
    pre-implementation behavior."""
    import re as _re
    import datetime as _dt
    import urllib.request
    import urllib.parse
    try:
        hdrs = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=hdrs)
        text = urllib.request.urlopen(req, timeout=4).read().decode(
            "utf-8", "replace")
        if ("#EXT-X-PROGRAM-DATE-TIME" not in text
                and "#EXT-X-STREAM-INF" in text):
            for line in text.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    sub = urllib.parse.urljoin(url, line)
                    req = urllib.request.Request(sub, headers=hdrs)
                    text = urllib.request.urlopen(req, timeout=4).read() \
                        .decode("utf-8", "replace")
                    break
        pdts = _re.findall(r"#EXT-X-PROGRAM-DATE-TIME:(\S+)", text)
        if not pdts:
            return
        ts = _dt.datetime.fromisoformat(
            pdts[-1].replace("Z", "+00:00")).timestamp()
        off = time.time() - ts
        # Sanity band: a negative offset is clock skew, >60s is a stale
        # manifest - both are worse than no correction at all.
        if 0.0 <= off <= 60.0:
            STREAM_PDT_OFFSET[key] = off
            print(f"live-analysis: PDT offset {key} = {off:.1f}s")
    except Exception:
        pass
JPEG_MAX_W = 960
JPEG_QUALITY = 80
# Replay ring (decision D3, 2026-08-23, scoped small on purpose): the
# last REPLAY_RING_S seconds of published (annotated) frames at
# REPLAY_FPS, so a gallery/strip event can be replayed in context.
# 15 s x 2 fps x ~60 KB JPEG = ~2 MB per session - safe on the 8 GB
# host. Raw pre-annotation frames are NOT kept; the replay shows what
# the operator's live view showed.
REPLAY_RING_S = 15.0
REPLAY_FPS = 2.0
REPLAY_RING_FRAMES = int(REPLAY_RING_S * REPLAY_FPS)
TRACK_KEEP = 48           # per-track box history cap (live runs are open-ended)
GRAB_FAIL_REFRESH = 3     # consecutive grab failures before re-resolving
# Parking-spot probe: parked two-wheelers at night rarely clear the
# tracker's confirmation gates (audit 2026-08-14: spots visibly full read
# "0/2 occupied"), so the parking layer additionally re-detects each spot
# on a 2x-upscaled crop of the spot itself every PARKING_PROBE_EVERY_S.
# A fresh hit feeds the same per-spot hysteresis as a track candidate.
PARKING_PROBE_EVERY_S = 12.0
PARKING_PROBE_FRESH_S = 30.0
PARKING_PROBE_CONF = 0.30

# Default counting line for cameras without a configured "line" (the local
# picker's cameras): horizontal, at 62% height - the sidewalk band on a
# typical street view. Same normalized [[x,y],[x,y]] convention as
# cameras.py; crossing negative -> positive side of A->B counts as "in".
DEFAULT_LINE = [[0.10, 0.62], [0.90, 0.62]]
# Hot-reload cadence: every N seconds the session restats the line JSON
# and picks up any change the operator saved while a session is running,
# so redrawing the line takes effect without a stop/start round-trip.
LINE_RELOAD_POLL_S = 5.0
# Per-tid crossing cooldown: a foot point that jitters within a few pixels
# of the line can produce a real neg->pos->neg burst in one second. This
# rejects any crossing for a tid that already crossed within N seconds,
# regardless of direction. 2 s is small enough that a person who really
# doubled back is still counted twice, and large enough to eat the jitter.
CROSSING_COOLDOWN_S = 2.0

# Serializes EVERY model call in this process (detection + pose, live
# sessions + the one-shot deep window): ultralytics predict is not
# thread-safe on a shared model object.
INFER_LOCK = threading.Lock()
# 2026-08-23 (C1a): serialize read-modify-write on saved.json across
# every LiveSession + every save_event caller thread. Without this the
# gallery got duplicate rows when two auto-saves landed within one
# tick boundary (see bug #2 in AUDIT_2026-08-23.md).
_SAVED_JSON_LOCK = threading.Lock()

# Body layer sudden-motion gate on wrist/ankle keypoint velocity. The
# pose model returns COCO-17 kps; wrists (9,10) and ankles (15,16) are
# the limbs a theft/escape/punch swings hard. Per-track ring of the last
# N tick displacements, flag when the newest hop exceeds
# BODY_SUDDEN_RATIO x the median of that ring.
#
# 2026-08-18 tightened significantly after operator report that the layer
# was flagging normal walking / talking / standing as anomalies:
#   ratio 1.8 -> 2.4  (needs a much clearer spike above baseline)
#   ring  10  -> 10   (unchanged, enough history)
#   floor 0.20 -> 0.45 (hop must be 45% of box diagonal, not 20%)
#   min-samples 3 -> 6 (need 6 ticks of history before a first flag)
#   cooldown 3.0s -> 8.0s (don't re-flag the same person every few ticks)
BODY_SUDDEN_KP_IDX = (9, 10, 15, 16)
# Operator decision 2026-08-23: ratio 3.0 -> 8.0 (aggressive). The
# 4-limb mean displacement must now be EIGHT times the trailing median
# before a sudden-motion fires - only true violence-grade bursts
# (snatch, punch, fall) clear it; every normal-motion pattern that was
# still slipping through at 3.0 is out.
BODY_SUDDEN_RATIO = 8.0
BODY_SUDDEN_RING = 10
BODY_SUDDEN_FLOOR = 0.55
BODY_SUDDEN_MIN_SAMPLES = 6
BODY_SUDDEN_MIN_CONF = 0.35       # ignore low-conf kps entirely
BODY_SUDDEN_COOLDOWN_S = 8.0
# bbox-centroid velocity fallback when pose keypoints are absent
# (small / distant person that failed the pose crop gate). A track whose
# centroid darts >= BODY_BBOX_SUDDEN_FRAC of its own diagonal in one
# tick counts as sudden motion too. Raised 0.35 -> 0.55 so brisk walking
# doesn't fire; only actual running / bolting motion trips this.
BODY_BBOX_SUDDEN_FRAC = 0.65
# Operator decision 2026-08-23: on this hardware the body pass may
# sample a person only once every few seconds, so ordinary walking
# looks like a centroid "teleport" between two sparse frames. Demand a
# real observation window - at least 10 samples spanning 10 seconds -
# before any bbox hop is trusted.
BODY_BBOX_SUDDEN_MIN_SAMPLES = 10
BODY_BBOX_WINDOW_S = 10.0
# Debounce (operator decision 2026-08-23): a spike must persist for 2
# consecutive ticks before it may flag - single-frame jitter never fires.
BODY_SUDDEN_STREAK = 2
# Fighting detector (operator decision 2026-08-23): two person tracks
# within 60 px of each other whose sudden-motion bursts happened within
# the same short pair window AND both at genuinely high speed. Distance
# 90 -> 60 px kills hugs / handshakes / close conversation; the pair
# window + per-side speed floor demand that BOTH parties were bursting
# essentially simultaneously, not one runner passing a stander.
# Fall detector (decision D4, 2026-08-23; folded INTO the Body layer
# per operator direction - it is not a separate picker entry). Posture
# first, because on
# this host the tick rate is too sparse for velocity to be trustworthy
# (the decision-6 lesson): a person is a FALL SUSPECT when their pose
# torso (shoulder-center -> hip-center) leans more than FALL_ANGLE_DEG
# from vertical AND their bbox is wider than tall (lying footprint),
# after having been seen upright earlier in the same track. Without
# keypoints the aspect flip alone (upright history -> wide box) is
# used. FALL_CONFIRM_TICKS consecutive ticks are required, same
# debounce philosophy as decision 8.
FALL_ANGLE_DEG = 65.0
FALL_ASPECT_W_OVER_H = 1.15
FALL_CONFIRM_TICKS = 2
FALL_UPRIGHT_MIN_TICKS = 3
FALL_COOLDOWN_S = 10.0

BODY_FIGHT_MAX_DIST_PX = 60
BODY_FIGHT_PAIR_WINDOW_S = 2.0
BODY_FIGHT_COOLDOWN_S = 8.0

# Plate auto-save: how many unique-OCR-text plate reads a live session
# is willing to write to the Investigation gallery before it stops. The
# cap keeps saved.json scannable in the UI; a fresh session starts a
# new pool. Deduped by uppercase OCR text so a single passing car isn't
# saved once per tick.
PLATE_AUTO_SAVE_CAP = 200


# ---- Append-only event sink (2026-08-17) -------------------------------
# Every _emit_event call from a running session lands here as one JSON
# line: {"ts", "cam", "layer", "text", "box"|null}. Downstream:
#   * /api/events.jsonl streams the raw log (newest-first, capped).
#   * /api/export.csv turns it into a CSV suitable for spreadsheets.
# The sink is a single append per call - the write is short and file
# locks are handled by the OS's O_APPEND semantics; readers can tail
# the file safely without coordination. Rotated when the file grows
# past _EVENTS_MAX_BYTES so a long-lived session cannot balloon the
# operator's disk.
_EVENTS_DIR = _SRC_ROOT / "data" / "events"
_EVENTS_MAX_BYTES = 2_000_000     # ~2 MB per camera before rotation
_EVENTS_KEEP_LINES = 4000         # trim to the newest N lines on rotation


def _events_path(cam_id: str) -> Path:
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_"
                   for ch in (cam_id or "cam"))
    return _EVENTS_DIR / f"{safe}.jsonl"


def _append_event_jsonl(cam_id: str, layer: str, text: str,
                        box: dict | None = None) -> None:
    """Append one event line to the sink. Silent-on-failure - a disk
    hiccup must never take down a running detection tick."""
    try:
        _EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        p = _events_path(cam_id)
        payload = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cam": cam_id,
            "layer": layer,
            "text": text,
            "box": ({"x1": float(box["x1"]),
                     "y1": float(box["y1"]),
                     "x2": float(box["x2"]),
                     "y2": float(box["y2"]),
                     "cls": box.get("cls"),
                     "tid": box.get("tid")}
                    if box else None),
        }
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        # Rotate when the file grows too large. Keep the newest slice so
        # the CSV export continues to see recent history after the trim.
        if p.stat().st_size > _EVENTS_MAX_BYTES:
            keep = p.read_text(encoding="utf-8").splitlines()[-_EVENTS_KEEP_LINES:]
            p.write_text("\n".join(keep) + "\n", encoding="utf-8")
    except Exception:
        pass


def read_events(cam_id: str, limit: int = 500) -> list[dict]:
    """Return the newest `limit` events for a camera, oldest-first
    within the returned slice. Missing sink = []."""
    p = _events_path(cam_id)
    if not p.exists():
        return []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[dict] = []
    for line in lines[-max(1, int(limit)):]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


class BusyError(RuntimeError):
    """All MAX_SESSIONS live-analysis slots are taken."""


# ---------------------------------------------------------------------------
# Camera resolution: registry cameras by id, local-picker slots by slot_id.
# ---------------------------------------------------------------------------

def resolve_cam(cam_id: str, grid_path: Path | None = None) -> dict:
    """Return an analyzable camera dict for `cam_id`.

    Registry cameras (app/cameras.py) win; then the local picker's
    web/local_grid.json is searched by slot_id and a stream-resolvable
    dict is synthesized from the slot's embed/HLS/page fields; finally
    the uploaded-file store (`upload_*` ids from `/api/upload-video`)
    is searched under `src/data/uploads/`. Raises ValueError when the
    id is unknown or the slot has no usable stream.
    """
    from app.cameras import CAMERAS
    cam = CAMERAS.get(cam_id)
    if cam is not None:
        return {"id": cam_id, **cam}
    p = grid_path or (_SRC_ROOT / "web" / "local_grid.json")
    if p.exists():
        try:
            slots = json.loads(p.read_text(encoding="utf-8")).get("slots") or []
        except (OSError, ValueError):
            slots = []
        for slot in slots:
            if slot.get("slot_id") == cam_id:
                return _cam_from_slot(slot)
    # Uploaded local files: /api/upload-video saves upload_<hex>.<ext>
    # under src/data/uploads/ and the picker offers the stem as cam_id.
    # Match any allowed extension so MKV/MP4/MOV/etc. all resolve.
    if cam_id.startswith("upload_"):
        upload_dir = _SRC_ROOT / "data" / "uploads"
        for ext in (".mp4", ".mkv", ".mov", ".avi", ".webm"):
            candidate = upload_dir / f"{cam_id}{ext}"
            if candidate.is_file():
                return {"id": cam_id, "name": candidate.name,
                        "kind": "local_file", "path": str(candidate),
                        "country": "local"}
    raise ValueError(f"unknown camera {cam_id!r}")


def _cam_from_slot(slot: dict) -> dict:
    cam_id = slot["slot_id"]
    name = slot.get("placeholder_name") or cam_id
    # When the picker recorded which catalog camera backs this slot,
    # carry it as stream_id: the session then resolves + pools its
    # stream under the SAME key the producers use, so one camera never
    # runs two decoders (that duplication measurably starved the CPU).
    extra = {}
    if slot.get("cam_id"):
        extra["stream_id"] = slot["cam_id"]
    emb = slot.get("placeholder_embed") or ""
    m = re.search(r"/embed/([\w-]{11})", emb)
    if m:
        return {"id": cam_id, "name": name, "kind": "youtube",
                "url": f"https://www.youtube.com/watch?v={m.group(1)}",
                **extra}
    hls = slot.get("placeholder_hls") or ""
    m = re.match(r"^/tvkur/([^/]+)/", hls)
    if m:
        # The dashboard plays tvkur through its local proxy; the analysis
        # loop talks to the upstream directly (grab_frame carries the
        # Referer/Origin the host demands).
        return {"id": cam_id, "name": name, "kind": "hls",
                "url": f"https://content.tvkur.com/l/{m.group(1)}/master.m3u8"}
    if hls.startswith("http"):
        return {"id": cam_id, "name": name, "kind": "hls", "url": hls}
    page = slot.get("placeholder_page") or ""
    if "youtube.com/watch" in page:
        return {"id": cam_id, "name": name, "kind": "youtube", "url": page}
    if "webcamera24.com" in page:
        return {"id": cam_id, "name": name, "kind": "webcamera24",
                "url": page, "page": page}
    # skylinewebcams pages resolve through detect_core.resolve_skyline; the
    # picker writes them as a plain page link (no HLS/embed hint), so match
    # on the host and hand back a kind="skyline" dict.
    if "skylinewebcams.com" in page:
        return {"id": cam_id, "name": name, "kind": "skyline",
                "url": page, "page": page}
    raise ValueError(f"camera {cam_id!r} has no analyzable stream")


# ---------------------------------------------------------------------------
# Shared accumulators (pure - unit-testable without streams or a model).
# ---------------------------------------------------------------------------

def bump_heat(grid: list, boxes: list[dict], frame_shape, weight: float) -> None:
    """Bank each box's foot point into the session dwell grid.

    Set env HEAT_DEBUG=1 to log per-tick totals (2026-08-16) - operator
    reported the frontend peak stayed ~0 after 30s+ of dwell; the log
    tells us whether the backend accumulator is firing at all vs a
    JSON-transport gap on the way to the canvas.
    """
    import os as _os
    H, W = frame_shape[:2]
    if not (H and W):
        return
    banked = 0
    for b in boxes:
        fx = (b["x1"] + b["x2"]) / 2.0
        fy = b["y2"]
        if not (0 <= fx <= W and 0 <= fy <= H):
            continue
        gx = min(GRID_W - 1, int(fx / W * GRID_W))
        gy = min(GRID_H - 1, int(fy / H * GRID_H))
        grid[gy][gx] += weight
        banked += 1
    if _os.environ.get("HEAT_DEBUG"):
        total_weight = sum(v for row in grid for v in row)
        peak = max((max(row) for row in grid), default=0.0)
        print(f"bump_heat: banked {banked}/{len(boxes)} boxes weight={weight:.2f} "
              f"grid_total={total_weight:.2f} peak={peak:.2f}")


def grid_from_tracks(tracks, frame_shape) -> list:
    """One-shot dwell grid from a closed window's tracks (behavior.py's
    heat layer - same accumulation, no session)."""
    grid = [[0.0] * GRID_W for _ in range(GRID_H)]
    for tr in tracks:
        bump_heat(grid, tr.boxes, frame_shape, 1.0)
    return grid


def update_crossings(side_state: dict, tracks, frame_shape, line: list,
                     cross: dict, on_event=None, frame=None,
                     cam_id: str | None = None,
                     classes: list | set | None = None,
                     last_cross_ts: dict | None = None,
                     cooldown_s: float = CROSSING_COOLDOWN_S,
                     now: float | None = None) -> None:
    """Advance the session in/out counters from each visible track's
    NEWEST foot point. `side_state` remembers the last STRICTLY-signed
    side per track id (side == 0 means "on the line" and is stored as
    None - the next tick with a real sign starts the comparison from
    there). A crossing = a strict sign flip between two consecutive
    signed observations. Same convention as
    line-crossing convention: negative -> positive side of the
    A->B line = "in".

    on_event(direction, track, frame): optional callback fired on each
    crossing so the caller can persist an event + snapshot to
    data/crossings/<cam>.jsonl (see log_crossing_event below). cam_id +
    frame are forwarded to the callback so it can crop the mover for the
    event image. Absent callback -> counters only, backward-compatible.

    `classes`: iterable of class names to count (None = every class).
    Tracks whose `cls` is not in the set are skipped BEFORE side tracking
    so their sign changes never update `side_state` and never fire a
    counter or event.

    `last_cross_ts` + `cooldown_s`: per-tid cooldown to swallow the
    jitter burst you get when a foot point rides right on the line. If a
    tid already crossed within `cooldown_s` seconds, the next crossing
    (either direction) is dropped. Pass None for `last_cross_ts` to
    disable cooldown (the pre-cooldown behavior)."""
    from app.detect_core import _line_side
    H, W = frame_shape[:2]
    if not (H and W):
        return
    cls_filter = set(classes) if classes else None
    if now is None:
        now = time.time()
    for tr in tracks:
        if getattr(tr, "misses", 0):
            continue
        if cls_filter is not None and getattr(tr, "cls", None) not in cls_filter:
            continue
        b = tr.boxes[-1]
        fx = (b["x1"] + b["x2"]) / 2.0
        fy = b["y2"]
        nx, ny = fx / W, fy / H
        side = _line_side(nx, ny, line)
        prev = side_state.get(tr.tid)
        prev_side = prev[0] if isinstance(prev, tuple) else prev
        prev_pt = prev[1] if isinstance(prev, tuple) else None
        # Landing exactly on the line is ambiguous: don't classify it as
        # either side, and don't reset the last known side either - a
        # track that jitters neg -> 0 -> neg should count zero crossings.
        if side == 0:
            continue
        side_state[tr.tid] = (side, (nx, ny))
        if prev_side is None or prev_side == 0:
            continue
        direction = None
        if prev_side < 0 and side > 0:
            direction = "in"
        elif prev_side > 0 and side < 0:
            direction = "out"
        if not direction:
            continue
        # Industry crossing test (Ultralytics ObjectCounter pattern): a
        # sign flip alone also fires when a track jumps laterally past
        # the line's INFINITE extension. Require the finite movement
        # segment to actually intersect the finite counting line.
        if prev_pt is not None and not _segments_intersect(
                prev_pt, (nx, ny),
                (line[0][0], line[0][1]), (line[1][0], line[1][1])):
            continue
        # Eligibility gates: a 1-tick-old track or a sub-jitter hop must
        # not count (sparse-tick anti-double-count per DeepStream /
        # supervision practice - re-cast as displacement + age because a
        # confirmation tick costs seconds here).
        if getattr(tr, "hits", 99) < 2:
            continue
        if prev_pt is not None:
            disp = ((nx - prev_pt[0]) ** 2 + (ny - prev_pt[1]) ** 2) ** 0.5
            if disp < 0.01:
                continue
        if last_cross_ts is not None:
            prev_ts = last_cross_ts.get(tr.tid)
            if prev_ts is not None and (now - prev_ts) < cooldown_s:
                # Jitter suppression: same tid crossed less than
                # cooldown_s ago. Skip without touching the counter or
                # firing an event, but keep the newest side in
                # side_state so the tid can cross again once it moves
                # off the line.
                continue
            last_cross_ts[tr.tid] = now
        if direction == "in":
            cross["in"] = cross.get("in", 0) + 1
        else:
            cross["out"] = cross.get("out", 0) + 1
        if on_event is not None:
            try:
                on_event(direction=direction, track=tr, frame=frame,
                         cam_id=cam_id)
            except Exception as e:
                # Event persistence must never break the session's counter
                # loop. Log and move on.
                print(f"live_analysis: crossing on_event failed: "
                      f"{type(e).__name__}: {e}")


# ---- crossing-event log --------------------------------------------------

# Per-camera JSONL of the most recent line-crossing events. The dashboard's
# /api/crossings?cam=<id> reads this to render toasts + a history strip on
# the Line layer. Bounded rewrite: keep the newest CROSSING_LOG_KEEP rows;
# a full rewrite of ~50 rows is cheap and avoids indefinite growth on a
# camera with heavy traffic.
CROSSING_LOG_KEEP = 50


def _crossings_dir():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent / "data" / "crossings"


def log_crossing_event(cam_id: str, direction: str, track, frame=None) -> None:
    """Append a crossing event to data/crossings/<cam>.jsonl (bounded).
    When `frame` is provided we also save a small jpeg crop of the mover.

    The event fields the frontend reads:
      ts        - ISO-8601 UTC
      direction - "in" | "out"
      cls       - track class (person/car/bus/...)
      snap      - relative URL of the crop, or None
    """
    import json as _json
    import time as _t
    d = _crossings_dir()
    d.mkdir(parents=True, exist_ok=True)
    ts = _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime())
    snap_rel = None
    if frame is not None:
        try:
            import cv2 as _cv
            from pathlib import Path as _P
            b = track.boxes[-1]
            H, W = frame.shape[:2]
            pad = 20
            x1 = max(0, int(b["x1"]) - pad); y1 = max(0, int(b["y1"]) - pad)
            x2 = min(W, int(b["x2"]) + pad); y2 = min(H, int(b["y2"]) + pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                # Snaps go under src/web/snapshots/crossings/ so the
                # dashboard's static handler serves them at
                # /snapshots/crossings/<cam>/<file>.jpg with no extra route.
                web_snaps = (_P(__file__).resolve().parent.parent
                             / "web" / "snapshots" / "crossings" / cam_id)
                web_snaps.mkdir(parents=True, exist_ok=True)
                fname = f"{ts.replace(':', '')}_{track.tid}_{direction}.jpg"
                _cv.imwrite(str(web_snaps / fname), crop,
                            [_cv.IMWRITE_JPEG_QUALITY, 72])
                snap_rel = f"snapshots/crossings/{cam_id}/{fname}"
        except Exception as e:
            print(f"log_crossing_event: snap failed: {type(e).__name__}: {e}")
    row = {"ts": ts, "direction": direction,
           "cls": getattr(track, "cls", None),
           "tid": getattr(track, "tid", None), "snap": snap_rel}

    log = d / f"{cam_id}.jsonl"
    lines = []
    if log.exists():
        try:
            lines = log.read_text().splitlines()[-CROSSING_LOG_KEEP + 1:]
        except OSError:
            lines = []
    lines.append(_json.dumps(row))
    tmp = log.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(log)


def read_crossing_events(cam_id: str, limit: int = 20) -> list:
    """Newest-first read of the last N crossing events for a camera.
    Returns [] if the log doesn't exist (nothing has crossed yet)."""
    import json as _json
    log = _crossings_dir() / f"{cam_id}.jsonl"
    if not log.exists():
        return []
    try:
        rows = [_json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        return []
    return list(reversed(rows))[:limit]


# ---------------------------------------------------------------------------
# Fire detection: lightweight lazy loader for a dedicated fire/smoke YOLO
# checkpoint. The layer is optional: if the weights file is missing at
# FIRE_MODEL_PATH the loader records a friendly error and every subsequent
# call short-circuits (no repeated file/network lookups per tick). The
# session's fire pass reads the "err" field and shows an honest caption
# instead of degrading silently.
# ---------------------------------------------------------------------------

def load_fire_model():
    """Return the cached fire-detection YOLO model, or None on failure.

    The loader is idempotent and thread-safe. First call attempts to
    instantiate `ultralytics.YOLO(str(FIRE_MODEL_PATH))`; failures are
    recorded in _FIRE_MODEL_CACHE['err'] and surfaced by the caller in
    the layer caption. The layer's tick never blocks on loader retries.
    """
    if _FIRE_MODEL_CACHE.get("loaded"):
        return _FIRE_MODEL_CACHE.get("model")
    with _FIRE_MODEL_LOCK:
        if _FIRE_MODEL_CACHE.get("loaded"):
            return _FIRE_MODEL_CACHE.get("model")
        try:
            if not FIRE_MODEL_PATH.exists():
                _FIRE_MODEL_CACHE.update(
                    loaded=True, model=None,
                    err=(f"fire model missing - place a fire/smoke YOLO at "
                         f"{FIRE_MODEL_PATH.name} (any ultralytics-compatible "
                         f".pt with a 'fire' class)"))
                return None
            from ultralytics import YOLO
            model = YOLO(str(FIRE_MODEL_PATH))
            _FIRE_MODEL_CACHE.update(loaded=True, model=model, err=None)
            return model
        except Exception as e:  # noqa: BLE001
            _FIRE_MODEL_CACHE.update(
                loaded=True, model=None,
                err=f"fire model load failed: {type(e).__name__}: {e}")
            return None


def run_fire_inference(frame, conf: float = FIRE_CONF) -> list[dict]:
    """Return a list of fire/smoke detection dicts for the given frame.

    Each detection dict:  {"x1","y1","x2","y2","cls","conf"}.
    Empty list when the model is unavailable or nothing was detected;
    the caller consults load_fire_model()'s cached err for UX.
    """
    model = load_fire_model()
    if model is None:
        return []
    try:
        with INFER_LOCK:
            res = model.predict(frame, imgsz=FIRE_IMGSZ, conf=conf,
                                verbose=False)[0]
    except Exception as e:  # noqa: BLE001
        # A transient predict failure must not crash the session; note
        # it once and return an empty result. The cached err surfaces
        # in the caption.
        prev = _FIRE_MODEL_CACHE.get("err")
        note = f"fire predict failed: {type(e).__name__}: {e}"
        if prev != note:
            _FIRE_MODEL_CACHE["err"] = note
        return []
    if res is None or res.boxes is None:
        return []
    hits: list[dict] = []
    names = getattr(model, "names", {}) or {}
    for bb, ci, cf in zip(res.boxes.xyxy.tolist(),
                          res.boxes.cls.tolist(),
                          res.boxes.conf.tolist()):
        cls = str(names.get(int(ci), "fire")).lower()
        # 2026-08-24: the deployed detector (yolov26 fire finetune)
        # carries an "other" background class - only actual fire/smoke
        # verdicts may reach the layer.
        if cls not in ("fire", "smoke"):
            continue
        hits.append({
            "x1": int(bb[0]), "y1": int(bb[1]),
            "x2": int(bb[2]), "y2": int(bb[3]),
            "cls": cls,
            "conf": float(cf),
        })
    return hits
# Per-layer drawing lives in app/layers/draw.py since the 2026-08-23
# split (decision D1); the engine imports the drawers + shared constants.
from app.layers.draw import (BODY_ANOMALY_LABELS, FIRE_CONFIRM_TICKS, TRAIL_MAX_PTS,
    _caption, _chip, _hud_panel, _alert_banner,
    draw_paths_layer, draw_fire_layer, draw_zones_layer, draw_pose_layer, draw_plates_layer,
    draw_gestures_layer, draw_body_layer, draw_faces_layer_img, draw_heat_layer, draw_line_layer, box_overlap_over_spot,
    _segments_intersect, _clip_poly_by_halfplane, _poly_area)


def _static_postures(kps: list) -> list:
    """Single-frame postures provable from COCO-17 keypoints.

    hand_raised: a wrist confidently above its OWN shoulder line (and
    above the nose when the nose is confident) with the elbow between
    them - the classroom/audience "hand-raiser" geometry. Sequence
    gestures (waving etc.) are not attempted: they need >=4-10 fps.
    """
    def ok(i):
        return (i < len(kps) and kps[i] and len(kps[i]) >= 3
                and kps[i][2] >= 0.35)
    out = []
    NOSE, LSH, RSH, LEL, REL, LWR, RWR = 0, 5, 6, 7, 8, 9, 10
    for wr, el, sh in ((LWR, LEL, LSH), (RWR, REL, RSH)):
        if not (ok(wr) and ok(el) and ok(sh)):
            continue
        wrist_above_shoulder = kps[wr][1] < kps[sh][1] - 4
        above_nose = (not ok(NOSE)) or kps[wr][1] < kps[NOSE][1] + 6
        elbow_between = kps[el][1] < kps[sh][1] + 10
        if wrist_above_shoulder and above_nose and elbow_between:
            out.append("hand_raised")
            break
    return out


def _face_from_head_kps(kps: list) -> dict | None:
    """Rough face rectangle from COCO head keypoints (nose, eyes, ears).
    Fallback for the faces layer when the dedicated face detector returns
    nothing (2026-08-16): a pose-model that anchors a nose+one eye can
    approximate a face bbox where YuNet gives up. Width comes from the
    widest confident pair among the head kps (ideally ear-to-ear, else
    eye-to-eye, else a padded nose point); height is width * 1.3. Returns
    None when there aren't enough confident head kps to draw anything
    honest (single low-conf nose alone doesn't cut it)."""
    CONF_MIN = 0.35
    NOSE_I, LE, RE, LEA, REA = 0, 1, 2, 3, 4
    def get(i):
        if i < len(kps) and kps[i] and len(kps[i]) >= 3 and kps[i][2] >= CONF_MIN:
            return (float(kps[i][0]), float(kps[i][1]), float(kps[i][2]))
        return None
    nose = get(NOSE_I)
    le, re = get(LE), get(RE)
    lear, rear = get(LEA), get(REA)
    # Priority: ear-to-ear > eye-to-eye > single eye + nose > nose only
    width = None
    cx = cy = None
    confs = []
    if lear and rear:
        width = abs(lear[0] - rear[0])
        cx = (lear[0] + rear[0]) / 2
        cy = (lear[1] + rear[1]) / 2
        confs = [lear[2], rear[2]]
    elif le and re:
        # eye-to-eye is ~0.4 of ear-to-ear; scale up for face box.
        width = abs(le[0] - re[0]) * 2.5
        cx = (le[0] + re[0]) / 2
        cy = (le[1] + re[1]) / 2 + width * 0.15
        confs = [le[2], re[2]]
    elif nose and (le or re or lear or rear):
        anchor = le or re or lear or rear
        width = abs(nose[0] - anchor[0]) * 3.0
        cx = nose[0]
        cy = nose[1]
        confs = [nose[2], anchor[2]]
    else:
        return None
    if not width or width < 8:
        return None
    height = width * 1.3
    x1 = cx - width / 2
    y1 = cy - height * 0.45
    x2 = cx + width / 2
    y2 = cy + height * 0.55
    return {"x1": round(x1, 1), "y1": round(y1, 1),
            "x2": round(x2, 1), "y2": round(y2, 1),
            "conf": round(sum(confs) / len(confs), 3),
            "source": "pose_kps"}


def _pt_in_poly(x: float, y: float, pts: list) -> bool:
    """Ray-casting point-in-polygon on normalized coords."""
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i][0], pts[i][1]
        xj, yj = pts[j][0], pts[j][1]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


class _InferBatcher(threading.Thread):
    """Coalesces concurrent sessions' YOLO requests into one batched call.

    With N sessions free-running behind INFER_LOCK each session ticked
    every N x 1.4 s (measured: 4 cams -> a new frame every 5-8 s). Here a
    session hands its frame in and blocks; the batcher waits a short
    COLLECT_S for the other active sessions (they pace off the same
    previous round, so after one round they arrive nearly together), then
    runs detect_with_boxes_batch once and fans the per-frame results
    back. Single-session cost: +COLLECT_S. Four-session cost: one
    batch-of-4 forward (~2.5x single) instead of 4x serial.
    """

    COLLECT_S = 0.10

    def __init__(self):
        super().__init__(daemon=True, name="live-infer-batcher")
        self._lock = threading.Lock()
        self._reqs: list[dict] = []
        self._wake = threading.Event()
        self.start()

    def infer(self, model, frame, conf: float, gates: dict) -> list[dict]:
        req = {"model": model, "frame": frame, "conf": conf,
               "gates": gates, "done": threading.Event(),
               "out": None, "err": None}
        with self._lock:
            self._reqs.append(req)
        self._wake.set()
        req["done"].wait(timeout=90)
        if req["err"] is not None:
            raise req["err"]
        return req["out"] or []

    def run(self) -> None:
        from app.detect_core import detect_with_boxes_batch
        while True:
            self._wake.wait()
            time.sleep(self.COLLECT_S)
            with self._lock:
                batch, self._reqs = self._reqs, []
                self._wake.clear()
            if not batch:
                continue
            try:
                with INFER_LOCK:
                    outs = detect_with_boxes_batch(
                        batch[0]["model"],
                        [r["frame"] for r in batch],
                        imgsz=LIVE_IMGSZ,
                        per_class_conf_list=[r["gates"] for r in batch],
                        conf_list=[r["conf"] for r in batch],
                        classes=LIVE_CLASSES,
                        agnostic_nms=True)
                for r, (_counts, boxes) in zip(batch, outs):
                    r["out"] = boxes
            except Exception as e:  # noqa: BLE001 - deliver, don't die
                for r in batch:
                    r["err"] = e
            finally:
                for r in batch:
                    r["done"].set()


BATCHER = _InferBatcher()


class _StreamReader(threading.Thread):
    """Continuously tracks the live edge of one direct-HLS stream.

    The old per-tick path opened a fresh cv2.VideoCapture for every
    analyzed frame - measured at 1.0-2.1 s per open on the operator's
    laptop, which alone made a ~2.5 s floor between analysis updates.
    This thread opens the capture ONCE, then grab()s every source frame
    (decode-only, no BGR conversion - the cheap half of read()) to stay
    pinned to the live edge, and retrieve()s a full frame a few times a
    second into `latest`. The analysis tick takes the newest frame
    instantly instead of paying the open cost again and again.
    """

    RETRIEVE_EVERY = 6      # ~4 fresh BGR frames/s at a 25 fps source

    def __init__(self, url: str):
        super().__init__(daemon=True, name="live-analysis-reader")
        self.url = url
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.latest = None          # newest BGR frame (ndarray)
        self.latest_ts = 0.0
        self.dead = False

    def run(self) -> None:
        from app.detect_core import _open_cap   # applies ffmpeg timeouts
        cap = _open_cap(self.url)
        if not cap.isOpened():
            self.dead = True
            return
        n = 0
        try:
            while not self.stop_event.is_set():
                if not cap.grab():
                    # Live edge starved or the signed manifest rotated:
                    # one in-place reopen attempt, then declare dead and
                    # let LiveSession._grab rebuild us on a fresh URL.
                    cap.release()
                    time.sleep(0.5)
                    cap = _open_cap(self.url)
                    if not cap.isOpened() or not cap.grab():
                        self.dead = True
                        return
                n += 1
                if n % self.RETRIEVE_EVERY == 0:
                    ok, frame = cap.retrieve()
                    if ok and frame is not None:
                        with self.lock:
                            self.latest = frame
                            self.latest_ts = time.time()
        finally:
            cap.release()

    def snapshot(self):
        self.last_used = time.time()
        with self.lock:
            return self.latest

    def snapshot_wait(self, timeout: float = 8.0):
        """snapshot(), but block up to `timeout` for the FIRST frame of a
        freshly-started reader instead of returning None."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            fr = self.snapshot()
            if fr is not None or self.dead:
                return fr
            time.sleep(0.1)
        return self.snapshot()

    def stop(self) -> None:
        self.stop_event.set()


# ---- Shared reader pool ----------------------------------------------------
# ONE persistent live-edge reader per camera, shared by every consumer:
# live-analysis sessions AND local_producers' KPI rounds. Before the pool,
# producers re-opened the HLS stream from scratch every 8 s round (measured
# 1-2 s per open, times four cams) while the sessions each held their own
# open reader for the very same streams - double infrastructure that
# saturated the CPU and slowed everyone's ticks. Readers idle-stop after
# _READER_IDLE_STOP_S without a snapshot() so an unwatched camera doesn't
# decode forever.

_READER_POOL: dict[str, _StreamReader] = {}
_READER_POOL_LOCK = threading.Lock()
_READER_IDLE_STOP_S = 180.0
# Consecutive staleness-rebuilds per camera. googlevideo occasionally
# retires a segment-edge pool BEFORE the manifest's expire= stamp
# (observed live: ffmpeg error -138 on every rr*.googlevideo.com edge
# while the cached manifest URL still had hours of validity) - rebuilding
# the reader on the SAME manifest re-knocks the same dead edges forever.
_STALE_REBUILDS: dict[str, int] = {}


def get_shared_reader(cam: dict, cam_id: str):
    """The pool's reader for this camera, (re)built as needed.

    Returns None for header-required hosts (tvkur/ibb/skyline) - those
    can't ride a plain VideoCapture and keep their per-tick segment path.
    Raises when the stream can't be resolved (caller counts the failure;
    resolve_stream's negative cache keeps the retry cost near zero).
    """
    from app.detect_core import HEADER_HOSTS, resolve_stream
    url = resolve_stream(cam)
    if any(h in url for h in HEADER_HOSTS):
        return None
    now = time.time()
    with _READER_POOL_LOCK:
        # Idle-stop readers nobody snapshots anymore.
        for key, r in list(_READER_POOL.items()):
            if key != cam_id and now - getattr(r, "last_used", now) \
                    > _READER_IDLE_STOP_S:
                r.stop()
                _READER_POOL.pop(key, None)
        r = _READER_POOL.get(cam_id)
        # 20s staleness bound (2 HLS segment periods): live segments land
        # in ~5s bursts, and a 10s bound tripped on burst boundaries,
        # needlessly rebuilding healthy readers and costing whole rounds.
        stale = (r is not None and r.latest is not None
                 and now - r.latest_ts > 20)
        if r is not None and not stale and not r.dead and r.is_alive():
            _STALE_REBUILDS[cam_id] = 0
        if r is None or r.dead or not r.is_alive() or r.url != url or stale:
            if r is not None:
                r.stop()
            # Decision 21: measure the manifest's PDT live-edge offset in
            # the background on every fresh build - never on the tick path.
            threading.Thread(target=_measure_pdt_offset, args=(url, cam_id),
                             daemon=True, name="pdt-probe").start()
            if stale:
                n = _STALE_REBUILDS.get(cam_id, 0) + 1
                _STALE_REBUILDS[cam_id] = n
                if n >= 2:
                    # Two staleness rebuilds in a row: the manifest's
                    # segment edges are dead even though expire= says
                    # valid. Force a fresh resolve so the new reader gets
                    # a NEW edge assignment instead of the dead pool.
                    from app.detect_core import invalidate_stream as _inv
                    _inv(cam_id)
                    try:
                        url = resolve_stream(cam)
                    except Exception:
                        pass   # keep the old URL; the next round retries
                    _STALE_REBUILDS[cam_id] = 0
            r = _StreamReader(url)
            r.last_used = now
            r.start()
            _READER_POOL[cam_id] = r
        return r


def _rider_person_tids(tracks) -> set:
    """Person tracks that are RIDING a two-wheeler: their box overlaps a
    vehicle track's box by most of its own area. Behavior verdicts
    (walking / crouching / erratic) are meaningless for a mounted rider -
    audit 2026-08-14 caught a rider labeled "WALKING crouching"."""
    out: set = set()
    veh = [t.boxes[-1] for t in tracks
           if t.cls in _VEHICLE_CLASSES and not t.misses]
    if not veh:
        return out
    for t in tracks:
        if t.cls != "person" or t.misses:
            continue
        p = t.boxes[-1]
        pa = max(1.0, (p["x2"] - p["x1"]) * (p["y2"] - p["y1"]))
        for v in veh:
            ix = min(p["x2"], v["x2"]) - max(p["x1"], v["x1"])
            iy = min(p["y2"], v["y2"]) - max(p["y1"], v["y1"])
            if ix > 0 and iy > 0 and (ix * iy) / pa >= 0.45:
                out.add(t.tid)
                break
    return out


class LiveSession(threading.Thread):
    """One camera's live analysis: stream -> detect -> track -> layer."""

    def __init__(self, cam: dict, model, layer: str):
        super().__init__(daemon=True, name=f"live-analysis-{cam['id']}")
        self.cam = cam
        self.cam_id = cam["id"]
        self.cam_name = cam.get("name", cam["id"])
        # Stream identity for resolve-cache + shared reader pool. When the
        # slot maps to a catalog camera this is the catalog id (shared
        # with local_producers -> ONE reader per physical camera); the
        # session's own cam_id stays the slot id for zones/lines/API.
        from app.cameras import CAMERAS as _CAMS
        sid = cam.get("stream_id")
        if sid and sid in _CAMS:
            self.stream_key = sid
            self.stream_cam = {"id": sid, **_CAMS[sid]}
        else:
            self.stream_key = self.cam_id
            self.stream_cam = cam
        self.model = model
        self.layer = layer            # mutated by the manager on switch
        self.created = time.time()
        self.last_poll = time.time()  # touched by every /frame poll
        self.stop_event = threading.Event()
        self.lock = threading.Lock()  # guards latest/seq/note
        self.latest: bytes | None = None
        # Structured snapshot for the canvas-overlay renderer. The
        # frontend fetches this every ~800 ms and draws boxes/heat/line
        # on a canvas positioned over the live iframe, so the video
        # stays 25 fps while the analysis overlay ticks at YOLO's pace.
        # Same PID as the poll handler, so a plain dict is safe.
        self.latest_data: dict | None = None
        self.seq = 0
        self.note = "starting stream..."
        self.err: str | None = None
        # 2026-08-23 (B2): per-stage tick timing exposed on the JSON
        # payload so the HUD can show real end-to-end latency instead
        # of just the inter-tick gap.
        self._last_stage_ms: dict = {}
        # Rolling state that SURVIVES layer switches (fix 2 point 9):
        self.tracker = None
        self.heat = [[0.0] * GRID_W for _ in range(GRID_H)]
        self.heat_since: float | None = None
        # User-drawn override (data/lines/<cam>.json) wins over cameras.py.
        # The line + its class filter are hot-reloaded on the fly (see
        # _maybe_reload_line) so redrawing while a session runs takes
        # effect within LINE_RELOAD_POLL_S seconds without restart.
        from app.cameras import resolve_line as _resolve_line
        from app.cameras import resolve_line_classes as _resolve_classes
        from app.cameras import resolve_zones as _resolve_zones
        self.line = _resolve_line(self.cam_id) or cam.get("line") or DEFAULT_LINE
        self.line_classes = _resolve_classes(self.cam_id)
        self._line_mtime = self._line_json_mtime()
        self._next_line_check = time.time() + LINE_RELOAD_POLL_S
        # User-drawn zones (loiter areas + parking spots) - same hot-reload
        # contract as the counting line.
        self.zones = _resolve_zones(self.cam_id)
        self._zones_mtime = self._zones_json_mtime()
        self._next_zones_check = time.time() + LINE_RELOAD_POLL_S
        self._zone_since: dict[tuple, float] = {}   # (tid, zone_idx) -> t0
        # 2026-08-17: parking layer no longer REQUIRES operator polygons.
        # If no manual parking zones exist for this camera, the auto-
        # parking bootstrap observes ~3 minutes of live footage, clusters
        # stationary vehicle foot-points into a 24x14 grid, and promotes
        # any cell that saw >= MIN_HITS stationary observations to a
        # parking spot. Manual polygons keep their priority - if the
        # operator draws one, the auto-inferred spots are ignored.
        from app.auto_parking import AutoParkingBootstrap
        self._auto_parking = AutoParkingBootstrap(self.cam_id)
        self.cross = {"in": 0, "out": 0}
        self._line_sides: dict[int, float] = {}
        self._last_cross_ts: dict[int, float] = {}
        self.gesture_counts: dict[str, int] = {}
        self._track_gestures: dict[int, set] = {}
        self._faces_ok: bool | None = None
        self._fail = 0
        self._last_tick: float | None = None
        # Fire layer per-session state: last tick's detections + a
        # consecutive-positive-tick streak. Confirmation only fires when
        # streak reaches FIRE_CONFIRM_TICKS (matches the operator's spec
        # of "any hit confirmed for >2 consecutive ticks" - two ticks in
        # a row means the alert appears on the second tick).
        self._fire_hits: list[dict] = []
        self._fire_streak = 0
        self._fire_confirmed = False

    # -- lifecycle ---------------------------------------------------------

    def run(self) -> None:
        # Per-tick guard so one bad frame (e.g. a new class the downstream
        # colour/config map hasn't been extended for - the KeyError:'animal'
        # observed on 2026-08-20) does not kill the whole session. The
        # outer try still catches process-fatal errors; the inner one logs
        # once and continues on the next tick.
        _tick_err_logged = False
        try:
            while (not self.stop_event.is_set()
                   and time.time() - self.last_poll < IDLE_STOP_S):
                t0 = time.time()
                # 2026-08-23 (B2): per-stage timing exposed to the HUD so
                # the operator can see real end-to-end latency, not just
                # the inter-tick delivery gap. All stages measured in ms.
                _st_grab = _st_infer = _st_render = _st_publish = 0.0
                try:
                    _t = time.time()
                    frame = self._grab()
                    _st_grab = (time.time() - _t) * 1000.0
                    if frame is None:
                        self._publish_note("stream unavailable - retrying...")
                        if self.stop_event.wait(2.0):
                            break
                        continue
                    self._last_frame = frame   # event thumbs crop from here
                    _t = time.time()
                    boxes = self._infer(frame)
                    _st_infer = (time.time() - _t) * 1000.0
                    now = time.time()
                    if self.tracker is None:
                        from app.tracker import BurstTracker
                        self.tracker = BurstTracker(frame.shape)
                    self.tracker.update(boxes, now)
                    self._trim()
                    layer = self.layer
                    if layer in ("pose", "gestures", "body"):
                        self._pose_pass(frame, boxes)
                    if layer == "gestures":
                        # Hand-landmark pass (2026-08-24): open palm /
                        # fist / pointing on wrist crops, budgeted, and
                        # silently absent when mediapipe or the model
                        # file is not available on this machine.
                        try:
                            from app.hands import analyze_hands
                            analyze_hands(frame, boxes)
                        except Exception:
                            pass
                    if layer == "plates":
                        # 2026-08-17: operator wants LPR to run on live streams
                        # regardless of kind. YouTube pixels are compressed and
                        # OCR often fails on them, but the failure is honest
                        # (empty plate string) and the ATTEMPT is what the
                        # operator asked for. The old YouTube-blanket skip is
                        # gone; a per-track cache + confidence gate inside
                        # attach_plates still stops the pass from wasting cycles
                        # on unreadable crops.
                        self._plates_pass(frame)
                    if layer == "parking":
                        self._parking_probe(frame)
                        self._auto_parking_tick(frame.shape, now)
                    if layer == "fire":
                        self._fire_pass(frame)
                    faces_list: list[dict] = []
                    if layer == "faces":
                        faces_list = self._faces_pass(frame, boxes)
                    self._accumulate(frame, boxes, now)
                    _t = time.time()
                    img = self._render(frame, faces_list, layer)
                    _st_render = (time.time() - _t) * 1000.0
                    _t = time.time()
                    self._publish(img)
                    self._publish_data(frame.shape, boxes, layer, faces_list)
                    _st_publish = (time.time() - _t) * 1000.0
                    # Latch the per-stage numbers so _publish_data picks
                    # them up next tick (safe to read even on first tick).
                    self._last_stage_ms = {
                        "grab": round(_st_grab, 1),
                        "infer": round(_st_infer, 1),
                        "render": round(_st_render, 1),
                        "publish": round(_st_publish, 1),
                        "tick": round((time.time() - t0) * 1000.0, 1),
                    }
                except Exception as tick_err:  # noqa: BLE001
                    if not _tick_err_logged:
                        _tick_err_logged = True
                        import traceback as _tb
                        print(f"live-analysis {self.cam_id}: tick error "
                              f"(surviving next frame) "
                              f"{type(tick_err).__name__}: {tick_err}")
                        _tb.print_exc()
                    self._publish_note(
                        f"tick error, retrying: "
                        f"{type(tick_err).__name__}")
                dt = time.time() - t0
                wait = max(0.0, TICK_TARGET_S - dt)
                if wait and self.stop_event.wait(wait):
                    break
        except Exception as e:  # noqa: BLE001 - the session must not die silently
            self.err = f"{type(e).__name__}: {e}"
            self._publish_note(f"analysis stopped: {self.err}")
            print(f"live-analysis {self.cam_id}: crashed ({self.err})")
        # No reader cleanup here: readers live in the shared pool now and
        # idle-stop on their own when nothing snapshots them anymore.

    # -- pipeline stages ---------------------------------------------------

    def _grab(self):
        from app.detect_core import (HEADER_HOSTS, grab_frame,
                                     invalidate_stream, resolve_stream)
        try:
            url = resolve_stream(self.stream_cam)
        except Exception:
            self._fail += 1
            return None
        # Screen-capture sentinel bypasses the persistent VideoCapture
        # entirely: grab_frame() routes screen:// straight into
        # screen_capture.capture(). Ticks per-frame instead of a
        # persistent reader (there's no HLS stream to demux).
        if url and url.startswith("screen://"):
            frame = grab_frame(url)
            if frame is None:
                self._fail += 1
            else:
                self._fail = 0
                self._last_frame_ts = time.time()
            return frame
        # Header-required hosts (tvkur, ibb, skyline) can't ride a plain
        # persistent VideoCapture - every segment request needs Referer/
        # Origin headers - so they keep the old per-tick segment path.
        if any(h in url for h in HEADER_HOSTS):
            frame = grab_frame(url)
            if frame is None:
                self._fail += 1
                if self._fail % GRAB_FAIL_REFRESH == 0:
                    invalidate_stream(self.stream_key)
            else:
                self._fail = 0
                self._last_frame_ts = time.time()
            return frame
        try:
            r = get_shared_reader(self.stream_cam, self.stream_key)
        except Exception:
            self._fail += 1
            return None
        if r is None:      # header host slipped through - segment path
            return grab_frame(url)
        frame = r.snapshot_wait(timeout=4.0)
        # A reader whose frames stopped aging forward is wedged (stalled
        # stream that still holds its last decode) - rebuild next tick.
        # 20s bound matches the pool getter (HLS bursts every ~5s).
        if frame is not None and time.time() - r.latest_ts > 20:
            r.stop()
            r.dead = True
            frame = None
        if frame is None:
            self._fail += 1
            if self._fail % GRAB_FAIL_REFRESH == 0:
                # Expired manifest / rotated token: force a fresh resolve.
                invalidate_stream(self.stream_key)
        else:
            self._fail = 0
            self._last_frame_ts = r.latest_ts
        return frame

    def _night_profile(self, frame) -> bool:
        """True when the night profile is active for this frame. Mean gray
        with ON/OFF hysteresis; recomputed cheaply on a 4x-decimated view."""
        import cv2
        g = cv2.cvtColor(frame[::4, ::4], cv2.COLOR_BGR2GRAY)
        luma = float(g.mean())
        prev = getattr(self, "_night_on", False)
        if prev and luma > NIGHT_LUMA_OFF:
            self._night_on = False
        elif not prev and luma < NIGHT_LUMA_ON:
            self._night_on = True
        return getattr(self, "_night_on", False)

    def _infer(self, frame) -> list[dict]:
        import cv2
        from app.detect_core import DEFAULT_PER_CLASS_CONF, filter_boxes_roi
        # Night profile: CLAHE on the L channel lifts dark pedestrians
        # into the detector's working range (the most consistently
        # effective classical enhancer in the night-CCTV literature).
        if self._night_profile(frame):
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = getattr(self, "_clahe", None)
            if clahe is None:
                clahe = self._clahe = cv2.createCLAHE(
                    clipLimit=2.5, tileGridSize=(8, 8))
            frame = cv2.cvtColor(
                cv2.merge((clahe.apply(l_ch), a_ch, b_ch)),
                cv2.COLOR_LAB2BGR)
        gates = dict(self.cam.get("per_class_conf") or DEFAULT_PER_CLASS_CONF)
        gates = {k: max(LIVE_CONF_FLOOR, v * LIVE_GATE_SCALE)
                 for k, v in gates.items()}
        # All sessions funnel through the batcher: concurrent ticks share
        # one batched forward pass instead of queueing on INFER_LOCK.
        boxes = BATCHER.infer(self.model, frame, LIVE_CONF_FLOOR, gates)
        if (self.cam.get("roi") or self.cam.get("roi_exclude")
                or self.cam.get("roi_exclude_class")):
            boxes = filter_boxes_roi(boxes, frame.shape, self.cam.get("roi"),
                                     self.cam.get("roi_exclude"),
                                     self.cam.get("roi_exclude_class"))
        return boxes

    def _pose_pass(self, frame, boxes) -> None:
        # NOT under INFER_LOCK: the pose pass serializes on detect_core's
        # _PREDICT_LOCK inside attach_keypoints_crops. Holding the batcher
        # lock across it as well queued every other session's DETECTION
        # behind this session's pose (measured: 12-20s infer waits with 4
        # pose sessions vs ~3-6s without).
        from app.pose import attach_keypoints_crops, load_pose_model
        attach_keypoints_crops(load_pose_model(), frame, boxes,
                               min_box_h=KPS_MIN_BOX_H,
                               max_crops=POSE_MAX_CROPS)

    def _plates_pass(self, frame) -> None:
        # Serializes on _PREDICT_LOCK inside attach_plates (same OpenVINO
        # single-InferRequest rule as detection and pose). Per-track read
        # cache lives on the session so a plate is OCRed once per track.
        from app.plates import attach_plates, load_ocr, load_plate_model
        if not hasattr(self, "_plate_reads"):
            self._plate_reads: dict[int, dict] = {}
        # SC feedback-loop gate: when the user tabs away from the video
        # tab, screen capture keeps grabbing whatever is currently on
        # the primary display - the chat pane, another window, the
        # taskbar. Those "frames" occasionally clear YOLO's motorcycle/
        # car threshold on text UI elements, and then the plate reader
        # OCRs strings out of the surrounding text and calls them
        # plates. Skip the plates pass entirely on frames that look like
        # solid-color UI content instead of a street scene: a real
        # camera frame has ~30-60% pixel variance across colours; chat/
        # editor frames are >80% one dominant colour (usually near-
        # white or the editor's dark theme). Bail on both extremes.
        try:
            import numpy as _np
            _gray = frame.mean(axis=2) if frame.ndim == 3 else frame
            _dominant = float(_np.mean(
                ((_gray < 40) | (_gray > 230)).astype(_np.float32)))
            if _dominant > 0.80:
                # A chat/editor screenshot is >80% white or >80% dark.
                # Don't waste OCR budget on it AND don't pollute the
                # Investigation gallery with UI screenshots.
                return
        except Exception:
            pass
        try:
            attach_plates(load_plate_model(), load_ocr(), frame,
                          self.tracker, self._plate_reads,
                          cam_id=self.cam_id)
        except Exception as e:
            # A missing/corrupt model must degrade to an honest empty
            # layer, not kill the session.
            if not getattr(self, "_plates_err_once", False):
                self._plates_err_once = True
                print(f"live-analysis {self.cam_id}: plates pass disabled "
                      f"({type(e).__name__}: {e})")
            return
        # Auto-save plate crops to the Investigation gallery (operator
        # request 2026-08-18). One event per unique plate text (OCR-
        # deduped) - a busy street with 20 vehicles doesn't flood the
        # gallery with 20 identical reads of the same passing car. Cap
        # the session pool at PLATE_AUTO_SAVE_CAP to keep saved.json
        # scannable in the UI.
        #
        # 2026-08-21 (operator report): the text-based dedup let the SAME
        # track re-save whenever the OCR "text" wobbled (an extra space,
        # a look-alike digit, a raised confidence). Switch to per-tid
        # dedup with a long backoff: once a tracker id (`tid`) has been
        # saved for this session, don't save it AGAIN until at least
        # PLATE_TID_DEDUP_S seconds passed (default 300 s = 5 min). A
        # new tid = a new vehicle in the tracker's own accounting, so
        # this cleanly separates "same car staying visible" from
        # "different car passing later".
        PLATE_TID_DEDUP_S = 300.0
        # 2026-08-23 (C1b): cross-track string dedup. If a plate string
        # was already emitted within PLATE_STR_DEDUP_S seconds, suppress
        # it even if a NEW tid produced it (happens when the tracker
        # splits a car under obstruction bail / fast motion). Normalized
        # to fold common OCR mixups (O/0, I/1, S/5, B/8) so "20H6863"
        # and "20H68G3" dedupe to the same slot.
        PLATE_STR_DEDUP_S = 30.0
        # 2026-08-23 (C1c): temporal agreement gate. A tid's read must
        # be observed at least AGREEMENT_MIN times (across OCR ticks)
        # before we emit it to the gallery. Cuts single-frame OCR
        # hallucinations without needing a higher confidence gate.
        AGREEMENT_MIN = 2
        if not hasattr(self, "_plate_emitted"):
            # tid -> last_saved_ts (float)
            self._plate_emitted: dict[int, float] = {}
        if not hasattr(self, "_plate_str_emitted"):
            # normalized_text -> last_saved_ts (float)
            self._plate_str_emitted: dict[str, float] = {}

        def _normalize_plate(s: str) -> str:
            t = "".join(ch for ch in (s or "").upper()
                        if ch.isalnum())
            # Fold common OCR mixups so near-duplicates dedupe.
            t = (t.replace("O", "0").replace("I", "1")
                  .replace("S", "5").replace("B", "8"))
            return t

        _now_dedup = time.time()
        if len(self._plate_emitted) >= PLATE_AUTO_SAVE_CAP:
            return
        for tr in (self.tracker.open if self.tracker else []):
            if not tr.boxes:
                continue
            b = tr.boxes[-1]
            text = b.get("plate")
            pbox = b.get("plate_box")
            if not text or not pbox:
                continue
            tid = tr.tid
            _last_ts = self._plate_emitted.get(tid, 0.0)
            if _now_dedup - _last_ts < PLATE_TID_DEDUP_S:
                continue
            # C1c: require the current text to have been read at least
            # AGREEMENT_MIN times for this tid before we trust it enough
            # to emit. `text_counts` is populated in plates.attach_plates
            # on every OCR pass (any read, not only best-so-far).
            _reads = self._plate_reads.get(tid, {}) if hasattr(self, "_plate_reads") else {}
            _counts = _reads.get("text_counts", {}) if isinstance(_reads, dict) else {}
            if _counts.get(text, 0) < AGREEMENT_MIN:
                continue
            # C1b: cross-track dedup on the normalized string.
            _norm = _normalize_plate(text)
            _last_str_ts = self._plate_str_emitted.get(_norm, 0.0)
            if _norm and _now_dedup - _last_str_ts < PLATE_STR_DEDUP_S:
                # Still counts as "handled" for this tid so we don't
                # burn CPU re-checking every tick until the window ends.
                self._plate_emitted[tid] = _now_dedup
                continue
            self._plate_emitted[tid] = _now_dedup
            if _norm:
                self._plate_str_emitted[_norm] = _now_dedup
            box = {"x1": pbox[0], "y1": pbox[1],
                   "x2": pbox[2], "y2": pbox[3]}
            try:
                self._emit_event("plates",
                                 f"plate: {text} "
                                 f"({b.get('plate_conf', 0):.2f})",
                                 box, auto_save=True, tight_crop=True)
            except Exception:
                pass
            if len(self._plate_emitted) >= PLATE_AUTO_SAVE_CAP:
                break

    def _faces_pass(self, frame, boxes: list[dict] | None = None
                     ) -> list[dict]:
        from app import faces as _faces
        if self._faces_ok is None:
            self._faces_ok = _faces.available()
        faces_from_yunet: list[dict] = []
        if self._faces_ok:
            faces_from_yunet = _faces.detect_faces(frame)
            # YuNet's own shipped threshold is 0.60 (see FACE_SCORE in
            # app/faces.py). The earlier 0.9 + 24px filter was tuned for
            # night far-field cameras and killed every face on day cameras
            # at street-mid range - the exact distance the earlier repo's
            # face demo used. Relaxed to YuNet's own 0.6 + a 16px floor,
            # kept the 32-face hard cap so a busy day frame stays snappy.
            faces_from_yunet = [
                f for f in faces_from_yunet
                if float(f.get("conf") or 0) >= 0.6
                and (f["x2"] - f["x1"]) >= 16
                and (f["y2"] - f["y1"]) >= 16]
            faces_from_yunet.sort(key=lambda f: -float(f.get("conf") or 0))
        if faces_from_yunet:
            kept = faces_from_yunet[:32]
            for _n, _f in enumerate(kept, 1):
                _f["n"] = _n
            # Face-crop audit trail (2026-08-24): same idea as the plate
            # crops - padded crops land under src/data/face_crops/<cam>
            # with a per-cell cooldown and a rolling cap, so the
            # Investigation flow can pull and SAVE any face later.
            try:
                _faces.save_face_crops(self.cam_id, frame, kept)
            except Exception:
                pass
            return kept
        # Pose-keypoint fallback (2026-08-16, operator: "old version
        # detected faces here"). YuNet often returns zero on far-field or
        # partly-turned faces the pose model can still anchor on. We run
        # the crop-based pose pass on person boxes and derive a rough
        # face box from the head-cluster keypoints (nose, eyes, ears).
        # Better a rough rectangle than a hard zero - the caption clearly
        # tags these as pose-derived so no one confuses them for YuNet
        # detections.
        if not boxes:
            return []
        try:
            from app.pose import attach_keypoints_crops, load_pose_model
            attach_keypoints_crops(load_pose_model(), frame, boxes,
                                   min_box_h=KPS_MIN_BOX_H,
                                   max_crops=POSE_MAX_CROPS)
        except Exception as e:
            if not getattr(self, "_faces_pose_err_once", False):
                self._faces_pose_err_once = True
                print(f"live-analysis {self.cam_id}: face pose fallback "
                      f"disabled ({type(e).__name__}: {e})")
            return []
        derived: list[dict] = []
        for b in boxes:
            if b.get("cls") != "person" or not b.get("kps"):
                continue
            face = _face_from_head_kps(b["kps"])
            if face is not None:
                derived.append(face)
        derived.sort(key=lambda f: -float(f.get("conf") or 0))
        return derived[:32]

    def _zones_json_mtime(self) -> float | None:
        from app.cameras import _zones_dir
        p = _zones_dir() / f"{self.cam_id}.json"
        try:
            return p.stat().st_mtime
        except OSError:
            return None

    def _maybe_reload_zones(self, now: float) -> None:
        """Hot-reload user-drawn zones on the same cadence as the line.
        The dwell clocks restart on an edit (indices may have shifted);
        occupancy recovers within one tick."""
        if now < self._next_zones_check:
            return
        self._next_zones_check = now + LINE_RELOAD_POLL_S
        mtime = self._zones_json_mtime()
        if mtime == self._zones_mtime:
            return
        self._zones_mtime = mtime
        from app.cameras import resolve_zones as _resolve_zones
        self.zones = _resolve_zones(self.cam_id)
        self._zone_since.clear()

    def _line_json_mtime(self) -> float | None:
        """Current mtime of data/lines/<cam>.json, or None when the file
        does not exist. Used by _maybe_reload_line to detect a fresh save
        or a clear that happened while the session is running."""
        from app.cameras import _lines_dir
        p = _lines_dir() / f"{self.cam_id}.json"
        try:
            return p.stat().st_mtime
        except OSError:
            return None

    def _maybe_reload_line(self, now: float) -> None:
        """Hot-reload the line + class filter if the JSON has been
        rewritten (or removed) since the last check.

        Cadence is bounded by LINE_RELOAD_POLL_S so this is at most one
        stat call every few seconds - cheap next to the inference tick.
        When the line moves, side_state is dropped so stale side
        observations from the OLD line can't fabricate a fake crossing
        against the NEW line; the counter itself is preserved (session
        totals keep accumulating across edits). The per-tid cooldown
        map is dropped too - it's per-line by design."""
        if now < self._next_line_check:
            return
        self._next_line_check = now + LINE_RELOAD_POLL_S
        mtime = self._line_json_mtime()
        if mtime == self._line_mtime:
            return
        self._line_mtime = mtime
        from app.cameras import (resolve_line as _resolve_line,
                                 resolve_line_classes as _resolve_classes)
        new_line = (_resolve_line(self.cam_id)
                    or self.cam.get("line") or DEFAULT_LINE)
        new_classes = _resolve_classes(self.cam_id)
        if new_line != self.line or new_classes != self.line_classes:
            self.line = new_line
            self.line_classes = new_classes
            self._line_sides.clear()
            self._last_cross_ts.clear()

    def _accumulate(self, frame, boxes, now: float) -> None:
        # First tick has no prior timestamp to measure against, so it
        # borrows the pacing target instead of the old arbitrary 1.0 - the
        # boot sample now weighs about as much as a normal tick (TICK_TARGET_S
        # is the pacing floor the run loop already sleeps to). Later ticks
        # use the real elapsed time, clamped so a long stall doesn't inflate
        # one bin.
        frame_shape = frame.shape
        if self._last_tick is None:
            w = TICK_TARGET_S
        else:
            w = min(5.0, max(0.2, now - self._last_tick))
        self._last_tick = now
        if self.heat_since is None:
            self.heat_since = now
        # Half-life decay before banking the new tick: the heatmap reads
        # as RECENT activity (industry "decay factor" / windowed-view
        # pattern) instead of an ever-brightening all-time integral in
        # which one busy corner eventually crushes the whole colormap.
        if w > 0:
            decay = 0.5 ** (w / HEAT_HALF_LIFE_S)
            for row in self.heat:
                for gx in range(len(row)):
                    row[gx] *= decay
        bump_heat(self.heat, boxes, frame_shape, w)
        self._maybe_reload_line(now)
        self._maybe_reload_zones(now)
        # Persist an event per crossing (bounded JSONL + optional snap).
        # The dashboard's Line layer polls /api/crossings?cam=<id> for the
        # toast + history strip. Frame is passed so the crop of the mover
        # captures the moment they crossed.
        def _on_cross(direction, track, frame, cam_id):
            log_crossing_event(cam_id, direction, track, frame=frame)
        update_crossings(self._line_sides, self.tracker.open, frame_shape,
                         self.line, self.cross,
                         on_event=_on_cross, frame=frame,
                         cam_id=self.cam_id,
                         classes=self.line_classes,
                         last_cross_ts=self._last_cross_ts, now=now)

    def _trim(self) -> None:
        for tr in self.tracker.open:
            if len(tr.boxes) > TRACK_KEEP:
                del tr.boxes[:-TRACK_KEEP]
                del tr.times[:-TRACK_KEEP]
        # Retired tracks are never revisited live - drop them.
        if self.tracker.done:
            self.tracker.done.clear()
        if len(self._line_sides) > 256 or len(self._track_gestures) > 256:
            keep = {tr.tid for tr in self.tracker.open}
            for store in (self._line_sides, self._track_gestures):
                for k in list(store):
                    if k not in keep:
                        store.pop(k, None)

    def _render(self, frame, faces_list: list[dict], layer: str):
        img = frame.copy()
        open_now = list(self.tracker.open)
        # Same display gates as the canvas JSON path - the JPEG fallback
        # used to draw every open track (0.16-conf ghosts included).
        visible = [tr.boxes[-1] for tr in open_now
                   if not tr.misses
                   and tr.hits >= (1 if layer == "paths"
                                   else DISPLAY_MIN_HITS)
                   and max(float(b.get("conf") or 0)
                           for b in tr.boxes[-2:]) >= DISPLAY_MIN_CONF
                   and tr.cls not in DISPLAY_CLASS_BLACKLIST]
        stats_by_id: dict[int, dict] = {}
        if layer in ("paths", "gestures", "body"):
            from app.behavior import track_stats
            riders = _rider_person_tids(open_now)
            for tr in open_now:
                if tr.misses:
                    continue
                row = track_stats(tr.cls, tr.boxes, tr.times, frame.shape)
                row["id"] = tr.tid
                if (layer in ("gestures", "body")
                        and not (tr.cls == "person" and tr.tid in riders)):
                    # behavior_labels removed with Category C; keep gestures
                    # + kinematic stats without the running/erratic labels.
                    from app.gestures import detect_gestures
                    kseq = [b.get("kps") for b in tr.boxes[-16:]]
                    has_kps = any(kseq)
                    row["gestures"] = detect_gestures(kseq) if has_kps else []
                    for g in row["gestures"]:
                        seen = self._track_gestures.setdefault(tr.tid, set())
                        if g not in seen:
                            seen.add(g)
                            self.gesture_counts[g] = \
                                self.gesture_counts.get(g, 0) + 1
                stats_by_id[tr.tid] = row
        if layer == "paths":
            return draw_paths_layer(img, open_now, visible, stats_by_id)
        if layer == "pose":
            return draw_pose_layer(img, visible)
        if layer == "gestures":
            return draw_gestures_layer(img, visible, stats_by_id,
                                       self.gesture_counts)
        if layer == "body":
            # Reuse whichever tids already tripped the sudden-motion gate
            # during this session (populated by _sudden_motion_check inside
            # the JSON publish for the same tick).
            sudden_tids = {tid for tid, ts in
                           getattr(self, "_body_kp_flag_ts", {}).items()
                           if time.time() - ts < BODY_SUDDEN_COOLDOWN_S}
            # Fall suspects paint red through the same drawer - fall
            # detection lives INSIDE the Body layer (operator direction).
            sudden_tids |= {tid for tid, st in
                            getattr(self, "_fall_state", {}).items()
                            if time.time() - st[2] < FALL_COOLDOWN_S}
            return draw_body_layer(img, visible, stats_by_id, sudden_tids)
        if layer == "faces":
            return draw_faces_layer_img(img, faces_list,
                                        available=bool(self._faces_ok))
        if layer == "heat":
            return draw_heat_layer(img, self.heat, since=self.heat_since)
        if layer == "line":
            return draw_line_layer(img, self.line, self.cross)
        if layer == "fire":
            hits = getattr(self, "_fire_hits", []) or []
            confirmed = bool(getattr(self, "_fire_confirmed", False))
            model_err = _FIRE_MODEL_CACHE.get("err") if not \
                _FIRE_MODEL_CACHE.get("model") else None
            return draw_fire_layer(img, hits, confirmed, model_err)
        if layer == "parking":
            _lo, pk, _dwell = self._zone_stats(frame.shape)
            out = draw_zones_layer(img, pk, "parking")
            # If auto-parking is still bootstrapping and no spots exist
            # yet, overwrite the default "nothing drawn yet" caption with
            # progress info so the operator sees what is happening (no
            # more silent 3-minute wait).
            if not pk:
                ap = getattr(self, "_auto_parking", None)
                if ap is not None:
                    st = ap.status(time.time())
                    if not st["done"]:
                        el = int(st["elapsed"])
                        rem = int(st["remaining"])
                        return _caption(out, [
                            f"Parking auto-detect - observing stationary "
                            f"vehicles ({el}s elapsed, ~{rem}s remaining, "
                            f"{st['candidate_cells']} candidate cells so far)"])
                    else:
                        return _caption(out, [
                            "Parking auto-detect - bootstrap done but no "
                            "spots emerged (no vehicles parked long enough "
                            "during the window). Move the camera or draw "
                            "manual polygons via the Draw zones button."])
            return out
        if layer == "plates":
            # 2026-08-17: LPR now attempts on any stream kind (including
            # YouTube). Compressed pixels often defeat OCR - the caption
            # inside draw_plates_layer says "X in range / Y read" so the
            # operator can see the attempt rate vs the success rate live.
            return draw_plates_layer(img, visible)
        return img

    def _fire_pass(self, frame) -> None:
        """Run the dedicated fire/smoke detector on the current frame and
        update this session's confirmation streak. Keeps zero state when
        the fire model is unavailable so the layer degrades to an honest
        "model not loaded" caption without crashing the session."""
        try:
            hits = run_fire_inference(frame, conf=FIRE_CONF)
        except Exception as e:  # noqa: BLE001 - keep the tick alive
            if not getattr(self, "_fire_err_once", False):
                self._fire_err_once = True
                print(f"live-analysis {self.cam_id}: fire pass disabled "
                      f"({type(e).__name__}: {e})")
            hits = []
        self._fire_hits = hits
        if hits:
            self._fire_streak = int(getattr(self, "_fire_streak", 0)) + 1
        else:
            self._fire_streak = 0
        self._fire_confirmed = self._fire_streak >= FIRE_CONFIRM_TICKS

    def _auto_parking_tick(self, frame_shape: tuple, now: float) -> None:
        """Feed one tick's stationary vehicles into the bootstrap grid.
        When the operator has already drawn parking polygons manually,
        auto-detection is skipped entirely (manual wins). Otherwise:
        sample -> emerge (after ~3 min) -> the inferred spots are
        merged into self.zones for the standard draw + occupancy path.
        """
        manual_parking = any(z.get("kind") == "parking"
                              and not z.get("auto")
                              for z in (self.zones or []))
        if manual_parking:
            return
        ap = getattr(self, "_auto_parking", None)
        if ap is None or self.tracker is None:
            return
        ap.sample(self.tracker.open, frame_shape, now)
        emerged = ap.to_zones(now)
        if emerged:
            # Idempotent merge: replace any existing auto entries with
            # the (possibly grown) fresh set. Manual zones are already
            # ruled out above so no interleave conflict.
            self.zones = [z for z in (self.zones or []) if not z.get("auto")]
            self.zones.extend(emerged)

    def _parking_probe(self, frame) -> None:
        """Trackerless occupancy assist (parking layer only): re-detect
        vehicles inside each parking polygon on a 2x-upscaled crop of the
        spot. A parked scooter that never becomes a confirmed track still
        shows up here; fresh hits feed _zone_stats' per-spot hysteresis
        exactly like track candidates."""
        now = time.time()
        if now < getattr(self, "_park_probe_next", 0):
            return
        self._park_probe_next = now + PARKING_PROBE_EVERY_S
        zones = [(zi, z) for zi, z in enumerate(self.zones or [])
                 if z.get("kind") == "parking"]
        if not zones:
            return
        import cv2
        from app.detect_core import (CLASSES_OF_INTEREST, NAME_BY_ID,
                                     _PREDICT_LOCK)
        H, W = frame.shape[:2]
        veh_ids = [v for k, v in CLASSES_OF_INTEREST.items()
                   if k in _VEHICLE_CLASSES]
        hits: dict = {}
        for zi, z in zones:
            xs = [p[0] for p in z["points"]]
            ys = [p[1] for p in z["points"]]
            x1 = max(0, int(min(xs) * W) - 12)
            x2 = min(W, int(max(xs) * W) + 12)
            y1 = max(0, int(min(ys) * H) - 12)
            y2 = min(H, int(max(ys) * H) + 12)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or min(crop.shape[:2]) < 16:
                continue
            crop2 = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2),
                               interpolation=cv2.INTER_CUBIC)
            try:
                with _PREDICT_LOCK:
                    res = self.model.predict(
                        crop2, imgsz=320, conf=PARKING_PROBE_CONF,
                        classes=veh_ids, verbose=False)[0]
            except Exception:
                continue
            if res.boxes is None:
                continue
            for bb, ci in zip(res.boxes.xyxy.tolist(),
                              res.boxes.cls.tolist()):
                # center: crop2 coords /2 back to crop, then + crop origin
                cx = x1 + (bb[0] + bb[2]) / 4.0
                cy = y1 + (bb[1] + bb[3]) / 4.0
                if _pt_in_poly(cx / W, cy / H, z["points"]):
                    hits[zi] = NAME_BY_ID.get(int(ci), "vehicle")
                    break
        self._park_probe = {"ts": now, "hits": hits}

    def _zone_stats(self, frame_shape):
        """Occupancy + dwell for loiter zones, occupancy for parking spots,
        computed from the tracker's confirmed tracks. Cached per tick
        (keyed on the frame's capture stamp) so the JPEG render and the
        JSON publish share ONE computation instead of walking every
        track against every polygon twice."""
        key = getattr(self, "_last_frame_ts", None)
        if key is not None and getattr(self, "_zone_cache_key", None) == key:
            return self._zone_cache
        H, W = int(frame_shape[0]), int(frame_shape[1])
        now_t = time.time()
        loiter, parking = [], []
        for zi, z in enumerate(self.zones or []):
            entry = {"name": z.get("name") or f"Z{zi + 1}",
                     "points": z["points"]}
            if z.get("kind") == "parking":
                entry.update(occupied=False, cls=None)
                parking.append((zi, entry))
            else:
                entry.update(count=0, max_dwell=0.0, alert=False,
                             dwell_s=float(z.get("dwell_s")
                                           or DEFAULT_LOITER_DWELL_S))
                loiter.append((zi, entry))
        if not hasattr(self, "_zone_streak"):
            self._zone_streak: dict[tuple, int] = {}
            self._zone_last_seen: dict[tuple, float] = {}
            self._spot_state: dict[int, dict] = {}
            # 2026-08-17: per (tid, zone_idx) - has this person been
            # flagged INSIDE this parking spot yet? Prevents an alert
            # storm on every tick the person stays inside.
            self._parking_person_alerted: set[tuple] = set()
        dwell_by_tid: dict[int, float] = {}
        active: set[tuple] = set()
        spot_cand: dict[int, str] = {}   # zone_idx -> vehicle cls this tick
        for tr in (self.tracker.open if self.tracker else []):
            if tr.misses > DISPLAY_MAX_MISSES or tr.hits < DISPLAY_MIN_HITS:
                continue
            b = tr.boxes[-1]
            if tr.cls == "person" and loiter:
                cx = ((b["x1"] + b["x2"]) / 2) / W
                by = b["y2"] / H            # feet, not head
                for zi, e in loiter:
                    if _pt_in_poly(cx, by, e["points"]):
                        # zkey, NOT key: `key` above is the per-tick cache
                        # key stored at the end - shadowing it here made
                        # the cache miss whenever someone stood in a zone
                        # (render+publish recomputed, double-stepping the
                        # entry-debounce streak).
                        zkey = (tr.tid, zi)
                        active.add(zkey)
                        self._zone_last_seen[zkey] = now_t
                        streak = self._zone_streak.get(zkey, 0) + 1
                        self._zone_streak[zkey] = streak
                        # Entry debounce (Frigate inertia / Bosch
                        # debounce): the dwell clock arms only on the
                        # SECOND consecutive tick inside - one grazing
                        # tick never starts a loitering countdown.
                        if streak < 2:
                            e["count"] += 1
                            continue
                        dw = now_t - self._zone_since.setdefault(zkey, now_t)
                        e["count"] += 1
                        e["max_dwell"] = max(e["max_dwell"], dw)
                        if dw >= e["dwell_s"]:
                            e["alert"] = True
                        dwell_by_tid[tr.tid] = max(
                            dwell_by_tid.get(tr.tid, 0.0), dw)
            # 2026-08-17: person entering a parking polygon fires an
            # explicit alert (operator asked for this - a person walking
            # into a marked parking zone is exactly the "someone messing
            # with the parked cars" trigger the layer is meant to catch).
            # Alert fires ONCE per (person_tid, spot) so a lingering
            # person doesn't spam the event log.
            if tr.cls == "person" and parking:
                fx = ((b["x1"] + b["x2"]) / 2) / W
                fy = b["y2"] / H
                for zi, e in parking:
                    if _pt_in_poly(fx, fy, e["points"]):
                        e["person_alert"] = True
                        key_pp = (tr.tid, zi)
                        if key_pp not in self._parking_person_alerted:
                            self._parking_person_alerted.add(key_pp)
                            try:
                                self._emit_event("parking",
                                    f"person #{tr.tid} entered spot "
                                    f"'{e['name']}'", b)
                            except Exception:
                                pass  # never let the emit failure kill the tick
            if tr.cls in _VEHICLE_CLASSES and parking:
                # Industry association: substantial AREA overlap with the
                # spot (>=30% of the spot covered), argmax spot per
                # vehicle - never the center-point test that let passing
                # traffic "occupy" a shopfront. Plus a stationarity gate:
                # only a vehicle that has been near-still for a while
                # can PARK (displacement under ~35% of its own diagonal
                # over a 45s+ track).
                bn = (b["x1"] / W, b["y1"] / H, b["x2"] / W, b["y2"] / H)
                span = tr.times[-1] - tr.times[0]
                c0 = ((tr.boxes[0]["x1"] + tr.boxes[0]["x2"]) / 2,
                      (tr.boxes[0]["y1"] + tr.boxes[0]["y2"]) / 2)
                c1 = ((b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2)
                disp = ((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2) ** 0.5
                diag = ((b["x2"] - b["x1"]) ** 2
                        + (b["y2"] - b["y1"]) ** 2) ** 0.5
                stationary = span >= 45 and disp < 0.35 * max(diag, 1)
                if not stationary:
                    continue
                best_zi, best_ov = None, 0.0
                for zi, e in parking:
                    ov = box_overlap_over_spot(bn, e["points"])
                    if ov > best_ov:
                        best_zi, best_ov = zi, ov
                if best_zi is not None and best_ov >= 0.30:
                    spot_cand[best_zi] = tr.cls
        # Trackerless probe assist (_parking_probe): a fresh probe hit
        # counts as a candidate for the spot exactly like a track - this
        # is what lets a parked scooter with no confirmed track occupy
        # its spot.
        pp = getattr(self, "_park_probe", None)
        if pp and now_t - pp.get("ts", 0) <= PARKING_PROBE_FRESH_S:
            for zi_p, cls_p in (pp.get("hits") or {}).items():
                spot_cand.setdefault(zi_p, cls_p)
        # Per-spot asymmetric hysteresis: 2 consecutive positive ticks to
        # flip OCCUPIED, 4 consecutive negatives to flip back - night
        # detector flicker must not toggle a spot, and a missed
        # detection is weak evidence of vacancy.
        for zi, e in parking:
            st = self._spot_state.setdefault(
                zi, {"occ": False, "pos": 0, "neg": 0, "cls": None,
                     "ever": False})
            if zi in spot_cand:
                st["pos"] += 1
                st["neg"] = 0
                st["cls"] = spot_cand[zi]
                st["ever"] = True
                if st["pos"] >= 2:
                    st["occ"] = True
            else:
                st["neg"] += 1
                st["pos"] = 0
                if st["neg"] >= 4:
                    st["occ"] = False
            e["occupied"] = st["occ"]
            e["cls"] = st["cls"] if st["occ"] else None
            e["seen_vehicle"] = st["ever"]
        # Loiter clocks survive a short track loss (grace) so one missed
        # tick doesn't reset a 25s dwell; a real exit (grace expired)
        # clears the clock and the streak.
        GRACE_S = 12.0
        for zkey in list(self._zone_since.keys()):
            if zkey not in active and \
                    now_t - self._zone_last_seen.get(zkey, 0) > GRACE_S:
                self._zone_since.pop(zkey, None)
                self._zone_streak.pop(zkey, None)
                self._zone_last_seen.pop(zkey, None)
        for zkey in list(self._zone_streak.keys()):
            if zkey not in active and \
                    now_t - self._zone_last_seen.get(zkey, 0) > GRACE_S:
                self._zone_streak.pop(zkey, None)
        result = ([e for _, e in loiter], [e for _, e in parking],
                  dwell_by_tid)
        self._zone_cache_key = key
        self._zone_cache = result
        return result

    def _fall_check(self, tid: int, kps, box: tuple, now: float) -> bool:
        """True on the tick a track is confirmed as a fall suspect (D4).

        Posture evidence per tick: torso lean (pose path) or a
        wider-than-tall bbox (fallback path), counted only after the
        track has a short upright history. FALL_CONFIRM_TICKS
        consecutive lying ticks confirm; a per-tid cooldown stops
        re-alerting on someone who stays down."""
        if not hasattr(self, "_fall_state"):
            # tid -> [upright_ticks, lying_streak, last_flag_ts]
            self._fall_state: dict[int, list] = {}
        st = self._fall_state.setdefault(tid, [0, 0, 0.0])
        x1, y1, x2, y2 = box
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        lying = w / h >= FALL_ASPECT_W_OVER_H
        if kps and len(kps) > 12:
            try:
                sh = [(kps[5][0] + kps[6][0]) / 2.0,
                      (kps[5][1] + kps[6][1]) / 2.0]
                hp = [(kps[11][0] + kps[12][0]) / 2.0,
                      (kps[11][1] + kps[12][1]) / 2.0]
                ok_conf = (min(kps[5][2], kps[6][2], kps[11][2],
                               kps[12][2]) >= BODY_SUDDEN_MIN_CONF)
                if ok_conf:
                    import math
                    dx = hp[0] - sh[0]
                    dy = hp[1] - sh[1]
                    # angle from vertical: 0 = standing, 90 = horizontal
                    ang = math.degrees(math.atan2(abs(dx), abs(dy) or 1e-6))
                    lying = ang >= FALL_ANGLE_DEG
            except Exception:
                pass
        if not lying:
            st[0] += 1          # accumulating upright history
            st[1] = 0
            return False
        if st[0] < FALL_UPRIGHT_MIN_TICKS:
            return False        # never seen upright - could be a bench
        st[1] += 1
        if st[1] < FALL_CONFIRM_TICKS:
            return False
        if now - st[2] < FALL_COOLDOWN_S:
            return False
        st[2] = now
        return True

    def _bbox_sudden_check(self, tid: int, box: tuple, now: float) -> bool:
        """Fallback sudden-motion detector when pose keypoints are absent.

        Watches the bbox centroid. A track whose centroid darts by
        BODY_BBOX_SUDDEN_FRAC of its own diagonal from
        one tick to the next - AFTER at least a few baseline samples -
        counts as sudden motion. Catches people RUNNING / FLEEING that
        pose can't see at their box size.
        """
        if not hasattr(self, "_body_bbox_hist"):
            self._body_bbox_hist: dict[int, list[float]] = {}
            self._body_bbox_prev: dict[int, tuple] = {}
        prev = self._body_bbox_prev.get(tid)
        self._body_bbox_prev[tid] = box
        x1, y1, x2, y2 = box
        diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if diag < 8 or prev is None:
            return False
        cx0 = (prev[0] + prev[2]) / 2
        cy0 = (prev[1] + prev[3]) / 2
        cx1 = (x1 + x2) / 2
        cy1 = (y1 + y2) / 2
        hop_frac = (((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2) ** 0.5) / diag
        ring = self._body_bbox_hist.setdefault(tid, [])
        ring.append((now, hop_frac))
        _cap = max(BODY_SUDDEN_RING, BODY_BBOX_SUDDEN_MIN_SAMPLES)
        if len(ring) > _cap:
            del ring[:-_cap]
        # Sparse-sampling guard: without enough samples over enough real
        # time a "hop" is just two frames far apart, not fast motion.
        if len(ring) < BODY_BBOX_SUDDEN_MIN_SAMPLES:
            return False
        if now - ring[0][0] < BODY_BBOX_WINDOW_S:
            return False
        trip = hop_frac >= BODY_BBOX_SUDDEN_FRAC
        if not hasattr(self, "_body_bbox_streak"):
            self._body_bbox_streak = {}
        self._body_bbox_streak[tid] = (
            self._body_bbox_streak.get(tid, 0) + 1 if trip else 0)
        if not trip or self._body_bbox_streak[tid] < BODY_SUDDEN_STREAK:
            return False
        if not hasattr(self, "_body_kp_flag_ts"):
            self._body_kp_flag_ts = {}
        last_flag = self._body_kp_flag_ts.get(tid, 0.0)
        if now - last_flag < BODY_SUDDEN_COOLDOWN_S:
            return False
        self._body_kp_flag_ts[tid] = now
        if not hasattr(self, "_body_flag_speed"):
            self._body_flag_speed = {}
        self._body_flag_speed[tid] = hop_frac
        return True

    def _sudden_motion_check(self, tid: int, kps: list,
                              box: tuple, now: float) -> bool:
        """True on the tick a track's wrist/ankle keypoint velocity spikes
        far above its own recent baseline (theft snatch, punch, kick, run
        burst). Per-tid ring of the last BODY_SUDDEN_RING mean-hop
        displacements between the previous kp position and the newest one,
        measured across the four fast-limb keypoints (wrists, ankles) in
        normalized (box-diagonal) units so a person 30 px tall in the
        distance and one 400 px tall up close are comparable. Fires only
        when both the ratio AND the absolute floor gates trip, plus a
        per-tid cooldown so one anomaly doesn't paint every following
        tick red until the streak decays."""
        # Per-dict init: _body_kp_flag_ts is SHARED with _bbox_sudden_check
        # (bbox fallback path) which also flags fighting. Guarding the three
        # dicts under a single hasattr on _body_kp_hist wiped the flag_ts
        # dict on the first pose-path call, silently dropping every fighting
        # event whose first-seen track had no pose keypoints.
        if not hasattr(self, "_body_kp_hist"):
            self._body_kp_hist: dict[int, list[float]] = {}
        if not hasattr(self, "_body_kp_prev"):
            self._body_kp_prev: dict[int, list] = {}
        if not hasattr(self, "_body_kp_flag_ts"):
            self._body_kp_flag_ts: dict[int, float] = {}
        prev = self._body_kp_prev.get(tid)
        self._body_kp_prev[tid] = kps
        if not prev or len(prev) != len(kps):
            return False
        x1, y1, x2, y2 = box
        diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if diag < 8:
            return False
        hops = []
        for idx in BODY_SUDDEN_KP_IDX:
            if idx >= len(kps) or idx >= len(prev):
                continue
            p, q = prev[idx], kps[idx]
            if (not p or not q or len(p) < 3 or len(q) < 3
                    or p[2] < BODY_SUDDEN_MIN_CONF
                    or q[2] < BODY_SUDDEN_MIN_CONF):
                continue
            dx = q[0] - p[0]
            dy = q[1] - p[1]
            hops.append(((dx * dx + dy * dy) ** 0.5) / diag)
        if not hops:
            return False
        mean_hop = sum(hops) / len(hops)
        ring = self._body_kp_hist.setdefault(tid, [])
        ring.append(mean_hop)
        if len(ring) > BODY_SUDDEN_RING:
            del ring[:-BODY_SUDDEN_RING]
        # Need at least a few observations for a meaningful baseline.
        if len(ring) < BODY_SUDDEN_MIN_SAMPLES:
            return False
        # Median of the ring EXCLUDING this newest hop is the baseline.
        baseline_sorted = sorted(ring[:-1])
        median = baseline_sorted[len(baseline_sorted) // 2] or 1e-6
        trip = (mean_hop >= BODY_SUDDEN_FLOOR
                and mean_hop >= BODY_SUDDEN_RATIO * median)
        if not hasattr(self, "_body_kp_streak"):
            self._body_kp_streak = {}
        self._body_kp_streak[tid] = (
            self._body_kp_streak.get(tid, 0) + 1 if trip else 0)
        if not trip or self._body_kp_streak[tid] < BODY_SUDDEN_STREAK:
            return False
        last_flag = self._body_kp_flag_ts.get(tid, 0.0)
        if now - last_flag < BODY_SUDDEN_COOLDOWN_S:
            return False
        self._body_kp_flag_ts[tid] = now
        if not hasattr(self, "_body_flag_speed"):
            self._body_flag_speed = {}
        self._body_flag_speed[tid] = mean_hop
        return True

    def _publish(self, img) -> None:
        import cv2
        H, W = img.shape[:2]
        if W > JPEG_MAX_W:
            img = cv2.resize(img, (JPEG_MAX_W, int(H * JPEG_MAX_W / W)),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            _now = time.time()
            with self.lock:
                self.latest = buf.tobytes()
                self.seq += 1
                self.note = ""
                # Replay ring (D3): throttled copy of the published frame.
                if not hasattr(self, "replay_ring"):
                    from collections import deque as _dq
                    self.replay_ring = _dq(maxlen=REPLAY_RING_FRAMES)
                    self._replay_last = 0.0
                if _now - self._replay_last >= 1.0 / REPLAY_FPS:
                    self._replay_last = _now
                    self.replay_ring.append((_now, self.latest))

    def _publish_note(self, note: str) -> None:
        with self.lock:
            self.note = note

    def _publish_data(self, frame_shape, boxes, layer: str,
                      faces_list: list[dict] | None = None) -> None:
        """Snapshot the just-inferred tick as JSON for the overlay canvas.

        Boxes come from the TRACKER, not the raw detections: only objects
        confirmed across DISPLAY_MIN_HITS ticks are published, each with
        its track id and centroid velocity (px/s) so the client can
        extrapolate positions between ticks and glide boxes with the
        video instead of letting them sit on vacated pixels. `at` is the
        capture time translated into the stream's PROGRAM-DATE-TIME clock
        (when known), which is the clock hls.js reports for the frame the
        operator is currently watching.
        """
        H, W = int(frame_shape[0]), int(frame_shape[1])
        js_boxes = []
        # Paths is the MOVERS layer: at 5-8s ticks under 4-session load a
        # motorcycle crosses the whole view inside one or two ticks, so
        # the 2-hit confirmation gate deleted exactly the objects the
        # layer exists for. First-tick display there; everything else
        # keeps the flicker-suppressing 2-hit gate.
        min_hits = 1 if layer == "paths" else DISPLAY_MIN_HITS
        riders_pub = (_rider_person_tids(self.tracker.open)
                      if self.tracker and layer in ("body", "gestures")
                      else set())
        for tr in (self.tracker.open if self.tracker else []):
            last = tr.boxes[-1]
            conf = max(float(b.get("conf") or 0) for b in tr.boxes[-2:])
            if (tr.hits < min_hits
                    or tr.misses > DISPLAY_MAX_MISSES
                    or conf < DISPLAY_MIN_CONF
                    or (tr.cls or "?") in DISPLAY_CLASS_BLACKLIST):
                continue
            jb = {
                "tid": tr.tid,
                "x1": int(last.get("x1", 0)),
                "y1": int(last.get("y1", 0)),
                "x2": int(last.get("x2", 0)),
                "y2": int(last.get("y2", 0)),
                "cls": tr.cls or "?",
                "conf": round(conf, 3),
                "vx": round(float(tr.vx), 1),
                "vy": round(float(tr.vy), 1),
            }
            if tr.misses:
                # Coasting through a missed detection - the client draws
                # it dashed so a box gliding on prediction alone is
                # visually distinct from an observed one.
                jb["coast"] = tr.misses
            # Layer-specific extras ride on each box so the canvas can
            # draw the REAL layer, not just generic rectangles - this
            # was the "analysis is not logically right" gap: trails,
            # skeletons, gesture chips and anomaly flags existed only
            # inside the server-rendered JPEG.
            if layer == "paths":
                jb["trail"] = [
                    [int((b["x1"] + b["x2"]) / 2),
                     int((b["y1"] + b["y2"]) / 2)]
                    for b in tr.boxes[-12:]]
                # Relative speed tiers instead of km/h: without a
                # ground-plane calibration a km/h number is a guess with
                # 20-30% scale error baked in (dimension-prior variance),
                # and it printed absurdities like 0.1 km/h on parked
                # bikes. Speed in BODY LENGTHS per second is
                # perspective-robust and honest.
                spd = (tr.vx ** 2 + tr.vy ** 2) ** 0.5
                diag = ((last["x2"] - last["x1"]) ** 2
                        + (last["y2"] - last["y1"]) ** 2) ** 0.5
                blps = spd / max(diag, 1.0)
                # "static" is a claim about a TRACK, not a frame: it
                # needs 3 confirmations AND 4s of observed age, or a
                # rider matched for two ticks at similar spots gets
                # branded static mid-ride.
                age = tr.times[-1] - tr.times[0] if len(tr.times) > 1 else 0
                if blps < 0.05 and tr.hits >= 3 and age >= 4:
                    jb["tier"] = "static"
                elif blps < 0.25:
                    jb["tier"] = "slow"
                elif blps < 0.8:
                    jb["tier"] = "moving"
                else:
                    jb["tier"] = "fast"
                # Also expose a numeric speed the frontend can print
                # next to the tier chip (2026-08-16 - operator asked for
                # a speed number, not just a category). blps is honest
                # even without ground-plane calibration; px/s is exposed
                # too for those who want a raw pixel rate.
                jb["speed_blps"] = round(float(blps), 2)
                jb["speed_pxs"] = int(spd)
            elif layer == "plates":
                if last.get("plate"):
                    jb["plate"] = last["plate"]
                    jb["plate_conf"] = last.get("plate_conf")
            elif layer in ("pose", "gestures", "body"):
                kps = last.get("kps")
                # COCO keypoints are only annotated for medium+ people;
                # below KPS_MIN_BOX_H skeletons are guesswork, so the
                # envelope gate simply withholds them (the note in the
                # payload tells the operator how many were gated).
                if kps and (jb["y2"] - jb["y1"]) >= KPS_MIN_BOX_H:
                    jb["kps"] = [[int(k[0]), int(k[1]), round(k[2], 2)]
                                 for k in kps]
                if layer == "gestures" and tr.cls == "person" \
                        and jb.get("kps"):
                    # Static single-frame postures. Sequence-based hand
                    # gestures are PHYSICALLY absent at 0.2-1 fps (a 1-2
                    # Hz wave aliases to noise below ~4 fps), so the
                    # layer detects what one frame can prove: a raised
                    # hand (wrist above shoulder with confident arm
                    # keypoints) - the analytic vendors actually ship.
                    g = _static_postures(jb["kps"])
                    if g:
                        jb["gestures"] = g
                if (layer == "body" and tr.cls == "person"
                        and tr.tid not in riders_pub):
                    try:
                        # Fall check first (operator: fall detection is
                        # PART of the Body layer, not its own entry) -
                        # a confirmed fall outranks a sudden-motion tag.
                        _bbx0 = (jb["x1"], jb["y1"], jb["x2"], jb["y2"])
                        if self._fall_check(tr.tid, jb.get("kps"),
                                            _bbx0, time.time()):
                            jb["flag"] = "fall_suspect"
                            jb["alert"] = True
                            jb["flags"] = ["fall posture confirmed"]
                    except Exception:
                        pass
                    try:
                        # Sudden-motion gate. Two paths:
                        # (1) with pose keypoints -> per-track ring of
                        #     wrist/ankle hop distances (BODY_SUDDEN_*).
                        #     Fires on martial-arts kicks / punches /
                        #     dance moves / fall bursts.
                        # (2) without keypoints (small / distant person
                        #     that missed pose's crop gate) -> bbox
                        #     centroid velocity fallback. Fires on a
                        #     person suddenly RUNNING or fleeing across
                        #     the frame.
                        # Either path can flag; both share the same
                        # per-tid cooldown so one event never spams.
                        _bbx = (jb["x1"], jb["y1"], jb["x2"], jb["y2"])
                        _now = time.time()
                        flagged = False
                        if jb.get("kps"):
                            flagged = self._sudden_motion_check(
                                tr.tid, jb["kps"], _bbx, _now)
                            reason = "wrist/ankle burst"
                        if not flagged:
                            flagged = self._bbox_sudden_check(
                                tr.tid, _bbx, _now)
                            if flagged:
                                reason = "sudden displacement"
                        if flagged:
                            jb["flag"] = "sudden_motion"
                            jb["alert"] = True
                            jb["flags"] = [reason]
                    except Exception:
                        pass
            js_boxes.append(jb)
        # Fighting pass: any two person tracks that BOTH tripped the
        # sudden-motion gate within the same BODY_FIGHT_PAIR_WINDOW_S
        # AND are within BODY_FIGHT_MAX_DIST_PX centroid-to-centroid,
        # AND both flagged at genuinely high speed (>= the sudden
        # floor) - near-simultaneous violent bursts from BOTH sides,
        # not one runner passing a stander. Upgrades "sudden_motion"
        # to "fighting" so the layer can tell a lone kick from an
        # altercation.
        if layer == "body":
            try:
                _fnow = time.time()
                _speeds = getattr(self, "_body_flag_speed", {})
                sudden_active = {
                    tid for tid, ts in
                    getattr(self, "_body_kp_flag_ts", {}).items()
                    if _fnow - ts < BODY_FIGHT_PAIR_WINDOW_S
                    and _speeds.get(tid, 0.0) >= BODY_SUDDEN_FLOOR
                }
                if len(sudden_active) >= 2:
                    person_jbs = [b for b in js_boxes
                                  if b.get("cls") == "person"
                                  and b.get("track_id") in sudden_active]
                    for i, b1 in enumerate(person_jbs):
                        cx1 = (b1["x1"] + b1["x2"]) / 2.0
                        cy1 = (b1["y1"] + b1["y2"]) / 2.0
                        for b2 in person_jbs[i + 1:]:
                            cx2 = (b2["x1"] + b2["x2"]) / 2.0
                            cy2 = (b2["y1"] + b2["y2"]) / 2.0
                            d = ((cx1 - cx2) ** 2
                                 + (cy1 - cy2) ** 2) ** 0.5
                            if d <= BODY_FIGHT_MAX_DIST_PX:
                                for b in (b1, b2):
                                    b["flag"] = "fighting"
                                    b["alert"] = True
                                    b["flags"] = ["pair burst < "
                                                  f"{BODY_FIGHT_MAX_DIST_PX}px"]
            except Exception:
                pass
        cap_ts = getattr(self, "_last_frame_ts", None) or time.time()
        # The ytproxy measures the offset under the CATALOG id (that is the
        # ?cam= it serves); for a local-picker slot self.cam_id is the slot
        # id, so look up by stream_key first or the measured value is
        # silently ignored and the 3.0 s default always wins.
        # Default 0.0 (was 3.0): for the iframe path we don't route through
        # /ytproxy, so STREAM_PDT_OFFSET is never populated for these cams -
        # and the 3-second fixed subtraction offset `at` earlier than the
        # actual capture wall clock, which the operator saw as boxes
        # perpetually 3 s ahead of the moving object. When the hls.js path
        # IS used, /ytproxy measures the real PDT offset and overwrites
        # this default within one playlist refresh.
        pdt_off = STREAM_PDT_OFFSET.get(
            self.stream_key, STREAM_PDT_OFFSET.get(self.cam_id, 0.0))
        data: dict = {
            "seq": self.seq + 1,        # matches _publish's post-bump seq
            "layer": layer,
            "frame_w": W,
            "frame_h": H,
            # capture time on the video's own clock; clamp the measured
            # ingest offset to something sane so one bad manifest parse
            # can't shove every box seconds off.
            "at": round(cap_ts - min(15.0, max(0.0, pdt_off)), 3),
            "boxes": js_boxes,
            "person": sum(1 for b in (boxes or [])
                          if b.get("cls") == "person"),
            "vehicles": sum(1 for b in (boxes or [])
                            if b.get("cls") in ("car", "truck", "bus",
                                                "motorcycle", "bicycle")),
        }
        if layer == "heat":
            # Snapshot copy (list-of-list clone) so JSON serialization
            # never sees a mid-mutation view from the tick loop that
            # runs concurrently with the /api/analysis/data poll. Plain
            # lists of floats - cheap.
            data["heat"] = [list(row) for row in self.heat]
            data["heat_peak"] = max((max(row) for row in self.heat),
                                     default=0.0)
            data["heat_nonzero"] = sum(1 for row in self.heat
                                        for v in row if v > 0)
        if layer == "line":
            data["line"] = self.line
            data["cross"] = dict(self.cross)
        if layer == "gestures" and self.gesture_counts:
            data["gesture_counts"] = dict(self.gesture_counts)
        if layer == "fire":
            hits = getattr(self, "_fire_hits", []) or []
            data["fire"] = {
                "hits": [{"x1": h["x1"], "y1": h["y1"],
                          "x2": h["x2"], "y2": h["y2"],
                          "cls": h.get("cls", "fire"),
                          "conf": round(float(h.get("conf") or 0), 3)}
                         for h in hits],
                "confirmed": bool(getattr(self, "_fire_confirmed", False)),
                "streak": int(getattr(self, "_fire_streak", 0)),
                "err": (_FIRE_MODEL_CACHE.get("err")
                        if not _FIRE_MODEL_CACHE.get("model") else None),
            }
        if layer == "parking":
            _lo, pk, _dwell = self._zone_stats(frame_shape)
            data["spots"] = pk
            data["parking"] = {
                "total": len(pk),
                "occupied": sum(1 for e in pk if e["occupied"])}
        if layer == "faces":
            data["faces"] = [
                {"x1": int(f["x1"]), "y1": int(f["y1"]),
                 "x2": int(f["x2"]), "y2": int(f["y2"]),
                 "conf": round(float(f.get("conf") or 0), 2)}
                for f in (faces_list or [])]
            data["faces_ok"] = bool(self._faces_ok)
        if layer == "plates":
            # 2026-08-17: YouTube gate removed; LPR attempts on every
            # stream kind. Envelope now reports the real counts (vehicles
            # in frame, in plate-range, and actually read) - operator sees
            # attempt vs success live instead of a blanket "disabled".
            from app.plates import MIN_VEHICLE_W, PLATE_VEHICLE_CLASSES
            veh = [b for b in js_boxes
                   if b["cls"] in PLATE_VEHICLE_CLASSES]
            in_range = [b for b in veh
                        if (b["x2"] - b["x1"]) >= MIN_VEHICLE_W]
            read = [b for b in veh if b.get("plate")]
            import os as _os_env
            _langs_env = (_os_env.environ.get("PLATE_OCR_LANGS") or "").lower()
            _extra_langs = [l for l in _langs_env.split(",")
                            if l.strip() and l.strip() not in ("latin", "en")]
            _script_hint = (f"digits+Latin+[{','.join(_extra_langs)}]"
                            if _extra_langs
                            else "digits+Latin; non-Latin scripts disabled "
                                 "(set PLATE_OCR_LANGS=latin,th,ar,ja to enable)")
            data["envelope"] = (
                f"{len(veh)} vehicles · {len(in_range)} in plate range "
                f"(>={MIN_VEHICLE_W}px wide) · {len(read)} read ({_script_hint})")
        # Operating-envelope note per pose-family layer: how many people
        # were in scene vs how many passed the size gates, so an empty
        # overlay reads as an honest "out of range", not a failure.
        if layer in ("pose", "gestures", "body", "faces"):
            persons = [b for b in js_boxes if b["cls"] == "person"]
            with_kps = [b for b in persons if b.get("kps")]
            if layer == "faces":
                pose_derived = sum(1 for f in (faces_list or [])
                                   if f.get("source") == "pose_kps")
                yunet_n = len(faces_list or []) - pose_derived
                tail = (f"({pose_derived} pose-derived fallback)"
                        if pose_derived and not yunet_n
                        else f"({pose_derived} pose-derived fallback)"
                        if pose_derived else "")
                data["envelope"] = (
                    f"{len(persons)} people · {len(faces_list or [])} "
                    f"faces >=16px @conf .6 {tail}").strip()
            else:
                data["envelope"] = (
                    f"{len(persons)} people · skeletons on "
                    f"{len(with_kps)} (>={KPS_MIN_BOX_H}px only)")
        try:
            self._detect_events(js_boxes, layer, data, faces_list)
        except Exception as e:  # events must never cost a tick
            if not getattr(self, "_ev_err_once", False):
                self._ev_err_once = True
                print(f"live-analysis {self.cam_id}: event strip disabled "
                      f"({type(e).__name__}: {e})")
        try:
            self._append_local_history(js_boxes)
        except Exception:
            pass   # history is best-effort; never costs a tick
        # 2026-08-23 (B2): per-stage timing from the last completed tick.
        # Falls back to empty dict on the very first publish.
        if self._last_stage_ms:
            data["_stage_ms"] = self._last_stage_ms
        with self.lock:
            self.latest_data = data

    def _append_local_history(self, js_boxes) -> None:
        """One footfall-history sample per ~30s per camera, written by the
        SESSION: the local producers (the usual writers of these files)
        yield the CPU whenever an analysis session runs, so a dashboard
        that analyzes continuously would otherwise chart nothing."""
        now = time.time()
        if now - getattr(self, "_hist_last", 0) < 30:
            return
        self._hist_last = now
        person = sum(1 for b in js_boxes if b["cls"] == "person")
        vehicles = sum(1 for b in js_boxes
                       if b["cls"] in _VEHICLE_CLASSES)
        base = _SRC_ROOT / "web" / "snapshots" / "model_view"
        base.mkdir(parents=True, exist_ok=True)
        hist = base / f"{self.cam_id}_history.jsonl"
        row = json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                              time.gmtime()),
                          "person": person, "vehicles": vehicles,
                          "ok": True})
        with hist.open("a", encoding="utf-8") as hf:
            hf.write(row + "\n")
        if hist.stat().st_size > 300_000:
            keep = hist.read_text(encoding="utf-8").splitlines()[-2880:]
            hist.write_text("\n".join(keep) + "\n", encoding="utf-8")

    # -- detection event strip (hot feed + save/recall) --------------------
    # Every layer publishes STATE CHANGES (a plate read, a crossing, a
    # loiter alert firing, a spot flipping...) into a bounded ring the
    # client renders as a rolling strip under the video. New events push
    # old ones out; only an explicit save writes the full frame to disk.

    EV_RING = 50

    def _emit_event(self, layer: str, text: str, box: dict | None = None,
                    *, auto_save: bool = False,
                    tight_crop: bool = False):
        """Push one detection event into the session's ring buffer.

        auto_save   - immediately persist the event to the on-disk
                      gallery (Investigation tab); default False so the
                      operator still has to click Save for most layers.
                      Set True by the plates layer for its auto-extract-
                      plate-crop pipeline (2026-08-18).
        tight_crop  - crop stored on disk uses NO padding around the
                      given box (the box IS the object of interest, eg
                      a plate). Default False keeps the historic 25%
                      padding so a whole-vehicle proof crop still shows
                      surrounding context.
        """
        # Persist the raw event to the on-disk sink FIRST - even if the
        # frame is None (a caption-only alert), the event still belongs
        # in the append-only log the CSV export reads from.
        _append_event_jsonl(self.cam_id, layer, text, box)
        import base64
        import cv2
        frame = getattr(self, "_last_frame", None)
        if frame is None:
            return
        H, W = frame.shape[:2]
        # Annotate a COPY so the saved proof is self-contained: the
        # detection box plus a caption bar (what | layer | camera | when)
        # burn into the image itself - a bare frame in a gallery proves
        # nothing.
        annotated = frame.copy()
        if box is not None:
            cv2.rectangle(annotated,
                          (int(box["x1"]), int(box["y1"])),
                          (int(box["x2"]), int(box["y2"])),
                          (80, 220, 80), 3)
        cap = (f"{text} | {LAYER_TITLES.get(layer, layer)} | "
               f"{self.cam_name} | {time.strftime('%H:%M:%S')}")
        (cw_, ch_), _ = cv2.getTextSize(cap, cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, 2)
        cv2.rectangle(annotated, (0, H - ch_ - 16),
                      (min(W, cw_ + 16), H), (42, 23, 15), -1)
        cv2.putText(annotated, cap, (8, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (250, 245, 240), 2,
                    cv2.LINE_AA)
        crop = annotated
        if box is not None:
            # tight_crop=True: the box IS the object (a plate crop).
            # Otherwise pad 25% around it so the gallery thumbnail keeps
            # some context (which car, which side of the frame, etc).
            _pad_frac = 0.0 if tight_crop else 0.25
            bw, bh = box["x2"] - box["x1"], box["y2"] - box["y1"]
            px, py = bw * _pad_frac, bh * _pad_frac
            x1 = max(0, int(box["x1"] - px)); y1 = max(0, int(box["y1"] - py))
            x2 = min(W, int(box["x2"] + px)); y2 = min(H, int(box["y2"] + py))
            if x2 - x1 > 4 and y2 - y1 > 4:
                crop = annotated[y1:y2, x1:x2]
        th = 90
        tw = max(1, min(240, int(crop.shape[1] * th / max(1, crop.shape[0]))))
        thumb = cv2.resize(crop, (tw, th))
        ok1, tj = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        ok2, fj = cv2.imencode(".jpg", annotated,
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not (ok1 and ok2):
            return
        ts = time.time()
        ev = {"id": str(int(ts * 1000)), "ts": ts, "layer": layer,
              "text": text,
              "thumb": base64.b64encode(tj.tobytes()).decode("ascii"),
              "saved": False,
              "_full": fj.tobytes()}   # server-side only, dropped on GET
        with self.lock:
            self.events.append(ev)
        if auto_save:
            # Skip the user's Save click - persist immediately to the
            # Investigation gallery. Silent-on-failure so a disk hiccup
            # never takes down the tick.
            try:
                self.save_event(ev["id"])
            except Exception:
                pass

    def _detect_events(self, js_boxes, layer, data, faces_list) -> None:
        """Layer-specific state changes -> the event ring. A tick that
        merely re-observes the same scene emits nothing."""
        if not hasattr(self, "events"):
            from collections import deque
            self.events = deque(maxlen=self.EV_RING)
            self._ev = {"plates": set(), "gest": set(), "body": set(),
                        "pose": set(), "fast": set(), "cross": None,
                        "fire_on": False, "spots": {}, "faces_n": 0,
                        "faces_t": 0.0, "heat_t": 0.0,
                        "obstruction": None}
        st = self._ev
        now = time.time()

        # ---- Cross-layer: camera obstruction ---------------------------
        # ModelViewProducer used to compute this badge, but it skips its
        # whole round while _analysis_active() returns True - i.e. exactly
        # when the operator is looking at the live tile. Moving the check
        # inside every Advanced Analysis tick means an obstructed camera
        # (bus parked on the lens, truck at the junction) fires the badge
        # on the SAME layer the operator is watching, not only on the
        # background producer. Same gate: box area >= 50% of frame with
        # detector conf >= 0.45. Emitted once per transition, not per tick.
        W = int(data.get("frame_w") or 0)
        H = int(data.get("frame_h") or 0)
        if W and H:
            frame_area = float(W) * float(H)
            obstructed = None
            for b in js_boxes:
                bw = max(0.0, float(b.get("x2", 0)) - float(b.get("x1", 0)))
                bh = max(0.0, float(b.get("y2", 0)) - float(b.get("y1", 0)))
                frac = (bw * bh) / frame_area if frame_area else 0.0
                if frac >= 0.5 and float(b.get("conf") or 0.0) >= 0.45:
                    obstructed = {"cls":  b.get("cls", "?"),
                                  "frac": round(frac, 2),
                                  "tid":  b.get("tid")}
                    break
            data["obstructed"] = obstructed
            prev = st.get("obstruction")
            if obstructed and obstructed != prev:
                self._emit_event(
                    "obstruction",
                    f"camera obstructed: {obstructed['cls']} covers "
                    f"{int(obstructed['frac'] * 100)}% of view")
            st["obstruction"] = obstructed

        if layer == "plates":
            for b in js_boxes:
                t = b.get("plate")
                if t and (b.get("tid"), t) not in st["plates"]:
                    st["plates"].add((b.get("tid"), t))
                    self._emit_event(layer,
                                     f"plate {t} "
                                     f"({b.get('plate_conf', 0):.2f}) "
                                     f"{b.get('cls', '')}", b)
        elif layer == "line":
            prev, cur = st["cross"], dict(self.cross)
            if prev is not None:
                for d_ in ("in", "out"):
                    if cur.get(d_, 0) > prev.get(d_, 0):
                        self._emit_event(layer, f"crossing {d_.upper()} "
                                         f"(total {cur[d_]})")
            st["cross"] = cur
        elif layer == "fire":
            info = data.get("fire") or {}
            confirmed = bool(info.get("confirmed"))
            prev = bool(st.get("fire_on", False))
            if confirmed and not prev:
                hits = info.get("hits") or []
                top = max(hits, key=lambda h: h.get("conf", 0)) \
                    if hits else None
                self._emit_event(
                    layer,
                    f"FIRE confirmed - {len(hits)} region(s), "
                    f"top conf {float(top.get('conf', 0)):.2f}"
                    if top else "FIRE confirmed",
                    top)
            elif prev and not confirmed:
                self._emit_event(layer, "fire cleared - no confirmed "
                                 "hits this tick")
            st["fire_on"] = confirmed
        elif layer == "parking":
            for s in data.get("spots") or []:
                name, occ = s.get("name"), bool(s.get("occupied"))
                if name in st["spots"] and st["spots"][name] != occ:
                    self._emit_event(layer, f"{name} " +
                                     (f"occupied ({s.get('cls')})" if occ
                                      else "vacated"))
                st["spots"][name] = occ
        elif layer == "gestures":
            for b in js_boxes:
                for g in b.get("gestures") or []:
                    k = (b.get("tid"), g)
                    if k not in st["gest"]:
                        st["gest"].add(k)
                        self._emit_event(layer, f"#{b.get('tid')} {g}", b)
        elif layer == "body":
            for b in js_boxes:
                f = b.get("flag")
                if f and (b.get("tid"), f) not in st["body"]:
                    st["body"].add((b.get("tid"), f))
                    self._emit_event(layer, f"#{b.get('tid')} "
                                     f"{str(f).upper()}"
                                     + (" ALERT" if b.get("alert") else ""),
                                     b)
        elif layer == "pose":
            for b in js_boxes:
                if b.get("kps") and b.get("tid") not in st["pose"]:
                    st["pose"].add(b.get("tid"))
                    self._emit_event(layer, f"#{b.get('tid')} skeleton "
                                     f"acquired", b)
        elif layer == "paths":
            for b in js_boxes:
                if b.get("tier") == "fast" and b.get("tid") not in st["fast"]:
                    st["fast"].add(b.get("tid"))
                    self._emit_event(layer, f"#{b.get('tid')} fast "
                                     f"({b.get('cls')})", b)
        elif layer == "faces":
            n = len(faces_list or [])
            if n > st["faces_n"] and now - st["faces_t"] >= 30:
                st["faces_t"] = now
                self._emit_event(layer, f"{n} face(s) in frame")
            st["faces_n"] = n
        elif layer == "heat":
            if now - st["heat_t"] >= 120:
                raw = getattr(self, "heat", None)
                if raw is not None:
                    import numpy as np
                    # self.heat is a plain list-of-lists; asarray first
                    # (hm.shape on the raw list raised AttributeError and
                    # silently killed every heat event).
                    hm = np.asarray(raw, dtype=float)
                    if hm.ndim == 2 and float(hm.max()) > 0:
                        iy, ix = divmod(int(hm.argmax()), hm.shape[1])
                        st["heat_t"] = now
                        self._emit_event(layer, "hotspot at "
                                         f"{int(100 * ix / hm.shape[1])}%,"
                                         f"{int(100 * iy / hm.shape[0])}% "
                                         f"of frame")
        # Long sessions: bound the dedup sets.
        for k in ("plates", "gest", "body", "pose", "fast"):
            if len(st[k]) > 512:
                st[k].clear()

    def snapshot_events(self) -> list[dict]:
        with self.lock:
            events = getattr(self, "events", None)
            if not events:
                return []
            return [{k: v for k, v in e.items() if k != "_full"}
                    for e in reversed(events)]

    def save_event(self, event_id: str) -> dict | None:
        """Persist one ring event (full frame) to disk for later study."""
        with self.lock:
            ev = next((e for e in getattr(self, "events", [])
                       if e["id"] == event_id), None)
            full = ev.get("_full") if ev else None
        if ev is None:
            return None
        from pathlib import Path
        base = (Path(__file__).resolve().parent.parent / "web"
                / "snapshots" / "detections")
        base.mkdir(parents=True, exist_ok=True)
        fn = f"{self.cam_id}_{ev['id']}.jpg"
        if full:
            (base / fn).write_bytes(full)
        row = {"id": ev["id"], "cam": self.cam_id,
               "cam_name": self.cam_name, "layer": ev["layer"],
               "text": ev["text"], "ts": ev["ts"],
               "image": f"snapshots/detections/{fn}"}
        man = base / "saved.json"
        # 2026-08-23 (C1a): serialize the read-modify-write of saved.json
        # under a module-level lock. Two auto-save emits from consecutive
        # ticks (the same plate that survived dedup - or two different
        # events that share a tick boundary) could both read stale
        # items, both insert their row, then race on write - the second
        # write would clobber the first row. Observed: `UU730 (0.63)`
        # saved twice with ts 62 ms apart (bug #2 in AUDIT_2026-08-23).
        with _SAVED_JSON_LOCK:
            try:
                items = json.loads(man.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                items = []
            items.insert(0, row)
            man.write_text(json.dumps(items[:500], ensure_ascii=False),
                           encoding="utf-8")
        with self.lock:
            ev["saved"] = True
        return row


# ---------------------------------------------------------------------------
# The manager the dashboard endpoints talk to.
# ---------------------------------------------------------------------------

class LiveAnalysisManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: dict[str, LiveSession] = {}
        # Last crash reason per cam_id, kept until the NEXT frame() poll so
        # the client can render "analysis stopped: <reason>" instead of the
        # bare 404 the old reap loop returned. Bounded to MAX_SESSIONS
        # entries by _remember_error_locked (one per possible camera slot).
        self._errors: dict[str, str] = {}

    def start(self, cam_id: str, layer: str, model) -> dict:
        """Start a session for `cam_id`, or switch the layer of a running
        one (stream + accumulators survive the switch)."""
        if layer not in LIVE_LAYERS:
            raise ValueError(f"unknown layer {layer!r}")
        cam = resolve_cam(cam_id)     # raises ValueError on unknown ids
        with self._lock:
            self._reap_locked()
            # A fresh start clears any stale error remembered from the
            # previous session on this camera.
            self._errors.pop(cam_id, None)
            s = self._sessions.get(cam_id)
            if s is not None and s.is_alive():
                switched = s.layer != layer
                s.layer = layer
                s.last_poll = time.time()
                return {"cam": cam_id, "cam_name": s.cam_name,
                        "layer": layer, "switched": switched,
                        "active": len(self._sessions)}
            if len(self._sessions) >= MAX_SESSIONS:
                raise BusyError(
                    f"{MAX_SESSIONS} live analyses already running - "
                    f"stop one first")
            s = LiveSession(cam, model, layer)
            self._sessions[cam_id] = s
            s.start()
            return {"cam": cam_id, "cam_name": s.cam_name, "layer": layer,
                    "switched": False, "active": len(self._sessions)}

    def frame(self, cam_id: str) -> dict | None:
        """Latest JPEG + metadata, or None when no session runs. Every
        call refreshes the idle clock. A session that crashed is popped
        AND its error is returned once via {"error": "..."} so the client
        sees WHY analysis stopped instead of a bare 404."""
        with self._lock:
            s = self._sessions.get(cam_id)
            if s is None:
                err = self._errors.pop(cam_id, None)
                return {"error": err} if err else None
            if not s.is_alive():
                self._sessions.pop(cam_id, None)
                self._remember_error_locked(cam_id, s.err
                                            or "session ended unexpectedly")
                return {"error": self._errors.pop(cam_id, None)}
        s.last_poll = time.time()
        with s.lock:
            return {"jpeg": s.latest, "seq": s.seq, "layer": s.layer,
                    "note": s.note}

    def any_alive(self) -> bool:
        """True while at least one session thread is actually running.
        local_producers reads this to yield the CPU during analysis;
        thread state (not a bookkeeping set) means an idle-timed-out
        session releases the pause automatically."""
        with self._lock:
            return any(s.is_alive() for s in self._sessions.values())

    def clear_saved_state(self) -> None:
        """Reset per-session dedup + event-ring state that /api/analysis/
        saved-clear invalidated. Without this, already-seen plates stay
        in _plate_emitted for the rest of the session so the gallery
        cannot refill on the same passing car, and the event strip keeps
        offering Save on rows whose disk copies were just deleted."""
        with self._lock:
            sessions = list(self._sessions.values())
        for s in sessions:
            try:
                with s.lock:
                    if hasattr(s, "_plate_emitted"):
                        s._plate_emitted.clear()
                    if hasattr(s, "events"):
                        s.events.clear()
            except Exception:
                pass

    def data(self, cam_id: str) -> dict | None:
        """Same idle-clock refresh as frame(), but returns the JSON snapshot
        used by the canvas-overlay renderer instead of the annotated JPEG."""
        with self._lock:
            s = self._sessions.get(cam_id)
            if s is None:
                err = self._errors.pop(cam_id, None)
                return {"error": err} if err else None
            if not s.is_alive():
                self._sessions.pop(cam_id, None)
                self._remember_error_locked(cam_id, s.err
                                            or "session ended unexpectedly")
                return {"error": self._errors.pop(cam_id, None)}
        s.last_poll = time.time()
        with s.lock:
            return {"data": s.latest_data, "seq": s.seq, "layer": s.layer,
                    "note": s.note}

    def replay(self, cam_id: str, ts: float | None = None) -> dict | None:
        """The session's replay ring as base64 JPEGs (D3). With `ts`,
        only frames within REPLAY_RING_S/2 of it; without, the whole
        ring. None when no session runs for the camera."""
        with self._lock:
            s = self._sessions.get(cam_id)
        if s is None or not s.is_alive():
            return None
        s.last_poll = time.time()
        import base64
        with s.lock:
            frames = list(getattr(s, "replay_ring", []) or [])
        if ts:
            half = REPLAY_RING_S / 2.0
            frames = [f for f in frames if abs(f[0] - ts) <= half]
        return {
            "fps": REPLAY_FPS,
            "window_s": REPLAY_RING_S,
            "frames": [
                {"ts": round(t, 3),
                 "jpeg": base64.b64encode(b).decode("ascii")}
                for t, b in frames],
        }

    def events(self, cam_id: str) -> list[dict] | None:
        """The session's detection-event ring, newest first (no full
        frames - those stay server-side until an explicit save)."""
        with self._lock:
            s = self._sessions.get(cam_id)
        if s is None or not s.is_alive():
            return None
        s.last_poll = time.time()
        return s.snapshot_events()

    def save_event(self, cam_id: str, event_id: str) -> dict | None:
        with self._lock:
            s = self._sessions.get(cam_id)
        if s is None:
            return None
        return s.save_event(event_id)

    def stop(self, cam_id: str) -> bool:
        with self._lock:
            s = self._sessions.pop(cam_id, None)
            # An operator-initiated stop is not a crash; drop any pending
            # error so the next start on this camera is a clean slate.
            self._errors.pop(cam_id, None)
        if s is None:
            return False
        s.stop_event.set()
        return True

    def stop_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._errors.clear()
        for s in sessions:
            s.stop_event.set()

    def _reap_locked(self) -> None:
        for cam_id in [c for c, s in self._sessions.items()
                       if not s.is_alive()]:
            s = self._sessions.pop(cam_id, None)
            if s is not None:
                self._remember_error_locked(
                    cam_id, getattr(s, "err", None) or "session ended unexpectedly")

    def _remember_error_locked(self, cam_id: str, err: str) -> None:
        """Cap the error dict at MAX_SESSIONS entries (one per possible
        camera slot) so a runaway crash loop can never grow it unbounded."""
        self._errors[cam_id] = err
        if len(self._errors) > MAX_SESSIONS:
            # FIFO eviction: pop the oldest remembered error.
            oldest = next(iter(self._errors))
            if oldest != cam_id:
                self._errors.pop(oldest, None)


MANAGER = LiveAnalysisManager()
