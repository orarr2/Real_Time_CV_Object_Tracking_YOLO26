"""Deep-window behavior analysis: WHAT each individual did, on demand.

The collector's routine burst (3 frames) answers "how many"; this module
answers "who did what". It grabs a LONGER window from one camera (default
12 frames ~0.5s apart), runs the same gated detection per frame, threads
the detections into per-individual tracks (app/tracker.py - position +
motion, the only signal that separates look-alikes), and computes a
behavior profile per individual:

  * path            - the foot-point trajectory (normalized, JSON-safe);
  * distance/speed  - path length, net displacement, mean/max px/s, and a
                      km/h estimate for vehicles (class-length ruler, same
                      +-30-50% honesty band as the burst speed pass);
  * moving_frac     - the fraction of its steps it actually moved
                      ("stood still 80% of the window");
  * direction       - dominant screen direction of the net displacement;
  * zones           - which heatmap grid cells it visited (ties the
                      trajectory to the long-horizon dwell map);
  * nn_min/mean_px  - closest same-class neighbor over the window
                      (crowding/pairing signal);
  * label           - one readable behavior verdict per individual
                      (walking / running / erratic / fall_suspect ... -
                      app/behavior_labels.py) with its evidence;
  * gestures        - arm-level gestures over the window (hand raised,
                      both up, wave - app/gestures.py), pose mode only.

Opt-in extras per request: `pose=True` runs the keypoint pass
(app/pose.py) and enriches labels/gestures with posture; `want_faces`
adds face DETECTION boxes on the final frame (app/faces.py - rectangles
only, no identification); `lock` picks one individual (a numeric id or
"auto") and renders a crosshair target-lock overlay, returning the
target's normalized offset from frame center - the exact signal a
pan/tilt loop would consume if local hardware ever existed. All three
default off, so an unadorned call behaves exactly as before.

Cost model: n_frames extra inferences on ONE camera, so this NEVER runs
inside the collector's round. It is operator-triggered - the dashboard's
"analyze window" button (POST /api/deep-analyze) or the CLI
(tools/analyze_window.py) - and the annotated result (trails + ids) plus
the JSON profile land under snapshots/behavior/ with a small LRU cap.
"""
from __future__ import annotations

import datetime as _dt
import json
import time
from pathlib import Path

from app.cameras import CAMERAS
from app.detect_core import (
    DEFAULT_PER_CLASS_CONF,
    VEHICLE_LENGTH_M,
    detect_with_boxes,
    draw_boxes,
    filter_boxes_roi,
    grab_burst,
    last_stream_fps,
    resolve_stream,
)
from app.heatmap import GRID_H, GRID_W
from app.tracker import Track, assign_burst_ids, _centroid

_SRC_ROOT = Path(__file__).resolve().parent.parent
BEHAVIOR_DIR = _SRC_ROOT / "web" / "snapshots" / "behavior"
BEHAVIOR_MAX_FILES = 40           # jpg+json pairs; oldest evicted first

DEFAULT_FRAMES = 12
DEFAULT_STRIDE = 12               # ~0.5s between frames at 25 fps
DEFAULT_IMGSZ = 640               # deep window favors coverage over recall

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


def attach_neighbor_stats(tracks: list[Track], stats: list[dict]) -> None:
    """Closest SAME-CLASS neighbor per individual over the window - the
    crowding signal ("these two moved as a pair" / "this one kept apart").
    Mutates `stats` rows in place; solitary individuals stay None."""
    # time -> [(track_idx, cx, cy, cls)]
    by_time: dict[float, list[tuple[int, float, float, str | None]]] = {}
    for i, tr in enumerate(tracks):
        for t, b in zip(tr.times, tr.boxes):
            cx, cy = _centroid(b)
            by_time.setdefault(round(t, 3), []).append((i, cx, cy, tr.cls))
    dists: dict[int, list[float]] = {}
    for entries in by_time.values():
        for i, cx, cy, cls in entries:
            best = None
            for j, ox, oy, ocls in entries:
                if j == i or ocls != cls:
                    continue
                d = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
                if best is None or d < best:
                    best = d
            if best is not None:
                dists.setdefault(i, []).append(best)
    for i, row in enumerate(stats):
        ds = dists.get(i)
        row["nn_min_px"] = round(min(ds), 1) if ds else None
        row["nn_mean_px"] = round(sum(ds) / len(ds), 1) if ds else None


# Trail palette (BGR) - cycled by track id so neighboring ids differ.
_TRAIL_COLORS = ((80, 175, 76), (60, 130, 246), (0, 200, 255),
                 (200, 60, 200), (0, 90, 230), (255, 160, 0),
                 (180, 220, 40), (140, 100, 255))


def render_window(frames, tracks: list[Track], stats: list[dict] | None = None,
                  lock: dict | None = None, faces: list[dict] | None = None):
    """Annotate the LAST frame with every individual's trail + numbered
    boxes. Trails run through centroids; the final box (drawn by
    draw_boxes) already carries `#id` in its label. The optional layers
    stack on top in reading order: skeletons (when boxes carry `kps`),
    behavior-label chips (when `stats` is given), face boxes, and the
    target-lock crosshair last so nothing covers it."""
    import cv2

    base = frames[-1]
    out = base.copy()
    for tr in tracks:
        color = _TRAIL_COLORS[(tr.tid - 1) % len(_TRAIL_COLORS)]
        pts = [(int(cx), int(cy))
               for cx, cy in (_centroid(b) for b in tr.boxes)]
        for p0, p1 in zip(pts, pts[1:]):
            cv2.line(out, p0, p1, color, 2, cv2.LINE_AA)
        if pts:
            cv2.circle(out, pts[0], 4, color, -1, cv2.LINE_AA)  # birth dot
    # Boxes of the final frame (they hold the ids) on top of the trails.
    last_boxes = _boxes_of_last_frame(tracks)
    out = draw_boxes(out, last_boxes)

    if any("kps" in b for b in last_boxes):
        from app.pose import draw_skeleton
        out = draw_skeleton(out, last_boxes)

    if stats:
        _draw_label_chips(out, last_boxes, stats)
    if faces:
        from app.faces import draw_faces
        out = draw_faces(out, faces)
    if lock:
        _draw_target_lock(out, lock)
    return out


def _draw_label_chips(out, last_boxes: list[dict], stats: list[dict],
                      text_of=None) -> None:
    """Behavior verdict under each surviving individual's box - the demo
    convention (a colored chip with the label), red when alerting.
    `text_of(stats_row) -> str | None` overrides the chip text (the
    gestures layer shows only gestures, the body layer only the label)."""
    import cv2

    def _default_text(s):
        txt = s.get("label") or ""
        if s.get("gestures"):
            txt += " +" + "+".join(s["gestures"])
        return txt or None

    text_of = text_of or _default_text
    by_id = {s.get("id"): s for s in stats}
    for b in last_boxes:
        s = by_id.get(b.get("track_id"))
        if not s:
            continue
        txt = text_of(s)
        if not txt:
            continue
        color = (0, 0, 220) if s.get("alert") else (90, 90, 90)
        x1, y2 = int(b["x1"]), int(b["y2"])
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y2 + 2), (x1 + tw + 6, y2 + th + 8),
                      color, -1)
        cv2.putText(out, txt, (x1 + 3, y2 + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)


def _draw_target_lock(out, lock: dict) -> None:
    """Full-frame crosshair through the locked individual's centroid."""
    import cv2

    H, W = out.shape[:2]
    cx, cy = int(lock["cx"]), int(lock["cy"])
    red = (0, 0, 230)
    cv2.line(out, (0, cy), (W, cy), red, 1, cv2.LINE_AA)
    cv2.line(out, (cx, 0), (cx, H), red, 1, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), 26, red, 2, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), 4, red, -1, cv2.LINE_AA)
    cv2.putText(out, f"TARGET LOCKED #{lock['track_id']}",
                (max(8, W - 320), 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                red, 2, cv2.LINE_AA)
    cv2.putText(out, f"[{cx} {cy}]", (cx + 32, max(20, cy - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, red, 1, cv2.LINE_AA)


def _choose_lock(tracks: list[Track], stats: list[dict], want: str,
                 frame_shape) -> dict | None:
    """Resolve the lock request to one track. "auto" prefers the first
    alerting individual, then the largest person, then the largest
    anything; a numeric id targets that exact track. None when the
    request matches nothing (an id that never existed, an empty scene)."""
    if not tracks:
        return None
    by_id = {tr.tid: tr for tr in tracks}
    tid: int | None = None
    if want == "auto":
        alerted = [s["id"] for s in stats if s.get("alert")]
        if alerted:
            tid = alerted[0]
        else:
            def _area(tr: Track) -> float:
                b = tr.boxes[-1]
                return (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
            persons = [tr for tr in tracks if tr.cls == "person"]
            pool = persons or tracks
            tid = max(pool, key=_area).tid
    else:
        try:
            tid = int(want)
        except (TypeError, ValueError):
            return None
    tr = by_id.get(tid)
    if tr is None:
        return None
    H, W = frame_shape[:2]
    cx, cy = _centroid(tr.boxes[-1])
    return {
        "track_id": tr.tid,
        "cx": round(cx, 1), "cy": round(cy, 1),
        # Normalized offset from frame center in [-0.5, 0.5] - the error
        # signal a pan/tilt controller would zero out. Kept even though no
        # such hardware exists here: it makes the lock testable and honest
        # about what it is (an annotation, not an actuator).
        "dx": round(cx / W - 0.5, 3), "dy": round(cy / H - 0.5, 3),
    }


def _boxes_of_last_frame(tracks: list[Track]) -> list[dict]:
    """The most recent box of every track that survived to the window's
    final timestamp (== max over all tracks)."""
    if not tracks:
        return []
    t_end = max(tr.times[-1] for tr in tracks)
    return [tr.boxes[-1] for tr in tracks if tr.times[-1] >= t_end - 1e-9]


def _enforce_cap(root: Path, cap: int = BEHAVIOR_MAX_FILES) -> None:
    files = sorted(root.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
    for p in files[:-cap] if len(files) > cap else []:
        p.unlink(missing_ok=True)
        p.with_suffix(".json").unlink(missing_ok=True)


def analyze_window(cam_id: str, model,
                   n_frames: int = DEFAULT_FRAMES,
                   stride: int = DEFAULT_STRIDE,
                   imgsz: int | None = DEFAULT_IMGSZ,
                   save: bool = True,
                   frames=None,
                   pose: bool = False,
                   lock: str | None = None,
                   want_faces: bool = False) -> dict:
    """Grab a window from `cam_id`, profile every individual in it.

    `frames` overrides the live grab (tests / offline replays feed frames
    directly). Raises ValueError on an unknown camera, RuntimeError when
    the stream yields fewer than 2 frames. `pose` / `lock` / `want_faces`
    are the opt-in layers documented in the module docstring; all off by
    default, leaving the base behavior byte-identical.
    """
    cam = CAMERAS.get(cam_id)
    if cam is None:
        raise ValueError(f"unknown cam_id {cam_id!r}")
    if frames is None:
        frames = grab_burst(resolve_stream(cam), n=n_frames, stride=stride)
    if len(frames) < 2:
        raise RuntimeError(f"{cam_id}: needed >= 2 frames, "
                           f"got {len(frames)}")

    gates = dict(cam.get("per_class_conf") or DEFAULT_PER_CLASS_CONF)
    per_boxes: list[list[dict]] = []
    for fr in frames:
        _c, b = detect_with_boxes(model, fr, conf=cam.get("conf", 0.30),
                                  imgsz=imgsz, per_class_conf=gates)
        if cam.get("roi") or cam.get("roi_exclude") \
                or cam.get("roi_exclude_class"):
            b = filter_boxes_roi(b, fr.shape, cam.get("roi"),
                                 cam.get("roi_exclude"),
                                 cam.get("roi_exclude_class"))
        per_boxes.append(b)

    # Optional pose pass: attach keypoints to the person boxes BEFORE the
    # tracker threads them - the tracker keeps the same dicts, so every
    # track's box history carries its skeletons for free.
    pose_persons = 0
    if pose:
        # Top-down (per-crop) pose: the full-frame pass at window imgsz saw
        # ~15px of person on wide street shots and attached nothing - the
        # crop variant is what makes gestures/fall/crouch actually fire at
        # street-cam distance.
        from app.pose import attach_keypoints_crops, load_pose_model
        pose_model = load_pose_model()
        for fr, b in zip(frames, per_boxes):
            pose_persons += attach_keypoints_crops(pose_model, fr, b)

    # Real container fps of the stream just grabbed (falls back to the
    # 25fps assumption for direct-frame calls in tests/replays).
    dt = stride / last_stream_fps()
    tracks = assign_burst_ids(per_boxes, frames[0].shape, dt=dt)
    stats = []
    for tr in tracks:
        row = track_stats(tr.cls, tr.boxes, tr.times, frames[0].shape)
        row["id"] = tr.tid
        stats.append(row)
    attach_neighbor_stats(tracks, stats)

    # Behavior verdict per individual - kinematic rules always, posture
    # rules when this track's boxes carry keypoints.
    from app.behavior_labels import label_track
    from app.gestures import detect_gestures
    for tr, row in zip(tracks, stats):
        kps_seq = [b.get("kps") for b in tr.boxes]
        has_kps = any(kps_seq)
        row.update(label_track(row, frames[0].shape,
                               kps_seq if has_kps else None))
        row["gestures"] = detect_gestures(kps_seq) if has_kps else []

    faces_list: list[dict] = []
    faces_available = None
    if want_faces:
        from app import faces as _faces
        faces_available = _faces.available()
        if faces_available:
            faces_list = _faces.detect_faces(frames[-1])

    lock_info = _choose_lock(tracks, stats, lock, frames[0].shape) \
        if lock else None

    moving = [s for s in stats if not s["stationary"]]
    result = {
        "cam_id": cam_id,
        "cam_name": cam.get("name", cam_id),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "frames": len(frames),
        "window_sec": round((len(frames) - 1) * dt, 1),
        "individuals": len(stats),
        "moving": len(moving),
        "stationary": len(stats) - len(moving),
        "tracks": stats,
    }

    label_counts: dict[str, int] = {}
    for s in stats:
        label_counts[s["label"]] = label_counts.get(s["label"], 0) + 1
    result["labels"] = label_counts
    result["alerts"] = [s["id"] for s in stats if s.get("alert")]
    if pose:
        result["pose_persons"] = pose_persons
    if want_faces:
        result["faces_available"] = faces_available
        result["faces"] = len(faces_list)
    if lock_info:
        result["lock"] = lock_info

    if save:
        import cv2
        BEHAVIOR_DIR.mkdir(parents=True, exist_ok=True)
        stem = (f"{cam_id}_"
                f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}")
        annotated = render_window(frames, tracks, stats=stats,
                                  lock=lock_info,
                                  faces=faces_list or None)
        okj, buf = cv2.imencode(".jpg", annotated,
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
        if okj:
            (BEHAVIOR_DIR / f"{stem}.jpg").write_bytes(buf.tobytes())
            result["image_url"] = f"/snapshots/behavior/{stem}.jpg"
        (BEHAVIOR_DIR / f"{stem}.json").write_text(
            json.dumps(result, indent=1), encoding="utf-8")
        result["json_url"] = f"/snapshots/behavior/{stem}.json"
        _enforce_cap(BEHAVIOR_DIR)

    return result
