"""Local dashboard-data producers for Repo #2 single-camera live mode.

The dashboard reads its Model view - live strip and its Review-
detections tab from files on disk. This module runs two lightweight
background threads that produce those files LOCALLY, one round per
picked cam (the single-camera Repo #2 mode passes a one-element list):

  * ``ModelViewProducer`` - every N seconds, grab a frame from each
    picked cam, run YOLO on it, draw the boxes and save the annotated
    JPEG + a small JSON with counts under
    ``web/snapshots/model_view/<slot_id>.jpg`` / ``.json``. The
    dashboard's Model view - live strip reads from THAT path.
  * ``ReviewFrameProducer`` - every M seconds, grab a frame from a
    round-robin picked cam, run YOLO on it, and save the frame plus
    its box metadata under
    ``web/snapshots/review_frames/<slot_id>/<ts>.jpg`` / ``.json``.
    The existing Review-detections tab picks it up automatically the
    next time the sampler runs.

Both producers are strictly ADDITIVE: no external services, no cloud
writes, no-op silently when a grab or an inference fails so a bad
stream cannot crash the notebook.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = _SRC_ROOT / "web"
SNAPSHOTS_DIR = WEB_DIR / "snapshots"
MODEL_VIEW_DIR = SNAPSHOTS_DIR / "model_view"
REVIEW_FRAMES_DIR = SNAPSHOTS_DIR / "review_frames"

DEFAULT_MODEL_VIEW_INTERVAL_S = 30.0
DEFAULT_REVIEW_FRAME_INTERVAL_S = 60.0

_LAST_ERR_PRINT: dict[str, float] = {}


def _print_throttled(key: str, msg: str, every: float = 60.0) -> None:
    """print(), but at most once per `every` seconds per key - a dead
    camera used to write the same failure line every round for hours."""
    now = time.time()
    if now - _LAST_ERR_PRINT.get(key, 0.0) >= every:
        _LAST_ERR_PRINT[key] = now
        print(msg)


def _analysis_active() -> bool:
    """True while any Advanced Analysis session is running.

    Both producers skip their whole round then, yielding the CPU so the
    analyzed stream ticks as fast as the model allows - on a 4-core
    laptop the producers' grab+infer rounds on the OTHER cams were
    measured to saturate the CPU and roughly double the analysis tick
    time. KPI badges freeze during analysis and resume on Stop. Reading
    live thread state (not a bookkeeping set) means a session that dies
    on idle-timeout releases the pause by itself.
    """
    try:
        from app.live_analysis import MANAGER
        return MANAGER.any_alive()
    except Exception:
        return False


def _grab_and_detect(cam_dict: dict, model,
                     conf: float = 0.30, imgsz: int = 640):
    """Grab the freshest frame + run YOLO. Returns ``(frame, boxes)`` or
    ``(None, None)`` on any failure.

    Frames come from live_analysis's SHARED reader pool when possible -
    one persistent decoder per camera serving both the producers and any
    analysis session - instead of the old open-read-close VideoCapture
    that cost 1-2 s of stream handshake per camera per round. Header-
    required hosts (no plain VideoCapture possible) and pool failures
    fall back to the old one-shot segment path.
    """
    from app.detect_core import (grab_frame, resolve_stream,
                                 detect_with_boxes,
                                 DEFAULT_PER_CLASS_CONF)
    frame = None
    cam_id = (cam_dict.get("id") or cam_dict.get("cam_id")
              or cam_dict.get("slot_id"))
    if cam_id:
        try:
            from app.live_analysis import get_shared_reader
            r = get_shared_reader(cam_dict, cam_id)
            if r is not None:
                frame = r.snapshot_wait(timeout=8.0)
        except Exception:
            frame = None
    if frame is None:
        try:
            url = resolve_stream(cam_dict)
        except Exception:
            return None, None
        frame = grab_frame(url)
        if frame is None:
            return None, None
    try:
        gates = dict(cam_dict.get("per_class_conf")
                     or DEFAULT_PER_CLASS_CONF)
        _c, boxes = detect_with_boxes(model, frame, conf=conf,
                                      imgsz=imgsz, per_class_conf=gates)
    except Exception:
        return frame, []
    return frame, boxes


def _draw_boxes(frame, boxes):
    """Draw class-labeled boxes on a copy of frame, return the copy."""
    import cv2
    img = frame.copy()
    for b in boxes or []:
        x1, y1 = int(b.get("x1", 0)), int(b.get("y1", 0))
        x2, y2 = int(b.get("x2", 0)), int(b.get("y2", 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), (79, 140, 255), 2)
        cls = b.get("cls", "?")
        conf = float(b.get("conf") or 0)
        label = f"{cls} {int(conf * 100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 4)),
                      (x1 + tw + 4, y1), (15, 23, 42), -1)
        cv2.putText(img, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


class ModelViewProducer(threading.Thread):
    """Refresh web/snapshots/model_view/<slot_id>.jpg every N seconds."""

    def __init__(self, picked_slots: list[dict], model,
                 interval_s: float = DEFAULT_MODEL_VIEW_INTERVAL_S):
        super().__init__(daemon=True, name="local-model-view-producer")
        self.picked_slots = list(picked_slots)
        self.model = model
        self.interval_s = float(interval_s)
        self.stop_event = threading.Event()

    def run(self) -> None:
        # Wrap the whole loop so a crash gets logged instead of dying silent.
        try:
            self._loop()
        except Exception as e:
            print(f"[local_producers.ModelViewProducer] crashed: "
                  f"{type(e).__name__}: {e}")

    def _loop(self) -> None:
        import cv2
        from app.cameras import CAMERAS
        MODEL_VIEW_DIR.mkdir(parents=True, exist_ok=True)
        while not self.stop_event.is_set():
            t0 = time.time()
            if _analysis_active():
                # Advanced Analysis owns the CPU - skip the whole round.
                if self.stop_event.wait(2.0):
                    break
                continue
            for slot in self.picked_slots:
                if self.stop_event.is_set():
                    break
                slot_id = slot.get("slot_id") or slot.get("id")
                if not slot_id:
                    continue
                cam_id = slot.get("cam_id") or slot_id
                cam = CAMERAS.get(cam_id) or slot
                try:
                    frame, boxes = _grab_and_detect(cam, self.model)
                    if frame is None:
                        continue
                    img = _draw_boxes(frame, boxes)
                    jpg_path = MODEL_VIEW_DIR / f"{slot_id}.jpg"
                    # Skip atomic rename on Windows: replace() into a path an
                    # HTTP handler may have open for read raises PermissionError
                    # and killed the producer thread. Direct overwrite is fine
                    # here - the frontend polls with a cache-buster and reads
                    # whichever byte it gets first.
                    cv2.imwrite(str(jpg_path), img,
                                [cv2.IMWRITE_JPEG_QUALITY, 82])
                    counts = {"person": 0, "vehicles": 0}
                    for b in boxes or []:
                        if b.get("cls") == "person":
                            counts["person"] += 1
                        elif b.get("cls") in ("car", "truck", "bus", "motorcycle",
                                              "bicycle"):
                            counts["vehicles"] += 1
                    # camera_obstructed: one confident box covering half the
                    # view (a bus parked on the lens, a truck at the junction
                    # mouth) is an interference the operator should see as a
                    # tile badge. Gates: area >= 0.5, conf >= 0.45.
                    H, W = frame.shape[:2]
                    obstructed = None
                    for b in boxes or []:
                        frac = (max(0, b["x2"] - b["x1"])
                                * max(0, b["y2"] - b["y1"])) / max(1, W * H)
                        if frac >= 0.5 and float(b.get("conf") or 0) >= 0.45:
                            obstructed = {"cls": b.get("cls", "?"),
                                          "frac": round(frac, 2)}
                            break
                    # camera_dark: mean luma near black = covered lens or
                    # power cut (night streets still average 25-60; a
                    # genuinely dead view sits under ~10).
                    luma = float(frame[::8, ::8].mean())
                    payload = {"slot_id": slot_id, "cam_id": cam_id,
                               "cam_name": slot.get("placeholder_name")
                                          or cam.get("name") or cam_id,
                               "counts": counts,
                               "at": time.time()}
                    if luma < 10:
                        payload["dark"] = round(luma, 1)
                    if obstructed:
                        payload["obstructed"] = obstructed
                    (MODEL_VIEW_DIR / f"{slot_id}.json").write_text(
                        json.dumps(payload))
                    # Rolling 24h footfall history for the PICKED cams -
                    # the source for the combined 24h chart and the tile
                    # KPI aggregates. One JSON line per producer round;
                    # trimmed to 24h of 30s rounds.
                    hist = MODEL_VIEW_DIR / f"{slot_id}_history.jsonl"
                    row = json.dumps(
                        {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()),
                         "person": counts["person"],
                         "vehicles": counts["vehicles"], "ok": True})
                    with hist.open("a", encoding="utf-8") as hf:
                        hf.write(row + "\n")
                    if hist.stat().st_size > 300_000:
                        keep = hist.read_text(
                            encoding="utf-8").splitlines()[-2880:]
                        hist.write_text("\n".join(keep) + "\n",
                                        encoding="utf-8")
                except Exception as e:
                    # A single-cam glitch (network, decoder, disk-lock) must
                    # never kill the loop for the OTHER cams.
                    _print_throttled(f"mv:{slot_id}",
                                     f"[model-view] slot {slot_id}: "
                                     f"{type(e).__name__}: {e}")
            elapsed = time.time() - t0
            if self.stop_event.wait(max(1.0, self.interval_s - elapsed)):
                break


class ReviewFrameProducer(threading.Thread):
    """Save one review-eligible frame per picked cam every M seconds.

    Rotates round-robin so no camera monopolizes the save quota; each
    save produces the frame JPEG PLUS a metadata JSON that the existing
    review_frames.list_frames() picks up with zero extra wiring.
    """

    def __init__(self, picked_slots: list[dict], model,
                 interval_s: float = DEFAULT_REVIEW_FRAME_INTERVAL_S,
                 max_per_cam: int = 20):
        super().__init__(daemon=True, name="local-review-frame-producer")
        self.picked_slots = list(picked_slots)
        self.model = model
        self.interval_s = float(interval_s)
        self.max_per_cam = int(max_per_cam)
        self.stop_event = threading.Event()

    def run(self) -> None:
        try:
            self._loop()
        except Exception as e:
            print(f"[local_producers.ReviewFrameProducer] crashed: "
                  f"{type(e).__name__}: {e}")

    def _loop(self) -> None:
        import cv2
        from app.cameras import CAMERAS
        REVIEW_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        idx = 0
        while not self.stop_event.is_set():
            t0 = time.time()
            if not self.picked_slots:
                if self.stop_event.wait(self.interval_s):
                    break
                continue
            if _analysis_active():
                # Advanced Analysis owns the CPU - skip this round.
                if self.stop_event.wait(self.interval_s):
                    break
                continue
            slot = self.picked_slots[idx % len(self.picked_slots)]
            idx += 1
            slot_id = slot.get("slot_id") or slot.get("id")
            cam_id = slot.get("cam_id") or slot_id
            cam = CAMERAS.get(cam_id) or slot
            frame, boxes = _grab_and_detect(cam, self.model)
            if frame is None:
                if self.stop_event.wait(self.interval_s):
                    break
                continue
            cam_dir = REVIEW_FRAMES_DIR / slot_id
            cam_dir.mkdir(parents=True, exist_ok=True)
            # Cap disk usage: at most max_per_cam frames per cam.
            existing = sorted(cam_dir.glob("*.jpg"),
                              key=lambda p: p.stat().st_mtime)
            while len(existing) >= self.max_per_cam:
                oldest = existing.pop(0)
                for ext in (".jpg", ".json"):
                    try: oldest.with_suffix(ext).unlink()
                    except OSError: pass
            ts_ns = int(time.time() * 1000_000)   # ns-ish unique key
            jpg = cam_dir / f"{ts_ns}.jpg"
            js = cam_dir / f"{ts_ns}.json"
            cv2.imwrite(str(jpg), frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            H, W = frame.shape[:2]
            meta_boxes = []
            for bi, b in enumerate(boxes or []):
                meta_boxes.append({
                    "id":   bi,
                    "cls":  b.get("cls") or "?",
                    "conf": round(float(b.get("conf") or 0), 3),
                    "x1":   int(b.get("x1", 0)),
                    "y1":   int(b.get("y1", 0)),
                    "x2":   int(b.get("x2", 0)),
                    "y2":   int(b.get("y2", 0)),
                })
            js.write_text(json.dumps({
                "cam_id":   slot_id,
                "slot_id":  slot_id,
                "cam_name": slot.get("placeholder_name")
                           or cam.get("name") or cam_id,
                "at":       time.time(),
                "img_w":    int(W),
                "img_h":    int(H),
                "boxes":    meta_boxes,
            }))
            elapsed = time.time() - t0
            if self.stop_event.wait(max(1.0, self.interval_s - elapsed)):
                break


_RUNNING: dict[str, threading.Thread] = {}


def start_all(picked_slots: list[dict], model,
              model_view_interval_s: float = DEFAULT_MODEL_VIEW_INTERVAL_S,
              review_interval_s: float = DEFAULT_REVIEW_FRAME_INTERVAL_S,
              ) -> dict[str, threading.Thread]:
    """Idempotently launch both producers. Second calls are ignored while
    the previous instances are still alive - the notebook's Section 7
    can safely re-run without piling up duplicates."""
    stop_all()   # any previous instances get torn down first
    mv = ModelViewProducer(picked_slots, model,
                           interval_s=model_view_interval_s)
    rf = ReviewFrameProducer(picked_slots, model,
                             interval_s=review_interval_s)
    mv.start()
    rf.start()
    _RUNNING["model_view"] = mv
    _RUNNING["review_frames"] = rf
    return dict(_RUNNING)


def stop_all() -> None:
    for name, t in list(_RUNNING.items()):
        try:
            t.stop_event.set()
        except Exception:
            pass
        _RUNNING.pop(name, None)
