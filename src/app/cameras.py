"""Camera catalog for the single-camera live dashboard.

Each entry: `kind` is one of
  "hls"          direct .m3u8 (used as-is)
  "youtube"      YouTube live URL, iframe embed on frontend; backend requires
                 yt-dlp cookies to resolve HLS for detection
  "skyline"      skylinewebcams.com page, resolved via detect_core.resolve_skyline
  "webcamera24"  webcamera24.com page, resolved via detect_core.resolve_webcamera24
  "local_file"   absolute path to an uploaded MP4/MKV/MOV/AVI/WEBM

Optional per-entry keys:
  page  - the human-facing webcam page (also the resolver input for skyline/webcamera24)
  embed - iframe URL for the live player (auto-derived from `url` for youtube kind)
  conf  - per-camera YOLO confidence override
  line  - virtual counting line [[x1,y1], [x2,y2]] in normalized 0..1 coords
  loiter_person_sec / loiter_vehicle_sec - override the presence dwell thresholds
"""
from __future__ import annotations

import json
import time
from pathlib import Path


CAMERAS: dict[str, dict] = {
    # --- Thailand: street / beach-road / nightlife / traffic ---
    "th_sukhumvit": {
        "name":  "Sukhumvit Rd (Bangkok)", "city": "Bangkok", "country": "thailand",
        "kind":  "youtube",
        "url":   "https://www.youtube.com/watch?v=Q71sLS8h9a4",
        "page":  "https://webcamera24.com/camera/thailand/sukhumvit-street/",
        "embed": "https://www.youtube.com/embed/Q71sLS8h9a4?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "th_chaweng_hooters": {
        "name":  "Chaweng Beach Rd (Koh Samui)", "city": "Koh Samui", "country": "thailand",
        "kind":  "youtube",
        "url":   "https://www.youtube.com/watch?v=VR-x3HdhKLQ",
        "page":  "https://webcamera24.com/camera/thailand/7108-hooters-cam-chaweng-live-street-webcam-stream-p-hd/",
        "embed": "https://www.youtube.com/embed/VR-x3HdhKLQ?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "th_nanai_road": {
        "name":  "Nanai Rd (Patong)", "city": "Patong", "country": "thailand",
        "kind":  "youtube",
        "url":   "https://www.youtube.com/watch?v=WSm_r0eNl1E",
        "page":  "https://webcamera24.com/camera/thailand/nanai-road-cam/",
        "embed": "https://www.youtube.com/embed/WSm_r0eNl1E?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "th_patong_sainamyen": {
        "name":  "Sainamyen Rd (Patong)", "city": "Patong", "country": "thailand",
        "kind":  "youtube",
        "url":   "https://www.youtube.com/watch?v=_nvG0c9keWI",
        "page":  "https://webcamera24.com/camera/thailand/patong-sainamyen-rd-cam/",
        "embed": "https://www.youtube.com/embed/_nvG0c9keWI?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "th_petchaburi_traffic": {
        "name":  "Petchaburi Rd traffic (Bangkok)", "city": "Bangkok", "country": "thailand",
        "kind":  "youtube",
        "url":   "https://www.youtube.com/watch?v=a_bUVExv_Cg",
        "page":  "https://webcamera24.com/camera/thailand/petchaburi-road-traffic-cam/",
        "embed": "https://www.youtube.com/embed/a_bUVExv_Cg?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "th_green_mango": {
        "name":  "Soi Green Mango (Chaweng)", "city": "Koh Samui", "country": "thailand",
        "kind":  "youtube",
        "url":   "https://www.youtube.com/watch?v=DwKCna1mumk",
        "page":  "https://webcamera24.com/camera/thailand/7098-hush-bar-soi-green-mango-chaweng-live-street-webcam-stream-p-hd/",
        "embed": "https://www.youtube.com/embed/DwKCna1mumk?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "th_sukhumvit_soi11": {
        "name":  "Sukhumvit Soi 11 - El Gaucho (Bangkok)", "city": "Bangkok", "country": "thailand",
        "kind":  "youtube",
        "url":   "https://www.youtube.com/watch?v=UemFRPrl1hk",
        "page":  "https://www.youtube.com/watch?v=UemFRPrl1hk",
        "embed": "https://www.youtube.com/embed/UemFRPrl1hk?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "th_bophut_el_gaucho": {
        "name":  "El Gaucho - Fisherman's Village (Bophut, Koh Samui)", "city": "Koh Samui", "country": "thailand",
        "kind":  "youtube",
        "url":   "https://www.youtube.com/watch?v=FyFAqPHBKiQ",
        "page":  "https://www.youtube.com/watch?v=FyFAqPHBKiQ",
        "embed": "https://www.youtube.com/embed/FyFAqPHBKiQ?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "th_chaweng_pancake": {
        "name":  "Chaweng - Pancake Man (Koh Samui)", "city": "Koh Samui", "country": "thailand",
        "kind":  "youtube",
        "url":   "https://www.youtube.com/watch?v=e9T0L_POAOk",
        "page":  "https://www.youtube.com/watch?v=e9T0L_POAOk",
        "embed": "https://www.youtube.com/embed/e9T0L_POAOk?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "th_chaweng_murphys": {
        "name":  "Chaweng - Murphy's Irish Pub (Koh Samui)", "city": "Koh Samui", "country": "thailand",
        "kind":  "youtube",
        "url":   "https://www.youtube.com/watch?v=OBJ5Q0lWbqk",
        "page":  "https://www.youtube.com/watch?v=OBJ5Q0lWbqk",
        "embed": "https://www.youtube.com/embed/OBJ5Q0lWbqk?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
}

# Stamp id + country on every entry so every consumer sees them without
# having to look up the key.
for _cid, _cam in CAMERAS.items():
    _cam.setdefault("id", _cid)
    _cam.setdefault("country", "thailand")


def active_cameras() -> dict[str, dict]:
    """Cameras that have a usable URL (skips placeholders)."""
    return {k: v for k, v in CAMERAS.items() if v.get("url")}


# ---------------------------------------------------------------------------
# User-drawn counting line: dashboard POSTs to /api/lines, the analysis loop
# picks up the file within a few seconds. Lives at src/data/lines/<cam>.json.
# ---------------------------------------------------------------------------

LINE_ALLOWED_CLASSES = frozenset({
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
})


def _lines_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "lines"


def _valid_line_shape(line) -> bool:
    return (isinstance(line, list) and len(line) == 2
            and all(isinstance(pt, list) and len(pt) == 2
                    and all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0
                            for v in pt)
                    for pt in line))


def _valid_classes(classes) -> bool:
    """None means "count every class"; a non-empty list of allowed names
    means "count only these". An empty list is rejected."""
    if classes is None:
        return True
    if not isinstance(classes, list) or not classes:
        return False
    return all(isinstance(c, str) and c in LINE_ALLOWED_CLASSES
               for c in classes)


def resolve_line(cam_id: str) -> list | None:
    """Return the counting line to use. User override beats CAMERAS[cam]["line"];
    None when neither exists. Malformed overrides fall back silently."""
    p = _lines_dir() / f"{cam_id}.json"
    if p.exists():
        try:
            data = json.loads(p.read_text())
            line = data.get("line")
            if _valid_line_shape(line):
                return line
        except (OSError, ValueError):
            pass
    return CAMERAS.get(cam_id, {}).get("line")


def resolve_line_classes(cam_id: str) -> list | None:
    """None = count every tracked class; a list = only these classes."""
    p = _lines_dir() / f"{cam_id}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        classes = data.get("classes")
        if classes is None:
            return None
        if _valid_classes(classes):
            return list(classes)
    except (OSError, ValueError):
        pass
    return None


def save_line(cam_id: str, line: list, classes: list | None = None) -> None:
    if not _valid_line_shape(line):
        raise ValueError(
            "line must be exactly two [x, y] points with 0 <= x,y <= 1")
    if not _valid_classes(classes):
        raise ValueError(
            f"classes must be null or a non-empty list of names from "
            f"{sorted(LINE_ALLOWED_CLASSES)}")
    d = _lines_dir()
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "line": [[float(line[0][0]), float(line[0][1])],
                 [float(line[1][0]), float(line[1][1])]],
        "set_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if classes is not None:
        payload["classes"] = list(classes)
    (d / f"{cam_id}.json").write_text(json.dumps(payload))


def clear_line(cam_id: str) -> bool:
    p = _lines_dir() / f"{cam_id}.json"
    try:
        p.unlink()
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# User-drawn analysis zones (loiter / parking polygons). Lives at
# src/data/zones/<cam>.json. Same write-from-dashboard / read-from-analysis
# contract as the lines file.
# ---------------------------------------------------------------------------

ZONE_KINDS = frozenset({"loiter", "parking"})


def _zones_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "zones"


def _valid_zone(z) -> bool:
    if not isinstance(z, dict):
        return False
    if z.get("kind") not in ZONE_KINDS:
        return False
    pts = z.get("points")
    if not (isinstance(pts, list) and len(pts) >= 3
            and all(isinstance(p, list) and len(p) == 2
                    and all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0
                            for v in p)
                    for p in pts)):
        return False
    d = z.get("dwell_s")
    if d is not None and not (isinstance(d, (int, float)) and 5 <= d <= 3600):
        return False
    name = z.get("name")
    if name is not None and not (isinstance(name, str) and len(name) <= 24):
        return False
    return True


def resolve_zones(cam_id: str) -> list:
    """The user-drawn zones for this camera ([] when none / malformed)."""
    p = _zones_dir() / f"{cam_id}.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return []
    zones = data.get("zones")
    if not isinstance(zones, list):
        return []
    return [z for z in zones if _valid_zone(z)]


def save_zones(cam_id: str, zones: list) -> None:
    if not isinstance(zones, list) or len(zones) > 24:
        raise ValueError("zones must be a list of at most 24 entries")
    if not all(_valid_zone(z) for z in zones):
        raise ValueError("invalid zone entry (kind/points/dwell_s/name)")
    d = _zones_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{cam_id}.json").write_text(
        json.dumps({"zones": zones, "set_at": time.time()}))


def clear_zones(cam_id: str) -> bool:
    p = _zones_dir() / f"{cam_id}.json"
    try:
        p.unlink()
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Per-camera confidence-calibration file (tools/calibrate_conf.py output).
# Optional; loaded once at import and re-run by reload_review_overrides()
# so hot-swap of a fresh calibration takes effect without a restart.
# ---------------------------------------------------------------------------

PER_CAMERA_CONF_PATH = (Path(__file__).resolve().parent.parent
                        / "data" / "per_camera_conf.json")


def _merge_per_camera_conf(data: dict | None = None) -> None:
    if data is None:
        try:
            data = json.loads(PER_CAMERA_CONF_PATH.read_text())
        except (OSError, ValueError):
            return
    for cam_id, cls_map in (data.get("cameras") or {}).items():
        cam = CAMERAS.get(cam_id)
        if not cam:
            continue
        pcc = dict(cam.get("per_class_conf") or {})
        for cls, entry in (cls_map or {}).items():
            try:
                pcc[cls] = float(entry["conf"])
            except (KeyError, TypeError, ValueError):
                continue
        if pcc:
            cam["per_class_conf"] = pcc


def reload_review_overrides() -> None:
    """Hot-reload hook (called by the collector timer). Category B/C were
    cut so only per-camera confidence calibration remains."""
    _merge_per_camera_conf()


_merge_per_camera_conf()
