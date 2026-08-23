"""Shared detection + stream-access core.

Imported by the notebook, the collector daemon, and the Streamlit app so the
detection logic lives in exactly one place.
"""
from __future__ import annotations

import os
import re
import ssl
import tempfile
import threading
import time
import urllib.request

import cv2
import numpy as np

# HLS/RTSP timeouts. Without them a stream that opens but never delivers
# packets (CDN routing weirdness, tvkur backend hung, geo-blocking that
# stalls the response mid-handshake) leaves `cv2.VideoCapture.read()`
# blocked for tens of seconds. The collector runs slots serially, so one
# stuck stream drags every other camera's round with it and the operator
# sees "empty frame" on all four tiles.
#
# The env-var route works for the ffmpeg backend (the one used for HLS
# over http): it needs to be set BEFORE the first VideoCapture so ffmpeg
# picks it up at library init. Later, the explicit CAP_PROP_*_TIMEOUT_MSEC
# properties nudge cases the env var doesn't cover.
# 20s default, env-tunable. 8s proved too tight in production: kamerayayin
# segments range 2.5-5.5 MB per stream, and from GCP us-east1 the heavier
# ones (beyazit 4.3MB, buyuk camlica 5.4MB) do not finish downloading in
# 8s - so the VM decoded ONLY the lightest stream (taksim, 2.5MB) and
# every other Istanbul camera MISSed with an empty frame. 20s still bounds
# a genuinely-hung stream (the reason these timeouts exist) while letting
# heavy-but-alive segments through.
_STREAM_OPEN_TIMEOUT_MS = int(os.environ.get("STREAM_OPEN_TIMEOUT_MS") or 20000)
_STREAM_READ_TIMEOUT_MS = int(os.environ.get("STREAM_READ_TIMEOUT_MS") or 20000)
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    (f"stimeout;{_STREAM_OPEN_TIMEOUT_MS * 1000}"      # microseconds
     f"|rw_timeout;{_STREAM_READ_TIMEOUT_MS * 1000}"
     "|reconnect;1|reconnect_streamed;1|reconnect_delay_max;5"
     # Single-threaded H.264 decode per capture: with four persistent
     # 1080p readers, ffmpeg's default per-stream thread pool (cores)
     # oversubscribed the 4-core laptop so badly that OpenVINO inference
     # stretched from 0.66s (isolated) to 6-12s inside the server
     # process. One decode thread per stream keeps 25fps comfortably
     # and leaves the cores to the model.
     "|threads;1"),
)

# Same oversubscription control for the compute libraries: OpenVINO owns
# the heavy math; torch only does pre/post-processing here, and cv2's own
# parallel ops (CLAHE, resize) are short - neither deserves a full pool
# that preempts the inference threads.
try:
    cv2.setNumThreads(2)
except Exception:
    pass
try:
    import torch as _torch
    _torch.set_num_threads(1)
except Exception:
    pass


def _open_cap(url_or_path: str) -> "cv2.VideoCapture":
    """cv2.VideoCapture(url) with the timeouts applied.

    A stream that never delivers packets used to block .read() for tens of
    seconds; the collector's serial slot loop then stalled every camera.
    hasattr guards keep this working on older OpenCV builds where the
    CAP_PROP_*_TIMEOUT_MSEC properties don't exist.
    """
    cap = cv2.VideoCapture(url_or_path, cv2.CAP_FFMPEG)
    for name, ms in (("CAP_PROP_OPEN_TIMEOUT_MSEC", _STREAM_OPEN_TIMEOUT_MS),
                     ("CAP_PROP_READ_TIMEOUT_MSEC", _STREAM_READ_TIMEOUT_MS)):
        prop = getattr(cv2, name, None)
        if prop is not None:
            try:
                cap.set(prop, ms)
            except cv2.error:
                pass
    # Record the container's real frame rate for the speed/track dt. Every
    # km/h in the pipeline used to divide by an ASSUMED 25 fps - a 12.5 fps
    # stream then reported doubled speeds systematically. The collector and
    # the deep window both run grab -> analyze serially, so "fps of the last
    # opened capture" is the fps of the frames being analyzed.
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if 4.0 <= fps <= 65.0:
            global _LAST_STREAM_FPS
            _LAST_STREAM_FPS = fps
    except cv2.error:
        pass
    return cap

# COCO class ids we care about for *business activity* (footfall + vehicles
# + rail). `train` was added after a metro train crossing the frame went
# unlabeled - the model classifies it as class 6, but without id 6 in the
# `classes=` filter YOLO silently drops it before we ever see the box.
# Serializes every model.predict in this process - see the comment at the
# call site in detect_with_boxes. RLock so a locked caller may re-enter.
_PREDICT_LOCK = threading.RLock()

# Torch checkpoints loaded on demand when a static-shape OpenVINO engine
# rejects an explicit imgsz (keyed by the .pt path; see detect_with_boxes).
_TORCH_FALLBACK: dict = {}

# Toggle: when yt-dlp is blocked by YouTube's bot check, resolve to a
# sentinel URL that grab_frame handles by capturing pixels from the
# operator's primary display via mss (see src/app/screen_capture.py).
_SCREEN_CAPTURE_FALLBACK = (
    os.environ.get("SCREEN_CAPTURE_FALLBACK") or "").strip().lower() in (
        "1", "true", "yes", "on")

CLASSES_OF_INTEREST = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "train": 6,
    "truck": 7,
}
NAME_BY_ID = {v: k for k, v in CLASSES_OF_INTEREST.items()}
# `train` is intentionally excluded from `vehicles`: a metro/tram flows at a
# completely different rate than road traffic and mixing them would corrupt
# the per-camera baselines. It shows up as its own count on cameras that
# look at rail.
VEHICLE_NAMES = ("bicycle", "car", "motorcycle", "bus", "truck")

# Animal consolidation (2026-08-17, operator request). COCO ships six
# distinct animal ids (14 bird, 15 cat, 16 dog, 17 horse, 18 sheep,
# 19 cow); on a street cam the distinction is noise - one "animal"
# count reads more cleanly on the line and event log than six seldom-
# occupied per-species buckets. NAME_BY_ID maps every animal id to
# the single label "animal" so any downstream code that reads
# box["cls"] sees the same string regardless of species. EXTRA_CLASSES
# still wins for anyone who wants species labels back: assigning
# NAME_BY_ID[14] = "bird" through _apply_extra_classes overrides the
# animal mapping for that specific id.
_ANIMAL_LABEL = "animal"
_ANIMAL_COCO_IDS = frozenset({14, 15, 16, 17, 18, 19})
for _aid in _ANIMAL_COCO_IDS:
    NAME_BY_ID[_aid] = _ANIMAL_LABEL


# ---- Inference-backend selector (2026-08-17) ---------------------------
# Reports which runtime is actually available on this machine so the
# operator gets a real answer from the dashboard capability chip
# instead of a rumor. Prefers OpenVINO on Intel (2-3x faster inference
# on the CPUs this project runs on) and falls back to plain CPU torch
# when OpenVINO is not installed / has no devices exposed. The pick is
# CACHED to ~/.yolo26_pref.json so a re-invocation returns the same
# answer without re-probing (probe cost is negligible - the cache is
# more about giving the operator a stable label between restarts).
_BACKEND_PREF_PATH = None


def _backend_pref_path():
    global _BACKEND_PREF_PATH
    if _BACKEND_PREF_PATH is None:
        try:
            from pathlib import Path as _P
            _BACKEND_PREF_PATH = _P.home() / ".yolo26_pref.json"
        except Exception:
            _BACKEND_PREF_PATH = None
    return _BACKEND_PREF_PATH


def _probe_openvino_devices() -> list[str]:
    """Return the list of OpenVINO CPU/GPU devices the runtime can see.
    Empty list on any failure (openvino not installed, no compatible
    device). Cheap - milliseconds - but only called once per session."""
    try:
        import openvino as ov
        return list(ov.Core().available_devices)
    except Exception:
        return []


def select_backend() -> dict:
    """Choose the inference backend for this process. Returns a dict:
        {"backend": "openvino"|"cpu",
         "device":  "CPU"|"GPU.0"|"cpu-torch",
         "openvino_devices": [...],
         "cached":  bool}
    Cached pref is only trusted when its cached openvino_devices still
    match today's probe (an operator installing / uninstalling OpenVINO
    between runs would otherwise get a stale label)."""
    devices = _probe_openvino_devices()
    p = _backend_pref_path()
    cached: dict | None = None
    if p is not None:
        try:
            import json as _json
            cached = _json.loads(p.read_text())
        except (OSError, ValueError):
            cached = None
    if (cached
            and cached.get("openvino_devices") == devices
            and cached.get("backend") in ("openvino", "cpu")):
        cached["cached"] = True
        return cached
    if devices:
        # Prefer GPU when the runtime lists one (Intel iGPU / discrete);
        # otherwise CPU. Both are still under the openvino backend so
        # load_model()'s _openvino_model detection continues to work.
        pref = next((d for d in devices if d.startswith("GPU")), "CPU")
        chosen = {"backend": "openvino", "device": pref,
                  "openvino_devices": devices, "cached": False}
    else:
        chosen = {"backend": "cpu", "device": "cpu-torch",
                  "openvino_devices": [], "cached": False}
    if p is not None:
        try:
            import json as _json
            p.write_text(_json.dumps(chosen))
        except OSError:
            pass
    return chosen


def system_info() -> dict:
    """Compact dashboard payload. Pairs the backend chip with the
    canonical detector name (yolo26m) so the operator sees BOTH the
    runtime and the weights it feeds - a fast backend paired with the
    wrong weights is worth surfacing."""
    b = select_backend()
    label = "YOLO26 (CPU)"
    if b["backend"] == "openvino":
        label = f"YOLO26 OpenVINO ({b['device']})"
    sc_info: dict = {"enabled": bool(_SCREEN_CAPTURE_FALLBACK), "bbox": None}
    if _SCREEN_CAPTURE_FALLBACK:
        try:
            from app.screen_capture import get_region
            r = get_region()
            sc_info["bbox"] = list(r) if r else None
        except Exception:
            pass
    return {
        "model": label,
        "backend": b["backend"],
        "device": b["device"],
        "openvino_devices": b["openvino_devices"],
        "detector_weight": "yolo26m",
        "cached": b["cached"],
        "screen_capture": sc_info,
    }


def load_model(weights: str = "yolov8s.pt"):
    """Load a YOLO model once and reuse it.

    Default is `yolov8s` (small) rather than `yolov8n` (nano). Nano's
    recall on the wide overhead street views these cameras produce is too
    low: it silently drops distant/static vehicles, mis-fires `person` on
    upright thin road furniture, and often mis-classifies a partially-cropped
    car at the frame edge as `bicycle`. Small is the smallest tier where those
    three failure modes back off to acceptable levels. CPU cost is ~3x nano
    per burst - still a fraction of the collector's sampling interval.

    If the active-learning loop has promoted a trained Detect head
    (data/adapters/current.json), it is overlaid here; no adapter file
    means the base weights run untouched, bit-identical (plan D6).
    """
    from ultralytics import YOLO

    # Prefer a sibling OpenVINO export when one exists: on the Intel CPUs
    # this project actually runs on it cuts inference time by ~2-3x for
    # bit-comparable outputs (Ultralytics' documented OpenVINO gain),
    # which is the difference between 1 and 3 usable analysis ticks per
    # second-scale interval. Export once with:
    #   YOLO("src/yolo26m.pt").export(format="openvino", imgsz=640)
    # The RL adapter overlay only applies to torch checkpoints, so when
    # the OpenVINO path is taken the adapter (none is promoted anyway)
    # is skipped with a log line rather than a crash.
    if str(weights).endswith(".pt"):
        import os as _os
        _ov_dir = str(weights)[:-3] + "_openvino_model"
        if _os.path.isdir(_ov_dir):
            model = YOLO(_ov_dir)
            # Remember the torch checkpoint so detect_with_boxes can fall
            # back to it when a caller asks for an imgsz the static engine
            # was not exported for (e.g. the calibration cells' 960 on a
            # 640 engine, LIVE_IMGSZ=512 on a 640 engine). The .pt lives
            # in the repo root by default (notebook downloads it there),
            # so accept the file from either src/ or root - anything to
            # avoid the mid-tick 113 MB Ultralytics auto-download.
            _pt = str(weights)
            if not _os.path.isfile(_pt):
                from pathlib import Path as _Path
                _bare = _Path(_pt).name
                for _c in (_Path(__file__).resolve().parent.parent / _bare,
                           _Path(__file__).resolve().parent.parent.parent
                             / _bare):
                    if _c.is_file():
                        _pt = str(_c)
                        break
            if _os.path.isfile(_pt):
                model._pt_fallback_path = _pt
                print(f"detect: OpenVINO engine loaded ({_ov_dir}) - "
                      f"torch fallback ready at {_pt}")
            else:
                print(f"detect: OpenVINO engine loaded ({_ov_dir}) - "
                      f"torch fallback disabled (no .pt on disk; "
                      f"non-native imgsz will raise instead of downloading)")
            print(f"detect: OpenVINO engine loaded ({_ov_dir}) - "
                  f"adapter overlay skipped (torch-only)")
            return model

    model = YOLO(weights)
    try:
        # adapters was removed with Category C; the loader is kept in a
        # try/except so a future re-introduction reads its file without
        # touching this call site.
        raise ImportError("adapters removed with Category C")
    except Exception as e:
        print(f"load_model: adapter overlay skipped ({type(e).__name__}: {e})")
    return model


# yt-dlp innertube clients tried in order until one hands back a real HLS
# manifest. YouTube retires/breaks specific clients on a rolling schedule
# so the list is kept broad on purpose - the try loop stops at the first
# success. Auth-cookie-friendly clients FIRST so a session with a cookies
# file wins on its first attempt; legacy anonymous clients at the tail.
_YT_PLAYER_CLIENTS = (
    os.environ.get("YT_PLAYER_CLIENTS")
    or "web,web_safari,android_producer,ios_music,default,android,ios,tv"
).split(",")

# PO-token provider (2026-07-29): YouTube starves Google-datacenter IPs -
# streams resolve, but googlevideo serves them no data ("opened but
# produced no frames" on every YT camera from the VM, while the same
# streams play 1080p from a residential IP). The remedy yt-dlp documents
# is a PO token minted by the bgutil provider; in SCRIPT mode (a node
# script invoked per resolution - nothing resident, which matters on the
# 1 GB e2-micro) the plugin only needs this env var pointing at the
# transpiled generate_once.js. Unset = the exact previous behavior; the
# plugin package must also be pip-installed for the arg to matter (see
# deploy/gcp-vm/setup_pot_provider.sh).
_YT_POT_SCRIPT = (os.environ.get("YT_POT_SCRIPT") or "").strip()

# Authenticated-session cookies (YT_COOKIES_FILE): the last free lever
# against the YouTube bot-check ("Sign in to confirm you're not a bot").
# A Netscape-format cookies.txt exported from a logged-in browser session.
# When the var is unset or the file is missing, yt-dlp is anonymous.
_YT_COOKIES_FILE = (os.environ.get("YT_COOKIES_FILE") or "").strip()


def _yt_extractor_args(client: str) -> dict:
    args = {"youtube": {"player_client": [client.strip()]}}
    if _YT_POT_SCRIPT:
        args["youtubepot-bgutilscript"] = {"script_path": [_YT_POT_SCRIPT]}
    return args


def _yt_opts(client: str) -> dict:
    """yt-dlp options for one resolution attempt. Cookies attach only
    when the env-referenced file really exists; every call re-reads the
    env so a notebook that sets YT_COOKIES_FILE after import picks it up
    on the very next resolve."""
    opts = {"quiet": True, "no_warnings": True,
            "format": "best[protocol^=m3u8]/best",
            "extractor_args": _yt_extractor_args(client)}
    cookies = (os.environ.get("YT_COOKIES_FILE")
               or _YT_COOKIES_FILE or "").strip()
    if cookies and os.path.isfile(cookies):
        opts["cookiefile"] = cookies
    return opts


_YT_HLS_MANIFEST_RE = re.compile(r'"hlsManifestUrl"\s*:\s*"([^"]+)"')


def _yt_scrape_hls(url: str) -> str | None:
    """Last-resort YouTube resolver: fetch the watch page like a normal
    browser and pull `hlsManifestUrl` out of the player JSON.

    Exists because yt-dlp's innertube calls periodically hit YouTube's
    "Sign in to confirm you're not a bot" wall (observed killing the
    Sainamyen cam mid-session), while a plain browser-UA page fetch of
    the same video keeps working. No cookies, no yt-dlp.
    """
    try:
        html = _http_get(url, _BROWSER_HEADERS).decode("utf-8", "replace")
    except Exception:
        return None
    m = _YT_HLS_MANIFEST_RE.search(html)
    if not m:
        return None
    return (m.group(1)
            .replace("\\/", "/")
            .replace("\\u0026", "&"))


def resolve_youtube(url: str) -> str:
    """Resolve a YouTube Live (or webcamera24 YouTube-backed) page to an HLS
    .m3u8 URL. Tries each configured innertube client until one yields a
    stream, then falls back to scraping the watch page itself - the scrape
    frequently survives the bot-wall that blocks yt-dlp's API clients."""
    import yt_dlp

    if not url.startswith("http"):                      # bare 11-char video id
        url = f"https://www.youtube.com/watch?v={url}"
    last = None
    for client in _YT_PLAYER_CLIENTS:
        opts = _yt_opts(client)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
            if info.get("url"):
                return info["url"]
        except Exception as e:
            last = e
    scraped = _yt_scrape_hls(url)
    if scraped:
        return scraped
    raise RuntimeError(f"youtube: no client resolved a stream ({last})")


# ---- Resolved-stream cache ------------------------------------------------
# resolve_stream() is called every sampling round for every slot. For direct
# HLS that is free, but a YouTube resolve shells out to yt-dlp (~3-5 s on the
# e2-micro) and a webcamera24 resolve fetches+scrapes a page. Doing that four
# times a round, every 40 s, would dominate the loop and hammer the origin.
# The googlevideo manifest URL carries its own `expire=<unixts>`; tvkur/skyline
# tokens rotate on a similar timescale. Cache the resolved URL per camera and
# reuse it until shortly before it expires (or a grab fails and clears it).
_RESOLVE_TTL_FALLBACK = int(os.environ.get("RESOLVE_TTL_FALLBACK") or 900)
_RESOLVE_FAIL_BACKOFF_S = int(os.environ.get("RESOLVE_FAIL_BACKOFF") or 60)


def _resolve_cache_file():
    from pathlib import Path
    return (Path(__file__).resolve().parent.parent / "data"
            / "resolve_cache.json")


def _load_resolve_cache() -> dict:
    """Warm the resolve cache from disk at import.

    Signed googlevideo URLs stay valid for ~6 h; before this, a server
    restart threw every one of them away and forced fresh resolves - and
    the night YouTube's bot-wall was up, that one restart took down ALL
    four cameras at once even though their old URLs were still perfectly
    good. Only positive, unexpired entries are loaded.
    """
    import json as _json
    try:
        raw = _json.loads(_resolve_cache_file().read_text())
    except (OSError, ValueError):
        return {}
    now = time.time()
    out = {}
    for k, v in raw.items():
        if (isinstance(v, list) and len(v) == 2 and v[0]
                and isinstance(v[1], (int, float)) and v[1] > now):
            out[k] = (v[0], float(v[1]))
    return out


def _persist_resolve_cache() -> None:
    import json as _json
    try:
        p = _resolve_cache_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps(
            {k: [v[0], v[1]] for k, v in _RESOLVE_CACHE.items() if v[0]}))
    except OSError:
        pass


_RESOLVE_CACHE: dict = _load_resolve_cache()
                                   # cam_id -> (url | None, good_until)
                                   # url None = negative entry (backoff)
_EXPIRE_RE = re.compile(r"[?&/]expire[/=](\d{10})")


def _expiry_of(url: str, now: float) -> float:
    m = _EXPIRE_RE.search(url)
    if m:
        # Re-resolve 2 min before the manifest actually expires.
        return int(m.group(1)) - 120
    return now + _RESOLVE_TTL_FALLBACK


def invalidate_resolved(cam_id: str) -> None:
    """Drop a camera's cached stream URL - call after a failed grab so the
    next round re-resolves instead of retrying a stale token/manifest."""
    _RESOLVE_CACHE.pop(cam_id, None)


# Browser-ish headers: webcamera24 and skylinewebcams both 403 bare urllib fetchers.
_BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

# skylinewebcams page -> the tokenized HLS it points at (token rotates, so resolve live).
_SKYLINE_RE = re.compile(r'(?:source|src)\s*[:=]\s*["\']([^"\']*?live[^"\']*?\.m3u8[^"\']*)["\']',
                         re.IGNORECASE)
_SKYLINE_HOST = "https://hd-auth.skylinewebcams.com/"

# webcamera24 pages embed a tvkur player; pull its id and build the master playlist.
_TVKUR_ID_RE = re.compile(r'(?:player\.tvkur\.com/l/|content\.tvkur\.com/l/)([a-z0-9]+)',
                          re.IGNORECASE)
_YOUTUBE_RE = re.compile(r'(?:youtube\.com/(?:embed/|watch\?v=)|youtu\.be/)([\w-]{11})')


def resolve_skyline(page_url: str) -> str:
    """Resolve a skylinewebcams.com webcam page to its tokenized HLS .m3u8 URL.

    The page embeds the playlist as `source:"livee.m3u8?a=<token>"` (relative to
    hd-auth.skylinewebcams.com). The token rotates, so call this each cycle.
    """
    html = _http_get(page_url, _BROWSER_HEADERS).decode("utf-8", "replace")
    m = _SKYLINE_RE.search(html)
    if not m:
        raise RuntimeError("skyline: no HLS source found on page (layout changed or geo-blocked)")
    src = m.group(1)
    if src.startswith("http"):
        return src
    return _SKYLINE_HOST + src.lstrip("/")


def resolve_webcamera24(page_url: str) -> str:
    """Resolve a webcamera24.com page to an HLS URL via its embedded tvkur/YouTube player."""
    html = _http_get(page_url, _BROWSER_HEADERS).decode("utf-8", "replace")
    m = _TVKUR_ID_RE.search(html)
    if m:
        return f"https://content.tvkur.com/l/{m.group(1)}/master.m3u8"
    y = _YOUTUBE_RE.search(html)
    if y:
        return resolve_youtube(f"https://www.youtube.com/watch?v={y.group(1)}")
    raise RuntimeError("webcamera24: no tvkur/YouTube player found on page")


def _resolve_uncached(cam: dict) -> str:
    kind = cam.get("kind", "hls")
    # Local uploaded files carry `path` instead of `url`; hand the path
    # straight through so OpenCV VideoCapture opens it as a file source.
    if kind == "local_file":
        return cam.get("path") or cam.get("url") or ""
    url = cam["url"]
    if kind == "hls":
        return url
    if kind == "youtube":
        # With SCREEN_CAPTURE_FALLBACK=1 skip yt-dlp entirely and go straight
        # to screen capture. yt-dlp sometimes succeeds on this machine and
        # returns a manifest URL, but the googlevideo CDN then blocks the
        # actual .ts segment fetch (30-second FFmpeg timeout) - so relying on
        # yt-dlp gives an unusable stream. When the operator opted into the
        # screen-capture fallback they've already accepted that the video is
        # rendered by the visible iframe player; capturing that is more
        # reliable than fighting the CDN.
        if _SCREEN_CAPTURE_FALLBACK:
            from app.screen_capture import SCREEN_CAPTURE_SENTINEL
            return SCREEN_CAPTURE_SENTINEL
        # Primary path: yt-dlp resolves the manifest URL directly. Without
        # the fallback, a yt-dlp failure raises so the caller can honestly
        # say the camera is unreachable and the operator can pick another.
        return resolve_youtube(url)
    if kind == "skyline":
        return resolve_skyline(cam.get("page", url))
    if kind == "webcamera24":
        return resolve_webcamera24(cam.get("page", url))
    if kind == "screen":
        # Explicit screen-capture camera. Register in cameras.py with
        # url: "screen://primary" and optionally an env-provided bbox.
        from app.screen_capture import SCREEN_CAPTURE_SENTINEL
        return SCREEN_CAPTURE_SENTINEL
    raise ValueError(f"unknown camera kind: {kind!r}")


def resolve_stream(cam: dict, now: float | None = None) -> str:
    """Resolve any catalog camera dict to a directly-openable stream URL by
    `kind`. Direct HLS is returned as-is; YouTube/skyline/webcamera24 pages
    are resolved live and CACHED per camera until the manifest nears expiry,
    so the collector pays the yt-dlp / page-scrape cost once per token
    lifetime rather than once per sampling round. Pass a stable `cam['id']`
    to enable caching; without an id the resolve is always live."""
    kind = cam.get("kind", "hls")
    if kind == "hls":
        return cam["url"]
    if kind == "local_file":
        return cam.get("path") or cam.get("url") or ""
    cam_id = cam.get("id")
    now = time.time() if now is None else now
    if cam_id:
        hit = _RESOLVE_CACHE.get(cam_id)
        if hit and now < hit[1]:
            if hit[0] is None:
                # Negative-cache hit: the last resolve failed. Raising
                # instantly (instead of shelling out to yt-dlp again)
                # is what stops a dead camera from spawning a resolver
                # subprocess every two seconds for hours.
                raise RuntimeError("resolve backing off after failure "
                                   f"({cam_id})")
            return hit[0]
    try:
        resolved = _resolve_uncached(cam)
    except Exception:
        if cam_id:
            _RESOLVE_CACHE[cam_id] = (None, now + _RESOLVE_FAIL_BACKOFF_S)
        raise
    if cam_id:
        _RESOLVE_CACHE[cam_id] = (resolved, _expiry_of(resolved, now))
        _persist_resolve_cache()
    return resolved


def invalidate_stream(cam_id: str) -> None:
    """Drop the cached resolve for one camera. The live-analysis loop
    calls this after repeated grab failures so the next resolve_stream()
    re-runs yt-dlp/page-scraping instead of re-knocking an expired
    manifest until its natural expiry."""
    _RESOLVE_CACHE.pop(cam_id, None)


_SSL_CTX = ssl._create_unverified_context()

# Some live-CDN HLS endpoints (e.g. content.tvkur.com) require a Referer/Origin header
# that ffmpeg-via-cv2 can't always pass on Windows. For those hosts we fetch the latest
# .ts segment manually and decode locally.
HEADER_HOSTS = {
    "content.tvkur.com":          {"Referer": "https://player.tvkur.com/",
                                   "Origin":  "https://player.tvkur.com"},
    "livestream.ibb.gov.tr":      {"Referer": "https://istanbuluseyret.ibb.gov.tr/",
                                   "Origin":  "https://istanbuluseyret.ibb.gov.tr"},
    "kamerayayin.ibb.istanbul":   {"Referer": "https://istanbuluseyret.ibb.gov.tr/",
                                   "Origin":  "https://istanbuluseyret.ibb.gov.tr"},
    "skylinewebcams.com":         {"Referer": "https://www.skylinewebcams.com/",
                                   "Origin":  "https://www.skylinewebcams.com"},
}

# Manual segment downloads (the header-host path): 30s ceiling, env-tunable.
# 15s proved too tight for kamerayayin - its playlists offer a SINGLE 1080p
# rendition with 2.5-5.5 MB segments, and from GCP us-east1 the heavier ones
# did not finish in 15s, so only the lightest camera (taksim) ever delivered.
_SEGMENT_HTTP_TIMEOUT_S = int(os.environ.get("SEGMENT_HTTP_TIMEOUT_S") or 30)


# ---- IBB proxy relay (Cloudflare Worker) --------------------------------
# `kamerayayin.ibb.istanbul` returns HTTP 403 to every GCP IP range - it
# treats Google Cloud as a scraping ASN. Every other origin (residential,
# Cloudflare edge, other clouds) sees 200 with the full HLS chain. When
# IBB_PROXY_URL is set, every IBB request is rewritten to go through the
# operator's Cloudflare Worker (see deploy/cloudflare-proxy/) which
# fetches from a non-GCP address and hands the bytes back. The URL is
# encoded as the worker's PATH so response bodies with relative segment
# names still resolve through the worker on the follow-up fetch.
# Empty env vars = no proxy, current behavior preserved.
_IBB_PROXY_URL = (os.environ.get("IBB_PROXY_URL") or "").rstrip("/")
_IBB_PROXY_SECRET = os.environ.get("IBB_PROXY_SECRET") or ""
_IBB_PROXY_HOSTS = ("kamerayayin.ibb.istanbul",)


def _apply_ibb_proxy(url: str,
                     extra_headers: dict | None) -> tuple[str, dict | None]:
    """When configured, route an IBB URL through the Cloudflare Worker.

    Rewrites `https://kamerayayin.ibb.istanbul/X.m3u8` to
    `<worker>/https://kamerayayin.ibb.istanbul/X.m3u8` and adds the
    shared-secret header so the worker accepts the call. All other URLs
    pass through unchanged - the worker's own allow-list would refuse them
    anyway, but we save the round trip.
    """
    if not (_IBB_PROXY_URL and _IBB_PROXY_SECRET):
        return url, extra_headers
    from urllib.parse import urlparse
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return url, extra_headers
    if host not in _IBB_PROXY_HOSTS:
        return url, extra_headers
    hdrs = dict(extra_headers or {})
    hdrs["X-Proxy-Secret"] = _IBB_PROXY_SECRET
    return f"{_IBB_PROXY_URL}/{url}", hdrs

def _http_get(url: str, extra_headers: dict | None = None,
              max_bytes: int | None = None) -> bytes:
    """GET with browser-ish headers. `max_bytes` truncates the body - the
    Konya-era trick of decoding a LIGHT rendition, recreated for hosts that
    only serve 1080p: a frame grab needs the first ~MB of a segment (the
    leading I-frame), not the whole 5 MB."""
    url, extra_headers = _apply_ibb_proxy(url, extra_headers)
    h = {"User-Agent": "Mozilla/5.0"}
    if extra_headers:
        h.update(extra_headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=_SEGMENT_HTTP_TIMEOUT_S,
                                context=_SSL_CTX) as r:
        if max_bytes is None:
            return r.read()
        chunks: list[bytes] = []
        got = 0
        while got < max_bytes:
            c = r.read(min(262_144, max_bytes - got))
            if not c:
                break
            chunks.append(c)
            got += len(c)
        return b"".join(chunks)

# How much of a segment a short grab downloads. ~2.5 MB holds well over a
# second of 1080p video - plenty for a single frame or a strided 2-frame
# burst - while capping the transfer at what the lightest kamerayayin
# camera (the only one that reliably worked) used to cost.
_SEGMENT_BYTE_BUDGET = int(os.environ.get("SEGMENT_BYTE_BUDGET") or 2_500_000)


# ---- Grab-failure reporting ---------------------------------------------
# Every fetch/decode problem below used to be swallowed (`except: return`),
# so the collector journal showed one opaque "MISS (RuntimeError: empty
# frame)" for FOUR different failure modes. During the 2026-07-16 all-miss
# incident that mask hid the actual cause for hours: diagnosing it required
# knowing WHICH stage (playlist / chunklist / segment / decode) died and
# with what HTTP code. These helpers keep the LAST failure so the collector
# can append it to its MISS line. Single-threaded use only (the collector
# samples slots serially); grab_burst() resets it at the start of a grab.
_LAST_GRAB_ERROR: str | None = None
_LAST_GRAB_STAGE: str | None = None
_LAST_GRAB_HTTP: int | None = None


def last_grab_error() -> str | None:
    """Stage + cause of the most recent failed grab (None after a clean one)."""
    return _LAST_GRAB_ERROR


def last_grab_http() -> tuple[str | None, int | None]:
    """(stage, http_status) of the most recent failed grab. The status is
    set only when the failure was an HTTP error - it lets the collector
    tell an ACCESS REFUSAL (403/429, the host blocking this address) from
    a dead channel (404) or a network problem, and drive the host-level
    circuit breaker off the real signal instead of string-matching logs."""
    return _LAST_GRAB_STAGE, _LAST_GRAB_HTTP


def _reset_grab_error() -> None:
    global _LAST_GRAB_ERROR, _LAST_GRAB_STAGE, _LAST_GRAB_HTTP
    _LAST_GRAB_ERROR = None
    _LAST_GRAB_STAGE = None
    _LAST_GRAB_HTTP = None


def _note_grab_failure(stage: str, err) -> None:
    global _LAST_GRAB_ERROR, _LAST_GRAB_STAGE, _LAST_GRAB_HTTP
    detail = f"{type(err).__name__}: {err}" if isinstance(err, Exception) else str(err)
    _LAST_GRAB_ERROR = f"{stage}: {detail}"
    _LAST_GRAB_STAGE = stage
    _LAST_GRAB_HTTP = getattr(err, "code", None) if isinstance(err, Exception) else None


# Playlists must be revalidated, never served stale: wowza rotates the
# chunklist token (chunklist_w<session>.m3u8) on restart and the segment
# list every few seconds, so a stale edge-cached playlist chains into an
# instant 404 one level down - four cameras failing in ~1.7s each with
# "empty frame" is exactly what that looks like. Segments themselves are
# immutable and stay cacheable.
def _no_cache(headers: dict) -> dict:
    h = dict(headers)
    h["Cache-Control"] = "no-cache"
    return h


_STREAM_INF_RES_RE = re.compile(r"RESOLUTION=(\d+)x(\d+)")
_STREAM_INF_BW_RE  = re.compile(r"BANDWIDTH=(\d+)")


def _pick_variant(pl: str, min_height: int = 640) -> str | None:
    """Choose which rendition of a master playlist to decode.

    The old behavior took the FIRST variant, which CDNs list as the highest
    bitrate - so the collector was downloading and H.264-decoding 1080p
    only to letterbox it down to a 512-640px YOLO input. On the e2-micro's
    shared cores that decode dominates the whole sampling round.

    Detection quality is set by the model's imgsz, not the source pixels,
    as long as the source is at least ~as tall as the input: pick the
    SMALLEST rendition whose height still covers `min_height`, breaking
    ties by bandwidth. When nothing is tall enough (or the playlist lists
    no RESOLUTION), fall back to the tallest thing offered.
    """
    cands: list[tuple[int, int, str]] = []   # (height, bandwidth, uri)
    lines = pl.splitlines()
    for i, l in enumerate(lines):
        if not l.startswith("#EXT-X-STREAM-INF"):
            continue
        uri = next((x.strip() for x in lines[i + 1:]
                    if x.strip() and not x.startswith("#")), None)
        if not uri:
            continue
        m = _STREAM_INF_RES_RE.search(l)
        h = int(m.group(2)) if m else 0
        b = _STREAM_INF_BW_RE.search(l)
        bw = int(b.group(1)) if b else 0
        cands.append((h, bw, uri))
    if not cands:
        return None
    tall = [c for c in cands if c[0] >= min_height]
    if tall:
        tall.sort(key=lambda c: (c[0], c[1]))
        return tall[0][2]
    cands.sort(key=lambda c: (-c[0], c[1]))
    return cands[0][2]

def _grab_via_segment(stream_url: str, headers: dict) -> np.ndarray | None:
    """Download the most recent .ts segment with the right headers and decode it."""
    base = stream_url.rsplit("/", 1)[0] + "/"
    pl = _http_get(stream_url, _no_cache(headers)).decode("utf-8", "replace")
    if "#EXT-X-STREAM-INF" in pl:
        variant = _pick_variant(pl) or next(
            (l.strip() for l in pl.splitlines()
             if l.strip() and not l.startswith("#")), None)
        if not variant:
            return None
        variant_url = variant if variant.startswith("http") else base + variant
        pl = _http_get(variant_url, _no_cache(headers)).decode("utf-8", "replace")
        base = variant_url.rsplit("/", 1)[0] + "/"
    segs = [l.strip() for l in pl.splitlines() if l.strip() and not l.startswith("#")]
    if not segs:
        return None
    # Newest segment first; one already rotated off the CDN edge 404s
    # instantly, so walk back up to two older siblings before giving up.
    for seg in segs[::-1][:3]:
        seg_url = seg if seg.startswith("http") else base + seg
        try:
            # One frame needs the segment's leading I-frame, not all 5 MB.
            data = _http_get(seg_url, headers, max_bytes=_SEGMENT_BYTE_BUDGET)
        except Exception as e:
            _note_grab_failure("segment", e)
            continue
        with tempfile.NamedTemporaryFile(suffix=".ts", delete=False) as f:
            f.write(data); tmp = f.name
        try:
            cap = _open_cap(tmp)
            ok, frame = cap.read()
            cap.release()
            if ok:
                return frame
        finally:
            try: os.unlink(tmp)
            except OSError: pass
    return None


def grab_frame(stream_url: str):
    """Open an HLS/RTSP stream, read a single frame (BGR ndarray), close. None on failure.

    For hosts that need referer/origin headers, route via _grab_via_segment.
    For the screen-capture sentinel, grab pixels from the primary display.
    """
    if stream_url and stream_url.startswith("screen://"):
        try:
            from app.screen_capture import capture as _screen_capture
            return _screen_capture()
        except Exception as e:
            if _LAST_GRAB_ERROR is None:
                _note_grab_failure("screen capture", e)
            return None
    for host, headers in HEADER_HOSTS.items():
        if host in stream_url:
            try:
                return _grab_via_segment(stream_url, headers)
            except Exception as e:
                # Keep the FIRST failure of this grab: iter_frames already
                # stage-tagged it; the retry usually hits the same wall.
                if _LAST_GRAB_ERROR is None:
                    _note_grab_failure("single-frame grab", e)
                return None
    cap = _open_cap(stream_url)
    try:
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def detect_with_boxes_batch(model, frames, conf: float = 0.35,
                            imgsz: int | None = None,
                            per_class_conf_list=None,
                            conf_list=None,
                            classes: list | None = None,
                            agnostic_nms: bool = False):
    """Batched detect_with_boxes: ONE model forward over N frames.

    On the CPU-only laptop a batch-of-4 forward costs ~2.5x a single frame
    instead of 4x, which is what makes four concurrent live-analysis
    sessions tick at a usable rate. Post-filtering (per-class gates,
    person plausibility, rider rescue) is reused verbatim from
    detect_with_boxes via its `_res` shortcut, one frame at a time, so
    batch results are bit-identical to serial ones.

    Returns a list of (counts, boxes) tuples, one per input frame.
    """
    frames = list(frames)
    if not frames:
        return []
    gates_list = list(per_class_conf_list or [None] * len(frames))
    confs = list(conf_list or [conf] * len(frames))
    # The shared model gate must be the loosest any frame needs, so no
    # frame's candidates are dropped before its own filters get to look.
    floors = [conf]
    for g, c in zip(gates_list, confs):
        floors.append(min(g.values()) if g else c)
    model_gate = max(0.001, min(floors))
    # Sequential per-frame inference, deliberately: OpenVINO's static
    # batch-1 engine is the FAST configuration on CPU (a dynamic-shape
    # export measured ~4x slower per frame), and torch's cross-frame
    # batching gain was modest. The batcher still coalesces sessions -
    # one INFER_LOCK hold, synchronized rounds - it just walks the
    # frames one forward pass at a time.
    return [detect_with_boxes(model, frames[i], conf=confs[i], imgsz=imgsz,
                              per_class_conf=gates_list[i],
                              classes=classes, agnostic_nms=agnostic_nms)
            for i in range(len(frames))]


def iter_frames(stream_url: str, max_frames: int, stride: int = 1):
    """Yield up to `max_frames` frames from a live HLS stream, `stride`
    source-frames apart (stride=1 keeps the old consecutive behavior).

    For header-required hosts (content.tvkur.com, livestream.ibb.gov.tr, skylinewebcams.com)
    cv2.VideoCapture(url) can't pass Referer/Origin on Windows, so we download the latest
    few .ts segments with the right headers and decode them locally - yielding frames in
    arrival order. For normal HLS we open the URL directly with cv2 and read.

    Skipped frames use cap.grab() rather than cap.read(): H.264 still has to
    decode them (P-frames reference their predecessors) but grab() skips the
    BGR conversion + copy, which is a solid slice of the per-frame cost on
    the collector's shared vCPUs. At burst stride 13 that's 12 cheap grabs
    per kept frame.

    Used with stride=1 by the dwell-time / tracking section of the notebook
    so ByteTrack can see the consecutive frames it needs.
    """
    stride = max(1, int(stride))
    _reset_grab_error()
    # header-required host: fetch enough segments to cover the strided span
    matching_headers = None
    for host, headers in HEADER_HOSTS.items():
        if host in stream_url:
            matching_headers = headers
            break

    if matching_headers is not None:
        base = stream_url.rsplit("/", 1)[0] + "/"
        try:
            pl = _http_get(stream_url, _no_cache(matching_headers)).decode("utf-8", "replace")
        except Exception as e:
            _note_grab_failure("playlist", e)
            return
        if "#EXT-X-STREAM-INF" in pl:
            variant = _pick_variant(pl) or next(
                (l.strip() for l in pl.splitlines()
                 if l.strip() and not l.startswith("#")), None)
            if not variant:
                _note_grab_failure("playlist", "master playlist lists no variant")
                return
            variant_url = variant if variant.startswith("http") else base + variant
            try:
                pl = _http_get(variant_url, _no_cache(matching_headers)).decode("utf-8", "replace")
            except Exception as e:
                _note_grab_failure("chunklist", e)
                return
            base = variant_url.rsplit("/", 1)[0] + "/"

        segs = [l.strip() for l in pl.splitlines() if l.strip() and not l.startswith("#")]
        if not segs:
            _note_grab_failure("chunklist", "no segments listed")
            return
        # tail segments give the freshest live view; pull ~enough to cover
        # the whole strided span ((max_frames-1)*stride + 1 source frames)
        approx_frames_per_seg = 60   # 2 s @ 30 fps is a typical segment
        span = (max_frames - 1) * stride + 1
        n_segs = max(1, min(len(segs), -(-span // approx_frames_per_seg)))
        # Short spans (the collector's 2-frame strided burst) live in the
        # first second of a segment - cap the transfer like the Konya-era
        # light-rendition path did. Dense spans (the notebook's stride=1
        # dwell burst) need the whole segment.
        budget = _SEGMENT_BYTE_BUDGET if span <= approx_frames_per_seg else None
        # The newest segment can rotate off the CDN edge between the
        # chunklist fetch and the .ts fetch (an instant 404). Keep two older
        # siblings as a rescue path, used ONLY when the tail produced
        # nothing at all - so dense dwell bursts keep their time order.
        tail = segs[-n_segs:]
        rescue = segs[:-n_segs][-2:][::-1]          # next-freshest first
        yielded = 0
        idx = 0
        fetched_ok = 0
        for pos, seg in enumerate(tail + rescue):
            if yielded >= max_frames:
                break
            if pos >= len(tail) and yielded:
                break                     # rescue only a totally-empty grab
            seg_url = seg if seg.startswith("http") else base + seg
            try:
                data = _http_get(seg_url, matching_headers, max_bytes=budget)
            except Exception as e:
                _note_grab_failure("segment", e)
                continue
            fetched_ok += 1
            with tempfile.NamedTemporaryFile(suffix=".ts", delete=False) as f:
                f.write(data); tmp = f.name
            try:
                cap = _open_cap(tmp)
                try:
                    while yielded < max_frames:
                        if idx % stride == 0:
                            ok, fr = cap.read()
                            if not ok:
                                break
                            yielded += 1
                            idx += 1
                            yield fr
                        else:
                            if not cap.grab():
                                break
                            idx += 1
                finally:
                    # Must release even when the consumer closes the generator
                    # mid-yield (grab_burst stops after n frames) - otherwise
                    # the capture still holds the file and the unlink below
                    # fails on Windows, leaking one temp .ts per burst.
                    cap.release()
            finally:
                try: os.unlink(tmp)
                except OSError: pass
        if yielded == 0 and fetched_ok:
            _note_grab_failure(
                "decode", f"{fetched_ok} segment(s) downloaded, 0 frames decoded")
        return

    # normal HLS / RTSP: stream directly
    cap = _open_cap(stream_url)
    yielded = 0
    idx = 0
    try:
        while yielded < max_frames:
            if idx % stride == 0:
                ok, fr = cap.read()
                if not ok:
                    break
                yielded += 1
                idx += 1
                yield fr
            else:
                if not cap.grab():
                    break
                idx += 1
        if yielded == 0:
            _note_grab_failure("stream", "opened but produced no frames")
    finally:
        cap.release()


# ---- Region-of-interest (ROI) filtering --------------------------------------
# A camera entry may carry a "roi" polygon (and/or "roi_exclude" polygons) in
# NORMALIZED coordinates (0..1 relative to frame width/height), so one config
# works across stream resolutions. A detection belongs to the ROI when its
# FOOT POINT - bottom-center of the box, where the object touches the ground -
# is inside the polygon. That excludes parked-car lots, sky, and neighboring
# roofs without clipping pedestrians whose heads poke outside the zone.

def point_in_polygon(x: float, y: float, poly: list) -> bool:
    """Ray-casting test; poly is [[x1,y1], [x2,y2], ...] in any unit."""
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _foot_point(b: dict) -> tuple[float, float]:
    return (b["x1"] + b["x2"]) / 2.0, b["y2"]


def filter_boxes_roi(boxes: list[dict], frame_shape,
                     roi: list | None,
                     roi_exclude: list | None = None,
                     roi_exclude_class: dict | None = None) -> list[dict]:
    """Keep boxes whose foot point is inside `roi` (if set) and outside every
    polygon in `roi_exclude`. Polygons use normalized 0..1 coordinates.

    `roi_exclude_class` is per-class: `{cls_name: [poly, poly, ...]}`. A box
    is dropped when its foot point falls inside ANY polygon listed for its
    own class. This lets a camera's config say "never accept `person` in the
    top-left corner (there's a lamp post there)" without hiding real cars
    that pass through the same pixels.
    """
    if not roi and not roi_exclude and not roi_exclude_class:
        return boxes
    H, W = frame_shape[:2]
    kept = []
    for b in boxes:
        fx, fy = _foot_point(b)
        nx, ny = fx / W, fy / H
        if roi and not point_in_polygon(nx, ny, roi):
            continue
        if roi_exclude and any(point_in_polygon(nx, ny, p) for p in roi_exclude):
            continue
        if roi_exclude_class:
            polys = roi_exclude_class.get(b.get("cls")) or ()
            if any(point_in_polygon(nx, ny, p) for p in polys):
                continue
        kept.append(b)
    return kept


def counts_from_boxes(boxes: list[dict]) -> dict:
    """Recompute the per-class count dict (incl. 'vehicles') from a box list."""
    counts = {name: 0 for name in CLASSES_OF_INTEREST}
    for b in boxes:
        if b.get("cls") in counts:
            counts[b["cls"]] += 1
    counts["vehicles"] = sum(counts[v] for v in VEHICLE_NAMES)
    return counts


# ---- Burst tracking + virtual-line crossing -----------------------------------
# The burst gives a short consecutive window (~n frames, ~1s apart). Matching
# detections across those frames by nearest centroid yields short tracks; a
# camera with a configured "line" ([[x1,y1],[x2,y2]] normalized) then counts
# how many tracks CROSSED it, and in which direction. Because the collector
# only observes ~2-3s out of every interval, the numbers are a SAMPLED flow
# rate - comparable over time on the same camera, not an absolute turnstile.

def _centroid(b: dict) -> tuple[float, float]:
    return (b["x1"] + b["x2"]) / 2.0, (b["y1"] + b["y2"]) / 2.0


# ---- Burst-based vehicle speed estimation --------------------------------------
# Rides the same 3-frame burst (frames ~stride/fps apart) and the same greedy
# centroid tracks. The trick that avoids per-camera calibration: every vehicle
# is its own ruler. Its class has a typical real-world length, so the box's
# pixel extent calibrates meters-per-pixel AT THAT SPOT in the frame, and the
# centroid displacement between burst frames converts straight to m/s. This
# is an ESTIMATE: viewing angle folds the true length into the projection
# (a head-on car shows its 1.8 m width, not its 4.5 m length), so the error
# band is roughly +-30-50%. Good for "crawling vs city-speed vs fast", not
# for issuing speeding tickets - the UI labels it accordingly.

# Typical HLS street cam frame rate; grab_burst spaces frames `stride` frames
# apart, so dt between burst frames = stride / fps. Fallback only - _open_cap
# records each stream's REAL container fps into _LAST_STREAM_FPS and
# last_stream_fps() serves it to the speed/track dt computations.
BURST_FPS_ASSUMED = 25.0
_LAST_STREAM_FPS: float | None = None


def last_stream_fps(default: float = BURST_FPS_ASSUMED) -> float:
    """Frame rate of the most recently opened stream (sanity-clamped at
    open time), or `default` when nothing was measured yet. Valid because
    both callers (collector round, deep window) grab and analyze one
    camera at a time."""
    return _LAST_STREAM_FPS if _LAST_STREAM_FPS else default


def _line_side(px: float, py: float, line: list) -> float:
    """Signed side of point vs the line A->B (cross product z)."""
    (ax, ay), (bx, by) = line
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


# Night-time gate bump. Sodium/LED point-lights after dark make storefronts
# read as `bus` and shrubbery as `motorcycle` (both observed on operator
# screenshots), while the review-driven boosts are tuned on daylight scenes
# and sit too low at night. When the frame reads as night (see the
# collector's NIGHT_MEAN_GRAY), every class gate rises by this much.
NIGHT_CONF_BUMP = 0.08


def night_adjusted_conf(per_class_conf: dict,
                        bump: float = NIGHT_CONF_BUMP) -> dict:
    """Return a copy of a per-class gate map with the night bump applied,
    clamped to 0.8 so a class can never become undetectable."""
    return {c: min(0.8, float(v) + bump) for c, v in per_class_conf.items()}


# Per-class confidence gate applied AFTER the model's own conf filter.
# `person`, `car`, `bus`, `truck` at 0.35 keep the model honest on the classes
# it's usually confident about. `train` sits at 0.25 because a partial-view
# tram or metro car at street-camera angles rarely lands above 0.35 in
# practice, and losing it entirely (as the user reported) is a worse failure
# than the occasional low-confidence false positive at that class.
DEFAULT_PER_CLASS_CONF = {
    "person":     0.35,
    "bicycle":    0.22,
    "motorcycle": 0.22,
    "car":        0.35,
    "bus":        0.35,
    "train":      0.25,
    "truck":      0.35,
    # 2026-08-17: animals opted into LIVE_CLASSES (line layer needs to
    # count them). Every COCO animal id (14-19) shows up under the
    # single "animal" label by way of NAME_BY_ID - see the animal
    # consolidation block above. One gate at 0.30 covers all of
    # them; permissive enough that a street dog / cat / horse
    # actually clears, tight enough that texture noise doesn't
    # hallucinate one on an empty road.
    "animal":     0.30,
}

# ---- Opt-in extra classes (EXTRA_CLASSES env) --------------------------------
# The seven-class business set above is deliberate; everything downstream
# (review, relabel, calibration, training export) is built around it. But
# some scenes have a legitimate extra subject - a square full of pigeons,
# a canal with boats - and losing them at the `classes=` filter means no
# counting, no heatmap, no tracking. EXTRA_CLASSES adds COCO classes to
# DETECTION ONLY: "bird" or "bird:0.30,dog" (optional per-class gate,
# default 0.30). Counts/heatmap/tracker pick them up automatically;
# the review/training chain intentionally does NOT - extras never mint
# labels, so they cannot skew the fine-tuning loop. Unset env = the exact
# seven-class behavior, byte for byte.
_EXTRA_CLASS_IDS = {
    # COCO ids for the extras that make sense on a street/square camera.
    "bird": 14, "cat": 15, "dog": 16, "boat": 8,
    "backpack": 24, "umbrella": 25, "handbag": 26, "suitcase": 28,
}


def _apply_extra_classes(spec: str | None = None) -> list[str]:
    """Parse EXTRA_CLASSES and extend the class tables in place.

    Returns the class names actually added (tests use it directly and
    undo by popping the returned names from the three tables).
    """
    spec = (os.environ.get("EXTRA_CLASSES") or "").strip() if spec is None \
        else spec
    added: list[str] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, gate = item.partition(":")
        name = name.strip()
        cid = _EXTRA_CLASS_IDS.get(name)
        if cid is None or name in CLASSES_OF_INTEREST:
            if cid is None and name:
                print(f"EXTRA_CLASSES: unknown class {name!r} ignored "
                      f"(known: {', '.join(sorted(_EXTRA_CLASS_IDS))})")
            continue
        CLASSES_OF_INTEREST[name] = cid
        NAME_BY_ID[cid] = name
        try:
            DEFAULT_PER_CLASS_CONF[name] = float(gate) if gate else 0.30
        except ValueError:
            DEFAULT_PER_CLASS_CONF[name] = 0.30
        added.append(name)
    if added:
        print(f"detect_core: extra classes enabled: {', '.join(added)} "
              "(detection/counts/heatmap only - never the training chain)")
    return added


_apply_extra_classes()

# Person shape / size gates. A real pedestrian on a street cam is TALLER
# than wide (aspect >= 0.90) but NOT wildly so (aspect <= 3.0); has at
# least a couple of dozen pixels of vertical extent; and never spans more
# than a fraction of the frame's width (a "person" box wider than that is
# almost always the model misfiring on a metro car, a bus at close range,
# or a large signage board mistaken for a human).
#
# Values below were pulled in after user reports of
#   * a metro / light-rail car (spanning ~half the frame) labeled `person`;
#   * a separator pole (very narrow, tall) labeled `person` at conf 0.4;
#   * false positives on distant road furniture that clear MIN_ASPECT but
#     look nothing like a person.
DEFAULT_PERSON_MIN_ASPECT = 0.90
DEFAULT_PERSON_MAX_ASPECT = 3.0
DEFAULT_PERSON_MIN_HEIGHT_PX = 24     # smaller than this and there's no
                                       # meaningful person signal anyway
DEFAULT_PERSON_MAX_WIDTH_FRAC = 0.30   # any "person" wider than 30% of
                                       # frame width is misfiring

# "Rider co-detection": if a person box overlaps a two-wheeler that YOLO
# reported ABOVE its own gate but BELOW the per-class gate, resurrect the
# two-wheeler - a person on a motorcycle is a rider AND a vehicle, but the
# nano model often reports the person confidently and the vehicle just below
# threshold, so counting only the person under-reports vehicle traffic.
DEFAULT_RIDER_IOU = 0.30
_TWO_WHEELERS = ("bicycle", "motorcycle")


def _box_wh(b: dict) -> tuple[float, float]:
    return b["x2"] - b["x1"], b["y2"] - b["y1"]


def detect_and_count(model, frame, conf: float = 0.35, imgsz: int | None = None) -> dict:
    """Run YOLO on one frame -> {class_name: count} for the classes we track."""
    counts, _ = detect_with_boxes(model, frame, conf=conf, imgsz=imgsz)
    return counts


def detect_with_boxes(model, frame, conf: float = 0.35,
                      imgsz: int | None = None,
                      per_class_conf: dict | None = None,
                      person_min_aspect: float | None = DEFAULT_PERSON_MIN_ASPECT,
                      person_max_aspect: float | None = DEFAULT_PERSON_MAX_ASPECT,
                      person_min_height_px: int | None = DEFAULT_PERSON_MIN_HEIGHT_PX,
                      person_max_width_frac: float | None = DEFAULT_PERSON_MAX_WIDTH_FRAC,
                      rider_iou: float | None = DEFAULT_RIDER_IOU,
                      classes: list | None = None,
                      agnostic_nms: bool = False,
                      _res=None,
                      ) -> tuple[dict, list[dict]]:
    """Like detect_and_count but also returns per-detection boxes.

    Detection is a two-stage filter: the model runs at the MOST PERMISSIVE
    threshold in `per_class_conf` (so nothing that any class needs is dropped
    before we can see it), then each raw detection is kept iff its confidence
    clears that class's per-class threshold. `person_min_aspect` and
    `person_max_aspect` reject person boxes whose height/width falls outside
    the plausible band - respectively the "stroller mis-read as person" case
    and the "lamp post/traffic sign mis-read as person" case. `rider_iou`
    resurrects a two-wheeler box that survived the model gate but not its
    per-class gate when it overlaps a surviving person box - a rider is a
    person AND a vehicle.

    Set `per_class_conf=None` to fall back to the single `conf` (legacy).
    Set `person_min_aspect=None` / `person_max_aspect=None` / `rider_iou=None`
    to skip that filter.

    Returns:
        counts: {class_name: int, "vehicles": int}
        boxes:  [{x1,y1,x2,y2,cls,conf}, ...] in pixel coords (BGR frame).
    """
    # Assemble the effective per-class gate. When the caller passes a single
    # legacy `conf`, per_class_conf=None means "same threshold everywhere" -
    # older callers keep their exact behavior. When per_class_conf is used,
    # the incoming `conf` is a global floor (nothing below is asked for).
    if per_class_conf is None:
        per_cls_gate = {c: conf for c in CLASSES_OF_INTEREST}
        model_gate = conf
    else:
        per_cls_gate = {c: float(per_class_conf.get(c, conf))
                        for c in CLASSES_OF_INTEREST}
        model_gate = max(0.001, min(min(per_cls_gate.values()), conf))

    kwargs = dict(conf=model_gate,
                  classes=(classes if classes is not None
                           else list(CLASSES_OF_INTEREST.values())),
                  agnostic_nms=agnostic_nms, verbose=False)
    if imgsz:
        kwargs["imgsz"] = imgsz
    # _res lets detect_with_boxes_batch run ONE batched forward pass over
    # several frames and reuse this function purely for the per-frame
    # post-filtering (per-class gates, person plausibility, rider rescue).
    if _res is not None:
        res = _res
    else:
        # ONE predict at a time, process-wide. A YOLO() instance shares a
        # single predictor (and, under OpenVINO, a single InferRequest);
        # concurrent predict() calls from different threads - sessions,
        # producers, warmup, search - deadlocked the OV request PERMANENTLY
        # (py-spy: batcher parked inside openvino infer forever, all four
        # sessions starved behind it). torch merely tolerated the race;
        # OpenVINO does not. The lock costs microseconds next to the
        # ~0.7s forward pass.
        with _PREDICT_LOCK:
            try:
                res = model.predict(frame, **kwargs)[0]
            except RuntimeError as e:
                # A static-shape OpenVINO engine rejects any imgsz other
                # than the one it was exported with. Retry the call on the
                # sibling torch checkpoint (loaded once, cached) so
                # explicit-imgsz callers - the calibration cells, per-cam
                # imgsz overrides - work instead of crashing the run.
                ptp = getattr(model, "_pt_fallback_path", None)
                if not ptp or "compatible" not in str(e):
                    raise
                # Only fall back to torch when the .pt actually exists on
                # disk. Passing a missing bare name to YOLO() triggers a
                # ~113 MB Ultralytics auto-download mid-tick (silent, no
                # network policy prompt) - re-raise the original OpenVINO
                # error instead so the caller sees a clean failure.
                if not os.path.isfile(ptp):
                    print(f"detect: OpenVINO engine rejected "
                          f"imgsz={kwargs.get('imgsz')} but torch fallback "
                          f"({ptp}) is missing - re-raising")
                    raise
                fb = _TORCH_FALLBACK.get(ptp)
                if fb is None:
                    from ultralytics import YOLO as _YOLO
                    fb = _TORCH_FALLBACK[ptp] = _YOLO(ptp)
                    print(f"detect: OpenVINO engine rejected "
                          f"imgsz={kwargs.get('imgsz')} - torch fallback "
                          f"({ptp}) for non-native sizes")
                res = fb.predict(frame, **kwargs)[0]
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()

    # Stage 1: build the raw candidate list (everything the model returned).
    raw: list[dict] = []
    for i, c in enumerate(cls_ids):
        name = NAME_BY_ID.get(int(c))
        if not name:
            continue
        x1, y1, x2, y2 = xyxy[i].tolist()
        raw.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cls": name, "conf": float(confs[i])})

    # Stage 2: apply the per-class gate + person shape filter.
    kept: list[dict] = []
    below_gate: list[dict] = []
    for b in raw:
        gate = per_cls_gate.get(b["cls"], conf)
        if b["conf"] < gate:
            below_gate.append(b)
            continue
        if b["cls"] == "person":
            w, h = _box_wh(b)
            if person_min_height_px is not None and h < person_min_height_px:
                # Below this the box is too small to carry meaningful person
                # signal; usually a false pop on textured background.
                b["_dropped_reason"] = "person_too_small"
                continue
            if person_max_width_frac is not None:
                frame_w = frame.shape[1] if hasattr(frame, "shape") else None
                if frame_w and w > frame_w * person_max_width_frac:
                    # A "person" box wider than 30% of the frame is a metro
                    # car, a bus at close range, or a large signage board -
                    # never an actual pedestrian.
                    b["_dropped_reason"] = "person_too_wide"
                    continue
            if w > 0 and (person_min_aspect is not None
                          or person_max_aspect is not None):
                aspect = h / w
                if person_min_aspect is not None and aspect < person_min_aspect:
                    # Stroller / banner / cart shaped like a person to the model.
                    b["_dropped_reason"] = "person_aspect_low"
                    continue
                if person_max_aspect is not None and aspect > person_max_aspect:
                    # Lamp post / traffic sign / bollard: taller-and-thinner
                    # than any real pedestrian is.
                    b["_dropped_reason"] = "person_aspect_high"
                    continue
        kept.append(b)

    # Stage 3: rider co-detection - resurrect below-gate two-wheelers that
    # overlap a surviving person box. Person + motorcycle is a rider, and both
    # should be counted; without this the rider inflates the person count but
    # the vehicle disappears.
    if rider_iou is not None:
        persons = [b for b in kept if b["cls"] == "person"]
        for b in below_gate:
            if b["cls"] not in _TWO_WHEELERS:
                continue
            for p in persons:
                if box_iou(p, b) >= rider_iou:
                    b["_rescued_by_rider"] = True
                    kept.append(b)
                    break

    counts = {name: 0 for name in CLASSES_OF_INTEREST}
    counts["animal"] = 0  # synthetic label from NAME_BY_ID (COCO ids 14-19)
    boxes: list[dict] = []
    for b in kept:
        counts[b["cls"]] = counts.get(b["cls"], 0) + 1
        boxes.append({k: b[k] for k in ("x1", "y1", "x2", "y2", "cls", "conf")})
    counts["vehicles"] = sum(counts[v] for v in VEHICLE_NAMES)
    return counts, boxes


def annotate(model, frame, conf: float = 0.35, imgsz: int | None = None):
    """Run detection and return the annotated frame (BGR ndarray).

    Runs a FRESH inference - fine for notebook one-offs. The collector calls
    draw_boxes() with the detections it already has instead, so an anomalous
    sample doesn't cost a second model pass on the VM. Both paths render via
    draw_boxes so calibration images and dashboard snapshots look identical.
    """
    _, boxes = detect_with_boxes(model, frame, conf=conf, imgsz=imgsz)
    return draw_boxes(frame, boxes)


def box_iou(a: dict | None, b: dict | None) -> float:
    """IoU of two {x1,y1,x2,y2} boxes; 0.0 if either is missing/degenerate."""
    if not a or not b:
        return 0.0
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _box_diagonal(box: dict) -> float:
    """Length of the box's diagonal in pixels."""
    import math
    w = max(0.0, box["x2"] - box["x1"])
    h = max(0.0, box["y2"] - box["y1"])
    return math.hypot(w, h)


def _point_to_box_distance(px: float, py: float, box: dict) -> float:
    """Shortest distance from a point to the nearest edge of a box; 0 if
    the point is inside."""
    dx = max(box["x1"] - px, 0.0, px - box["x2"])
    dy = max(box["y1"] - py, 0.0, py - box["y2"])
    return (dx * dx + dy * dy) ** 0.5


def associate_by_iou_or_distance(anchor: dict, item: dict,
                                 iou_floor: float = 0.02,
                                 distance_frac: float = 0.6) -> bool:
    """True when `item` belongs to `anchor` under a two-stage geometric
    test - the pattern adapted from Nawaf-Rayhan585's PPE compliance
    monitor. Reason IoU alone fails for accessories (bag, phone, helmet)
    is they usually protrude past the person's box; a distance fallback
    keyed off the anchor's diagonal catches those without dragging in
    far-away detections.

    * `anchor` - the "owner" box (person, vehicle).
    * `item`   - the box we want to check ownership of (bag, PPE, plate).
    * `iou_floor` - IoU threshold; 0.02 is very permissive on purpose,
      the distance test still gates far-away items.
    * `distance_frac` - max center-to-anchor-edge distance, as a fraction
      of the anchor's diagonal. 0.6 = "within one radius of the anchor
      edge".

    Both boxes use the same {x1,y1,x2,y2} contract as `box_iou`.
    """
    if not anchor or not item:
        return False
    if box_iou(anchor, item) >= iou_floor:
        return True
    icx = (item["x1"] + item["x2"]) / 2.0
    icy = (item["y1"] + item["y2"]) / 2.0
    diag = _box_diagonal(anchor)
    if diag <= 0:
        return False
    return _point_to_box_distance(icx, icy, anchor) <= distance_frac * diag


_BOX_COLORS = {
    "person":     (80, 175, 76),    # green (BGR)
    "bicycle":    (200, 130, 0),
    "car":        (60, 130, 246),
    "motorcycle": (200, 130, 0),
    "bus":        (0, 90, 230),
    "train":      (200, 60, 200),   # magenta - rail is neither road nor foot
    "truck":      (0, 90, 230),
    "animal":     (60, 200, 220),   # amber - one color for every species
}


def draw_boxes(frame: np.ndarray, boxes: list[dict]) -> np.ndarray:
    """Annotate a COPY of `frame` with already-computed detection boxes.

    Same information as annotate() (class + confidence per box) without
    re-running the model.
    """
    out = frame.copy()
    for b in boxes:
        x1, y1 = int(b["x1"]), int(b["y1"])
        x2, y2 = int(b["x2"]), int(b["y2"])
        color = _BOX_COLORS.get(b.get("cls", ""), (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f'{b.get("cls", "?")} {b.get("conf", 0):.2f}'
        if "track_id" in b:
            # Within-burst individual number (app/tracker.py) - lets the
            # viewer tell look-alike objects apart on the annotated frame.
            label = f'#{b["track_id"]} ' + label
        if "kmh" in b:
            # burst-based estimate - "~" marks it as
            # such; 0 means matched-but-not-moving, i.e. parked.
            label += f' ~{b["kmh"]:.0f}km/h' if b["kmh"] > 0 else " parked"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = y1 - 4 if y1 - th - 6 >= 0 else y2 + th + 4
        cv2.rectangle(out, (x1, ty - th - 3), (x1 + tw + 4, ty + 3), color, -1)
        cv2.putText(out, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---- Burst sampling: several frames per sample, median count -----------------
# A single frame is a noisy estimator: a pedestrian occluded for one moment, or
# a car sitting at the edge of the confidence band, flips the count between
# consecutive frames. The collector therefore detects on a short burst and
# keeps the MEDIAN count - per-frame flicker cancels out while "now" still
# means "this handful of seconds".

def grab_burst(stream_url: str, n: int = 3, stride: int = 25) -> list[np.ndarray]:
    """Grab up to `n` frames spaced ~`stride` frames (~1 s at 25 fps) apart.

    Rides on iter_frames(), which already handles the header-required HLS hosts
    (tvkur / IBB / skyline) by decoding recent .ts segments locally. Falls back
    to the single-frame grab if the iterator yields nothing. May return fewer
    than `n` frames (short segments); callers should handle 1..n.
    """
    _reset_grab_error()
    if n <= 1:
        f = grab_frame(stream_url)
        return [] if f is None else [f]
    frames: list[np.ndarray] = []
    try:
        # Striding happens INSIDE iter_frames now: skipped frames only pay
        # cap.grab() (decode without the BGR convert+copy), not a full read.
        for fr in iter_frames(stream_url, max_frames=n, stride=stride):
            frames.append(fr)
            if len(frames) >= n:
                break
    except Exception as e:
        _note_grab_failure("burst decode", e)
    if not frames:
        # A playlist-level failure (404 dead channel, 403 access block)
        # will refuse the single-frame retry identically - same URL, same
        # headers. Skipping the duplicate knock halves the request rate
        # exactly when a blocking host is counting them.
        if _LAST_GRAB_STAGE == "playlist":
            return frames
        f = grab_frame(stream_url)
        if f is not None:
            frames = [f]
    return frames


if __name__ == "__main__":  # one-time stream-resolution check (run on an open network)
    import argparse

    from app.cameras import CAMERAS

    ap = argparse.ArgumentParser(description="Resolve a catalog camera to its live HLS URL")
    ap.add_argument("--resolve", default="", help="comma-separated cam ids (default: all)")
    args = ap.parse_args()

    ids = [c.strip() for c in args.resolve.split(",") if c.strip()] or list(CAMERAS)
    for cid in ids:
        cam = CAMERAS.get(cid)
        if not cam:
            print(f"{cid:16s} -> UNKNOWN camera id")
            continue
        try:
            print(f"{cid:16s} -> {resolve_stream(cam)}")
        except Exception as e:
            print(f"{cid:16s} -> FAILED ({e})")
