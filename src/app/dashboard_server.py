"""Dashboard HTTP server building blocks shared by serve.py and the notebook.

Serves web/ statically AND proxies tvkur/IBB streams the browser can't reach
directly due to Referer/CORS requirements:

    GET /tvkur/<stream_id>/<path>           -> content.tvkur.com/l/<stream_id>/<path>
                                               with Referer/Origin=player.tvkur.com
    GET /snapshots/...                      -> web/snapshots/... (saved detections)

The proxy adds Access-Control-Allow-Origin:* so hls.js in the dashboard can
fetch the master playlist and segments without browser CORS errors.

Env knobs (optional):
    SEARCH_YOLO  detector weights for live analysis (default yolo26x.pt;
                 set to "off" to skip loading a model).
"""
from __future__ import annotations

import http.server
import json
import os
import socket
import socketserver
import ssl
import sys
import threading
import time
import urllib.request
from pathlib import Path

# ThreadingHTTPServer is what we need: with 4 cameras each polling the HLS
# chunklist and pulling new .ts segments every few seconds (8-12 concurrent
# requests bursting in parallel), a single-threaded TCPServer queues them
# serially and the videos stall on "loading...". ThreadingHTTPServer hands
# each request to its own thread, which is what hls.js expects from a CDN.

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
SNAPSHOTS_DIR = WEB_DIR / "snapshots"
# 2026-08-23 (C2): the plates layer writes per-attempt crops to
# src/data/plate_crops/<cam>/<ts>_<tid>_<text>_<conf>.jpg as an audit
# trail. The Investigation LPR pipeline modal reads from here through
# a dedicated /plate-crops/<cam>/<file> path so B/C/D panels show real
# per-stage crops instead of CSS crops of the same saved JPEG.
PLATE_CROPS_DIR = ROOT / "data" / "plate_crops"

_TVKUR_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; turkey-footfall-dashboard)",
    "Referer":    "https://player.tvkur.com/",
    "Origin":     "https://player.tvkur.com",
}
_SSL_CTX = ssl._create_unverified_context()

# ===== Repo #2 additions: MP4/MKV upload for local file analysis =====
import uuid as _uuid
from pathlib import Path as _Path
_UPLOAD_DIR = _Path(__file__).resolve().parent.parent / "data" / "uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_ALLOWED_VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
_MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB per file


def _parse_time(v: str) -> float | None:
    """Accept ISO-8601 (`2026-07-06T18:00:00Z`), the browser's datetime-local
    format (`2026-07-06T18:00`), or a bare epoch-seconds number. Return
    epoch seconds. Empty / unparseable input returns None (open bound)."""
    v = (v or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        pass
    import datetime as _dt
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            t = _dt.datetime.strptime(v, fmt)
            # datetime-local sends naive strings; treat as UTC so the API
            # is timezone-stable across browsers.
            return t.replace(tzinfo=_dt.timezone.utc).timestamp()
        except ValueError:
            continue
    return None


class _ModelState:
    """Process-wide detector holder. The YOLO model loads once, behind a
    lock (ThreadingHTTPServer would otherwise race concurrent first
    requests into loading it twice), and every consumer - live analysis,
    local producers - shares the same instance."""

    def __init__(self):
        self._model_lock = threading.Lock()
        self._model_ready = False
        self.model = None

    def get_model(self):
        """The YOLO model, loaded on first call (~5-15s).

        The dashboard's Analyze button ends up here to load the detector
        for live analysis. `yolo26x.pt` (extra-large) ships with the repo
        along with its OpenVINO IR under src/yolo26x_openvino_model/. Env
        override kept so a slower host can drop back to yolo26m or v8s.
        """
        with self._model_lock:
            if not self._model_ready:
                weights = os.environ.get("SEARCH_YOLO", "yolo26x.pt")
                if weights.lower() not in ("off", "none", ""):
                    try:
                        from app.detect_core import load_model
                        # src-relative first (the OpenVINO IR lives there);
                        # raw-name fallback lets ultralytics auto-download.
                        src_relative = str(
                            Path(__file__).resolve().parent.parent / weights)
                        self.model = load_model(src_relative)
                    except Exception as e:
                        print(f"model warmup: YOLO unavailable ({e})")
                self._model_ready = True
        return self.model


_MODEL_STATE = _ModelState()


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Static handler for web/ + transparent tvkur HLS proxy.

    Browsers can't fetch content.tvkur.com directly:
    1. The CDN returns 403 without a Referer header (the browser sets Referer
       to the page origin, not player.tvkur.com).
    2. The CDN does NOT send Access-Control-Allow-Origin, so even if we got
       past 403, hls.js's fetch would fail browser CORS.

    Solution: when the browser asks for /tvkur/<id>/master.m3u8 we relay it
    server-side with the right Referer and add ACAO:* on the way back.
    """

    def end_headers(self) -> None:
        # No-cache for static files so JS edits show on reload (the proxy
        # path sets its own headers and returns early before reaching here).
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        # Under a Jupyter kernel this access log lands inside the notebook
        # cell's output area AND gets saved into the .ipynb (hundreds of
        # "GET /snapshots/..." lines per run bloated the file). Terminal
        # runs (serve.py) keep the log; notebook runs stay quiet.
        if "ipykernel" in sys.modules:
            return
        sys.stdout.write("  " + (fmt % args) + "\n")

    def do_GET(self) -> None:
        if self.path.startswith("/tvkur/"):
            self._proxy_tvkur()
            return
        # 2026-08-23 (C2): serve the per-attempt plate crops for the
        # Investigation LPR pipeline modal. Path layout is
        # /plate-crops/<cam>/<file>.jpg -> src/data/plate_crops/<cam>/<file>.
        if self.path.startswith("/plate-crops/"):
            self._serve_plate_crop()
            return
        path = self.path.split("?")[0]
        if path == "/api/catalog":
            self._catalog()
            return
        if path == "/api/analysis/plate-crops":
            self._analysis_plate_crops()
            return
        if path == "/api/uploaded-videos":
            self._uploaded_videos()
            return
        if path == "/api/local-file":
            self._local_file()
            return
        if path == "/api/ping":
            # Capability probe: only THIS private server answers it, so the
            # frontend can tell "operator dashboard with a backend" from the
            # hosted public copy without sniffing hostnames (which lied
            # behind proxies). Gates the send-report field + live analysis.
            self._send_json(200, {"ok": True, "private": True})
            return
        if path == "/api/analysis/frame":
            self._analysis_frame()
            return
        if path == "/api/analysis/data":
            self._analysis_data()
            return
        if path == "/api/analysis/events":
            self._analysis_events()
            return
        if path == "/api/analysis/replay":
            self._analysis_replay()
            return
        if path == "/api/analysis/saved":
            self._analysis_saved()
            return
        if path == "/api/model-metrics":
            self._model_metrics()
            return
        if path == "/api/lines":
            self._get_line()
            return
        if path == "/api/zones":
            self._get_zones()
            return
        if path == "/api/crossings":
            self._get_crossings()
            return
        if path == "/api/events.jsonl":
            self._events_jsonl()
            return
        if path == "/api/export.csv":
            self._export_csv()
            return
        if path == "/api/system":
            self._system_info()
            return
        if path == "/api/system/live":
            self._system_live()
            return
        if path == "/api/models/info":
            self._models_info()
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        if path == "/api/upload-video":
            self._upload_video()
            return
        if path == "/api/analysis/start":
            self._analysis_start()
            return
        if path == "/api/analysis/stop":
            self._analysis_stop()
            return
        if path == "/api/analysis/event/save":
            self._analysis_event_save()
            return
        if path == "/api/analysis/saved-clear":
            self._analysis_saved_clear()
            return
        if path == "/api/analysis/saved-delete":
            self._analysis_saved_delete()
            return
        if path == "/api/screen-capture/bbox":
            self._screen_capture_bbox()
            return
        if path == "/api/lines":
            self._save_line()
            return
        if path == "/api/lines/clear":
            self._clear_line()
            return
        if path == "/api/zones":
            self._save_zones()
            return
        if path == "/api/zones/clear":
            self._clear_zones()
            return
        self.send_error(404, "unknown POST endpoint")

    def do_DELETE(self) -> None:
        path = self.path.split("?")[0]
        if path == "/api/uploaded-video":
            self._delete_uploaded_video()
            return
        self.send_error(404, "unknown DELETE endpoint")

    # ---- Line-crossing config + event log --------------------------------
    # The Line layer in the dashboard lets the operator draw a virtual
    # counting line on a snapshot; every crossing then produces a toast +
    # a crop in the history strip. Three endpoints:
    #   GET  /api/lines?cam=<id>       -> {"line": [[x,y],[x,y]] | null, "set_at": ...}
    #   POST /api/lines?cam=<id>       body: {"line": [[x,y],[x,y]]}
    #   POST /api/lines/clear?cam=<id> -> delete the override, back to cameras.py
    #   GET  /api/crossings?cam=<id>&limit=20 -> newest-first events

    def _q_cam(self):
        """Extract the ?cam= query arg. Returns cam_id or None (and writes 400)."""
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(self.path).query)
        cam = (q.get("cam") or [""])[0].strip()
        if not cam:
            self.send_error(400, "missing ?cam=")
            return None
        return cam

    def _get_line(self) -> None:
        cam = self._q_cam()
        if cam is None:
            return
        # Route through resolve_line + resolve_line_classes so a malformed
        # override on disk falls back to the CAMERAS catalog silently -
        # the same rule the collector follows on the next round. Reading
        # the JSON here without the validator would let a bad hand-edit
        # paint a line the frontend believes in but the counter never uses.
        from app.cameras import (LINE_ALLOWED_CLASSES, _lines_dir,
                                 resolve_line, resolve_line_classes)
        p = _lines_dir() / f"{cam}.json"
        set_at = None
        if p.exists():
            try:
                set_at = json.loads(p.read_text()).get("set_at")
            except (OSError, ValueError):
                set_at = None
        line = resolve_line(cam)
        classes = resolve_line_classes(cam)
        body = json.dumps({"cam": cam, "line": line,
                           "classes": classes,
                           "allowed_classes": sorted(LINE_ALLOWED_CLASSES),
                           "set_at": set_at,
                           "user_override": p.exists()}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def _save_line(self) -> None:
        cam = self._q_cam()
        if cam is None:
            return
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > 1024:
            self.send_error(400, "empty or oversized body"); return
        try:
            data = json.loads(self.rfile.read(n).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self.send_error(400, "body must be JSON"); return
        line = data.get("line")
        classes = data.get("classes")
        from app.cameras import save_line
        try:
            save_line(cam, line, classes=classes)
        except ValueError as e:
            self.send_error(400, str(e)); return
        body = json.dumps({"ok": True, "cam": cam, "line": line,
                           "classes": classes}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def _clear_line(self) -> None:
        cam = self._q_cam()
        if cam is None:
            return
        from app.cameras import clear_line
        removed = clear_line(cam)
        body = json.dumps({"ok": True, "cam": cam, "removed": removed}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    # ---- Analysis zones (loiter areas + parking spots) -------------------
    #   GET  /api/zones?cam=<id>        -> {"zones": [...]}
    #   POST /api/zones?cam=<id>        body: {"zones": [...]} (full replace)
    #   POST /api/zones/clear?cam=<id>  -> delete the file
    # Running loiter/parking sessions hot-reload within a few seconds.

    def _get_zones(self) -> None:
        cam = self._q_cam()
        if cam is None:
            return
        from app.cameras import resolve_zones
        self._send_json(200, {"zones": resolve_zones(cam)})

    def _save_zones(self) -> None:
        cam = self._q_cam()
        if cam is None:
            return
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > 64 * 1024:
            self.send_error(400, "empty or oversized body"); return
        try:
            data = json.loads(self.rfile.read(n).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self.send_error(400, "body must be JSON"); return
        from app.cameras import save_zones
        try:
            save_zones(cam, data.get("zones"))
        except ValueError as e:
            self.send_error(400, str(e)); return
        self._send_json(200, {"ok": True, "cam": cam,
                              "count": len(data.get("zones") or [])})

    def _clear_zones(self) -> None:
        cam = self._q_cam()
        if cam is None:
            return
        from app.cameras import clear_zones
        self._send_json(200, {"ok": True, "removed": clear_zones(cam)})

    def _get_crossings(self) -> None:
        cam = self._q_cam()
        if cam is None:
            return
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(self.path).query)
        limit = 20
        try:
            limit = max(1, min(200, int((q.get("limit") or ["20"])[0])))
        except ValueError:
            pass
        from app.live_analysis import read_crossing_events
        events = read_crossing_events(cam, limit=limit)
        body = json.dumps({"cam": cam, "events": events}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def _system_info(self) -> None:
        """GET /api/system - inference backend + detector weight the
        server is running with. Answers the operator's "am I on the
        fast path?" question without opening the terminal."""
        try:
            from app.detect_core import system_info
            self._send_json(200, {"ok": True, **system_info()})
        except Exception as e:
            self._send_json(500, {"ok": False,
                                  "error": f"{type(e).__name__}: {e}"})

    def _system_live(self) -> None:
        """GET /api/system/live - CPU / RAM / GPU utilisation snapshot
        for the status-bar HUD above the video player. Polled every few
        seconds by the frontend. psutil is the only hard dep (already in
        requirements.txt via ultralytics); pynvml is best-effort so
        machines without an NVIDIA GPU still get a clean payload.
        """
        out: dict = {"ok": True}
        try:
            import psutil
            # Short-interval sample (0.15 s) so the first call returns a
            # real percent instead of 0.0 (psutil.cpu_percent(interval=None)
            # returns 0.0 on the FIRST call of each new client - it needs
            # two samples in the same process to compute a delta).
            out["cpu_pct"] = float(psutil.cpu_percent(interval=0.15))
            out["cpu_count"] = psutil.cpu_count(logical=True)
            vm = psutil.virtual_memory()
            out["ram_pct"] = float(vm.percent)
            out["ram_used_gb"] = round(vm.used / (1024 ** 3), 1)
            out["ram_total_gb"] = round(vm.total / (1024 ** 3), 1)
        except Exception as e:
            out["cpu_err"] = f"{type(e).__name__}: {e}"
        # GPU discovery in preference order:
        #   1. NVIDIA via pynvml (utilisation % + memory)
        #   2. OpenVINO device list (identifies the Intel iGPU that
        #      OpenVINO would use for accelerated inference - no live
        #      utilisation numbers, just presence + name)
        #   3. Windows WMIC as a last-resort name-only fallback
        # If none work, GPU stays empty and the UI shows "N/A".
        gpus = []
        try:
            import pynvml
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            for i in range(n):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode()
                gpus.append({
                    "name": name, "util_pct": float(util.gpu),
                    "mem_pct": round(100.0 * mem.used / max(1, mem.total), 1),
                    "mem_used_gb": round(mem.used / (1024 ** 3), 1),
                    "mem_total_gb": round(mem.total / (1024 ** 3), 1),
                    "source": "nvidia",
                })
            try:
                pynvml.nvmlShutdown()
            except Exception as _shut:
                # Cleanup best-effort; a failed shutdown is not fatal.
                if os.environ.get("HW_PROBE_DEBUG"):
                    print(f"hw-probe: nvml shutdown failed: "
                          f"{type(_shut).__name__}: {_shut}")
        except Exception as _nv:
            # Anticipated: no NVIDIA driver / pynvml missing / no GPU.
            # OpenVINO discovery below covers the Intel iGPU case. Set
            # HW_PROBE_DEBUG=1 to see the specific pynvml failure.
            if os.environ.get("HW_PROBE_DEBUG"):
                print(f"hw-probe: pynvml unavailable: "
                      f"{type(_nv).__name__}: {_nv}")
        if not gpus:
            try:
                import openvino as ov
                core = ov.Core()
                for dev in core.available_devices:
                    if dev.startswith("GPU"):
                        try:
                            name = core.get_property(dev, "FULL_DEVICE_NAME")
                        except Exception:
                            name = dev
                        gpus.append({"name": str(name), "source": "openvino"})
            except Exception:
                pass
        if not gpus and os.name == "nt":
            try:
                import subprocess
                r = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "name"],
                    capture_output=True, text=True, timeout=3, check=False)
                for line in (r.stdout or "").splitlines()[1:]:
                    s = line.strip()
                    if s:
                        gpus.append({"name": s, "source": "wmic"})
            except Exception:
                pass
        out["gpus"] = gpus
        self._send_json(200, out)

    def _models_info(self) -> None:
        """GET /api/models/info - static + measured metrics on every
        weight file the pipeline consumes, plus a benchmark of the live
        inference latency on this machine. Feeds the Model Information
        tab. Cached in memory once computed since file sizes and static
        reference metrics don't change between requests.
        """
        try:
            from app.model_metrics import gather_models_info
            self._send_json(200, {"ok": True, **gather_models_info()})
        except Exception as e:
            self._send_json(500, {"ok": False,
                                  "error": f"{type(e).__name__}: {e}"})

    def _events_jsonl(self) -> None:
        """GET /api/events.jsonl?cam=<id>[&limit=<N>] - append-only event
        log served as newline-delimited JSON, newest last (chronological
        order matches the CSV export). Missing sink = 200 empty body."""
        cam = self._q_cam()
        if cam is None:
            return
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        try:
            limit = max(1, min(5000, int((q.get("limit") or ["500"])[0])))
        except ValueError:
            limit = 500
        from app.live_analysis import read_events
        events = read_events(cam, limit=limit)
        body = "".join(json.dumps(e, ensure_ascii=False) + "\n"
                       for e in events).encode("utf-8")
        self._send_bytes(200, "application/x-ndjson", body)

    def _export_csv(self) -> None:
        """GET /api/export.csv?cam=<id>[&limit=<N>] - the same event log
        as /api/events.jsonl serialized as CSV so an operator can drop
        it into Excel / Google Sheets without a JSON round-trip. Column
        order: iso, ts, cam, layer, text, cls, tid, x1, y1, x2, y2. An
        alert with no bbox (e.g. a face count summary) gets empty
        coordinate columns."""
        cam = self._q_cam()
        if cam is None:
            return
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        try:
            limit = max(1, min(50_000, int((q.get("limit") or ["2000"])[0])))
        except ValueError:
            limit = 2000
        from app.live_analysis import read_events
        events = read_events(cam, limit=limit)
        import csv
        import io
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["iso", "ts", "cam", "layer", "text",
                    "cls", "tid", "x1", "y1", "x2", "y2"])
        for e in events:
            box = e.get("box") or {}
            w.writerow([
                e.get("iso", ""),
                e.get("ts", ""),
                e.get("cam", ""),
                e.get("layer", ""),
                (e.get("text") or "").replace("\n", " "),
                box.get("cls", "") or "",
                box.get("tid", "") if box.get("tid") is not None else "",
                box.get("x1", "") if box else "",
                box.get("y1", "") if box else "",
                box.get("x2", "") if box else "",
                box.get("y2", "") if box else "",
            ])
        payload = buf.getvalue().encode("utf-8")
        # Content-Disposition primes the browser to save-as when the
        # operator opens the URL directly. Overridden if the caller
        # asks with fetch() from JS.
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header(
            "Content-Disposition",
            f'attachment; filename="events_{cam}.csv"')
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()
        self.wfile.write(payload)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()   # skip our no-cache re-header dance
        self.wfile.write(body)

    def _send_bytes(self, status: int, content_type: str,
                    data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()
        self.wfile.write(data)



    # -- fix 2: live advanced analysis (app/live_analysis.py) --------------

    def _analysis_start(self) -> None:
        """POST /api/analysis/start?cam=<id>&layer=<layer>

        Starts a live-analysis session on ONE camera (registry id or a
        local-picker slot id), or switches the layer of a running session
        in place - stream, tracker and accumulators survive the switch.
        At most live_analysis.MAX_SESSIONS run concurrently (409 beyond).
        """
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        cam = (q.get("cam") or [""])[0]
        layer = (q.get("layer") or [""])[0]
        if not cam or not layer:
            self._send_json(400, {"error": "need ?cam= and ?layer="})
            return
        model = _MODEL_STATE.get_model()
        if model is None:
            self._send_json(503, {"error": "no detection model loaded "
                                           "(SEARCH_YOLO=off?)"})
            return
        from app.live_analysis import MANAGER, BusyError
        try:
            info = MANAGER.start(cam, layer, model)
            self._send_json(200, {"ok": True, **info})
        except BusyError as e:
            self._send_json(409, {"error": str(e)})
        except ValueError as e:
            self._send_json(404, {"error": str(e)})
        except Exception as e:
            self._send_json(502, {"error": f"{type(e).__name__}: {e}"})

    def _analysis_frame(self) -> None:
        """GET /api/analysis/frame?cam=<id>

        Latest analyzed JPEG of the session (200 image/jpeg with X-Seq /
        X-Layer / X-Note headers), 202 JSON while the first frame is
        still being produced, 410 JSON when the session died with a
        reported reason (so the operator sees WHY, not a bare 404), and
        404 when no session ever ran for this camera. Polling this keeps
        the session's idle clock alive.
        """
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        cam = (q.get("cam") or [""])[0]
        from app.live_analysis import MANAGER
        fr = MANAGER.frame(cam) if cam else None
        if fr is None:
            self._send_json(404, {"error": "no live analysis for this "
                                           "camera"})
            return
        if fr.get("error"):
            # Session crashed / ended - report the reason once so the UI
            # can distinguish a fatal analysis error from "never started".
            self._send_json(410, {"error": fr["error"], "ended": True})
            return
        if not fr["jpeg"]:
            self._send_json(202, {"ok": True, "pending": True,
                                  "note": fr["note"] or "starting..."})
            return
        body = fr["jpeg"]
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Seq", str(fr["seq"]))
        self.send_header("X-Layer", fr["layer"])
        note = (fr["note"] or "").encode("ascii", "replace").decode("ascii")
        if note:
            self.send_header("X-Note", note)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # poller gave up mid-frame - the next poll catches up

    def _analysis_data(self) -> None:
        """GET /api/analysis/data?cam=<id>

        JSON snapshot for the canvas-overlay renderer: boxes+heat+line
        instead of a rendered JPEG. The client draws these on a canvas
        positioned over the live iframe so the video stays smooth while
        the overlay ticks at YOLO pace. Same idle-clock behaviour as
        /api/analysis/frame: polling keeps the session alive.
        """
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        cam = (q.get("cam") or [""])[0]
        from app.live_analysis import MANAGER
        d = MANAGER.data(cam) if cam else None
        if d is None:
            self._send_json(404, {"error": "no live analysis for this "
                                           "camera"})
            return
        if d.get("error"):
            self._send_json(410, {"error": d["error"], "ended": True})
            return
        if not d.get("data"):
            self._send_json(202, {"ok": True, "pending": True,
                                  "note": d.get("note") or "starting..."})
            return
        payload = dict(d["data"])
        payload["seq"] = d["seq"]
        payload["layer"] = d["layer"]
        self._send_json(200, payload)

    def _analysis_stop(self) -> None:
        """POST /api/analysis/stop?cam=<id> - back to plain video."""
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        cam = (q.get("cam") or [""])[0]
        from app.live_analysis import MANAGER
        stopped = MANAGER.stop(cam) if cam else False
        self._send_json(200, {"ok": True, "stopped": stopped})

    def _screen_capture_bbox(self) -> None:
        """POST /api/screen-capture/bbox
        Body: {"x1": int, "y1": int, "x2": int, "y2": int}  (physical pixels)
        Sets the screen-capture region so the fallback grabs the video
        area only, not the whole desktop. Pass an empty body or all-zero
        values to clear back to full primary display."""
        import json
        raw = b""
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b""
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception as e:
            self._send_json(400, {"error": f"bad JSON: {e}"})
            return
        try:
            from app.screen_capture import set_region
        except Exception as e:
            self._send_json(500, {"error": f"screen_capture unavailable: {e}"})
            return
        bbox = None
        if payload and all(k in payload for k in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = (int(payload["x1"]), int(payload["y1"]),
                              int(payload["x2"]), int(payload["y2"]))
            if x2 > x1 and y2 > y1:
                bbox = (x1, y1, x2, y2)
        try:
            set_region(bbox)
        except Exception as e:
            self._send_json(400, {"error": f"invalid bbox: {e}"})
            return
        self._send_json(200, {"ok": True, "bbox": list(bbox) if bbox else None})

    def _analysis_events(self) -> None:
        """GET /api/analysis/events?cam=<id> - the session's detection
        event ring (newest first, thumbs only; full frames stay on the
        server until an explicit save)."""
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        cam = (q.get("cam") or [""])[0]
        from app.live_analysis import MANAGER
        evs = MANAGER.events(cam) if cam else None
        if evs is None:
            self._send_json(404, {"error": "no live analysis for this "
                                           "camera"})
            return
        self._send_json(200, {"events": evs})

    def _analysis_replay(self) -> None:
        """GET /api/analysis/replay?cam=<id>[&ts=<epoch>] - the session's
        last-15s annotated-frame ring (base64 JPEGs @2 fps, decision D3).
        With ts, only frames near that moment."""
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        cam = (q.get("cam") or [""])[0]
        try:
            ts = float((q.get("ts") or ["0"])[0]) or None
        except ValueError:
            ts = None
        from app.live_analysis import MANAGER
        clip = MANAGER.replay(cam, ts) if cam else None
        if clip is None:
            self._send_json(404, {"error": "no live analysis for this "
                                           "camera"})
            return
        self._send_json(200, {"ok": True, **clip})

    def _analysis_event_save(self) -> None:
        """POST /api/analysis/event/save?cam=<id>&id=<event_id> - persist
        one ring event (full frame + manifest row) for later study."""
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        cam = (q.get("cam") or [""])[0]
        eid = (q.get("id") or [""])[0]
        from app.live_analysis import MANAGER
        row = MANAGER.save_event(cam, eid) if cam and eid else None
        if row is None:
            self._send_json(404, {"error": "event not found (ring may "
                                           "have rolled past it)"})
            return
        self._send_json(200, {"ok": True, "saved": row})

    def _analysis_saved(self) -> None:
        """GET /api/analysis/saved - manifest of saved detection events."""
        import json as _json
        from pathlib import Path
        man = (Path(__file__).resolve().parent.parent / "web" / "snapshots"
               / "detections" / "saved.json")
        try:
            items = _json.loads(man.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            items = []
        self._send_json(200, {"items": items})

    def _analysis_plate_crops(self) -> None:
        """GET /api/analysis/plate-crops?cam=X[&tid=Y][&ts=T][&limit=N]
        Return the newest saved plate crops for a camera, optionally
        filtered to a specific track id or to entries whose ts (in the
        filename) is near the requested wall-clock. Payload:
          { "ok": True, "cam": "X", "count": N,
            "items": [{"url": "/plate-crops/<cam>/<file>",
                       "ts": 1787..., "tid": 42, "text": "AB123",
                       "conf_pct": 63}, ...] }
        The Investigation LPR pipeline modal calls this to fill panels
        B/C/D with the REAL per-attempt crops instead of CSS crops of
        the single saved event JPEG (see AUDIT_2026-08-23.md bug #3).
        """
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        cam = (q.get("cam") or [""])[0].strip()
        tid_s = (q.get("tid") or [""])[0].strip()
        ts_s = (q.get("ts") or [""])[0].strip()
        limit = 5
        try:
            limit = max(1, min(20, int((q.get("limit") or ["5"])[0])))
        except ValueError:
            pass
        if not cam:
            self._send_json(400, {"error": "cam required"})
            return
        import re as _re
        _safe = _re.compile(r"[^A-Za-z0-9._-]+")
        safe_cam = _safe.sub("_", cam) or "cam"
        cam_dir = PLATE_CROPS_DIR / safe_cam
        try:
            files = sorted(cam_dir.glob("*.jpg"),
                           key=lambda p: p.stat().st_mtime,
                           reverse=True)
        except OSError:
            files = []
        items = []
        fn_re = _re.compile(r"^(\d+)_(\d+)_(.+)_(\d{2})\.jpg$")
        want_tid = None
        try:
            want_tid = int(tid_s) if tid_s else None
        except ValueError:
            pass
        want_ts_ms = None
        try:
            want_ts_ms = int(float(ts_s) * 1000) if ts_s else None
        except ValueError:
            pass
        for p in files:
            m = fn_re.match(p.name)
            if not m:
                continue
            ts_ms, tid, text, conf_pct = m.groups()
            tid_i = int(tid)
            if want_tid is not None and tid_i != want_tid:
                continue
            if want_ts_ms is not None:
                # Include crops within +/- 10 s of the requested ts. LPR
                # ticks buffer up to 5 sharpest crops per track over a
                # few seconds, so this window comfortably covers one
                # detection's whole burst.
                if abs(int(ts_ms) - want_ts_ms) > 10_000:
                    continue
            items.append({
                "url": f"/plate-crops/{safe_cam}/{p.name}",
                "ts": int(ts_ms) / 1000.0,
                "tid": tid_i,
                "text": text,
                "conf_pct": int(conf_pct),
            })
            if len(items) >= limit:
                break
        self._send_json(200, {"ok": True, "cam": cam,
                              "count": len(items), "items": items})

    def _serve_plate_crop(self) -> None:
        """GET /plate-crops/<cam>/<file>.jpg -> serve from
        src/data/plate_crops/<cam>/<file>. Read-only, jpg-only."""
        from urllib.parse import unquote
        raw = self.path.split("?")[0]
        rel = unquote(raw[len("/plate-crops/"):])
        parts = rel.split("/")
        if (len(parts) != 2 or ".." in parts
                or not parts[0] or not parts[1]
                or not parts[1].endswith(".jpg")):
            self.send_error(400, "bad plate crop path")
            return
        import re as _re
        _safe = _re.compile(r"[^A-Za-z0-9._-]+")
        cam, fname = parts
        if _safe.sub("", cam) != cam or _safe.sub("", fname) != fname:
            self.send_error(400, "bad plate crop path")
            return
        p = PLATE_CROPS_DIR / cam / fname
        try:
            body = p.read_bytes()
        except OSError:
            self.send_error(404, "plate crop not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=60")
        self.end_headers()
        self.wfile.write(body)

    def _analysis_saved_delete(self) -> None:
        """POST /api/analysis/saved-delete?id=<event_id> - remove ONE saved
        event: drop its row from saved.json and unlink its jpg. Operator-
        triggered from the per-tile delete button in the Investigation
        gallery."""
        import json as _json
        from pathlib import Path
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        event_id = (q.get("id") or [""])[0].strip()
        if not event_id:
            self._send_json(400, {"ok": False, "error": "missing ?id="})
            return
        det_dir = (Path(__file__).resolve().parent.parent / "web"
                   / "snapshots" / "detections")
        man = det_dir / "saved.json"
        try:
            items = _json.loads(man.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            items = []
        # 2026-08-23 (C1a): guard the whole read-modify-write of saved.json
        # under the shared lock so a concurrent auto-save (from the plates
        # tick loop) cannot re-insert the row we are about to delete.
        try:
            from app.live_analysis import _SAVED_JSON_LOCK
        except Exception:
            _SAVED_JSON_LOCK = None
        removed_row = None
        kept = []
        for it in items:
            if str(it.get("id")) == event_id:
                removed_row = it
                continue
            kept.append(it)
        removed_files = 0
        if removed_row:
            img_rel = removed_row.get("image") or ""
            # image field is "snapshots/detections/<cam>_<id>.jpg"; anchor
            # at web/ and unlink. Guard against traversal (any '..').
            if img_rel and ".." not in img_rel:
                web_root = det_dir.parent.parent
                p = web_root / img_rel.replace("/", "\\") \
                    if "\\" in str(web_root) else web_root / img_rel
                # Cross-platform: also try forward-slash join.
                try_paths = [web_root / img_rel, det_dir /
                             img_rel.rsplit("/", 1)[-1]]
                for candidate in try_paths:
                    try:
                        if candidate.is_file():
                            candidate.unlink()
                            removed_files += 1
                    except OSError:
                        pass
        try:
            if _SAVED_JSON_LOCK is not None:
                with _SAVED_JSON_LOCK:
                    man.write_text(_json.dumps(kept), encoding="utf-8")
            else:
                man.write_text(_json.dumps(kept), encoding="utf-8")
        except OSError as e:
            self._send_json(500, {"ok": False,
                                  "error": f"{type(e).__name__}: {e}"})
            return
        # Clear the per-tid dedup on the running plates session so a
        # re-passing vehicle with the same tid can re-emit. 2026-08-21:
        # _plate_emitted is a dict keyed by tid, not a set of text keys,
        # so we cannot target one entry from the row alone - operator
        # intent (delete one gallery card, allow it to re-appear next
        # time the car goes by) is best served by clearing the whole
        # dedup dict on the session. A busy street then re-fills naturally
        # over the next few minutes.
        try:
            from app.live_analysis import MANAGER as _MGR
            for s in _MGR._sessions.values():
                if hasattr(s, "_plate_emitted"):
                    try:
                        s._plate_emitted.clear()
                    except Exception:
                        pass
        except Exception:
            pass
        self._send_json(200, {"ok": True, "removed": bool(removed_row),
                              "files_unlinked": removed_files})

    def _analysis_saved_clear(self) -> None:
        """POST /api/analysis/saved-clear - delete every saved detection
        crop from disk and truncate the manifest. Operator-triggered from
        the Investigation tab's Clear all button."""
        import json as _json
        from pathlib import Path
        det_dir = (Path(__file__).resolve().parent.parent / "web"
                   / "snapshots" / "detections")
        removed = 0
        try:
            if det_dir.is_dir():
                for p in det_dir.iterdir():
                    if p.is_file() and p.name != "saved.json" \
                            and p.name != ".gitkeep":
                        try:
                            p.unlink()
                            removed += 1
                        except OSError:
                            pass
            man = det_dir / "saved.json"
            man.parent.mkdir(parents=True, exist_ok=True)
            # 2026-08-23 (C1a): same lock as save_event so a concurrent
            # auto-save cannot slip a row in between our read of an
            # empty tree and our write of an empty manifest.
            try:
                from app.live_analysis import _SAVED_JSON_LOCK
            except Exception:
                _SAVED_JSON_LOCK = None
            if _SAVED_JSON_LOCK is not None:
                with _SAVED_JSON_LOCK:
                    man.write_text(_json.dumps([]), encoding="utf-8")
            else:
                man.write_text(_json.dumps([]), encoding="utf-8")
        except Exception as e:
            self._send_json(500, {"ok": False,
                                  "error": f"{type(e).__name__}: {e}"})
            return
        # Wipe in-memory dedup so already-seen plates can re-emit and the
        # event strip no longer offers Save on rows whose disk copies just
        # got deleted. Best-effort: a missing MANAGER (module not imported
        # yet) is a no-op, same as before.
        try:
            from app.live_analysis import MANAGER as _MGR
            _MGR.clear_saved_state()
        except Exception:
            pass
        self._send_json(200, {"ok": True, "removed": removed})


    # ---- human-in-the-loop review endpoints ------------------------------
    # Backing the "Review detections" panel in index.html. The user is shown
    # one un-reviewed crop with its current label and picks correct /
    # wrong-label / not-an-object. Answers persist to data/reviews.json via
    # ReviewStore. Sampler: REVIEW_SAMPLER=badge|naive (default naive),
    # overridable per request with ?strategy= (plan WS2). The response
    # carries "sampler" so the UI can badge it, and the server remembers
    # what it served so the submit row records sampler +
    # uncertainty_at_selection (spec 9.1) without any client change.







    # ---- Frame-based review endpoints ----------------------------------
    # The new canvas UX: one frame carries multiple detections, the user
    # gives a verdict per BOX, plus optional "missed" boxes drawn on the
    # canvas. That last piece is what finally gives us FN → recall → F1.

    @staticmethod
    def _local_pick_ids() -> set:
        """The MAIN edition's camera universe: the picked slots' ids plus
        their catalog ids (frames get saved under either)."""
        import json as _json
        try:
            grid = _json.loads((WEB_DIR / "local_grid.json").read_text(
                encoding="utf-8"))
        except (OSError, ValueError):
            return set()
        out = set()
        for s in grid.get("slots") or []:
            if s.get("slot_id"):
                out.add(s["slot_id"])
            if s.get("cam_id"):
                out.add(s["cam_id"])
        return out

    @staticmethod
    def _frame_cam(f: dict) -> str:
        cam = f.get("cam_id") or f.get("cam") or ""
        if not cam:
            parts = str(f.get("frame_path") or f.get("path") or "").split("/")
            if len(parts) >= 2:
                cam = parts[1]
        return cam








    def _model_metrics(self) -> None:
        """Scoreboard endpoint driving the header line. Cheap - it just
        walks the in-memory review store and does arithmetic. Safe to poll
        every 10s from the browser. Since Category B (Review system) was
        removed, this returns an empty scoreboard - the UI treats missing
        counts as zero."""
        self._send_json(200, {
            "reviews": 0, "correct": 0, "wrong": 0, "unsure": 0,
            "header_line": "reviews: 0",
            "curve": [],
            "note": "review system removed with Category B",
        })

    def do_HEAD(self) -> None:
        # Browsers use GET (not HEAD) for <video>/HLS, so this matters only to
        # dev tools like `curl -I`. Route it through the same proxy so the dev
        # check gets a real status code instead of a 404 from the static handler.
        if self.path.startswith("/tvkur/"):
            self._proxy_tvkur()
            return
        super().do_HEAD()

    # ---- Repo #2: upload a local video file for analysis ------------------

    def _upload_video(self) -> None:
        """POST /api/upload-video: multipart/form-data with field 'video'.
        Saves to src/data/uploads/<uuid>.<ext>. Returns cam dict for CAMERAS."""
        try:
            ctype = self.headers.get("Content-Type", "")
            if not ctype.startswith("multipart/form-data"):
                self._send_json(400, {"error": "expect multipart/form-data"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > _MAX_UPLOAD_BYTES:
                self._send_json(413, {"error": f"file too large or empty (max {_MAX_UPLOAD_BYTES // (1024*1024)} MB)"})
                return
            # Parse multipart using email/cgi (stdlib)
            import cgi
            env = {"REQUEST_METHOD": "POST", "CONTENT_TYPE": ctype, "CONTENT_LENGTH": str(length)}
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ=env, keep_blank_values=True)
            if "video" not in form:
                self._send_json(400, {"error": "missing 'video' field"})
                return
            item = form["video"]
            filename = getattr(item, "filename", "") or "upload"
            ext = _Path(filename).suffix.lower() or ".mp4"
            if ext not in _ALLOWED_VIDEO_EXT:
                self._send_json(415, {"error": f"unsupported extension {ext}"})
                return
            cam_id = f"upload_{_uuid.uuid4().hex[:8]}"
            dst = _UPLOAD_DIR / f"{cam_id}{ext}"
            with open(dst, "wb") as w:
                w.write(item.file.read())
            cam = {"id": cam_id, "name": filename, "kind": "local_file", "path": str(dst), "country": "local"}
            self._send_json(200, {"ok": True, "cam": cam})
        except Exception as e:
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _catalog(self) -> None:
        """GET /api/catalog: list catalog cameras (active URLs only) for the picker.

        Includes `url` so the frontend can render the source directly - a
        `youtube` kind camera needs its watch URL to embed the iframe player
        without a backend yt-dlp resolve.
        """
        try:
            from app.cameras import active_cameras
            cams = active_cameras()
            items = []
            for cid, cam in cams.items():
                items.append({
                    "id":      cid,
                    "name":    cam.get("name") or cid,
                    "kind":    cam.get("kind") or "hls",
                    "url":     cam.get("url") or "",
                    "hls":     cam.get("hls") or "",
                    "area":    cam.get("city") or cam.get("area") or "",
                    "country": cam.get("country") or "",
                })
            self._send_json(200, {"ok": True, "cameras": items})
        except Exception as e:
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _uploaded_videos(self) -> None:
        """GET /api/uploaded-videos: list files in src/data/uploads/."""
        try:
            items = []
            for p in _UPLOAD_DIR.iterdir():
                if p.is_file() and p.suffix.lower() in _ALLOWED_VIDEO_EXT:
                    st = p.stat()
                    items.append({"cam_id": p.stem, "name": p.name, "path": str(p),
                                  "size": st.st_size, "mtime": int(st.st_mtime)})
            items.sort(key=lambda x: -x["mtime"])
            self._send_json(200, {"ok": True, "items": items})
        except Exception as e:
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _local_file(self) -> None:
        """GET /api/local-file?cam=<upload_hex>: serve an uploaded video.

        The upload endpoint writes files under `src/data/uploads/` which
        is NOT inside the dashboard's static `web/` root, so a plain
        static handler cannot reach them. This route resolves the
        `<cam_id>` back to `src/data/uploads/<cam_id>.<ext>` (any
        allowed extension) and streams the bytes with Range support so
        the browser <video> element can seek.
        """
        try:
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            cam_id = (q.get("cam") or [""])[0].strip()
            if not cam_id or not cam_id.startswith("upload_") \
                    or "/" in cam_id or "\\" in cam_id or ".." in cam_id:
                self.send_error(400, "invalid ?cam=")
                return
            path = None
            for ext in _ALLOWED_VIDEO_EXT:
                candidate = _UPLOAD_DIR / f"{cam_id}{ext}"
                if candidate.is_file():
                    path = candidate
                    break
            if path is None:
                self.send_error(404, f"no upload {cam_id!r}")
                return
            size = path.stat().st_size
            # Video MIME - honour known extensions; fall back to a
            # container-neutral default so unknown containers still play
            # in browsers that sniff.
            mime = {".mp4":  "video/mp4",
                    ".mkv":  "video/x-matroska",
                    ".webm": "video/webm",
                    ".mov":  "video/quicktime",
                    ".avi":  "video/x-msvideo"}.get(path.suffix.lower(),
                                                    "application/octet-stream")
            rng = self.headers.get("Range") or ""
            start, end = 0, size - 1
            partial = False
            if rng.startswith("bytes="):
                try:
                    s, e = rng[6:].split("-", 1)
                    if s:
                        start = int(s)
                    if e:
                        end = int(e)
                    if start < 0 or start >= size or end >= size or end < start:
                        # RFC 7233 recommends 416 on unsatisfiable Range.
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.end_headers()
                        return
                    partial = True
                except ValueError:
                    partial = False
            length = end - start + 1
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", mime)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if partial:
                self.send_header("Content-Range",
                                 f"bytes {start}-{end}/{size}")
            # Uploaded files are private to this session - discourage
            # caching so the operator seeing a stale copy after re-upload
            # under the same id is not a possibility.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                remaining = length
                chunk = 64 * 1024
                while remaining > 0:
                    buf = f.read(min(chunk, remaining))
                    if not buf:
                        break
                    try:
                        self.wfile.write(buf)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    remaining -= len(buf)
        except Exception as e:  # noqa: BLE001
            try:
                self.send_error(500,
                                f"{type(e).__name__}: {e}")
            except Exception:
                pass

    def _delete_uploaded_video(self) -> None:
        """DELETE /api/uploaded-video?cam_id=X: remove one upload (whitelisted to uploads dir)."""
        try:
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            cam_id = (q.get("cam_id") or [""])[0]
            if not cam_id or not cam_id.startswith("upload_"):
                self._send_json(400, {"error": "invalid cam_id"})
                return
            for p in _UPLOAD_DIR.iterdir():
                if p.stem == cam_id:
                    p.unlink()
                    self._send_json(200, {"ok": True, "removed": p.name})
                    return
            self._send_json(404, {"error": "not found"})
        except Exception as e:
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _proxy_tvkur(self) -> None:
        # /tvkur/<stream_id>/<path...> -> content.tvkur.com/l/<stream_id>/<path...>
        # Strip any ?query so we mirror exactly what the browser asked for.
        path = self.path[len("/tvkur/"):]
        upstream = "https://content.tvkur.com/l/" + path
        try:
            req = urllib.request.Request(upstream, headers=_TVKUR_HEADERS)
            with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as r:
                self.send_response(r.status)
                ct = r.headers.get("Content-Type")
                if ct:
                    self.send_header("Content-Type", ct)
                # CORS open + short cache so hls.js can refresh the chunklist.
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                # stream the body in chunks - .ts segments are several MB
                while True:
                    chunk = r.read(64 * 1024)
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return  # the browser closed the segment fetch - fine
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(f"tvkur proxy error: {type(e).__name__}: {e}".encode())


def make_handler_factory(directory: Path | None = None):
    """Return a handler class bound to a serving directory (defaults to web/)."""
    d = str(directory or WEB_DIR)
    return lambda *a, **k: DashboardHandler(*a, directory=d, **k)


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return True
        except OSError:
            return False


def _warm_model_async() -> None:
    """Load the detector in a background daemon thread so the first
    Analyze click does not pay the 5-15s model load."""
    def _run() -> None:
        try:
            _MODEL_STATE.get_model()
        except Exception as e:
            print(f"  ! model warmup failed: {type(e).__name__}: {e}")
    threading.Thread(target=_run, daemon=True,
                     name="model-warmup").start()


def bind(port: int, directory: Path | None = None) -> http.server.ThreadingHTTPServer:
    """Threaded server so simultaneous video segment requests don't queue.

    Also fires an async warmup that loads YOLO in the background and
    bootstraps the review pool from fixture frames, so the first user to
    open the dashboard finds material to review already sitting there.
    """
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    http.server.ThreadingHTTPServer.daemon_threads = True
    server = http.server.ThreadingHTTPServer(("", port), make_handler_factory(directory))
    _warm_model_async()
    # Auto-start local ModelViewProducer + ReviewFrameProducer if a picker
    # has already written web/local_grid.json. The producers run inside this
    # Python process, share the same YOLO model as the warmup
    # + live-analysis sessions, and write annotated JPEGs + counts JSON that
    # the frontend's LOCAL_MODE poll reads from
    # /snapshots/model_view/local_*.json.
    def _start_local_producers_when_ready():
        _lg = WEB_DIR / "local_grid.json"
        if not _lg.exists():
            return
        try:
            import json as _json
            grid = _json.loads(_lg.read_text(encoding="utf-8"))
            slots = grid.get("slots") or []
            if not slots:
                return
            # ONE model for the whole process: producers share the
            # warmed detector (yolo26x / its OpenVINO engine) instead
            # of loading a second yolo26m engine. On the 8GB laptop the
            # two-engine setup measurably exhausted RAM (88% used, 1GB
            # free) and pushed inference into pagefile thrash - ticks
            # ballooned to 15s. Wait for the warmup to finish loading,
            # bounded so a broken warmup doesn't hang the hook forever.
            import time as _t
            _model = None
            for _ in range(300):
                _model = _MODEL_STATE.model
                if _model is not None:
                    break
                _t.sleep(1)
            if _model is None:
                print("  ! local_producers not started: model not loaded "
                      "within 5 min")
                return
            from app.local_producers import start_all as _start_all
            _start_all(slots, _model,
                       model_view_interval_s=8, review_interval_s=60)
            print(f"local_producers running: {len(slots)} slots -> "
                  f"web/snapshots/model_view/local_*.jpg (~8s per round)")
        except Exception as e:
            print(f"  ! local_producers not started: {type(e).__name__}: {e}")

    import threading as _threading
    _threading.Thread(target=_start_local_producers_when_ready,
                      daemon=True,
                      name="local-producers-autostart").start()
    return server
