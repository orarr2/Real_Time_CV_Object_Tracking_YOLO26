# Project guide - Real-Time CV Object Tracking (YOLO26)

> **Changes in the 2026-08-23 session** (full decision log in
> `docs/AUDIT_2026-08-23.md`): ~1,900 lines of dead code removed (dead server
> endpoints, burst-analytics chain, blur path, live_samples /
> anomaly_crops / calibrate_conf); plate detector upgraded to
> yolov11-S + OCR to fast-plate-ocr `cct_s_v2` + pose to `yolov8s-pose`;
> body-anomaly gates hardened (ratio 8, 10-sample/10-s bbox window,
> 2-tick debounce, 60-px both-fast fighting rule); fall detection
> folded into the Body layer; per-country plate grammar; hot-trail decay +
> 15-s replay ring; PDT clock-sync probe implemented; per-layer drawers
> split into `app/layers/draw.py`. Thresholds reference:
> `src/docs/DECISION_THRESHOLDS_HE.md`.


Deep-dive companion to `README.md` at the repo root. This file covers
the actual runtime pipeline, every tunable configuration knob, the
mechanics of each of the ten analysis layers, and the extension points
for adding sources, models, or layers of your own.

Target reader: someone who has already run `python serve.py` at least
once, watched the live tile move, and now wants to understand where the
numbers come from - or how to plug in a new module.

## Table of contents

- [1. The tick pipeline](#1-the-tick-pipeline)
  - [1.1 Source resolution](#11-source-resolution)
  - [1.2 Frame grab](#12-frame-grab)
  - [1.3 YOLO26 inference](#13-yolo26-inference)
  - [1.4 BurstTracker](#14-bursttracker)
  - [1.5 Layer render](#15-layer-render)
- [2. Analysis layer mechanics](#2-analysis-layer-mechanics)
- [3. Confidence thresholds and display gates](#3-confidence-thresholds-and-display-gates)
- [4. Frontend architecture](#4-frontend-architecture)
- [5. Backend APIs](#5-backend-apis)
- [6. Adding a source](#6-adding-a-source)
- [7. Adding a layer](#7-adding-a-layer)
- [8. Swapping the detector model](#8-swapping-the-detector-model)
- [9. Investigation + saved detections](#9-investigation--saved-detections)
- [10. Performance envelope](#10-performance-envelope)

## 1. The tick pipeline

Every processing cycle of the live-analysis loop walks the same five
stages. Each stage is a small, testable Python function under
`src/app/`.

### 1.1 Source resolution

- **Input:** a `cam_id` from the picker.
- **Output:** a playable stream URL or local file path, with a
  per-camera cache entry.

`resolve_stream(cam)` in `detect_core.py` dispatches on the catalog
entry's `kind` field:

| `kind`         | Resolution                                                                                                              |
| -------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `hls`          | The URL is used directly.                                                                                               |
| `skyline`      | Fetch the skylinewebcams page HTML, extract the signed `.m3u8` from the embedded JavaScript, cache until token expires. |
| `webcamera24`  | Fetch the webcamera24 page HTML, extract the embedded HLS URL, cache the same way.                                      |
| `youtube`      | Call `yt-dlp` with the ANDROID player client to get the current live HLS manifest. May require exported cookies. When the bot-check blocks even cookies and `SCREEN_CAPTURE_FALLBACK=1` is set, the resolver returns the `screen://primary` sentinel (see [1.2](#12-frame-grab)). |
| `local_file`   | Return the path unchanged.                                                                                              |

Resolutions land in `data/resolve_cache.json`. A cache hit costs one
disk read; a miss costs one HTTP GET plus (for YouTube) a yt-dlp
invocation.

### 1.2 Frame grab

- **Input:** the resolved URL or path, plus a wall-clock deadline.
- **Output:** a single decoded BGR ndarray, or `None` on failure.

`grab_frame(url, timeout)` opens an OpenCV `VideoCapture`, seeks to
the newest packet (segment tail for HLS, `frame_count - 1` for a
file), decodes once, and releases. The call is hard-bounded by the
timeout so one stalled camera cannot freeze the tick loop.

Failures return a compact `GrabError` record:

- `kind` - one of `timeout`, `open_failed`, `decode_failed`,
  `empty_frame`.
- `url` - the resolved source that failed.
- `elapsed_ms` - milliseconds spent before giving up.

`src/tests/test_grab_error_reporting.py` locks the shape.

**Screen-capture fallback.** When the resolved URL is `screen://primary`
(YouTube bot-check dead-end, see [1.1](#11-source-resolution)),
`grab_frame` short-circuits to `screen_capture.capture()` in
`src/app/screen_capture.py`. That module uses `mss` (DXGI Desktop
Duplication on Windows, XShm on Linux, Quartz on macOS) with
`PIL.ImageGrab` and then a cached last-good frame as fallbacks. The
region defaults to the full primary display but is normally cropped to
the browser's iframe rectangle - the frontend POSTs the physical-pixel
bbox to `/api/screen-capture/bbox` on every analysis start
(`iframe.getBoundingClientRect() * devicePixelRatio + screenY +
chromeH`). Requires `SCREEN_CAPTURE_FALLBACK=1` in the environment.

### 1.3 YOLO26 inference

- **Input:** one frame plus a per-class confidence table.
- **Output:** a list of box dicts
  (`{cls, conf, x1, y1, x2, y2, cx, cy, w, h}`) already filtered to
  the classes we care about.

The model loads once at process start. When `yolo26m_openvino_model/`
sits next to `yolo26m.pt`, the engine picks the OpenVINO IR; on Intel
CPUs this is measurably faster (~1.8x on UHD 620) than the raw
PyTorch weights. Without the IR, the engine transparently falls back
to `.pt`.

Per-class confidence thresholds live at the module level of
`detect_core.py` as `DEFAULT_PER_CLASS_CONF`. Defaults are conservative
(0.22-0.35 depending on class) so the tracker sees a clean upper band
to start tracks from. See [3. Confidence thresholds](#3-confidence-thresholds-and-display-gates)
for the full picture.

### 1.4 BurstTracker

- **Input:** the current frame's boxes plus the tracker state from
  the previous frame.
- **Output:** the same boxes, each stamped with a stable `track_id`
  that persists across frames of a burst.

`tracker.py` implements a Python-only motion-based tracker inspired
by ByteTrack:

1. High-confidence detections (`conf >= 0.45`) claim tracks via
   constant-velocity prediction plus a matching radius of 30% of the
   frame diagonal.
2. Low-confidence detections may only **extend** existing tracks
   (radius 15% of the diagonal); they never start new tracks.

Tracks coast on their predicted path for `TRACK_MAX_MISSES = 2`
frames of no match before retiring. Velocity is smoothed via an EMA
with weight `TRACK_VEL_SMOOTH = 0.6`.

### 1.5 Layer render

Only one layer runs at a time (operator constraint). The active layer
consumes the boxes and produces its own overlay data - a heat grid
cell contribution, a per-track trail point, a face rectangle, a plate
crop, etc. `live_analysis.py::_publish_data` collects the overlay
data into a JSON blob and the raw frame into a JPEG buffer; the
frontend polls both.

## 2. Analysis layer mechanics

Each layer is one file under `src/app/`. `live_analysis.py`
orchestrates render order (heat first so boxes draw on top; faces
last so they never get covered by trails). The final overlay data
plus a compact JSON telemetry blob (per-class counts, alerts, plate
reads) is what the dashboard tab sees over its polling loop.

| Layer      | File                        | Notes                                                                                          |
| ---------- | --------------------------- | ---------------------------------------------------------------------------------------------- |
| `paths`    | `live_analysis.py` (inline) | Per-track trail history bounded by `TRAIL_MAX_PTS = 40` points. Speed tier from body-lengths/s. |
| `pose`     | `pose.py`                   | Top-down COCO-17 keypoints on person crops with box height >= `KPS_MIN_BOX_H = 96 px`.        |
| `gestures` | `gestures.py`               | Temporal window over recent keypoints; emits `hand_raised`, `both_hands_up`, `wave`.          |
| `body`     | `live_analysis.py` (inline) | Kinematic flags + gesture flags per person track.                                             |
| `faces`    | `faces.py`                  | YuNet CV DNN. `FACE_SCORE = 0.60`, box side >= 24 px. Empty result on far-field cameras is by design. |
| `line`     | `cameras.py`, live_analysis | Counting line drawn by the operator. In / out determined by sign flip across A -> B.          |
| `loiter`   | `live_analysis.py` (zones)  | Polygon dwell detection with per-track cooldown (folds into the parking layer's zone engine). |
| `parking`  | `live_analysis.py` (inline) | Occupancy flip on operator-drawn spots plus a 12 s per-spot re-probe.                         |
| `plates`   | `plates.py`                 | Two-stage LPR (yolov11-S + fast-plate-ocr cct_s_v2) with per-track cache, multi-frame integration and a per-country grammar gate. |
| `heat`     | `heatmap.py`                | 48x27 grid, foot-point accumulation, 180 s half-life decay.                                   |

## 3. Confidence thresholds and display gates

Detection thresholds are layered:

1. **Detector floor** (`LIVE_CONF_FLOOR = 0.12`) - any detection
   below this is dropped by the model outright.
2. **Per-class gate** (`DEFAULT_PER_CLASS_CONF` in `detect_core.py`,
   scaled by `LIVE_GATE_SCALE = 0.7` in the live path). Prevents
   COCO categories that make no sense on a street cam ("train" on a
   fence) from surviving.
3. **Tracker gate** (`TRACK_HIGH_CONF = 0.45` in `tracker.py`).
   Above this a detection can start a new track; below it, only
   extends an existing one.
4. **Display gate** (`DISPLAY_MIN_CONF = 0.32`,
   `DISPLAY_MIN_HITS = 1` in `live_analysis.py`). The overlay
   publishes tracker-confirmed objects only.

A calibration workflow in the notebook (Section 10) lets you sample
frames from your camera, label the ground truth, and derive
per-camera confidence overrides that get merged into `CAMERAS` at
import via `_merge_per_camera_conf()`.

## 4. Frontend architecture

`src/web/index.html` is a single static page loaded by the HTTP server.
`app.js` handles the picker, tile builder, analysis loop and canvas
overlay:

- **Picker** - fetches `/api/catalog` and `/api/uploaded-videos`,
  populates the dropdown. Uploading an MP4/MKV posts to
  `/api/upload-video` and reloads the picker.
- **Video element** - three modes chosen by `cam.kind`:
  1. `local_file` -> `<video src>` served by `/api/local-file`.
  2. `youtube` -> `<iframe>` embedding the YouTube player.
  3. `hls` (or anything with `cam.hls`) -> `<video data-hls>` played
     via `hls.js`.
- **Analysis loop** (`beginTileAnalysis`) - creates a canvas overlay
  and polls `/api/analysis/data?cam=<id>` every
  `ANALYSIS_POLL_MS = 500 ms`. Between backend ticks it interpolates
  box positions via linear-blend + velocity extrapolation. When the
  video is playing, the underlying `<video>` / iframe is visible and
  the canvas draws boxes on top. When the video stalls, a fallback
  JPEG (`<img data-analysis-bg>`) drawn by the backend takes over.

Client-side interpolation caps at
`EXTRAP_MAX_S = 3 s` and `EXTRAP_MAX_DIAG = 0.5` (a box cannot slide
more than half its own diagonal per second). Past that window, the
box fades out until the next backend tick provides a fresh position.

## 5. Backend APIs

`src/app/dashboard_server.py` serves the static `web/` directory plus
the following JSON endpoints:

| Endpoint                                     | Purpose                                                                          |
| -------------------------------------------- | -------------------------------------------------------------------------------- |
| `GET  /api/catalog`                          | Camera catalog (`active_cameras()`).                                             |
| `GET  /api/uploaded-videos`                  | Uploaded MP4/MKV list under `data/uploads/`.                                     |
| `POST /api/upload-video`                     | Multipart upload; new file appears as a `local_file` camera.                     |
| `GET  /api/ping`                             | Capability probe returning `{ok: true, private: true}` for the private backend.  |
| `GET  /api/analysis/data?cam=<id>`           | The current tick's overlay JSON (boxes, trails, heat, alerts).                   |
| `GET  /api/analysis/frame?cam=<id>`          | The current tick's rendered JPEG (fallback for stalled video).                   |
| `GET  /api/analysis/events?cam=<id>`         | Rolling ring of alert events for the events strip.                               |
| `POST /api/analysis/start?cam=<id>&layer=X`  | Start (or switch) a live session on camera `id` with layer `X`.                  |
| `POST /api/analysis/stop?cam=<id>`           | Stop the running session.                                                        |
| `POST /api/analysis/event/save?cam=<id>&id=` | Persist one event's full frame to `web/snapshots/detections/`.                   |
| `GET  /api/analysis/saved`                   | Manifest of saved detections (Investigation gallery).                            |
| `GET  /api/lines?cam=<id>`                   | Current counting-line config for the Line layer.                                 |
| `POST /api/lines?cam=<id>`                   | Save a counting line drawn on the snapshot.                                      |
| `POST /api/lines/clear?cam=<id>`             | Remove a per-camera line override.                                               |
| `GET  /api/zones?cam=<id>`                   | Current loiter / parking polygon config.                                         |
| `POST /api/zones?cam=<id>`                   | Save polygons drawn on the snapshot.                                             |
| `POST /api/zones/clear?cam=<id>`             | Remove all polygons for a camera.                                                |
| `GET  /api/crossings?cam=<id>`               | Rolling line-crossing event log.                                                 |
| `GET  /api/heatmap?cam=<id>`                 | Persisted heatmap overlay (rendered JPEG or JSON).                               |

## 6. Adding a source

To add a camera whose `kind` already exists, drop an entry into
`CAMERAS` in `src/app/cameras.py` with the right `url` and reload
the server. `/api/catalog` picks it up automatically.

To support a **new** kind:

1. Add a branch to `resolve_stream(cam)` in `detect_core.py` that
   returns a playable URL or path for that kind.
2. Add rendering in `app.js::buildVideoInto` if the new kind needs a
   different player element (iframe, `<img>`, WebRTC).
3. Optional: add a scraper helper in `detect_core.py` (see
   `resolve_skyline`, `resolve_webcamera24` for the shape).

## 7. Adding a layer

1. Create `src/app/mylayer.py` exporting an `apply(frame, boxes, state)`
   function that mutates `state` and returns overlay data.
2. Register the layer name in `LIVE_LAYERS` in `live_analysis.py`.
3. Wire the render call inside `_publish_data` alongside the other
   layers.
4. Add a radio option in `web/app.js` `ANALYSIS_LAYER_DEFS`.
5. Add the canvas render for the new overlay data in
   `_drawAnalysisOverlay`.

## 8. Swapping the detector model

`MODEL_WEIGHTS` in `real_time_cv.ipynb` and `LIVE_MODEL_WEIGHTS`
(if set) in `detect_core.py` control which weights load. Swap
options in increasing order of cost:

- **yolo26x** - drop-in, +4.4 mAP over yolo26m, ~2.4x per-tick
  latency on CPU. Recommended once you have discrete GPU.
- **rtdetr-x** - transformer detector, stronger on small / distant
  objects (license plates from far cameras) but ~2x latency compared
  to yolo26m and a completely different weight file.
- **rf-detr-large** - >60 mAP but practically requires a GPU. `pip
  install rfdetr` plus a small wrapper in `detect_core.py`.

Always export the OpenVINO IR after downloading:

```python
from ultralytics import YOLO
YOLO('yolo26x.pt').export(format='openvino', imgsz=640)
```

## 9. Investigation + saved detections

The Investigation tab renders a persistent grid from `/api/analysis/saved`.
Each entry references a saved event JPEG under
`web/snapshots/detections/`. Saving happens two ways:

- **Manual** - press "save" on any chip in the events strip during a
  live session. The event's `_full` JPEG gets written and a manifest
  row appended.
- **Automatic** - the `LiveSession._detect_events` state machine
  emits an event to the ring on each layer-relevant state change
  (new plate read, gesture flip, loiter alert, extreme load).

The manifest is `web/snapshots/detections/saved.json`, capped at 500
newest entries.

## 10. Performance envelope

Measured on an Intel i5-8250U (4 cores, 8 threads, no discrete GPU)
with the OpenVINO IR loaded:

| Workload                              | Tick / latency          |
| ------------------------------------- | ----------------------- |
| yolo26m + Paths & speeds              | 0.8 - 1.5 s per tick    |
| yolo26m + Pose                        | 1.5 - 3 s per tick      |
| yolo26m + Plates (single vehicle)     | 2 - 4 s per tick        |
| Two concurrent live sessions          | Not supported (single-session cap - `MAX_SESSIONS = 1`) |

With a discrete GPU, ticks typically drop below 200 ms and the
frontend interpolation ceiling (`EXTRAP_MAX_S = 3 s`) becomes
irrelevant - boxes always land inside the extrapolation window.
