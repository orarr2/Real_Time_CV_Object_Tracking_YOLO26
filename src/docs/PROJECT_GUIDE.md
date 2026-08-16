# Project guide - Real-Time CV Object Tracking (YOLO26)

This is the deep-dive companion to the top-level `README.md`. It covers
the full runtime pipeline, every configuration knob, the mechanics of
each analysis layer, and the extension points for adding your own
sources, models and layers.

Target reader: someone who has already run `python serve.py` at least
once, seen the live tile move, and now wants to know why the numbers
look the way they do or how to add a new module.

---

## 1. Full runtime pipeline

Every "tick" of the live dashboard walks the same five stages. Each
stage is a small, testable Python function in `src/app/`.

### 1.1 Source resolve

Input: a `cam_id` string chosen in the dashboard.
Output: a playable stream URL or local path plus a per-camera cache
entry.

The resolver in `detect_core.py` dispatches on the `kind` field of the
catalog row:

- `hls` - no work, use the URL directly.
- `skyline` - fetch the `skylinewebcams.com` page HTML, extract the
  signed `.m3u8` from the embedded JavaScript, cache the result for the
  duration of that page's token (~1 hour).
- `webcamera24` - fetch the `webcamera24.com` page HTML, extract the
  HLS from the embedded player config, cache similarly.
- `local_file` - the path is already absolute; nothing to resolve.

Resolves are memoized in `data/resolve_cache.json`. A cache hit costs a
disk read; a cache miss costs one HTTP GET.

### 1.2 Frame grab

Input: the resolved URL/path plus a target-time budget.
Output: a single decoded BGR ndarray, or `None` on failure.

`grab_frame(url, timeout)` opens an OpenCV `VideoCapture`, seeks to the
newest available packet (for HLS: the tail of the segment list; for
files: `CAP_PROP_POS_FRAMES = frame_count - 1`), decodes once and
releases. The whole call has a hard timeout so a stalled camera cannot
freeze the tick loop.

Failure modes are reported with a compact `GrabError` record - kind
(`timeout`, `open_failed`, `decode_failed`, `empty_frame`), the URL,
and elapsed ms. `src/tests/test_grab_error_reporting.py` locks the
shape.

### 1.3 YOLO26 inference

Input: one frame plus a per-class confidence table.
Output: a list of box dicts (`{cls, conf, x1, y1, x2, y2, cx, cy, w,
h}`) already filtered to the classes-of-interest.

The model is loaded once at process start. If the OpenVINO IR
(`yolo26m_openvino_model/`) is present next to `yolo26m.pt`, the engine
prefers it for CPU inference - roughly 1.8x faster than raw PyTorch on
an Intel UHD 620 class chip. If the IR is missing the engine falls
back to `.pt` and prints a one-line notice.

Per-class gates come from `labels.py`. The defaults are conservative
(0.22-0.35) so the tracker gets a clean high-band to bootstrap from.

### 1.4 BurstTracker

Input: a frame's boxes plus the previous frame's tracks.
Output: the same boxes, each stamped with a `track_id` that is stable
within the observation window.

`src/app/tracker.py` is a pure-Python, position-and-motion tracker
inspired by BYTE. It runs a two-stage association:

1. High-confidence detections (conf >= 0.45) claim tracks using
   constant-velocity prediction + a matching radius of 30% of the
   frame diagonal.
2. Leftover low-confidence detections may only EXTEND existing tracks
   (radius 15%), never START one.

Tracks coast for `TRACK_MAX_MISSES = 2` unmatched frames before
retiring, so a bus passing in front of a pedestrian does not spawn a
new id.

The tracker does NOT attempt cross-window identity. That is `reid.py`'s
job (OSNet embedding + HSV fallback), and only fires when the operator
explicitly asks for "have I seen this person before".

### 1.5 Layer render

Input: boxes-with-track-ids plus the layer-enabled bitmask.
Output: a rendered overlay drawn on the same frame.

Each of the 10 layers is one Python module. `live_analysis.py`
orchestrates the render order (heat first so boxes draw on top; faces
last so they never get covered by trails). The final composite frame
plus a compact JSON telemetry blob (per-class counts, alerts, plate
reads) is what the dashboard tab sees over its WebSocket.

Total per-tick cost on a stock CPU:

| Stage | Typical CPU ms | Notes |
|-------|----------------|-------|
| Source resolve | 0-2 | cache hit dominant path |
| Frame grab | 30-120 | dominated by network for HLS |
| YOLO26 inference | 180-260 | OpenVINO on Intel iGPU-class |
| BurstTracker | 1-3 | pure python, few hundred comparisons |
| Layer render | 15-50 | scales with N tracks + enabled layers |
| **Total** | **~230-400 ms** | one tick every ~1s at four enabled layers |

---

## 2. Configuration knobs

### 2.1 Environment variables

| Var | Default | What it does |
|-----|---------|--------------|
| `RTCV_DEVICE` | `cpu` | Pass `cpu` for OpenVINO, or a CUDA index like `0` for GPU |
| `RTCV_INFER_SIZE` | `640` | YOLO input square. 512 is faster on weak CPUs; 960 is sharper but ~2x slower |
| `RTCV_MAX_TRACKS` | `256` | Hard cap on live tracks per camera |
| `RTCV_HEAT_HALFLIFE` | `180` | Seconds until a heat cell decays to half weight |
| `RTCV_PLATES_EASYOCR` | `0` | Set to `1` to enable the non-Latin fallback |
| `RTCV_UPLOAD_MAX_MB` | `500` | Max size of an uploaded local video |
| `RTCV_LOG_LEVEL` | `INFO` | Standard Python logging level |

None are required. Leaving all defaults is a valid path.

### 2.2 `cameras.py` structure

```python
CAMERAS: dict[str, dict] = {
    "cam_id": {
        "id":      "cam_id",         # must equal the key
        "name":    "Human label",
        "kind":    "hls|skyline|webcamera24|local_file",
        "url":     "https://...",    # or omit for local_file
        "path":    "/abs/path.mp4",  # only for local_file
        "area":    "Neighborhood",
        "country": "generic",         # free-form; unused by engine
    },
    ...
}
```

Uploads are injected at runtime by `active_cameras()`, so the static
dict is only the curated list. `country_pool()` returns all keys and
exists purely for notebook back-compat - Repo #2 has no country ladder.

### 2.3 Per-layer thresholds

Each layer's tunables live at the top of its module. The intent is that
you never need to edit code to adjust behavior - see `pose.PERSON_MIN_H`,
`faces.YUNET_MIN_CONF`, `heatmap.DAILY_DECAY`, `tracker.TRACK_HIGH_CONF`,
and so on. Every value has a one-line comment explaining what happens
when you nudge it.

---

## 3. MP4/MKV upload flow

The dashboard exposes `POST /api/upload-video` (multipart). The flow:

1. Browser posts the file to the endpoint.
2. `dashboard_server.py` writes the bytes to
   `src/data/uploads/<sha1prefix>_<origname>` with size cap
   `RTCV_UPLOAD_MAX_MB`.
3. On the next call to `active_cameras()`, the file is enumerated and
   appears with `cam_id = <sha1prefix>_<origname>` (no extension), kind
   `local_file`, path filled in.
4. The dashboard picker refreshes; the new file appears at the bottom
   of the list and can be selected like any other camera.

Deleting the file from disk removes it from the next enumeration.

Supported extensions: `.mp4 .mkv .mov .avi .webm`. Anything else is
rejected with a 415.

---

## 4. Adding a new HLS camera

Two files, small edits:

1. `src/app/cameras.py` - add one entry to `CAMERAS`:
   ```python
   "my_new_cam": {
       "id": "my_new_cam",
       "name": "My street corner",
       "kind": "hls",
       "url": "https://cams.example.com/live/street.m3u8",
       "area": "Downtown",
       "country": "generic",
   },
   ```
2. `src/web/cameras.js` - only edit if you want a nicer default label
   in the picker; the engine already reads `cameras.py` for the
   canonical list.

Restart `serve.py` and the new camera is in the picker.

For a `skyline` or `webcamera24` page URL, use the corresponding `kind`
and let the scraper find the `.m3u8` at runtime.

---

## 5. Troubleshooting

### 5.1 `cv2.error` on frame grab

Almost always a network-side stall on the HLS. Check the printed
`GrabError` timing; if it hits the timeout (default 8s), the camera is
degraded. Try a different `cam_id` first before blaming the code.

### 5.2 OpenVINO IR not loading

Ultralytics prints a one-line hint. Common causes:

- The `_openvino_model/` folder was not fetched with LFS - `git lfs
  pull`.
- `openvino` package too old - `pip install -U openvino openvino-dev`.
- Model file is a partial download - delete the IR folder and let the
  engine regenerate it from the `.pt`.

If everything else fails, delete the IR folder; the engine falls back
to `.pt` cleanly.

### 5.3 FSRCNN silently no-op

The plate layer prints `FSRCNN unavailable - install opencv-contrib`
on the first sample if `cv2.dnn_superres` is missing. Fix:

```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-contrib-python
```

Only ONE opencv wheel should be installed at a time; mixing them
leaves `dnn_superres` missing silently.

### 5.4 EasyOCR download stall

The first non-Latin plate triggers a ~150 MB download per language.
On a slow link the tick times out. Pre-warm it:

```python
import easyocr
easyocr.Reader(["th"])   # or ["ar"], ["ja"], ...
```

### 5.5 Port already in use

`serve.py` scans the next 20 ports if 8000 is busy. To pin a port:

```bash
python serve.py --port 8765
```

---

## 6. Testing

`pytest src/tests/` runs the full suite (28 files, ~180 tests).
No network access is required - HLS resolves are stubbed with
recorded fixtures under `src/tests/fixtures/`.

Key files:

- `test_adapters.py` - source resolvers (skyline, webcamera24, local).
- `test_detect_filters.py` - class filter + confidence gates.
- `test_tracker.py` - two-stage association, coasting, id stability.
- `test_live_analysis.py` - full tick orchestration on stub frames.
- `test_operational.py` - end-to-end dashboard tick on a fixture stream.
- `test_report_pdf.py` - PDF report generator.
- `test_review_queue.py` - active-learning queue admission/eviction.

CI is intentionally not wired up in this repo - the tests are meant
to be run locally before you commit.

---

## 7. Model inventory (the 10)

| # | Model | Role | Format | Location | Honest limitations |
|---|-------|------|--------|----------|--------------------|
| 1 | yolo26m | Primary detection (person, vehicle, train) | PyTorch + OpenVINO | `src/yolo26m_openvino_model/`, `yolo26m.pt` | ~220 ms/frame on CPU (Intel UHD 620 class); misses objects <24 px tall; no CUDA required |
| 2 | yolov8n-plate | LPR stage 1: locate plate boxes inside vehicles | PyTorch + OpenVINO | `yolov8n-plate.pt` + `_openvino_model/` | Best when plate width >=32 px; loosened to 16 px with FSRCNN; conf 0.30 |
| 3 | plate_ocr_global.onnx | LPR stage 2: Latin OCR (digits + A-Z) | ONNX CTC 9-slot | `plate_ocr_global.onnx` | Alphabet is 0-9 A-Z only. Non-Latin scripts are honestly out-of-alphabet |
| 4 | easyocr (optional) | LPR fallback for non-Latin scripts (Thai / Arabic / Japanese) | Python pkg | `~/.EasyOCR/model/` (~150 MB per lang) | Only fires if Latin OCR conf<0.90. Slow on CPU. Uncomment in requirements.txt |
| 5 | FSRCNN_x4.pb | 4x super-resolution on plate / vehicle crops | TF frozen graph via OpenCV dnn_superres | `models/FSRCNN_x4.pb` | Requires `opencv-contrib-python`. Silently no-op if missing |
| 6 | yolov8n-pose | Top-down keypoints (17 COCO joints) on person crops | PyTorch + OpenVINO | `yolov8n-pose.pt` + `_openvino_model/` | Only fires on person boxes >=96 px tall. Below that: bare rectangle |
| 7 | YuNet (face) | Face bounding-box detection (no identification) | ONNX via OpenCV | `src/data/face_detection_yunet_2023mar.onnx` | Deliberately strict: conf>=0.9, box>=24x24. Empty on far-field is by design |
| 8 | OSNet x0.25 msmt17 | Person re-identification embedding | ONNX | `src/data/osnet_x0_25_msmt17.onnx` | 5-10 ms per crop CPU. Falls back to HSV histogram if missing |
| 9 | BurstTracker (custom) | Track IDs across frames (BYTE-style, EMA velocity) | Pure Python | `src/app/tracker.py` | NOT Ultralytics native ByteTrack. Coasting tracks may freeze between ticks |
| 10 | (extension slot) | Reserved for future models (per-script LP recognizers, Jetson variants) | - | - | - |

---

## 8. Layer mechanics (the 10)

### 8.1 `paths` - trails and speed tiers

Mechanism: per-track ring buffer of the last N centroids (default 32).
Render draws the polyline plus a speed badge computed as
`recent_distance_px / recent_elapsed_s / body_length_px`. Speed tiers
are `still / walking / brisk / running`, colored respectively.

Threshold: minimum trail length 4 samples before a badge is drawn.

Why: raw distance in pixels is meaningless across zoom levels; body
lengths per second is roughly comparable across cameras.

### 8.2 `pose` - COCO-17 skeleton

Mechanism: `yolov8n-pose` runs on each person crop above 96 px tall.
Keypoints are drawn as a stick figure with per-limb color.

Threshold: `PERSON_MIN_H = 96` (px). Below that the pose model
produces noise, so the layer skips and draws just the person box.

Why: pose is expensive relative to detection; gating on box height
keeps the total cost bounded and honest.

### 8.3 `gestures` - temporal arm gestures

Mechanism: three-frame rolling analysis of pose keypoints per track
id. Detects `hand_raised` (wrist above shoulder), `both_hands_up`
(both wrists above head), `wave` (wrist oscillates around shoulder
line for >=3 frames).

Threshold: gestures need >=2 consecutive positive frames to fire, to
kill single-frame noise.

Why: single-frame gesture detection is unreliable; the temporal
integration is what makes it usable in a live tile.

### 8.4 `body` - behavior verdicts (alert-only)

Mechanism: per-track feature vector (speed, direction stability,
sudden vertical drop, pose posture) fed into small
hand-tuned classifiers. Emits `running`, `erratic`, `fall_suspect`.

Threshold: `body.MIN_TRACK_LEN = 6` frames. Verdicts are alert-only -
the layer never draws a green "everything fine" badge; only real
alerts show.

Why: silent when boring is the point. Operators tune out systems that
label everything.

### 8.5 `faces` - YuNet boxes, no identification

Mechanism: YuNet ONNX face detector runs on each person crop above
96 px. Draws a small rectangle only.

Thresholds: `YUNET_MIN_CONF = 0.9`, `YUNET_MIN_BOX = 24x24`. Strict on
purpose - a public street cam gives noisy faces and cranking the
threshold up makes the "someone is looking at camera" cue reliable.

Why: this layer is a visual cue only. It never identifies anyone; the
model does not do that. Empty far-field is by design.

### 8.6 `line` - crossing counter on ground contact

Mechanism: operator draws a two-point line on the tile. Each track's
FOOT POINT (bottom-center of its box) is checked against the line
segment; a sign-flip in the signed distance is a crossing. Counters
per direction (`up`, `down`) are maintained.

Threshold: `MIN_TRACK_LEN = 3` samples before a track can register a
crossing.

Why: foot point matches real-world ground contact better than
centroid, so the count roughly equals pedestrians who crossed a
painted road stripe.

### 8.7 `loiter` - dwell alerts on polygons

Mechanism: operator draws polygons. For each track, accumulate seconds
spent inside each polygon; alert once dwell exceeds
`LOITER_MIN_SEC = 30` and clear when the track leaves.

Threshold: 30 s default; tunable from the tab.

Why: too-short thresholds fire on the first passer-by; 30 s catches
actual dwellers without noise.

### 8.8 `parking` - occupancy flips

Mechanism: operator draws polygons around parking bays. Each tick,
compute vehicle-box IoU with each bay; a state flip (empty->filled or
filled->empty) is a discrete event that pipes into the alerts stream.

Threshold: IoU >= 0.35 for "vehicle occupies bay"; two consecutive
positive ticks required to flip to occupied (kills single-frame
detection noise).

Why: bay-level truth is what people expect; per-frame IoU raw is too
jumpy to be usable directly.

### 8.9 `plates` - two-stage LPR

Mechanism:

1. Stage 1 (`yolov8n-plate`): for each vehicle box, run the plate
   detector on the enlarged vehicle crop. Extract plate boxes >=16 px
   wide (with FSRCNN 4x upscale rescuing 16-32 px cases).
2. Stage 2 (`plate_ocr_global.onnx`): CTC decoder on the plate crop
   into a 9-slot A-Z 0-9 alphabet.
3. Per-track cache: results are pooled across the last K frames per
   track id (majority vote on the top-1 read) so a single-frame OCR
   miss does not overwrite a good read.
4. Optional easyocr fallback: if Latin OCR confidence <0.90 AND
   `RTCV_PLATES_EASYOCR=1`, run easyocr in the enabled language and
   overwrite if the fallback confidence is materially higher.

Threshold: plate box conf 0.30 for stage 1, OCR conf 0.65 to accept a
read at all.

Why: two-stage LPR is more robust than end-to-end when the plate
occupies <5% of the vehicle box - which is the typical street-cam
case.

### 8.10 `heat` - 48x27 grid recent-activity map

Mechanism: each observed track's FOOT POINT bumps its 48x27 cell by a
weight equal to the elapsed seconds since the previous sample.
Rendering interpolates the grid to the tile size with a JET colormap,
alpha-blended at 0.35.

Threshold: half-life 180 s (`RTCV_HEAT_HALFLIFE`). Older weight decays
exponentially.

Why: raw counts favor already-busy spots forever; time-weighted
short-horizon map surfaces where activity is NOW.

---

## 9. Hot trail + Investigation gallery

The dashboard maintains two rolling surfaces alongside the live tile.

**Hot trail** - a small horizontal strip of the last N (default 12)
person crops that scored highest on any anomaly signal (fast motion,
gesture, fall_suspect). Clicking a crop scrubs the live tile to the
frame it was taken from.

**Investigation gallery** - full snapshots of any tick that carried at
least one alert. Written to `src/web/snapshots/<cam>/<epoch>.jpg` with
sidecar JSON describing the alert. The gallery tab paginates the
newest first, no server-side filtering.

Both are meant to be lightweight visual memory; neither replaces a
proper VMS.

---

## 10. Active learning + Review workflow

`review_frames.py` maintains a small queue of the boxes the engine was
least confident about. Admission is `uncertainty.py`'s job:

- boxes near the class confidence gate (within +-0.05),
- boxes with volatile class assignment across nearby frames,
- boxes flagged by the operator via the tab.

The review tab shows one box at a time with the picked class + a
one-click relabel. Confirmed labels are appended to
`data/relabel_export.jsonl` for later fine-tuning; nothing is auto-
retrained on-device.

`tools/export_labels.py` converts the JSONL into COCO or YOLO format;
`tools/cv_train.py` and `tools/train_head.py` are thin wrappers that
can fine-tune a head on a small labeled set. Neither is run
automatically; they exist for the operator who wants to close the
loop.

---

## 11. Extensibility

### 11.1 Adding a new layer

1. Create `src/app/my_layer.py` exposing:
   - `render(frame, boxes, tracks, state) -> frame`
   - `initial_state() -> dict`
   - optional `on_operator_edit(state, edit) -> state` for drawn shapes.
2. Register the layer in `live_analysis.LAYERS` (order matters -
   earlier = drawn first, so later layers can cover it).
3. Add a toggle to the live-analysis tab.
4. Write a test under `src/tests/test_my_layer.py` covering the render
   with an empty state.

Nothing else. There is no plugin loader magic; the LAYERS list is the
whole registry.

### 11.2 Adding a new source kind

1. Add the new `kind` string to `cameras.CAMERAS`.
2. Add a resolver branch in `detect_core.resolve_source(kind, row)`.
3. Test with a fixture in `src/tests/test_adapters.py`.

Local files, HLS, and the two scraped page kinds cover almost
everything; the pattern for a new "REST snapshot every 5s" kind is
maybe 30 lines.

### 11.3 Adding a new model

The engine expects the model to expose one of: an Ultralytics YOLO
interface (for detection), an OpenCV `dnn` net, or an
`onnxruntime.InferenceSession`. Wrap yours in a small adapter under
`src/app/` and pipe its outputs into a new layer (see 11.1).

---

## 12. Where to look next

- `README.md` at repo root - shortest path from clone to a live tile.
- `src/docs/PROJECT_GUIDE_HE.md` - Hebrew edition of this document.
- `src/docs/NOTEBOOK_MAIN_HE.md` - cell-by-cell guide for the
  exploratory notebook.
- `src/app/detect_core.py` - the source of truth for the runtime
  pipeline. If any doc contradicts the code, the code wins.
