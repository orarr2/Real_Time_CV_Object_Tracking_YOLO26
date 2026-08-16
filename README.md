# Real-Time CV Object Tracking - YOLO26

Local live computer-vision dashboard: pick a single camera (public HLS, webcamera24, skylinewebcams, or your own uploaded MP4/MKV) and run YOLO26 detection with 10 analysis layers on the same tile in real time.

**No cloud dependencies. No datacenter. No account signup.** Everything runs on your machine.

Repository layout is intentionally minimal: one notebook for exploratory work, one static HTML dashboard, one Python detection engine, one shared model folder. All weights ship in-tree so a fresh clone can produce a live tile without a second download step.

---

## Quick start

```bash
# 1. Clone and enter
git clone <this-repo> real-time-cv-yolo26
cd real-time-cv-yolo26

# 2. Create a virtual environment (Python 3.10+ recommended)
python -m venv .venv
.\.venv\Scripts\activate            # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux

# 3. Install runtime dependencies
pip install -r src/requirements.txt

# 4a. Launch the dashboard (single-camera live tile + 3 tabs)
cd src
python serve.py                     # opens http://localhost:8000

# 4b. OR open the notebook for exploratory work
jupyter lab real_time_cv.ipynb      # sits at repo root
```

The first inference call downloads any missing OpenVINO cache for `yolo26m.pt`. The `.pt` file itself is already tracked in the repo (`yolo26m.pt`, ~42 MB).

---

## What the model predicts

`yolo26m` is trained on COCO 80 classes; the engine keeps the classes that are relevant for street footfall and vehicle tracking:

| Group | COCO classes kept |
|-------|-------------------|
| person | `person` |
| vehicles (road) | `car`, `motorcycle`, `bus`, `truck`, `bicycle` |
| extras | `train` (rail), `dog` (optional) |

Everything else is discarded before the layers run so cost and clutter both stay bounded. The dashboard also draws boxes for each kept class in its own color and passes the boxes to every analysis layer that consumes them.

The notebook and the dashboard share the SAME `yolo26m.pt` weights and the SAME analysis code (`src/app/`). The notebook is the exploratory surface; the dashboard is the operator surface.

---

## Video sources supported

The engine resolves a `cam_id` to a playable stream through one of four adapters:

- `hls` - direct `.m3u8` URL. Use for any camera that already exposes an HLS manifest.
- `skyline` - a `skylinewebcams.com/...` page URL. The engine scrapes the current signed HLS out of the page HTML on demand.
- `webcamera24` - a `webcamera24.com/...` page URL. Same idea, different scraper.
- `local_file` - an absolute path to an MP4/MKV/MOV/AVI/WEBM on disk. The dashboard has an upload button that drops the file into `src/data/uploads/` and immediately registers it as a new camera under `upload_<hash>`.

Adding a new HLS camera is a one-line edit to `src/app/cameras.py`. No collector, no rotation, no country pool - a single tile at a time.

---

## The 10 analysis layers

Every layer draws on top of the SAME frame the operator is watching. Any subset can be toggled on or off from the dashboard tab bar.

1. `paths` - per-track trail history + speed tiers (body-lengths per second).
2. `pose` - COCO-17 top-down skeleton on each person crop.
3. `gestures` - temporal arm gestures across frames (hand_raised, both_hands_up, wave).
4. `body` - behavior verdicts, alert-only (running, erratic, fall_suspect).
5. `faces` - face bounding-boxes only (YuNet). No identification.
6. `line` - crossing counter on ground contact - the operator draws a line, the engine bumps a counter each time a track's foot point crosses.
7. `loiter` - dwell zone alerts on operator-drawn polygons.
8. `parking` - occupancy flips (empty <-> filled) on operator-drawn polygons.
9. `plates` - two-stage license-plate recognition with per-track cache, FSRCNN 4x upscale and optional non-Latin OCR fallback.
10. `heat` - 48x27 grid recent-activity map with 180s half-life decay.

Each layer is one Python module under `src/app/` and is documented in detail in `src/docs/PROJECT_GUIDE.md`.

---

## The 10 ML models used

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
| 10 | (extension slot) | Reserved for future models (per-script LP recognizers or Jetson-optimized variants) | - | - | - |

---

## Notebook

Open `real_time_cv.ipynb` at the repo root. The notebook walks a linear path:

1. Setup and imports (loads `yolo26m.pt`).
2. Pick ONE camera from the catalog (`PICK=N`).
3. Grab a single frame, run inference, plot the result.
4. Loop-sample the camera over a short window and write per-second footfall to CSV.
5. Cross a manual line, render heat, run the plates pipeline on demand, and so on for every layer.

The notebook is the same detection code you run from the dashboard - it is not a second implementation. See `src/docs/NOTEBOOK_MAIN_HE.md` for the cell-by-cell walkthrough (Hebrew).

---

## Dashboard

`python serve.py` binds a local HTTP server on `http://localhost:8000` and opens the default browser. The UI is a single static page (`src/web/index.html`) with:

- ONE full-width live tile at the top - the picked camera plus every enabled layer drawn as an overlay.
- A camera picker (drop-down of catalog cameras + uploaded local files).
- An upload button that accepts MP4 / MKV / MOV / AVI / WEBM and registers the file as a new camera.
- Three tabs below the tile:
  1. **Live analysis** - toggle any of the 10 layers on/off, adjust conf/gate thresholds, draw lines and polygons.
  2. **Investigation gallery** - snapshots of unusual detections + the Hot trail last-seen strip.
  3. **Review queue** - active-learning: label the boxes the engine was least sure of, keep the queue small.

The dashboard is single-tile by design. If you want multi-camera surveillance, run one dashboard per camera.

---

## Repo map

```
real-time-cv-yolo26/
  README.md                         (this file)
  yolo26m.pt                        primary detector, 42 MB
  yolo26m_openvino_model/           <- inside src/, OpenVINO IR cache
  yolov8n-plate.pt + _openvino_model/
  yolov8n-pose.pt  + _openvino_model/
  plate_ocr_global.onnx             Latin plate OCR
  models/
    FSRCNN_x4.pb                    optional 4x super-resolution
  real_time_cv.ipynb                exploratory notebook (root)
  src/
    serve.py                        one-shot dashboard launcher
    requirements.txt
    app/
      detect_core.py                model load + resolve + grab_frame
      live_analysis.py              per-tick orchestration
      dashboard_server.py           HTTP + WebSocket routes
      cameras.py                    camera catalog
      tracker.py                    BurstTracker
      plates.py, pose.py, faces.py, gestures.py, heatmap.py, ...
    data/
      face_detection_yunet_2023mar.onnx
      osnet_x0_25_msmt17.onnx
      uploads/                      uploaded local videos land here
    web/
      index.html                    dashboard UI (single page)
      cameras.js
      snapshots/                    investigation gallery images
    tests/                          pytest suite - 28 test files
    tools/                          calibration + reporting CLIs
    docs/
      PROJECT_GUIDE.md              deep-dive (English)
      PROJECT_GUIDE_HE.md           deep-dive (Hebrew, RTL)
      NOTEBOOK_MAIN_HE.md           cell-by-cell notebook guide (Hebrew, RTL)
```

For anything deeper - configuration knobs, adding sources, layer mechanics, troubleshooting - see `src/docs/PROJECT_GUIDE.md`.
