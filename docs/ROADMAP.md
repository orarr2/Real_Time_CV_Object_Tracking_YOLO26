# Real-Time CV YOLO26 - Technical Roadmap

> **Changes in the 2026-08-23 session** (full decision log in
> `AUDIT_2026-08-23.md`): ~1,900 lines of dead code removed (dead server
> endpoints, burst-analytics chain, blur path, live_samples /
> anomaly_crops / calibrate_conf); plate detector upgraded to
> yolov11-L + OCR to fast-plate-ocr `cct_s_v2` + pose to `yolov8s-pose`;
> body-anomaly gates hardened (ratio 8, 10-sample/10-s bbox window,
> 2-tick debounce, 60-px both-fast fighting rule); NEW Fall-detection
> layer (11 layers now); per-country plate grammar; hot-trail decay +
> 15-s replay ring; PDT clock-sync probe implemented; per-layer drawers
> split into `app/layers/draw.py`. Thresholds reference:
> `src/docs/DECISION_THRESHOLDS_HE.md`.

> **Roadmap status after the 2026-08-23 session:** items now DONE and
> no longer open here: per-layer draw split (first slice of the
> live_analysis refactor), hot-trail strip decay, replay ring
> (15 s scoped variant), fall-detection layer, PDT clock-sync probe,
> per-country plate grammar, model upgrades (pose-S / plate-L /
> OCR cct_s_v2), body-anomaly hardening. Remaining items below are
> unchanged planning text and may reference the pre-split layout.


Source: multi-agent CV/YOLO research pass (2026-08-17). Eight parallel
research strands scanned leading channels (OpenViewer, AI Dev Guy, Mohsin
Ali, Roboflow, SkalskiP, Ultralytics, LearnOpenCV, JetsonHacks) and the
2025-2026 detector/tracker/enterprise-CV landscape (RF-DETR, YOLOE-26,
BoT-SORT, SOLIDER, Ambient / Actuate / Verkada). 56 candidate ideas were
adversarially verified; the survivors are specified below.

This is a **specification** document, not a task list. Each entry
describes what to build, where it hooks into today's pipeline, and the
concrete delta it delivers.

## Table of contents

- [How to read each entry](#how-to-read-each-entry)
- [Top 3 next moves - highest ROI](#top-3-next-moves--highest-roi)
- [Quick wins (hours)](#quick-wins-hours)
- [Medium features (days)](#medium-features-days)
- [Larger initiatives (weeks+)](#larger-initiatives-weeks)
- [Competitive gap map](#competitive-gap-map)
- [Stack upgrade recommendations](#stack-upgrade-recommendations)
- [UX + marketing polish](#ux--marketing-polish)

## How to read each entry

| Field | Meaning |
|---|---|
| **What** | One-line description of the feature |
| **Contribution** | The concrete delta vs today's pipeline - what an operator gains |
| **Where** | Existing file to touch OR new module to add |
| **How** | Implementation sketch + code example |
| **Delta** | before → after in one line |
| **Effort** | rough size (h = hours, d = days, w = weeks) |
| **Deps** | pip packages / weights / external services |
| **Status** | ⬜ not started · 🟡 partial (some scaffolding exists) · ✅ done in-tree |

## Top 3 next moves - highest ROI

1. **Move 1 - foundation fixes** (≈2 days): Q1 + Q3 + Q7. Replace the
   homegrown `BurstTracker` with Ultralytics `model.track(persist=True)`,
   quantize YOLO26 to INT8 with NNCF, add a capability chip that shows
   which OpenVINO device/precision is actually running. Doubles baseline
   FPS on i5-class hardware and stabilises track IDs across all 10
   layers.
2. **Move 2 - viewer → product** (≈2 weeks): M5 + M6 + Q9. Alert bus
   over SSE, auto-clip recorder that persists a 15 s clip on any flagged
   event, PolygonZone editor with named zone counters. Closes the loop
   `zone-defined → event fired → clip saved → phone notified` - the
   feature set Ambient/Verkada charge $$$ for, built on primitives
   already in the tree.
3. **Move 3 - skills velocity** (≈1 month): L1 + M3 + M10. YAML-driven
   "skills" loader (drop in a new YOLO checkpoint + a manifest, get a
   new dashboard layer), close out the fire/smoke layer (~80 % wired),
   YOLOE-26 open-vocab "Ask" layer (`model.set_classes([...])` free-text
   prompt). Moves the shipping cadence from "one new layer per
   sprint" to "one new layer per YAML".

---

# Quick wins (hours)

## Q1. BoT-SORT native tracker

**What** - Replace `src/app/tracker.py` (BurstTracker) with Ultralytics'
built-in `model.track(persist=True, tracker='botsort.yaml')`.

**Contribution** - Today's tracker matches purely on position + velocity
(deliberately, to avoid appearance mismatch on look-alikes). BoT-SORT
adds a lightweight Kalman filter + optional ReID head and is
industry-standard on MOT20 / DanceTrack. Every layer that reads
`track_id` (paths, line, loiter, plates, heat, body, gestures) benefits
immediately: ID-switches on crowded scenes drop measurably, no per-layer
code change needed.

**Where** - `src/app/detect_core.py` (inference call), `src/app/tracker.py`
(deprecate but keep as fallback for the notebook's dwell burst).

**How**

```python
# src/app/detect_core.py
_TRACKER_CFG = str(Path(__file__).parent.parent / "configs" / "botsort.yaml")

def track_frame(model, frame, imgsz=640, conf=0.30):
    """Drop-in replacement for detect_with_boxes that returns tracked boxes
    with stable track_id from Ultralytics' built-in ByteTrack/BoT-SORT."""
    r = model.track(frame, persist=True, tracker=_TRACKER_CFG,
                    conf=conf, imgsz=imgsz, verbose=False, classes=LIVE_CLASSES)
    r = r[0] if isinstance(r, (list, tuple)) else r
    boxes = []
    if r.boxes is None or r.boxes.id is None:
        return boxes
    xyxy = r.boxes.xyxy.cpu().numpy()
    ids  = r.boxes.id.int().cpu().numpy()
    cls  = r.boxes.cls.int().cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    for (x1,y1,x2,y2), tid, c, cf in zip(xyxy, ids, cls, confs):
        boxes.append({
            "x1": float(x1), "y1": float(y1),
            "x2": float(x2), "y2": float(y2),
            "cls": r.names[int(c)],
            "conf": float(cf),
            "track_id": int(tid),
        })
    return boxes
```

```yaml
# configs/botsort.yaml
tracker_type: botsort
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 30
match_thresh: 0.8
gmc_method: sparseOptFlow
with_reid: false          # flip to true after Q1+M1 (SOLIDER)
```

**Delta** - Homegrown position tracker → industry standard. Track ID
stability across occlusion improves markedly on crowded street scenes.

**Effort** - ~4 h (swap + regression test the 11 layers). **Deps** -
none new (bundled with `ultralytics>=8.4`). **Status** - ⬜

---

## Q2. Foot-point + directional line counters

**Status** - ✅ Already in-tree. `update_crossings` in
`src/app/live_analysis.py` uses foot-point `(x, y2)` and splits IN/OUT
via a signed cross-product against the line vector. The top-center
`IN N / OUT N` HUD already renders. No work required - flagged here
for completeness against the research doc.

---

## Q3. INT8 OpenVINO PTQ for YOLO26

**What** - Post-training-quantize the FP16 OpenVINO IR to INT8 with
NNCF. Auto-select the INT8 IR when `YOLO26_INT8=1` env is set.

**Contribution** - Current baseline: yolo26m on OpenVINO CPU ≈ 220 ms/frame
on Intel UHD 620 class. INT8 typically halves latency and reduces RAM by
~4×. This is the single-highest-ROI move on CPU-only baselines because
YOLO inference is 90 %+ of the tick budget.

**Where** - new `src/tools/quantize_openvino.py`; loader in
`src/app/detect_core.py:load_model`.

**How**

```python
# src/tools/quantize_openvino.py
import cv2, nncf
from pathlib import Path
from ultralytics import YOLO

# 1. Export FP16 IR (idempotent - skip if already present).
weights = "yolo26m.pt"
YOLO(weights).export(format="openvino", half=True)
fp_dir  = Path("yolo26m_openvino_model")
xml_fp  = fp_dir / "yolo26m.xml"
xml_int8 = fp_dir / "yolo26m.int8.xml"

# 2. Calibration dataset - ~200 frames from data/heatmap_cache or
#    a stashed sample directory. Doesn't need labels.
def calib_gen():
    for p in sorted(Path("src/data/calibration").glob("*.jpg"))[:200]:
        img = cv2.imread(str(p))
        img = cv2.resize(img, (640, 640))
        img = img.transpose(2, 0, 1).astype("float32") / 255.0
        yield {"images": img[None, ...]}

import openvino as ov
model = ov.Core().read_model(str(xml_fp))
q = nncf.quantize(model, nncf.Dataset(calib_gen()),
                  preset=nncf.QuantizationPreset.MIXED,
                  subset_size=200)
ov.save_model(q, str(xml_int8))
print(f"wrote {xml_int8.stat().st_size/1e6:.1f} MB INT8 model")
```

```python
# detect_core.py loader tweak
def load_model(weights: str):
    if os.environ.get("YOLO26_INT8"):
        p = Path(weights).parent / "yolo26m.int8.xml"
        if p.exists():
            return YOLO(str(p), task="detect")
    return YOLO(weights)
```

**Delta** - 220 ms/frame FP16 → ~110 ms/frame INT8 on same CPU. Frees
half the budget for additional layers.

**Effort** - 1 day incl. accuracy A/B on Section 10 fixtures. **Deps** -
`pip install nncf openvino` (openvino already transitively required).
**Status** - ⬜

---

## Q4. Fall detection on the pose layer

**What** - State-machine that flags a "fall" event when a tracked
person's torso rotates past 60° AND the hip drops by ≥40 % of the
snapshot torso length within 0.7 s AND stays down for 2 s.

**Contribution** - Turns the already-rendering pose keypoints (COCO-17)
into an actionable safety alert. Slots into the same red banner +
"#TID FALL" HUD the body-anomaly layer already uses. Demo-strong on
CCTV footage; no new model download.

**Where** - new `src/app/fall.py` consumed from `_render` in
`live_analysis.py` when `layer == "body"` AND `pose` results present.

**How**

```python
# src/app/fall.py
import time
from dataclasses import dataclass, field
from math import atan2, degrees

FALL_TORSO_ANGLE_DEG = 60      # from vertical
FALL_HIP_DROP_FRAC   = 0.40    # of snapshot torso length
FALL_WINDOW_S        = 0.7
FALL_HOLD_S          = 2.0
FALL_COOLDOWN_S      = 3.0

@dataclass
class _State:
    torso_len_snap: float = 0.0
    fell_at: float | None = None
    alerted_at: float | None = None
    events: list = field(default_factory=list)

_states: dict[int, _State] = {}

def _torso(kps):
    # COCO-17: 5 L-shoulder, 6 R-shoulder, 11 L-hip, 12 R-hip
    (sx, sy) = ((kps[5][0]+kps[6][0])/2, (kps[5][1]+kps[6][1])/2)
    (hx, hy) = ((kps[11][0]+kps[12][0])/2, (kps[11][1]+kps[12][1])/2)
    length = ((sx-hx)**2 + (sy-hy)**2) ** 0.5
    angle = abs(90 - degrees(atan2(sy-hy, sx-hx)))
    return length, angle, (hx, hy)

def check(tid: int, kps, now: float) -> bool:
    """Return True the tick a fall is confirmed for this track."""
    st = _states.setdefault(tid, _State())
    length, angle, (hx, hy) = _torso(kps)
    if st.torso_len_snap == 0.0 or length > st.torso_len_snap:
        st.torso_len_snap = length
    dropped = (angle > FALL_TORSO_ANGLE_DEG)
    if dropped and st.fell_at is None:
        st.fell_at = now
    if not dropped:
        st.fell_at = None
        return False
    if st.fell_at and (now - st.fell_at) >= FALL_HOLD_S:
        if not st.alerted_at or (now - st.alerted_at) > FALL_COOLDOWN_S:
            st.alerted_at = now
            return True
    return False
```

```python
# live_analysis.py - inside _render for layer == "body"
from app.fall import check as _fall_check
sudden = set()
for tr in visible:
    kps = tr.boxes[-1].get("kps")
    if kps and _fall_check(tr.tid, kps, time.time()):
        sudden.add(tr.tid)
        self._emit_event("body", f"FALL detected on #{tr.tid}", tr.boxes[-1])
```

**Delta** - pose skeletons drawn but silent → skeletons + red "FALL"
banner + JSON event that flows through the alert bus (once M5 lands).

**Effort** - 4-6 h incl. false-positive tuning. **Deps** - none new
(pose model already auto-downloads). **Status** - ⬜

---

## Q5. Premium HUD sidebar

**What** - Semi-transparent dark rectangle in the top-left corner of the
canvas overlay with rows of KPI ticks (People, FPS, Alerts, Camera,
Model, Uptime).

**Contribution** - The single most impactful visual polish move - every
viral CV reel (Mohsin Ali PPE, OpenViewer wildfire) uses this pattern.
Turns raw detection into a demo-worthy shot without changing a single
inference path.

**Where** - pure frontend: `src/web/app.js` canvas overlay.

**How**

```js
// src/web/app.js - inside the render loop
function drawHUD(ctx, w, h, d) {
  const rows = [
    ["People",  d.person || 0],
    ["Vehicles", d.vehicles || 0],
    ["FPS",     (1 / (d.tick_dt || 1)).toFixed(1)],
    ["Alerts",  d.alerts_open || 0],
    ["Model",   d.model_id || "yolo26m"],
    ["Camera",  d.cam_name || "-"],
  ];
  const pad = 12, rowH = 22, colW = 220;
  ctx.save();
  ctx.fillStyle = "rgba(15,23,42,0.72)";
  ctx.strokeStyle = "rgba(59,130,246,0.85)";
  ctx.lineWidth = 2;
  const hudH = rows.length * rowH + pad * 2;
  ctx.beginPath();
  ctx.roundRect(20, 20, colW, hudH, 8);
  ctx.fill(); ctx.stroke();
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillStyle = "#f1f5f9";
  rows.forEach(([k, v], i) => {
    const y = 20 + pad + i * rowH + 14;
    ctx.fillText(k, 34, y);
    ctx.fillText(String(v), 34 + 110, y);
  });
  ctx.restore();
}
```

Toggle via a header button (`HUD [on|off]`). Persist choice in
`localStorage`.

**Delta** - Bare video + boxes → CCTV-command-center aesthetic without
touching detection at all.

**Effort** - 2-3 h. **Deps** - none. **Status** - ⬜

---

## Q6. `/api/events.jsonl` sink + `/api/export.csv`

**What** - Append-only JSONL log of every emitted event
(`_emit_event` call site) + a CSV export endpoint with `?from=&to=&layer=`
filtering.

**Contribution** - Today every event dies with the session. This makes
the dashboard the front door of a real evidence pipeline - operator
"exports the last hour" and hands the CSV to their manager. Prerequisite
for M6 (auto-clip).

**Where** - `src/app/live_analysis.py:_emit_event` (add file append);
`src/app/dashboard_server.py` (new route).

**How**

```python
# live_analysis.py - extend _emit_event
_EVENTS_LOG = Path("src/data/events.jsonl")

def _emit_event(self, layer, msg, box=None):
    ev = {"ts": time.time(), "cam": self.cam_id, "layer": layer,
          "msg": msg, "box": box}
    self.events.appendleft(ev)          # existing in-memory ring
    try:
        _EVENTS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _EVENTS_LOG.open("a") as f:
            f.write(json.dumps(ev, default=str) + "\n")
        # rolling size cap - trim if > 10 MB
        if _EVENTS_LOG.stat().st_size > 10_000_000:
            lines = _EVENTS_LOG.read_text().splitlines()[-50_000:]
            _EVENTS_LOG.write_text("\n".join(lines) + "\n")
    except OSError:
        pass
```

```python
# dashboard_server.py
if path == "/api/export.csv":
    q = parse_qs(query)
    fr = float(q.get("from", [0])[0])
    to = float(q.get("to", [time.time()+1])[0])
    layer = q.get("layer", [None])[0]
    self.send_response(200)
    self.send_header("Content-Type", "text/csv")
    self.end_headers()
    self.wfile.write(b"ts,cam,layer,msg,x1,y1,x2,y2\n")
    with _EVENTS_LOG.open() as f:
        for line in f:
            ev = json.loads(line)
            if not (fr <= ev["ts"] <= to): continue
            if layer and ev["layer"] != layer: continue
            b = ev.get("box") or {}
            row = f'{ev["ts"]:.3f},{ev["cam"]},{ev["layer"]},"{ev["msg"]}",'\
                  f'{b.get("x1","")},{b.get("y1","")},{b.get("x2","")},{b.get("y2","")}\n'
            self.wfile.write(row.encode())
    return
```

**Delta** - Events die with the session → events survive across
restarts and can be handed off as a CSV artifact.

**Effort** - 3-4 h. **Deps** - none. **Status** - ⬜

---

## Q7. Startup backend selector + capability chip

**What** - On serve.py boot, probe `openvino.Core().available_devices`,
warm up 20 frames on each, pick the fastest that fits in RAM, cache
choice in `~/.yolo26_pref.json`. Publish `{device, precision,
ms_per_frame}` to a `/api/system` endpoint that renders a small chip in
the dashboard header.

**Contribution** - Today the operator has no idea if inference is
running on GPU/NPU/CPU. The chip settles the biggest silent perf
mystery: "is it fast because it's on my iGPU or slow because it's on
CPU?".

**Where** - `src/app/detect_core.py:load_model`,
`src/app/dashboard_server.py` (new route), `src/web/index.html` (chip).

**How**

```python
# detect_core.py
import openvino as ov
def select_backend(weights="yolo26m.pt"):
    pref = Path.home() / ".yolo26_pref.json"
    if pref.exists():
        return json.loads(pref.read_text())
    devices = ov.Core().available_devices  # ['CPU','GPU','NPU',...]
    best = None
    for dev in devices:
        try:
            model = YOLO(weights)
            model.to(dev.lower())
            t0 = time.time()
            for _ in range(20):
                model.predict(np.zeros((640,640,3), dtype=np.uint8), verbose=False)
            ms = (time.time()-t0)/20 * 1000
            if best is None or ms < best["ms"]:
                best = {"device": dev, "precision": "fp16", "ms": ms}
        except Exception:
            continue
    pref.write_text(json.dumps(best))
    return best
```

**Delta** - Silent inference → visible chip like
`OpenVINO/CPU · FP16 · 220 ms/frame` in dashboard header.

**Effort** - 4-6 h. **Deps** - `openvino` (already required for the IR
cache). **Status** - ⬜

---

## Q8. LPR OCR robustness

**Status** - 🟡 Partially in-tree already:
- Per-track cache exists (a plate reads once per track).
- Vehicle-width gate exists (`MIN_VEHICLE_W = 60`, `MIN_VEHICLE_W_MOTO = 40`).
- YouTube-blanket skip removed 2026-08-17 (`b9b5176`).

**Remaining work** - Frame-skip (`PLATES_OCR_EVERY_N=5`) + confidence
gate on candidate reads (`conf<0.6 or len<4 → reject`). See M9 for the
GDPR-safe persistence layer.

**Effort** - 2 h. **Deps** - none. **Status** - 🟡

---

## Q9. PolygonZone editor

**What** - Frontend polygon editor on the live canvas (click to place
vertices, double-click to close). POST to `/api/zones`, persist to
`data/zones/<cam>.json`.

**Contribution** - Today's zones (loiter, parking) are configured by
hand-editing JSON. An in-canvas editor turns zone setup from
5-minutes-of-console-work into 10-seconds-of-clicks. Prerequisite for
M7 (density per zone) and L4 (rules catalog).

**Where** - `src/web/app.js` (editor UI) + `src/app/dashboard_server.py`
(POST /api/zones already partly wired via `save_zones` in cameras.py).

**How** - Backend already exposes `save_zones` and `resolve_zones`. The
missing piece is the JS editor:

```js
// src/web/app.js - polygon editor
let drawState = { active: false, pts: [], zone_kind: "loiter" };
canvas.addEventListener("click", (e) => {
  if (!drawState.active) return;
  const rect = canvas.getBoundingClientRect();
  const nx = (e.clientX - rect.left) / rect.width;
  const ny = (e.clientY - rect.top) / rect.height;
  drawState.pts.push([nx, ny]);
});
canvas.addEventListener("dblclick", async () => {
  if (drawState.pts.length < 3) return;
  await fetch("/api/zones", {
    method: "POST",
    body: JSON.stringify({
      cam: currentCam,
      zones: [{kind: drawState.zone_kind, points: drawState.pts,
               dwell_s: 30, name: prompt("zone name?")}],
    }),
  });
  drawState = { active: false, pts: [], zone_kind: "loiter" };
});
```

**Delta** - JSON hand-edit → live click-to-draw with immediate takeup
by the analysis loop.

**Effort** - 6-8 h. **Deps** - none. **Status** - 🟡 (backend done,
frontend editor missing).

---

## Q10. Boot smoke test + dependency check

**What** - `tests/test_smoke_boot.py` that imports every module under
`src/app/`, parses `src/requirements.txt`, and runs the app factory
against a 640×360 black stub frame.

**Contribution** - Catches import breakage / signature drift on every
CI run. Zero runtime cost.

**Where** - `src/tests/test_smoke_boot.py`.

**How**

```python
import importlib, pkgutil
import numpy as np
from pathlib import Path

def test_every_app_module_imports():
    pkg = importlib.import_module("app")
    for m in pkgutil.walk_packages(pkg.__path__, "app."):
        importlib.import_module(m.name)

def test_load_model_on_stub_frame():
    from app.detect_core import load_model, detect_and_count
    model = load_model("yolo26m.pt")
    counts = detect_and_count(model, np.zeros((360,640,3), dtype=np.uint8))
    assert set(counts) >= {"person", "vehicles"}
```

**Delta** - Silent import breakage → CI red-light on first commit that
breaks the API surface.

**Effort** - 2 h. **Deps** - `pytest` (already in dev tree).
**Status** - ⬜

---

# Medium features (days)

## M1. SOLIDER-Swin-Tiny Re-ID

**What** - Replace `osnet_x0_25_msmt17.onnx` with SOLIDER-Swin-Tiny
(Apache-2.0, ~15 MB INT8) as the Re-ID backbone.

**Contribution** - Current OSNet-x0_25 is the WEAKEST checkpoint of an
older backbone (2019). SOLIDER-Swin-Tiny (2023) is +8-12 % Rank-1 on
MSMT17 at similar cost. Direct win for the notebook's Section 5b
"unique visitors" counter and any future FAISS gallery (M2).

**Where** - new class `SoliderEmbedder` in `src/app/reid_embed.py`
alongside the existing OSNet embedder; select via
`REID_BACKBONE=solider_swin_t` env var.

**How**

```python
# reid_embed.py
class SoliderEmbedder:
    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path,
                                          providers=["CPUExecutionProvider"])
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def embed(self, crop_bgr):
        img = cv2.resize(crop_bgr, (128, 384))   # SOLIDER default
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2,0,1)[None, ...]
        out = self.sess.run(None, {"images": img})[0][0]
        return out / (np.linalg.norm(out) + 1e-9)
```

**Delta** - Rank-1 ~62 % (OSNet) → ~74 % (SOLIDER) on MSMT17, same CPU
budget. **Effort** - 1-2 d. **Deps** - `onnxruntime` + SOLIDER ONNX
export. **Status** - ⬜

---

## M2. FAISS gallery for returning visitors

**What** - On track birth, compute the Re-ID embedding + cosine-search a
FAISS index. Match > 0.75 → reuse existing `person_uid` (EMA-update the
stored vector); else insert. Persist to `data/reid.sqlite`.

**Contribution** - Answers "how many UNIQUE customers walked past
today?" across days, not just per-session. The infrastructure the
notebook already has (Section 5b re-ID) becomes a persistent visitor
identity service.

**Where** - extend `src/app/reid.py` with a `FaissGallery` class;
new endpoint `GET /api/reid/gallery-stats`.

**How**

```python
import faiss, sqlite3, numpy as np

class FaissGallery:
    def __init__(self, path="src/data/reid.sqlite", dim=768):
        self.db = sqlite3.connect(path)
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS entities(
            uid INTEGER PRIMARY KEY AUTOINCREMENT,
            cls TEXT, first_seen REAL, last_seen REAL,
            sightings INT DEFAULT 1, vec BLOB);
          """)
        self.index = faiss.IndexFlatIP(dim)
        for uid, vec in self.db.execute(
                "SELECT uid, vec FROM entities"):
            self.index.add(np.frombuffer(vec, dtype=np.float32)[None,:])

    def match_or_insert(self, embedding, cls, thresh=0.75):
        v = embedding.astype(np.float32)[None,:]
        if self.index.ntotal > 0:
            D, I = self.index.search(v, k=1)
            if D[0][0] >= thresh:
                uid = int(I[0][0]) + 1  # FAISS row → sqlite uid
                self.db.execute("UPDATE entities SET last_seen=?, "
                                "sightings=sightings+1 WHERE uid=?",
                                (time.time(), uid))
                self.db.commit()
                return uid, False
        cur = self.db.execute("INSERT INTO entities(cls, first_seen, "
                              "last_seen, vec) VALUES (?,?,?,?)",
                              (cls, time.time(), time.time(), v.tobytes()))
        self.db.commit()
        self.index.add(v)
        return cur.lastrowid, True
```

**Delta** - Per-session identities → cross-session persistent IDs with
sightings + first/last-seen. **Effort** - 2-3 d. **Deps** - `faiss-cpu`
(~2 MB). **Status** - ⬜

---

## M3. Fire / smoke layer completion

**Status** - 🟡 80 % wired already. `_fire_pass`, `FIRE_MODEL_PATH`,
`FIRE_CONFIRM_TICKS = 2` and `draw_fire_layer` all exist in
`live_analysis.py`. Missing: the weights file.

**Remaining work** - Fetch a permissive-license fire/smoke detector
(FIgLib-trained, or Roboflow smoke checkpoint), OpenVINO-convert, wire
the download into the notebook setup cell (like the LPR weights).

**Effort** - 4-6 h once the weights source is settled. **Status** - 🟡

---

## M4. PPE compliance layer

**What** - 11th detection layer: run a construction-safety YOLO (~50 MB,
10 classes: hat, vest, gloves, mask, etc.) on person crops. Match hat/
vest boxes back to `track_id` via IoU. Render per-person `OK` / `NF`
badge next to the pose skeleton and a KPI in the HUD.

**Contribution** - The single most-common viral demo in the Mohsin Ali /
OpenViewer content niche. Direct pitch for construction/warehouse
operators.

**Where** - new `src/app/ppe.py` (mirrors the two-stage pattern of
`plates.py`); layer entry in `live_analysis.py:LIVE_LAYERS`.

**How** - Two-stage exactly like plates: outer YOLO26 finds `person`
boxes; inner PPE YOLO runs on each person crop; IoU-associate each PPE
class to the person; render badge.

```python
# src/app/ppe.py
from ultralytics import YOLO
PPE_CLASSES = ("hat", "vest", "gloves", "mask")

def attach_ppe(person_boxes, frame, ppe_model, min_conf=0.35):
    for pb in person_boxes:
        crop = frame[int(pb["y1"]):int(pb["y2"]),
                     int(pb["x1"]):int(pb["x2"])]
        if crop.size == 0: continue
        r = ppe_model.predict(crop, conf=min_conf, verbose=False)[0]
        found = set(r.names[int(c)] for c in r.boxes.cls.int().cpu().numpy())
        pb["ppe"] = {cls: (cls in found) for cls in PPE_CLASSES}
        pb["ppe_ok"] = all(pb["ppe"].values())
```

**Delta** - Empty operator gap → construction-vertical demo the moment
weights land. **Effort** - 2-3 d. **Deps** - Roboflow construction-safety
weights + `PPE_MODEL_PATH` env. **Status** - ⬜

---

## M5. Detection → Rule → Alert bus (SSE + ntfy)

**What** - Central event bus. Every `_emit_event` push goes into a
ring-buffered `Event` dataclass, streamed over `/api/events` (SSE) and
optionally POSTed to `ntfy.sh` for phone push.

**Contribution** - The wire that connects everything else - plate
match, fall, PPE violation, zone entry all become "events" the operator
subscribes to (or forwards to their phone).

**Where** - new `src/app/alerts.py`; new `/api/events` SSE endpoint in
`dashboard_server.py`; hot-reloadable rules in
`src/data/alert_rules.json`.

**How**

```python
# src/app/alerts.py
from dataclasses import dataclass, asdict
from collections import deque
import json, time, threading
import urllib.request

@dataclass
class Event:
    kind: str           # "fall" | "plate" | "zone_entry" | ...
    cam: str
    subject_id: int | None
    box: dict | None
    ts: float
    meta: dict

class AlertBus:
    def __init__(self, cap=500):
        self.ring = deque(maxlen=cap)
        self.lock = threading.Lock()
        self.subscribers = set()  # sse queues
        self.rules = self._load_rules()

    def emit(self, ev: Event):
        with self.lock:
            self.ring.append(ev)
            for q in list(self.subscribers):
                q.put_nowait(asdict(ev))
        for r in self.rules:
            if r["kind"] == ev.kind and eval(r.get("when","True"), {}, asdict(ev)):
                self._notify(r, ev)

    def _notify(self, rule, ev):
        if rule.get("ntfy"):
            urllib.request.urlopen(rule["ntfy"],
                data=f'{ev.kind} on {ev.cam}: {ev.meta.get("msg","")}'.encode(),
                timeout=3)
```

```python
# dashboard_server.py - SSE stream
if path == "/api/events":
    self.send_response(200)
    self.send_header("Content-Type", "text/event-stream")
    self.send_header("Cache-Control", "no-cache")
    self.end_headers()
    q = queue.Queue()
    _BUS.subscribers.add(q)
    try:
        while True:
            ev = q.get(timeout=15)
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            self.wfile.flush()
    finally:
        _BUS.subscribers.discard(q)
    return
```

**Delta** - In-memory events lost on session end → cross-session,
push-notifiable event stream.

**Effort** - 3-4 d. **Deps** - none (SSE is stdlib). ⚠️
Privacy: run YuNet face-blur on any snapshot before POSTing to ntfy.
**Status** - ⬜

---

## M6. Auto-clip recorder

**What** - 15 s pre-event rolling buffer of annotated frames. On any
alert-bus event, drain the buffer + record 5 s post → mp4v file under
`src/web/snapshots/clips/`. `/api/clips` lists them with thumbnails.

**Contribution** - Turns any live alert into forensic evidence with
provenance. The single feature the operator asked about most in the
research pass.

**Where** - new `src/app/clip_recorder.py`, hooked to the alert bus M5.

**How**

```python
# src/app/clip_recorder.py
from collections import deque
import cv2, hashlib, time
from pathlib import Path

class ClipRecorder:
    def __init__(self, fps=8, pre_s=15, post_s=5):
        self.buf = deque(maxlen=int(pre_s * fps))
        self.fps = fps
        self.post_frames = int(post_s * fps)
        self.pending = []
        self.dir = Path("src/web/snapshots/clips")
        self.dir.mkdir(parents=True, exist_ok=True)

    def tick(self, annotated_frame):
        self.buf.append(annotated_frame.copy())
        for p in list(self.pending):
            p["frames"].append(annotated_frame.copy())
            if len(p["frames"]) >= self.post_frames:
                self._flush(p)
                self.pending.remove(p)

    def on_event(self, event):
        self.pending.append({"event": event, "frames": list(self.buf)})

    def _flush(self, p):
        stem = f'{int(p["event"].ts)}_{p["event"].kind}_{p["event"].cam}'
        path = self.dir / f"{stem}.mp4"
        h, w = p["frames"][0].shape[:2]
        vw = cv2.VideoWriter(str(path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             self.fps, (w, h))
        for f in p["frames"]:
            vw.write(f)
        vw.release()
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.dir / f"{stem}.sha256").write_text(sha)
```

**Delta** - Screen-only alerts → mp4 clips + SHA-256 provenance sidecar
that can be handed to authorities. **Effort** - 2-3 d. **Deps** - none
(cv2 already required). **Status** - ⬜

---

## M7. Zone-scoped density + queue wait-time

**What** - For each zone (Q9), publish `{count, median_dwell_s,
rolling_60s_median}`. Chart.js sparkline in dashboard sidebar.

**Contribution** - Direct queue-analytics parity with Bosch / Verkada /
Actuate marketing pages. Reuses zone + presence code already in
`src/app/presence.py` and `src/app/live_analysis.py`.

**Where** - new `src/app/density.py`; new endpoint `/api/density`;
Chart.js sparkline in dashboard.

**Delta** - Zone events flip only (occupied/vacant) → per-zone density
+ wait-time time-series. **Effort** - 2-3 d (depends on Q1 + Q9).
**Deps** - none. **Status** - ⬜

---

## M8. Plate super-res: FSRCNN → SwinIR-lite (feature-flagged)

**What** - Add `SwinIR-lite` as an alternative super-res backend behind
`PLATE_SR=swinir|fsrcnn|off`. Default stays `fsrcnn` until A/B on a
fixed 30-plate fixture shows the swap is worth it.

**Contribution** - SwinIR generally beats FSRCNN by 1-2 dB PSNR on
plate crops; the difference is often the boundary between "unreadable
blur" and "8-of-9 characters correct".

**Where** - `src/app/plates.py:_upscale_for_ocr` - add a backend switch.

**How**

```python
_SR_BACKEND = os.environ.get("PLATE_SR", "fsrcnn")

def _upscale_for_ocr(plate_bgr):
    if _SR_BACKEND == "off":
        return plate_bgr
    if _SR_BACKEND == "swinir":
        return _swinir_upscale(plate_bgr)   # OpenVINO IR
    return _fsrcnn_upscale(plate_bgr)       # legacy default
```

**Delta** - FSRCNN-only → pluggable SR backends with a documented A/B
harness. **Effort** - 2 d + 1 d benchmark. **Deps** - SwinIR-lite ONNX
(~10 MB) + `openvino`. **Status** - ⬜

---

## M9. HMAC-hashed plate persistence (GDPR-safe)

**What** - Never persist a plate string in plaintext. Store
`(hmac_sha256(plate, key), first_seen, last_seen, count, vehicle_class)`.
Key at `src/data/.plate_key` (chmod 600, gitignored).

**Contribution** - Plate persistence today is off-by-default because it's
a compliance grenade. HMAC makes it safe to persist - you can still count
unique plates, detect "same plate returning", but the plain text never
touches disk.

**Where** - `src/app/plates.py` - extend the read-cache with a persist
step through an HMAC helper.

**How**

```python
# plates.py
import hmac, hashlib, secrets
from pathlib import Path

_KEY_PATH = Path("src/data/.plate_key")
def _key():
    if not _KEY_PATH.exists():
        _KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _KEY_PATH.write_bytes(secrets.token_bytes(32))
        _KEY_PATH.chmod(0o600)
    return _KEY_PATH.read_bytes()

def hash_plate(plate: str) -> str:
    return hmac.new(_key(), plate.encode(), hashlib.sha256).hexdigest()
```

**Delta** - No persistence (compliance risk) → GDPR/KVKK-safe unique-plate
counter. **Effort** - 1-2 d. **Deps** - none. **Status** - ⬜

---

## M10. YOLOE-26 open-vocabulary "Ask" layer

**What** - New 11th layer that accepts a free-text list of prompts:
"red backpack, cigarette, smoke, delivery van, person with umbrella".
`model.set_classes(prompts)` sets the label vocabulary; detections then
flow through the same overlay path.

**Contribution** - Closes fire + smoking + arbitrary-prompt gaps in
ONE model download instead of one weight per category. Direct answer to
"can it detect X?" without training.

**Where** - new `src/app/askvocab.py`; layer entry in `LIVE_LAYERS`;
`POST /api/prompt` in `dashboard_server.py`.

**How**

```python
# askvocab.py
from ultralytics import YOLO
_MODEL = None
def load_yoloe():
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO("yoloe-11s-seg.pt")   # ~40 MB
    return _MODEL

def detect_by_prompt(frame, prompts):
    m = load_yoloe()
    m.set_classes(prompts, m.get_text_pe(prompts))
    r = m.predict(frame, verbose=False)[0]
    return _to_boxes(r)
```

**Delta** - Fixed 80 COCO classes → operator types "cigarette, backpack"
and sees live detections. **Effort** - 3-4 d. **Deps** - YOLOE-11s-seg
weights (~40 MB), Ultralytics ≥ 8.4. **Status** - ⬜

---

## M11. "Attention" chip per person (neutral labeling)

**What** - Fuse loiter dwell + gesture flags + path curvature into a
per-person score in `[0, 1]`. Render a floating chip above the box
labelled `Attention 0.72` with hover-tooltip listing the reasons.

**Contribution** - Direct match for the AI-Dev-Guy "Normal / Suspicious"
viral pattern - but with **neutral wording** (`Attention`, not
`Suspicious`) to avoid false-positive branding of real individuals.

**Where** - `src/app/behavior.py` + frontend chip in `app.js`.

**How** - Score formula runs on the existing behaviour signals:

```python
def attention_score(dwell_s, gesture_count, path_curvature):
    dwell_bonus = min(1.0, dwell_s / 30.0)          # linger 30s+ = 1.0
    gesture_bonus = min(1.0, gesture_count / 3.0)   # 3+ gestures = 1.0
    curvature_bonus = min(1.0, path_curvature / 2.0)
    score = 0.4 * dwell_bonus + 0.3 * gesture_bonus + 0.3 * curvature_bonus
    reasons = []
    if dwell_bonus > 0.5: reasons.append(f"dwelling {int(dwell_s)}s")
    if gesture_bonus > 0.5: reasons.append(f"{gesture_count} gestures")
    if curvature_bonus > 0.5: reasons.append("erratic path")
    return {"score": round(score, 2), "reasons": reasons}
```

**Delta** - Individual raw signals → a single interpretable per-person
number the operator can act on. **Effort** - 3-4 d. **Deps** - none.
**Status** - ⬜

---

## M12. Ground-plane homography → km/h speed

**What** - 4-point calibrator (operator clicks a known rectangle on the
street). `cv2.getPerspectiveTransform` gives the pixel→metre mapping.
Each track's centroid in metres gives real km/h speed instead of
pixels/sec.

**Contribution** - Turns today's `slow/moving/fast` chip into an actual
number (with the honest ±20 % accuracy caveat). Match for the AI-Dev-Guy
speed-cam viral demo.

**Where** - `src/app/speed.py` + a small calibrator dialog in
`app.js`.

**How** - `cv2.getPerspectiveTransform([4-image-points], [4-metre-points])`
once per camera, persist in `cameras.py`. Then every track's centroid
maps through the matrix; a 1-second moving average smooths out the
tracker jitter.

**Delta** - px/s → km/h with per-camera calibration. **Effort** - 2-3 d.
**Deps** - none. **Status** - ⬜

---

# Larger initiatives (weeks+)

## L1. Pluggable "skills" loader

**What** - YAML manifest → dynamic YOLO layer. Drop a checkpoint in
`skills/<name>/model.pt` and a manifest with `{classes, colors, notify_on}`
next to it; a new dropdown entry appears in the dashboard without a
Python change.

**Contribution** - Reshapes the shipping cadence. Currently every new
layer = a new Python module (`plates.py`, `faces.py`, ...). With L1, a
new layer = a directory + YAML. This is exactly how OpenViewer ships a
new reel per week.

**Where** - new `src/app/skills.py`; per-skill directories under
`skills/`.

**How** - Manifest:

```yaml
# skills/wildfire_smoke/manifest.yaml
name: wildfire_smoke
weights: model.pt          # relative to this directory
classes: ["smoke", "fire"]
colors: {smoke: "#94a3b8", fire: "#dc2626"}
min_conf: 0.35
confirm_ticks: 2
notify_on: ["fire"]
```

Loader iterates `skills/*/manifest.yaml` at boot, registers each as an
optional layer in `LIVE_LAYERS`.

**Delta** - Add-a-layer = 1 day of Python + 1 PR → drop a directory,
restart, done. **Effort** - 1-2 w. **Deps** - none. **Status** - ⬜

---

## L2. SAM 2 click-to-track promptable mask overlay

**What** - Operator clicks any object in the video → SAM 2 tiny propagates
a mask + track through subsequent frames. Overlay a coloured mask with
smoothed contour instead of a bounding box.

**Contribution** - The single most-viral live demo of the 2024-2025
season. Ideal for hands-on demos.

**Where** - new `src/app/sam2.py`; WebSocket channel for click events;
frontend click handler in `app.js`.

**Deps** - `sam2_t.pt` (~40 MB); careful memory management on CPU.
**Effort** - 2-3 w. **Status** - ⬜ (flagship demo, not a critical path).

---

## L3. Split-view multi-source (2 cameras concurrently)

**What** - Lift `MAX_SESSIONS = 1` to 2, 2-column CSS grid,
per-view id namespace. No cross-camera Re-ID at first - that comes with
M2 (FAISS gallery).

**Effort** - 1 w. **Deps** - none. Requires ~2× CPU budget so wire it
behind a config flag. **Status** - ⬜

---

## L4. Rules catalog UI ("30+ signatures")

**What** - Ship 20-30 preset alert rules (`"after-hours perimeter
breach"`, `"checkout queue > 4"`, `"PPE violation"`, `"parking spot
occupied > 30 min"`). Compose them in a UI, save to
`data/alert_rules.json`, hot-reloaded by M5's alert bus.

**Contribution** - This is the *repackaging* that Ambient / Actuate /
Vaidio charge for. The primitives (line, loiter, parking, PPE, plates)
already exist - the catalog turns them into "signatures".

**Effort** - 2 w after L1 + M5. **Deps** - none. **Status** - ⬜

---

## L5. Natural-language track search (CLIP / SigLIP)

**What** - Store a CLIP embedding of the sharpest crop per completed
track in SQLite. Query the dashboard with `"person in red jacket"` /
`"white SUV"` → cosine top-K → grid of matching track thumbnails.

**Contribution** - Direct match for Verkada's flagship "AI Search"
feature - the single most operator-requested capability across the
enterprise survey.

**Where** - new `src/app/track_search.py`; `POST /api/search`.

**How**

```python
# track_search.py - using open_clip
import open_clip, torch
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

def embed_crop(crop_bgr):
    img = preprocess(Image.fromarray(
        cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)))
    with torch.no_grad():
        return model.encode_image(img[None]).squeeze().numpy()

def embed_text(query):
    with torch.no_grad():
        return model.encode_text(tokenizer([query])).squeeze().numpy()
```

**Effort** - 2 w. **Deps** - `open_clip_torch` (~150 MB weights).
**Status** - ⬜

---

## L6. Local Ollama VLM narrator (opt-in only)

**What** - Every N minutes, feed the last few flagged events + a couple
of representative frames to a local VLM (SmolVLM-2.2B) that returns a
one-paragraph summary. Rendered as a banner in the dashboard sidebar.

**Contribution** - Turns raw event lists into human-readable situational
reports.

**Effort** - 3-4 w after M5. **Deps** - Ollama + SmolVLM (~2.2 GB).
Opt-in only. Explicit "AI-generated, may be inaccurate" badge.
**Status** - ⬜

---

## L7. YOLO26-seg masks for path/loiter/parking/heat

**What** - Swap YOLO26 detection weights for `yolo26m-seg.pt`; use the
segmentation mask contour centroid instead of bbox foot-point.

**Contribution** - Foot-point today biases upward on long silhouettes
(person with a large backpack, someone riding a scooter). Contour
centroid is exact.

**Effort** - 1 w. **Deps** - `yolo26m-seg.pt` weights (~50 MB).
**Status** - ⬜

---

## L8. RF-DETR vs YOLO26 A/B benchmark cell

**What** - Add a Section 11 to the notebook that runs `rfdetr-nano`
and `yolo26m` on the same 30 calibration frames, plots MAE and latency
side by side. Persist to `data/bench_YYYYMMDD.json`.

**Contribution** - Turns model-choice from "trend-following" into a
reproducible benchmark documented in the repo.

**Effort** - 3-5 d. **Deps** - `pip install rfdetr`. **Status** - ⬜

---

# Competitive gap map

| Capability | Where competitors have it | Where we are | Gap-closing move |
|---|---|---|---|
| Fire / smoke | OpenViewer, Actuate, Ambient | Skeleton wired, weights missing | **M3** |
| PPE compliance | Mohsin Ali, Ambient, Vaidio | Not started | **M4** |
| Fall / collapse | Actuate, Verkada | Keypoints flow, no state machine | **Q4** |
| Weapon detection | Actuate, Ambient (flagship) | Not started; no permissive weights | *deferred* - needs responsible source |
| ALPR advanced (make / model) | Vaidio, Bosch, Milestone | Plate text only | *skipped* - better ROI in Q4 / M11 |
| Retail queue + wait-time | Actuate, Bosch, Verkada | Not started | **M7** |
| AI Search (natural language) | Verkada (flagship), Motorola Avigilon | Not started | **L5** |
| Appearance-attribute filters | Bosch IVA Pro | Not started | *composed of L5 + Q10-style chips* |
| Signed evidence export | Hayden AI, enterprise | Not started | **M6 + Q6** |
| Zone editor with named alerts | Roboflow, Verkada, Prosegur | Backend done, editor missing | **Q9** |
| Open-vocab detection | Roboflow YOLOE, Grounding DINO | Not started | **M10** |
| Rules composer / signatures marketplace | Ambient (150 signatures), Vaidio (30+) | 10 hardcoded layers | **L4** |

---

# Stack upgrade recommendations

## Models

| Model | Recommendation | Reason |
|---|---|---|
| **YOLO26** (current primary) | Keep | Still the CPU sweet spot |
| **RF-DETR-L / X** | Skip on CPU | Only meaningful on GPU |
| **RF-DETR-Nano** | Consider for L8 A/B only | ONNX runs on CPU |
| **YOLOE-26** (new secondary) | Add via **M10** | Open-vocab prompts unlock fire + smoking + arbitrary in one model |
| **YOLO26-seg** | Consider via **L7** | Mask centroid > foot-point for long silhouettes |
| **OSNet → SOLIDER-Swin-Tiny** | Swap via **M1** | +10 pp Rank-1 at same CPU cost |
| **SAM 2 tiny** | Only for L2 demo | Not production-ready on CPU-only |
| **CLIP ViT-B/32 / SigLIP** | Add via **L5** | Enables track search + attribute chips |

## Runtime

- **INT8 PTQ with NNCF** (**Q3**) - the single highest-ROI move on
  CPU-only baselines. ~2× FPS, ~4× RAM.
- **OpenVINO Core probe + auto-select** (**Q7**) - becomes meaningful
  after Q3 (nothing to select between if only one precision exists).
- **TensorRT** - skip until there's discrete NVIDIA GPU. Zero value on
  Intel iGPU.
- **onnxruntime with OpenVINO EP** - already used for the Re-ID head.
  Reuse the same pattern for any additional side model.

## Tracking

- **BoT-SORT native** via `model.track(persist=True)` (**Q1**) - removes
  the homegrown tracker from the maintenance surface and stabilises
  every downstream layer.

## Infrastructure

- **`ultralytics >= 8.4`** - needed for both `model.track` and YOLOE.
- **Python 3.10+** - already stated as required; yt-dlp dropped 3.9.
- **`faiss-cpu`** (**M2**) - 2 MB library, worth it for >1k identity
  gallery.
- **Roboflow `supervision`** - replace hand-rolled `PolygonZone` /
  `LineZone` primitives once M5/L4 arrive.

---

# UX + marketing polish

The gap between the dashboard and a viral reel from
OpenViewer / AI Dev Guy is entirely **visual polish**, not detection
quality. These are the shots that push a demo from "developer preview"
to "product":

1. **Premium HUD sidebar** (**Q5**) - the screenshot-worthy layer.
2. **Attention chip per person** (**M11**) - pill above box with hover
   reasons. Must use neutral wording - "Attention", not "Suspicious".
3. **Auto-clip + evidence export** (**M6** + **Q6**) - turns "seen and
   lost" into "here is the mp4 with SHA-256".
4. **Red pulsing banner + top-bar alert chip** (**M3**, **M5**) -
   the OpenViewer / Actuate standard.
5. **Chart.js sparklines in every KPI tile** (**M7 + Q2**) - looks
   like analytics, not print statements.
6. **Privacy blur default-on** + `PRIVACY: ON` badge - ethical AND a
   maturity signal for enterprise buyers.
7. **Notebook Section 11 with A/B chart** (**L8**) - positions the
   YOLO26 choice as scientific instead of trend-following.
8. **Skill dropdown in dashboard header** (**L1**) - each new demo
   becomes a chip. Changes the narrative from "another layer" to "new
   plugin in the marketplace".

---

*Document generated from the 2026-08-17 multi-agent CV research pass
(35 subagents, 8 research strands, 56 candidate ideas, 10 verified
survivors). See the top of this file for methodology; see individual
entries for `Where` + `How` + `Deps`.*
