// Repo #2 single-camera live dashboard.
//
// One camera at a time, full viewport width. Picker at the top lists
// catalog cameras + uploaded MP4/MKV files; Upload button posts to
// /api/upload-video; Start reloads with ?cam=<id>. Everything else on
// the page (analysis picker, hot trail chip strip, gallery, review,
// model metrics) is unchanged from the legacy dashboard.
//
// No cloud subscriptions, no 4-tile grid, no cross-camera aggregation.

"use strict";

// ---------- Endpoints ------------------------------------------------------
const CATALOG_ENDPOINT   = "/api/catalog";
const UPLOADED_ENDPOINT  = "/api/uploaded-videos";
const UPLOAD_ENDPOINT    = "/api/upload-video";
const LOCAL_FILE_ENDPOINT = "/api/local-file";
const MODEL_VIEW_JSON    = (camId) => `/snapshots/model_view/${camId}.json`;

// ---------- Constants ------------------------------------------------------
const STALE_AGE_S = 120;

const ACTIVITY_BANDS = [
  { max: 0,   idx: 0  },
  { max: 2,   idx: 1  },
  { max: 5,   idx: 2  },
  { max: 8,   idx: 3  },
  { max: 12,  idx: 5  },
  { max: 18,  idx: 6  },
  { max: 25,  idx: 7  },
  { max: 35,  idx: 8  },
  { max: 50,  idx: 9  },
  { max: 1e9, idx: 10 },
];
const VEHICLE_LOAD_WEIGHTS = {
  car: 1.0, truck: 2.5, bus: 2.5, motorcycle: 0.5, bicycle: 0.3, train: 3.0,
};
const VEHICLE_BANDS = [
  { max: 0,   idx: 0  },
  { max: 1,   idx: 1  },
  { max: 3,   idx: 2  },
  { max: 5,   idx: 3  },
  { max: 8,   idx: 5  },
  { max: 12,  idx: 6  },
  { max: 18,  idx: 7  },
  { max: 26,  idx: 8  },
  { max: 38,  idx: 9  },
  { max: 1e9, idx: 10 },
];

// The single-tile state, one entry keyed by cam_id. Mirrors the shape of the
// legacy tileState so downstream helpers (draw loop, extrapolation, editors)
// keep working without renames.
const tileState = {};
let SINGLE_CAM_ID = null;
let SINGLE_CAM = null;      // {id, name, kind, ...}
const tilesEl = null;       // preserved name for stopTileAnalysis (unused here)

// ---------- Camera picker + upload ----------------------------------------

async function refreshCameraPicker() {
  const sel = document.getElementById("cam-picker");
  if (!sel) return;
  sel.innerHTML = "";
  let cams = [];
  try {
    const r = await fetch(CATALOG_ENDPOINT, { cache: "no-store" });
    if (r.ok) {
      const j = await r.json();
      cams = j.cameras || j.items || [];
    }
  } catch (_) { /* catalog absent, keep going */ }
  try {
    const r = await fetch(UPLOADED_ENDPOINT, { cache: "no-store" });
    if (r.ok) {
      const j = await r.json();
      for (const it of (j.items || [])) {
        cams.push({ id: it.cam_id || it.id, name: it.name,
                    kind: "local_file", path: it.path });
      }
    }
  } catch (_) { /* uploads absent */ }
  if (!cams.length) {
    const opt = document.createElement("option");
    opt.textContent = "(no cameras - upload a video or add to cameras.py)";
    opt.disabled = true;
    sel.appendChild(opt);
    return cams;
  }
  for (const c of cams) {
    const opt = document.createElement("option");
    opt.value = c.id || c.cam_id;
    opt.textContent = `${c.name || c.id} [${c.kind || "hls"}]`;
    opt.dataset.kind = c.kind || "hls";
    if (c.area) opt.dataset.area = c.area;
    sel.appendChild(opt);
  }
  const urlCam = new URLSearchParams(location.search).get("cam");
  if (urlCam && [...sel.options].some((o) => o.value === urlCam)) {
    sel.value = urlCam;
  }
  return cams;
}

function initUpload() {
  const btn = document.getElementById("upload-btn");
  const inp = document.getElementById("upload-input");
  if (!btn || !inp) return;
  btn.addEventListener("click", () => inp.click());
  inp.addEventListener("change", async () => {
    if (!inp.files || !inp.files[0]) return;
    const file = inp.files[0];
    const fd = new FormData();
    fd.append("video", file);
    btn.disabled = true;
    const origText = btn.textContent;
    btn.textContent = "Uploading...";
    try {
      const r = await fetch(UPLOAD_ENDPOINT, { method: "POST", body: fd });
      const j = await r.json();
      if (j.ok) {
        btn.textContent = "Uploaded";
        await refreshCameraPicker();
        const sel = document.getElementById("cam-picker");
        if (sel && j.cam) sel.value = j.cam.id || j.cam.cam_id;
      } else {
        alert("Upload failed: " + (j.error || "unknown"));
      }
    } catch (e) {
      alert("Upload error: " + e);
    } finally {
      setTimeout(() => {
        btn.textContent = origText;
        btn.disabled = false;
        inp.value = "";
      }, 1500);
    }
  });
}

function initStartCamera() {
  const btn = document.getElementById("start-cam");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const sel = document.getElementById("cam-picker");
    const camId = sel ? sel.value : null;
    if (!camId) return;
    const url = new URL(location.href);
    url.searchParams.set("cam", camId);
    location.href = url.toString();
  });
}

// ---------- Tile builder (single) -----------------------------------------

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function buildSingleTile(cam) {
  const host = document.getElementById("tile-single");
  if (!host) return null;
  host.innerHTML = `
    <div class="tile-head">
      <div class="tile-head-left">
        <h2 data-cam-name>${escapeHtml(cam.name || cam.id)}</h2>
        <div class="city" data-cam-area>${escapeHtml(cam.area || cam.kind || "")}</div>
      </div>
      <div class="tile-head-right">
        <button class="analyze-btn" data-analyze
                title="Live advanced analysis - pick one layer"
                style="cursor:pointer;border:1px solid #334155;background:#1e293b;color:#e2e8f0;border-radius:6px;padding:2px 8px;font-size:13px">Analyze</button>
        <span class="activity-badge act-unknown" data-activity>
          <span class="dot"></span><span data-activity-text>-/10</span>
        </span>
        <span class="anomaly-badge unk" data-anomaly title="no data yet">
          <span class="dot"></span><span data-anomaly-text>-</span>
        </span>
      </div>
    </div>
    <div class="video-wrap" data-video-wrap>
      <div class="video-overlay-bottom" data-overlay>
        <span class="kpi"><span class="lbl">People</span>
          <span class="val" data-k="person">-</span></span>
        <span class="kpi vehicles"><span class="lbl">Vehicles</span>
          <span class="val" data-k="vehicles">-</span></span>
        <span class="age" data-age title="age of the counts"></span>
      </div>
    </div>
  `;
  const st = {
    slot: { slot_id: cam.id, placeholder_name: cam.name || cam.id,
            display_area: cam.area || "" },
    cam,
    tile:          host,
    camNameEl:     host.querySelector("[data-cam-name]"),
    camAreaEl:     host.querySelector("[data-cam-area]"),
    videoWrap:     host.querySelector("[data-video-wrap]"),
    overlay:       host.querySelector("[data-overlay]"),
    latestVals:    host.querySelectorAll("[data-k]"),
    activityBadge: host.querySelector("[data-activity]"),
    activityText:  host.querySelector("[data-activity-text]"),
    anomalyBadge:  host.querySelector("[data-anomaly]"),
    anomalyText:   host.querySelector("[data-anomaly-text]"),
    ageEl:         host.querySelector("[data-age]"),
    history: [],
    lastSampleMs: null,
    currentHlsInstance: null,
    analysis: null,
  };
  tileState[cam.id] = st;
  host.querySelector("[data-analyze]").addEventListener("click",
    () => openAnalysisPicker(st));
  buildVideoInto(st, cam);
  return st;
}

function buildVideoInto(st, cam) {
  st.lastVideoBuild = { cam };
  if (st.currentHlsInstance) {
    try { st.currentHlsInstance.destroy(); } catch (_) {}
    st.currentHlsInstance = null;
  }
  for (const el of Array.from(st.videoWrap.children)) {
    if (el !== st.overlay) el.remove();
  }
  let markup = "";
  if (cam.kind === "local_file") {
    const src = `${LOCAL_FILE_ENDPOINT}?cam=${encodeURIComponent(cam.id)}`;
    markup = `<video src="${src}" controls muted autoplay playsinline
                     preload="auto"
                     controlsList="nodownload noremoteplayback"
                     style="width:100%;height:100%;object-fit:contain;background:#0c0e13"></video>`;
  } else if (cam.hls || cam.active_hls) {
    const hlsUrl = cam.hls || cam.active_hls;
    markup = `<video data-hls="${hlsUrl}" autoplay muted playsinline
                     controls controlsList="nodownload noremoteplayback"
                     preload="auto"></video>`;
  } else {
    markup = `<div class="video-fallback">
                Waiting for the analyzer to render the first frame -
                click Analyze above and pick a layer to start the live pipeline.
              </div>`;
  }
  st.videoWrap.insertAdjacentHTML("afterbegin", markup);
  const video = st.videoWrap.querySelector("video[data-hls]");
  if (video) attachHls(st, video);
}

function attachHls(st, video) {
  const src = video.dataset.hls;
  if (window.Hls && window.Hls.isSupported()) {
    const hls = new window.Hls({ lowLatencyMode: true, liveSyncDuration: 3 });
    hls.loadSource(src);
    hls.attachMedia(video);
    hls.on(window.Hls.Events.MANIFEST_PARSED, () => {
      const p = video.play();
      if (p && p.catch) p.catch(() => {});
    });
    hls.on(window.Hls.Events.ERROR, (_, data) => {
      if (!data.fatal) return;
      console.warn("hls.js fatal error on", src, data);
    });
    st.currentHlsInstance = hls;
  } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
    video.src = src;
    video.addEventListener("loadedmetadata", () => {
      const p = video.play();
      if (p && p.catch) p.catch(() => {});
    }, { once: true });
  } else {
    console.warn("No HLS playback support in this browser for", src);
  }
}

// ---------- Private-backend probe -----------------------------------------
// dashboard_server.py answers /api/ping with {private:true}; a hosted static
// copy has no server. The probe reveals the Analyze button.
let PRIVATE_BACKEND = false;
const _privateProbe = (async () => {
  try {
    const r = await fetch("/api/ping", { cache: "no-store" });
    if (r.ok) {
      const j = await r.json();
      PRIVATE_BACKEND = !!(j && j.private === true);
    }
  } catch (_) { PRIVATE_BACKEND = false; }
  document.body.classList.toggle("private-backend", PRIVATE_BACKEND);
  return PRIVATE_BACKEND;
})();

// ---------- Live advanced analysis ----------------------------------------
// One layer at a time. Picker morphs the tile in place: the video is
// overlaid by a canvas of analyzed frames of the SAME camera, polled from
// the local server. Analysis keeps the extrapolation/lerp helpers below so
// boxes glide with tracked objects between server ticks.
const ANALYSIS_LAYER_DEFS = [
  ["heat",     "Heat signature"],
  ["paths",    "Paths & speeds"],
  ["pose",     "Pose & skeleton"],
  ["gestures", "Static postures"],
  ["body",     "Body anomalies"],
  ["faces",    "Face detection"],
  ["line",     "Line crossing"],
  ["loiter",   "Zone & loitering"],
  ["parking",  "Parking occupancy"],
  ["plates",   "License plates (LPR)"],
];
const DRAWABLE_LAYERS = { line: "Draw line",
                          loiter: "Draw zones",
                          parking: "Draw spots" };
const ANALYSIS_POLL_MS = 500;

const analysisPanel = document.createElement("div");
analysisPanel.style.cssText =
  "display:none;position:fixed;inset:0;z-index:60;background:rgba(2,6,23,.72);" +
  "align-items:center;justify-content:center";
analysisPanel.innerHTML = `
  <div style="background:#0f172a;border:1px solid #334155;border-radius:12px;
              padding:20px 22px;max-width:440px;width:92%;color:#e2e8f0;
              font-size:15px">
    <h3 style="margin:0 0 4px;font-size:17px">Live analysis -
      <span data-an-cam></span></h3>
    <div style="color:#94a3b8;font-size:13px;margin-bottom:12px">
      Pick a layer to run. Only ONE layer at a time - switching swaps in
      place without restarting the stream (accumulators kept).</div>
    <div data-an-boxes style="display:grid;grid-template-columns:1fr 1fr;
         gap:8px 14px;margin-bottom:14px"></div>
    <div data-an-err style="color:#f87171;font-size:13px;min-height:18px"></div>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <button data-an-run style="cursor:pointer;background:#2563eb;border:0;
              color:#fff;border-radius:8px;padding:7px 18px;font-size:14px">
        Start</button>
      <button data-an-editline style="cursor:pointer;background:#334155;border:0;
              color:#fff;border-radius:8px;padding:7px 14px;font-size:14px;
              display:none">Edit counting line</button>
      <button data-an-cancel style="cursor:pointer;background:#1e293b;
              border:1px solid #334155;color:#e2e8f0;border-radius:8px;
              padding:7px 14px;font-size:14px">Cancel</button>
    </div>
  </div>`;
document.body.appendChild(analysisPanel);

analysisPanel.addEventListener("change", (e) => {
  if (e.target && e.target.name === "an-layer") {
    const el = analysisPanel.querySelector("[data-an-editline]");
    el.style.display = (e.target.value === "line") ? "" : "none";
  }
});

analysisPanel.querySelector("[data-an-editline]").addEventListener("click",
  () => {
    if (!_anTarget) return;
    const cam = tileAnalysisCamId(_anTarget);
    if (!cam) { alert("No active camera yet"); return; }
    analysisPanel.style.display = "none";
    window.openLineEditor(cam,
      `/api/analysis/frame?cam=${encodeURIComponent(cam)}&_=${Date.now()}`);
  });

const _anBoxes = analysisPanel.querySelector("[data-an-boxes]");
for (const [key, label] of ANALYSIS_LAYER_DEFS) {
  const lab = document.createElement("label");
  lab.style.cssText = "display:flex;gap:7px;align-items:center;cursor:pointer";
  lab.innerHTML = `<input type="radio" name="an-layer" value="${key}"> ${label}`;
  _anBoxes.appendChild(lab);
}

let _anTarget = null;

// Client-side singleton guard: only one analysis session at a time. Repo #2
// has one tile, so this collapses to "no other running session on the same
// tile" - kept for symmetry with the multi-tile edition.
function _findActiveAnalysisTile(exceptSt) {
  for (const key of Object.keys(tileState)) {
    const other = tileState[key];
    if (other !== exceptSt && other.analysis) return other;
  }
  return null;
}

function tileAnalysisCamId(st) {
  return st && st.cam ? st.cam.id : null;
}

function openAnalysisPicker(st) {
  _anTarget = st;
  analysisPanel.querySelector("[data-an-cam]").textContent =
    st.camNameEl.textContent || st.cam.id;
  const errEl = analysisPanel.querySelector("[data-an-err]");
  const runBtn = analysisPanel.querySelector("[data-an-run]");
  const editBtn = analysisPanel.querySelector("[data-an-editline]");
  const otherActive = _findActiveAnalysisTile(st);
  if (otherActive) {
    const otherName =
      otherActive.camNameEl.textContent || otherActive.slot.slot_id;
    errEl.style.color = "#fbbf24";
    errEl.textContent =
      `Advanced analysis is already running on "${otherName}". ` +
      `Only ONE camera at a time - stop it first, then start here.`;
    for (const rb of _anBoxes.querySelectorAll("input")) {
      rb.checked = false;
      rb.disabled = true;
    }
    runBtn.disabled = true;
    editBtn.style.display = "none";
    analysisPanel.style.display = "flex";
    return;
  }
  errEl.style.color = "#f87171";
  errEl.textContent = "";
  for (const rb of _anBoxes.querySelectorAll("input")) rb.disabled = false;
  runBtn.disabled = false;
  const current = st.analysis ? st.analysis.layer : null;
  for (const rb of _anBoxes.querySelectorAll("input"))
    rb.checked = rb.value === current;
  runBtn.textContent = st.analysis ? "Switch layer" : "Start";
  editBtn.style.display = (current === "line") ? "" : "none";
  analysisPanel.style.display = "flex";
}

analysisPanel.querySelector("[data-an-cancel]").addEventListener("click",
  () => { analysisPanel.style.display = "none"; });

analysisPanel.querySelector("[data-an-run]").addEventListener("click",
  async () => {
    const errEl = analysisPanel.querySelector("[data-an-err]");
    const picked = _anBoxes.querySelector("input:checked");
    if (!picked) {
      errEl.textContent = "Pick a layer";
      return;
    }
    const st = _anTarget;
    const cam = st && tileAnalysisCamId(st);
    if (!cam) {
      errEl.textContent = "Camera not identified yet";
      return;
    }
    const otherActive = _findActiveAnalysisTile(st);
    if (otherActive && !st.analysis) {
      errEl.style.color = "#fbbf24";
      errEl.textContent = "Blocked - stop the other analysis first";
      return;
    }
    const runBtn = analysisPanel.querySelector("[data-an-run]");
    runBtn.disabled = true;
    errEl.style.color = "#f87171";
    errEl.textContent = "";
    try {
      const r = await fetch(
        `/api/analysis/start?cam=${encodeURIComponent(cam)}` +
        `&layer=${encodeURIComponent(picked.value)}`,
        { method: "POST" });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || r.status);
      if (data.layer && data.layer !== picked.value) {
        await new Promise((res) => setTimeout(res, 300));
        const r2 = await fetch(
          `/api/analysis/start?cam=${encodeURIComponent(cam)}` +
          `&layer=${encodeURIComponent(picked.value)}`,
          { method: "POST" });
        const d2 = await r2.json();
        if (!r2.ok || (d2.layer && d2.layer !== picked.value))
          throw new Error("layer switch did not take - try again");
      }
      beginTileAnalysis(st, cam, picked.value);
      analysisPanel.style.display = "none";
      setTimeout(async () => {
        try {
          const chk = await fetch(
            `/api/analysis/data?cam=${encodeURIComponent(cam)}`)
            .then((x) => x.status === 200 ? x.json() : null);
          if (chk && chk.layer && chk.layer !== picked.value) {
            await fetch(
              `/api/analysis/start?cam=${encodeURIComponent(cam)}` +
              `&layer=${encodeURIComponent(picked.value)}`,
              { method: "POST" });
          }
        } catch (_) {}
      }, 1200);
    } catch (e) {
      errEl.textContent = "Failed to start: " + e.message;
    } finally {
      runBtn.disabled = false;
    }
  });

const _layerLabel = Object.fromEntries(ANALYSIS_LAYER_DEFS);

function beginTileAnalysis(st, cam, layer) {
  if (st.analysis) {
    st.analysis.layer = layer;
    st.analysis.tickBuf.length = 0;
    const tag = st.videoWrap.querySelector(".analysis-live-tag");
    if (tag) tag.textContent = `LIVE ANALYSIS - ${_layerLabel[layer] || layer}`;
    const lb = st.videoWrap.querySelector(".analysis-drawline");
    if (lb) {
      lb.style.display = DRAWABLE_LAYERS[layer] ? "" : "none";
      if (DRAWABLE_LAYERS[layer]) lb.textContent = DRAWABLE_LAYERS[layer];
    }
    return;
  }
  st._overlayWasHidden = st.overlay.style.display === "none";
  st.overlay.style.display = "none";
  const wrap = document.createElement("div");
  wrap.className = "analysis-wrap analysis-overlay-mode";
  wrap.style.cssText = "position:absolute;inset:0;pointer-events:none;"
                     + "z-index:4;background:transparent;";
  wrap.innerHTML = `
    <img class="analysis-bg" alt="" draggable="false"
         style="position:absolute;inset:0;width:100%;height:100%;
                object-fit:contain;background:#0f172a;display:block;">
    <canvas class="analysis-canvas"
            style="position:absolute;inset:0;width:100%;height:100%;
                   pointer-events:none;background:transparent;"></canvas>
    <div class="analysis-status"
         style="position:absolute;left:8px;top:8px;padding:4px 10px;
                background:rgba(15,23,42,0.85);color:#e2e8f0;border-radius:6px;
                font-size:12px;pointer-events:none;">starting live analysis...</div>
    <span class="analysis-live-tag"
          style="position:absolute;right:8px;top:8px;padding:4px 10px;
                 background:rgba(37,99,235,0.9);color:#f8fafc;border-radius:6px;
                 font-size:12px;font-weight:600;pointer-events:none;">LIVE -
      ${escapeHtml(_layerLabel[layer] || layer)}</span>
    <button class="analysis-drawline"
            style="position:absolute;right:78px;bottom:8px;padding:6px 12px;
                   background:#2563eb;color:#f8fafc;border:0;border-radius:6px;
                   cursor:pointer;font-size:13px;pointer-events:auto;
                   display:${DRAWABLE_LAYERS[layer] ? "" : "none"};">
      ${DRAWABLE_LAYERS[layer] || "Draw"}</button>
    <button class="analysis-stop"
            style="position:absolute;right:8px;bottom:8px;padding:6px 12px;
                   background:#dc2626;color:#f8fafc;border:0;border-radius:6px;
                   cursor:pointer;font-size:13px;pointer-events:auto;">
      Stop</button>`;
  st.videoWrap.style.position = st.videoWrap.style.position || "relative";
  st.videoWrap.appendChild(wrap);
  wrap.querySelector(".analysis-stop").addEventListener("click",
    () => stopTileAnalysis(st));
  wrap.querySelector(".analysis-drawline").addEventListener("click", () => {
    const snap =
      `/api/analysis/frame?cam=${encodeURIComponent(cam)}&_=${Date.now()}`;
    const lay = st.analysis ? st.analysis.layer : layer;
    if (lay === "line") window.openLineEditor(cam, snap);
    else openZoneEditor(cam, lay, snap);
  });
  st.analysis = {
    cam, layer,
    wrap,
    bg: wrap.querySelector(".analysis-bg"),
    canvas: wrap.querySelector(".analysis-canvas"),
    status: wrap.querySelector(".analysis-status"),
    lastBgUrl: null,
    tickBuf: [],
    failures: 0, lastRestart: 0, inflight: false, lastSeq: -1,
    evSeen: new Set(),
    evTimer: setInterval(() => pollAnalysisEvents(st), 2500),
    timer: setInterval(() => pollAnalysisFrame(st), ANALYSIS_POLL_MS),
    videoStateTimer: setInterval(() => _syncAnalysisBgVisibility(st), 500),
  };
  st.analysis.bg.style.display = "block";
  st.analysis.canvas.style.display = "none";
  const evStrip = document.createElement("div");
  evStrip.className = "events-strip";
  evStrip.style.cssText =
    "display:flex;gap:6px;align-items:stretch;overflow-x:auto;" +
    "padding:6px 4px;background:#0b1220;border-radius:6px;margin-top:6px;" +
    "min-height:76px;scrollbar-width:thin;";
  evStrip.innerHTML = `<button class="events-saved-btn" style="flex:0 0 auto;
      background:#1e293b;color:#94a3b8;border:0;border-radius:6px;
      padding:0 10px;cursor:pointer;font-size:12px">Saved</button>
    <div class="events-empty" style="color:#475569;font-size:12px;
      align-self:center;padding:0 8px">detections will appear here...</div>`;
  evStrip.querySelector(".events-saved-btn")
    .addEventListener("click", openSavedDetections);
  st.videoWrap.insertAdjacentElement("afterend", evStrip);
  st.analysis.evStrip = evStrip;
  pollAnalysisFrame(st);
  pollAnalysisEvents(st);
  _refreshAnalysisBg(st.analysis);
  _analysisDrawLoop(st, st.analysis);
}

async function pollAnalysisEvents(st) {
  const a = st.analysis;
  if (!a || !a.evStrip) return;
  let d;
  try {
    const r = await fetch(
      `/api/analysis/events?cam=${encodeURIComponent(a.cam)}`);
    if (!r.ok) return;
    d = await r.json();
  } catch (_) { return; }
  if (!st.analysis || st.analysis !== a) return;
  const evs = d.events || [];
  if (evs.length) {
    const empty = a.evStrip.querySelector(".events-empty");
    if (empty) empty.remove();
  }
  const anchor = a.evStrip.querySelector(".events-saved-btn");
  for (let i = evs.length - 1; i >= 0; i--) {
    const ev = evs[i];
    if (a.evSeen.has(ev.id)) continue;
    a.evSeen.add(ev.id);
    anchor.insertAdjacentElement("afterend", _eventChip(a, ev));
  }
  const chips = a.evStrip.querySelectorAll(".event-chip");
  for (let i = 30; i < chips.length; i++) chips[i].remove();
  const cur = a.actualLayer || a.layer;
  for (const c of a.evStrip.querySelectorAll(".event-chip")) {
    c.style.display = (!c.dataset.layer || c.dataset.layer === cur)
      ? "" : "none";
  }
}

function _eventChip(a, ev) {
  const chip = document.createElement("div");
  chip.className = "event-chip";
  chip.dataset.layer = ev.layer || "";
  chip.style.cssText =
    "flex:0 0 auto;display:flex;gap:6px;align-items:center;" +
    "background:#111a2e;border:1px solid #1e293b;border-radius:6px;" +
    "padding:4px 6px;max-width:250px;";
  const t = new Date(ev.ts * 1000);
  const pad = (n) => String(n).padStart(2, "0");
  chip.innerHTML = `
    <img src="data:image/jpeg;base64,${ev.thumb}" alt=""
         style="height:56px;border-radius:4px;flex:0 0 auto">
    <div style="min-width:0">
      <div style="font-size:11px;color:#e2e8f0;white-space:nowrap;
                  overflow:hidden;text-overflow:ellipsis">${escapeHtml(ev.text)}</div>
      <div style="font-size:10px;color:#64748b">${pad(t.getHours())}:${pad(t.getMinutes())}:${pad(t.getSeconds())}</div>
    </div>
    <button class="event-save" title="save this detection for later study"
            style="background:#1d4ed8;color:#fff;border:0;border-radius:5px;
                   padding:4px 7px;cursor:pointer;font-size:11px;flex:0 0 auto">
      ${ev.saved ? "saved" : "save"}</button>`;
  const btn = chip.querySelector(".event-save");
  if (ev.saved) btn.disabled = true;
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    try {
      const r = await fetch(
        `/api/analysis/event/save?cam=${encodeURIComponent(a.cam)}` +
        `&id=${encodeURIComponent(ev.id)}`, { method: "POST" });
      btn.textContent = r.ok ? "saved" : "err";
      if (!r.ok) btn.disabled = false;
    } catch (_) { btn.textContent = "err"; btn.disabled = false; }
  });
  return chip;
}

async function openSavedDetections() {
  let items = [];
  try {
    const r = await fetch("/api/analysis/saved");
    items = (await r.json()).items || [];
  } catch (_) {}
  const bg = document.createElement("div");
  bg.style.cssText = "position:fixed;inset:0;background:rgba(2,6,23,0.8);" +
    "z-index:60;display:flex;align-items:center;justify-content:center";
  const box = document.createElement("div");
  box.style.cssText = "background:#0f172a;border:1px solid #1e293b;" +
    "border-radius:10px;max-width:820px;width:92%;max-height:80vh;" +
    "overflow:auto;padding:16px";
  box.innerHTML = `<div style="display:flex;justify-content:space-between;
      align-items:center;margin-bottom:10px">
      <b style="color:#e2e8f0">Saved detections (${items.length})</b>
      <button class="saved-close" style="background:#1e293b;color:#94a3b8;
        border:0;border-radius:6px;padding:6px 12px;cursor:pointer">close</button>
    </div>` + (items.length ? "" :
    `<div style="color:#64748b;font-size:13px">nothing saved yet - use the
     save button on a detection chip</div>`);
  for (const it of items) {
    const row = document.createElement("div");
    row.style.cssText = "display:flex;gap:10px;align-items:center;" +
      "border-top:1px solid #1e293b;padding:8px 0";
    const t = new Date(it.ts * 1000);
    row.innerHTML = `
      <a href="${it.image}" target="_blank" rel="noopener">
        <img src="${it.image}" alt="" style="height:64px;border-radius:6px"></a>
      <div style="min-width:0">
        <div style="color:#e2e8f0;font-size:13px">${escapeHtml(it.text)}</div>
        <div style="color:#64748b;font-size:11px">${escapeHtml(it.cam_name || it.cam)}
          - ${escapeHtml(it.layer)} - ${t.toLocaleString()}</div>
      </div>`;
    box.appendChild(row);
  }
  bg.appendChild(box);
  bg.addEventListener("click", (e) => { if (e.target === bg) bg.remove(); });
  box.querySelector(".saved-close").addEventListener("click",
    () => bg.remove());
  document.body.appendChild(bg);
}

async function renderGallery() {
  const wrap = document.getElementById("gallery-wrap");
  const title = document.getElementById("gallery-title");
  if (!wrap) return;
  let items = [];
  try {
    const r = await fetch("/api/analysis/saved", { cache: "no-store" });
    items = (await r.json()).items || [];
  } catch (_) { return; }
  if (title) title.textContent =
    `Detections gallery - ${items.length} saved sample(s)`;
  if (!items.length) {
    wrap.innerHTML = `<div class="sub">nothing saved yet - press save on a
      live detection chip, or let the proof collector fill this up.</div>`;
    return;
  }
  const order = ["plates", "line", "loiter", "parking", "gestures",
                 "body", "pose", "faces", "heat", "paths"];
  items.sort((a, b) => order.indexOf(a.layer) - order.indexOf(b.layer)
                       || (b.ts || 0) - (a.ts || 0));
  wrap.innerHTML = items.map((it) => {
    const t = new Date((it.ts || 0) * 1000);
    return `<figure style="margin:0;background:#0c0e13;border:1px solid
        #232733;border-radius:8px;overflow:hidden">
      <a href="${it.image}" target="_blank" rel="noopener">
        <img src="${it.image}" alt="" loading="lazy"
             style="width:100%;height:130px;object-fit:cover;display:block"></a>
      <figcaption style="padding:6px 8px">
        <div style="font-size:11px;color:#e7e9ee;white-space:nowrap;
             overflow:hidden;text-overflow:ellipsis">${escapeHtml(it.text)}</div>
        <div style="font-size:10px;color:#8b909a">${escapeHtml(it.layer)} -
          ${escapeHtml(it.cam_name || it.cam)} - ${t.toLocaleTimeString()}</div>
      </figcaption>
    </figure>`;
  }).join("");
}
renderGallery();
setInterval(renderGallery, 20000);

// ---------- Draw loop + extrapolation -------------------------------------

function _analysisDrawLoop(st, a) {
  if (!st.analysis || st.analysis !== a) return;
  requestAnimationFrame(() => _analysisDrawLoop(st, a));
  const nowMs = performance.now();
  if (a._lastDrawMs && nowMs - a._lastDrawMs < 66) return;
  a._lastDrawMs = nowMs;
  if (a.canvas.style.display === "none") return;
  const buf = a.tickBuf;
  if (!buf.length) return;
  let merged = buf[buf.length - 1];
  const hls = st.currentHlsInstance;
  const pd = hls && hls.playingDate;
  let vidT = null;
  if (pd instanceof Date && !isNaN(pd)) {
    vidT = pd.getTime() / 1000;
  } else {
    const video = st.videoWrap.querySelector("video");
    if (video && video.currentTime > 0) {
      const newest = buf[buf.length - 1];
      if (!a._vPin) a._vPin = { ct: video.currentTime,
                                at: newest ? (newest.at || Date.now() / 1000)
                                           : Date.now() / 1000 };
      vidT = a._vPin.at + (video.currentTime - a._vPin.ct);
    }
  }
  const EXTRAP_MAX_S = 1.5;
  const EXTRAP_MAX_DIAG = 0.5;
  const _capShift = (b, dt) => {
    if (b.coast || !dt || dt <= 0) return _shiftBox(b, 0);
    const diag = Math.hypot(b.x2 - b.x1, b.y2 - b.y1) || 1;
    const disp = Math.hypot((b.vx || 0) * dt, (b.vy || 0) * dt);
    const cap = EXTRAP_MAX_DIAG * diag;
    const useDt = disp > cap && disp > 0 ? dt * (cap / disp) : dt;
    return _shiftBox(b, Math.min(useDt, EXTRAP_MAX_S));
  };
  if (vidT != null) {
    let i = buf.length - 1;
    while (i > 0 && (buf[i].at || 0) > vidT + 0.25) i--;
    const d = buf[i];
    const nxt = i + 1 < buf.length ? buf[i + 1] : null;
    const t0 = d.at || vidT;
    let fade = 1;
    const boxes = [];
    if (nxt && (nxt.at || 0) > t0 + 0.05) {
      const al = Math.max(0, Math.min(1, (vidT - t0) / ((nxt.at || 0) - t0)));
      const byTid = new Map();
      for (const nb of nxt.boxes || []) {
        if (nb.tid !== undefined) byTid.set(nb.tid, nb);
      }
      for (const b of d.boxes || []) {
        const nb = b.tid !== undefined ? byTid.get(b.tid) : null;
        boxes.push(nb ? _lerpBox(b, nb, al) : _capShift(b, vidT - t0));
      }
    } else {
      const rawDt = Math.max(0, vidT - t0);
      if (rawDt <= EXTRAP_MAX_S) {
        for (const b of d.boxes || []) boxes.push(_capShift(b, rawDt));
        if (rawDt > EXTRAP_MAX_S * 0.7) {
          fade = 1 - 0.5 * (rawDt - EXTRAP_MAX_S * 0.7)
                     / (EXTRAP_MAX_S * 0.3);
        }
      }
    }
    merged = Object.assign({}, d, { boxes, _fade: fade });
  }
  _drawAnalysisOverlay(a.canvas, merged, 0);
}

function _lerpBox(b, nb, al) {
  const o = Object.assign({}, nb);
  o.x1 = b.x1 + (nb.x1 - b.x1) * al;
  o.y1 = b.y1 + (nb.y1 - b.y1) * al;
  o.x2 = b.x2 + (nb.x2 - b.x2) * al;
  o.y2 = b.y2 + (nb.y2 - b.y2) * al;
  const dx = (o.x1 + o.x2 - b.x1 - b.x2) / 2;
  const dy = (o.y1 + o.y2 - b.y1 - b.y2) / 2;
  if (b.kps && nb.kps && nb.kps.length === b.kps.length) {
    o.kps = b.kps.map((k, j) => [k[0] + (nb.kps[j][0] - k[0]) * al,
                                 k[1] + (nb.kps[j][1] - k[1]) * al,
                                 Math.min(k[2], nb.kps[j][2])]);
  } else if (b.kps) {
    o.kps = b.kps.map((k) => [k[0] + dx, k[1] + dy, k[2]]);
  }
  if (b.trail) o.trail = b.trail;
  o.vx = 0; o.vy = 0;
  return o;
}

function _shiftBox(b, dt) {
  const dx = (b.vx || 0) * dt, dy = (b.vy || 0) * dt;
  const o = Object.assign({}, b);
  o.x1 = b.x1 + dx; o.y1 = b.y1 + dy;
  o.x2 = b.x2 + dx; o.y2 = b.y2 + dy;
  if (b.kps) o.kps = b.kps.map((k) => [k[0] + dx, k[1] + dy, k[2]]);
  o.vx = 0; o.vy = 0;
  return o;
}

function _syncAnalysisBgVisibility(st) {
  const a = st.analysis;
  if (!a || !a.bg) return;
  let playing = false;
  const v = st.videoWrap.querySelector("video");
  if (v && !v.paused && !v.ended && v.readyState >= 2) {
    const t = v.currentTime;
    playing = (a._lastVidT !== undefined) && (t > a._lastVidT + 0.05);
    a._lastVidT = t;
  } else {
    a._lastVidT = undefined;
  }
  const want = playing ? "none" : "block";
  if (a.bg.style.display !== want) {
    a.bg.style.display = want;
    a.canvas.style.display = playing ? "" : "none";
    if (want === "block") _refreshAnalysisBg(a);
  }
}

async function pollAnalysisFrame(st) {
  const a = st.analysis;
  if (!a || a.inflight) return;
  a.inflight = true;
  try {
    const r = await fetch(
      `/api/analysis/data?cam=${encodeURIComponent(a.cam)}&_=${Date.now()}`,
      { cache: "no-store" });
    if (r.status === 200) {
      const d = await r.json();
      a.actualLayer = d.layer;
      const liveTag = st.videoWrap.querySelector(".analysis-live-tag");
      if (d.layer === a.layer) {
        if (liveTag) liveTag.textContent =
          `LIVE - ${_layerLabel[d.layer] || d.layer}`;
      } else {
        if (liveTag) liveTag.textContent =
          `switching to ${_layerLabel[a.layer] || a.layer}...`;
        if (Date.now() - (a._switchPost || 0) > 4000) {
          a._switchPost = Date.now();
          fetch(`/api/analysis/start?cam=${encodeURIComponent(a.cam)}`
                + `&layer=${encodeURIComponent(a.layer)}`,
                { method: "POST" }).catch(() => {});
        }
      }
      if (d.seq !== a.lastSeq) {
        a.lastSeq = d.seq;
        const prevTick = a.tickBuf[a.tickBuf.length - 1];
        if (prevTick) {
          const gap = (d.at || 0) - (prevTick.at || 0);
          if (gap > 0.2 && gap < 60) {
            a._gapEma = a._gapEma ? 0.7 * a._gapEma + 0.3 * gap : gap;
          }
        }
        a.tickBuf.push(d);
        if (a.tickBuf.length > 24) a.tickBuf.shift();
        if (a.bg && a.bg.style.display !== "none") _refreshAnalysisBg(a);
        if (d.layer === "loiter" && Array.isArray(d.zones)) {
          a._loiterAlerted = a._loiterAlerted || new Set();
          for (const z of d.zones) {
            if (z.alert && !a._loiterAlerted.has(z.name)) {
              a._loiterAlerted.add(z.name);
              showCrossToast(`loitering in ${z.name} - ${z.max_dwell}s`);
              const t = st.tile;
              if (t) {
                t.style.outline = "3px solid #ef4444";
                setTimeout(() => { t.style.outline = ""; }, 2500);
              }
            } else if (!z.alert) {
              a._loiterAlerted.delete(z.name);
            }
          }
        }
      }
      a.status.style.display = "none";
      a.failures = 0;
    } else if (r.status === 202) {
      const j = await r.json();
      a.status.style.display = "";
      a.status.textContent = j.note || "starting...";
    } else if (r.status === 404) {
      a.failures += 1;
      if (Date.now() - a.lastRestart > 5000) {
        a.lastRestart = Date.now();
        a.status.style.display = "";
        a.status.textContent = "analysis session ended - restarting...";
        fetch(`/api/analysis/start?cam=${encodeURIComponent(a.cam)}`
              + `&layer=${encodeURIComponent(a.layer)}`,
              { method: "POST" }).catch(() => {});
      }
    } else if (r.status === 410) {
      let reason = "";
      try { reason = (await r.json()).error || ""; } catch (_) {}
      a.status.style.display = "";
      a.status.textContent = "analysis ended"
        + (reason ? ` - ${reason}` : "") + " - pick a layer to restart";
      a.failures += 1;
    } else {
      a.failures += 1;
    }
  } catch (_) {
    a.failures += 1;
  } finally {
    a.inflight = false;
  }
  if (a.failures > 8) {
    a.status.style.display = "";
    a.status.textContent =
      "analysis unreachable - press Stop to return to video";
  }
}

function _drawAnalysisOverlay(canvas, d, dtExtra = 0) {
  const nowMs = performance.now();
  let m = canvas._sizeCache;
  if (!m || nowMs - m.t > 1500) {
    const rect = canvas.parentElement.getBoundingClientRect();
    m = canvas._sizeCache = {
      t: nowMs,
      cw: Math.max(1, Math.round(rect.width)),
      ch: Math.max(1, Math.round(rect.height)),
    };
  }
  const cw = m.cw, ch = m.ch;
  if (canvas.width !== cw)  canvas.width = cw;
  if (canvas.height !== ch) canvas.height = ch;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, cw, ch);
  const fw = Math.max(1, Number(d.frame_w) || cw);
  const fh = Math.max(1, Number(d.frame_h) || ch);
  const sx = cw / fw, sy = ch / fh;

  if (d.layer === "heat" && Array.isArray(d.heat) && d.heat.length) {
    let hc = canvas._heatCache;
    if (!hc || hc.seq !== d.seq || hc.cw !== cw || hc.ch !== ch) {
      const off = document.createElement("canvas");
      off.width = cw; off.height = ch;
      const octx = off.getContext("2d");
      const gh = d.heat.length, gw = d.heat[0].length;
      const cellW = cw / gw, cellH = ch / gh;
      const vals = [];
      for (const row of d.heat) for (const v of row) if (v > 0) vals.push(v);
      vals.sort((a, b) => a - b);
      const peak = vals.length
        ? vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.99))]
        : 0;
      if (peak > 0) {
        for (let gy = 0; gy < gh; gy++) {
          for (let gx = 0; gx < gw; gx++) {
            const v = Math.min(1, d.heat[gy][gx] / peak);
            if (v < 0.05) continue;
            const alpha = Math.min(0.65, v * 0.7);
            octx.fillStyle = _heatColor(v, alpha);
            octx.fillRect(gx * cellW, gy * cellH, cellW + 1, cellH + 1);
          }
        }
      }
      hc = canvas._heatCache = { seq: d.seq, cw, ch, off };
    }
    ctx.drawImage(hc.off, 0, 0);
  }

  if (d.layer === "line" && Array.isArray(d.line) && d.line.length === 2) {
    const norm = d.line.every((p) => p[0] <= 1.001 && p[1] <= 1.001);
    const lx = (p) => norm ? p[0] * cw : p[0] * sx;
    const ly = (p) => norm ? p[1] * ch : p[1] * sy;
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(59,130,246,0.95)";
    ctx.setLineDash([10, 6]);
    ctx.beginPath();
    ctx.moveTo(lx(d.line[0]), ly(d.line[0]));
    ctx.lineTo(lx(d.line[1]), ly(d.line[1]));
    ctx.stroke();
    ctx.setLineDash([]);
    if (d.cross) {
      ctx.fillStyle = "rgba(15,23,42,0.85)";
      ctx.fillRect(8, ch - 30, 150, 22);
      ctx.fillStyle = "#f8fafc";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText(`in: ${d.cross.in || 0}   out: ${d.cross.out || 0}`,
                   14, ch - 14);
    }
  }

  if ((d.layer === "loiter" && Array.isArray(d.zones))
      || (d.layer === "parking" && Array.isArray(d.spots))) {
    const entries = d.layer === "loiter" ? d.zones : d.spots;
    ctx.font = "12px system-ui, sans-serif";
    for (const z of entries) {
      const hot = d.layer === "loiter" ? z.alert : z.occupied;
      const col = hot ? "239,68,68" : "74,222,128";
      ctx.beginPath();
      ctx.moveTo(z.points[0][0] * cw, z.points[0][1] * ch);
      for (let i = 1; i < z.points.length; i++)
        ctx.lineTo(z.points[i][0] * cw, z.points[i][1] * ch);
      ctx.closePath();
      ctx.fillStyle = `rgba(${col},${hot ? 0.22 : 0.12})`;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = `rgba(${col},0.95)`;
      ctx.stroke();
      const label = d.layer === "loiter"
        ? `${z.name}: ${z.count} inside - max ${z.max_dwell}s`
        : `${z.name}: ${z.occupied ? (z.cls || "occupied") : "free"}`;
      const zx = z.points[0][0] * cw, zy = z.points[0][1] * ch;
      const tw = ctx.measureText(label).width + 8;
      ctx.fillStyle = "rgba(15,23,42,0.85)";
      ctx.fillRect(zx, Math.max(0, zy - 16), tw, 16);
      ctx.fillStyle = "#f8fafc";
      ctx.fillText(label, zx + 4, Math.max(12, zy - 4));
    }
    if (d.layer === "parking" && d.parking) {
      const t = `parking: ${d.parking.occupied}/${d.parking.total} occupied`;
      ctx.fillStyle = "rgba(15,23,42,0.85)";
      ctx.fillRect(8, ch - 30, ctx.measureText(t).width + 14, 22);
      ctx.fillStyle = "#f8fafc";
      ctx.fillText(t, 14, ch - 14);
    }
    if (!entries.length) {
      const t = "no zones drawn yet - press the Draw button";
      ctx.fillStyle = "rgba(15,23,42,0.85)";
      ctx.fillRect(8, 8, ctx.measureText(t).width + 14, 22);
      ctx.fillStyle = "#fbbf24";
      ctx.fillText(t, 14, 23);
    }
  }

  if (d.layer === "paths") {
    for (const b of d.boxes || []) {
      if (!Array.isArray(b.trail) || b.trail.length < 2) continue;
      ctx.lineWidth = 2;
      ctx.strokeStyle = _trailColor(b.tid || 0);
      ctx.beginPath();
      ctx.moveTo(b.trail[0][0] * sx, b.trail[0][1] * sy);
      for (let i = 1; i < b.trail.length; i++)
        ctx.lineTo(b.trail[i][0] * sx, b.trail[i][1] * sy);
      ctx.lineTo((b.x1 + b.x2) / 2 * sx + (b.vx || 0) * dtExtra * sx,
                 (b.y1 + b.y2) / 2 * sy + (b.vy || 0) * dtExtra * sy);
      ctx.stroke();
    }
  }

  if (d.layer === "faces") {
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(250,204,21,0.95)";
    for (const f of d.faces || [])
      ctx.strokeRect(f.x1 * sx, f.y1 * sy,
                     (f.x2 - f.x1) * sx, (f.y2 - f.y1) * sy);
    if (d.faces_ok === false) {
      ctx.fillStyle = "rgba(15,23,42,0.85)";
      ctx.fillRect(8, 8, 220, 22);
      ctx.fillStyle = "#fbbf24";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText("face backend unavailable", 14, 23);
    }
  }

  ctx.font = "12px system-ui, sans-serif";
  const boxAlpha = Number(d._fade) || 1;
  if (boxAlpha < 1) ctx.globalAlpha = boxAlpha;
  let alertOn = false;
  for (const b of d.boxes || []) {
    if (d.layer === "faces") continue;
    if (d.layer === "heat") break;
    const isPose = (d.layer === "pose" || d.layer === "gestures");
    if (isPose && b.cls !== "person") continue;
    if (d.layer === "body" && !b.flag) continue;
    if (d.layer === "gestures" && !b.gestures && !b.kps) continue;
    if (d.layer === "plates" && b.cls === "person") continue;
    const ox = (b.vx || 0) * dtExtra, oy = (b.vy || 0) * dtExtra;
    const x = (b.x1 + ox) * sx, y = (b.y1 + oy) * sy;
    const w = (b.x2 - b.x1) * sx, h = (b.y2 - b.y1) * sy;
    if (x + w < 0 || y + h < 0 || x > cw || y > ch) continue;
    let color = b.cls === "person"
      ? "rgba(74,222,128,0.95)" : "rgba(251,146,60,0.95)";
    let label = `${b.cls} ${Math.round((b.conf || 0) * 100)}%`;
    if (d.layer === "paths" && b.tier)
      label += ` - ${b.tier}`;
    if (d.layer === "gestures" && b.gestures)
      label = `#${b.tid} ${b.gestures.join("+")}`;
    if (d.layer === "body" && b.flag) {
      color = b.alert ? "rgba(239,68,68,0.95)" : "rgba(234,140,8,0.95)";
      label = `#${b.tid} ${String(b.flag).toUpperCase()}`
        + (b.flags ? " " + b.flags.join("+") : "");
      if (b.alert) alertOn = true;
    }
    if (d.layer === "loiter" && b.dwell != null) {
      label += ` - ${b.dwell}s in zone`;
      if ((d.zones || []).some((z) => z.alert)) color = "rgba(239,68,68,0.95)";
    }
    if (d.layer === "plates" && b.plate) {
      color = "rgba(74,222,128,0.95)";
      label = `${b.plate} - ${Math.round((b.plate_conf || 0) * 100)}%`;
    }
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    if (b.coast) ctx.setLineDash([6, 5]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    if (b.kps) _drawSkeleton(ctx, b.kps, sx, sy, ox, oy);
    const tw = ctx.measureText(label).width + 8;
    ctx.fillStyle = "rgba(15,23,42,0.85)";
    ctx.fillRect(x, Math.max(0, y - 16), tw, 16);
    ctx.fillStyle = "#f8fafc";
    ctx.fillText(label, x + 4, Math.max(12, y - 4));
  }
  if (boxAlpha < 1) ctx.globalAlpha = 1;

  if (d.envelope) {
    ctx.font = "11px system-ui, sans-serif";
    const tw = ctx.measureText(d.envelope).width + 14;
    ctx.fillStyle = "rgba(15,23,42,0.8)";
    ctx.fillRect(8, 8, tw, 20);
    ctx.fillStyle = "#94a3b8";
    ctx.fillText(d.envelope, 15, 22);
    ctx.font = "12px system-ui, sans-serif";
  }

  if (d.layer === "gestures" && d.gesture_counts) {
    const txt = "session: " + Object.entries(d.gesture_counts)
      .map(([g, n]) => `${g} x${n}`).join(", ");
    ctx.fillStyle = "rgba(15,23,42,0.85)";
    ctx.fillRect(8, ch - 30, ctx.measureText(txt).width + 14, 22);
    ctx.fillStyle = "#f8fafc";
    ctx.fillText(txt, 14, ch - 14);
  }

  if (alertOn) {
    ctx.lineWidth = 6;
    ctx.strokeStyle = "rgba(239,68,68,0.9)";
    ctx.strokeRect(3, 3, cw - 6, ch - 6);
  }
}

const _SKELETON_EDGES = [
  [5, 7], [7, 9], [6, 8], [8, 10], [5, 6], [5, 11], [6, 12],
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [0, 5], [0, 6],
];

function _drawSkeleton(ctx, kps, sx, sy, ox, oy) {
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(96,165,250,0.95)";
  for (const [a, b] of _SKELETON_EDGES) {
    const p = kps[a], q = kps[b];
    if (!p || !q || p[2] < 0.3 || q[2] < 0.3) continue;
    ctx.beginPath();
    ctx.moveTo((p[0] + ox) * sx, (p[1] + oy) * sy);
    ctx.lineTo((q[0] + ox) * sx, (q[1] + oy) * sy);
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(219,234,254,0.95)";
  for (const k of kps) {
    if (!k || k[2] < 0.3) continue;
    ctx.beginPath();
    ctx.arc((k[0] + ox) * sx, (k[1] + oy) * sy, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

const _TRAIL_PALETTE = [
  "rgba(96,165,250,0.9)", "rgba(74,222,128,0.9)", "rgba(251,146,60,0.9)",
  "rgba(232,121,249,0.9)", "rgba(250,204,21,0.9)", "rgba(45,212,191,0.9)",
];
function _trailColor(tid) {
  return _TRAIL_PALETTE[Math.abs(tid) % _TRAIL_PALETTE.length];
}

async function _refreshAnalysisBg(a) {
  if (!a || !a.bg) return;
  try {
    const r = await fetch(
      `/api/analysis/frame?cam=${encodeURIComponent(a.cam)}&_=${Date.now()}`,
      { cache: "no-store" });
    if (r.status !== 200
        || !(r.headers.get("Content-Type") || "").includes("image")) return;
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    a.bg.src = url;
    if (a.lastBgUrl) URL.revokeObjectURL(a.lastBgUrl);
    a.lastBgUrl = url;
  } catch (_) { /* transient */ }
}

function _heatColor(v, alpha) {
  const r = Math.round(255 * Math.min(1, v * 2));
  const g = Math.round(255 * Math.min(1, (1 - Math.abs(v - 0.5) * 2)));
  const b = Math.round(255 * Math.max(0, 1 - v * 2));
  return `rgba(${r},${g},${b},${alpha})`;
}

function stopTileAnalysis(st) {
  const a = st.analysis;
  if (!a) return;
  clearInterval(a.timer);
  if (a.videoStateTimer) clearInterval(a.videoStateTimer);
  if (a.evTimer) clearInterval(a.evTimer);
  if (a.evStrip) a.evStrip.remove();
  if (a.lastBgUrl) URL.revokeObjectURL(a.lastBgUrl);
  st.analysis = null;
  fetch(`/api/analysis/stop?cam=${encodeURIComponent(a.cam)}`,
        { method: "POST" }).catch(() => {});
  const wrap = st.videoWrap.querySelector(".analysis-wrap");
  if (wrap) wrap.remove();
  const strip = st.tile && st.tile.querySelector(".crossings-strip");
  if (strip) strip.remove();
  if (!st._overlayWasHidden) st.overlay.style.display = "";
}

// ---------- Activity + anomaly badges (per-camera local snapshot) --------

function _bandIndex(n, bands = ACTIVITY_BANDS) {
  for (const b of bands) if (n <= b.max) return b.idx;
  return 10;
}
function _vehicleLoad(r) {
  const c = r.counts;
  if (c && typeof c === "object") {
    let load = 0, seen = false;
    for (const [cls, w] of Object.entries(VEHICLE_LOAD_WEIGHTS)) {
      const n = c[cls];
      if (typeof n === "number" && n > 0) { load += w * n; }
      if (n != null) seen = true;
    }
    if (seen) return load;
  }
  return (r.vehicles ?? 0) * 1.0;
}
function _median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
}
function computeActivity(rows) {
  if (!rows.length) return null;
  const tail   = rows.slice(-3);
  const people = Math.round(_median(tail.map((r) => Math.max(0, r.person ?? 0))));
  const load   = _median(tail.map(_vehicleLoad));
  const pIdx   = _bandIndex(people, ACTIVITY_BANDS);
  const vIdx   = _bandIndex(load, VEHICLE_BANDS);
  const idx    = Math.max(pIdx, vIdx);
  const label = idx <= 3 ? "Quiet"
              : idx <= 6 ? "Moderate"
              : idx <= 8 ? "Busy"
              : "Crowded";
  const last = rows[rows.length - 1];
  return { idx, label, pIdx, vIdx,
           now: last.person ?? 0,
           veh: last.vehicles ?? 0,
           load: Math.round(load * 10) / 10 };
}

function setActivityBadge(st, act) {
  const badge = st.activityBadge;
  const text  = st.activityText;
  if (!act) {
    badge.className = "activity-badge act-unknown";
    text.textContent = "-/10";
    badge.title = "activity index - not enough samples yet";
    return;
  }
  const cls = act.label.toLowerCase();
  badge.className = `activity-badge act-${cls}`;
  text.textContent = `${act.idx}/10`;
  badge.title = `activity ${act.idx}/10 - ${act.label} - ` +
      `people ${act.now} (${act.pIdx}/10) - ` +
      `vehicle load ${act.load} (${act.vIdx}/10)`;
}

function renderSampleAge(st) {
  if (!st.ageEl) return;
  if (!st.lastSampleMs) { st.ageEl.textContent = ""; return; }
  const ageS = Math.max(0, Math.round((Date.now() - st.lastSampleMs) / 1000));
  const stale = ageS > STALE_AGE_S;
  const label = ageS < 90 ? `${ageS}s ago`
              : `${Math.round(ageS / 60)}m ago`;
  const memo = label + (stale ? "!" : "");
  if (memo !== st._ageMemo) {
    st._ageMemo = memo;
    st.ageEl.textContent = label;
    st.ageEl.classList.toggle("stale", stale);
  }
}

setInterval(() => {
  for (const st of Object.values(tileState)) renderSampleAge(st);
}, 1000);

const _LOCAL_ACTIVITY_WINDOW = 6;
const _LOCAL_HISTORY = Object.create(null);

function _updateLocalTileBadges(camId, j) {
  const st = tileState[camId];
  if (!st) return;
  const person   = Number(j?.counts?.person   ?? 0);
  const vehicles = Number(j?.counts?.vehicles ?? 0);
  const ts       = j?.at ? j.at * 1000 : Date.now();
  const hist = _LOCAL_HISTORY[camId] || (_LOCAL_HISTORY[camId] = []);
  hist.push({ person, vehicles, ts, counts: j?.counts || null });
  if (hist.length > _LOCAL_ACTIVITY_WINDOW) hist.shift();

  const act = computeActivity(hist);
  st.activityBadge.style.display = "";
  setActivityBadge(st, act);

  const load = act?.load ?? 0;
  const now  = act?.now  ?? 0;
  let mood = "unk", msg = "no data yet";
  if (hist.length >= 2) {
    mood = "ok";
    msg = act ? `no anomaly - activity ${act.idx}/10 ${act.label}` : "ok";
    if (now >= 50 || load >= 38) {
      mood = "spike";
      msg = `extreme load - ${now} people + ${load} veh-load units`;
    }
  }
  if (j?.dark != null) {
    mood = "spike";
    msg = `camera dark - view went black (mean luma ${j.dark})`;
  }
  if (j?.obstructed) {
    mood = "spike";
    msg = `camera obstructed - ${j.obstructed.cls} covers `
        + `${Math.round((j.obstructed.frac || 0) * 100)}% of view`;
  }
  st.anomalyBadge.style.display = "";
  st.anomalyBadge.className = `anomaly-badge ${mood}`;
  if (st.anomalyText) {
    st.anomalyText.textContent = mood === "spike" ? "!"
                              : mood === "unk"   ? "-"
                              : "ok";
  }
  st.anomalyBadge.title = msg;

  if (!st.analysis) st.overlay.style.display = "";
  const setK = (k, v) => {
    const el = [...st.latestVals].find((x) => x.dataset.k === k);
    if (el) el.textContent = v != null ? v : "-";
  };
  setK("person", person);
  setK("vehicles", vehicles);
  st.lastSampleMs = ts;
  renderSampleAge(st);
}

async function pollLocalModelView() {
  if (!SINGLE_CAM_ID) return;
  const url = MODEL_VIEW_JSON(SINGLE_CAM_ID) + `?_=${Date.now()}`;
  try {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) return;
    const j = await r.json();
    _updateLocalTileBadges(SINGLE_CAM_ID, j);
  } catch (_) { /* not yet written */ }
}

// ---------- Line editor (counting-line drawing) ---------------------------

const lineEditor = document.createElement("div");
lineEditor.style.cssText =
  "display:none;position:fixed;inset:0;z-index:70;background:rgba(2,6,23,.82);" +
  "align-items:center;justify-content:center";
lineEditor.innerHTML = `
  <div style="background:#0f172a;border:1px solid #334155;border-radius:12px;
              padding:16px 18px;max-width:800px;width:94%;color:#e2e8f0">
    <h3 style="margin:0 0 4px;font-size:17px">Counting line -
      <span data-le-cam></span></h3>
    <div style="color:#94a3b8;font-size:13px;margin-bottom:10px">
      Drag on the snapshot to place a counting line. Save persists it
      per-camera; a running Line-layer session picks it up within a few
      seconds without a restart.</div>
    <div style="position:relative;background:#020617;border:1px solid #334155;
                border-radius:8px;overflow:hidden">
      <img data-le-img style="display:block;width:100%;height:auto;
                              user-select:none;-webkit-user-drag:none">
      <canvas data-le-canvas style="position:absolute;inset:0;width:100%;
                                     height:100%;cursor:crosshair"></canvas>
    </div>
    <div data-le-classes style="display:flex;flex-wrap:wrap;gap:10px 16px;
                                 margin-top:10px;font-size:13px;
                                 color:#cbd5e1"></div>
    <div style="color:#94a3b8;font-size:12px;margin-top:4px">
      Nothing checked = count every tracked class.</div>
    <div data-le-err style="color:#f87171;font-size:13px;min-height:18px;
                            margin-top:8px"></div>
    <div style="display:flex;gap:10px;margin-top:6px">
      <button data-le-save style="cursor:pointer;background:#2563eb;border:0;
              color:#fff;border-radius:8px;padding:7px 18px">Save line</button>
      <button data-le-clear style="cursor:pointer;background:#334155;border:0;
              color:#fff;border-radius:8px;padding:7px 14px">Clear override</button>
      <button data-le-cancel style="cursor:pointer;background:#1e293b;
              border:1px solid #334155;color:#e2e8f0;border-radius:8px;
              padding:7px 14px">Close</button>
    </div>
  </div>`;
document.body.appendChild(lineEditor);

const _leImg = lineEditor.querySelector("[data-le-img]");
const _leCanvas = lineEditor.querySelector("[data-le-canvas]");
const _leErr = lineEditor.querySelector("[data-le-err]");
const _leClasses = lineEditor.querySelector("[data-le-classes]");
let _leCam = null;
let _lePts = [];

function _leRenderClasses(allowed, picked) {
  _leClasses.innerHTML = "";
  const set = new Set(picked || []);
  for (const name of (allowed || [])) {
    const id = "le-cls-" + name;
    const lab = document.createElement("label");
    lab.style.cssText = "display:inline-flex;align-items:center;gap:5px;" +
                        "cursor:pointer";
    lab.innerHTML = `<input type="checkbox" data-le-cls value="${name}" ` +
      `id="${id}"${set.has(name) ? " checked" : ""}> ${name}`;
    _leClasses.appendChild(lab);
  }
}

function _leCollectClasses() {
  const boxes = _leClasses.querySelectorAll("[data-le-cls]:checked");
  if (!boxes.length) return null;
  return Array.from(boxes, (b) => b.value);
}

function _leDraw() {
  const c = _leCanvas;
  c.width = _leImg.clientWidth || _leImg.naturalWidth || 640;
  c.height = _leImg.clientHeight || _leImg.naturalHeight || 360;
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  if (_lePts.length === 0) return;
  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 3;
  const p0 = [_lePts[0][0] * c.width, _lePts[0][1] * c.height];
  if (_lePts.length === 1) {
    ctx.beginPath(); ctx.arc(p0[0], p0[1], 6, 0, Math.PI * 2); ctx.fillStyle = "#f59e0b"; ctx.fill();
  } else {
    const p1 = [_lePts[1][0] * c.width, _lePts[1][1] * c.height];
    ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]); ctx.stroke();
    for (const [x, y] of [p0, p1]) {
      ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fillStyle = "#f59e0b"; ctx.fill();
    }
  }
}

let _leDragging = false;
_leCanvas.addEventListener("mousedown", (e) => {
  const r = _leCanvas.getBoundingClientRect();
  _lePts = [[(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]];
  _leDragging = true;
  _leDraw();
});
_leCanvas.addEventListener("mousemove", (e) => {
  if (!_leDragging) return;
  const r = _leCanvas.getBoundingClientRect();
  const pt = [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
  _lePts = _lePts.length === 0 ? [pt] : [_lePts[0], pt];
  _leDraw();
});
_leCanvas.addEventListener("mouseup", () => { _leDragging = false; });
window.addEventListener("resize", _leDraw);

lineEditor.querySelector("[data-le-cancel]").addEventListener("click",
  () => { lineEditor.style.display = "none"; });

lineEditor.querySelector("[data-le-save]").addEventListener("click", async () => {
  _leErr.textContent = "";
  if (_lePts.length !== 2) { _leErr.textContent = "Draw a line first (drag on the image)"; return; }
  const classes = _leCollectClasses();
  try {
    const r = await fetch(`/api/lines?cam=${encodeURIComponent(_leCam)}`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({line: _lePts, classes}),
    });
    if (!r.ok) throw new Error(await r.text());
    _leErr.style.color = "#4ade80";
    _leErr.textContent = "Saved - a running session picks it up in a few seconds";
    setTimeout(() => { _leErr.style.color = "#f87171"; _leErr.textContent = ""; }, 2500);
  } catch (e) {
    _leErr.style.color = "#f87171";
    _leErr.textContent = "Save failed: " + e.message;
  }
});

lineEditor.querySelector("[data-le-clear]").addEventListener("click", async () => {
  _leErr.textContent = "";
  try {
    const r = await fetch(`/api/lines/clear?cam=${encodeURIComponent(_leCam)}`,
                          {method: "POST"});
    if (!r.ok) throw new Error(await r.text());
    _lePts = [];
    _leRenderClasses(_leLastAllowed, []);
    _leDraw();
    _leErr.style.color = "#4ade80";
    _leErr.textContent = "Override cleared - falling back to cameras.py";
    setTimeout(() => { _leErr.style.color = "#f87171"; _leErr.textContent = ""; }, 2000);
  } catch (e) {
    _leErr.style.color = "#f87171";
    _leErr.textContent = "Clear failed: " + e.message;
  }
});

let _leLastAllowed = ["person", "bicycle", "car", "motorcycle", "bus", "truck"];

async function openLineEditor(cam, snapshotUrl) {
  _leCam = cam;
  _lePts = [];
  lineEditor.querySelector("[data-le-cam]").textContent = cam;
  _leImg.src = snapshotUrl || `/api/analysis/frame?cam=${encodeURIComponent(cam)}&_=${Date.now()}`;
  _leRenderClasses(_leLastAllowed, []);
  _leImg.onload = () => {
    fetch(`/api/lines?cam=${encodeURIComponent(cam)}`).then(r => r.json()).then(d => {
      if (d && d.line) _lePts = d.line;
      if (d && Array.isArray(d.allowed_classes) && d.allowed_classes.length)
        _leLastAllowed = d.allowed_classes;
      _leRenderClasses(_leLastAllowed, (d && d.classes) || []);
      _leDraw();
    }).catch(() => _leDraw());
  };
  lineEditor.style.display = "flex";
}
window.openLineEditor = openLineEditor;

// ---------- Zones editor (loiter/parking polygons) ------------------------

const zoneEditor = document.createElement("div");
zoneEditor.style.cssText =
  "display:none;position:fixed;inset:0;z-index:70;background:rgba(2,6,23,.82);" +
  "align-items:center;justify-content:center";
zoneEditor.innerHTML = `
  <div style="background:#0f172a;border:1px solid #334155;border-radius:12px;
              padding:16px 18px;max-width:800px;width:94%;color:#e2e8f0">
    <h3 style="margin:0 0 4px;font-size:17px"><span data-ze-title></span> -
      <span data-ze-cam></span></h3>
    <div style="color:#94a3b8;font-size:13px;margin-bottom:10px">
      Click the snapshot to drop polygon corners; double-click (or the
      button) closes the shape. Repeat for more zones, then Save.</div>
    <div style="position:relative;background:#020617;border:1px solid #334155;
                border-radius:8px;overflow:hidden">
      <img data-ze-img style="display:block;width:100%;height:auto;
                              user-select:none;-webkit-user-drag:none">
      <canvas data-ze-canvas style="position:absolute;inset:0;width:100%;
                                     height:100%;cursor:crosshair"></canvas>
    </div>
    <div data-ze-dwellrow style="margin-top:10px;font-size:13px;color:#cbd5e1">
      Loiter alert after <input data-ze-dwell type="number" min="5" max="3600"
        value="30" style="width:70px;background:#1e293b;color:#e2e8f0;
        border:1px solid #334155;border-radius:6px;padding:3px 6px"> seconds
      inside a zone.</div>
    <div data-ze-err style="color:#f87171;font-size:13px;min-height:18px;
                            margin-top:8px"></div>
    <div style="display:flex;gap:10px;margin-top:6px;flex-wrap:wrap">
      <button data-ze-closepoly style="cursor:pointer;background:#334155;
              border:0;color:#fff;border-radius:8px;padding:7px 14px">
        Close polygon</button>
      <button data-ze-undo style="cursor:pointer;background:#334155;border:0;
              color:#fff;border-radius:8px;padding:7px 14px">Undo point</button>
      <button data-ze-clear style="cursor:pointer;background:#7f1d1d;border:0;
              color:#fff;border-radius:8px;padding:7px 14px">Clear all</button>
      <button data-ze-save style="cursor:pointer;background:#2563eb;border:0;
              color:#fff;border-radius:8px;padding:7px 18px">Save</button>
      <button data-ze-cancel style="cursor:pointer;background:#1e293b;
              border:1px solid #334155;color:#e2e8f0;border-radius:8px;
              padding:7px 14px">Close</button>
    </div>
  </div>`;
document.body.appendChild(zoneEditor);

const _zeImg = zoneEditor.querySelector("[data-ze-img]");
const _zeCanvas = zoneEditor.querySelector("[data-ze-canvas]");
const _zeErr = zoneEditor.querySelector("[data-ze-err]");
let _zeCam = null, _zeKind = "loiter";
let _zeZones = [];
let _zeOthers = [];
let _zeCurrent = [];

function _zeRedraw() {
  const r = _zeImg.getBoundingClientRect();
  _zeCanvas.width = Math.max(1, Math.round(r.width));
  _zeCanvas.height = Math.max(1, Math.round(r.height));
  const cw = _zeCanvas.width, ch = _zeCanvas.height;
  const ctx = _zeCanvas.getContext("2d");
  ctx.clearRect(0, 0, cw, ch);
  ctx.font = "12px system-ui, sans-serif";
  for (const z of _zeZones) {
    ctx.beginPath();
    ctx.moveTo(z.points[0][0] * cw, z.points[0][1] * ch);
    for (let i = 1; i < z.points.length; i++)
      ctx.lineTo(z.points[i][0] * cw, z.points[i][1] * ch);
    ctx.closePath();
    ctx.fillStyle = "rgba(74,222,128,0.15)";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(74,222,128,0.95)";
    ctx.stroke();
    ctx.fillStyle = "#f8fafc";
    ctx.fillText(z.name || "?",
                 z.points[0][0] * cw + 4, z.points[0][1] * ch + 14);
  }
  if (_zeCurrent.length) {
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(250,204,21,0.95)";
    ctx.beginPath();
    ctx.moveTo(_zeCurrent[0][0] * cw, _zeCurrent[0][1] * ch);
    for (let i = 1; i < _zeCurrent.length; i++)
      ctx.lineTo(_zeCurrent[i][0] * cw, _zeCurrent[i][1] * ch);
    ctx.stroke();
    ctx.fillStyle = "rgba(250,204,21,0.95)";
    for (const p of _zeCurrent) {
      ctx.beginPath();
      ctx.arc(p[0] * cw, p[1] * ch, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function _zeClosePoly() {
  if (_zeCurrent.length < 3) {
    _zeErr.textContent = "a polygon needs at least 3 points";
    return;
  }
  const prefix = _zeKind === "parking" ? "P" : "Z";
  _zeZones.push({
    kind: _zeKind,
    name: prefix + (_zeZones.length + 1),
    points: _zeCurrent.slice(),
  });
  _zeCurrent = [];
  _zeErr.textContent = "";
  _zeRedraw();
}

_zeCanvas.addEventListener("click", (e) => {
  const r = _zeCanvas.getBoundingClientRect();
  _zeCurrent.push([
    Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
    Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))]);
  _zeRedraw();
});
_zeCanvas.addEventListener("dblclick", (e) => {
  e.preventDefault();
  if (_zeCurrent.length >= 2) _zeCurrent.pop();
  _zeClosePoly();
});
zoneEditor.querySelector("[data-ze-closepoly]")
  .addEventListener("click", _zeClosePoly);
zoneEditor.querySelector("[data-ze-undo]").addEventListener("click", () => {
  if (_zeCurrent.length) _zeCurrent.pop();
  else _zeZones.pop();
  _zeRedraw();
});
zoneEditor.querySelector("[data-ze-clear]").addEventListener("click", () => {
  _zeZones = []; _zeCurrent = [];
  _zeRedraw();
});
zoneEditor.querySelector("[data-ze-cancel]").addEventListener("click",
  () => { zoneEditor.style.display = "none"; });
zoneEditor.querySelector("[data-ze-save]").addEventListener("click",
  async () => {
    if (_zeCurrent.length) _zeClosePoly();
    const dwell = Number(zoneEditor.querySelector("[data-ze-dwell]").value)
                  || 30;
    const mine = _zeZones.map((z) => (_zeKind === "loiter"
      ? { ...z, dwell_s: Math.min(3600, Math.max(5, dwell)) } : z));
    try {
      const r = await fetch(`/api/zones?cam=${encodeURIComponent(_zeCam)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ zones: [..._zeOthers, ...mine] }),
      });
      if (!r.ok) throw new Error((await r.text()).slice(0, 120));
      zoneEditor.style.display = "none";
    } catch (e) {
      _zeErr.textContent = "save failed: " + e.message;
    }
  });

async function openZoneEditor(cam, kind, snapshotUrl) {
  _zeCam = cam;
  _zeKind = kind === "parking" ? "parking" : "loiter";
  _zeCurrent = [];
  _zeErr.textContent = "";
  zoneEditor.querySelector("[data-ze-title]").textContent =
    _zeKind === "parking" ? "Parking spots" : "Loitering zones";
  zoneEditor.querySelector("[data-ze-cam]").textContent = cam;
  zoneEditor.querySelector("[data-ze-dwellrow]").style.display =
    _zeKind === "loiter" ? "" : "none";
  try {
    const j = await fetch(`/api/zones?cam=${encodeURIComponent(cam)}`)
      .then((r) => r.json());
    const all = Array.isArray(j.zones) ? j.zones : [];
    _zeZones = all.filter((z) => z.kind === _zeKind);
    _zeOthers = all.filter((z) => z.kind !== _zeKind);
    const dz = _zeZones.find((z) => z.dwell_s);
    if (dz) zoneEditor.querySelector("[data-ze-dwell]").value = dz.dwell_s;
  } catch (_) { _zeZones = []; _zeOthers = []; }
  _zeImg.onload = _zeRedraw;
  _zeImg.src = snapshotUrl;
  zoneEditor.style.display = "flex";
  if (_zeImg.complete) _zeRedraw();
}

// ---------- Crossings toast + strip ---------------------------------------

const _crossToast = document.createElement("div");
_crossToast.style.cssText =
  "position:fixed;top:16px;right:16px;z-index:80;display:none;" +
  "background:#dc2626;color:#fff;padding:10px 14px;border-radius:8px;" +
  "font-size:14px;font-weight:600;box-shadow:0 6px 20px rgba(0,0,0,.4);" +
  "max-width:320px";
document.body.appendChild(_crossToast);

let _crossToastTimer = null;
function showCrossToast(msg) {
  _crossToast.textContent = msg;
  _crossToast.style.display = "block";
  if (_crossToastTimer) clearTimeout(_crossToastTimer);
  _crossToastTimer = setTimeout(() => { _crossToast.style.display = "none"; }, 3500);
}

const _seenCrossings = new Map();
const CROSSINGS_STRIP_MAX = 8;
const CROSSINGS_POLL_LIMIT = CROSSINGS_STRIP_MAX + 4;

function _ensureCrossingsStrip(st) {
  if (!st || !st.tile) return null;
  let strip = st.tile.querySelector(".crossings-strip");
  if (strip) return strip;
  strip = document.createElement("div");
  strip.className = "crossings-strip";
  strip.style.cssText =
    "display:flex;gap:6px;overflow-x:auto;padding:6px 8px;background:#0b1220;" +
    "border-top:1px solid #1f2937";
  strip.dataset.empty = "1";
  strip.textContent = "waiting for the first crossing...";
  strip.style.color = "#64748b";
  strip.style.fontSize = "12px";
  st.tile.appendChild(strip);
  return strip;
}

function _renderCrossingsStrip(strip, events) {
  if (!strip) return;
  const top = events[0];
  const topKey = top ? (top.ts + "|" + top.tid + "|" + top.direction) : "";
  if (strip.dataset.topKey === topKey) return;
  strip.dataset.topKey = topKey;
  strip.style.color = "";
  strip.style.fontSize = "";
  strip.innerHTML = "";
  if (!top) {
    strip.dataset.empty = "1";
    strip.style.color = "#64748b";
    strip.style.fontSize = "12px";
    strip.textContent = "waiting for the first crossing...";
    return;
  }
  strip.dataset.empty = "0";
  for (const ev of events.slice(0, CROSSINGS_STRIP_MAX)) {
    const card = document.createElement("div");
    card.style.cssText =
      "flex:0 0 auto;width:88px;background:#111827;border:1px solid #1f2937;" +
      "border-radius:6px;overflow:hidden;text-align:center";
    const dir = (ev.direction === "in") ? "IN" : "OUT";
    const color = (ev.direction === "in") ? "#22c55e" : "#f97316";
    const hhmmss = (ev.ts || "").substr(11, 8);
    if (ev.snap) {
      const img = document.createElement("img");
      img.src = "/" + ev.snap;
      img.alt = dir;
      img.style.cssText = "display:block;width:100%;height:56px;object-fit:cover";
      card.appendChild(img);
    } else {
      const placeholder = document.createElement("div");
      placeholder.style.cssText = "height:56px;background:#020617;" +
        "display:flex;align-items:center;justify-content:center;" +
        "color:#475569;font-size:11px";
      placeholder.textContent = "no crop";
      card.appendChild(placeholder);
    }
    const meta = document.createElement("div");
    meta.style.cssText = "padding:3px 4px;font-size:11px;line-height:1.2;" +
                         "color:#e2e8f0";
    meta.innerHTML =
      `<div style="color:${color};font-weight:600">${dir} - ${escapeHtml(ev.cls || "obj")}</div>` +
      `<div style="color:#94a3b8">${escapeHtml(hhmmss)}</div>`;
    card.appendChild(meta);
    strip.appendChild(card);
  }
}

setInterval(async () => {
  for (const st of Object.values(tileState)) {
    if (!st) continue;
    const onLine = st.analysis && st.analysis.layer === "line";
    if (!onLine) {
      const stale = st.tile && st.tile.querySelector(".crossings-strip");
      if (stale) stale.remove();
      continue;
    }
    const cam = st.analysis.cam;
    if (!cam) continue;
    let seen = _seenCrossings.get(cam);
    if (!seen) { seen = new Set(); _seenCrossings.set(cam, seen); }
    const strip = _ensureCrossingsStrip(st);
    try {
      const r = await fetch(`/api/crossings?cam=${encodeURIComponent(cam)}` +
                            `&limit=${CROSSINGS_POLL_LIMIT}`);
      if (!r.ok) continue;
      const data = await r.json();
      const eventsNewestFirst = data.events || [];
      _renderCrossingsStrip(strip, eventsNewestFirst);
      const events = eventsNewestFirst.slice().reverse();
      const boot = seen.size === 0;
      for (const ev of events) {
        const key = ev.ts + "|" + ev.tid + "|" + ev.direction;
        if (seen.has(key)) continue;
        seen.add(key);
        if (boot) continue;
        showCrossToast(`${ev.direction === "in" ? "IN" : "OUT"}  ` +
                       `${ev.cls || "object"}  @ ${ev.ts.substr(11, 8)}`);
        const el = st.tile || st.videoWrap || null;
        if (el && el.style) {
          const prev = el.style.boxShadow;
          el.style.boxShadow = "0 0 0 4px #dc2626 inset";
          setTimeout(() => { el.style.boxShadow = prev; }, 800);
        }
      }
      if (seen.size > 200) {
        const arr = Array.from(seen);
        _seenCrossings.set(cam, new Set(arr.slice(arr.length - 100)));
      }
    } catch (_e) { /* transient */ }
  }
}, 4000);

// ---------- Boot -----------------------------------------------------------

async function boot() {
  const cams = await refreshCameraPicker();
  initUpload();
  initStartCamera();
  const urlCam = new URLSearchParams(location.search).get("cam");
  const status = document.getElementById("picker-status");
  if (!urlCam) {
    if (status) status.textContent =
      "pick a camera above (or upload a video) and press Start to load it.";
    const host = document.getElementById("tile-single");
    if (host) host.innerHTML =
      `<div class="video-fallback" style="min-height:240px;padding:40px;text-align:center;color:#8b909a">
         No camera selected. Pick one from the dropdown above and press Start.
       </div>`;
    return;
  }
  let cam = (cams || []).find((c) => (c.id || c.cam_id) === urlCam);
  if (!cam) {
    try {
      const r = await fetch(`${CATALOG_ENDPOINT}?cam=${encodeURIComponent(urlCam)}`,
                            { cache: "no-store" });
      if (r.ok) {
        const j = await r.json();
        cam = j.camera || j;
      }
    } catch (_) {}
  }
  if (!cam) {
    cam = { id: urlCam, name: urlCam, kind: "hls" };
  }
  SINGLE_CAM_ID = cam.id || cam.cam_id || urlCam;
  SINGLE_CAM = cam;
  buildSingleTile(cam);
  if (status) status.textContent = "";
  pollLocalModelView();
  setInterval(pollLocalModelView, 8000);
}

document.addEventListener("DOMContentLoaded", boot);
if (document.readyState !== "loading") boot();

// Diagnostic hook: return the current tile's liveness at a glance.
window.__tileLiveDebug = () => Object.fromEntries(
  Object.entries(tileState).map(([sid, st]) => {
    const video = st.videoWrap && st.videoWrap.querySelector("video");
    return [sid, {
      cam: st.cam && st.cam.id,
      kind: st.cam && st.cam.kind,
      hasVideo: !!video,
      curTime: video ? Math.round(video.currentTime) : null,
      analysis: st.analysis ? st.analysis.layer : null,
      lastSampleMs: st.lastSampleMs,
    }];
  }));
