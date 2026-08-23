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
  // Hide the standalone #picker-bar - its contents move into the tile
  // header so the row reads: [cam name] [picker in middle] [Analyze / -/10 / OK]
  const oldPicker = document.getElementById("picker-bar");
  if (oldPicker) oldPicker.style.display = "none";
  host.innerHTML = `
    <div class="tile-head" style="display:flex;align-items:center;
         gap:16px;flex-wrap:wrap">
      <div class="tile-head-left" style="flex-shrink:0;min-width:180px">
        <h2 data-cam-name style="margin:0">${escapeHtml(cam.name || cam.id)}</h2>
        <div class="city" data-cam-area style="color:#94a3b8;font-size:12px">
          ${escapeHtml(cam.area || cam.kind || "")}</div>
      </div>
      <div class="tile-head-picker" style="flex:1;display:flex;
           align-items:center;gap:8px;flex-wrap:wrap;justify-content:center;
           min-width:0">
        <label for="cam-picker-inline" class="sub" style="flex-shrink:0;
             color:#94a3b8;font-size:12px">Camera:</label>
        <select id="cam-picker-inline" data-cam-picker-inline
                style="background:#0c0e13;color:#e7e9ee;
                border:1px solid #2c3140;border-radius:6px;padding:6px 10px;
                font-size:13px;flex:1;min-width:200px;max-width:420px">
          <option>loading...</option>
        </select>
        <button data-upload-inline title="upload a local MP4/MKV file"
                style="background:#1e293b;color:#e2e8f0;border:1px solid #334155;
                border-radius:6px;padding:6px 12px;font-size:13px;cursor:pointer;
                flex-shrink:0">Upload</button>
        <button data-start-inline title="load the picked camera"
                style="background:#2563eb;color:#fff;border:1px solid #2563eb;
                border-radius:6px;padding:6px 14px;font-size:13px;cursor:pointer;
                flex-shrink:0">Start</button>
      </div>
      <div class="tile-head-right" style="flex-shrink:0;display:flex;
           align-items:center;gap:8px">
        <button class="analyze-btn" data-analyze
                title="Live advanced analysis - pick one layer"
                style="cursor:pointer;border:1px solid #334155;background:#1e293b;color:#e2e8f0;border-radius:6px;padding:2px 8px;font-size:13px">Analyze</button>
        <button class="cinema-btn" data-cinema
                title="Cinema mode - hide UI, video fills the tab, plate crops strip at bottom"
                style="cursor:pointer;border:1px solid #334155;background:#1e293b;color:#e2e8f0;border-radius:6px;padding:2px 8px;font-size:13px">Cinema</button>
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
  // Mirror the inline picker to the same options as the (now hidden)
  // top-of-page picker so clicking Start in the header does the same
  // thing as the old picker bar. Wire the three inline controls to the
  // existing handlers on the standalone elements.
  const inlineSel = host.querySelector("[data-cam-picker-inline]");
  const origSel = document.getElementById("cam-picker");
  if (inlineSel && origSel) {
    // Mirror the current options once, then keep in sync: /api/cameras
    // is async and can resolve AFTER buildSingleTile runs, so the plain
    // one-shot snapshot leaves the inline picker frozen on "loading...".
    // A MutationObserver on the origin picker re-mirrors options + value
    // every time the top-of-page picker is repopulated.
    const _mirror = () => {
      const keep = inlineSel.value;
      inlineSel.innerHTML = origSel.innerHTML;
      // Preserve the operator's current selection when the mirror rewrites
      // the option list, else fall through to whatever origSel currently
      // has selected.
      const hasKeep = keep && Array.from(inlineSel.options)
                                 .some((o) => o.value === keep);
      inlineSel.value = hasKeep ? keep : origSel.value;
    };
    _mirror();
    try {
      const _obs = new MutationObserver(_mirror);
      _obs.observe(origSel, { childList: true });
      inlineSel._pickerObs = _obs;
    } catch (_e) { /* older browsers - one-shot snapshot is the fallback */ }
    inlineSel.addEventListener("change", () => {
      origSel.value = inlineSel.value;
      origSel.dispatchEvent(new Event("change", {bubbles: true}));
    });
  }
  const inlineUpload = host.querySelector("[data-upload-inline]");
  const origUpload = document.getElementById("upload-btn");
  if (inlineUpload && origUpload) {
    inlineUpload.addEventListener("click", () => origUpload.click());
  }
  const inlineStart = host.querySelector("[data-start-inline]");
  const origStart = document.getElementById("start-cam");
  if (inlineStart && origStart) {
    inlineStart.addEventListener("click", () => origStart.click());
  }
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
  const cinemaBtn = host.querySelector("[data-cinema]");
  if (cinemaBtn) cinemaBtn.addEventListener("click", () => toggleCinemaMode(st));
  buildVideoInto(st, cam);
  return st;
}

// ---------- Cinema mode + saved plate crops strip -----------------------
// Operator ask 2026-08-20: LPR needs bigger pixels than the default tile
// gives (~800px wide video → 20-40px plates → OCR borderline). Cinema mode
// stretches the video to the whole tab so YouTube's iframe renders at
// 1600-1920px wide, tripling the plate pixels. The canvas overlay stays
// drawn on top of the video (unlike YouTube's own fullscreen which nukes
// it), and a horizontal strip of the most recent saved plate crops is
// pinned at the bottom so the operator sees the LPR extract land in real
// time without leaving Cinema.
(function installCinemaMode() {
  const style = document.createElement("style");
  style.textContent = `
    body.cinema-active > header,
    body.cinema-active > nav.tabbar,
    body.cinema-active .tile-head,
    body.cinema-active #picker-bar,
    body.cinema-active [data-tab="investigation"],
    body.cinema-active [data-tab="models"] { display: none !important; }
    body.cinema-active main { padding: 0 !important; margin: 0 !important; max-width: none !important; }
    body.cinema-active #tile-single {
      position: fixed !important; inset: 0 !important;
      margin: 0 !important; padding: 0 !important;
      background: #000; z-index: 40;
      width: 100vw !important; height: 100vh !important;
    }
    body.cinema-active #tile-single .video-wrap {
      position: absolute !important; left: 0 !important; right: 0 !important;
      top: 0 !important; bottom: 96px !important;
      width: auto !important; height: auto !important;
      max-height: none !important; max-width: none !important;
      aspect-ratio: auto !important; margin: 0 !important; padding: 0 !important;
    }
    body.cinema-active #tile-single .video-wrap > iframe,
    body.cinema-active #tile-single .video-wrap > video,
    body.cinema-active #tile-single .video-wrap > canvas,
    body.cinema-active #tile-single .video-wrap > img {
      position: absolute !important; inset: 0 !important;
      width: 100% !important; height: 100% !important; object-fit: contain;
    }
    body.cinema-active .cinema-exit-btn {
      position: fixed; top: 12px; right: 16px; z-index: 60;
      background: rgba(15,23,42,.85); color: #f1f5f9; border: 1px solid #475569;
      border-radius: 999px; padding: 6px 14px; cursor: pointer; font-size: 13px;
    }
    body.cinema-active .cinema-analyze-btn,
    body.native-fullscreen .cinema-analyze-btn,
    body.native-fullscreen .cinema-exit-btn {
      position: fixed; top: 12px; right: 16px; z-index: 2147483646;
      background: rgba(37,99,235,.9); color: #fff; border: 1px solid #2563eb;
      border-radius: 999px; padding: 6px 18px; cursor: pointer; font-size: 13px;
    }
    body.cinema-active .cinema-analyze-btn { right: 140px; }
    body.native-fullscreen .cinema-exit-btn {
      background: rgba(15,23,42,.9); border-color: #475569; color: #f1f5f9;
      right: 16px;
    }
    body.native-fullscreen .cinema-analyze-btn { right: 150px; }
    body.cinema-active #plate-crops-strip {
      position: fixed; left: 0; right: 0; bottom: 0; height: 96px;
      display: flex !important; align-items: center; gap: 8px;
      padding: 8px 12px; z-index: 50; background: rgba(15,23,42,.72);
      border-top: 1px solid #334155; overflow-x: auto;
    }
    #plate-crops-strip { display: none; }
    #plate-crops-strip .lbl { color: #cbd5e1; font-size: 11px; font-weight: 600;
      writing-mode: vertical-lr; text-orientation: mixed; padding: 0 6px;
      flex-shrink: 0; letter-spacing: 0.08em; }
    #plate-crops-strip img { height: 80px; width: auto; border: 1px solid #475569;
      border-radius: 4px; flex-shrink: 0; background: #0f172a; }
    #plate-crops-strip .empty { color: #94a3b8; font-size: 12px; padding: 0 12px; }
  `;
  document.head.appendChild(style);

  const strip = document.createElement("div");
  strip.id = "plate-crops-strip";
  strip.innerHTML = `<span class="lbl">SAVED PLATE CROPS</span>
                     <span class="empty">no plate saves yet - waiting for the LPR pass to land one</span>`;
  document.body.appendChild(strip);

  const exitBtn = document.createElement("button");
  exitBtn.className = "cinema-exit-btn";
  exitBtn.textContent = "Exit Cinema";
  exitBtn.style.display = "none";
  exitBtn.addEventListener("click", () => toggleCinemaMode(null, false));
  document.body.appendChild(exitBtn);

  // Cinema-mode Analyze button: the normal Analyze in .tile-head gets
  // hidden by the cinema-active CSS, so operators lose access to the
  // layer picker mid-analysis. Mirror it as a fixed button next to
  // Exit Cinema so switching layers works without leaving Cinema.
  const analyzeBtn = document.createElement("button");
  analyzeBtn.className = "cinema-analyze-btn";
  analyzeBtn.textContent = "Analyze";
  analyzeBtn.style.display = "none";
  analyzeBtn.addEventListener("click", () => {
    // Route the click at the ACTIVE tile's real state so
    // openAnalysisPicker gets a fully-populated `st` (camNameEl, cam,
    // analysis). SINGLE_CAM_ID + tileState live in the module scope
    // that surrounds this handler - safe closure reference, not the
    // window global (which is undefined for `let` module vars).
    const sid = SINGLE_CAM_ID;
    const st = sid ? tileState[sid] : null;
    if (st) {
      openAnalysisPicker(st);
    } else {
      console.warn("[cinema-analyze] no active tile - pick a camera first");
    }
  });
  document.body.appendChild(analyzeBtn);

  window.__cinemaExitBtn = exitBtn;
  window.__cinemaAnalyzeBtn = analyzeBtn;
  window.__cinemaStrip = strip;

  // Browser-native fullscreen (F11 or Element.requestFullscreen inside an
  // iframe like YouTube's) hides the dashboard's normal Analyze button
  // because the fullscreen element covers everything else. Track the
  // change and reveal the floating Analyze / Exit buttons on the same
  // z-index as the fullscreen surface so the operator keeps the layer
  // picker within reach without leaving fullscreen.
  const onFsChange = () => {
    const fsEl = document.fullscreenElement || document.webkitFullscreenElement;
    document.body.classList.toggle("native-fullscreen", !!fsEl);
    const showFloat = !!fsEl || document.body.classList.contains("cinema-active");
    if (analyzeBtn) analyzeBtn.style.display = showFloat ? "block" : "none";
    if (exitBtn) exitBtn.style.display = showFloat ? "block" : "none";
    // Wire the exit button to leave fullscreen too, not just Cinema.
    if (fsEl && exitBtn) {
      exitBtn.textContent = "Exit Fullscreen";
      exitBtn.onclick = () => {
        if (document.exitFullscreen) document.exitFullscreen();
        else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
      };
    } else if (exitBtn) {
      exitBtn.textContent = "Exit Cinema";
      exitBtn.onclick = null;
      exitBtn.addEventListener("click", () => toggleCinemaMode(null, false), {once: true});
    }
  };
  document.addEventListener("fullscreenchange", onFsChange);
  document.addEventListener("webkitfullscreenchange", onFsChange);
})();

function toggleCinemaMode(st, force) {
  const active = force !== undefined
    ? force
    : !document.body.classList.contains("cinema-active");
  document.body.classList.toggle("cinema-active", active);
  const exitBtn = window.__cinemaExitBtn;
  if (exitBtn) exitBtn.style.display = active ? "block" : "none";
  const analyzeBtn = window.__cinemaAnalyzeBtn;
  if (analyzeBtn) analyzeBtn.style.display = active ? "block" : "none";
  // Re-target SC bbox after the CSS layout has committed. Waiting one
  // full second is generous, but that's the price of never firing on
  // the pre-transition size (which was the "stuck video" symptom).
  if (active) {
    setTimeout(() => updateScCaptureBbox(st), 1000);
  } else if (_hudSystemInfo && _hudSystemInfo.screen_capture
             && _hudSystemInfo.screen_capture.enabled) {
    fetch("/api/screen-capture/bbox", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    }).catch(() => {});
  }
}

// updateScCaptureBbox: POST the video's on-screen rectangle to the
// screen-capture endpoint so YOLO reads pixels from ONLY the video
// area, not the black letterbox or the browser chrome around it.
// Called from cinema toggle, native-fullscreen change, resize (debounced)
// and analyze-start. Guards against degenerate rects (pre-transition).
function updateScCaptureBbox(st) {
  try {
    if (!_hudSystemInfo || !_hudSystemInfo.screen_capture
        || !_hudSystemInfo.screen_capture.enabled) return;
    // A hidden / minimised tab zeroes out outerHeight/outerWidth and
    // screenX/screenY in Chrome. Publishing a bbox computed from those
    // zeros produces off-screen coordinates (y1 goes negative) and the
    // capturer starts grabbing the wrong region. Keep the last known
    // good bbox instead by short-circuiting whenever the window is not
    // actually visible on the desktop.
    if (document.visibilityState !== "visible") return;
    if (!window.outerHeight || !window.outerWidth) return;
    // Pick the active tile if st was not provided.
    if (!st) {
      const sid = SINGLE_CAM_ID;
      st = sid ? tileState[sid] : null;
    }
    if (!st || !st.videoWrap) return;
    const el = st.videoWrap.querySelector("iframe")
             || st.videoWrap.querySelector("video");
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (rect.width < 100 || rect.height < 100) return;
    const dpr = window.devicePixelRatio || 1;
    // window.screenY is the OUTER top of the browser window (including
    // any browser chrome above). window.innerHeight excludes chrome, so
    // the vertical delta between them is the address bar + tabs. Add
    // that to rect.top (viewport-relative) to land on the physical Y.
    // Under browser fullscreen (F11) both deltas are zero so this
    // reduces to rect.top + screenY, which is correct.
    const chromeV = window.outerHeight - window.innerHeight;
    const bbox = {
      x1: Math.round((rect.left + window.screenX) * dpr),
      y1: Math.round((rect.top  + window.screenY + chromeV) * dpr),
      x2: Math.round((rect.right + window.screenX) * dpr),
      y2: Math.round((rect.bottom + window.screenY + chromeV) * dpr),
    };
    // Reject bboxes with any negative coordinate (a sign the math went
    // off the rails - typically hidden-tab false positive that slipped
    // past the visibility check).
    if (bbox.x1 < 0 || bbox.y1 < 0 || bbox.x2 < 0 || bbox.y2 < 0) return;
    fetch("/api/screen-capture/bbox", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(bbox),
    }).catch(() => {});
  } catch (_e) { /* SC bbox is best-effort, never blocks the loop */ }
}

// Re-target the bbox any time the window resizes / moves (debounced 400 ms).
let _bboxResizeTimer = null;
function _bboxResizeSchedule() {
  clearTimeout(_bboxResizeTimer);
  _bboxResizeTimer = setTimeout(() => updateScCaptureBbox(null), 400);
}
window.addEventListener("resize", _bboxResizeSchedule);
document.addEventListener("fullscreenchange", () => {
  setTimeout(() => updateScCaptureBbox(null), 500);
});

// Poll /api/analysis/saved and populate the strip with plate crops only.
// Runs whenever cinema is active (once every 3 s) so a fresh save lands
// in the strip within a tick.
setInterval(async () => {
  const strip = window.__cinemaStrip;
  if (!strip || !document.body.classList.contains("cinema-active")) return;
  try {
    const r = await fetch("/api/analysis/saved", { cache: "no-store" });
    if (!r.ok) return;
    const j = await r.json();
    const rows = (j.items || j.saved || []).filter(x =>
      (x.layer === "plates") || /plate/i.test(x.file || x.name || ""));
    if (!rows.length) return;
    const label = strip.querySelector(".lbl");
    strip.innerHTML = "";
    if (label) strip.appendChild(label);
    for (const row of rows.slice(0, 12)) {
      const src = row.url || row.file || row.path;
      if (!src) continue;
      const img = document.createElement("img");
      img.src = src.startsWith("/") ? src : "/" + src;
      img.alt = row.plate || row.text || "plate";
      img.title = `${row.plate || row.text || "unread"} · ${row.ts || row.at || ""}`;
      strip.appendChild(img);
    }
  } catch (_e) { /* keep last known strip on transient errors */ }
}, 3000);

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
  } else if (cam.kind === "youtube") {
    // yt-dlp may be blocked (YouTube bot check on Chrome 127+ hosts): even
    // when the backend cannot resolve an HLS the operator still needs to
    // SEE the stream. Embed the YouTube iframe directly - the player runs
    // in its own context, does not depend on yt-dlp, and the canvas
    // overlay above still receives whatever boxes the backend can produce.
    const m = String(cam.url || "").match(
        /(?:youtube\.com\/(?:watch\?v=|embed\/|live\/)|youtu\.be\/)([\w-]{11})/);
    const vid = m ? m[1] : "";
    if (vid) {
      // `vq=hd2160` is legacy but some players still honour it; the
      // authoritative quality push happens after load via postMessage.
      markup = `<iframe data-youtube="${vid}"
                        src="https://www.youtube.com/embed/${vid}?autoplay=1&mute=1&playsinline=1&controls=1&enablejsapi=1&vq=hd2160&rel=0&modestbranding=1"
                        allow="autoplay; encrypted-media; picture-in-picture; fullscreen"
                        allowfullscreen
                        style="border:0;background:#0c0e13"></iframe>`;
    } else {
      markup = `<div class="video-fallback">
                  YouTube URL missing a video id - check cameras.py.
                </div>`;
    }
  } else if (cam.hls || cam.active_hls
             || (cam.kind === "hls" && cam.url)) {
    const hlsUrl = cam.hls || cam.active_hls || cam.url;
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
  const yt = st.videoWrap.querySelector("iframe[data-youtube]");
  if (yt) {
    yt.addEventListener("load", () => {
      // Ask YouTube for the highest available quality; the player picks
      // the top the bandwidth allows (4K when the source has it).
      try {
        yt.contentWindow.postMessage(
          JSON.stringify({event: "command",
                          func: "setPlaybackQuality",
                          args: ["hd2160"]}), "*");
      } catch (_) { /* cross-origin postMessage may reject silently */ }
    }, { once: true });
  }
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
  ["gestures", "Hand gestures"],
  ["body",     "Body anomalies"],
  ["fall",     "Fall detection"],
  ["faces",    "Face detection"],
  ["line",     "Line crossing"],
  ["fire",     "Fire detection"],
  ["parking",  "Parking occupancy"],
  ["plates",   "License plates (LPR)"],
];
const DRAWABLE_LAYERS = { line: "Draw line",
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
      <button data-an-cancel style="cursor:pointer;background:#1e293b;
              border:1px solid #334155;color:#e2e8f0;border-radius:8px;
              padding:7px 14px;font-size:14px">Cancel</button>
    </div>
  </div>`;
document.body.appendChild(analysisPanel);

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
    // Hide the picker overlay before kicking off analysis so it
    // doesn't linger on top of the video during startup.
    analysisPanel.style.display = "none";
    // SC bbox reroute: send the video's on-screen rectangle to the
    // capturer so YOLO reads only the video pixels (not the black
    // letterbox / browser chrome / etc.). Uses the shared helper so the
    // logic and math stay consistent across normal + Cinema + native
    // fullscreen modes.
    updateScCaptureBbox(st);
    let _startOk = false;
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
      _startOk = true;
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
      // Start failed - re-open the picker so the operator sees the error
      // and can pick another layer.
      analysisPanel.style.display = "flex";
    } finally {
      runBtn.disabled = false;
      // Absolute guarantee: after success the panel is never left open.
      if (_startOk) analysisPanel.style.display = "none";
    }
  });

const _layerLabel = Object.fromEntries(ANALYSIS_LAYER_DEFS);

function beginTileAnalysis(st, cam, layer) {
  if (st.analysis) {
    st.analysis.layer = layer;
    st.analysis.tickBuf.length = 0;
    const barRoot = st.analysis.bar || st.videoWrap;
    const tag = barRoot.querySelector(".analysis-live-tag");
    if (tag) tag.textContent = `LIVE - ${_layerLabel[layer] || layer}`;
    const lb = barRoot.querySelector(".analysis-drawline");
    if (lb) {
      lb.style.display = DRAWABLE_LAYERS[layer] ? "" : "none";
      if (DRAWABLE_LAYERS[layer]) lb.textContent = DRAWABLE_LAYERS[layer];
    }
    return;
  }
  st._overlayWasHidden = st.overlay.style.display === "none";
  st.overlay.style.display = "none";
  // Operator request 2026-08-18: no more buttons on top of the video.
  // Everything the operator needs to see or click while analysis is
  // running lives in a status bar ABOVE the player. Only the actual
  // detection drawing (canvas + still-frame background for iframe
  // sources) stays inside the video-wrap.
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
                   pointer-events:none;background:transparent;"></canvas>`;
  st.videoWrap.style.position = st.videoWrap.style.position || "relative";
  st.videoWrap.appendChild(wrap);

  // Status bar sibling ABOVE the video. ONE horizontal row, three
  // logical zones separated by dot dividers:
  //   LEFT  - Model / Camera / CPU / RAM / GPU (system + selected cam)
  //   MID   - LIVE tag + People / Vehicles / Tick / Alerts (KPIs)
  //   RIGHT - Draw line (only for drawable layers) + Stop
  // Everything on one line, aligned baseline. Wraps only when the
  // viewport is genuinely too narrow to keep everything inline.
  const bar = document.createElement("div");
  bar.className = "tile-status-bar";
  bar.style.cssText =
    "display:flex;flex-wrap:wrap;align-items:center;gap:6px 14px;"
    + "padding:9px 14px;margin:8px 0;background:linear-gradient(90deg,"
    + "#0b1220 0%,#111a2e 100%);border:1px solid #1e293b;"
    + "border-radius:10px;font:12.5px/1.4 system-ui,sans-serif;"
    + "color:#e2e8f0;box-shadow:0 2px 8px rgba(0,0,0,0.35)";
  const _pill = "background:#0f172a;border:1px solid #1e293b;"
              + "border-radius:999px;padding:3px 9px;white-space:nowrap;"
              + "display:inline-flex;align-items:baseline;gap:5px";
  const _lbl = "color:#94a3b8;font-size:11px;letter-spacing:0.02em";
  bar.innerHTML = `
    <span class="analysis-live-tag" style="padding:4px 11px;
       background:linear-gradient(90deg,#2563eb,#3b82f6);color:#f8fafc;
       border-radius:999px;font-size:12px;font-weight:700;
       letter-spacing:0.03em;box-shadow:0 1px 4px rgba(37,99,235,0.5)">
       LIVE - ${escapeHtml(_layerLabel[layer] || layer)}</span>
    <span style="${_pill}"><span style="${_lbl}">Model</span>
       <b class="hud-model" style="color:#cbd5e1">-</b></span>
    <span style="${_pill}"><span style="${_lbl}">Cam</span>
       <b class="hud-cam" style="color:#cbd5e1;max-width:180px;
       overflow:hidden;text-overflow:ellipsis"
       title="${escapeHtml(cam)}">${escapeHtml(cam)}</b></span>
    <span style="${_pill}"><span style="${_lbl}">CPU</span>
       <b class="hud-cpu" style="color:#fbbf24">-</b></span>
    <span style="${_pill}"><span style="${_lbl}">RAM</span>
       <b class="hud-ram" style="color:#fbbf24">-</b></span>
    <span style="${_pill}"><span style="${_lbl}">GPU</span>
       <b class="hud-gpu" style="color:#a5f3fc">N/A</b></span>
    <span style="flex:1"></span>
    <span style="${_pill}"><span style="${_lbl}">People</span>
       <b class="hud-people" style="color:#4ade80">-</b></span>
    <span style="${_pill}"><span style="${_lbl}">Vehicles</span>
       <b class="hud-veh" style="color:#60a5fa">-</b></span>
    <span style="${_pill}"><span style="${_lbl}">Tick</span>
       <b class="hud-tick" style="color:#fbbf24">-</b></span>
    <span style="${_pill}" title="Per-stage tick time (ms): grab / infer / render / publish"><span style="${_lbl}">Stages ms</span>
       <b class="hud-stages" style="color:#c4b5fd">-</b></span>
    <span style="${_pill}"><span style="${_lbl}">Alerts</span>
       <b class="hud-alerts" style="color:#f97316">0</b></span>
    <span class="analysis-status" style="color:#64748b;font-size:11px;
       font-style:italic">starting live analysis...</span>
    <button class="analysis-drawline" style="padding:6px 14px;
       background:#2563eb;color:#f8fafc;border:0;border-radius:6px;
       cursor:pointer;font-size:12.5px;font-weight:600;
       display:${DRAWABLE_LAYERS[layer] ? "" : "none"}">
       ${DRAWABLE_LAYERS[layer] || "Draw"}</button>
    <button class="analysis-stop" style="padding:6px 14px;
       background:#dc2626;color:#f8fafc;border:0;border-radius:6px;
       cursor:pointer;font-size:12.5px;font-weight:600">Stop</button>`;
  st.videoWrap.insertAdjacentElement("beforebegin", bar);
  st.analysisBar = bar;
  bar.querySelector(".analysis-stop").addEventListener("click",
    () => stopTileAnalysis(st));
  bar.querySelector(".analysis-drawline").addEventListener("click", () => {
    const snap =
      `/api/analysis/frame?cam=${encodeURIComponent(cam)}&_=${Date.now()}`;
    const lay = st.analysis ? st.analysis.layer : layer;
    if (lay === "line") window.openLineEditor(cam, snap);
    else openZoneEditor(cam, lay, snap);
  });
  st.analysis = {
    cam, layer,
    wrap,
    bar,
    bg: wrap.querySelector(".analysis-bg"),
    canvas: wrap.querySelector(".analysis-canvas"),
    status: bar.querySelector(".analysis-status"),
    lastBgUrl: null,
    tickBuf: [],
    failures: 0, lastRestart: 0, inflight: false, lastSeq: -1,
    evSeen: new Set(),
    evTimer: setInterval(() => pollAnalysisEvents(st), 2500),
    timer: setInterval(() => pollAnalysisFrame(st), ANALYSIS_POLL_MS),
    videoStateTimer: setInterval(() => _syncAnalysisBgVisibility(st), 500),
    sysTimer: setInterval(() => _pollSystemLive(st), 3000),
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
  // Hot-trail decay (decision D2, 2026-08-23): a chip is a ROLLING
  // recent-events trail, not an archive - the gallery keeps saved ones.
  // Fade after EV_CHIP_FADE_S, drop entirely after EV_CHIP_TTL_S.
  const nowS = Date.now() / 1000;
  for (const c of a.evStrip.querySelectorAll(".event-chip")) {
    const ts = Number(c.dataset.ts || 0);
    if (!ts) continue;
    const age = nowS - ts;
    if (age > EV_CHIP_TTL_S) c.remove();
    else c.style.opacity = age > EV_CHIP_FADE_S ? "0.35" : "";
  }
  const cur = a.actualLayer || a.layer;
  let visible = 0;
  // 2026-08-23 (B6): obstruction events fire in live_analysis regardless
  // of the active layer (>=50% frame coverage, conf>=0.45) - they were
  // being hidden here because "obstruction" is not in ANALYSIS_LAYER_DEFS.
  // Treat obstruction as always-visible so operators actually see them.
  const CROSS_LAYER_EVENTS = new Set(["obstruction"]);
  for (const c of a.evStrip.querySelectorAll(".event-chip")) {
    const lay = c.dataset.layer;
    const show = (!lay || lay === cur || CROSS_LAYER_EVENTS.has(lay));
    c.style.display = show ? "" : "none";
    if (show) visible += 1;
  }
  try { _updateAnalysisHudAlerts(a, visible); }
  catch (_) { /* HUD is decorative */ }
}

// Hot-trail tuning (decision D2): chips fade at 60s and vanish at 90s;
// a fresh arrival pulses once so the operator's eye is drawn to it.
const EV_CHIP_FADE_S = 60;
const EV_CHIP_TTL_S = 90;
(() => {
  const st = document.createElement("style");
  st.textContent = `@keyframes evPulse {
      0% { box-shadow: 0 0 0 0 rgba(37,99,235,.75); }
    100% { box-shadow: 0 0 0 12px rgba(37,99,235,0); } }
  .event-chip.ev-new { animation: evPulse 1.2s ease-out 2; }`;
  document.head.appendChild(st);
})();

function _eventChip(a, ev) {
  const chip = document.createElement("div");
  chip.className = "event-chip ev-new";
  chip.dataset.layer = ev.layer || "";
  chip.dataset.ts = String(ev.ts || "");
  chip.addEventListener("animationend", () => chip.classList.remove("ev-new"),
                        { once: true });
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
      ${ev.saved ? "saved" : "save"}</button>
    <button class="event-replay" title="replay the seconds around this event"
            style="background:#334155;color:#e2e8f0;border:0;border-radius:5px;
                   padding:4px 7px;cursor:pointer;font-size:11px;flex:0 0 auto">
      replay</button>`;
  chip.querySelector(".event-replay").addEventListener("click",
      () => openReplayModal(a.cam, ev.ts));
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

async function openReplayModal(cam, ts) {
  // D3 (2026-08-23): minimal in-page player over the last-15s annotated
  // ring - frames come as base64 JPEGs at the server's replay fps.
  let d;
  try {
    const r = await fetch(`/api/analysis/replay?cam=${encodeURIComponent(cam)}`
                          + (ts ? `&ts=${ts}` : ""));
    d = await r.json();
    if (!r.ok || !d.ok) throw new Error(d.error || r.status);
  } catch (e) {
    alert("replay unavailable: " + e.message);
    return;
  }
  const frames = d.frames || [];
  if (!frames.length) {
    alert("replay ring is empty for this moment (it holds ~15s)");
    return;
  }
  const ov = document.createElement("div");
  ov.style.cssText = "position:fixed;inset:0;background:rgba(2,6,17,.88);" +
    "z-index:9999;display:flex;flex-direction:column;align-items:center;" +
    "justify-content:center;gap:10px";
  ov.innerHTML = `
    <img style="max-width:92vw;max-height:82vh;border-radius:8px;
                border:1px solid #1e293b">
    <div style="color:#94a3b8;font-size:13px">
      <span class="rp-pos"></span> - click anywhere to close</div>`;
  const img = ov.querySelector("img");
  const pos = ov.querySelector(".rp-pos");
  let i = 0, timer = null;
  const step = () => {
    const f = frames[i % frames.length];
    img.src = "data:image/jpeg;base64," + f.jpeg;
    const t = new Date(f.ts * 1000);
    const p = (n) => String(n).padStart(2, "0");
    pos.textContent = `frame ${(i % frames.length) + 1}/${frames.length}` +
      ` @ ${p(t.getHours())}:${p(t.getMinutes())}:${p(t.getSeconds())}`;
    i += 1;
  };
  step();
  timer = setInterval(step, 1000 / (d.fps || 2));
  ov.addEventListener("click", () => { clearInterval(timer); ov.remove(); });
  document.body.appendChild(ov);
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
  const order = ["plates", "line", "fire", "parking", "gestures",
                 "body", "pose", "faces", "heat", "paths"];
  items.sort((a, b) => order.indexOf(a.layer) - order.indexOf(b.layer)
                       || (b.ts || 0) - (a.ts || 0));
  wrap.innerHTML = items.map((it) => {
    const t = new Date((it.ts || 0) * 1000);
    const idSafe = escapeHtml(String(it.id || ""));
    return `<figure data-saved-id="${idSafe}" style="margin:0;position:relative;
        background:#0c0e13;border:1px solid #232733;border-radius:6px;
        overflow:hidden">
      <button data-saved-delete="${idSafe}" title="Delete this saved crop"
        style="position:absolute;top:4px;right:4px;z-index:2;width:22px;
        height:22px;padding:0;line-height:20px;border:0;border-radius:11px;
        background:rgba(2,6,23,.72);color:#fca5a5;font-size:14px;
        cursor:pointer">×</button>
      <a href="#" data-pipeline-expand="${idSafe}">
        <img src="${it.image}" alt="" loading="lazy"
             style="width:100%;height:96px;object-fit:cover;display:block"></a>
      <figcaption style="padding:5px 6px">
        <div style="font-size:10px;color:#e7e9ee;white-space:nowrap;
             overflow:hidden;text-overflow:ellipsis"
             title="${escapeHtml(it.text)}">${escapeHtml(it.text)}</div>
        <div style="font-size:9px;color:#8b909a;white-space:nowrap;
             overflow:hidden;text-overflow:ellipsis">
          ${escapeHtml(it.layer)} - ${t.toLocaleTimeString()}</div>
      </figcaption>
    </figure>`;
  }).join("");
  // Wire per-tile delete
  wrap.querySelectorAll("[data-saved-delete]").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.preventDefault(); e.stopPropagation();
      const id = btn.getAttribute("data-saved-delete");
      if (!id) return;
      btn.disabled = true; btn.style.opacity = "0.4";
      try {
        const r = await fetch(
          `/api/analysis/saved-delete?id=${encodeURIComponent(id)}`,
          { method: "POST" });
        if (!r.ok) throw new Error("HTTP " + r.status);
        renderGallery();
      } catch (err) {
        btn.disabled = false; btn.style.opacity = "1";
        btn.textContent = "!";
      }
    });
  });
  // Wire per-tile expand -> 4-stage pipeline modal
  wrap.querySelectorAll("[data-pipeline-expand]").forEach((a) => {
    a.addEventListener("click", (e) => {
      e.preventDefault();
      const id = a.getAttribute("data-pipeline-expand");
      const it = items.find((x) => String(x.id) === id);
      if (it) openPipelineModal(it);
    });
  });
}

// 4-stage LPR pipeline viewer: click a saved plate in the gallery to
// open a modal that walks the crop through Vehicle -> Plate -> Enhanced
// -> Final. Stages B/C/D are computed client-side from the same saved
// annotated frame; no additional backend calls beyond the existing
// image asset.
function openPipelineModal(it) {
  const bg = document.createElement("div");
  bg.style.cssText =
    "position:fixed;inset:0;background:rgba(2,6,23,.82);z-index:80;" +
    "display:flex;align-items:center;justify-content:center;padding:20px;" +
    "box-sizing:border-box";
  const box = document.createElement("div");
  box.style.cssText =
    "background:#0f172a;border:1px solid #334155;border-radius:14px;" +
    "max-width:1080px;width:100%;max-height:92vh;color:#e2e8f0;" +
    "display:flex;flex-direction:column;overflow:hidden";
  const rawText = String(it.text || "").replace(/^plate:?\s*/i, "");
  box.innerHTML = `
    <div style="padding:14px 20px;flex:0 0 auto;border-bottom:1px solid #1e293b;
                display:flex;align-items:center;gap:14px">
      <div style="flex:1">
        <div style="font-weight:700;font-size:16px">
          LPR pipeline &mdash; <span style="color:#93c5fd">
            ${escapeHtml(rawText)}</span>
        </div>
        <div style="font-size:12px;color:#94a3b8;margin-top:2px">
          ${escapeHtml(it.layer || "plates")} &middot;
          ${escapeHtml(it.cam_name || it.cam || "")} &middot;
          ${new Date((it.ts || 0) * 1000).toLocaleString()}
        </div>
      </div>
      <label style="font-size:12px;color:#94a3b8;
                   display:flex;align-items:center;gap:6px;cursor:pointer">
        <input type="checkbox" data-lpr-enhance>
        Enhance plate (contrast &middot; brightness)
      </label>
      <button data-lpr-close style="background:#334155;color:#e2e8f0;border:0;
              border-radius:8px;padding:6px 14px;cursor:pointer">Close</button>
    </div>
    <div style="padding:16px 20px;overflow:auto;display:grid;
                grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
                gap:14px">
      ${["A. Vehicle crop", "B. Plate crop",
         "C. Enhanced plate", "D. Final view"].map((title, i) => `
        <div style="background:#0b1223;border:1px solid #1e293b;
                    border-radius:10px;overflow:hidden;
                    display:flex;flex-direction:column">
          <div style="padding:8px 12px;font-size:12px;color:#94a3b8;
                     background:#0a0f1c;border-bottom:1px solid #1e293b">
            ${title}
          </div>
          <div style="flex:1;position:relative;background:#000;min-height:170px">
            <canvas data-lpr-stage="${i}"
              style="display:block;width:100%;height:auto"></canvas>
          </div>
        </div>`).join("")}
    </div>`;
  bg.appendChild(box);
  document.body.appendChild(bg);
  bg.addEventListener("click", (e) => { if (e.target === bg) bg.remove(); });
  box.querySelector("[data-lpr-close]").addEventListener("click",
    () => bg.remove());
  const stageEls = box.querySelectorAll("[data-lpr-stage]");
  const enhanceInput = box.querySelector("[data-lpr-enhance]");
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => paintStages(img, stageEls, enhanceInput.checked);
  enhanceInput.addEventListener("change",
    () => paintStages(img, stageEls, enhanceInput.checked));
  img.src = it.image;
  // 2026-08-23 (C2): try to fetch the REAL per-attempt plate crops
  // saved during the OCR pass. When found, overlay panels B/C/D with
  // actual plate crops (with the OCR text baked in as a caption bar).
  // Failure is soft: the CSS-cropped fallback above still renders.
  (async () => {
    if (!it || !it.cam) return;
    try {
      const params = new URLSearchParams({ cam: it.cam, limit: "5" });
      if (it.ts) params.set("ts", String(it.ts));
      const r = await fetch(`/api/analysis/plate-crops?${params}`,
                            { cache: "no-store" });
      if (!r.ok) return;
      const j = await r.json();
      const items = (j && j.items) || [];
      if (!items.length) return;
      const label = box.querySelector("[data-lpr-close]").previousElementSibling;
      const captionRow = document.createElement("div");
      captionRow.style.cssText =
        "grid-column:1/-1;padding:6px 10px;background:#0a0f1c;" +
        "border:1px solid #1e293b;border-radius:8px;color:#94a3b8;" +
        "font-size:11px;line-height:1.4";
      captionRow.textContent =
        `Real plate crops (src/data/plate_crops): ${items.length} sample(s) ` +
        `for this detection. Panels B/C/D now show the actual crops the ` +
        `OCR pass saw instead of a CSS crop of the saved event frame.`;
      const grid = box.querySelector(
        "div[style*='grid-template-columns']");
      if (grid) grid.prepend(captionRow);
      const overlayCrop = (idx, url) => {
        const cvs = stageEls[idx];
        if (!cvs || !url) return;
        const im = new Image();
        im.crossOrigin = "anonymous";
        im.onload = () => {
          const w = im.naturalWidth || 1;
          const h = im.naturalHeight || 1;
          cvs.width = w; cvs.height = h;
          cvs.style.aspectRatio = `${w}/${h}`;
          const ctx = cvs.getContext("2d");
          ctx.filter = "none";
          ctx.drawImage(im, 0, 0);
        };
        im.src = url;
      };
      // Panel B = newest crop. Panel C = second-newest. Panel D =
      // third-newest OR a 2x-upscaled newest for eyeball comparison.
      overlayCrop(1, items[0] && items[0].url);
      overlayCrop(2, (items[1] || items[0]) && (items[1] || items[0]).url);
      // Panel D: pick the crop with the HIGHEST conf_pct (best read).
      const best = items.slice().sort(
        (a, b) => (b.conf_pct || 0) - (a.conf_pct || 0))[0];
      overlayCrop(3, best && best.url);
    } catch (_) { /* silent: fallback stays visible */ }
  })();
}

function paintStages(img, canvases, enhance) {
  // Stage A - full annotated frame. Stages B/D zoom into the bottom-
  // center 40% (where the plate + caption sit) since the backend saved
  // an annotated crop centred on the vehicle. Stage C = Stage B with
  // contrast/brightness boost applied via canvas filter (equivalent to
  // the ESPCN + CLAHE preprocessing the OCR pass runs internally).
  const iw = img.naturalWidth, ih = img.naturalHeight;
  if (!iw || !ih) return;
  const setSize = (c, w, h) => {
    c.width = w; c.height = h;
    c.style.aspectRatio = `${w}/${h}`;
  };
  // Stage A: full image
  const cA = canvases[0];
  setSize(cA, iw, ih);
  cA.getContext("2d").drawImage(img, 0, 0);
  // Stages B/C/D: crop bottom-center where plate lives.
  // For a typical vehicle crop the plate + caption sit in the bottom
  // 45%. Take the FULL width but crop the top 55%.
  const bx = 0, by = Math.round(ih * 0.30);
  const bw = iw, bh = Math.max(1, ih - by);
  const drawCropped = (c, filter) => {
    setSize(c, bw, bh);
    const ctx = c.getContext("2d");
    ctx.filter = filter || "none";
    ctx.drawImage(img, bx, by, bw, bh, 0, 0, bw, bh);
    ctx.filter = "none";
  };
  drawCropped(canvases[1], "none");
  const enhFilter = enhance
    ? "contrast(1.6) brightness(1.1) saturate(1.2)"
    : "contrast(1.25) brightness(1.05)";
  drawCropped(canvases[2], enhFilter);
  // Stage D: same as C but rendered larger (2x pixel density) for
  // eyeball verification of digit shapes.
  const c = canvases[3];
  setSize(c, bw * 2, bh * 2);
  const ctx = c.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.filter = enhFilter;
  ctx.drawImage(img, bx, by, bw, bh, 0, 0, bw * 2, bh * 2);
  ctx.filter = "none";
}
renderGallery();
setInterval(renderGallery, 20000);

// Clear all saved detections - one confirm, then wipe the gallery.
(function initGalleryClear() {
  const btn = document.getElementById("gallery-clear");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    if (!confirm("Delete every saved detection crop from this machine? "
                 + "This cannot be undone.")) return;
    btn.disabled = true;
    const orig = btn.textContent;
    btn.textContent = "clearing...";
    try {
      const r = await fetch("/api/analysis/saved-clear", { method: "POST" });
      const d = await r.json();
      if (!r.ok) throw new Error(d && d.error || ("HTTP " + r.status));
      btn.textContent = `cleared ${d.removed || 0}`;
    } catch (e) {
      btn.textContent = "err: " + e.message;
    }
    setTimeout(() => { btn.textContent = orig; btn.disabled = false; }, 2000);
    renderGallery();
  });
})();

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
  // Layer-switching placeholder (2026-08-16): the server keeps publishing
  // the OLD layer for a few ticks after the client asks for a new one
  // (Analyze picker sends the switch, the current in-flight tick still
  // uses the old layer). Drawing the old layer's overlay while the LIVE
  // tag says "switching to X..." misled the operator ("why are skeletons
  // showing when I picked paths"). When the server's layer disagrees
  // with what the client wants, clear the canvas and paint a small
  // placeholder instead of the stale layer's geometry.
  const latestTick = buf[buf.length - 1];
  if (latestTick && a.layer && latestTick.layer
      && latestTick.layer !== a.layer) {
    const ctx = a.canvas.getContext("2d");
    const rect = a.canvas.parentElement.getBoundingClientRect();
    const cw = Math.max(1, Math.round(rect.width));
    const ch = Math.max(1, Math.round(rect.height));
    if (a.canvas.width !== cw)  a.canvas.width = cw;
    if (a.canvas.height !== ch) a.canvas.height = ch;
    ctx.clearRect(0, 0, cw, ch);
    ctx.font = "14px system-ui, sans-serif";
    const msg = `switching to ${_layerLabel[a.layer] || a.layer}...`;
    const tw = ctx.measureText(msg).width + 20;
    ctx.fillStyle = "rgba(15,23,42,0.75)";
    ctx.fillRect((cw - tw) / 2, ch / 2 - 16, tw, 32);
    ctx.fillStyle = "#f8fafc";
    ctx.fillText(msg, (cw - tw) / 2 + 10, ch / 2 + 4);
    return;
  }
  let merged = latestTick;
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
  const EXTRAP_MAX_S = 3.0;         // was 1.5; wider window keeps boxes on
                                    // screen when the backend tick is slow
                                    // on CPU-only hosts (5-10 s per tick)
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

// Screen-capture fallback removed 2026-08-18 - it burned whatever was
// on the operator's desktop into frames instead of the actual video.
// Kept as a no-op stub in case any lingering caller still references it.
async function _postScreenCaptureBbox(_st) {
  return;
}

// Premium HUD sidebar (2026-08-17). Reads from the tick payload +
// backend /api/system chip so the operator sees the live pulse of the
// analysis (people, vehicles, tick rate, alert count, active model
// backend, camera id) without opening a dev console. Non-invasive:
// the HUD lives INSIDE the analysis-wrap overlay, so it appears only
// while a session is live and vanishes as soon as Stop is clicked.
let _hudSystemInfo = null;
let _hudSystemInflight = false;

function _hudLoadSystemInfo() {
  if (_hudSystemInfo !== null || _hudSystemInflight) return;
  _hudSystemInflight = true;
  fetch("/api/system", { cache: "no-store" })
    .then((r) => (r.ok ? r.json() : null))
    .then((j) => {
      _hudSystemInflight = false;
      if (!j) return;
      _hudSystemInfo = j;
      // Refresh any live HUD tiles with the model chip we just learned.
      for (const key of Object.keys(tileState || {})) {
        const st = tileState[key];
        const a = st && st.analysis;
        if (!a || !a.wrap) continue;
        const m = (a.bar || a.wrap).querySelector(".hud-model");
        if (m) m.textContent = _hudSystemInfo.model || "-";
      }
      // Populate the top-right capability chip once - color the
      // background by backend so an OpenVINO GPU install jumps out
      // relative to plain CPU torch. The chip stays hidden when
      // /api/system fails so a stale label is never displayed.
      const chip = document.getElementById("backend-chip");
      if (chip) {
        chip.textContent = _hudSystemInfo.model || _hudSystemInfo.backend;
        chip.title = "device: " + (_hudSystemInfo.device || "?");
        chip.style.display = "";
        const bk = _hudSystemInfo.backend;
        const dev = _hudSystemInfo.device || "";
        if (bk === "openvino" && dev.startsWith("GPU")) {
          chip.style.background = "#052e16";  // green - fast + accelerated
          chip.style.borderColor = "#166534";
          chip.style.color = "#bbf7d0";
        } else if (bk === "openvino") {
          chip.style.background = "#082f49";  // blue - fast CPU path
          chip.style.borderColor = "#075985";
          chip.style.color = "#bae6fd";
        } else {
          chip.style.background = "#292524";  // amber - plain torch
          chip.style.borderColor = "#78350f";
          chip.style.color = "#fed7aa";
        }
        // Screen-capture indicator: only rendered when the backend has
        // SCREEN_CAPTURE_FALLBACK=1. Operators need to know the analysis
        // is running on desktop pixels (not YouTube stream) so they keep
        // the browser tab with the iframe visible on the primary display.
        const sc = _hudSystemInfo.screen_capture;
        let scChip = document.getElementById("sc-chip");
        if (sc && sc.enabled) {
          if (!scChip) {
            scChip = document.createElement("div");
            scChip.id = "sc-chip";
            scChip.className = "sub";
            scChip.style.cssText =
              "display:inline-block;margin-left:8px;padding:2px 10px;" +
              "border-radius:999px;font-size:12px;background:#3f1d1d;" +
              "border:1px solid #7f1d1d;color:#fecaca;cursor:help";
            chip.parentNode.insertBefore(scChip, chip.nextSibling);
          }
          scChip.textContent = sc.bbox
            ? "SCREEN CAPTURE (bbox)"
            : "SCREEN CAPTURE (full display)";
          scChip.title = sc.bbox
            ? `capturing bbox ${sc.bbox.join(",")} - keep the browser iframe visible on the primary display`
            : "capturing the primary display fully - keep the browser iframe visible";
          scChip.style.display = "";
        } else if (scChip) {
          scChip.style.display = "none";
        }
      }
    })
    .catch(() => { _hudSystemInflight = false; });
}
_hudLoadSystemInfo();

// Header line updater. The "Model: loading..." placeholder used to be
// swapped out by the removed RL script; without a replacement it stayed
// forever. Now we pull the scoreboard from /api/model-metrics and format
// a plain one-liner. Refreshes every 10 s.
function _refreshHeaderMetricsLine() {
  const el = document.getElementById("model-metrics-line");
  if (!el) return;
  fetch("/api/model-metrics", { cache: "no-store" })
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => {
      if (!d) return;
      const line = d.header_line
        || `reviews: ${d.reviews || 0}`;
      const backend = (_hudSystemInfo && _hudSystemInfo.model)
        ? ` - detector: ${_hudSystemInfo.model}` : "";
      el.textContent = line + backend;
    })
    .catch(() => {
      // On failure show the detector chip at least, don't stay stuck.
      if (_hudSystemInfo && _hudSystemInfo.model) {
        el.textContent = "detector: " + _hudSystemInfo.model;
      } else {
        el.textContent = "model info unavailable";
      }
    });
}
_refreshHeaderMetricsLine();
setInterval(_refreshHeaderMetricsLine, 10000);

function _updateAnalysisHud(a, d) {
  if (!a || !(a.bar || a.wrap)) return;
  const root = (a.bar || a.wrap);
  const p = root.querySelector(".hud-people");
  const v = root.querySelector(".hud-veh");
  const t = root.querySelector(".hud-tick");
  const s = root.querySelector(".hud-stages");
  const m = root.querySelector(".hud-model");
  if (p) p.textContent = String(d.person ?? "-");
  // Decision 13 (2026-08-23): the second pill follows the ACTIVE layer
  // instead of always saying "Vehicles" - on the faces layer a vehicle
  // count reads as noise (the audit's "vehicles on faces" confusion was
  // exactly this text). Label element = the pill's <span> before the
  // value <b>.
  if (v) {
    let lbl = "Vehicles", val = d.vehicles ?? "-";
    if (d.layer === "faces") {
      lbl = "Faces"; val = (d.faces || []).length;
    } else if (d.layer === "plates") {
      lbl = "Plates";
      val = (d.boxes || []).filter((b) => b.plate).length;
    } else if (d.layer === "fire") {
      lbl = "Fire";
      const fh = d.fire && (d.fire.hits || d.fire);
      val = Array.isArray(fh) ? fh.length : 0;
    } else if (d.layer === "body") {
      lbl = "Flagged";
      val = (d.boxes || []).filter((b) => b.alert).length;
    }
    v.textContent = String(val);
    const lblEl = v.previousElementSibling;
    if (lblEl && lblEl.textContent !== lbl) lblEl.textContent = lbl;
  }
  if (t) {
    // Prefer the smoothed inter-tick gap the client already tracks
    // (a._gapEma); fall back to a per-tick delta when only two ticks
    // exist. Rendered as SECONDS with one decimal - "0.9s" reads more
    // clearly than the FPS reciprocal on a CPU-only pipeline where a
    // tick often runs 0.5-4s apart.
    const gap = a._gapEma || null;
    t.textContent = gap ? gap.toFixed(1) + "s / tick" : "-";
  }
  // 2026-08-23 (B2): per-stage tick ms from the backend, so operators
  // can see WHICH stage is the bottleneck instead of a single number.
  // Format: "grab / infer / render / publish" (ms each, no decimals).
  if (s) {
    const sm = d._stage_ms;
    if (sm && typeof sm === "object") {
      const fmt = (x) => x == null ? "-" : Math.round(x);
      s.textContent = `${fmt(sm.grab)} / ${fmt(sm.infer)} / ${fmt(sm.render)} / ${fmt(sm.publish)}`;
    } else {
      s.textContent = "-";
    }
  }
  if (m && _hudSystemInfo && _hudSystemInfo.model) {
    m.textContent = _hudSystemInfo.model;
  }
}

function _updateAnalysisHudAlerts(a, count) {
  if (!a || !(a.bar || a.wrap)) return;
  const el = (a.bar || a.wrap).querySelector(".hud-alerts");
  if (el) el.textContent = String(count);
}

// Poll CPU/RAM/GPU utilisation into the status-bar HUD above the
// player. Silent-on-failure - stale numbers just linger, they don't
// break the analysis loop.
function _pollSystemLive(st) {
  const a = st && st.analysis;
  if (!a || !a.bar) return;
  fetch("/api/system/live", { cache: "no-store" })
    .then((r) => (r.ok ? r.json() : null))
    .then((j) => {
      if (!j || !a.bar || !a.bar.isConnected) return;
      const cpu = a.bar.querySelector(".hud-cpu");
      const ram = a.bar.querySelector(".hud-ram");
      const gpu = a.bar.querySelector(".hud-gpu");
      if (cpu) cpu.textContent = (j.cpu_pct != null)
        ? j.cpu_pct.toFixed(0) + "%" : "-";
      if (ram) ram.textContent = (j.ram_pct != null)
        ? `${j.ram_pct.toFixed(0)}% (${j.ram_used_gb}/${j.ram_total_gb}GB)`
        : "-";
      if (gpu) {
        if (j.gpus && j.gpus.length) {
          const g = j.gpus[0];
          if (g.util_pct != null && g.mem_used_gb != null) {
            gpu.textContent = `${g.util_pct.toFixed(0)}% `
              + `(${g.mem_used_gb}/${g.mem_total_gb}GB)`;
          } else {
            // OpenVINO / WMIC: name only, no util numbers
            const short = (g.name || "GPU")
              .replace(/\(R\)/g, "").replace(/\(TM\)/g, "")
              .replace(/Intel /i, "").trim();
            gpu.textContent = short.slice(0, 32);
          }
          gpu.title = g.name + " (via " + (g.source || "?") + ")";
        } else {
          gpu.textContent = "N/A";
          gpu.title = "no GPU detected";
        }
      }
    })
    .catch(() => {});
}

function _syncAnalysisBgVisibility(st) {
  const a = st.analysis;
  if (!a || !a.bg) return;

  // 2026-08-17: heat is a special case - the colormap is BAKED into the
  // backend JPEG by draw_heat_layer(). Trying to render it on the canvas
  // overlay from JSON on top of the YouTube iframe hits three problems
  // at once: (a) the JSON heat is empty until the backend has accumulated
  // for a few ticks, (b) the iframe covers the JPEG entirely on YouTube
  // streams, (c) double-rendering (canvas + JPEG) produces a subtly
  // wrong result on non-YouTube. Solution: force the JPEG visible on
  // heat regardless of video state, hide the canvas overlay entirely.
  // The operator sees the authoritative backend-rendered heat frame.
  if (a.layer === "heat") {
    if (a.bg.style.display !== "block") {
      a.bg.style.display = "block";
      _refreshAnalysisBg(a);
    }
    if (a.canvas.style.display !== "none") {
      a.canvas.style.display = "none";
    }
    return;
  }

  let playing = false;
  // YouTube iframe (yt-dlp-independent playback): assume it's playing so
  // the backend JPEG fallback stays hidden and the canvas overlay draws
  // boxes on top of the iframe video the operator actually sees.
  const yt = st.videoWrap.querySelector("iframe[data-youtube]");
  if (yt) {
    playing = true;
  } else {
    const v = st.videoWrap.querySelector("video");
    if (v && !v.paused && !v.ended && v.readyState >= 2) {
      const t = v.currentTime;
      playing = (a._lastVidT !== undefined) && (t > a._lastVidT + 0.05);
      a._lastVidT = t;
    } else {
      a._lastVidT = undefined;
    }
  }
  const want = playing ? "none" : "block";
  if (a.bg.style.display !== want) {
    a.bg.style.display = want;
    a.canvas.style.display = playing ? "" : "none";
    if (want === "block") _refreshAnalysisBg(a);
  }
  // 2026-08-23 (B3): stream-stalled ribbon. If no new analysis payload
  // in > 3s (backend cache is looping on _LAST_GOOD_FRAME, or the
  // screen-capture branch is silently reusing the same pixels), show
  // an obvious red bar over the video so operators do not stare at a
  // "live" chip while the pipeline is dead.
  const STALE_MS = 3000;
  const fresh = a._lastFreshMs || 0;
  const stale = fresh > 0 && (Date.now() - fresh) > STALE_MS;
  let ribbon = st.videoWrap.querySelector(".analysis-stall-ribbon");
  if (stale) {
    if (!ribbon) {
      ribbon = document.createElement("div");
      ribbon.className = "analysis-stall-ribbon";
      ribbon.style.cssText =
        "position:absolute;top:0;left:0;right:0;padding:8px 12px;" +
        "background:rgba(220,38,38,0.92);color:#fff;font:600 13px system-ui;" +
        "text-align:center;z-index:70;pointer-events:none;";
      st.videoWrap.style.position = st.videoWrap.style.position || "relative";
      st.videoWrap.appendChild(ribbon);
    }
    const ageS = Math.round((Date.now() - fresh) / 1000);
    ribbon.textContent =
      `Stream stalled - no fresh frame in ${ageS}s ` +
      `(check backend or reload the tab)`;
  } else if (ribbon) {
    ribbon.remove();
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
      const liveTag = (st.analysisBar || st.videoWrap)
        .querySelector(".analysis-live-tag");
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
        // 2026-08-23 (B3): record wall-clock of the last real payload
        // update; a client-side check below uses this to show a
        // "Stream stalled" ribbon when the analysis loop stops
        // producing new frames.
        a._lastFreshMs = Date.now();
        const prevTick = a.tickBuf[a.tickBuf.length - 1];
        if (prevTick) {
          const gap = (d.at || 0) - (prevTick.at || 0);
          if (gap > 0.2 && gap < 60) {
            a._gapEma = a._gapEma ? 0.7 * a._gapEma + 0.3 * gap : gap;
          }
        }
        a.tickBuf.push(d);
        if (a.tickBuf.length > 24) a.tickBuf.shift();
        // Heat: bg is always visible (see _syncAnalysisBgVisibility) so
        // refresh unconditionally to pick up the newest backend-baked
        // colormap frame. Other layers: only refresh when bg is visible.
        if (a.layer === "heat"
            || (a.bg && a.bg.style.display !== "none")) {
          _refreshAnalysisBg(a);
        }
        // Feed the tile's activity + anomaly pills from the live-analysis
        // tick so the "-/10" placeholder becomes a real reading whenever a
        // session is running. Previously only pollLocalModelView (which
        // reads model_view/<cam>.json - only the notebook Section 7 writes
        // that file) updated these, so a standalone-dashboard operator
        // stared at "-/10" forever.
        try {
          _updateLocalTileBadges(a.cam, {
            counts: { person: d.person, vehicles: d.vehicles },
            at: d.at,
            dark: d.dark,
            obstructed: d.obstructed,
          });
        } catch (_) { /* pills are decorative - never break the poll */ }
        try { _updateAnalysisHud(a, d); }
        catch (_) { /* HUD is decorative - never break the poll */ }
        if (d.layer === "fire" && d.fire) {
          const confirmed = !!d.fire.confirmed;
          const prev = !!a._fireConfirmed;
          if (confirmed && !prev) {
            const n = (d.fire.hits || []).length;
            showCrossToast(`FIRE DETECTED - ${n} region(s)`);
            const t = st.tile;
            if (t) {
              t.style.outline = "3px solid #f97316";
              setTimeout(() => { t.style.outline = ""; }, 3000);
            }
          }
          a._fireConfirmed = confirmed;
        }
        if (d.layer === "parking" && Array.isArray(d.spots)) {
          // Occupancy-transition toasts + tile flash. Loiter alerts fire
          // on sustained presence; parking fires on the SPOT'S state
          // flipping (empty <-> filled), so a spot that frees up (which
          // is often the operationally interesting event: "space #3 just
          // opened") gets its own toast instead of only living in the
          // event feed.
          a._parkState = a._parkState || {};
          for (const z of d.spots) {
            const cur = !!z.occupied;
            const prev = a._parkState[z.name];
            if (prev !== undefined && prev !== cur) {
              const msg = cur
                ? `${z.name} occupied${z.cls ? " (" + z.cls + ")" : ""}`
                : `${z.name} just freed up`;
              showCrossToast(msg);
              const t = st.tile;
              if (t) {
                const col = cur ? "#ef4444" : "#22c55e";
                t.style.outline = `3px solid ${col}`;
                setTimeout(() => { t.style.outline = ""; }, 2000);
              }
            }
            a._parkState[z.name] = cur;
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
    // Faithful port of the original heatmap.overlay() (Turkey-era, the
    // one the operator explicitly wants back): raw dwell grid -> sqrt
    // tone curve -> bilinear resize -> Gaussian blur -> TURBO colormap
    // -> signal-modulated alpha blend. The empty street stays a photo;
    // only where activity accumulated does color bloom in.
    //
    // 2026-08-16 tuning per operator ("peak stays ~0 in d.heat"):
    //   * peak = plain max, not 99th percentile - the 99th tail cut the
    //     ONE cell that had the fresh foot point on a first-few-ticks
    //     dwell (with 1-2 non-zero cells the 99th index rounded to 0);
    //   * raw floor lowered from 0.02 to 0.005 so a lone cell one tick
    //     after a person arrived still lights up faintly (was silent);
    //   * alpha eased so tiny early accumulation is visible sooner.
    let hc = canvas._heatCache;
    if (!hc || hc.seq !== d.seq || hc.cw !== cw || hc.ch !== ch) {
      const gh = d.heat.length, gw = d.heat[0].length;
      const src = document.createElement("canvas");
      src.width = gw; src.height = gh;
      const sctx = src.getContext("2d");
      let peak = 0;
      for (const row of d.heat)
        for (const v of row) if (v > peak) peak = v;
      const off = document.createElement("canvas");
      off.width = cw; off.height = ch;
      const octx = off.getContext("2d");
      if (peak > 0) {
        // 1) Paint the grid at native resolution using the TURBO
        //    colormap over a sqrt-normalized dwell value (one busy
        //    corner does not crush the walking routes into invisibility).
        for (let gy = 0; gy < gh; gy++) {
          for (let gx = 0; gx < gw; gx++) {
            const raw = d.heat[gy][gx] / peak;
            if (raw < 0.005) continue;
            const v = Math.sqrt(Math.min(1, raw));
            const alpha = Math.min(1, 0.35 + 0.65 * v);
            const [rr, gg, bb] = _turboRGB(v);
            sctx.fillStyle = `rgba(${rr},${gg},${bb},${alpha})`;
            sctx.fillRect(gx, gy, 1, 1);
          }
        }
        // 2) Upscale bilinear + heavy Gaussian blur to the full frame
        //    so the coarse grid melts into smooth blooms.
        octx.imageSmoothingEnabled = true;
        octx.imageSmoothingQuality = "high";
        octx.filter = `blur(${Math.max(10, Math.round(cw / 60))}px)`;
        octx.drawImage(src, 0, 0, cw, ch);
        octx.filter = "none";
      }
      hc = canvas._heatCache = { seq: d.seq, cw, ch, off, peak };
    }
    // 3) Signal-modulated alpha blend onto the live iframe canvas. The
    //    heat canvas already carries per-pixel alpha (blur preserves it),
    //    so a straight drawImage with a moderate globalAlpha is enough.
    if (hc.peak > 0) {
      const prevAlpha = ctx.globalAlpha;
      ctx.globalAlpha = 0.72;
      ctx.drawImage(hc.off, 0, 0);
      ctx.globalAlpha = prevAlpha;
    } else {
      // Empty grid: reassure the operator this layer is running by
      // painting a small "warming up..." caption instead of a blank
      // canvas that reads as broken.
      ctx.font = "12px system-ui, sans-serif";
      const t = "heat signature - accumulating dwell (nothing banked yet)";
      const tw = ctx.measureText(t).width + 14;
      ctx.fillStyle = "rgba(15,23,42,0.8)";
      ctx.fillRect(8, ch - 30, tw, 22);
      ctx.fillStyle = "#fbbf24";
      ctx.fillText(t, 14, ch - 14);
    }
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
    // Big top-center IN/OUT readout (2026-08-16): the counter used to sit
    // small in the bottom-left, hidden behind YouTube's playback UI. Now
    // rendered at 48pt at the top of the frame, IN in glowing green and
    // OUT in glowing red - operator reads it at a glance no matter where
    // the line itself was drawn.
    if (d.cross) {
      const nIn = d.cross.in || 0, nOut = d.cross.out || 0;
      const fontPx = Math.max(24, Math.min(48, Math.round(cw / 22)));
      ctx.save();
      ctx.font = `800 ${fontPx}px system-ui, -apple-system, sans-serif`;
      const txtIn = `IN ${nIn}`;
      const txtSep = "   ";
      const txtOut = `OUT ${nOut}`;
      const wIn = ctx.measureText(txtIn).width;
      const wSep = ctx.measureText(txtSep).width;
      const wOut = ctx.measureText(txtOut).width;
      const totalW = wIn + wSep + wOut;
      const x0 = Math.max(8, (cw - totalW) / 2);
      const y0 = 12 + fontPx;
      // Dark backdrop so both colors read on bright frames too.
      ctx.fillStyle = "rgba(15,23,42,0.72)";
      ctx.fillRect(x0 - 18, 4,
                   Math.min(cw - 4, totalW + 36),
                   fontPx + 22);
      // Glow (blur+wide stroke) + solid fill on top.
      ctx.shadowColor = "rgba(34,197,94,0.85)";
      ctx.shadowBlur = 18;
      ctx.fillStyle = "#22c55e";
      ctx.fillText(txtIn, x0, y0);
      ctx.shadowColor = "rgba(239,68,68,0.85)";
      ctx.fillStyle = "#ef4444";
      ctx.fillText(txtOut, x0 + wIn + wSep, y0);
      ctx.restore();
    }
  }

  if (d.layer === "parking" && Array.isArray(d.spots)) {
    const entries = d.spots;
    ctx.font = "12px system-ui, sans-serif";
    for (const z of entries) {
      const hot = z.occupied;
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
      const label = `${z.name}: ${z.occupied ? (z.cls || "occupied") : "free"}`;
      const zx = z.points[0][0] * cw, zy = z.points[0][1] * ch;
      const tw = ctx.measureText(label).width + 8;
      ctx.fillStyle = "rgba(15,23,42,0.85)";
      ctx.fillRect(zx, Math.max(0, zy - 16), tw, 16);
      ctx.fillStyle = "#f8fafc";
      ctx.fillText(label, zx + 4, Math.max(12, zy - 4));
    }
    if (d.parking) {
      const t = `parking: ${d.parking.occupied}/${d.parking.total} occupied`;
      ctx.fillStyle = "rgba(15,23,42,0.85)";
      ctx.fillRect(8, ch - 30, ctx.measureText(t).width + 14, 22);
      ctx.fillStyle = "#f8fafc";
      ctx.fillText(t, 14, ch - 14);
    }
    if (!entries.length) {
      // Parking on a camera with no spots is not a mistake to fix, it is
      // just "this camera is not a lot" - a friendlier hint (2026-08-16
      // per operator: Green Mango was showing the generic "no zones"
      // string when parking was picked, which read as broken).
      const t = "no parking spots configured for this camera - draw spots to enable";
      ctx.fillStyle = "rgba(15,23,42,0.85)";
      ctx.fillRect(8, 8, ctx.measureText(t).width + 14, 22);
      ctx.fillStyle = "#fbbf24";
      ctx.fillText(t, 14, 23);
    }
  }

  if (d.layer === "fire" && d.fire) {
    const hits = Array.isArray(d.fire.hits) ? d.fire.hits : [];
    ctx.font = "600 13px system-ui, sans-serif";
    for (const h of hits) {
      const x = h.x1 * sx, y = h.y1 * sy;
      const w = (h.x2 - h.x1) * sx, hh = (h.y2 - h.y1) * sy;
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(249,115,22,0.95)";
      ctx.strokeRect(x, y, w, hh);
      const label = `${h.cls || "fire"} ${(h.conf || 0).toFixed(2)}`;
      const tw = ctx.measureText(label).width + 8;
      ctx.fillStyle = "rgba(15,23,42,0.85)";
      ctx.fillRect(x, Math.max(0, y - 16), tw, 16);
      ctx.fillStyle = "#fde68a";
      ctx.fillText(label, x + 4, Math.max(12, y - 4));
    }
    if (d.fire.confirmed) {
      const banner = "FIRE DETECTED";
      const fontPx = Math.max(20, Math.min(40, Math.round(cw / 26)));
      ctx.save();
      ctx.font = `800 ${fontPx}px system-ui, sans-serif`;
      const tw = ctx.measureText(banner).width + 24;
      const bx = Math.max(8, (cw - tw) / 2);
      ctx.fillStyle = "rgba(239,68,68,0.92)";
      ctx.fillRect(bx, 8, tw, fontPx + 16);
      ctx.fillStyle = "#fff";
      ctx.fillText(banner, bx + 12, 8 + fontPx + 4);
      ctx.restore();
    } else if (d.fire.err) {
      const t = d.fire.err;
      ctx.fillStyle = "rgba(15,23,42,0.85)";
      ctx.fillRect(8, 8, ctx.measureText(t).width + 14, 22);
      ctx.fillStyle = "#fbbf24";
      ctx.fillText(t, 14, 23);
    }
  }

  if (d.layer === "paths") {
    // Trails behind each tracked mover + a small dot at the current
    // centroid. Trail line width is 3 (was 2) so it reads on a busy
    // street scene, and each trail carries a small end-cap dot so a
    // one-tick track without enough points to draw a line still shows.
    for (const b of d.boxes || []) {
      const trail = Array.isArray(b.trail) ? b.trail : [];
      const color = _trailColor(b.tid || 0);
      const cxNow = ((b.x1 + b.x2) / 2 + (b.vx || 0) * dtExtra) * sx;
      const cyNow = ((b.y1 + b.y2) / 2 + (b.vy || 0) * dtExtra) * sy;
      if (trail.length >= 2) {
        ctx.lineWidth = 3;
        ctx.lineCap = "round";
        ctx.strokeStyle = color;
        ctx.beginPath();
        ctx.moveTo(trail[0][0] * sx, trail[0][1] * sy);
        for (let i = 1; i < trail.length; i++)
          ctx.lineTo(trail[i][0] * sx, trail[i][1] * sy);
        ctx.lineTo(cxNow, cyNow);
        ctx.stroke();
      }
      // Head dot at current position: makes the newest end of the
      // trail obvious and gives brand-new one-tick tracks a mark.
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(cxNow, cyNow, 4, 0, Math.PI * 2);
      ctx.fill();
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
    // 2026-08-23 (B4): line/parking/fire layers used to paint generic
    // detection boxes on the canvas that the backend JPEG never draws;
    // the mismatch looked like phantom "extra" boxes to operators. Skip
    // the generic pass on those layers - the layer-specific renderers
    // above already draw the pieces the operator cares about
    // (crossings/pills, occupancy polygons, fire hits).
    if (d.layer === "line" || d.layer === "parking" || d.layer === "fire") continue;
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
    if (d.layer === "paths" && b.tier) {
      // Speed number next to the tier chip (2026-08-16). BL/s is
      // perspective-honest (body-lengths per second); the raw px/s is
      // shown too so operators who prefer a pixel rate have it.
      label += ` - ${b.tier}`;
      if (b.speed_blps != null) {
        label += ` ${b.speed_blps.toFixed(1)} BL/s`;
        if (b.speed_pxs) label += ` (${b.speed_pxs} px/s)`;
      }
    }
    if (d.layer === "gestures" && b.gestures)
      label = `#${b.tid} ${b.gestures.join("+")}`;
    if (d.layer === "body" && b.flag) {
      color = b.alert ? "rgba(239,68,68,0.95)" : "rgba(234,140,8,0.95)";
      const flagTxt = String(b.flag).toUpperCase().replace(/_/g, " ");
      label = `#${b.tid} ${flagTxt}`
        + (b.flags ? " " + b.flags.join("+") : "");
      if (b.alert) alertOn = true;
      // Sudden-motion (wrist/ankle burst): draw a red HALO around the
      // person so a snatch/punch/kick is unmistakable at a glance -
      // matches draw_body_layer in the backend JPEG fallback.
      if (b.flag === "sudden_motion") {
        const cx = (b.x1 + b.x2) / 2 * sx + ox * sx;
        const cy = (b.y1 + b.y2) / 2 * sy + oy * sy;
        const rHalo = Math.max(w, h) * 0.75;
        ctx.save();
        ctx.lineWidth = 4;
        ctx.strokeStyle = "rgba(239,68,68,0.9)";
        ctx.beginPath();
        ctx.arc(cx, cy, rHalo, 0, Math.PI * 2);
        ctx.stroke();
        ctx.lineWidth = 1;
        ctx.strokeStyle = "rgba(120,20,20,0.9)";
        ctx.beginPath();
        ctx.arc(cx, cy, rHalo + 4, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
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

// Skeleton in four region colors (2026-08-16, matches app/pose.py's
// _BONE_GROUPS): head/face cluster yellow (nose+eyes+ears), arms cyan
// (shoulders-elbows-wrists), torso trunk green, legs magenta. Each
// person carries four clearly separated colors at once.
const _SKELETON_GROUPS = [
  // head/face
  { color: "rgba(250,204,21,0.95)",
    edges: [[0, 1], [0, 2], [1, 3], [2, 4]] },
  // arms
  { color: "rgba(34,211,238,0.95)",
    edges: [[5, 7], [7, 9], [6, 8], [8, 10]] },
  // torso trunk (incl. neck to shoulders)
  { color: "rgba(74,222,128,0.95)",
    edges: [[0, 5], [0, 6], [5, 6], [5, 11], [6, 12], [11, 12]] },
  // legs
  { color: "rgba(232,121,249,0.95)",
    edges: [[11, 13], [13, 15], [12, 14], [14, 16]] },
];
// Keypoint index -> region color for the joint dots (kept in sync with
// _SKELETON_GROUPS above; head kps 0-4, arms 5-10, legs 11-16).
const _KP_COLOR = {
  0: "rgba(250,204,21,0.95)", 1: "rgba(250,204,21,0.95)",
  2: "rgba(250,204,21,0.95)", 3: "rgba(250,204,21,0.95)",
  4: "rgba(250,204,21,0.95)",
  5: "rgba(34,211,238,0.95)", 6: "rgba(34,211,238,0.95)",
  7: "rgba(34,211,238,0.95)", 8: "rgba(34,211,238,0.95)",
  9: "rgba(34,211,238,0.95)", 10: "rgba(34,211,238,0.95)",
  11: "rgba(232,121,249,0.95)", 12: "rgba(232,121,249,0.95)",
  13: "rgba(232,121,249,0.95)", 14: "rgba(232,121,249,0.95)",
  15: "rgba(232,121,249,0.95)", 16: "rgba(232,121,249,0.95)",
};

function _drawSkeleton(ctx, kps, sx, sy, ox, oy) {
  ctx.lineWidth = 2;
  for (const grp of _SKELETON_GROUPS) {
    ctx.strokeStyle = grp.color;
    for (const [a, b] of grp.edges) {
      const p = kps[a], q = kps[b];
      if (!p || !q || p[2] < 0.3 || q[2] < 0.3) continue;
      ctx.beginPath();
      ctx.moveTo((p[0] + ox) * sx, (p[1] + oy) * sy);
      ctx.lineTo((q[0] + ox) * sx, (q[1] + oy) * sy);
      ctx.stroke();
    }
  }
  for (let i = 0; i < kps.length; i++) {
    const k = kps[i];
    if (!k || k[2] < 0.3) continue;
    ctx.fillStyle = _KP_COLOR[i] || "rgba(219,234,254,0.95)";
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
  const [r, g, b] = _turboRGB(v);
  return `rgba(${r},${g},${b},${alpha})`;
}

// TURBO colormap sampled at 10 stops (Google's improved-rainbow palette,
// same one cv2.COLORMAP_TURBO produces). Restores the exact look the
// original Turkey-era heatmap.overlay() had - dark blue at the cold end,
// glowing red at the hot end, smooth through the middle without the
// perceptual quirks of rainbow.
function _turboRGB(v) {
  const stops = [
    [0.00,  48,  18,  59],
    [0.11,  68,  84, 210],
    [0.22,  65, 155, 250],
    [0.34,  33, 208, 218],
    [0.46,  60, 236, 138],
    [0.58, 156, 246,  67],
    [0.70, 226, 213,  38],
    [0.82, 249, 145,  20],
    [0.94, 235,  76,  14],
    [1.00, 144,  12,  20],
  ];
  const x = Math.max(0, Math.min(1, v));
  for (let i = 0; i < stops.length - 1; i++) {
    const a = stops[i], b = stops[i + 1];
    if (x <= b[0]) {
      const t = (x - a[0]) / Math.max(1e-6, b[0] - a[0]);
      return [
        Math.round(a[1] + (b[1] - a[1]) * t),
        Math.round(a[2] + (b[2] - a[2]) * t),
        Math.round(a[3] + (b[3] - a[3]) * t),
      ];
    }
  }
  return [144, 12, 20];
}

function stopTileAnalysis(st) {
  const a = st.analysis;
  if (!a) return;
  clearInterval(a.timer);
  if (a.videoStateTimer) clearInterval(a.videoStateTimer);
  if (a.evTimer) clearInterval(a.evTimer);
  if (a.sysTimer) clearInterval(a.sysTimer);
  if (a.evStrip) a.evStrip.remove();
  if (a.lastBgUrl) URL.revokeObjectURL(a.lastBgUrl);
  st.analysis = null;
  fetch(`/api/analysis/stop?cam=${encodeURIComponent(a.cam)}`,
        { method: "POST" }).catch(() => {});
  const wrap = st.videoWrap.querySelector(".analysis-wrap");
  if (wrap) wrap.remove();
  if (st.analysisBar) {
    st.analysisBar.remove();
    st.analysisBar = null;
  }
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
  // 2026-08-23 (B1b + decision 11): the old fixed "Quiet / Busy" word
  // alone hid the actual numbers, and raw numbers alone lose the
  // at-a-glance color. Show BOTH: the raw median counts always, and
  // the band word only after enough history rows exist (warmup) so a
  // cold tile never flashes a meaningless "Quiet".
  const last = rows[rows.length - 1];
  const veh = last.vehicles ?? 0;
  const band = idx <= 2 ? "quiet" : idx <= 5 ? "moderate"
             : idx <= 8 ? "busy" : "crowded";
  const warm = rows.length >= ACTIVITY_WARMUP_ROWS;
  const label = `${veh} veh · ${people} ppl`
              + (warm ? ` · ${band}` : "");
  return { idx, label, band, warm, pIdx, vIdx,
           now: last.person ?? 0,
           veh,
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
  // CSS band class comes from act.band (quiet/moderate/busy/crowded) -
  // the label itself now carries numbers and is not a valid class name.
  badge.className = `activity-badge act-${act.band || "unknown"}`;
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
// Rows of history required before the band word (quiet/busy/...) joins
// the raw counts on the activity badge (decision 11 warmup gate).
const ACTIVITY_WARMUP_ROWS = 5;
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

// After N consecutive 404s per camera, back off to a much slower cadence
// so the dashboard doesn't flood serve.py logs with 404s when the notebook
// Section 7 (the only writer of model_view/<cam>.json) is not running -
// which is the standalone-dashboard operator's default state.
const _MODEL_VIEW_MAX_MISS = 3;
const _modelViewMisses = new Map();          // cam_id -> miss count
const _modelViewNextTry = new Map();         // cam_id -> earliest Date.now() to poll again

async function pollLocalModelView() {
  if (!SINGLE_CAM_ID) return;
  const cam = SINGLE_CAM_ID;
  const now = Date.now();
  const nextTry = _modelViewNextTry.get(cam) || 0;
  if (now < nextTry) return;                 // backing off
  const url = MODEL_VIEW_JSON(cam) + `?_=${now}`;
  try {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) {
      const misses = (_modelViewMisses.get(cam) || 0) + 1;
      _modelViewMisses.set(cam, misses);
      if (misses >= _MODEL_VIEW_MAX_MISS) {
        // Cold-start standalone dashboard: no notebook writer, no file.
        // Back off to once per 5 minutes; the notebook can still turn it
        // on later and the next hit resets the counter.
        _modelViewNextTry.set(cam, now + 5 * 60 * 1000);
      }
      return;
    }
    _modelViewMisses.set(cam, 0);
    _modelViewNextTry.set(cam, 0);
    const j = await r.json();
    _updateLocalTileBadges(cam, j);
  } catch (_) { /* not yet written */ }
}

// ---------- Line editor (counting-line drawing) ---------------------------

const lineEditor = document.createElement("div");
// 2026-08-17 (round 2): strict flex-column layout - fixed header top,
// scrollable body middle, PINNED footer bottom - so the Save/Cancel
// row is unconditionally reachable regardless of snapshot size or
// load state. Explicit loading + error placeholders inside the image
// container so the modal is never "just black" while the snapshot
// fetches (previously an in-flight or 404-ing snapshot left the img
// container 0-height, which made the whole modal collapse to only
// the header + darkness).
lineEditor.style.cssText =
  "display:none;position:fixed;inset:0;z-index:70;background:rgba(2,6,23,.55);" +
  "align-items:center;justify-content:center;padding:16px;box-sizing:border-box";
lineEditor.innerHTML = `
  <div style="background:#0f172a;border:1px solid #334155;border-radius:12px;
              max-width:800px;width:94%;max-height:92vh;color:#e2e8f0;
              display:flex;flex-direction:column;overflow:hidden">
    <div style="padding:14px 18px 6px;flex:0 0 auto">
      <h3 style="margin:0;font-size:17px">Counting line -
        <span data-le-cam></span></h3>
      <div style="color:#94a3b8;font-size:13px;margin-top:4px">
        Drag on the snapshot to place a counting line. Save persists
        it per-camera and closes this dialog; a running Line-layer
        session picks it up within a few seconds without a restart.
      </div>
    </div>
    <div style="padding:6px 18px;flex:1 1 auto;overflow-y:auto;
                display:flex;flex-direction:column;gap:8px">
      <div style="position:relative;background:#020617;
                  border:1px solid #334155;border-radius:8px;
                  overflow:hidden;min-height:200px;
                  display:flex;justify-content:center;align-items:center">
        <div data-le-placeholder style="color:#94a3b8;font-size:14px;
             text-align:center;padding:16px">Loading snapshot...</div>
        <img data-le-img style="display:none;max-width:100%;max-height:55vh;
                                width:auto;height:auto;object-fit:contain;
                                user-select:none;-webkit-user-drag:none">
        <canvas data-le-canvas style="display:none;position:absolute;inset:0;
                                       width:100%;height:100%;
                                       cursor:crosshair"></canvas>
      </div>
      <div data-le-classes style="display:flex;flex-wrap:wrap;
                                   gap:10px 16px;font-size:13px;
                                   color:#cbd5e1"></div>
      <div style="color:#94a3b8;font-size:12px">
        Nothing checked = count every tracked class.</div>
      <div data-le-err style="color:#f87171;font-size:13px;
                              min-height:18px"></div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;flex:0 0 auto;
                padding:10px 18px 14px;background:#0f172a;
                border-top:1px solid #1e293b">
      <button data-le-save style="cursor:pointer;background:#2563eb;border:0;
              color:#fff;border-radius:8px;padding:9px 22px;font-weight:600">
        Save &amp; Close</button>
      <button data-le-flip style="cursor:pointer;background:#0369a1;border:0;
              color:#fff;border-radius:8px;padding:9px 14px" title=
              "Swap the two line endpoints so IN/OUT reverse">
        Flip direction</button>
      <button data-le-clear style="cursor:pointer;background:#334155;border:0;
              color:#fff;border-radius:8px;padding:9px 14px">
        Clear override</button>
      <button data-le-cancel style="cursor:pointer;background:#1e293b;
              border:1px solid #334155;color:#e2e8f0;border-radius:8px;
              padding:9px 14px;margin-left:auto">Cancel</button>
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

lineEditor.querySelector("[data-le-flip]").addEventListener("click", () => {
  // Swap the two endpoints so the crossing direction reverses. IN
  // (neg->pos cross of A->B) becomes OUT after the swap. Purely
  // client-side; nothing is persisted until Save fires with the new
  // point order.
  if (_lePts.length !== 2) {
    _leErr.style.color = "#f87171";
    _leErr.textContent = "Draw a line first, then flip.";
    return;
  }
  _lePts = [_lePts[1], _lePts[0]];
  _leDraw();
  _leErr.style.color = "#4ade80";
  _leErr.textContent = "Direction flipped - Save to persist.";
  setTimeout(() => {
    _leErr.style.color = "#f87171";
    _leErr.textContent = "";
  }, 1500);
});

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
    // 2026-08-17: Save now auto-closes the modal on success so the
    // operator returns to the live tile immediately - the persisted
    // line is picked up by the running session within a few seconds
    // via LINE_RELOAD_POLL_S.
    _leErr.style.color = "#4ade80";
    _leErr.textContent = "Saved - closing dialog...";
    setTimeout(() => {
      lineEditor.style.display = "none";
      _leErr.style.color = "#f87171";
      _leErr.textContent = "";
    }, 700);
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
  const placeholder = lineEditor.querySelector("[data-le-placeholder]");
  placeholder.style.display = "";
  placeholder.style.color = "#94a3b8";
  placeholder.textContent = "Loading snapshot from the running analysis...";
  _leImg.style.display = "none";
  _leCanvas.style.display = "none";
  _leRenderClasses(_leLastAllowed, []);
  // Show the modal IMMEDIATELY so the operator sees the header +
  // buttons + a "loading..." caption instead of a black screen while
  // the JPEG fetch is in flight (or 404s on a not-yet-started session).
  lineEditor.style.display = "flex";
  // Attach handlers BEFORE setting src so a cached image can't miss
  // its onload race.
  _leImg.onload = () => {
    placeholder.style.display = "none";
    _leImg.style.display = "block";
    _leCanvas.style.display = "block";
    fetch(`/api/lines?cam=${encodeURIComponent(cam)}`).then(r => r.json()).then(d => {
      if (d && d.line) _lePts = d.line;
      if (d && Array.isArray(d.allowed_classes) && d.allowed_classes.length)
        _leLastAllowed = d.allowed_classes;
      _leRenderClasses(_leLastAllowed, (d && d.classes) || []);
      _leDraw();
    }).catch(() => _leDraw());
  };
  _leImg.onerror = () => {
    _leImg.style.display = "none";
    _leCanvas.style.display = "none";
    placeholder.style.color = "#f87171";
    placeholder.textContent =
      "No snapshot available yet. Start the Line layer on this camera " +
      "for a few seconds, then click Draw line again.";
  };
  _leImg.src = snapshotUrl || `/api/analysis/frame?cam=${encodeURIComponent(cam)}&_=${Date.now()}`;
}
window.openLineEditor = openLineEditor;

// ---------- Zones editor (loiter/parking polygons) ------------------------

const zoneEditor = document.createElement("div");
zoneEditor.style.cssText =
  "display:none;position:fixed;inset:0;z-index:70;background:rgba(2,6,23,.82);" +
  "align-items:center;justify-content:center";
zoneEditor.innerHTML = `
  <div style="background:#0f172a;border:1px solid #334155;border-radius:12px;
              padding:16px 18px;max-width:960px;width:96%;color:#e2e8f0;
              max-height:92vh;display:flex;flex-direction:column">
    <h3 style="margin:0 0 4px;font-size:17px"><span data-ze-title></span> -
      <span data-ze-cam></span></h3>
    <div style="color:#94a3b8;font-size:13px;margin-bottom:10px">
      Click the snapshot to drop polygon corners; double-click (or the
      button) closes the shape. The list on the right shows every saved
      zone - rename or delete individual ones there, then Save.</div>
    <div style="display:grid;grid-template-columns:minmax(0,1fr) 260px;
                gap:14px;flex:1 1 auto;overflow:hidden">
      <div>
        <div style="position:relative;background:#020617;
                    border:1px solid #334155;border-radius:8px;overflow:hidden">
          <img data-ze-img style="display:block;width:100%;height:auto;
                                  user-select:none;-webkit-user-drag:none">
          <canvas data-ze-canvas style="position:absolute;inset:0;width:100%;
                                         height:100%;cursor:crosshair"></canvas>
        </div>
        <div data-ze-dwellrow style="margin-top:10px;font-size:13px;
                                      color:#cbd5e1">
          Loiter alert after <input data-ze-dwell type="number" min="5"
            max="3600" value="30" style="width:70px;background:#1e293b;
            color:#e2e8f0;border:1px solid #334155;border-radius:6px;
            padding:3px 6px"> seconds inside a zone.</div>
      </div>
      <div style="background:#0b1220;border:1px solid #1e293b;border-radius:8px;
                  padding:8px;display:flex;flex-direction:column;
                  overflow:hidden;min-height:0">
        <div style="font-size:12px;color:#94a3b8;padding:2px 4px 6px;
                    border-bottom:1px solid #1e293b;margin-bottom:6px">
          Saved zones (<span data-ze-count>0</span>)
        </div>
        <div data-ze-list style="overflow-y:auto;display:flex;flex-direction:
                                  column;gap:4px;flex:1 1 auto;min-height:0"></div>
        <div style="color:#64748b;font-size:11px;padding:6px 4px 0;
                    border-top:1px solid #1e293b;margin-top:6px">
          Deletions and renames apply on Save. Cancel throws them away.
        </div>
      </div>
    </div>
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

function _zeRenderList() {
  const list = zoneEditor.querySelector("[data-ze-list]");
  const count = zoneEditor.querySelector("[data-ze-count]");
  if (!list || !count) return;
  count.textContent = String(_zeZones.length);
  list.innerHTML = "";
  if (!_zeZones.length) {
    const empty = document.createElement("div");
    empty.style.cssText = "color:#475569;font-size:12px;padding:6px 4px";
    empty.textContent = "no zones yet - draw one on the snapshot.";
    list.appendChild(empty);
    return;
  }
  for (let i = 0; i < _zeZones.length; i++) {
    const z = _zeZones[i];
    const row = document.createElement("div");
    row.style.cssText =
      "display:flex;gap:6px;align-items:center;background:#1e293b;" +
      "border-radius:6px;padding:6px 8px";
    const nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.value = z.name || "";
    nameInput.maxLength = 24;
    nameInput.style.cssText =
      "flex:1 1 auto;min-width:0;background:#0b1220;color:#e2e8f0;" +
      "border:1px solid #334155;border-radius:4px;padding:3px 6px;" +
      "font-size:12px";
    nameInput.addEventListener("input", () => {
      _zeZones[i].name = nameInput.value.trim().slice(0, 24);
      _zeRedraw();   // redraw uses z.name for the on-canvas label
    });
    const delBtn = document.createElement("button");
    delBtn.textContent = "delete";
    delBtn.style.cssText =
      "cursor:pointer;background:#7f1d1d;color:#fff;border:0;" +
      "border-radius:4px;padding:4px 8px;font-size:11px;flex:0 0 auto";
    delBtn.addEventListener("click", () => {
      _zeZones.splice(i, 1);
      _zeRedraw();      // canvas needs the removal reflected too
      _zeRenderList();
    });
    row.appendChild(nameInput);
    row.appendChild(delBtn);
    list.appendChild(row);
  }
}

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
  _zeRenderList();
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
  _zeRenderList();
});
zoneEditor.querySelector("[data-ze-clear]").addEventListener("click", () => {
  _zeZones = []; _zeCurrent = [];
  _zeRedraw();
  _zeRenderList();
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
  _zeImg.onload = () => { _zeRedraw(); _zeRenderList(); };
  _zeImg.src = snapshotUrl;
  zoneEditor.style.display = "flex";
  if (_zeImg.complete) { _zeRedraw(); _zeRenderList(); }
  _zeRenderList();
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
  const st = buildSingleTile(cam);
  if (status) status.textContent = "";
  pollLocalModelView();
  setInterval(pollLocalModelView, 8000);
  // ?cinema=1 auto-toggle removed 2026-08-20 per operator request:
  // Cinema should always be an explicit user action - the auto-toggle
  // surprised operators who wanted the normal dashboard first.
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
