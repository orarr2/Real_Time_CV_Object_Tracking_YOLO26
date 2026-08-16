// Slot config for the HTML dashboard.
//
// Since the fallback refactor, the SOURCE OF TRUTH for which camera is
// currently backing each grid slot is Firestore `config/grid`. The cloud
// collector updates it whenever a slot switches to a fallback cam (or
// back to its primary).
//
// This file only carries the initial layout used until the first onSnapshot
// callback lands - the four slot ids + fallback URLs for local HLS playback,
// mirrored from app/cameras.py GRID_SLOTS. Everything else (active_cam_name,
// active_embed, active_hls) comes from the Firestore doc live.

// Helper: tvkur HLS via the local /tvkur/ proxy (dashboard_server.py rewrites
// the Referer + adds Access-Control-Allow-Origin so hls.js can play it).
const tvkurHls = (id) => `/tvkur/${id}/master.m3u8`;

// The four slots the dashboard renders. Order = display order (2x2 grid).
// `placeholder_*` fields are what the tile shows before Firestore's
// config/grid doc arrives; they get replaced on the first snapshot.
//
// 2026-07: the Istanbul slots (Sultanahmet + Taksim) were swapped for two
// more Konya webcamera24/tvkur cams because IBB tightened its geo-block to
// Turkey-only. Keep in sync with GRID_SLOTS in src/app/cameras.py.
const CLOUD_SLOTS = [
  {
    slot_id:          "slot_konya_hukumet",
    display_area:     "Konya - Hükümet",
    placeholder_name: "Konya - Hükümet Meydanı",
    placeholder_hls:  tvkurHls("c77i84vbb2nj4i0fr80g"),
    placeholder_page: "https://webcamera24.com/camera/turkey/8043-sarraflar-yeralti-carsisi/",
  },
  {
    slot_id:          "slot_otogar",
    display_area:     "Konya - Otogar",
    placeholder_name: "Konya - Otogar Kavşağı",
    placeholder_hls:  tvkurHls("c77i91vbb2nj4i0fr81g"),
    placeholder_page: "https://webcamera24.com/camera/turkey/8044-otogar-kavsagi/",
  },
  {
    slot_id:          "slot_kulturpark",
    display_area:     "Konya - Kültürpark",
    placeholder_name: "Konya - Kültürpark",
    placeholder_hls:  tvkurHls("c77i6hb84cnrb6mlji3g"),
    placeholder_page: "https://webcamera24.com/camera/turkey/8058-kulturpark/",
  },
  {
    slot_id:          "slot_millet_caddesi",
    display_area:     "Konya - Millet Caddesi",
    placeholder_name: "Konya - Millet Caddesi / Hastane Kavşağı",
    placeholder_hls:  tvkurHls("c77i9cfbb2nj4i0fr82g"),
    placeholder_page: "https://webcamera24.com/camera/turkey/8046-millet-caddesi/",
  },
  // konya_ince_minareli (tram view, tvkur c77ib8vbb2nj4i0fr8bg) is cataloged
  // in app/cameras.py but kept OUT of the grid - four slots is the cadence
  // budget on the free-tier VM.
];

// Local-mode override. When the notebook's Section 7 serves this dashboard it
// writes `web/local_grid.json` = the four cameras the user PICKED (distinct
// `local_*` slot_ids, each with its own embed/HLS). If that file is present we
// render THOSE instead of the cloud Konya grid - so "I picked 4, I see 4".
// The cloud Firestore `config/grid` keys by the VM's slot ids, which do NOT
// match `local_*`, so its onSnapshot naturally leaves these tiles alone.
// When the file is absent (the public cloud dashboard), we fall back to the
// cloud grid unchanged. Top-level await is fine in an ES module.
let GRID_SLOTS = CLOUD_SLOTS;
let LOCAL_MODE = false;
try {
  const r = await fetch("./local_grid.json?_=" + Date.now(), { cache: "no-store" });
  if (r.ok) {
    const j = await r.json();
    if (Array.isArray(j?.slots) && j.slots.length) {
      // YouTube picks play through the NATIVE YouTube iframe (YT.Player) -
      // reverted 2026-08-15 after the operator confirmed hls.js via the
      // local /ytproxy relay lags visibly vs. youtube.com itself: the
      // extra Python-relay hop, hls.js lowLatencyMode churn, and MSE
      // appends competing with the canvas draw on the main thread stack
      // together into stutters no tuning erased. YouTube's own decoder,
      // JIT-buffered by their own CDN, is what "runs like YouTube"
      // literally means. Overlay alignment now uses YT IFrame API's
      // getCurrentTime as the video clock (see _analysisDrawLoop's
      // ytClockEpoch branch). For non-YouTube picks (tvkur / IBB) hls.js
      // stays - they have no iframe alternative.
      GRID_SLOTS = j.slots;
      LOCAL_MODE = true;
    }
  }
} catch (_) { /* no local_grid.json -> cloud grid */ }

export { GRID_SLOTS, LOCAL_MODE };

// Given an active_cam field from Firestore, return the correct HLS/embed URL.
// For tvkur-backed cams we always route through the local /tvkur/ proxy so
// hls.js can play them (content.tvkur.com refuses direct requests).
export function hlsUrlForActiveCam(cfg) {
  if (!cfg) return null;
  if (cfg.active_embed && cfg.active_embed.includes("player.tvkur.com/l/")) {
    // player.tvkur.com/l/<id> -> /tvkur/<id>/master.m3u8
    const id = cfg.active_embed.split("/l/")[1].split("/")[0];
    return tvkurHls(id);
  }
  return cfg.active_hls || null;
}
