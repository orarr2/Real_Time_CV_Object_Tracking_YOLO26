"""Verified Turkey commercial / market / square camera catalog.

IBB streams (Istanbulu Seyret) migrated 2024-2025: the legacy `cam_trsk_*` prefix is gone,
and the current player config (bradmaxPlayerConfig in each page on istanbuluseyret.ibb.gov.tr)
points at `livestream.ibb.gov.tr/cam_turistik/b_*.stream/playlist.m3u8`. The Eminonu and
Istiklal cameras are no longer listed and have been removed.

Note: livestream.ibb.gov.tr returns HTTP 404 (not 403) for these stream paths when accessed
from non-Turkey IPs - they appear to be geo-restricted. Run from a Turkey-routed IP for live
data; on any other network you'll just see MISS rows in the collector.

Each entry: kind = one of
  "hls"          direct .m3u8 (used as-is)
  "youtube"      a YouTube / YouTube-backed page, resolved via yt-dlp
  "skyline"      a skylinewebcams.com page, resolved via detect_core.resolve_skyline
  "webcamera24"  a webcamera24.com page, resolved via detect_core.resolve_webcamera24

Optional per-entry keys:
  "page"   public webcam page (human-facing, also the resolver input for skyline/webcamera24)
  "embed"  iframe URL for the live player shown in the dashboard grid (None -> the grid
           falls back to the latest annotated YOLO frame as the "live" tile)
  "conf"   per-camera YOLO confidence override. The collector's global default
           (--conf) fits most scenes; set this on a camera whose calibration run
           (notebook section 10) shows a systematic bias - lower it when the
           camera undercounts (distant/small objects), raise it on false
           positives. Example: "conf": 0.25.

Operational-analysis keys (all coordinates NORMALIZED 0..1 relative to the
frame; run `python -m tools.roi_grid <cam_id>` to capture a frame with a
coordinate grid overlay and read the points off it):
  "roi"          include-polygon [[x,y], ...]: detections whose FOOT POINT
                 (bottom-center) falls outside are ignored entirely - use it
                 to cut parking lots / sky / a neighboring street out of the
                 business-activity counts.
  "roi_exclude"  list of exclude-polygons carved out of the ROI.
  "line"         virtual counting line [[x1,y1],[x2,y2]]: each sampling burst
                 counts tracked objects crossing it, split into in/out
                 (crossing from the right side of A->B is "in" - order the two
                 points so "in" points into your area of interest). The
                 dashboard shows the sampled flow; it is a rate indicator,
                 not a turnstile.
  "loiter_roi"   polygon for prolonged-presence alerts (defaults to the whole
                 frame when omitted while loitering is enabled).
  "loiter_person_sec" / "loiter_vehicle_sec"
                 per-camera override of how long a person/vehicle must stay
                 (matched in place across samples) before a loiter event
                 fires. Defaults: 300s person / 900s vehicle.

Example (uncomment and tune per scene):
  # "roi":  [[0.05, 0.35], [0.95, 0.35], [0.95, 0.98], [0.05, 0.98]],
  # "line": [[0.10, 0.60], [0.90, 0.55]],
  # "loiter_person_sec": 240,
"""
from __future__ import annotations

import json
import time
from pathlib import Path

# Header set IBB's nginx accepts. ffmpeg/OpenCV honor this via OPENCV_FFMPEG_CAPTURE_OPTIONS.
IBB_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer":    "https://istanbuluseyret.ibb.gov.tr/",
    "Origin":     "https://istanbuluseyret.ibb.gov.tr",
}

CAMERAS = {
    # --- Konya: crowded square/market. webcamera24 entry 8043-sarraflar-yeralti-carsisi
    # embeds a tvkur.com live player; the underlying HLS master is on content.tvkur.com.
    # detect_core.grab_frame() handles the Referer/Origin requirements for this host. ---
    "konya_hukumet": {
        "name": "Konya - Hukumet Meydani / Sarraflar Yeralti Carsisi",
        "city": "Konya",
        "kind": "hls",
        "url": "https://content.tvkur.com/l/c77i84vbb2nj4i0fr80g/master.m3u8",
        "page": "https://webcamera24.com/camera/turkey/8043-sarraflar-yeralti-carsisi/",
        "embed": "https://player.tvkur.com/l/c77i84vbb2nj4i0fr80g",
        "type": "square/market",
    },
    # --- Istanbul: high-footfall commerce & markets (IBB public HLS, new b_* prefix) ---
    "taksim": {
        "name": "Taksim Meydani (legacy host - 404 outside Turkey)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://livestream.ibb.gov.tr/cam_turistik/b_taksim_meydan.stream/playlist.m3u8",
        "type": "square/retail",
    },
    # --- IBB migrated to kamerayayin.ibb.istanbul (2025-2026). The new -yeni pages
    # use this domain, return CORS *, and are reachable from any country - much
    # better than the legacy livestream.ibb.gov.tr endpoints above. ---
    "taksim_yeni": {
        "name": "Taksim Meydani (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/taksim.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/taksim-yeni/",
        # the public page sets X-Frame-Options: SAMEORIGIN, so no iframe embed.
        # The web/ dashboard plays this HLS directly with hls.js in its own <video>.
        "embed": None,
        "type": "square/retail",
        # The metro entrance canopy + its "M" logo (bottom-right of the
        # plaza) fired as bus/car for two straight weeks and owned the
        # anomaly feed (registry entities 164830: 'bus', 1181 sightings;
        # 165327: 'car', 254 sightings). The first polygon stopped at
        # x=0.80 and the canopy's 'bus' box kept escaping by widening to
        # the frame edge (x 0.66-1.0 -> foot center 0.83, caught again in
        # the 07.08 digest), so `bus` is now excluded across the WHOLE
        # bottom strip - no real bus ever stands with its wheels on the
        # pedestrian plaza (the terminal lanes' foot points sit at
        # y 0.55-0.75). `car`/`truck` keep the left edge at 0.55 because
        # real vans do park on the bottom-LEFT apron (seen 02.08).
        # 08.08: the canopy escaped UPWARD too - the 05:39Z loiter box
        # bottomed out at foot y=0.786, above the 0.84 strip - so a second
        # polygon covers the canopy's right-shoulder drift range
        # (x>=0.75, y>=0.70; the terminal lanes on the LEFT never enter it).
        # `person` is deliberately NOT excluded: people walk in front of
        # the canopy all day.
        # 09.08: vendor carts (simit/corn stands with white canopies) park on
        # the bottom-CENTER walkway (foot x 0.23-0.55, y>=0.90) and COCO has
        # no cart class - a canopied cart reads as 'car', stood 5+ min, and
        # produced a "Static object left · car" card that was actually a
        # vendor cart. Real vans keep the bottom-LEFT apron (x < 0.23).
        "roi_exclude_class": {
            "bus":   [[[0.0, 0.84], [1.0, 0.84], [1.0, 1.0], [0.0, 1.0]],
                      [[0.75, 0.70], [1.0, 0.70], [1.0, 1.0], [0.75, 1.0]]],
            "car":   [[[0.55, 0.84], [1.0, 0.84], [1.0, 1.0], [0.55, 1.0]],
                      [[0.75, 0.70], [1.0, 0.70], [1.0, 1.0], [0.75, 1.0]],
                      [[0.23, 0.90], [0.55, 0.90], [0.55, 1.0], [0.23, 1.0]]],
            "truck": [[[0.55, 0.84], [1.0, 0.84], [1.0, 1.0], [0.55, 1.0]],
                      [[0.75, 0.70], [1.0, 0.70], [1.0, 1.0], [0.75, 1.0]],
                      [[0.23, 0.90], [0.55, 0.90], [0.55, 1.0], [0.23, 1.0]]],
        },
        # Measured 09.08 live: person 8 -> 16 at 960 (the wide plaza's far
        # half simply does not exist at 640). Same per-cam mechanism as
        # Sarachane; Sultanahmet measured no gain and stays on the global.
        "imgsz": 960,
        # fix1-A10: pedestrian flow line across the plaza's main walking
        # axis. A->B left-to-right => "in" = crossing DOWN-screen (toward
        # the metro / McDonald's side).
        "line": [[0.15, 0.74], [0.85, 0.74]],
    },
    "beyazit_meydan": {
        "name": "Beyazit Meydani",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://livestream.ibb.gov.tr/cam_turistik/b_beyazitmeydani.stream/playlist.m3u8",
        "type": "square/market-gateway",
    },
    "kapali_carsi": {
        "name": "Kapali Carsi (Grand Bazaar)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://livestream.ibb.gov.tr/cam_turistik/b_kapalicarsi.stream/playlist.m3u8",
        "type": "market",
    },
    "misir_carsisi": {
        "name": "Misir Carsisi (Spice Bazaar)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://livestream.ibb.gov.tr/cam_turistik/b_misircarsisi.stream/playlist.m3u8",
        "type": "market",
    },
    "sultanahmet_1": {
        "name": "Sultanahmet (legacy host - 404 outside Turkey)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://livestream.ibb.gov.tr/cam_turistik/b_sultanahmet.stream/playlist.m3u8",
        "type": "tourist square",
    },
    "sultanahmet_1_yeni": {
        "name": "Sultanahmet (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/sultanahmet1.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/sultanahmet-1-yeni/",
        "embed": None,
        "type": "tourist square",
        # fix1-A10: the main promenade at the bottom of the park view.
        # A->B left-to-right => "in" = crossing down-screen (toward the
        # camera / tram side).
        "line": [[0.05, 0.82], [0.95, 0.82]],
    },
    # --- Three more IBB kamerayayin cameras added 2026-07-14 after the
    # Konya tvkur backend went 404. All three verified HTTP 200 at add time
    # and returned as CORS: * so the web dashboard can play them directly
    # via hls.js. Their -yeni public pages are on istanbuluseyret. ---
    "beyazit_meydan_yeni": {
        "name": "Beyazit Meydani (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/beyazitmeydan.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/beyazit-meydani-yeni/",
        "embed": None,
        "type": "square/market-gateway",
        # Measured 09.08 live: person 14 -> 16 at 960 - the far half of the
        # university plaza gains meaningfully; same per-cam mechanism as
        # Sarachane/Taksim.
        "imgsz": 960,
        # fix1-A10: gate corridor line by the university entrance (left
        # side). A->B bottom-to-top => "in" = crossing left-to-right =
        # out of the gate INTO the plaza.
        "line": [[0.33, 0.95], [0.33, 0.40]],
    },
    "eyup_sultan_yeni": {
        "name": "Eyup Sultan (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/eyupsultan.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/eyup-sultan-yeni/",
        "embed": None,
        "type": "religious square",
    },
    "buyuk_camlica_yeni": {
        "name": "Buyuk Camlica (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/buyukcamlica.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/buyuk-camlica-yeni/",
        "embed": None,
        "type": "park/vista",
    },
    # --- Remaining kamerayayin.ibb.istanbul cameras (all HTTP 200 on
    # 2026-07-16). Tier-3 fallbacks for the collector: when Konya (tvkur)
    # AND the four preferred Istanbul cams are down, the pool keeps walking
    # this list so no slot ever settles on an empty frame. ---
    "sarachane_yeni": {
        "name": "Sarachane (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/sarachane.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/sarachane-yeni/",
        "embed": None,
        "type": "civic square",
        # The camera hangs far above the Valens aqueduct junction - people on
        # the sidewalks are ~10px tall at the global imgsz 640, and the
        # 07-08.08 digests peaked at "2 people" while the preview showed
        # dozens (vehicles on the near road detected fine: 11-14). Running
        # THIS cam at 960 recovers the pedestrian half of the scene for
        # ~2.25x its share of round compute; the other cams stay at 640.
        "imgsz": 960,
        # fix1-A10: vertical tripwire across the main road - counts the
        # traffic flow. A->B bottom-to-top => "in" = left-to-right traffic.
        "line": [[0.35, 0.62], [0.35, 0.42]],
    },
    "sultanahmet_2_yeni": {
        "name": "Sultanahmet 2 (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/sultanahmet2.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/sultanahmet-2-yeni/",
        "embed": None,
        "type": "tourist square",
    },
    "uskudar_yeni": {
        "name": "Uskudar (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/uskudar.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/uskudar-yeni/",
        "embed": None,
        "type": "square/transport",
    },
    "salacak_yeni": {
        "name": "Salacak (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/salacak.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/salacak-yeni/",
        "embed": None,
        "type": "waterfront promenade",
    },
    "kucukcekmece_yeni": {
        "name": "Kucukcekmece (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/kucukcekmece.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/kucukcekmece-yeni/",
        "embed": None,
        "type": "lakeside park",
    },
    "ulus_parki_yeni": {
        "name": "Ulus Parki (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/ulusparki.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/ulus-parki-yeni/",
        "embed": None,
        "type": "park/vista",
    },
    "pierre_lotti_yeni": {
        "name": "Pierre Lotti (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/pierreloti.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/pierre-lotti-yeni/",
        "embed": None,
        "type": "hilltop cafe/vista",
    },
    "emirgan_yeni": {
        "name": "Emirgan (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/emirgan.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/emirgan-yeni/",
        "embed": None,
        "type": "park",
    },
    "kiz_kulesi_yeni": {
        "name": "Kiz Kulesi (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/kizkulesi.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/kiz-kulesi-yeni/",
        "embed": None,
        "type": "waterfront landmark",
    },
    "hidiv_kasri_yeni": {
        "name": "Hidiv Kasri (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/hidivkasri.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/hidiv-kasri-yeni/",
        "embed": None,
        "type": "palace grounds",
    },
    "dragos_yeni": {
        "name": "Dragos (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/dragos.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/dragos-yeni/",
        "embed": None,
        "type": "coastal vista",
    },
    "kadikoy": {
        "name": "Kadikoy",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://livestream.ibb.gov.tr/cam_turistik/b_kadikoy.stream/chunklist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/kadikoy/",
        # IBB pages set X-Frame-Options, so no reliable iframe embed; the dashboard grid
        # shows the latest annotated YOLO frame for this tile instead.
        "embed": None,
        "type": "commerce/transit",
    },
    "eyup_sultan": {
        "name": "Eyup Sultan",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://livestream.ibb.gov.tr/cam_turistik/b_eyupsultan.stream/playlist.m3u8",
        "type": "tourist square",
    },
    "uskudar": {
        "name": "Uskudar",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://livestream.ibb.gov.tr/cam_turistik/b_uskudar.stream/playlist.m3u8",
        "type": "square/transport",
    },
    # --- Otogar Kavsagi (webcamera24, entry 8044). Like the Konya cam this page embeds a
    # tvkur live player. tvkur player id resolved 2026-06: c77i91vbb2nj4i0fr81g. ---
    "otogar_kavsagi": {
        "name": "Otogar Kavsagi",
        "city": "Konya",
        "kind": "webcamera24",
        "url": "https://webcamera24.com/camera/turkey/8044-otogar-kavsagi/",
        "page": "https://webcamera24.com/camera/turkey/8044-otogar-kavsagi/",
        "embed": "https://player.tvkur.com/l/c77i91vbb2nj4i0fr81g",
        "type": "junction/transit",
        # Counting line perpendicular to the diagonal road axis (top-left
        # cluster -> bottom-right exit), calibrated on
        # docs/images/model_view_otogar_kavsagi.jpg. A->B ordered so "in"
        # = crossing down-right = toward the tram crossing / otogar side.
        # Same sampled-rate caveat as every "line".
        "line": [[0.35, 0.65], [0.95, 0.25]],
    },
    # --- Konya Kulturpark (webcamera24 8058) - replaces Giresun in the grid because
    # skylinewebcams geo-blocks Israel. tvkur id: c77i6hb84cnrb6mlji3g. ---
    "konya_kulturpark": {
        "name": "Konya - Kulturpark",
        "city": "Konya",
        "kind": "webcamera24",
        "url": "https://webcamera24.com/camera/turkey/8058-kulturpark/",
        "page": "https://webcamera24.com/camera/turkey/8058-kulturpark/",
        "embed": "https://player.tvkur.com/l/c77i6hb84cnrb6mlji3g",
        "type": "park/commercial",
    },
    # --- Konya Millet Caddesi / Hastane Kavsagi (webcamera24 8046) - replaces Kadikoy
    # in the grid because IBB streams geo-block Israel. tvkur id: c77i9cfbb2nj4i0fr82g. ---
    "konya_millet_caddesi": {
        "name": "Konya - Millet Caddesi / Hastane Kavsagi",
        "city": "Konya",
        "kind": "webcamera24",
        "url": "https://webcamera24.com/camera/turkey/8046-millet-caddesi/",
        "page": "https://webcamera24.com/camera/turkey/8046-millet-caddesi/",
        "embed": "https://player.tvkur.com/l/c77i9cfbb2nj4i0fr82g",
        "type": "hospital junction / vehicular",
        # Virtual counting line across the junction mouth, calibrated on
        # docs/images/model_view_konya_millet_caddesi.jpg: the main street
        # runs from the top-center vanishing point toward the camera, so a
        # near-horizontal cut at mid-frame catches the whole flow. A->B is
        # ordered left-to-right, which makes "in" = crossing DOWNWARD =
        # toward the camera / into the junction foreground. Sampled rate,
        # not a turnstile (the burst sees ~2-3s of every interval).
        "line": [[0.03, 0.46], [0.95, 0.30]],
        # The blue keep-right sign on the traffic island reads as `person`
        # (~0.38) again and again - operator flagged it twice from live
        # screenshots. Static street furniture, so a static exclude: any
        # `person` whose foot-point lands on the island sign is dropped.
        "roi_exclude_class": {
            "person": [[[0.385, 0.17], [0.445, 0.17],
                        [0.445, 0.30], [0.385, 0.30]]],
        },
    },
    # --- Konya Ince Minareli Medrese (webcamera24 8033). The city's TRAM line
    # runs on dedicated tracks along the left of the frame (Alaaddin loop),
    # with a pedestrian plaza and a road in view - the first camera in the
    # grid that can actually produce `train`-class detections. Added after
    # the operator reported metro/tram traffic was invisible to the system
    # (no rail-view camera existed). tvkur id resolved 2026-07:
    # c77ib8vbb2nj4i0fr8bg, frame 1920x1080, reachable from any country. ---
    "konya_ince_minareli": {
        "name": "Konya - Ince Minareli Medrese (tram line)",
        "city": "Konya",
        "kind": "hls",
        "url": "https://content.tvkur.com/l/c77ib8vbb2nj4i0fr8bg/master.m3u8",
        "page": "https://webcamera24.com/camera/turkey/8033-ince-minareli-medrese/",
        "embed": "https://player.tvkur.com/l/c77ib8vbb2nj4i0fr8bg",
        "type": "tram line / plaza",
    },
    # --- Kept for runs from a Turkey-routed IP, where these unblock. Not in the grid. ---
    "giresun_gazi": {
        "name": "Giresun - Gazi Caddesi",
        "city": "Giresun",
        "kind": "skyline",
        "url": "https://www.skylinewebcams.com/en/webcam/turkey/giresun/giresun/gazi-street.html",
        "page": "https://www.skylinewebcams.com/en/webcam/turkey/giresun/giresun/gazi-street.html",
        "embed": "https://www.skylinewebcams.com/en/embed/turkey/giresun/giresun/gazi-street.html",
        "type": "commercial street (geo-restricted)",
    },

    # --- Turkey YouTube-Live tier (2026-07-21, added after the GCP geo-block
    # verification). Every IBB stream (kamerayayin.ibb.istanbul) and every
    # tvkur.com-backed webcamera24 Turkey entry returns HTTP 403 on the raw
    # HLS layer from us-east1 - verified end-to-end with tools/probe_country
    # (21/21 DEAD, all http_403). YouTube-Live is the only path that gives
    # the collector actual Turkish coverage from the free-tier VM: these
    # three streams are webcamera24-listed but backed by youtube.com/watch
    # embeds (verified 2026-07-21 with yt-dlp is_live=True), so they resolve
    # exactly like the Thailand/Japan/USA cameras that have been running
    # for weeks. Ordered first in TURKEY_POOL below so the collector starts
    # here instead of walking through the 21 blocked entries.
    "tr_bulancak_meydan": {
        "name": "Bulancak Meydani (Giresun)", "city": "Giresun", "country": "turkey",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=vn702Owd5Kk",
        "page": "https://webcamera24.com/camera/turkey/bulancak-square-cam/",
        "embed": "https://www.youtube.com/embed/vn702Owd5Kk?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
        # A bench-lined town square: sitting for 5+ minutes is the NORMAL
        # use of the place, and the default 300s person threshold turned a
        # summer afternoon into 10 loiter alerts (the full daily budget,
        # 07.08). Alert-worthy dwell here starts at 15 minutes.
        "loiter_person_sec": 900,
    },
    "tr_golden_horn": {
        "name": "Golden Horn (Istanbul)", "city": "Istanbul", "country": "turkey",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=7VCk0oB0pDo",
        "page": "https://webcamera24.com/camera/turkey/clarionhotelgoldenhorn-cam/",
        "embed": "https://www.youtube.com/embed/7VCk0oB0pDo?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "tr_giresun_kalesi": {
        "name": "Giresun Kalesi (Castle)", "city": "Giresun", "country": "turkey",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=MMw0F-b-Q7c",
        "page": "https://webcamera24.com/camera/turkey/giresun-castle-cam/",
        "embed": "https://www.youtube.com/embed/MMw0F-b-Q7c?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    # Ankara addition (2026-07-22 deeper webcamera24 re-scan across every
    # Turkish city index): outdoor city park in Yenimahalle/Ankara, hosted
    # on the Ankara municipality-adjacent channel and confirmed
    # yt-dlp is_live=True. Adds a capital-city option to the Turkey grid
    # beyond the Istanbul/Giresun cluster.
    "tr_ankara_kivircik_park": {
        "name": "Kivircik Ali Parki (Ankara)", "city": "Ankara", "country": "turkey",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=jJlZiD3hZ80",
        "page": "https://webcamera24.com/camera/turkey/7984-ali-ozutemiz-kivircik-ali-parki-yenimahalle-ankara-canli-yayin/",
        "embed": "https://www.youtube.com/embed/jJlZiD3hZ80?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },

    # ================= Multi-country street/traffic cameras =================
    # Added 2026-07-17 for the country-generic collector. All resolve to a
    # YouTube Live via yt-dlp's ANDROID innertube client (the web client
    # started returning "page needs to be reloaded" on live streams). Every
    # one was verified end-to-end (resolve -> segment -> cv2 decode) on
    # 2026-07-17. `country` groups them into the collector's country ladder;
    # `tz` (where present) overrides the country default for the digest's
    # hour-of-week profile and day/night gate. YouTube manifests carry an
    # `expire=` timestamp - detect_core.resolve_stream caches per camera and
    # re-resolves shortly before expiry.
    #
    # --- Thailand (street / beach-road / traffic) ---
    "th_sukhumvit": {
        "name": "Sukhumvit Rd (Bangkok)", "city": "Bangkok", "country": "thailand",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=Q71sLS8h9a4",
        "page": "https://webcamera24.com/camera/thailand/sukhumvit-street/",
        "embed": "https://www.youtube.com/embed/Q71sLS8h9a4?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "th_chaweng_hooters": {
        "name": "Chaweng Beach Rd (Koh Samui)", "city": "Koh Samui", "country": "thailand",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=VR-x3HdhKLQ",
        "page": "https://webcamera24.com/camera/thailand/7108-hooters-cam-chaweng-live-street-webcam-stream-p-hd/",
        "embed": "https://www.youtube.com/embed/VR-x3HdhKLQ?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "th_nanai_road": {
        "name": "Nanai Rd (Patong)", "city": "Patong", "country": "thailand",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=WSm_r0eNl1E",
        "page": "https://webcamera24.com/camera/thailand/nanai-road-cam/",
        "embed": "https://www.youtube.com/embed/WSm_r0eNl1E?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "th_patong_sainamyen": {
        "name": "Sainamyen Rd (Patong)", "city": "Patong", "country": "thailand",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=_nvG0c9keWI",
        "page": "https://webcamera24.com/camera/thailand/patong-sainamyen-rd-cam/",
        "embed": "https://www.youtube.com/embed/_nvG0c9keWI?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "th_petchaburi_traffic": {
        "name": "Petchaburi Rd traffic (Bangkok)", "city": "Bangkok", "country": "thailand",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=a_bUVExv_Cg",
        "page": "https://webcamera24.com/camera/thailand/petchaburi-road-traffic-cam/",
        "embed": "https://www.youtube.com/embed/a_bUVExv_Cg?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "th_green_mango": {
        "name": "Soi Green Mango (Chaweng)", "city": "Koh Samui", "country": "thailand",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=DwKCna1mumk",
        "page": "https://webcamera24.com/camera/thailand/7098-hush-bar-soi-green-mango-chaweng-live-street-webcam-stream-p-hd/",
        "embed": "https://www.youtube.com/embed/DwKCna1mumk?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    # Operator additions 2026-07-17 (verified live 1080p, YouTube android-client).
    "th_sukhumvit_soi11": {
        "name": "Sukhumvit Soi 11 - El Gaucho (Bangkok)", "city": "Bangkok", "country": "thailand",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=UemFRPrl1hk",
        "page": "https://www.youtube.com/watch?v=UemFRPrl1hk",
        "embed": "https://www.youtube.com/embed/UemFRPrl1hk?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "th_chaweng_pancake": {
        "name": "Chaweng - Pancake Man (Koh Samui)", "city": "Koh Samui", "country": "thailand",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=e9T0L_POAOk",
        "page": "https://www.youtube.com/watch?v=e9T0L_POAOk",
        "embed": "https://www.youtube.com/embed/e9T0L_POAOk?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "th_chaweng_murphys": {
        "name": "Chaweng - Murphy's Irish Pub (Koh Samui)", "city": "Koh Samui", "country": "thailand",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=OBJ5Q0lWbqk",
        "page": "https://www.youtube.com/watch?v=OBJ5Q0lWbqk",
        "embed": "https://www.youtube.com/embed/OBJ5Q0lWbqk?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },

    # --- Japan (crossings / downtown streets) ---
    "jp_shinsaibashi": {
        "name": "Shinsaibashi (Osaka)", "city": "Osaka", "country": "japan",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=aVAO2wSUsPo",
        "page": "https://webcamera24.com/camera/japan/shinsaibashi-cam/",
        "embed": "https://www.youtube.com/embed/aVAO2wSUsPo?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "jp_kabukicho_crossing": {
        "name": "Kabukicho Crossing (Tokyo)", "city": "Tokyo", "country": "japan",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=ErHJBXTmm2Q",
        "page": "https://webcamera24.com/camera/japan/kabukicho-crossing/",
        "embed": "https://www.youtube.com/embed/ErHJBXTmm2Q?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "jp_kabukicho_shinjuku": {
        "name": "Kabukicho (Shinjuku, Tokyo)", "city": "Tokyo", "country": "japan",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=DjdUEyjx8GM",
        "page": "https://webcamera24.com/camera/japan/kabukicho-shinjuku-cam/",
        "embed": "https://www.youtube.com/embed/DjdUEyjx8GM?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "jp_cross_space": {
        "name": "Cross Space (Shinjuku)", "city": "Tokyo", "country": "japan",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=Zhmmh7l6KEw",
        "page": "https://webcamera24.com/camera/japan/cross-space-shinjuku/",
        "embed": "https://www.youtube.com/embed/Zhmmh7l6KEw?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "jp_shibuya": {
        "name": "Shibuya (Tokyo)", "city": "Tokyo", "country": "japan",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=ocQygJpZnhU",
        "page": "https://webcamera24.com/camera/japan/shibuya/",
        "embed": "https://www.youtube.com/embed/ocQygJpZnhU?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "jp_seibu_shinjuku": {
        "name": "Seibu-Shinjuku Station (Tokyo)", "city": "Tokyo", "country": "japan",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=lA6TaaMGgDo",
        "page": "https://webcamera24.com/camera/japan/seibu-shinjuku-station-cam/",
        "embed": "https://www.youtube.com/embed/lA6TaaMGgDo?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "jp_tenjin": {
        "name": "Tenjin Watanabe-dori (Fukuoka)", "city": "Fukuoka", "country": "japan",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=p326sZfmwHM",
        "page": "https://webcamera24.com/camera/japan/tenjin-watanabe-dori-avenue/",
        "embed": "https://www.youtube.com/embed/p326sZfmwHM?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "jp_kyoto_station": {
        "name": "Kyoto Station Bus Terminal", "city": "Kyoto", "country": "japan",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=v9rQqa_VTEY",
        "page": "https://webcamera24.com/camera/japan/kyoto-station-bus-terminal-cam/",
        "embed": "https://www.youtube.com/embed/v9rQqa_VTEY?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },

    # --- USA (Times Square / downtowns / main streets). tz set per camera:
    # this bench spans Eastern, Central and Pacific. ---
    "us_north_conway": {
        "name": "North Conway (NH)", "city": "North Conway", "country": "usa",
        "tz": "America/New_York",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=H8bFFw-0ZQE",
        "page": "https://webcamera24.com/camera/usa/north-conway/",
        "embed": "https://www.youtube.com/embed/H8bFFw-0ZQE?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "us_boston_common": {
        "name": "Boston Common (MA)", "city": "Boston", "country": "usa",
        "tz": "America/New_York",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=sWF5RQ_OzpM",
        "page": "https://webcamera24.com/camera/usa/boston-common-cam/",
        "embed": "https://www.youtube.com/embed/sWF5RQ_OzpM?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "us_times_square": {
        "name": "Times Square (NYC)", "city": "New York", "country": "usa",
        "tz": "America/New_York",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=z-jYdOIKcTQ",
        "page": "https://webcamera24.com/camera/usa/times-square-manhattan/",
        "embed": "https://www.youtube.com/embed/z-jYdOIKcTQ?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "us_bellevue_2nd": {
        "name": "Bellevue 2nd St (WA)", "city": "Bellevue", "country": "usa",
        "tz": "America/Los_Angeles",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=to8iWyVHNM4",
        "page": "https://webcamera24.com/camera/usa/bellevue-2ndstreet-station-cam/",
        "embed": "https://www.youtube.com/embed/to8iWyVHNM4?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "us_church_st_burlington": {
        "name": "Church St (Burlington, VT)", "city": "Burlington", "country": "usa",
        "tz": "America/New_York",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=zl1woMXGGmQ",
        "page": "https://webcamera24.com/camera/usa/church-street-burlington/",
        "embed": "https://www.youtube.com/embed/zl1woMXGGmQ?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "us_houston_downtown": {
        "name": "Houston Downtown (TX)", "city": "Houston", "country": "usa",
        "tz": "America/Chicago",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=wUQc3RoLAPs",
        "page": "https://webcamera24.com/camera/usa/houston-downtown/",
        "embed": "https://www.youtube.com/embed/wUQc3RoLAPs?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "us_apex_main": {
        "name": "Main St (Apex, NC)", "city": "Apex", "country": "usa",
        "tz": "America/New_York",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=xaHSBtKtWTs",
        "page": "https://webcamera24.com/camera/usa/main-street-apex-town/",
        "embed": "https://www.youtube.com/embed/xaHSBtKtWTs?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
    "us_putnam_square": {
        "name": "Putnam County Sq (Cookeville, TN)", "city": "Cookeville", "country": "usa",
        "tz": "America/Chicago",
        "kind": "youtube", "url": "https://www.youtube.com/watch?v=z8HYmP_gOhY",
        "page": "https://webcamera24.com/camera/usa/putnam-county-square-live/",
        "embed": "https://www.youtube.com/embed/z8HYmP_gOhY?autoplay=1&mute=1&playsinline=1&enablejsapi=1",
    },
}

# Live dashboard grid, in display order. Each slot has a primary camera and an
# ordered fallback chain: if the collector can't grab a frame from the current
# cam N times in a row (see CameraPool in collector.py), the shared pool
# advances down the priority ladder. Every FALLBACK_RETRY_MINUTES a resting
# camera is re-probed; if it recovers, it takes its ladder spot back.
#
# 2026-07 update: IBB (both livestream.ibb.gov.tr and kamerayayin.ibb.istanbul)
# tightened its geo-block to Turkey-only in early July 2026. Every IBB HLS now
# returns 403/502 from GCP us-east1 AND from any non-Turkey network - not just
# an outbound-IP issue with the VM. The two "Istanbul" grid slots have therefore
# been swapped for two more webcamera24/tvkur Konya cams that this project
# already had in the catalog and that work from any country. The IBB entries
# stay in CAMERAS above for anyone running from a Turkey-routed IP; if they
# unblock again they can be put back in this list.
#
# Rule for the fallback chains (updated 2026-07 after the 48h empty-run):
# every chain now spans MULTIPLE backends. The first three entries stay on
# webcamera24 / tvkur so a single-camera outage swaps into a sibling that
# reaches the same host; the last two are IBB `kamerayayin.ibb.istanbul`
# cameras plus `konya_ince_minareli` (also tvkur but on a different stream
# id) so a full tvkur or webcamera24 backend outage still has somewhere to
# go. Before this change every fallback lived on the same backend, so when
# tvkur went 404 all four grid slots died together and the report still
# said "reporting normally".
#
# IBB `_yeni` entries were geo-blocked from GCP us-east1 in early July 2026
# per the note above. They still belong at the tail of the chain: they cost
# nothing when they fail (the picker skips them within a few rounds) and
# unblock as soon as IBB relaxes the block or the VM is behind a TR proxy.
# ===================== Country priority ladders =========================
# 2026-07-17: the collector became country-generic. The grid always runs
# FOUR cameras from ONE country; each country has its own priority ladder
# (the CameraPool walks it top-down, assigning the first four cameras that
# are currently delivering frames). A country is only abandoned for the
# next one when its WHOLE ladder is exhausted (every camera resting or its
# host blocked) - a single dead camera just backfills from deeper in the
# same country's bench. Country order below IS the fallback order; the
# CountryDirector re-probes higher-priority countries shortly before each
# report and switches back if one has recovered.
#
# Turkey ladder (re-revised 2026-07-29): IBB FIRST, YouTube second.
# The 2026-07-21 revision put the YouTube tier at the head because IBB
# returned 403 from GCP. Both facts have since flipped: IBB delivers
# reliably through the Cloudflare-Worker relay, while YouTube starves
# Google-datacenter IPs at the SEGMENT level - resolves succeed, then
# googlevideo serves no data, so every YT attempt burns ~90s of open
# timeouts before failing. With YT at the head that cost ~15-20 min of
# empty grid after every service restart (measured live 2026-07-29,
# including with the bgutil PO-token provider installed - the CDN
# starvation beat the unauthenticated-token route too). IBB-first makes
# every boot settle within one round; the YT tier stays next in line so
# it is probed automatically whenever IBB thins, which is also the
# passive detector for a future YouTube thaw. Konya `tvkur` cams are the
# fast-fail lane (see collector.CameraPool): one miss rests them.
TURKEY_YT = [
    "tr_bulancak_meydan", "tr_golden_horn", "tr_giresun_kalesi",
    "tr_ankara_kivircik_park",
]
TURKEY_IBB = [
    # Operator-approved IBB set (2026-07-21): Taksim -> Beyazit -> Sarachane
    # -> Sultanahmet. Eyup Sultan moved down to the tail; Sarachane
    # promoted up from the tail. All four resolve through kamerayayin.ibb.istanbul,
    # so they ride the Cloudflare-Worker relay in one call.
    "taksim_yeni", "beyazit_meydan_yeni", "sarachane_yeni", "sultanahmet_1_yeni",
]
TURKEY_KONYA = [
    "konya_hukumet", "otogar_kavsagi", "konya_kulturpark", "konya_millet_caddesi",
]
TURKEY_TAIL = [
    "buyuk_camlica_yeni", "konya_ince_minareli", "eyup_sultan_yeni",
    "sultanahmet_2_yeni", "uskudar_yeni", "salacak_yeni", "kucukcekmece_yeni",
    "ulus_parki_yeni", "pierre_lotti_yeni", "emirgan_yeni", "kiz_kulesi_yeni",
    "hidiv_kasri_yeni", "dragos_yeni",
]
TURKEY_POOL = TURKEY_IBB + TURKEY_YT + TURKEY_KONYA + TURKEY_TAIL

# Foreign ladders: the operator's four per country first (verified live
# 2026-07-17), then the spares discovered from the same webcamera24 country
# listing (street / traffic / crossing views), deepest bench so a dead
# camera backfills without leaving the country.
THAILAND_POOL = [
    "th_sukhumvit", "th_chaweng_hooters", "th_nanai_road", "th_patong_sainamyen",
    "th_petchaburi_traffic", "th_green_mango",
    "th_sukhumvit_soi11", "th_chaweng_pancake", "th_chaweng_murphys",
]
JAPAN_POOL = [
    "jp_shinsaibashi", "jp_kabukicho_crossing", "jp_kabukicho_shinjuku", "jp_cross_space",
    "jp_shibuya", "jp_seibu_shinjuku", "jp_tenjin", "jp_kyoto_station",
]
USA_POOL = [
    "us_north_conway", "us_boston_common", "us_times_square", "us_bellevue_2nd",
    "us_church_st_burlington", "us_houston_downtown", "us_apex_main", "us_putnam_square",
]

# Ordered country ladder. `tz` is the DEFAULT timezone for the country's
# reports/day-night gate; a camera may override it with its own "tz"
# (the US bench spans Eastern/Central/Pacific).
COUNTRIES = {
    "turkey":   {"display": "Turkey",   "flag": "TR", "tz": "Europe/Istanbul", "pool": TURKEY_POOL},
    "thailand": {"display": "Thailand", "flag": "TH", "tz": "Asia/Bangkok",    "pool": THAILAND_POOL},
    "japan":    {"display": "Japan",    "flag": "JP", "tz": "Asia/Tokyo",      "pool": JAPAN_POOL},
    "usa":      {"display": "USA",      "flag": "US", "tz": "America/New_York", "pool": USA_POOL},
}
COUNTRY_ORDER = list(COUNTRIES)


def country_pool(country: str) -> list[str]:
    """The ordered camera ladder for one country (validated against CAMERAS)."""
    return [c for c in COUNTRIES[country]["pool"] if c in CAMERAS]


def camera_timezone(cam_id: str) -> str:
    """Camera's own tz if set, else its country default, else Istanbul."""
    cam = CAMERAS.get(cam_id, {})
    if cam.get("tz"):
        return cam["tz"]
    ctry = cam.get("country")
    if ctry and ctry in COUNTRIES:
        return COUNTRIES[ctry]["tz"]
    return "Europe/Istanbul"


# Country-level (lat, lon) used as the weather-fetch anchor by the
# forecasting notebook. Individual cameras don't need pinpoint accuracy -
# a Konya vs Istanbul afternoon differ mostly by 1-2 C and rain-yes/no,
# which the country capital captures fine for the 15-min forecast bins.
# Override per-camera by setting "lat"/"lon" on the CAMERAS entry.
COUNTRY_LATLON = {
    "turkey":   (41.0082, 28.9784),   # Istanbul
    "thailand": (13.7563, 100.5018),  # Bangkok
    "japan":    (35.6895, 139.6917),  # Tokyo
    "usa":      (40.7128, -74.0060),  # New York City (Eastern-anchored bench)
}


def camera_latlon(cam_id: str) -> tuple[float, float] | None:
    """The (lat, lon) to use when fetching weather for this camera.

    Per-camera "lat"/"lon" beats the country default. Cameras with no
    country and no explicit coords return None - the caller should skip
    them (the forecaster does so silently)."""
    cam = CAMERAS.get(cam_id, {})
    if cam.get("lat") is not None and cam.get("lon") is not None:
        return (float(cam["lat"]), float(cam["lon"]))
    ctry = cam.get("country")
    if ctry and ctry in COUNTRY_LATLON:
        return COUNTRY_LATLON[ctry]
    return None


# ISO 3166-1 alpha-2 country codes for the `holidays` library.
COUNTRY_ISO2 = {
    "turkey":   "TR",
    "thailand": "TH",
    "japan":    "JP",
    "usa":      "US",
}


def camera_country_iso2(cam_id: str) -> str | None:
    """Country ISO-2 code for the `holidays` library. None if unknown."""
    ctry = CAMERAS.get(cam_id, {}).get("country")
    return COUNTRY_ISO2.get(ctry)


# Per-camera counting-line override written by the dashboard UI (a user
# draws a line on a snapshot -> POST /api/lines/<cam>). Overrides the
# CAMERAS[cam]["line"] if present. Location is relative to the src/
# directory the collector runs from; the dashboard writes to the same
# path so both processes see the same source of truth.
def _lines_dir() -> Path:
    """The user-drawn line-config directory (src/data/lines/). Created
    lazily on first write."""
    return Path(__file__).resolve().parent.parent / "data" / "lines"


# Classes the crossing counter is allowed to filter on. Kept narrow on
# purpose - the picker checkboxes mirror this list, and an unknown token
# in a hand-edited override is rejected instead of silently dropping every
# real detection.
LINE_ALLOWED_CLASSES = frozenset({
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
})


def _valid_line_shape(line) -> bool:
    return (isinstance(line, list) and len(line) == 2
            and all(isinstance(pt, list) and len(pt) == 2
                    and all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0
                            for v in pt)
                    for pt in line))


def _valid_classes(classes) -> bool:
    """None means "count every class"; a list means "count only these".
    An empty list is rejected - the operator meant something, and an empty
    filter would silently count nothing."""
    if classes is None:
        return True
    if not isinstance(classes, list) or not classes:
        return False
    return all(isinstance(c, str) and c in LINE_ALLOWED_CLASSES
               for c in classes)


def _zones_dir() -> Path:
    """User-drawn analysis zones (src/data/zones/<cam>.json). Same
    write-from-dashboard / read-from-analysis contract as _lines_dir."""
    return Path(__file__).resolve().parent.parent / "data" / "zones"


ZONE_KINDS = frozenset({"loiter", "parking"})


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
    """The user-drawn zones for this camera ([] when none / malformed).
    Malformed files are ignored entry-by-entry - one bad polygon must not
    take down the good ones."""
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
    """Persist the user-drawn zones for one camera (full replace).
    Raises ValueError on a malformed payload - the API surfaces it as 400."""
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


def resolve_line(cam_id: str) -> list | None:
    """Return the counting line to use for this camera.

    Precedence: user override (data/lines/<cam>.json) beats the CAMERAS
    catalog entry. Returns None when neither exists. Malformed overrides
    fall back to the catalog silently - the collector must never crash
    because the dashboard wrote a bad file.
    """
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
    """Class filter for the crossing counter, if the user set one.

    None (the default) means "count every tracked class"; a validated
    list of class names means "count only these". Malformed overrides
    fall back to None so the counter degrades to the old permissive
    behavior instead of going silent."""
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
    """Persist a user-drawn line for this camera. Validates shape + the
    optional class filter; raises ValueError on garbage input so the API
    layer can return 400.

    `classes`: None keeps the current behavior (count every class); a list
    like ["person"] restricts the counter to the named classes (must all
    be in LINE_ALLOWED_CLASSES, and the list must be non-empty)."""
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
    """Delete a user-drawn line override; returns True if a file was
    removed, False if none existed."""
    p = _lines_dir() / f"{cam_id}.json"
    try:
        p.unlink()
        return True
    except OSError:
        return False


# Stamp every catalog entry with its own id (needed by the resolve cache in
# detect_core) and a country. The foreign cameras carry an explicit country;
# every other catalog entry predates the country field and is Turkish. Done
# once at import so every consumer - collector, notebook, digest - sees it.
for _cid, _cam in CAMERAS.items():
    _cam.setdefault("id", _cid)
    _cam.setdefault("country", "turkey")

# Back-compat: the old single global ladder is Turkey's ladder. Modules that
# still import FALLBACK_POOL (tests, host breaker) get Turkey unchanged.
FALLBACK_POOL = TURKEY_POOL


def _pool_fallbacks(primary: str, pool: list[str] | None = None) -> list[str]:
    pool = pool if pool is not None else FALLBACK_POOL
    return [c for c in pool if c != primary]


def build_grid_slots(country: str = "turkey") -> list[dict]:
    """Four generic grid slots for a country. slot_id stays generic
    (slot_1..slot_4) so the Firestore grid schema and the dashboard are
    country-agnostic; the human label comes from the active camera at
    runtime (see collector._slot_metadata)."""
    pool = country_pool(country)
    primaries = (pool + pool)[:4]              # pad if a country has < 4 cams
    slots = []
    for i, primary in enumerate(primaries, 1):
        slots.append({
            "slot_id":      f"slot_{i}",
            "display_area": f"Slot {i}",
            "primary":      primary,
            "fallbacks":    _pool_fallbacks(primary, pool),
        })
    return slots


# Default grid = the top-priority country (Turkey). The collector rebuilds
# these when the active country changes.
GRID_SLOTS = build_grid_slots("turkey")

# Backward compat for the viewer notebook / smoke tests: the four primary cams.
GRID_CAMERAS = [s["primary"] for s in GRID_SLOTS]


# --- auto-blacklist merge -----------------------------------------------------
# When the review UI accumulates enough "wrong" verdicts in a coherent screen
# area (see app/auto_blacklist.py), a polygon is appended to
# data/blacklist_auto.json. Merge those into cam["roi_exclude_class"] on
# import so the next collector burst drops the same false positives - no
# code change, no restart, no GPU.
def _merge_auto_blacklist() -> None:
    try:
        from app.auto_blacklist import load_auto_blacklist
    except ImportError:
        return    # tests or minimal envs without the module - just skip
    try:
        by_cam = load_auto_blacklist()
    except Exception:
        return
    for cam_id, cls_map in by_cam.items():
        cam = CAMERAS.get(cam_id)
        if not cam:
            continue
        # Deep-copy the class lists and dedupe before appending: this merge
        # re-runs on the collector's hot-reload timer, and the old in-place
        # extend() re-appended every auto polygon on every reload - the
        # lists grew by one duplicate per cycle and static catalog entries
        # shared their list objects with the merged result.
        existing = {c: list(p)
                    for c, p in (cam.get("roi_exclude_class") or {}).items()}
        for cls, polys in cls_map.items():
            cur = existing.setdefault(cls, [])
            for poly in polys:
                if poly not in cur:
                    cur.append(poly)
        cam["roi_exclude_class"] = existing


# --- confidence-boost merge ---------------------------------------------------
# Per-camera per-class confidence adjustment learned from the review UI's
# correct/wrong verdicts (see app/confidence_boost.py). Applied as a delta
# on top of DEFAULT_PER_CLASS_CONF; the collector clamps the effective
# value at read time. Live-reloaded by collector.main() every few rounds.
def _merge_confidence_boost() -> None:
    try:
        from app.confidence_boost import load_boosts
        from app.detect_core import DEFAULT_PER_CLASS_CONF
    except ImportError:
        return
    try:
        by_cam = load_boosts()
    except Exception:
        return
    for cam_id, cls_map in by_cam.items():
        cam = CAMERAS.get(cam_id)
        if not cam:
            continue
        pcc = dict(cam.get("per_class_conf") or DEFAULT_PER_CLASS_CONF)
        for cls, delta in cls_map.items():
            try:
                base = float(pcc.get(cls, DEFAULT_PER_CLASS_CONF.get(cls, 0.35)))
                pcc[cls] = max(0.10, min(0.80, base + float(delta)))
            except (TypeError, ValueError):
                continue
        cam["per_class_conf"] = pcc


# --- per-camera calibration merge ---------------------------------------------
# tools/calibrate_conf.py distills the review confusion matrix into an
# explicit conf gate per (cam, cls) at a target precision (plan WS4).
# Runs AFTER the boost merge and OVERRIDES it per pair: a calibrated gate
# beats a heuristic nudge; the nudge keeps covering pairs that don't have
# 30+ verdicts yet.
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
    """Public entry point for the collector's hot-reload timer. Re-runs
    the merges without importing/reloading the whole module. Order
    matters: calibration LAST so it overrides the boost delta per pair."""
    _merge_auto_blacklist()
    _merge_confidence_boost()
    _merge_per_camera_conf()


_merge_auto_blacklist()
_merge_confidence_boost()
_merge_per_camera_conf()


def active_cameras():
    """Cameras that have a usable URL (skips placeholders awaiting a YouTube id)."""
    return {k: v for k, v in CAMERAS.items() if v.get("url")}
