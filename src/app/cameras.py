"""Multi-country camera catalog for the single-camera live dashboard.

The dashboard analyses ONE camera at a time (picked in the notebook or the
dashboard UI); this file is the pool of choices — Thailand, Turkey, Japan
and USA public webcams.

Geo-restriction note: Turkey IBB streams (livestream.ibb.gov.tr, cam ids
starting with taksim/beyazit/eyup/etc.) return HTTP 404 for requests from
non-Turkey IPs — they are geo-restricted at the origin. Run those from a
Turkey-routed IP for live data; otherwise the collector logs MISS and the
dashboard shows "stream unavailable". The Thailand/Japan/USA cameras have
no such restriction.

(Was: "Camera catalog for the single-camera live dashboard.")

Each entry: `kind` is one of
  "hls"          direct .m3u8 (used as-is)
  "youtube"      YouTube live URL, iframe embed on frontend; backend requires
                 yt-dlp cookies to resolve HLS for detection
  "skyline"      skylinewebcams.com page, resolved via detect_core.resolve_skyline
  "webcamera24"  webcamera24.com page, resolved via detect_core.resolve_webcamera24
  "local_file"   absolute path to an uploaded MP4/MKV/MOV/AVI/WEBM

Optional per-entry keys:
  page  - the human-facing webcam page (also the resolver input for skyline/webcamera24)
  embed - iframe URL for the live player (auto-derived from `url` for youtube kind)
  conf  - per-camera YOLO confidence override
  line  - virtual counting line [[x1,y1], [x2,y2]] in normalized 0..1 coords
  loiter_person_sec / loiter_vehicle_sec - override the presence dwell thresholds
"""
from __future__ import annotations

import json
import time
from pathlib import Path


CAMERAS: dict[str, dict] = {
    # --- Thailand: street / beach-road / nightlife / traffic ---
    'th_sukhumvit': {
        'name': 'Sukhumvit Rd (Bangkok)',
        'city': 'Bangkok',
        'country': 'thailand',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=Q71sLS8h9a4',
        'page': 'https://webcamera24.com/camera/thailand/sukhumvit-street/',
        'embed': 'https://www.youtube.com/embed/Q71sLS8h9a4?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
    },
    'th_chaweng_hooters': {
        'name': 'Chaweng Beach Rd (Koh Samui)',
        'city': 'Koh Samui',
        'country': 'thailand',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=VR-x3HdhKLQ',
        'page': 'https://webcamera24.com/camera/thailand/7108-hooters-cam-chaweng-live-street-webcam-stream-p-hd/',
        'embed': 'https://www.youtube.com/embed/VR-x3HdhKLQ?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
    },
    'th_nanai_road': {
        'name': 'Nanai Rd (Patong)',
        'city': 'Patong',
        'country': 'thailand',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=WSm_r0eNl1E',
        'page': 'https://webcamera24.com/camera/thailand/nanai-road-cam/',
        'embed': 'https://www.youtube.com/embed/WSm_r0eNl1E?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
    },
    'th_patong_sainamyen': {
        'name': 'Sainamyen Rd (Patong)',
        'city': 'Patong',
        'country': 'thailand',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=_nvG0c9keWI',
        'page': 'https://webcamera24.com/camera/thailand/patong-sainamyen-rd-cam/',
        'embed': 'https://www.youtube.com/embed/_nvG0c9keWI?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
    },
    'th_petchaburi_traffic': {
        'name': 'Petchaburi Rd traffic (Bangkok)',
        'city': 'Bangkok',
        'country': 'thailand',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=a_bUVExv_Cg',
        'page': 'https://webcamera24.com/camera/thailand/petchaburi-road-traffic-cam/',
        'embed': 'https://www.youtube.com/embed/a_bUVExv_Cg?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
    },
    'th_green_mango': {
        'name': 'Soi Green Mango (Chaweng)',
        'city': 'Koh Samui',
        'country': 'thailand',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=DwKCna1mumk',
        'page': 'https://webcamera24.com/camera/thailand/7098-hush-bar-soi-green-mango-chaweng-live-street-webcam-stream-p-hd/',
        'embed': 'https://www.youtube.com/embed/DwKCna1mumk?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
    },
    'th_sukhumvit_soi11': {
        'name': 'Sukhumvit Soi 11 - El Gaucho (Bangkok)',
        'city': 'Bangkok',
        'country': 'thailand',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=UemFRPrl1hk',
        'page': 'https://www.youtube.com/watch?v=UemFRPrl1hk',
        'embed': 'https://www.youtube.com/embed/UemFRPrl1hk?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
    },
    'th_bophut_el_gaucho': {
        'name': "El Gaucho - Fisherman's Village (Bophut, Koh Samui)",
        'city': 'Koh Samui',
        'country': 'thailand',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=FyFAqPHBKiQ',
        'page': 'https://www.youtube.com/watch?v=FyFAqPHBKiQ',
        'embed': 'https://www.youtube.com/embed/FyFAqPHBKiQ?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
    },
    'th_chaweng_pancake': {
        'name': 'Chaweng - Pancake Man (Koh Samui)',
        'city': 'Koh Samui',
        'country': 'thailand',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=e9T0L_POAOk',
        'page': 'https://www.youtube.com/watch?v=e9T0L_POAOk',
        'embed': 'https://www.youtube.com/embed/e9T0L_POAOk?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
    },
    'th_chaweng_murphys': {
        'name': "Chaweng - Murphy's Irish Pub (Koh Samui)",
        'city': 'Koh Samui',
        'country': 'thailand',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=OBJ5Q0lWbqk',
        'page': 'https://www.youtube.com/watch?v=OBJ5Q0lWbqk',
        'embed': 'https://www.youtube.com/embed/OBJ5Q0lWbqk?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
    },
    # --- Turkey: Istanbul / Konya / Ankara / Giresun squares + markets ---
    'konya_hukumet': {
        'name': 'Konya - Hukumet Meydani / Sarraflar Yeralti Carsisi',
        'city': 'Konya',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://content.tvkur.com/l/c77i84vbb2nj4i0fr80g/master.m3u8',
        'page': 'https://webcamera24.com/camera/turkey/8043-sarraflar-yeralti-carsisi/',
        'embed': 'https://player.tvkur.com/l/c77i84vbb2nj4i0fr80g',
        'type': 'square/market',
    },
    'taksim': {
        'name': 'Taksim Meydani (legacy host - 404 outside Turkey)',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://livestream.ibb.gov.tr/cam_turistik/b_taksim_meydan.stream/playlist.m3u8',
        'type': 'square/retail',
    },
    'taksim_yeni': {
        'name': 'Taksim Meydani (live)',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://kamerayayin.ibb.istanbul/turistikcam/taksim.stream/playlist.m3u8',
        'page': 'https://istanbuluseyret.ibb.gov.tr/taksim-yeni/',
        'embed': None,
        'line': [[0.15, 0.74], [0.85, 0.74]],
        'type': 'square/retail',
        'roi_exclude_class': {'bus': [[[0.0, 0.84], [1.0, 0.84], [1.0, 1.0], [0.0, 1.0]], [[0.75, 0.7], [1.0, 0.7], [1.0, 1.0], [0.75, 1.0]]], 'car': [[[0.55, 0.84], [1.0, 0.84], [1.0, 1.0], [0.55, 1.0]], [[0.75, 0.7], [1.0, 0.7], [1.0, 1.0], [0.75, 1.0]], [[0.23, 0.9], [0.55, 0.9], [0.55, 1.0], [0.23, 1.0]]], 'truck': [[[0.55, 0.84], [1.0, 0.84], [1.0, 1.0], [0.55, 1.0]], [[0.75, 0.7], [1.0, 0.7], [1.0, 1.0], [0.75, 1.0]], [[0.23, 0.9], [0.55, 0.9], [0.55, 1.0], [0.23, 1.0]]]},
        'imgsz': 960,
    },
    'beyazit_meydan': {
        'name': 'Beyazit Meydani',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://livestream.ibb.gov.tr/cam_turistik/b_beyazitmeydani.stream/playlist.m3u8',
        'type': 'square/market-gateway',
    },
    'kapali_carsi': {
        'name': 'Kapali Carsi (Grand Bazaar)',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://livestream.ibb.gov.tr/cam_turistik/b_kapalicarsi.stream/playlist.m3u8',
        'type': 'market',
    },
    'misir_carsisi': {
        'name': 'Misir Carsisi (Spice Bazaar)',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://livestream.ibb.gov.tr/cam_turistik/b_misircarsisi.stream/playlist.m3u8',
        'type': 'market',
    },
    'sultanahmet_1': {
        'name': 'Sultanahmet (legacy host - 404 outside Turkey)',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://livestream.ibb.gov.tr/cam_turistik/b_sultanahmet.stream/playlist.m3u8',
        'type': 'tourist square',
    },
    'sultanahmet_1_yeni': {
        'name': 'Sultanahmet (live)',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://kamerayayin.ibb.istanbul/turistikcam/sultanahmet1.stream/playlist.m3u8',
        'page': 'https://istanbuluseyret.ibb.gov.tr/sultanahmet-1-yeni/',
        'embed': None,
        'line': [[0.05, 0.82], [0.95, 0.82]],
        'type': 'tourist square',
    },
    'beyazit_meydan_yeni': {
        'name': 'Beyazit Meydani (live)',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://kamerayayin.ibb.istanbul/turistikcam/beyazitmeydan.stream/playlist.m3u8',
        'page': 'https://istanbuluseyret.ibb.gov.tr/beyazit-meydani-yeni/',
        'embed': None,
        'line': [[0.33, 0.95], [0.33, 0.4]],
        'type': 'square/market-gateway',
        'imgsz': 960,
    },
    'eyup_sultan_yeni': {
        'name': 'Eyup Sultan (live)',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://kamerayayin.ibb.istanbul/turistikcam/eyupsultan.stream/playlist.m3u8',
        'page': 'https://istanbuluseyret.ibb.gov.tr/eyup-sultan-yeni/',
        'embed': None,
        'type': 'religious square',
    },
    'buyuk_camlica_yeni': {
        'name': 'Buyuk Camlica (live)',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://kamerayayin.ibb.istanbul/turistikcam/buyukcamlica.stream/playlist.m3u8',
        'page': 'https://istanbuluseyret.ibb.gov.tr/buyuk-camlica-yeni/',
        'embed': None,
        'type': 'park/vista',
    },
    'sarachane_yeni': {
        'name': 'Sarachane (live)',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://kamerayayin.ibb.istanbul/turistikcam/sarachane.stream/playlist.m3u8',
        'page': 'https://istanbuluseyret.ibb.gov.tr/sarachane-yeni/',
        'embed': None,
        'line': [[0.35, 0.62], [0.35, 0.42]],
        'type': 'civic square',
        'imgsz': 960,
    },
    'sultanahmet_2_yeni': {
        'name': 'Sultanahmet 2 (live)',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://kamerayayin.ibb.istanbul/turistikcam/sultanahmet2.stream/playlist.m3u8',
        'page': 'https://istanbuluseyret.ibb.gov.tr/sultanahmet-2-yeni/',
        'embed': None,
        'type': 'tourist square',
    },
    'uskudar_yeni': {
        'name': 'Uskudar (live)',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://kamerayayin.ibb.istanbul/turistikcam/uskudar.stream/playlist.m3u8',
        'page': 'https://istanbuluseyret.ibb.gov.tr/uskudar-yeni/',
        'embed': None,
        'type': 'square/transport',
    },
    'salacak_yeni': {
        'name': 'Salacak (live)',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://kamerayayin.ibb.istanbul/turistikcam/salacak.stream/playlist.m3u8',
        'page': 'https://istanbuluseyret.ibb.gov.tr/salacak-yeni/',
        'embed': None,
        'type': 'waterfront promenade',
    },
    'kucukcekmece_yeni': {
        'name': 'Kucukcekmece (live)',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://kamerayayin.ibb.istanbul/turistikcam/kucukcekmece.stream/playlist.m3u8',
        'page': 'https://istanbuluseyret.ibb.gov.tr/kucukcekmece-yeni/',
        'embed': None,
        'type': 'lakeside park',
    },
    'ulus_parki_yeni': {
        'name': 'Ulus Parki (live)',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://kamerayayin.ibb.istanbul/turistikcam/ulusparki.stream/playlist.m3u8',
        'page': 'https://istanbuluseyret.ibb.gov.tr/ulus-parki-yeni/',
        'embed': None,
        'type': 'park/vista',
    },
    'pierre_lotti_yeni': {
        'name': 'Pierre Lotti (live)',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://kamerayayin.ibb.istanbul/turistikcam/pierreloti.stream/playlist.m3u8',
        'page': 'https://istanbuluseyret.ibb.gov.tr/pierre-lotti-yeni/',
        'embed': None,
        'type': 'hilltop cafe/vista',
    },
    'emirgan_yeni': {
        'name': 'Emirgan (live)',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://kamerayayin.ibb.istanbul/turistikcam/emirgan.stream/playlist.m3u8',
        'page': 'https://istanbuluseyret.ibb.gov.tr/emirgan-yeni/',
        'embed': None,
        'type': 'park',
    },
    'kiz_kulesi_yeni': {
        'name': 'Kiz Kulesi (live)',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://kamerayayin.ibb.istanbul/turistikcam/kizkulesi.stream/playlist.m3u8',
        'page': 'https://istanbuluseyret.ibb.gov.tr/kiz-kulesi-yeni/',
        'embed': None,
        'type': 'waterfront landmark',
    },
    'hidiv_kasri_yeni': {
        'name': 'Hidiv Kasri (live)',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://kamerayayin.ibb.istanbul/turistikcam/hidivkasri.stream/playlist.m3u8',
        'page': 'https://istanbuluseyret.ibb.gov.tr/hidiv-kasri-yeni/',
        'embed': None,
        'type': 'palace grounds',
    },
    'dragos_yeni': {
        'name': 'Dragos (live)',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://kamerayayin.ibb.istanbul/turistikcam/dragos.stream/playlist.m3u8',
        'page': 'https://istanbuluseyret.ibb.gov.tr/dragos-yeni/',
        'embed': None,
        'type': 'coastal vista',
    },
    'kadikoy': {
        'name': 'Kadikoy',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://livestream.ibb.gov.tr/cam_turistik/b_kadikoy.stream/chunklist.m3u8',
        'page': 'https://istanbuluseyret.ibb.gov.tr/kadikoy/',
        'embed': None,
        'type': 'commerce/transit',
    },
    'eyup_sultan': {
        'name': 'Eyup Sultan',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://livestream.ibb.gov.tr/cam_turistik/b_eyupsultan.stream/playlist.m3u8',
        'type': 'tourist square',
    },
    'uskudar': {
        'name': 'Uskudar',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://livestream.ibb.gov.tr/cam_turistik/b_uskudar.stream/playlist.m3u8',
        'type': 'square/transport',
    },
    'otogar_kavsagi': {
        'name': 'Otogar Kavsagi',
        'city': 'Konya',
        'country': 'turkey',
        'kind': 'webcamera24',
        'url': 'https://webcamera24.com/camera/turkey/8044-otogar-kavsagi/',
        'page': 'https://webcamera24.com/camera/turkey/8044-otogar-kavsagi/',
        'embed': 'https://player.tvkur.com/l/c77i91vbb2nj4i0fr81g',
        'line': [[0.35, 0.65], [0.95, 0.25]],
        'type': 'junction/transit',
    },
    'konya_kulturpark': {
        'name': 'Konya - Kulturpark',
        'city': 'Konya',
        'country': 'turkey',
        'kind': 'webcamera24',
        'url': 'https://webcamera24.com/camera/turkey/8058-kulturpark/',
        'page': 'https://webcamera24.com/camera/turkey/8058-kulturpark/',
        'embed': 'https://player.tvkur.com/l/c77i6hb84cnrb6mlji3g',
        'type': 'park/commercial',
    },
    'konya_millet_caddesi': {
        'name': 'Konya - Millet Caddesi / Hastane Kavsagi',
        'city': 'Konya',
        'country': 'turkey',
        'kind': 'webcamera24',
        'url': 'https://webcamera24.com/camera/turkey/8046-millet-caddesi/',
        'page': 'https://webcamera24.com/camera/turkey/8046-millet-caddesi/',
        'embed': 'https://player.tvkur.com/l/c77i9cfbb2nj4i0fr82g',
        'line': [[0.03, 0.46], [0.95, 0.3]],
        'type': 'hospital junction / vehicular',
        'roi_exclude_class': {'person': [[[0.385, 0.17], [0.445, 0.17], [0.445, 0.3], [0.385, 0.3]]]},
    },
    'konya_ince_minareli': {
        'name': 'Konya - Ince Minareli Medrese (tram line)',
        'city': 'Konya',
        'country': 'turkey',
        'kind': 'hls',
        'url': 'https://content.tvkur.com/l/c77ib8vbb2nj4i0fr8bg/master.m3u8',
        'page': 'https://webcamera24.com/camera/turkey/8033-ince-minareli-medrese/',
        'embed': 'https://player.tvkur.com/l/c77ib8vbb2nj4i0fr8bg',
        'type': 'tram line / plaza',
    },
    'giresun_gazi': {
        'name': 'Giresun - Gazi Caddesi',
        'city': 'Giresun',
        'country': 'turkey',
        'kind': 'skyline',
        'url': 'https://www.skylinewebcams.com/en/webcam/turkey/giresun/giresun/gazi-street.html',
        'page': 'https://www.skylinewebcams.com/en/webcam/turkey/giresun/giresun/gazi-street.html',
        'embed': 'https://www.skylinewebcams.com/en/embed/turkey/giresun/giresun/gazi-street.html',
        'type': 'commercial street (geo-restricted)',
    },
    'tr_bulancak_meydan': {
        'name': 'Bulancak Meydani (Giresun)',
        'city': 'Giresun',
        'country': 'turkey',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=vn702Owd5Kk',
        'page': 'https://webcamera24.com/camera/turkey/bulancak-square-cam/',
        'embed': 'https://www.youtube.com/embed/vn702Owd5Kk?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
        'loiter_person_sec': 900,
    },
    'tr_golden_horn': {
        'name': 'Golden Horn (Istanbul)',
        'city': 'Istanbul',
        'country': 'turkey',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=7VCk0oB0pDo',
        'page': 'https://webcamera24.com/camera/turkey/clarionhotelgoldenhorn-cam/',
        'embed': 'https://www.youtube.com/embed/7VCk0oB0pDo?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
    },
    'tr_giresun_kalesi': {
        'name': 'Giresun Kalesi (Castle)',
        'city': 'Giresun',
        'country': 'turkey',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=MMw0F-b-Q7c',
        'page': 'https://webcamera24.com/camera/turkey/giresun-castle-cam/',
        'embed': 'https://www.youtube.com/embed/MMw0F-b-Q7c?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
    },
    'tr_ankara_kivircik_park': {
        'name': 'Kivircik Ali Parki (Ankara)',
        'city': 'Ankara',
        'country': 'turkey',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=jJlZiD3hZ80',
        'page': 'https://webcamera24.com/camera/turkey/7984-ali-ozutemiz-kivircik-ali-parki-yenimahalle-ankara-canli-yayin/',
        'embed': 'https://www.youtube.com/embed/jJlZiD3hZ80?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
    },
    # --- Japan: Osaka / Tokyo / Kyoto crossings + stations ---
    'jp_shinsaibashi': {
        'name': 'Shinsaibashi (Osaka)',
        'city': 'Osaka',
        'country': 'japan',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=aVAO2wSUsPo',
        'page': 'https://webcamera24.com/camera/japan/shinsaibashi-cam/',
        'embed': 'https://www.youtube.com/embed/aVAO2wSUsPo?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
    },
    'jp_kabukicho_crossing': {
        'name': 'Kabukicho Crossing (Tokyo)',
        'city': 'Tokyo',
        'country': 'japan',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=ErHJBXTmm2Q',
        'page': 'https://webcamera24.com/camera/japan/kabukicho-crossing/',
        'embed': 'https://www.youtube.com/embed/ErHJBXTmm2Q?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
    },
    'jp_kabukicho_shinjuku': {
        'name': 'Kabukicho (Shinjuku, Tokyo)',
        'city': 'Tokyo',
        'country': 'japan',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=DjdUEyjx8GM',
        'page': 'https://webcamera24.com/camera/japan/kabukicho-shinjuku-cam/',
        'embed': 'https://www.youtube.com/embed/DjdUEyjx8GM?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
    },
    'jp_cross_space': {
        'name': 'Cross Space (Shinjuku)',
        'city': 'Tokyo',
        'country': 'japan',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=Zhmmh7l6KEw',
        'page': 'https://webcamera24.com/camera/japan/cross-space-shinjuku/',
        'embed': 'https://www.youtube.com/embed/Zhmmh7l6KEw?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
    },
    'jp_shibuya': {
        'name': 'Shibuya (Tokyo)',
        'city': 'Tokyo',
        'country': 'japan',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=ocQygJpZnhU',
        'page': 'https://webcamera24.com/camera/japan/shibuya/',
        'embed': 'https://www.youtube.com/embed/ocQygJpZnhU?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
    },
    'jp_seibu_shinjuku': {
        'name': 'Seibu-Shinjuku Station (Tokyo)',
        'city': 'Tokyo',
        'country': 'japan',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=lA6TaaMGgDo',
        'page': 'https://webcamera24.com/camera/japan/seibu-shinjuku-station-cam/',
        'embed': 'https://www.youtube.com/embed/lA6TaaMGgDo?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
    },
    'jp_tenjin': {
        'name': 'Tenjin Watanabe-dori (Fukuoka)',
        'city': 'Fukuoka',
        'country': 'japan',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=p326sZfmwHM',
        'page': 'https://webcamera24.com/camera/japan/tenjin-watanabe-dori-avenue/',
        'embed': 'https://www.youtube.com/embed/p326sZfmwHM?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
    },
    'jp_kyoto_station': {
        'name': 'Kyoto Station Bus Terminal',
        'city': 'Kyoto',
        'country': 'japan',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=v9rQqa_VTEY',
        'page': 'https://webcamera24.com/camera/japan/kyoto-station-bus-terminal-cam/',
        'embed': 'https://www.youtube.com/embed/v9rQqa_VTEY?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
    },
    # --- USA: town centers + traffic ---
    'us_north_conway': {
        'name': 'North Conway (NH)',
        'city': 'North Conway',
        'country': 'usa',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=H8bFFw-0ZQE',
        'page': 'https://webcamera24.com/camera/usa/north-conway/',
        'embed': 'https://www.youtube.com/embed/H8bFFw-0ZQE?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
        'tz': 'America/New_York',
    },
    'us_boston_common': {
        'name': 'Boston Common (MA)',
        'city': 'Boston',
        'country': 'usa',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=sWF5RQ_OzpM',
        'page': 'https://webcamera24.com/camera/usa/boston-common-cam/',
        'embed': 'https://www.youtube.com/embed/sWF5RQ_OzpM?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
        'tz': 'America/New_York',
    },
    'us_times_square': {
        'name': 'Times Square (NYC)',
        'city': 'New York',
        'country': 'usa',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=z-jYdOIKcTQ',
        'page': 'https://webcamera24.com/camera/usa/times-square-manhattan/',
        'embed': 'https://www.youtube.com/embed/z-jYdOIKcTQ?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
        'tz': 'America/New_York',
    },
    'us_bellevue_2nd': {
        'name': 'Bellevue 2nd St (WA)',
        'city': 'Bellevue',
        'country': 'usa',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=to8iWyVHNM4',
        'page': 'https://webcamera24.com/camera/usa/bellevue-2ndstreet-station-cam/',
        'embed': 'https://www.youtube.com/embed/to8iWyVHNM4?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
        'tz': 'America/Los_Angeles',
    },
    'us_church_st_burlington': {
        'name': 'Church St (Burlington, VT)',
        'city': 'Burlington',
        'country': 'usa',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=zl1woMXGGmQ',
        'page': 'https://webcamera24.com/camera/usa/church-street-burlington/',
        'embed': 'https://www.youtube.com/embed/zl1woMXGGmQ?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
        'tz': 'America/New_York',
    },
    'us_houston_downtown': {
        'name': 'Houston Downtown (TX)',
        'city': 'Houston',
        'country': 'usa',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=wUQc3RoLAPs',
        'page': 'https://webcamera24.com/camera/usa/houston-downtown/',
        'embed': 'https://www.youtube.com/embed/wUQc3RoLAPs?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
        'tz': 'America/Chicago',
    },
    'us_apex_main': {
        'name': 'Main St (Apex, NC)',
        'city': 'Apex',
        'country': 'usa',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=xaHSBtKtWTs',
        'page': 'https://webcamera24.com/camera/usa/main-street-apex-town/',
        'embed': 'https://www.youtube.com/embed/xaHSBtKtWTs?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
        'tz': 'America/New_York',
    },
    'us_putnam_square': {
        'name': 'Putnam County Sq (Cookeville, TN)',
        'city': 'Cookeville',
        'country': 'usa',
        'kind': 'youtube',
        'url': 'https://www.youtube.com/watch?v=z8HYmP_gOhY',
        'page': 'https://webcamera24.com/camera/usa/putnam-county-square-live/',
        'embed': 'https://www.youtube.com/embed/z8HYmP_gOhY?autoplay=1&mute=1&playsinline=1&enablejsapi=1',
        'tz': 'America/Chicago',
    },
}


# Stamp id + country on every entry so every consumer sees them without
# having to look up the key.
for _cid, _cam in CAMERAS.items():
    _cam.setdefault("id", _cid)
    _cam.setdefault("country", "thailand")


def active_cameras() -> dict[str, dict]:
    """Cameras that have a usable URL (skips placeholders)."""
    return {k: v for k, v in CAMERAS.items() if v.get("url")}


# ---------------------------------------------------------------------------
# User-drawn counting line: dashboard POSTs to /api/lines, the analysis loop
# picks up the file within a few seconds. Lives at src/data/lines/<cam>.json.
# ---------------------------------------------------------------------------

LINE_ALLOWED_CLASSES = frozenset({
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
})


def _lines_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "lines"


def _valid_line_shape(line) -> bool:
    return (isinstance(line, list) and len(line) == 2
            and all(isinstance(pt, list) and len(pt) == 2
                    and all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0
                            for v in pt)
                    for pt in line))


def _valid_classes(classes) -> bool:
    """None means "count every class"; a non-empty list of allowed names
    means "count only these". An empty list is rejected."""
    if classes is None:
        return True
    if not isinstance(classes, list) or not classes:
        return False
    return all(isinstance(c, str) and c in LINE_ALLOWED_CLASSES
               for c in classes)


def resolve_line(cam_id: str) -> list | None:
    """Return the counting line to use. User override beats CAMERAS[cam]["line"];
    None when neither exists. Malformed overrides fall back silently."""
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
    """None = count every tracked class; a list = only these classes."""
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
    p = _lines_dir() / f"{cam_id}.json"
    try:
        p.unlink()
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# User-drawn analysis zones (loiter / parking polygons). Lives at
# src/data/zones/<cam>.json. Same write-from-dashboard / read-from-analysis
# contract as the lines file.
# ---------------------------------------------------------------------------

ZONE_KINDS = frozenset({"loiter", "parking"})


def _zones_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "zones"


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
    """The user-drawn zones for this camera ([] when none / malformed)."""
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


# ---------------------------------------------------------------------------
# Per-camera confidence-calibration file (tools/calibrate_conf.py output).
# Optional; loaded once at import and re-run by reload_review_overrides()
# so hot-swap of a fresh calibration takes effect without a restart.
# ---------------------------------------------------------------------------

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
    """Hot-reload hook (called by the collector timer). Category B/C were
    cut so only per-camera confidence calibration remains."""
    _merge_per_camera_conf()


_merge_per_camera_conf()
