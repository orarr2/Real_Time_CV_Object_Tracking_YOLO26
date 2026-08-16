"""HLS rendition picker: decode the lightest stream that still covers the
YOLO input, never the 1080p default the CDN lists first."""
from app.detect_core import _pick_variant

MASTER = """#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=4200000,RESOLUTION=1920x1080
1080p/index.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=1800000,RESOLUTION=1280x720
720p/index.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=700000,RESOLUTION=640x360
360p/index.m3u8
"""


def test_picks_smallest_tall_enough():
    assert _pick_variant(MASTER, min_height=640) == "720p/index.m3u8"


def test_falls_back_to_tallest_when_none_cover():
    only_small = """#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=700000,RESOLUTION=640x360
360p/index.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=400000,RESOLUTION=426x240
240p/index.m3u8
"""
    assert _pick_variant(only_small, min_height=640) == "360p/index.m3u8"


def test_single_variant_without_resolution():
    pl = """#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=2500000
live/index.m3u8
"""
    assert _pick_variant(pl) == "live/index.m3u8"


def test_media_playlist_returns_none():
    pl = """#EXTM3U
#EXTINF:2.0,
seg001.ts
"""
    assert _pick_variant(pl) is None
