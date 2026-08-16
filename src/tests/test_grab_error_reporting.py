"""Grab-failure reporting + stale-playlist defenses.

The 2026-07-16 all-miss incident: every camera failed in ~1.7s with an
opaque "MISS (RuntimeError: empty frame)" while the exact same code path,
run standalone, fetched and decoded fine. The mask was detect_core
swallowing the real error at four different stages. These tests pin the
fixes:
  * last_grab_error() names the stage (playlist/chunklist/segment/decode)
    and the underlying exception, so the journal shows the true cause;
  * a rotated-out newest segment (instant 404) walks BACK to an older
    sibling instead of reporting an empty grab.
"""
import urllib.error

import numpy as np

from app import detect_core as dc

URL = "https://kamerayayin.ibb.istanbul/cam/x/playlist.m3u8"
MASTER = ("#EXTM3U\n"
          "#EXT-X-STREAM-INF:BANDWIDTH=2000000,RESOLUTION=1920x1080\n"
          "chunklist_w123.m3u8\n")
CHUNKS = ("#EXTM3U\n"
          "#EXT-X-TARGETDURATION:10\n"
          "media_1.ts\nmedia_2.ts\nmedia_3.ts\n")


class FakeCap:
    """Stand-in for cv2.VideoCapture over a downloaded .ts: serves `n`
    read()/grab() operations, then reports end-of-stream."""

    def __init__(self, n):
        self.n = n

    def read(self):
        if self.n <= 0:
            return False, None
        self.n -= 1
        return True, np.zeros((4, 4, 3), np.uint8)

    def grab(self):
        if self.n <= 0:
            return False
        self.n -= 1
        return True

    def release(self):
        pass


def http_router(fail_stage, segment_ok=(), fetched=None):
    """Fake _http_get: 404s the requested stage, records every URL."""
    def fake(url, extra_headers=None, max_bytes=None):
        if fetched is not None:
            fetched.append(url)
        if url.endswith("playlist.m3u8"):
            if fail_stage == "playlist":
                raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
            return MASTER.encode()
        if "chunklist" in url:
            if fail_stage == "chunklist":
                raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
            return CHUNKS.encode()
        name = url.rsplit("/", 1)[1]
        if fail_stage == "segment" and name not in segment_ok:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        return b"\x47" * 188
    return fake


def test_playlist_404_is_named(monkeypatch):
    monkeypatch.setattr(dc, "_http_get", http_router("playlist"))
    assert dc.grab_burst(URL, n=2, stride=13) == []
    err = dc.last_grab_error()
    assert err.startswith("playlist") and "404" in err


def test_chunklist_404_is_named(monkeypatch):
    """The prime stale-cache suspect: master playlist points at a rotated
    wowza chunklist token -> instant 404 one level down."""
    monkeypatch.setattr(dc, "_http_get", http_router("chunklist"))
    assert dc.grab_burst(URL, n=2, stride=13) == []
    err = dc.last_grab_error()
    assert err.startswith("chunklist") and "404" in err


def test_segment_404_walks_back_to_an_older_sibling(monkeypatch):
    """Newest .ts already purged from the edge: the grab must rescue itself
    from the next-freshest segment, not report empty."""
    fetched = []
    monkeypatch.setattr(dc, "_http_get",
                        http_router("segment", segment_ok={"media_2.ts"},
                                    fetched=fetched))
    monkeypatch.setattr(dc, "_open_cap", lambda p: FakeCap(n=20))
    frames = dc.grab_burst(URL, n=2, stride=13)
    assert len(frames) == 2
    assert dc.last_grab_error() is not None   # the 404 was still recorded
    seg_tries = [u.rsplit("/", 1)[1] for u in fetched if u.endswith(".ts")]
    assert seg_tries[0] == "media_3.ts"       # newest attempted first
    assert "media_2.ts" in seg_tries          # ...then the rescue sibling


def test_all_segments_404_reports_segment_stage(monkeypatch):
    monkeypatch.setattr(dc, "_http_get", http_router("segment"))
    assert dc.grab_burst(URL, n=2, stride=13) == []
    err = dc.last_grab_error()
    assert err.startswith("segment") and "404" in err


def test_downloaded_but_undecodable_reports_decode_stage(monkeypatch):
    monkeypatch.setattr(dc, "_http_get", http_router(None))
    monkeypatch.setattr(dc, "_open_cap", lambda p: FakeCap(n=0))
    assert dc.grab_burst(URL, n=2, stride=13) == []
    assert "decode" in dc.last_grab_error()


def test_playlist_failure_skips_the_duplicate_single_frame_retry(monkeypatch):
    """A refused playlist refuses the single-frame retry identically -
    grab_burst must knock ONCE, not twice (halves the request rate exactly
    when a blocking host is counting knocks)."""
    fetched = []
    monkeypatch.setattr(dc, "_http_get", http_router("playlist", fetched=fetched))
    assert dc.grab_burst(URL, n=2, stride=13) == []
    assert len(fetched) == 1


def test_successful_grab_leaves_no_error(monkeypatch):
    monkeypatch.setattr(dc, "_http_get", http_router(None))
    monkeypatch.setattr(dc, "_open_cap", lambda p: FakeCap(n=20))
    frames = dc.grab_burst(URL, n=2, stride=13)
    assert len(frames) == 2
    assert dc.last_grab_error() is None


def test_playlists_are_fetched_with_no_cache_header():
    h = dc._no_cache({"Referer": "https://istanbuluseyret.ibb.gov.tr/"})
    assert h["Cache-Control"] == "no-cache"
    assert h["Referer"] == "https://istanbuluseyret.ibb.gov.tr/"
