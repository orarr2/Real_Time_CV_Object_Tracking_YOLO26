"""Resolved-stream cache + YouTube android-client resolution.

The collector resolves a stream URL every sampling round for every slot.
Direct HLS is free, but a YouTube resolve shells out to yt-dlp (~3-5s on
the e2-micro) and a webcamera24 resolve scrapes a page. Caching the
resolved URL until its manifest nears expiry is what keeps a four-slot,
40-second loop from being dominated by resolution cost. These tests pin:
  * direct HLS is passed through untouched and never cached;
  * a resolved YouTube/webcamera24 URL is reused within its TTL and
    re-resolved after it (or after an explicit invalidation);
  * the googlevideo `expire=` timestamp drives the TTL when present.
"""
import re

from app import detect_core as dc


def setup_function(_):
    dc._RESOLVE_CACHE.clear()


def test_direct_hls_passthrough_not_cached():
    cam = {"id": "x", "kind": "hls", "url": "https://cdn/live.m3u8"}
    assert dc.resolve_stream(cam, now=1000) == "https://cdn/live.m3u8"
    assert "x" not in dc._RESOLVE_CACHE


def test_youtube_resolve_is_cached_within_ttl(monkeypatch):
    calls = []

    def fake(cam):
        calls.append(cam["id"])
        return "https://manifest.googlevideo.com/x/expire/2000/hls.m3u8"

    monkeypatch.setattr(dc, "_resolve_uncached", fake)
    cam = {"id": "jp_shibuya", "kind": "youtube", "url": "https://youtu.be/abc"}
    # First call resolves; second (within TTL) is served from cache.
    assert dc.resolve_stream(cam, now=1000).endswith("hls.m3u8")
    assert dc.resolve_stream(cam, now=1500).endswith("hls.m3u8")
    assert calls == ["jp_shibuya"]


def test_expire_timestamp_drives_reresolution(monkeypatch):
    calls = []
    EXP = 1784313381                                 # a real googlevideo expiry

    def fake(cam):
        calls.append(1)
        return f"https://manifest.googlevideo.com/api/expire/{EXP}/n{len(calls)}.m3u8"

    monkeypatch.setattr(dc, "_resolve_uncached", fake)
    cam = {"id": "th_x", "kind": "youtube", "url": "u"}
    dc.resolve_stream(cam, now=EXP - 600)            # good_until = EXP-120
    dc.resolve_stream(cam, now=EXP - 121)            # still fresh
    assert len(calls) == 1
    dc.resolve_stream(cam, now=EXP - 119)            # past re-resolve point
    assert len(calls) == 2


def test_fallback_ttl_when_no_expire(monkeypatch):
    monkeypatch.setattr(dc, "_resolve_uncached", lambda cam: "https://tvkur/master.m3u8")
    monkeypatch.setattr(dc, "_RESOLVE_TTL_FALLBACK", 900)
    cam = {"id": "tr_k", "kind": "webcamera24", "url": "u", "page": "p"}
    dc.resolve_stream(cam, now=1000)
    _, good_until = dc._RESOLVE_CACHE["tr_k"]
    assert good_until == 1000 + 900


def test_invalidate_forces_reresolution(monkeypatch):
    calls = []
    monkeypatch.setattr(dc, "_resolve_uncached",
                        lambda cam: f"https://x/expire/9999999999/{len(calls)}.m3u8" or calls.append(1))

    def fake(cam):
        calls.append(1)
        return "https://x/expire/9999999999/live.m3u8"

    monkeypatch.setattr(dc, "_resolve_uncached", fake)
    cam = {"id": "us_1", "kind": "youtube", "url": "u"}
    dc.resolve_stream(cam, now=1000)
    dc.invalidate_resolved("us_1")
    dc.resolve_stream(cam, now=1001)
    assert len(calls) == 2


def test_no_id_means_always_live(monkeypatch):
    calls = []
    monkeypatch.setattr(dc, "_resolve_uncached",
                        lambda cam: calls.append(1) or "https://x/live.m3u8")
    cam = {"kind": "youtube", "url": "u"}       # no id -> uncacheable
    dc.resolve_stream(cam, now=1000)
    dc.resolve_stream(cam, now=1001)
    assert len(calls) == 2


def test_resolve_with_default_now_uses_wall_clock(monkeypatch):
    """resolve_stream(cam) with no `now` must fall back to time.time() -
    the notebook and any ad-hoc caller invoke it that way. (Regressed once:
    detect_core called time.time() without importing time.)"""
    monkeypatch.setattr(dc, "_resolve_uncached",
                        lambda cam: "https://x/expire/9999999999/live.m3u8")
    cam = {"id": "def_now", "kind": "youtube", "url": "u"}
    url = dc.resolve_stream(cam)                  # no now= -> wall clock
    assert url.endswith("live.m3u8")
    assert "def_now" in dc._RESOLVE_CACHE


def test_expire_regex_matches_googlevideo_shapes():
    for url, exp in [
        ("https://m.googlevideo.com/api/manifest/hls_playlist/expire/1784313381/ei/x", 1784313381),
        ("https://host/path?expire=1700000000&foo=bar", 1700000000),
    ]:
        m = dc._EXPIRE_RE.search(url)
        assert m and int(m.group(1)) == exp
