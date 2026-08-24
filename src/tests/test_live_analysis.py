"""fix 2 live-analysis engine: accumulators, layer semantics, manager.

Run from src/:  python -m pytest tests -q
"""
import json
import threading

import numpy as np
import pytest

from app import live_analysis as la
from app.tracker import Track

SHAPE = (360, 640)


def _box(x, y, w=30, h=60, cls="person", conf=0.9):
    return {"x1": x, "y1": y, "x2": x + w, "y2": y + h,
            "cls": cls, "conf": conf}


def _kps_for(b, conf=0.9):
    """17 plausible keypoints inside a box (COCO order not important for
    drawing - draw_skeleton only needs x,y,conf per index)."""
    x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
    cx = (x1 + x2) / 2
    return [[cx + (i % 3 - 1) * 5, y1 + (y2 - y1) * (i / 16.0), conf]
            for i in range(17)]


# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------

def test_update_crossings_counts_in_then_out():
    # DEFAULT_LINE is horizontal at y=0.62: crossing downward = "in".
    line = la.DEFAULT_LINE
    tr = Track(1, _box(300, 180 - 60), 0.0)          # foot y=180 (0.50) above
    sides, cross = {}, {"in": 0, "out": 0}
    la.update_crossings(sides, [tr], SHAPE, line, cross)
    assert cross == {"in": 0, "out": 0}              # first sighting: no flip
    tr.add(_box(300, 252 - 60), 1.0)                 # foot y=252 (0.70) below
    la.update_crossings(sides, [tr], SHAPE, line, cross)
    assert cross == {"in": 1, "out": 0}
    tr.add(_box(300, 180 - 60), 2.0)                 # back up
    la.update_crossings(sides, [tr], SHAPE, line, cross)
    assert cross == {"in": 1, "out": 1}


def test_update_crossings_skips_coasting_tracks():
    line = la.DEFAULT_LINE
    tr = Track(1, _box(300, 120), 0.0)
    tr.misses = 1                                    # coasting - no fresh box
    sides, cross = {}, {}
    la.update_crossings(sides, [tr], SHAPE, line, cross)
    assert sides == {} and cross == {}


def test_update_crossings_ignores_landing_exactly_on_line():
    # A foot point that lands EXACTLY on the line has no signed side; the
    # old convention treated 0 as "positive" and double-counted a track
    # that jittered neg -> 0 -> neg. Now side == 0 is skipped entirely
    # (neither counted nor allowed to reset the last known side), so the
    # touch-and-return produces zero crossings.
    # Build a line ON the exact centroid row of our synthetic boxes:
    # foot_y = 180, SHAPE height = 360, so ny = 0.5 lands right on it.
    line = [[0.10, 0.50], [0.90, 0.50]]
    tr = Track(1, _box(300, 180 - 60), 0.0)          # foot y = 180 (on)
    sides, cross = {}, {"in": 0, "out": 0}
    la.update_crossings(sides, [tr], SHAPE, line, cross)
    assert cross == {"in": 0, "out": 0}
    # A subsequent frame ABOVE the line establishes a real signed side.
    tr.add(_box(300, 170 - 60), 1.0)
    la.update_crossings(sides, [tr], SHAPE, line, cross)
    # Coming from "on the line" back to a side is NOT a crossing -
    # prev was skipped so there is no signed previous state yet.
    assert cross == {"in": 0, "out": 0}
    # Now cross for real - established side -> the other side.
    tr.add(_box(300, 260 - 60), 2.0)
    la.update_crossings(sides, [tr], SHAPE, line, cross)
    assert cross["in"] + cross["out"] == 1


def test_bump_heat_and_grid_from_tracks():
    grid = [[0.0] * la.GRID_W for _ in range(la.GRID_H)]
    b = _box(320 - 15, 180 - 60)                     # foot at frame center
    la.bump_heat(grid, [b], SHAPE, 2.5)
    assert sum(v for row in grid for v in row) == pytest.approx(2.5)
    tr = Track(1, b, 0.0)
    tr.add(_box(400, 200), 1.0)
    g2 = la.grid_from_tracks([tr], SHAPE)
    assert sum(v for row in g2 for v in row) == pytest.approx(2.0)


def test_first_tick_heat_weight_uses_pacing_target(monkeypatch):
    # The first _accumulate call has no prior timestamp; the old code
    # weighted it with an arbitrary 1.0, so the boot sample carried
    # ~25% more heat than the ~0.8s ticks that followed. Now it borrows
    # TICK_TARGET_S, matching what a normal tick banks.
    from app.tracker import BurstTracker
    sess = la.LiveSession.__new__(la.LiveSession)    # bypass __init__ (needs cam/model)
    sess.cam = {"id": "camX"}
    sess.cam_id = "camX"
    sess.tracker = BurstTracker(SHAPE)
    sess.heat = [[0.0] * la.GRID_W for _ in range(la.GRID_H)]
    sess.heat_since = None
    sess.line = la.DEFAULT_LINE
    sess.line_classes = None
    sess.cross = {"in": 0, "out": 0}
    sess._line_sides = {}
    sess._last_cross_ts = {}
    sess._last_tick = None
    # Skip the hot-reload path on this fixture-only tick: pretend the
    # next check is far in the future so _maybe_reload_line and
    # _maybe_reload_zones short-circuit.
    sess._line_mtime = None
    sess._next_line_check = 1e18
    sess._zones_mtime = None
    sess._next_zones_check = 1e18
    sess.zones = []
    frame = np.zeros((*SHAPE, 3), dtype=np.uint8)    # crossing-snap needs the array
    boxes = [_box(320 - 15, 180 - 60)]               # foot at frame center
    sess._accumulate(frame, boxes, now=100.0)
    total = sum(v for row in sess.heat for v in row)
    assert total == pytest.approx(la.TICK_TARGET_S)
    assert sess.heat_since == 100.0
    assert sess._last_tick == 100.0


# ---------------------------------------------------------------------------
# Camera resolution
# ---------------------------------------------------------------------------

def test_resolve_cam_registry():
    cam = la.resolve_cam("th_green_mango")
    assert cam["id"] == "th_green_mango"
    assert cam["url"].startswith("http")


def test_resolve_cam_local_slots(tmp_path):
    p = tmp_path / "local_grid.json"
    p.write_text(json.dumps({"slots": [
        {"slot_id": "local_0", "placeholder_name": "Sukhumvit Rd",
         "placeholder_embed": "https://www.youtube.com/embed/Q71sLS8h9a4?x=1"},
        {"slot_id": "local_1", "placeholder_name": "Konya",
         "placeholder_hls": "/tvkur/abc123/master.m3u8"},
    ]}), encoding="utf-8")
    yt = la.resolve_cam("local_0", grid_path=p)
    assert yt["kind"] == "youtube"
    assert yt["url"] == "https://www.youtube.com/watch?v=Q71sLS8h9a4"
    hls = la.resolve_cam("local_1", grid_path=p)
    assert hls["kind"] == "hls"
    assert hls["url"] == "https://content.tvkur.com/l/abc123/master.m3u8"
    with pytest.raises(ValueError):
        la.resolve_cam("local_9", grid_path=p)


def test_resolve_cam_local_skyline_slot(tmp_path):
    # A skyline camera picked via the notebook picker only carries a
    # placeholder_page (no HLS, no embed) - the picker doesn't know how to
    # resolve its rotating token. Before the fix the slot fell through to
    # ValueError("no analyzable stream") and the "analyze" button on that
    # tile returned 404; now the slot resolves to kind="skyline" so
    # detect_core.resolve_stream can chase the token live.
    p = tmp_path / "local_grid.json"
    p.write_text(json.dumps({"slots": [
        {"slot_id": "local_sky",
         "placeholder_name": "Gazi Street, Giresun",
         "placeholder_page": ("https://www.skylinewebcams.com/en/webcam/"
                              "turkey/giresun/giresun/gazi-street.html")},
    ]}), encoding="utf-8")
    sk = la.resolve_cam("local_sky", grid_path=p)
    assert sk["kind"] == "skyline"
    assert sk["page"] == sk["url"]
    assert "skylinewebcams.com" in sk["url"]


# ---------------------------------------------------------------------------
# Layer semantics (fix 2: each layer draws ONLY its own information)
# ---------------------------------------------------------------------------

CAP_H = 60   # everything below this row must be untouched by caption-only layers


def test_pose_layer_draws_no_detection_boxes():
    # 2026-08-24 contract update: every PERSON gets a stable per-track
    # colored box (color pairs the object with its side card), with or
    # without a skeleton. Vehicles stay unannotated - the layer is still
    # people-only.
    img = np.zeros((*SHAPE, 3), dtype=np.uint8)
    person = _box(100, 100)                          # no kps -> too far
    person["tid"] = 7
    car = _box(400, 200, w=100, h=40, cls="car")
    out = la.draw_pose_layer(img.copy(), [person, car])
    assert out[100:170, 90:140].sum() > 0            # person box drawn
    assert out[200:240, 400:500].sum() == 0          # car untouched
    person["kps"] = _kps_for(person)
    out2 = la.draw_pose_layer(img.copy(), [person, car])
    assert out2[CAP_H:].sum() > out[CAP_H:].sum()    # skeleton added ink
    # ...and the car region still has no annotation.
    assert out2[200:240, 400:500].sum() == 0


def test_gestures_layer_honest_when_empty():
    img = np.zeros((*SHAPE, 3), dtype=np.uint8)
    person = _box(100, 100)
    person["kps"] = _kps_for(person)
    person["track_id"] = 1
    stats = {1: {"id": 1, "gestures": []}}
    out = la.draw_gestures_layer(img.copy(), [person], stats, {})
    assert out[CAP_H:].sum() == 0                    # nothing to show, says so
    stats[1]["gestures"] = ["hand_up"]
    out2 = la.draw_gestures_layer(img.copy(), [person], stats, {"hand_up": 1})
    assert out2[CAP_H:].sum() > 0                    # skeleton + chip


def test_body_layer_flags_only_anomalies():
    # 2026-08-24 contract: EVERY person gets their stable per-track
    # COLOR box (pairs with the side card by color); alert-grade flags
    # still overlay a thicker red box + banner on top. The walker's box
    # must therefore match track_color(1) exactly - not the alert red.
    from app.layers.draw import track_color
    img = np.zeros((*SHAPE, 3), dtype=np.uint8)
    walker = _box(100, 100); walker["track_id"] = 1
    faller = _box(300, 100); faller["track_id"] = 2
    stats = {1: {"id": 1, "label": "walking", "alert": False},
             2: {"id": 2, "label": "fall_suspect", "alert": True}}
    out = la.draw_body_layer(img.copy(), [walker, faller], stats)
    walker_pixels = out[100:170, 90:140]
    faller_pixels = out[100:170, 295:340]
    assert walker_pixels.sum() > 0                   # walker: colored box
    # Walker's ink is its track color, NOT the (0,0,220) alert red:
    # collect the walker box's non-black pixels and check the palette
    # color appears while pure alert red does not dominate.
    wcol = np.array(track_color(1), dtype=np.uint8)
    nz = walker_pixels[walker_pixels.sum(axis=2) > 0]
    assert (nz == wcol).all(axis=1).any()            # palette color present
    alert_red = (nz[:, 2] > 200) & (nz[:, 1] < 60) & (nz[:, 0] < 60)
    assert not alert_red.any()                       # no alert red on walker
    assert faller_pixels.sum() > walker_pixels.sum() # faller more heavily drawn
    assert faller_pixels[..., 2].max() > 180         # faller box in red
    assert out[10:40, 200:460, 2].max() > 150        # red ALERT banner
    # No alert-grade flag -> banner absent, walker still shown in color.
    calm = la.draw_body_layer(img.copy(), [walker],
                              {1: {"id": 1, "label": "walking",
                                   "alert": False}})
    assert calm[100:170, 90:140].sum() > 0           # walker box drawn
    assert calm[10:40, 300:460].sum() == 0           # no red banner


def test_line_layer_draws_line_and_counts():
    # 2026-08-18: IN/OUT pills moved to BOTTOM-LEFT (was top-center) so
    # the numbers don't fight the tile's own Stop / Draw-line control row
    # above the frame. Assert the pills appear in the lower band instead
    # of the top - line and counters still on the same frame.
    img = np.zeros((*SHAPE, 3), dtype=np.uint8)
    out = la.draw_line_layer(img.copy(), la.DEFAULT_LINE, {"in": 3, "out": 1})
    y = int(0.62 * SHAPE[0])
    assert out[y - 3:y + 4].sum() > 0                # the line itself
    assert out[-80:].sum() > 0                       # bottom pills present


def test_heat_layer_is_signal_overlay():
    # 2026-08-17 operator request: the layer paints a FULL-FRAME thermal
    # recolour so an operator sees IMMEDIATELY the layer is running
    # (previously a bare photo, indistinguishable from "not working").
    # Empty grid = INFERNO recolour of the base; a dwell cell overlays
    # TURBO hotspots on top. Both states must differ from the flat input.
    img = np.full((*SHAPE, 3), 40, dtype=np.uint8)
    grid = [[0.0] * la.GRID_W for _ in range(la.GRID_H)]
    out = la.draw_heat_layer(img.copy(), grid)
    assert out.shape == img.shape
    # No dwell -> whole frame recoloured (INFERNO thermal), not the flat
    # input any more.
    assert not (out[CAP_H:] == img[CAP_H:]).all()
    grid[la.GRID_H // 2][la.GRID_W // 2] = 50.0
    out2 = la.draw_heat_layer(img.copy(), grid)
    # Dwell zone -> body of the frame ALSO differs, and the two rendered
    # frames must differ from each other (TURBO hotspot vs. plain thermal).
    assert not (out2[CAP_H:] == img[CAP_H:]).all()
    assert not (out2[CAP_H:] == out[CAP_H:]).all()


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class _StubSession:
    def __init__(self, cam, model, layer):
        self.cam = cam
        self.cam_id = cam["id"]
        self.cam_name = cam.get("name", cam["id"])
        self.model = model
        self.layer = layer
        self.last_poll = 0.0
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.seq = 0
        self.note = "starting"
        self._alive = True

    def start(self):
        pass

    def is_alive(self):
        return self._alive


@pytest.fixture()
def stub_manager(monkeypatch):
    monkeypatch.setattr(la, "LiveSession", _StubSession)
    return la.LiveAnalysisManager()


def test_manager_rejects_unknown_layer(stub_manager):
    with pytest.raises(ValueError):
        stub_manager.start("th_green_mango", "xray", model=None)


def test_manager_switch_keeps_session(stub_manager):
    a = stub_manager.start("th_green_mango", "heat", model=None)
    assert a["switched"] is False and a["active"] == 1
    b = stub_manager.start("th_green_mango", "gestures", model=None)
    assert b["switched"] is True and b["active"] == 1
    fr = stub_manager.frame("th_green_mango")
    assert fr["layer"] == "gestures" and fr["jpeg"] is None


def test_manager_caps_sessions_and_reaps(stub_manager):
    # Single-camera design (MAX_SESSIONS = 1): starting one is fine,
    # starting a second on a DIFFERENT camera raises BusyError. When the
    # first dies its slot frees on the next start, and the reaped
    # session's crash reason surfaces ONCE on the follow-up frame poll.
    stub_manager.start("th_green_mango", "paths", model=None)
    with pytest.raises(la.BusyError):
        stub_manager.start("th_nanai_road", "paths", model=None)
    stub_manager._sessions["th_green_mango"]._alive = False
    ok = stub_manager.start("th_nanai_road", "paths", model=None)
    assert ok["active"] == la.MAX_SESSIONS
    ended = stub_manager.frame("th_green_mango")
    assert ended == {"error": "session ended unexpectedly"}
    assert stub_manager.frame("th_green_mango") is None


def test_manager_surfaces_run_error_to_next_poll(stub_manager):
    # A session that died with a specific reason should hand that reason
    # to the very next frame poll instead of being silently reaped as 404.
    stub_manager.start("th_green_mango", "paths", model=None)
    s = stub_manager._sessions["th_green_mango"]
    s.err = "RuntimeError: pose model failed to load"
    s._alive = False
    fr = stub_manager.frame("th_green_mango")
    assert fr == {"error": "RuntimeError: pose model failed to load"}
    assert stub_manager.frame("th_green_mango") is None


def test_manager_stop_clears_pending_error(stub_manager):
    # An operator-initiated stop is not a crash; any pending error from a
    # previous incarnation must not resurface on the next start.
    stub_manager.start("th_green_mango", "paths", model=None)
    s = stub_manager._sessions["th_green_mango"]
    s.err = "boom"; s._alive = False
    stub_manager.frame("th_green_mango")               # drains once
    stub_manager.start("th_green_mango", "paths", model=None)   # fresh restart
    fr = stub_manager.frame("th_green_mango")
    assert fr.get("error") is None
    assert fr["layer"] == "paths"


def test_manager_stop(stub_manager):
    stub_manager.start("th_green_mango", "line", model=None)
    assert stub_manager.stop("th_green_mango") is True
    assert stub_manager.stop("th_green_mango") is False
    assert stub_manager.frame("th_green_mango") is None
