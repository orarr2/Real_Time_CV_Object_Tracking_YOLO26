"""Presence heatmap: accumulation, dayparts, decay, persistence, render.

Run from src/:  python -m pytest tests -q
"""
import datetime as dt
import json

import pytest

from app import heatmap

SHAPE = (360, 640)          # H, W
TZ = dt.timezone.utc


@pytest.fixture(autouse=True)
def _fresh_state(tmp_path, monkeypatch):
    # Redirect the default data dir so no test can write into src/data.
    monkeypatch.setattr(heatmap, "DATA_DIR", tmp_path)
    heatmap.reset()
    yield
    heatmap.reset()


def _box(x, y, w=20, h=40, cls="person"):
    return {"x1": x, "y1": y, "x2": x + w, "y2": y + h, "cls": cls}


def _noon(day=1):
    return dt.datetime(2026, 7, day, 12, 0, tzinfo=TZ).timestamp()


def _foot_cell_of_test_box():
    """Grid cell of _box(100, 100)'s foot point (x=110, y=140 on 640x360),
    computed from the CURRENT grid constants so a resolution change moves
    the expectation with it."""
    gx = min(heatmap.GRID_W - 1, int(110 / 640 * heatmap.GRID_W))
    gy = min(heatmap.GRID_H - 1, int(140 / 360 * heatmap.GRID_H))
    return gx, gy


def test_accumulate_lands_in_the_foot_cell(tmp_path):
    heatmap.accumulate("camA", [_box(100, 100)], SHAPE,
                       now=_noon(), tz=TZ, root=tmp_path)
    grid = heatmap.grid_for("camA", layer="person", root=tmp_path)
    gx, gy = _foot_cell_of_test_box()
    assert grid[gy][gx] == pytest.approx(heatmap.WEIGHT_DEFAULT_S)
    assert sum(sum(r) for r in grid) == pytest.approx(heatmap.WEIGHT_DEFAULT_S)


def test_layer_routing():
    now = _noon()
    boxes = [_box(0, 0, cls="person"), _box(100, 100, cls="car"),
             _box(200, 200, cls="train"), _box(300, 100, cls="bird")]
    heatmap.accumulate("camA", boxes, SHAPE, now=now, tz=TZ)
    p = sum(sum(r) for r in heatmap.grid_for("camA", layer="person"))
    v = sum(sum(r) for r in heatmap.grid_for("camA", layer="vehicles"))
    o = sum(sum(r) for r in heatmap.grid_for("camA", layer="other"))
    w = heatmap.WEIGHT_DEFAULT_S
    assert (p, v, o) == (pytest.approx(w), pytest.approx(w),
                         pytest.approx(2 * w))   # train + bird


def test_weight_is_observed_interval_clamped():
    heatmap.accumulate("camA", [_box(0, 0)], SHAPE, now=_noon(), tz=TZ)
    # 60s later: weight 60. 1h later: clamped to WEIGHT_MAX_S.
    heatmap.accumulate("camA", [_box(0, 0)], SHAPE, now=_noon() + 60, tz=TZ)
    heatmap.accumulate("camA", [_box(0, 0)], SHAPE,
                       now=_noon() + 60 + 3600, tz=TZ)
    total = sum(sum(r) for r in heatmap.grid_for("camA", layer="person"))
    assert total == pytest.approx(
        heatmap.WEIGHT_DEFAULT_S + 60 + heatmap.WEIGHT_MAX_S)


def test_daypart_split():
    d = dt.datetime(2026, 7, 1, tzinfo=TZ)
    for hour, part in ((3, "night"), (8, "morning"),
                       (14, "afternoon"), (20, "evening")):
        heatmap.accumulate("camA", [_box(0, 0)], SHAPE,
                           now=d.replace(hour=hour).timestamp(), tz=TZ)
    for part in heatmap.DAYPARTS:
        g = heatmap.grid_for("camA", layer="person", daypart=part)
        assert sum(sum(r) for r in g) > 0, part


def test_daily_decay():
    heatmap.accumulate("camA", [_box(100, 100)], SHAPE,
                       now=_noon(1), tz=TZ)
    before = sum(sum(r) for r in heatmap.grid_for("camA", layer="person"))
    # Ten days later a new sample triggers the decay pass first.
    heatmap.accumulate("camA", [_box(500, 300)], SHAPE,
                       now=_noon(11), tz=TZ)
    grid = heatmap.grid_for("camA", layer="person")
    gx, gy = _foot_cell_of_test_box()
    decayed_cell = grid[gy][gx]
    assert decayed_cell == pytest.approx(
        before * heatmap.DAILY_DECAY ** 10, rel=1e-6)


def test_persistence_roundtrip(tmp_path):
    heatmap.accumulate("camA", [_box(100, 100)], SHAPE,
                       now=_noon(), tz=TZ, root=tmp_path)
    heatmap.save("camA", root=tmp_path)
    files = list(tmp_path.glob("heatmap_camA.json"))
    assert files, "expected heatmap_camA.json on disk"
    payload = json.loads(files[0].read_text())
    assert payload["samples"] == 1
    # Fresh process: state reloads from disk.
    heatmap.reset()
    grid = heatmap.grid_for("camA", layer="person", root=tmp_path)
    gx, gy = _foot_cell_of_test_box()
    assert grid[gy][gx] == pytest.approx(heatmap.WEIGHT_DEFAULT_S, abs=0.01)


def test_render_due_cadence():
    assert heatmap.render_due("nope") is False        # never accumulated
    heatmap.accumulate("camA", [_box(0, 0)], SHAPE, now=_noon(), tz=TZ)
    assert heatmap.render_due("camA") is True          # first sample
    for i in range(1, heatmap.RENDER_EVERY_SAMPLES - 1):
        heatmap.accumulate("camA", [_box(0, 0)], SHAPE,
                           now=_noon() + i, tz=TZ)
        assert heatmap.render_due("camA") is False
    heatmap.accumulate("camA", [_box(0, 0)], SHAPE,
                       now=_noon() + 100, tz=TZ)
    assert heatmap.render_due("camA") is True          # sample #30


def test_render_shapes_and_empty_map():
    import numpy as np
    # Empty map on a dark canvas: just the canvas back.
    img = heatmap.render("virgin_cam")
    assert img.shape == (360, 640, 3)
    # With signal + a base frame: overlay keeps the base's shape.
    heatmap.accumulate("camA", [_box(100, 100)], SHAPE, now=_noon(), tz=TZ)
    base = np.zeros((360, 640, 3), dtype=np.uint8)
    out = heatmap.render("camA", base_frame=base)
    assert out.shape == base.shape
    assert out.sum() > 0        # something was painted


def test_stats():
    heatmap.accumulate("camA", [_box(100, 100), _box(400, 200)], SHAPE,
                       now=_noon(), tz=TZ)
    st = heatmap.stats("camA")
    assert st["samples"] == 1
    assert st["total_weight_s"] == pytest.approx(2 * heatmap.WEIGHT_DEFAULT_S)
    assert 0 < st["coverage_frac"] < 0.02
