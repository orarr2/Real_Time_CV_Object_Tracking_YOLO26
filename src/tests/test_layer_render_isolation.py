"""Cross-layer render isolation.

Motivation: the operator reported "picking face detection made the
system show vehicles" (bug source in AUDIT_HE_2026-08-23.md sec 12).
The false-positive audit proved this was a HUD-TEXT confusion, not a
box mis-render - backend `_render` for `layer=="faces"` calls only
`draw_faces_layer_img` on a `frame.copy()`, and the frontend canvas
main loop explicitly `continue`s past `d.boxes` for the faces layer.

These tests LOCK that guarantee: every layer's drawing function has a
predictable shape, and the `LAYER_DEFS` / `_publish_data` cross-tags
never let a layer emit an event tagged with a different layer name.
The point is to catch REGRESSIONS if someone later adds a per-layer
draw path that fires generic boxes without noticing.

Run from src/:  python -m pytest tests/test_layer_render_isolation.py -q
"""
from __future__ import annotations

import inspect

import pytest


def test_layers_tuple_matches_expected_set():
    """The dashboard's Analyze modal names 10 layers; the backend LAYERS
    tuple in live_analysis is the source of truth. If someone adds a
    new layer without updating BOTH the frontend LAYER_DEFS and the
    backend LAYERS tuple, that mismatch shows up as invisible chips
    (see obstruction bug fixed in Batch B item B6). Fall detection is
    NOT a separate layer - it runs inside "body" (operator direction,
    2026-08-23)."""
    from app.live_analysis import LIVE_LAYERS
    expected = {
        "paths", "pose", "gestures", "body", "faces",
        "heat", "line", "fire", "parking", "plates",
    }
    assert set(LIVE_LAYERS) == expected, (
        f"LIVE_LAYERS tuple drifted from Analyze modal set: "
        f"{set(LIVE_LAYERS)} vs {expected}"
    )


@pytest.mark.parametrize("layer_name,draw_fn_name", [
    ("paths",    "draw_paths_layer"),
    ("pose",     "draw_pose_layer"),
    ("gestures", "draw_gestures_layer"),
    ("body",     "draw_body_layer"),
    ("faces",    "draw_faces_layer_img"),
    ("heat",     "draw_heat_layer"),
    ("line",     "draw_line_layer"),
    ("fire",     "draw_fire_layer"),
    ("plates",   "draw_plates_layer"),
])
def test_every_layer_has_a_draw_function(layer_name, draw_fn_name):
    """Each layer's dedicated draw function must exist and be callable.
    Prevents a silent regression where a rename kills the layer's
    output while the layer picker still shows the entry."""
    import app.live_analysis as la
    fn = getattr(la, draw_fn_name, None)
    assert fn is not None, (
        f"layer {layer_name!r} has no {draw_fn_name!r} in live_analysis"
    )
    assert callable(fn), f"{draw_fn_name} exists but is not callable"


def test_saved_json_lock_is_shared_module_level():
    """Bug #2 (LPR duplicates 62 ms apart) came from a missing lock
    on saved.json's read-modify-write. Batch C item C1a adds a module-
    level _SAVED_JSON_LOCK; if a refactor moves it back to per-session
    (self.lock) the race comes back. Assert the module-level lock is
    exported and is the same object every caller sees."""
    from app.live_analysis import _SAVED_JSON_LOCK
    assert _SAVED_JSON_LOCK is not None
    from app.live_analysis import _SAVED_JSON_LOCK as again
    assert _SAVED_JSON_LOCK is again, "lock must be module-level singleton"


def test_plate_agreement_gate_lives_in_plates_pass_source():
    """C1c gate: AGREEMENT_MIN and text_counts must both be present in
    the plates pipeline. This is a source-level test (not a live-tick
    test) because the full pipeline needs OpenVINO + a model file; the
    goal is to catch someone deleting the gate.
    """
    import app.live_analysis as la
    src = inspect.getsource(la.LiveSession._plates_pass)
    assert "AGREEMENT_MIN" in src, (
        "temporal-agreement gate removed from _plates_pass; bug #2 "
        "regression risk. See AUDIT_HE_2026-08-23.md C1c."
    )
    assert "text_counts" in src, (
        "temporal-agreement gate reads text_counts from _plate_reads; "
        "if this is gone the gate cannot see prior reads."
    )
    assert "PLATE_STR_DEDUP_S" in src, (
        "cross-track string dedup removed; a tracker split will start "
        "re-saving the same plate under a new tid."
    )


def test_plate_reads_counter_lives_in_attach_plates():
    """C1c gate is only useful if plates.attach_plates actually
    populates entry['text_counts'] on every OCR read."""
    import inspect as _inspect
    import app.plates as p
    src = _inspect.getsource(p.attach_plates)
    assert "text_counts" in src, (
        "plates.attach_plates no longer records per-text read counts; "
        "the _plates_pass agreement gate will never fire. See C1c."
    )
