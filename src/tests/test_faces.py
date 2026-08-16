"""Faces module: the no-op guarantees matter more than the detection.

A missing/unset FACE_MODEL must cost nothing - no exception, no changed
frame, no blocked sample. Detection itself needs the ONNX file and real
frames, so it is exercised manually, not here.

Run from src/:  python -m pytest tests -q
"""
import importlib

import app.faces as faces


def _fresh():
    """Reload to reset the cached detector/failure state between tests."""
    importlib.reload(faces)
    return faces


def test_unset_env_falls_back_to_bundled_model(monkeypatch):
    # The YuNet ONNX ships in data/ since 2026-08-08; with FACE_MODEL unset
    # the module must pick it up (before that, no machine had the file and
    # every face feature was silently dead). On a checkout without the
    # model the module still degrades to unavailable, never raises.
    m = _fresh()
    monkeypatch.delenv(m.FACE_MODEL_ENV, raising=False)
    assert m.available() is m.FACE_MODEL_DEFAULT.is_file()
    assert m.detect_faces(object()) == []      # junk input never raises


def test_missing_file_is_unavailable(monkeypatch, tmp_path):
    m = _fresh()
    monkeypatch.setenv(m.FACE_MODEL_ENV, str(tmp_path / "nope.onnx"))
    assert m.available() is False


def test_maybe_blur_disabled_returns_same_object():
    m = _fresh()
    frame = object()                      # never touched when disabled
    assert m.BLUR_ENABLED is False
    assert m.maybe_blur(frame) is frame


def test_maybe_blur_enabled_but_unavailable_is_noop(monkeypatch):
    m = _fresh()
    monkeypatch.delenv(m.FACE_MODEL_ENV, raising=False)
    m.BLUR_ENABLED = True
    frame = object()                      # would explode if cv2 touched it
    assert m.maybe_blur(frame) is frame


def test_maybe_blur_never_raises(monkeypatch):
    m = _fresh()
    m.BLUR_ENABLED = True
    monkeypatch.setattr(m, "blur_faces",
                        lambda f: (_ for _ in ()).throw(RuntimeError("boom")))
    frame = object()
    assert m.maybe_blur(frame) is frame
