"""Head-adapter loop: pointer/history, gate, tensor overlay, Storage sync."""
import json

import pytest
from conftest import FakeBucket as _FakeBucket

from app import adapters


# ---- pointer / history / rollback (no torch) ---------------------------------

def test_promote_rollback_chain(tmp_path):
    (tmp_path / "head_a.pt").write_bytes(b"a")
    (tmp_path / "head_b.pt").write_bytes(b"b")
    e1 = adapters.promote("head_a.pt", {"map50": 0.5}, tmp_path)
    assert e1["previous"] is None
    e2 = adapters.promote("head_b.pt", {"map50": 0.6}, tmp_path)
    assert e2["previous"] == "head_a.pt"
    assert adapters.read_pointer(tmp_path)["file"] == "head_b.pt"
    # rollback -> head_a
    back = adapters.rollback(tmp_path)
    assert back["file"] == "head_a.pt"
    # rollback again -> nothing older exists -> base model (pointer gone)
    assert adapters.rollback(tmp_path) is None
    assert adapters.read_pointer(tmp_path) is None
    events = [h["event"] for h in adapters.read_history(tmp_path)]
    assert events == ["promoted", "promoted", "rollback", "rollback"]


def test_gate_decision_rules():
    base = {"map50": 0.50, "per_class": {"person": 0.60, "car": 0.55,
                                         "bus": 0.40}}
    # +1pp, no drops -> promoted
    ok, reasons = adapters.gate_decision(
        base, {"map50": 0.51, "per_class": {"person": 0.60, "car": 0.55,
                                            "bus": 0.40}})
    assert ok and "no class regressions" in reasons[0]
    # +0.2pp gain -> under the +0.5pp bar
    ok, reasons = adapters.gate_decision(base, {"map50": 0.502,
                                                "per_class": {}})
    assert not ok and "gain" in reasons[0]
    # person may not drop AT ALL, even with a big global gain
    ok, reasons = adapters.gate_decision(
        base, {"map50": 0.58, "per_class": {"person": 0.59}})
    assert not ok and any("person" in r for r in reasons)
    # bus may drop up to 2pp...
    ok, _ = adapters.gate_decision(
        base, {"map50": 0.52, "per_class": {"bus": 0.385}})
    assert ok
    # ...but not 3pp
    ok, reasons = adapters.gate_decision(
        base, {"map50": 0.52, "per_class": {"bus": 0.37}})
    assert not ok and any("bus" in r for r in reasons)


# ---- head tensors (torch required) --------------------------------------------

def _tiny_yolo():
    """Shape-compatible stand-in for YOLO(...): .model is the DetectionModel
    whose .model is the module list; the last module plays the Detect head."""
    import torch

    class _TinyDet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Conv2d(1, 2, 1),
                torch.nn.Conv2d(2, 2, 1),
            )

    class _FakeYOLO:
        def __init__(self):
            self.model = _TinyDet()

    return _FakeYOLO()


def test_head_save_load_overlay_equivalence(tmp_path):
    torch = pytest.importorskip("torch")
    a, b = _tiny_yolo(), _tiny_yolo()
    head = adapters.extract_head(a.model)
    assert set(head) == {"model.1.weight", "model.1.bias"}
    p = tmp_path / "head.pt"
    adapters.save_head(head, p)
    loaded = adapters.load_head(p)              # weights_only=True path
    n = adapters.overlay_head(b.model, loaded)
    assert n == 2
    sa, sb = a.model.state_dict(), b.model.state_dict()
    for k in head:                              # head now byte-identical
        assert torch.equal(sa[k], sb[k])
    assert not torch.equal(sa["model.0.weight"], sb["model.0.weight"])


def test_overlay_refuses_shape_mismatch():
    torch = pytest.importorskip("torch")
    y = _tiny_yolo()
    bad = {"model.1.weight": torch.zeros(3, 3, 1, 1)}
    with pytest.raises(ValueError, match="shape mismatch"):
        adapters.overlay_head(y.model, bad)
    with pytest.raises(ValueError, match="not in model"):
        adapters.overlay_head(y.model, {"model.9.weight": torch.zeros(1)})


def test_apply_current_paths(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    y = _tiny_yolo()
    # no pointer -> base model untouched
    assert adapters.apply_current(y, tmp_path) == 0
    # promoted head -> overlaid
    head = adapters.extract_head(_tiny_yolo().model)
    adapters.save_head(head, tmp_path / "head_x.pt")
    adapters.promote("head_x.pt", {"map50": 0.9}, tmp_path)
    assert adapters.apply_current(y, tmp_path) == 2
    # kill-switch env
    monkeypatch.setenv("ADAPTERS_DISABLE", "1")
    assert adapters.apply_current(y, tmp_path) == 0


# ---- Storage sync (fake bucket, no torch) --------------------------------------

def test_refresh_from_storage_downloads_once(tmp_path):
    bucket = _FakeBucket()
    ptr = {"file": "head_a.pt", "promoted_at": "2026-07-11T00:00:00Z",
           "metrics": {"map50": 0.6}, "previous": None}
    bucket.store["training/adapter_current.json"] = json.dumps(ptr).encode()
    bucket.store["training/adapters/head_a.pt"] = b"tensorbytes"
    # first poll: fetches file + pointer
    assert adapters.refresh_from_storage(bucket, tmp_path) == "head_a.pt"
    assert (tmp_path / "head_a.pt").read_bytes() == b"tensorbytes"
    assert adapters.read_pointer(tmp_path)["file"] == "head_a.pt"
    # second poll: nothing new
    assert adapters.refresh_from_storage(bucket, tmp_path) is None
    # trainer promotes head_b -> picked up
    ptr2 = dict(ptr, file="head_b.pt", previous="head_a.pt")
    bucket.store["training/adapter_current.json"] = json.dumps(ptr2).encode()
    bucket.store["training/adapters/head_b.pt"] = b"tensorbytes2"
    assert adapters.refresh_from_storage(bucket, tmp_path) == "head_b.pt"
    # no bucket / missing pointer are quiet no-ops
    assert adapters.refresh_from_storage(None, tmp_path) is None
    assert adapters.refresh_from_storage(_FakeBucket(), tmp_path) is None


def test_refresh_rejects_path_traversal(tmp_path):
    bucket = _FakeBucket()
    bucket.store["training/adapter_current.json"] = json.dumps(
        {"file": "../../evil.pt"}).encode()
    assert adapters.refresh_from_storage(bucket, tmp_path) is None
    assert not (tmp_path.parent.parent / "evil.pt").exists()


def test_publish_full_vs_history_only(tmp_path):
    bucket = _FakeBucket()
    (tmp_path / "head_a.pt").write_bytes(b"a")
    adapters.promote("head_a.pt", {"map50": 0.7}, tmp_path)
    n = adapters.publish_to_storage(bucket, tmp_path)
    assert n == 3      # head + pointer + history
    assert "training/adapters/head_a.pt" in bucket.store
    assert "training/adapter_current.json" in bucket.store
    assert "training/history.jsonl" in bucket.store
    # a rejected run publishes ONLY the history
    b2 = _FakeBucket()
    adapters.append_history({"event": "gate", "promoted": False}, tmp_path)
    n2 = adapters.publish_to_storage(b2, tmp_path, history_only=True)
    assert n2 == 1
    assert list(b2.store) == ["training/history.jsonl"]
    assert b"'promoted': false" in b2.store["training/history.jsonl"].lower() \
        or b'"promoted": false' in b2.store["training/history.jsonl"]


def test_apply_current_refuses_foreign_base(tmp_path, capsys):
    """A head promoted for one base model must not overlay onto another -
    the guard fires BEFORE any torch load, so a dummy model object works."""
    (tmp_path / "head_n.pt").write_bytes(b"n")
    adapters.promote("head_n.pt", {"map50": 0.5}, tmp_path, base="yolov8n.pt")
    assert adapters.read_pointer(tmp_path)["base"] == "yolov8n.pt"
    n = adapters.apply_current(object(), tmp_path, expected_base="yolov8s.pt")
    assert n == 0
    out = capsys.readouterr().out
    assert "trained for 'yolov8n.pt'" in out and "yolov8s.pt" in out
    # legacy pointer without a base field -> guard stays out of the way
    adapters.write_pointer({"file": "head_n.pt"}, tmp_path)
    # (overlay itself then fails on the dummy object and is caught -> 0)
    assert adapters.apply_current(object(), tmp_path,
                                  expected_base="yolov8s.pt") == 0
