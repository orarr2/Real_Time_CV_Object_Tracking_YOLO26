"""Per-camera-fleet Detect-head "adapters": save, promote, overlay, hot-load.

The active-learning loop (see the "Active-learning loop" chapter of
src/docs/PROJECT_GUIDE.md, or PROJECT_GUIDE_HE.md in Hebrew) fine-tunes
ONLY the Detect head of the production base model (yolov8n, the VM's
pinned weights) on operator-reviewed frames - the D2 replacement for LoRA:
the backbone stays frozen, so the artifact is just the head's tensors
(~4-6 MB) and loading is "load base, overlay head". No adapter file
present means the base model runs untouched - bit-identical behavior (D6).

Filesystem contract (data/adapters/):
  head_YYYYMMDD_HHMMSS.pt   trained head state dicts (full retention - D9)
  current.json              pointer: which head is live + its gate metrics
  history.jsonl             append-only log of every promote/reject/rollback

Storage contract (training/ prefix in the project bucket):
  training/adapter_current.json   mutable pointer (no-store, like manifests)
  training/adapters/<file>.pt     immutable head artifacts
The GitHub Actions trainer publishes there; the VM collector polls the
pointer every few rounds and hot-swaps the head IN PLACE (load_state_dict
on the existing module - no model reload, no restart, no RAM spike).

torch imports live inside functions: the dashboard and most tests import
this module for the pointer/gate logic only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent
ADAPTERS_DIR = _SRC_ROOT / "data" / "adapters"
POINTER_NAME = "current.json"
HISTORY_NAME = "history.jsonl"

STORAGE_PREFIX = "training"
STORAGE_POINTER = f"{STORAGE_PREFIX}/adapter_current.json"

# A head artifact is a few MB; anything bigger than this is not ours.
MAX_HEAD_BYTES = 64 * 1024 * 1024

# Promotion gate (plan WS3): candidate must beat baseline mAP50 by at least
# MIN_GAIN, no class may drop more than MAX_CLASS_DROP, and the two classes
# the operator cares most about may not drop AT ALL.
MIN_GAIN = 0.005            # +0.5 percentage points
MAX_CLASS_DROP = 0.02       # -2 pp tolerated on secondary classes
ZERO_DROP_CLASSES = ("person", "car")


# ---- pointer / history (pure json - no torch) -----------------------------

def read_pointer(adapters_dir: str | Path = ADAPTERS_DIR) -> dict | None:
    p = Path(adapters_dir) / POINTER_NAME
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) and d.get("file") else None
    except (OSError, ValueError):
        return None


def write_pointer(entry: dict, adapters_dir: str | Path = ADAPTERS_DIR) -> None:
    """Atomic pointer update; keeps one step of `previous` for rollback."""
    d = Path(adapters_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / POINTER_NAME
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entry, indent=1))
    tmp.replace(p)


def append_history(record: dict, adapters_dir: str | Path = ADAPTERS_DIR) -> None:
    d = Path(adapters_dir)
    d.mkdir(parents=True, exist_ok=True)
    record = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record}
    with (d / HISTORY_NAME).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_history(adapters_dir: str | Path = ADAPTERS_DIR) -> list[dict]:
    out: list[dict] = []
    try:
        for line in (Path(adapters_dir) / HISTORY_NAME).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    return out


def promote(head_file: str, metrics: dict,
            adapters_dir: str | Path = ADAPTERS_DIR,
            base: str | None = None) -> dict:
    """Point `current` at head_file (name only, must live in adapters_dir).

    `base` records which base weights the head was trained/gated against
    (e.g. "yolov8n.pt") so every loader can refuse a head that does not
    belong to the model it is running - a v8s head silently failing to
    overlay onto the VM's v8n was exactly the failure this prevents."""
    prev = read_pointer(adapters_dir)
    entry = {
        "file": Path(head_file).name,
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": metrics,
        "previous": (prev or {}).get("file"),
    }
    if base:
        entry["base"] = Path(base).name
    write_pointer(entry, adapters_dir)
    append_history({"event": "promoted", **entry}, adapters_dir)
    return entry


def rollback(adapters_dir: str | Path = ADAPTERS_DIR) -> dict | None:
    """Restore the previous pointer. Returns the new pointer (or None when
    there is nothing to roll back to - pointer removed, base model runs)."""
    cur = read_pointer(adapters_dir)
    if cur is None:
        return None
    prev_file = cur.get("previous")
    if prev_file and (Path(adapters_dir) / prev_file).is_file():
        entry = {"file": prev_file,
                 "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "metrics": {"note": "rollback"},
                 "previous": None}
        write_pointer(entry, adapters_dir)
        append_history({"event": "rollback", "from": cur.get("file"),
                        "to": prev_file}, adapters_dir)
        return entry
    # No previous artifact -> drop the pointer entirely: base model.
    try:
        (Path(adapters_dir) / POINTER_NAME).unlink()
    except OSError:
        pass
    append_history({"event": "rollback", "from": cur.get("file"),
                    "to": None}, adapters_dir)
    return None


# ---- promotion gate (pure math - no torch) ---------------------------------

def gate_decision(base: dict, cand: dict,
                  min_gain: float = MIN_GAIN,
                  max_class_drop: float = MAX_CLASS_DROP,
                  zero_drop_classes=ZERO_DROP_CLASSES) -> tuple[bool, list[str]]:
    """Decide promote/reject from two {"map50": x, "per_class": {cls: ap50}}.

    Returns (ok, reasons). Reasons always explain the verdict - they go to
    history.jsonl so a rejected run is auditable months later.
    """
    reasons: list[str] = []
    b_map, c_map = float(base.get("map50") or 0), float(cand.get("map50") or 0)
    gain = c_map - b_map
    if gain < min_gain:
        reasons.append(f"mAP50 gain {gain * 100:+.2f}pp < required "
                       f"{min_gain * 100:+.2f}pp (base {b_map:.4f} -> "
                       f"cand {c_map:.4f})")
    bc = base.get("per_class") or {}
    cc = cand.get("per_class") or {}
    for cls in sorted(set(bc) & set(cc)):
        drop = float(bc[cls]) - float(cc[cls])
        allowed = 0.0 if cls in zero_drop_classes else max_class_drop
        if drop > allowed + 1e-9:
            reasons.append(f"class '{cls}' dropped {drop * 100:.2f}pp "
                           f"(allowed {allowed * 100:.2f}pp)")
    if not reasons:
        reasons.append(f"mAP50 {b_map:.4f} -> {c_map:.4f} "
                       f"({gain * 100:+.2f}pp), no class regressions")
        return True, reasons
    return False, reasons


def al_curve_payload(adapters_dir: str | Path = ADAPTERS_DIR) -> dict:
    """Data for the dashboard's "labels vs quality" chart (plan WS5).

    Reads history.jsonl gate records - one point per training run, promoted
    or rejected (rejected renders grey). Baseline = the baseline mAP50 of
    the most recent gate run (each run re-validates the clean base on the
    CURRENT chronological val split, so the latest number is the honest
    comparator for the whole curve).
    """
    points = []
    baseline = None
    for rec in read_history(adapters_dir):
        if rec.get("event") != "gate":
            continue
        base_map = ((rec.get("baseline") or {}).get("map50"))
        cand_map = ((rec.get("metrics") or {}).get("map50"))
        if cand_map is None:
            continue
        if base_map is not None:
            baseline = base_map
        points.append({
            "at":           rec.get("at"),
            "adapter":      rec.get("candidate"),
            "labels_total": rec.get("labels_total"),
            "map50":        cand_map,
            "promoted":     bool(rec.get("promoted")),
        })
    ptr = read_pointer(adapters_dir)
    return {
        "baseline_map50": baseline,
        "points":         points,
        "current":        (ptr or {}).get("file"),
    }


# ---- head tensors: extract / save / load / overlay (torch inside) ----------

def detect_head_index(det_model) -> int:
    """Index of the Detect module inside DetectionModel.model. Falls back to
    the last module - which IS the Detect head in every yolov8 layout."""
    seq = getattr(det_model, "model", det_model)
    idx = len(seq) - 1
    for i, m in enumerate(seq):
        if type(m).__name__ == "Detect":
            idx = i
    return idx


def extract_head(det_model, head_idx: int | None = None) -> dict:
    """Head-only state dict (cloned, cpu, fp32) keyed as the DetectionModel
    sees it (``model.<idx>.*``), so overlay is a plain load_state_dict."""
    if head_idx is None:
        head_idx = detect_head_index(det_model)
    prefix = f"model.{head_idx}."
    out = {}
    for k, v in det_model.state_dict().items():
        if k.startswith(prefix):
            out[k] = v.detach().float().cpu().clone()
    if not out:
        raise ValueError(f"no tensors under '{prefix}' - not a detection model?")
    return out


def save_head(head_state: dict, path: str | Path) -> None:
    import torch
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(head_state, tmp)
    tmp.replace(p)


def load_head(path: str | Path) -> dict:
    """Plain-tensor load. weights_only=True: the artifact travels through
    cloud Storage, so it must never be able to execute pickled code."""
    import torch
    return torch.load(str(path), map_location="cpu", weights_only=True)


def overlay_head(det_model, head_state: dict) -> int:
    """Load head tensors into the model IN PLACE. Every key must exist with
    a matching shape (a shape mismatch means base-model version drift - we
    refuse rather than half-load). Returns the number of tensors applied."""
    current = det_model.state_dict()
    for k, v in head_state.items():
        if k not in current:
            raise ValueError(f"head tensor '{k}' not in model")
        if tuple(current[k].shape) != tuple(v.shape):
            raise ValueError(f"shape mismatch for '{k}': model "
                             f"{tuple(current[k].shape)} vs head "
                             f"{tuple(v.shape)}")
    det_model.load_state_dict(head_state, strict=False)
    return len(head_state)


def apply_current(yolo_model, adapters_dir: str | Path = ADAPTERS_DIR,
                  expected_base: str | None = None) -> int:
    """Overlay the promoted head onto a loaded YOLO model, if one exists.
    Returns tensor count (0 = no adapter / not applicable). Never raises -
    a broken artifact must not take the collector down; it logs and the
    base model keeps running (D6).

    `expected_base` is the weights name of the model being overlaid (the
    collector passes its --weights). When the pointer records which base it
    was trained against and the two disagree, the overlay is refused with a
    loud, actionable line instead of failing on a tensor-shape mismatch."""
    import os
    if os.environ.get("ADAPTERS_DISABLE") == "1":
        return 0
    ptr = read_pointer(adapters_dir)
    if not ptr:
        return 0
    ptr_base = ptr.get("base")
    if expected_base and ptr_base \
            and Path(str(expected_base)).name != str(ptr_base):
        print(f"adapters: promoted head was trained for '{ptr_base}' but "
              f"this process runs '{Path(str(expected_base)).name}' - "
              f"skipping overlay (retrain with --base "
              f"{Path(str(expected_base)).name})")
        return 0
    head_path = Path(adapters_dir) / ptr["file"]
    if not head_path.is_file():
        return 0
    try:
        det = getattr(yolo_model, "model", yolo_model)
        n = overlay_head(det, load_head(head_path))
        print(f"adapters: overlaid {n} head tensors from {ptr['file']} "
              f"(promoted {ptr.get('promoted_at', '?')})")
        return n
    except Exception as e:
        print(f"adapters: overlay failed ({type(e).__name__}: {e}) - "
              f"running base model")
        return 0


# ---- VM side: poll Storage for a newly promoted head -----------------------

def refresh_from_storage(bucket,
                         adapters_dir: str | Path = ADAPTERS_DIR) -> str | None:
    """Check the bucket's training/adapter_current.json; when it names a head
    we don't have as current, download it and update the local pointer.

    `bucket` is the firebase_admin storage bucket (the collector's
    ``firebase.storage``); None -> no-op. Returns the new head filename when
    something changed, else None. Never raises."""
    if bucket is None:
        return None
    try:
        raw = bucket.blob(STORAGE_POINTER).download_as_bytes()
        remote = json.loads(raw.decode("utf-8"))
        fname = remote.get("file")
        if not fname or "/" in fname or "\\" in fname or ".." in fname:
            return None
        local = read_pointer(adapters_dir)
        if local and local.get("file") == fname \
                and (Path(adapters_dir) / fname).is_file():
            return None
        blob = bucket.blob(f"{STORAGE_PREFIX}/adapters/{fname}")
        data = blob.download_as_bytes()
        if len(data) > MAX_HEAD_BYTES:
            print(f"adapters: remote head {fname} is {len(data)} bytes - "
                  f"over cap, ignoring")
            return None
        d = Path(adapters_dir)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / (fname + ".part")
        tmp.write_bytes(data)
        tmp.replace(d / fname)
        write_pointer(remote, d)
        append_history({"event": "fetched", "file": fname,
                        "size": len(data)}, d)
        return fname
    except Exception as e:
        # Missing pointer object = trainer never ran yet; normal, stay quiet.
        if "404" not in str(e) and "Not Found" not in str(e):
            print(f"adapters: storage refresh failed "
                  f"({type(e).__name__}: {e})")
        return None


def publish_to_storage(bucket, adapters_dir: str | Path = ADAPTERS_DIR,
                       history_only: bool = False) -> int:
    """Upload pointer + its head file + history to the bucket. Called by the
    trainer. ``history_only`` covers REJECTED runs: the gate verdict must
    still reach the cumulative cloud history (full retention - D9) even
    though the pointer and head stay untouched. Returns objects uploaded."""
    if bucket is None:
        return 0
    d = Path(adapters_dir)
    n = 0
    ptr = read_pointer(adapters_dir)
    if not history_only and ptr is not None:
        head = d / ptr["file"]
        if head.is_file():
            blob = bucket.blob(f"{STORAGE_PREFIX}/adapters/{ptr['file']}")
            blob.upload_from_string(head.read_bytes(),
                                    content_type="application/octet-stream")
            blob.make_public()
            n += 1
        pb = bucket.blob(STORAGE_POINTER)
        pb.cache_control = "no-store"   # mutable name - same rule as manifests
        pb.upload_from_string(json.dumps(ptr), content_type="application/json")
        pb.make_public()
        n += 1
    hist = d / HISTORY_NAME
    if hist.is_file():
        hb = bucket.blob(f"{STORAGE_PREFIX}/{HISTORY_NAME}")
        hb.cache_control = "no-store"
        hb.upload_from_string(hist.read_bytes(),
                              content_type="application/octet-stream")
        hb.make_public()
        n += 1
    return n
