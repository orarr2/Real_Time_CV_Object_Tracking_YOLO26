"""Per-camera per-class confidence tuning from user review verdicts.

Closes the positive-and-negative feedback loop the auto-blacklist alone
was missing: ``auto_blacklist`` builds polygons for "wrong" verdicts in
specific areas, but a "correct" verdict is not just its opposite - it
tells the system "for this class on this camera, the threshold is too
strict and you're missing real ones", and a "wrong" verdict spread over
the whole frame tells it "for this class on this camera, the threshold
is too loose and you're firing on random stuff".

Every review submit calls ``apply_review()``:

* verdict == "correct"     -> ``delta -= STEP``  (lower the confidence
  bar so the same-looking box passes next time). Floor: ``MIN_CONF``.
* verdict == "wrong_label" / "not_an_object" -> ``delta += STEP``
  (raise the bar). Ceiling: ``MAX_CONF``.

The delta is persisted to ``data/confidence_boost.json``. The collector's
``cameras.py`` merges it into each camera's ``per_class_conf`` on import,
and ``collector.main`` re-imports every few rounds so live edits from the
review UI take effect without a service restart.

Store shape:

    {
      "updated_at": "...",
      "cams": {
        "<cam_id>": {
          "<cls>": {
            "delta":    +0.06,
            "approved": 2,
            "rejected": 5,
            "updated_at": "..."
          },
          ...
        },
        ...
      }
    }

The ``delta`` is what actually gets added to the class's default
``per_class_conf`` value on merge. Approved/rejected counts are book-
keeping so someone auditing the file understands why a class ended up
where it did.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORE_PATH = _SRC_ROOT / "data" / "confidence_boost.json"

# Balanced learning rate for the new full-frame review UX, where each
# frame yields 5-10 verdicts (not one). At delta=0.015 a typical five-box
# frame moves the effective conf by 0.075 - still enough to nudge the model
# after every focused review pass, but no longer flooded by a burst of
# verdicts on a single scene.
STEP = 0.015
MIN_CONF = 0.20
MAX_CONF = 0.60

_LOCK = threading.Lock()

_CAM_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_CLS_ALLOWED = {"person", "bicycle", "car", "motorcycle", "bus", "train", "truck"}


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {"updated_at": "", "cams": {}}


def _save(store: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    store["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(store, indent=2))
    tmp.replace(path)


def apply_review(cam_id: str, cls: str, verdict: str, *,
                 store_path: str | Path | None = None,
                 base_conf: float = 0.35,
                 step: float = STEP,
                 min_conf: float = MIN_CONF,
                 max_conf: float = MAX_CONF) -> dict | None:
    """Nudge (cam, cls) confidence based on one verdict.

    Returns the updated per-(cam, cls) record, or None when the caller
    passed an invalid cam_id / cls / verdict. Validation is intentional:
    the endpoint accepts JSON from the browser and unknown values should
    not silently create ghost entries in the store.
    """
    if not cam_id or not _CAM_ID_RE.match(cam_id):
        return None
    if cls not in _CLS_ALLOWED:
        return None
    if verdict == "correct":
        change = -step
    elif verdict in ("wrong_label", "not_an_object", "wrong"):
        change = +step
    else:
        return None

    path = Path(store_path if store_path is not None else DEFAULT_STORE_PATH)
    with _LOCK:
        store = _load(path)
        cams = store.setdefault("cams", {})
        cam_rec = cams.setdefault(cam_id, {})
        cls_rec = cam_rec.setdefault(cls, {"delta": 0.0, "approved": 0,
                                            "rejected": 0, "updated_at": ""})
        # Clamp the effective conf, not the delta itself, so the record
        # always says "how far from base we've drifted" honestly.
        new_delta = float(cls_rec.get("delta") or 0.0) + change
        effective = max(min_conf, min(max_conf, base_conf + new_delta))
        cls_rec["delta"] = round(effective - base_conf, 4)
        if change < 0:
            cls_rec["approved"] = int(cls_rec.get("approved") or 0) + 1
        else:
            cls_rec["rejected"] = int(cls_rec.get("rejected") or 0) + 1
        cls_rec["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _save(store, path)
        return dict(cls_rec)


def load_boosts(path: str | Path | None = None) -> dict[str, dict[str, float]]:
    """Return ``{cam_id: {cls: delta}}`` for cameras.py to merge into each
    camera's ``per_class_conf``. Missing entries mean "no adjustment".
    Reads at call time so hot reloads from the collector loop pick up
    edits between rounds.
    """
    p = Path(path if path is not None else DEFAULT_STORE_PATH)
    store = _load(p)
    out: dict[str, dict[str, float]] = {}
    for cam_id, cls_map in (store.get("cams") or {}).items():
        for cls, rec in (cls_map or {}).items():
            try:
                delta = float(rec.get("delta") or 0.0)
            except (TypeError, ValueError):
                continue
            if delta == 0.0:
                continue
            out.setdefault(cam_id, {})[cls] = delta
    return out


def summary(path: str | Path | None = None) -> dict:
    """Small aggregate view used by the review-stats endpoint."""
    p = Path(path if path is not None else DEFAULT_STORE_PATH)
    store = _load(p)
    cams = store.get("cams") or {}
    total_appr = total_rej = adjusted = 0
    for cam_map in cams.values():
        for rec in cam_map.values():
            total_appr += int(rec.get("approved") or 0)
            total_rej  += int(rec.get("rejected") or 0)
            if float(rec.get("delta") or 0.0) != 0.0:
                adjusted += 1
    return {
        "approved":     total_appr,
        "rejected":     total_rej,
        "adjusted_cls": adjusted,
        "updated_at":   store.get("updated_at") or "",
    }


def details(path: str | Path | None = None,
            baseline: dict[str, float] | None = None) -> dict:
    """Per-(cam, cls) baseline vs. current confidence with review counts.

    Powers the dashboard's "Learning proof" panel so the user can watch
    each verdict shift the threshold. Reports:

      * baseline - the shipped DEFAULT_PER_CLASS_CONF value for the class,
        so the delta has a reference point the user can read off directly;
      * current - baseline + delta, clamped to [MIN_CONF, MAX_CONF];
      * direction - "stricter" when delta > 0 (fewer false positives),
        "looser" when delta < 0 (fewer missed real objects), "unchanged"
        when the class has never been reviewed on this camera;
      * approved / rejected counts + last-updated timestamp per row.

    Rows are emitted only for (cam, cls) pairs the user has actually
    touched - not every camera x every class - so a fresh install with no
    reviews returns an empty ``rows`` list rather than 40 zero-delta rows.
    """
    p = Path(path if path is not None else DEFAULT_STORE_PATH)
    store = _load(p)
    cams = store.get("cams") or {}
    if baseline is None:
        try:
            from app.detect_core import DEFAULT_PER_CLASS_CONF
            baseline = dict(DEFAULT_PER_CLASS_CONF)
        except Exception:
            baseline = {}
    rows = []
    for cam_id, cam_map in sorted(cams.items()):
        for cls, rec in sorted(cam_map.items()):
            delta = float(rec.get("delta") or 0.0)
            base  = float(baseline.get(cls, 0.35))
            current = max(MIN_CONF, min(MAX_CONF, base + delta))
            if delta > 1e-6:
                direction = "stricter"
            elif delta < -1e-6:
                direction = "looser"
            else:
                direction = "unchanged"
            rows.append({
                "cam_id":     cam_id,
                "cls":        cls,
                "baseline":   round(base, 3),
                "delta":      round(delta, 3),
                "current":    round(current, 3),
                "direction":  direction,
                "approved":   int(rec.get("approved") or 0),
                "rejected":   int(rec.get("rejected") or 0),
                "updated_at": rec.get("updated_at") or "",
            })
    return {
        "rows":       rows,
        "updated_at": store.get("updated_at") or "",
        "step":       STEP,
        "min_conf":   MIN_CONF,
        "max_conf":   MAX_CONF,
    }
