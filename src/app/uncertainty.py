"""Per-box uncertainty at capture time (plan WS1, decision D1).

Every crop/frame box the collector stores carries ``uncertainty`` in [0, 1]:
how much a human verdict on this detection would teach the system. Two
components, both nearly free on the e2-micro:

  * margin  - distance of the box conf from its EFFECTIVE class gate (the
    boosted/night-adjusted one actually used for the burst, not the shipped
    default): 1.0 at the gate, falling linearly to 0 at gate +- span. A box
    at its threshold is exactly the box the model was on the fence about.
  * flip delta (optional, sampled bursts only, UNCERTAINTY_FLIP=1) - ONE
    extra pass on the horizontally-flipped frame; per-box IoU-matched conf
    delta. A detection whose confidence moves (or vanishes) under a mirror
    flip is unstable evidence. Costs a single extra inference on the ~1-in-N
    bursts that get sampled, not T=10 on all (the spec's MC-Dropout was
    rejected in D1: yolov8 detection models carry zero Dropout modules).

Aggregate: ``0.6 * margin + 0.4 * flip`` (flip term 0 when disabled), same
downstream contract the spec asked for. Persistence is the callers' job:
review_frames writes the field into the sidecar json, live_samples encodes
it as a ``_uNN`` filename suffix.
"""
from __future__ import annotations

MARGIN_SPAN = 0.25
MARGIN_WEIGHT = 0.6
FLIP_WEIGHT = 0.4
# Below-gate detections a flip pass may consider when re-matching boxes.
FLIP_PREDICT_CONF = 0.05


def margin_score(conf: float, gate: float, span: float = MARGIN_SPAN) -> float:
    """1.0 when conf sits exactly on the gate, linear to 0.0 at gate +- span."""
    try:
        d = abs(float(conf) - float(gate))
    except (TypeError, ValueError):
        return 0.0
    if span <= 0:
        return 0.0
    return max(0.0, 1.0 - d / span)


def box_iou(a: dict, b: dict) -> float:
    """IoU of two {x1,y1,x2,y2} dicts (pixel space)."""
    ax1, ay1, ax2, ay2 = (float(a[k]) for k in ("x1", "y1", "x2", "y2"))
    bx1, by1, bx2, by2 = (float(b[k]) for k in ("x1", "y1", "x2", "y2"))
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = ((ax2 - ax1) * (ay2 - ay1)) + ((bx2 - bx1) * (by2 - by1)) - inter
    return inter / union if union > 0 else 0.0


def _predict_boxes(model, frame, imgsz: int | None) -> list[dict]:
    """One raw low-conf pass -> [{cls, conf, x1..y2}]. Seam for tests."""
    kwargs = {"conf": FLIP_PREDICT_CONF, "verbose": False}
    if imgsz:
        kwargs["imgsz"] = imgsz
    # Shares the live model with sessions/producers - must hold the
    # process-wide predict lock (OpenVINO InferRequest is not reentrant).
    from app.detect_core import _PREDICT_LOCK
    with _PREDICT_LOCK:
        res = model.predict(frame, **kwargs)[0]
    names = getattr(model, "names", {}) or {}
    out: list[dict] = []
    for bx in res.boxes:
        x1, y1, x2, y2 = (float(v) for v in bx.xyxy[0].tolist())
        out.append({"cls": str(names.get(int(bx.cls[0]), int(bx.cls[0]))),
                    "conf": float(bx.conf[0]),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return out


def flip_delta(model, frame, boxes: list[dict],
               imgsz: int | None = None,
               min_iou: float = 0.30) -> dict[int, float]:
    """{box_index: normalized conf delta} from ONE horizontally-flipped pass.

    Each original box is mirrored in x and greedily matched (same class,
    best IoU >= min_iou) against the flipped frame's detections. Matched ->
    |conf - conf_flipped| (already in [0,1]); unmatched -> 1.0, the
    detection did not survive a mirror flip at all.
    """
    if not boxes:
        return {}
    W = frame.shape[1]
    flipped = frame[:, ::-1]
    try:
        cand = _predict_boxes(model, flipped, imgsz)
    except Exception:
        return {}
    taken: set[int] = set()
    out: dict[int, float] = {}
    for i, b in enumerate(boxes):
        mirrored = {"x1": W - float(b["x2"]), "x2": W - float(b["x1"]),
                    "y1": float(b["y1"]), "y2": float(b["y2"])}
        best_j, best_iou = None, min_iou
        for j, c in enumerate(cand):
            if j in taken or c["cls"] != b.get("cls"):
                continue
            iou = box_iou(mirrored, c)
            if iou >= best_iou:
                best_j, best_iou = j, iou
        if best_j is None:
            out[i] = 1.0
        else:
            taken.add(best_j)
            out[i] = min(1.0, abs(float(b.get("conf") or 0.0)
                                  - cand[best_j]["conf"]))
    return out


def attach_uncertainty(boxes: list[dict], gates: dict,
                       flip: dict[int, float] | None = None,
                       default_gate: float = 0.35) -> None:
    """Write ``uncertainty`` into every box dict IN PLACE.

    ``gates`` is the effective per-class conf map the burst actually ran
    with (boosted + night-adjusted) - scoring against the shipped defaults
    would mis-rank cameras whose gates the review loop has already moved.
    """
    for i, b in enumerate(boxes):
        gate = gates.get(b.get("cls"), default_gate)
        m = margin_score(b.get("conf") or 0.0, gate)
        f = (flip or {}).get(i, 0.0)
        b["uncertainty"] = round(MARGIN_WEIGHT * m + FLIP_WEIGHT * f, 4)
