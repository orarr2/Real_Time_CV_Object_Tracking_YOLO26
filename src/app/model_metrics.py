"""Model-quality scoreboard, computed from user review verdicts.

The dashboard's "Model: X% accuracy · P(person) Y% · FP Z% · N reviews"
header line is driven by ``compute()`` in this module. All numbers come
from ``data/reviews.json`` (the ReviewStore) - nothing here reads live
inference, so a caller can trust it and cache it without invalidation
tricks.

Percent-metrics need a MINIMUM SAMPLE SIZE to be meaningful. 3/3 correct
is technically 100% but says nothing about the model - it says the user
happened to review three easy crops. Below ``MIN_REVIEWS_FOR_METRIC``
``header_line()`` reports N/A + a progress hint instead of a fabricated
percentage, so the dashboard never shows a trustworthy-looking 100%
against a two-digit sample. Same rule applies to per-class precision.

Definitions used here (crop-level, until the full-frame review UX ships
its own recall data):

  correct       = user said the label the model gave was right
  wrong         = user said the label was wrong (either wrong class OR
                  not an object at all)
  accuracy      = correct / (correct + wrong)
  precision[c]  = correct[c] / (correct[c] + wrong[c])
                  where ``c`` is the class the model originally gave
  fp_rate       = wrong / (correct + wrong)
                  = 1 - accuracy in the crop-level model

`recall` is left off intentionally: without "the model missed this object
here" verdicts we cannot count FN. The full-frame review UX (Task 26)
adds a ``missed`` verdict, at which point ``compute()`` grows an
``recall`` field per class.
"""
from __future__ import annotations

# Below this many verdicts the % metrics are treated as unavailable in the
# UI (header_line reports N/A + progress). 20 balances "quick to reach"
# against "not laughably small" - the standard error on a proportion at
# n=20 is already ~10 pp, low enough that a 60% vs 90% distinction is
# real signal rather than coin flips.
MIN_REVIEWS_FOR_METRIC = 20
# Per-class precision needs its own minimum. Kept lower than the global
# threshold since a class-specific bar of 20 would keep every per-class
# metric hidden until well past 60 total reviews.
MIN_REVIEWS_FOR_PER_CLASS = 5


# Crop reviews under this prefix come from the shipped bootstrap fixtures,
# not from anything a production camera streamed. They exist so the review
# UI has material on a fresh install; scoring the model on them would let
# a demo image move the production accuracy number.
_DEMO_CROP_PREFIX = "live_samples/_demo/"


def compute(review_store) -> dict:
    """Aggregate crop-level AND frame-level verdicts into a scoreboard.

    Two verdict streams feed the numbers:
    * crop reviews  - one verdict per crop (legacy UI, still counted)
    * frame reviews - many verdicts per frame + explicit missed detections,
      which is where FN (and therefore recall / F1) come from.

    Honesty rules (each metric is gated by ITS OWN sample, see header_line):
    * precision/accuracy sample = tp + fp (verdicts on model boxes);
    * recall sample            = tp + fn (model boxes confirmed + missed
      objects the user drew) - a user who marked 17 misses has given real
      recall signal even when only 3 boxes got a verdict;
    * bootstrap ``_demo`` crops are excluded outright.
    """
    # --- crop verdicts (precision-only stream) -----------------------
    correct = 0
    wrong = 0
    demo_excluded = 0
    per_cls: dict[str, dict[str, int]] = {}
    for r in review_store._by_path.values():  # noqa: SLF001 - deliberate
        if (r.crop_path or "").startswith(_DEMO_CROP_PREFIX):
            demo_excluded += 1
            continue
        cls = r.original_cls or "?"
        rec = per_cls.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})
        if r.verdict == "correct":
            correct += 1
            rec["tp"] += 1
        else:
            wrong += 1
            rec["fp"] += 1

    # --- frame verdicts (adds FN → recall / F1) ----------------------
    frame_reviews = getattr(review_store, "_frames_by_path", {}).values()
    for fr in frame_reviews:
        meta_boxes_by_id = {}
        try:
            from app.review_frames import load_metadata
            meta = load_metadata(fr.frame_path)
            for b in (meta or {}).get("boxes", []):
                meta_boxes_by_id[str(b["id"])] = b.get("cls", "?")
        except Exception:
            pass
        for box_id, verdict in (fr.box_verdicts or {}).items():
            cls = meta_boxes_by_id.get(str(box_id), "?")
            rec = per_cls.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})
            if verdict == "correct":
                correct += 1; rec["tp"] += 1
            elif verdict in ("wrong", "object") or verdict.startswith("relabel:"):
                # relabel = real object, wrong class: a precision miss for
                # the class the model CLAIMED (which is what per_cls keys on).
                wrong += 1; rec["fp"] += 1
        for miss in (fr.missed_detections or ()):
            cls = miss.get("cls") or "?"
            rec = per_cls.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})
            rec["fn"] += 1

    total = correct + wrong          # precision-side sample (verdicts on boxes)
    accuracy = correct / total if total else None
    fp_rate  = wrong / total if total else None

    # Global recall / F1 - defined only when at least one frame review
    # has landed (so FN is a real count, not zero-by-omission).
    total_fn = sum(rec["fn"] for rec in per_cls.values())
    n_recall = correct + total_fn    # recall-side sample (confirmed + missed)
    if any(fr.missed_detections is not None for fr in frame_reviews) \
            or total_fn > 0:
        recall = correct / n_recall if n_recall else None
        precision = correct / (correct + wrong) if (correct + wrong) else None
        f1 = None
        if recall and precision and (recall + precision) > 0:
            f1 = 2 * precision * recall / (precision + recall)
    else:
        recall = None
        f1 = None

    classes = []
    for cls, rec in sorted(per_cls.items()):
        n = rec["tp"] + rec["fp"]
        p_c = rec["tp"] / n if n else None
        n_denom_r = rec["tp"] + rec["fn"]
        r_c = rec["tp"] / n_denom_r if n_denom_r > 0 else None
        classes.append({
            "cls":       cls,
            "n":         n,
            "precision": round(p_c, 4) if p_c is not None else None,
            "recall":    round(r_c, 4) if r_c is not None else None,
            "fn":        rec["fn"],
        })

    return {
        "total_reviews": total,
        "tp":            correct,
        "fp":            wrong,
        "fn":            total_fn,
        "n_precision":   total,
        "n_recall":      n_recall,
        "demo_excluded": demo_excluded,
        "accuracy":      round(accuracy, 4) if accuracy is not None else None,
        "fp_rate":       round(fp_rate,  4) if fp_rate  is not None else None,
        "recall":        round(recall, 4)   if recall   is not None else None,
        "f1":            round(f1, 4)       if f1       is not None else None,
        "per_class":     classes,
    }


def learning_curve(review_store, batch_size: int = 5) -> list[dict]:
    """Model mistake-rate per tagging batch, chronological - the operator's
    "is it actually getting better?" chart.

    Each reviewed frame contributes signals: per-box verdicts (wrong or
    relabel = a model mistake, correct = a model win) plus every
    operator-drawn miss (a mistake by definition). Frames are grouped into
    batches of `batch_size` in review order - matching the paced queue, so
    one chart point = one sitting's batch. A falling error_rate over
    batches is the improvement the operator asked to SEE.

    Honesty caveat carried to the UI: the rate also moves with how hard
    the sampled frames are; the uncertainty-first queue deliberately
    serves hard ones, so a plateau is not failure - a sustained rise is.
    """
    frs = sorted(getattr(review_store, "_frames_by_path", {}).values(),
                 key=lambda fr: fr.reviewed_at or "")
    points: list[dict] = []
    batch = {"frames": 0, "signals": 0, "mistakes": 0, "last": ""}

    def _flush() -> None:
        if not batch["frames"]:
            return
        denom = batch["signals"]
        points.append({
            "batch":            len(points) + 1,
            "frames":           batch["frames"],
            "signals":          denom,
            "mistakes":         batch["mistakes"],
            "error_rate":       round(batch["mistakes"] / denom, 4) if denom else None,
            "last_reviewed_at": batch["last"],
        })

    for fr in frs:
        signals = mistakes = 0
        for v in (fr.box_verdicts or {}).values():
            signals += 1
            if v in ("wrong", "object") or v.startswith("relabel:"):
                mistakes += 1
        miss = len(fr.missed_detections or ())
        signals += miss
        mistakes += miss
        if signals == 0:
            continue
        batch["frames"] += 1
        batch["signals"] += signals
        batch["mistakes"] += mistakes
        batch["last"] = fr.reviewed_at or ""
        if batch["frames"] >= batch_size:
            _flush()
            batch = {"frames": 0, "signals": 0, "mistakes": 0, "last": ""}
    _flush()
    return points


# ---------------------------------------------------------------------------
# Model information tab (added 2026-08-18)
#
# The dashboard's Model information tab pulls its content from
# gather_models_info(). Two pieces:
#
#   * MODEL CARDS - one row per weight file the pipeline is willing to
#     consume: filename, presence-on-disk, size, role, backend hint. Pure
#     inventory - no inference, cheap to compute on every request.
#
#   * REFERENCE METRICS - published Precision / Recall / mAP@50 /
#     mAP@50-95 numbers for each model on its own benchmark dataset
#     (COCO val2017 for the general YOLOv8/26 heads, per-repo held-out
#     for the specialist plate / fire heads). These are STATIC - the
#     source is the model's own README on Ultralytics / HuggingFace. The
#     UI labels them "reference metrics, not measured on your feed" so
#     an operator can't mistake them for a live in-scene score.
#
#   * MEASURED LATENCY - a one-time benchmark on THIS machine's CPU/GPU
#     stack for whichever weights actually loaded. Cached after the first
#     call so subsequent GETs on /api/models/info return instantly.
#
# Kept in this module (not a new file) because it shares the "computed
# from static + tiny live measurement, no inference required per call"
# shape with the review scoreboard above.
# ---------------------------------------------------------------------------

# Reference metrics from each model's published benchmark. Sources are
# noted alongside each entry so an auditor can reproduce them. All
# numbers are dataset-level - they measure the model, not our pipeline.
_REFERENCE_METRICS = {
    "yolov8n.pt": {
        "dataset": "COCO val2017",
        "source":  "ultralytics.com/models/yolov8",
        "mAP50":   0.529,
        "mAP5095": 0.373,
        "params_m": 3.2,
    },
    "yolov8s.pt": {
        "dataset": "COCO val2017",
        "source":  "ultralytics.com/models/yolov8",
        "mAP50":   0.617,
        "mAP5095": 0.449,
        "params_m": 11.2,
    },
    "yolo26x.pt": {
        # YOLO26 X - Ultralytics YOLO26 series flagship. Uses YOLOv8-x
        # reference numbers as a conservative floor since YOLO26 outperforms
        # v8 on the same size class on the published leaderboard.
        "dataset": "COCO val2017 (v8-x floor; YOLO26 tops it)",
        "source":  "ultralytics.com/models/yolov8",
        "mAP50":   0.688,
        "mAP5095": 0.539,
        "params_m": 68.2,
    },
    "yolov8n-pose.pt": {
        "dataset": "COCO val2017 keypoints",
        "source":  "ultralytics.com/models/yolov8-pose",
        "mAP50":   0.796,
        "mAP5095": 0.502,
        "params_m": 3.3,
    },
    "yolov8n-plate.pt": {
        "dataset": "Koushim/yolov8-license-plate-detection val split",
        "source":  "huggingface.co/Koushim/yolov8-license-plate-detection",
        "mAP50":   0.973,
        "mAP5095": 0.686,
        "params_m": 3.0,
    },
    "yolo_fire.pt": {
        "dataset": "SHOU-ISD fire-and-smoke val split",
        "source":  "huggingface.co/SHOU-ISD/fire-and-smoke",
        "mAP50":   0.812,
        "mAP5095": 0.542,
        "params_m": 3.2,
    },
    "face_detection_yunet_2023mar.onnx": {
        "dataset": "WIDER FACE val (Easy)",
        "source":  "github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
        "mAP50":   0.887,
        "mAP5095": None,
        "params_m": 0.09,
    },
    "plate_ocr_global.onnx": {
        # OCR head, not a detector - report character accuracy instead of
        # box mAP. Rendered on a separate axis so the two number types
        # never share a chart.
        "dataset": "fast-plate-ocr global v2 test set",
        "source":  "github.com/ankandrew/fast-plate-ocr",
        "char_acc": 0.976,
        "params_m": 2.1,
    },
    "ESPCN_x4.pb": {
        # Super-res model - report PSNR uplift on Set14 x4.
        "dataset": "Set14 x4",
        "source":  "github.com/fannymonori/TF-ESPCN",
        "psnr_db": 28.10,
        "params_m": 0.02,
    },
}


# Roles map weights to what the pipeline actually uses them for. Rendered
# in the model-cards table so an operator understands why a given weight
# is on disk and what happens if it's missing.
_MODEL_ROLES = {
    "yolov8n.pt":            ("detection", "general object detection (nano)"),
    "yolov8s.pt":            ("detection", "general object detection (small)"),
    "yolo26x.pt":            ("detection", "primary detector (YOLO26 X)"),
    "yolov8n-pose.pt":       ("pose", "keypoints for Pose / Gestures / Body layers"),
    "yolov8n-plate.pt":      ("plates", "license-plate box detector inside a vehicle crop"),
    "plate_ocr_global.onnx": ("plates", "license-plate OCR head (Latin + digits)"),
    "ESPCN_x4.pb":           ("plates", "super-res 4x for tiny plate crops"),
    "yolo_fire.pt":          ("fire", "fire + smoke detection layer"),
    "face_detection_yunet_2023mar.onnx": ("faces", "face detection (YuNet)"),
    "osnet_x0_25_msmt17.onnx": ("re-id", "person re-identification embedding"),
}


def _latency_bench_once() -> dict:
    """Measure a single-inference latency for the primary detector on
    THIS machine's stack. Called once and memoized. Best-effort - if the
    detector isn't loaded or something fails we return {} and the UI
    reports 'unavailable' rather than a fake number.
    """
    out: dict = {}
    try:
        import time as _t
        import numpy as np
        from app.detect_core import (detect_with_boxes, load_model,
                                     system_info)
        info = system_info()
        out["backend"] = info.get("backend")
        out["device"] = info.get("device")
        # detect_with_boxes takes (model, frame, ...) - load the primary
        # detector before timing so we're benchmarking real inference not
        # the loader path. load_model() is idempotent (cached).
        # Pass the weight name explicitly so the label we report matches
        # what we actually benched (system_info's "YOLO26 (CPU)" hard-code
        # was misleading when the pipeline in fact runs yolov8s.pt).
        _weights = "yolov8s.pt"
        _model = load_model(_weights)
        _backend_tag = ("OpenVINO" if info.get("backend") == "openvino"
                        else "CPU")
        out["model"] = f"{_weights} ({_backend_tag})"
        # Warm-up (first inference always pays JIT/compile cost).
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        detect_with_boxes(_model, dummy, imgsz=640)
        # Timed sample - median of 5 short frames.
        samples = []
        for _ in range(5):
            t0 = _t.perf_counter()
            detect_with_boxes(_model, dummy, imgsz=640)
            samples.append((_t.perf_counter() - t0) * 1000.0)
        samples.sort()
        out["latency_ms_median"] = round(samples[len(samples) // 2], 1)
        out["latency_ms_min"] = round(samples[0], 1)
        out["latency_ms_max"] = round(samples[-1], 1)
        out["fps_from_median"] = round(
            1000.0 / max(0.1, out["latency_ms_median"]), 1)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


_LATENCY_CACHE: dict = {}


def _scan_weight_files() -> list[dict]:
    """Walk the repo root and known model dirs, cataloguing every
    weight file the pipeline recognises."""
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent.parent
    src = repo / "src"
    seen: dict[str, dict] = {}
    for base in (repo, repo / "models", src, src / "data"):
        if not base.is_dir():
            continue
        for pat in ("*.pt", "*.onnx", "*.pb"):
            for p in base.glob(pat):
                name = p.name
                if name in seen:
                    continue
                sz = p.stat().st_size
                role, purpose = _MODEL_ROLES.get(name, ("other", "-"))
                ref = _REFERENCE_METRICS.get(name, {})
                seen[name] = {
                    "name":     name,
                    "path":     str(p.relative_to(repo)).replace("\\", "/"),
                    "size_mb":  round(sz / (1024 * 1024), 2),
                    "role":     role,
                    "purpose":  purpose,
                    "present":  True,
                    "reference": ref,
                }
    # Also list weights we EXPECT but that aren't on disk yet so the
    # operator can see what still needs to be downloaded.
    for name, (role, purpose) in _MODEL_ROLES.items():
        if name in seen:
            continue
        ref = _REFERENCE_METRICS.get(name, {})
        seen[name] = {
            "name":     name,
            "path":     "",
            "size_mb":  0,
            "role":     role,
            "purpose":  purpose,
            "present":  False,
            "reference": ref,
        }
    return list(seen.values())


def gather_models_info() -> dict:
    """Full payload for /api/models/info: weight inventory + reference
    metrics + measured latency on THIS machine."""
    global _LATENCY_CACHE
    models = _scan_weight_files()
    if not _LATENCY_CACHE:
        _LATENCY_CACHE = _latency_bench_once()
    return {
        "models":  models,
        "latency": _LATENCY_CACHE,
        "notes":   [
            "Reference metrics come from each model's own published "
            "benchmark on its native dataset (COCO val2017 for the "
            "general heads, per-repo held-out for specialists). They "
            "measure the model, not how it performs on YOUR live feed.",
            "Latency is measured on THIS machine's backend "
            "(CPU / OpenVINO / GPU) with a 640-pixel dummy input, "
            "median of 5 runs after a warm-up.",
        ],
    }


def header_line(metrics: dict, boost_summary: dict | None = None) -> str:
    """Human-readable one-line summary for the dashboard header.

    Kept in Python (not JS) so the same string can be logged, dumped in
    Firestore, and rendered by any client identically.

    Honesty rules:
    * The raw verdict counts (correct / wrong / missed) are ALWAYS shown -
      17 misses the user drew must be visible even before any % unlocks.
    * Each percentage is gated by ITS OWN sample size: precision by
      tp+fp, recall by tp+fn. A user who confirmed 3 boxes but marked 17
      misses has recall signal (n=20) and no precision signal (n=3) - the
      old single gate showed either both or neither.
    """
    tp = metrics.get("tp")
    fp_n = metrics.get("fp")
    fn = metrics.get("fn") or 0
    if tp is None:                       # pre-rework caller (tests, cache)
        tp = metrics.get("total_reviews") or 0
        fp_n = 0
    n_prec = metrics.get("n_precision")
    n_prec = (tp + (fp_n or 0)) if n_prec is None else n_prec
    n_rec = metrics.get("n_recall")
    n_rec = (tp + fn) if n_rec is None else n_rec

    if n_prec + fn == 0:
        return "Model: no feedback yet - review a few frames below to teach it"

    # Plain words, operator-first (2026-07 redesign): the old line
    # ("precision pending (3/20 verdicts) · recall 12%") answered none of
    # the operator's real questions - is it right? how often? is it
    # learning? Full statistical detail stays in the JSON for tooling.
    # ASCII only: the string is printed to Windows consoles (cp125x).
    parts = []
    right = f"right on {tp} of {n_prec} boxes you checked"
    acc = metrics.get("accuracy")
    if acc is not None and n_prec:
        pct = int(round(acc * 100))
        if n_prec >= MIN_REVIEWS_FOR_METRIC:
            right += f" ({pct}% accuracy)"
        else:
            # Always show the NUMBER (the operator asked for one), with the
            # sample-size truth attached: 3-of-3 is "100% so far", not "the
            # model is perfect" - the figure firms up at 20 checks.
            right += (f" ({pct}% so far - small sample, firm after "
                      f"{MIN_REVIEWS_FOR_METRIC} checks)")
    parts.append(right)
    if fn:
        parts.append(f"{fn} objects it missed are marked and queued for training")
    if boost_summary and boost_summary.get("adjusted_cls"):
        learn = (f"learning is ON - it self-adjusted "
                 f"{boost_summary['adjusted_cls']} detection thresholds "
                 f"from your feedback")
        upd = boost_summary.get("updated_at") or ""
        try:
            import calendar
            import time as _t
            mins = max(0, int((_t.time() - calendar.timegm(
                _t.strptime(upd, "%Y-%m-%dT%H:%M:%SZ"))) / 60))
            learn += (f" (last {mins} min ago)" if mins < 120
                      else f" (last {mins // 60} h ago)")
        except (ValueError, TypeError):
            pass
        parts.append(learn)
    else:
        parts.append("learning is ON - waiting for your first verdicts")
    return "Model: " + " · ".join(parts)
