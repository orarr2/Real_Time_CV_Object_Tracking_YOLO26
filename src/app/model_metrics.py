"""Model information backend for the dashboard's Model information tab.

gather_models_info() returns the model-cards inventory (one row per
weight the pipeline consumes), each model's published reference metrics,
and a one-time latency benchmark measured on this machine. The review-
verdict scoreboard that used to live here (compute / learning_curve /
header_line) was removed in the 2026-08-23 cleanup together with the
review UI it served.
"""
from __future__ import annotations


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
    "yolov8s-pose.pt": {
        "dataset": "COCO val2017 keypoints",
        "source":  "ultralytics.com/models/yolov8-pose",
        "mAP50":   0.821,
        "mAP5095": 0.599,
        "params_m": 11.6,
    },
    "yolov11s-plate.pt": {
        # Specialist finetune - the HF card publishes no val-split mAP,
        # so no number is shown rather than a fabricated one.
        "dataset": "morsetechlab/yolov11-license-plate-detection (S)",
        "source":  "huggingface.co/morsetechlab/yolov11-license-plate-detection",
        "mAP50":   None,
        "mAP5095": None,
        "params_m": 9.4,
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
        # OCR head (cct_s_v2_global since 2026-08-23), not a detector -
        # its author publishes region-F1, not char accuracy, so no
        # number is shown rather than a fabricated one.
        "dataset": "fast-plate-ocr global v2 (cct-s)",
        "source":  "github.com/ankandrew/fast-plate-ocr",
        "char_acc": None,
        "params_m": 1.8,
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
    "yolov8s-pose.pt":       ("pose", "keypoints for Pose / Gestures / Body layers"),
    "yolov11s-plate.pt":     ("plates", "license-plate box detector inside a vehicle crop"),
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


