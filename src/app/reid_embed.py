"""Pluggable appearance embedders for the re-ID registry.

Two implementations behind one interface:

  * HistogramEmbedder - the original dependency-free HSV color histogram
    (514-d). Works for stationary objects and same-lighting matches; collapses
    across lighting changes (a relit object scores ~0 cosine vs itself).
  * OsnetEmbedder - a real person/vehicle re-ID CNN (OSNet) exported to ONNX,
    run with onnxruntime on CPU (~5-10 ms per crop for osnet_x0_25). Survives
    lighting/pose change - the piece the histogram fundamentally cannot do.
    Produce the .onnx once with torchreid on any machine with internet, copy
    it to the VM, run the collector with --reid-model /path/to/osnet.onnx.

Embedders carry an `embedder_id`; the registry stores it and RESETS itself
when it changes (embeddings from different embedders are not comparable -
different dimensions AND different metric scales).
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

PERSON_CROP  = (64, 128)   # w x h - histogram embedder
VEHICLE_CROP = (96, 96)
OSNET_INPUT  = (128, 256)  # w x h - standard torchreid input

# The conventional drop location tools/setup_reid.sh downloads to. When the
# file exists it is picked up automatically (see resolve_model_path), so one
# script run upgrades the collector AND the dashboard with zero config edits.
_SRC_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OSNET_PATH = _SRC_ROOT / "data" / "osnet_x0_25_msmt17.onnx"


def resolve_model_path(model_path: str | None = None) -> str | None:
    """Resolve which re-ID model to load: explicit argument first, then the
    REID_MODEL env var, then the conventional data/ path if the ONNX has
    been downloaded there. None means 'histogram fallback'."""
    if model_path:
        return model_path
    env = os.environ.get("REID_MODEL")
    if env:
        return env
    if DEFAULT_OSNET_PATH.is_file():
        return str(DEFAULT_OSNET_PATH)
    return None

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _l2norm(vec: np.ndarray) -> np.ndarray | None:
    n = np.linalg.norm(vec)
    return (vec / n).astype(np.float32) if n > 0 else None


class HistogramEmbedder:
    """Masked HSV color histogram + geometry: the original demo-grade signature."""

    embedder_id = "hsv_hist_v1"
    default_threshold = 0.92

    def embed(self, crop_bgr: np.ndarray, cls: str) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < 8 or w < 8:
            return None
        target = PERSON_CROP if cls == "person" else VEHICLE_CROP
        resized = cv2.resize(crop_bgr, target, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        # mask out very dark pixels (night-light gutter) so the signature
        # reflects the object
        mask = cv2.inRange(hsv[..., 2], 30, 255)
        if int(mask.sum()) == 0:
            mask = None  # crop is entirely dark - use everything
        hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8],
                            [0, 180, 0, 256, 0, 256]).flatten().astype(np.float32)
        aspect = w / max(1, h)
        area = (w * h) / (1920 * 1080)
        vec = np.concatenate([hist, np.array([aspect, area], dtype=np.float32)])
        return _l2norm(vec)


class OsnetEmbedder:
    """OSNet (or any 1-input/1-output re-ID CNN) via ONNX Runtime on CPU."""

    default_threshold = 0.65   # cosine on OSNet features; tune with real data

    def __init__(self, model_path: str | Path, num_threads: int = 2):
        import onnxruntime as ort
        self.path = Path(model_path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.path), sess_options=so,
                                            providers=["CPUExecutionProvider"])
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        shape = self.session.get_inputs()[0].shape   # [N,3,H,W] (may be dynamic)
        self.in_h = int(shape[2]) if isinstance(shape[2], int) else OSNET_INPUT[1]
        self.in_w = int(shape[3]) if isinstance(shape[3], int) else OSNET_INPUT[0]
        self.embedder_id = f"osnet_onnx_v1:{self.path.name}"

    def embed(self, crop_bgr: np.ndarray, cls: str) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < 8 or w < 8:
            return None
        img = cv2.resize(crop_bgr, (self.in_w, self.in_h),
                         interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        blob = img.transpose(2, 0, 1)[None]           # NCHW
        out = self.session.run([self.output_name],
                               {self.input_name: blob})[0][0]
        return _l2norm(np.asarray(out, dtype=np.float32))


def make_embedder(model_path: str | None = None):
    """Build the best available embedder.

    Model resolution: explicit arg > REID_MODEL env > the conventional
    ``data/osnet_x0_25_msmt17.onnx`` drop path (see resolve_model_path).
    With a model path: OSNet via onnxruntime; if the file is missing or
    onnxruntime isn't installed, WARN LOUDLY and fall back to the histogram -
    a silently degraded re-ID would invalidate the returning-visitor feature
    without anyone noticing.
    """
    model_path = resolve_model_path(model_path)
    if model_path:
        try:
            emb = OsnetEmbedder(model_path)
            print(f"re-ID embedder: OSNet ONNX ({model_path}), "
                  f"input {emb.in_w}x{emb.in_h}, "
                  f"default threshold {emb.default_threshold}")
            return emb
        except Exception as e:
            print(f"  !! re-ID model {model_path!r} unavailable ({e}) - "
                  f"FALLING BACK to HSV histogram. Returning-visitor matching "
                  f"will NOT survive lighting changes until this is fixed.")
    return HistogramEmbedder()
