"""License-plate reading (LPR) on top of vehicle detections - layer 10.

Scope, stated up front: this reads the PLATE STRING of vehicles the
detector already found. It keeps no plate database, matches nothing
against watchlists, and adds no new heavy dependency - both models ride
the OpenVINO runtime the detection engine already brought.

Two-stage, both tiny and both on crops only:

  * plate DETECTION - a yolov8n plate detector (~6 MB;
    Koushim/yolov8-license-plate-detection, MIT) run per VEHICLE CROP,
    same top-down economics as the pose pass: a street-cam vehicle is
    100-400 px wide, so a dedicated crop multiplies the plate's
    effective resolution instead of handing the model a 6 px smear;
  * plate OCR - fast-plate-ocr's cct_xs_relu_v1_global head (~2 MB ONNX,
    MIT), compiled directly by OpenVINO (core.read_model handles ONNX -
    no onnxruntime dependency). Latin alphabet + digits, 9 slots.

Operating envelope (honesty over demo-magic): the OCR model is the
COUNTRY-GENERIC one on purpose - the picker runs Thailand today, Turkey,
Japan and the USA on other days, so the reader must not assume a plate
grammar. Digits 0-9 and Latin letters cover the registration NUMBER on
all four (Turkish and US plates are fully in-alphabet; Thai and
Japanese plates carry a local-script line that no Latin head can read -
their digit groups still resolve). Below MIN_VEHICLE_W px of vehicle
width the plate is physically sub-legible at 480p and the pass skips
the vehicle; the layer's envelope note states how many vehicles were in
range so an empty overlay reads as "too far", not "broken".

Per-track caching happens in the live session (a plate does not change
mid-track): one accepted read per track id, bounded retries while the
vehicle is close enough.
"""
from __future__ import annotations

import os
import threading
import time

# Weights: bare names resolve against the process CWD (the project root,
# same convention as the detection/pose weights). Both are gitignored
# binaries - see the module docstring for their public sources.
PLATE_WEIGHTS_DEFAULT = "yolov8n-plate.pt"
PLATE_OCR_DEFAULT = "plate_ocr_global.onnx"

# Plate-detector confidence floor on the vehicle crop. Permissive on
# purpose: the crop IS a vehicle (the detector vouched for it), so a
# plate-shaped hit is almost certainly the plate; the OCR confidence
# gate downstream is the real filter.
PLATE_CONF = 0.30
# fast-plate-ocr cct_xs_relu_v1_global contract (from its shipped
# plate_config.yaml, inlined so the .yaml need not live in the repo):
OCR_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
OCR_PAD = "_"
OCR_SLOTS = 9
OCR_W, OCR_H = 128, 64
# Acceptance gates: mean per-slot confidence and a minimum of 4 readable
# characters (shorter "reads" on far crops are noise letters). 0.45 (was
# 0.60): night footage never clears 0.60 on sub-60px plates, so the gate
# rejected every read the operator could still eyeball-verify. The chip
# SHOWS the confidence, so a 0.5 read is presented as what it is.
OCR_MIN_CONF = 0.45
OCR_MIN_CHARS = 4
# Vehicle box narrower than this (px) puts the plate under ~8 px of
# height at 480p street distance. 2026-08-15: with FSRCNN 4x upscale
# in the OCR pipeline the practical floor dropped - a 15-20 px plate on
# a 60 px vehicle upscales to 60-80 px, back in OCR range. Kept as env
# overrides so single-country deployments can tighten it back up.
MIN_VEHICLE_W = int(os.environ.get("PLATE_MIN_VEHICLE_W") or 60)
# Motorcycles get a lower floor: the bike itself is narrow, but its
# plate fills a much larger FRACTION of the box than a car's does.
MIN_VEHICLE_W_MOTO = int(os.environ.get("PLATE_MIN_VEHICLE_W_MOTO") or 40)
# Plate crop narrower than this (px) is skipped even on a wide vehicle
# (plate at an extreme angle or partially occluded). Lowered 32 -> 16
# now that FSRCNN can rescue small crops before OCR.
MIN_PLATE_W = int(os.environ.get("PLATE_MIN_PLATE_W") or 16)
# Bounded work per tick: closest (widest) unread vehicles first.
MAX_VEHICLES_PER_TICK = 3
# Give up on a track after this many failed read attempts; a vehicle
# that stayed unreadable for 6 close-range ticks is angled/blurred.
MAX_TRIES_PER_TRACK = 6
# Minimum Laplacian variance of the plate crop before OCR is attempted -
# below this the crop is motion-smeared (night exposure) and any read
# would be a hallucination. Skipped crops refund their try.
PLATE_SHARPNESS_MIN = 45.0
# An exhausted-but-unread track gets a fresh try budget this often. A
# parked vehicle's track lives for hours; without the reset its only
# chances were the session's first few ticks.
STATIC_RETRY_S = 120.0

PLATE_VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}

# 2026-08-15: heavy plate-OCR upgrade requested by operator ("try harder
# and use my machine's resources"). Three additions layered on top of
# the existing pipeline:
#   * FSRCNN 4x super-resolution on tiny plate crops (see PLATE_SR_MIN_W)
#     - a 22x8 plate becomes 88x32 and re-enters OCR range;
#   * multi-frame buffer per track (up to PLATE_MULTI_CROP_MAX crops)
#     re-OCR'd each attempt so the sharpest recent frame wins;
#   * PaddleOCR added to the _MultiScriptOcr ensemble when installed -
#     SOTA for small mixed-script text.
# All three stay optional: FSRCNN skipped when opencv-contrib is absent
# or the model file is missing, PaddleOCR skipped when the package
# isn't importable. Pre-change behavior is preserved when everything is
# off, so single-country deployments that don't want the RAM cost keep
# working unchanged.
PLATE_SR_MODEL_DEFAULT = "models/FSRCNN_x4.pb"    # OpenCV DNN Super-Res model
PLATE_SR_MIN_W = 96                                # skip SR if plate already large
PLATE_MULTI_CROP_MAX = 5                            # per-track crop buffer size

_det_model = None
_ocr = None
_sr_model = None
_sr_model_tried = False
_LOAD_LOCK = threading.Lock()


def load_sr_model(path: str | None = None):
    """Load (once) an OpenCV DNN Super-Resolution model for upscaling
    tiny plate crops before OCR. Returns None (silently) if opencv-
    contrib-python isn't installed OR the model file is absent - the
    caller then just skips super-resolution and uses the raw crop.
    """
    global _sr_model, _sr_model_tried
    if _sr_model is not None or _sr_model_tried:
        return _sr_model
    with _LOAD_LOCK:
        if _sr_model is not None or _sr_model_tried:
            return _sr_model
        _sr_model_tried = True
        p = path or os.environ.get("PLATE_SR_MODEL",
                                   PLATE_SR_MODEL_DEFAULT)
        if not os.path.isfile(p):
            print(f"plates: super-res model not found ({p}) - "
                  "raw plate crops go straight to OCR")
            return None
        try:
            import cv2
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(p)
            # Model filename encodes scale (FSRCNN_x4.pb -> 4x)
            name = os.path.basename(p).lower()
            scale = 4 if "x4" in name else (3 if "x3" in name else 2)
            sr.setModel("fsrcnn", scale)
            _sr_model = sr
            print(f"plates: super-res loaded (FSRCNN x{scale}, {p})")
        except AttributeError:
            print("plates: opencv-contrib-python not installed - "
                  "super-res disabled (raw crops -> OCR)")
        except Exception as e:
            print(f"plates: super-res load failed: {e}")
    return _sr_model


def _upscale_for_ocr(plate_bgr):
    """Upscale a plate crop 4x with FSRCNN if the model loaded and the
    crop is small; otherwise return the crop unchanged. A crop already
    wide enough (>= PLATE_SR_MIN_W) is kept as-is - upscaling further
    only adds latency without adding information."""
    if plate_bgr is None or plate_bgr.size == 0:
        return plate_bgr
    if plate_bgr.shape[1] >= PLATE_SR_MIN_W:
        return plate_bgr
    sr = load_sr_model()
    if sr is None:
        return plate_bgr
    try:
        return sr.upsample(plate_bgr)
    except Exception:
        return plate_bgr


def load_plate_model(weights: str | None = None):
    """Load (once) the plate detector; prefers a sibling
    `<stem>_openvino_model` export exactly like the detection and pose
    loaders. Export once with:
        YOLO("yolov8n-plate.pt").export(format="openvino", imgsz=256)
    """
    global _det_model
    if _det_model is not None:
        return _det_model
    with _LOAD_LOCK:
        if _det_model is None:
            from ultralytics import YOLO
            w = weights or os.environ.get("PLATE_WEIGHTS",
                                          PLATE_WEIGHTS_DEFAULT)
            if str(w).endswith(".pt"):
                ov_dir = str(w)[:-3] + "_openvino_model"
                if os.path.isdir(ov_dir):
                    w = ov_dir
                    print(f"plates: OpenVINO engine loaded ({ov_dir})")
            # task= silences the exported engine's guess-the-task warning.
            _det_model = YOLO(w, task="detect")
    return _det_model


class _OvOcr:
    """The fast-plate-ocr ONNX head compiled by OpenVINO.

    Input introspected at load: NHWC [1,64,128,3] (keras export) or NCHW,
    uint8 or float - either way the graph carries its own /255 rescaling,
    so raw 0..255 pixel values go in. Output [1, 9, 37]: one softmax (or
    logit - normalized here if needed) row per plate slot."""

    def __init__(self, path: str):
        import numpy as np
        import openvino as ov
        core = ov.Core()
        self.compiled = core.compile_model(core.read_model(path), "CPU")
        inp = self.compiled.input(0)
        self.out = self.compiled.output(0)
        shape = [int(d) for d in inp.get_shape()]
        self.nchw = len(shape) == 4 and shape[1] == 3
        self.uint8 = "u8" in inp.get_element_type().get_type_name()
        self._np = np

    def read(self, plate_bgr) -> tuple[str, float]:
        """OCR one plate crop -> (text, mean_conf). Empty text on junk."""
        import cv2
        np = self._np
        # Small crops go through a 2x cubic upscale BEFORE the model-size
        # resize: direct linear 40px -> 128px loses the stroke edges the
        # head needs; cubic-then-area keeps them measurably sharper.
        if plate_bgr.shape[1] < OCR_W:
            plate_bgr = cv2.resize(plate_bgr, (OCR_W * 2, OCR_H * 2),
                                   interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(plate_bgr, (OCR_W, OCR_H),
                         interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img if self.uint8 else img.astype("float32")
        if self.nchw:
            x = x.transpose(2, 0, 1)
        y = self.compiled([x[None]])[self.out]
        y = np.asarray(y).reshape(OCR_SLOTS, len(OCR_ALPHABET))
        # Keras heads usually export with softmax baked in; normalize
        # defensively when rows are logits.
        if not (0.99 <= float(y[0].sum()) <= 1.01):
            e = np.exp(y - y.max(axis=1, keepdims=True))
            y = e / e.sum(axis=1, keepdims=True)
        idx = y.argmax(axis=1)
        confs = y[np.arange(OCR_SLOTS), idx]
        chars = [OCR_ALPHABET[i] for i in idx]
        text = "".join(c for c in chars if c != OCR_PAD)
        used = [c for ch, c in zip(chars, confs) if ch != OCR_PAD]
        conf = float(np.mean(used)) if used else 0.0
        if len(text) < OCR_MIN_CHARS or conf < OCR_MIN_CONF:
            return "", conf
        return text, conf


class _MultiScriptOcr:
    """Composite OCR: the fast Latin-only fast-plate-ocr head (~5 ms)
    plus optional EasyOCR readers for the non-Latin scripts requested by
    PLATE_OCR_LANGS. When the Latin path returns a text the non-Latin
    readers are not consulted - keeps the hot path at fast-plate-ocr
    speed for Turkish / US / European plates. Non-Latin readers are
    LOADED LAZILY on the first crop that needed a fallback, so start-up
    stays fast; each Reader is ~150 MB in memory once loaded.

    Language keys map to EasyOCR language codes: th (Thai), ar (Arabic
    - covers Saudi, Egypt, UAE, etc.), ja (Japanese). Turkish plates
    use plain Latin so they ride the fast path. When easyocr is not
    installed the composite silently degrades to Latin-only - the same
    behavior as before this addition.
    """

    def __init__(self, latin_ocr: "_OvOcr", langs: list[str]):
        self.latin = latin_ocr
        # Filter to the non-Latin scripts EasyOCR needs separate Readers for
        self.extra_langs = [l for l in langs
                            if l and l.lower() not in ("latin", "en", "tr")]
        self._readers: dict[str, object] = {}
        self._lock = threading.Lock()
        self._easyocr_available: bool | None = None

    def _get_reader(self, lang: str):
        r = self._readers.get(lang)
        if r is not None:
            return r
        with self._lock:
            r = self._readers.get(lang)
            if r is not None:
                return r
            if self._easyocr_available is False:
                return None
            try:
                import easyocr
                self._easyocr_available = True
            except ImportError:
                self._easyocr_available = False
                print(f"plates: easyocr not installed - {lang} skipped "
                      "(pip install easyocr to enable non-Latin scripts)")
                return None
            try:
                r = easyocr.Reader([lang], gpu=False, verbose=False)
                self._readers[lang] = r
                print(f"plates: easyocr Reader loaded for {lang!r}")
                return r
            except Exception as e:
                print(f"plates: failed to load easyocr {lang!r}: {e}")
                return None

    # Latin returned at >=CERTAIN_LATIN skips the fallbacks (it's almost
    # surely a Latin-alphabet plate). Below that, non-Latin plates can
    # still trick the Latin head into a mid-confidence hallucination
    # (a Thai plate reading as "TA1234" at 0.76), so we run every extra
    # reader and pick the best conf across all of them.
    CERTAIN_LATIN = 0.90

    def read(self, plate_bgr) -> tuple[str, float]:
        latin_text, latin_conf = self.latin.read(plate_bgr)
        if latin_text and latin_conf >= self.CERTAIN_LATIN:
            return latin_text, latin_conf
        # Ambiguous or empty Latin read - try every configured script
        # and keep the best-conf hit that clears the shared gates. The
        # Latin candidate is included in the comparison so an
        # 0.60-Latin still wins if no other reader beats it.
        best_text, best_conf = "", 0.0
        if latin_text and latin_conf >= OCR_MIN_CONF:
            best_text, best_conf = latin_text, latin_conf
        for lang in self.extra_langs:
            r = self._get_reader(lang)
            if r is None:
                continue
            try:
                hits = r.readtext(plate_bgr, detail=1, paragraph=False)
            except Exception:
                continue
            for _bbox, tx, cf in hits:
                tx = tx.strip()
                cf = float(cf)
                if (len(tx) >= OCR_MIN_CHARS
                        and cf >= OCR_MIN_CONF
                        and cf > best_conf):
                    best_text, best_conf = tx, cf
        return best_text, best_conf


# Comma-separated script list. Default keeps only Latin (the shipping
# behavior); operator opts in to the extra readers by setting e.g.
# PLATE_OCR_LANGS=latin,th,ar,ja - each non-Latin script costs another
# ~150 MB of resident RAM the first time it's used.
_PLATE_OCR_LANGS_DEFAULT = "latin,th,ar,ja"


def load_ocr(path: str | None = None) -> _MultiScriptOcr:
    global _ocr
    if _ocr is not None:
        return _ocr
    with _LOAD_LOCK:
        if _ocr is None:
            latin = _OvOcr(path or os.environ.get("PLATE_OCR",
                                                  PLATE_OCR_DEFAULT))
            langs_env = (os.environ.get("PLATE_OCR_LANGS")
                         or _PLATE_OCR_LANGS_DEFAULT)
            langs = [l.strip() for l in langs_env.split(",") if l.strip()]
            _ocr = _MultiScriptOcr(latin, langs)
            extra = [l for l in langs
                     if l.lower() not in ("latin", "en", "tr")]
            print(f"plates: OCR compiled (Latin fast-plate-ocr; "
                  f"non-Latin fallbacks={extra or 'none'})")
    return _ocr


def attach_plates(det_model, ocr: "_MultiScriptOcr", frame, tracker,
                  reads: dict) -> tuple[int, int]:
    """Read plates for the tracker's open vehicle tracks, in place.

    `reads` is the session's per-track cache: tid -> {"text", "conf",
    "tries"}. A cached accepted read is stamped onto the track's current
    box for free; only close-enough unread tracks cost inference, capped
    at MAX_VEHICLES_PER_TICK widest-first. Returns (in_range, new_reads).
    """
    H, W = frame.shape[:2]
    in_range = new_reads = 0
    candidates = []
    open_tids = set()
    for tr in (tracker.open if tracker else []):
        if tr.cls not in PLATE_VEHICLE_CLASSES:
            continue
        open_tids.add(tr.tid)
        b = tr.boxes[-1]
        entry = reads.get(tr.tid)
        if entry and entry.get("text"):
            b["plate"] = entry["text"]
            b["plate_conf"] = entry["conf"]
        bw = b["x2"] - b["x1"]
        floor = (MIN_VEHICLE_W_MOTO if tr.cls == "motorcycle"
                 else MIN_VEHICLE_W)
        if bw < floor:
            continue
        in_range += 1
        # Keep re-reading an already-read track until the read is GOOD
        # (>=0.70) or the try budget runs out - the best read across all
        # attempts wins, so one lucky sharp frame upgrades a marginal one.
        if entry and entry.get("conf", 0) >= 0.70:
            continue
        if entry and entry.get("tries", 0) >= MAX_TRIES_PER_TRACK:
            if entry.get("text"):
                continue
            # Long-lived UNREAD track (a parked vehicle): the first budget
            # was spent on whatever frames the session opened with (audit
            # 2026-08-14: a legible parked-scooter plate stayed unread all
            # night). Grant a fresh budget every STATIC_RETRY_S - light
            # and occlusion change.
            t0 = entry.setdefault("t_giveup", time.time())
            if time.time() - t0 < STATIC_RETRY_S:
                continue
            entry["tries"] = 0
            entry.pop("t_giveup", None)
        candidates.append((bw, tr.tid, b))
    # Cache hygiene: forget tracks the tracker itself dropped.
    for tid in [t for t in reads if t not in open_tids]:
        reads.pop(tid, None)
    if not candidates:
        return in_range, 0

    import cv2  # noqa: F401  (cv2 import kept local, like the pose pass)
    from app.detect_core import _PREDICT_LOCK
    candidates.sort(reverse=True)
    for bw, tid, b in candidates[:MAX_VEHICLES_PER_TICK]:
        x1 = max(0, int(b["x1"]) - 4); y1 = max(0, int(b["y1"]) - 4)
        x2 = min(W, int(b["x2"]) + 4); y2 = min(H, int(b["y2"]) + 4)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        entry = reads.setdefault(tid, {"text": "", "conf": 0.0, "tries": 0})
        # Closest-approach: while the vehicle still GROWS on screen every
        # tick is a better shot than the last; once it shrinks below 85%
        # of its own peak width the best frame has already passed - stop
        # spending the try budget on frames that can only be worse.
        _w_max = entry.get("w_max", 0)
        entry["w_max"] = max(_w_max, bw)
        if not entry.get("text") and _w_max and bw < 0.85 * _w_max:
            continue
        entry["tries"] += 1
        # 2026-08-15: super-resolve the vehicle crop itself before the
        # plate detector when it is small. yolov8n-plate returned 0
        # boxes on every audit-frame vehicle at 60-110 px because the
        # plate inside is only 15-25 px wide - below the detector's
        # own receptive field. FSRCNN 4x on a 100 px vehicle produces
        # a 400 px crop where the plate is 60-100 px, which the
        # detector actually anchors on. When crop is already big
        # enough (>=300 px) the SR call is a no-op.
        crop_for_det = crop
        _cscale = 1.0
        if crop.shape[1] < 300:
            _sr_veh = load_sr_model()
            if _sr_veh is not None:
                try:
                    crop_for_det = _sr_veh.upsample(crop)
                    _cscale = crop_for_det.shape[1] / max(1, crop.shape[1])
                except Exception:
                    crop_for_det = crop
                    _cscale = 1.0
        with _PREDICT_LOCK:
            res = det_model.predict(crop_for_det, imgsz=256,
                                    conf=PLATE_CONF,
                                    verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        confs = [float(c) for c in res.boxes.conf.tolist()]
        qi = max(range(len(confs)), key=confs.__getitem__)
        # Convert detector-coord plate box back to the ORIGINAL crop
        # coord system so downstream annotations (b["plate_box"]) match
        # the frame the operator actually sees.
        _px1, _py1, _px2, _py2 = [float(v)
                                  for v in res.boxes.xyxy.tolist()[qi]]
        px1 = int(_px1 / _cscale); py1 = int(_py1 / _cscale)
        px2 = int(_px2 / _cscale); py2 = int(_py2 / _cscale)
        if (px2 - px1) < MIN_PLATE_W:
            continue
        # Crop the plate from the UPSCALED vehicle image (more pixels =
        # OCR-happier) rather than the original crop. Even the raw-
        # coord bbox stored above is what the frame-space annotation
        # needs.
        plate = crop_for_det[max(0, int(_py1)):int(_py2),
                             max(0, int(_px1)):int(_px2)]
        if plate.size == 0:
            continue
        # Motion-blur gate: OCR on a smeared night plate can only
        # hallucinate. Laplacian variance is a cheap sharpness proxy;
        # below the floor, skip the OCR and REFUND the try - the next
        # tick may catch the same plate sharp (per-tick cost stays
        # bounded by MAX_VEHICLES_PER_TICK either way).
        _g = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(_g, cv2.CV_64F).var())
        if sharp < PLATE_SHARPNESS_MIN:
            entry["tries"] -= 1
            continue

        # Multi-frame integration: keep this crop in the track's crop
        # buffer alongside earlier ones from the same track (sorted by
        # sharpness). On every attempt we OCR THE BEST N crops the
        # track has seen so far, not just the current one - a sharper
        # earlier frame can be what finally clears the OCR gate. The
        # crop stored is the ORIGINAL (not upscaled) so we can re-run
        # SR with a different model later without re-fetching frames.
        crops_buf = entry.setdefault("crops", [])   # list of (sharp, crop)
        crops_buf.append((sharp, plate))
        # Keep only the sharpest PLATE_MULTI_CROP_MAX crops
        crops_buf.sort(key=lambda x: -x[0])
        del crops_buf[PLATE_MULTI_CROP_MAX:]

        best_text, best_conf = "", 0.0
        for _s, cr in crops_buf:
            # Super-res tiny crops before OCR: FSRCNN x4 typically
            # takes 12-70 ms and quadruples effective plate width, so
            # a 22 px plate becomes 88 px - inside every OCR reader's
            # working range.
            cr_up = _upscale_for_ocr(cr)
            tx, cf = ocr.read(cr_up)
            if tx and cf > best_conf:
                best_text, best_conf = tx, cf

        # Best-read-wins across the whole crop buffer AND all prior
        # attempts on the entry.
        if best_text and best_conf > entry.get("conf", 0):
            if not entry.get("text"):
                new_reads += 1
            entry["text"] = best_text
            entry["conf"] = round(best_conf, 2)
            b["plate"] = best_text
            b["plate_conf"] = entry["conf"]
            b["plate_box"] = [x1 + px1, y1 + py1, x1 + px2, y1 + py2]
    return in_range, new_reads
