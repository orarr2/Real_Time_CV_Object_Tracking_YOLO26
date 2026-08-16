"""Per-camera confidence calibration from reviewed boxes (plan WS4, SPEC 4.8).

    python -m tools.calibrate_conf                 # write data/per_camera_conf.json
    python -m tools.calibrate_conf --dry-run       # print, write nothing
    python -m tools.calibrate_conf --target-precision 0.92 --min-reviews 40

For every (camera, class) pair with enough verdicts, find the LOWEST
confidence gate ``conf_star`` at which the reviewed boxes reach the target
precision - lowest, because every notch above the minimum needlessly
sacrifices recall. Pairs with fewer than ``--min-reviews`` verdicts keep
their current gate (a tiny sample over-fits); pairs that cannot reach the
target under ``--max-conf`` are left alone too (the camera is noisy or the
reviews are still too few) - both per the spec.

Evidence sources (conf attached to every verdict):
  * frame reviews - box verdicts from the canvas UI, joined with the conf
    each box carries in its ``review_frames/.../<ts>.json`` sidecar;
    ``correct`` counts as TP, ``wrong``/``relabel:*`` as FP.
  * crop reviews - live_samples crops encode conf in the filename
    (``<ts>_<cls>_<NN>[_uNN].jpg``); ``correct`` -> TP, ``wrong_label`` /
    ``not_an_object`` -> FP. Crops from other pools carry no conf and are
    skipped.

Output (spec 9.3): ``data/per_camera_conf.json``. ``cameras.py`` merges it
AFTER the confidence-boost delta, so a calibrated gate OVERRIDES the
heuristic nudge for that pair - calibration beats nudging; the nudge stays
as warm-up for pairs calibration has not reached yet.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = _SRC_ROOT / "data" / "per_camera_conf.json"

TARGET_PRECISION = 0.90
MIN_REVIEWS = 30
MAX_CONF = 0.60

_CROP_NAME = re.compile(r"^\d+_([a-z]+)_(\d{2,3})(?:_u\d{2,3})?\.jpg$")


def collect_verdicts(store, snapshots_root) -> dict[tuple[str, str], list]:
    """(cam_id, cls) -> [(conf, is_tp), ...] from both review surfaces."""
    from app.review_frames import load_metadata

    out: dict[tuple[str, str], list] = {}

    def add(cam, cls, conf, is_tp):
        out.setdefault((cam, cls), []).append((float(conf), bool(is_tp)))

    # Frame reviews: conf lives in the sidecar json, verdicts key by box id.
    for fr in store._frames_by_path.values():  # noqa: SLF001 - same family
        meta = load_metadata(fr.frame_path, snapshots_root)
        if not meta:
            continue
        by_id = {str(b.get("id")): b for b in meta.get("boxes") or []}
        for box_id, verdict in fr.box_verdicts.items():
            b = by_id.get(str(box_id))
            if not b or b.get("conf") is None:
                continue
            add(fr.cam_id, b.get("cls", "?"), b["conf"], verdict == "correct")

    # Crop reviews: only live_samples filenames carry the conf.
    for r in store._by_path.values():  # noqa: SLF001
        parts = r.crop_path.split("/")
        if len(parts) < 3 or parts[0] != "live_samples":
            continue
        m = _CROP_NAME.match(parts[-1])
        if not m:
            continue
        add(parts[1], r.original_cls if r.original_cls != "?" else m.group(1),
            int(m.group(2)) / 100.0, r.verdict == "correct")
    return out


def conf_star(verdicts: list, target_precision: float = TARGET_PRECISION,
              max_conf: float = MAX_CONF) -> float | None:
    """Lowest gate <= max_conf whose surviving boxes reach the target
    precision. None when no gate qualifies (or nothing would survive)."""
    best = None
    for cand in sorted({round(c, 4) for c, _ in verdicts}):
        if cand > max_conf:
            break
        kept = [(c, tp) for c, tp in verdicts if c >= cand]
        if not kept:
            continue
        precision = sum(1 for _, tp in kept if tp) / len(kept)
        if precision >= target_precision:
            best = cand
            break     # sorted ascending: the first hit is the lowest gate
    return best


def calibrate(store, snapshots_root,
              target_precision: float = TARGET_PRECISION,
              min_reviews: int = MIN_REVIEWS,
              max_conf: float = MAX_CONF) -> dict:
    """Spec-9.3-shaped payload for every pair that qualifies."""
    cameras: dict[str, dict] = {}
    for (cam, cls), rows in sorted(collect_verdicts(store,
                                                    snapshots_root).items()):
        if len(rows) < min_reviews:
            continue
        star = conf_star(rows, target_precision, max_conf)
        if star is None:
            continue
        cameras.setdefault(cam, {})[cls] = {
            "conf": star,
            "target_precision": target_precision,
            "n_reviews": len(rows),
        }
    return {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cameras": cameras,
    }


def main() -> None:
    from app.labels import ReviewStore
    from app.visual_search import SNAPSHOTS_ROOT

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target-precision", type=float, default=TARGET_PRECISION)
    ap.add_argument("--min-reviews", type=int, default=MIN_REVIEWS)
    ap.add_argument("--max-conf", type=float, default=MAX_CONF)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    payload = calibrate(ReviewStore(), SNAPSHOTS_ROOT,
                        args.target_precision, args.min_reviews,
                        args.max_conf)
    n_pairs = sum(len(v) for v in payload["cameras"].values())
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        print(f"dry-run: {n_pairs} calibrated pair(s), nothing written")
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(out)
    print(f"calibrate_conf: {n_pairs} pair(s) -> {out} "
          f"(collector picks it up on its next hot-reload)")


if __name__ == "__main__":
    main()
