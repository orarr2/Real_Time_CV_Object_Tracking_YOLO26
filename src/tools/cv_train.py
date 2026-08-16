"""Camera-grouped cross-validation for the head trainer.

    python -m tools.cv_train --folds 4                    # full CV report
    python -m tools.cv_train --folds 3 --epochs 4         # quicker smoke

Why this exists: the dataset is a few dozen frames from a handful of FIXED
cameras. A single split (any split) judges the model on one lucky/unlucky
scene, and frames from the same camera are near-duplicates - so the only
honest score at this scale is k-fold with WHOLE CAMERAS held out per fold
(GroupKFold by cam_id). This tool:

  1. exports the reviewed frames into k camera-grouped fold layouts
     (tools/export_labels.export_folds);
  2. per fold: trains a Detect head on the fold's train split
     (tools/train_head, backbone frozen) and validates BASE vs CANDIDATE
     on the fold's held-out cameras;
  3. prints the per-fold table + mean/std mAP50 gain, runs the promotion
     gate per fold, and appends one {"event": "cv"} record to
     data/adapters/history.jsonl.

It never promotes anything itself - promotion stays with
tools/promote_adapter.py on the single camera-grouped split; this is the
measurement that says whether a promotion attempt is even worth making
(rule of thumb: a mean gain below the gate's +0.5pp with fold gains of
mixed sign is noise, not learning).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent


def _train_fold(fold_dir: Path, base: str, epochs: int, imgsz: int,
                batch: int, device: str) -> Path | None:
    """Run tools/train_head on one fold in a subprocess (its own ultralytics
    run dir, its own memory lifecycle). Returns the head artifact or None."""
    out_file = fold_dir / "head.pt"
    cmd = [sys.executable, "-m", "tools.train_head",
           "--data", str(fold_dir / "dataset.yaml"),
           "--base", base, "--epochs", str(epochs),
           "--imgsz", str(imgsz), "--batch", str(batch),
           "--device", device,
           "--out-file", str(out_file),
           "--runs-dir", str(fold_dir / "runs")]
    res = subprocess.run(cmd, cwd=_SRC_ROOT)
    return out_file if res.returncode == 0 and out_file.is_file() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reviews", default=str(_SRC_ROOT / "data" / "reviews.json"))
    ap.add_argument("--snapshots", default=str(_SRC_ROOT / "web" / "snapshots"))
    ap.add_argument("--out", default=str(_SRC_ROOT / "data" / "labels_export_cv"))
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--base", default="yolov8s.pt",
                    help="must match the VM's pinned weights "
                         "(deploy/gcp-vm/collector.service)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="matches the collector's production inference size")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--adapters-dir", default=None,
                    help="where the {'event': 'cv'} history record lands "
                         "(default: data/adapters - pass a scratch dir for "
                         "smoke runs)")
    args = ap.parse_args()

    from tools.export_labels import collect_examples, export_folds

    skipped: list[str] = []
    examples = collect_examples(Path(args.reviews), Path(args.snapshots),
                                skipped=skipped)
    if skipped:
        print(f"cv: !! {len(skipped)} reviewed frame(s) missing on disk - "
              f"not part of any fold")
    cams = sorted({ex["cam_id"] for ex in examples})
    if len(cams) < 2:
        raise SystemExit(f"cv: need frames from >= 2 cameras to group folds "
                         f"(have {len(cams)}: {', '.join(cams) or '-'})")

    out_root = Path(args.out)
    summaries = export_folds(out_root, examples, args.folds)
    print(f"cv: {len(examples)} frame(s) from {len(cams)} camera(s) -> "
          f"{len(summaries)} fold(s)")

    from tools.promote_adapter import _val_metrics
    from ultralytics import YOLO
    from app import adapters

    rows: list[dict] = []
    for s in summaries:
        fold_dir = out_root / f"fold_{s['fold']}"
        data_yaml = str(fold_dir / "dataset.yaml")
        head = _train_fold(fold_dir, args.base, args.epochs, args.imgsz,
                           args.batch, args.device)
        if head is None:
            print(f"cv: fold_{s['fold']}: training failed - skipped")
            continue
        base_model = YOLO(args.base)
        base_m = _val_metrics(base_model, data_yaml, args.imgsz)
        cand_model = YOLO(args.base)
        adapters.overlay_head(cand_model.model, adapters.load_head(head))
        cand_m = _val_metrics(cand_model, data_yaml, args.imgsz)
        ok, reasons = adapters.gate_decision(base_m, cand_m)
        rows.append({"fold": s["fold"], "val_cams": s["val_cams"],
                     "val_frames": s["val_frames"],
                     "base_map50": base_m["map50"],
                     "cand_map50": cand_m["map50"],
                     "gain_pp": round((cand_m["map50"] - base_m["map50"]) * 100, 2),
                     "gate_ok": ok, "reasons": reasons})

    if not rows:
        raise SystemExit("cv: no fold produced a result")

    print(f"\n{'fold':>4} {'val frames':>10} {'base':>7} {'cand':>7} "
          f"{'gain pp':>8} {'gate':>5}  val cams")
    for r in rows:
        print(f"{r['fold']:>4} {r['val_frames']:>10} {r['base_map50']:>7} "
              f"{r['cand_map50']:>7} {r['gain_pp']:>8} "
              f"{'PASS' if r['gate_ok'] else 'fail':>5}  "
              f"{', '.join(r['val_cams'])}")
    gains = [r["gain_pp"] for r in rows]
    mean = sum(gains) / len(gains)
    std = (sum((g - mean) ** 2 for g in gains) / len(gains)) ** 0.5
    print(f"\ncv: mAP50 gain mean {mean:+.2f}pp, std {std:.2f}pp over "
          f"{len(rows)} fold(s); gate passes: "
          f"{sum(1 for r in rows if r['gate_ok'])}/{len(rows)}")
    verdict = ("promotion attempt is justified"
               if mean >= 0.5 and min(gains) > -0.5
               else "not enough signal - label more frames before promoting")
    print(f"cv: {verdict}")

    record = {"event": "cv", "folds": rows,
              "mean_gain_pp": round(mean, 2), "std_gain_pp": round(std, 2),
              "n_frames": len(examples), "n_cams": len(cams),
              "base": Path(args.base).name, "epochs": args.epochs,
              "imgsz": args.imgsz}
    adapters.append_history(record, Path(args.adapters_dir)
                            if args.adapters_dir else adapters.ADAPTERS_DIR)
    (out_root / "cv_report.json").write_text(json.dumps(record, indent=1))
    print(f"cv: report -> {out_root / 'cv_report.json'} (+history.jsonl)")


if __name__ == "__main__":
    main()
