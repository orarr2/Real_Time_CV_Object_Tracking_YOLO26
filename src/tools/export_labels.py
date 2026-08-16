"""Export reviewed frames as a YOLO training dataset.

    python -m tools.export_labels                       # everything reviewed
    python -m tools.export_labels --cam konya_hukumet   # one camera
    python -m tools.export_labels --reviewed-boxes-only # drop unverified boxes
    python -m tools.export_labels --out data/labels_export --val-frac 0.1

This is the missing final leg of the feedback loop: verdicts used to stop
at threshold nudges, and the class corrections the operator typed were
never consumed by anything. This tool turns ``data/reviews.json``'s frame
reviews + the ``review_frames/`` metadata into the standard Ultralytics
layout, ready for a fine-tune run on any machine with a GPU (or Colab):

    <out>/
      dataset.yaml
      images/train/*.jpg   images/val/*.jpg
      labels/train/*.txt   labels/val/*.txt

Label mapping per reviewed box:
  correct          -> keep the model's class
  wrong / object   -> DROP the annotation (false positive / static thing;
                      both teach background)
  relabel:<cls>    -> keep the box with the operator's class
  (no verdict)     -> keep the model's class as a weak label, unless
                      ``--reviewed-boxes-only`` - dropping an unverified real
                      car would teach the model "that is background", which
                      is worse than trusting the detector's own label.
  missed_detections-> add a new annotation with the operator's class

Split is CHRONOLOGICAL (val = the most recent ``--val-frac``), so
validation measures drift toward the present instead of leaking
near-duplicate frames from train.

Crop-level reviews (the legacy single-crop UI) carry no frame coordinates,
so they cannot become YOLO rows; they keep feeding the confidence tuner
only.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent

# NATIVE COCO ids, not a compact 0..6 remap. This is what makes the
# head-adapter loop work: training against nc=80 keeps the Detect head the
# EXACT shape of the stock base model (yolov8n), so (a) the fine-tune starts from the
# pretrained head instead of a random 7-class one - which matters enormously
# on tiny operator-labeled datasets - and (b) the emitted head tensors
# overlay cleanly onto the base model at inference (a 7-class head is
# shape-incompatible and the promotion gate rightly refuses it).
# Classes the operator never labels simply contribute no examples.
EXPORT_CLASSES = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3,
                  "bus": 5, "train": 6, "truck": 7}
_CLS_ID = EXPORT_CLASSES

# Full COCO-80 name table for dataset.yaml (ultralytics needs every id the
# model predicts named, even the 73 we never label).
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def _yolo_row(cls: str, box: list[float], W: int, H: int) -> str | None:
    """(x1,y1,x2,y2) pixels -> 'id cx cy w h' normalized, clamped to [0,1]."""
    cid = _CLS_ID.get(cls)
    if cid is None or W <= 0 or H <= 0:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = max(0.0, min(x1, x2)), min(float(W), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(H), max(y1, y2))
    w, h = x2 - x1, y2 - y1
    if w < 2 or h < 2:
        return None
    cx, cy = (x1 + x2) / 2.0 / W, (y1 + y2) / 2.0 / H
    return f"{cid} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}"


def collect_examples(reviews_path: Path, snapshots_root: Path,
                     cam: str | None = None,
                     reviewed_boxes_only: bool = False,
                     skipped: list | None = None) -> list[dict]:
    """One entry per reviewed frame that still exists on disk:
    {frame_abs, frame_rel, cam_id, rows: [yolo line, ...], stats {...}}

    A reviewed frame whose image/metadata is gone is appended to `skipped`
    (when given) instead of vanishing silently - silent skips are how a
    36-frame labeling effort once exported as 3 train + 1 val images and
    two training runs measured +0.00pp without anyone noticing why."""
    try:
        data = json.loads(reviews_path.read_text())
    except (OSError, ValueError):
        return []
    out: list[dict] = []
    for fr in data.get("frame_reviews", []):
        rel = fr.get("frame_path") or ""
        cam_id = fr.get("cam_id") or "?"
        if cam and cam_id != cam:
            continue
        frame_abs = snapshots_root / rel
        meta_abs = frame_abs.with_suffix(".json")
        if not frame_abs.is_file() or not meta_abs.is_file():
            if skipped is not None:
                skipped.append(rel)
            continue   # frame evicted/cleared since the review - image is gone
        try:
            meta = json.loads(meta_abs.read_text())
        except (OSError, ValueError):
            continue
        W = int(meta.get("frame_w") or 0)
        H = int(meta.get("frame_h") or 0)
        verdicts = fr.get("box_verdicts") or {}
        rows: list[str] = []
        kept = dropped = relabeled = weak = 0
        for b in meta.get("boxes") or []:
            v = verdicts.get(str(b.get("id")))
            cls = b.get("cls") or "?"
            if v in ("wrong", "object"):
                dropped += 1
                continue
            if isinstance(v, str) and v.startswith("relabel:"):
                cls = v.split(":", 1)[1]
                relabeled += 1
            elif v == "correct":
                kept += 1
            else:
                if reviewed_boxes_only:
                    dropped += 1
                    continue
                weak += 1
            row = _yolo_row(cls, b.get("box") or [], W, H)
            if row:
                rows.append(row)
        added = 0
        for miss in fr.get("missed_detections") or []:
            row = _yolo_row(miss.get("cls") or "?", miss.get("box") or [], W, H)
            if row:
                rows.append(row)
                added += 1
        out.append({
            "frame_abs": frame_abs, "frame_rel": rel, "cam_id": cam_id,
            "rows": rows,
            "stats": {"kept": kept, "dropped": dropped,
                      "relabeled": relabeled, "weak": weak, "added_fn": added},
        })
    # Chronological order - frame filenames are microsecond timestamps.
    out.sort(key=lambda e: e["frame_rel"])
    return out


def _write_split(out_dir: Path, train: list[dict], val: list[dict]) -> dict:
    """Write one Ultralytics train/val layout. Returns summary counts."""
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    totals = {"frames": 0, "val_frames": 0, "labels": 0,
              "kept": 0, "dropped": 0, "relabeled": 0, "weak": 0, "added_fn": 0}
    for part, part_examples in (("train", train), ("val", val)):
        for ex in part_examples:
            # cam prefix keeps two cams' identical microsecond stamps distinct
            stem = f"{ex['cam_id']}_{ex['frame_abs'].stem}"
            shutil.copy2(ex["frame_abs"],
                         out_dir / "images" / part / f"{stem}.jpg")
            (out_dir / "labels" / part / f"{stem}.txt").write_text(
                "\n".join(ex["rows"]) + ("\n" if ex["rows"] else ""))
            totals["frames"] += 1
            totals["val_frames"] += (part == "val")
            totals["labels"] += len(ex["rows"])
            for k, v in ex["stats"].items():
                totals[k] += v
    yaml = ["# Generated by tools/export_labels.py - reviewed frames only",
            f"path: {out_dir.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:"]
    yaml += [f"  {i}: {c}" for i, c in enumerate(COCO80)]
    (out_dir / "dataset.yaml").write_text("\n".join(yaml) + "\n")
    return totals


def group_cameras(examples: list[dict], k: int) -> list[list[str]]:
    """Partition cam_ids into k groups, balancing frame counts (greedy:
    largest camera into the currently-smallest group). Deterministic."""
    from collections import Counter
    counts = Counter(ex["cam_id"] for ex in examples)
    k = max(1, min(k, len(counts)))
    groups: list[list[str]] = [[] for _ in range(k)]
    sizes = [0] * k
    for cam_id, n in sorted(counts.items(), key=lambda cv: (-cv[1], cv[0])):
        i = sizes.index(min(sizes))
        groups[i].append(cam_id)
        sizes[i] += n
    return [sorted(g) for g in groups if g]


def split_by_camera(examples: list[dict],
                    val_frac: float = 0.1) -> tuple[list[dict], list[dict]]:
    """GROUPED split: whole cameras go to val, never single frames.

    Frames from a fixed camera are near-duplicates of each other (same
    static background minutes apart), so any within-camera split leaks the
    scene into val and inflates the score. Held-out cameras are picked
    smallest-first until val_frac of the frames is covered (at least one
    camera when there are two or more)."""
    from collections import Counter
    counts = Counter(ex["cam_id"] for ex in examples)
    if len(counts) < 2:
        return list(examples), []      # one camera: grouping cannot help
    target = max(1, round(len(examples) * val_frac))
    val_cams: set[str] = set()
    covered = 0
    for cam_id, n in sorted(counts.items(), key=lambda cv: (cv[1], cv[0])):
        if covered >= target or len(val_cams) == len(counts) - 1:
            break
        val_cams.add(cam_id)
        covered += n
    train = [ex for ex in examples if ex["cam_id"] not in val_cams]
    val = [ex for ex in examples if ex["cam_id"] in val_cams]
    return train, val


def export(out_dir: Path, examples: list[dict], val_frac: float = 0.1,
           split: str = "chrono") -> dict:
    """Write the Ultralytics layout. Returns summary counts.

    split="chrono": val = the most recent val_frac (legacy behavior).
    split="camera": val = whole held-out cameras (no same-scene leakage).
    """
    if not examples:
        return {"frames": 0}
    if split == "camera":
        train, val = split_by_camera(examples, val_frac)
    else:
        n_val = (max(1, round(len(examples) * val_frac))
                 if len(examples) > 1 else 0)
        split_at = len(examples) - n_val
        train, val = examples[:split_at], examples[split_at:]
    totals = _write_split(out_dir, train, val)
    totals["split"] = split
    totals["val_cams"] = sorted({ex["cam_id"] for ex in val})
    return totals


def export_folds(out_dir: Path, examples: list[dict], folds: int) -> list[dict]:
    """Camera-grouped k-fold layouts for cross-validation: fold_i's val =
    the i-th camera group, train = every other group. Writes
    <out>/fold_<i>/{images,labels,dataset.yaml} + <out>/folds.json."""
    groups = group_cameras(examples, folds)
    summaries: list[dict] = []
    for i, cams in enumerate(groups):
        cams_set = set(cams)
        train = [ex for ex in examples if ex["cam_id"] not in cams_set]
        val = [ex for ex in examples if ex["cam_id"] in cams_set]
        totals = _write_split(out_dir / f"fold_{i}", train, val)
        totals.update({"fold": i, "val_cams": cams})
        summaries.append(totals)
    (out_dir / "folds.json").write_text(json.dumps(summaries, indent=1))
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reviews", default=str(_SRC_ROOT / "data" / "reviews.json"))
    ap.add_argument("--snapshots", default=str(_SRC_ROOT / "web" / "snapshots"))
    ap.add_argument("--out", default=str(_SRC_ROOT / "data" / "labels_export"))
    ap.add_argument("--cam", default=None, help="restrict to one cam_id")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="validation fraction (default 0.1)")
    ap.add_argument("--split", choices=("chrono", "camera"), default="camera",
                    help="val selection: 'camera' holds out WHOLE cameras "
                         "(no same-scene leakage - the default), 'chrono' "
                         "is the legacy most-recent slice")
    ap.add_argument("--folds", type=int, default=0,
                    help="also write <out>/fold_i camera-grouped k-fold "
                         "layouts for cross-validation (see tools/cv_train)")
    ap.add_argument("--reviewed-boxes-only", action="store_true",
                    help="drop boxes without an explicit verdict instead of "
                         "keeping them as weak labels")
    args = ap.parse_args()

    skipped: list[str] = []
    examples = collect_examples(Path(args.reviews), Path(args.snapshots),
                                cam=args.cam,
                                reviewed_boxes_only=args.reviewed_boxes_only,
                                skipped=skipped)
    if skipped:
        print(f"!! {len(skipped)} reviewed frame(s) SKIPPED - image/metadata "
              f"no longer on disk. Each one is a labeled example the trainer "
              f"cannot see:")
        for rel in skipped[:10]:
            print(f"     missing: {rel}")
        if len(skipped) > 10:
            print(f"     ... and {len(skipped) - 10} more")
    if not examples:
        print("nothing to export: no reviewed frames whose image still exists "
              "(review some frames in the dashboard first)")
        return
    totals = export(Path(args.out), examples, val_frac=args.val_frac,
                    split=args.split)
    print(f"exported {totals['frames']} frame(s) "
          f"({totals['val_frames']} val) with {totals['labels']} label row(s) "
          f"-> {args.out} [split={totals['split']}, "
          f"val cams: {', '.join(totals['val_cams']) or '-'}]")
    print(f"  verdicts: {totals['kept']} kept · {totals['dropped']} dropped · "
          f"{totals['relabeled']} relabeled · {totals['weak']} weak · "
          f"{totals['added_fn']} operator-added")
    if args.folds and args.folds > 1:
        summaries = export_folds(Path(args.out), examples, args.folds)
        for s in summaries:
            print(f"  fold_{s['fold']}: {s['frames']} frames "
                  f"({s['val_frames']} val) - val cams: "
                  f"{', '.join(s['val_cams'])}")
    print("  train: yolo detect train model=yolov8n.pt "
          f"data={Path(args.out) / 'dataset.yaml'} epochs=10 imgsz=960")


if __name__ == "__main__":
    main()
