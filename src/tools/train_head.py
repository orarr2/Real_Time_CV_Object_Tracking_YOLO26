"""Fine-tune ONLY the Detect head of the production model on the reviewed-frame export.

    python -m tools.export_labels                      # produce the dataset
    python -m tools.train_head                         # train on it (CPU ok)
    python -m tools.train_head --epochs 6 --imgsz 512 --out-file data/adapters/head_ci.pt

Backbone frozen (D2: the LoRA replacement) - every layer up to the Detect
module gets requires_grad=False via Ultralytics' own ``freeze=``, so the
artifact this emits is the head's tensors only (~4-6 MB), saved through
``app.adapters.save_head``. Mosaic/mixup are off (tiny dataset of real
frames from FIXED viewpoints - collage augmentation fabricates geometry
these cameras never see); HSV jitter + horizontal flip stay on. Epochs are
clamped to 10 with early stopping, per the plan's budget.

The trained head is NOT promoted here - run tools/promote_adapter.py, which
gates it against the baseline on the export's val split.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent

MAX_EPOCHS = 10


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(_SRC_ROOT / "data" / "labels_export"
                                          / "dataset.yaml"))
    ap.add_argument("--base", default="yolov8s.pt",
                    help="MUST match the weights the consumer runs "
                         "(the VM collector is pinned to yolov8s.pt in "
                         "deploy/gcp-vm/collector.service since 2026-08-05) "
                         "- a head trained on one base cannot overlay onto "
                         "another")
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="matches the collector's production inference size")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-file", default=None,
                    help="head artifact path (default: data/adapters/"
                         "head_<UTC timestamp>.pt)")
    ap.add_argument("--runs-dir", default=str(_SRC_ROOT / "data" / "runs"),
                    help="ultralytics working dir (checkpoints, curves)")
    args = ap.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.is_file():
        raise SystemExit(f"dataset yaml not found: {data_yaml} - run "
                         f"`python -m tools.export_labels` first")

    from ultralytics import YOLO
    from app import adapters

    model = YOLO(args.base)
    head_idx = adapters.detect_head_index(model.model)
    epochs = max(1, min(MAX_EPOCHS, args.epochs))
    print(f"train_head: base={args.base} head_idx={head_idx} "
          f"epochs={epochs} imgsz={args.imgsz} device={args.device}")

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        freeze=head_idx,          # freezes model.0 .. model.<head_idx-1>
        mosaic=0.0,
        mixup=0.0,
        patience=5,
        workers=2,
        plots=False,
        project=args.runs_dir,
        name=f"head_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}",
        exist_ok=True,
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"training produced no best.pt under {results.save_dir}")

    trained = YOLO(str(best))
    head_state = adapters.extract_head(trained.model, head_idx=head_idx)
    # Pre-flight: the artifact must overlay onto a FRESH base model - if the
    # dataset yaml ever drifts from the COCO-80 name table the trained head
    # changes shape, and that must fail HERE with one clear sentence, not
    # two steps later inside the promotion gate.
    fresh = YOLO(args.base)
    try:
        adapters.overlay_head(fresh.model, head_state)
    except ValueError as e:
        raise SystemExit(
            f"trained head is not shape-compatible with {args.base}: {e}\n"
            f"(the dataset yaml must keep the full COCO-80 name table - "
            f"see tools/export_labels.py)")
    out_file = Path(args.out_file) if args.out_file else (
        adapters.ADAPTERS_DIR
        / f"head_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.pt")
    adapters.save_head(head_state, out_file)
    print(f"train_head: saved {len(head_state)} head tensors -> {out_file}")
    print(f"HEAD_ARTIFACT={out_file}")


if __name__ == "__main__":
    main()
