"""CLI for search-by-example - the terminal twin of web/search.html.

Search everything the system has saved for things that look like a query photo:

    # search the saved snapshots + re-ID registry with an uploaded/query image
    python -m tools.search_by_image --query /path/to/what_im_looking_for.jpg

    # no collector history yet? seed a demo index from still images first:
    python -m tools.search_by_image --seed-images "docs/images/*.jpg"
    python -m tools.search_by_image --query crop.jpg --no-registry

    # OSNet signature instead of the HSV histogram (provide the .onnx yourself)
    python -m tools.search_by_image --query crop.jpg --reid-model osnet.onnx

Run from src/ (same convention as the other tools).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2

from app.reid_embed import make_embedder
from app.visual_search import (
    DEFAULT_DB,
    SNAPSHOTS_ROOT,
    SnapshotIndex,
    extract_query_objects,
    search_registry,
    seed_index_from_images,
)


def _load_yolo(weights: str):
    if weights.lower() in ("off", "none", ""):
        return None
    try:
        from app.detect_core import load_model
        return load_model(weights)
    except Exception as e:
        print(f"YOLO unavailable ({e}); the query image will be embedded "
              f"whole instead of per detected object.")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--query", help="image of what you're looking for")
    ap.add_argument("--seed-images",
                    help="glob of still images to detect + index as demo snapshots")
    ap.add_argument("--snapshots", default=str(SNAPSHOTS_ROOT),
                    help="snapshots root to search (default web/snapshots)")
    ap.add_argument("--db", default=str(DEFAULT_DB),
                    help="re-ID registry path (default data/reid.db)")
    ap.add_argument("--no-registry", action="store_true",
                    help="skip the reid.db entity search")
    ap.add_argument("--reid-model", default=None,
                    help="OSNet .onnx for the similarity signature "
                         "(default: HSV histogram)")
    ap.add_argument("--yolo", default="yolov8s.pt",
                    help='YOLO weights for object extraction ("off" to disable)')
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--min-sim", type=float, default=0.30)
    ap.add_argument("--classes", default="",
                    help="restrict query objects, e.g. person or car,truck")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    if not args.query and not args.seed_images:
        ap.error("nothing to do: pass --query and/or --seed-images")

    embedder = make_embedder(args.reid_model)
    model = _load_yolo(args.yolo)

    if args.seed_images:
        paths = sorted(glob.glob(args.seed_images))
        if not paths:
            sys.exit(f"--seed-images matched nothing: {args.seed_images!r}")
        if model is None:
            sys.exit("--seed-images needs YOLO (pip install ultralytics)")
        saved = seed_index_from_images(paths, model, embedder=embedder,
                                       snapshots_root=args.snapshots,
                                       conf=args.conf)
        print(f"seeded {len(saved)} crops from {len(paths)} images "
              f"under {args.snapshots}/events/demo/")

    if not args.query:
        return

    img = cv2.imread(args.query)
    if img is None:
        sys.exit(f"cannot read query image: {args.query}")
    classes = ({c.strip() for c in args.classes.split(",") if c.strip()}
               or None)
    queries = extract_query_objects(img, model=model, embedder=embedder,
                                    conf=args.conf, classes=classes)
    index = SnapshotIndex(args.snapshots, embedder=embedder)
    n_new = index.refresh()

    out = {"query": args.query, "embedder_id": index.embedder_id,
           "index_size": len(index), "newly_embedded": n_new, "results": []}
    for q in queries:
        snap = index.search(q, top_n=args.top, min_sim=args.min_sim)
        reg = ([] if args.no_registry else
               search_registry(q, db_path=args.db, embedder=embedder,
                               top_n=args.top, min_sim=args.min_sim))
        out["results"].append({
            "query_object": q.to_public(),
            "snapshots": [m.to_public() for m in snap],
            "registry": [m.to_public() for m in reg],
        })

    if args.json:
        print(json.dumps(out, indent=2))
        return

    print(f"\nembedder: {out['embedder_id']}   snapshot index: "
          f"{out['index_size']} crops ({n_new} newly embedded)")
    for res in out["results"]:
        qo = res["query_object"]
        label = (f"{qo['cls']} conf {qo.get('conf')}" if qo["cls"] != "image"
                 else "whole image (no detection)")
        print(f"\n=== query object: {label} ===")
        if not res["snapshots"]:
            print("  snapshots: nothing above min similarity")
        for m in res["snapshots"]:
            tag = "MATCH  " if m["strong"] else "similar"
            print(f"  {tag} {m['similarity']:.3f}  [{m['cls']}] {m['path']}")
        for m in res["registry"]:
            tag = "MATCH  " if m["strong"] else "similar"
            print(f"  {tag} {m['similarity']:.3f}  registry entity "
                  f"#{m['entity_id']} [{m['cls']}] cam={m['cam_id']} "
                  f"sightings={m['sightings']} last={m['last_seen']}")


if __name__ == "__main__":
    main()
