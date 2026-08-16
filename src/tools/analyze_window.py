"""Deep-window behavior analysis from the terminal.

Grabs a longer frame window from ONE catalog camera, threads detections
into per-individual tracks (position + motion - the signal that separates
look-alike objects), and prints each individual's behavior profile. The
annotated trail image + full JSON land under web/snapshots/behavior/.

Usage (from src/):
    python -m tools.analyze_window --cam taksim_meydani
    python -m tools.analyze_window --cam taksim_meydani --frames 16 --stride 10
    python -m tools.analyze_window --cam taksim_meydani --json-only

This costs `--frames` inferences on one camera - an operator action, not
something the collector loop ever does on its own.
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Per-individual behavior profile over a deep window")
    ap.add_argument("--cam", required=True, help="catalog cam_id (cameras.py)")
    ap.add_argument("--frames", type=int, default=12,
                    help="frames in the window (default 12)")
    ap.add_argument("--stride", type=int, default=12,
                    help="source frames between grabs (~0.5s at 25fps)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--weights", default="yolov8s.pt",
                    help="YOLO weights (default yolov8s.pt)")
    # fix 3: the panel-only extras are CLI-reachable too.
    ap.add_argument("--pose", action="store_true",
                    help="per-crop skeleton pass (posture labels + gestures)")
    ap.add_argument("--faces", action="store_true",
                    help="face-detection boxes on the final frame")
    ap.add_argument("--lock", default=None, metavar="auto|<track id>",
                    help="target-lock crosshair on one individual")
    ap.add_argument("--json-only", action="store_true",
                    help="print raw JSON instead of the table")
    args = ap.parse_args(argv)

    from app.behavior import analyze_window
    from app.detect_core import load_model

    model = load_model(args.weights)
    try:
        result = analyze_window(args.cam, model, n_frames=args.frames,
                                stride=args.stride, imgsz=args.imgsz,
                                pose=args.pose, want_faces=args.faces,
                                lock=args.lock)
    except (ValueError, RuntimeError) as e:
        print(f"analyze failed: {e}", file=sys.stderr)
        return 1

    if args.json_only:
        print(json.dumps(result, indent=1))
        return 0

    print(f"{result['cam_name']} ({result['cam_id']}): "
          f"{result['individuals']} individual(s) over "
          f"{result['window_sec']}s / {result['frames']} frames - "
          f"{result['moving']} moving, {result['stationary']} stationary")
    if args.pose:
        print(f"pose: skeletons on {result.get('pose_persons', 0)} "
              f"person sighting(s)")
    if args.faces:
        print(f"faces: {result.get('faces', 0)} detected"
              + ("" if result.get("faces_available")
                 else " (face model not available)"))
    if result.get("lock"):
        lk = result["lock"]
        print(f"target lock: #{lk['track_id']} at ({lk['cx']}, {lk['cy']}) "
              f"offset ({lk['dx']}, {lk['dy']})")
    if result.get("image_url"):
        print(f"trails image: web{result['image_url']}")
    hdr = (f"{'id':>3} {'cls':<10} {'seen':>4} {'move%':>5} {'dir':<10} "
           f"{'px/s':>6} {'km/h':>5} {'nn_px':>6} zones")
    print(hdr)
    print("-" * len(hdr))
    for t in result["tracks"]:
        print(f"{t['id']:>3} {t['cls'] or '?':<10} {t['sightings']:>4} "
              f"{int(t['moving_frac'] * 100):>4}% "
              f"{t['direction'] or ('parked' if t['stationary'] else '-'):<10} "
              f"{t['mean_speed_px_s']:>6.1f} "
              f"{t['kmh_est'] if t['kmh_est'] is not None else '-':>5} "
              f"{t['nn_min_px'] if t['nn_min_px'] is not None else '-':>6} "
              f"{len(t['zones'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
