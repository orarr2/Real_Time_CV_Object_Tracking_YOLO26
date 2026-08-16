"""Capture a frame from a camera with a normalized-coordinate grid overlay.

The ROI / line / loiter zones in app/cameras.py use normalized 0..1
coordinates. This tool grabs a live frame, draws a labeled 10x10 grid on it,
and saves it as <cam_id>_grid.jpg - open the image, read the polygon corner
coordinates off the grid, and paste them into the camera's config:

    python -m tools.roi_grid konya_hukumet
    # -> writes konya_hukumet_grid.jpg next to where you ran it

Requires the camera's stream to resolve from this network.
"""
from __future__ import annotations

import argparse

import cv2

from app.cameras import CAMERAS
from app.detect_core import grab_frame, resolve_stream


def draw_grid(frame, steps: int = 10):
    H, W = frame.shape[:2]
    out = frame.copy()
    for i in range(1, steps):
        x = int(W * i / steps)
        y = int(H * i / steps)
        cv2.line(out, (x, 0), (x, H), (0, 255, 255), 1)
        cv2.line(out, (0, y), (W, y), (0, 255, 255), 1)
        cv2.putText(out, f"{i/steps:.1f}", (x + 3, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, f"{i/steps:.1f}", (3, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cam_id", help="camera id from app/cameras.py")
    ap.add_argument("--out", default=None, help="output jpg (default <cam>_grid.jpg)")
    args = ap.parse_args()

    cam = CAMERAS.get(args.cam_id)
    if cam is None:
        print(f"unknown cam_id {args.cam_id!r}; known: {', '.join(CAMERAS)}")
        return 2
    frame = grab_frame(resolve_stream(cam))
    if frame is None:
        print("could not grab a frame (stream down or geo-blocked on this network)")
        return 1
    out = args.out or f"{args.cam_id}_grid.jpg"
    cv2.imwrite(out, draw_grid(frame))
    print(f"wrote {out}  ({frame.shape[1]}x{frame.shape[0]}) - read the "
          f"normalized coordinates off the yellow grid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
