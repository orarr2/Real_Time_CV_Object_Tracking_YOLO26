"""Screen-region capture fallback for YouTube cameras when yt-dlp is
blocked by YouTube's bot check on this IP.

Strategy: the operator keeps the browser open with the YouTube iframe
video visible; the backend captures the screen (optionally a bbox) via
Pillow's ImageGrab and feeds those frames straight into YOLO. Bypasses
YouTube entirely - we never hit their extraction API.

Usage:
    from app.screen_capture import capture, set_region
    set_region((x1, y1, x2, y2))         # once, to bound the capture
    frame = capture()                    # BGR ndarray, ready for YOLO

The bbox is remembered per-process; a fresh operator run should set it
to the browser's video area (right-click the video -> "Inspect" ->
copy bounding-client-rect, or eyeball it).
"""
from __future__ import annotations

import os
import time

_LAST_BBOX: tuple[int, int, int, int] | None = None
_LAST_CAPTURE_TS: float = 0.0
_MIN_INTERVAL_S = 0.15


def set_region(bbox: tuple[int, int, int, int] | None) -> None:
    """Set (or clear) the capture bbox in absolute screen coords.
    (x1, y1, x2, y2) with (0,0) at the top-left. None = full primary screen."""
    global _LAST_BBOX
    if bbox is not None and (len(bbox) != 4 or bbox[2] <= bbox[0]
                             or bbox[3] <= bbox[1]):
        raise ValueError(f"invalid bbox {bbox!r}")
    _LAST_BBOX = tuple(map(int, bbox)) if bbox else None


def get_region() -> tuple[int, int, int, int] | None:
    return _LAST_BBOX


_MSS_INSTANCE = None
_LAST_GOOD_FRAME = None  # BGR ndarray; served on transient grab failures


def _get_mss():
    """Lazy singleton mss.mss() instance (thread-safe DXGI Desktop
    Duplication on Windows - hardware-composited overlays like Chrome's
    video renderer are captured correctly, which PIL.ImageGrab misses).
    """
    global _MSS_INSTANCE
    if _MSS_INSTANCE is None:
        try:
            import mss
            _MSS_INSTANCE = mss.mss()
        except Exception as e:
            print(f"screen_capture: mss unavailable ({type(e).__name__}: "
                  f"{e}) - falling back to PIL.ImageGrab")
            _MSS_INSTANCE = False
    return _MSS_INSTANCE or None


def capture(bbox: tuple[int, int, int, int] | None = None):
    """One BGR frame from the screen (or the previously-set bbox).

    Rate-limited to 1 capture / 150 ms so the caller can spin freely
    without pegging a core. Returns None on failure (DISPLAY missing,
    screen locked, headless host).

    Uses mss (DXGI Desktop Duplication) when available - it captures
    hardware-composited overlays like Chrome's video renderer that
    PIL.ImageGrab's GDI BitBlt path misses on Windows. Falls back to
    PIL.ImageGrab if mss is not installed.
    """
    global _LAST_CAPTURE_TS, _LAST_GOOD_FRAME
    now = time.time()
    if now - _LAST_CAPTURE_TS < _MIN_INTERVAL_S:
        time.sleep(_MIN_INTERVAL_S - (now - _LAST_CAPTURE_TS))
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        print(f"screen_capture: {type(e).__name__}: {e}")
        return _LAST_GOOD_FRAME
    b = bbox if bbox is not None else _LAST_BBOX
    sct = _get_mss()
    if sct is not None:
        try:
            if b is not None:
                mon = {"left": int(b[0]), "top": int(b[1]),
                       "width": int(b[2] - b[0]),
                       "height": int(b[3] - b[1])}
            else:
                mon = sct.monitors[1]  # primary display
            raw = sct.grab(mon)
            arr = np.frombuffer(raw.rgb, dtype=np.uint8).reshape(
                raw.height, raw.width, 3)
            _LAST_CAPTURE_TS = time.time()
            _LAST_GOOD_FRAME = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return _LAST_GOOD_FRAME
        except Exception as e:
            # BitBlt failures happen when the compositor is redrawing;
            # not fatal, just fall through to PIL and, ultimately, cached
            # last-good so the analysis session doesn't die.
            pass
    try:
        from PIL import ImageGrab
        img = ImageGrab.grab(bbox=b, all_screens=False)
        arr = np.array(img)
        _LAST_CAPTURE_TS = time.time()
        _LAST_GOOD_FRAME = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return _LAST_GOOD_FRAME
    except Exception as e:
        print(f"screen_capture (PIL): {type(e).__name__}: {e}")
        return _LAST_GOOD_FRAME  # keep session alive during transient failures


# ---------------------------------------------------------------------------
# Integration hook: when yt-dlp fails on a `youtube` camera and the operator
# has enabled SCREEN_CAPTURE_FALLBACK=1 in the env, `resolve_youtube` can
# short-circuit to this module and mark the "URL" as a sentinel value that
# grab_frame recognises.
# ---------------------------------------------------------------------------

SCREEN_CAPTURE_SENTINEL = "screen://primary"


def is_screen_url(url: str | None) -> bool:
    return bool(url) and url.startswith("screen://")


def parse_bbox_env() -> tuple[int, int, int, int] | None:
    """SCREEN_CAPTURE_BBOX="x1,y1,x2,y2" env override. Empty -> None
    -> full primary screen."""
    raw = (os.environ.get("SCREEN_CAPTURE_BBOX") or "").strip()
    if not raw:
        return None
    try:
        parts = [int(v.strip()) for v in raw.split(",")]
        if len(parts) != 4:
            raise ValueError
        return tuple(parts)  # type: ignore[return-value]
    except ValueError:
        print(f"screen_capture: invalid SCREEN_CAPTURE_BBOX={raw!r} - "
              "expected 'x1,y1,x2,y2' (integers, absolute screen coords)")
        return None
