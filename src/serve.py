"""One-shot launcher for the local live-analysis dashboard.

    python serve.py            # serve web/ on http://localhost:8000 and open the browser
    python serve.py --port 8765
    python serve.py --no-browser

The dashboard at web/index.html is a static page served with a small
threaded HTTP handler. It talks to the local /api/* endpoints (analysis,
uploads, model info, gallery) provided by
app/dashboard_server.py - no cloud, no Firestore, no external SDK.

Pick a camera in the header dropdown (or upload an MP4/MKV via the upload
button), then click Start. The live tile renders full-width; click the
advanced-analysis icon to pick one of the 10 layers.
"""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

from app.dashboard_server import WEB_DIR, bind, port_is_free


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8000, help="port to serve on (default 8000)")
    ap.add_argument("--no-browser", action="store_true", help="do not open the default browser")
    args = ap.parse_args()

    if not WEB_DIR.is_dir():
        sys.exit(f"web/ folder not found at {WEB_DIR}")

    port = args.port
    if not port_is_free(port):
        for candidate in range(port + 1, port + 21):
            if port_is_free(candidate):
                print(f"Port {port} busy; falling back to {candidate}.")
                port = candidate
                break
        else:
            sys.exit(f"Port {port} is busy and no nearby port is free. Use --port to pick one.")

    url = f"http://localhost:{port}/"
    server = bind(port)
    print(f"Serving {WEB_DIR} at {url}")
    print("Routes: /                     -> web/index.html (single-camera live dashboard)")
    print("        /api/upload-video     -> POST an MP4/MKV/MOV/AVI/WEBM to analyze")
    print("        /api/uploaded-videos  -> list your uploaded files")
    print("        /api/analysis/*       -> start/stop/data for the 10-layer advanced analysis")
    print("        /snapshots            -> local snapshots (saved detections, review crops)\n")
    print("Press Ctrl+C to stop.\n")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url, new=2)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
