"""No-op anomaly-crops extractor - the legacy full extractor was cut with
Category B (visual_search + Review system) but dashboard_server.py still
tries to import `refresh` on the first `/api/search` request. Keeping a
silent stub lets that request return an empty result without a 500.

If the extractor is ever reintroduced, replace this file with the real
implementation - the calling site takes any exception silently, so a
richer signature is fine as long as `refresh(model, embedder, snapshots_dir)`
stays callable.
"""
from __future__ import annotations


def refresh(model=None, embedder=None, snapshots_dir=None) -> dict:
    """Return an empty summary. Never raises."""
    return {"scanned": 0, "written": 0, "note": "anomaly-crops extractor removed with Category B"}


def usage_stats(snapshots_dir=None) -> dict:
    """Empty stats matching the old contract - used by
    /api/anomaly-crops-stats. Returns zeros so the UI badge shows 0/0
    instead of crashing with 500."""
    return {"count": 0, "bytes": 0, "oldest": None, "newest": None,
            "note": "anomaly-crops removed with Category B"}


def clear_all(snapshots_dir=None) -> dict:
    """No-op for /api/anomaly-crops-clear - nothing to clear."""
    return {"ok": True, "removed": 0,
            "note": "anomaly-crops removed with Category B"}
