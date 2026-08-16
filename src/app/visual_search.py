"""Search-by-example: the user uploads a photo of WHAT they are looking for
("this person", "a car like this") and we rank everything the system has seen
by visual similarity to it.

Pipeline (all pieces reuse the existing re-ID machinery so the query image is
scored with exactly the same signature the collector stores):

  1. extract_query_objects() - YOLO on the uploaded image -> one query object
     per detected person/vehicle. If nothing is detected (or ultralytics isn't
     available) the WHOLE image becomes the query, so a tight user-cropped
     photo still works.
  2. Each query crop is embedded with the same pluggable embedder family the
     registry uses (HSV histogram by default, OSNet when --reid-model is set).
  3. The query embedding is matched against two sources:
       * the re-ID registry (data/reid.db): "has an entity that looks like
         this been seen - where, when, how many times". Only consulted when
         the registry was built by the SAME embedder_id (vectors from
         different embedders are not comparable - same rule ReidStore
         enforces on itself).
       * saved snapshot crops under web/snapshots/ (returning/ + events/):
         actual viewable images, ranked by cosine similarity. Embeddings are
         cached next to the snapshots so repeated searches don't re-embed.

Similarity semantics: scores are cosine on L2-normalized vectors (higher is
more similar). A match is tagged `strong` when it clears the embedder's own
matching threshold - the same bar the collector uses to say "same entity".
Below that the ranking is still useful ("most similar things we have"), it
just isn't an identity claim.

NOTE the honest limitation inherited from the histogram embedder: it is a
color/shape signature, not semantic similarity. It finds "the same-looking
object again", especially under similar lighting; it does not understand
"a man with a hat" in the abstract. Plug an OSNet ONNX model (see
reid_embed.py) for lighting/pose-robust matching.
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.detect_core import CLASSES_OF_INTEREST, DEFAULT_IMGSZ, draw_boxes
from app.reid_embed import make_embedder

# Where the collector's local-mode snapshots live (crops we can search AND
# serve back to the browser as /snapshots/... urls).
_SRC_ROOT      = Path(__file__).resolve().parent.parent
SNAPSHOTS_ROOT = _SRC_ROOT / "web" / "snapshots"
DEFAULT_DB     = _SRC_ROOT / "data" / "reid.db"

# Snapshot subtrees that contain per-object CROPS. `anomalies/` holds full
# annotated frames (whole street), so it can't be matched against an
# object-shaped query directly - instead ``app.anomaly_crops`` pre-extracts
# per-object crops from every anomaly frame into ``anomalies_crops/`` (with
# an LRU size cap and de-dup), and that sibling directory joins the index.
# `live_samples/` is the collector's continuous pool: one random detection
# crop per LIVE_SAMPLE_EVERY_N bursts per camera, so the review UI has
# fresh material even on cameras that never fire returning / events /
# anomalies. See app.live_samples for the writer side.
# `review_crops/` is the per-object extraction of the review_frames pool
# (see app.frame_crops) - the frames the collector captured in the past,
# finally searchable. Its crops OUTLIVE the 100-frame LRU of the source
# pool, so this subtree is where most of the searchable history lives.
CROP_SUBDIRS = ("returning", "events", "anomalies_crops", "live_samples",
                "review_crops")
ANOMALY_CROPS_SUBDIR = "anomalies_crops"


def _is_from_anomaly(rel: str) -> bool:
    """True when the crop's rel path is under the anomaly-crops subtree.
    Used by the review UI to decide whether to prompt "was this really an
    anomaly?" alongside the label verdict."""
    return rel.startswith(ANOMALY_CROPS_SUBDIR + "/")

# Results below this cosine are noise for every embedder we ship; the
# per-embedder "strong" threshold sits far above it.
#
# Raised from 0.30 to 0.55 as part of the search-hardening pass. At 0.30
# the HSV histogram embedder returned mid-tier color-similar crops that
# were often not semantically the same object (a red car vs a red banner).
# At 0.55 only crops with meaningfully overlapping content clear the bar;
# the "similar" tier stays useful, the "match" tier stays honest.
# Callers who want the old floor can still pass min_sim=0.30 explicitly.
MIN_SIMILARITY_FLOOR = 0.55


def _iso(mtime: float) -> str:
    """Epoch seconds -> UTC ISO-8601. Kept short so it fits result cards."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(mtime))


@dataclass
class QueryObject:
    """One thing the user is searching for, extracted from their upload."""
    cls: str                     # 'person' | 'car' | ... | 'image' (whole-image fallback)
    embedding: np.ndarray
    box: dict | None = None     # {x1,y1,x2,y2,conf} on the uploaded image, None for fallback
    crop_bgr: np.ndarray | None = None

    def to_public(self) -> dict:
        d = {"cls": self.cls}
        if self.box:
            d["box"] = {k: round(float(self.box[k]), 1) for k in ("x1", "y1", "x2", "y2")}
            d["conf"] = round(float(self.box.get("conf", 0.0)), 3)
        return d


@dataclass
class Match:
    source: str                  # 'snapshot' | 'registry'
    similarity: float
    cls: str
    strong: bool                 # cleared the embedder's identity threshold
    query_cls: str
    extra: dict = field(default_factory=dict)   # url/path or entity metadata

    def to_public(self) -> dict:
        return {"source": self.source, "similarity": round(self.similarity, 4),
                "cls": self.cls, "strong": self.strong,
                "query_cls": self.query_cls, **self.extra}


def _embed_cls_for(cls: str) -> str:
    """Map a query/candidate class onto the embedder's crop geometry."""
    return "person" if cls in ("person", "image") else "vehicle"


import re as _re

_EXPECTED_DIM_RE = _re.compile(r"Expected:\s*(\d+)")


def _detect(model, image_bgr: np.ndarray, conf: float,
            imgsz: int | None) -> list[dict]:
    """detect_with_boxes with a static-input fallback: an ONNX/OpenVINO export
    with a fixed input size rejects any other imgsz - retry at the size the
    runtime says it expects (ultralytics caches imgsz in its predictor, so the
    retry must pass the native size EXPLICITLY, not just omit the kwarg).

    Runs with the collector's per-class confidence + person plausibility
    filters (see DEFAULT_PER_CLASS_CONF) so a user uploading a photo of a
    rider gets the same "person + motorcycle" pair the collector would
    record, and a stroller in a query photo doesn't fan out into a bogus
    'person' search."""
    from app.detect_core import detect_with_boxes, DEFAULT_PER_CLASS_CONF
    try:
        _, boxes = detect_with_boxes(model, image_bgr, conf=conf, imgsz=imgsz,
                                     per_class_conf=DEFAULT_PER_CLASS_CONF)
    except Exception as e:
        m = _EXPECTED_DIM_RE.search(str(e))
        native = int(m.group(1)) if m else 640
        if imgsz == native:
            raise
        _, boxes = detect_with_boxes(model, image_bgr, conf=conf, imgsz=native,
                                     per_class_conf=DEFAULT_PER_CLASS_CONF)
    return boxes


# ---- 1. query extraction ----------------------------------------------------

def extract_query_objects(image_bgr: np.ndarray, model=None, embedder=None,
                          conf: float = 0.30, imgsz: int | None = DEFAULT_IMGSZ,
                          classes: set[str] | None = None,
                          max_objects: int = 8) -> list[QueryObject]:
    """Turn an uploaded image into embedded query objects.

    With a YOLO `model`: each detected person/vehicle becomes a query object
    (largest first, capped at `max_objects` so a crowded photo doesn't fan out
    into dozens of searches). Without a model, or when nothing is detected,
    the whole image is embedded as a single fallback query - the right
    behavior for the common case of a user uploading an already-cropped
    picture of their target.
    """
    if embedder is None:
        embedder = make_embedder()
    objects: list[QueryObject] = []
    if model is not None:
        boxes = _detect(model, image_bgr, conf=conf, imgsz=imgsz)
        if classes:
            boxes = [b for b in boxes if b["cls"] in classes]
        boxes.sort(key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]),
                   reverse=True)
        H, W = image_bgr.shape[:2]
        for b in boxes[:max_objects]:
            x1 = max(0, int(b["x1"])); y1 = max(0, int(b["y1"]))
            x2 = min(W, int(b["x2"])); y2 = min(H, int(b["y2"]))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image_bgr[y1:y2, x1:x2]
            emb = embedder.embed(crop, _embed_cls_for(b["cls"]))
            if emb is None:
                continue
            objects.append(QueryObject(cls=b["cls"], embedding=emb, box=b,
                                       crop_bgr=crop))
    if not objects:
        emb = embedder.embed(image_bgr, "person")
        if emb is not None:
            objects.append(QueryObject(cls="image", embedding=emb,
                                       crop_bgr=image_bgr))
    return objects


# ---- 2. snapshot index -------------------------------------------------------

class SnapshotIndex:
    """Embeddings for every saved crop under web/snapshots/{returning,events}.

    The cache file (.search_cache.json inside the snapshots root) is keyed by
    (relative path, mtime, embedder_id): refresh() only embeds new/changed
    crops, and a cache produced by a different embedder is discarded outright
    - the exact invalidation rule the re-ID registry applies to itself.
    """

    CACHE_NAME = ".search_cache.json"

    def __init__(self, root: str | Path = SNAPSHOTS_ROOT, embedder=None):
        self.root = Path(root)
        self.embedder = embedder if embedder is not None else make_embedder()
        self.embedder_id = getattr(self.embedder, "embedder_id", "unknown")
        # rel_path -> {"mtime": float, "cls": str, "vec": np.ndarray}
        self._entries: dict[str, dict] = {}
        self._load_cache()

    # -- crop discovery --

    def _iter_crop_files(self):
        for sub in CROP_SUBDIRS:
            base = self.root / sub
            if not base.is_dir():
                continue
            for p in sorted(base.rglob("*.jpg")):
                if p.name.endswith("_full.jpg"):   # full frames aren't objects
                    continue
                yield p

    def _manifest_cls(self) -> dict[str, str]:
        """rel_path -> cls. Reads three manifest formats:
        * ``manifest.json`` under ``returning/`` (list of items with
          crop_url + cls);
        * ``.anomaly_crops.json`` under ``anomalies_crops/`` written by
          ``app.anomaly_crops`` (dict keyed by rel_crop_path);
        * ``.review_crops.json`` under ``review_crops/`` written by
          ``app.frame_crops`` (same schema as the anomaly one)."""
        out: dict[str, str] = {}
        # returning/ format
        for manifest in self.root.rglob("manifest.json"):
            try:
                items = json.loads(manifest.read_text())
            except Exception:
                continue
            for it in items if isinstance(items, list) else []:
                url = it.get("crop_url") or ""
                cls = it.get("cls")
                if url.startswith("/snapshots/") and cls:
                    out[url[len("/snapshots/"):]] = cls
        # crops-map format (anomaly_crops + frame_crops share the schema)
        for name in (".anomaly_crops.json", ".review_crops.json"):
            for manifest in self.root.rglob(name):
                try:
                    data = json.loads(manifest.read_text())
                except Exception:
                    continue
                for rel, meta in (data.get("crops") or {}).items():
                    cls = meta.get("cls")
                    if cls:
                        out[rel] = cls
        return out

    @staticmethod
    def _guess_cls(path: Path, crop_shape) -> str:
        """Fallback class when no manifest covers the crop: filename hints,
        then aspect (people are taller than wide on street cams)."""
        name = path.name.lower()
        for cand in CLASSES_OF_INTEREST:
            if cand in name:
                return cand
        h, w = crop_shape[:2]
        return "person" if h >= 1.2 * w else "car"

    # -- cache --

    def _cache_path(self) -> Path:
        return self.root / self.CACHE_NAME

    def _load_cache(self) -> None:
        try:
            data = json.loads(self._cache_path().read_text())
        except Exception:
            return
        if data.get("embedder_id") != self.embedder_id:
            return   # different signature space - rebuild from scratch
        for rel, e in (data.get("entries") or {}).items():
            try:
                vec = np.asarray(e["vec"], dtype=np.float32)
                self._entries[rel] = {"mtime": float(e["mtime"]),
                                      "cls": str(e["cls"]), "vec": vec}
            except (KeyError, TypeError, ValueError):
                continue

    def _save_cache(self) -> None:
        payload = {
            "embedder_id": self.embedder_id,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "entries": {rel: {"mtime": e["mtime"], "cls": e["cls"],
                              "vec": [round(float(v), 6) for v in e["vec"]]}
                        for rel, e in self._entries.items()},
        }
        try:
            self._cache_path().write_text(json.dumps(payload))
        except OSError:
            pass   # read-only snapshots dir: the index still works, just uncached

    # -- public API --

    def refresh(self) -> int:
        """Sync the index with the snapshot tree. Returns #crops (re)embedded."""
        manifest_cls = self._manifest_cls()
        seen: set[str] = set()
        embedded = 0
        for p in self._iter_crop_files():
            rel = str(p.relative_to(self.root)).replace("\\", "/")
            seen.add(rel)
            mtime = p.stat().st_mtime
            cached = self._entries.get(rel)
            if cached is not None and cached["mtime"] == mtime:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            cls = manifest_cls.get(rel) or self._guess_cls(p, img.shape)
            vec = self.embedder.embed(img, _embed_cls_for(cls))
            if vec is None:
                continue
            self._entries[rel] = {"mtime": mtime, "cls": cls, "vec": vec}
            embedded += 1
        removed = set(self._entries) - seen
        for rel in removed:
            del self._entries[rel]
        if embedded or removed:
            self._save_cache()
        return embedded

    def __len__(self) -> int:
        return len(self._entries)

    def search(self, query: QueryObject, top_n: int = 12,
               min_sim: float = MIN_SIMILARITY_FLOOR,
               same_class_only: bool = True,
               classes: set[str] | None = None,
               time_from: float | None = None,
               time_to: float | None = None) -> list[Match]:
        """Rank indexed crops by cosine similarity to one query object.

        `classes` (optional set of class names) filters candidates to those
        classes regardless of the query object's own class - lets a
        multi-class chip selector ("person or car") match either class
        against a single upload. `time_from`/`time_to` (epoch seconds; None
        = open bound) restrict candidates by file mtime.
        """
        strong_at = getattr(self.embedder, "default_threshold", 0.9)
        scored: list[Match] = []
        for rel, e in self._entries.items():
            if classes and e["cls"] not in classes:
                continue
            if (same_class_only and not classes and query.cls != "image"
                    and e["cls"] != query.cls):
                continue
            if time_from is not None and e["mtime"] < time_from:
                continue
            if time_to is not None and e["mtime"] > time_to:
                continue
            if e["vec"].shape != query.embedding.shape:
                continue
            sim = float(np.dot(e["vec"], query.embedding))
            if sim < min_sim:
                continue
            scored.append(Match(
                source="snapshot", similarity=sim, cls=e["cls"],
                strong=sim >= strong_at, query_cls=query.cls,
                extra={"url": f"/snapshots/{rel}", "path": rel,
                       "seen_at": _iso(e["mtime"]),
                       "from_anomaly": _is_from_anomaly(rel)}))
        scored.sort(key=lambda m: m.similarity, reverse=True)
        return scored[:top_n]

    def browse(self, classes: set[str] | None = None,
               time_from: float | None = None,
               time_to: float | None = None,
               limit: int = 24,
               order: str = "time_desc") -> list[Match]:
        """Return crops matching filters WITHOUT a query image.

        The counterpart to `search()` for the "just show me the last three
        cars between 18:00 and 22:00" case. Ordering is `time_desc` (most
        recent first) or `time_asc`. Similarity is 0 (not defined for
        browse), and each match carries `seen_at` (ISO-8601 UTC).
        """
        cands: list[tuple[float, str, dict]] = []
        for rel, e in self._entries.items():
            if classes and e["cls"] not in classes:
                continue
            if time_from is not None and e["mtime"] < time_from:
                continue
            if time_to is not None and e["mtime"] > time_to:
                continue
            cands.append((e["mtime"], rel, e))
        cands.sort(key=lambda t: t[0], reverse=(order != "time_asc"))
        out: list[Match] = []
        for mtime, rel, e in cands[:limit]:
            out.append(Match(
                source="snapshot", similarity=0.0, cls=e["cls"],
                strong=False, query_cls="browse",
                extra={"url": f"/snapshots/{rel}", "path": rel,
                       "seen_at": _iso(mtime), "browse_only": True,
                       "from_anomaly": _is_from_anomaly(rel)}))
        return out


# ---- 3. registry search -------------------------------------------------------

def search_registry(query: QueryObject, db_path: str | Path = DEFAULT_DB,
                    embedder=None, top_n: int = 12,
                    min_sim: float = MIN_SIMILARITY_FLOOR,
                    cam_id: str | None = None) -> list[Match]:
    """Rank re-ID registry entities against one query object.

    Read-only: unlike ReidStore.query() this never inserts or updates - the
    user browsing for something must not pollute the appearance memory.
    Returns [] when the DB is missing or was built by a different embedder.
    """
    import sqlite3
    path = Path(db_path)
    if not path.is_file():
        return []
    if embedder is None:
        embedder = make_embedder()
    eid = getattr(embedder, "embedder_id", "unknown")
    strong_at = getattr(embedder, "default_threshold", 0.9)
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key='embedder_id'").fetchone()
        if row and row[0] != eid:
            return []   # incomparable vector spaces
        where, args = "", []
        if query.cls != "image":
            where = "WHERE cls=?"
            args.append(query.cls)
        if cam_id:
            where = (where + " AND cam_id=?") if where else "WHERE cam_id=?"
            args.append(cam_id)
        cur = conn.execute(
            f"SELECT entity_id, cam_id, cls, first_seen, last_seen, sightings, "
            f"embedding FROM entities {where}", args)
        scored: list[Match] = []
        for ent_id, cam, cls, first, last, sightings, blob in cur:
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.shape != query.embedding.shape:
                continue
            sim = float(np.dot(vec, query.embedding))
            if sim < min_sim:
                continue
            scored.append(Match(
                source="registry", similarity=sim, cls=cls,
                strong=sim >= strong_at, query_cls=query.cls,
                extra={"entity_id": ent_id, "cam_id": cam,
                       "first_seen": first, "last_seen": last,
                       "sightings": sightings}))
        scored.sort(key=lambda m: m.similarity, reverse=True)
        return scored[:top_n]
    except sqlite3.DatabaseError:
        return []
    finally:
        conn.close()


# ---- 4. the one-call entry point ----------------------------------------------

def search_image(image_bgr: np.ndarray, *, model=None, embedder=None,
                 snapshots_root: str | Path = SNAPSHOTS_ROOT,
                 db_path: str | Path = DEFAULT_DB,
                 snapshot_index: SnapshotIndex | None = None,
                 top_n: int = 12, min_sim: float = MIN_SIMILARITY_FLOOR,
                 classes: set[str] | None = None,
                 time_from: float | None = None,
                 time_to: float | None = None,
                 include_preview: bool = True) -> dict:
    """Full pipeline for one uploaded image. Returns a JSON-serializable dict:

        {"mode": "image", "query_objects": [...], "snapshot_matches": [...],
         "registry_matches": [...], "index_size": int,
         "query_preview_jpeg_b64": "..."}    # detections drawn on the upload

    `snapshot_index` lets a server reuse one warm index across requests.
    `classes` restricts BOTH the query-object extraction (which classes to
    pull from the upload) AND the candidate filter on the snapshot index.
    `time_from`/`time_to` (epoch seconds) restrict snapshot candidates by
    file mtime.
    """
    if embedder is None:
        embedder = make_embedder()
    queries = extract_query_objects(image_bgr, model=model, embedder=embedder,
                                    classes=classes)
    idx = snapshot_index
    if idx is None:
        idx = SnapshotIndex(snapshots_root, embedder=embedder)
    idx.refresh()

    snapshot_matches: list[dict] = []
    registry_matches: list[dict] = []
    seen_snap: set[str] = set()
    seen_ent: set[int] = set()
    for q in queries:
        # Rail queries relax the same-class rule: the street pools contain
        # nearly zero `train` crops (no rail camera feeds them), so a metro
        # photo matched same-class-only returned nothing BY CONSTRUCTION.
        # Relaxed, it at least surfaces the bus/large-vehicle lookalikes the
        # pool does hold - an honest "closest things the system has seen".
        relax_cls = (q.cls == "train" and not classes)
        for m in idx.search(q, top_n=top_n, min_sim=min_sim,
                            same_class_only=not relax_cls,
                            classes=classes,
                            time_from=time_from, time_to=time_to):
            if m.extra["path"] in seen_snap:
                continue
            seen_snap.add(m.extra["path"])
            snapshot_matches.append(m.to_public())
        for m in search_registry(q, db_path=db_path, embedder=embedder,
                                 top_n=top_n, min_sim=min_sim):
            if m.extra["entity_id"] in seen_ent:
                continue
            seen_ent.add(m.extra["entity_id"])
            registry_matches.append(m.to_public())
    snapshot_matches.sort(key=lambda m: m["similarity"], reverse=True)
    registry_matches.sort(key=lambda m: m["similarity"], reverse=True)

    out = {
        "mode": "image",
        "embedder_id": getattr(embedder, "embedder_id", "unknown"),
        "strong_threshold": getattr(embedder, "default_threshold", None),
        "query_objects": [q.to_public() for q in queries],
        "snapshot_matches": snapshot_matches[:top_n],
        "registry_matches": registry_matches[:top_n],
        "index_size": len(idx),
    }
    if include_preview:
        boxes = [q.box for q in queries if q.box]
        preview = draw_boxes(image_bgr, boxes) if boxes else image_bgr
        ok, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            out["query_preview_jpeg_b64"] = base64.b64encode(
                buf.tobytes()).decode("ascii")
    return out


def search_image_bytes(data: bytes, **kwargs) -> dict:
    """Decode an uploaded image (any cv2-supported format) and search it."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode the uploaded image")
    # Bound the working size: phone photos arrive at 4000px+, and the largest
    # useful signal for a 640-960px YOLO pass + a 128px embed crop is far below
    # that. Downscale keeps request latency flat.
    h, w = img.shape[:2]
    if max(h, w) > 1920:
        s = 1920.0 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)),
                         interpolation=cv2.INTER_AREA)
    return search_image(img, **kwargs)


def browse_snapshots(*, embedder=None,
                     snapshots_root: str | Path = SNAPSHOTS_ROOT,
                     snapshot_index: SnapshotIndex | None = None,
                     classes: set[str] | None = None,
                     time_from: float | None = None,
                     time_to: float | None = None,
                     limit: int = 24,
                     order: str = "time_desc") -> dict:
    """Query-less browse: list saved crops matching class/time filters.

    Same result shape as `search_image` (so the frontend renders it with the
    same code path) but with an empty `query_objects` list and every match
    tagged `browse_only=True` in its extras. Used for "show me the three
    most recent cars" without needing a reference photo.
    """
    if embedder is None:
        embedder = make_embedder()
    idx = snapshot_index
    if idx is None:
        idx = SnapshotIndex(snapshots_root, embedder=embedder)
    idx.refresh()
    matches = [m.to_public() for m in idx.browse(
        classes=classes, time_from=time_from, time_to=time_to,
        limit=limit, order=order)]
    return {
        "mode": "browse",
        "embedder_id": getattr(embedder, "embedder_id", "unknown"),
        "query_objects": [],
        "snapshot_matches": matches,
        "registry_matches": [],
        "index_size": len(idx),
    }


# ---- demo/index seeding (testing without a running collector) ------------------

def seed_index_from_images(image_paths, model, *, embedder=None,
                           snapshots_root: str | Path = SNAPSHOTS_ROOT,
                           conf: float = 0.30,
                           imgsz: int | None = DEFAULT_IMGSZ) -> list[dict]:
    """Detect objects in still images and save their crops under
    snapshots/events/demo/<image-stem>/ so they become searchable exactly like
    collector-saved snapshots. Returns one manifest dict per saved crop.

    This exists so search-by-image can be exercised (tools/search_by_image.py
    --seed-images ...) before the collector has accumulated real snapshots.
    """
    if embedder is None:
        embedder = make_embedder()
    root = Path(snapshots_root)
    saved: list[dict] = []
    for path in image_paths:
        p = Path(path)
        img = cv2.imread(str(p))
        if img is None:
            print(f"seed: cannot read {p}, skipping")
            continue
        boxes = _detect(model, img, conf=conf, imgsz=imgsz)
        out_dir = root / "events" / "demo" / p.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        H, W = img.shape[:2]
        n = 0
        for i, b in enumerate(boxes):
            x1 = max(0, int(b["x1"])); y1 = max(0, int(b["y1"]))
            x2 = min(W, int(b["x2"])); y2 = min(H, int(b["y2"]))
            if x2 - x1 < 12 or y2 - y1 < 12:
                continue
            crop = img[y1:y2, x1:x2]
            name = f"obj{i:02d}_{b['cls']}.jpg"
            if not cv2.imwrite(str(out_dir / name), crop,
                               [cv2.IMWRITE_JPEG_QUALITY, 90]):
                continue
            n += 1
            saved.append({"source_image": str(p), "cls": b["cls"],
                          "conf": round(b["conf"], 3),
                          "box": {k: round(b[k], 1) for k in ("x1", "y1", "x2", "y2")},
                          "crop": str(out_dir / name)})
        print(f"seed: {p.name}: {n} crops -> {out_dir}")
    return saved
