"""Rich English PDF report - the visual evidence layer over daily_digest.

The plain-text/HTML digest is fine for a phone glance ("Kulturpark loitering
x14"), but every ambiguous event begs the same question: WHICH crop? Which
frame? WHICH pole did the model call a person? The events collection already
carries a public ``fullframe_url`` per event; this module downloads those
into a paginated PDF - tables for the counts, actual images for the
top-priority scenes - and returns the file so daily_digest can attach it to
the email.

Design decisions:

* Layout - A4 portrait, English throughout, LTR. Section headers left-
  aligned; page title centered.
* Image selection - one representative event per anomaly KIND, priority-
  ordered (obstructed > dark > extreme_load > loiter > returning). Cap at
  8 pictures so the PDF stays under ~1 MB for Gmail's inline preview.
* Failure mode - never raise. A blank snapshot URL, a slow bucket, or a
  PIL error degrades the section (skip the image) but the caller still
  gets a PDF path back.
"""
from __future__ import annotations

import datetime as dt
from io import BytesIO
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

# Kinds in order of "operator wants to see this first" - a blocked camera is
# strictly more urgent than yet another loiter alert. Kinds not listed here
# get appended in first-seen order.
KIND_PRIORITY = (
    "camera_obstructed", "camera_dark", "extreme_load",
    "loiter", "returning",
)

# Cap so the emailed PDF stays inline-previewable on the phone.
MAX_IMAGES = 8
MAX_IMAGE_BYTES = 4 * 1024 * 1024
IMAGE_DOWNLOAD_TIMEOUT_S = 15
# Downscale + re-encode: 8 fullframe HD snapshots at q=75 -> ~600 KB PDF vs
# 3+ MB raw, so Gmail's phone preview still opens instantly.
PDF_IMAGE_MAX_WIDTH_PX = 900
PDF_IMAGE_QUALITY = 75


def pick_event_samples(events: list[dict], max_total: int = MAX_IMAGES
                       ) -> list[dict]:
    """Legacy sample picker: one event per KIND, priority-ordered.
    Kept for the tests that pin its behavior; new callers use
    ``pick_group_samples`` which ties each picked event back to its
    aggregated row so the numbered badge lines up."""
    by_kind: dict[str, list[dict]] = {}
    for e in events:
        if not e.get("snapshot_url") and not e.get("fullframe_url"):
            continue
        by_kind.setdefault(str(e.get("kind") or "?"), []).append(e)
    for group in by_kind.values():
        group.sort(key=lambda e: str(e.get("ts") or ""), reverse=True)

    kinds_ordered = [k for k in KIND_PRIORITY if k in by_kind] + \
                    [k for k in by_kind if k not in KIND_PRIORITY]
    picks: list[dict] = []
    round_idx = 0
    while len(picks) < max_total and kinds_ordered:
        progress = False
        for k in list(kinds_ordered):
            if round_idx < len(by_kind[k]):
                picks.append(by_kind[k][round_idx])
                progress = True
                if len(picks) >= max_total:
                    break
        if not progress:
            break
        round_idx += 1
    return picks


def pick_group_samples(groups: list[dict], max_total: int = MAX_IMAGES
                       ) -> list[dict]:
    """Pick which aggregated ROWS deserve a visual card, priority-ordered.

    The report's promise is: every image in the evidence pages traces back
    to a specific #N row in the anomalies table. We pick groups (not
    events) so the mapping is unambiguous - one image per row, at most.
    Priority ordering (obstructed > dark > extreme > loiter > returning)
    beats volume; ties within a priority level use recency."""
    by_kind: dict[str, list[dict]] = {}
    for g in groups:
        if not g.get("latest_event"):
            continue
        e = g["latest_event"]
        if not e.get("snapshot_url") and not e.get("fullframe_url"):
            continue
        by_kind.setdefault(g["kind"], []).append(g)
    for gs in by_kind.values():
        gs.sort(key=lambda g: str(g["last_ts"]), reverse=True)

    kinds_ordered = [k for k in KIND_PRIORITY if k in by_kind] + \
                    [k for k in by_kind if k not in KIND_PRIORITY]
    picks: list[dict] = []
    round_idx = 0
    while len(picks) < max_total and kinds_ordered:
        progress = False
        for k in list(kinds_ordered):
            if round_idx < len(by_kind[k]):
                picks.append(by_kind[k][round_idx])
                progress = True
                if len(picks) >= max_total:
                    break
        if not progress:
            break
        round_idx += 1
    return picks


def find_first_sighting(bucket_name: str, cam_id: str, entity_id,
                        downloader=None) -> str | None:
    """The oldest entity-gallery image URL for a (cam, entity) pair, from
    the review_sync manifest. Returning-visitor events without this look
    up the current crop only; with it, the report can show 'first seen' +
    'now' side-by-side and the operator can eyeball the identity claim."""
    import json
    import time as _t
    if downloader is None:
        downloader = _http_bytes
    try:
        raw = downloader(
            f"https://storage.googleapis.com/{bucket_name}/"
            f"review_sync/manifest.json?t={int(_t.time())}")
        if not raw:
            return None
        manifest = json.loads(raw.decode("utf-8"))
        prefix = f"entities/{cam_id}/{entity_id}/"
        candidates = [(rel, meta) for rel, meta in
                      (manifest.get("files") or {}).items()
                      if rel.startswith(prefix) and rel.endswith(".jpg")]
        if not candidates:
            return None
        candidates.sort(key=lambda kv: float(kv[1].get("mtime", 0)))
        return candidates[0][1].get("url")
    except Exception:
        return None


def _http_bytes(url: str, timeout: float = IMAGE_DOWNLOAD_TIMEOUT_S
                ) -> bytes | None:
    """Fetch a JPEG. The lower bound was 1024 bytes originally to reject
    broken/empty responses, but valid crops of small subjects (a distant
    person, an entity-gallery portrait) come in around 800-1000 bytes
    JPEG-encoded - a real loiter-person crop at 971 bytes was being
    silently dropped, leaving the evidence card with only the fullframe.
    256 bytes still guards against zero-byte / html-error payloads while
    accepting every real JPEG the collector produces."""
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "digest/2"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        return data if 256 < len(data) < MAX_IMAGE_BYTES else None
    except Exception:
        return None


def _shrink_jpeg(data: bytes,
                 max_width: int = PDF_IMAGE_MAX_WIDTH_PX,
                 quality: int = PDF_IMAGE_QUALITY) -> bytes:
    """Downscale + re-encode. Silently returns the input on any PIL error,
    since a slightly-heavy PDF still ships where a raised exception would
    drop the whole report."""
    try:
        from PIL import Image as PILImage
        im = PILImage.open(BytesIO(data))
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        if im.width > max_width:
            new_h = int(im.height * max_width / im.width)
            im = im.resize((max_width, new_h), PILImage.LANCZOS)
        buf = BytesIO()
        im.save(buf, "JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception:
        return data


def _draw_box_on_frame(data: bytes, box, frame_w: int, frame_h: int,
                       cls: str | None = None) -> bytes:
    """Overlay a red bounding rectangle on the fullframe JPEG so the operator
    can see WHICH object was flagged - the caption alone gave 'someone
    loitered somewhere in this plaza'; the box says 'this specific person'.

    ``box`` is [x1, y1, x2, y2] in the same pixel space as ``frame_w x
    frame_h`` (whatever the collector emitted). We rescale to the actual
    decoded image size, in case the fullframe was resized between capture
    and upload. Returns the original bytes when PIL is missing, when the
    coordinates are unusable, or when any step raises - a labeled image is
    a nicety, an unlabeled one still ships the report."""
    if not box or frame_w <= 0 or frame_h <= 0:
        return data
    try:
        x1, y1, x2, y2 = (float(v) for v in box[:4])
    except (TypeError, ValueError):
        return data
    if x2 <= x1 or y2 <= y1:
        return data
    try:
        from PIL import Image as PILImage, ImageDraw, ImageFont
        im = PILImage.open(BytesIO(data))
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        sx = w / float(frame_w)
        sy = h / float(frame_h)
        rx1, ry1 = int(max(0, x1 * sx)), int(max(0, y1 * sy))
        rx2, ry2 = int(min(w - 1, x2 * sx)), int(min(h - 1, y2 * sy))
        if rx2 <= rx1 or ry2 <= ry1:
            return data
        draw = ImageDraw.Draw(im)
        # Bright red outline, thick enough to stay visible after the
        # 900px downscale (~3-4px absolute at that width).
        thickness = max(3, int(0.005 * max(w, h)))
        for i in range(thickness):
            draw.rectangle([rx1 - i, ry1 - i, rx2 + i, ry2 + i],
                           outline=(220, 38, 38))
        # Class label chip above the box; falls back to no chip if font
        # loading fails (Debian VMs have DejaVu; dev machines vary).
        if cls:
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except (OSError, IOError):
                font = ImageFont.load_default()
            text = str(cls)
            # Backwards-compat with old PIL: use textlength+getmetrics
            # instead of textbbox when necessary.
            try:
                tw = draw.textlength(text, font=font)
                th = font.size + 4
            except AttributeError:
                tw, th = draw.textsize(text, font=font)
            pad = 4
            chip_y0 = max(0, ry1 - th - 2 * pad)
            draw.rectangle([rx1, chip_y0, rx1 + tw + 2 * pad, chip_y0 + th + 2 * pad],
                           fill=(220, 38, 38))
            draw.text((rx1 + pad, chip_y0 + pad), text,
                      fill=(255, 255, 255), font=font)
        buf = BytesIO()
        im.save(buf, "JPEG", quality=PDF_IMAGE_QUALITY)
        return buf.getvalue()
    except Exception:
        return data


def fetch_snapshots(picks: list[dict],
                    downloader=_http_bytes,
                    shrink: bool = True) -> list[tuple[dict, bytes]]:
    """Download each pick's image. Prefer fullframe_url (context), fall back
    to snapshot_url (crop-only). Silently drops picks whose downloads fail."""
    out: list[tuple[dict, bytes]] = []
    for e in picks:
        for key in ("fullframe_url", "snapshot_url"):
            url = e.get(key)
            if not url:
                continue
            data = downloader(url)
            if data:
                out.append((e, _shrink_jpeg(data) if shrink else data))
                break
    return out


def _fmt_ts(ts_iso: str) -> str:
    """UTC ISO -> Israel-time HH:MM (the operator's clock)."""
    try:
        from zoneinfo import ZoneInfo
        t = dt.datetime.strptime(str(ts_iso)[:19], "%Y-%m-%dT%H:%M:%S")
        t = t.replace(tzinfo=dt.timezone.utc).astimezone(ZoneInfo("Asia/Jerusalem"))
        return t.strftime("%H:%M")
    except (ValueError, TypeError):
        return str(ts_iso)[:16]


def _fmt_date(now: dt.datetime) -> str:
    return now.strftime("%d.%m.%Y")


def _table_style() -> TableStyle:
    return TableStyle([
        ("FONTNAME",       (0, 0),  (-1, -1), "Helvetica"),
        ("FONTNAME",       (0, 0),  (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0),  (-1, -1), 10),
        ("BACKGROUND",     (0, 0),  (-1, 0),  colors.HexColor("#0f172a")),
        ("TEXTCOLOR",      (0, 0),  (-1, 0),  colors.white),
        ("ALIGN",          (0, 0),  (-1, 0),  "CENTER"),
        ("ALIGN",          (0, 1),  (0, -1),  "LEFT"),
        ("ALIGN",          (1, 1),  (-1, -1), "CENTER"),
        ("VALIGN",         (0, 0),  (-1, -1), "MIDDLE"),
        ("TOPPADDING",     (0, 0),  (-1, -1), 6),
        ("BOTTOMPADDING",  (0, 0),  (-1, -1), 6),
        ("GRID",           (0, 0),  (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1),  (-1, -1), [colors.white,
                                               colors.HexColor("#f8fafc")]),
    ])


def _event_caption(e: dict, kind_labels: dict[str, str]) -> str:
    """One-line English caption for the image evidence page."""
    kind = str(e.get("kind") or "?")
    label = kind_labels.get(kind, kind)
    cam = str(e.get("cam_name") or e.get("cam_id") or e.get("slot") or "?")
    parts: list[str] = [label, cam, _fmt_ts(e.get("ts"))]
    dur = e.get("duration_sec")
    if isinstance(dur, (int, float)) and dur > 0:
        parts.append(f"{int(dur)} sec")
    cls = e.get("cls")
    if cls:
        parts.append(str(cls))
    return " · ".join(parts)


# A slot thumbnail older than this is stale scenery, not a live preview.
THUMB_MAX_AGE_S = 3 * 3600.0


def fetch_grid_thumbnails(grid: dict | None, latest: list[dict],
                          bucket_name: str | None,
                          downloader=None) -> list[dict]:
    """One small annotated frame per active grid slot for the report cover.

    The collector overwrites `snapshots/live/<slot>.jpg` on every sample
    (see collector._save_live_view - 24h Storage lifecycle keeps it fresh);
    when Storage is off, `latest.live_url` still points at the last one.
    Returns a list of {name, city, country, jpeg} - the caller renders it
    as a 2x2 strip. Silently drops slots without a resolvable image.
    """
    if downloader is None:
        downloader = _http_bytes
    if not grid or not bucket_name:
        return []
    latest_by_slot = {str(d.get("slot") or ""): d for d in latest}
    out: list[dict] = []
    for s in (grid or {}).get("slots") or []:
        slot_id = str(s.get("slot_id") or "")
        # Staleness gate (2026-07-31): slot_N.jpg is only overwritten on a
        # SUCCESSFUL sample, so when a slot's latest doc is a miss (or old)
        # the stored image is whatever camera last succeeded there - the
        # Japan midday report attached Taksim/Beyazit frames under Japanese
        # camera names after an all-miss night. No fresh success = no
        # thumbnail; an honest gap beats a wrong picture.
        d = latest_by_slot.get(slot_id) or {}
        fresh = False
        if d.get("ok") == 1 and d.get("ts"):
            try:
                age = (dt.datetime.now(dt.timezone.utc)
                       - dt.datetime.fromisoformat(str(d["ts"]))).total_seconds()
                fresh = 0 <= age <= THUMB_MAX_AGE_S
            except ValueError:
                fresh = False
        if not fresh:
            continue
        # The live-view URL when the collector wrote one; otherwise fall
        # back to the fixed Storage path each slot's live tile writes to.
        live_url = (d.get("live_url")
                    or f"https://storage.googleapis.com/{bucket_name}/"
                       f"snapshots/live/{slot_id}.jpg")
        data = downloader(live_url)
        if not data:
            continue
        out.append({
            "name":    s.get("active_cam_name") or s.get("active_cam") or "?",
            "city":    s.get("city") or "",
            "country": s.get("country") or "",
            "jpeg":    _shrink_jpeg(data, max_width=520, quality=70),
        })
    return out


def fetch_snapshots_for_groups(groups: list[dict],
                               bucket_name: str | None = None,
                               downloader=None,
                               shrink: bool = True) -> list[dict]:
    """Download the visual assets for each picked group. Returns groups
    augmented with ``fullframe_jpeg`` (context), ``crop_jpeg`` (the specific
    object) and, for returning-visitor entries, ``first_jpeg`` (oldest
    sighting of that entity from the entity gallery).

    Groups whose downloads all fail are dropped - a card without images is
    just repeated text."""
    if downloader is None:
        downloader = _http_bytes
    out: list[dict] = []
    for g in groups:
        e = g.get("latest_event") or {}
        entry = dict(g)

        ff = e.get("fullframe_url")
        cr = e.get("snapshot_url")
        bf = e.get("before_url")
        ff_bytes = downloader(ff) if ff else None
        cr_bytes = downloader(cr) if cr else None
        bf_bytes = downloader(bf) if bf else None
        if not ff_bytes and not cr_bytes and not bf_bytes:
            continue
        # If we have only the crop, promote it to fullframe slot so the
        # card layout still fills its primary image row.
        if not ff_bytes and cr_bytes:
            ff_bytes = cr_bytes
            cr_bytes = None

        # Draw the red rectangle on the fullframe BEFORE the downscale so
        # the line-width scales down with the image. When the event does
        # not carry box coords (pre-v6 collector output), the helper is a
        # no-op and the fullframe ships as-is.
        if ff_bytes:
            ff_bytes = _draw_box_on_frame(
                ff_bytes,
                e.get("box"),
                int(e.get("frame_w") or 0),
                int(e.get("frame_h") or 0),
                cls=e.get("cls"))

        # fix1-A6: the settle-time SCENE (the object still standing, in
        # context) - boxed like the fullframe so the card can lead with it.
        if bf_bytes:
            bf_bytes = _draw_box_on_frame(
                bf_bytes, e.get("box"),
                int(e.get("frame_w") or 0), int(e.get("frame_h") or 0),
                cls=e.get("cls"))

        entry["fullframe_jpeg"] = _shrink_jpeg(ff_bytes) if shrink and ff_bytes else ff_bytes
        entry["crop_jpeg"] = _shrink_jpeg(cr_bytes) if shrink and cr_bytes else cr_bytes
        entry["before_jpeg"] = _shrink_jpeg(bf_bytes) if shrink and bf_bytes else bf_bytes

        # First-sighting lookup for returning AND loitering visitors:
        # both are "this same entity" claims that read as unfounded without
        # a comparison frame. Missing entity_id, missing bucket, or a stale
        # manifest all resolve to "no first image" - the card degrades
        # gracefully to a single object crop.
        if (g.get("kind") in ("returning", "loiter") and bucket_name
                and e.get("entity_id") is not None):
            first_url = find_first_sighting(bucket_name,
                                            str(e.get("cam_id") or ""),
                                            e.get("entity_id"),
                                            downloader=downloader)
            if first_url:
                first_bytes = downloader(first_url)
                if first_bytes:
                    entry["first_jpeg"] = (_shrink_jpeg(first_bytes)
                                           if shrink else first_bytes)
        out.append(entry)
    return out


def _evidence_card(group: dict, kind_labels: dict[str, str]) -> Table:
    """Framed layout: header line with the #N badge + primary fullframe +
    the small crop of the object (and, for returning visitors, the first
    sighting side-by-side with the current crop)."""
    e = group.get("latest_event") or {}
    ref = group.get("ref")
    header_style = ParagraphStyle(
        "ev_head", fontName="Helvetica-Bold", fontSize=12, alignment=TA_LEFT,
        leading=15, textColor=colors.HexColor("#0f172a"))
    body_style = ParagraphStyle(
        "ev_body", fontName="Helvetica", fontSize=10, alignment=TA_LEFT,
        leading=13, textColor=colors.HexColor("#334155"))
    small_style = ParagraphStyle(
        "ev_small", fontName="Helvetica", fontSize=9, alignment=TA_LEFT,
        leading=11, textColor=colors.HexColor("#64748b"))

    header_bits = []
    if ref:
        header_bits.append(f"<font color='#b91c1c'>#{ref}</font>")
    header_bits.append(group.get("label") or "?")
    header_bits.append(group.get("cam") or "?")
    if group.get("count", 0) > 1:
        header_bits.append(f"x{group['count']}")
    header_bits.append(f"latest {_fmt_ts(group.get('last_ts'))}")
    header = Paragraph(" · ".join(header_bits), header_style)

    extras = []
    if e.get("fall_suspect"):
        extras.append("possible FALL - torso read horizontal")
    dur = e.get("duration_sec")
    if isinstance(dur, (int, float)) and dur > 0:
        extras.append(f"duration {int(dur)} sec")
    cls = e.get("cls")
    if cls:
        extras.append(f"class '{cls}'")
    eid = e.get("entity_id")
    if eid is not None:
        extras.append(f"entity #{eid}")
    sub = Paragraph(" · ".join(extras) if extras else "-", body_style)

    rows: list[list] = [[header], [sub]]

    # A static-departed card's big scene is the AFTER frame - the object is
    # gone and the red box floats over whatever now occupies the spot.
    # Without a caption the box reads as a live (mis)detection of the
    # pedestrians standing there (operator complaint, 09.08: the vendor
    # cart's departure card looked like "the model calls these people a
    # car"). Say what each image IS.
    is_static = group.get("kind") == "static_departed"
    is_unattended = group.get("kind") == "unattended_object"
    before_bytes = group.get("before_jpeg")
    ff = group.get("fullframe_jpeg")

    if before_bytes:
        # fix1-A6 layout: lead with the settle-time scene - the object
        # STANDING there, boxed in context (the operator's "what happened
        # before"); the current frame joins the small pair row below.
        rows.append([Paragraph(
            "The scene while the object stood there - the red box marks it:",
            small_style)])
        rows.append([Image(BytesIO(before_bytes), width=15*cm,
                           height=8.5*cm, kind="proportional")])
    elif ff:
        if is_static:
            rows.append([Paragraph(
                "The scene AFTER departure - the red box marks the spot "
                "where the object had been standing:", small_style)])
        rows.append([Image(BytesIO(ff), width=15*cm, height=8.5*cm,
                           kind="proportional")])

    # Small pair row: for static/unattended with a before-scene, the pair
    # is NOW-scene + object crop; for returning/loiter it stays first
    # sighting + current crop.
    crop_bytes = group.get("crop_jpeg")
    first_bytes = group.get("first_jpeg")
    pair_left = None
    if before_bytes and ff:
        pair_left = (("The spot NOW - the object is gone:" if is_static
                      else "The scene NOW - the object is still there:"), ff)
    elif first_bytes:
        pair_left = ("First seen previously:", first_bytes)
    if crop_bytes or pair_left:
        image_cells: list[list] = [[], []]
        if pair_left:
            image_cells[0].append(Paragraph(pair_left[0], small_style))
            image_cells[0].append(Image(BytesIO(pair_left[1]),
                                        width=6*cm, height=4.5*cm,
                                        kind="proportional"))
        else:
            image_cells[0].append(Paragraph("", small_style))
        if crop_bytes:
            label = ("Same object now:" if first_bytes
                     else "The object, up close:"
                     if (is_static or is_unattended)
                     else "Specific object flagged:")
            image_cells[1].append(Paragraph(label, small_style))
            image_cells[1].append(Image(BytesIO(crop_bytes),
                                        width=6*cm, height=4.5*cm,
                                        kind="proportional"))
        else:
            image_cells[1].append(Paragraph("", small_style))
        pair = Table([image_cells], colWidths=[8*cm, 8*cm])
        pair.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        rows.append([pair])

    frame = Table(rows, colWidths=[16.2*cm])
    frame.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX",           (0, 0), (-1, -1), 0.75, colors.HexColor("#cbd5e1")),
        ("LINEBELOW",     (0, 1), (-1, 1),  0.5,  colors.HexColor("#e2e8f0")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    return frame


def compose_pdf(out_path: str | Path, *,
                now_il: dt.datetime,
                window_hours: int,
                events_by_kind: list[dict],
                cam_stats: list[dict],
                training: dict | None,
                stale_slots: list[dict],
                snapshots: Iterable[dict],
                kind_labels: dict[str, str],
                total_events: int,
                total_samples: int,
                training_lines: list[str] | None = None,
                country_label: str = "Live grid",
                grid_country_label: str | None = None,
                grid_cameras: list[dict] | None = None,
                grid_thumbnails: list[dict] | None = None,
                dominant: str | None = None,
                current_country: str | None = None) -> Path:
    """Compose the phone-oriented English PDF and return its path.

    ``snapshots`` is a list of picked GROUP dicts (from
    ``pick_group_samples``) - each carries a ``ref`` linking back to the
    numbered row in the anomalies table, plus the primary image bytes
    (``fullframe_jpeg``), optional object crop (``crop_jpeg``), and optional
    first-sighting crop (``first_jpeg``) for returning-visitor entries."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    styles = {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=20,
                                alignment=TA_CENTER, leading=26, spaceAfter=10),
        "sub":   ParagraphStyle("sub", fontName="Helvetica", fontSize=11,
                                alignment=TA_CENTER, leading=14, spaceAfter=16,
                                textColor=colors.HexColor("#475569")),
        "h":     ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=14,
                                alignment=TA_LEFT, spaceBefore=14,
                                spaceAfter=6, textColor=colors.HexColor("#0f172a")),
        "body":  ParagraphStyle("body", fontName="Helvetica", fontSize=11,
                                alignment=TA_LEFT, leading=15),
        "cap":   ParagraphStyle("cap", fontName="Helvetica", fontSize=10,
                                alignment=TA_LEFT,
                                textColor=colors.HexColor("#475569"),
                                spaceAfter=10),
        "warn":  ParagraphStyle("warn", fontName="Helvetica-Bold", fontSize=11,
                                alignment=TA_LEFT, spaceAfter=4,
                                textColor=colors.HexColor("#b91c1c")),
        "ok":    ParagraphStyle("ok", fontName="Helvetica", fontSize=11,
                                alignment=TA_LEFT,
                                textColor=colors.HexColor("#166534")),
    }

    story = []
    part = "Midday Report" if now_il.hour < 16 else "Evening Report"
    story.append(Paragraph(f"{country_label} - Activity Summary - {part}",
                           styles["title"]))
    story.append(Paragraph(f"{_fmt_date(now_il)}  ·  last {window_hours} hours",
                           styles["sub"]))

    # "Currently watching" callout when the window's data majority and the
    # grid at digest time disagree - an overnight ladder walk would title
    # the report by the country that owned the WINDOW, and this line names
    # where the collector actually is right now.
    grid_label_txt = grid_country_label or country_label
    if (dominant and current_country
            and dominant != current_country):
        story.append(Paragraph(
            f"This window's data is mostly from {country_label}. "
            f"The grid moved to {grid_label_txt} at digest time - the "
            f"section below reflects the current grid, not the window's "
            f"majority.",
            styles["cap"]))

    # Live-grid strip: annotated thumbnails of what the four active
    # cameras look like right now. The report used to be text-only for
    # this section; a picture answers the "does the grid actually work?"
    # question at a glance.
    if grid_thumbnails:
        story.append(Paragraph(
            f"Live grid preview - {grid_label_txt}", styles["h"]))
        _thumb_cells = []
        cap_style = ParagraphStyle(
            "thumbcap", fontName="Helvetica", fontSize=9,
            alignment=TA_CENTER, leading=11,
            textColor=colors.HexColor("#334155"))
        for t in grid_thumbnails[:4]:
            name = str(t.get("name") or "?")
            city = str(t.get("city") or "")
            caption = name + (f" · {city}" if city else "")
            _thumb_cells.append([
                Image(BytesIO(t["jpeg"]), width=7.5*cm, height=4.2*cm,
                      kind="proportional"),
                Paragraph(caption, cap_style),
            ])
        # Two-column table, each cell an image over its caption.
        col = []
        for i in range(0, len(_thumb_cells), 2):
            row = _thumb_cells[i:i+2]
            while len(row) < 2:
                row.append([Paragraph("", cap_style)])
            col.append(row)
        tt = Table(col, colWidths=[8.1*cm, 8.1*cm])
        tt.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(tt)
        story.append(Spacer(1, 4))

    # Live grid: which cameras / country the collector is running RIGHT NOW.
    # The collector is country-generic (4 cameras from one country, rotating
    # through a ladder when a country is blocked), so this section makes the
    # report self-describing - the reader sees exactly what was watched.
    if grid_cameras:
        story.append(Paragraph(f"Live grid - {grid_label_txt} "
                               f"({len(grid_cameras)} cameras)", styles["h"]))
        grid_rows = [["Camera", "City", "Country"]]
        for gc in grid_cameras:
            grid_rows.append([str(gc.get("name") or "?"),
                              str(gc.get("city") or "-"),
                              str(gc.get("country") or "-").title()])
        gt = Table(grid_rows, colWidths=[9*cm, 5*cm, 4*cm])
        gt.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("LINEBELOW", (0, 0), (-1, 0), 0.75, colors.HexColor("#94a3b8")),
            ("LINEBELOW", (0, 1), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(gt)
        story.append(Spacer(1, 10))

    # KPI cards - two columns
    events_word = "anomaly" if total_events == 1 else "anomalies"
    samples_word = "sample" if total_samples == 1 else "samples"
    kpi_rows = [[f"{total_events} {events_word}",
                 f"{total_samples} {samples_word}"]]
    kpi = Table(kpi_rows, colWidths=[9*cm, 9*cm])
    kpi.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 13),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (0, 0),   colors.HexColor("#fef3c7")),
        ("BACKGROUND", (1, 0), (1, 0),   colors.HexColor("#dbeafe")),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("BOX",        (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("LINEAFTER",  (0, 0), (0, 0),   0.25, colors.HexColor("#cbd5e1")),
    ]))
    story.append(kpi)

    story.append(Paragraph("Camera Status", styles["h"]))
    if stale_slots:
        for s in stale_slots:
            reason = s.get("reason")
            if reason:
                msg = f"[!] {s['cam']} - {reason}"
            else:
                msg = (f"[!] {s['cam']} - not reporting for "
                       f"{s['age_min']} minutes")
            story.append(Paragraph(msg, styles["warn"]))
    elif cam_stats and all(c["peak_person"] == 0 and c["peak_vehicles"] == 0
                           for c in cam_stats):
        # Silent-miss guard: every camera looks fresh but nothing was
        # detected across the window - the streams are almost certainly
        # dead or geo-blocked. Without this the report reads "all cameras
        # active and reporting normally" while every peak is 0.
        story.append(Paragraph(
            f"[!] {len(cam_stats)} camera(s) reporting but detected 0 people "
            "and 0 vehicles across the entire window - streams may be dead, "
            "geo-blocked or obscured. Check the VM journal for repeated "
            "MISS lines.",
            styles["warn"]))
    else:
        story.append(Paragraph("✓  All cameras active and reporting normally",
                               styles["ok"]))

    # Camera names in these tables are long ("Konya - Hukumet Meydani /
    # Sarraflar Yeralti Carsisi"); wrapping them as Paragraph flowables
    # keeps the cell inside its column width instead of running over its
    # neighbor. Plain strings do not wrap in reportlab tables.
    cell_style = ParagraphStyle("cell", fontName="Helvetica", fontSize=10,
                                alignment=TA_LEFT, leading=12)

    # Aggregated anomalies table with a "#" column that ties every row to
    # its evidence card on the pages below. A row without a matching image
    # simply has no card - the operator can still see the count and location.
    story.append(Paragraph("Anomalies by Type and Camera", styles["h"]))
    if events_by_kind:
        header = ["#", "Type", "Camera", "Last", "Count"]
        body = [header]
        for g in events_by_kind:
            body.append([str(g.get("ref") or ""),
                         Paragraph(g["label"], cell_style),
                         Paragraph(g["cam"], cell_style),
                         _fmt_ts(g["last_ts"]), str(g["count"])])
        tbl = Table(body, colWidths=[1*cm, 5*cm, 6*cm, 2.2*cm, 2.4*cm],
                    repeatRows=1)
        tbl.setStyle(_table_style())
        story.append(tbl)
        story.append(Paragraph(
            "Numbered rows have a matching image on the next pages.",
            styles["cap"]))
    else:
        story.append(Paragraph("Quiet - no anomalies in this window.",
                               styles["body"]))

    # Camera activity peaks - only cameras that actually delivered frames.
    # The ladder walk probes dead cameras across every country and each
    # probe writes an ok=0 doc; listing those buried the real story under
    # 20+ rows of zeros. Cameras from the reporting-country majority sit
    # in their own table; cameras from earlier legs of the same window get
    # a compact "earlier in this window" table underneath (context without
    # noise). Dead-probed cameras collapse to one honest footnote.
    _visible = [c for c in cam_stats if c.get("samples", 0) > 0]
    _dark = len(cam_stats) - len(_visible)
    try:
        from app.cameras import CAMERAS as _CAM_CAT
    except Exception:
        _CAM_CAT = {}
    _dom = dominant or current_country
    _dom_rows = [c for c in _visible
                 if (_CAM_CAT.get(str(c.get("cam_id"))) or {}).get(
                     "country", _dom) == _dom]
    _other_rows = [c for c in _visible if c not in _dom_rows]

    def _peaks_table(rows, title):
        story.append(Paragraph(title, styles["h"]))
        header = ["Camera", "Peak People", "Peak Vehicles",
                  "Typical Traffic", "Crossings in/out"]
        body = [header]
        for c in rows:
            people = str(c["peak_person"])
            if c["peak_person_ts"]:
                people += f" at {_fmt_ts(c['peak_person_ts'])}"
            spd = f"~{c['typ_kmh']:.0f} km/h" if c["typ_kmh"] > 0 else "-"
            cross = (f"{c.get('cross_in', 0)} / {c.get('cross_out', 0)}"
                     if c.get("has_line") else "-")
            body.append([Paragraph(c["cam"], cell_style),
                         people, str(c["peak_vehicles"]), spd, cross])
        tbl = Table(body, colWidths=[5.6*cm, 3.2*cm, 2.3*cm, 2.7*cm, 2.8*cm],
                    repeatRows=1)
        tbl.setStyle(_table_style())
        story.append(tbl)
        # fix1-A5: EXTRA_CLASSES peaks (birds, bags, umbrellas...) - counted
        # by the collector but absent from the person/vehicle columns; one
        # compact line keeps them visible without widening the table.
        parts = []
        for c in rows:
            ep = c.get("extra_peaks") or {}
            if ep:
                inner = ", ".join(f"{cls} {n}" for cls, n in
                                  sorted(ep.items(), key=lambda kv: -kv[1]))
                parts.append(f"{c['cam']}: {inner}")
        if parts:
            story.append(Paragraph(
                "Other objects (window peaks) - " + "; ".join(parts),
                styles["cap"]))

    if _dom_rows:
        _peaks_table(_dom_rows,
                     f"Activity Peaks - {country_label}")
    if _other_rows:
        _peaks_table(_other_rows,
                     "Earlier in this window (before the grid settled here)")
    if not _visible:
        story.append(Paragraph("Activity Peaks by Camera", styles["h"]))
        story.append(Paragraph("No footfall samples in this window.",
                               styles["body"]))
    if _dark:
        story.append(Paragraph(
            f"{_dark} more camera(s) were probed but delivered no frames "
            "(dead / geo-blocked) - omitted above.",
            styles["cap"]))

    # Training status
    story.append(Paragraph("Model Training Status", styles["h"]))
    for line in (training_lines or []):
        story.append(Paragraph(line, styles["body"]))
    if not training_lines:
        # Fallback for callers that predate the training_lines parameter -
        # keep the report shipping instead of leaving a blank section.
        if training:
            verdict = ("PROMOTED new head" if training.get("promoted")
                       else "did not clear the gate yet")
            cand = training.get("candidate") or training.get("file") or "?"
            when = str(training.get("at") or "")[:10]
            story.append(Paragraph(
                f"Last cloud training ({when}): {verdict} - {cand}.",
                styles["body"]))
        else:
            story.append(Paragraph(
                "No cloud training run has executed yet.", styles["body"]))

    # Visual evidence pages - each card carries the #N badge that appears
    # in the anomalies table above, so the operator can look at an image
    # and tell exactly which row it belongs to (no more caption hunting).
    snap_list = list(snapshots)
    if snap_list:
        story.append(PageBreak())
        story.append(Paragraph(
            "Evidence - What the Model Actually Flagged",
            styles["h"]))
        story.append(Paragraph(
            "Each card matches one numbered row in the Anomalies table. "
            "The full scene shows context; the small crop is the specific "
            "object the model flagged. For returning visitors, the earlier "
            "sighting of the same identity is included so you can eyeball "
            "the identity claim.",
            styles["cap"]))
        for g in snap_list:
            story.append(_evidence_card(g, kind_labels))
            story.append(Spacer(1, 0.4*cm))
    else:
        story.append(Paragraph(
            "No evidence images attached (all events lacked snapshot URLs "
            "or the bucket was unreachable).", styles["cap"]))

    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.4*cm, bottomMargin=1.4*cm,
                            title=f"{country_label} activity report {_fmt_date(now_il)}",
                            author="turkey-collector")
    doc.build(story)
    return out_path
