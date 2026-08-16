"""English report: sample selection, snapshot fetch, PDF composition."""
import datetime as dt

import pytest

reportlab = pytest.importorskip("reportlab")
report_pdf = pytest.importorskip("tools.report_pdf")


def _mk(kind, cam, ts, full=None, snap=None, **extra):
    e = {"kind": kind, "cam_id": cam, "ts": ts, **extra}
    if full: e["fullframe_url"] = full
    if snap: e["snapshot_url"] = snap
    return e


def test_pick_event_samples_priority_and_recency():
    """Priority (obstructed > dark > extreme > loiter > returning) beats
    volume; within a kind, the newest event wins the first slot."""
    events = [
        # a flood of loiter events...
        *[_mk("loiter", "cam_A", f"2026-07-11T0{i}:00:00Z", snap="s") for i in range(5)],
        _mk("camera_obstructed", "cam_B", "2026-07-10T00:00:00Z", snap="s"),
        _mk("camera_dark", "cam_B", "2026-07-10T01:00:00Z", snap="s"),
        _mk("extreme_load", "cam_C", "2026-07-10T02:00:00Z", full="f"),
        _mk("returning", "cam_A", "2026-07-11T00:00:00Z", snap="s"),
    ]
    picks = report_pdf.pick_event_samples(events, max_total=6)
    kinds = [p["kind"] for p in picks]
    # First pass: one per priority-ordered kind that has any events.
    assert kinds[:5] == ["camera_obstructed", "camera_dark",
                         "extreme_load", "loiter", "returning"]
    # 6th slot goes back around; loiter has 5 events so a second round-robin
    # gives another loiter (the SECOND-newest of the batch).
    assert kinds[5] == "loiter"
    # newest loiter first
    assert picks[3]["ts"] == "2026-07-11T04:00:00Z"


def test_pick_event_samples_skips_events_without_any_url():
    events = [
        _mk("loiter", "cam_A", "2026-07-11T02:00:00Z"),        # no url
        _mk("loiter", "cam_A", "2026-07-11T01:00:00Z", snap="s"),
    ]
    picks = report_pdf.pick_event_samples(events, max_total=5)
    assert len(picks) == 1
    assert picks[0]["ts"] == "2026-07-11T01:00:00Z"


def test_pick_event_samples_max_cap():
    events = [_mk("loiter", "cam_A", f"2026-07-11T{h:02d}:00:00Z", snap="s")
              for h in range(20)]
    assert len(report_pdf.pick_event_samples(events, max_total=3)) == 3


def test_fetch_snapshots_uses_fullframe_then_snapshot():
    """We prefer the wide fullframe (context) - the object crop is only the
    fallback. A URL whose downloader returns None must fall through, not
    poison the pick."""
    served = {"https://ff.jpg": b"x" * 5000, "https://snap.jpg": b"y" * 5000}
    seen = []

    def dl(url):
        seen.append(url)
        return served.get(url)

    picks = [
        _mk("loiter", "cam_A", "t", full="https://ff.jpg", snap="https://snap.jpg"),
        _mk("loiter", "cam_A", "t", full="https://gone.jpg",
            snap="https://snap.jpg"),                 # fullframe 404 -> use snap
        _mk("loiter", "cam_A", "t", full="https://gone.jpg"),  # both gone
    ]
    # shrink=False so the byte comparison stays exact
    out = report_pdf.fetch_snapshots(picks, downloader=dl, shrink=False)
    assert [b for _, b in out] == [b"x" * 5000, b"y" * 5000]
    assert seen == ["https://ff.jpg", "https://gone.jpg",
                    "https://snap.jpg", "https://gone.jpg"]


def _jpeg(color=(180, 180, 180)):
    """A 320x180 solid-color JPEG so reportlab has something to embed."""
    pil = pytest.importorskip("PIL.Image")
    from io import BytesIO
    im = pil.new("RGB", (320, 180), color)
    buf = BytesIO()
    im.save(buf, "JPEG", quality=70)
    return buf.getvalue()


def test_event_caption_english():
    """Caption is a simple dot-joined line: label · camera · time · duration
    · class, with duration and class dropped when not applicable."""
    cap = report_pdf._event_caption(
        {"kind": "loiter", "cam_id": "konya_millet",
         "ts": "2026-07-12T15:30:00Z", "duration_sec": 320, "cls": "person"},
        {"loiter": "Loitering"})
    assert "Loitering" in cap
    assert "konya_millet" in cap
    assert "320 sec" in cap
    assert "person" in cap
    # Missing duration/class - just three parts.
    cap2 = report_pdf._event_caption(
        {"kind": "returning", "cam_id": "cam_x", "ts": "2026-07-12T07:00:00Z"},
        {"returning": "Returning visitor"})
    assert cap2.count(" · ") == 2


def test_draw_box_marks_the_image():
    """The overlay draws a red rectangle at the given coordinates. A
    downstream reader (PIL) sees red pixels where the box was drawn and
    none in the untouched center of a grey source image."""
    pil = pytest.importorskip("PIL.Image")
    from io import BytesIO
    im = pil.new("RGB", (400, 300), (128, 128, 128))
    buf = BytesIO(); im.save(buf, "JPEG", quality=90)
    src = buf.getvalue()

    out = report_pdf._draw_box_on_frame(
        src, box=[100, 60, 200, 180], frame_w=400, frame_h=300, cls="person")
    assert out != src and len(out) > 100
    marked = pil.open(BytesIO(out))
    # Sample a pixel exactly on the rectangle border - should be red-ish.
    r, g, b = marked.getpixel((100, 60))[:3]
    assert r > 180 and g < 80 and b < 80, f"border pixel not red: ({r},{g},{b})"
    # Interior of the box stays the original grey (we only outline).
    r2, g2, b2 = marked.getpixel((150, 120))[:3]
    assert abs(r2 - 128) < 20 and abs(g2 - 128) < 20


def test_draw_box_noop_on_bad_input():
    """Missing coords or a corrupt image must not raise - the report ships
    the fullframe as-is when overlay data is unusable."""
    b = b"not-a-jpeg"
    assert report_pdf._draw_box_on_frame(b, None, 100, 100) is b
    assert report_pdf._draw_box_on_frame(b, [0, 0, 10, 10], 0, 100) is b
    # inverted box (x2 < x1) is rejected before PIL sees it
    pil = pytest.importorskip("PIL.Image")
    from io import BytesIO
    im = pil.new("RGB", (100, 100), (50, 50, 50))
    buf = BytesIO(); im.save(buf, "JPEG", quality=80)
    src = buf.getvalue()
    assert report_pdf._draw_box_on_frame(src, [50, 50, 10, 10], 100, 100) is src


def test_pick_group_samples_uses_priority_and_recency():
    """Groups (aggregated by kind+cam) are picked by kind priority, not by
    raw event count - a flood of loiter events at one camera is one row
    that gets one visual card, freeing the other slots for other kinds."""
    groups = [
        {"kind": "loiter", "cam": "A", "count": 20,
         "last_ts": "2026-07-11T09:00:00Z",
         "latest_event": {"kind": "loiter", "snapshot_url": "s"}},
        {"kind": "camera_obstructed", "cam": "B", "count": 1,
         "last_ts": "2026-07-11T02:00:00Z",
         "latest_event": {"kind": "camera_obstructed", "snapshot_url": "s"}},
        {"kind": "returning", "cam": "A", "count": 1,
         "last_ts": "2026-07-11T04:00:00Z",
         "latest_event": {"kind": "returning", "snapshot_url": "s"}},
        # No URL -> disqualified
        {"kind": "camera_dark", "cam": "C", "count": 1,
         "last_ts": "2026-07-11T05:00:00Z",
         "latest_event": {"kind": "camera_dark"}},
    ]
    picks = report_pdf.pick_group_samples(groups, max_total=4)
    kinds = [p["kind"] for p in picks]
    assert "camera_obstructed" in kinds        # priority beats recency
    assert "loiter" in kinds and "returning" in kinds
    assert "camera_dark" not in kinds          # no image = no card


def test_find_first_sighting(monkeypatch):
    """Manifest lookup returns the URL of the oldest entity-gallery jpg for
    the given (cam, entity)."""
    import json
    manifest = {"files": {
        "entities/camA/7/2000.jpg": {"mtime": 2000.0, "url": "u2"},
        "entities/camA/7/1000.jpg": {"mtime": 1000.0, "url": "u1"},
        "entities/camA/7/1500.jpg": {"mtime": 1500.0, "url": "u1_5"},
        "entities/camB/9/9999.jpg": {"mtime": 9999.0, "url": "u_other"},
    }}
    served = {"manifest": json.dumps(manifest).encode()}

    def dl(url):
        if "manifest" in url: return served["manifest"]
        return None

    url = report_pdf.find_first_sighting("bkt", "camA", 7, downloader=dl)
    assert url == "u1"
    # unknown entity -> None, no exception
    assert report_pdf.find_first_sighting("bkt", "camA", 999, downloader=dl) is None


def test_fetch_snapshots_for_groups_wires_crop_and_first_sighting():
    served = {
        "https://ff.jpg":  b"F" * 5000,
        "https://cr.jpg":  b"C" * 5000,
        "https://first.jpg": b"P" * 5000,
        "https://storage.googleapis.com/bkt/review_sync/manifest.json?t=":
            None,
    }
    import json

    def dl(url):
        # coarse match ignoring cache-buster
        if url.startswith("https://ff.jpg"): return served["https://ff.jpg"]
        if url.startswith("https://cr.jpg"): return served["https://cr.jpg"]
        if url.startswith("https://first.jpg"): return served["https://first.jpg"]
        if "manifest" in url:
            return json.dumps({"files": {
                "entities/camA/7/1000.jpg": {"mtime": 1000.0,
                                             "url": "https://first.jpg"},
            }}).encode()
        return None

    groups = [
        {"ref": 1, "kind": "returning", "cam": "camA",
         "label": "Returning visitor", "count": 1,
         "last_ts": "2026-07-11T06:00:00Z",
         "latest_event": {"kind": "returning", "cam_id": "camA",
                          "entity_id": 7,
                          "fullframe_url": "https://ff.jpg",
                          "snapshot_url":  "https://cr.jpg"}},
    ]
    out = report_pdf.fetch_snapshots_for_groups(groups, bucket_name="bkt",
                                                downloader=dl, shrink=False)
    assert len(out) == 1
    g = out[0]
    assert g["fullframe_jpeg"] == b"F" * 5000
    assert g["crop_jpeg"] == b"C" * 5000
    assert g.get("first_jpeg") == b"P" * 5000


def test_compose_pdf_end_to_end(tmp_path):
    """Compose a real PDF from realistic inputs and validate: file exists,
    is a valid PDF, has both summary and evidence pages."""
    out = tmp_path / "report.pdf"
    now = dt.datetime(2026, 7, 12, 20, 0)
    groups = [
        {"ref": 1, "kind": "loiter", "label": "Loitering",
         "cam": "Konya - Millet Caddesi", "count": 14,
         "last_ts": "2026-07-12T15:30:00Z",
         "latest_event": {"kind": "loiter", "cam_id": "konya_millet",
                          "ts": "2026-07-12T15:30:00Z",
                          "duration_sec": 420, "cls": "person"}},
        {"ref": 2, "kind": "camera_obstructed", "label": "Camera blocked",
         "cam": "Otogar Kavsagi", "count": 2,
         "last_ts": "2026-07-12T13:00:00Z",
         "latest_event": {"kind": "camera_obstructed", "cam_id": "otogar",
                          "ts": "2026-07-12T13:00:00Z", "cls": "train"}},
    ]
    cam_stats = [
        {"cam": "Konya - Hukumet", "peak_person": 14,
         "peak_person_ts": "2026-07-12T09:07:00Z",
         "peak_vehicles": 13, "typ_kmh": 22.0, "samples": 500},
    ]
    stale = []
    snaps = [dict(groups[0], fullframe_jpeg=_jpeg((100, 140, 200)),
                  crop_jpeg=_jpeg((200, 200, 200))),
             dict(groups[1], fullframe_jpeg=_jpeg((60, 60, 60)),
                  crop_jpeg=None)]
    LABELS = {"loiter": "Loitering", "camera_obstructed": "Camera blocked"}
    p = report_pdf.compose_pdf(
        out, now_il=now, window_hours=12,
        events_by_kind=groups, cam_stats=cam_stats,
        training=None, stale_slots=stale,
        snapshots=snaps, kind_labels=LABELS,
        total_events=16, total_samples=980,
        training_lines=["You have labeled 5 frames so far.",
                        "Cloud training did not improve on the baseline yet."])
    assert p == out and p.is_file()
    head = p.read_bytes()[:8]
    assert head.startswith(b"%PDF-"), f"not a PDF: {head!r}"
    size_kb = p.stat().st_size / 1024
    assert 5 < size_kb < 800, f"unexpected PDF size {size_kb:.0f} KB"


def test_compose_pdf_handles_empty_window(tmp_path):
    """Off-hours or a fresh install produces zero events, zero snapshots -
    the report must still render without raising."""
    out = tmp_path / "empty.pdf"
    now = dt.datetime(2026, 7, 12, 12, 0)
    p = report_pdf.compose_pdf(
        out, now_il=now, window_hours=12,
        events_by_kind=[], cam_stats=[],
        training=None, stale_slots=[],
        snapshots=[], kind_labels={},
        total_events=0, total_samples=0)
    assert p.is_file() and p.read_bytes()[:5] == b"%PDF-"
