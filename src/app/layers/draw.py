"""Per-layer frame drawing for the live analysis pipeline.

Extracted from live_analysis.py in the 2026-08-23 split (decision D1):
everything here is STATELESS - pure functions from (image, layer state)
to an annotated image, plus the small geometry helpers the line/parking
drawers share. Session state, grabbing, inference and publishing stay
in live_analysis.LiveSession; new layers add a draw_<layer>_layer here
and a render branch there.
"""
from __future__ import annotations

import time

TRAIL_MAX_PTS = 40
FIRE_CONFIRM_TICKS = 2
# Body-anomaly layer: which behavior labels count as an anomaly worth
# drawing. "running" was removed 2026-08-18: normal fast walking often
# labeled as running, causing constant red flags for legitimate street
# behaviour. Real escapes trip the bbox-centroid velocity fallback below
# which is set high enough to only catch actual bolt-away motion.
BODY_ANOMALY_LABELS = frozenset({"fall_suspect", "erratic",
                                 "sudden_motion", "fighting"})

# Stable per-track colors (operator direction 2026-08-24): every tracked
# object keeps ONE distinct color for its box, trail, and side card, so
# the eye pairs card-to-object by color instead of chasing leader lines.
# 12 well-separated hues (BGR); tid hashes into the palette, so a track
# keeps its color for its whole life and collisions only start when more
# than 12 objects share the frame.
TRACK_PALETTE = (
    (80, 200, 255),   # amber
    (90, 90, 245),    # red
    (220, 160, 70),   # blue
    (120, 210, 120),  # green
    (230, 100, 200),  # purple
    (90, 220, 220),   # yellow
    (200, 200, 120),  # aqua
    (140, 120, 245),  # rose
    (245, 190, 140),  # light blue
    (100, 160, 230),  # orange
    (200, 140, 170),  # violet
    (150, 220, 170),  # light green
)


def track_color(tid) -> tuple:
    """Stable BGR color for a track id."""
    try:
        idx = int(tid) % len(TRACK_PALETTE)
    except (TypeError, ValueError):
        idx = 0
    return TRACK_PALETTE[idx]


# ---------------------------------------------------------------------------
# Layer renderers - each draws ONLY its layer's semantics + an honest
# caption. All mutate/return the given BGR frame.
# ---------------------------------------------------------------------------

def _caption(img, lines) -> "object":
    """Darkened strip at the top with the layer verdict ("no gestures
    detected right now" is a legitimate, expected outcome - fix 2)."""
    import cv2
    if isinstance(lines, str):
        lines = [lines]
    lh, pad = 22, 8
    h = min(img.shape[0], pad * 2 + lh * len(lines) - 8)
    img[0:h] = (img[0:h] * 0.35).astype(img.dtype)
    y = pad + 12
    for i, t in enumerate(lines):
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 if i == 0 else 0.46,
                    (255, 255, 255) if i == 0 else (205, 205, 205),
                    1, cv2.LINE_AA)
        y += lh
    return img


def _chip(img, b: dict, txt: str, color) -> None:
    import cv2
    x1, y2 = int(b["x1"]), int(b["y2"])
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y2 + 2), (x1 + tw + 6, y2 + th + 8), color, -1)
    cv2.putText(img, txt, (x1 + 3, y2 + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)


def _hud_panel(img, lines: list[str], alert: bool = False) -> None:
    """Bordered status panel, top-left - the fall-detection-reference HUD
    (system name, persons in view, flagged count). Red border on alert."""
    import cv2
    lh, pad = 18, 8
    w = 240
    h = pad * 2 + lh * len(lines) - 4
    x0, y0 = 8, 8
    roi = img[y0:y0 + h, x0:x0 + w]
    roi[:] = (roi * 0.25).astype(img.dtype)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h),
                  (0, 0, 230) if alert else (160, 160, 160), 2)
    y = y0 + pad + 8
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x0 + 8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48 if i == 0 else 0.42,
                    (255, 255, 255) if i == 0 else (210, 210, 210),
                    1, cv2.LINE_AA)
        y += lh


def _alert_banner(img, txt: str) -> None:
    """Loud red banner, top-center - fires only while an alert-grade flag
    (fall/erratic) is live, exactly like the operator's reference clip."""
    import cv2
    H, W = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    x0 = max(8, (W - tw) // 2 - 12)
    y0 = 8
    cv2.rectangle(img, (x0, y0), (min(W - 8, x0 + tw + 24), y0 + th + 18),
                  (0, 0, 210), -1)
    cv2.putText(img, txt, (x0 + 12, y0 + th + 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
                cv2.LINE_AA)


def draw_paths_layer(img, tracks, last_boxes: list[dict],
                     stats_by_id: dict):
    """Trails + id boxes + km/h chips - the one layer that legitimately
    shows detection boxes for every class."""
    import cv2
    for tr in tracks:
        # A track in miss state has no current match - its trail floating
        # over vacated pixels reads as ghost spaghetti. Draw matched only.
        if tr.misses:
            continue
        # Trail shares the track's stable color (2026-08-24) so box,
        # trail and side card all pair by one hue.
        color = track_color(tr.tid)
        pts = [(int((b["x1"] + b["x2"]) / 2), int((b["y1"] + b["y2"]) / 2))
               for b in tr.boxes[-TRAIL_MAX_PTS:]]
        for p0, p1 in zip(pts, pts[1:]):
            cv2.line(img, p0, p1, color, 2, cv2.LINE_AA)
        if pts:
            cv2.circle(img, pts[0], 4, color, -1, cv2.LINE_AA)
    for b in last_boxes:
        # Per-track colored box + id/class tag (replaces the class-color
        # draw_boxes pass, operator direction 2026-08-24).
        _tid = b.get("track_id", b.get("tid"))
        col = track_color(_tid)
        x1, y1 = int(b.get("x1", 0)), int(b.get("y1", 0))
        cv2.rectangle(img, (x1, y1),
                      (int(b.get("x2", 0)), int(b.get("y2", 0))), col, 2)
        tag = f"#{_tid} {b.get('cls', '')}".strip()
        cv2.putText(img, tag, (x1 + 4, max(y1 + 16, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2, cv2.LINE_AA)
    for b in last_boxes:
        s = stats_by_id.get(b.get("track_id"))
        # km/h honesty gate (audit 2026-08-14: moving bikes chipped
        # "2.3 km/h"): below ~8 km/h the class-length ruler at sampled
        # ticks is inside its own noise band, and a short track has no
        # statistical mass - show nothing rather than a wrong number.
        if (s and s.get("kmh_est") and s["kmh_est"] >= 8
                and int(s.get("sightings") or 0) >= 5):
            _chip(img, b, f"{s['kmh_est']} km/h", (90, 90, 90))
    note = (f"Paths & speeds - {len(last_boxes)} tracked now"
            if last_boxes else "Paths & speeds - nothing tracked yet")
    return _caption(img, [note])


def draw_fire_layer(img, hits: list[dict], confirmed: bool,
                    model_err: str | None = None):
    """Bright-orange boxes on any fire/smoke detection + a top banner
    when the detection has been present for FIRE_CONFIRM_TICKS in a
    row. When the dedicated fire model failed to load, `model_err`
    surfaces in the caption instead of a silent empty frame."""
    import cv2
    for h in (hits or []):
        p1 = (int(h["x1"]), int(h["y1"]))
        p2 = (int(h["x2"]), int(h["y2"]))
        # Bright saturated orange (BGR 20,140,255) - stands out against
        # both night and day scenes without collinding with the tracker's
        # cyan / green / red palette used by other layers.
        cv2.rectangle(img, p1, p2, (20, 140, 255), 3, cv2.LINE_AA)
        label = f"{h.get('cls', 'fire')} {float(h.get('conf', 0)):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      0.55, 2)
        ty = max(th + 6, p1[1] - 4)
        cv2.rectangle(img, (p1[0], ty - th - 6),
                      (p1[0] + tw + 8, ty + 2), (30, 30, 30), -1)
        cv2.putText(img, label, (p1[0] + 4, ty - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 200, 255), 2,
                    cv2.LINE_AA)
    if confirmed:
        _alert_banner(img, "FIRE DETECTED")
    if model_err:
        note = f"Fire detection - {model_err}"
    elif not hits:
        note = "Fire detection - no fire/smoke in view"
    elif not confirmed:
        note = (f"Fire detection - {len(hits)} candidate(s), "
                f"awaiting {FIRE_CONFIRM_TICKS}-tick confirmation")
    else:
        note = (f"Fire detection - {len(hits)} confirmed hit(s) "
                f"(alert active)")
    return _caption(img, [note])


def draw_zones_layer(img, entries: list[dict], kind: str):
    """Polygons + occupancy caption for the loiter / parking layers - the
    JPEG-fallback rendering; the canvas overlay is the primary view.

    Semantics that distinguish "zones & loitering" from "line crossing":

    * geometry - loiter is POLYGONAL (any convex/concave shape drawn on
      the frame), line crossing is a single ORIENTED SEGMENT. A closed
      polygon can gate an alcove or shopfront a straight line cannot;
    * signal - loiter fires on DWELL (the person has been inside for
      >= dwell_s seconds), line crossing fires on TRAJECTORY (a track
      that transitioned from one side of the line to the other, once
      per direction);
    * alert cardinality - loiter alerts ONCE per (track, zone) while the
      dwell exceeds threshold, and clears when the person leaves; line
      crossings increment per crossing (a single track can cross N times
      and be counted N times).

    If a customer only cares about counting foot traffic past a threshold,
    line crossing is the right layer. Zones + loitering is the right
    layer when the question is "who lingered where, for how long".
    """
    import cv2
    import numpy as np
    H, W = img.shape[:2]
    overlay = img.copy()
    for e in entries:
        pts = np.array([[int(p[0] * W), int(p[1] * H)]
                        for p in e["points"]], dtype=np.int32)
        if kind == "parking":
            occ = bool(e.get("occupied"))
            person_alert = bool(e.get("person_alert"))
            hot = occ or person_alert
            if person_alert:
                label = f"{e['name']}: PERSON INSIDE (alert)"
            else:
                label = f"{e['name']}: {'occupied' if occ else 'free'}"
        else:
            hot = bool(e.get("alert"))
            person_alert = False
            label = (f"{e['name']}: {e.get('count', 0)} inside"
                     f", max {int(e.get('max_dwell', 0))}s")
        # Person-in-parking uses a distinct RED (not the occupied-vehicle
        # RED) so operators can tell "car parked" from "someone snooping
        # around the parked car" at a glance.
        if person_alert:
            color = (0, 0, 255)      # bright red, more saturated than occ
        elif hot:
            color = (0, 0, 220)
        else:
            color = (0, 200, 80)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
        x0, y0 = int(pts[0][0]), int(pts[0][1])
        cv2.putText(img, label, (x0, max(14, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
    if not entries:
        note = (f"{'Parking' if kind == 'parking' else 'Zone & loitering'}"
                " - nothing drawn yet (use the Draw zones button)")
    elif kind == "parking":
        occ = sum(1 for e in entries if e.get("occupied"))
        person_alerts = sum(1 for e in entries if e.get("person_alert"))
        auto_n = sum(1 for e in entries if e.get("auto"))
        source = ("auto-detected" if auto_n == len(entries)
                  else "operator-drawn" if auto_n == 0
                  else f"{auto_n} auto + {len(entries) - auto_n} manual")
        alert_tail = (f" | PERSON ALERT in {person_alerts} spot(s)"
                      if person_alerts else "")
        note = (f"Parking - {occ}/{len(entries)} occupied ({source}; "
                f"state flips emit events){alert_tail}")
    else:
        note = (f"Zone & loitering - "
                f"{sum(e.get('count', 0) for e in entries)} inside, "
                f"{sum(1 for e in entries if e.get('alert'))} alert(s) "
                "(sustained presence in polygon; body-anomalies is the "
                "kinematic-per-person layer, this one is region-based)")
    return _caption(img, [note])


def draw_pose_layer(img, boxes: list[dict]):
    """Skeletons on people close enough for the per-crop pose pass, each
    person framed in their stable track color with a #id tag (operator
    direction 2026-08-24: unique color pairs the object with its side
    card). No vehicles - fix 2's core layer-correctness complaint."""
    import cv2
    from app.pose import draw_skeleton
    persons = [b for b in boxes if b.get("cls") == "person"]
    withk = [b for b in persons if b.get("kps")]
    for b in persons:
        # Colored box only - the #id lives on the side card, not inside
        # the video (operator direction 2026-08-24: no double labeling).
        col = track_color(b.get("tid"))
        x1, y1 = int(b.get("x1", 0)), int(b.get("y1", 0))
        x2, y2 = int(b.get("x2", 0)), int(b.get("y2", 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
    if withk:
        draw_skeleton(img, withk)
    if not persons:
        note = "Pose & skeleton - no people in frame"
    elif not withk:
        note = (f"Pose & skeleton - no skeletons "
                f"({len(persons)} people too far/small for pose)")
    else:
        note = (f"Pose & skeleton - skeletons on {len(withk)} "
                f"of {len(persons)} people")
        if len(withk) < len(persons):
            note += " (rest too far)"
    return _caption(img, [note])


def draw_plates_layer(img, boxes: list[dict]):
    """Vehicle boxes + plate strings. GREEN = read succeeded (OCR text
    shown); AMBER = in plate range (pipeline recognises the vehicle as
    a candidate) but OCR has not landed a read yet. Amber gives the
    operator visual feedback that the LPR stage-1 detection is working
    even when stage-2 OCR fails on hard frames."""
    import cv2
    from app.plates import MIN_VEHICLE_W, PLATE_VEHICLE_CLASSES
    veh = [b for b in boxes if b.get("cls") in PLATE_VEHICLE_CLASSES]
    read = [b for b in veh if b.get("plate")]
    # AMBER (thin) - in plate range, no read yet
    for b in veh:
        if b.get("plate"):
            continue  # green drawing below wins
        w_px = int(b.get("x2", 0)) - int(b.get("x1", 0))
        if w_px < MIN_VEHICLE_W:
            continue
        p1 = (int(b["x1"]), int(b["y1"]))
        p2 = (int(b["x2"]), int(b["y2"]))
        cv2.rectangle(img, p1, p2, (0, 165, 255), 1)  # amber (BGR)
        tag = f"{b.get('cls','?')} {b.get('conf', 0):.2f}  no-read"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX,
                                      0.4, 1)
        ty = max(th + 4, p1[1] - 2)
        cv2.rectangle(img, (p1[0], ty - th - 4), (p1[0] + tw + 4, ty + 1),
                      (24, 30, 44), -1)
        cv2.putText(img, tag, (p1[0] + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1,
                    cv2.LINE_AA)
    # GREEN (thick) - read succeeded, OCR text shown
    for b in read:
        p1 = (int(b["x1"]), int(b["y1"]))
        p2 = (int(b["x2"]), int(b["y2"]))
        cv2.rectangle(img, p1, p2, (80, 220, 80), 2)
        label = f"{b['plate']} {b.get('plate_conf', 0):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      0.55, 2)
        ty = max(th + 6, p1[1] - 4)
        cv2.rectangle(img, (p1[0], ty - th - 6), (p1[0] + tw + 8, ty + 2),
                      (30, 30, 30), -1)
        cv2.putText(img, label, (p1[0] + 4, ty - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 120), 2,
                    cv2.LINE_AA)
    in_range = sum(1 for b in veh
                   if (b["x2"] - b["x1"]) >= MIN_VEHICLE_W)
    if not veh:
        note = "License plates - no vehicles in frame"
    elif not in_range:
        note = (f"License plates - {len(veh)} vehicles, all too far "
                f"for plate read (<{MIN_VEHICLE_W}px)")
    else:
        note = (f"License plates - {len(read)} read / {in_range} in "
                f"range / {len(veh)} vehicles")
    return _caption(img, [note])


def draw_gestures_layer(img, boxes: list[dict], stats_by_id: dict,
                        session_counts: dict | None = None):
    """Skeleton + gesture chip only for people with a DETECTED gesture.

    Two gesture families since 2026-08-24: arm poses from the skeleton
    (hand raised / both hands up / wave) and hand-landmark verdicts from
    app.hands stamped on the box (`hand_gesture`: open_palm / fist /
    pointing + `hand_dir`)."""
    from app.pose import draw_skeleton
    _HAND_WORDS = {"open_palm": "OPEN PALM", "fist": "FIST",
                   "pointing": "POINTING"}
    active = []
    for b in boxes:
        if b.get("cls") != "person" or not b.get("kps"):
            continue
        s = stats_by_id.get(b.get("track_id"))
        labels = list((s or {}).get("gestures") or ())
        hg = b.get("hand_gesture")
        if hg:
            word = _HAND_WORDS.get(hg, hg.upper())
            if b.get("hand_dir"):
                word += f" {b['hand_dir']}"
            labels.append(word)
        if labels:
            active.append((b, s or {}, labels))
    for b, s, labels in active:
        draw_skeleton(img, [b])
        _chip(img, b, "+".join(labels), (190, 120, 0))
    note = ("Hand gestures - "
            + ", ".join(f"#{s.get('id', b.get('track_id', '?'))} "
                        f"{'+'.join(labels)}"
                        for b, s, labels in active)
            if active else "Hand gestures - none detected right now")
    lines = [note]
    if session_counts is not None:
        tot = ", ".join(f"{g} x{n}"
                        for g, n in sorted(session_counts.items()))
        lines.append(f"session: {tot}" if tot else "session: none yet")
    return _caption(img, lines)


def draw_body_layer(img, boxes: list[dict], stats_by_id: dict,
                    sudden_tids: set | None = None):
    """Body-anomaly view (2026-08-16):
    * every person still gets their detection box drawn so operators can
      see the scene, but the box color STANDS OUT only when flagged;
    * FAST/SUDDEN motion (wrist/ankle burst - theft, escape, punch)
      draws a red HALO circle around the person and a "SUDDEN MOTION"
      tag - a persistent verdict caption, not just a count in the HUD;
    * behavior-label flags (fall_suspect / erratic / running / etc)
      keep their box + skeleton + chip;
    * an ALERT banner burns while any alert-grade flag is live.
    Normal street life stays unmarked (grey box, no tag)."""
    import cv2
    sudden = set(sudden_tids or ())
    persons = [b for b in boxes if b.get("cls") == "person"]
    flagged = []
    for b in persons:
        s = stats_by_id.get(b.get("track_id"))
        is_sudden = b.get("track_id") in sudden
        if s and (s.get("label") in BODY_ANOMALY_LABELS
                  or s.get("pose_flags")):
            flagged.append((b, s, is_sudden))
        elif is_sudden:
            flagged.append((b, {}, True))
    for b in persons:
        # Stable per-track color (operator direction 2026-08-24) so each
        # person pairs with their side card by color; the #id lives on
        # the card only, and flagged persons get the red overlay on top.
        col = track_color(b.get("track_id", b.get("tid")))
        cv2.rectangle(img, (int(b["x1"]), int(b["y1"])),
                      (int(b["x2"]), int(b["y2"])), col, 2)
    # Skeletons on EVERY person with keypoints (operator direction
    # 2026-08-24): the anomaly verdict is judged visually against the
    # full-scene pose picture, not only on flagged tracks.
    _withk = [b for b in persons if b.get("kps")]
    if _withk:
        from app.pose import draw_skeleton
        draw_skeleton(img, _withk)
    for b, s, is_sudden in flagged:
        color = (0, 0, 220) if (s.get("alert") or is_sudden) else (0, 150, 230)
        cv2.rectangle(img, (int(b["x1"]), int(b["y1"])),
                      (int(b["x2"]), int(b["y2"])), color, 3)
        if is_sudden:
            # Red halo (wide anti-aliased circle around the person) so a
            # sudden-motion flag is unmistakable at a glance.
            cx = int((b["x1"] + b["x2"]) / 2)
            cy = int((b["y1"] + b["y2"]) / 2)
            r_halo = int(max(b["x2"] - b["x1"], b["y2"] - b["y1"]) * 0.75)
            cv2.circle(img, (cx, cy), r_halo, (0, 0, 220), 4,
                       cv2.LINE_AA)
            cv2.circle(img, (cx, cy), r_halo + 4, (0, 0, 120), 1,
                       cv2.LINE_AA)
        parts = [f"#{s.get('id', b.get('track_id', '?'))}"]
        if is_sudden:
            parts.append("SUDDEN MOTION")
        if s.get("label"):
            parts.append(str(s["label"]).upper())
        extra = [f for f in (s.get("pose_flags") or [])
                 if f and f != s.get("label")]
        if extra:
            parts.append("+".join(extra))
        _chip(img, b, " ".join(parts), color)
    alerts = [s for _, s, _ in flagged if s.get("alert")]
    sudden_count = sum(1 for _, _, sud in flagged if sud)
    with_kps = sum(1 for b in persons if b.get("kps"))
    _hud_panel(img, ["BODY ANOMALIES",
                     f"persons in view: {len(persons)} ({with_kps} w/ pose)",
                     f"flagged: {len(flagged)}"
                     + (f" ({sudden_count} sudden)" if sudden_count
                        else "" if flagged else " (none right now)"),
                     ("watching: fast punches/kicks (pose), "
                      "sudden displacement (bbox fallback)")],
               alert=bool(alerts or sudden_count))
    if alerts or sudden_count:
        kinds: dict[str, int] = {}
        for s in alerts:
            k = (s.get("label") or "?").upper().replace("_", " ")
            kinds[k] = kinds.get(k, 0) + 1
        if sudden_count:
            kinds["SUDDEN MOTION"] = sudden_count
        _alert_banner(img, "ALERT! " + ", ".join(
            f"{n} {k}" for k, n in sorted(kinds.items())))
    return img


def draw_faces_layer_img(img, faces_list: list[dict], available: bool = True):
    if faces_list:
        from app.faces import draw_faces
        draw_faces(img, faces_list)
        note = f"Face detection - {len(faces_list)} face(s)"
    elif available:
        note = "Face detection - no faces at this distance/resolution"
    else:
        note = "Face detection - face model not available on this machine"
    return _caption(img, [note])


def draw_heat_layer(img, grid: list, since: float | None = None):
    """FULL-FRAME thermal recolor + hot-blob overlay for the heat layer.

    2026-08-17: operator explicitly asked for the entire frame to
    become a thermal view (like a real thermal-camera image), with
    warm blobs on top marking accumulated dwell. `overlay_thermal`
    (heatmap.py) does the two-layer composite: INFERNO base over the
    frame's luminance, TURBO override where dwell signal is present.
    The result reads unmistakably as "thermal" even before any dwell
    accumulates - no more silent photo-with-nothing-on-it."""
    from app.heatmap import overlay_thermal
    out = overlay_thermal(grid, base_frame=img)
    if since:
        el = int(time.time() - since)
        mm, ss = divmod(el, 60)
        note = (f"Heat signature - dwell accumulating since "
                f"{time.strftime('%H:%M:%S', time.localtime(since))} "
                f"({mm}m{ss:02d}s)")
    else:
        note = "Heat signature - dwell over this window"
    peak = max((max(row) for row in grid), default=0.0)
    # Diagnostic: show peak + non-zero cell count on the frame so the
    # operator can see whether backend accumulation is happening (peak
    # rising over time = working). 2026-08-17: the layer had been
    # reported "not working"; making the state visible in the JPEG
    # caption itself decouples backend-accumulation debugging from
    # frontend-canvas rendering.
    nonzero = sum(1 for row in grid for v in row if v > 0)
    note += f" | peak={peak:.2f}, nonzero_cells={nonzero}"
    if peak <= 0:
        note += " (no activity banked yet - wait for people to appear in frame)"
        return _caption(out, [note])
    # Verbal legend (operator request 2026-08-24): name WHERE the
    # hottest concentration sits, as a 3x3 compass region of the frame.
    py, px, best = 0, 0, -1.0
    for gy, row in enumerate(grid):
        for gx, v in enumerate(row):
            if v > best:
                py, px, best = gy, gx, v
    rows_n, cols_n = max(len(grid), 1), max(len(grid[0]), 1)
    vert = ("top", "center", "bottom")[min(2, py * 3 // rows_n)]
    horz = ("left", "center", "right")[min(2, px * 3 // cols_n)]
    region = "center" if (vert, horz) == ("center", "center") \
        else f"{vert}-{horz}"
    return _caption(out, [note, f"hottest concentration: {region} of frame"])


def draw_line_layer(img, line: list, cross: dict):
    """Counting line + BOTTOM-LEFT IN/OUT counters (2026-08-17).

    Placement history: originally near the line midpoint (invisible under
    YouTube hover controls); then top-center (competed with the layer
    title strip and any browser overlay chrome). Now bottom-left in two
    separate color-coded pills - IN in a green pill, OUT in a red pill -
    so the operator can scan the state at a glance without the readout
    fighting the tile's own Stop / Draw-line control row above the frame.
    Each pill sits its own gap apart so the two counters read as two
    distinct channels rather than one glyph blob.
    """
    import cv2
    H, W = img.shape[:2]
    (ax, ay), (bx, by) = line
    p0 = (int(ax * W), int(ay * H))
    p1 = (int(bx * W), int(by * H))
    cv2.line(img, p0, p1, (0, 215, 255), 3, cv2.LINE_AA)
    for p in (p0, p1):
        cv2.circle(img, p, 6, (0, 215, 255), -1, cv2.LINE_AA)
    # Direction hint - a small arrowhead near B pointing perpendicular
    # to A->B, on the "positive" (IN) side. Line dir is A->B; rotate
    # 90 degrees anti-clockwise for the IN-normal. Helps the operator
    # see which side the "in" count corresponds to before crossings
    # start accumulating.
    dxl, dyl = (bx - ax) * W, (by - ay) * H
    mag = (dxl * dxl + dyl * dyl) ** 0.5
    if mag > 8:
        nx, ny = -dyl / mag, dxl / mag        # rotate +90 (screen-space)
        cx, cy = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        tip = (int(cx + nx * 22), int(cy + ny * 22))
        base = (int(cx), int(cy))
        cv2.arrowedLine(img, base, tip, (0, 200, 60), 2, cv2.LINE_AA,
                        tipLength=0.35)
    n_in = int(cross.get("in", 0))
    n_out = int(cross.get("out", 0))
    # Font size scales with frame width (matches the previous top-center
    # readout so the numbers stay readable on both small embeds and
    # dashboard-full views).
    fs = max(0.7, min(1.4, W / 960.0 * 1.05))
    thick = 2 if fs >= 1.0 else 2
    txt_in = f"IN {n_in}"
    txt_out = f"OUT {n_out}"
    (tw_i, th_i), _ = cv2.getTextSize(txt_in, cv2.FONT_HERSHEY_SIMPLEX,
                                       fs, thick)
    (tw_o, th_o), _ = cv2.getTextSize(txt_out, cv2.FONT_HERSHEY_SIMPLEX,
                                       fs, thick)
    pad_x, pad_y = 10, 6
    gap = 8
    pill_h_i = th_i + pad_y * 2
    pill_h_o = th_o + pad_y * 2
    pill_h = max(pill_h_i, pill_h_o)
    # Bottom-left anchor - stack IN pill above OUT pill so each reads
    # in its own color band.
    margin = 12
    y_bot = H - margin
    y_top_out = y_bot - pill_h
    y_top_in = y_top_out - gap - pill_h
    x_left = margin
    # IN pill (green fill, white text).
    cv2.rectangle(img,
                  (x_left, y_top_in),
                  (x_left + tw_i + pad_x * 2, y_top_in + pill_h),
                  (0, 165, 45), -1)
    cv2.rectangle(img,
                  (x_left, y_top_in),
                  (x_left + tw_i + pad_x * 2, y_top_in + pill_h),
                  (0, 90, 25), 1, cv2.LINE_AA)
    cv2.putText(img, txt_in,
                (x_left + pad_x, y_top_in + pad_y + th_i),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), thick,
                cv2.LINE_AA)
    # OUT pill (red fill, white text).
    cv2.rectangle(img,
                  (x_left, y_top_out),
                  (x_left + tw_o + pad_x * 2, y_top_out + pill_h),
                  (55, 55, 205), -1)
    cv2.rectangle(img,
                  (x_left, y_top_out),
                  (x_left + tw_o + pad_x * 2, y_top_out + pill_h),
                  (25, 25, 90), 1, cv2.LINE_AA)
    cv2.putText(img, txt_out,
                (x_left + pad_x, y_top_out + pad_y + th_o),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), thick,
                cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# The live session.
# ---------------------------------------------------------------------------

def _segments_intersect(p1, p2, q1, q2) -> bool:
    """True when finite segments p1-p2 and q1-q2 properly intersect.
    Standard orientation test; collinear grazing counts as a miss (a
    foot point sliding ALONG the line is not a crossing)."""
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)
    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)
    return o1 != o2 and o3 != o4 and 0 not in (o1, o2, o3, o4)


def _clip_poly_by_halfplane(poly, a, b):
    """Sutherland-Hodgman step: keep the part of `poly` left of a->b."""
    def side(p):
        return ((b[0] - a[0]) * (p[1] - a[1])
                - (b[1] - a[1]) * (p[0] - a[0]))
    out = []
    n = len(poly)
    for i in range(n):
        cur, nxt = poly[i], poly[(i + 1) % n]
        sc, sn = side(cur), side(nxt)
        if sc >= 0:
            out.append(cur)
        if (sc >= 0) != (sn >= 0):
            t = sc / (sc - sn)
            out.append((cur[0] + t * (nxt[0] - cur[0]),
                        cur[1] + t * (nxt[1] - cur[1])))
    return out


def _poly_area(poly) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def box_overlap_over_spot(box_norm, spot_pts) -> float:
    """area(box INTERSECT spot) / area(spot), all in normalized coords.

    The industry association metric for parking (IoU/overlap thresholds
    0.15-0.5 in the PKLot/Frigate/Roboflow lineage) - a vehicle CENTER
    inside a polygon is how a shopfront ends up 'occupied' by a passing
    bike; substantial areal overlap is much harder to fake."""
    spot = [(float(p[0]), float(p[1])) for p in spot_pts]
    x1, y1, x2, y2 = box_norm
    # Clip the SPOT by the box's four half-planes (box is convex).
    for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                 ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        spot = _clip_poly_by_halfplane(spot, a, b)
        if not spot:
            return 0.0
    denom = _poly_area([(float(p[0]), float(p[1])) for p in spot_pts])
    return (_poly_area(spot) / denom) if denom > 1e-9 else 0.0
