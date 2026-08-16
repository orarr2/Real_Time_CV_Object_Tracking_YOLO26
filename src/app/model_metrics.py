"""Model-quality scoreboard, computed from user review verdicts.

The dashboard's "Model: X% accuracy · P(person) Y% · FP Z% · N reviews"
header line is driven by ``compute()`` in this module. All numbers come
from ``data/reviews.json`` (the ReviewStore) - nothing here reads live
inference, so a caller can trust it and cache it without invalidation
tricks.

Percent-metrics need a MINIMUM SAMPLE SIZE to be meaningful. 3/3 correct
is technically 100% but says nothing about the model - it says the user
happened to review three easy crops. Below ``MIN_REVIEWS_FOR_METRIC``
``header_line()`` reports N/A + a progress hint instead of a fabricated
percentage, so the dashboard never shows a trustworthy-looking 100%
against a two-digit sample. Same rule applies to per-class precision.

Definitions used here (crop-level, until the full-frame review UX ships
its own recall data):

  correct       = user said the label the model gave was right
  wrong         = user said the label was wrong (either wrong class OR
                  not an object at all)
  accuracy      = correct / (correct + wrong)
  precision[c]  = correct[c] / (correct[c] + wrong[c])
                  where ``c`` is the class the model originally gave
  fp_rate       = wrong / (correct + wrong)
                  = 1 - accuracy in the crop-level model

`recall` is left off intentionally: without "the model missed this object
here" verdicts we cannot count FN. The full-frame review UX (Task 26)
adds a ``missed`` verdict, at which point ``compute()`` grows an
``recall`` field per class.
"""
from __future__ import annotations

# Below this many verdicts the % metrics are treated as unavailable in the
# UI (header_line reports N/A + progress). 20 balances "quick to reach"
# against "not laughably small" - the standard error on a proportion at
# n=20 is already ~10 pp, low enough that a 60% vs 90% distinction is
# real signal rather than coin flips.
MIN_REVIEWS_FOR_METRIC = 20
# Per-class precision needs its own minimum. Kept lower than the global
# threshold since a class-specific bar of 20 would keep every per-class
# metric hidden until well past 60 total reviews.
MIN_REVIEWS_FOR_PER_CLASS = 5


# Crop reviews under this prefix come from the shipped bootstrap fixtures,
# not from anything a production camera streamed. They exist so the review
# UI has material on a fresh install; scoring the model on them would let
# a demo image move the production accuracy number.
_DEMO_CROP_PREFIX = "live_samples/_demo/"


def compute(review_store) -> dict:
    """Aggregate crop-level AND frame-level verdicts into a scoreboard.

    Two verdict streams feed the numbers:
    * crop reviews  - one verdict per crop (legacy UI, still counted)
    * frame reviews - many verdicts per frame + explicit missed detections,
      which is where FN (and therefore recall / F1) come from.

    Honesty rules (each metric is gated by ITS OWN sample, see header_line):
    * precision/accuracy sample = tp + fp (verdicts on model boxes);
    * recall sample            = tp + fn (model boxes confirmed + missed
      objects the user drew) - a user who marked 17 misses has given real
      recall signal even when only 3 boxes got a verdict;
    * bootstrap ``_demo`` crops are excluded outright.
    """
    # --- crop verdicts (precision-only stream) -----------------------
    correct = 0
    wrong = 0
    demo_excluded = 0
    per_cls: dict[str, dict[str, int]] = {}
    for r in review_store._by_path.values():  # noqa: SLF001 - deliberate
        if (r.crop_path or "").startswith(_DEMO_CROP_PREFIX):
            demo_excluded += 1
            continue
        cls = r.original_cls or "?"
        rec = per_cls.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})
        if r.verdict == "correct":
            correct += 1
            rec["tp"] += 1
        else:
            wrong += 1
            rec["fp"] += 1

    # --- frame verdicts (adds FN → recall / F1) ----------------------
    frame_reviews = getattr(review_store, "_frames_by_path", {}).values()
    for fr in frame_reviews:
        meta_boxes_by_id = {}
        try:
            from app.review_frames import load_metadata
            meta = load_metadata(fr.frame_path)
            for b in (meta or {}).get("boxes", []):
                meta_boxes_by_id[str(b["id"])] = b.get("cls", "?")
        except Exception:
            pass
        for box_id, verdict in (fr.box_verdicts or {}).items():
            cls = meta_boxes_by_id.get(str(box_id), "?")
            rec = per_cls.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})
            if verdict == "correct":
                correct += 1; rec["tp"] += 1
            elif verdict in ("wrong", "object") or verdict.startswith("relabel:"):
                # relabel = real object, wrong class: a precision miss for
                # the class the model CLAIMED (which is what per_cls keys on).
                wrong += 1; rec["fp"] += 1
        for miss in (fr.missed_detections or ()):
            cls = miss.get("cls") or "?"
            rec = per_cls.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})
            rec["fn"] += 1

    total = correct + wrong          # precision-side sample (verdicts on boxes)
    accuracy = correct / total if total else None
    fp_rate  = wrong / total if total else None

    # Global recall / F1 - defined only when at least one frame review
    # has landed (so FN is a real count, not zero-by-omission).
    total_fn = sum(rec["fn"] for rec in per_cls.values())
    n_recall = correct + total_fn    # recall-side sample (confirmed + missed)
    if any(fr.missed_detections is not None for fr in frame_reviews) \
            or total_fn > 0:
        recall = correct / n_recall if n_recall else None
        precision = correct / (correct + wrong) if (correct + wrong) else None
        f1 = None
        if recall and precision and (recall + precision) > 0:
            f1 = 2 * precision * recall / (precision + recall)
    else:
        recall = None
        f1 = None

    classes = []
    for cls, rec in sorted(per_cls.items()):
        n = rec["tp"] + rec["fp"]
        p_c = rec["tp"] / n if n else None
        n_denom_r = rec["tp"] + rec["fn"]
        r_c = rec["tp"] / n_denom_r if n_denom_r > 0 else None
        classes.append({
            "cls":       cls,
            "n":         n,
            "precision": round(p_c, 4) if p_c is not None else None,
            "recall":    round(r_c, 4) if r_c is not None else None,
            "fn":        rec["fn"],
        })

    return {
        "total_reviews": total,
        "tp":            correct,
        "fp":            wrong,
        "fn":            total_fn,
        "n_precision":   total,
        "n_recall":      n_recall,
        "demo_excluded": demo_excluded,
        "accuracy":      round(accuracy, 4) if accuracy is not None else None,
        "fp_rate":       round(fp_rate,  4) if fp_rate  is not None else None,
        "recall":        round(recall, 4)   if recall   is not None else None,
        "f1":            round(f1, 4)       if f1       is not None else None,
        "per_class":     classes,
    }


def learning_curve(review_store, batch_size: int = 5) -> list[dict]:
    """Model mistake-rate per tagging batch, chronological - the operator's
    "is it actually getting better?" chart.

    Each reviewed frame contributes signals: per-box verdicts (wrong or
    relabel = a model mistake, correct = a model win) plus every
    operator-drawn miss (a mistake by definition). Frames are grouped into
    batches of `batch_size` in review order - matching the paced queue, so
    one chart point = one sitting's batch. A falling error_rate over
    batches is the improvement the operator asked to SEE.

    Honesty caveat carried to the UI: the rate also moves with how hard
    the sampled frames are; the uncertainty-first queue deliberately
    serves hard ones, so a plateau is not failure - a sustained rise is.
    """
    frs = sorted(getattr(review_store, "_frames_by_path", {}).values(),
                 key=lambda fr: fr.reviewed_at or "")
    points: list[dict] = []
    batch = {"frames": 0, "signals": 0, "mistakes": 0, "last": ""}

    def _flush() -> None:
        if not batch["frames"]:
            return
        denom = batch["signals"]
        points.append({
            "batch":            len(points) + 1,
            "frames":           batch["frames"],
            "signals":          denom,
            "mistakes":         batch["mistakes"],
            "error_rate":       round(batch["mistakes"] / denom, 4) if denom else None,
            "last_reviewed_at": batch["last"],
        })

    for fr in frs:
        signals = mistakes = 0
        for v in (fr.box_verdicts or {}).values():
            signals += 1
            if v in ("wrong", "object") or v.startswith("relabel:"):
                mistakes += 1
        miss = len(fr.missed_detections or ())
        signals += miss
        mistakes += miss
        if signals == 0:
            continue
        batch["frames"] += 1
        batch["signals"] += signals
        batch["mistakes"] += mistakes
        batch["last"] = fr.reviewed_at or ""
        if batch["frames"] >= batch_size:
            _flush()
            batch = {"frames": 0, "signals": 0, "mistakes": 0, "last": ""}
    _flush()
    return points


def header_line(metrics: dict, boost_summary: dict | None = None) -> str:
    """Human-readable one-line summary for the dashboard header.

    Kept in Python (not JS) so the same string can be logged, dumped in
    Firestore, and rendered by any client identically.

    Honesty rules:
    * The raw verdict counts (correct / wrong / missed) are ALWAYS shown -
      17 misses the user drew must be visible even before any % unlocks.
    * Each percentage is gated by ITS OWN sample size: precision by
      tp+fp, recall by tp+fn. A user who confirmed 3 boxes but marked 17
      misses has recall signal (n=20) and no precision signal (n=3) - the
      old single gate showed either both or neither.
    """
    tp = metrics.get("tp")
    fp_n = metrics.get("fp")
    fn = metrics.get("fn") or 0
    if tp is None:                       # pre-rework caller (tests, cache)
        tp = metrics.get("total_reviews") or 0
        fp_n = 0
    n_prec = metrics.get("n_precision")
    n_prec = (tp + (fp_n or 0)) if n_prec is None else n_prec
    n_rec = metrics.get("n_recall")
    n_rec = (tp + fn) if n_rec is None else n_rec

    if n_prec + fn == 0:
        return "Model: no feedback yet - review a few frames below to teach it"

    # Plain words, operator-first (2026-07 redesign): the old line
    # ("precision pending (3/20 verdicts) · recall 12%") answered none of
    # the operator's real questions - is it right? how often? is it
    # learning? Full statistical detail stays in the JSON for tooling.
    # ASCII only: the string is printed to Windows consoles (cp125x).
    parts = []
    right = f"right on {tp} of {n_prec} boxes you checked"
    acc = metrics.get("accuracy")
    if acc is not None and n_prec:
        pct = int(round(acc * 100))
        if n_prec >= MIN_REVIEWS_FOR_METRIC:
            right += f" ({pct}% accuracy)"
        else:
            # Always show the NUMBER (the operator asked for one), with the
            # sample-size truth attached: 3-of-3 is "100% so far", not "the
            # model is perfect" - the figure firms up at 20 checks.
            right += (f" ({pct}% so far - small sample, firm after "
                      f"{MIN_REVIEWS_FOR_METRIC} checks)")
    parts.append(right)
    if fn:
        parts.append(f"{fn} objects it missed are marked and queued for training")
    if boost_summary and boost_summary.get("adjusted_cls"):
        learn = (f"learning is ON - it self-adjusted "
                 f"{boost_summary['adjusted_cls']} detection thresholds "
                 f"from your feedback")
        upd = boost_summary.get("updated_at") or ""
        try:
            import calendar
            import time as _t
            mins = max(0, int((_t.time() - calendar.timegm(
                _t.strptime(upd, "%Y-%m-%dT%H:%M:%SZ"))) / 60))
            learn += (f" (last {mins} min ago)" if mins < 120
                      else f" (last {mins // 60} h ago)")
        except (ValueError, TypeError):
            pass
        parts.append(learn)
    else:
        parts.append("learning is ON - waiting for your first verdicts")
    return "Model: " + " · ".join(parts)
