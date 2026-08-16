"""Gate a trained head against the baseline; promote it only if it earns it.

    python -m tools.promote_adapter --candidate data/adapters/head_X.pt
    python -m tools.promote_adapter --candidate ... --publish   # + upload
    python -m tools.promote_adapter --rollback                  # undo last

Both models (plain base and base+candidate head, yolov8n by default - the
VM's pinned weights) are validated on the SAME split: the chronological val
slice tools/export_labels.py wrote. Gate
(plan WS3): mAP50 must improve by >= +0.5pp AND no class may drop more than
2pp - person and car may not drop at all. Every verdict (promoted or
rejected) is appended to data/adapters/history.jsonl, so the "it rejected a
regressing run" evidence the plan's DoD asks for is a grep away.

``--publish`` uploads the promoted head + pointer + history to Firebase
Storage under ``training/`` (needs FIREBASE_CREDENTIALS +
FIREBASE_STORAGE_BUCKET, or the auto-detected key on the operator's
machine) - that is what the VM's hot-load polls.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent


def _val_metrics(model, data_yaml: str, imgsz: int) -> dict:
    """Run ultralytics val, reduce to the gate's shape:
    {"map50": float, "per_class": {cls: ap50}}."""
    m = model.val(data=data_yaml, imgsz=imgsz, device="cpu", plots=False,
                  verbose=False)
    per_class: dict[str, float] = {}
    try:
        names = model.names or {}
        # NOTE: `arr or []` on a numpy array raises "truth value of an
        # array is ambiguous" - which the old except swallowed, silently
        # emptying per_class and DISABLING the per-class regression gate
        # (person/car zero-drop was never enforced). Length-check instead.
        idxs = getattr(m.box, "ap_class_index", None)
        ap50 = getattr(m.box, "ap50", None)
        if idxs is not None and ap50 is not None:
            for ci, ap in zip(list(idxs), list(ap50)):
                per_class[str(names.get(int(ci), ci))] = round(float(ap), 4)
    except Exception as e:
        print(f"promote: per-class metrics unavailable "
              f"({type(e).__name__}: {e})")
    return {"map50": round(float(m.box.map50), 4), "per_class": per_class}


def _bucket_name_from_web_config() -> str | None:
    """storageBucket out of web/firebase-config.js - inlined (no app.pool_sync
    import) so the slim CI env never drags that module's dependency chain
    into the publish path."""
    import re
    cfg = _SRC_ROOT / "web" / "firebase-config.js"
    try:
        m = re.search(r"storageBucket:\s*[\"']([^\"']+)[\"']", cfg.read_text())
        return m.group(1) if m else None
    except OSError:
        return None


def _storage_bucket():
    """Firebase Storage bucket via the Admin SDK (write access needed)."""
    import firebase_admin
    from firebase_admin import credentials, storage
    cred = os.environ.get("FIREBASE_CREDENTIALS")
    bucket = (os.environ.get("FIREBASE_STORAGE_BUCKET")
              or _bucket_name_from_web_config())
    if not cred or not Path(cred).is_file():
        raise SystemExit("--publish needs FIREBASE_CREDENTIALS pointing at "
                         "the service-account json")
    if not bucket:
        raise SystemExit("--publish: no storage bucket (env or "
                         "web/firebase-config.js)")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(cred),
                                      {"storageBucket": bucket})
    # Pass the name EXPLICITLY: when the training_events mirror ran first
    # it already initialized the app WITHOUT a storageBucket option, and a
    # bare storage.bucket() then raises "bucket name not specified" - that
    # collision silently killed the history upload on every run since the
    # mirror shipped (runs 2-4) while run 1, which predates it, published.
    return storage.bucket(bucket)


def main() -> None:
    from app import adapters

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate", default=None,
                    help="head artifact from tools/train_head.py")
    ap.add_argument("--data", default=str(_SRC_ROOT / "data" / "labels_export"
                                          / "dataset.yaml"))
    ap.add_argument("--base", default="yolov8s.pt",
                    help="same base the head was trained on AND the VM runs "
                         "(pinned in deploy/gcp-vm/collector.service)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--adapters-dir", default=str(adapters.ADAPTERS_DIR))
    ap.add_argument("--publish", action="store_true",
                    help="on promotion, upload head+pointer+history to "
                         "Storage training/")
    ap.add_argument("--rollback", action="store_true",
                    help="restore the previous promoted head and exit")
    args = ap.parse_args()
    adir = Path(args.adapters_dir)

    if args.rollback:
        entry = adapters.rollback(adir)
        print(f"rollback -> {entry['file'] if entry else 'base model (no pointer)'}")
        if args.publish:
            n = adapters.publish_to_storage(_storage_bucket(), adir)
            print(f"published {n} object(s) to Storage training/")
        return

    if not args.candidate:
        raise SystemExit("--candidate is required (or --rollback)")
    cand_path = Path(args.candidate)
    if not cand_path.is_file():
        raise SystemExit(f"candidate not found: {cand_path}")
    if not Path(args.data).is_file():
        raise SystemExit(f"dataset yaml not found: {args.data} - run "
                         f"tools/export_labels first")

    from ultralytics import YOLO

    print(f"promote: validating BASELINE {args.base} on {args.data}")
    base_model = YOLO(args.base)
    base_metrics = _val_metrics(base_model, args.data, args.imgsz)
    print(f"  baseline: mAP50={base_metrics['map50']} "
          f"per-class={base_metrics['per_class']}")

    print(f"promote: validating CANDIDATE {cand_path.name}")
    cand_model = YOLO(args.base)
    adapters.overlay_head(cand_model.model, adapters.load_head(cand_path))
    cand_metrics = _val_metrics(cand_model, args.data, args.imgsz)
    print(f"  candidate: mAP50={cand_metrics['map50']} "
          f"per-class={cand_metrics['per_class']}")

    ok, reasons = adapters.gate_decision(base_metrics, cand_metrics)
    record = {
        "event": "gate",
        "candidate": cand_path.name,
        "base": Path(args.base).name,
        "promoted": ok,
        "labels_total": _count_labels(Path(args.data).parent),
        "baseline": base_metrics,
        "metrics": cand_metrics,
        "reasons": reasons,
    }
    adapters.append_history(record, adir)
    _mirror_training_event(record)

    if not ok:
        print("REJECTED:")
        for r in reasons:
            print(f"  - {r}")
        if args.publish:
            # The reject verdict still belongs in the cumulative cloud
            # history (D9 full retention); pointer and head stay untouched.
            # A history-upload hiccup must NOT fail the run - the verdict
            # is already in history.jsonl + the run artifact, and the next
            # run re-publishes the cumulative history anyway.
            try:
                adapters.publish_to_storage(_storage_bucket(), adir,
                                            history_only=True)
            except BaseException as e:
                import traceback
                traceback.print_exc()
                print(f"promote: history publish failed ({type(e).__name__}) "
                      f"- verdict preserved locally, continuing")
        _gh_summary(record)
        return

    # Keep the artifact inside adapters_dir (full retention - D9).
    dest = adir / cand_path.name
    if dest.resolve() != cand_path.resolve():
        adir.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(cand_path.read_bytes())
    entry = adapters.promote(dest.name, cand_metrics, adir,
                             base=Path(args.base).name)
    print(f"PROMOTED {entry['file']}:")
    for r in reasons:
        print(f"  - {r}")
    if args.publish:
        # A PROMOTED head that fails to publish is a real failure - the VM
        # would never see it. Fail loudly with the full cause.
        try:
            n = adapters.publish_to_storage(_storage_bucket(), adir)
        except BaseException:
            import traceback
            traceback.print_exc()
            raise SystemExit("promote: PUBLISH FAILED for a promoted head - "
                             "the VM cannot hot-load it; see traceback above")
        print(f"published {n} object(s) to Storage training/ - the VM "
              f"hot-loads it within a few rounds")
    _gh_summary(record)


def _count_labels(export_dir: Path) -> int | None:
    """Label files in the export (train + val) - the X axis of the WS5
    labels-vs-quality curve. None when the layout isn't there."""
    try:
        n = sum(1 for split in ("train", "val")
                for _ in (export_dir / "labels" / split).glob("*.txt"))
        return n or None
    except OSError:
        return None


def _mirror_training_event(record: dict) -> None:
    """Best-effort copy of the gate record into Firestore `training_events`
    (D7: one write per training run) so the HOSTED dashboard can render the
    AL curve without the operator's disk. Quiet no-op without credentials."""
    try:
        import datetime as _dt

        import firebase_admin
        from firebase_admin import credentials, firestore
        cred = os.environ.get("FIREBASE_CREDENTIALS")
        if not cred or not Path(cred).is_file():
            return
        if not firebase_admin._apps:
            firebase_admin.initialize_app(credentials.Certificate(cred))
        doc = {**record,
               "at": _dt.datetime.now(_dt.timezone.utc)
               .strftime("%Y-%m-%dT%H:%M:%SZ"),
               "expire_at": _dt.datetime.now(_dt.timezone.utc)
               + _dt.timedelta(days=30)}
        firestore.client().collection("training_events").add(doc)
        print("promote: mirrored gate record -> Firestore training_events")
    except Exception as e:
        print(f"promote: training_events mirror skipped "
              f"({type(e).__name__}: {e})")


def _gh_summary(record: dict) -> None:
    """One table row in the GitHub Actions job summary, when running there."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        verdict = "PROMOTED" if record["promoted"] else "REJECTED"
        lines = [f"## Adapter gate: {verdict}",
                 f"- candidate: `{record['candidate']}`",
                 f"- baseline mAP50: {record['baseline']['map50']}",
                 f"- candidate mAP50: {record['metrics']['map50']}",
                 "- reasons:"]
        lines += [f"  - {r}" for r in record["reasons"]]
        lines += ["", "<details><summary>per-class AP50</summary>", "",
                  "```json",
                  json.dumps({"baseline": record["baseline"]["per_class"],
                              "candidate": record["metrics"]["per_class"]},
                             indent=1),
                  "```", "</details>"]
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass


if __name__ == "__main__":
    main()
