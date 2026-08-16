"""Pose constants: the COCO-17 wiring every consumer relies on.

The full-frame pose pass and its IoU pairing were deleted in fix 3 -
the per-crop pass (attach_keypoints_crops) is the only production path,
and it claims boxes by crop ownership, not IoU. What remains here are
the pure constants the gesture/label/drawing rules index into.

Run from src/:  python -m pytest tests -q
"""
from app.pose import KEYPOINT_NAMES, SKELETON


def test_keypoint_names_are_coco17():
    assert len(KEYPOINT_NAMES) == 17
    assert KEYPOINT_NAMES[0] == "nose"


def test_skeleton_indices_are_valid():
    assert all(0 <= i < 17 and 0 <= j < 17 for i, j in SKELETON)
    assert len(set(SKELETON)) == len(SKELETON)   # no duplicate bones
