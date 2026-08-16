"""Burst-median count aggregation.

Run from src/:  python -m pytest tests -q
"""
from app.detect_core import median_counts


def test_median_counts_odd():
    counts = median_counts([
        {"person": 5, "vehicles": 2},
        {"person": 7, "vehicles": 3},
        {"person": 6, "vehicles": 9},   # one glitchy frame can't move the result
    ])
    assert counts == {"person": 6, "vehicles": 3}


def test_median_counts_even_rounds_to_int():
    counts = median_counts([{"person": 5}, {"person": 6}])
    assert isinstance(counts["person"], int)
    assert counts["person"] in (5, 6)


def test_median_counts_missing_keys_and_none_are_zero():
    counts = median_counts([
        {"person": 5, "car": None},
        {"person": 6},
        {"person": 7, "car": 4},
    ])
    assert counts["person"] == 6
    assert counts["car"] == 0


def test_median_counts_single_frame_passthrough():
    counts = median_counts([{"person": 3, "vehicles": 1}])
    assert counts == {"person": 3, "vehicles": 1}


def test_median_counts_empty():
    assert median_counts([]) == {}
