"""associate_by_iou_or_distance - Nawaf-adapted two-stage association
pattern from PPE compliance. IoU >= floor OR item center within
distance_frac * anchor diagonal from anchor edge."""
from app.detect_core import associate_by_iou_or_distance


def _box(x1, y1, x2, y2):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def test_overlapping_boxes_pass_via_iou():
    person = _box(100, 100, 200, 400)
    bag = _box(150, 200, 180, 260)  # fully inside person
    assert associate_by_iou_or_distance(person, bag)


def test_far_away_item_rejected():
    person = _box(100, 100, 200, 400)
    other = _box(800, 800, 900, 900)
    assert not associate_by_iou_or_distance(person, other)


def test_adjacent_item_passes_via_distance_fallback():
    # Item center right next to the person's edge (helmet-above-head shape),
    # zero IoU but well inside distance_frac * diagonal.
    person = _box(100, 100, 200, 400)  # diagonal ~316px
    helmet = _box(140, 60, 170, 90)    # sits above head, no IoU
    assert associate_by_iou_or_distance(person, helmet)


def test_item_beyond_distance_frac_rejected():
    # Same person, but the item is 1.5x diagonal away - the 0.6*diagonal
    # gate should reject it even though there is no IoU.
    person = _box(100, 100, 200, 400)  # diagonal ~316px
    far = _box(500, 700, 520, 720)     # far below and to the right
    assert not associate_by_iou_or_distance(person, far)


def test_tight_iou_floor_disables_iou_stage():
    # A grazing corner overlap (~1px sliver) passes the permissive IoU
    # floor but a strict 0.5 IoU + zero distance_frac rejects it as long
    # as the item is clearly OUTSIDE the anchor.
    person = _box(100, 100, 200, 400)
    grazing = _box(199, 399, 201, 401)  # tiny sliver overlap
    assert associate_by_iou_or_distance(person, grazing,
                                        iou_floor=0.0001, distance_frac=0.0)
    fully_outside = _box(240, 100, 260, 120)   # no overlap, few px away
    assert not associate_by_iou_or_distance(person, fully_outside,
                                            iou_floor=0.5,
                                            distance_frac=0.0)


def test_missing_boxes_return_false():
    assert not associate_by_iou_or_distance(None, _box(0, 0, 10, 10))
    assert not associate_by_iou_or_distance(_box(0, 0, 10, 10), None)
    assert not associate_by_iou_or_distance(None, None)
