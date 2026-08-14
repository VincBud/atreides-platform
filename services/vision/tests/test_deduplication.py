"""Rule-B3 duplicate-detection suppression.

The safety property under test throughout: the rule must remove near-identical
boxes on one person and must leave genuinely distinct people alone.

B3 is a three-axis conjunction — IoU >= 0.85 AND area similarity >= 0.95 AND
normalised centre distance <= 0.03 — so each axis is exercised in isolation.
Constructing a pair that varies one axis while holding the other two safely
inside the gate takes deliberate geometry, so the boundary tests below assert
the geometry they built before asserting the policy's verdict on it.
"""

from __future__ import annotations

import math

import pytest

from atreides_vision.detection.deduplication import (
    RULE_B3,
    DeduplicationPolicy,
    deduplicate_detections,
    deduplicate_with_report,
)
from atreides_vision.models import BoundingBox, Detection


def box(x1: float, y1: float, x2: float, y2: float, conf: float = 0.5) -> Detection:
    return Detection(bbox_xyxy=BoundingBox(x1, y1, x2, y2), confidence=conf, class_id=1)


def person(x: float, y: float, w: float = 40.0, h: float = 120.0, conf: float = 0.5) -> Detection:
    """A person-shaped box with its top-left at ``(x, y)``."""
    return box(x, y, x + w, y + h, conf)


def geometry(a: Detection, b: Detection) -> tuple[float, float, float]:
    """``(iou, area_similarity, normalised_centre_distance)`` for a pair.

    Independently reimplemented from the production predicate so a boundary
    test asserts against an outside measurement, not the code under test.
    """
    ba, bb = a.bbox_xyxy, b.bbox_xyxy
    ix1, iy1 = max(ba.x1, bb.x1), max(ba.y1, bb.y1)
    ix2, iy2 = min(ba.x2, bb.x2), min(ba.y2, bb.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = ba.area + bb.area - intersection
    iou = intersection / union if union > 0 else 0.0
    area_similarity = min(ba.area, bb.area) / max(ba.area, bb.area)
    separation = math.hypot(ba.center.x - bb.center.x, ba.center.y - bb.center.y)
    mean_diagonal = (math.hypot(ba.width, ba.height) + math.hypot(bb.width, bb.height)) / 2
    return iou, area_similarity, separation / mean_diagonal


def side_by_side(offset: float, w: float, h: float) -> tuple[Detection, Detection]:
    """Two identically sized boxes offset horizontally by ``offset``.

    Area similarity is exactly 1.0, so the pair varies IoU and centre distance
    together and isolates them from the area axis.
    """
    return box(0.0, 0.0, w, h, conf=0.7), box(offset, 0.0, w + offset, h, conf=0.4)


def concentric(scale: float, w: float = 74.0, h: float = 240.0) -> tuple[Detection, Detection]:
    """A box and a smaller copy sharing its centre.

    Centre distance is exactly 0, and IoU equals area similarity equals
    ``scale ** 2``, which isolates the area axis.
    """
    outer = box(0.0, 0.0, w, h, conf=0.7)
    dx, dy = w * (1 - scale) / 2, h * (1 - scale) / 2
    return outer, box(dx, dy, w - dx, h - dy, conf=0.4)


class TestPolicy:
    def test_rule_b3_constants(self) -> None:
        assert RULE_B3.min_iou == 0.85
        assert RULE_B3.min_area_similarity == 0.95
        assert RULE_B3.max_center_distance == 0.03


class TestIouAxis:
    """IoU >= 0.85, holding area similarity and centre distance inside the gate."""

    def test_at_the_gate_suppresses(self) -> None:
        # 74 wide offset by 6 gives intersection 68/80 of the union: exactly 0.85.
        a, b = side_by_side(offset=6.0, w=74.0, h=240.0)
        iou, area_similarity, centre = geometry(a, b)
        assert iou == 0.85, "fixture must sit exactly on the gate"
        assert area_similarity == 1.0
        assert centre < RULE_B3.max_center_distance
        assert deduplicate_detections([a, b]) == [a]

    def test_just_above_the_gate_suppresses(self) -> None:
        a, b = side_by_side(offset=5.0, w=74.0, h=240.0)
        iou, area_similarity, centre = geometry(a, b)
        assert RULE_B3.min_iou < iou < 0.90, "above B3's gate but below the old B0 gate"
        assert area_similarity >= RULE_B3.min_area_similarity
        assert centre <= RULE_B3.max_center_distance
        assert deduplicate_detections([a, b]) == [a]

    @pytest.mark.parametrize("offset", [6.5, 7.0])
    def test_just_below_the_gate_is_kept(self, offset: float) -> None:
        a, b = side_by_side(offset=offset, w=74.0, h=240.0)
        iou, area_similarity, centre = geometry(a, b)
        assert iou < RULE_B3.min_iou, "fixture must fail the IoU axis"
        assert area_similarity >= RULE_B3.min_area_similarity, "and only the IoU axis"
        assert centre <= RULE_B3.max_center_distance
        assert len(deduplicate_detections([a, b])) == 2

    def test_the_old_b0_gate_no_longer_binds(self) -> None:
        """Geometry between the B0 and B3 IoU gates is exactly what B3 adds."""
        a, b = side_by_side(offset=6.0, w=74.0, h=240.0)
        iou, _, _ = geometry(a, b)
        assert 0.85 <= iou < 0.90
        b0 = DeduplicationPolicy(min_iou=0.90, min_area_similarity=0.95, max_center_distance=0.02)
        assert len(deduplicate_detections([a, b], b0)) == 2, "B0 left this pair alone"
        assert len(deduplicate_detections([a, b])) == 1, "B3 suppresses it"


class TestAreaSimilarityAxis:
    """Area similarity >= 0.95, holding IoU and centre distance inside the gate."""

    def test_at_the_gate_suppresses(self) -> None:
        a, b = concentric(scale=math.sqrt(0.96))
        iou, area_similarity, centre = geometry(a, b)
        assert iou >= RULE_B3.min_iou
        assert area_similarity >= RULE_B3.min_area_similarity
        assert centre == pytest.approx(0.0)
        assert deduplicate_detections([a, b]) == [a]

    @pytest.mark.parametrize("area_ratio", [0.94, 0.90, 0.86])
    def test_below_the_gate_is_kept(self, area_ratio: float) -> None:
        a, b = concentric(scale=math.sqrt(area_ratio))
        iou, area_similarity, centre = geometry(a, b)
        assert area_similarity < RULE_B3.min_area_similarity, "fixture must fail the area axis"
        assert iou >= RULE_B3.min_iou, "and only the area axis"
        assert centre <= RULE_B3.max_center_distance
        assert len(deduplicate_detections([a, b])) == 2


class TestCentreDistanceAxis:
    """Centre distance <= 0.03, holding IoU and area similarity inside the gate."""

    def test_above_the_gate_is_kept(self) -> None:
        # A squatter box shortens the diagonal, so a small offset still clears
        # the centre gate while IoU stays comfortably above 0.85.
        a, b = side_by_side(offset=5.0, w=74.0, h=120.0)
        iou, area_similarity, centre = geometry(a, b)
        assert centre > RULE_B3.max_center_distance, "fixture must fail the centre axis"
        assert iou >= RULE_B3.min_iou, "and only the centre axis"
        assert area_similarity >= RULE_B3.min_area_similarity
        assert len(deduplicate_detections([a, b])) == 2

    def test_inside_the_gate_suppresses(self) -> None:
        a, b = side_by_side(offset=4.0, w=74.0, h=120.0)
        iou, area_similarity, centre = geometry(a, b)
        assert centre <= RULE_B3.max_center_distance
        assert iou >= RULE_B3.min_iou
        assert area_similarity >= RULE_B3.min_area_similarity
        assert deduplicate_detections([a, b]) == [a]

    def test_vertical_offset_beyond_the_gate_is_kept(self) -> None:
        """Head-to-toe offsets are how a duplicate and a real neighbour differ."""
        a = person(100, 100, w=40, h=200, conf=0.8)
        b = person(100, 112, w=40, h=200, conf=0.7)
        _, _, centre = geometry(a, b)
        assert centre > RULE_B3.max_center_distance
        assert len(deduplicate_detections([a, b])) == 2


#: Real geometry from the cross-clip marginal set, every pair confirmed by eye
#: to be two boxes on one person. These seven are exactly what B3 adds over B0,
#: so they are the regression fixtures that matter most.
CONFIRMED_DUPLICATES: list[tuple[str, tuple[float, ...], tuple[float, ...]]] = [
    # (name, retained box, suppressed box)
    (
        "dense f511",
        (823.47, 339.26, 894.60, 552.00, 0.3769),
        (828.38, 338.95, 897.69, 551.75, 0.3019),
    ),
    (
        "dense f1249",
        (1152.71, 609.17, 1233.57, 849.40, 0.4109),
        (1148.79, 607.83, 1228.56, 842.21, 0.3012),
    ),
    (
        "night f898",
        (968.63, 431.65, 1035.16, 657.03, 0.5433),
        (971.05, 427.85, 1038.86, 656.63, 0.4846),
    ),
    (
        "night f863",
        (423.91, 150.30, 478.53, 319.03, 0.4846),
        (422.22, 150.75, 474.56, 319.90, 0.4418),
    ),
    (
        "night f37",
        (864.73, 145.39, 917.96, 323.22, 0.3643),
        (866.47, 150.22, 920.50, 324.20, 0.3332),
    ),
    (
        "night f926",
        (947.35, 394.50, 1013.54, 614.85, 0.4185),
        (951.27, 395.05, 1018.48, 610.77, 0.3999),
    ),
    (
        "night f927",
        (947.00, 395.72, 1012.05, 614.05, 0.4655),
        (950.72, 394.98, 1017.78, 610.83, 0.3128),
    ),
]

#: Real geometry from pairs manually confirmed to be two different people.
#: Every one must survive; these are the cases a looser rule would collapse.
CONFIRMED_TWO_REAL_PEOPLE: list[tuple[str, tuple[float, ...], tuple[float, ...]]] = [
    (
        "dense IDs 190+213",
        (1252.97, 650.31, 1322.59, 859.82, 0.6),
        (1239.64, 598.15, 1318.56, 859.40, 0.5),
    ),
    (
        "dense IDs 7+74",
        (303.50, 799.67, 393.16, 1068.05, 0.6),
        (284.66, 741.57, 383.04, 1015.45, 0.5),
    ),
    (
        "dense IDs 141+226",
        (1166.31, 902.15, 1256.05, 1080.00, 0.6),
        (1167.24, 784.70, 1270.63, 1078.34, 0.5),
    ),
    (
        "dense IDs 99+110",
        (1076.69, 823.62, 1162.93, 1077.25, 0.6),
        (1075.90, 921.95, 1154.30, 1078.99, 0.5),
    ),
]

DUPLICATE_IDS = [c[0] for c in CONFIRMED_DUPLICATES]
TWO_PEOPLE_IDS = [c[0] for c in CONFIRMED_TWO_REAL_PEOPLE]


class TestConfirmedDuplicateGeometry:
    """The seven marginal pairs B3 adds over B0 must all be suppressed."""

    @pytest.mark.parametrize("name,keep,drop", CONFIRMED_DUPLICATES, ids=DUPLICATE_IDS)
    def test_confirmed_duplicate_is_suppressed(
        self, name: str, keep: tuple[float, ...], drop: tuple[float, ...]
    ) -> None:
        strong, weak = box(*keep), box(*drop)
        assert deduplicate_detections([strong, weak]) == [strong], name
        assert deduplicate_detections([weak, strong]) == [strong], f"{name}, reversed"

    @pytest.mark.parametrize("name,keep,drop", CONFIRMED_DUPLICATES, ids=DUPLICATE_IDS)
    def test_confirmed_duplicate_sits_between_the_b0_and_b3_gates(
        self, name: str, keep: tuple[float, ...], drop: tuple[float, ...]
    ) -> None:
        iou, area_similarity, centre = geometry(box(*keep), box(*drop))
        assert 0.85 <= iou < 0.90, name
        assert area_similarity >= 0.95, name
        assert centre <= 0.03, name


class TestConfirmedTwoRealPeopleGeometry:
    """Pairs confirmed to be two different people must always survive."""

    @pytest.mark.parametrize("name,first,second", CONFIRMED_TWO_REAL_PEOPLE, ids=TWO_PEOPLE_IDS)
    def test_two_real_people_are_never_merged(
        self, name: str, first: tuple[float, ...], second: tuple[float, ...]
    ) -> None:
        a, b = box(*first), box(*second)
        assert len(deduplicate_detections([a, b])) == 2, name

    @pytest.mark.parametrize("name,first,second", CONFIRMED_TWO_REAL_PEOPLE, ids=TWO_PEOPLE_IDS)
    def test_two_real_people_stay_clear_of_the_gate(
        self, name: str, first: tuple[float, ...], second: tuple[float, ...]
    ) -> None:
        """Quantify the safety margin, so a future relaxation cannot erode it silently."""
        iou, _, centre = geometry(box(*first), box(*second))
        assert iou < RULE_B3.min_iou - 0.15, f"{name}: IoU margin"
        assert centre > RULE_B3.max_center_distance * 3, f"{name}: centre margin"


class TestNonDuplicatesAreUntouched:
    def test_empty_input(self) -> None:
        assert deduplicate_detections([]) == []

    def test_single_detection(self) -> None:
        one = person(10, 10)
        assert deduplicate_detections([one]) == [one]

    def test_disjoint_boxes(self) -> None:
        people = [person(0, 0), person(500, 0), person(1000, 0)]
        assert deduplicate_detections(people) == people

    def test_adjacent_people_touching_but_not_overlapping(self) -> None:
        people = [person(0, 0, w=40), person(40, 0, w=40)]
        assert len(deduplicate_detections(people)) == 2

    def test_partially_overlapping_people_are_kept(self) -> None:
        """Two people side by side overlap but nowhere near Rule-B3 geometry."""
        people = [person(0, 0, w=40, h=120), person(20, 0, w=40, h=120)]  # IoU = 1/3
        assert len(deduplicate_detections(people)) == 2

    def test_one_person_in_front_of_another_is_kept(self) -> None:
        """High overlap but very different sizes: a near person and a far one."""
        near = box(100, 100, 200, 400, conf=0.9)
        far = box(120, 120, 170, 260, conf=0.6)
        assert len(deduplicate_detections([near, far])) == 2


class TestDuplicatesAreSuppressed:
    def test_identical_boxes_collapse(self) -> None:
        a = person(100, 100, conf=0.7)
        b = person(100, 100, conf=0.4)
        assert deduplicate_detections([a, b]) == [a]

    def test_near_identical_boxes_collapse(self) -> None:
        """Geometry matching the observed real duplicates."""
        a = box(1000.0, 400.0, 1060.0, 600.0, conf=0.45)
        b = box(1001.5, 401.0, 1061.0, 601.5, conf=0.35)
        kept = deduplicate_detections([a, b])
        assert kept == [a]

    def test_higher_confidence_is_retained(self) -> None:
        weak = person(100, 100, conf=0.31)
        strong = person(100, 100, conf=0.69)
        assert deduplicate_detections([weak, strong]) == [strong]
        assert deduplicate_detections([strong, weak]) == [strong]

    def test_retained_detection_is_the_original_object_unmodified(self) -> None:
        strong = person(100, 100, conf=0.7)
        weak = person(100, 100, conf=0.4)
        kept = deduplicate_detections([strong, weak])
        assert kept[0] is strong
        assert kept[0].confidence == 0.7
        assert kept[0].class_id == 1
        assert kept[0].bbox_xyxy == BoundingBox(100, 100, 140, 220)

    def test_only_the_duplicate_pair_is_affected(self) -> None:
        dup_a = person(100, 100, conf=0.6)
        dup_b = person(100, 100, conf=0.4)
        other = person(600, 100, conf=0.5)
        kept = deduplicate_detections([dup_a, other, dup_b])
        assert kept == [dup_a, other]


class TestDeterminism:
    def test_equal_confidence_tie_keeps_the_earliest(self) -> None:
        first = person(100, 100, conf=0.5)
        second = person(100, 100, conf=0.5)
        assert deduplicate_detections([first, second])[0] is first
        assert deduplicate_detections([second, first])[0] is second

    def test_repeated_runs_agree(self) -> None:
        dets = [person(100, 100, conf=0.5), person(100, 100, conf=0.5), person(500, 0, conf=0.9)]
        results = [deduplicate_detections(list(dets)) for _ in range(20)]
        assert all(r == results[0] for r in results)

    def test_output_preserves_input_order(self) -> None:
        a = person(0, 0, conf=0.9)
        b = person(300, 0, conf=0.8)
        dup = person(300, 0, conf=0.4)
        c = person(700, 0, conf=0.7)
        assert deduplicate_detections([a, b, dup, c]) == [a, b, c]


class TestConnectedGroups:
    def test_three_mutually_duplicate_boxes_collapse_to_one(self) -> None:
        a = person(100, 100, conf=0.4)
        b = person(100, 100, conf=0.8)
        c = person(100, 100, conf=0.6)
        kept = deduplicate_detections([a, b, c])
        assert kept == [b]

    def test_chained_group_collapses_once(self) -> None:
        """A~B and B~C but A!~C still forms a single group, not two pairs."""
        a = box(1000.0, 400.0, 1060.0, 600.0, conf=0.4)
        b = box(1001.0, 401.0, 1061.0, 601.0, conf=0.8)
        c = box(1002.0, 402.0, 1062.0, 602.0, conf=0.6)
        report = deduplicate_with_report([a, b, c])
        assert report.kept == (b,)
        assert report.removed_count == 2

    def test_chained_group_is_order_independent(self) -> None:
        a = box(1000.0, 400.0, 1060.0, 600.0, conf=0.4)
        b = box(1001.0, 401.0, 1061.0, 601.0, conf=0.8)
        c = box(1002.0, 402.0, 1062.0, 602.0, conf=0.6)
        for ordering in ([a, b, c], [c, b, a], [b, a, c], [c, a, b]):
            assert deduplicate_detections(list(ordering)) == [b]

    def test_two_independent_groups(self) -> None:
        g1 = [person(100, 100, conf=0.7), person(100, 100, conf=0.3)]
        g2 = [person(800, 100, conf=0.4), person(800, 100, conf=0.9)]
        kept = deduplicate_detections([*g1, *g2])
        assert kept == [g1[0], g2[1]]


class TestPurity:
    def test_input_list_is_not_mutated(self) -> None:
        dets = [person(100, 100, conf=0.7), person(100, 100, conf=0.4), person(600, 0)]
        snapshot = list(dets)
        deduplicate_detections(dets)
        assert dets == snapshot
        assert len(dets) == 3

    def test_returns_a_new_list(self) -> None:
        dets = [person(0, 0), person(500, 0)]
        assert deduplicate_detections(dets) is not dets

    def test_report_exposes_removed_detections(self) -> None:
        strong = person(100, 100, conf=0.7)
        weak = person(100, 100, conf=0.4)
        report = deduplicate_with_report([strong, weak])
        assert report.kept == (strong,)
        assert report.removed == (weak,)
        assert report.removed_count == 1

    def test_no_duplicates_reports_nothing_removed(self) -> None:
        report = deduplicate_with_report([person(0, 0), person(500, 0)])
        assert report.removed == ()
        assert report.removed_count == 0


class TestPolicyIsHonoured:
    def test_a_stricter_policy_suppresses_less(self) -> None:
        a = box(1000.0, 400.0, 1060.0, 600.0, conf=0.5)
        b = box(1002.0, 402.0, 1062.0, 602.0, conf=0.4)
        assert len(deduplicate_detections([a, b], RULE_B3)) == 1
        strict = DeduplicationPolicy(
            min_iou=0.999, min_area_similarity=0.999, max_center_distance=0.0001
        )
        assert len(deduplicate_detections([a, b], strict)) == 2

    def test_degenerate_zero_area_boxes_are_not_merged(self) -> None:
        a = box(10.0, 10.0, 10.0, 10.0, conf=0.5)
        b = box(10.0, 10.0, 10.0, 10.0, conf=0.4)
        assert len(deduplicate_detections([a, b])) == 2
