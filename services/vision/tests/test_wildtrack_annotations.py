"""WILDTRACK ground-truth annotation domain models and JSON parsing."""

from __future__ import annotations

import pytest

from atreides_vision.errors import WildtrackDatasetError
from atreides_vision.venue_validation.datasets.wildtrack.annotations import (
    GRID_HEIGHT,
    GRID_ORIGIN_X_M,
    GRID_ORIGIN_Y_M,
    GRID_SPACING_M,
    GRID_WIDTH,
    MAX_POSITION_ID,
    WildtrackBoundingBox,
    WildtrackFrameAnnotations,
    WildtrackGroundPosition,
    WildtrackPersonAnnotation,
    WildtrackPersonId,
    ground_position_from_position_id,
    load_frame_annotations,
)
from atreides_vision.venue_validation.datasets.wildtrack.cameras import CAMERA_IDS
from atreides_vision.venue_validation.datasets.wildtrack.time import WildtrackFrameId
from wildtrack_fixtures import make_view, write_annotation_json


def all_not_visible_views() -> list[dict[str, object]]:
    return [make_view(i, None) for i in range(7)]


def person_record(
    *, person_id: int = 1, position_id: int = 0, views: list[dict[str, object]] | None = None
) -> dict[str, object]:
    return {
        "personID": person_id,
        "positionID": position_id,
        "views": views if views is not None else all_not_visible_views(),
    }


class TestGroundPositionFormula:
    def test_origin_cell(self) -> None:
        assert ground_position_from_position_id(0) == (GRID_ORIGIN_X_M, GRID_ORIGIN_Y_M)

    def test_one_step_in_x(self) -> None:
        x, y = ground_position_from_position_id(1)
        assert x == pytest.approx(GRID_ORIGIN_X_M + GRID_SPACING_M)
        assert y == pytest.approx(GRID_ORIGIN_Y_M)

    def test_one_row_up_in_y(self) -> None:
        x, y = ground_position_from_position_id(GRID_WIDTH)
        assert x == pytest.approx(GRID_ORIGIN_X_M)
        assert y == pytest.approx(GRID_ORIGIN_Y_M + GRID_SPACING_M)

    def test_max_position_id_valid(self) -> None:
        x, y = ground_position_from_position_id(MAX_POSITION_ID)
        assert x == pytest.approx(GRID_ORIGIN_X_M + GRID_SPACING_M * (GRID_WIDTH - 1))
        assert y == pytest.approx(GRID_ORIGIN_Y_M + GRID_SPACING_M * (GRID_HEIGHT - 1))

    def test_negative_position_id_rejected(self) -> None:
        with pytest.raises(WildtrackDatasetError):
            ground_position_from_position_id(-1)

    def test_position_id_beyond_grid_rejected(self) -> None:
        with pytest.raises(WildtrackDatasetError):
            ground_position_from_position_id(MAX_POSITION_ID + 1)


class TestWildtrackPersonId:
    def test_valid(self) -> None:
        assert WildtrackPersonId(0).value == 0

    def test_negative_rejected(self) -> None:
        with pytest.raises(WildtrackDatasetError):
            WildtrackPersonId(-1)

    def test_ordered_and_distinct(self) -> None:
        assert WildtrackPersonId(1) < WildtrackPersonId(2)
        assert WildtrackPersonId(1) != WildtrackPersonId(2)


class TestWildtrackGroundPosition:
    def test_from_position_id_matches_formula(self) -> None:
        position = WildtrackGroundPosition.from_position_id(5)
        x, y = ground_position_from_position_id(5)
        assert position.x_m == pytest.approx(x)
        assert position.y_m == pytest.approx(y)

    def test_out_of_range_position_id_rejected(self) -> None:
        with pytest.raises(WildtrackDatasetError):
            WildtrackGroundPosition(position_id=-1, x_m=0.0, y_m=0.0)

    def test_non_finite_coordinates_rejected(self) -> None:
        with pytest.raises(WildtrackDatasetError):
            WildtrackGroundPosition(position_id=0, x_m=float("nan"), y_m=0.0)


class TestWildtrackBoundingBox:
    def test_valid(self) -> None:
        box = WildtrackBoundingBox(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0)
        assert box.to_record() == {"xmin": 1.0, "ymin": 2.0, "xmax": 3.0, "ymax": 4.0}

    def test_non_finite_rejected(self) -> None:
        with pytest.raises(WildtrackDatasetError):
            WildtrackBoundingBox(xmin=float("inf"), ymin=0.0, xmax=1.0, ymax=1.0)

    def test_max_before_min_rejected(self) -> None:
        with pytest.raises(WildtrackDatasetError):
            WildtrackBoundingBox(xmin=5.0, ymin=0.0, xmax=1.0, ymax=1.0)


def _person(
    views: dict[str, WildtrackBoundingBox | None] | None = None,
) -> WildtrackPersonAnnotation:
    full_views: dict[str, WildtrackBoundingBox | None] = {c: None for c in CAMERA_IDS}
    if views:
        full_views.update(views)
    return WildtrackPersonAnnotation(
        person_id=WildtrackPersonId(1),
        position=WildtrackGroundPosition.from_position_id(0),
        views=full_views,
    )


class TestWildtrackPersonAnnotation:
    def test_missing_camera_view_rejected(self) -> None:
        incomplete = {c: None for c in CAMERA_IDS[:-1]}
        with pytest.raises(WildtrackDatasetError):
            WildtrackPersonAnnotation(
                person_id=WildtrackPersonId(1),
                position=WildtrackGroundPosition.from_position_id(0),
                views=incomplete,
            )

    def test_extra_camera_view_rejected(self) -> None:
        extra = {c: None for c in CAMERA_IDS}
        extra["C99"] = None
        with pytest.raises(WildtrackDatasetError):
            WildtrackPersonAnnotation(
                person_id=WildtrackPersonId(1),
                position=WildtrackGroundPosition.from_position_id(0),
                views=extra,
            )

    def test_visible_in_and_visible_camera_ids(self) -> None:
        bbox = WildtrackBoundingBox(xmin=0.0, ymin=0.0, xmax=1.0, ymax=1.0)
        person = _person(views={"C3": bbox, "C1": bbox})
        assert person.visible_in("C1") is True
        assert person.visible_in("C2") is False
        # Canonical order, not insertion order.
        assert person.visible_camera_ids == ("C1", "C3")

    def test_bbox_for_unknown_camera_raises(self) -> None:
        person = _person()
        with pytest.raises(WildtrackDatasetError):
            person.bbox_for("C99")

    def test_to_record_shape(self) -> None:
        record = _person().to_record()
        assert record["person_id"] == 1
        assert set(record["views"]) == set(CAMERA_IDS)


class TestWildtrackFrameAnnotations:
    def test_duplicate_person_id_rejected(self) -> None:
        person = _person()
        with pytest.raises(WildtrackDatasetError):
            WildtrackFrameAnnotations(frame_id=WildtrackFrameId(0), persons=(person, person))

    def test_non_canonical_order_rejected(self) -> None:
        first = WildtrackPersonAnnotation(
            person_id=WildtrackPersonId(2),
            position=WildtrackGroundPosition.from_position_id(0),
            views={c: None for c in CAMERA_IDS},
        )
        second = WildtrackPersonAnnotation(
            person_id=WildtrackPersonId(1),
            position=WildtrackGroundPosition.from_position_id(0),
            views={c: None for c in CAMERA_IDS},
        )
        with pytest.raises(WildtrackDatasetError):
            WildtrackFrameAnnotations(frame_id=WildtrackFrameId(0), persons=(first, second))

    def test_person_lookup(self) -> None:
        person = _person()
        frame = WildtrackFrameAnnotations(frame_id=WildtrackFrameId(0), persons=(person,))
        assert frame.person(1) is person
        assert frame.person(99) is None
        assert frame.person_count == 1


class TestFromJsonHappyPath:
    def test_single_visible_person(self) -> None:
        views = all_not_visible_views()
        views[0] = make_view(0, (1, 2, 3, 4))
        data = [person_record(person_id=7, position_id=100, views=views)]
        frame = WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), data)
        assert frame.person_count == 1
        person = frame.persons[0]
        assert person.person_id.value == 7
        assert person.visible_camera_ids == ("C1",)
        assert person.bbox_for("C1") == WildtrackBoundingBox(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0)

    def test_sorts_by_person_id_regardless_of_input_order(self) -> None:
        data = [person_record(person_id=5), person_record(person_id=2), person_record(person_id=9)]
        frame = WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), data)
        assert [p.person_id.value for p in frame.persons] == [2, 5, 9]

    def test_empty_frame(self) -> None:
        frame = WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), [])
        assert frame.person_count == 0


class TestVisibilitySentinelSemantics:
    """Official WILDTRACK annotations-viewer rule: visible iff no -1 field.

    ``-1 not in _box.values()`` — any occurrence of -1 on any one of the four
    bbox fields makes the view non-visible, not only the all-four-``-1``
    shape.
    """

    def test_all_four_fields_sentinel_is_not_visible(self) -> None:
        views = all_not_visible_views()
        views[0] = {"viewNum": 0, "xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1}
        entry = person_record(views=views)
        frame = WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), [entry])
        person = frame.persons[0]
        assert person.visible_in("C1") is False
        assert person.bbox_for("C1") is None

    def test_sentinel_on_xmin_only_is_not_visible(self) -> None:
        views = all_not_visible_views()
        views[0] = {"viewNum": 0, "xmin": -1, "ymin": 100, "xmax": 200, "ymax": 300}
        entry = person_record(views=views)
        frame = WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), [entry])
        person = frame.persons[0]
        assert person.visible_in("C1") is False
        assert person.bbox_for("C1") is None

    def test_sentinel_on_ymin_only_is_not_visible(self) -> None:
        views = all_not_visible_views()
        views[0] = {"viewNum": 0, "xmin": 100, "ymin": -1, "xmax": 200, "ymax": 300}
        entry = person_record(views=views)
        frame = WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), [entry])
        person = frame.persons[0]
        assert person.visible_in("C1") is False
        assert person.bbox_for("C1") is None

    def test_sentinel_on_xmax_only_is_not_visible(self) -> None:
        views = all_not_visible_views()
        views[0] = {"viewNum": 0, "xmin": 100, "ymin": 200, "xmax": -1, "ymax": 300}
        entry = person_record(views=views)
        frame = WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), [entry])
        person = frame.persons[0]
        assert person.visible_in("C1") is False
        assert person.bbox_for("C1") is None

    def test_sentinel_on_ymax_only_is_not_visible(self) -> None:
        views = all_not_visible_views()
        views[0] = {"viewNum": 0, "xmin": 100, "ymin": 200, "xmax": 300, "ymax": -1}
        entry = person_record(views=views)
        frame = WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), [entry])
        person = frame.persons[0]
        assert person.visible_in("C1") is False
        assert person.bbox_for("C1") is None

    def test_ordinary_valid_bbox_is_visible(self) -> None:
        views = all_not_visible_views()
        views[0] = {"viewNum": 0, "xmin": 100, "ymin": 200, "xmax": 300, "ymax": 400}
        entry = person_record(views=views)
        frame = WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), [entry])
        person = frame.persons[0]
        assert person.visible_in("C1") is True
        assert person.bbox_for("C1") == WildtrackBoundingBox(
            xmin=100.0, ymin=200.0, xmax=300.0, ymax=400.0
        )

    def test_visible_malformed_bbox_without_sentinel_still_rejected(self) -> None:
        views = all_not_visible_views()
        # No -1 present, so this is a genuine "visible" record subject to
        # ordinary bbox validation — max-before-min is still an error.
        views[0] = {"viewNum": 0, "xmin": 300, "ymin": 400, "xmax": 100, "ymax": 200}
        entry = person_record(views=views)
        with pytest.raises(WildtrackDatasetError):
            WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), [entry])


class TestFromJsonMalformedInput:
    def test_top_level_not_an_array(self) -> None:
        with pytest.raises(WildtrackDatasetError):
            WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), {"not": "an array"})

    def test_top_level_string_rejected(self) -> None:
        with pytest.raises(WildtrackDatasetError):
            WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), "00000000")

    def test_entry_not_an_object(self) -> None:
        with pytest.raises(WildtrackDatasetError):
            WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), ["not an object"])

    def test_missing_person_id_field(self) -> None:
        entry = person_record()
        del entry["personID"]
        with pytest.raises(WildtrackDatasetError):
            WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), [entry])

    def test_missing_views_field(self) -> None:
        entry = person_record()
        del entry["views"]
        with pytest.raises(WildtrackDatasetError):
            WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), [entry])

    def test_person_id_wrong_type(self) -> None:
        entry = person_record()
        entry["personID"] = "1"
        with pytest.raises(WildtrackDatasetError):
            WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), [entry])

    def test_person_id_bool_rejected(self) -> None:
        entry = person_record()
        entry["personID"] = True
        with pytest.raises(WildtrackDatasetError):
            WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), [entry])

    def test_views_not_an_array(self) -> None:
        entry = person_record()
        entry["views"] = "not an array"
        with pytest.raises(WildtrackDatasetError):
            WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), [entry])

    def test_wrong_view_count(self) -> None:
        entry = person_record(views=all_not_visible_views()[:-1])
        with pytest.raises(WildtrackDatasetError):
            WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), [entry])

    def test_duplicate_view_num(self) -> None:
        views = all_not_visible_views()
        views[1]["viewNum"] = 0
        entry = person_record(views=views)
        with pytest.raises(WildtrackDatasetError):
            WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), [entry])

    def test_view_missing_field(self) -> None:
        views = all_not_visible_views()
        del views[0]["xmin"]
        entry = person_record(views=views)
        with pytest.raises(WildtrackDatasetError):
            WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), [entry])

    def test_view_num_out_of_range(self) -> None:
        views = all_not_visible_views()
        views[0]["viewNum"] = 99
        entry = person_record(views=views)
        with pytest.raises(WildtrackDatasetError):
            WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), [entry])

    def test_duplicate_person_id_across_records(self) -> None:
        entries = [person_record(person_id=1), person_record(person_id=1)]
        with pytest.raises(WildtrackDatasetError):
            WildtrackFrameAnnotations.from_json(WildtrackFrameId(0), entries)


class TestLoadFrameAnnotations:
    def test_missing_file_raises(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(WildtrackDatasetError):
            load_frame_annotations(tmp_path / "00000000.json", WildtrackFrameId(0))

    def test_malformed_json_raises(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "00000000.json"
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(WildtrackDatasetError):
            load_frame_annotations(path, WildtrackFrameId(0))

    def test_happy_path_round_trip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "00000000.json"
        write_annotation_json(path, [person_record(person_id=3)])
        frame = load_frame_annotations(path, WildtrackFrameId(0))
        assert frame.person_count == 1
        assert frame.persons[0].person_id.value == 3

    def test_determinism_across_shuffled_input_order(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        write_annotation_json(path_a, [person_record(person_id=3), person_record(person_id=1)])
        write_annotation_json(path_b, [person_record(person_id=1), person_record(person_id=3)])
        frame_a = load_frame_annotations(path_a, WildtrackFrameId(0))
        frame_b = load_frame_annotations(path_b, WildtrackFrameId(0))
        assert frame_a.to_record() == frame_b.to_record()
