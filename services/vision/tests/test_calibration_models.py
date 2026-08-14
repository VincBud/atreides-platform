"""Calibration domain models: venue points, reference frames, validation."""

from __future__ import annotations

import pytest

from atreides_vision.calibration.models import (
    CalibrationQuality,
    GroundPlaneCalibration,
    PointCorrespondence,
    ReferenceFrame,
    VenuePoint,
)
from atreides_vision.errors import CalibrationError
from atreides_vision.models import Point
from synthetic_cameras import CAMERA_A, FRAME_A, VENUE_ID, calibration_for

RECTANGLE = [(0.0, 0.0), (12.0, 0.0), (12.0, 9.0), (0.0, 9.0)]


def correspondence(
    image_x: float, image_y: float, venue_x: float, venue_y: float
) -> PointCorrespondence:
    return PointCorrespondence(image=Point(image_x, image_y), venue=VenuePoint(venue_x, venue_y))


def calibration(
    correspondences: list[PointCorrespondence], **overrides: object
) -> GroundPlaneCalibration:
    kwargs: dict[str, object] = {
        "camera_id": "cam_01",
        "venue_id": VENUE_ID,
        "reference_frame": FRAME_A,
        "correspondences": tuple(correspondences),
    }
    kwargs.update(overrides)
    return GroundPlaneCalibration(**kwargs)  # type: ignore[arg-type]


SQUARE = [
    correspondence(100.0, 900.0, 0.0, 0.0),
    correspondence(900.0, 900.0, 6.0, 0.0),
    correspondence(800.0, 500.0, 6.0, 8.0),
    correspondence(200.0, 500.0, 0.0, 8.0),
]


class TestVenuePoint:
    def test_holds_metres(self) -> None:
        point = VenuePoint(12.4, 7.1)
        assert (point.x, point.y) == (12.4, 7.1)
        assert point.as_tuple() == (12.4, 7.1)

    def test_distance_is_in_metres(self) -> None:
        assert VenuePoint(0.0, 0.0).distance_to(VenuePoint(3.0, 4.0)) == pytest.approx(5.0)

    def test_allows_negative_coordinates(self) -> None:
        """The venue origin is wherever the operator put it."""
        assert VenuePoint(-4.0, -2.5).x == -4.0

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_rejects_non_finite(self, bad: float) -> None:
        with pytest.raises(CalibrationError, match="must be finite metres"):
            VenuePoint(bad, 0.0)

    def test_is_immutable(self) -> None:
        with pytest.raises(AttributeError):
            VenuePoint(1.0, 2.0).x = 5.0  # type: ignore[misc]


class TestReferenceFrame:
    def test_records_authoring_dimensions(self) -> None:
        frame = ReferenceFrame(1920, 1080)
        assert (frame.width, frame.height) == (1920, 1080)
        assert frame.aspect_ratio == pytest.approx(16 / 9)
        assert str(frame) == "1920x1080"

    @pytest.mark.parametrize(("width", "height"), [(0, 100), (100, 0), (-1, 10), (0, 0)])
    def test_rejects_invalid_dimensions(self, width: int, height: int) -> None:
        with pytest.raises(CalibrationError, match="must be positive"):
            ReferenceFrame(width, height)


class TestPointCorrespondence:
    def test_pairs_pixels_with_metres(self) -> None:
        pair = correspondence(100.0, 200.0, 1.5, 2.5)
        assert pair.image == Point(100.0, 200.0)
        assert pair.venue == VenuePoint(1.5, 2.5)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_rejects_non_finite_image_coordinates(self, bad: float) -> None:
        with pytest.raises(CalibrationError, match="must be finite pixels"):
            correspondence(bad, 10.0, 1.0, 1.0)

    def test_round_trips_to_a_record(self) -> None:
        assert correspondence(1.0, 2.0, 3.0, 4.0).to_record() == {
            "image": [1.0, 2.0],
            "venue": [3.0, 4.0],
        }


class TestCalibrationValidation:
    def test_accepts_four_points(self) -> None:
        assert calibration(SQUARE).point_count == 4
        assert calibration(SQUARE).is_overdetermined is False

    def test_accepts_more_than_four(self) -> None:
        extra = [*SQUARE, correspondence(500.0, 700.0, 3.0, 4.0)]
        assert calibration(extra).is_overdetermined is True

    @pytest.mark.parametrize("count", [0, 1, 2, 3])
    def test_rejects_too_few_points(self, count: int) -> None:
        with pytest.raises(CalibrationError, match="at least 4 are required"):
            calibration(SQUARE[:count])

    @pytest.mark.parametrize("field", ["camera_id", "venue_id", "surface_id"])
    @pytest.mark.parametrize("value", ["", "   "])
    def test_rejects_empty_identifiers(self, field: str, value: str) -> None:
        with pytest.raises(CalibrationError, match=f"{field} must be a non-empty string"):
            calibration(SQUARE, **{field: value})

    def test_rejects_duplicate_image_points(self) -> None:
        duplicated = [*SQUARE[:3], correspondence(100.0, 900.0, 3.0, 5.0)]
        with pytest.raises(CalibrationError, match="same image position"):
            calibration(duplicated)

    def test_rejects_duplicate_venue_points(self) -> None:
        duplicated = [*SQUARE[:3], correspondence(400.0, 700.0, 0.0, 0.0)]
        with pytest.raises(CalibrationError, match="same venue position"):
            calibration(duplicated)

    def test_rejects_three_collinear_of_four(self) -> None:
        collinear = [
            correspondence(100.0, 900.0, 0.0, 0.0),
            correspondence(500.0, 900.0, 3.0, 0.0),
            correspondence(900.0, 900.0, 6.0, 0.0),
            correspondence(500.0, 500.0, 3.0, 8.0),
        ]
        with pytest.raises(CalibrationError, match="lie on a straight line"):
            calibration(collinear)

    def test_allows_three_collinear_when_more_points_exist(self) -> None:
        """With five points, a collinear triple is not fatal."""
        five = [
            correspondence(100.0, 900.0, 0.0, 0.0),
            correspondence(500.0, 900.0, 3.0, 0.0),
            correspondence(900.0, 900.0, 6.0, 0.0),
            correspondence(500.0, 500.0, 3.0, 8.0),
            correspondence(150.0, 500.0, 0.0, 8.0),
        ]
        assert calibration(five).point_count == 5

    def test_rejects_wholly_collinear_sets(self) -> None:
        line = [correspondence(100.0 * i, 900.0, float(i), 0.0) for i in range(1, 7)]
        with pytest.raises(CalibrationError, match="all lie on a straight line"):
            calibration(line)


class TestCalibrationGeometry:
    def test_exposes_point_arrays(self) -> None:
        model = calibration(SQUARE)
        assert model.image_points().shape == (4, 2)
        assert model.venue_points().shape == (4, 2)
        assert model.venue_points()[1].tolist() == [6.0, 0.0]

    def test_venue_extent_is_the_bounding_box_diagonal(self) -> None:
        assert calibration(SQUARE).venue_extent_metres() == pytest.approx(10.0)

    def test_extent_scales_with_the_venue(self) -> None:
        """Nothing assumes a venue size — a field and a doorway both work."""
        small = calibration_for(CAMERA_A, [(0.0, 0.0), (2.0, 0.0), (2.0, 1.5), (0.0, 1.5)])
        large = calibration_for(CAMERA_A, [(0.0, 0.0), (80.0, 0.0), (80.0, 60.0), (0.0, 60.0)])
        assert small.venue_extent_metres() == pytest.approx(2.5)
        assert large.venue_extent_metres() == pytest.approx(100.0)

    def test_iterates_and_counts(self) -> None:
        model = calibration(SQUARE)
        assert len(model) == 4
        assert len(list(model)) == 4


class TestVenueCompatibility:
    def test_same_venue_and_surface_are_comparable(self) -> None:
        left = calibration_for(CAMERA_A, RECTANGLE, camera_id="cam_a")
        right = calibration_for(CAMERA_A, RECTANGLE, camera_id="cam_b")
        assert left.covers_same_venue_as(right)

    def test_different_venues_are_not_comparable(self) -> None:
        left = calibration_for(CAMERA_A, RECTANGLE, camera_id="cam_a", venue_id="club")
        right = calibration_for(CAMERA_A, RECTANGLE, camera_id="cam_b", venue_id="festival")
        assert not left.covers_same_venue_as(right)

    def test_different_surfaces_are_not_comparable(self) -> None:
        """A balcony and the floor are different planes even in one venue."""
        floor = calibration_for(CAMERA_A, RECTANGLE, camera_id="cam_a")
        balcony = calibration_for(CAMERA_A, RECTANGLE, camera_id="cam_b", surface_id="balcony")
        assert not floor.covers_same_venue_as(balcony)


class TestSerialization:
    def test_round_trips_through_a_record(self) -> None:
        original = calibration(SQUARE, description="test rig")
        restored = GroundPlaneCalibration.from_mapping(original.to_record())
        assert restored == original

    def test_surface_id_defaults_to_ground(self) -> None:
        record = calibration(SQUARE).to_record()
        del record["surface_id"]
        assert GroundPlaneCalibration.from_mapping(record).surface_id == "ground"

    @pytest.mark.parametrize(
        "field", ["camera_id", "venue_id", "reference_frame", "correspondences"]
    )
    def test_rejects_missing_required_fields(self, field: str) -> None:
        record = calibration(SQUARE).to_record()
        del record[field]
        with pytest.raises(CalibrationError, match=f"missing required field '{field}'"):
            GroundPlaneCalibration.from_mapping(record)

    def test_rejects_a_non_object(self) -> None:
        with pytest.raises(CalibrationError, match="must be a JSON object"):
            GroundPlaneCalibration.from_mapping([])  # type: ignore[arg-type]

    def test_rejects_non_string_identifiers(self) -> None:
        record = calibration(SQUARE).to_record()
        record["camera_id"] = 7
        with pytest.raises(CalibrationError, match="camera_id must be a string"):
            GroundPlaneCalibration.from_mapping(record)

    @pytest.mark.parametrize("field", ["width", "height"])
    def test_rejects_missing_frame_dimensions(self, field: str) -> None:
        record = calibration(SQUARE).to_record()
        del record["reference_frame"][field]
        with pytest.raises(CalibrationError, match=f"missing required field '{field}'"):
            GroundPlaneCalibration.from_mapping(record)

    @pytest.mark.parametrize("value", [1920.5, "1920", True, None])
    def test_rejects_non_integer_frame_dimensions(self, value: object) -> None:
        record = calibration(SQUARE).to_record()
        record["reference_frame"]["width"] = value
        with pytest.raises(CalibrationError, match="must be an integer number of pixels"):
            GroundPlaneCalibration.from_mapping(record)

    def test_rejects_a_non_object_reference_frame(self) -> None:
        record = calibration(SQUARE).to_record()
        record["reference_frame"] = [1920, 1080]
        with pytest.raises(CalibrationError, match="reference_frame must be an object"):
            GroundPlaneCalibration.from_mapping(record)

    def test_rejects_correspondences_that_are_not_a_list(self) -> None:
        record = calibration(SQUARE).to_record()
        record["correspondences"] = "four points"
        with pytest.raises(CalibrationError, match="must be an array"):
            GroundPlaneCalibration.from_mapping(record)

    @pytest.mark.parametrize("field", ["image", "venue"])
    def test_rejects_missing_correspondence_fields(self, field: str) -> None:
        record = calibration(SQUARE).to_record()
        del record["correspondences"][1][field]
        with pytest.raises(
            CalibrationError, match=rf"correspondences\[1\] is missing required field '{field}'"
        ):
            GroundPlaneCalibration.from_mapping(record)

    @pytest.mark.parametrize("bad", [[1.0], [1.0, 2.0, 3.0], []])
    def test_rejects_wrong_length_pairs(self, bad: list[float]) -> None:
        record = calibration(SQUARE).to_record()
        record["correspondences"][0]["venue"] = bad
        with pytest.raises(CalibrationError, match="must have exactly 2 values"):
            GroundPlaneCalibration.from_mapping(record)

    @pytest.mark.parametrize("bad", ["x", True, None])
    def test_rejects_non_numeric_coordinates(self, bad: object) -> None:
        record = calibration(SQUARE).to_record()
        record["correspondences"][0]["venue"] = [bad, 2.0]
        with pytest.raises(CalibrationError, match="must contain two numbers"):
            GroundPlaneCalibration.from_mapping(record)

    def test_rejects_non_finite_coordinates(self) -> None:
        record = calibration(SQUARE).to_record()
        record["correspondences"][0]["venue"] = [float("inf"), 2.0]
        with pytest.raises(CalibrationError, match="must contain finite numbers"):
            GroundPlaneCalibration.from_mapping(record)

    def test_rejects_a_non_string_description(self) -> None:
        record = calibration(SQUARE).to_record()
        record["description"] = 42
        with pytest.raises(CalibrationError, match="description must be a string"):
            GroundPlaneCalibration.from_mapping(record)


class TestCalibrationQuality:
    def make(self, mean_m: float = 0.01, extent: float = 10.0) -> CalibrationQuality:
        return CalibrationQuality(
            point_count=4,
            mean_venue_error_m=mean_m,
            max_venue_error_m=mean_m * 2,
            mean_image_error_px=1.0,
            max_image_error_px=2.0,
            venue_extent_m=extent,
        )

    def test_relative_error_is_scale_free(self) -> None:
        """1 cm over a 2 m doorway is not 1 cm over a 60 m field."""
        doorway = self.make(mean_m=0.01, extent=2.0)
        field = self.make(mean_m=0.01, extent=60.0)
        assert doorway.relative_error == pytest.approx(0.005)
        assert field.relative_error == pytest.approx(0.01 / 60.0)

    def test_relative_error_is_zero_for_a_degenerate_extent(self) -> None:
        assert self.make(extent=0.0).relative_error == 0.0

    def test_is_exact_only_for_a_perfect_fit(self) -> None:
        assert self.make(mean_m=0.0).is_exact
        assert not self.make(mean_m=0.01).is_exact

    def test_record_reports_both_units(self) -> None:
        record = self.make().to_record()
        assert set(record) == {
            "point_count",
            "mean_venue_error_m",
            "max_venue_error_m",
            "mean_image_error_px",
            "max_image_error_px",
            "venue_extent_m",
            "relative_error",
        }
