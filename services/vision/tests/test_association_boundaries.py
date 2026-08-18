"""Architectural guards for V0.4.1: layer boundaries, and the absence of identity.

These are not style checks. Cross-camera logic must live in exactly one place —
above the camera-local tracker, never inside it — and V0.4.1 deliberately builds
no venue-local identity, which is only credible if nothing here quietly grows
one.

What these guards do and do not see
-----------------------------------
Every module **anywhere under** ``association/`` is walked, subpackages
included: a future ``association/identity/store.py`` is exactly the file these
guards exist to catch, and a non-recursive scan would never look at it.
:class:`TestTheGuardsThemselves` proves that by running each check against a
deliberately offending nested module.

The allowed set is narrow on purpose. ``association/`` consumes Atreides domain
*values* — a subject reference, a stream reference, a venue time and its
uncertainty, a projected position, a camera's trust and calibrated region — and
imports none of the machinery that produced them. That is what keeps OpenCV, and
every pixel, out of this layer transitively: ``calibration.integrity`` reaches
``cv2``, and ``venue`` reaches it through ``video.source``.

**Residual, documented rather than silent:** an AST walk sees ``import`` and
``from ... import`` statements only. A dynamic ``importlib.import_module(...)``
would be invisible to every check here. Nothing in the codebase does that today.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from association_builders import GATING, observation
from atreides_vision.association import (
    AssociationOutcome,
    CandidatePair,
    ObservationGating,
    TrackObservation,
    gate_observation,
)
from atreides_vision.behavior.models import SubjectRef
from atreides_vision.errors import AssociationError
from atreides_vision.timing import StreamRef

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "atreides_vision"
ASSOCIATION_ROOT = SOURCE_ROOT / "association"

#: The association layer may reach these and nothing else inside the package.
#: It consumes the shared coordinate vocabulary, the camera-local subject
#: reference, and the venue time domain. Everything else — calibration, venue,
#: video, and every signal layer — arrives as values from a composition root.
ALLOWED_INTERNAL_PREFIXES = (
    "atreides_vision.association",
    "atreides_vision.behavior.coordinates",
    "atreides_vision.behavior.models",
    "atreides_vision.timing",
    "atreides_vision.errors",
)

#: Sibling signal layers and the pixel-facing layers. ``calibration`` and
#: ``venue`` are here because both transitively import OpenCV; association takes
#: their conclusions as plain values instead.
FORBIDDEN_INTERNAL = (
    "atreides_vision.interaction",
    "atreides_vision.crowd",
    "atreides_vision.zones",
    "atreides_vision.calibration",
    "atreides_vision.tracking",
    "atreides_vision.detection",
    "atreides_vision.video",
    "atreides_vision.venue",
    "atreides_vision.validation",
    "atreides_vision.visualization",
)

FORBIDDEN_LIBRARIES = ("cv2", "rfdetr", "trackers", "supervision", "torch")

#: Names V0.4.1 must not define. Venue-local identity, fusion state, appearance
#: and any headcount are later capabilities or none at all; a name is the first
#: thing to appear when one starts leaking in.
FORBIDDEN_NAMES = (
    "venuesubject",
    "venue_subject",
    "person_id",
    "same_person",
    "is_same",
    "identity_of",
    "merge",
    "fuse",
    "fusion",
    "embedding",
    "appearance",
    "descriptor",
    "template",
    "histogram",
    "reid",
    "re_id",
    "biometric",
    "duplicate",
    "dedup",
    "unique_people",
    "unique_person",
    "visitor",
    "headcount",
)


def imported_modules(path: Path) -> set[str]:
    """Every module name imported by one file, at any level."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def python_modules(root: Path) -> list[Path]:
    """Every Python module under ``root``, subpackages included."""
    return sorted(root.rglob("*.py"))


def association_modules() -> list[Path]:
    modules = python_modules(ASSOCIATION_ROOT)
    assert modules, "no association modules found — has the package moved?"
    return modules


def unexpected_internal_imports(path: Path) -> set[str]:
    internal = {name for name in imported_modules(path) if name.startswith("atreides_vision")}
    return {name for name in internal if not name.startswith(ALLOWED_INTERNAL_PREFIXES)}


def forbidden_internal_imports(path: Path) -> set[str]:
    imported = imported_modules(path)
    return {
        forbidden
        for forbidden in FORBIDDEN_INTERNAL
        if any(name.startswith(forbidden) for name in imported)
    }


def forbidden_library_imports(path: Path) -> set[str]:
    imported = imported_modules(path)
    return {
        forbidden
        for forbidden in FORBIDDEN_LIBRARIES
        if any(name == forbidden or name.startswith(f"{forbidden}.") for name in imported)
    }


def defined_names(path: Path) -> set[str]:
    """Every class, function and module-level name a file defines."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    }
    defined |= {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    defined |= {
        node.target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    return defined


def identity_shaped_names(path: Path, forbidden: str) -> set[str]:
    return {name for name in defined_names(path) if forbidden in name.lower()}


class TestImportBoundaries:
    @pytest.mark.parametrize("path", association_modules(), ids=lambda path: path.name)
    def test_association_reaches_only_the_shared_domain_vocabulary(self, path: Path) -> None:
        unexpected = unexpected_internal_imports(path)

        assert unexpected == set(), (
            f"{path.name} imports outside the association layer: {unexpected}"
        )

    @pytest.mark.parametrize("path", association_modules(), ids=lambda path: path.name)
    def test_association_imports_no_sibling_signal_or_pixel_layer(self, path: Path) -> None:
        offending = forbidden_internal_imports(path)

        assert offending == set(), f"{path.name} imports {sorted(offending)}"

    @pytest.mark.parametrize("path", association_modules(), ids=lambda path: path.name)
    def test_association_imports_no_model_library(self, path: Path) -> None:
        offending = forbidden_library_imports(path)

        assert offending == set(), (
            f"{path.name} imports {sorted(offending)}; model libraries stay in their adapter"
        )

    @pytest.mark.parametrize(
        "layer", ["tracking", "detection", "behavior", "interaction", "crowd", "zones", "venue"]
    )
    def test_no_lower_layer_imports_association(self, layer: str) -> None:
        """Cross-camera logic never leaks down into a camera-local layer."""
        for path in python_modules(SOURCE_ROOT / layer):
            reached = {
                name
                for name in imported_modules(path)
                if name.startswith("atreides_vision.association")
            }
            assert reached == set(), f"{layer}/{path.name} imports {sorted(reached)}"


class TestNoVenueLocalIdentity:
    def test_subject_ref_is_still_exactly_camera_and_track(self) -> None:
        """The camera-local subject is untouched by the layer built above it."""
        assert [field.name for field in dataclasses.fields(SubjectRef)] == [
            "camera_id",
            "track_id",
        ]

    def test_an_observation_keeps_its_camera_and_stream_provenance(self) -> None:
        """Fusion must never destroy provenance — so nothing collapses it here."""
        subject = observation("cam_b", 4, stream_epoch=3)

        assert subject.subject == SubjectRef("cam_b", 4)
        assert subject.stream == StreamRef("cam_b", 3)

    @pytest.mark.parametrize("forbidden", FORBIDDEN_NAMES)
    def test_no_association_module_defines_an_identity_shaped_name(self, forbidden: str) -> None:
        """Names, not prose: a docstring may say what is *not* built."""
        for path in association_modules():
            offending = identity_shaped_names(path, forbidden)
            assert offending == set(), f"{path.name} defines {offending}"

    def test_no_type_here_carries_a_venue_wide_identifier(self) -> None:
        fields = {
            field.name
            for model in (TrackObservation, CandidatePair, ObservationGating)
            for field in dataclasses.fields(model)
        }

        assert not any("venue_subject" in name or "person" in name for name in fields)

    def test_geometry_cannot_produce_an_association(self) -> None:
        with pytest.raises(AssociationError, match="cannot establish ASSOCIATED"):
            ObservationGating(
                observation=observation(),
                geometric_outcome=AssociationOutcome.ASSOCIATED,
            )

    def test_no_gating_call_returns_associated(self) -> None:
        pool = [observation("cam_b", 4, x=0.0, y=0.0), observation("cam_c", 5, x=0.0, y=0.0)]

        gating = gate_observation(observation("cam_a", 1, x=0.0, y=0.0), pool, GATING)

        assert gating.geometric_outcome is not AssociationOutcome.ASSOCIATED

    def test_the_layer_exposes_no_is_same_person_predicate(self) -> None:
        exposed = {name for name in dir(CandidatePair) if not name.startswith("_")}

        assert exposed == {
            "cameras",
            "first",
            "key",
            "other_than",
            "second",
            "subjects",
            "to_record",
        }

    def test_a_gating_result_exposes_no_merge_or_count(self) -> None:
        exposed = {name for name in dir(ObservationGating) if not name.startswith("_")}

        assert exposed == {
            "ambiguous_cameras",
            "assessments",
            "candidates",
            "comparable",
            "geometric_outcome",
            "observation",
            "subject",
            "to_record",
        }


class TestTheGuardsThemselves:
    """The guards above pass. These prove they would fail if they should."""

    OFFENDING = """
from atreides_vision.calibration.integrity.models import IntegrityState
from atreides_vision.tracking.base import MultiObjectTracker
import cv2


class VenueSubjectId:
    pass


def merge_tracks(left: object, right: object) -> bool:
    return True
"""

    def nested_module(self, tmp_path: Path) -> Path:
        """A module one package deeper than the association root."""
        package = tmp_path / "association" / "identity"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        module = package / "store.py"
        module.write_text(self.OFFENDING, encoding="utf-8")
        return module

    def test_a_nested_module_is_collected_at_all(self, tmp_path: Path) -> None:
        module = self.nested_module(tmp_path)

        collected = python_modules(tmp_path / "association")

        assert module in collected
        assert module not in (tmp_path / "association").glob("*.py")

    def test_the_import_boundary_check_catches_it(self, tmp_path: Path) -> None:
        unexpected = unexpected_internal_imports(self.nested_module(tmp_path))

        assert unexpected == {
            "atreides_vision.calibration.integrity.models",
            "atreides_vision.tracking.base",
        }

    def test_the_layer_check_catches_it(self, tmp_path: Path) -> None:
        offending = forbidden_internal_imports(self.nested_module(tmp_path))

        assert offending == {"atreides_vision.calibration", "atreides_vision.tracking"}

    def test_the_model_library_check_catches_it(self, tmp_path: Path) -> None:
        assert forbidden_library_imports(self.nested_module(tmp_path)) == {"cv2"}

    def test_the_identity_name_check_catches_it(self, tmp_path: Path) -> None:
        module = self.nested_module(tmp_path)

        assert identity_shaped_names(module, "venuesubject") == {"VenueSubjectId"}
        assert identity_shaped_names(module, "merge") == {"merge_tracks"}

    def test_the_calibration_package_really_does_reach_opencv(self) -> None:
        """Why calibration is forbidden rather than merely discouraged."""
        registration = SOURCE_ROOT / "calibration" / "integrity" / "registration.py"

        assert "cv2" in imported_modules(registration)
        assert "atreides_vision.calibration.integrity.registration" in imported_modules(
            SOURCE_ROOT / "calibration" / "integrity" / "__init__.py"
        )
