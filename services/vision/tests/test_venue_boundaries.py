"""Architectural guards for V0.4.0: layer boundaries, and the absence of association.

These are not style checks. The venue layer sits below every signal layer, so an
import in the wrong direction would let ingestion depend on measurement; and
V0.4.0 deliberately performs no cross-camera association, which is only credible
if nothing here quietly grows one.

What these guards do and do not see
-----------------------------------
Every module **anywhere under** ``venue/`` is walked, subpackages included: a
future ``venue/association/matcher.py`` is exactly the file these guards exist
to catch, and a non-recursive scan would never look at it.
:class:`TestTheGuardsThemselves` proves that by running each check against a
deliberately offending nested module.

The allowed set is narrowed to the specific modules ``venue/`` actually reaches.
``atreides_vision.video`` as a whole would be far too wide: ``video.processor``
transitively imports every signal layer, so permitting the package would let any
of them in through one hop.

**Residual, documented rather than silent:** an AST walk sees ``import`` and
``from ... import`` statements only. A dynamic ``importlib.import_module(...)``
would be invisible to all four checks. Nothing in the codebase does that today,
and closing it would need a runtime import graph rather than a source scan.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import pytest

from atreides_vision.behavior.models import SubjectRef
from atreides_vision.timing import FrameRateSource, StreamRef, TimeUncertainty
from atreides_vision.venue.models import VenueTick
from atreides_vision.venue.reporting import (
    JsonlVenueSessionWriter,
    VenueSessionMetadata,
    VenueSessionSummary,
    VenueStreamProvenance,
)
from atreides_vision.venue.session import VenueSession, VideoSourceCursor
from atreides_vision.venue.timebase import VenueTimebase
from venue_fakes import CountingSource

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "atreides_vision"
VENUE_ROOT = SOURCE_ROOT / "venue"

#: The venue layer may reach these and nothing else inside the package. It
#: consumes video and time; it must never consume a measurement layer, because
#: ingestion happens below all of them.
#: Narrowed to the exact modules venue/ reaches. Not "atreides_vision.video":
#: video.processor transitively imports every layer below, so allowing the
#: package would readmit all of them through a single hop.
ALLOWED_INTERNAL_PREFIXES = (
    "atreides_vision.venue",
    "atreides_vision.video.source",
    "atreides_vision.timing",
    "atreides_vision.errors",
    "atreides_vision.jsonl",
)

FORBIDDEN_INTERNAL = (
    "atreides_vision.behavior",
    "atreides_vision.interaction",
    "atreides_vision.crowd",
    "atreides_vision.zones",
    "atreides_vision.calibration",
    "atreides_vision.tracking",
    "atreides_vision.detection",
    "atreides_vision.validation",
    "atreides_vision.visualization",
)

#: Model libraries stay in their adapter files. Ingestion decodes through the
#: video layer's interface, so none of these belongs here.
FORBIDDEN_LIBRARIES = ("cv2", "rfdetr", "trackers", "supervision", "torch")


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
    """Every Python module under ``root``, subpackages included.

    ``rglob``, deliberately: a guard that only sees the top level of a package
    stops guarding the moment somebody adds a subpackage — which is the shape a
    future association layer would arrive in.
    """
    return sorted(root.rglob("*.py"))


def venue_modules() -> list[Path]:
    modules = python_modules(VENUE_ROOT)
    assert modules, "no venue modules found — has the package moved?"
    return modules


def unexpected_internal_imports(path: Path) -> set[str]:
    """Atreides imports this module makes that the venue layer may not make."""
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


def association_shaped_names(path: Path, forbidden: str) -> set[str]:
    return {name for name in defined_names(path) if forbidden in name.lower()}


class TestImportBoundaries:
    @pytest.mark.parametrize("path", venue_modules(), ids=lambda path: path.name)
    def test_venue_modules_import_only_video_source_timing_errors_and_jsonl(
        self, path: Path
    ) -> None:
        unexpected = unexpected_internal_imports(path)

        assert unexpected == set(), f"{path.name} imports outside the venue layer: {unexpected}"

    @pytest.mark.parametrize("path", venue_modules(), ids=lambda path: path.name)
    def test_venue_modules_import_no_signal_layer(self, path: Path) -> None:
        offending = forbidden_internal_imports(path)

        assert offending == set(), f"{path.name} imports {sorted(offending)}"

    @pytest.mark.parametrize("path", venue_modules(), ids=lambda path: path.name)
    def test_venue_modules_import_no_model_library(self, path: Path) -> None:
        offending = forbidden_library_imports(path)

        assert offending == set(), (
            f"{path.name} imports {sorted(offending)}; model libraries stay in their adapter"
        )

    def test_timing_depends_on_nothing_but_the_error_hierarchy(self) -> None:
        internal = {
            name
            for name in imported_modules(SOURCE_ROOT / "timing.py")
            if name.startswith("atreides_vision")
        }

        assert internal == {"atreides_vision.errors"}


class TestNoAssociation:
    def test_subject_ref_is_still_exactly_camera_and_track(self) -> None:
        """Venue identity must not have leaked into the camera-local subject."""
        assert [field.name for field in dataclasses.fields(SubjectRef)] == [
            "camera_id",
            "track_id",
        ]

    def test_a_stream_reference_is_not_a_subject_reference(self) -> None:
        assert not isinstance(StreamRef("cam_a", 0), SubjectRef)
        assert str(StreamRef("cam_a", 7)) != str(SubjectRef("cam_a", 7))
        assert "#" in str(StreamRef("cam_a", 7))
        assert "/" in str(SubjectRef("cam_a", 7))

    @pytest.mark.parametrize(
        "forbidden",
        [
            "venuesubject",
            "venue_subject",
            "person_id",
            "same_person",
            "is_same",
            "embedding",
            "appearance",
            "descriptor",
            "reid",
            "re_id",
            "match_score",
            "candidate_pair",
            "distance_matrix",
            "duplicate_count",
            "deduplicate",
            "unique_people",
            "unique_person",
            "visitor",
            "biometric",
            "identity_of",
        ],
    )
    def test_no_venue_module_defines_an_association_shaped_name(self, forbidden: str) -> None:
        """Names, not prose: a docstring may say what is *not* built."""
        for path in [*venue_modules(), SOURCE_ROOT / "timing.py"]:
            offending = association_shaped_names(path, forbidden)
            assert offending == set(), f"{path.name} defines {offending}"

    def test_a_tick_exposes_streams_and_never_a_merged_entity(self) -> None:
        exposed = {name for name in dir(VenueTick) if not name.startswith("_")}

        assert exposed == {
            "absent",
            "activity_of",
            "alignment",
            "cameras",
            "contributions",
            "sequence",
            "venue_micros",
            "venue_seconds",
        }

    def test_uncertainty_yields_no_boolean_a_merge_rule_could_use(self) -> None:
        """``is_established`` says whether anything is known — not whether to merge."""
        exposed = {name for name in dir(TimeUncertainty) if not name.startswith("_")}

        assert exposed == {
            "basis",
            "declared",
            "half_width_seconds",
            "is_established",
            "unknown",
            "worst_of",
        }

    def test_two_cameras_at_one_instant_produce_two_observations(self) -> None:
        """The whole point: co-occurrence is recorded, never resolved."""
        timebase = VenueTimebase()
        streams = (StreamRef("cam_a", 0), StreamRef("cam_b", 0))
        for stream in streams:
            timebase.register(stream, fps_source=FrameRateSource.CONTAINER)
        session = VenueSession(
            "session_a",
            timebase,
            [
                VideoSourceCursor(stream, CountingSource(frame_count=3, fps=25.0), timebase)
                for stream in streams
            ],
        )

        tick = next(iter(session.ticks()))

        assert len(tick.contributions) == 2
        assert tick.cameras() == ("cam_a", "cam_b")
        # Two camera-local observations. Nothing says they are one person, and
        # there is no count of people anywhere to be wrong about.
        assert {contribution.stream for contribution in tick.contributions} == set(streams)

    def test_the_session_record_contains_no_identity_field(self, tmp_path: Path) -> None:
        stream = StreamRef("cam_a", 0)
        timebase = VenueTimebase()
        timebase.register(stream, fps_source=FrameRateSource.CONTAINER)
        cursor = VideoSourceCursor(stream, CountingSource(frame_count=2, fps=25.0), timebase)
        session = VenueSession("session_a", timebase, [cursor])
        path = tmp_path / "session.jsonl"

        with JsonlVenueSessionWriter(
            path,
            VenueSessionMetadata(
                session_id=session.session_id,
                streams=(
                    VenueStreamProvenance(
                        stream=stream,
                        input_path="/clips/a.mp4",
                        width=8,
                        height=6,
                        fps=25.0,
                        fps_source=FrameRateSource.CONTAINER,
                        offset_seconds=0.0,
                        timing=timebase.uncertainty(stream),
                    ),
                ),
            ),
        ) as writer:
            for tick in session.ticks():
                writer.write_tick(tick)
            writer.write_summary(
                VenueSessionSummary(
                    session_id=session.session_id,
                    tick_count=session.tick_count,
                    frames_by_stream=session.frames_by_stream,
                    first_venue_micros=session.first_venue_micros,
                    last_venue_micros=session.last_venue_micros,
                )
            )

        keys = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            keys |= _keys(json.loads(line))

        for forbidden in (
            "venue_subject_id",
            "person_id",
            "subject_id",
            "track_id",
            "identity",
            "match",
            "embedding",
            "appearance",
            "unique_people",
            "duplicates_suppressed",
        ):
            assert forbidden not in keys, f"the session record exposes '{forbidden}'"


def _keys(value: object) -> set[str]:
    """Every key anywhere in a nested JSON record."""
    if isinstance(value, dict):
        found = set(value.keys())
        for item in value.values():
            found |= _keys(item)
        return found
    if isinstance(value, list):
        found = set()
        for item in value:
            found |= _keys(item)
        return found
    return set()


class TestTheGuardsThemselves:
    """The guards above pass. These prove they would fail if they should.

    A structural check that never fails is indistinguishable from one that
    cannot fail, and the failure mode here is specific: a non-recursive scan
    silently stops covering a package the moment a subpackage appears. So each
    check is run against a deliberately offending module in the shape it would
    actually arrive in — ``venue/association/matcher.py``.
    """

    OFFENDING = """
from atreides_vision.behavior.models import SubjectRef
from atreides_vision.tracking.base import MultiObjectTracker
import cv2


class VenueSubjectId:
    pass


def is_same_person(left: SubjectRef, right: SubjectRef) -> bool:
    return True
"""

    def nested_module(self, tmp_path: Path) -> Path:
        """A module one package deeper than the venue root, as a new layer would be."""
        package = tmp_path / "venue" / "association"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        module = package / "matcher.py"
        module.write_text(self.OFFENDING, encoding="utf-8")
        return module

    def test_a_nested_module_is_collected_at_all(self, tmp_path: Path) -> None:
        """The rglob/glob difference, which is the whole failure mode."""
        module = self.nested_module(tmp_path)

        collected = python_modules(tmp_path / "venue")

        assert module in collected
        assert module not in (tmp_path / "venue").glob("*.py")

    def test_the_import_boundary_check_catches_it(self, tmp_path: Path) -> None:
        unexpected = unexpected_internal_imports(self.nested_module(tmp_path))

        assert unexpected == {
            "atreides_vision.behavior.models",
            "atreides_vision.tracking.base",
        }

    def test_the_signal_layer_check_catches_it(self, tmp_path: Path) -> None:
        offending = forbidden_internal_imports(self.nested_module(tmp_path))

        assert offending == {"atreides_vision.behavior", "atreides_vision.tracking"}

    def test_the_model_library_check_catches_it(self, tmp_path: Path) -> None:
        assert forbidden_library_imports(self.nested_module(tmp_path)) == {"cv2"}

    def test_the_association_name_check_catches_it(self, tmp_path: Path) -> None:
        module = self.nested_module(tmp_path)

        assert association_shaped_names(module, "venuesubject") == {"VenueSubjectId"}
        assert association_shaped_names(module, "same_person") == {"is_same_person"}

    def test_the_video_package_is_not_wholesale_allowed(self) -> None:
        """video.processor reaches every layer below; permitting it readmits them."""
        processor = SOURCE_ROOT / "video" / "processor.py"
        reached = {
            forbidden
            for forbidden in FORBIDDEN_INTERNAL
            if any(name.startswith(forbidden) for name in imported_modules(processor))
        }

        assert reached, "video/processor.py no longer imports any signal layer"
        assert not "atreides_vision.video".startswith(ALLOWED_INTERNAL_PREFIXES)
        assert "atreides_vision.video.source".startswith(ALLOWED_INTERNAL_PREFIXES)
