"""Architectural guards for V0.4.6: occupancy consumes, it never widens the reach.

``occupancy`` sits above ``venue_pipeline`` and reads only its already-produced
:class:`~atreides_vision.venue_pipeline.VenueInstantResult` and
:class:`~atreides_vision.identity.registry.VenueSubjectRegistry` state, plus
the camera-local track reference and ordering helpers association already
defines. These guards check the package reaches no further: no model
library, no sibling signal layer, no fusion internals of its own, and no
other layer reaching back into it uninvited.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "atreides_vision"
OCCUPANCY_ROOT = SOURCE_ROOT / "occupancy"

#: occupancy reads association's observation/ordering vocabulary, identity's
#: registry and venue subject types, the camera-local subject reference,
#: the pipeline result type it consumes, the shared jsonl plumbing for its
#: own reporting seam, and the shared error hierarchy. Nothing else — not
#: even fusion, whose decisions occupancy never re-derives or inspects
#: directly.
ALLOWED_INTERNAL_PREFIXES = (
    "atreides_vision.occupancy",
    "atreides_vision.association",
    "atreides_vision.identity",
    "atreides_vision.behavior.models",
    "atreides_vision.venue_pipeline",
    "atreides_vision.jsonl",
    "atreides_vision.errors",
)

FORBIDDEN_INTERNAL = (
    "atreides_vision.fusion",
    "atreides_vision.interaction",
    "atreides_vision.crowd",
    "atreides_vision.zones",
    "atreides_vision.calibration",
    "atreides_vision.tracking",
    "atreides_vision.detection",
    "atreides_vision.video",
    "atreides_vision.venue.",
    "atreides_vision.timing",
    "atreides_vision.venue_pipeline_reporting",
)

FORBIDDEN_LIBRARIES = ("cv2", "rfdetr", "trackers", "supervision", "torch")

#: This milestone deduplicates already-decided venue identity. It invents no
#: second, competing notion of identity or matching evidence.
FORBIDDEN_NAMES = (
    "embedding",
    "appearance",
    "descriptor",
    "reid",
    "re_id",
    "biometric",
)


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def occupancy_modules() -> list[Path]:
    modules = sorted(OCCUPANCY_ROOT.rglob("*.py"))
    assert modules, "no occupancy modules found — has the package moved?"
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
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    }


class TestImportBoundaries:
    @pytest.mark.parametrize("path", occupancy_modules(), ids=lambda path: path.name)
    def test_occupancy_reaches_only_its_allowed_dependencies(self, path: Path) -> None:
        unexpected = unexpected_internal_imports(path)
        assert unexpected == set(), f"{path.name} imports outside its allowed reach: {unexpected}"

    @pytest.mark.parametrize("path", occupancy_modules(), ids=lambda path: path.name)
    def test_occupancy_does_not_import_fusion_or_a_sibling_signal_layer(self, path: Path) -> None:
        offending = forbidden_internal_imports(path)
        assert offending == set(), f"{path.name} imports {sorted(offending)}"

    @pytest.mark.parametrize("path", occupancy_modules(), ids=lambda path: path.name)
    def test_occupancy_imports_no_model_library(self, path: Path) -> None:
        offending = forbidden_library_imports(path)
        assert offending == set(), (
            f"{path.name} imports {sorted(offending)}; model libraries stay in their adapter"
        )

    @pytest.mark.parametrize(
        "layer",
        ["tracking", "detection", "behavior", "interaction", "crowd", "zones"],
    )
    def test_no_other_layer_imports_occupancy(self, layer: str) -> None:
        """Nothing below or beside this seam reaches into it — this
        milestone is a terminal consumer, never an input to a signal layer."""
        layer_root = SOURCE_ROOT / layer
        if not layer_root.exists():
            return
        for path in sorted(layer_root.rglob("*.py")):
            reached = {
                name
                for name in imported_modules(path)
                if name.startswith("atreides_vision.occupancy")
            }
            assert reached == set(), f"{layer}/{path.name} imports {sorted(reached)}"

    def test_venue_pipeline_reporting_does_not_import_occupancy(self) -> None:
        """V0.4.5's schema stays frozen: occupancy is additive, not a change
        to the existing report."""
        path = SOURCE_ROOT / "venue_pipeline_reporting.py"
        reached = {
            name for name in imported_modules(path) if name.startswith("atreides_vision.occupancy")
        }
        assert reached == set(), f"venue_pipeline_reporting.py imports {sorted(reached)}"


class TestNoReidOrBiometricShapedNames:
    @pytest.mark.parametrize("forbidden", FORBIDDEN_NAMES)
    def test_no_occupancy_module_defines_a_biometric_shaped_name(self, forbidden: str) -> None:
        for path in occupancy_modules():
            offending = {name for name in defined_names(path) if forbidden in name.lower()}
            assert offending == set(), f"{path.name} defines {offending}"
