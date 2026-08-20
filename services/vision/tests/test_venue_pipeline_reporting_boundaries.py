"""Architectural guards for V0.4.5: reporting observes, it never widens reach.

``venue_pipeline_reporting`` reads a :class:`~atreides_vision.venue_pipeline.VenueInstantResult`
and a :class:`~atreides_vision.identity.registry.VenueSubjectRegistry` and
turns them into records. It has no reason to reach any further than
``venue_pipeline`` itself already reaches, plus ``venue_pipeline`` and
``jsonl`` for the pieces it adds. These guards check the module reaches no
further: no model library, no sibling signal layer, no camera-local
track/video/calibration access, and no signal layer reaching back into it
uninvited.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "atreides_vision"
REPORTING_MODULE = SOURCE_ROOT / "venue_pipeline_reporting.py"

#: Reporting reads association/fusion/identity output and the pipeline result
#: type that bundles them, writes through the shared jsonl plumbing, and
#: raises the shared error hierarchy. Nothing else.
ALLOWED_INTERNAL_PREFIXES = (
    "atreides_vision.association",
    "atreides_vision.fusion",
    "atreides_vision.identity",
    "atreides_vision.venue_pipeline",
    "atreides_vision.errors",
    "atreides_vision.jsonl",
)

FORBIDDEN_INTERNAL = (
    "atreides_vision.interaction",
    "atreides_vision.crowd",
    "atreides_vision.zones",
    "atreides_vision.calibration",
    "atreides_vision.tracking",
    "atreides_vision.detection",
    "atreides_vision.video",
    "atreides_vision.venue.",
    "atreides_vision.behavior",
    "atreides_vision.timing",
)

FORBIDDEN_LIBRARIES = ("cv2", "rfdetr", "trackers", "supervision", "torch")

#: This milestone reports already-built association/fusion/identity outcomes.
#: It invents no second, competing notion of identity or matching evidence.
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
    def test_reporting_reaches_only_its_allowed_dependencies(self) -> None:
        unexpected = unexpected_internal_imports(REPORTING_MODULE)
        assert unexpected == set(), (
            f"venue_pipeline_reporting.py imports beyond its allowed reach: {unexpected}"
        )

    def test_reporting_does_not_import_a_sibling_signal_layer(self) -> None:
        offending = forbidden_internal_imports(REPORTING_MODULE)
        assert offending == set(), f"venue_pipeline_reporting.py imports {sorted(offending)}"

    def test_reporting_imports_no_model_library(self) -> None:
        offending = forbidden_library_imports(REPORTING_MODULE)
        assert offending == set(), (
            f"venue_pipeline_reporting.py imports {sorted(offending)}; "
            "model libraries stay in their adapter"
        )

    @pytest.mark.parametrize(
        "layer",
        ["tracking", "detection", "behavior", "interaction", "crowd", "zones"],
    )
    def test_no_other_layer_imports_the_reporting_module(self, layer: str) -> None:
        """Nothing below or beside this seam reaches into it — this
        milestone reports to an external consumer, never a signal layer."""
        layer_root = SOURCE_ROOT / layer
        if not layer_root.exists():
            return
        for path in sorted(layer_root.rglob("*.py")):
            reached = {
                name
                for name in imported_modules(path)
                if name.startswith("atreides_vision.venue_pipeline_reporting")
            }
            assert reached == set(), f"{layer}/{path.name} imports {sorted(reached)}"


class TestNoReidOrBiometricShapedNames:
    @pytest.mark.parametrize("forbidden", FORBIDDEN_NAMES)
    def test_the_reporting_module_defines_no_biometric_shaped_name(self, forbidden: str) -> None:
        offending = {name for name in defined_names(REPORTING_MODULE) if forbidden in name.lower()}
        assert offending == set(), f"venue_pipeline_reporting.py defines {offending}"
