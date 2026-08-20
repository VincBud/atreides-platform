"""Architectural guards for V0.4.4: the pipeline wires, it never widens the reach.

``venue_pipeline`` is the runtime seam that calls ``association`` and
``fusion`` in sequence for one venue session — the same three packages
``fusion`` itself is already permitted to see, plus ``fusion`` itself. These
guards check the module reaches no further: no model library, no sibling
signal layer, no camera-local track/video/calibration access, and no other
layer reaching back into it uninvited.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "atreides_vision"
PIPELINE_MODULE = SOURCE_ROOT / "venue_pipeline.py"

#: The pipeline wires association's decisions into identity through fusion's
#: own policy, plus the shared error hierarchy. Nothing else.
ALLOWED_INTERNAL_PREFIXES = (
    "atreides_vision.association",
    "atreides_vision.fusion",
    "atreides_vision.identity",
    "atreides_vision.errors",
)

FORBIDDEN_INTERNAL = (
    "atreides_vision.interaction",
    "atreides_vision.crowd",
    "atreides_vision.zones",
    "atreides_vision.calibration",
    "atreides_vision.tracking",
    "atreides_vision.detection",
    "atreides_vision.video",
    "atreides_vision.venue",
    "atreides_vision.behavior",
    "atreides_vision.timing",
)

FORBIDDEN_LIBRARIES = ("cv2", "rfdetr", "trackers", "supervision", "torch")

#: This milestone wires already-built observations through already-built
#: association and fusion policy. It invents no second, competing notion of
#: identity or matching.
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
    def test_pipeline_reaches_only_association_fusion_identity_and_errors(self) -> None:
        unexpected = unexpected_internal_imports(PIPELINE_MODULE)
        assert unexpected == set(), (
            f"venue_pipeline.py imports beyond its allowed reach: {unexpected}"
        )

    def test_pipeline_does_not_import_a_sibling_signal_layer(self) -> None:
        offending = forbidden_internal_imports(PIPELINE_MODULE)
        assert offending == set(), f"venue_pipeline.py imports {sorted(offending)}"

    def test_pipeline_imports_no_model_library(self) -> None:
        offending = forbidden_library_imports(PIPELINE_MODULE)
        assert offending == set(), (
            f"venue_pipeline.py imports {sorted(offending)}; model libraries stay in their adapter"
        )

    @pytest.mark.parametrize(
        "layer",
        ["tracking", "detection", "behavior", "interaction", "crowd", "zones"],
    )
    def test_no_other_layer_imports_the_pipeline(self, layer: str) -> None:
        """Nothing below or beside this seam reaches into it — this
        milestone wires no consumer of its own."""
        layer_root = SOURCE_ROOT / layer
        if not layer_root.exists():
            return
        for path in sorted(layer_root.rglob("*.py")):
            reached = {
                name
                for name in imported_modules(path)
                if name.startswith("atreides_vision.venue_pipeline")
            }
            assert reached == set(), f"{layer}/{path.name} imports {sorted(reached)}"


class TestNoReidOrBiometricShapedNames:
    @pytest.mark.parametrize("forbidden", FORBIDDEN_NAMES)
    def test_the_pipeline_module_defines_no_biometric_shaped_name(self, forbidden: str) -> None:
        offending = {name for name in defined_names(PIPELINE_MODULE) if forbidden in name.lower()}
        assert offending == set(), f"venue_pipeline.py defines {offending}"
