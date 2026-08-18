"""Architectural guards for V0.4.3b: the integration layer stays a seam, not a shortcut.

``fusion`` is the one place ``association`` and ``identity`` are both in
scope — that is its entire purpose — so it may import both. These guards
check it reaches no further: no sibling signal layer, no model library, and
no other layer reaching back into it uninvited.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "atreides_vision"
FUSION_ROOT = SOURCE_ROOT / "fusion"

#: fusion/ is the seam between association and identity, plus its own
#: package and the shared error hierarchy. Nothing else.
ALLOWED_INTERNAL_PREFIXES = (
    "atreides_vision.fusion",
    "atreides_vision.association",
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

#: This milestone applies association's own decisions to identity's own
#: registry. It never invents a second, competing notion of identity.
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


def fusion_modules() -> list[Path]:
    modules = sorted(FUSION_ROOT.rglob("*.py"))
    assert modules, "no fusion modules found — has the package moved?"
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
    @pytest.mark.parametrize("path", fusion_modules(), ids=lambda path: path.name)
    def test_fusion_reaches_only_association_identity_and_errors(self, path: Path) -> None:
        unexpected = unexpected_internal_imports(path)
        assert unexpected == set(), f"{path.name} imports outside its allowed reach: {unexpected}"

    @pytest.mark.parametrize("path", fusion_modules(), ids=lambda path: path.name)
    def test_fusion_does_not_import_a_sibling_signal_layer(self, path: Path) -> None:
        offending = forbidden_internal_imports(path)
        assert offending == set(), f"{path.name} imports {sorted(offending)}"

    @pytest.mark.parametrize("path", fusion_modules(), ids=lambda path: path.name)
    def test_fusion_imports_no_model_library(self, path: Path) -> None:
        offending = forbidden_library_imports(path)
        assert offending == set(), (
            f"{path.name} imports {sorted(offending)}; model libraries stay in their adapter"
        )

    @pytest.mark.parametrize(
        "layer",
        ["tracking", "detection", "behavior", "interaction", "crowd", "zones"],
    )
    def test_no_other_layer_imports_fusion(self, layer: str) -> None:
        """Nothing below or beside this layer reaches into it — this
        milestone wires no consumer of its own."""
        for path in sorted((SOURCE_ROOT / layer).rglob("*.py")):
            reached = {
                name for name in imported_modules(path) if name.startswith("atreides_vision.fusion")
            }
            assert reached == set(), f"{layer}/{path.name} imports {sorted(reached)}"


class TestNoReidOrBiometricShapedNames:
    @pytest.mark.parametrize("forbidden", FORBIDDEN_NAMES)
    def test_no_fusion_module_defines_a_biometric_shaped_name(self, forbidden: str) -> None:
        for path in fusion_modules():
            offending = {name for name in defined_names(path) if forbidden in name.lower()}
            assert offending == set(), f"{path.name} defines {offending}"
