"""Architectural guards for V0.4.3a: the identity layer stays self-contained.

This milestone deliberately does not consume ``association`` — see
``identity/__init__.py``. These guards make that a checked fact, not just a
docstring claim, and keep model libraries and sibling signal layers out too.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "atreides_vision"
IDENTITY_ROOT = SOURCE_ROOT / "identity"

#: identity/ consumes the camera-local subject reference and the venue time
#: domain, plus its own package. Nothing else — not even association, which
#: is the whole point of this sub-milestone.
ALLOWED_INTERNAL_PREFIXES = (
    "atreides_vision.identity",
    "atreides_vision.behavior.models",
    "atreides_vision.timing",
    "atreides_vision.errors",
)

FORBIDDEN_INTERNAL = (
    "atreides_vision.association",
    "atreides_vision.interaction",
    "atreides_vision.crowd",
    "atreides_vision.zones",
    "atreides_vision.calibration",
    "atreides_vision.tracking",
    "atreides_vision.detection",
    "atreides_vision.video",
    "atreides_vision.venue",
)

FORBIDDEN_LIBRARIES = ("cv2", "rfdetr", "trackers", "supervision", "torch")

#: This milestone builds representation and registry state only — no decision
#: about which tracklets belong together, no automatic merge/split/dedup.
FORBIDDEN_NAMES = (
    "merge",
    "fuse",
    "fusion",
    "embedding",
    "appearance",
    "descriptor",
    "reid",
    "re_id",
    "biometric",
    "dedup",
    "duplicate_suppress",
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


def identity_modules() -> list[Path]:
    modules = sorted(IDENTITY_ROOT.rglob("*.py"))
    assert modules, "no identity modules found — has the package moved?"
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
    @pytest.mark.parametrize("path", identity_modules(), ids=lambda path: path.name)
    def test_identity_reaches_only_the_shared_domain_vocabulary(self, path: Path) -> None:
        unexpected = unexpected_internal_imports(path)
        assert unexpected == set(), f"{path.name} imports outside its allowed reach: {unexpected}"

    @pytest.mark.parametrize("path", identity_modules(), ids=lambda path: path.name)
    def test_identity_does_not_import_association_or_a_sibling_layer(self, path: Path) -> None:
        offending = forbidden_internal_imports(path)
        assert offending == set(), f"{path.name} imports {sorted(offending)}"

    @pytest.mark.parametrize("path", identity_modules(), ids=lambda path: path.name)
    def test_identity_imports_no_model_library(self, path: Path) -> None:
        offending = forbidden_library_imports(path)
        assert offending == set(), (
            f"{path.name} imports {sorted(offending)}; model libraries stay in their adapter"
        )

    @pytest.mark.parametrize(
        "layer",
        ["tracking", "detection", "behavior", "association", "interaction", "crowd", "zones"],
    )
    def test_no_other_layer_imports_identity(self, layer: str) -> None:
        """Nothing below or beside this layer reaches into it yet — this
        milestone wires no consumer."""
        for path in sorted((SOURCE_ROOT / layer).rglob("*.py")):
            reached = {
                name
                for name in imported_modules(path)
                if name.startswith("atreides_vision.identity")
            }
            assert reached == set(), f"{layer}/{path.name} imports {sorted(reached)}"


class TestNoMergeOrFusionShapedNames:
    @pytest.mark.parametrize("forbidden", FORBIDDEN_NAMES)
    def test_no_identity_module_defines_a_fusion_shaped_name(self, forbidden: str) -> None:
        for path in identity_modules():
            offending = {name for name in defined_names(path) if forbidden in name.lower()}
            assert offending == set(), f"{path.name} defines {offending}"
