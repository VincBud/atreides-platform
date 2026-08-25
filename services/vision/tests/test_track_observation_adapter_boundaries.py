"""Architectural guards for V0.4.7: the adapter converts, it never widens the reach.

``track_observation_adapter`` sits between camera-local runtime/calibration/
timing state and the association contract. These guards check it reaches no
further than the runtime types it is documented to read: no fusion, identity,
occupancy or risk layer, no sibling signal layer, no model library, and no
ReID/embedding/biometric-shaped name anywhere in it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "atreides_vision"
ADAPTER_MODULE = SOURCE_ROOT / "track_observation_adapter.py"

#: The adapter reads camera-local behaviour output, venue-time framing and
#: calibration/integrity state, and writes association's own observation
#: type. Nothing else.
ALLOWED_INTERNAL_PREFIXES = (
    "atreides_vision.association",
    "atreides_vision.behavior",
    "atreides_vision.calibration",
    "atreides_vision.timing",
    "atreides_vision.errors",
)

FORBIDDEN_INTERNAL = (
    "atreides_vision.interaction",
    "atreides_vision.crowd",
    "atreides_vision.zones",
    "atreides_vision.tracking",
    "atreides_vision.detection",
    "atreides_vision.video",
    "atreides_vision.venue",
    "atreides_vision.identity",
    "atreides_vision.fusion",
    "atreides_vision.venue_pipeline",
)

FORBIDDEN_LIBRARIES = ("cv2", "rfdetr", "trackers", "supervision", "torch")

#: This milestone converts already-produced runtime state. It invents no
#: appearance, embedding or biometric evidence of its own.
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
    def test_adapter_reaches_only_its_documented_runtime_sources(self) -> None:
        unexpected = unexpected_internal_imports(ADAPTER_MODULE)
        assert unexpected == set(), (
            f"track_observation_adapter.py imports beyond its allowed reach: {unexpected}"
        )

    def test_adapter_does_not_import_fusion_identity_occupancy_or_the_pipeline(self) -> None:
        offending = forbidden_internal_imports(ADAPTER_MODULE)
        assert offending == set(), f"track_observation_adapter.py imports {sorted(offending)}"

    def test_adapter_does_not_import_a_sibling_signal_layer(self) -> None:
        offending = forbidden_internal_imports(ADAPTER_MODULE)
        assert offending == set(), f"track_observation_adapter.py imports {sorted(offending)}"

    def test_adapter_imports_no_model_library(self) -> None:
        offending = forbidden_library_imports(ADAPTER_MODULE)
        assert offending == set(), (
            f"track_observation_adapter.py imports {sorted(offending)}; "
            "model libraries stay in their adapter"
        )

    @pytest.mark.parametrize(
        "layer",
        ["tracking", "detection", "interaction", "crowd", "zones", "identity", "fusion"],
    )
    def test_no_other_layer_imports_the_adapter(self, layer: str) -> None:
        """Nothing below or beside this seam reaches into it."""
        layer_root = SOURCE_ROOT / layer
        if not layer_root.exists():
            return
        for path in sorted(layer_root.rglob("*.py")):
            reached = {
                name
                for name in imported_modules(path)
                if name.startswith("atreides_vision.track_observation_adapter")
            }
            assert reached == set(), f"{layer}/{path.name} imports {sorted(reached)}"


class TestNoReidOrBiometricShapedNames:
    @pytest.mark.parametrize("forbidden", FORBIDDEN_NAMES)
    def test_the_adapter_module_defines_no_biometric_shaped_name(self, forbidden: str) -> None:
        offending = {name for name in defined_names(ADAPTER_MODULE) if forbidden in name.lower()}
        assert offending == set(), f"track_observation_adapter.py defines {offending}"
