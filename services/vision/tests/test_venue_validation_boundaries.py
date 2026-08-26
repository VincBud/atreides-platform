"""Architectural guards for V0.4.8: the runner coordinates, it never absorbs.

``venue_validation`` sits at the top of the runtime as a composition root. It is
allowed to reach *down* into the layers it wires together — that is its whole
job — but three things must stay true:

* nothing below it ever reaches back up into it;
* it builds observations only through the V0.4.7 adapter, never by hand;
* it stays a validation harness, with no risk engine, dashboard, ReID or
  persistent-identity ambition leaking into it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "atreides_vision"
PACKAGE_ROOT = SOURCE_ROOT / "venue_validation"
RUNNER_MODULE = PACKAGE_ROOT / "runner.py"
MODULES = sorted(PACKAGE_ROOT.rglob("*.py"))

#: A composition root wires layers together, so its allowed reach is wide. It is
#: bounded by what the documented end-to-end flow actually needs.
ALLOWED_INTERNAL_PREFIXES = (
    "atreides_vision.association",
    "atreides_vision.behavior",
    "atreides_vision.calibration",
    "atreides_vision.config",
    "atreides_vision.detection",
    "atreides_vision.errors",
    "atreides_vision.fusion",
    "atreides_vision.identity",
    "atreides_vision.occupancy",
    "atreides_vision.timing",
    "atreides_vision.track_observation_adapter",
    "atreides_vision.tracking",
    "atreides_vision.venue",
    "atreides_vision.venue_pipeline",
    "atreides_vision.venue_validation",
    "atreides_vision.video",
)

#: Layers that must never appear here: V1 work this milestone does not do, and
#: product surfaces a validation harness has no business touching.
FORBIDDEN_INTERNAL = (
    "atreides_vision.risk",
    "atreides_vision.decision",
    "atreides_vision.dashboard",
    "atreides_vision.api",
)

#: The runner drives the real camera-local pipeline, but through its owner. Model
#: libraries stay in their adapters, several layers below this one.
FORBIDDEN_LIBRARIES = ("cv2", "rfdetr", "trackers", "supervision", "torch")

FORBIDDEN_NAMES = (
    "embedding",
    "appearance",
    "descriptor",
    "reid",
    "re_id",
    "biometric",
    "face",
)

#: Every layer beneath the harness. None of them may import it: a lower layer
#: that knows about its validation runner has been inverted.
LOWER_LAYERS = (
    "association",
    "behavior",
    "calibration",
    "crowd",
    "detection",
    "fusion",
    "identity",
    "interaction",
    "occupancy",
    "tracking",
    "venue",
    "video",
    "zones",
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


def defined_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    }


def called_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


class TestImportBoundaries:
    @pytest.mark.parametrize("module", MODULES, ids=lambda path: path.name)
    def test_the_harness_reaches_only_the_layers_it_wires(self, module: Path) -> None:
        internal = {name for name in imported_modules(module) if name.startswith("atreides_vision")}
        unexpected = {name for name in internal if not name.startswith(ALLOWED_INTERNAL_PREFIXES)}
        assert unexpected == set(), f"{module.name} imports beyond its allowed reach: {unexpected}"

    @pytest.mark.parametrize("module", MODULES, ids=lambda path: path.name)
    def test_no_risk_decision_or_product_surface(self, module: Path) -> None:
        imported = imported_modules(module)
        offending = {
            forbidden
            for forbidden in FORBIDDEN_INTERNAL
            if any(name.startswith(forbidden) for name in imported)
        }
        assert offending == set(), f"{module.name} imports {sorted(offending)}"

    @pytest.mark.parametrize("module", MODULES, ids=lambda path: path.name)
    def test_no_model_library(self, module: Path) -> None:
        imported = imported_modules(module)
        offending = {
            forbidden
            for forbidden in FORBIDDEN_LIBRARIES
            if any(name == forbidden or name.startswith(f"{forbidden}.") for name in imported)
        }
        assert offending == set(), (
            f"{module.name} imports {sorted(offending)}; model libraries stay in their adapter"
        )

    @pytest.mark.parametrize("layer", LOWER_LAYERS)
    def test_no_lower_layer_imports_the_harness(self, layer: str) -> None:
        layer_root = SOURCE_ROOT / layer
        if not layer_root.exists():
            return
        for path in sorted(layer_root.rglob("*.py")):
            reached = {
                name
                for name in imported_modules(path)
                if name.startswith("atreides_vision.venue_validation")
            }
            assert reached == set(), f"{layer}/{path.name} imports {sorted(reached)}"

    def test_neither_the_adapter_nor_the_pipeline_imports_the_harness(self) -> None:
        for module in ("track_observation_adapter.py", "venue_pipeline.py"):
            reached = {
                name
                for name in imported_modules(SOURCE_ROOT / module)
                if name.startswith("atreides_vision.venue_validation")
            }
            assert reached == set(), f"{module} imports {sorted(reached)}"


class TestAdapterIsTheOnlySeam:
    def test_the_runner_calls_the_v0_4_7_adapter(self) -> None:
        assert "build_track_observation" in called_names(RUNNER_MODULE)

    @pytest.mark.parametrize("module", MODULES, ids=lambda path: path.name)
    def test_the_harness_constructs_no_observation_or_spatial_context_itself(
        self, module: Path
    ) -> None:
        """Association's runtime types are built by their owner, not here."""
        constructed = called_names(module) & {
            "TrackObservation",
            "CameraSpatialContext",
            "VenueRegion",
            "VenuePosition",
        }
        assert constructed == set(), f"{module.name} constructs {sorted(constructed)} directly"

    @pytest.mark.parametrize("module", MODULES, ids=lambda path: path.name)
    def test_the_harness_makes_no_association_or_identity_decision(self, module: Path) -> None:
        """Merging, scoring and identity allocation belong to the layers below."""
        decisions = called_names(module) & {
            "assess_pair",
            "decide",
            "merge",
            "allocate",
            "new_subject",
        }
        assert decisions == set(), f"{module.name} decides {sorted(decisions)} for itself"


class TestNoReidOrBiometricShapedNames:
    @pytest.mark.parametrize("module", MODULES, ids=lambda path: path.name)
    def test_the_harness_defines_no_biometric_shaped_name(self, module: Path) -> None:
        for forbidden in FORBIDDEN_NAMES:
            offending = {name for name in defined_names(module) if forbidden in name.lower()}
            assert offending == set(), f"{module.name} defines {offending}"
