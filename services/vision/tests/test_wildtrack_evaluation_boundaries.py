"""Layer-boundary guard for the offline WILDTRACK evaluator package (V0.4.9b-D1).

The evaluator consumes the runtime's frozen domain models and WILDTRACK ground
truth only. It must never reach into the association engine, candidate builder,
gates or matcher, and must never import a model library.
"""

from __future__ import annotations

from pathlib import Path

from atreides_vision.venue_validation.datasets.wildtrack import evaluation

_PACKAGE_DIR = Path(evaluation.__file__).parent

_FORBIDDEN = (
    "association.engine",
    "association.candidates",
    "association.gates",
    "association.matching",
    "import rfdetr",
    "from rfdetr",
    "import cv2",
    "import supervision",
    "from supervision",
    "import trackers",
    "from trackers",
)


def test_no_evaluation_module_crosses_a_layer_boundary() -> None:
    offenders: dict[str, list[str]] = {}
    for source in sorted(_PACKAGE_DIR.glob("*.py")):
        text = source.read_text(encoding="utf-8")
        hits = [token for token in _FORBIDDEN if token in text]
        if hits:
            offenders[source.name] = hits
    assert offenders == {}


def test_the_package_exposes_the_evaluator_surface() -> None:
    for name in ("evaluator", "reporting", "models", "config", "ground_truth"):
        assert (_PACKAGE_DIR / f"{name}.py").is_file()
