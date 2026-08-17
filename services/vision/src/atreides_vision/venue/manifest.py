"""The venue session manifest: which cameras, which recordings, and when.

A manifest rather than a directory scan or a naming convention, because
everything in it is *deployment configuration* that Atreides must never assume:
how many cameras there are, what they are called, how many recordings each
produced, whether any of them start at the same moment, and what frame rate they
really run at. Nothing here has a default that describes a venue.

What a manifest asserts, and what it does not
---------------------------------------------
It asserts that these files came from these cameras during one session, and —
where an operator says so — how their clocks relate. It asserts **nothing**
about the cameras' geometry: no layout, no overlap, no adjacency, no coverage.
Non-overlapping cameras are an ordinary case, not a missing configuration.

Several sources per camera
--------------------------
A camera that stops and starts during a session contributes several sources.
Their position in the list is the ``stream_epoch``, and crossing that boundary
breaks track identity: ``(cam_01, track_7)`` before and after are unrelated
tracklets. Because those recordings are sequential in venue time rather than
simultaneous, a camera with more than one source **must** declare where each one
starts; otherwise every epoch would be placed at venue zero, stacking a whole
session on top of itself.

Offsets and their accuracy travel together
------------------------------------------
``start_offset_seconds`` places a source on the session timeline.
``offset_accuracy_seconds`` says how well it is known, and is **required**
whenever an offset is declared: an offset with no stated accuracy would either
be read as exact — which no manually declared offset is — or be quietly
defaulted to zero uncertainty, which is worse. Declaring neither is the honest
and supported case: the source starts at venue zero and its alignment is
``NOT_ESTABLISHED``.

``declared_fps`` exists because real CCTV exports routinely report a frame rate
they do not have, and every source-relative timestamp is derived from it. It is
an operator statement about their own recording, recorded as such.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atreides_vision.errors import VenueSessionError

__all__ = [
    "VENUE_MANIFEST_VERSION",
    "VenueCamera",
    "VenueSessionManifest",
    "VenueStreamEntry",
    "load_venue_manifest",
]

#: Version of the manifest contract this loader understands. A manifest naming a
#: different one is rejected rather than parsed on a guess.
VENUE_MANIFEST_VERSION = "0.4.0"


@dataclass(frozen=True, slots=True)
class VenueStreamEntry:
    """One recording from one camera, and what the operator states about it."""

    input_path: Path
    #: Where this recording starts on the session timeline. ``None`` means it
    #: starts at venue zero with its alignment ``NOT_ESTABLISHED`` — a default,
    #: never a fallback for a declaration that failed to parse.
    start_offset_seconds: float | None = None
    #: Symmetric accuracy of ``start_offset_seconds``. Required with it, and
    #: meaningless without it.
    offset_accuracy_seconds: float | None = None
    #: The recording's real frame rate, overriding whatever the container says.
    declared_fps: float | None = None

    def __post_init__(self) -> None:
        if self.start_offset_seconds is None and self.offset_accuracy_seconds is not None:
            raise VenueSessionError(
                f"{self.input_path}: 'offset_accuracy_seconds' describes the accuracy of a "
                "'start_offset_seconds' that was not declared. Declare both, or neither."
            )
        if self.start_offset_seconds is not None:
            if not math.isfinite(self.start_offset_seconds) or self.start_offset_seconds < 0:
                raise VenueSessionError(
                    f"{self.input_path}: 'start_offset_seconds' must be finite and "
                    f"non-negative, got {self.start_offset_seconds!r}. Venue time is "
                    "session-relative, so nothing starts before zero."
                )
            if self.offset_accuracy_seconds is None:
                raise VenueSessionError(
                    f"{self.input_path}: a declared 'start_offset_seconds' needs an "
                    "'offset_accuracy_seconds'. An offset with no stated accuracy would be "
                    "read as exact, and no manually declared offset is. Omit both to run this "
                    "source with its alignment NOT_ESTABLISHED."
                )
            if not math.isfinite(self.offset_accuracy_seconds) or self.offset_accuracy_seconds <= 0:
                raise VenueSessionError(
                    f"{self.input_path}: 'offset_accuracy_seconds' must be finite and strictly "
                    f"positive, got {self.offset_accuracy_seconds!r}. Zero would claim the "
                    "offset is exact."
                )
        if self.declared_fps is not None and (
            not math.isfinite(self.declared_fps) or self.declared_fps <= 0
        ):
            raise VenueSessionError(
                f"{self.input_path}: 'declared_fps' must be finite and positive, got "
                f"{self.declared_fps!r}. Omit it to use the container's own frame rate."
            )

    @property
    def has_declared_offset(self) -> bool:
        return self.start_offset_seconds is not None


@dataclass(frozen=True, slots=True)
class VenueCamera:
    """One camera and its recordings, in the order that defines their epochs.

    A camera's sources are **sequential**: it recorded one, stopped, and
    recorded the next. That is enforced as far as it can be — every source must
    declare where it starts, and those starts must strictly increase, because a
    camera cannot record two things at once and two segments sharing a start
    would contribute two frames at every venue instant while looking like one
    camera to anything reading per-camera state.

    What is **not** checked: that segment *N + 1* begins after segment *N*
    actually ends. That needs each segment's duration, which means trusting the
    container's ``frame_count``, and this layer deliberately never does — the
    schedule is built by reading until a stream is exhausted. So overlapping
    recordings whose declared starts happen to increase are accepted, and the
    ordering guarantee here is strictly weaker than non-overlap.
    """

    camera_id: str
    sources: tuple[VenueStreamEntry, ...]

    def __post_init__(self) -> None:
        if not self.camera_id or not self.camera_id.strip():
            raise VenueSessionError("A venue camera needs a non-empty camera_id.")
        if not self.sources:
            raise VenueSessionError(
                f"Camera '{self.camera_id}' has no sources. A camera with nothing to read is "
                "configuration nobody can act on."
            )
        if len(self.sources) == 1:
            return

        offsets = [source.start_offset_seconds for source in self.sources]
        if not all(offset is not None for offset in offsets):
            raise VenueSessionError(
                f"Camera '{self.camera_id}' has {len(self.sources)} sources, so every one of "
                "them needs a 'start_offset_seconds'. Sequential recordings from one camera "
                "start at different points in the session; without offsets they would all be "
                "placed at venue zero and play over each other."
            )

        declared = [offset for offset in offsets if offset is not None]
        for epoch in range(1, len(declared)):
            if declared[epoch] > declared[epoch - 1]:
                continue
            raise VenueSessionError(
                f"Camera '{self.camera_id}' source #{epoch + 1} starts at "
                f"{declared[epoch]} s, which is not after source #{epoch} at "
                f"{declared[epoch - 1]} s. A camera cannot record two things at once: its "
                "sources are sequential recordings, and equal or decreasing offsets would lay "
                "two of them over each other, contributing two frames at one venue instant "
                "from one camera. List them in recording order with strictly increasing "
                "offsets."
            )

    @property
    def is_multi_source(self) -> bool:
        """Whether this camera stopped and restarted during the session."""
        return len(self.sources) > 1

    def __len__(self) -> int:
        return len(self.sources)

    def __iter__(self) -> Iterator[VenueStreamEntry]:
        return iter(self.sources)


@dataclass(frozen=True, slots=True)
class VenueSessionManifest:
    """Every camera in one venue session, in the manifest's own order.

    Order is preserved rather than sorted, because it is how an operator wrote
    their venue down and it belongs in logs. It must never reach an output
    record: everything serialised is in canonical stream order, so reversing the
    cameras here changes nothing downstream.
    """

    session_id: str
    cameras: tuple[VenueCamera, ...]
    #: Where the manifest was loaded from, recorded in the session record so a
    #: run can be traced back to the exact input list that produced it.
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise VenueSessionError("A venue session manifest needs a non-empty session_id.")
        if not self.cameras:
            raise VenueSessionError("A venue session manifest must contain at least one camera.")

        counts = Counter(camera.camera_id for camera in self.cameras)
        duplicated = sorted(name for name, count in counts.items() if count > 1)
        if duplicated:
            raise VenueSessionError(
                f"Duplicate camera_id in venue manifest: {duplicated}. A camera id is the "
                "namespace every one of that camera's tracks lives in; two entries sharing "
                "one would make their tracklets indistinguishable. List a camera's several "
                "recordings as several 'sources' of one entry instead — that is what stream "
                "epochs are for."
            )

    @property
    def camera_ids(self) -> tuple[str, ...]:
        """Camera ids in manifest order."""
        return tuple(camera.camera_id for camera in self.cameras)

    @property
    def stream_count(self) -> int:
        """Total recordings across every camera — the number of streams."""
        return sum(len(camera.sources) for camera in self.cameras)

    def camera(self, camera_id: str) -> VenueCamera:
        for camera in self.cameras:
            if camera.camera_id == camera_id:
                return camera
        known = ", ".join(self.camera_ids)
        raise VenueSessionError(
            f"No camera '{camera_id}' in this session manifest. Known cameras: {known}."
        )

    def __len__(self) -> int:
        return len(self.cameras)

    def __iter__(self) -> Iterator[VenueCamera]:
        return iter(self.cameras)


def load_venue_manifest(path: Path | str, base_dir: Path | None = None) -> VenueSessionManifest:
    """Load and validate a venue session manifest, resolving relative paths.

    Relative ``input_path`` values resolve against ``base_dir`` (the current
    working directory by default), so a manifest committed to the repository
    works from a checkout without absolute paths in version control.

    Every input file must exist at load time. A session that fails on the fourth
    camera after opening three is a worse failure than one that never starts.

    Args:
        path: The manifest JSON.
        base_dir: Root that relative source paths resolve against.

    Returns:
        The validated manifest, in its own camera order.

    Raises:
        VenueSessionError: for a missing or malformed manifest, an unknown
            manifest version, a duplicate camera id, an offset declared without
            an accuracy, or a source file that does not exist.
    """
    manifest_path = Path(path)
    root = Path(base_dir) if base_dir is not None else Path.cwd()

    if not manifest_path.exists():
        raise VenueSessionError(f"Venue manifest not found: {manifest_path}")
    if not manifest_path.is_file():
        raise VenueSessionError(f"Venue manifest path is not a file: {manifest_path}")

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VenueSessionError(f"Venue manifest {manifest_path} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise VenueSessionError(f"Could not read venue manifest {manifest_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise VenueSessionError(
            f"Venue manifest {manifest_path} must be a JSON object with a 'cameras' list."
        )

    version = raw.get("manifest_version")
    if version != VENUE_MANIFEST_VERSION:
        raise VenueSessionError(
            f"Venue manifest {manifest_path} declares manifest_version {version!r}; this build "
            f"understands {VENUE_MANIFEST_VERSION!r}. Refusing to guess at the shape."
        )

    session_id = _require_str(raw, "session_id", str(manifest_path))

    entries = raw.get("cameras")
    if not isinstance(entries, list) or not entries:
        raise VenueSessionError(
            f"Venue manifest {manifest_path} must contain a non-empty 'cameras' list."
        )

    cameras: list[VenueCamera] = []
    for index, entry in enumerate(entries):
        where = f"{manifest_path} camera #{index + 1}"
        if not isinstance(entry, dict):
            raise VenueSessionError(
                f"{where}: each camera must be a JSON object, got {type(entry)}"
            )

        camera_id = _require_str(entry, "camera_id", where)
        raw_sources = entry.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise VenueSessionError(
                f"{where} ('{camera_id}'): needs a non-empty 'sources' list. One recording is "
                "one entry; a camera that stopped and restarted has several."
            )

        sources = tuple(
            _load_source(source, f"{where} ('{camera_id}') source #{epoch + 1}", root)
            for epoch, source in enumerate(raw_sources)
        )
        cameras.append(VenueCamera(camera_id=camera_id, sources=sources))

    return VenueSessionManifest(
        session_id=session_id, cameras=tuple(cameras), source_path=manifest_path
    )


def _load_source(entry: object, where: str, root: Path) -> VenueStreamEntry:
    if not isinstance(entry, dict):
        raise VenueSessionError(f"{where}: each source must be a JSON object, got {type(entry)}")

    input_path = _resolve(_require_str(entry, "input_path", where), root)
    if not input_path.exists():
        raise VenueSessionError(
            f"{where}: input file not found: {input_path}. A venue session runs on footage that "
            "is already present; nothing is downloaded."
        )
    if not input_path.is_file():
        raise VenueSessionError(f"{where}: input path is not a file: {input_path}")

    return VenueStreamEntry(
        input_path=input_path,
        start_offset_seconds=_optional_float(entry, "start_offset_seconds", where),
        offset_accuracy_seconds=_optional_float(entry, "offset_accuracy_seconds", where),
        declared_fps=_optional_float(entry, "declared_fps", where),
    )


def _require_str(mapping: dict[str, Any], key: str, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VenueSessionError(f"{where}: '{key}' must be a non-empty string, got {value!r}")
    return value.strip()


def _optional_float(mapping: dict[str, Any], key: str, where: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    # ``bool`` is an ``int``, and ``true`` in a manifest is a mistake, not a 1.
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise VenueSessionError(f"{where}: '{key}' must be a number when present, got {value!r}")
    return float(value)


def _resolve(value: str, root: Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (root / candidate)
