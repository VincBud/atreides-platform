"""The validation input manifest: which clips, from which camera, of what kind.

A manifest rather than a hard-coded clip list so footage can be added, removed
or re-scened without touching a line of validation logic. It is also where a
camera identity is established for validation footage, which is not cosmetic:
every subject reference in every output stream is *(camera_id, track_id)*.

Two rules the loader enforces rather than trusts:

**Clip ids are unique.** They name output directories; a duplicate would have
one run overwrite another's artifacts and the summary would report the survivor
twice under different scene types.

**Input files must exist at load time.** A three-clip run that fails on the
third after twenty minutes of processing is a worse failure than one that never
starts.

Camera ids are deliberately *not* unique
----------------------------------------
**Several clips may share one ``camera_id``, and that is the expected case.** A
real fixed camera produces many independent validation clips — different times
of day, different lighting, different crowd densities, different days, different
test scenarios — and forcing a fabricated distinct id per clip would misdescribe
the deployment the harness exists to characterise.

Sharing a camera id asserts exactly one thing: *these clips came from the same
physical camera*. It asserts **nothing** about identity across them, and the
harness never treats it as if it did:

* each clip is processed by its own detector, tracker and analyzers, built fresh
  per clip;
* each clip writes its own output streams into its own run directory;
* each clip is measured by its own accumulators, into its own result;
* nothing anywhere merges, joins or compares track ids across clips.

So ``camera_01/clip_A/track_7`` and ``camera_01/clip_B/track_7`` are **two
independent tracklets that happen to share an integer**, exactly as
``cam_a/track_7`` and ``cam_b/track_7`` are. Track identity does not survive a
clip boundary, and no persistent identity is introduced by this milestone.

The one consequence a reader must hold: per-clip counts from clips sharing a
camera id **must not be added together**. Two clips of one camera reporting 40
unique track ids each are not 80 people, or even 80 tracklets of one
continuously-observed scene. The report says so wherever it applies.

Calibration is optional and, today, unused. The field exists so a future
physically measured scene can be validated by adding a manifest entry rather
than by changing this harness — see
``docs/vision/V0.3.6_REAL_WORLD_VALIDATION.md``. **No current validation clip
supplies one**, so every current run measures in normalized image units and no
physical quantity is reported for any of them.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atreides_vision.errors import ValidationError

__all__ = [
    "MANIFEST_VERSION",
    "ValidationClip",
    "ValidationManifest",
    "load_manifest",
]

#: Version of the manifest contract this loader understands. A manifest naming a
#: different one is rejected rather than parsed on a guess.
MANIFEST_VERSION = "0.3.6"

#: Characters a clip id may contain. Clip ids become directory names, so a value
#: containing a separator would write artifacts outside the run directory.
_ID_ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")


def _require_str(mapping: dict[str, Any], key: str, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{where}: '{key}' must be a non-empty string, got {value!r}")
    return value.strip()


def _optional_str(mapping: dict[str, Any], key: str, where: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            f"{where}: '{key}' must be a non-empty string when present, got {value!r}"
        )
    return value.strip()


@dataclass(frozen=True, slots=True)
class ValidationClip:
    """One clip to validate, and the identity its records are namespaced by.

    ``scene_type`` is a free-form label used only to group the comparison table.
    It carries no thresholds and selects no behaviour: a clip labelled ``dense``
    is processed exactly as one labelled ``control``.
    """

    clip_id: str
    input_path: Path
    scene_type: str
    camera_id: str
    notes: str | None = None
    #: Ground-plane calibration for this clip, if one was physically measured.
    #: ``None`` — the case for every current clip — means the run measures in
    #: normalized image units and reports no physical quantity.
    calibration_path: Path | None = None

    def __post_init__(self) -> None:
        for name, value in (("clip_id", self.clip_id), ("camera_id", self.camera_id)):
            if not value or not value.strip():
                raise ValidationError(f"A validation clip needs a non-empty {name}.")
        invalid = sorted(set(self.clip_id) - _ID_ALLOWED)
        if invalid:
            raise ValidationError(
                f"clip_id '{self.clip_id}' contains characters that cannot be used in a "
                f"directory name: {invalid}. Use letters, digits, '_', '-' or '.'."
            )
        if not self.scene_type.strip():
            raise ValidationError(f"Clip '{self.clip_id}' needs a non-empty scene_type.")

    @property
    def is_calibrated(self) -> bool:
        """Whether this clip has a measured calibration, and so metric output."""
        return self.calibration_path is not None

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "clip_id": self.clip_id,
            "input_path": str(self.input_path),
            "scene_type": self.scene_type,
            "camera_id": self.camera_id,
        }
        if self.notes is not None:
            record["notes"] = self.notes
        record["calibration_path"] = (
            None if self.calibration_path is None else str(self.calibration_path)
        )
        return record


@dataclass(frozen=True, slots=True)
class ValidationManifest:
    """Every clip in one validation run, in the manifest's own order.

    Order is preserved rather than sorted: the manifest author controls the
    order of the comparison table, and a run's report should read the way the
    manifest was written.
    """

    clips: tuple[ValidationClip, ...]
    #: Where the manifest was loaded from, recorded in the summary so a report
    #: can be traced back to the exact input list that produced it.
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.clips:
            raise ValidationError("A validation manifest must contain at least one clip.")
        # Clip ids only. Camera ids are deliberately allowed to repeat — see the
        # module docstring — because one physical camera legitimately produces
        # many independent validation clips.
        _reject_duplicate_clip_ids([clip.clip_id for clip in self.clips])

    @property
    def camera_ids(self) -> tuple[str, ...]:
        """Distinct camera ids, in first-appearance order."""
        seen: dict[str, None] = {}
        for clip in self.clips:
            seen.setdefault(clip.camera_id, None)
        return tuple(seen)

    @property
    def shared_camera_ids(self) -> tuple[str, ...]:
        """Camera ids used by more than one clip, sorted.

        Not an error — it is the expected case for repeated captures from one
        fixed camera. Exposed so the report can state, where it applies, that
        per-clip counts from these clips must not be added together.
        """
        counts = Counter(clip.camera_id for clip in self.clips)
        return tuple(sorted(camera for camera, count in counts.items() if count > 1))

    def clips_for_camera(self, camera_id: str) -> tuple[ValidationClip, ...]:
        """Every clip captured by one camera, in manifest order.

        A grouping for reporting only. It does **not** imply that the clips'
        tracks relate to each other in any way.
        """
        return tuple(clip for clip in self.clips if clip.camera_id == camera_id)

    def clip(self, clip_id: str) -> ValidationClip:
        for clip in self.clips:
            if clip.clip_id == clip_id:
                return clip
        known = ", ".join(clip.clip_id for clip in self.clips)
        raise ValidationError(f"No clip '{clip_id}' in this manifest. Known clips: {known}.")

    def selecting(self, clip_ids: Sequence[str]) -> ValidationManifest:
        """A manifest restricted to ``clip_ids``, in manifest order.

        Used by ``--clip`` to re-run one scene without disturbing the others'
        artifacts. An unknown id raises rather than silently selecting nothing.
        """
        if not clip_ids:
            return self
        wanted = set(clip_ids)
        for clip_id in wanted:
            self.clip(clip_id)
        return ValidationManifest(
            clips=tuple(clip for clip in self.clips if clip.clip_id in wanted),
            source_path=self.source_path,
        )

    def __len__(self) -> int:
        return len(self.clips)

    def __iter__(self) -> Iterator[ValidationClip]:
        return iter(self.clips)


def _reject_duplicate_clip_ids(values: Sequence[str]) -> None:
    """Clip ids must be unique. Camera ids deliberately need not be.

    A duplicate clip id is a genuine collision: clip ids name run directories, so
    two clips sharing one would have the second overwrite the first's streams and
    the summary would report the survivor twice under two scene types.
    """
    counts = Counter(values)
    duplicated = sorted(value for value, count in counts.items() if count > 1)
    if not duplicated:
        return
    raise ValidationError(
        f"Duplicate clip_id in manifest: {duplicated}. Clip ids name output directories, so a "
        "duplicate would have one run overwrite another's artifacts. Note that camera_id may "
        "repeat freely — one camera can supply many clips — but each clip needs its own id."
    )


def load_manifest(path: Path | str, base_dir: Path | None = None) -> ValidationManifest:
    """Load and validate a manifest, resolving relative paths.

    Relative ``input_path`` values resolve against ``base_dir`` (the current
    working directory by default), so a manifest committed to the repository
    works from a repository checkout without absolute paths in version control.

    Args:
        path: The manifest JSON.
        base_dir: Root that relative clip paths resolve against.

    Returns:
        The validated manifest.

    Raises:
        ValidationError: for a missing or malformed manifest, an unknown
            manifest version, a duplicate clip or camera id, or a clip whose
            input file does not exist.
    """
    manifest_path = Path(path)
    root = Path(base_dir) if base_dir is not None else Path.cwd()

    if not manifest_path.exists():
        raise ValidationError(f"Validation manifest not found: {manifest_path}")
    if not manifest_path.is_file():
        raise ValidationError(f"Validation manifest path is not a file: {manifest_path}")

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"Validation manifest {manifest_path} is not valid JSON: {exc}"
        ) from exc
    except OSError as exc:
        raise ValidationError(f"Could not read validation manifest {manifest_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValidationError(
            f"Validation manifest {manifest_path} must be a JSON object with a 'clips' list."
        )

    version = raw.get("manifest_version")
    if version != MANIFEST_VERSION:
        raise ValidationError(
            f"Validation manifest {manifest_path} declares manifest_version {version!r}; "
            f"this build understands {MANIFEST_VERSION!r}. Refusing to guess at the shape."
        )

    entries = raw.get("clips")
    if not isinstance(entries, list) or not entries:
        raise ValidationError(
            f"Validation manifest {manifest_path} must contain a non-empty 'clips' list."
        )

    clips: list[ValidationClip] = []
    for index, entry in enumerate(entries):
        where = f"{manifest_path} clip #{index + 1}"
        if not isinstance(entry, dict):
            raise ValidationError(f"{where}: each clip must be a JSON object, got {type(entry)}.")

        clip_id = _require_str(entry, "clip_id", where)
        input_value = _require_str(entry, "input_path", where)
        input_path = _resolve(input_value, root)
        if not input_path.exists():
            raise ValidationError(
                f"{where} ('{clip_id}'): input file not found: {input_path}. Validation runs on "
                "real footage that must already be present; nothing is downloaded."
            )
        if not input_path.is_file():
            raise ValidationError(f"{where} ('{clip_id}'): input path is not a file: {input_path}")

        calibration_value = _optional_str(entry, "calibration_path", where)
        calibration_path = None if calibration_value is None else _resolve(calibration_value, root)
        if calibration_path is not None and not calibration_path.exists():
            raise ValidationError(
                f"{where} ('{clip_id}'): calibration file not found: {calibration_path}. Omit "
                "'calibration_path' to validate this clip in normalized image units."
            )

        clips.append(
            ValidationClip(
                clip_id=clip_id,
                input_path=input_path,
                scene_type=_require_str(entry, "scene_type", where),
                camera_id=_require_str(entry, "camera_id", where),
                notes=_optional_str(entry, "notes", where),
                calibration_path=calibration_path,
            )
        )

    return ValidationManifest(clips=tuple(clips), source_path=manifest_path)


def _resolve(value: str, root: Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (root / candidate)
