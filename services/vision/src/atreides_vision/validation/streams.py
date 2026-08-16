"""Reading the V0.3 output streams back, one record at a time.

Validation clips produce output streams far larger than the clips themselves —
a 50 s dense scene writes hundreds of thousands of observation records. Nothing
here ever holds a whole stream: :func:`iter_jsonl` yields one decoded record at
a time and the accumulators that consume it keep bounded state.

Gzip is transparent. The pipeline selects compression from the ``.gz`` suffix,
so validation reads the suffix back the same way rather than recording a flag
somewhere that could disagree with the file on disk.
"""

from __future__ import annotations

import gzip
import json
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

from atreides_vision.errors import ValidationError
from atreides_vision.jsonl import GZIP_SUFFIX

__all__ = ["iter_jsonl", "stream_size_bytes", "uncompressed_size_bytes"]


def _open_text(path: Path) -> IO[str]:
    if path.suffix.lower() == GZIP_SUFFIX:
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield each record of a JSON Lines stream, decompressing ``.gz`` in step.

    Blank lines are skipped: a run interrupted between writes can leave a
    trailing newline, and refusing to read the 400,000 good records before it
    would serve nobody.

    Args:
        path: A ``.jsonl`` or ``.jsonl.gz`` file.

    Yields:
        One decoded record per line, in file order.

    Raises:
        ValidationError: if the file is missing, unreadable, or contains a line
            that is not a JSON object. A malformed line is never skipped —
            silently dropping records would understate every count derived from
            the stream.
    """
    stream_path = Path(path)
    if not stream_path.exists():
        raise ValidationError(f"Output stream not found: {stream_path}")

    try:
        handle = _open_text(stream_path)
    except OSError as exc:
        raise ValidationError(f"Could not open output stream {stream_path}: {exc}") from exc

    with handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"{stream_path} line {line_number} is not valid JSON: {exc}"
                ) from exc
            except (OSError, EOFError) as exc:
                raise ValidationError(f"Could not read {stream_path}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValidationError(
                    f"{stream_path} line {line_number} is a {type(record).__name__}, not a "
                    "JSON object. Every record in an Atreides stream is an object."
                )
            yield record


def stream_size_bytes(path: Path | str) -> int:
    """Size of a stream on disk, as stored.

    For a compressed stream this is the compressed size, which is what the file
    actually costs. Read from the filesystem, never by decoding the file.
    """
    stream_path = Path(path)
    if not stream_path.exists():
        raise ValidationError(f"Output stream not found: {stream_path}")
    return stream_path.stat().st_size


def uncompressed_size_bytes(path: Path | str) -> int | None:
    """Uncompressed size of a gzip stream from its trailer, without decoding it.

    A gzip member records its uncompressed length in the final four bytes, so
    this is two seeks rather than a decompression pass — which matters when the
    file is a few hundred megabytes of observations.

    Returns:
        The uncompressed byte count, or ``None`` when it cannot be established
        cheaply: an uncompressed file (whose stored size is already the answer),
        a multi-member archive, or a member over 4 GiB, where the trailer holds
        the length modulo 2³² and would report a confidently wrong number.
    """
    stream_path = Path(path)
    if stream_path.suffix.lower() != GZIP_SUFFIX:
        return None
    stored = stream_path.stat().st_size
    if stored < 18:  # Smaller than an empty gzip member: nothing to read.
        return None
    try:
        with stream_path.open("rb") as handle:
            handle.seek(-4, 2)
            (isize,) = struct.unpack("<I", handle.read(4))
    except OSError as exc:
        raise ValidationError(f"Could not read the gzip trailer of {stream_path}: {exc}") from exc

    size = int(isize)
    # The trailer is modulo 2**32. A member whose compressed size already
    # exceeds the reported uncompressed size is either wrapped or incompressible
    # noise; either way the number would mislead, so decline to report it.
    if size < stored:
        return None
    return size
