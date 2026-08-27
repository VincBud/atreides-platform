"""Dataset-specific ground-truth adapters for offline venue validation.

Everything under this package is validation infrastructure: it teaches the
validation harness how to read one particular public dataset's own file
formats and produce typed, dataset-scoped ground truth. It is never imported
by camera-local detection or tracking, by any signal layer (``behavior``,
``interaction``, ``crowd``, ``zones``), or by ``association``, ``fusion``,
``identity`` or ``occupancy`` — those modules run against real or synthetic
video and must stay blind to which, if any, public dataset a validation run
happens to be replaying.
"""

from __future__ import annotations
