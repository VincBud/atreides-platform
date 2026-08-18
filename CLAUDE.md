# Atreides — Project Context

Architectural source of truth. Where this file and an older document disagree, this file
wins; where it and the code disagree, that is a finding to report, not something to
silently "fix" in either direction.

Milestone workflow, gate procedure and report format live in the manual skill
`/atreides-milestone` (`.claude/skills/atreides-milestone/SKILL.md`) — invoke it instead of
restating that workflow in a prompt.

## Mission

AI-assisted venue security intelligence. Atreides consumes video from a venue's cameras and
gives human security operators situational awareness they cannot get from a wall of
monitors. It measures, associates, aggregates evidence and supports a decision. **It does
not make the decision, and it does not identify people.**

## Engineering versions vs product generations

Two numbering schemes; never conflate them.

- **Engineering milestone (`V0.x`)** — a bounded increment of this repository, each with a
  doc in `docs/vision/` and its own validation. `V0.1` SEE + TRACK · `V0.2` zones ·
  `V0.3.0–V0.3.6` behaviour, calibration, interaction, crowd, real-world validation.
- **Product generation (`V1`, `V2`, `V3`)** — a generation of the product, defining scope
  and deployment class.

The repository is mid-`V0.3`. That is **the state of the implementation, not the boundary
of V1 scope** — V1 capabilities are in-scope requirements whether or not code exists yet.

## One product family, one core

V1, V2 and V3 are **not** branches, forks or parallel pipelines. No `v2/` package, no
`if generation == ...`, no duplicated pipeline. Generations differ by **modular
capabilities** (a stronger detector, an association strategy, an extra modality) and
**deployment profiles**, composed at the edges — never by rewriting the core. A capability
a deployment lacks degrades cleanly: the system runs with one camera or with forty, with
calibration or without, and says which.

## V1 — CCTV multi-camera decision support

V1 runs on a venue's **existing CCTV infrastructure** and is explicitly multi-camera. In
scope: multi-camera ingestion · camera-local detection and tracking · calibration / spatial
context where required · behaviour, interaction and crowd signals · **Multi-Camera
Association / Fusion** · **temporary venue-local identity** · temporal evidence aggregation
· Risk / Decision Engine foundation · Decision-Support events · the validation and
observability a real pilot deployment requires.

**Multi-Camera Association / Fusion is a V1 product capability** — not future work, not
V2-only, not architecturally hypothetical. It is unbuilt today and gets built when a
bounded, approved implementation task calls for it: the same rule as any other scoped work
here, not a special prohibition.

Single-camera is a degenerate case of the multi-camera path, never a separate mode.

## Identity architecture

```text
detection → camera-local tracking → calibration / spatial context
  → Multi-Camera Association / Fusion → venue-local temporary identity
  → venue-level signals / evidence → risk / decision support

camera_01 / track_17 ─┐
                      ├── venue_person_0042
camera_02 / track_08 ─┘
```

**Camera-local track identity** is what a tracker produces: a tracklet on one camera in one
session. It is not a person and is not unique across cameras or clips.

**Venue-local temporary identity** is what association produces: a hypothesis that several
camera-local tracklets are the same real person right now, in this venue, in this session —
including so one person is not counted twice at venue level.

Venue-local identity is, without exception: **temporary** (dies with the session) ·
**venue/session scoped** · **confidence-aware** (every association carries its evidence and
its uncertainty) · **a distinct type and namespace from tracker IDs** (never a reused
integer, never "lowest track ID wins") · **non-persistent** (never written to a durable
identity store, never carried between sessions or venues) · **non-biometric** (no face, no
gait signature, no template, ever).

Rules that follow:

- **Cross-camera identity logic never lives inside a camera-local tracker.** ByteTrack,
  OC-SORT or any successor stays camera-local; association is its own module above them.
- **Prefer no merge over a wrong merge.** On insufficient evidence the correct outputs are
  "not associated" or "ambiguous", never a forced merge to tidy a count. A wrong merge
  corrupts every venue-level number above it and is harder to detect than a missed one.
- **Fusion must never destroy provenance.** A venue-local subject always resolves back to
  its contributing `(camera, track)` observations with the evidence and confidence that
  produced the association.
- **No persistent real-world identity, ever** — and no facial recognition or biometric
  matching as an engineering shortcut to easier association.

## Scalability constraints

Nothing anywhere may hardcode: a number of cameras, or assume exactly one · a camera layout,
quadrant scheme or naming convention · coverage area, venue dimensions or floor extent · an
overlap pattern between cameras, or that overlap exists at all · mounting height, tilt,
focal length, resolution or frame rate.

Geometry enters through calibration and the projector seam. Topology, when known, is
configuration — never an assumption baked into a signal, a metric or a threshold.
Non-overlapping cameras are a supported case, not a bug report.

## Signal → evidence → decision

**Signals** are neutral measurements — speed, separation, cluster spread, dwell; no labels,
scores or judgement. **Evidence** is signals aggregated over time and, after association,
across cameras, with provenance and confidence attached. **Decision support** turns evidence
into something an operator sees, with the reasoning visible.

A measurement must never be renamed into a conclusion as it moves up: `motion_level` is not
a risk score, a cluster is not a fight, a fused identity is not a person's name.

## Human decision-support boundaries

- The system produces a view; a human decides and acts.
- No autonomous enforcement, no automated alerting presented as truth, no ranking of people,
  no behaviour classification, no intent/emotion/aggression/intoxication inference, no
  demographic classification.
- Every operator-facing conclusion is traceable to the evidence and confidence beneath it,
  including which cameras contributed.
- Uncertainty is shown, not hidden. "We don't know" is a valid, first-class output.

## Privacy

- No identification of people. No biometrics, face recognition or demographics.
- Venue-local identity is temporary, session-scoped and discarded — no identity database,
  no cross-session or cross-venue linkage.
- Raw video stays local. The only network call in the project is the first-run model weight
  download.
- Persistence is the exception, not the default: what is stored must be justified, minimal
  and non-identifying.

## Calibration and integrity failure semantics

- One homography describes **one plane** — stairs, balconies and multiple floors are not
  covered by one.
- Uncalibrated footage produces **no physical units**: no metre, m/s or subjects/m² is ever
  derived from, or estimated for, an uncalibrated scene.
- When integrity monitoring detects that a camera moved, the system **stops trusting its
  metres** rather than continuing with quietly wrong ones. Degraded state is reported, not
  swallowed.
- Loss of calibration or integrity degrades association and venue-level output explicitly —
  it lowers confidence, it does not silently keep merging.

## Validation philosophy

- **Synthetic correctness ≠ real-world robustness.** Closed-form tests prove a formula; they
  say nothing about occlusion, crowding, distance, low light, glare or weather.
- **Proxy/system metrics and human ground truth never mix.** Figures derived from Atreides'
  own output are labelled proxies. Ground truth appears only where a human actually
  annotated; un-annotated material reports `NOT_ANNOTATED` and emits no recall, precision or
  ID-switch figure at all — never a zero standing in for a measurement.
- **No verdict beyond the evidence.** "Not established" is acceptable and often correct.
  Ratios and measurements, not grades.
- **Frozen scope during validation.** Detector, tracker, deduplication rule, signal
  definitions and schemas under active validation are not retuned in response to a
  validation figure; changing them requires a new milestone and re-validation.
- Association is validated on its own terms, with **false merges the most serious class**
  (checklist in `atreides-validation-engineer`).

## Quality gates

`uv run pytest` (no weights, network or GPU) · `uv run ruff check .` ·
`uv run ruff format .` · `uv run mypy` (strict, src and tests). All four must pass, plus
determinism: same input, same output, byte-for-byte.

**No gate is ever weakened to make a change pass** — no loosened assertion, deleted or
skipped test, blanket `# type: ignore`, widened mypy override, new ruff exemption, or
lowered threshold. If a gate looks wrong, that is a human decision, raised explicitly and
separately.

## Repository layout

- `services/vision/src/atreides_vision/` — implementation; `services/vision/tests/` — tests
- `docs/vision/` — one document per engineering milestone, with its limitations
- `config/` — zone and validation manifests · `artifacts/`, `data/input/` — git-ignored
- `services/api/`, `services/intelligence/`, `apps/operator-dashboard/`,
  `packages/contracts/` — placeholders

Module conventions: pure maths modules with no state, `models.py` for frozen domain types,
`config.py` for tunables carrying a unit and a rationale, `history.py` for bounded state,
`analyzer.py` as the facade the pipeline orchestrates, `reporting.py` for versioned JSONL.
Model libraries (`rfdetr`, `trackers`, `supervision`, `cv2`) stay confined to their adapter
file; no signal layer imports one, or another signal layer.

## Roadmap discipline

**V2 — extreme vision and advanced association** strengthens V1; it does not introduce
association from scratch. Areas: extreme low light, strobe, motion blur, severe occlusion,
track continuity, stronger ReID, long-gap association, harder non-overlapping transitions,
advanced fusion, ambiguity handling, first-class confidence and observability.

**V3 — Atreides certified deployment** may introduce Atreides-specified RGB/NIR hardware,
optimised sensors and optics, known topology, designed overlap, standardised installation
and calibration procedures, deployment verification, extra sensor modalities. None of it is
built during V1.

Discipline both ways: do not defer a V1 capability by calling it V2 — association and
venue-local identity are V1; and do not pull V2/V3 sophistication into V1 — V1 needs the
**simplest sufficiently robust** association that works on real CCTV, not embeddings,
training pipelines or GPU dependence adopted for their own sake. Choose V1 designs that
leave room for stronger V2 methods behind the same seam. Scope changes come from a human.

## Routing policy

**Default: do the work in the current session.** One Claude context is the cheapest correct
answer for most tasks. Do not create Agent Teams. Do not spawn an agent merely because it
exists.

Use a specialized Atreides agent only when its expertise materially improves the task:

| Agent | Model | Use only for |
| --- | --- | --- |
| `atreides-architecture-guardian` | opus | Consequential architectural changes — boundaries, public contracts, association/identity shape, suspected V2/V3 leakage |
| `atreides-implementation-engineer` | opus | Primary implementation role for substantial coding inside an approved, bounded scope |
| `atreides-validation-engineer` | sonnet | Independent validation when warranted — not every edit |
| `atreides-adversarial-reviewer` | opus | High-risk architectural, security or release boundaries — not every milestone |
| `atreides-vision-researcher` | opus | Genuine unresolved CV / perception research questions |

These model assignments are deliberate. They are not downgraded, and these roles are not
weakened, to save tokens.

**Agent Teams are an exception, not a workflow.** Use one only when multiple agents genuinely
need peer-to-peer communication or parallel competing investigation. Never create a Team
solely to execute sequential architecture → implementation → validation work — that is a
hand-off between single sessions. Teams are not globally enabled in `.claude/settings.json`
and must not be enabled through another mechanism.

No agent commits, pushes, or runs destructive git commands. Product code changes come from
an approved implementation task — never as a side effect of review, validation or research.
