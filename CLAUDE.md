# Atreides — Project Context

This file is the architectural source of truth for work in this repository. Agents,
sessions and reviews defer to it. Where this file and an older document disagree, this
file wins; where it and the code disagree, the disagreement is a finding to report, not
something to silently "fix" in either direction.

## Mission

Atreides is an AI-assisted venue security intelligence platform. It consumes video from a
venue's cameras and gives human security operators situational awareness they cannot get
by watching a wall of monitors. It measures, associates, aggregates evidence and supports
a decision. **It does not make the decision, and it does not identify people.**

## Engineering versions vs product generations

Two numbering schemes exist and must not be conflated.

| | Meaning | Examples |
| --- | --- | --- |
| **Engineering milestone** (`V0.x`) | A bounded increment of this repository's implementation, each with a doc in `docs/vision/` and its own validation | `V0.1` SEE + TRACK · `V0.2` zones · `V0.3.0–V0.3.6` behaviour, calibration, interaction, crowd, real-world validation |
| **Product generation** (`V1`, `V2`, `V3`) | A generation of the Atreides product, defining scope and deployment class | `V1` CCTV multi-camera decision support · `V2` extreme vision + advanced association · `V3` Atreides certified deployment |

The repository is currently mid-`V0.3` engineering work. That is **the state of the
implementation, not the boundary of V1 scope.** Capabilities listed under V1 below are
in-scope product requirements whether or not code exists for them yet.

## One product family, one core

Atreides is one software family. V1, V2 and V3 are **not** branches, forks, or parallel
pipelines.

- No `v2/` package, no `if generation == ...`, no duplicated pipeline.
- Generations differ by **modular capabilities** (a stronger detector, an association
  strategy, an extra modality) and **deployment profiles** (what hardware and topology a
  site actually has), composed at the edges — not by rewriting the core.
- A capability that a deployment lacks must degrade cleanly: the system runs with one
  camera or with forty, with calibration or without, and says which.

## V1 — CCTV multi-camera decision support

V1 runs on a venue's **existing CCTV infrastructure** and is explicitly multi-camera.

In scope for V1:

- multi-camera video ingestion
- camera-local detection and tracking
- calibration / spatial context where required
- behaviour signals
- interaction signals
- crowd signals
- **Multi-Camera Association / Fusion**
- **temporary venue-local identity**
- temporal evidence aggregation
- Risk / Decision Engine foundation
- Decision-Support events
- the validation and observability a real pilot deployment requires

**Multi-Camera Association / Fusion is a V1 product capability.** It is not future work,
not V2-only, and not something to keep architecturally hypothetical. It is unbuilt today
and gets built when a bounded, approved implementation task calls for it — the same rule
as any other scoped work here, not a special prohibition.

The architecture must support single-camera and multi-camera deployments naturally. One
camera is a degenerate case of the multi-camera path, not a separate mode.

## Identity architecture

The layer boundary that everything else depends on:

```text
Detection
  ↓
Camera-local tracking
  ↓
Calibration / spatial context
  ↓
Multi-Camera Association / Fusion
  ↓
Venue-local temporary identity
  ↓
Venue-level signals / evidence
  ↓
Risk / Decision Support
```

```text
camera_01 / track_17 ─┐
                      ├── venue_person_0042
camera_02 / track_08 ─┘
```

**Camera-local track identity** is what a tracker produces: a tracklet on one camera in
one session. It is not a person and is not unique across cameras or clips.

**Venue-local temporary identity** is what association produces: a hypothesis that
several camera-local tracklets are the same real person right now, in this venue, in this
session. Its purpose includes preventing the same person from being counted twice at
venue level when several cameras see them.

Venue-local identity is, without exception:

- **temporary** — it exists for the session and dies with it
- **venue/session scoped** — meaningless outside this venue and session
- **confidence-aware** — every association carries its evidence and its uncertainty
- **distinct from tracker IDs** — a different type, a different namespace, never a
  reused integer, never "the lowest track ID wins"
- **non-persistent** — never written to a durable identity store, never carried between
  sessions or venues
- **non-biometric** — no face, no gait signature, no biometric template, ever

Rules that follow:

- **Cross-camera identity logic never lives inside a camera-local tracker.** ByteTrack,
  OC-SORT or any successor stays camera-local. Association is its own module, above them.
- **Prefer no merge over a wrong merge.** When evidence is insufficient, the correct
  outputs are "not associated" or "ambiguous" — never a forced merge to make a count look
  tidy. A wrong merge corrupts every venue-level number above it and is harder to detect
  than a missed one.
- **Fusion must never destroy provenance.** A venue-local subject always resolves back to
  its contributing `(camera, track)` observations, with the evidence and confidence that
  produced the association. Nothing downstream may consume a fused identity without being
  able to see what it was built from.
- **No persistent real-world identity, ever**, and no facial recognition or biometric
  matching as an engineering shortcut to make association easier.

## Scalability constraints

Nothing anywhere may hardcode:

- a number of cameras, or assume exactly one
- a camera layout, quadrant scheme or naming convention
- physical coverage area, venue dimensions or floor extent
- an overlap pattern between cameras, or that overlap exists at all
- mounting height, tilt, focal length, resolution or frame rate

Geometry enters through calibration and the projector seam. Topology, when it is known,
is configuration — never an assumption baked into a signal, a metric or a threshold.
Non-overlapping cameras must be a supported case, not a bug report.

## Signal → evidence → decision

Three distinct stages, kept distinct:

1. **Signals** are neutral measurements — speed, separation, cluster spread, dwell.
   No labels, no scores, no judgement.
2. **Evidence** is signals aggregated over time and, after association, across cameras —
   with its provenance and confidence attached.
3. **Decision support** turns evidence into something an operator sees, with the
   reasoning visible.

A measurement must never be renamed into a conclusion as it moves up. `motion_level` is
not a risk score; a cluster is not a fight; a fused identity is not a person's name.

## Human decision-support boundaries

- The system produces a view; a human decides and acts.
- No autonomous enforcement, no automated alerting presented as truth, no ranking of
  people, no behaviour classification, no intent/emotion/aggression/intoxication
  inference, no demographic classification.
- Every operator-facing conclusion is traceable to the evidence and confidence beneath
  it, including which cameras contributed.
- Uncertainty is shown, not hidden. "We don't know" is a valid, first-class output.

## Privacy

- No identification of people. No biometrics. No face recognition. No demographics.
- Venue-local identity is temporary, session-scoped and discarded — there is no identity
  database and no cross-session or cross-venue linkage.
- Raw video stays local. The only network call in the project is the first-run model
  weight download.
- Persistence is the exception, not the default: what is stored must be justified,
  minimal and non-identifying.

## Calibration and integrity failure semantics

- Calibration converts camera-local image positions into venue coordinates. One
  homography describes **one plane** — stairs, balconies and multiple floors are not
  covered by one.
- Uncalibrated footage produces **no physical units**. No metre, m/s or subjects/m² is
  ever derived from, or estimated for, an uncalibrated scene.
- When integrity monitoring detects that a camera moved, the system **stops trusting its
  metres** rather than continuing with quietly wrong ones. Degraded state is reported,
  not swallowed.
- Loss of calibration or integrity must degrade association and venue-level output
  explicitly — it lowers confidence, it does not silently keep merging.

## Validation philosophy

- **Synthetic correctness ≠ real-world robustness.** Closed-form tests prove a formula.
  They say nothing about occlusion, crowding, distance, low light, glare or weather.
- **Proxy/system metrics and human ground truth never mix.** Automated figures derived
  from Atreides' own output are labelled as proxies. Ground truth appears only where a
  human actually annotated; un-annotated material reports `NOT_ANNOTATED` and emits no
  recall, precision or ID-switch figure at all — never a zero standing in for a
  measurement.
- **No verdict beyond the evidence.** "Not established" is an acceptable, often correct,
  result. Ratios and measurements, not grades.
- **Frozen scope during validation.** Detector, tracker, deduplication rule, signal
  definitions and schemas under active validation are not retuned in response to a
  validation figure. Changing them requires a new milestone and re-validation.
- Association, when built, is validated on its own terms: true associations, missed
  associations, **incorrect merges**, ambiguous cases, duplicate-count suppression and
  provenance integrity — with false merges treated as the most serious class.

## Quality gates

```bash
uv run pytest          # unit tests — no weights, no network, no GPU
uv run ruff check .
uv run ruff format .
uv run mypy            # strict, over src and tests
```

All four must pass. **No gate is ever weakened to make a change pass** — no loosened
assertion, deleted or skipped test, blanket `# type: ignore`, widened mypy override, new
ruff exemption, or lowered threshold. If a gate looks wrong, that is a human decision,
raised explicitly and separately.

Determinism is a gate too: same input, same output, byte-for-byte.

## Repository layout

- `services/vision/src/atreides_vision/` — implementation; `services/vision/tests/` — tests
- `docs/vision/` — one document per engineering milestone, with its limitations
- `config/` — zone and validation manifests · `artifacts/`, `data/input/` — git-ignored
- `services/api/`, `services/intelligence/`, `apps/operator-dashboard/`,
  `packages/contracts/` — placeholders

Module conventions: pure maths modules with no state, `models.py` for frozen domain
types, `config.py` for tunables carrying a unit and a rationale, `history.py` for bounded
state, `analyzer.py` as the facade the pipeline orchestrates, `reporting.py` for versioned
JSONL. Model libraries (`rfdetr`, `trackers`, `supervision`, `cv2`) stay confined to their
adapter file; no signal layer imports one, or another signal layer.

## Roadmap discipline

**V2 — extreme vision and advanced association** strengthens V1; it does not introduce
association from scratch. V2 areas: extreme low-light, strobe and rapidly changing
lighting, motion blur, severe occlusion, better track continuity, stronger ReID, long-gap
association, harder non-overlapping transitions, advanced cross-camera fusion, stronger
ambiguity handling, first-class confidence and observability.

**V3 — Atreides certified deployment** may introduce Atreides-specified RGB/NIR hardware,
optimised sensors, controlled optics, known camera topology, designed overlap,
standardised installation, calibration procedures, deployment verification and additional
sensor modalities. None of it is built during V1.

Discipline in both directions:

- Do not defer a V1 capability by calling it V2. Association and venue-local identity are
  V1.
- Do not pull V2/V3 sophistication into V1. V1 needs the **simplest sufficiently robust**
  association that works on real CCTV — not embeddings, training pipelines or GPU
  dependence adopted for their own sake.
- Choose V1 designs that leave room for stronger V2 methods behind the same seam.
- Build what an approved, bounded task asks for. Scope changes come from a human.

## Agent operating model

Project agents live in `.claude/agents/` and hold only role-specific rules; this file is
their shared context.

| Agent | Model | Role |
| --- | --- | --- |
| `atreides-architecture-guardian` | opus | Architectural review and planning before risky or boundary-touching work |
| `atreides-implementation-engineer` | opus | Implements approved, bounded scopes with tests |
| `atreides-validation-engineer` | sonnet | Independent validation — gates, regressions, edge cases, real footage |
| `atreides-adversarial-reviewer` | opus | Tries to falsify a design or implementation |
| `atreides-vision-researcher` | opus | Perception/CV analysis and algorithm recommendations |

The primary product-code-writing agent runs on Opus deliberately: implementation quality
and architectural compliance outrank speed and token cost.

Typical flow for non-trivial work: guardian decides the shape → implementation engineer
builds it → validation engineer measures it → adversarial reviewer attacks it. The vision
researcher informs the first step when the question is a perception one.

Agent Teams are enabled project-locally
(`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in `.claude/settings.json`) for work that
genuinely needs several independent sessions. Do not spin one up for ordinary tasks.

No agent commits, pushes, or runs destructive git commands. Product code changes come
from an approved implementation task — never as a side effect of review, validation or
research.
