# Atreides — V0.3.6: REAL-WORLD VALIDATION

Atreides is an AI-assisted venue security intelligence platform. It analyses existing
CCTV/video streams and gives human security operators situational awareness.

**Two numbering schemes appear in this project and mean different things.** `V1`, `V2`
and `V3` are *product generations* — what the product is meant to contain. `V0.x` are
*engineering milestones* — bounded increments of this repository, each with its own
document and its own validation. This repository is at engineering milestone **V0.3.6**,
working inside product generation **V1**. A V1 capability that is not implemented here
yet is unbuilt, not out of scope — and not thereby a V2 capability. See
[Product roadmap](#product-roadmap--v1-v2-v3) below; `CLAUDE.md` is the authoritative
architecture document.

**This repository contains the first four technical milestones.** They are
deliberately narrow:

- **V0.1 — SEE + TRACK.** ✅ Find people in a video and follow them frame to frame.
- **V0.2 — UNDERSTAND WHERE ACTIVITY OCCURS.** ✅ Measure crowd activity inside
  operator-defined regions of the camera view.
- **V0.2.1 — FIXED-CCTV VALIDATION.** ✅ Choose and justify the detector/tracker
  defaults by measurement on three materially different fixed cameras.
  Verdict: **PASS WITH DOCUMENTED LIMITATIONS** —
  see [docs/vision/V0.2.1_FIXED_CCTV_VALIDATION.md](docs/vision/V0.2.1_FIXED_CCTV_VALIDATION.md).
- **V0.3 — BEHAVIOR SIGNALS.** 🔄 In progress.
  - **V0.3.0 — behaviour signal foundation.** ✅ Neutral kinematic signals — speed,
    acceleration, turn rate, stationary duration — measured in an abstract coordinate
    space rather than pixels.
    See [docs/vision/V0.3_BEHAVIOR_SIGNALS.md](docs/vision/V0.3_BEHAVIOR_SIGNALS.md).
  - **V0.3.1 — ground-plane calibration.** ✅ Turns camera-local image positions into
    **venue-local metres** shared across cameras, so speed is `m/s` rather than frame
    fractions per second.
    See [docs/vision/V0.3.1_GROUND_PLANE_CALIBRATION.md](docs/vision/V0.3.1_GROUND_PLANE_CALIBRATION.md).
  - **V0.3.2 — calibration pipeline.** ✅ Exposes both through the CLI, adds a versioned
    JSONL output contract, and adds calibration validation and visual diagnostics.
    See [docs/vision/V0.3.2_CALIBRATION_PIPELINE.md](docs/vision/V0.3.2_CALIBRATION_PIPELINE.md).
  - **V0.3.3 — calibration integrity & spatial context.** ✅ Detects when a camera has
    moved and stops trusting its metres, passes zone labels into behaviour output, and
    adds output sampling and compression.
    See [docs/vision/V0.3.3_CALIBRATION_INTEGRITY_SPATIAL_CONTEXT.md](docs/vision/V0.3.3_CALIBRATION_INTEGRITY_SPATIAL_CONTEXT.md).
  - **V0.3.4 — interaction signals.** ✅ Neutral pairwise measurement between two
    subjects on one camera: separation, closing speed, relative direction of travel,
    time spent close.
    See [docs/vision/V0.3.4_INTERACTION_SIGNALS.md](docs/vision/V0.3.4_INTERACTION_SIGNALS.md).
  - **V0.3.5 — spatial cluster & crowd dynamics.** ✅ Neutral frame-local measurement
    across several subjects on one camera: which are spatially connected, how spread
    out, how locally dense, how aligned in motion, and for how long.
    See [docs/vision/V0.3.5_SPATIAL_CLUSTER_CROWD_DYNAMICS.md](docs/vision/V0.3.5_SPATIAL_CLUSTER_CROWD_DYNAMICS.md).
  - **V0.3.6 — real-world validation.** ✅ A reproducible harness that measures the
    whole V0.3 stack on real footage — tracking stability, signal availability, output
    volume, local throughput. It adds no signal, tunes nothing, and issues **no
    pass/fail verdict**. Automated system/proxy validation is complete; manual ground
    truth and physical metric validation are **pending**, and final signal-stack
    acceptance waits on both.
    See [docs/vision/V0.3.6_REAL_WORLD_VALIDATION.md](docs/vision/V0.3.6_REAL_WORLD_VALIDATION.md).

> **Read the V0.2.1 limitations before trusting any count.** In particular, a
> unique track ID is *not* a unique person: same-person ID multiplicity means the
> unique-ID count is an upper bound, not a census.

> **V0.3.6 measures proxies, not ground truth.** Everything the validation harness
> derives from Atreides' own output is a *proxy / system metric*. `track_births_per_minute`
> is **not** an ID-switch rate, a short-track fraction is **not** proof of fragmentation,
> and nothing automated anywhere is detection recall. True recall and true ID switches
> require human annotation, which the harness generates templates for and reports in a
> separate section — or not at all.

## What it does

```text
MP4  →  person detection  →  tracking  →  zone assignment  →  crowd metrics  →  annotated MP4
```

Zones are optional. Without `--zones` the pipeline is exactly V0.1.

The annotated output shows, per frame:

- person bounding boxes with persistent track IDs (`ID 7`) and confidence
- a short trajectory trail from each track's recent positions
- each configured zone's boundary, name and live metrics
- a top-left status panel:

```text
ATREIDES // ZONE INTELLIGENCE      (or ATREIDES // SEE + TRACK without zones)
Frame: 1240
Processing: 6.4 FPS
Tracks: 18
Zones: 4
```

Everything runs locally on a normal development machine. No NVIDIA hardware is
required — CPU works, Apple Silicon (MPS) is used automatically when available.

## What it intentionally does not do

These are product and privacy boundaries, not missing features. They hold in **every**
generation — V1, V2 and V3 alike:

- **It does not identify people.** A track ID is a handle on a box moving through
  one video. It is not a person, a name, or a record.
- **It does not perform face recognition** — or any biometric recognition.
- **It does not infer aggression**, intent, emotion, or intoxication. V0.3
  *measures* movement — speed, acceleration, turn rate, time spent still — and
  stops there. A measurement is not a judgement, and none of these signals is
  labelled, scored or ranked.
- **It does not classify demographics.**
- **It does not make security decisions.** It produces a view for a human operator.
- **It does not send video to a cloud service.** Raw video stays on your machine.
- **It does not persist tracks.** There is no database. State lives in memory for the
  duration of a run and is discarded when the process exits.
- **It does not carry identity beyond a session.** Nothing links a person between runs,
  between venues, or to a real-world record — in any generation.

**Different thing: what is simply not built yet.** Today every track ID is local to one
camera and one session and means nothing outside it, so this implementation cannot tell
that two cameras are watching the same person. That is the state of the V0.3.x
implementation, not a product boundary — multi-camera association and a *temporary*
venue-local identity are [V1 capabilities](#product-roadmap--v1-v2-v3). What that
identity will be when it exists: session-scoped, confidence-aware, held in memory,
discarded when the process exits, and used to avoid counting one person twice at venue
level. What it will never be: a name, a stored record, a face, or any biometric template.

Zone analytics operate only on anonymous tracking geometry: boxes, positions and
elapsed time. The metrics stream contains zone counts and durations, never track
identifiers or positions.

There is also no dashboard, API, alerting, risk scoring, or anomaly detection in
these milestones. **`motion_level` is not a risk score** — see below.

> The one network call in the whole project is the first-run download of the RF-DETR
> model weights. Your video is never uploaded.

## Product roadmap — V1, V2, V3

One product family, one core. Generations differ by capability and deployment profile,
not by forking the pipeline. `CLAUDE.md` holds the full architecture.

### V1 — CCTV multi-camera decision support

V1 runs on a venue's **existing** CCTV. It is multi-camera by definition, and its
required capabilities are:

| V1 capability | In this repository today |
| --- | --- |
| Multi-camera ingestion | one video per run; the validation harness processes many clips, each independently |
| Camera-local detection and tracking | ✅ V0.1, chosen and justified by measurement in V0.2.1 |
| Calibration and spatial context | ✅ V0.3.1 – V0.3.3, validated against synthetic geometry only |
| Behaviour, interaction and crowd signals | ✅ V0.3.0, V0.3.4, V0.3.5 |
| **Multi-camera association / fusion** | not implemented |
| **Temporary venue-local identity** | not implemented |
| Evidence aggregation over time | partial — bounded per-subject, per-pair and per-cluster history, within one camera and one run |
| Risk / decision engine foundation | not implemented |
| Decision-support events | not implemented |
| Validation and observability for pilots | in progress — V0.3.6 |

**Multi-camera association is a V1 capability**, not a V2 one. It is not implemented in
this repository yet, and the milestone sections below say so wherever it is relevant.
Its job is to decide when camera-local tracklets are the same real person right now —
so that one person walking past three cameras is not counted three times — and to
produce a temporary venue-local identity above the trackers, never inside them:

```text
camera_01 / track_17 ─┐
                      ├── venue_person_0042   (temporary, session-scoped, confidence-aware)
camera_02 / track_08 ─┘
```

Where the evidence is insufficient, the intended answer is *no merge* or *ambiguous* —
a missed association is recoverable, a wrong one silently corrupts every venue-level
number above it. Association also never destroys provenance: a venue-local subject
always resolves back to the `(camera, track)` observations that produced it.

### V2 — extreme vision and advanced association

V2 **strengthens V1**; it does not introduce association from scratch. Its areas:
extreme low-light robustness, strobe and rapidly changing lighting, stronger
motion-blur handling, severe occlusion, improved tracking continuity, stronger ReID,
long-gap association, difficult non-overlapping camera transitions, advanced
multi-camera fusion, and first-class confidence and observability.

"ReID" here means appearance-based matching methods that make association work in hard
conditions. It does not mean persistent or biometric identity, which is excluded in
every generation.

### V3 — Atreides certified deployment

V3 is where Atreides stops inheriting whatever cameras a venue already has: specified
RGB/NIR sensors, controlled optics, defined camera topology and overlap, standardised
installation, calibration procedures, deployment validation, and certified deployments.
None of it is built during V1, and nothing in this repository assumes it.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv installs it for you).

```bash
# 1. Install uv, if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies (creates .venv/ automatically)
uv sync
```

## Run

Put your own MP4 into `data/input/`, then:

```bash
# V0.1 — detection and tracking only
uv run atreides-vision run \
  --input data/input/test.mp4 \
  --output artifacts/tracked.mp4

# V0.2 — with zones and crowd metrics
uv run atreides-vision run \
  --input data/input/test.mp4 \
  --output artifacts/zones.mp4 \
  --zones config/zones.example.json \
  --metrics-output artifacts/zone_metrics.jsonl

# V0.3 — pairwise interaction measurement (metric)
uv run atreides-vision run \
  --input data/input/test.mp4 \
  --output artifacts/tracked.mp4 \
  --interaction-output artifacts/interactions.jsonl \
  --calibration config/calibration.example.json

# V0.3 — spatial cluster / crowd measurement (metric)
uv run atreides-vision run \
  --input data/input/test.mp4 \
  --output artifacts/tracked.mp4 \
  --crowd-output artifacts/crowd.jsonl \
  --calibration config/calibration.example.json

# V0.3 — with behaviour signals (camera-local, normalized units)
uv run atreides-vision run \
  --input data/input/test.mp4 \
  --output artifacts/tracked.mp4 \
  --behavior-output artifacts/behavior.jsonl \
  --camera-id cam_01

# V0.3.6 — real-world validation over a manifest of clips
uv run atreides-vision validation run \
  --manifest config/validation/v0.3.6.json \
  --output artifacts/v0.3.6-validation

# V0.3 — with calibrated venue metres
uv run atreides-vision run \
  --input data/input/test.mp4 \
  --output artifacts/tracked.mp4 \
  --behavior-output artifacts/behavior.jsonl \
  --calibration config/calibration.example.json
```

Behaviour analysis is off unless `--behavior-output` is given, and writing it changes
nothing about the rendered video.

Inspect a calibration before trusting metres from it:

```bash
uv run atreides-vision calibration validate --calibration config/calibration.example.json

uv run atreides-vision calibration render \
  --calibration config/calibration.example.json \
  --input data/input/test.mp4 \
  --output artifacts/calibration.png
```

`render` draws the venue grid, axes and reference points onto a frame. Do the grid
lines follow the real floor? That is the one check reprojection error cannot perform.

The first run downloads the RF-DETR Nano weights (~349 MB) to `~/.roboflow/models/`.
Every run after that is fully offline.

### Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--input` | required | Source video path |
| `--output` | required | Annotated MP4 destination |
| `--confidence` | `0.30` | Minimum detection confidence, in `(0, 1]` (V0.2.1 validated) |
| `--device` | `auto` | `auto`, `cpu`, `mps`, or `cuda` |
| `--tracker` | `ocsort` | Tracker implementation: `ocsort` or `bytetrack` |
| `--deduplicate-detections` | on | Rule-B3 duplicate suppression. `--no-deduplicate-detections` disables |
| `--ocsort-high-conf-threshold` | `0.30` | Confidence above which OC-SORT may start a track |
| `--tracking-debug-output` | off | Per-frame tracker internals as JSON Lines (observational) |
| `--max-frames` | all | Stop after N frames — useful for a quick look at a long video |
| `--zones` | off | Zone configuration JSON. Omit for plain V0.1 behaviour |
| `--metrics-output` | off | Write zone metrics as JSON Lines. Requires `--zones` |
| `--metrics-interval` | `1.0` | Seconds of **source** time between metric records |
| `--behavior-output` | off | Write behaviour signals as JSON Lines. Enables behaviour analysis |
| `--calibration` | off | Ground-plane calibration JSON. Switches signals to metres. Needs `--behavior-output` |
| `--camera-id` | from config | Camera namespace for behaviour records |
| `--calibration-reference` | off | Reference frame enabling camera-drift monitoring. Needs `--calibration` |
| `--integrity-report-only` | off | Keep emitting after drift is detected (diagnostic; output may be wrong) |
| `--behavior-output-interval` | every frame | Seconds of source time between emitted records, on all three streams |
| `--interaction-output` | off | Write pairwise interaction measurements as JSON Lines. Enables behaviour analysis internally |
| `--crowd-output` | off | Write spatial-cluster and crowd measurements as JSON Lines. Enables behaviour analysis internally |
| `--trail-length` | `25` | Centre points kept and drawn per track |
| `--no-confidence-labels` | off | Draw `ID <n>` without the score |
| `--log-every` | `100` | Progress log interval, in frames |
| `--verbose` | off | Debug logging |

#### `validation` options (V0.3.6)

`validation run` and `validation report` share `--manifest`, `--output`, `--clip`
(repeatable) and `--verbose`. `run` adds:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--device` | `auto` | `auto`, `cpu`, `mps`, or `cuda` |
| `--max-frames` | all | Per-clip cap for a smoke run. Recorded in every result, because a capped run's metrics are **not** comparable with a full one's |
| `--sample-frames` | `10` | Deterministic still frames sampled per clip for human review |
| `--continuity-windows` | `3` | Short windows sampled per clip for track-continuity review |
| `--continuity-window-seconds` | `8.0` | Length of each continuity window |

**There is no detector or tracker option** — not one that is accepted and ignored,
there is no parameter to pass. V0.3.6 measures the frozen V0.2.1 defaults, and a
harness that took overrides would eventually be used to tune against its own report.

`auto` prefers CUDA, then MPS, then CPU. Asking for an accelerator that isn't
available fails immediately with an explanation instead of an obscure error from
deep inside torch:

```bash
uv run atreides-vision run --input in.mp4 --output out.mp4 --device cuda
# ERROR  CUDA was requested but no CUDA device is available on this machine.
#        Re-run with --device cpu (always supported), --device mps on Apple
#        Silicon, or --device auto to pick the best available backend.
```

Expect single-digit FPS on CPU. This milestone is about correctness, not throughput.

### Choosing a tracker

OC-SORT is the default since V0.2.1. It scores association on direction consistency
as well as IoU, and recovers a track from its last observation before a gap rather
than from a drifted Kalman prediction — which is what ByteTrack could not do when a
person briefly occluded by someone else came back with a new track id.

ByteTrack remains fully supported:

```bash
uv run atreides-vision run --input in.mp4 --output out.mp4 --tracker bytetrack
```

Note that ByteTrack's own `track_activation_threshold` is still `0.50`, so combined
with the `0.30` detector default it will not start tracks from detections scoring
between 0.30 and 0.50. That is ByteTrack's baseline, left untouched by V0.2.1.

Both are motion-only. Neither computes appearance descriptors, and neither uses
pixels for identity, so the privacy boundaries above hold for either choice.
Because they sit behind the same `MultiObjectTracker` interface and share one
conversion layer, swapping them changes the tracker and nothing else — which is
what makes an A/B on the same clip meaningful.

## Zones

### Defining zones

Zones live in a JSON file, never in Python. Copy `config/zones.example.json` and
edit it for your camera:

```json
{
  "camera_id": "cam_01",
  "zones": [
    {
      "id": "dancefloor",
      "name": "Dancefloor",
      "polygon": [[0.22, 0.30], [0.80, 0.30], [0.88, 0.80], [0.14, 0.80]]
    }
  ]
}
```

**Coordinates are normalized to the frame**, not pixels:

```text
x = pixel_x / frame_width      0.0 = left edge,  1.0 = right edge
y = pixel_y / frame_height     0.0 = top edge,   1.0 = bottom edge
```

So `[0.5, 0.5]` is the centre of the frame at any resolution, and the same
configuration works whether the camera streams at 720p or 4K.

To find coordinates for a real camera, grab a frame and divide the pixel position
of each corner by the frame size. A polygon may have any number of points from
three upward — a four-point trapezoid usually fits a floor seen at an angle better
than a rectangle.

Validation is strict and errors name the offending field: at least three points,
every coordinate within `[0, 1]`, unique zone ids, non-empty names. A bad zone file
fails before the detector loads, so you find out immediately.

**Overlapping zones are allowed and are not an error.** Each zone is evaluated
independently, so one person can count toward several zones at once. A small
"entrance" polygon inside a larger "lounge" polygon reports occupancy for both.

### Person position

A person's zone is decided by the **bottom-centre of their bounding box** — roughly
where their feet are — not the box centre:

```text
┌───────────┐
│           │
│  PERSON   │
│           │
└─────●─────┘
      ↑ this point decides the zone
```

For a camera looking across a floor, the box centre sits at chest height and drifts
into whatever is behind the person. The bottom edge is the floor position.

A point exactly on a zone boundary counts as **inside**, so two zones sharing an
edge both claim someone standing on the line rather than neither claiming them.

### Metrics

Per zone, per frame:

| Metric | Meaning |
| --- | --- |
| `occupancy` | Tracked people currently in the zone |
| `entries` | Cumulative outside → inside transitions this session |
| `exits` | Cumulative inside → outside transitions this session |
| `average_dwell_seconds` | Mean time in the zone, over people currently inside |
| `max_dwell_seconds` | Longest current stay |
| `mean_normalized_speed` | Mean image-plane speed of those people |
| `motion_level` | Smoothed 0.0–1.0 descriptor of movement activity |
| `completed_visits` | Stays that have ended this session |

`occupancy == entries - exits` holds at every frame.

**Entries and exits use a grace period.** A track counts as present from the first
frame it is seen inside until it has gone `exit_grace_frames` (default 8, about a
third of a second at 25 FPS) without being seen inside. This is what stops a dropped
detection from producing a phantom exit-then-re-entry pair, and stops someone
loitering on a boundary from churning the counters. The cost: a real exit is counted
that many frames late. The completed visit is still timestamped at the person's last
actual sighting, so dwell durations stay truthful.

**Dwell and speed come from source timestamps**, derived from the video's real frame
rate. Nothing assumes 30 FPS; a 10 FPS and a 60 FPS recording of the same scene
report the same durations and speeds.

#### Speed is image-plane movement, not physical speed

`mean_normalized_speed` is measured in **normalized frame units per second**: `0.1`
means a person's floor point moved a tenth of the frame per second. It is **not**
metres per second and cannot be converted to one without camera calibration. A
person walking near the camera crosses far more of the frame per step than someone
the same speed at the back of the room. Treat it as a relative, per-camera quantity.

It is computed as path length over the recent window, not straight-line
displacement, so someone dancing on the spot registers as moving.

#### `motion_level` is a descriptive heuristic, not a risk score

`motion_level` is mean speed divided by a reference speed, clamped to `[0, 1]` and
smoothed over recent frames. That is the whole formula.

It describes **how much movement there is**, relative to what counts as brisk on
this camera. A busy dancefloor scores high because dancing is movement. It is
**not** a danger, threat, aggression, anomaly or incident score, carries no
judgement about anybody, and must not be presented as one.

The reference speed (`motion_reference_speed`, default `0.15`) is **provisional**: a
presentation constant chosen by eye, not a calibrated measurement. It needs re-tuning
per camera — if your `motion_level` sits pinned at `1.00`, the reference is too low
for that camera's geometry.

### Metrics output

`--metrics-output` writes JSON Lines, one record per sampled frame. Metrics are
computed every frame; sampling only controls how often they are written, defaulting
to once per second of source time.

```json
{"frame":76,"timestamp_seconds":3.0,"zones":[{"id":"dancefloor","name":"Dancefloor","occupancy":27,"entries":8,"exits":4,"mean_normalized_speed":0.083,"motion_level":0.54,"average_dwell_seconds":2.8,"max_dwell_seconds":4.0,"completed_visits":4}]}
```

### Tuning

Defaults live in `ZoneAnalysisConfig` (`services/vision/src/atreides_vision/zones/config.py`),
each with a comment explaining the choice:

| Setting | Default | Purpose |
| --- | --- | --- |
| `exit_grace_frames` | `8` | Missing frames tolerated before an exit is counted |
| `speed_window_seconds` | `1.0` | Window over which speed is averaged |
| `motion_reference_speed` | `0.15` | Provisional speed treated as fully active |
| `motion_smoothing_frames` | `25` | Smoothing span for `motion_level` |
| `metrics_sample_interval_seconds` | `1.0` | Metrics stream sampling interval |
| `stale_track_ttl_frames` | `150` | When unseen tracks' movement history is dropped |

### A note on camera movement

Zones are fixed regions of the frame, so they assume a **fixed camera**. On handheld
or panning footage the zones stay put while the venue moves underneath them, and
both zone membership and measured speed pick up the camera's motion as well as the
crowd's. Zone analysis is meant for mounted CCTV.

## Test

```bash
uv run pytest          # unit tests — no model weights, no network, no GPU
uv run ruff check .    # lint
uv run ruff format .   # format
uv run mypy            # strict type check
```

The test suite runs in a few seconds and never downloads anything. It covers the
domain models, configuration and device policy, video metadata and sink lifecycle,
trajectory cleanup, both ML adapters (via fakes), zone geometry and validation, zone
state transitions and the grace period, dwell and speed at several frame rates,
motion-level bounds, memory cleanup, the overlay, the CLI, and the full pipeline
end-to-end with fake detector and tracker implementations — including a regression
guard that a run without `--zones` still behaves as V0.1.

V0.3 adds coordinate projection, the kinematic formulas checked against closed-form
synthetic trajectories, motion segments and gap handling, per-subject state
isolation, and — the important one — **resolution invariance**: the same trajectory
at 384×288 and 1920×1080 must produce identical signals, where a pixel-based
measurement would have differed fivefold.

V0.3.1 adds homography estimation against synthetic cameras with known geometry,
degeneracy rejection, runtime-resolution rescaling, and two proofs that matter:
**perspective correction** (the same physical metre measured 2.1× differently in
image space, 1.000× after calibration) and **multi-camera agreement** (two cameras
with different geometry and resolution reporting the same speed to 4×10⁻¹⁵ m/s,
while their observations stay strictly separate).

V0.3.6 adds the validation harness's own tests: every metric checked against synthetic
structured records with analytic answers, the manifest's identity rules, the manual
annotation parser (including the cases that must **not** produce a number — zero
visible people, a false-positive-only frame, a half-filled row), deterministic sample
points, the runner end-to-end on a synthetic MP4 with fake detector and tracker, the
run -> report round trip, and the report guarantees themselves: deterministic output,
proxy and ground truth never sharing a label, no physical unit attached to any value
for uncalibrated footage, and no pass/fail verdict anywhere.

## Architecture

```text
atreides-platform/
├── config/zones.example.json    # zone configuration to copy and edit
├── config/validation/v0.3.6.json # V0.3.6 validation input manifest
├── apps/operator-dashboard/     # placeholder — not this milestone
├── packages/contracts/          # placeholder — shared contracts, later
├── services/
│   ├── vision/                  # ← everything in V0.1 and V0.2 lives here
│   │   ├── src/atreides_vision/
│   │   │   ├── models.py            # Detection, Track, BoundingBox, Point
│   │   │   ├── config.py            # config objects + device resolution
│   │   │   ├── errors.py            # operator-readable error hierarchy
│   │   │   ├── cli.py               # composition root
│   │   │   ├── detection/           # PersonDetector interface + RF-DETR adapter
│   │   │   ├── tracking/            # MultiObjectTracker interface + ByteTrack adapter
│   │   │   ├── video/               # source, sink, trajectories, processor
│   │   │   ├── zones/               # V0.2 zone intelligence
│   │   │   │   ├── geometry.py          # floor point, normalization, point-in-polygon
│   │   │   │   ├── models.py            # Zone, ZoneLayout, ZoneMembership, ZoneSnapshot
│   │   │   │   ├── loader.py            # JSON loading and validation
│   │   │   │   ├── config.py            # tunables, each with a rationale
│   │   │   │   ├── assignment.py        # tracks -> zones, per frame
│   │   │   │   ├── state.py             # membership, dwell, movement history
│   │   │   │   ├── metrics.py           # motion level, snapshot assembly
│   │   │   │   ├── analyzer.py          # the facade VideoProcessor orchestrates
│   │   │   │   └── reporting.py         # JSONL metrics stream
│   │   │   ├── behavior/            # V0.3.0 behaviour signals
│   │   │   │   ├── coordinates.py       # MotionSpace, projectors — never raw pixels
│   │   │   │   ├── models.py            # SubjectRef, BehaviorObservation, BehaviorSignal
│   │   │   │   ├── config.py            # tunables, each with a unit and rationale
│   │   │   │   ├── history.py           # bounded per-subject motion segments
│   │   │   │   ├── kinematics.py        # pure formulas, no state
│   │   │   │   ├── analyzer.py          # the facade VideoProcessor orchestrates
│   │   │   │   └── reporting.py         # V0.3.2 versioned JSONL output contract
│   │   │   ├── interaction/         # V0.3.4 pairwise measurement
│   │   │   │   ├── models.py             # SubjectPairRef, SpatialContext, snapshots
│   │   │   │   ├── geometry.py           # pure pairwise maths
│   │   │   │   ├── history.py            # bounded per-pair separation history
│   │   │   │   ├── analyzer.py           # the facade VideoProcessor orchestrates
│   │   │   │   └── reporting.py          # schema 0.3.4 JSONL stream
│   │   │   ├── crowd/               # V0.3.5 spatial cluster measurement
│   │   │   │   ├── models.py             # ClusterMembers, snapshots, CrowdFrame
│   │   │   │   ├── geometry.py           # pure spatial maths (spread, density)
│   │   │   │   ├── clustering.py         # radius graph and connected components
│   │   │   │   ├── history.py            # per-composition persistence, bounded
│   │   │   │   ├── analyzer.py           # the facade VideoProcessor orchestrates
│   │   │   │   └── reporting.py          # schema 0.3.5 JSONL stream
│   │   │   ├── validation/          # V0.3.6 real-world validation harness
│   │   │   │   ├── manifest.py           # which clips, which camera, what kind
│   │   │   │   ├── storage.py            # artifact layout + deterministic writers
│   │   │   │   ├── streams.py            # streaming JSONL reader (gzip transparent)
│   │   │   │   ├── distributions.py      # one percentile implementation, shared
│   │   │   │   ├── track_metrics.py      # presence, lifetime, churn proxies
│   │   │   │   ├── behavior_metrics.py   # signal coverage and warm-up
│   │   │   │   ├── interaction_metrics.py# pair episodes and churn propagation
│   │   │   │   ├── crowd_metrics.py      # composition episodes and persistence
│   │   │   │   ├── ground_truth.py       # manual annotation: the ONLY human input
│   │   │   │   ├── review_samples.py     # deterministic frames + contact sheets
│   │   │   │   ├── runner.py             # composes the frozen pipeline, times it
│   │   │   │   ├── analysis.py           # run artifacts -> ClipValidationResult
│   │   │   │   ├── models.py             # result types and serialised shapes
│   │   │   │   └── reporting.py          # summary JSON, CSV, Markdown report
│   │   │   ├── calibration/         # V0.3.1 ground-plane calibration
│   │   │   │   ├── homography.py        # normalized DLT — pure array maths
│   │   │   │   ├── models.py            # VenuePoint, correspondences, quality
│   │   │   │   ├── fitting.py           # solve + measure + safeguard
│   │   │   │   ├── projector.py         # GroundPlaneProjector (a MotionProjector)
│   │   │   │   ├── loader.py            # JSON loading and validation
│   │   │   │   └── integrity/            # V0.3.3 camera-drift detection
│   │   │   │       ├── registration.py       # ORB scene matching — the only OpenCV here
│   │   │   │       ├── monitor.py            # evidence -> state (measures)
│   │   │   │       └── policy.py             # guard (decides)
│   │   │   └── visualization/       # overlay drawing + calibration diagnostics
│   │   └── tests/
│   ├── intelligence/            # placeholder
│   └── api/                     # placeholder
├── data/input/                  # your footage (git-ignored)
└── artifacts/                   # rendered output (git-ignored)
```

### How zone analysis attaches to the pipeline

```text
read frame → detect → track → trajectories → zone assignment
           → zone state → zone metrics → overlay → write frame
```

`VideoProcessor` gained one optional collaborator, `ZoneAnalyzer`, rather than a
second job. The analyzer owns polygons, grace periods, dwell timing and smoothing;
the processor calls `analyzer.update(tracks, ...)` and passes the result to the
annotator. With no analyzer the loop is byte-for-byte the V0.1 pipeline.

The zone layer consumes `Track` objects and nothing else. It has no import of
RF-DETR, ByteTrack or any model, so swapping either leaves zone intelligence
untouched — the same property the detector and tracker adapters exist to provide.

V0.3's `BehaviorAnalyzer` attaches the same way and holds to the same rule: it
consumes `Track` objects and a coordinate projector, imports no model *and not the
zone layer either*, and is measurement-only — a run with one attached renders
byte-identical output to one without.

### Why the detector and tracker sit behind adapters

RF-DETR is the *initial* detector. ByteTrack is the *initial* tracker. Both will be
replaced — by a fine-tuned model, a different architecture, or hardware-accelerated
inference — and that replacement must not ripple through the pipeline.

So the pipeline depends on two Atreides-owned interfaces:

```python
class PersonDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class MultiObjectTracker(Protocol):
    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]: ...
```

and on Atreides-owned domain models. `Detection` and `Track` are the only things that
cross a module boundary. Concretely:

- `rfdetr` is imported in exactly one file: `detection/rfdetr.py`. That file owns the
  BGR→RGB conversion, the COCO label space, and the person-class filter.
- `trackers` and `supervision` are imported in exactly one file:
  `tracking/bytetrack.py`. That file owns both directions of the
  `Detection ↔ sv.Detections ↔ Track` conversion.
- `VideoProcessor` imports neither. It reads frames, calls two interfaces, updates
  trajectories, annotates, and writes.

Swapping a detector means writing one new adapter file and one line in `cli.py`.
The payoff is already visible in the tests: the entire pipeline is verified with fake
detector and tracker implementations, with no neural network involved.

Two further design notes:

- **State is bounded on every axis.** Trajectory trails, zone membership and movement
  history all expire on a TTL and have a hard ceiling that evicts the
  least-recently-seen entry. A long run cannot grow memory without limit, which is
  covered by tests that churn a thousand distinct track ids.
- **No hidden global state.** Configuration is constructed in `cli.py`, validated at
  construction, and passed down explicitly.

## Dependencies

| Package | Why |
| --- | --- |
| `rfdetr` | RF-DETR Nano, the initial person detector |
| `trackers` | ByteTrack, the initial multi-object tracker |
| `opencv-python` | Video decode/encode and all drawing |
| `numpy` | Array interchange between the layers |

V0.2 added no dependencies. Point-in-polygon is ~20 lines of ray casting rather than
a geometry library, which keeps the boundary contract ours to define and test.

V0.3 added none either — the behaviour package uses only the standard library, and
not even NumPy or OpenCV. V0.3.1's calibration package uses NumPy for the one-off
homography solve and plain arithmetic per point; homography estimation is ~40 lines
of normalized DLT rather than a call into OpenCV, for the same reason
point-in-polygon is ours.

Dev: `pytest`, `ruff`, `mypy`. Deliberately absent: Ultralytics, cloud SDKs, database
libraries, web frameworks, Docker.

## Current milestone — V0.3: BEHAVIOR SIGNALS

V0.2.1 closed the question of *which* configuration to run. V0.3 turns tracks into
behaviour signals. **V0.3.0 through V0.3.5 are built**: the signal foundation,
ground-plane calibration, runtime exposure, calibration-integrity monitoring,
pairwise interaction measurement, and spatial-cluster measurement.

Four neutral measurements per tracked subject per frame: `speed`, `acceleration`,
`heading_change_rate` and `stationary_duration`. Each carries its subject, timestamp,
unit, coordinate space and an estimate-support quality. None carries a label,
score or judgement.

The point of the milestone is the architecture underneath. Behaviour maths never
touches a pixel: an injectable `MotionProjector` converts a person's bottom-centre
anchor into an abstract `MotionSpace`, so nothing encodes a venue size, a camera
coverage area, a mounting height or a resolution.

### V0.3.1 — ground-plane calibration

That abstraction is what made **V0.3.1** cheap. Four or more operator-supplied
correspondences — *"this pixel is that spot on the floor"* — determine a planar
homography, and `GroundPlaneProjector` implements the same `MotionProjector`
protocol. **No kinematics, history or signal code changed:** speed simply starts
arriving in `m/s` and acceleration in `m/s²`, because units are derived from the
coordinate space rather than hard-coded.

Why it was necessary, measured on a synthetic camera — one person, walking exactly
one metre in one second, at two depths in the same shot:

| | Near the camera | Far from the camera |
| --- | --- | --- |
| Normalized image speed | `0.0436` frame-widths/s | `0.0206` frame-widths/s |
| Calibrated ground plane | **`1.000 m/s`** | **`1.000 m/s`** |

Image space disagrees with itself by **2.1×** for the same physical metre. That is
why a normalized threshold silently encodes one camera and one spot on its floor.

Venue coordinates are **shared across cameras**: the operator picks one origin for
the venue and expresses every camera's reference points in it. Two synthetic cameras
with different geometry, different lenses and different resolutions report the same
walk to within `4×10⁻¹⁵ m/s`.

**Spatial agreement is not identity.** Two cameras placing someone at the same
coordinates are two independent observations — `(cam_a, 7)` and `(cam_b, 7)` remain
two distinct subjects, and nothing in this implementation associates them. A
camera-local track ID is still not a unique venue person: that is limitation (2) from
the V0.2.1 report, unfixed and not pretended otherwise.

Shared venue coordinates are what a future association layer will reason over, but they
are not a substitute for it. Multi-camera association is a V1 capability that this
milestone does not implement; when it does arrive it sits **above** tracking and
calibration as its own layer, producing a temporary venue-local identity. It will not
change what a camera-local track ID means — a tracklet stays a tracklet.

One homography describes **one plane**. Stairs, balconies, platforms and multiple
floors are not supported. No calibration has been validated against a real measured
venue — every number above is synthetic geometry.

### V0.3.2 — calibration pipeline

V0.3.2 makes all of the above reachable from the runtime. `--behavior-output` turns
behaviour analysis on; `--calibration` switches it from frame fractions to venue
metres. Signals leave the process as **versioned JSONL** (`schema_version 0.3.2`),
one record per subject per frame, carrying position, units, coordinate space and
calibration provenance — so a future intelligence layer consumes structured data
rather than parsing logs.

Behaviour analysis attaches to `VideoProcessor` as an optional collaborator, exactly
as zones do, and is measurement-only: with it enabled the rendered video is
**byte-identical**. Nothing is drawn.

Two calibration inspection commands were added: `calibration validate` reports the
fit numerically (with no GOOD/BAD verdict — no validated threshold exists), and
`calibration render` draws the venue grid onto a frame, which is the only check that
can catch a reference point measured against the wrong spot.

### V0.3.3 — calibration integrity & spatial context

A ground-plane calibration is only valid while the camera stays put. Nudge it, rotate
it, re-zoom or re-crop the stream, and the homography keeps producing confident metre
values from a premise that is no longer true — and nothing downstream can tell.

`--calibration-reference` enables monitoring against a reference frame captured from
the calibrated geometry. ORB feature matching (no new dependency, no neural model,
static scene only — never people) runs every 5 s of source time and reports one of
four states. On detected drift, **metric observations stop for the rest of the run**
and the stream records why. Precise-looking metres from a moved camera are worse than
none.

Drift is measured as **corner displacement relative to the frame diagonal**, never in
pixels — so one threshold serves a doorway camera and a festival field, and a
proportional decode resize correctly scores zero.

`CONSISTENT` means only that *the view has not changed*. It is not a claim that the
calibration's metres were ever correct — a badly measured calibration is perfectly
consistent, just wrong. And "could not tell" is reported as `INSUFFICIENT_EVIDENCE`,
never quietly as "unchanged".

Zone membership now reaches behaviour output as neutral `zone_ids` context — opaque
labels only. Zones remain camera-local image-space polygons; a label attached to a
metric position does not make it venue geometry.

Behaviour output also gained volume control: `--behavior-output-interval` and gzip via
a `.jsonl.gz` path. **Analysis is never sampled** — every frame is still analysed in
full, so signals are unaffected and only serialisation is thinned.

### V0.3.4 — interaction signals

V0.3.0–V0.3.3 measured people one at a time. V0.3.4 measures how two observed
subjects relate: **separation**, **closing speed** (positive means the gap is
shrinking), **relative direction of travel**, and **time spent within a proximity
threshold**. `--interaction-output` writes them to their own versioned JSONL stream
(schema `0.3.4`) and enables behaviour analysis internally, so no behaviour file is
required to get one.

Verified against arithmetic: two people walking head-on at 1 m/s each close at
exactly `+2.0000 m/s`; walking apart gives `−2.0000`; walking in parallel gives `0`.
And two people a fixed 2 m apart measure **more than 1.5× differently in image space**
depending on their depth in the frame, while calibration recovers `2.0000 m` at both.

**Pairs are same-camera only, and that is deliberate.** `cam_A/track_7` is never
paired with `cam_B/track_12` even when both project to nearly the same venue
coordinates: with no association layer to say whether those are two people or one, a
single person seen by two cameras would be paired *with themselves*. Cross-camera
pairing is rejected at construction, not merely absent. That guard is correct for as
long as association is unimplemented — and it stays correct afterwards, because
cross-camera pairing only becomes meaningful through venue-local identity, one layer up,
never by relaxing this rule.

Everything is a measurement. There is no confrontation, aggression, threat, conflict
or escalation signal — those need evidence Atreides does not have, and a layer that
measured *and* judged would make both jobs impossible to audit.

### V0.3.5 — spatial cluster & crowd dynamics

V0.3.4 measured people two at a time. V0.3.5 measures the arrangement of *several*
at once: which subjects are **spatially connected**, the cluster's **centroid** and
**spread**, its **mean and maximum member separation**, the **local density** around
each member, how **aligned their directions of travel** are, and how long that exact
composition has **persisted**. `--crowd-output` writes them to their own versioned
JSONL stream (schema `0.3.5`) and enables behaviour analysis internally.

Clusters are connected components under a radius rule, so **they chain**: four people
1.4 m apart in a line are one cluster under a 1.5 m radius even though the ends are
4.2 m apart. That is the rule, not a defect — it is what makes a queue register as one
arrangement instead of fragmenting into pairs.

**A spatial cluster is not a social group.** It is a set of camera-local tracklets that
were geometrically connected at one instant, and `member_count` is a tracklet count,
not a headcount. Clustering is same-camera only and rejected across cameras, exactly
as pairing is.

Perspective is why calibration matters here: at a fixed normalized radius, the same
three people **cluster at the back of the room and do not at the front**, because a
fixed fraction of the frame covers less floor as they approach. Calibrated, both
depths recover the same arrangement to 1e-4.

There is no crowd risk score, density verdict, surge, crush, panic or unrest signal.

### V0.3.6 — real-world validation & signal-stack freeze

V0.3.0–V0.3.5 were verified against synthetic inputs with known answers, which checks
that a formula is correct. It does not say what the stack does on real footage. V0.3.6
answers that, reproducibly, and then freezes the scope.

```bash
# Process every clip in the manifest, measure, write all three artifacts.
uv run atreides-vision validation run \
  --manifest config/validation/v0.3.6.json \
  --output artifacts/v0.3.6-validation

# Regenerate the artifacts from an existing run — no footage re-processed.
# This is what picks up manual annotation added since the run.
uv run atreides-vision validation report \
  --manifest config/validation/v0.3.6.json \
  --output artifacts/v0.3.6-validation
```

It measures, per clip: track lifetime and occupancy, fragmentation and churn proxies,
observation gaps, behaviour-signal coverage and warm-up, pair-episode stability, how
often pair episodes end because a **member track ID stopped existing**, cluster
composition churn and persistence, output volume, and local wall-clock throughput.

**The two measurement classes never mix.** Automated metrics are labelled proxy/system
metrics everywhere they appear — in the types, the JSON, the `proxy_` CSV prefix and
the report. Manual ground truth appears only where a human actually annotated, under
`manual_`, and is **absent entirely** otherwise: an un-annotated clip reports
`NOT_ANNOTATED` and emits no recall, no precision and no ID-switch figure at all,
rather than a zero that would be charted next to a real one.

**No physical units for uncalibrated footage.** None of the validation clips has a
measured floor calibration, so all of them run in normalized image units and no metre,
m/s or subjects/m² figure is derived. No floor dimension is estimated from doors,
tiles, vehicles or people. A manifest entry can carry a `calibration_path` for a future
physically measured scene; none currently does.

**No verdict.** There is no pass, fail, grade or acceptance band anywhere — the evidence
does not support one. Where scenes differ, the report states a ratio between
measurements ("2.3× the sub-1 s track fraction"), never a judgement.

**Camera IDs may repeat.** One physical camera supplies many independent clips, so
`camera_id` is not required to be unique across a manifest; only `clip_id` is. Sharing
a camera ID asserts that the footage came from the same camera and **nothing more** —
each clip is processed, written and measured entirely separately, so
`camera_01/clip_A/track_7` and `camera_01/clip_B/track_7` are two independent
tracklets. Per-clip counts from clips sharing a camera must not be summed.

The harness is a *consumer* of the runtime: no measurement package imports it, the
runner composes the same objects `run` does, and it accepts **no** detector or tracker
option — not one that is ignored, there is no parameter to pass.

Full detail and the complete limitation lists:
[V0.3.0](docs/vision/V0.3_BEHAVIOR_SIGNALS.md) ·
[V0.3.1](docs/vision/V0.3.1_GROUND_PLANE_CALIBRATION.md) ·
[V0.3.2](docs/vision/V0.3.2_CALIBRATION_PIPELINE.md) ·
[V0.3.3](docs/vision/V0.3.3_CALIBRATION_INTEGRITY_SPATIAL_CONTEXT.md) ·
[V0.3.4](docs/vision/V0.3.4_INTERACTION_SIGNALS.md) ·
[V0.3.5](docs/vision/V0.3.5_SPATIAL_CLUSTER_CROWD_DYNAMICS.md) ·
[V0.3.6](docs/vision/V0.3.6_REAL_WORLD_VALIDATION.md).

Persistent identity, biometrics, demographics and behaviour *classification* remain out
of scope by design, in every generation. Cross-camera association, temporary venue-local
identity, evidence aggregation and the risk / decision-support layer are a different
case: they are V1 product capabilities that these milestones do not implement — out of
scope *for V0.3*, not out of scope for the product.

### V0.3 validation status

**V0.3.6 does not establish that the V0.3 signal stack is sufficient, accurate against
reality, or valid in production.** It completes one review of several:

| Stage | Status |
| --- | --- |
| Implementation scope (V0.3.0 – V0.3.5 signals) | **complete** |
| Automated system / proxy validation on real footage | **complete** |
| Manual detection & tracking ground truth | **pending** |
| Controlled physical metric validation | **pending** |
| Production validation | **not established** |
| Final signal-stack acceptance / freeze | **pending the reviews above** |

**Frozen for the duration of validation** — not to be tuned in response to any
validation figure, and not to be changed without a new milestone and a re-validation:

| | |
| --- | --- |
| Detector | RF-DETR Nano, confidence 0.30 |
| Tracker | OC-SORT — high-confidence 0.30, minimum IoU 0.30, lost-track buffer 30, minimum consecutive frames 2, direction-consistency weight 0.2, `delta_t` 3 |
| Deduplication | Rule B3, enabled |
| Signals | individual, pairwise and crowd sets as built in V0.3.0–V0.3.5 |
| Schemas | behaviour `0.3.3`, interaction `0.3.4`, crowd `0.3.5` |

Known limitations of the implemented stack:

1. **Camera-local track IDs are tracklets, not guaranteed unique people.** Occlusion
   fragments identity; when it does, behaviour history, pair history and exact cluster
   composition all reset.
2. **No physically validated metric output.** The projection is tested against
   synthetic cameras with known geometry; no real scene has been validated against
   measured floor distances.
3. **Camera-local only.** Each camera is processed independently: no multi-camera
   association, no fusion, no venue-local identity, no appearance embeddings. Association
   and *temporary* venue-local identity are V1 capabilities that are not yet built;
   stronger ReID and long-gap association are V2 improvements on top of them. Face
   recognition and persistent identity are excluded permanently, not pending.
4. **Measurement only.** No risk engine, no decision engine, no behaviour
   classification, no operator alerting. The risk / decision-support foundation is V1
   work that has not started.
5. **Detection is the floor for everything.** Missed detections in distant or dense
   regions propagate into every signal above them.

The validation harness itself is **not** frozen: adding a clip or a metric definition
cannot change what the pipeline produces.

One open finding, recorded and deliberately not acted on: **median exact-composition
cluster persistence measured 0.00–0.16 s** on this footage. That is V0.3.5's
exact-member-set rule behaving as specified — it measures exact composition stability,
not spatial crowd-region continuity. The V0.3.5 algorithm was **not changed**; whether
a separate neutral region-continuity primitive is warranted is a bounded review item to
settle *after* ground-truth review, and nothing has been implemented for it.
