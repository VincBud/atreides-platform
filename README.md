# Atreides — V0.3.2: CALIBRATION PIPELINE (in progress)

Atreides is an AI-assisted venue security intelligence platform. It analyses existing
CCTV/video streams and gives human security operators situational awareness.

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

> **Read the V0.2.1 limitations before trusting any count.** In particular, a
> unique track ID is *not* a unique person: same-person ID multiplicity means the
> unique-ID count is an upper bound, not a census.

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

These are product and privacy boundaries, not missing features:

- **It does not identify people.** A track ID is a handle on a box moving through
  one video. It is not a person, a name, or a record.
- **It does not perform face recognition** — or any biometric recognition.
- **It does not infer aggression**, intent, emotion, or intoxication. V0.3
  *measures* movement — speed, acceleration, turn rate, time spent still — and
  stops there. A measurement is not a judgement, and none of these signals is
  labelled, scored or ranked.
- **It does not classify demographics.**
- **It does not make security decisions.** It produces a view for a human operator.
- **It does not perform cross-camera re-identification.** Track IDs are local to one
  camera and one session, and mean nothing outside it.
- **It does not send video to a cloud service.** Raw video stays on your machine.
- **It does not persist tracks.** There is no database. State lives in memory for the
  duration of a run and is discarded when the process exits.

Zone analytics operate only on anonymous tracking geometry: boxes, positions and
elapsed time. The metrics stream contains zone counts and durations, never track
identifiers or positions.

There is also no dashboard, API, alerting, risk scoring, or anomaly detection in
these milestones. **`motion_level` is not a risk score** — see below.

> The one network call in the whole project is the first-run download of the RF-DETR
> model weights. Your video is never uploaded.

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

# V0.3 — with behaviour signals (camera-local, normalized units)
uv run atreides-vision run \
  --input data/input/test.mp4 \
  --output artifacts/tracked.mp4 \
  --behavior-output artifacts/behavior.jsonl \
  --camera-id cam_01

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
| `--trail-length` | `25` | Centre points kept and drawn per track |
| `--no-confidence-labels` | off | Draw `ID <n>` without the score |
| `--log-every` | `100` | Progress log interval, in frames |
| `--verbose` | off | Debug logging |

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

## Architecture

```text
atreides-platform/
├── config/zones.example.json    # zone configuration to copy and edit
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
│   │   │   ├── calibration/         # V0.3.1 ground-plane calibration
│   │   │   │   ├── homography.py        # normalized DLT — pure array maths
│   │   │   │   ├── models.py            # VenuePoint, correspondences, quality
│   │   │   │   ├── fitting.py           # solve + measure + safeguard
│   │   │   │   ├── projector.py         # GroundPlaneProjector (a MotionProjector)
│   │   │   │   └── loader.py            # JSON loading and validation
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
behaviour signals. **V0.3.0 (the signal foundation) and V0.3.1 (ground-plane
calibration) are both built.**

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
two distinct subjects, and nothing associates them. A camera-local track ID is still
not a unique venue person: that is limitation (2) from the V0.2.1 report, unfixed
and not pretended otherwise.

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

Full detail and the complete limitation lists:
[V0.3.0](docs/vision/V0.3_BEHAVIOR_SIGNALS.md) ·
[V0.3.1](docs/vision/V0.3.1_GROUND_PLANE_CALIBRATION.md) ·
[V0.3.2](docs/vision/V0.3.2_CALIBRATION_PIPELINE.md).

Identity, biometrics, demographics, cross-camera matching, risk scoring and
behaviour *classification* remain out of scope by design.
