# Atreides — V0.1: SEE + TRACK

Atreides is an AI-assisted venue security intelligence platform. It analyses existing
CCTV/video streams and gives human security operators situational awareness.

**This repository contains the first technical milestone only.** It is deliberately
narrow: a local video-processing pipeline that finds people in a video and follows
them from frame to frame.

## What it does

```text
MP4  →  person detection  →  multi-object tracking  →  annotated MP4
```

The annotated output shows, per frame:

- person bounding boxes
- persistent track IDs (`ID 7`)
- the detection confidence
- a short trajectory trail from each track's recent centre points
- a top-left status panel:

```text
ATREIDES // SEE + TRACK
Frame: 1240
Processing: 6.4 FPS
Tracks: 18
```

Everything runs locally on a normal development machine. No NVIDIA hardware is
required — CPU works, Apple Silicon (MPS) is used automatically when available.

## What it intentionally does not do

These are product and privacy boundaries, not missing features:

- **It does not identify people.** A track ID is a handle on a box moving through
  one video. It is not a person, a name, or a record.
- **It does not perform face recognition** — or any biometric recognition.
- **It does not infer aggression**, intent, emotion, or behaviour.
- **It does not make security decisions.** It produces a view for a human operator.
- **It does not perform cross-camera re-identification.** Track IDs are local to one
  camera and one session, and mean nothing outside it.
- **It does not send video to a cloud service.** Raw video stays on your machine.
- **It does not persist tracks.** There is no database. State lives in memory for the
  duration of a run and is discarded when the process exits.

There is also no dashboard, API, alerting, risk scoring, or zone analysis in this
milestone.

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
uv run atreides-vision run \
  --input data/input/test.mp4 \
  --output artifacts/tracked.mp4
```

The first run downloads the RF-DETR Nano weights (~349 MB) to `~/.roboflow/models/`.
Every run after that is fully offline.

### Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--input` | required | Source video path |
| `--output` | required | Annotated MP4 destination |
| `--confidence` | `0.5` | Minimum detection confidence, in `(0, 1]` |
| `--device` | `auto` | `auto`, `cpu`, `mps`, or `cuda` |
| `--max-frames` | all | Stop after N frames — useful for a quick look at a long video |
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

## Test

```bash
uv run pytest          # unit tests — no model weights, no network, no GPU
uv run ruff check .    # lint
uv run ruff format .   # format
uv run mypy            # strict type check
```

The test suite runs in a couple of seconds and never downloads anything. It covers
the domain models, configuration and device policy, video metadata and sink
lifecycle, trajectory cleanup, both ML adapters (via fakes), the overlay, the CLI,
and the full pipeline end-to-end with fake detector and tracker implementations.

## Architecture

```text
atreides-platform/
├── apps/operator-dashboard/     # placeholder — not this milestone
├── packages/contracts/          # placeholder — shared contracts, later
├── services/
│   ├── vision/                  # ← everything in V0.1 lives here
│   │   ├── src/atreides_vision/
│   │   │   ├── models.py            # Detection, Track, BoundingBox, Point
│   │   │   ├── config.py            # config objects + device resolution
│   │   │   ├── errors.py            # operator-readable error hierarchy
│   │   │   ├── cli.py               # composition root
│   │   │   ├── detection/           # PersonDetector interface + RF-DETR adapter
│   │   │   ├── tracking/            # MultiObjectTracker interface + ByteTrack adapter
│   │   │   ├── video/               # source, sink, trajectories, processor
│   │   │   └── visualization/       # overlay drawing
│   │   └── tests/
│   ├── intelligence/            # placeholder
│   └── api/                     # placeholder
├── data/input/                  # your footage (git-ignored)
└── artifacts/                   # rendered output (git-ignored)
```

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

- **Trajectory state is bounded on both axes.** Each track keeps a fixed-length ring
  of recent centre points; trails for tracks that have not been seen for `ttl_frames`
  are dropped, and a hard ceiling evicts the least-recently-seen trail. A long run
  cannot grow memory without limit.
- **No hidden global state.** Configuration is constructed in `cli.py`, validated at
  construction, and passed down explicitly.

## Dependencies

| Package | Why |
| --- | --- |
| `rfdetr` | RF-DETR Nano, the initial person detector |
| `trackers` | ByteTrack, the initial multi-object tracker |
| `opencv-python` | Video decode/encode and all drawing |
| `numpy` | Array interchange between the layers |

Dev: `pytest`, `ruff`, `mypy`. Deliberately absent: Ultralytics, cloud SDKs, database
libraries, web frameworks, Docker.

## Next milestone

V0.2 would most usefully add zone definition and dwell/occupancy counts on top of
tracks — the first step from "seeing" to "understanding" — plus detector throughput
work. Identity, biometrics, and cross-camera matching remain out of scope by design.
