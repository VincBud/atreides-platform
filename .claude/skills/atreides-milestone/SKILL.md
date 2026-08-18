---
name: atreides-milestone
description: Reusable workflow for implementing a bounded Atreides engineering milestone (V0.x) — scope framing, architectural boundaries, implementation, tests, docs, quality gate and final report. Manual invocation only.
disable-model-invocation: true
---

# Atreides milestone workflow

Run this in the current session. Do not create an Agent Team for it. Delegate only per the
routing policy in `CLAUDE.md` — typically nothing, or `atreides-implementation-engineer`
for a substantial coding scope.

Milestone-specific scope, target version and requirements arrive as: $ARGUMENTS

Work the stages in order. Fill each from `$ARGUMENTS` plus what the repository actually
shows; where `$ARGUMENTS` is silent, state the assumption rather than inventing scope.

## 1. CURRENT STATE

Read before planning: the modules the milestone touches, their tests, the preceding
`docs/vision/` milestone doc, and relevant `config/` manifests. State what exists today,
what it measures, and its known limitations — from the code, not from memory.

## 2. GOAL

One paragraph: what this milestone adds and what becomes true when it is done. Name the
engineering version (`V0.x`) and the V1 capability it serves.

## 3. PRODUCT / SCALABILITY REQUIREMENTS

Confirm against `CLAUDE.md`: no hardcoded camera count, layout, coverage area, overlap
pattern, mounting geometry, resolution or frame rate; single-camera and multi-camera both
run the same path; non-overlapping cameras supported; no physical units without calibration;
degraded calibration/integrity lowers confidence rather than being swallowed.

## 4. ARCHITECTURAL BOUNDARIES

State which layers are touched and which are not. Enforce: cross-camera identity logic never
inside a camera-local tracker; venue-local identity a distinct type and namespace, temporary,
session-scoped, confidence-aware, non-persistent, non-biometric; provenance back to
`(camera, track)` preserved through fusion; no forced merge below the evidence threshold;
signal layers import no model library and no sibling signal layer; measurements are not
renamed into conclusions. If the scope appears to require breaking a boundary, stop and
report it instead of breaking it.

## 5. IMPLEMENTATION

Match the local idiom and the module conventions in `CLAUDE.md`. Stay inside the given
scope — no drive-by refactors, no speculative abstraction, no config nobody sets. Typed
Python throughout; schema changes additive with `schema_version` respected; bounded state
gets a TTL and a ceiling; output ordering deterministic and independent of camera
enumeration order.

## 6. TESTS

`pytest`, no network, no weights, no GPU; fakes for detector and tracker. Cover the new
behaviour, boundary conditions, and a regression guard where existing behaviour changes.
Prefer analytic expectations over snapshotting current output. For association work, test the
failure directions too: a true association found, a genuine non-match refused, an ambiguous
case left unmerged, provenance recoverable after fusion, duplicate counting suppressed, and
identical results across two runs and across camera ordering.

## 7. DOCUMENTATION

One doc per milestone in `docs/vision/`: what was built, the contracts and units it emits,
how it was validated, and — mandatory — its limitations and what remains unproven. Label
every figure as synthetic, proxy/system metric, or human-verified ground truth;
un-annotated material is `NOT_ANNOTATED`, never a zero.

## 8. QUALITY GATE

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy
```

All four must pass on the change. Paste real output; never claim a gate you did not run.
Fix the code, never the gate — the prohibitions in `CLAUDE.md` § Quality gates apply in
full. If a gate looks genuinely wrong, stop and raise it as a human decision.

## 9. SCOPE EXCLUSIONS

List explicitly what this milestone does **not** do: deferred capabilities, V2/V3 work kept
out, anything noticed but deliberately left untouched. An exclusion stated here is not a
defect; an exclusion discovered later is.

## 10. FINAL REPORT

- Files changed and why; tests added.
- Gate results, verbatim.
- Boundary, identity or scalability decisions taken, and their reasoning.
- What was left out of scope, and what remains unvalidated.
- Concerns for the human, including anything that needs approval before it is built.

Do not commit or push. Leave the work in the working tree.

## TOKEN / EXECUTION DISCIPLINE

Keep implementation contexts small and avoid redundant work.

### During implementation
- Search before reading; read only relevant files/ranges.
- Do not repeatedly reread unchanged files.
- Run targeted tests for the code currently being changed.
- Prefer parameterized representative boundary coverage over exhaustive combinatorial test matrices.
- Every new test must protect a distinct behavior, invariant, regression, or boundary.
- More than approximately 80 new tests requires explicit justification before proceeding.
- Do not run the full repository test suite during ordinary implementation iterations.

### Final quality gate
Run the full quality gate once after targeted tests are green:
1. `uv run pytest`
2. `uv run ruff check .`
3. `uv run ruff format --check .`
4. `uv run mypy`

Do not repeat a successful full quality gate merely to demonstrate determinism.

After a failure:
- f- f- f- f- f- f- f- f- f- f- f-tly affected check/tests first,
- repeat the full gate only when the fix could plausibly affect unrelated code or when final verification requires it.

### Context discipline
- Avoid dumping complete large files, diffs, or logs when targeted inspection is sufficient.
- Keep tool output narrow with grep, sed ranges, targeted pytest paths, and concise git commands.
- Do not perform unrelated environment/setup investigations during a milestone.
- Keep the final report concise and avoid repeating the same findings in multiple forms.
