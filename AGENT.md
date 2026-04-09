# Agent Workflow

## Goal

Expand TinyTPU implementation coverage autonomously.

The default mission is to make the system handle more real workloads through the smallest defensible changes across:

- BSV hardware in `src/`
- BSV tests in `test/`
- tinygrad TinyTPU backend in `tinygrad/tinygrad/runtime/ops_tinytpu.py`
- supporting scripts and tests in `scripts/` and `tests/`

The agent should prefer changes that improve supported functionality, correctness, and debuggability without adding unnecessary complexity.

## Read First

Before starting a new implementation push, read the repo context:

1. `README.md`
2. The directly relevant design/spec doc under `doc/`
3. The source and test files in the area being changed
4. `tinygrad/tinygrad/runtime/ops_tinytpu.py` when the task touches lowering or co-sim

Do not make blind edits. Build context first, then choose the smallest useful slice.

## Primary Loop

Loop continuously until interrupted:

1. Check git state and confirm the current branch/commit.
2. Choose one bounded implementation gap.
3. Reproduce the gap with the smallest relevant test.
4. Implement the minimum coherent fix.
5. Run the narrowest useful verification first, then broader verification if warranted.
6. Keep the change only if it advances functionality or simplifies the code without regression.
7. Commit the improvement.
8. Repeat from the new head.

The loop is intended to keep expanding real capability, not just churn code.

## How To Choose Work

Preferred order:

1. A failing or missing end-to-end TinyTPU capability
2. A lowering/runtime gap in the tinygrad `TINYTPU` backend
3. A missing hardware instruction/data-path needed by software
4. Missing tests for behavior the repo claims to support
5. Simplifications that preserve behavior and reduce code or concepts

Good task shapes:

- Add one missing SXU/VPU/XLU/backend capability end to end
- Turn a current `NotImplementedError` path into working behavior
- Add a missing testbench case and implement the minimal hardware/software support
- Fix a real mismatch between the software contract and the BSV implementation

Avoid:

- Large speculative rewrites without a failing test or concrete target
- Changing multiple subsystems at once unless the boundary requires it
- Complexity that is not justified by clear capability gain

## ONNX -> tinygrad -> TinyTPU Triage

For model bring-up and backend expansion, use this rule:

1. Start from the failing ONNX model or tinygrad workload.
2. Compile through tinygrad targeting `TINYTPU`.
3. Inspect where execution fails in the TinyTPU renderer/runtime path.
4. First assume it is a software/lowering problem.
5. Escalate to a TinyTPU hardware issue only when the required behavior cannot be expressed with the current instruction set or data paths.

Expected output for each investigated workload:

- What tinygrad produced
- What the TinyTPU backend could and could not lower
- The exact missing instruction sequence, lowering rule, or hardware capability
- The chosen fix location: tinygrad software or TinyTPU hardware

## Implementation Rules

- Prefer test-driven expansion: add or tighten a failing test before the fix when practical.
- Keep edits local to the subsystem you are advancing.
- Preserve existing architectural boundaries unless there is a concrete reason to change them.
- Favor simple instruction sequences and explicit data movement over clever abstractions.
- If a tinygrad-side workaround can express the behavior cleanly with the current ISA, do that before inventing new hardware.
- If hardware is required, make the missing contract explicit in code and tests.

## Verification Rules

Use the narrowest command that proves the change:

- `make test-<unit>` for unit-level BSV work
- `make test` when cross-cutting changes justify the cost
- `python3 scripts/test_cosim.py` for end-to-end tinygrad co-sim
- `pytest tests/...` for Python-side tooling or profiler work

When fixing a bug:

1. Reproduce it with a targeted test.
2. Make the test pass.
3. Run adjacent tests that could plausibly regress.

If a change cannot be verified locally, state exactly what remains unverified and why.

## Keep / Discard Standard

Keep a change if it does at least one of these:

- Enables a new supported behavior
- Fixes an incorrect result
- Removes code while preserving behavior
- Improves debuggability or diagnostics for a real failure mode

Discard or rework a change if it:

- Adds complexity with negligible functional gain
- Leaves behavior ambiguous or untested
- Fixes one path by hard-coding around the design
- Regresses a nearby unit or end-to-end flow

## Commit Standard

Commit only coherent advances. Each commit should describe one implementation step, for example:

- `xlu: add transpose path for runtime bundles`
- `tinytpu: lower elementwise add through vpu sequence`
- `tensorcore: fix mxu completion handshake`

Do not mix unrelated cleanups into an implementation commit.

## Expected Output

For each completed loop iteration, leave the repo in a state where a human can see:

- What gap was targeted
- What test or workload reproduced it
- What was changed
- What verification passed
- What limitations remain, if any
