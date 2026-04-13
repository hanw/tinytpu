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
7. Update `TODO.md` when the iteration changes tinyspec coverage, milestone progress, known gaps, or recommended next work.
8. Update `results.tsv` with one tab-separated progress row for the completed iteration or iteration batch.
9. Commit the improvement.
10. Repeat from the new head.

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

After any successful model or workload run, explicitly check ISA support sufficiency:

- Did the workload run end to end on `TINYTPU` without host fallback?
- Did lowering use direct primitives or short regular SXU programs instead of large analyzer special-cases?
- Did any unsupported path, `UNSUPPORTED` descriptor, or simulator-side software convention appear?
- Did the generated bundle count and instruction mix look structurally reasonable for the workload?

If the answer to any of these is "no", record the exact missing primitive, lowering gap, or hardware contract in `TODO.md`. Do not treat "model ran" as equivalent to "ISA support is sufficient".

## Implementation Rules

- Prefer test-driven expansion: add or tighten a failing test before the fix when practical.
- Keep edits local to the subsystem you are advancing.
- Preserve existing architectural boundaries unless there is a concrete reason to change them.
- Favor simple instruction sequences and explicit data movement over clever abstractions.
- If a tinygrad-side workaround can express the behavior cleanly with the current ISA, do that before inventing new hardware.
- If hardware is required, make the missing contract explicit in code and tests.

## UOp-to-SXU Compilation Strategy

The target architecture for `ops_tinytpu.py` is a **UOp-walking renderer** that
emits SXU instructions directly from the tinygrad UOp graph — one SXU/VPU
instruction per UOp, like how CStyleLanguage emits one line of C per UOp.

When a UOp has no corresponding SXU/VPU instruction, **add a hardware
primitive** rather than writing complex UOp-graph pattern matching in
software. Prefer a new VPU opcode or SXU instruction over a multi-hundred-
line graph walker that reverse-engineers tinygrad's decomposition.

Concrete priority order:
1. **Add a hardware primitive** (new VPU opcode, new SXU instruction) that
   maps 1:1 to the tinygrad UOp. This is always preferred.
2. **Emit a short SXU microprogram** (2–4 existing instructions) when the
   pattern is simple and stable (e.g. WHERE via COPY+SELECT).
3. **Use pattern matching as a last resort** only when the tinygrad
   decomposition is too complex for a single hardware primitive and no
   short microprogram exists. Document why and plan the hardware fix.

Examples of hardware-first decisions:
- `VPU_SELECT` replaced a 4-instruction MUL/SUB/MUL/ADD WHERE sequence
- `VPU_NOT` replaced XOR-with-all-ones constant tile loading
- `VPU_MIN` is preferred over detecting XOR+MAX decomposition in software
- Remaining `analyze_tinytpu_uops` patterns (scalar-const DIV truncation,
  MOD via DIV+MUL+SUB) should be resolved by adding `VPU_TRUNC_DIV` and
  `VPU_MOD` hardware opcodes, not by more graph walking

Propose ISA additions in `TODO.md` and get user approval before implementing
(per the microarchitecture rule below). The goal is a 1:1 UOp→instruction
mapping that eliminates the `analyze_tinytpu_uops` pattern-matching layer.

## Unsupported Feature Triage

When a workload hits an unsupported op, dtype, or shape:

1. Record it in `TODO.md` under the appropriate coverage area with a concrete description.
2. Decide whether the fix belongs in **software** (tinygrad backend lowering) or **hardware** (BSV):
   - **Software first**: if the behavior can be expressed with existing SXU/VPU/MXU/XLU instructions, lower it in `ops_tinytpu.py`. Examples: new elementwise op via existing VPU opcodes, host fallback for unsupported dtypes, shape remapping.
   - **Hardware needed**: if no existing instruction sequence can express the behavior, or if a software workaround would be unreasonably slow. Examples: new VPU opcode, new SXU instruction, wider data paths. In this case, note the required BSV change in `TODO.md` and **ask the user before implementing** — do not expand the microarchitecture (new opcodes, new functional units, wider data paths, new SRAMs) without explicit approval.
3. If the unsupported feature blocks a real model (not just a synthetic test), prioritize it higher.
4. Do not silently skip unsupported features — always emit a clear diagnostic via the `UNSUPPORTED` descriptor path so the gap is visible.

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

When updating `TODO.md`, keep the edit scoped to the completed iteration:

- Mark newly supported behavior as complete.
- Add newly discovered gaps or limitations.
- Adjust recommended next iterations if the priority changed.
- Do not rewrite unrelated estimates or checklist sections unless the iteration made them obsolete.

When updating `results.tsv`, append one row per completed iteration or batch. Keep the file tab-separated with this schema:

```text
date	commit	iterations	scope	supported_delta	tests_passed	todo_delta	remaining_gap
```

Rules for `results.tsv`:

- Use ISO dates.
- Use the short commit hash after the commit is created.
- Keep fields concise and avoid tabs inside field values.
- Set `iterations` to the number of loop iterations represented by the row.
- Use `supported_delta` for newly supported behavior or `none` for docs-only/tooling-only work.
- Use `tests_passed` for the highest-signal verification command and result.
- Use `todo_delta` to summarize TODO progress changed by the iteration.
- Use `remaining_gap` for the next concrete blocker surfaced by the work.

## Expected Output

For each completed loop iteration, leave the repo in a state where a human can see:

- What gap was targeted
- What test or workload reproduced it
- What was changed
- What verification passed
- What `TODO.md` progress was updated, if applicable
- What `results.tsv` progress row was appended
- What limitations remain, if any
