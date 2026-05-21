# TinyTPU InstSel — Structural Slice Design

**Document status:** Draft v1
**Scope:** migrate the 12 remaining structural `_render_*_sxu_program` recognizers
in `ops_tinytpu.py` into the InstSel lowering layer
**Prerequisite:** the elementwise/transcendental/activation slice
(`doc/plan-tinytpu-instsel.md`) is complete — 8 iterations, 37 recognizers
deleted, `ops_tinytpu.py` 6984 → 3416 lines.

## 1. Problem Statement

The first InstSel slice replaced the elementwise/transcendental/activation
recognizer waterfall with a UOp-walking lowerer (`tinytpu_lowering.py`). Twelve
structural recognizers remain in `ops_tinytpu.py`, ~1,500 lines across five
categories:

| Category   | Recognizers                          | Lines |
|------------|--------------------------------------|-------|
| Reductions | reduction, colreduce, rowreduce      | 572   |
| Broadcasts | rowbc, colbc, colbc_where            | 299   |
| Trivial    | cast, copy, const_fill               | 320   |
| Movement   | pad, transpose                       | 208   |
| GEMM       | gemm_fallback + WMMA dispatch path   | 122+  |

These are not all the same problem. Reductions and broadcasts are genuine
pattern-archaeology (250-line op-counting, axis classification reverse-engineered
from index math). Trivial kernels are degenerate elementwise. GEMM is legitimate
instruction selection — `WMMA` is a real codegen op. Movement ops, per the
project's own analysis, should be consumed by Rangeify upstream.

The goal: `ops_tinytpu.py` ends with zero `_render_*_sxu_program` recognizers;
all kernel lowering lives behind one positive classifier.

## 2. Goals

1. Replace the recognizer dispatch chain with a positive, exhaustive
   `classify(uops)` and per-class lowerers.
2. Consolidate reductions and broadcasts into structured lowerers.
3. Relocate GEMM lowering into the lowering layer unchanged in behavior.
4. Resolve movement ops at the correct layer (upstream if possible).
5. Leave `ops_tinytpu.py` as allocator/runtime/device glue plus a `lower()` call.

## 3. Non-Goals

- Rewriting the GEMM tiling/PSUM/epilogue logic — it is correct and well tested.
- Changing the `SXU_PROGRAM` descriptor or `TinyTPUProgram` runtime.
- Fixing the broken `EXP2`/`LOG2`/`SIN` hardware (separate; see
  `doc/plan-primitive-ops-handoff.md`).
- Removing `analyze_tinytpu_uops` — it has an external consumer
  (`tests/onnx_tinytpu_trace/driver.py`).

## 4. Decisions (fixed)

Settled during design:

- **Scope:** all five categories.
- **Architecture:** classifier + per-class lowerers (not one mega-walker, not a
  graph-rewrite-to-IR).
- **Movement:** investigate a Rangeify-upstream fix before writing any
  renderer-side movement lowerer.
- **GEMM:** relocate and clean up — keep the WMMA logic, restructure it behind
  the classifier; do not rewrite.
- **No fallback.** `classify` is exhaustive; an unclassifiable kernel yields the
  `UNSUPPORTED` descriptor. Refactored code deletes dead code.

## 5. Architecture

`render()` calls `classify(uops) -> KernelClass`, then dispatches to the matching
lowerer. Every lowerer returns the existing `SXU_PROGRAM` JSON descriptor, so the
runtime is untouched.

`classify` is positive and exhaustive:

- `WMMA` UOp present → `GEMM`
- output smaller than an associative-reduce-tree input → `REDUCTION`
- output larger than input via structured replication → `BROADCAST`
- pure elementwise (the current `can_lower` set) → `ELEMENTWISE`
- otherwise → `UNSUPPORTED` (clear diagnostic, never silent)

`classify` subsumes `can_lower` — `ELEMENTWISE` is one of its outcomes. There is
no try-chain and no fallthrough between classes.

## 6. Module Layout — the `tinytpu_lowering` Package

`runtime/support/tinytpu_lowering.py` (350 lines today, ~1,000 after this slice)
becomes a package `runtime/support/tinytpu_lowering/`:

- `__init__.py` — `classify()`, the `lower(uops)` dispatch entry, public exports.
- `common.py` — `TpuInst`, `TpuKernel`, instruction encoders, the data-DAG walk
  (`_canon`/`_data_dag`), linear-scan register allocation, the InstSel
  `PatternMatcher`, shared graph helpers.
- `elementwise.py` — the current walker, moved unchanged.
- `reduction.py` — the reduction lowerer.
- `broadcast.py` — the broadcast lowerer.
- `gemm.py` — the GEMM lowerer, relocated from `ops_tinytpu.py`.
- `movement.py` — added only if the §9 spike shows movement must stay
  renderer-side.

`ops_tinytpu.py` imports `lower` and `classify` from the package. Trivial kernels
(cast, copy, const-fill) fold into `elementwise.py`: a cast is a per-element op
(`I2F`/`F2I`/transparent bool→int), a copy is identity (`LOAD`→`STORE`), a
const-fill is nullary (`CONST`→`STORE`). They extend the elementwise walker's
accepted shapes; they are not new lowerers.

## 7. Reduction Lowerer

`reduction.py` consolidates `_render_reduction`, `_render_rowreduce`, and
`_render_colreduce` (572 lines) into one structured lowerer (~200 lines).

A reduction kernel has an input of size S and an output of size O < S whose
stored value is an associative-op tree over loads of the input. The lowerer:

- Classifies the reduce op: `ADD`→`SUM`, `MAX`→`MAX`, `MUL`→`PROD`,
  `MAX`+`XOR`→`MIN` (integer min decomposition), float-min-via-negation→`FMIN`.
- Classifies the axis: scalar (`O == 1`), row, or column.
- Emits `VPU_*_REDUCE_TILE` / `VPU_*_REDUCE_COL` per tile, padding partial tiles
  with the reduction identity (`0` / `-inf` / `+inf` / `1.0`).
- Combines multi-tile partials with the matching combine op.
- Folds a post-reduction scalar op (`reduce(x) + c`, `reduce(x) * c`).

The op-counting in the legacy recognizers becomes a single classify-then-emit
pass with no `Counter`-based archaeology.

## 8. Broadcast Lowerer

`broadcast.py` consolidates `_render_rowbc`, `_render_colbc`, and
`_render_colbc_where`. A broadcast kernel replicates a smaller operand along an
axis into a larger output. The lowerer classifies the replication axis from the
operand's index relationship to the output ranges, then emits
`BROADCAST_SCALAR` / `BROADCAST_ROW` / `BROADCAST_COL`. The `colbc_where` case is
a broadcast feeding a `WHERE`, which the elementwise walker already lowers — the
broadcast lowerer handles the broadcast operand and defers the rest.

## 9. GEMM Lowerer

`gemm.py` relocates the `WMMA` dispatch path (currently in `_render_sxu_program`)
and `_render_gemm_fallback_sxu_program` into a single `lower_gemm()`. The tiling,
PSUM accumulation, and fused bias/ReLU epilogue logic is correct and the
most-tested path in the suite — it is moved and restructured to the lowerer
interface, not rewritten. Verification is the existing GEMM test set, which must
stay green with identical results.

## 10. Movement — Rangeify Spike

Before writing a movement lowerer, a time-boxed spike:

- Dump the UOp graphs for representative `pad` and `transpose` kernels.
- Determine whether `tinygrad/tinygrad/codegen/rangeify.py` already lowers these
  movement ops to index arithmetic for other backends, and whether the TinyTPU
  renderer can consume that form.

Outcomes:

- **Movement consumable upstream** → `_render_pad` and `_render_transpose` delete
  with nothing to replace; the package gains no `movement.py`. This is the
  architecturally correct result — movement ops should not reach the renderer.
- **Movement must stay renderer-side** → add `movement.py` with a pad lowerer
  (scatter into a padded tile) and a transpose lowerer
  (`SXU_DISPATCH_XLU_TRANSPOSE`).

The spike result is recorded in the implementation plan as a decision point;
the plan branches there.

## 11. Migration Order

Each step is one bounded, committed chunk, verified through the sim suite —
the same loop as the first slice's eight iterations.

1. **Package split.** Move `tinytpu_lowering.py` into the package; behavior
   neutral. Verify the suite is unchanged (810 pass / 88 fail).
2. **Classifier.** Introduce `classify()`; `render()` dispatches through it;
   the legacy dispatch chain for structural recognizers remains reachable only
   for not-yet-migrated classes.
3. **Reduction lowerer.** Add `reduction.py`; delete the 3 reduction recognizers.
4. **Broadcast lowerer.** Add `broadcast.py`; delete the 3 broadcast recognizers.
5. **Trivial kernels.** Extend `elementwise.py` for cast/copy/const-fill; delete
   the 3 recognizers.
6. **GEMM relocate.** Move GEMM into `gemm.py`; delete it from `ops_tinytpu.py`.
7. **Movement spike**, then the upstream fix or `movement.py`; delete the 2
   recognizers.

## 12. Verification

Unchanged from the first slice:

- Each step diffs the full `tests/test_tinytpu_backend.py::TestTinyTPUBackend`
  sim run against the captured base set.
- Hardware-exposure failures (broken `EXP2`/`LOG2`/`SIN`) are accepted and
  already documented; no new walker-bug regression is accepted.
- A recognizer is deleted only after its lowerer passes the full sim suite with
  no walker-bug regression.
- Steps 1 and 6 are behavior-neutral and must show an unchanged pass/fail count.
- One commit per step, with `TODO.md` and `results.tsv` updated per `AGENT.md`.

## 13. Risks

- **Package split churn.** Moving the 8-iteration module risks import errors.
  Mitigated: step 1 is behavior-neutral and gated on an unchanged suite.
- **GEMM regression.** GEMM is the most-tested path. Mitigated: relocate, do not
  rewrite; the GEMM suite is the gate.
- **Reduction semantic edge cases.** float-min-via-negation, post-op folding,
  partial-tile identity padding are subtle. Mitigated: the legacy recognizer is
  the behavioral reference; the reduction lowerer must match it on every
  currently-passing reduction test.
- **Movement spike inconclusive.** If the upstream path is unclear, the plan
  falls back to `movement.py` rather than blocking.

## 14. Exit Criteria

- `ops_tinytpu.py` contains no `_render_*_sxu_program` recognizers — only
  allocator, runtime, device glue, and the `lower()` / `classify()` calls.
- All kernel lowering lives in the `tinytpu_lowering` package behind `classify()`.
- `TinyTPUProgram` and the `SXU_PROGRAM` descriptor are unchanged.
- `analyze_tinytpu_uops` remains for its external consumer.
- The backend suite shows no walker-bug regression versus the captured base.
