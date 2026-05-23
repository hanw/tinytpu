# Tinyspec / TinyTPU Implementation Sync Plan

## Status

Draft implementation plan.

This plan is about syncing `tinygrad/spec/tinyspec.tex` with the current
TinyTPU backend implementation. It is not a plan to make TinyTPU implement the
entire tinygrad UOp surface immediately, and it is not a plan to turn
Tinyspec into the TinyTPU ISA manual.

The desired outcome is:

- `tinygrad/spec/tinyspec.tex` remains the source for core tinygrad tensor
  semantics.
- The spec gains a clear TinyTPU backend profile describing which Tinyspec
  semantics the current TinyTPU implementation supports, how they lower, and
  which gaps are deliberate.
- `doc/tinyspec_coverage.md` and `TODO.md` derive their status from that
  profile instead of drifting as separate truth.

## Current Implementation Snapshot

The TinyTPU backend entry point is `TinyTPURenderer` in
`tinygrad/tinygrad/runtime/ops_tinytpu.py`.

Current renderer dispatch:

- `KernelClass.ELEMENTWISE` -> `lower_kernel`
- `KernelClass.REDUCTION` -> `lower_reduction`
- `KernelClass.BROADCAST` -> `lower_broadcast`
- `KernelClass.MOVEMENT` -> `lower_movement`
- `KernelClass.GEMM` -> `lower_gemm`
- non-WMMA matmul fallback -> `lower_gemm_fallback`
- otherwise -> `UNSUPPORTED` descriptor, raised as `NotImplementedError`

Current TinyTPU lowerer modules:

- `tinygrad/tinygrad/renderer/tinytpu/elementwise.py`
- `tinygrad/tinygrad/renderer/tinytpu/reduction.py`
- `tinygrad/tinygrad/renderer/tinytpu/broadcast.py`
- `tinygrad/tinygrad/renderer/tinytpu/movement.py`
- `tinygrad/tinygrad/renderer/tinytpu/gemm.py`
- `tinygrad/tinygrad/renderer/tinytpu/classify.py`
- `tinygrad/tinygrad/renderer/tinytpu/common.py`

Current hardware-facing surfaces:

- VPU op table in `ops_tinytpu.py` includes integer, float, reduction,
  transcendental, packed-int8, bit utility, saturating, rotation, sign, abs
  diff, unsigned min/max, and pair-rotate opcodes.
- SXU op table includes VREG load/store, VPU dispatch, select, scalar/row/col
  broadcast, XLU transpose, MXU dispatch/result, PSUM, predicate skips,
  epilogue, and requant dispatch.
- `code_for_op` declares direct TinyTPU support for `EXP2`, `LOG2`, `SIN`,
  and `SQRT` so tinygrad does not decompose them before rendering.
- `tensor_cores` declares a 4x4x4 integer tensor-core path.

## Mismatch Classes

### 1. Tinyspec Omits TinyTPU Backend Profile

`tinygrad/spec/tinyspec.tex` describes core UOp semantics and general lowering
concepts, but it does not say which subset a concrete backend may implement or
how TinyTPU maps those semantics onto its tile ISA.

This causes two practical problems:

- A reader cannot tell whether a Tinyspec op is expected to run on TinyTPU
  today.
- Coverage documents duplicate support status and become stale.

### 2. TinyTPU Implements Hardware Primitives Not Named In Tinyspec

TinyTPU has backend and ISA primitives that are implementation details relative
to core tensor semantics:

- scalar, row, and column broadcast instructions
- XLU transpose
- VPU select
- tile, row, and column reduction opcodes
- PSUM operations
- MXU epilogue and requant instructions
- packed int8 helper ops
- bit utility ops such as CLZ, CTZ, POPCOUNT, byte reverse, rotate
- ARGMIN / ARGMAX and pair rotate

The spec should not promote every one of these to a core tensor op. It should
document them in the TinyTPU profile as backend lowering capabilities.

### 3. Tinyspec Includes Semantics TinyTPU Does Not Fully Implement

Examples:

- general movement semantics: arbitrary `Permute`, `Flip`, `Pad`, `Shrink`,
  `Expand`, and `Index`
- multi-device semantics: `Copy` to tuple devices, replicated axes, collectives
- general control flow and program ops: `Barrier`, `Special`, `If`, `Endif`,
  custom code, program/source/binary metadata
- atomics
- dtype-complete arithmetic across all integer and float widths
- general `Reduce(Mul, axes)` for float until all float product reducer paths
  are wired end to end

The spec sync should mark these as unsupported or partial in the TinyTPU
profile, with precise reasons.

## Target Spec Shape

Add a TinyTPU backend profile section to `tinygrad/spec/tinyspec.tex` after
the general lowering pipeline section.

Recommended structure:

```tex
\subsection*{TinyTPU Backend Profile}

\paragraph{Scope}
...

\paragraph{Supported Lowering Classes}
...

\paragraph{DType and Tile Model}
...

\paragraph{Supported Tinyspec Semantics}
...

\paragraph{Backend Primitives Used For Lowering}
...

\paragraph{Unsupported / Partial Semantics}
...
```

The profile should explicitly distinguish:

- **Core Tinyspec op**: semantic operation visible in the UOp graph.
- **TinyTPU primitive**: hardware or SXU/VPU/XLU/MXU instruction used to
  implement one or more core ops.
- **Lowering class**: renderer-owned category selected by `classify()`.

## Phase 1: Mechanical Inventory

Goal: produce an implementation-owned inventory that can be copied into the
spec without guessing.

Tasks:

- Extract the renderer classes from `classify.py`: `ELEMENTWISE`, `REDUCTION`,
  `BROADCAST`, `MOVEMENT`, `GEMM`, `UNSUPPORTED`.
- Extract the supported VPU op table from `_VPU_OPS` in
  `ops_tinytpu.py`.
- Extract the supported SXU op table from `_SXU_OPS` in
  `ops_tinytpu.py`.
- Extract `code_for_op` declarations from `TinyTPURenderer`.
- Extract tensor-core declaration from `TinyTPURenderer.tensor_cores`.
- Extract test-backed capability groups from `tests/test_tinytpu_backend.py`.
- Replace stale high-level statements in `doc/tinyspec_coverage.md` with a
  generated or manually audited table that matches the inventory.

Acceptance:

- `doc/tinyspec_coverage.md` no longer says current false statements such as
  "multi-tile elementwise open" if the current implementation supports that
  class.
- Every row in the coverage table links to a lowerer module or test group.

## Phase 2: Add TinyTPU Profile To Tinyspec

Goal: update `tinygrad/spec/tinyspec.tex` to describe what the current TinyTPU
implementation supports.

Add a TinyTPU backend profile with these tables.

### Supported Lowering Classes

| Class | Implementation | Tinyspec semantics covered |
| --- | --- | --- |
| Elementwise | `elementwise.py` | per-element ALU, `Where`, degenerate copy, const fill, supported casts, scalar broadcast |
| Reduction | `reduction.py` | `Reduce(Add/Max/Mul)` for supported integer/float row, col, and tile shapes |
| Broadcast | `broadcast.py` | row/column broadcast patterns and selected broadcast+where patterns |
| Movement | `movement.py` | supported pad/flip/non-affine single-tile scatter, 4x4 transpose, row-broadcast copy |
| GEMM | `gemm.py` | int8 MXU-backed 4x4 tiled matmul with int32 output and supported epilogues |

### DType And Shape Profile

Document the backend constraints:

- VMEM tile is 4x4 int32 lanes.
- TinyTPU elementwise stores bool as 0/1 values.
- Float32 is carried through int32 lane bits for supported VPU float ops.
- MXU input operands are int8, with int32 accumulation/output.
- Tensor-core path is 4x4x4.
- Some lowerers are multi-tile; some movement paths are still single-tile.

### Tinyspec Op Coverage

Document each core category:

- Source/storage ops: `Buffer`, `BufferView`, `Param`, `Const`, `Binary`.
- Movement ops: which movement forms are renderer-lowered today and which
  remain unsupported.
- Elementwise ops: supported integer, bool, float, transcendental, and
  decomposed forms.
- Reduce ops: supported reducer op/axis/dtype matrix.
- Call/function/program ops: which are handled by tinygrad before TinyTPU and
  which are not a TinyTPU runtime feature.
- Multi-device ops: current TinyTPU status.
- Codegen/control ops: current TinyTPU status.

Acceptance:

- A reader can answer "should this Tinyspec program run on TinyTPU today?"
  without reading renderer code.
- The spec does not imply TinyTPU supports all Tinyspec semantics.

## Phase 3: Reconcile Spec Vocabulary With TinyTPU Names

Goal: avoid confusing hardware instruction names with core op names.

Tasks:

- Use Tinyspec names for user-visible semantics: `Where`, `Reduce`, `Permute`,
  `Expand`, `Cast`, `Bitcast`, `Wmma`, etc.
- Use TinyTPU names only in the backend profile: `SXU_BROADCAST_ROW`,
  `VPU_FSUM_REDUCE_TILE`, `SXU_DISPATCH_XLU_TRANSPOSE`, `MXU`, `PSUM`.
- Add a small mapping table:

| Tinyspec semantic | TinyTPU lowering primitive |
| --- | --- |
| `Where` | `SXU_DISPATCH_SELECT` / `VPU_SELECT` |
| scalar expand | `SXU_BROADCAST_SCALAR` |
| row/col broadcast | `SXU_BROADCAST_ROW` / `SXU_BROADCAST_COL` |
| transpose 4x4 tile | `SXU_DISPATCH_XLU_TRANSPOSE` |
| `Reduce(Add/Max/Mul)` | VPU row/col/tile reduce opcodes |
| `Wmma`/matmul | MXU dispatch plus optional PSUM/epilogue/requant |

Acceptance:

- The core spec remains backend-neutral.
- The TinyTPU section is precise enough for implementation planning and test
  coverage.

## Phase 4: Unsupported Semantics Manifest

Goal: make unsupported areas explicit and testable.

Tasks:

- Add or update an unsupported manifest under `tests/` or `doc/` listing
  Tinyspec categories that TinyTPU intentionally rejects today.
- For each unsupported category, record:
  - Tinyspec op/category
  - TinyTPU reason
  - expected error path or skip/xfail test
  - likely implementation owner
- Convert ambiguous unsupported behavior into either:
  - a positive lowerer test, or
  - a clear `NotImplementedError` diagnostic test.

Initial unsupported buckets:

- arbitrary gather/scatter indexing
- general multi-device `Copy` and collectives
- atomics
- generic custom code injection
- general control flow beyond existing predicate skip primitives
- unsupported dtype widths and mixed precision
- movement shapes outside current lowerer constraints
- GEMM shapes/dtypes outside current int8->int32 tiled path

Acceptance:

- Unsupported behavior is deliberate and discoverable.
- `TODO.md` links unsupported work items to Tinyspec categories.

## Phase 5: Keep Spec And Implementation In Sync

Goal: prevent the spec from drifting again.

Tasks:

- Add a small script, for example `scripts/check_tinyspec_tinytpu_profile.py`,
  that reads `_VPU_OPS`, `_SXU_OPS`, and `KernelClass`, then checks that each
  appears in the TinyTPU profile table.
- Add a lightweight test that fails when a new TinyTPU lowering class or opcode
  is added without updating the profile or an explicit internal-only allowlist.
- Update `tinygrad/spec/tinyspec.pdf` whenever `tinyspec.tex` changes.
- Require every TinyTPU capability iteration to update one of:
  - Tinyspec TinyTPU profile
  - unsupported manifest
  - `doc/tinyspec_coverage.md`

Acceptance:

- Adding a new backend primitive creates an obvious documentation checkpoint.
- `doc/tinyspec_coverage.md` becomes a summary, not an independent source of
  truth.

## Patch Order

Recommended commit sequence:

1. Refresh `doc/tinyspec_coverage.md` from current TinyTPU lowerers and tests.
2. Add the TinyTPU backend profile to `tinygrad/spec/tinyspec.tex`.
3. Regenerate `tinygrad/spec/tinyspec.pdf`.
4. Add unsupported manifest entries for the largest known gaps.
5. Add the profile-sync checker.
6. Wire the checker into the local test command or CI-equivalent path.

## Non-Goals

- Do not delete core Tinyspec semantics just because TinyTPU does not support
  them yet.
- Do not rename TinyTPU hardware opcodes to match Tinyspec op names.
- Do not claim full TinyTPU support for a Tinyspec category unless there is a
  renderer path and test coverage.
- Do not put BSV microarchitecture details in the core spec; keep those in the
  TinyTPU architecture and ISA docs.

## Done Criteria

The sync is done when:

- `tinygrad/spec/tinyspec.tex` contains a TinyTPU backend profile that matches
  `TinyTPURenderer`, `_VPU_OPS`, `_SXU_OPS`, and the lowerer package.
- `doc/tinyspec_coverage.md` agrees with the profile.
- Unsupported TinyTPU Tinyspec areas are listed with explicit reasons.
- A checker prevents new TinyTPU opcodes or lowering classes from silently
  bypassing the profile.
- The rendered PDF is regenerated from the updated TeX source.
