# TinyTPU Instruction Selection Pass — Design

**Document status:** Draft v1
**Scope:** replace the elementwise/activation half of the `_render_*` waterfall in
`ops_tinytpu.py` with a real instruction-selection pass and a UOp-walking renderer
**Goal:** stop reverse-engineering tinygrad's decompositions; emit one TinyTPU
instruction per UOp through a clean compiler seam

## 1. Problem Statement

`tinygrad/tinygrad/runtime/ops_tinytpu.py` (~6,984 lines) has no real renderer.
`TinyTPURenderer.render()` delegates to `_render_sxu_program`, a ~47-deep waterfall
of `_render_<thing>_sxu_program(uops)` functions. Each one takes the whole UOp
list, op-counts it (`Counter(u.op.name for u in uops)`), checks for a high-level
*composite* signature (`tanh`, `sigmoid`, `elu`, `mish`, `swish`, `softplus`,
`logsigmoid`, `relu6`, `hardsigmoid`, …), and emits a hand-unrolled SXU program.

This is pattern *re-recognition*: tinygrad's `tensor.py` already decomposed those
activations into primitives upstream, and the renderer scans the decomposed
primitive graph trying to reconstruct "this is an elu." `_render_elu_sxu_program`
literally matches on the float constant `1.4426950408889634` (`log2(e)`) — the
fingerprint tinygrad injects when it lowers `exp` into `exp2(x·log2(e))`. Any
change to tinygrad's decomposition silently breaks ~20 recognizers.

A healthy renderer has one small pattern per primitive UOp and emits one
instruction per UOp (see `cstyle.py`). `ops_tinytpu.py` instead had ~47 patterns,
one per composite kernel, plus a legacy whole-kernel analyzer and a dozen
`_find_*` helpers — all pattern-sniffing infrastructure. There is no
`PatternMatcher`/`UPat` in the file at all.

## 2. Goals

1. Introduce a real instruction-selection pass for the TinyTPU backend.
2. Emit one TinyTPU instruction per tinygrad UOp (the strategy `AGENT.md`
   already prescribes under "UOp-to-SXU Compilation Strategy").
3. Delete the elementwise/unary/transcendental/activation recognizers — the
   walker replaces them; it does not fall back to them.
4. Establish a typed instruction seam (`TpuInst`) without disturbing the runtime.
5. Begin executing `software-nextgen-spec.md` Phase 3 (compiler-facing lowering).

## 3. Non-Goals

- Migrating GEMM/WMMA, reductions, structured broadcasts, pad, transpose, copy,
  const-fill. They keep their existing recognizers this slice.
- A separate optimizer IR or out-of-repo compiler.
- Changing the `SXU_PROGRAM` JSON descriptor or the runtime (`TinyTPUProgram`).
- Fused-epilogue InstSel (needed only by the out-of-scope structural slices).
- New hardware primitives. The walker targets the current ISA.

## 4. Decisions (fixed)

These were settled during design and are not open for revisiting:

- **Scope:** vertical slice — the elementwise/unary/transcendental/activation
  family only.
- **Hook point:** the pass runs as our own pass invoked from `render()`, not via
  tinygrad's `Renderer.extra_matcher`. Fully self-contained; no dependency on
  tinygrad's rewrite ordering.
- **Code location:** new module
  `tinygrad/tinygrad/runtime/support/tinytpu_lowering.py`.
- **Structure:** syntax-directed walker — a `PatternMatcher` op→emitter dispatch
  plus a normalization pass, a topological walk, and register allocation,
  producing a typed `TpuInst` list encoded to the existing JSON descriptor.
- **No fallbacks.** The walker is terminal for its kernel class. An unknown UOp
  or unsupported form raises and surfaces as the `UNSUPPORTED` descriptor; it is
  never silently rerouted. Refactored code deletes dead code rather than leaving
  it as a fallback.

## 5. Architecture

New module `tinygrad/tinygrad/runtime/support/tinytpu_lowering.py` owns:

- **`TpuInst`** — frozen dataclass, the typed form of one TinyTPU instruction:
  `op: str`, `dst: int`, `srcs: tuple[int, ...]`, `vpu_op: int | None`. This is
  the typed seam `software-nextgen-spec.md §9` asks for.
- **`TpuKernel`** — `instructions: list[TpuInst]`, `data_plan: list[dict]`,
  `outputs: list[dict]`. `.to_sxu_descriptor()` serializes to the **existing**
  `SXU_PROGRAM` JSON dict. The runtime (`TinyTPUProgram._exec_sxu_program`) is
  untouched, so migrated and not-yet-migrated kernels share one runtime.
- **`lower_kernel(sink_uop) -> TpuKernel`** — the InstSel pass plus the walker.

`render()` becomes a positive router (no try/None fallthrough for the migrated
family):

```
WMMA present        -> existing GEMM renderer          (kept, unchanged)
structural kernel   -> existing structural recognizer  (kept this slice)
otherwise           -> lower_kernel()  -- terminal, no fallback
```

## 6. The InstSel Pass

`lower_kernel` first runs a tinygrad `PatternMatcher` via `graph_rewrite` on the
kernel graph. This is the instruction-selection stage:

- **Normalization rules** — canonicalize tinygrad's decomposition so the walker
  sees clean primitives:
  - fold `ADD(MUL(x, CONST(-1)), y)` → `SUB(y, x)` (replaces the scattered
    `is_neg_add` special-casing in the old elementwise renderer)
  - drop identity `CAST`s (same dtype in and out)
  - canonicalize constant operand position for commutative ops
- **Selection rules:**
  - `WHERE(CMPLT(x, CONST(0)), CONST(0), x)` → a single `RELU` node
  - everything else is 1:1 and resolved by a static `Ops → _VPU_OPS` table
    during the walk

For this slice the pass is a normalizer plus a selection table. Subgraph-fusion
InstSel (MXU epilogues, fused reductions) is only needed by the out-of-scope
structural slices and is deliberately not built here.

## 7. The UOp Walker

For each output tile chunk (`ceil(out_size / _TILE_ELEMS)` chunks of 16
elements), topologically walk the rewritten DAG and emit instructions:

| UOp                       | emits                                              |
|---------------------------|----------------------------------------------------|
| input `LOAD`              | plan a VMEM tile, `LOAD vreg <- vmem`              |
| `CONST`                   | broadcast-const VMEM tile + `LOAD` (or `_vfill`)   |
| ALU (`ADD`/`MUL`/`MAX`/`CMPLT`/`AND`/`SHL`/`IDIV`/…) | one `VPU dst,a,op,b`     |
| `WHERE`                   | `SELECT dst,cond,lhs,rhs`                          |
| `EXP2` / `LOG2` / `SIN`   | `VPU_EXP2` / `VPU_LOG2` / `VPU_SIN`                |
| `SQRT` / `RSQRT` / `RECIPROCAL` | short fixed microprogram — an emit rule may emit >1 instruction (no single VPU sqrt opcode exists) |
| `CAST`                    | `I2F` / `F2I` (no-op casts already dropped in §6)  |
| `STORE`                   | `STORE vmem <- vreg`                               |

**Register allocation:** linear-scan with last-use over the straight-line
per-tile program. The allocator resets per tile chunk (chunks are independent).
If the live set exceeds the 16 VREGs (deep activations such as `mish`), **spill
to a VMEM scratch tile** via `STORE`/`LOAD`. The walker therefore always
succeeds — spilling is not a fallback, it is correct allocation. No other
fallback exists: an unknown UOp raises a clear error that `render()` converts to
the visible `UNSUPPORTED` descriptor.

**Why composite activations cost zero new walker code:** once the walker handles
binary/unary ALU, `WHERE`, the transcendentals, and `SQRT`, composite
activations (`elu`, `mish`, `sigmoid`, `tanh`, `swish`, `softplus`, …) need no
new code. tinygrad already decomposed them into exactly those primitives, so the
walker renders them by construction. They are deleted, not reimplemented — the
bulk of the ~37 recognizers in §8 fall out this way.

## 8. Scope Boundary — Deleted vs Kept

**Deleted** (the walker fully replaces these; verified no remaining callers):

`_render_elementwise_sxu_program`, `_render_multistep_sxu_program`,
`_render_where_sxu_program`, `_render_min_const_sxu_program`,
`_render_chained_const_sxu_program`, `_render_clip_sxu_program`,
`_render_clamp_single_bound_sxu_program`, `_render_relu6_sxu_program`,
`_render_hardsigmoid_sxu_program`, `_render_elu_sxu_program`,
`_render_leaky_relu_sxu_program`, `_render_softsign_sxu_program`,
`_render_swish_sxu_program`, `_render_sigmoid_sxu_program`,
`_render_tanh_sxu_program`, `_render_mish_sxu_program`,
`_render_softplus_sxu_program`, `_render_logsigmoid_sxu_program`,
`_render_tan_sxu_program`, `_render_cosh_sxu_program`,
`_render_sinh_sxu_program`, `_render_hyperbolic_sxu_program`,
`_render_logaddexp0_sxu_program`, `_render_self_cube_sxu_program`,
`_render_self_square_sxu_program`, `_render_exp2_sxu_program`,
`_render_scaled_exp2_sxu_program`, `_render_log2_sxu_program`,
`_render_scaled_log2_sxu_program`, `_render_sin_sxu_program`,
`_render_scaled_sin_sxu_program`, `_render_unary_transcendental_sxu_program`,
`_render_reciprocal_sxu_program`, `_render_rsqrt_sxu_program`,
`_render_sqrt_sxu_program`, `_render_trunc_sxu_program`,
`_render_scalar_const_divmod_sxu_program`.

Plus every `_find_*` / helper used **only** by the deleted functions — each
checked for remaining callers before deletion (dead-code removal, not a
fallback). The legacy `analyze_tinytpu_uops` diagnostic was removed later after
`tests/onnx_tinytpu_trace/driver.py` switched to renderer descriptors.

**Kept this slice** (own structural paths; future slices):
GEMM/WMMA path and `_render_gemm_fallback_sxu_program`,
`_render_reduction_sxu_program`, `_render_colreduce_sxu_program`,
`_render_rowreduce_sxu_program`, `_render_rowbc_sxu_program`,
`_render_colbc_sxu_program`, `_render_colbc_where_sxu_program`,
`_render_pad_sxu_program`, `_render_transpose_sxu_program`,
`_render_copy_sxu_program`, `_render_const_fill_sxu_program`,
`_render_cast_sxu_program`.

`_render_cast_sxu_program` is flagged: the walker handles pure VPU `I2F`/`F2I`/
bool converts; if `_render_cast` also does dtype-size *repacking* (memory
layout), that part stays. Resolved during implementation by reading the
function.

The kept structural recognizers retain their existing dispatch chain — that is
their routing, not a walker fallback. Converting them to walker-owned InstSel is
a later slice and out of scope here.

## 9. Verification

The repo already has dense numerical coverage (recent commits add mish, elu,
hardsigmoid tests). The correctness bar:

- All existing `tests/test_tinytpu_backend.py` and sim-backed
  `TestTinyTPUBackendGemm` tests pass unchanged. The walker must produce
  numerically identical results.
- A new lowering-dump assertion confirms the **walker path** was taken for the
  migrated family, not a (now-deleted) recognizer.
- TDD per `CLAUDE.md`: one commit per iteration with test, implementation,
  verification, and `TODO.md` / `results.tsv` rows per `AGENT.md`.
- Run from repo root via `.venv/bin/python3`:
  - `PYTHONPATH=tinygrad .venv/bin/python3 -m pytest tests/test_tinytpu_backend.py -x -v`
  - `python3 scripts/test_cosim.py`

## 10. Iteration Sequencing

Detailed step-by-step ordering is produced by the implementation plan. Intended
shape:

1. Module skeleton — `TpuInst`, `TpuKernel`, `.to_sxu_descriptor()` encoder, and
   the walker for binary/unary ALU plus `RELU`. Delete `_render_elementwise`.
2. `WHERE` → `SELECT`. Delete `_render_where`, `_render_min_const`,
   `_render_multistep`, `_render_chained_const`.
3. Transcendentals `EXP2`/`LOG2`/`SIN` (and scaled variants). Delete the
   matching recognizers.
4. `SQRT`/`RSQRT`/`RECIPROCAL` microprograms. Delete the matching recognizers.
5. Composite activations (`elu`, `mish`, `sigmoid`, `tanh`, `swish`, `softplus`,
   `logsigmoid`, `relu6`, `hardsigmoid`, `softsign`, `clip`, `tan`, `cosh`,
   `sinh`, self-square/cube) — delete *for free*; no new walker code.
6. `divmod` / `trunc`. Delete the matching recognizers.
7. Register-allocation spill path and cleanup of orphaned `_find_*` helpers.

## 11. Risks

- **Numeric divergence.** The walker emits one VPU op per decomposed primitive,
  so numerics follow tinygrad's decomposition exactly — the same thing the old
  recognizers and the numpy-reference tests already pin. Risk is low; the
  identical-result test gate catches any divergence.
- **VREG pressure.** Deep activations may exceed 16 live VREGs. Mitigated by the
  VMEM spill path (§7); a kernel can never become unrenderable.
- **`_render_cast` layout logic.** If cast does dtype-size repacking, splitting
  "pure convert" from "repack" must be done carefully. Mitigated by reading the
  function before touching it and keeping the repack path.
- **Helper deletion.** A `_find_*` helper might be shared with a kept
  recognizer. Mitigated by per-helper caller checks before deletion.

## 12. Exit Criteria

- The elementwise/unary/transcendental/activation family renders through
  `lower_kernel` — one TinyTPU instruction per UOp.
- The ~37 listed `_render_*` functions and their exclusive helpers are deleted.
- No fallback path exists for the migrated family; unsupported cases surface as
  `UNSUPPORTED`.
- All existing TinyTPU tests pass with numerically identical results.
- `TinyTPUProgram` and the `SXU_PROGRAM` descriptor are unchanged.
