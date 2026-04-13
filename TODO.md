# TinyTPU Tinyspec Coverage TODO

This TODO estimates the remaining work to support the tinyspec surface in
`doc/tinyspec.tex`. It uses the current repo iteration style: one narrow,
tested, committed behavior per iteration.

## Overall Estimate

- Broad functional tinyspec coverage: **250-400 iterations**
- Robust hardware-backed and well-tested coverage: **500-800 iterations**

Current coverage includes hardware-backed TinyTPU execution for:
GEMM (multi-WMMA, batched, deep-K, wide-N, with hardware fused bias+ReLU
epilogue); full int32/bool VPU elementwise (ADD/MUL/SUB/MAX/MIN/DIV/MOD/
ABS, CMP{LT,EQ,NE}, AND/OR/XOR/NOT, SHL/SHR, WHERE, RELU, clip, fused
add+relu); scalar-const variants and scalar broadcasting; all shapes with
numel>16 through multi-tile elementwise loops; scalar, row-wise, and
column-wise reductions (SUM/MAX/MIN/PROD) for any NxM through SXU_PROGRAM
emitting VPU_{SUM,MAX,MIN,MUL}_REDUCE{,_COL,_TILE}; movement ops
(reshape, contiguous slice/shrink, scalar expand, unrolled row expand);
XLU transpose reachable via SXU_DISPATCH_XLU_TRANSPOSE at runtime level;
TASM bundle assembler/disassembler; runtime bundle roundtrip + end-to-end
sim tests.

## Current Progress

- [x] Save tinyspec source in `doc/tinyspec.tex`
- [x] Runtime co-simulation bundle format for MXU programs
- [x] Runtime VMEM preload and VMEM result output path
- [x] Tinygrad GEMM lowering for supported tiled int32 cases through 4x4 MXU
- [x] Multi-row, batched, deep-K, wide-N GEMM test coverage
- [x] Tinygrad int32 single-tile VPU binary lowering
  - [x] `ADD`
  - [x] `MUL`
  - [x] `MAX`
  - [x] `SUB`
- [x] Tinygrad int32 single-tile VPU unary lowering
  - [x] `RELU`
- [x] Tinygrad 4-element int32 reduction lowering
  - [x] `SUM` via `VPU_SUM_REDUCE`
- [x] Full 16-lane VMEM tile coverage
  - [x] `ADD`
  - [x] `MUL`
  - [x] `MAX`
  - [x] `RELU`
- [x] Signed VPU test coverage
- [x] Runtime output validation for MXU and VMEM result lines
- [x] Remove generated `bdpi/tinytpu_io.o` from git tracking
- [x] TASM bundle assembler and disassembler (`scripts/tasm.py`, `doc/tinytpu_asm.md`)
- [x] Full-tile and multi-tile abs, IDIV, MOD coverage
- [x] Scalar broadcast (size-1 tensor) for MUL, SUB, MAX, multi-tile ADD/MUL
- [x] int32→bool and bool→int32 cast coverage
- [x] Clip (MIN+MAX program) full-tile and multi-tile
- [x] Fused add+relu full-tile and multi-tile
- [x] Tensor-tensor IDIV and MOD
- [x] Row-wise sum/max/min for NxM tensors — hardware-backed via VPU_{SUM,MAX,MIN}_REDUCE through SXU_PROGRAM for all N,M (legacy VPU_ROWSUM/HOST_ROWREDUCE removed)
- [x] Column-wise sum/max/min for NxM tensors — hardware-backed via VPU_*_REDUCE_COL for all N,M (legacy HOST_COLREDUCE removed)
- [x] Fix stale WAIT_MXU opcode in VPU-only test bundles
- [x] 2D tensor ops: all VPU binary/unary ops for arbitrary 2D shapes
- [x] Grouped scalar-const lowering for 2D/large tensors (NEG, x*c, x+c)
- [x] _tasm helper functions + bundle builders rewritten in readable assembly style
- [x] TASM bundle roundtrip tests for all bundle builders

## Legacy Descriptor Removal Milestones

- [ ] Milestone 1: remove legacy `GEMM4x4`
  - Route all remaining single-tile GEMM cases through `SXU_PROGRAM`
  - Delete the `GEMM4x4` descriptor/runtime path once coverage is equivalent
- [ ] Milestone 2: remove legacy `VPU_BINARY`
  - Migrate remaining simple elementwise/bool/broadcast cases to `SXU_PROGRAM`
  - Keep regressions for scalar, row, and column broadcast forms while deleting the descriptor path
- [ ] Milestone 3: remove legacy `VPU_PROGRAM`
  - Convert remaining analyzer-driven multi-step kernels to explicit `SXU_PROGRAM` renderers
  - Prefer hardware primitives over adding new analyzer branches
- [ ] Milestone 4: remove legacy `HOST_BINARY`
  - Either lower all remaining integer host fallbacks to hardware-backed SXU/VPU sequences
  - Or make unsupported integer behavior explicit instead of silently keeping a host escape hatch
- [ ] Milestone 5: remove legacy `HOST_UNARY`
  - Add hardware-backed support for float `TRUNC` / `RECIPROCAL`, or declare them unsupported on TinyTPU
  - Delete host-unary execution only after the policy is explicit and tested
- [ ] Milestone 6: simplify runtime after descriptor removal
  - Reduce `_SUPPORTED_OPS` to `SXU_PROGRAM` plus `UNSUPPORTED`
  - Delete `_render_legacy_descriptor(...)` and the non-SXU executors once no tests/workloads depend on them

## Coverage Estimate by Area

### Current VPU/Runtime Architecture: 20-40 iterations

- [x] Make VPU-only programs complete without dummy MXU dispatch
- [ ] Add first-class runtime bundle builder for VMEM/VPU programs
- [x] Support multiple VMEM output tiles (col-reduce/row-reduce/GEMM emit multi-tile outputs through SXU_PROGRAM)
- [x] Support multiple VPU instructions in one tinygrad lowered program (SXU_PROGRAM multi-step paths: abs, clip, MOD, WHERE, row/col reductions)
- [x] Add runtime tests for VMEM preload/output protocol
- [ ] Improve trace output for VPU-only programs
- [x] Document bundle records for VMEM input/output
- [ ] Fuse GEMM epilogues in model runs
  - Post-run review on `scripts/models/cnn_4_8_8_4.py` showed the model runs end to end without host fallback or `UNSUPPORTED`, but each layer still lowers as `MXU -> LOAD bias -> VPU ADD -> VPU RELU -> STORE` rather than a first-class fused `matmul + bias + relu` epilogue path.
  - Add a direct runtime/compiler path so model kernels stop depending on separate bias-load and VPU epilogue instructions after every MXU tile.

### General Elementwise Scalar/Tile Support: 30-60 iterations

- [x] Single-tile int32 `ADD`
- [x] Single-tile int32 `MUL`
- [x] Single-tile int32 `MAX`
- [x] Single-tile int32 `RELU`
- [x] Constants in VPU programs, e.g. `x + 1`, `x * 2`
  - [x] `x + scalar`
  - [x] `x * scalar`
  - [x] `maximum(x, scalar)`
  - [x] `minimum(x, scalar)`
  - [x] `x < scalar`
  - [x] `x != scalar`
- [x] Scalar broadcasting
  - [x] Add explicit SXU/XLU broadcast opcodes (scalar/row/col)
  - [x] Migrate scalar broadcast binary ops to SXU_PROGRAM via `BROADCAST_SCALAR`
- [x] Size-1 axis broadcasting (scalar-expand via BROADCAST_SCALAR; row-expand via BROADCAST_ROW for unrolled shapes)
- [x] Column-broadcast compare/select lowering for tinygrad workloads
  - `(4,4) < (4,1)` now lowers through `SXU_PROGRAM` with the `BROADCAST_COL` primitive instead of falling through to `UNSUPPORTED`.
  - The post-run review model path `mask.where(y, y * 2)` now closes through a dedicated single-tile `BROADCAST_COL_SELECT` SXU program when the compare and select stay fused in one kernel.
  - Follow-up: a separately realized bool mask still exposes a different mixed bool/int32 arithmetic gap in the generic `VPU_BINARY` path; keep that as a distinct issue rather than regressing the fused review-model path.
- [x] Arbitrary shapes with `numel <= 16` (1D/2D/nD elementwise + reshape/slice/expand covered by copy + elementwise renderers)
  - [x] Shape-preserving 2x2 elementwise coverage for supported VPU ops
- [x] Multi-tile elementwise loops for `numel > 16`
  - [x] Multi-tile AND/OR/XOR/NOT for bool tensors
- [x] Mixed VPU op chains without host round trips (via VPU_PROGRAM path)
- [x] Output shape preservation for scalar, vector, and small matrix cases (1D/2D/nD coverage via copy + elementwise + reduction renderers)

### More Elementwise Ops: 35-70 iterations

- [x] `SUB` as `ADD + NEG`, lowered to `VPU_SUB`
- [x] `NEG` as multiply by `-1`
- [x] `CMPLT`
- [x] `CMPNE`
- [x] `CMPEQ`
- [x] `WHERE`
- [x] `AND`
- [x] `OR`
- [x] `XOR`
- [x] `NOT`
- [x] `SHL`
- [x] `SHR`
- [x] `MOD` (via DIV+MUL+SUB bundle)
- [x] `IDIV` (via VPU_DIV, truncation semantics)
- [ ] `RECIP`
- [ ] `TRUNC`
- [x] Basic `CAST` (int32↔bool; int32↔float32 via VPU_I2F/F2I; fused same-dtype round-trips via COPY)
- [ ] Basic `BITCAST`
- [x] Basic `COPY` (_render_copy_sxu_program handles same-dtype identity-index kernels via LOAD/STORE pairs)

### Reductions: 30-60 iterations

- [x] 4-element int32 sum to scalar
- [x] Full-tile int32 sum to scalar (via VPU_SUM_REDUCE_TILE)
- [x] Multi-tile int32 sum to scalar (VPU_SUM_REDUCE_TILE per tile + VPU_ADD combine)
- [x] Row-wise sum/max/min over NxM tensor (SXU_PROGRAM row-reduce renderer, all N,M)
- [x] Column-wise sum/max/min over NxM tensor (SXU_PROGRAM col-reduce renderer, all N,M)
- [x] Full-tile sum (via VPU_SUM_REDUCE_TILE)
- [x] `MAX` reduction (4-elem, full-tile, multi-tile via VPU_MAX_REDUCE_TILE)
- [x] `MIN` reduction (4-elem, full-tile, multi-tile via VPU_MIN_REDUCE_TILE)
- [x] VPU col-reduce primitives (VPU_SUM/MAX/MIN_REDUCE_COL opcodes 29/30/31)
- [x] VPU tile-reduce primitives (VPU_SUM/MAX/MIN_REDUCE_TILE opcodes 32/33/34)
- [x] `MUL` reduction (VPU_MUL_REDUCE{,_COL,_TILE} hardware + tinygrad lowering for scalar/row/col prod)
- [x] `keepdim` behavior (rowsum/rowmax/rowmin keepdim tests pass)
- [x] Multi-tile reductions (scalar SUM/MAX/MIN with VPU_*_REDUCE_TILE per tile + VPU_ADD/MAX/MIN combine)
- [ ] Reduction axis shape validation
- [ ] Reduction result layout in VMEM

### Movement Ops: 40-80 iterations

- [x] `RESHAPE` (copy SXU_PROGRAM with identity LOAD/STORE index mapping)
- [ ] `PERMUTE`
- [ ] `TRANSPOSE` via XLU — hardware: SXU_DISPATCH_XLU_TRANSPOSE opcode (12) + runtime test done; tinygrad permute(1,0) lowering still open
- [x] `EXPAND` (scalar and unrolled row-broadcast via BROADCAST_SCALAR / BROADCAST_ROW; RANGE-loop variant still open)
- [x] `SHRINK` (contiguous slice via copy renderer with affine offset)
- [ ] `PAD`
- [ ] `FLIP`
- [ ] `CAT`
- [x] `INDEX` with simple affine patterns (contiguous slice with constant offset through copy renderer)
- [ ] Gather-like indexing
- [x] XLU-backed broadcast primitives (scalar/row/col)
- [ ] XLU-backed permutation paths
- [ ] Multi-tile movement kernels

### GEMM and Matmul Expansion: 25-50 iterations

- [x] 4x4 GEMM
- [x] Multi-row GEMM
- [x] Batched GEMM
- [x] Deep-K tiled GEMM
- [x] Wide-N tiled GEMM
- [x] Multi-WMMA lowering (multi-tile GEMM through WMMA path)
- [x] Bias epilogue (row-broadcast and full-tensor, hardware-backed for single-K-tile)
- [x] ReLU epilogue (hardware-backed for single-K-tile)
- [x] Fused bias+ReLU epilogue (hardware-backed via SXU_LOAD_MXU_RESULT → VPU)
- [ ] M/N/K tail handling
- [ ] Better unsupported shape diagnostics
- [ ] Add/mul epilogue
- [ ] Non-int8 operand policy
- [ ] Accumulation overflow tests
- [ ] Multi-output tile scheduling cleanup
- [ ] Hardware epilogue for multi-K-tile GEMM (currently falls back to numpy)

### Dtypes: 25-50 iterations

- [x] int32 VMEM values for VPU paths
- [x] int8 operands for MXU paths
- [x] bool comparison outputs
- [ ] int8 elementwise
- [ ] uint8 elementwise
- [ ] int16 elementwise
- [ ] uint16 elementwise
- [ ] uint32 elementwise
- [x] float32 policy (FADD/FSUB/FMUL/FMAX/FCMPLT dispatched via VPU_F* variants; scalar const and multi-tile covered; FRECIP+scalar fdiv work; tensor-tensor fdiv requires RECIPROCAL UOp detection, tracked)
- [ ] cast saturation/wrapping behavior
- [ ] comparison output dtype behavior
- [ ] dtype range diagnostics

### Call/Tuple/Ordering Semantics: 20-40 iterations

- [ ] `CALL`
- [ ] `TUPLE`
- [ ] `GETTUPLE`
- [ ] `AFTER`
- [ ] assign/store dependency ordering
- [ ] multiple outputs
- [ ] multiple stores in one kernel
- [ ] reusable captured graph fragments

### Multi-Kernel and Memory Planning: 20-50 iterations

- [ ] Multiple TinyTPU programs per tensor expression
- [ ] Intermediate VMEM allocation
- [ ] Spill/fill to HBM model
- [ ] Host/device copy scheduling
- [ ] Cross-kernel dependency tracking
- [ ] Runtime buffer lifetime tests
- [ ] Larger tensors split across VMEM tiles

### Tinygrad Backend Integration Quality: 40-80 iterations

#### Shrink ops_tinytpu.py (~2335 → target <1200 lines)

Current bloat sources:
- `analyze_tinytpu_uops` (~800 lines): old UOp-counting renderer, mostly
  superseded by WMMA path. Delete dead paths, merge remaining VPU detection
  into structural matchers.
- Bundle builders (~400 lines): `_build_vpu_binary_bundle`,
  `_build_vpu_where_bundle`, `_build_full_gemm_bundle`, etc. Unify into a
  single generic bundle builder with a template pattern.
- `_exec_*` methods (~400 lines): every op has the same chunk loop
  (`for chunk_start in range(0, num_elems, _TILE_ELEMS)`). Extract a shared
  `_run_tiled_vpu` helper.
- Duplicate output parsing: `_parse_vmem_output`, `_parse_multi_vmem_output`,
  `_parse_sim_output` share the same structure.

Cleanup plan — eliminate analyze_tinytpu_uops via SXU_PROGRAM migration:
- [x] Add VPU_NOT hardware opcode (BSV + testbench + Python table)
- [x] Migrate scalar-const binary ops (x+c, x*c, NEG, NOT) to SXU_PROGRAM
- [x] Migrate bool-typed ops (AND/OR/XOR/NOT on bool tensors) to SXU_PROGRAM
- [x] Migrate WHERE (ternary select) to SXU_PROGRAM (now via first-class SXU_DISPATCH_SELECT using VPU_SELECT)
- [x] Migrate multi-step VPU_PROGRAM patterns (abs, clip, MOD, CMPEQ) to SXU_PROGRAM
- [x] Migrate scalar reductions (SUM/MAX/MIN to scalar) to SXU_PROGRAM
- [x] Migrate row-wise reduce to SXU_PROGRAM (done: _render_rowreduce_sxu_program, legacy VPU_ROWSUM removed)
- [x] Migrate row-broadcast binary (VPU_ROWBC_BINARY) to SXU_PROGRAM
- [ ] Emit host fallbacks (HOST_*) directly from renderer without analyze_tinytpu_uops
- [x] Delete dead analyze blocks, _exec_vpu_where/unary, _build_vpu_where, _render_wmma_descriptor
- [ ] Delete remaining analyze_tinytpu_uops (now limited to scalar-const DIV/MOD)
- [x] Run selected upstream tinygrad tests on `TINYTPU` (2 tests pass via scripts/run_tinytpu_upstream_subset.py — expand list as coverage grows)
- [ ] Add skipped/xfail manifest for unsupported tinyspec areas
- [ ] Track coverage by tinyspec op category

### Profiler and Debuggability: 20-40 iterations

- [x] Trace parser validation
- [x] Perfetto emission for existing traces
- [ ] Full VPU-only trace coverage
- [ ] VMEM bundle dump utility
- [x] Lowering decision dump (TINYTPU_DUMP_LOWERING env var emits renderer JSON)
- [x] Runtime bundle round-trip tests for all record types
- [x] Per-op cycle reports
- [ ] MXU/VPU/VMEM utilization by lowered tinygrad op

### Optional/Advanced Tinyspec Surface: 50-150 iterations

- [ ] Multi-device `COPY`
- [ ] `REPLICATED`
- [ ] collectives: broadcast, scatter, gather, reduce, allgather
- [ ] reduce_scatter
- [ ] allreduce
- [ ] optimizer axis semantics
- [ ] `BARRIER`
- [ ] `SPECIAL`
- [ ] `IF` / `ENDIF`
- [ ] `WMMA` mapping beyond current MXU path
- [ ] `CUSTOM`
- [ ] `ATOMICADD`
- [ ] `CUSTOMFUNCTION`
- [ ] `PROGRAM` / `SOURCE` / `BINARY` metadata handling

## Milestones

### Milestone 1: Useful Single-Tile Tensor Programs

Estimate: **~50 total iterations**

- [x] Single-tile binary VPU ops
- [x] Single-tile ReLU
- [x] One simple reduction
- [x] Constants and scalar broadcasting
- [x] `WHERE` and comparisons
- [x] Basic reshape/transpose within one 4x4 tile (reshape via copy renderer; transpose reachable via SXU_DISPATCH_XLU_TRANSPOSE, tinygrad lowering pending)
- [x] VPU-only runtime completion cleanup

### Milestone 2: Multi-Tile Core Tensor Ops

Estimate: **~120 total iterations**

- [x] Multi-tile elementwise (ADD/MUL/SUB/MAX/MIN/compare/logic + scalar-const all cover numel>16 via tile-loop)
- [x] Multi-tile reductions (scalar via VPU_*_REDUCE_TILE; row/col via SXU_PROGRAM tile iteration)
- [x] Multi-tile movement ops (multi-tile reshape/slice/expand work through the copy renderer's tile loop)
- [x] GEMM epilogues (hardware-backed bias add, ReLU, fused bias+ReLU via SXU_LOAD_MXU_RESULT)
- [ ] More shape coverage
- [x] First selected upstream tinygrad test subset passing on `TINYTPU` (test_plus_int, test_plus_big)

### Milestone 3: Broad Core Tinyspec Semantics

Estimate: **~250 total iterations**

- [ ] Most movement ops
- [ ] Most integer elementwise ops
- [x] Add/max/mul reductions (SUM/MAX/MIN/PROD scalar, row, col via VPU_*_REDUCE{,_COL,_TILE})
- [ ] More dtype handling
- [ ] Multi-kernel scheduling
- [ ] Structural lowering pass

### Milestone 4: Robust Full-Spec Direction

Estimate: **400+ iterations**

- [ ] Broad tinyspec semantic coverage
- [ ] Well-tested runtime and lowering failures
- [ ] Robust memory planning
- [ ] Multi-device decisions documented
- [ ] Advanced codegen ops either implemented or explicitly out of scope

### Milestone 5: Serious Full-Spec Coverage

Estimate: **500-800 iterations**

- [ ] Full core tinyspec behavior
- [ ] Broad upstream tinygrad test coverage
- [ ] Hardware/runtime-backed execution where appropriate
- [ ] Clear software fallback policy where hardware is not appropriate
- [ ] Stable performance/profiling story

## Recommended Next Iterations (updated 2026-04-12)

Current state: 379 backend tests + 8 runtime bundle tests. Row/col/scalar
reductions (SUM/MAX/MIN/PROD) fully hardware-backed. Movement ops: reshape,
contiguous slice/shrink, scalar and unrolled row-expand working.
SXU_DISPATCH_XLU_TRANSPOSE opcode exists in hardware but tinygrad
permute/transpose lowering not yet wired.

Highest-value next work:
1. Wire tinygrad `permute(1,0)` to SXU_DISPATCH_XLU_TRANSPOSE. Detect the
   2D index-swap pattern in the UOp graph and emit LOAD/XLU_TRANSPOSE/STORE
   for single-tile cases; extend to multi-tile via a tile-of-tiles pass.
2. RANGE-loop variant of row-broadcast expand (shape (N, M) with M<_COLS
   and outer RANGE over rows) — current renderer handles only unrolled.
3. Hardware epilogue for multi-K-tile GEMM (currently still a numpy
   fallback).
4. Delete remaining `analyze_tinytpu_uops` (only scalar-const DIV/MOD
   left — requires matching tinygrad's WHERE+CMPLT+AND floor-div decomp).
5. Dtype expansion beyond int32/bool: int8/uint8/int16 elementwise policy.
6. Movement gaps: PAD (requires CMPLT bounds check), FLIP (LOAD index =
   const - STORE index), CAT (WHERE-based select between two buffers),
   true permute (non-transpose axis reorder).
7. Multi-output tile scheduling cleanup and intermediate VMEM allocation.
8. Fused kernel detection improvements for 2-op chains tinygrad emits
   (slice-then-sum currently fuses to wrong-offset sum).

### Model-blocking gaps (from mnist_gan.py bring-up)

- `leaky_relu(alpha)`: tinygrad emits float mul for the negative slope.
  SW path: integer approximation via SHR + WHERE select.
- `tanh`, `log_softmax`, float32 GEMM: require float datapath (major arch
  change). SW fallback via host numpy or replace with int/argmax where
  semantically acceptable for inference-only models.
- `backward` / autograd: training is not supported; export int8 weights and
  run inference only.
