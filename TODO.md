# TinyTPU Tinyspec Coverage TODO

This TODO estimates the remaining work to support the tinyspec surface in
`doc/tinyspec.tex`. It uses the current repo iteration style: one narrow,
tested, committed behavior per iteration.

## Overall Estimate

- Broad functional tinyspec coverage: **250-400 iterations**
- Robust hardware-backed and well-tested coverage: **500-800 iterations**

Current coverage is still a small slice of the full spec, but it now includes
real TinyTPU execution for GEMM, core int32 VPU binary ops, ReLU, equality and
comparison bool outputs, scalar constants, simple reductions, row-wise
sum/max/min, abs, clip, fused add+relu, IDIV, MOD, scalar broadcasting,
int32/bool casts, and the TASM bundle assembler/disassembler.

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
- [x] Row-wise sum/max/min for NxM tensors (VPU_ROWSUM for M=4, HOST_ROWREDUCE for other M)
- [x] Column-wise sum/max/min for NxM tensors (HOST_COLREDUCE)
- [x] Fix stale WAIT_MXU opcode in VPU-only test bundles
- [x] 2D tensor ops: all VPU binary/unary ops for arbitrary 2D shapes
- [x] Grouped scalar-const lowering for 2D/large tensors (NEG, x*c, x+c)
- [x] _tasm helper functions + bundle builders rewritten in readable assembly style
- [x] TASM bundle roundtrip tests for all bundle builders

## Coverage Estimate by Area

### Current VPU/Runtime Architecture: 20-40 iterations

- [x] Make VPU-only programs complete without dummy MXU dispatch
- [ ] Add first-class runtime bundle builder for VMEM/VPU programs
- [ ] Support multiple VMEM output tiles
- [ ] Support multiple VPU instructions in one tinygrad lowered program
- [x] Add runtime tests for VMEM preload/output protocol
- [ ] Improve trace output for VPU-only programs
- [x] Document bundle records for VMEM input/output

### General Elementwise Scalar/Tile Support: 30-60 iterations

- [x] Single-tile int32 `ADD`
- [x] Single-tile int32 `MUL`
- [x] Single-tile int32 `MAX`
- [x] Single-tile int32 `RELU`
- [ ] Constants in VPU programs, e.g. `x + 1`, `x * 2`
  - [x] `x + scalar`
  - [x] `x * scalar`
  - [x] `maximum(x, scalar)`
  - [x] `minimum(x, scalar)`
  - [x] `x < scalar`
  - [x] `x != scalar`
- [ ] Scalar broadcasting
  - [ ] Add VPU_BROADCAST BSV instruction (broadcast lane 0 to all lanes+sublanes)
  - [ ] Migrate scalar broadcast binary ops to SXU_PROGRAM
- [ ] Size-1 axis broadcasting
- [ ] Arbitrary shapes with `numel <= 16`
  - [x] Shape-preserving 2x2 elementwise coverage for supported VPU ops
- [x] Multi-tile elementwise loops for `numel > 16`
  - [x] Multi-tile AND/OR/XOR/NOT for bool tensors
- [x] Mixed VPU op chains without host round trips (via VPU_PROGRAM path)
- [ ] Output shape preservation for scalar, vector, and small matrix cases

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
- [x] Basic `CAST` (int32↔bool)
- [ ] Basic `BITCAST`
- [ ] Basic `COPY`

### Reductions: 30-60 iterations

- [x] 4-element int32 sum to scalar
- [x] Full-tile int32 sum to scalar
- [x] Multi-tile int32 sum to scalar
- [x] Row-wise sum/max/min over NxM tensor (VPU_ROWSUM for M=4, HOST_ROWREDUCE otherwise)
- [x] Column-wise sum/max/min over NxM tensor (HOST_COLREDUCE)
- [ ] Full-tile sum
- [x] `MAX` reduction (4-elem, full-tile, multi-tile via VPU_MAX_REDUCE)
- [x] `MIN` reduction (4-elem, full-tile, multi-tile via VPU_MIN_REDUCE)
- [ ] `MUL` reduction
- [ ] `keepdim` behavior
- [ ] Multi-tile reductions
- [ ] Reduction axis shape validation
- [ ] Reduction result layout in VMEM

### Movement Ops: 40-80 iterations

- [ ] `RESHAPE`
- [ ] `PERMUTE`
- [ ] `TRANSPOSE` via XLU
- [ ] `EXPAND`
- [ ] `SHRINK`
- [ ] `PAD`
- [ ] `FLIP`
- [ ] `CAT`
- [ ] `INDEX` with simple affine patterns
- [ ] Gather-like indexing
- [ ] XLU-backed broadcast/permutation paths
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
- [ ] float32 policy
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
- [x] Migrate WHERE (ternary select) to SXU_PROGRAM
- [x] Migrate multi-step VPU_PROGRAM patterns (abs, clip, MOD, CMPEQ) to SXU_PROGRAM
- [ ] Migrate scalar reductions (SUM/MAX/MIN to scalar) to SXU_PROGRAM
- [ ] Migrate row-wise reduce (VPU_ROWSUM) to SXU_PROGRAM
- [ ] Migrate row-broadcast binary (VPU_ROWBC_BINARY) to SXU_PROGRAM
- [ ] Emit host fallbacks (HOST_*) directly from renderer without analyze_tinytpu_uops
- [ ] Delete analyze_tinytpu_uops, old bundle builders, old _exec_* methods
- [ ] Run selected upstream tinygrad tests on `TINYTPU`
- [ ] Add skipped/xfail manifest for unsupported tinyspec areas
- [ ] Track coverage by tinyspec op category

### Profiler and Debuggability: 20-40 iterations

- [x] Trace parser validation
- [x] Perfetto emission for existing traces
- [ ] Full VPU-only trace coverage
- [ ] VMEM bundle dump utility
- [ ] Lowering decision dump
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
- [ ] Constants and scalar broadcasting
- [x] `WHERE` and comparisons
- [ ] Basic reshape/transpose within one 4x4 tile
- [x] VPU-only runtime completion cleanup

### Milestone 2: Multi-Tile Core Tensor Ops

Estimate: **~120 total iterations**

- [ ] Multi-tile elementwise
- [ ] Multi-tile reductions
- [ ] Multi-tile movement ops
- [ ] GEMM epilogues
- [ ] More shape coverage
- [ ] First selected upstream tinygrad test subset passing on `TINYTPU`

### Milestone 3: Broad Core Tinyspec Semantics

Estimate: **~250 total iterations**

- [ ] Most movement ops
- [ ] Most integer elementwise ops
- [ ] Add/max/mul reductions
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

## Recommended Next Iterations (updated 2026-04-10)

Current state: 340 tests, multi-WMMA + hardware epilogue batch completed.

Completed in this batch:
1. ~~GEMM + bias add epilogue~~ — done, hardware-backed
2. ~~GEMM + ReLU epilogue~~ — done, hardware-backed
3. ~~Multi-WMMA lowering~~ — done, all tiled GEMM shapes go through WMMA path
4. SXU_LOAD_MXU_RESULT instruction — new BSV hardware for MXU→VRF transfer

Highest-value next work:
1. Hardware epilogue for multi-K-tile GEMM (currently numpy fallback)
2. Column-wise SUM via VPU (use XLU TRANSPOSE + VPU_SUM_REDUCE instead of host numpy)
3. Movement ops: RESHAPE as no-copy, PERMUTE via XLU
4. XLU transpose hardware backing for col-wise reductions
5. Broader tinygrad upstream test coverage
6. Multi-output kernel support
7. Fused kernel detection improvements (tinygrad fuses many 2-op chains)

### Model-blocking gaps (from mnist_gan.py bring-up)

- `leaky_relu(alpha)`: tinygrad emits float mul for the negative slope.
  - SW path: integer approximation via shift (alpha=0.25 → SHR by 2) + WHERE select.
    Needs: VPU_SHR + VPU_CMPLT + WHERE sequence lowered as a VPU_PROGRAM.
  - HW path: not needed if SW approximation suffices.
- `tanh`: requires RECIPROCAL, EXP, float BITCAST — pure float.
  - SW path: host software fallback (`HOST_UNARY` with numpy). Or integer clamp approximation.
  - HW path: would need float datapath (major arch change, not recommended now).
- `log_softmax`: requires EXP, LOG, RECIPROCAL — pure float.
  - SW path: host software fallback. For inference-only int models, replace with argmax.
  - HW path: same as tanh — needs float (not recommended now).
- `float32` GEMM: GAN training uses float weights/activations throughout.
  - SW path: host fallback for float matmul. Or quantize-aware int8 inference.
  - HW path: would need float MXU (major arch change).
- `backward` / autograd: training is not supported on TinyTPU.
  - SW path: train on CPU/GPU, export int8 weights, run inference on TinyTPU.

## Old Recommended Next Iterations

1. Add multi-tile elementwise loop for `ADD` using the plan in `doc/tinytpu_multitile_add.md`.
2. Add multiple VMEM output tile support in the runtime protocol.
3. Add multi-instruction VPU bundles for two-pass reductions.
4. Add row-wise sum result compaction.
5. Add full-tile sum through two-pass VPU lowering.
6. Add `WHERE` as a first general select op.
7. Add single-tile transpose through XLU.
8. Add single-tile reshape/permute no-copy cases or document exact tinygrad movement UOps.
9. Run `scripts/run_tinytpu_upstream_subset.py` without `--dry-run` in CI once the selected tests are stable.
10. Replace remaining UOp-count matching with structural pattern helpers.

## Next 50 Iteration Plan

1. [x] Add scalar `x - c` coverage.
2. [x] Add reverse scalar `c - x` lowering.
3. [x] Add reverse scalar `c - x` coverage.
4. [x] Add `CMPEQ` VPU opcode.
5. [x] Lower tensor equality through `CMPEQ`.
6. [x] Lower scalar equality through `CMPEQ`.
7. [x] Add equality dtype coverage.
8. [x] Add full-tile `SUB` coverage.
9. [x] Add full-tile `CMPLT` coverage.
10. [x] Add full-tile `CMPNE` coverage.
11. [x] Add full-tile `CMPEQ` coverage.
12. [x] Add shape-preservation coverage for 2x2 elementwise ops.
13. [x] Mark arbitrary `numel <= 16` coverage for supported VPU ops.
14. [x] Add explicit unsupported test for `numel > 16`.
15. [x] Add scalar negative constant coverage for `ADD`.
16. [x] Add scalar negative constant coverage for `MUL`.
17. [x] Add scalar negative constant coverage for `MAX`.
18. [x] Add scalar `x != c` full-tile coverage.
19. [x] Add scalar `x < c` full-tile coverage.
20. [x] Add scalar `x == c` full-tile coverage.
21. [x] Add reverse scalar `c < x` diagnostics/coverage.
22. [x] Add VPU opcode table tests.
23. [x] Refactor bool-result opcode set into a helper.
24. [x] Refactor VPU opcode table into module constants.
25. [x] Refactor scalar-const descriptor fields.
26. [x] Add tests for descriptor JSON for VPU binary ops.
27. [x] Add tests for descriptor JSON for scalar const ops.
28. [x] Add tests for descriptor JSON for equality ops.
29. [x] Add runtime bundle builder helper for VPU binary programs.
30. [x] Reuse profiler `Bundle` for VPU bundle text where practical.
31. [x] Add VMEM/VPU bundle dump utility.
32. [x] Add lowering-decision dump behind an env var.
33. [x] Add selected upstream tinygrad VPU test list.
34. [x] Add selected upstream tinygrad invocation wrapper.
35. [x] Add skipped manifest for unsupported tinyspec areas.
36. [x] Add tinyspec coverage category table.
37. [x] Document VMEM input/output bundle records.
38. [x] Add runtime VMEM preload/output protocol test.
39. [x] Add trace parser coverage for VPU-only traces.
40. [x] Add per-op cycle report for VPU traces.
41. [x] Improve unsupported diagnostics for multi-tile elementwise.
42. [x] Improve unsupported diagnostics for `WHERE`.
43. [x] Improve unsupported diagnostics for movement ops.
44. [x] Add `RESHAPE` no-copy coverage if tinygrad lowers as copy.
45. [x] Add `PERMUTE` unsupported diagnostic coverage.
46. [x] Add row-wise reduction investigation note.
47. [x] Add full-tile sum investigation note.
48. [x] Add first multi-tile `ADD` design note.
49. [x] Re-estimate remaining tinyspec iterations after this batch.
50. [x] Update recommended next iterations from observed gaps.
