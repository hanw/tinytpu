# TinyTPU Tinyspec Coverage TODO

This TODO estimates the remaining work to support the tinyspec surface in
`doc/tinyspec.tex`. It uses the current repo iteration style: one narrow,
tested, committed behavior per iteration.

## Overall Estimate

- Broad functional tinyspec coverage: **250-400 iterations**
- Robust hardware-backed and well-tested coverage: **500-800 iterations**

Current coverage is still a small slice of the full spec, but it now includes
real TinyTPU execution for GEMM, VPU binary ops, ReLU, and a simple reduction.

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

## Coverage Estimate by Area

### Current VPU/Runtime Architecture: 20-40 iterations

- [x] Make VPU-only programs complete without dummy MXU dispatch
- [ ] Add first-class runtime bundle builder for VMEM/VPU programs
- [ ] Support multiple VMEM output tiles
- [ ] Support multiple VPU instructions in one tinygrad lowered program
- [ ] Add runtime tests for VMEM preload/output protocol
- [ ] Improve trace output for VPU-only programs
- [ ] Document bundle records for VMEM input/output

### General Elementwise Scalar/Tile Support: 30-60 iterations

- [x] Single-tile int32 `ADD`
- [x] Single-tile int32 `MUL`
- [x] Single-tile int32 `MAX`
- [x] Single-tile int32 `RELU`
- [ ] Constants in VPU programs, e.g. `x + 1`, `x * 2`
  - [x] `x + scalar`
  - [x] `x * scalar`
  - [x] `maximum(x, scalar)`
- [ ] Scalar broadcasting
- [ ] Size-1 axis broadcasting
- [ ] Arbitrary shapes with `numel <= 16`
- [ ] Multi-tile elementwise loops for `numel > 16`
- [ ] Mixed VPU op chains without host round trips
- [ ] Output shape preservation for scalar, vector, and small matrix cases

### More Elementwise Ops: 35-70 iterations

- [ ] `SUB` as `ADD + NEG`
- [ ] `NEG` as multiply by `-1`
- [x] `CMPLT`
- [x] `CMPNE`
- [ ] `WHERE`
- [ ] `AND`
- [ ] `OR`
- [ ] `XOR`
- [ ] `SHL`
- [ ] `SHR`
- [ ] `MOD`
- [ ] `IDIV`
- [ ] `RECIP`
- [ ] `TRUNC`
- [ ] Basic `CAST`
- [ ] Basic `BITCAST`
- [ ] Basic `COPY`

### Reductions: 30-60 iterations

- [x] 4-element int32 sum to scalar
- [ ] Row-wise sum over a 4x4 VMEM tile
- [ ] Column-wise sum over a 4x4 VMEM tile
- [ ] Full-tile sum
- [ ] `MAX` reduction
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
- [ ] M/N/K tail handling
- [ ] Better unsupported shape diagnostics
- [ ] Bias epilogue
- [ ] ReLU epilogue
- [ ] Add/mul epilogue
- [ ] Non-int8 operand policy
- [ ] Accumulation overflow tests
- [ ] Multi-output tile scheduling cleanup

### Dtypes: 25-50 iterations

- [x] int32 VMEM values for VPU paths
- [x] int8 operands for MXU paths
- [ ] bool
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

- [ ] Replace brittle UOp-count matching with structural pattern matching
- [ ] Build reusable lowering IR for TinyTPU bundles
- [ ] Table-driven VPU opcode lowering
- [ ] Shape-aware lowering helpers
- [ ] Dtype-aware lowering helpers
- [ ] Better diagnostics with selected lowering candidate
- [ ] Run selected upstream tinygrad tests on `TINYTPU`
- [ ] Add skipped/xfail manifest for unsupported tinyspec areas
- [ ] Track coverage by tinyspec op category

### Profiler and Debuggability: 20-40 iterations

- [x] Trace parser validation
- [x] Perfetto emission for existing traces
- [ ] Full VPU-only trace coverage
- [ ] VMEM bundle dump utility
- [ ] Lowering decision dump
- [ ] Runtime bundle round-trip tests for all record types
- [ ] Per-op cycle reports
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
- [ ] `WHERE` and comparisons
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

## Recommended Next Iterations

1. Add `WHERE`.
2. Add `SUB` and `NEG` decompositions.
3. Add single-tile transpose through XLU.
4. Add single-tile reshape/permute no-copy cases.
5. Add multi-tile elementwise loop for `ADD`.
6. Add runtime bundle round-trip tests for VMEM records.
7. Add selected upstream tinygrad test invocation for covered VPU ops.
8. Add structural VPU lowering helpers to replace UOp-count checks.
9. Add bool dtype coverage for comparison outputs.
10. Add scalar-constant comparison support.
