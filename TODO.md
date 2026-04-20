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

- [x] Milestone 1: remove legacy `GEMM4x4`
  - GEMM fallback (MULACC or scalar MUL+RANGE, no WMMA UOp) now emits `SXU_PROGRAM` with the same data_plan/instructions as the WMMA SXU path
  - `_exec_gemm4x4` executor deleted
- [x] Milestone 2: remove legacy `VPU_BINARY`
  - Scalar-const IDIV, tensor-tensor IDIV, and bool→int32 cast now lower to `SXU_PROGRAM`
  - `_exec_vpu_binary` executor and `VPU_BINARY` emitter deleted
- [x] Milestone 3: remove legacy `VPU_PROGRAM`
  - Scalar-const MOD, tensor-tensor MOD, CLIP, ABS, and fused-add-relu already flow through SXU renderers
  - `_exec_vpu_program` executor deleted; the analyzer no longer emits `VPU_PROGRAM`
- [x] Milestone 4: remove legacy `HOST_BINARY`
  - Dead code: no emitter ever produced `HOST_BINARY`; executor and supported-op entry deleted
- [x] Milestone 5: remove legacy `HOST_UNARY`
  - `RECIPROCAL` lowered via `FRECIP` SXU program; `TRUNC` lowered via `F2I+I2F` SXU program
  - `_exec_host_unary` executor and `HOST_UNARY` emitter deleted
- [x] Milestone 6: simplify runtime after descriptor removal
  - `_SUPPORTED_OPS = {"SXU_PROGRAM"}`
  - `_render_legacy_descriptor` now returns only SXU descriptors (the GEMM fallback builds an SXU_PROGRAM)
  - `analyze_tinytpu_uops` remains for external consumers in `tests/onnx_tinytpu_trace/driver.py` but is unreachable from the runtime dispatch

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
- [x] Hardware epilogue for multi-K-tile GEMM — landed via PSUM
      bucket bank. Multi-K-tile GEMMs now run `SXU_PSUM_CLEAR` → N ×
      `DISPATCH_MXU(psum_acc)` → `SXU_PSUM_READ_ROW` → bias/relu →
      STORE entirely in hardware. No numpy fallback path remains.

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

## Major Hardware Capability Gaps vs tinyspec

These are the remaining hardware-level gaps that cannot be solved by
software lowering alone. Ordered roughly by impact on real workloads.

### Data path & reductions

- [x] **Float reducer tile-scope primitives**
      `VPU_FSUM_REDUCE_TILE` (opcode 38), `VPU_FMAX_REDUCE_TILE` (39),
      `VPU_FMIN_REDUCE_TILE` (40). Float sum/max scalar reductions lower
      end-to-end; float min lowers for single-tile kernels via a
      negation-around-max pattern rewrite.
- [x] **Multi-tile float min combine** — `VPU_FMIN` ALU opcode (41)
      added; renderer emits per-tile `FMIN_REDUCE_TILE` plus `VPU_FMIN`
      across tiles. Works up to 32 elements; larger sizes hit the
      pre-existing RANGE-loop reduction gap shared with other scalar
      reductions.
- [x] **Float row-reduce primitives** (`VPU_FSUM_REDUCE` 42,
      `VPU_FMAX_REDUCE` 43, `VPU_FMIN_REDUCE` 44). Per-sublane float
      reducer ops added; row-sum and row-max are wired through the
      tinygrad row renderer. Row-min still needs negation-decomp
      detection in the row path.
- [x] **Float col-reduce primitives** (`VPU_FSUM_REDUCE_COL` 45,
      `VPU_FMAX_REDUCE_COL` 46, `VPU_FMIN_REDUCE_COL` 47). Landed via
      the "hoist once outside the case block" pattern (same as the
      integer col reducers) using shared `lane_f*` helpers. Col-sum
      and col-max wired through tinygrad. Col-min pending negation
      rewrite.
- [x] **Shared multi-cycle FP tile reducer** — `mkFpReducer`
      sub-module with one FP adder + one comparator driven by a small
      FSM. Tile-scope FP reducers now dispatch through it; SXU stalls
      on `vpu.isDone` until the walk completes. Runtime build went from
      1:48 to 1:29 after replacing the 3 combinational tile FP trees,
      and adding 3 col reducer opcodes on top cost only +7s (1:36),
      meeting the <10s/opcode acceptance bar.
- [ ] **Float prod reducer** — no opcode; tinyspec `Reduce(Mul, axes)`
      on float would need a `VPU_FMUL_REDUCE*` family.

### Build-time / bsc elaboration

- [x] **Shared multi-cycle FP reducer unit** — `src/FpReducer.bsv`
      landed. Tile-scope FP reducers now route through it; acceptance
      criterion met (+7s for 3 new col reducer opcodes). Row/col
      reducers still use the "hoist once" combinational pattern with
      shared lane_f* helpers, which is cheap at the current 4×4 lane
      count. If we scale lanes or add more reducer kinds (e.g. float
      prod), consider routing row and col through FpReducer too.
- [ ] **Narrow-dtype VPU lanes**: int8, uint8, int16, uint16, uint32
      elementwise. Current VPU is int32-only on the data path (except
      MXU which is int8). Blocks quantized-int8 inference kernels outside
      the MXU, and uint dtypes entirely.
- [ ] **Mixed-precision MXU**: MXU is int8-only today. No bf16/fp16 MAC
      unit, no fp32 accumulator path for floating GEMM. Float32 GEMM
      currently can't lower at all.

### Transcendental / math

- [x] **`Exp2`/`Log2`/`Sin`/`Cos` hardware — Remez + range reduction**.
      All four primitives live in `src/TranscUnit.bsv`:
      - Opcodes: `VPU_EXP2` (51), `VPU_LOG2` (52), `VPU_SIN` (53),
        `VPU_COS` (54). All dispatch through the multi-cycle walker;
        VPU `isDone` gates SXU collect.
      - Coefficients: Remez minimax for all four (EXP2 4× peak-error
        reduction, SIN 40×, COS 27×, LOG2 8× vs Taylor).
      - EXP2 range reduction: x = n + f via `tr_trunc`/`tr_fp_to_int`/
        `tr_pow2_int` helpers; poly(f) on [-1, 1] scaled by 2^n via
        exponent-bit construction. Integer x gives exact 2^n results.
      - SIN range reduction: mod-2π + quadrant fold + sign-aware
        round-to-nearest bias. sin(π), sin(2π) etc. zero out; sin(5)
        accurate to <0.3% rel.
      - Tinygrad `code_for_op` declares `Ops.EXP2`/`LOG2`/`SIN`/`SQRT`
        hardware-supported. Renderers land Tensor-level:
        `_render_scaled_exp2` (Tensor.exp), `_render_sigmoid`
        (FMUL+EXP2+FADD+FRECIP), `_render_tanh`, `_render_scaled_sin`
        (Tensor.cos), `_render_scaled_log2` (Tensor.log), plus plain
        `_render_{exp2,log2,sin,sqrt}_sxu_program`.
      - Guards in elementwise/reciprocal/multistep-scalar-divide
        renderers reject kernels with EXP2/LOG2/SIN/SQRT so missing
        composite renderers fail loud.
      - Standalone `TbTranscUnit` (make test-transcunit) locks the
        coefficient contract at the unit layer.
      - End-to-end backend tests cover wide ranges: tanh |x|≤2.5,
        exp [-3,3], sin/cos [-3π,3π], sqrt [0.25,16].

### Movement

- [~] **`PAD` primitive** — small unrolled pads lower through the
      `_render_pad_sxu_program` renderer (PAD_FILL VMEM preload).
      Multi-tile + RANGE-loop pads still open.
- [~] **`FLIP` primitive** — unrolled flip / non-affine permutes
      lower through the pad_fill renderer too. Multi-tile FLIP and
      RANGE-loop FLIP still open.
- [ ] **`CAT` primitive** — concatenate along an axis. Could be a
      two-source SXU copy program driven by a range split, but no
      primitive exists today.
- [ ] **General `Permute`** — only `permute(1,0)` (transpose) is wired
      through `SXU_DISPATCH_XLU_TRANSPOSE`. Non-transpose axis reorders
      for ≥3D tensors have no hardware path.

### GEMM & matmul

- [x] **Multi-K-tile hardware epilogue** — landed via PSUM bucket
      bank (Item #2 below). Multi-K-tile GEMM now zeros a PSUM bucket
      via `SXU_PSUM_CLEAR`, accumulates every K-tile into row 0 via
      `DISPATCH_MXU` with `psum_acc` mode, then extracts the row with
      `SXU_PSUM_READ_ROW` and runs the existing bias+ReLU epilogue on
      the PSUM result. No numpy fallback path remains for multi-K-tile
      bias/relu.

### Control flow & ordering

- [ ] **`BARRIER` / `IF` / `ENDIF` / `AFTER` SXU ops** — SXU today is a
      straight-line sequencer. No conditional execution, no explicit
      ordering barriers for RAW hazards across dispatches.
- [ ] **`ATOMICADD` unit on VMEM banks** — no RMW lane on the VMEM write
      port. Required for scatter-add style kernels.

### Multi-device & parallelism

- [ ] **Collective primitives** (broadcast/scatter/gather/allreduce/
      reduce\_scatter) expressed as NoC-backed hardware ops. NoC exists
      at the ring level but is not exposed as a tinyspec-shaped primitive.
- [ ] **`REPLICATED` / multi-device `COPY` model** — the compiler has no
      concept of cross-chip axis today.
- [ ] **`THREEFRY` PRNG datapath** — no ARX unit. All random init
      currently falls back to host.

### Memory planning

- [ ] **VMEM spill/fill to HBM** — compiler-managed spill path for
      tensors larger than the VMEM working set. Today everything must
      fit in VMEM tiles.

## Architectural Refactors (higher-impact, bigger than opcode additions)

These are the structural improvements that came out of reviewing the
TensorCore datapath against the NeuronCore reference and thinking
about what's actually bottlenecked today. Ordered by impact/effort
ratio — the top ones would most change what TinyTPU can run.

### Tier 1 — parallelism and data movement

- [ ] **Decouple SXU into per-engine command queues.** The single
      microsequencer is the visible biggest bottleneck: every engine
      shares one issue slot. MXU already overlaps via
      `DISPATCH_MXU`+`WAIT_MXU`, but VPU / FpReducer / XLU all
      serialize even though they touch disjoint state. Split the
      front-end into an SXU fetch/decode stage feeding four small
      per-engine FIFOs, each with its own execute-side controller; add
      a scoreboard keyed on VRegFile destination index for RAW/WAW.
      Closes the biggest visible gap vs NeuronCore's per-engine SEQs.
- [x] **Dedicated PSUM accumulator bank.** Landed. `src/PSUMBank.bsv`
      is an 8-bucket × 4×4 Int32 SRAM with `write`/`accumulate`/
      `writeRow`/`accumulateRow`/`readReq`/`readResp`/`peekRow`/
      `clear` and row-granular access for MXU. Shared between
      Controller (per-dispatch row deposit via `startPsum`) and SXU
      (opcodes 15-19: `PSUM_WRITE`, `PSUM_ACCUMULATE`, `PSUM_READ`,
      `PSUM_READ_ROW`, `PSUM_CLEAR`). The tinygrad renderer uses the
      bank for multi-K-tile GEMM accumulation, which closes the
      "Multi-K-tile hardware epilogue" gap above.
- [ ] **Engine-to-engine forwarding (bypass VRegFile).** Today every
      hand-off is `engine → resultReg → VRegFile → engine`, four
      cycles. Add a muxed forwarding network on engine inputs so
      `vs`/`vs2` can come straight from the MXU result reg / VPU
      resultReg / XLU output when SXU emits a `FWD` hint instead of a
      vreg index. The current `SXU_LOAD_MXU_RESULT` special case is
      the first half of this feature — generalize it.
- [~] **Dual-issue slot for VPU + XLU.** Scaffolding landed: SXU has
      an `sxu_is_xlu_slot()` classifier (identifies XLU-side dispatches)
      plus a pair of scoreboard registers — `xlu_busy` (is an XLU
      dispatch in flight) and `xlu_dst` (target vreg). FSM is still
      single-issue; the scoreboard is the first piece the dual-issue
      arbiter will consult. Remaining: second issue slot, stall on
      RAW hazard against `xlu_dst`, writeback arbitration, test.

### Tier 2 — memory hierarchy and programmability

- [~] **Double-buffered Weight/ActivSRAM + small DMA engine.** Scaffolding
      landed: `src/WeightSRAMDB.bsv` and `src/ActivationSRAMDB.bsv` each
      hold two banks with an `active` pointer. Writes target the
      inactive bank (background preload), reads serve from the active
      bank, and `swap` flips the pointer. Standalone testbenches
      (`make test-wsram-db`, `make test-asram-db`) cover read-after-
      swap, inactive-bank-write isolation, and second-swap. Remaining:
      wire the DB variants into the Controller behind a "preload
      parallel to dispatch" mode, and a small DMA engine that issues
      the background writes from HBM.
- [ ] **Transcendental / programmable SIMD unit.** `sqrt` / `log2` /
      `exp2` / `sin` are rejected today; NeuronCore covers these with
      its GPSIMD engine. Three options in increasing generality:
      (a) polynomial-approximator opcodes with fixed hardware LUTs,
      (b) a dedicated transcendental unit next to the VPU, (c) a tiny
      programmable SIMD lane (RISC-V style). (a) is the natural first
      iteration.
- [x] **Predicate register + `SKIP_IF_ZERO` (baby `IF` / `BARRIER`).**
      Landed. `src/ScalarUnit.bsv` gains a 1-bit `pred` Reg plus two
      opcodes: `SXU_SET_PRED_IF_ZERO` (opcode 20, pred := vreg[0][0]
      == 0) and `SXU_SKIP_IF_PRED` (opcode 21, advances pc by 2 when
      pred is set and auto-resets pred). TASM opcode table and Python
      wire helpers (`_set_pred_if_zero`, `_skip_if_pred`) landed;
      runtime sim tests cover both skip-taken and skip-not-taken
      paths. No tinygrad renderer is emitting these yet — they're
      infrastructure for future BARRIER / IF / ENDIF work.
- [~] **Output-stationary dataflow mode on MXU.** Scaffolding landed:
      Controller exposes a `DataflowMode` enum (`DF_WEIGHT_STATIONARY`,
      `DF_OUTPUT_STATIONARY`), a `dfModeReg` register, and a
      `getDataflowMode` interface method. No behavior change yet —
      always stays in WS. Remaining: `startOS()` method, Controller
      FSM variant that streams both operands, PE accumulator-hold
      mode, end-to-end test.

### Tier 3 — chip-level scale-out

- [ ] **Shared on-chip L2 between TensorCores.** NoC exists, but
      cores re-fetch overlapping weights from HBM for data-parallel
      kernels. An L2 cache or an NoC broadcast primitive amortizes
      weight traffic across cores.
- [~] **Generalize `SXU_LOAD_MXU_RESULT` into engine-to-engine read
      ports.** Hardware and TASM support landed:
      `SXU_LOAD_VPU_RESULT` (13) and `SXU_LOAD_XLU_RESULT` (14) both
      copy the engine's linger result register into a vreg. Bundle
      tests cover both. Remaining: tinygrad renderer is not yet using
      these opcodes to elide VRegFile round-trips in chained kernels
      (e.g. XLU-then-VPU, VPU chains that store and re-read).
- [ ] **Per-engine writeback queues on VRegFile.** Writeback bus is a
      single return path; if VPU and XLU both finish in the same
      cycle they compete. Multi-port VRF or small per-engine
      writeback FIFOs.

### Deferred (called out but probably not next)

- Mixed-precision MXU (bf16/fp16 MACs) — big surgery, unclear ROI
  without training workloads.
- Hardware collective primitives on the ring NoC — worth the software
  plumbing first; hardware once we have multi-core workloads.
- `ATOMICADD` on VMEM banks — only matters for scatter-add kernels
  which aren't on our critical path.

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

## Recommended Next Iterations (updated 2026-04-20, end of 40-iter push)

Current state: 1006 Python tests + 87 BSV unit tests (VPU 48 +
FpReducer 7 + PSUMBank 9 + SXU 6 + SxuPSUM 2 + CtrlPSUM 1 + CtrlOS 1 +
CtrlDB 2 + WSRAMDB 3 + ASRAMDB 3 + TranscUnit 5).

Landings in this push:
- Item #6 (Transcendentals): DONE with Remez + range reduction for
  EXP2/LOG2/SIN/COS. Tensor.exp, tanh, sigmoid, cos, log, sqrt, rsqrt
  all accurate across wide ranges.
- Item #8 (OS MXU): DISPATCH PATH LIVE via Controller.startOS() + SXU
  DISPATCH_MXU_OS opcode. PE accumulator-hold FSM pending.
- Item #5 (DB SRAM): CONTROLLER INTEGRATION. `.plain` sub-interface
  lets Controller consume DB SRAMs; preload-parallel pattern proven
  in TbCtrlDB. DMA stub + TensorCore default-wiring pending.
- Item #4 (Dual-issue): SCOREBOARD LIVE. `xlu_busy` / `xlu_dst` set
  on dispatch, cleared on collect. Parallel rule + RAW-stall pending.
- Renderer additions: Tensor.cos, Tensor.log, Tensor.rsqrt,
  Tensor.square, multi-tile wide-input tanh/exp/sigmoid/cos/sin tests.

Highest-leverage follow-ups (ranked):
1. **Finish Item #8** — PE accumulator-hold mode in SystolicArray +
   Controller FSM variant swapping operand roles so DISPATCH_MXU_OS
   delivers a genuinely distinct dataflow. ~6-8 iters.
2. **Finish Item #4** — parallel XLU issue rule + RAW-hazard stall
   consuming the existing scoreboard. ~4-6 iters.
3. **Finish Item #5** — DMA stub issuing background writes, plus
   TensorCore wiring to use DB SRAMs by default. ~4-6 iters.
4. **Engine-to-engine forwarding in renderers** — emit
   SXU_LOAD_VPU_RESULT / LOAD_XLU_RESULT to elide VRegFile round-trips
   in chained kernels. ~3 iters.
5. **Composite activations**: softplus (log(1+exp(x))), gelu
   (uses tanh), x**3 (MUL(x, MUL(x, x))). ~2-3 iters each.
6. **Narrow-dtype int8 VPU** — big scope. Blocks quantized INT8
   inference kernels outside the MXU. ~8-10 iters.

## Prior Recommended Next Iterations (2026-04-12)

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
