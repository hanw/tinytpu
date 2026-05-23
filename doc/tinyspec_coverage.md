# TinyTPU Tinyspec Coverage

This table summarizes current tinyspec-facing coverage for the TinyTPU backend.
`tinygrad/spec/tinyspec.tex` is the canonical Tinyspec source; this document is
the current TinyTPU backend profile summary. The detailed next-iteration
checklist lives in `TODO.md`, and the sync plan lives in
`doc/plan-tinyspec-tinytpu-sync.md`.

## Lowering Classes

| Class | Status | Implementation | Notes |
| --- | --- | --- | --- |
| `ELEMENTWISE` | Partial | `tinygrad/tinygrad/renderer/tinytpu/elementwise.py` | Per-element ALU/WHERE/casts/copy/const-fill/scalar broadcast over supported dtypes and shapes. |
| `REDUCTION` | Partial | `tinygrad/tinygrad/renderer/tinytpu/reduction.py` | Scalar, row, and column reductions for supported integer/float reducer forms. |
| `BROADCAST` | Partial | `tinygrad/tinygrad/renderer/tinytpu/broadcast.py` | Row/column broadcast and selected broadcast+WHERE patterns. |
| `MOVEMENT` | Partial | `tinygrad/tinygrad/renderer/tinytpu/movement.py` | Supported single-tile pad/flip/non-affine scatter, 4x4 transpose, and row-broadcast copy patterns. |
| `GEMM` | Partial | `tinygrad/tinygrad/renderer/tinytpu/gemm.py` | 4x4 tiled int8 MXU operands with int32 buffers, including supported batched/deep-K/wide-N paths and selected epilogues. |

## Tinyspec Areas

| Area | Status | Notes |
| --- | --- | --- |
| Source/storage ops | Partial | Host buffers are bytearrays; runtime binds `PARAM` buffers into VMEM, WMEM, and AMEM bundle records. |
| Elementwise ops | Partial | Integer, bool, float, transcendental, cast, select, scalar-const, and multi-tile paths are supported where the TinyTPU VPU has an opcode or InstSel rewrite. |
| Reductions | Partial | ADD/MAX/MIN/PROD variants lower through row/column/tile reducer opcodes when the reducer, dtype, and shape are recognized. |
| Broadcast | Partial | Scalar, row, and column broadcasts lower through TinyTPU broadcast primitives for recognized patterns. |
| Movement ops | Partial | Some movement lowers through VMEM prefill or XLU transpose; arbitrary gather/scatter/indexing remains unsupported. |
| GEMM / matmul | Partial | MXU path is int8 hardware operands with int32 accumulation/output buffers; unsupported shapes/dtypes return lowering diagnostics. |
| Multi-device semantics | Open | Tuple-device copy, replicated axes, and collectives are not currently TinyTPU runtime features. |
| Control/program ops | Open | General `BARRIER`, `SPECIAL`, `IF`/`ENDIF`, `CUSTOM`, `CUSTOM_FUNCTION`, and atomics are not exposed as TinyTPU backend semantics. |
| Profiler/debuggability | Partial | Bundle parsing, VMEM records, bundle dump, trace parsing, lowering dumps, cycle reports, and Perfetto emission exist. |

## TinyTPU Opcode Inventory

The names below are intentionally mirrored from the current runtime opcode
tables. `scripts/check_tinyspec_tinytpu_profile.py` checks that new names are
added here or to the sync plan.

### VPU

`ADD`, `MUL`, `MAX`, `SUM_REDUCE`, `CMPLT`, `CMPNE`, `SUB`, `CMPEQ`,
`MAX_REDUCE`, `SHL`, `SHR`, `MIN`, `MIN_REDUCE`, `DIV`, `AND`, `OR`, `XOR`,
`FADD`, `FMUL`, `FSUB`, `FMAX`, `FCMPLT`, `FRECIP`, `I2F`, `F2I`, `NOT`,
`SELECT`, `COPY`, `SUM_REDUCE_COL`, `MAX_REDUCE_COL`, `MIN_REDUCE_COL`,
`SUM_REDUCE_TILE`, `MAX_REDUCE_TILE`, `MIN_REDUCE_TILE`, `MUL_REDUCE`,
`MUL_REDUCE_COL`, `MUL_REDUCE_TILE`, `FSUM_REDUCE_TILE`,
`FMAX_REDUCE_TILE`, `FMIN_REDUCE_TILE`, `FMIN`, `FSUM_REDUCE`,
`FMAX_REDUCE`, `FMIN_REDUCE`, `FSUM_REDUCE_COL`, `FMAX_REDUCE_COL`,
`FMIN_REDUCE_COL`, `FPROD_REDUCE_TILE`, `FPROD_REDUCE`,
`FPROD_REDUCE_COL`, `EXP2`, `LOG2`, `SIN`, `COS`, `PACKED_I8_ADD`,
`PACKED_I8_SUB`, `PACKED_I8_MAX`, `PACKED_I8_MIN`, `PACKED_I8_NEG`,
`PACKED_I8_RELU`, `PACKED_I8_CMPLT`, `PACKED_I8_CMPEQ`,
`PACKED_I8_MUL_LOW`, `PACKED_I8_MUL_HIGH`, `PACKED_I8_ABS`, `SIGN`,
`PACKED_I8_SIGN`, `FSIGN`, `ARGMIN`, `ARGMAX`, `CLZ`, `POPCOUNT`, `CTZ`,
`BYTE_REVERSE`, `SAT_ADD_I32`, `SAT_SUB_I32`, `ABS_DIFF_I32`,
`PACKED_I8_ABS_DIFF`, `FABS`, `ROTL`, `ROTR`, `MIN_U32`, `MAX_U32`,
`PAIR_ROTATE`.

### SXU

`LOAD_VREG`, `STORE_VREG`, `DISPATCH_VPU`, `DISPATCH_XLU_BROADCAST`,
`DISPATCH_MXU`, `WAIT_MXU`, `LOAD_MXU_RESULT`, `HALT`, `DISPATCH_SELECT`,
`BROADCAST_SCALAR`, `BROADCAST_ROW`, `BROADCAST_COL`,
`DISPATCH_XLU_TRANSPOSE`, `LOAD_VPU_RESULT`, `LOAD_XLU_RESULT`,
`PSUM_WRITE`, `PSUM_ACCUMULATE`, `PSUM_READ`, `DISPATCH_MXU_EPILOGUE`,
`LOAD_EPILOGUE_STAT`, `SET_REQUANT_CONFIG`, `DISPATCH_MXU_REQUANT`.
