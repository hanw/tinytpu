# SXU_DISPATCH_MXU_EPILOGUE — Fused MXU-Drain Epilogue

**Status:** design approved 2026-05-22

## Context

This is **sub-project 2 of 3** in the program to close the architectural gaps
from the CODA-vs-TinyTPU comparison. Sub-project 1 (`VPU_PAIR_ROTATE`) is done
and pushed. The three sub-projects:

1. VPU pairwise-lane ops — **done**.
2. Fused MXU-drain epilogue (this spec).
3. VReg→MXU activation bypass — pending.

Today a `matmul + bias + ReLU` runs as an explicit SXU sequence:
`DISPATCH_MXU → WAIT_MXU → PSUM_READ_ROW → VPU ADD → VPU RELU → STORE` — ≥4
instructions of post-GEMM work per output row, each with its own dispatch
latency. The MXU `do_drain` rule in `src/Controller.bsv` already walks the
result rows; this sub-project folds bias-add, ReLU, and an optional row
reduction into that drain so the whole epilogue is one SXU dispatch.

`SXU_DISPATCH_MXU_EPILOGUE` is unimplemented today: the SXU has 42 opcodes
(0–41), none an epilogue; `Controller.bsv` has `start`/`startPsum`/
`startAccumulate`/`startOS`/`startOsAccumulate` but no `startEpilogue`.

## Goal

Add two SXU opcodes so a fused GEMM epilogue (row-broadcast bias add, optional
ReLU, optional per-row integer reduction) executes at the MXU drain in a single
dispatch, writing the result directly to a VReg or VMEM.

## The two opcodes

- **`SXU_DISPATCH_MXU_EPILOGUE`** (opcode 42) — runs an MXU GEMM and applies a
  fused epilogue at drain. One instruction: load weights → stream activations →
  drain-with-epilogue → writeback.
- **`SXU_LOAD_EPILOGUE_STAT`** (opcode 43) — reads the optional per-row INT64
  reduction result from the last epilogue into a VReg (companion read-out;
  needed only when the reduction is enabled).

## Instruction encoding — overload existing `SxuInstr` fields

The current `SxuInstr` struct (`src/ScalarUnit.bsv`) is:
`{ op, vmemAddr, vregDst, vregSrc, vpuOp, vregSrc2, mxuWBase, mxuABase,
mxuTLen }`. **The struct is NOT widened** — widening changes the wire format and
ripples into `scripts/tasm.py` and `test/TbTinyTPURuntime.bsv` bundle parsing.
There is precedent for per-opcode field reuse: the struct's own comment notes
`SELECT` overloads `mxuWBase` ("SELECT uses low bits as rhs vreg").

For `SXU_DISPATCH_MXU_EPILOGUE`, fields are interpreted as:

| Field | Meaning for `MXU_EPILOGUE` |
|---|---|
| `mxuWBase` | GEMM weight base address |
| `mxuABase` | GEMM activation base address |
| `mxuTLen` | GEMM tile length |
| `vregSrc` | bias source vreg (its row 0 is the bias vector) |
| `vregDst` | result destination — vreg index, or VMEM tile index when `writebackMode = VMEM` |
| `vpuOp` / `vregSrc2` | the epilogue config bits (see below), packed into these otherwise-unused fields |

Epilogue config bits: `biasEnable`, `reluEnable`, `reduceEnable`, `reduceOp`
(0 = SUM, 1 = SUMSQ), `writebackMode` (0 = VREG, 1 = VMEM). Five bits, packed
into the `vpuOp` field (`vpuOp` is unused for this opcode); if `vpuOp` cannot
hold all five, the overflow bit goes in `vregSrc2`. The exact bit layout is
fixed by the implementation and documented in both `src/ScalarUnit.bsv` and
`scripts/tasm.py`.

For `SXU_LOAD_EPILOGUE_STAT`, only `vregDst` is used (destination vreg).

## Controller epilogue stage

A new method on `Controller_IFC`:

```
method Action startEpilogue(weightBase, actBase, tileLen,
                            Vector#(cols, Int#(32)) biasVec,
                            EpilogueConfig epiCfg);
```

`EpilogueConfig` is a struct of `{ biasEnable, reluEnable, reduceEnable,
reduceOp }`. The bias is passed **by value** — the SXU reads row 0 of the bias
vreg and hands the Controller the `cols`-vector, so the Controller needs no new
memory port. `startEpilogue` reuses the existing weight-load / activation-stream
machinery; only the drain changes.

The `do_drain` rule (`src/Controller.bsv`) gains an epilogue branch. For each
drained result row `acc` (`Vector#(cols, Int#(32))`), in order:

1. **Bias** — if `biasEnable`: `acc[c] = acc[c] + biasVec[c]` for all `c`
   (row-broadcast: the same `biasVec` is added to every row).
2. **ReLU** — if `reluEnable`: `acc[c] = max(acc[c], 0)`.
3. **Reduce** — if `reduceEnable`: accumulate a per-row INT64 statistic over
   the post-bias-post-ReLU `acc`:
   - `reduceOp = SUM`:   `stat[r] = Σ_c acc[c]` (each `acc[c]` sign-extended to INT64)
   - `reduceOp = SUMSQ`: `stat[r] = Σ_c acc[c]·acc[c]` (INT64 products)
4. The epilogue'd `acc` row is the result row for writeback.

The per-row INT64 `stat` vector is held in Controller state and exposed by a
new `method Vector#(rows, Int#(64)) epilogueStat`.

## Writeback

`SXU_DISPATCH_MXU_EPILOGUE` writes the epilogue'd result matrix directly to the
destination, as part of executing the one instruction:
- `writebackMode = VREG`: into vreg `vregDst`.
- `writebackMode = VMEM`: to VMEM tile `vregDst`.

The legacy `DISPATCH_MXU` + `PSUM_READ_ROW` + VPU path is left intact as the
fallback; existing GEMM tests must keep passing unchanged.

## Reduction read-out

When `reduceEnable` was set, `SXU_LOAD_EPILOGUE_STAT vregX` writes the per-row
INT64 `epilogueStat` into vreg `vregX`: row `r`'s INT64 value occupies lanes
`(r, 0)` = low 32 bits and `(r, 1)` = high 32 bits; lanes `(r, 2)` and `(r, 3)`
are written zero. The downstream RMSNorm float tail (the existing VPU `I2F` /
`FRECIP` / `EXP2` / `LOG2` ops) consumes the lo/hi pair. SUMSQ of INT32 GEMM
outputs overflows INT32 at realistic magnitudes, which is why the statistic is
INT64 and read out as lo/hi rather than truncated.

## Files

- `src/ScalarUnit.bsv` — `SXU_DISPATCH_MXU_EPILOGUE` (42) + `SXU_LOAD_EPILOGUE_STAT`
  (43) opcodes; the per-opcode field decode; the SXU FSM sequencing
  (dispatch → wait → writeback; stat read-out).
- `src/Controller.bsv` — `startEpilogue`, `EpilogueConfig`, the `do_drain`
  epilogue branch, the `epilogueStat` method.
- `scripts/tasm.py` + `tests/test_tasm.py` — assembler/disassembler syntax and
  round-trip tests for both opcodes.
- `tinygrad/tinygrad/runtime/ops_tinytpu.py` (submodule) — bundle-builder
  helpers (`_mxu_epilogue`, `_load_epilogue_stat`) and the SXU opcode map.
- `tests/test_tinytpu_backend.py` — runtime numeric tests.

Two new opcodes plus a Controller datapath change make this the largest of the
three sub-projects.

## Verification

1. **Assembler** — `tests/test_tasm.py` round-trips both opcodes (all enable/
   mode combinations encode and disassemble correctly).
2. **Runtime numeric** — tests in `tests/test_tinytpu_backend.py`, each builds a
   bundle, runs it on the rebuilt simulator, and compares to a numpy reference:
   - `matmul + bias`, VReg writeback and VMEM writeback.
   - `matmul + bias + ReLU`.
   - `matmul + bias + ReLU + reduce=SUM`, with `SXU_LOAD_EPILOGUE_STAT`,
     verifying the per-row INT64 sum (lo/hi reconstructed).
   - `matmul + bias + ReLU + reduce=SUMSQ`, verifying the per-row INT64
     sum-of-squares including a case whose value exceeds INT32 range.
   - Bias-disabled / ReLU-disabled cases (each enable bit independently off).
3. **Equivalence** — a fused-epilogue result must equal the legacy
   `DISPATCH_MXU → PSUM_READ_ROW → VPU ADD → VPU RELU` sequence on the same
   inputs.
4. **No regression** — the simulator is rebuilt (`make runtime-tb`); the full
   `tests/test_tinytpu_backend.py` suite stays green (966 passing baseline +
   the new tests), and the BSV Controller unit tests (`make test-ctrl-psum`,
   `make test-ctrl-os`, `make test-ctrl-accumulate`) still pass.

## Out of scope for v1

- Column-broadcast bias (only row-broadcast).
- The float normalization itself — `I2F` / `1/sqrt` / rescale stay with the
  existing VPU ops; the epilogue produces only the integer statistic.
- Backward-pass / dual-accumulator support.
- Widening `SxuInstr` — explicitly avoided via per-opcode field overloading.
- The tinygrad-backend lowering that emits `SXU_DISPATCH_MXU_EPILOGUE` for a
  fused `matmul+bias+relu` UOp pattern — a separate compiler-side follow-on,
  after the hardware path is proven (matches the handoff doc's "hardware first,
  then one tinygrad lowering path").

## Risks

- **Controller datapath change** — `do_drain` is on the GEMM critical path.
  The epilogue branch must not alter timing or results for the non-epilogue
  `start`/`startPsum` paths; the Controller unit tests guard this.
- **Encoding pressure** — five config bits packed into `vpuOp`/`vregSrc2`. If
  they do not fit, the fallback is to widen `SxuInstr` after all (a larger
  change); the implementation must confirm the bits fit before proceeding.
- **INT64 reduction read-out** — the lo/hi-pair convention must be matched
  exactly by the downstream consumer; the runtime tests pin it.
- **Synthesis timing** — bias-add + ReLU + INT64 multiply-accumulate folded
  into the drain stage is heavier combinational logic; functionally correct for
  the Bluesim model, flagged for a future synthesis pass.
