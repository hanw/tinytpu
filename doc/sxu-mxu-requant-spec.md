# SXU_DISPATCH_MXU_REQUANT — Drain-Side Requantization Unit

**Status:** design approved 2026-05-22

## Context

This is **sub-project 3 of 3** in the program to close the architectural gaps
from the CODA-vs-TinyTPU review. SP1 (`VPU_PAIR_ROTATE`) and SP2
(`SXU_DISPATCH_MXU_EPILOGUE`) are done and pushed; this is the last.

The CODA review framed the gap as "VReg→MXU activation bypass to eliminate the
VMEM round-trip between chained GEMMs." Investigation found that framing
misdiagnoses the real cost: on TinyTPU the activation hop is already on-chip
(`ActivationSRAM`); the actual overhead in a chained
`GEMM → requant → GEMM` is the **VPU-side requantization** (multiply-by-scale,
shift, clamp, truncate — several VPU dispatches). The systolic array consumes
INT8 activations but produces INT32 accumulators, so a chain *must*
requantize, and today that work falls on the VPU.

The fix: add a small fixed-function requantization stage at the MXU drain
port. A drain-side `SXU_DISPATCH_MXU_REQUANT` writes the requantized INT8
result directly to `ActivationSRAM`, ready to feed the next GEMM. Zero VPU
cycles for the requant step.

## Goal

Add two SXU opcodes so a quantized chained GEMM runs as:
`SET_REQUANT_CONFIG → DISPATCH_MXU_REQUANT → DISPATCH_MXU (next layer)` —
replacing the legacy `DISPATCH_MXU → PSUM_READ_ROW → VPU mul-by-scale → VPU
shift → VPU clamp → VPU cast → STORE-to-ActivationSRAM` sequence.

## The two opcodes

- **`SXU_SET_REQUANT_CONFIG`** (opcode 44) — latches the Controller's scale
  registers `scaleMul:Int#(32)` and `scaleShift:UInt#(5)`. Typically issued
  once per quant-recipe change (per layer / per kernel).
- **`SXU_DISPATCH_MXU_REQUANT`** (opcode 45) — runs a GEMM and writes the
  drain-side requantized INT8 result directly to `ActivationSRAM` at a
  configured base address. **No parallel INT32 writeback in v1** — if both
  are needed, use a separate `DISPATCH_MXU` for the INT32 path.

## The requant pipeline (combinational, in `do_drain`)

For each drained INT32 lane `acc`:

```
wide     = (Int#(64)) signExtend(acc) × (Int#(64)) signExtend(scaleMul)
rounded  = wide + (Int#(64)) (1 << (scaleShift - 1))     // round-to-nearest
shifted  = rounded >>> scaleShift                        // arithmetic right shift
clamped  = (shifted >  127) ?  127 :
           (shifted < -128) ? -128 : shifted             // saturating clamp to INT8
out8     = truncate(clamped)                              // Int#(32) → Int#(8)
```

Per drained row: all `cols` lanes processed in parallel → one
`Vector#(rows, Int#(8))` (an activation vector) written to ActivationSRAM at
`requantTargetBase + requantTargetOffset`, then `requantTargetOffset += 1`.
The `cols × cols`-wide INT64 multiply is combinational logic — functionally
correct for the Bluesim model, flagged as a synthesis watch-item.

## Controller path

New registers in `mkController`:
- `Reg#(Int#(32)) scaleMul`, `Reg#(UInt#(5)) scaleShift` — latched by
  `setRequantConfig`.
- `Reg#(Bool) requantActive`, `Reg#(UInt#(8)) requantTargetBase`,
  `Reg#(UInt#(8)) requantTargetOffset`.

New `Controller_IFC` methods:
- `setRequantConfig(Int#(32) mul, UInt#(5) shift)` — guarded so it cannot
  fire mid-dispatch; writes the two scale registers.
- `startRequant(UInt#(TLog#(depth)) weightBase, UInt#(TLog#(depth)) actBase,
  UInt#(TLog#(depth)) tileLen, UInt#(8) asramTargetBase)` — mirrors
  `startPsum` (same `wBase`/`aBase`/`tLen`/`actIdx`/`streamCycle`/
  `firstActRead` writes, `dfModeReg <= DF_WEIGHT_STATIONARY`,
  `array.clearAll`, `cstate <= LoadWeights`), plus
  `requantActive <= True; requantTargetBase <= asramTargetBase;
  requantTargetOffset <= 0;`.
- All other start methods (`start`, `startPsum`, `startAccumulate`, `startOS`,
  `startOsAccumulate`, `startEpilogue`) set `requantActive <= False` — the
  same guard pattern SP2 uses for `epilogueActive`.

`do_drain` extension — alongside the existing PSUM and (SP2) epilogue
branches, add:
```bsv
   if (requantActive) begin
      Vector#(cols, Int#(8)) reqRow = newVector;
      for (Integer ci = 0; ci < valueOf(cols); ci = ci + 1) begin
         Int#(64) wide    = signExtend(drainRow[ci]) * signExtend(scaleMul);
         Int#(64) rounded = wide + (1 << (scaleShift - 1));
         Int#(64) shifted = rounded >>> scaleShift;
         Int#(64) clamped = (shifted >  127) ?  127 :
                            (shifted < -128) ? -128 : shifted;
         reqRow[ci] = truncate(clamped);
      end
      aSRAM.write(truncate(requantTargetBase + requantTargetOffset), reqRow);
      requantTargetOffset <= requantTargetOffset + 1;
   end
```
The existing PSUM/`resultsMatrix` paths are untouched; a requant dispatch in
v1 does not also write PSUM.

## Instruction encoding (no `SxuInstr` widening)

`SxuInstr` stays at its current 9 fields. Per-opcode field overloading,
established by `DISPATCH_MXU` (psum-mode in `vregSrc2`) and `SELECT`
(`mxuWBase` low-bits as rhs vreg).

**`SXU_SET_REQUANT_CONFIG` (op 44):** the INT32 `scaleMul` is packed
little-endian across the four 8-bit fields:
- `mxuWBase`  → `scaleMul[7:0]`
- `mxuABase`  → `scaleMul[15:8]`
- `mxuTLen`   → `scaleMul[23:16]`
- `vmemAddr`  → `scaleMul[31:24]`

`scaleShift` (5 bits) goes in the `vpuOp` field's low 5 bits (the 7-bit
`vpuOp` field comfortably holds 5 bits, same approach as SP2). All other
fields zero. The SXU reassembles `Int#(32) m = unpack({pack(vmemAddr),
pack(mxuTLen), pack(mxuABase), pack(mxuWBase)});` and reads `scaleShift`
via `truncate(pack(curInstr.vpuOp))`.

**`SXU_DISPATCH_MXU_REQUANT` (op 45):**
- `mxuWBase` / `mxuABase` / `mxuTLen` — GEMM weight base / act base / tile
  length (same as `DISPATCH_MXU`).
- `vmemAddr` — target ActivationSRAM base address for the requantized output.
- `vregDst` / `vregSrc` / `vregSrc2` / `vpuOp` — zero in v1.

## SXU FSM additions

New states in `SxuState`:
- `SXU_EXEC_SET_REQUANT_CONFIG` — single-cycle: extracts `scaleMul`/
  `scaleShift` from the packed fields, calls `ctrl.setRequantConfig(...)`,
  advances `pc`.
- `SXU_EXEC_MXU_REQUANT` — single-cycle: calls
  `ctrl.startRequant(mxuWBase, mxuABase, mxuTLen, vmemAddr)`, advances `pc`
  (fire-and-forget, like `do_mxu`). Following code uses `WAIT_MXU` to stall on
  `ctrl.isDone` before issuing the next dispatch (same convention as the
  legacy MXU path).

New decode arms in `do_fetch`:
- `SXU_DISPATCH_MXU_REQUANT: pc_state <= SXU_EXEC_MXU_REQUANT;`
- `SXU_SET_REQUANT_CONFIG: pc_state <= SXU_EXEC_SET_REQUANT_CONFIG;`

## Files

- `src/ScalarUnit.bsv` — opcodes 44/45 in `SxuOpCode`; states; decode arms;
  two new execute rules (`do_set_requant_config`, `do_mxu_requant`); TRACE
  instrumentation matching the surrounding rules.
- `src/Controller.bsv` — the four new registers; `setRequantConfig` and
  `startRequant` methods; the `do_drain` requant branch; the
  `requantActive <= False` guards in all other start methods.
- `scripts/tasm.py` + `tests/test_tasm.py` — assembler/disassembler syntax
  and round-trip tests for both opcodes.
- `tinygrad/tinygrad/renderer/tinytpu/common.py` (submodule) —
  `_mxu_requant` and `_set_requant_config` bundle helpers, packing the
  encoding above.
- `tinygrad/tinygrad/runtime/ops_tinytpu.py` (submodule) — `_SXU_OPS`
  entries for the two opcodes; helper re-exports.
- `tests/test_tinytpu_backend.py` — runtime numeric tests.

## Verification

1. **Assembler round-trip** — `tests/test_tasm.py` round-trips both opcodes,
   checking the packed `scaleMul` field reassembly is byte-correct and that
   `scaleShift` round-trips.
2. **Numeric vs numpy** — assemble a `SET_REQUANT_CONFIG` + `DISPATCH_MXU_REQUANT`
   bundle with known `scaleMul`/`scaleShift`/inputs; run on the simulator;
   read back the resulting ActivationSRAM contents; compare to a Python
   reference that applies the exact pipeline (multiply, round, shift, clamp,
   truncate).
3. **Equivalence-vs-legacy** — run the same GEMM+requant two ways and assert
   byte-identical ActivationSRAM contents:
   - (a) `SET_REQUANT_CONFIG` + `DISPATCH_MXU_REQUANT` — the fused path.
   - (b) `DISPATCH_MXU` + `PSUM_READ_ROW` per row + VPU sequence (multiply
     by scale, shift via SHR, clamp via MIN/MAX, truncate via PACKED_I8 cast)
     + `STORE` to ActivationSRAM — the legacy path. The VPU sequence must
     implement the *same* arithmetic the hardware unit does.
4. **Saturation** — drive inputs so that some lanes' post-scale value exceeds
   INT8 range; assert the output clamps to 127 / -128 (not garbage).
5. **Chained-GEMM end-to-end** — a follow-on `DISPATCH_MXU` reads the
   requantized result from ActivationSRAM and produces a second GEMM output;
   compare the final result to the same chain done legacy-style. This proves
   the requant output lands at the right ASRAM addresses and is consumable.
6. **No regression** — `make runtime-tb` rebuilds the sim; full
   `tests/test_tinytpu_backend.py` stays green (972 baseline + the new
   tests); `make test-ctrl-psum`, `test-ctrl-os`, `test-ctrl-accumulate`,
   `test-sxu` still pass.

## Out of scope for v1

- **Per-channel scales** — would need a `Vector#(cols, Int#(32))` scale tile
  and `cols` parallel multipliers. Future extension.
- **Asymmetric quantization** — no zero-point register; symmetric only.
- **OS-mode dispatch** — only WS in v1 (matches how `DISPATCH_MXU` is
  partitioned from `DISPATCH_MXU_OS`).
- **Parallel INT32 + INT8 writeback** — a single dispatch is either
  requant-INT8 OR PSUM-INT32, not both.
- **The tinygrad-backend lowering** that recognizes a `matmul → quantize →
  matmul` UOp pattern and emits the new opcodes — compiler-side follow-on,
  once the hardware is proven.

## Risks

- **Controller datapath change.** `do_drain` is on the GEMM critical path;
  the new branch is gated on `requantActive` (default False) so non-requant
  dispatches are byte-identical. The Controller unit tests
  (`test-ctrl-psum`/`test-ctrl-os`/`test-ctrl-accumulate`) guard this.
- **Rule-scheduling conflict.** Adding `aSRAM.write` inside `do_drain`
  (which already calls `psum.writeRow`/`accumulateRow` and writes
  `epilogueBuf`/`epilogueStatBuf`) may surface a scheduling conflict in
  `bsc`. The PSUM and epilogue branches are mutually exclusive at runtime;
  the requant branch is also mutually exclusive (only one of `psumModeReg
  != PSUM_OFF`, `epilogueActive`, `requantActive` is true per dispatch).
  If `bsc` cannot prove that statically, an explicit `if-else if-else`
  chain over those three flags resolves it.
- **Synthesis timing.** The combinational INT64 multiply + shift + clamp in
  the drain stage is heavier than the existing PSUM path. Functionally
  correct for the Bluesim model, flagged as a synth watch-item, not a
  v1 blocker.
- **Wire-format byte order for `scaleMul`.** The little-endian packing across
  four 8-bit fields must round-trip exactly through assemble/disassemble —
  the `tests/test_tasm.py` round-trip test pins it.
