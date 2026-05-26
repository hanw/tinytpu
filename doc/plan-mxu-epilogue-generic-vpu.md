# Generic-VPU MXU Epilogue Plan

> Sub-project 3 of the CODA-gap program. Sub-project 1 (`VPU_PAIR_ROTATE`)
> and 2 (`SXU_DISPATCH_MXU_EPILOGUE` bias/relu/reduce) are done.

**Goal:** Extend the fused MXU-drain epilogue so it can apply **any VPU
opcode** against a **tile-shape src2**, not just the hardcoded
bias(row) + relu + reduce pipeline. With this in place a CODA-style
single-bundle kernel becomes the default for every epilogue primitive
(residual, RoPE, SwiGLU, partial-RMS, cross-entropy, …) instead of the
multi-bundle GEMM→VMEM→VPU chain the runtime emits today.

**Architecture:** Add `SXU_DISPATCH_MXU_VPU_EPILOGUE` (op 46). The
Controller's drain stage embeds a VPU functional unit; on drain it
applies the configured VPU op lane-wise between the drained accumulator
matrix and a caller-supplied tile-shape src2. Result lands in
`epilogueBuf` exactly like the existing bias+relu path. Writeback
(VREG vs VMEM) reuses the existing `cfg.bit4` convention.

**Tech Stack:** Bluespec (`src/Controller.bsv`, `src/ScalarUnit.bsv`,
`src/VPU.bsv`), Python assembler (`scripts/tasm.py`), pytest, BSV sim.

**Spec:** This doc (no separate spec — small enough to inline).

---

## Why this is the structural win

The CODA paper's central claim is *composable epilogue primitives*:
every epilogue is a short program that runs once per GEMM output tile,
without ever spilling the accumulator to memory. Today's TinyTPU
epilogue can do exactly three things — bias-add, ReLU, INT64 reduce —
all of which must be enabled or disabled in lockstep by `EpilogueConfig`
bits. Anything else (rotate, gate, residual-add, scale) forces the
backend to drain to VMEM and dispatch a separate VPU bundle, which then
incurs:

1. An OUTPUT_VMEM emit + host re-upload between bundles (current runtime).
2. A second SXU dispatch sequence (LOAD_VREG x2 / VPU / STORE_VREG / HALT).

Both vanish if the same VPU op runs *during the drain*. Crucially this
is one local change that unlocks every CODA epilogue primitive — not a
per-primitive opcode (`IPAIR_ROTATE`, `SWIGLU`, …). The set of useful
epilogues is then exactly the set of useful VPU ops, which is exactly
what CODA's composability argument promises.

---

## Current state (verified read)

**`EpilogueConfig`** (`src/Controller.bsv:46-51`) is a 4-bool struct
encoded in `cfg[0:3]` of the SXU `vpuOp` field:

```bsv
typedef struct {
   Bool biasEnable;
   Bool reluEnable;
   Bool reduceEnable;
   Bool reduceSumsq;
} EpilogueConfig deriving (Bits, Eq, FShow);
```

**`startEpilogue`** (`src/Controller.bsv:70-74`) takes a single
`biasVec : Vector#(cols, Int#(32))` — one row, not a tile. Bias is
row-broadcast over all `rows` during drain.

**Drain loop** (`src/Controller.bsv:398-407`) is hardcoded:

```bsv
Int#(32) v = m[ri][ci];
if (epiCfgReg.biasEnable) v = v + biasReg[ci];
if (epiCfgReg.reluEnable && v < 0) v = 0;
outm[ri][ci] = v;
```

No VPU instance is involved — the bias+relu is open-coded combinational
logic. The same rule then computes the optional INT64 reduce stat
(lines 409-419).

**SXU dispatch** (`src/ScalarUnit.bsv:1040-1058`) reads `biasTile` from
the source vreg, takes **row 0** as the bias vector, packs the four
config bits, and calls `ctrl.startEpilogue(..., biasTile[0], ec)`. The
remaining rows of `biasTile` are discarded.

**Writeback** (`src/ScalarUnit.bsv:1071-1084`) picks vreg or VMEM based
on `cfg[4]`. `cfg[5]` and `cfg[6]` are unused today.

---

## Target state

Add **`SXU_DISPATCH_MXU_VPU_EPILOGUE`** (op 46) with the same
field overloading as op 42 except:

- `vpuOp` field carries the **actual `VpuOp`** (e.g. `VPU_PAIR_ROTATE`,
  `VPU_ADD`, `VPU_MUL`, …) — not the 4-bool config bit-packing.
- `vregSrc` holds **the full tile-shape src2** (all `rows × cols`
  lanes), not a single-row bias.
- Writeback mode (VREG vs VMEM) needs to move out of the `vpuOp` field
  into a different bit. Two options (see Decision below).
- Reduce stat is dropped from this opcode. If a downstream reduction is
  needed it's a separate dispatch (callers that needed bias+reduce keep
  using op 42).

**New `EpilogueConfig` variant** for the generic path:

```bsv
typedef struct {
   VpuOp                     op;         // applied lane-wise: out = vpu(drain, src2tile)
   Vector#(rows, Vector#(cols, Int#(32))) src2; // full tile
   Bool                      writebackVmem;
} GenericEpilogueConfig deriving (Bits, Eq, FShow);
```

The existing `startEpilogue` stays for backward compatibility; add
**`startGenericEpilogue`** that takes the new config.

**Drain loop change** in `Controller.bsv:398-407`: when the generic
epilogue is active, route `drainMatrix` and `src2Reg` through an
embedded VPU instance, write the VPU result to `epilogueBuf`. This is
the meaningful microarchitectural delta.

---

## ISA encoding (op 46)

| Field | Meaning |
|---|---|
| `mxuWBase` | GEMM weight base |
| `mxuABase` | GEMM activation base |
| `mxuTLen` | GEMM tile length |
| `vregSrc` | src2 tile vreg (all `rows × cols` lanes used) |
| `vregDst` | result vreg (when writeback = VREG) |
| `vmemAddr` | result VMEM address (when writeback = VMEM) |
| `vpuOp` | the actual `VpuOp` to apply lane-wise |
| `vregSrc2` | bit0 = writeback mode (0=VREG, 1=VMEM); bits 1-3 reserved |

**Decision needed:** writeback bit placement.

- **Option A** (chosen above): repurpose `vregSrc2`'s low bit as a
  writeback selector. Keeps `vpuOp` clean as a true `VpuOp`. Loses
  `vregSrc2` as a third operand (no current use case needs it for an
  MXU epilogue).
- **Option B**: split `vpuOp` field — say bit6 = writeback mode, bits
  0-5 = `VpuOp` (caps the enum at 64 values; currently 84).
  Doesn't fit — `VPU_PAIR_ROTATE` is opcode 84.

→ Option A.

---

## Implementation slices (one commit per slice)

### Slice 1: pure refactor, no behavior change
- Add `SXU_DISPATCH_MXU_VPU_EPILOGUE = 44` to `SxuOpCode` enum.
- Wire SXU rule that falls through to UNSUPPORTED (no behavior yet).
- Add Python opcode 46 to `_SXU_OPS`.
- Sanity: backend suite still passes (no kernel hits op 46 yet).

### Slice 2: Controller plumbing
- Add `startGenericEpilogue(..., src2Tile, vpuOp, wbVmem)` method
  returning when `cstate == Idle || cstate == Done`.
- Add `src2Reg`, `vpuOpReg`, `wbVmemReg` state.
- Drain rule: when `genericEpilogueActive`, dispatch `vpuOpReg` lane-
  wise against `drainMatrix` and `src2Reg`, write VPU result to
  `epilogueBuf`. Reuse the embedded VPU module already used elsewhere,
  OR (simpler) inline a per-lane combinational switch over the small
  set of int VPU ops the epilogue actually needs (ADD, MUL, MAX,
  IPAIR_ROTATE, IADD_RELU). Decide after profiling area/timing — start
  with the inline switch.

### Slice 3: SXU dispatch
- Implement `SXU_EXEC_MXU_VPU_EPILOGUE` state that:
  - Reads `vregSrc` as the full src2 tile (all sublanes).
  - Reads `vregSrc2.bit0` as writeback selector.
  - Calls `ctrl.startGenericEpilogue(..., src2Tile, curInstr.vpuOp, wbVmem)`.
- Writeback rule analogous to existing `do_mxu_epilogue_wb`.

### Slice 4: Assembler + Python bundle helper
- `scripts/tasm.py`: encode/decode op 46 with the new field semantics.
- New `_mxu_vpu_epilogue` helper alongside `_mxu_epilogue` in the
  tinygrad common bundle helpers.
- Round-trip test in `tests/test_tasm.py`.

### Slice 5: BSV unit test
- New cases in `test/TbController.bsv` (or a focused testbench) feeding
  a tiny GEMM + a tile-shape src2, asserting expected output for each
  of: VPU_ADD, VPU_MUL, VPU_IPAIR_ROTATE (after slice 6).
- Make target: `make test-controller`.

### Slice 6: Add VPU_IPAIR_ROTATE (deferred sub-project 1)
- Integer variant of `VPU_PAIR_ROTATE`. Trivial 10-line BSV change.
- Pulled in here because the first useful generic-epilogue test case
  is RoPE.

### Slice 7: Backend recognizer (deferred sub-project 2)
- ✅ **Residual `(D + R)`**: `_extract_wmma_epilogue` already detects a
  FULL-shape bias add; `lower_gemm` now sets `fuse_residual=True` when
  `bias_mode == "FULL" and num_k_tiles == 1 and not has_relu` and
  `_generate_gemm_sxu_instructions` emits one `_load(src2) +
  _mxu_vpu_epilogue` pair per output tile instead of the legacy
  `_mxu + _wait + _load_mxu_result + _load(bias) + _vpu(ADD) + _store`
  chain (6 → 2 instructions per tile). Verified by
  `TestResidualFusion`.
- 🟡 **RoPE `(D * C + swap(D) * S)`**: requires a UOp-tree recognizer
  for the multiplication + add + swap pattern. Infrastructure is
  in place (op 46 + `VPU_IPAIR_ROTATE` + Controller fusion path);
  only the lowering pattern is missing. Tracked as a follow-up — the
  recognizer is essentially a structural match in
  `tinygrad/renderer/tinytpu/`, similar in shape to
  `_extract_wmma_epilogue` but inspecting the post-WMMA UOp graph.

### Slice 8: End-to-end test
- 🟡 **RoPE 1-vs-5**: blocked on the RoPE recognizer in slice 7.
  When that lands, update `test_rope_pairwise_full_e2e` (or add a
  `_fused` variant) to assert the RoPE pipeline runs as **1 sim
  invocation**, not 5.
- ✅ **Residual correctness**: `tests/test_e2e_pipeclean.py`'s
  `test_residual_block_with_bias` already exercises the FULL-bias
  path and remains green after the slice-7 lowering change, which
  validates the fused op-46 output is byte-identical to the legacy
  chain.

---

## Risks / open questions

1. **VPU op set in the drain loop.** Inlining a switch over all 84
   `VpuOp` values inside the Controller's drain rule will blow up
   synthesis area and timing. Two paths:
   - Restrict to a curated subset (ADD, MUL, MAX, IPAIR_ROTATE, …
     ~10 ops). Document the restriction in the spec.
   - Truly invoke the VPU module — needs careful handshaking because
     the existing VPU is methodful, not combinational, and the drain
     is a single-cycle rule today.
   → Start with curated inline subset; revisit if subset grows >20.
2. **VRF read width.** Reading a full `rows × cols` tile from a vreg
   in one cycle is the same width as existing tile reads — verify.
3. **VMEM writeback bandwidth.** Writing a tile to VMEM in the drain
   cycle vs. a separate cycle. The existing `do_mxu_epilogue_wb` rule
   already does this; reuse it.
4. **Reduce stat with generic epilogue.** Out of scope; if a caller
   needs both, they use op 42 (bias+reduce) or chain a second
   dispatch.
5. **Backend pattern fragility.** Slice 7 recognizers will only fire
   for exactly the formulations they pattern-match. Out-of-pattern
   user code (e.g. `D + R` written as `D + 1*R`) won't fuse. Document
   the canonical forms.

---

## What success looks like

The same end-to-end RoPE test that today produces 5 VCDs produces
**one** VCD, and the bundle dump shows a single SXU dispatch:

```
2 46  <cfg> <dst> <src=cs_tile> 84(IPAIR_ROTATE) <wb_bit> <wbase> <abase> <tlen>
2 7   HALT
6 <out_addr>
4
```

The same recognizer fires for any of the CODA epilogue primitives whose
op is in the curated VPU subset. New primitives become a one-line
backend recognizer + a new VPU opcode if not already present.

---

## Decision gate

Per `AGENT.md`, BSV changes need explicit user approval. Approving this
plan implies approval for slices 1-6 (BSV + ISA work, ~300 lines BSV +
ISA wiring). Slices 7-8 are Python only.

Estimated effort: 4-6 iterations of the standard Primary Loop. Each
slice ends in a working commit with passing tests; the legacy bias+relu
path (op 42) is preserved throughout.
