# VPU_PAIR_ROTATE — Pairwise Lane-Rotation Primitive

**Status:** design approved 2026-05-21

## Context

This is **sub-project 1 of 3** in the program to close the architectural gaps
identified by the CODA-vs-TinyTPU comparison. The three sub-projects, run as
sequential design→build cycles:

1. **VPU pairwise-lane ops** (this spec) — a single-dispatch RoPE primitive.
2. Fused MXU-drain epilogue (`SXU_DISPATCH_MXU_EPILOGUE`).
3. VReg→MXU activation bypass for chained GEMMs.

The CODA review found the VPU is strictly lane-wise (`out[s][l] = a[s][l] op
b[s][l]`, see `src/VPU.bsv`), with no opcode coupling adjacent lanes. RoPE
inherently cross-couples adjacent feature pairs — `out[2k]` depends on both
`a[2k]` and `a[2k+1]` — so today it costs 5+ XLU/VPU instructions (rotate to
align partners, two multiplies, add/sub, rotate back). SwiGLU was found to need
no new hardware (it is `VPU_SILU` + `VPU_FMUL` on separate gate/linear tiles),
so this sub-project is scoped to **RoPE only**.

## Goal

Add one VPU opcode, `VPU_PAIR_ROTATE`, so a RoPE rotation over a tile lowers to
a **single VPU dispatch** instead of a multi-instruction XLU/VPU sequence.

## The operation

`VPU_PAIR_ROTATE` is a two-source float32 op. For each row `s` and each
adjacent column pair `p` (lanes `2p` and `2p+1`):

```
data_even = src1[s][2p]      coef carries the rotation angle θ_p:
data_odd  = src1[s][2p+1]    cos_p = src2[s][2p]
                             sin_p = src2[s][2p+1]

out[s][2p]   = data_even * cos_p  -  data_odd * sin_p
out[s][2p+1] = data_even * sin_p  +  data_odd * cos_p
```

- `src1` — the data tile (float32, IEEE-754 bits in each `Int#(32)` lane).
- `src2` — the coefficient tile: even lane of each pair holds `cos θ_p`, odd
  lane holds `sin θ_p`, both float32.
- The op is a standard counter-clockwise 2D rotation applied independently to
  each `(even, odd)` lane pair, independently per row.

This is the `data*cos + rotate_half(data)*sin` RoPE formulation, fused.

## Hardware design — `src/VPU.bsv`

1. **Enum.** Add `VPU_PAIR_ROTATE` to the `VpuOp` enum (it currently ends at
   `VPU_MIN_U32`). Group it with a comment near the float ops.

2. **Dispatch case.** Add a `VPU_PAIR_ROTATE:` arm to the per-row VPU case
   statement (alongside `VPU_FADD`/`VPU_FMUL`, currently ~line 791). Instead of
   the usual `for l in 0..lanes` lane loop, it iterates pairs
   `for p in 0..lanes/2`, reads `src1[s][2p]`, `src1[s][2p+1]`, `src2[s][2p]`,
   `src2[s][2p+1]`, and writes `row[2p]` and `row[2p+1]`. Use the existing
   float helpers: `bits2fp`, `multFP(_, _, Rnd_Nearest_Even)`,
   `addFP(_, _, Rnd_Nearest_Even)`, `fp2bits`, and the sign-flip idiom from
   `VPU_FSUB` (`b_neg.sign = !b_neg.sign`) for the subtraction.

3. **Timing / structure.** Single-cycle, like the other basic float ops — it
   is combinational logic in the case arm. It costs 4 float multiplies + 2
   float adds per pair; this is a synthesis-timing watch-item but is
   functionally correct for the Bluesim model and is not a blocker.

4. **Even-lanes invariant.** Pairing requires `lanes` to be even. Add a
   `staticAssert` (or equivalent compile-time check) that `valueOf(lanes)` is
   even, so an odd-width tile fails the build rather than silently dropping the
   last lane.

## Dispatch — no new SXU opcode

`VPU_PAIR_ROTATE` is reached through the **existing** `SXU_DISPATCH_VPU`
opcode with its `vpuOp` field set to `VPU_PAIR_ROTATE` — the same mechanism the
reduction ops (`VPU_SUM_REDUCE_COL`, etc.) already use. `src1` and `src2` are
vreg operands. No change to `src/ScalarUnit.bsv`'s opcode set, no change to the
`SxuInstr` struct.

## Coefficient tile

The coefficient tile (`src2`) is ordinary data. Software — a TASM program or
the backend — builds it: for each pair `p`, write `cos θ_p` into the even lane
and `sin θ_p` into the odd lane. The rotation angles θ_p depend on token
position and pair index; computing them is a software concern, out of hardware
scope. The hardware contract is purely "even lane = cos, odd lane = sin."

## Scope

In scope — a complete, directly-verifiable hardware slice:

- `src/VPU.bsv` — the `VPU_PAIR_ROTATE` enum value and dispatch case.
- `test/TbVPU.bsv` — a BSV unit test with directed vectors.
- `scripts/tasm.py` — assembler + disassembler syntax for the new op.
- `test/TbTinyTPURuntime.bsv` — opcode/`vpuOp` decode for the runtime testbench
  and one runtime bundle test.
- `tests/test_tinytpu_backend.py` — a runtime numeric test: assemble a bundle,
  run it on the rebuilt simulator, compare the result against a numpy RoPE
  reference.

Deferred follow-on (its own task, after the hardware op is proven):

- tinygrad backend lowering that recognizes a RoPE UOp pattern and emits
  `SXU_DISPATCH_VPU VPU_PAIR_ROTATE` plus the coefficient tile. This is
  compiler-side pattern work and matches the handoff doc's sequencing
  ("each new primitive lands in hardware first, then `tasm.py` and
  `TbTinyTPURuntime.bsv`, then one tinygrad lowering path").

## Verification

Three layers, mirroring the handoff doc's verification matrix:

1. **BSV unit** — `make test-vpu` (`test/TbVPU.bsv`). Directed vectors: a known
   `data` tile and a known `coef` tile (`cos θ`, `sin θ`) produce the expected
   rotated tile. Include θ = 0 (identity: out == data), θ = 90° (out[2p] =
   −data[2p+1], out[2p+1] = data[2p]), and a general angle. Cover all rows of
   the tile and both pairs per row.

2. **Runtime / TASM** — `scripts/tasm.py` assembles and disassembles a
   `VPU_PAIR_ROTATE` instruction (round-trips). `test/TbTinyTPURuntime.bsv`
   decodes the `vpuOp` and runs a numeric bundle.

3. **Runtime numeric** — a test in `tests/test_tinytpu_backend.py` that builds
   a bundle (load a data tile, load a coef tile, `SXU_DISPATCH_VPU
   VPU_PAIR_ROTATE`, store), runs it on the rebuilt simulator, and asserts the
   result matches a numpy RoPE reference within float tolerance.

The simulator must be rebuilt (`make runtime-tb`) after the BSV change; the
`tests/conftest.py` staleness guard enforces this.

Acceptance: all three layers pass, and the full `tests/test_tinytpu_backend.py`
suite stays green (965 passing, no regressions).

## Out of scope

- SwiGLU pairwise hardware — already expressible as `VPU_SILU` + `VPU_FMUL` on
  separate gate/linear tiles.
- A 3-source VPU encoding — the packed coefficient tile keeps the op 2-source.
- The tinygrad-backend RoPE lowering (deferred follow-on, above).
- Sub-projects 2 and 3 (fused MXU-drain epilogue; VReg→MXU bypass).

## Risks

- **Synthesis timing** — 4 float multiplies + 2 float adds in one combinational
  case arm is heavier than a plain `VPU_FMUL`. Functionally correct for the
  Bluesim model; flagged for a future synthesis pass, not a blocker here.
- **Pair-ordering convention** — the spec fixes "even lane = cos, odd lane =
  sin" and "pairs are adjacent lanes `(2p, 2p+1)` within a row." The deferred
  backend-lowering task must build coefficient tiles to match; the BSV unit
  test pins the convention so it cannot drift silently.
