# SXU_DISPATCH_MXU_REQUANT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a drain-side requantization unit so a chained GEMM runs `SET_REQUANT_CONFIG → DISPATCH_MXU_REQUANT → DISPATCH_MXU (next layer)`, eliminating the VPU requant dispatches between GEMMs.

**Architecture:** Two new SXU opcodes — `SXU_SET_REQUANT_CONFIG` (44) latches `scaleMul:Int#(32)`/`scaleShift:UInt#(5)` into the Controller; `SXU_DISPATCH_MXU_REQUANT` (45) runs a GEMM and writes drain-side INT8 directly to ActivationSRAM via a small fixed-function pipeline (INT64 multiply, round, arithmetic right-shift, saturating clamp, truncate). No `SxuInstr` widening — the INT32 scale packs across four 8-bit fields; per-opcode field overloading is the established pattern. Built end-to-end as one feature, then hardened with equivalence-vs-legacy and saturation tests.

**Tech Stack:** Bluespec SystemVerilog (`src/ScalarUnit.bsv`, `src/Controller.bsv`), Python assembler (`scripts/tasm.py`), pytest, the BSV simulator.

**Spec:** `doc/sxu-mxu-requant-spec.md`

---

## Background an engineer needs

This is sub-project 3 of 3 in the CODA-gap program (SP1 `VPU_PAIR_ROTATE` + SP2 `SXU_DISPATCH_MXU_EPILOGUE` are done and pushed).

**Cross-repo.** Parent (`src/*.bsv`, `scripts/tasm.py`, `tests/`) and tinygrad submodule (`tinygrad/tinygrad/renderer/tinytpu/common.py` bundle helpers, `tinygrad/tinygrad/runtime/ops_tinytpu.py` `_SXU_OPS`). Submodule commit lands first, then the parent records the pointer.

**Sim rebuild.** After any `src/*.bsv` change, `make runtime-tb` before running `tests/test_tinytpu_backend.py` — the staleness guard enforces it. Full suite baseline: **972 passing**.

**Opcode numbers.** `_SXU` ends at `LOAD_EPILOGUE_STAT: 43`. Next free: **44** (`SET_REQUANT_CONFIG`), **45** (`DISPATCH_MXU_REQUANT`).

**`SxuInstr` struct** (`src/ScalarUnit.bsv:147–157`, unchanged): `{ op, vmemAddr:UInt#(8), vregDst:UInt#(4), vregSrc:UInt#(4), vpuOp:VpuOp(7b), vregSrc2:UInt#(4), mxuWBase:UInt#(8), mxuABase:UInt#(8), mxuTLen:UInt#(8) }`. **No widening.** Per-opcode field reuse is established (`DISPATCH_MXU` overloads vreg fields for PSUM; `DISPATCH_MXU_EPILOGUE` packs 5 config bits in `vpuOp`).

**Encoding for `SET_REQUANT_CONFIG` (op 44):** INT32 `scaleMul` packed little-endian across `mxuWBase`/`mxuABase`/`mxuTLen`/`vmemAddr` (one byte each); `scaleShift` (5 bits) in the low bits of `vpuOp`. Reassembled in the SXU as `unpack({pack(vmemAddr), pack(mxuTLen), pack(mxuABase), pack(mxuWBase)})`.

**Encoding for `DISPATCH_MXU_REQUANT` (op 45):** `mxuWBase`/`mxuABase`/`mxuTLen` = GEMM operands (as `DISPATCH_MXU`); `vmemAddr` = target ActivationSRAM base address; other fields zero.

**Key templates — cite, do not guess:**
- `src/Controller.bsv:359–377` `startPsum` — template for `startRequant`.
- `src/Controller.bsv:341–...` `do_drain` rule (already has SP2's epilogue branch at line 357); the requant branch slots in alongside the existing branches, gated on `requantActive`.
- `src/Controller.bsv:184` and lines 411/434/470/491/512 — the existing `epilogueActive <= False` pattern in non-epilogue start methods; replicate for `requantActive`.
- `src/ScalarUnit.bsv:943–957` `do_mxu` (fire-and-forget MXU dispatch template — `pc++` immediately, Controller runs asynchronously). `SXU_DISPATCH_MXU_REQUANT` follows this exact shape.
- `src/ActivationSRAM.bsv`: `method Action write(UInt#(TLog#(depth)) addr, Vector#(rows, Int#(8)) data)` — the ASRAM write port the drain calls.

**The Controller has NO VRegFile access** (established in SP2): the SXU passes any needed value into the Controller methods.

Project rules: commit messages `subsystem: description`, no `Co-Authored-By`. Do NOT push. Do not pipe output through `head`/`tail`.

---

## File Structure

| File | Repo | Responsibility |
|---|---|---|
| `scripts/tasm.py` | parent | `_SXU` 44/45 entries; assemble/disassemble arms |
| `tests/test_tasm.py` | parent | round-trip tests for both opcodes |
| `src/Controller.bsv` | parent | scale registers, `setRequantConfig`, `startRequant`, `do_drain` requant branch, `requantActive <= False` in other starts |
| `src/ScalarUnit.bsv` | parent | opcodes 44/45 + 2 states + 2 rules |
| `tinygrad/.../renderer/tinytpu/common.py` | submodule | `_set_requant_config`, `_mxu_requant` bundle helpers |
| `tinygrad/.../runtime/ops_tinytpu.py` | submodule | `_SXU_OPS` 44/45 + helper re-exports |
| `tests/test_tinytpu_backend.py` | parent | runtime numeric + equivalence + saturation + chained tests |

Three feature-incremental tasks. T2 ends with the end-to-end requant path verified numerically vs a Python reference; T3 adds the equivalence-vs-legacy, saturation, and chained-GEMM tests.

---

## Task 1: Opcodes and assembler

**Files:** Modify `scripts/tasm.py`, `tests/test_tasm.py` (parent); `tinygrad/tinygrad/runtime/ops_tinytpu.py` (submodule).

- [ ] **Step 1: Write the failing round-trip tests**

In `tests/test_tasm.py`, near the other SXU round-trip tests, add:
```python
def test_sxu_set_requant_config_roundtrip():
    # SET_REQUANT_CONFIG scale_mul=0x12345678 scale_shift=7
    # scaleMul little-endian across mxuWBase/mxuABase/mxuTLen/vmemAddr; shift in vpuOp.
    prog = "SET_REQUANT_CONFIG scale_mul=305419896 scale_shift=7\nHALT\nEND\n"
    wire = assemble(prog)
    line = next(ln for ln in wire.strip().splitlines() if ln.startswith("2 44 "))
    f = line.split()
    # field order: 2 opc vmemAddr vregDst vregSrc vpuOp vregSrc2 mxuWBase mxuABase mxuTLen
    assert int(f[7]) == 0x78    # mxuWBase = scaleMul[7:0]
    assert int(f[8]) == 0x56    # mxuABase = scaleMul[15:8]
    assert int(f[9]) == 0x34    # mxuTLen  = scaleMul[23:16]
    assert int(f[2]) == 0x12    # vmemAddr = scaleMul[31:24]
    assert int(f[5]) == 7        # vpuOp low 5 bits = scaleShift
    assert "SET_REQUANT_CONFIG" in disassemble(wire)
    assert "scale_mul=305419896" in disassemble(wire)
    assert "scale_shift=7" in disassemble(wire)

def test_sxu_dispatch_mxu_requant_roundtrip():
    # DISPATCH_MXU_REQUANT runs GEMM(WMEM[<w>], AMEM[<a>], tiles=<t>) and writes
    # requantized INT8 result to ActivationSRAM[<dst>].
    prog = "DISPATCH_MXU_REQUANT WMEM[3] AMEM[5] tiles=2 ASRAM[9]\nHALT\nEND\n"
    wire = assemble(prog)
    line = next(ln for ln in wire.strip().splitlines() if ln.startswith("2 45 "))
    f = line.split()
    assert int(f[7]) == 3     # mxuWBase
    assert int(f[8]) == 5     # mxuABase
    assert int(f[9]) == 2     # mxuTLen
    assert int(f[2]) == 9     # vmemAddr = ASRAM target base
    dis = disassemble(wire)
    assert "DISPATCH_MXU_REQUANT" in dis
    assert "WMEM[3]" in dis and "AMEM[5]" in dis and "tiles=2" in dis and "ASRAM[9]" in dis
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/hanwang/p/tinytpu && .venv/bin/python3 -m pytest tests/test_tasm.py::test_sxu_set_requant_config_roundtrip tests/test_tasm.py::test_sxu_dispatch_mxu_requant_roundtrip -q`
Expected: FAIL — unknown opcodes.

- [ ] **Step 3: Add the opcodes to `_SXU`**

In `scripts/tasm.py`, after `"LOAD_EPILOGUE_STAT": 43,` add:
```python
    "LOAD_EPILOGUE_STAT":       43,
    "SET_REQUANT_CONFIG":       44,
    "DISPATCH_MXU_REQUANT":     45,
}
```

- [ ] **Step 4: Add assemble + disassemble arms**

In `scripts/tasm.py`, add assembler keyword arms:

For `SET_REQUANT_CONFIG`: parse `scale_mul=<int>` and `scale_shift=<int>`. Split `scaleMul` (mask to 32 bits — accept signed input) into four little-endian bytes; emit `_instr(_SXU["SET_REQUANT_CONFIG"], vmemAddr=(m>>24)&0xff, vpuOp=shift&0x1f, mxuWBase=m&0xff, mxuABase=(m>>8)&0xff, mxuTLen=(m>>16)&0xff)`.

For `DISPATCH_MXU_REQUANT`: parse `WMEM[<w>] AMEM[<a>] tiles=<t> ASRAM[<dst>]`. Emit `_instr(_SXU["DISPATCH_MXU_REQUANT"], vmemAddr=dst, mxuWBase=w, mxuABase=a, mxuTLen=t)`.

In the disassembler `case` on opcode, add arms for 44 and 45 reconstructing the mnemonics from the wire fields (reassemble `scaleMul` from the four bytes — `m = vmemAddr<<24 | mxuTLen<<16 | mxuABase<<8 | mxuWBase`, then sign-correct if `m & 0x80000000` so `disassemble` round-trips signed values). Follow the existing `DISPATCH_MXU` assemble/disassemble arms (`tasm.py:401–441`/`763–774`) for structure.

- [ ] **Step 5: Add to submodule `_SXU_OPS`**

In `tinygrad/tinygrad/runtime/ops_tinytpu.py`, `_SXU_OPS` currently ends `"DISPATCH_MXU_EPILOGUE": 42, "LOAD_EPILOGUE_STAT": 43}`. Change to:
```python
             ..., "LOAD_EPILOGUE_STAT": 43,
             "SET_REQUANT_CONFIG": 44, "DISPATCH_MXU_REQUANT": 45}
```

- [ ] **Step 6: Run the assembler tests**

Run: `cd /Users/hanwang/p/tinytpu && .venv/bin/python3 -m pytest tests/test_tasm.py -q`
Expected: ALL pass including the 2 new round-trips. If there is a `test_*ops_cover_full_range`-style assertion for the SXU opcode count, update it (the SP1 task reported there is none for SXU — verify).

- [ ] **Step 7: Commit (submodule first)**

```bash
cd /Users/hanwang/p/tinytpu/tinygrad && git add tinygrad/runtime/ops_tinytpu.py && git commit -m "tinytpu: add requant opcodes to _SXU_OPS map"
cd /Users/hanwang/p/tinytpu && git add scripts/tasm.py tests/test_tasm.py tinygrad && git commit -m "tasm: assembler support for requant opcodes"
```

---

## Task 2: Controller + SXU end-to-end + helpers + numeric test

**Files:**
- `src/Controller.bsv` (parent) — registers, 2 methods, `do_drain` requant branch, `requantActive <= False` in 6 other start methods.
- `src/ScalarUnit.bsv` (parent) — `SxuOpCode` 44/45, `SxuState` 2 new, `do_fetch` decode arms, 2 new rules.
- `tinygrad/.../renderer/tinytpu/common.py` (submodule) — 2 bundle helpers.
- `tinygrad/.../runtime/ops_tinytpu.py` (submodule) — helper re-exports.
- `tests/test_tinytpu_backend.py` (parent) — `test_mxu_requant_matches_reference`.

Drain rebuild is mandatory after the BSV changes.

- [ ] **Step 1: Write the failing runtime numeric test**

In `tests/test_tinytpu_backend.py`, class `TestTinyTPUSimOutputParsing`, add:
```python
  def test_mxu_requant_matches_reference(self):
    # Fused drain-side requant: GEMM result × scaleMul, round, arithmetic
    # right shift, saturating clamp to INT8, write to ActivationSRAM. Compare
    # to a Python reference applying the same pipeline.
    import struct
    sim = os.environ["TINYTPU_SIM"]
    from tinygrad.runtime.ops_tinytpu import _set_requant_config, _mxu_requant
    # Identity weights + activation row -> per-PE diag (matches the
    # established Task-2 epilogue test pattern); the requant unit then scales
    # and clamps each lane independently.
    a_row     = [3, -5, 7, -9]            # one activation vector
    scale_mul = 65536                     # 1.0 in fixed-point (mul + shift = 16)
    scale_sh  = 16
    asram_dst = 12
    bundle = _bundle(
      _wmem(0, [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),     # identity weights
      _amem(0, a_row),                                      # one activation row
      _set_requant_config(scale_mul, scale_sh),
      _mxu_requant(0, 0, 1, asram_dst),                    # tiles=1
      _output_asram(asram_dst),
      _halt(),
      _end(),
    )
    out = _run_bundle(sim, bundle)
    got = _parse_asram_output(out)        # Vector#(rows, Int#(8)) at slot
    # Python reference of the exact hardware pipeline:
    def requant(acc, mul, sh):
      wide = acc * mul
      rounded = wide + (1 << (sh - 1))
      shifted = rounded >> sh           # arithmetic right shift (Python //)
      return max(-128, min(127, shifted))
    expected = [requant(v, scale_mul, scale_sh) for v in a_row]
    self.assertEqual(got, expected, f"requant: got {got}, expected {expected}")
```
The helpers `_output_asram(slot)` and `_parse_asram_output(out)` mirror `_output_vmem`/`_parse_vmem_output` — if they do not exist in `tinygrad/.../runtime/ops_tinytpu.py` already, look for an existing ASRAM-output helper and use it; if there is none, the implementer adds a minimal one alongside the other `_output_*`/`_parse_*` helpers. (Read the surrounding helpers in `common.py`/`ops_tinytpu.py` and follow the established pattern.)

- [ ] **Step 2: Add the bundle helpers (submodule)**

In `tinygrad/tinygrad/renderer/tinytpu/common.py`, near `_mxu_epilogue`, add:
```python
def _set_requant_config(scale_mul: int, scale_shift: int) -> str:
    # opcode 44. scaleMul packed little-endian: byte 0 -> mxuWBase, byte 1 ->
    # mxuABase, byte 2 -> mxuTLen, byte 3 -> vmemAddr. scaleShift in vpuOp.
    m = scale_mul & 0xFFFFFFFF
    b0, b1, b2, b3 = m & 0xff, (m >> 8) & 0xff, (m >> 16) & 0xff, (m >> 24) & 0xff
    return f"2 44 {b3} 0 0 {scale_shift & 0x1f} 0 {b0} {b1} {b2}"

def _mxu_requant(wbase: int, abase: int, tiles: int, asram_dst: int) -> str:
    # opcode 45. ASRAM target base in vmemAddr; GEMM operands in mxu* fields.
    return f"2 45 {asram_dst} 0 0 0 0 {wbase} {abase} {tiles}"
```
Add both to the `from tinygrad.renderer.tinytpu.common import (...)` re-export in `ops_tinytpu.py`.

- [ ] **Step 3: Run the test to verify failure**

`cd /Users/hanwang/p/tinytpu && PYTHONPATH=tinygrad .venv/bin/python3 -m pytest "tests/test_tinytpu_backend.py::TestTinyTPUSimOutputParsing::test_mxu_requant_matches_reference" -q --tb=short` — expect FAIL (the BSV doesn't know about these opcodes yet).

- [ ] **Step 4: Add the opcodes to `SxuOpCode` and the new states to `SxuState`**

In `src/ScalarUnit.bsv`, the `SxuOpCode` enum currently ends with `SXU_DISPATCH_MXU_EPILOGUE, SXU_LOAD_EPILOGUE_STAT }`. Change to:
```bsv
               // (existing SP2 entries)
               SXU_DISPATCH_MXU_EPILOGUE, SXU_LOAD_EPILOGUE_STAT,
               // SP3: drain-side requantization
               SXU_SET_REQUANT_CONFIG, SXU_DISPATCH_MXU_REQUANT }
   SxuOpCode deriving (Bits, Eq, FShow);
```
In `SxuState`, append `SXU_EXEC_SET_REQUANT_CONFIG, SXU_EXEC_MXU_REQUANT` (alongside the SP2 epilogue states).

- [ ] **Step 5: Add the decode arms in `do_fetch`**

In `src/ScalarUnit.bsv`'s `do_fetch` rule, alongside the SP2 epilogue decode arms, add:
```bsv
                  SXU_SET_REQUANT_CONFIG:    pc_state <= SXU_EXEC_SET_REQUANT_CONFIG;
                  SXU_DISPATCH_MXU_REQUANT:  pc_state <= SXU_EXEC_MXU_REQUANT;
```

- [ ] **Step 6: Add Controller registers + interface methods**

In `src/Controller.bsv`:

a. Module-level registers (alongside the SP2 `epilogueActive` / `biasReg` / etc. registers near line 184):
```bsv
   Reg#(Int#(32)) scaleMul             <- mkReg(0);
   Reg#(UInt#(5)) scaleShift           <- mkReg(0);
   Reg#(Bool)     requantActive        <- mkReg(False);
   Reg#(UInt#(8)) requantTargetBase    <- mkReg(0);
   Reg#(UInt#(8)) requantTargetOffset  <- mkReg(0);
```

b. Add to `Controller_IFC` (after the SP2 epilogue method declarations):
```bsv
   method Action setRequantConfig(Int#(32) mul, UInt#(5) shift);
   method Action startRequant(UInt#(TLog#(depth)) weightBase,
                              UInt#(TLog#(depth)) actBase,
                              UInt#(TLog#(depth)) tileLen,
                              UInt#(8) asramTargetBase);
```

c. Implement (model `startRequant` exactly on `startPsum` at `Controller.bsv:359–377`):
```bsv
   method Action setRequantConfig(Int#(32) mul, UInt#(5) shift) if (cstate == Idle || cstate == Done);
      scaleMul   <= mul;
      scaleShift <= shift;
   endmethod

   method Action startRequant(UInt#(TLog#(depth)) weightBase,
                              UInt#(TLog#(depth)) actBase,
                              UInt#(TLog#(depth)) tileLen,
                              UInt#(8) asramTargetBase) if (cstate == Idle || cstate == Done);
      wBase        <= weightBase;
      aBase        <= actBase;
      tLen         <= tileLen;
      actIdx       <= 0;
      streamCycle  <= 0;
      firstActRead <= False;
      dfModeReg    <= DF_WEIGHT_STATIONARY;
      array.clearAll;
      requantActive       <= True;
      requantTargetBase   <= asramTargetBase;
      requantTargetOffset <= 0;
      cstate <= LoadWeights;
   endmethod
```

d. In every other start method (`start`, `startPsum`, `startAccumulate`, `startOS`, `startOsAccumulate`, `startEpilogue` — six methods), add `requantActive <= False;` (mirror the `epilogueActive <= False;` line already present in each).

- [ ] **Step 7: Add the `do_drain` requant branch**

In `src/Controller.bsv`'s `do_drain` rule (currently around line 341), alongside the SP2 epilogue branch (`if (epilogueActive) ...` around line 357), add — using the already-snapshotted `drainMatrix` local that the SP2 fix introduced (it's the lifted `array.getMatrix` read; if the variable name differs, match what's in the file):
```bsv
   if (requantActive) begin
      Vector#(cols, Int#(8)) reqRow = newVector;
      for (Integer ci = 0; ci < valueOf(cols); ci = ci + 1) begin
         Int#(32) acc     = drainRow[ci];   // drainRow = array.getResults — the single WS column-sum row
         Int#(64) wide    = signExtend(acc) * signExtend(scaleMul);
         Int#(64) one     = 1;
         Int#(64) rounded = wide + (one << (scaleShift - 1));
         Int#(64) shifted = rounded >> scaleShift;
         Int#(64) clamped = (shifted >  127) ?  127 :
                            (shifted < -128) ? -128 : shifted;
         reqRow[ci] = truncate(clamped);
      end
      aSRAM.write(truncate(requantTargetBase + requantTargetOffset), reqRow);
      requantTargetOffset <= requantTargetOffset + 1;
   end
```
Place this branch in the existing `if (epilogueActive) { ... }` / `case (psumModeReg)` sequence — they are mutually exclusive at runtime because each is gated by a flag set in `startEpilogue` / `startPsum` / `startRequant`. If `bsc` flags a rule-scheduling conflict from the multiple `if`s, restructure as `if (requantActive) ... else if (epilogueActive) ... else case (psumModeReg) ...`.

NOTE: WS `do_drain` uses `let r = array.getResults` (a `Vector#(cols, Int#(32))`) — the requant branch reduces over this single row. Use the local that already holds it (the SP2 fix lifted `array.getMatrix` to `drainMatrix` for the epilogue branch; the WS single-row result is the existing `r` / `outputBuf`-bound variable. Match the actual variable name in the file.)

- [ ] **Step 8: Add the two SXU rules**

In `src/ScalarUnit.bsv`, alongside the SP2 epilogue rules:
```bsv
rule do_set_requant_config (pc_state == SXU_EXEC_SET_REQUANT_CONFIG);
   `ifdef TRACE
   $display("TRACE cycle=%0d unit=SXU ev=SET_REQUANT_CONFIG", cyc);
   `endif
   Bit#(32) m = { pack(curInstr.vmemAddr), pack(curInstr.mxuTLen),
                  pack(curInstr.mxuABase), pack(curInstr.mxuWBase) };
   Bit#(5)  s = truncate(pack(curInstr.vpuOp));
   ctrl.setRequantConfig(unpack(m), unpack(s));
   pc <= pc + 1;
   pc_state <= SXU_FETCH;
endrule

rule do_mxu_requant (pc_state == SXU_EXEC_MXU_REQUANT);
   `ifdef TRACE
   $display("TRACE cycle=%0d unit=SXU ev=DISPATCH_MXU_REQUANT", cyc);
   `endif
   ctrl.startRequant(truncate(curInstr.mxuWBase),
                     truncate(curInstr.mxuABase),
                     truncate(curInstr.mxuTLen),
                     curInstr.vmemAddr);
   pc <= pc + 1;
   pc_state <= SXU_FETCH;
endrule
```
`do_mxu_requant` is fire-and-forget (mirrors `do_mxu` at `ScalarUnit.bsv:943–957`); a following `WAIT_MXU` or another dispatch then synchronizes.

- [ ] **Step 9: Rebuild the simulator**

`cd /Users/hanwang/p/tinytpu && make runtime-tb` — must succeed; if `bsc` errors on rule scheduling in `do_drain`, restructure per Step 7's fallback.

- [ ] **Step 10: Run the test — verify it passes; run the full suite**

```bash
cd /Users/hanwang/p/tinytpu
PYTHONPATH=tinygrad .venv/bin/python3 -m pytest "tests/test_tinytpu_backend.py::TestTinyTPUSimOutputParsing::test_mxu_requant_matches_reference" -q --tb=short
PYTHONPATH=tinygrad .venv/bin/python3 -m pytest tests/test_tinytpu_backend.py -q --tb=short
make test-ctrl-psum
make test-ctrl-os
```
Expect: numeric test passes; full suite **973 passed, 0 failed**; Controller unit tests pass. If a Controller test regresses, STOP and report BLOCKED.

- [ ] **Step 11: Commit**

```bash
cd /Users/hanwang/p/tinytpu/tinygrad && git add tinygrad/renderer/tinytpu/common.py tinygrad/runtime/ops_tinytpu.py && git commit -m "tinytpu: _set_requant_config / _mxu_requant bundle helpers"
cd /Users/hanwang/p/tinytpu && git add src/Controller.bsv src/ScalarUnit.bsv tests/test_tinytpu_backend.py tinygrad && git commit -m "controller,sxu: drain-side requant unit"
```

---

## Task 3: Saturation + parameterized numeric + chained-GEMM tests

**Files:** Modify `tests/test_tinytpu_backend.py` (parent). No BSV change.

**Note on test design.** The spec called for an "equivalence-vs-legacy" test, but there is no SXU runtime opcode that writes INT8 to ActivationSRAM through any legacy path — `_amem` is a bundle-setup record, not a runtime instruction, and no `STORE_ASRAM` SXU opcode exists. The drain-side requant *is* the enabling capability for SXU-runtime ASRAM writes, not a faster version of an existing one. The test design instead pairs (a) a saturation test, (b) a parameterized numeric test over multiple `scaleMul`/`scaleShift` combinations against the spec's Python pipeline (the spec is the independent reference), and (c) a chained-GEMM end-to-end test that proves the requantized output is consumable by a downstream `DISPATCH_MXU` and the final result matches a Python reference.

- [ ] **Step 1: Write the saturation test**

Add to `TestTinyTPUSimOutputParsing`:
```python
  def test_mxu_requant_saturates(self):
    # Drive scaled values outside INT8 range; assert clamp to 127 / -128.
    # Identity weights => per-PE result is diag(activation); the requant unit
    # then scales each diagonal value independently.
    sim = os.environ["TINYTPU_SIM"]
    from tinygrad.runtime.ops_tinytpu import _set_requant_config, _mxu_requant
    a_row     = [50, -50, 200, -200]    # GEMM result diag = a_row (identity W)
    scale_mul = 256                      # acc * 256
    scale_sh  = 8                        # then >> 8  →  effective identity
    asram_dst = 3
    bundle = _bundle(
      _wmem(0, [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
      _amem(0, a_row),
      _set_requant_config(scale_mul, scale_sh),
      _mxu_requant(0, 0, 1, asram_dst),
      _output_asram(asram_dst),
      _halt(), _end(),
    )
    out = _run_bundle(sim, bundle)
    got = _parse_asram_output(out)
    # 50 and -50 are in INT8 range; 200 clamps to 127; -200 clamps to -128.
    expected = [50, -50, 127, -128]
    self.assertEqual(got, expected, f"requant saturation: got {got}, expected {expected}")
```

- [ ] **Step 2: Write the parameterized numeric test**

Add to `TestTinyTPUSimOutputParsing`:
```python
  def test_mxu_requant_pipeline_matches_python(self):
    # The requant pipeline is fully specified: wide = acc * mul; rounded
    # = wide + (1 << (shift-1)); shifted = rounded >> shift; clamp [-128,127];
    # truncate to INT8. Exercise multiple (mul, shift, input) combinations
    # against an independent Python implementation of that pipeline.
    sim = os.environ["TINYTPU_SIM"]
    from tinygrad.runtime.ops_tinytpu import _set_requant_config, _mxu_requant
    def py_requant(acc, mul, sh):
      wide    = acc * mul
      rounded = wide + (1 << (sh - 1))
      shifted = rounded >> sh             # Python // is arithmetic right shift
      return max(-128, min(127, shifted))
    # (a_row, scale_mul, scale_shift) cases — covers identity, fractional
    # scaling, rounding boundaries, and a negative-input case.
    cases = [
      ([10, -7, 4, -3],   16384, 14),    # ~identity (mul/shift ≈ 1.0)
      ([100, 50, 25, 0],  32768, 16),    # mul=0.5
      ([1, 2, 3, 4],      65536, 15),    # mul=2.0
      ([-50, -1, 0, 1],   16384, 14),    # negatives at boundaries
    ]
    for ci, (a_row, mul, sh) in enumerate(cases):
      asram_dst = 16 + ci
      bundle = _bundle(
        _wmem(ci, [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
        _amem(ci, a_row),
        _set_requant_config(mul, sh),
        _mxu_requant(ci, ci, 1, asram_dst),
        _output_asram(asram_dst),
        _halt(), _end(),
      )
      out = _run_bundle(sim, bundle)
      got = _parse_asram_output(out)
      expected = [py_requant(v, mul, sh) for v in a_row]
      self.assertEqual(got, expected,
        f"case {ci} (mul={mul}, sh={sh}, a={a_row}): got {got}, expected {expected}")
```

- [ ] **Step 3: Write the chained-GEMM test**

Add to `TestTinyTPUSimOutputParsing`:
```python
  def test_mxu_requant_feeds_chained_gemm(self):
    # Two-layer chain proves the requant output is consumable by a downstream
    # DISPATCH_MXU. Sequence: GEMM1 → drain-requant → ASRAM[K] → GEMM2 reads
    # ASRAM[K] as activation. Final result must match the spec pipeline
    # applied between the two matmuls.
    import numpy as np
    sim = os.environ["TINYTPU_SIM"]
    from tinygrad.runtime.ops_tinytpu import _set_requant_config, _mxu_requant
    # Pick small enough magnitudes so the GEMM1 results stay well within the
    # post-scale INT8 range (no saturation in this test).
    W1   = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]   # identity (4x4)
    A1   = [3, -4, 5, -6]                          # one activation row
    W2   = [2,0,0,0, 0,3,0,0, 0,0,1,0, 0,0,0,2]   # diagonal scaling 4x4
    mul, sh = 16384, 14                            # ≈ identity scaling
    asram_chain = 24
    psum_addr   = 0
    bundle = _bundle(
      _wmem(0, W1),
      _amem(0, A1),
      _wmem(1, W2),
      _set_requant_config(mul, sh),
      _mxu_requant(0, 0, 1, asram_chain),          # GEMM1 + requant → ASRAM[24]
      _wait_mxu(),
      _mxu_psum_write(1, asram_chain, 1, psum_addr, 0),  # GEMM2 reads ASRAM[24]
      _wait_mxu(),
      _psum_read_row(2, psum_addr, 0),
      _store(2, 5),
      _halt(),
      _output_vmem(5),
      _end(),
    )
    out = _run_bundle(sim, bundle)
    got = _parse_vmem_output(out)
    # Python reference: GEMM1 (identity) gives A1; requantize each element;
    # GEMM2 with diag W2 multiplies each lane by its diagonal coefficient.
    def py_requant(acc, m, s):
      r = acc * m + (1 << (s - 1))
      return max(-128, min(127, r >> s))
    requantized = [py_requant(v, mul, sh) for v in A1]   # INT8 activations
    diag = [W2[i * 4 + i] for i in range(4)]              # diagonal of W2
    expected = [requantized[i] * diag[i] for i in range(4)]
    self.assertEqual(got[:4], expected,
      f"chained GEMM: got {got[:4]}, expected {expected}")
```
If the `_wait_mxu` / `_mxu_psum_write` / `_psum_read_row` / `_wmem` / `_amem` argument shapes differ slightly, model them on an existing PSUM/GEMM runtime test in the same file (e.g. `test_mxu_psum_accumulate_via_bundle` or `test_multi_wmma_8x8_at_8x8_matches_numpy`).

- [ ] **Step 4: Run all SP3 tests + full suite + Controller unit tests**

```bash
cd /Users/hanwang/p/tinytpu
PYTHONPATH=tinygrad .venv/bin/python3 -m pytest "tests/test_tinytpu_backend.py" -q --tb=short -k test_mxu_requant
PYTHONPATH=tinygrad .venv/bin/python3 -m pytest tests/test_tinytpu_backend.py -q --tb=short
make test-ctrl-psum
make test-ctrl-os
make test-ctrl-accumulate
make test-sxu
```
Expect: all `test_mxu_requant*` pass; full suite **976 passed, 0 failed** (972 baseline + 1 from Task 2 + 3 here = 976); all `make test-*` pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/hanwang/p/tinytpu && git add tests/test_tinytpu_backend.py && git commit -m "tests: drain-side requant equivalence + saturation + chained-GEMM"
```

---

## Notes for the executor

- **Submodule first** for Task 1 and Task 2 commits; the parent commit records the pointer (`git add tinygrad`). Do not push.
- **Rebuild discipline** — after any `src/*.bsv` change, `make runtime-tb` before pytest. The staleness guard will halt the suite otherwise.
- **`requantActive` gate** — every non-requant dispatch must set `requantActive <= False`. The six methods are `start`, `startPsum`, `startAccumulate`, `startOS`, `startOsAccumulate`, `startEpilogue`. A missed one would cause a non-requant GEMM to wrongly run the requant branch.
- **Three mutually-exclusive `do_drain` branches** — PSUM-mode case, `epilogueActive`, `requantActive`. If `bsc` flags a scheduling conflict, switch to `if/else if/else` ordering as noted in Task 2 Step 7.
- **BLOCKED conditions**: BSV build error, Controller-test regression, rule-scheduling conflict that the `if/else` fallback doesn't resolve. Report the exact `bsc` error; do not thrash.
- **Test counts**: 972 baseline → 973 after T2 → 976 after T3. Each task's full-suite run must show 0 failed.
- The deferred follow-on (out of scope for this plan) is the tinygrad-backend lowering that emits `SET_REQUANT_CONFIG` + `DISPATCH_MXU_REQUANT` for a quantized chained-GEMM UOp pattern.
