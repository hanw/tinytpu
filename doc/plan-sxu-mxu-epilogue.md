# SXU_DISPATCH_MXU_EPILOGUE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fused MXU-drain epilogue (`SXU_DISPATCH_MXU_EPILOGUE` + `SXU_LOAD_EPILOGUE_STAT`) so a GEMM with row-broadcast bias, optional ReLU, and an optional per-row INT64 reduction runs in one SXU dispatch.

**Architecture:** A new SXU opcode internally sequences dispatch → wait → writeback. The Controller gets a `startEpilogue` method and an epilogue branch in the existing `do_drain` rule (bias-add → ReLU → INT64 reduce, all combinational over the drained matrix). Built feature-incrementally: bias-only end-to-end first, then ReLU + VMEM writeback, then the reduction. The legacy `DISPATCH_MXU` path is untouched and stays the fallback.

**Tech Stack:** Bluespec SystemVerilog (`src/ScalarUnit.bsv`, `src/Controller.bsv`), Python assembler (`scripts/tasm.py`), pytest, the BSV simulator.

**Spec:** `doc/sxu-mxu-epilogue-spec.md`

---

## Background an engineer needs

This is sub-project 2 of the CODA-gap program (sub-project 1, `VPU_PAIR_ROTATE`, is done).

**Cross-repo.** Changes span the parent repo (`src/*.bsv`, `scripts/tasm.py`, `tests/`) and the tinygrad submodule (`tinygrad/tinygrad/renderer/tinytpu/common.py` bundle helpers, `tinygrad/tinygrad/runtime/ops_tinytpu.py` `_SXU_OPS` map). Submodule commits land first, then the parent records the pointer (`git add tinygrad`). The simulator (`build/mkTbTinyTPURuntime.bexe`) must be rebuilt with `make runtime-tb` after any `src/*.bsv` change — `tests/conftest.py` aborts the suite against a stale sim. Full `tests/test_tinytpu_backend.py` baseline: **966 passing**. Commit messages `subsystem: description`, no `Co-Authored-By`. Do not push. Do not pipe command output through `head`/`tail`.

**Opcode numbers.** The `SxuOpCode` enum in `src/ScalarUnit.bsv` (lines 12–145) ends at `SXU_DISPATCH_VPU_BG` = 41. Next free: **42** (`SXU_DISPATCH_MXU_EPILOGUE`), **43** (`SXU_LOAD_EPILOGUE_STAT`).

**`SxuInstr` struct** (`src/ScalarUnit.bsv:147–157`): `{ op, vmemAddr:UInt#(8), vregDst:UInt#(4), vregSrc:UInt#(4), vpuOp:VpuOp(7b), vregSrc2:UInt#(4), mxuWBase:UInt#(8), mxuABase:UInt#(8), mxuTLen:UInt#(8) }`. **Not widened.** Per-opcode field overloading is established (`SELECT` overloads `mxuWBase`; `DISPATCH_MXU` overloads `vregDst`/`vregSrc`/`vregSrc2` for PSUM).

**Encoding for `SXU_DISPATCH_MXU_EPILOGUE`** (op 42):

| Field | Meaning |
|---|---|
| `mxuWBase` | GEMM weight base |
| `mxuABase` | GEMM activation base |
| `mxuTLen` | GEMM tile length |
| `vregSrc` | bias source vreg (row 0 = bias vector) |
| `vregDst` | result vreg (when writeback = VREG) |
| `vmemAddr` | result VMEM address (when writeback = VMEM) |
| `vpuOp` | 7-bit config: bit0 `biasEnable`, bit1 `reluEnable`, bit2 `reduceEnable`, bit3 `reduceOp` (0=SUM,1=SUMSQ), bit4 `writebackMode` (0=VREG,1=VMEM) |
| `vregSrc2` | unused (0) |

Config values are 0–31, all below the 85-value `VpuOp` range, so they round-trip cleanly through the `VpuOp`-typed `vpuOp` field; the SXU reads them with `pack(curInstr.vpuOp)` as `Bit#(7)`.

**Encoding for `SXU_LOAD_EPILOGUE_STAT`** (op 43): only `vregDst` (destination vreg).

**Key templates** (cite these exact locations when implementing — do not guess):
- `Controller.bsv:359–377` — `startPsum` method body (template for `startEpilogue`).
- `Controller.bsv:308–340` — `do_drain` rule (where the epilogue branch inserts, before `cstate <= Done`).
- `Controller.bsv:14–24` — `ControlState` enum; `45–100` — `Controller_IFC`.
- `ScalarUnit.bsv:943–957` — `do_mxu` rule (template: how the SXU calls the Controller).
- `ScalarUnit.bsv:1012–1021` — `do_wait_mxu` rule (template: stall on `ctrl.isDone`).
- `ScalarUnit.bsv:768–778` — `do_load_mxu_result` rule (template: write `ctrl.results` to a vreg, "row 0 only, rest zeroed").
- `ScalarUnit.bsv:383–391` — `do_store` rule (template: `vmem.write` for VMEM writeback; `vrf.read` idiom).
- `ScalarUnit.bsv:190–207` — `SxuState` enum (new states append here); `295–350` — `do_fetch` decode case.

**Drained matrix.** In `do_drain`, `array.getMatrix` yields the full `Vector#(rows, Vector#(cols, Int#(32)))` result; `array.getResults` is one row. The epilogue operates on the **full matrix** from `getMatrix`.

**Controller has no VRegFile access** — the bias is passed *by value*: the SXU reads row 0 of the bias vreg and passes the `Vector#(cols, Int#(32))` into `startEpilogue`.

---

## File Structure

| File | Repo | Responsibility |
|---|---|---|
| `scripts/tasm.py` | parent | `_SXU` opcode entries + assemble/disassemble for the 2 opcodes |
| `tests/test_tasm.py` | parent | assembler round-trip tests |
| `src/Controller.bsv` | parent | `startEpilogue`, `EpilogueConfig`, the `do_drain` epilogue branch, `epilogueStat` |
| `src/ScalarUnit.bsv` | parent | the 2 opcodes, SXU FSM states + rules |
| `tinygrad/.../renderer/tinytpu/common.py` | submodule | `_mxu_epilogue`, `_load_epilogue_stat` bundle helpers |
| `tinygrad/.../runtime/ops_tinytpu.py` | submodule | `_SXU_OPS` entries + helper re-exports |
| `tests/test_tinytpu_backend.py` | parent | runtime numeric tests |

Four feature-incremental tasks. Tasks 2–4 each end with a green full suite.

---

## Task 1: Opcodes and assembler

**Files:** Modify `scripts/tasm.py`, `tests/test_tasm.py` (parent); `tinygrad/tinygrad/runtime/ops_tinytpu.py` (submodule).

- [ ] **Step 1: Write failing assembler round-trip tests**

In `tests/test_tasm.py`, near the other SXU round-trip tests, add:

```python
def test_sxu_mxu_epilogue_roundtrip():
    # opcode 42; config bits packed in the vpuOp field (index 5 of the wire instr)
    prog = "MXU_EPILOGUE v3 = GEMM(WMEM[0], AMEM[0], tiles=1) BIAS=v2 RELU DST_VREG\nHALT\nEND\n"
    wire = assemble(prog)
    line = next(ln for ln in wire.strip().splitlines() if ln.startswith("2 42 "))
    assert "MXU_EPILOGUE" in disassemble(wire)

def test_sxu_load_epilogue_stat_roundtrip():
    prog = "LOAD_EPILOGUE_STAT v5\nHALT\nEND\n"
    wire = assemble(prog)
    line = next(ln for ln in wire.strip().splitlines() if ln.startswith("2 43 "))
    assert line.split()[3] == "5"   # vregDst field
    assert "LOAD_EPILOGUE_STAT v5" in disassemble(wire)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/hanwang/p/tinytpu && .venv/bin/python3 -m pytest tests/test_tasm.py::test_sxu_mxu_epilogue_roundtrip tests/test_tasm.py::test_sxu_load_epilogue_stat_roundtrip -q`
Expected: FAIL — unknown opcodes.

- [ ] **Step 3: Add the opcodes to `scripts/tasm.py`**

In the `_SXU` dict (lines 31–74), after `"DISPATCH_VPU_BG": 41,`, add:
```python
    "DISPATCH_VPU_BG":        41,
    "DISPATCH_MXU_EPILOGUE":  42,
    "LOAD_EPILOGUE_STAT":     43,
```

- [ ] **Step 4: Add assemble + disassemble support in `scripts/tasm.py`**

In the assembler keyword dispatch, add an `elif` arm for `MXU_EPILOGUE` that parses `MXU_EPILOGUE v<dst> = GEMM(WMEM[<w>], AMEM[<a>], tiles=<t>) BIAS=v<b> [RELU] [REDUCE_SUM|REDUCE_SUMSQ] DST_VREG|DST_VMEM[<addr>]`, builds the 7-bit config (bit0 bias-enabled iff `BIAS=` present, bit1 `RELU`, bit2 reduce present, bit3 `REDUCE_SUMSQ`, bit4 `DST_VMEM`), and emits `_instr(_SXU["DISPATCH_MXU_EPILOGUE"], vmemAddr=<vmem_dst>, vregDst=<dst>, vregSrc=<b>, vpuOp=<config>, mxuWBase=<w>, mxuABase=<a>, mxuTLen=<t>)`. Add a `LOAD_EPILOGUE_STAT v<d>` arm emitting `_instr(_SXU["LOAD_EPILOGUE_STAT"], vregDst=<d>)`. In the disassembler `case` on opcode, add arms reconstructing both mnemonics from the fields. Follow the existing `DISPATCH_MXU` assemble/disassemble arms (`tasm.py:401–441`, `763–774`) for structure.

- [ ] **Step 5: Add the opcodes to the submodule `_SXU_OPS` map**

In `tinygrad/tinygrad/runtime/ops_tinytpu.py`, find the `_SXU_OPS` dict (~line 73) and add `"DISPATCH_MXU_EPILOGUE": 42, "LOAD_EPILOGUE_STAT": 43` so it stays in sync with `tasm.py`'s `_SXU`.

- [ ] **Step 6: Run the assembler tests**

Run: `cd /Users/hanwang/p/tinytpu && .venv/bin/python3 -m pytest tests/test_tasm.py -q`
Expected: PASS — all green including the two new round-trip tests.

- [ ] **Step 7: Commit**

```bash
cd /Users/hanwang/p/tinytpu/tinygrad && git add tinygrad/runtime/ops_tinytpu.py && git commit -m "tinytpu: add MXU epilogue opcodes to _SXU_OPS map"
cd /Users/hanwang/p/tinytpu && git add scripts/tasm.py tests/test_tasm.py tinygrad && git commit -m "tasm: assembler support for MXU epilogue opcodes"
```

---

## Task 2: Bias-only fused epilogue, end-to-end

This is the minimal end-to-end slice — it proves the encoding, the Controller method, the SXU FSM path, the drain branch, and VREG writeback on the simplest case (bias add only).

**Files:** Modify `src/Controller.bsv`, `src/ScalarUnit.bsv` (parent); `tinygrad/.../renderer/tinytpu/common.py` (submodule); `tests/test_tinytpu_backend.py` (parent).

- [ ] **Step 1: Add `EpilogueConfig` and `startEpilogue` to the Controller**

In `src/Controller.bsv`: define `typedef struct { Bool biasEnable; Bool reluEnable; Bool reduceEnable; Bool reduceSumsq; } EpilogueConfig deriving (Bits, Eq);` near `ControlState`. Add to `Controller_IFC` (after `startPsum`):
```bsv
   method Action startEpilogue(UInt#(TLog#(depth)) weightBase,
                               UInt#(TLog#(depth)) actBase,
                               UInt#(TLog#(depth)) tileLen,
                               Vector#(cols, Int#(32)) biasVec,
                               EpilogueConfig epiCfg);
```
Add module registers: `Reg#(Vector#(cols, Int#(32))) biasReg`, `Reg#(EpilogueConfig) epiCfgReg`, `Reg#(Bool) epilogueActive` (default False). Implement `startEpilogue` modeled exactly on `startPsum` (`Controller.bsv:359–377`) — same `wBase/aBase/tLen/actIdx/streamCycle/firstActRead` writes, `dfModeReg <= DF_WEIGHT_STATIONARY`, `array.clearAll`, `cstate <= LoadWeights` — plus `biasReg <= biasVec; epiCfgReg <= epiCfg; epilogueActive <= True;`. In `startPsum`/`start`/`startOS` etc., set `epilogueActive <= False` so non-epilogue dispatches are unaffected.

- [ ] **Step 2: Add the epilogue branch to `do_drain`**

In the `do_drain` rule (`Controller.bsv:308–340`), after `array.clearAll` and before `cstate <= Done`, add:
```bsv
   if (epilogueActive) begin
      Vector#(rows, Vector#(cols, Int#(32))) m = array.getMatrix;
      Vector#(rows, Vector#(cols, Int#(32))) outm = m;
      for (Integer ri = 0; ri < valueOf(rows); ri = ri + 1)
         for (Integer ci = 0; ci < valueOf(cols); ci = ci + 1) begin
            Int#(32) v = m[ri][ci];
            if (epiCfgReg.biasEnable) v = v + biasReg[ci];
            outm[ri][ci] = v;
         end
      epilogueBuf <= outm;
   end
```
Add `Reg#(Vector#(rows, Vector#(cols, Int#(32)))) epilogueBuf`. Add a method `method Vector#(rows, Vector#(cols, Int#(32))) epilogueResult if (cstate == Done); return epilogueBuf; endmethod` to the interface and module. The existing `psumModeReg` `case` stays — for an epilogue dispatch `psumModeReg` is whatever default leaves PSUM untouched (confirm `startEpilogue` sets `psumModeReg` to the no-op mode, matching how `start` leaves it).

- [ ] **Step 3: Add SXU states + the `SXU_DISPATCH_MXU_EPILOGUE` rule (bias path)**

In `src/ScalarUnit.bsv`: add `SXU_DISPATCH_MXU_EPILOGUE` (42) and `SXU_LOAD_EPILOGUE_STAT` (43) to `SxuOpCode` (after `SXU_DISPATCH_VPU_BG`). Add states `SXU_EXEC_MXU_EPILOGUE`, `SXU_WAIT_MXU_EPILOGUE`, `SXU_EXEC_MXU_EPILOGUE_WB`, `SXU_EXEC_LOAD_EPILOGUE_STAT` to `SxuState`. In `do_fetch`, add decode arms: `SXU_DISPATCH_MXU_EPILOGUE: pc_state <= SXU_EXEC_MXU_EPILOGUE;` and `SXU_LOAD_EPILOGUE_STAT: pc_state <= SXU_EXEC_LOAD_EPILOGUE_STAT;`.

Add the dispatch rule (model the `ctrl.start*` call on `do_mxu`, `ScalarUnit.bsv:943–957`):
```bsv
rule do_mxu_epilogue (pc_state == SXU_EXEC_MXU_EPILOGUE);
   Bit#(7) cfg = pack(curInstr.vpuOp);
   let biasTile = vrf.read(truncate(curInstr.vregSrc));
   EpilogueConfig ec = EpilogueConfig {
      biasEnable:  cfg[0] == 1, reluEnable:  cfg[1] == 1,
      reduceEnable: cfg[2] == 1, reduceSumsq: cfg[3] == 1 };
   ctrl.startEpilogue(truncate(curInstr.mxuWBase), truncate(curInstr.mxuABase),
                      truncate(curInstr.mxuTLen), biasTile[0], ec);
   pc_state <= SXU_WAIT_MXU_EPILOGUE;
endrule

rule do_wait_mxu_epilogue (pc_state == SXU_WAIT_MXU_EPILOGUE);
   if (ctrl.isDone) pc_state <= SXU_EXEC_MXU_EPILOGUE_WB;
endrule

rule do_mxu_epilogue_wb (pc_state == SXU_EXEC_MXU_EPILOGUE_WB);
   Bit#(7) cfg = pack(curInstr.vpuOp);
   let m = ctrl.epilogueResult;
   if (cfg[4] == 1) vmem.write(truncate(curInstr.vmemAddr), m);
   else             vrf.write(truncate(curInstr.vregDst), m);
   pc <= pc + 1;
   pc_state <= SXU_FETCH;
endrule
```
(`biasTile[0]` is row 0 of the bias vreg — the `Vector#(cols, Int#(32))` bias. `vmem.write`/`vrf.write` idioms from `do_store`/`do_load_mxu_result`.) Leave `SXU_EXEC_LOAD_EPILOGUE_STAT` as a stub rule that advances pc for now (filled in Task 4).

- [ ] **Step 4: Add the `_mxu_epilogue` bundle helper (submodule)**

In `tinygrad/tinygrad/renderer/tinytpu/common.py`, near `_mxu` (~line 304), add:
```python
def _mxu_epilogue(wbase, abase, tiles, bias_vreg, dst, bias=True, relu=False,
                  reduce=0, vmem_dst=False):
    # opcode 42. config bits in the vpuOp field: bit0 bias, bit1 relu,
    # bit2 reduce-enable, bit3 reduce-sumsq, bit4 vmem-writeback.
    cfg = (int(bias) | (int(relu) << 1) | (int(reduce != 0) << 2)
           | (int(reduce == 2) << 3) | (int(vmem_dst) << 4))
    vmem_addr = dst if vmem_dst else 0
    vreg_dst = 0 if vmem_dst else dst
    return f"2 42 {vmem_addr} {vreg_dst} {bias_vreg} {cfg} 0 {wbase} {abase} {tiles}"

def _load_epilogue_stat(vd):
    return f"2 43 0 {vd} 0 0 0 0 0 0"
```
Add both to the `from tinygrad.renderer.tinytpu.common import (...)` re-export list in `ops_tinytpu.py` (~lines 27–39).

- [ ] **Step 5: Write the failing runtime test**

In `tests/test_tinytpu_backend.py`, class `TestTinyTPUSimOutputParsing`, add:
```python
  def test_mxu_epilogue_bias_only(self):
    # SXU_DISPATCH_MXU_EPILOGUE: GEMM + row-broadcast bias in one dispatch.
    import numpy as np
    sim = os.environ["TINYTPU_SIM"]
    from tinygrad.runtime.ops_tinytpu import _mxu_epilogue
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [0, 1, 0, 1], [2, 2, 2, 2]], dtype=np.int32)
    w = np.eye(4, dtype=np.int32)
    bias = np.array([10, 20, 30, 40], dtype=np.int32)
    bundle = _bundle(
      _wmem(0, w.flatten().tolist()),
      _amem(0, a.flatten().tolist()),
      _vmem(0, bias.tolist() + [0] * 12),
      _load(0, 0),                                  # bias -> v0 row 0
      _mxu_epilogue(0, 0, 1, bias_vreg=0, dst=1, bias=True),
      _store(1, 1),
      _halt(),
      _output_vmem(1),
      _end(),
    )
    out = _run_bundle(sim, bundle)
    tile = _parse_vmem_output(out)
    expected = (a @ w + bias).flatten().tolist()
    self.assertEqual(tile, expected, f"epilogue bias: got {tile}")
```
If `_wmem`/`_amem` argument shapes differ, match an existing GEMM test in the file (e.g. `test_mxu_os_end_to_end_identity_matmul`).

- [ ] **Step 6: Run it, verify failure, then build**

Run the test (`pytest ... -k test_mxu_epilogue_bias_only`): expect FAIL (no hardware yet). Then `make runtime-tb` to rebuild after the BSV changes from Steps 1–3.

- [ ] **Step 7: Run the test and the full suite**

Run `pytest ... -k test_mxu_epilogue_bias_only` — expect PASS. Then full `tests/test_tinytpu_backend.py` — expect **967 passed, 0 failed**. Also run `make test-ctrl-psum` — expect PASS (the Controller change must not regress the PSUM path). If the BSV build fails or a Controller test regresses, STOP and report BLOCKED with the exact error.

- [ ] **Step 8: Commit**

```bash
cd /Users/hanwang/p/tinytpu/tinygrad && git add tinygrad/renderer/tinytpu/common.py tinygrad/runtime/ops_tinytpu.py && git commit -m "tinytpu: _mxu_epilogue / _load_epilogue_stat bundle helpers"
cd /Users/hanwang/p/tinytpu && git add src/Controller.bsv src/ScalarUnit.bsv tests/test_tinytpu_backend.py tinygrad && git commit -m "controller,sxu: fused MXU-drain epilogue — bias-only path"
```

---

## Task 3: ReLU and VMEM writeback

**Files:** Modify `src/Controller.bsv`, `tests/test_tinytpu_backend.py`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_tinytpu_backend.py`, class `TestTinyTPUSimOutputParsing`, add `test_mxu_epilogue_bias_relu` and `test_mxu_epilogue_vmem_writeback` and `test_mxu_epilogue_equiv_legacy`. The first builds a bundle with `_mxu_epilogue(..., bias=True, relu=True)` and asserts the result equals `np.maximum(a @ w + bias, 0)`. The second uses `_mxu_epilogue(..., dst=<vmem slot>, vmem_dst=True)` then `_output_vmem` (no `_store`), asserting the result landed in VMEM. The third runs the same GEMM both through `_mxu_epilogue(bias=True, relu=True)` and through the legacy `_mxu`/`_psum_read_row`/`_vpu(ADD)`/`_vpu(RELU)` sequence and asserts the two results are identical. Use negative inputs so ReLU actually clamps.

- [ ] **Step 2: Run to verify failure**

`pytest ... -k "test_mxu_epilogue_bias_relu or test_mxu_epilogue_vmem_writeback or test_mxu_epilogue_equiv_legacy"` — `bias_relu` and `equiv_legacy` fail (ReLU not applied); `vmem_writeback` may already pass (Task 2 Step 3 wired the `cfg[4]` VMEM path).

- [ ] **Step 3: Add ReLU to the `do_drain` epilogue branch**

In `src/Controller.bsv`, in the epilogue branch added in Task 2 Step 2, after the bias add and before `outm[ri][ci] = v;`, add:
```bsv
            if (epiCfgReg.reluEnable && v < 0) v = 0;
```

- [ ] **Step 4: Rebuild and verify**

`make runtime-tb`, then run the three tests — all PASS. Then full `tests/test_tinytpu_backend.py` — expect **970 passed, 0 failed** (967 after Task 2, plus Task 3's 3 new tests). Run `make test-ctrl-psum` — PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/hanwang/p/tinytpu && git add src/Controller.bsv tests/test_tinytpu_backend.py && git commit -m "controller: MXU epilogue — ReLU stage"
```

---

## Task 4: Per-row INT64 reduction + LOAD_EPILOGUE_STAT

**Files:** Modify `src/Controller.bsv`, `src/ScalarUnit.bsv`, `tests/test_tinytpu_backend.py`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_tinytpu_backend.py`, add `test_mxu_epilogue_reduce_sum` and `test_mxu_epilogue_reduce_sumsq`. Each builds `_mxu_epilogue(..., bias=True, relu=True, reduce=1)` (SUM) or `reduce=2` (SUMSQ), then `_load_epilogue_stat(vd)`, then `_store(vd, slot)` and `_output_vmem(slot)`. Reconstruct each row's INT64 statistic from lanes `(r,0)`=lo and `(r,1)`=hi: `stat = (hi << 32) | (lo & 0xFFFFFFFF)`, interpreted signed. Assert against the numpy reference — for SUM, `row.sum()` over the post-bias-post-ReLU row; for SUMSQ, `(row.astype(np.int64) ** 2).sum()`. Include a SUMSQ case whose value exceeds 2**31 to prove the INT64 width is real.

- [ ] **Step 2: Run to verify failure**

`pytest ... -k "test_mxu_epilogue_reduce"` — FAIL (reduction not implemented; `LOAD_EPILOGUE_STAT` is a stub).

- [ ] **Step 3: Add the INT64 reduction to `do_drain`**

In `src/Controller.bsv`: add `Reg#(Vector#(rows, Int#(64))) epilogueStatBuf`. In the epilogue branch of `do_drain`, after computing `outm`, add:
```bsv
   if (epiCfgReg.reduceEnable) begin
      Vector#(rows, Int#(64)) stat = replicate(0);
      for (Integer ri = 0; ri < valueOf(rows); ri = ri + 1) begin
         Int#(64) acc = 0;
         for (Integer ci = 0; ci < valueOf(cols); ci = ci + 1) begin
            Int#(64) e = signExtend(outm[ri][ci]);
            acc = acc + (epiCfgReg.reduceSumsq ? (e * e) : e);
         end
         stat[ri] = acc;
      end
      epilogueStatBuf <= stat;
   end
```
Add `method Vector#(rows, Int#(64)) epilogueStat if (cstate == Done); return epilogueStatBuf; endmethod` to the interface and module.

- [ ] **Step 4: Implement the `SXU_LOAD_EPILOGUE_STAT` rule**

In `src/ScalarUnit.bsv`, replace the Task 2 stub rule for `SXU_EXEC_LOAD_EPILOGUE_STAT` with:
```bsv
rule do_load_epilogue_stat (pc_state == SXU_EXEC_LOAD_EPILOGUE_STAT);
   let stat = ctrl.epilogueStat;
   Vector#(sublanes, Vector#(lanes, Int#(32))) v = replicate(replicate(0));
   for (Integer ri = 0; ri < valueOf(sublanes); ri = ri + 1) begin
      Bit#(64) b = pack(stat[ri]);
      v[ri][0] = unpack(b[31:0]);
      v[ri][1] = unpack(b[63:32]);
   end
   vrf.write(truncate(curInstr.vregDst), v);
   pc <= pc + 1;
   pc_state <= SXU_FETCH;
endrule
```
(`sublanes`/`lanes` are the SXU's tile dims; if `rows` from the Controller differs in name, the per-row count is the same value — match the existing `do_load_mxu_result` row indexing.)

- [ ] **Step 5: Rebuild and verify**

`make runtime-tb`, then run the reduce tests — PASS. Full `tests/test_tinytpu_backend.py` — expect **972 passed** (970 + 2), 0 failed. `make test-ctrl-psum` — PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/hanwang/p/tinytpu && git add src/Controller.bsv src/ScalarUnit.bsv tests/test_tinytpu_backend.py && git commit -m "controller,sxu: MXU epilogue — INT64 per-row reduction + LOAD_EPILOGUE_STAT"
```

---

## Notes for the executor

- **Submodule first.** Task 1 and Task 2 each commit the submodule (`tinygrad/`) before the parent, and the parent commit records the pointer (`git add tinygrad`). Do not push.
- **Rebuild discipline.** After every `src/*.bsv` edit, `make runtime-tb` before running `tests/test_tinytpu_backend.py` — the staleness guard enforces it.
- **The legacy path must stay green.** Existing GEMM/PSUM tests (e.g. `test_fused_gemm_bias_relu_matches_numpy`, `test_deep_k4_gemm_through_psum_matches_numpy`) and `make test-ctrl-psum` must keep passing — `epilogueActive` gates all new behavior so non-epilogue dispatches are byte-identical.
- **BLOCKED conditions.** A BSV build error, a Controller-test regression, or a rule-scheduling conflict in `do_drain` are real BLOCKED conditions — report the exact `bsc` error; do not thrash. If the epilogue branch causes a `do_drain` scheduling conflict, the fallback is a dedicated post-drain `EpilogueDrain` Controller state — escalate before taking that larger path.
- **Test counts** (966 baseline): Task 2 → 967, Task 3 → 970, Task 4 → 972. Each task's full-suite run must show 0 failed.
