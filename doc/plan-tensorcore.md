# TensorCore Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate all TensorCore sub-units — SXU (Scalar Unit), MXU (SystolicArray), VPU, XLU, VRegFile, and VMEM — into a single `mkTensorCore` top-level module exposing a simple program-load + start + done + output interface.

**Architecture:** TensorCore instantiates all sub-units and wires them together. The SXU drives VMEM, VRegFile, VPU, and XLU. MXU is driven by a dedicated MXU controller (using the existing `mkController` pattern) that the SXU triggers. The TensorCore interface exposes: VMEM data preload, program load, start, isDone, and getOutput (last VPU/MXU result vreg). This replaces the existing `mkTensorAccelerator`.

**Tech Stack:** BSV, BSC, Bluesim, GNU Make. Depends on: `src/VMEM.bsv`, `src/VRegFile.bsv`, `src/VPU.bsv`, `src/XLU.bsv`, `src/ScalarUnit.bsv`, `src/SystolicArray.bsv`, `src/Controller.bsv`.

---

## Architecture Diagram

```
mkTensorCore
├── mkVMEM            (unified scratchpad)
├── mkVRegFile        (vreg working memory)
├── mkSystolicArray   (MXU: existing)
├── mkController      (MXU controller: existing, driven by SXU)
├── mkVPU             (element-wise ops)
├── mkXLU             (lane permutation)
└── mkScalarUnit      (SXU: orchestrates all others via microprogram)
```

The SXU's instruction set is extended with `SXU_DISPATCH_MXU` and `SXU_WAIT_MXU` to drive the Controller.

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/ScalarUnit.bsv` | Modify | Add `SXU_DISPATCH_MXU` and `SXU_WAIT_MXU` opcodes and handling |
| `src/TensorCore.bsv` | Create | `TensorCore_IFC`, `mkTensorCore` top-level integration |
| `test/TbTensorCore.bsv` | Create | End-to-end test: load weights, load activations, GEMM via MXU, RELU via VPU |
| `Makefile` | Modify | Add `test-tc` target |

---

## Task 1: Extend ScalarUnit with MXU dispatch instructions

**Files:**
- Modify: `src/ScalarUnit.bsv`

The SXU needs two new opcodes to trigger the existing `mkController`:

- `SXU_DISPATCH_MXU` — calls `ctrl.start(wBase, aBase, tileLen)` using fields from the instruction
- `SXU_WAIT_MXU` — stalls until `ctrl.isDone` is true

- [ ] **Step 1: Add fields to `SxuInstr` struct**

In `src/ScalarUnit.bsv`, extend `SxuOpCode` and `SxuInstr`:

```bsv
typedef enum {
   SXU_LOAD_VREG,
   SXU_STORE_VREG,
   SXU_DISPATCH_VPU,
   SXU_WAIT_VPU,
   SXU_DISPATCH_MXU,
   SXU_WAIT_MXU,
   SXU_HALT
} SxuOpCode deriving (Bits, Eq, FShow);

typedef struct {
   SxuOpCode op;
   UInt#(8)  vmemAddr;
   UInt#(4)  vregDst;
   UInt#(4)  vregSrc;
   VpuOp     vpuOp;
   UInt#(4)  vregSrc2;
   UInt#(8)  mxuWBase;   // MXU weight base addr in VMEM
   UInt#(8)  mxuABase;   // MXU activation base addr in VMEM
   UInt#(8)  mxuTLen;    // MXU tile length
} SxuInstr deriving (Bits, Eq);
```

- [ ] **Step 2: Update `mkScalarUnit` signature to accept Controller**

Change module signature to accept a `Controller_IFC` parameter. In the module:

```bsv
module mkScalarUnit#(
   VMEM_IFC#(vmDepth, sublanes, lanes)           vmem,
   VRegFile_IFC#(numRegs, sublanes, lanes)        vrf,
   VPU_IFC#(sublanes, lanes)                      vpu,
   XLU_IFC#(sublanes, lanes)                      xlu,
   Controller_IFC#(sublanes, lanes, vmDepth)      ctrl
)(SXU_IFC#(progDepth, vmDepth, numRegs, sublanes, lanes))
```

Add `Controller` import at top of file.

- [ ] **Step 3: Add DISPATCH_MXU and WAIT_MXU rules**

In the `do_fetch` case statement, add:
```bsv
SXU_DISPATCH_MXU: pc_state <= SXU_EXEC_MXU;
SXU_WAIT_MXU:     pc_state <= SXU_WAIT_MXU_STATE;
```

Add two new state enum values: `SXU_EXEC_MXU` and `SXU_WAIT_MXU_STATE`.

Add rules:
```bsv
rule do_mxu (pc_state == SXU_EXEC_MXU);
   ctrl.start(extend(curInstr.mxuWBase),
              extend(curInstr.mxuABase),
              extend(curInstr.mxuTLen));
   pc <= pc + 1;
   pc_state <= SXU_FETCH;
endrule

rule do_wait_mxu (pc_state == SXU_WAIT_MXU_STATE && ctrl.isDone);
   pc <= pc + 1;
   pc_state <= SXU_FETCH;
endrule
```

- [ ] **Step 4: Update existing TbScalarUnit.bsv to pass a dummy Controller**

Since `mkScalarUnit` now requires a Controller parameter, the existing testbench must be updated. Create a small stub:

```bsv
// In TbScalarUnit.bsv — replace sxu instantiation:
// Use mkTensorAccelerator's internal controller pattern:
// Instantiate SystolicArray + WeightSRAM + ActivationSRAM + Controller as stubs
// Or: create a simple stub controller that is always isDone and accepts start().
```

For the stub, add to `test/TbScalarUnit.bsv`:
```bsv
import SystolicArray :: *;
import WeightSRAM :: *;
import ActivationSRAM :: *;
import Controller :: *;

SystolicArray_IFC#(4, 4)   arr   <- mkSystolicArray;
WeightSRAM_IFC#(16, 4, 4)  wsram <- mkWeightSRAM;
ActivationSRAM_IFC#(16, 4) asram <- mkActivationSRAM;
Controller_IFC#(4, 4, 16)  ctrl  <- mkController(arr, wsram, asram);

SXU_IFC#(8, 16, 8, 4, 4) sxu <- mkScalarUnit(vmem, vrf, vpu, xlu, ctrl);
```

- [ ] **Step 5: Run existing sxu test to verify nothing broke**

```bash
cd /home/hanwang/p/tinytpu && make test-sxu
```
Expected: still passes.

- [ ] **Step 6: Commit ScalarUnit extension**

```bash
git add src/ScalarUnit.bsv test/TbScalarUnit.bsv
git commit --author="Han Wang <h1337h4x0r@gmail.com>" -m "feat: extend ScalarUnit with DISPATCH_MXU/WAIT_MXU opcodes"
```

---

## Task 2: Write failing TensorCore test

**Files:**
- Create: `test/TbTensorCore.bsv`
- Modify: `Makefile`

Test scenario: 4×4 GEMM followed by RELU.
1. Preload weight tile into VMEM[0]
2. Preload activation tile into VMEM[1]
3. SXU program: DISPATCH_MXU(wBase=0, aBase=1, tLen=1) → WAIT_MXU → read MXU result → store to VMEM[2] → HALT
4. Verify VMEM[2] contains correct GEMM output

- [ ] **Step 1: Add Makefile entries**

```makefile
$(BUILDDIR)/TensorCore.bo: $(BUILDDIR)/ScalarUnit.bo $(BUILDDIR)/SystolicArray.bo $(BUILDDIR)/VMEM.bo $(BUILDDIR)/VRegFile.bo $(BUILDDIR)/VPU.bo $(BUILDDIR)/XLU.bo $(BUILDDIR)/Controller.bo $(BUILDDIR)/WeightSRAM.bo $(BUILDDIR)/ActivationSRAM.bo
$(BUILDDIR)/TbTensorCore.bo: $(BUILDDIR)/TensorCore.bo
```

Link rule:
```makefile
$(BUILDDIR)/mkTbTensorCore.bexe: $(BUILDDIR)/TbTensorCore.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbTensorCore $(BUILDDIR)/mkTbTensorCore.ba
```

Test target:
```makefile
test-tc: $(BUILDDIR)/mkTbTensorCore.bexe
	$<
```

Add `test-tc` to `.PHONY` and `test` target.

- [ ] **Step 2: Write failing test**

Create `test/TbTensorCore.bsv`:

```bsv
package TbTensorCore;

import Vector :: *;
import TensorCore :: *;

(* synthesize *)
module mkTbTensorCore();

   // rows=4, cols=4, depth=16 (VMEM depth), numRegs=8, progDepth=16
   TensorCore_IFC#(4, 4, 16) tc <- mkTensorCore;

   Reg#(UInt#(8)) cycle  <- mkReg(0);
   Reg#(UInt#(8)) passed <- mkReg(0);
   Reg#(UInt#(8)) failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 200) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   // Load 4×4 identity weight matrix into VMEM[0]
   rule load_weights (cycle == 0);
      Vector#(4, Vector#(4, Int#(8))) w = replicate(replicate(0));
      w[0][0] = 1; w[1][1] = 1; w[2][2] = 1; w[3][3] = 1;
      tc.loadWeightTile(0, w);
      $display("Cycle %0d: loaded identity weights", cycle);
   endrule

   // Load activation [1, 2, 3, 4] into VMEM[1]
   rule load_acts (cycle == 1);
      Vector#(4, Int#(8)) a = newVector;
      a[0]=1; a[1]=2; a[2]=3; a[3]=4;
      tc.loadActivationTile(1, a);
      $display("Cycle %0d: loaded activations", cycle);
   endrule

   // Load SXU program: DISPATCH_MXU wBase=0 aBase=1 tLen=1 → WAIT_MXU → HALT
   rule load_prog (cycle == 2);
      tc.loadProgram(0, TCInstr { op: TC_DISPATCH_MXU, mxuWBase:0, mxuABase:1, mxuTLen:1,
                                  vmemAddr:0, vregDst:0, vregSrc:0, vpuOp:TC_VPU_RELU, vregSrc2:0 });
      tc.loadProgram(1, TCInstr { op: TC_WAIT_MXU,     mxuWBase:0, mxuABase:0, mxuTLen:0,
                                  vmemAddr:0, vregDst:0, vregSrc:0, vpuOp:TC_VPU_RELU, vregSrc2:0 });
      tc.loadProgram(2, TCInstr { op: TC_HALT,          mxuWBase:0, mxuABase:0, mxuTLen:0,
                                  vmemAddr:0, vregDst:0, vregSrc:0, vpuOp:TC_VPU_RELU, vregSrc2:0 });
      $display("Cycle %0d: program loaded", cycle);
   endrule

   rule start_tc (cycle == 3);
      tc.start(3);
      $display("Cycle %0d: TensorCore started", cycle);
   endrule

   rule wait_done (cycle > 3 && !tc.isDone);
      $display("Cycle %0d: computing...", cycle);
   endrule

   rule check_done (cycle > 3 && tc.isDone);
      Vector#(4, Int#(32)) out = tc.getMxuResult;
      // Identity × [1,2,3,4] = [1,2,3,4]
      Bool ok = (out[0] == 1 && out[1] == 2 && out[2] == 3 && out[3] == 4);
      if (ok) begin
         $display("PASS TensorCore GEMM [%0d,%0d,%0d,%0d]",
            out[0], out[1], out[2], out[3]);
         passed <= passed + 1;
      end else begin
         $display("FAIL TensorCore expected [1,2,3,4] got [%0d,%0d,%0d,%0d]",
            out[0], out[1], out[2], out[3]);
         failed <= failed + 1;
      end
      $display("Results: %0d passed, %0d failed",
         passed + (ok ? 1 : 0), failed + (ok ? 0 : 1));
      if (ok) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
```

- [ ] **Step 3: Run — expect compile error**

```bash
cd /home/hanwang/p/tinytpu && make test-tc
```

---

## Task 3: Implement `src/TensorCore.bsv`

**Files:**
- Create: `src/TensorCore.bsv`

- [ ] **Step 1: Write TensorCore.bsv**

```bsv
package TensorCore;

import Vector :: *;
import VMEM :: *;
import VRegFile :: *;
import VPU :: *;
import XLU :: *;
import ScalarUnit :: *;
import SystolicArray :: *;
import WeightSRAM :: *;
import ActivationSRAM :: *;
import Controller :: *;

// Expose a simplified instruction type that wraps SxuInstr
// (reuses SxuInstr from ScalarUnit; TCInstr is an alias)
typedef SxuInstr TCInstr;
typedef SxuOpCode TCOpCode;
// Expose opcode aliases for testbench use
TCOpCode tc_dispatch_mxu = SXU_DISPATCH_MXU;
TCOpCode tc_wait_mxu     = SXU_WAIT_MXU;
TCOpCode tc_halt         = SXU_HALT;

interface TensorCore_IFC#(numeric type rows, numeric type cols, numeric type depth);
   // Pre-load weight tile into VMEM at addr (uses WeightSRAM internally for MXU)
   method Action loadWeightTile(UInt#(TLog#(depth)) addr,
                                Vector#(rows, Vector#(cols, Int#(8))) data);
   // Pre-load activation vector into VMEM (via ActivationSRAM)
   method Action loadActivationTile(UInt#(TLog#(depth)) addr,
                                    Vector#(rows, Int#(8)) data);
   // Load SXU microprogram instruction
   method Action loadProgram(UInt#(4) pc, TCInstr instr);
   // Start SXU execution with given program length
   method Action start(UInt#(4) len);
   method Bool isDone;
   // MXU result (from Controller.results)
   method Vector#(cols, Int#(32)) getMxuResult;
endinterface

module mkTensorCore(TensorCore_IFC#(rows, cols, depth))
   provisos(
      Add#(1, r_, rows),
      Add#(1, c_, cols),
      Add#(1, d_, depth),
      Log#(depth, logDepth),
      Add#(logd_, TLog#(depth), 32),
      Bits#(Vector#(rows, Vector#(cols, Int#(8))), wsz),
      Bits#(Vector#(rows, Int#(8)), asz),
      Add#(0, rows, cols),  // square (for XLU)
      Add#(1, s_, rows),
      Log#(rows, logRows),
      Bits#(Vector#(rows, Vector#(rows, Int#(32))), vrsz),
      Bits#(SxuInstr, isz)
   );

   // MXU sub-system (reuses existing modules)
   SystolicArray_IFC#(rows, cols) array <- mkSystolicArray;
   WeightSRAM_IFC#(depth, rows, cols) wsram <- mkWeightSRAM;
   ActivationSRAM_IFC#(depth, rows)  asram <- mkActivationSRAM;
   Controller_IFC#(rows, cols, depth) ctrl  <- mkController(array, wsram, asram);

   // VPU/XLU sub-system
   VMEM_IFC#(depth, rows, rows)    vmem <- mkVMEM;
   VRegFile_IFC#(16, rows, rows)   vrf  <- mkVRegFile;
   VPU_IFC#(rows, rows)            vpu  <- mkVPU;
   XLU_IFC#(rows, rows)            xlu  <- mkXLU;

   // Scalar Unit — drives everything
   SXU_IFC#(16, depth, 16, rows, rows) sxu <-
      mkScalarUnit(vmem, vrf, vpu, xlu, ctrl);

   method Action loadWeightTile(UInt#(TLog#(depth)) addr,
                                Vector#(rows, Vector#(cols, Int#(8))) data);
      wsram.write(addr, data);
   endmethod

   method Action loadActivationTile(UInt#(TLog#(depth)) addr,
                                    Vector#(rows, Int#(8)) data);
      asram.write(addr, data);
   endmethod

   method Action loadProgram(UInt#(4) pc, TCInstr instr);
      sxu.loadInstr(extend(pc), instr);
   endmethod

   method Action start(UInt#(4) len);
      sxu.start(extend(len));
   endmethod

   method Bool isDone = sxu.isDone;

   method Vector#(cols, Int#(32)) getMxuResult;
      return ctrl.results;
   endmethod

endmodule

export TCInstr(..);
export TensorCore_IFC(..);
export mkTensorCore;
export TC_DISPATCH_MXU = SXU_DISPATCH_MXU;
export TC_WAIT_MXU     = SXU_WAIT_MXU;
export TC_HALT         = SXU_HALT;
export TC_VPU_RELU     = VPU_RELU;

endpackage
```

**Note:** The BSV export aliases for enum values require careful handling. If BSV doesn't support `export X = Y` syntax, expose `SxuOpCode` and `VpuOp` directly and use them in the testbench. Update the testbench to import `ScalarUnit` and `VPU` directly for the opcode values.

- [ ] **Step 2: Run test — iterate on compile errors**

```bash
cd /home/hanwang/p/tinytpu && make test-tc 2>&1
```

Fix any proviso or type errors. Common issues:
- `Add#(0, rows, cols)` proviso for XLU: may need `TensorCore` to be square-only
- BSV RegFile size limits: ensure `numRegs` is power of 2
- Missing `Bits` instances: add as needed to provisos

- [ ] **Step 3: Run — expect PASS**

Expected:
```
Cycle 0: loaded identity weights
Cycle 1: loaded activations
Cycle 2: program loaded
Cycle 3: TensorCore started
Cycle N: computing...
PASS TensorCore GEMM [1,2,3,4]
Results: 1 passed, 0 failed
```

- [ ] **Step 4: Run full regression**

```bash
cd /home/hanwang/p/tinytpu && make test
```
All existing tests should still pass.

- [ ] **Step 5: Commit**

```bash
git add src/TensorCore.bsv test/TbTensorCore.bsv Makefile
git commit --author="Han Wang <h1337h4x0r@gmail.com>" -m "feat: add TensorCore integrating SXU/MXU/VPU/XLU/VRegFile/VMEM"
```

---

*Plan created: 2026-04-08*
