# Scalar Unit (SXU) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a simplified Scalar Unit (SXU) — a microprogram sequencer that executes a fixed-instruction-set program to control VRegFile loads/stores from VMEM and dispatch operations to the MXU, VPU, and XLU.

**Architecture:** The SXU holds a program in a RegFile (program memory). Each cycle it fetches the next instruction, decodes it, and drives control signals to other units. Instructions: LOAD_VREG (VMEM→VRegFile), STORE_VREG (VRegFile→VMEM), DISPATCH_MXU, DISPATCH_VPU, DISPATCH_XLU, WAIT (stall until pending unit is done), HALT. The program is loaded from outside before `start` is called. This is the "CPU" of the TensorCore; it replaces the existing `mkController` FSM with a programmable sequencer.

**Tech Stack:** Bluespec SystemVerilog (BSV), BSC compiler, Bluesim simulator, GNU Make. Follows pattern of `src/Controller.bsv` (FSM-based rule sequencer) but driven by a program array.

---

## Instruction Set

```
LOAD_VREG  vmemAddr vregDst      — VRegFile[vregDst] ← VMEM[vmemAddr] (1+1 cycle)
STORE_VREG vmemAddr vregSrc      — VMEM[vmemAddr] ← VRegFile[vregSrc] (1 cycle)
DISPATCH_MXU wBase aBase tLen    — Start MXU matrix multiply (fires mkController logic)
DISPATCH_VPU op vsrc1 vsrc2 vdst — VPU op on VRegFile vregs → write result to vdst
DISPATCH_XLU op vsrc vdst [amt]  — XLU op on VRegFile vreg → write result to vdst
WAIT_MXU                         — Stall PC until MXU isDone
WAIT_VPU                         — Stall PC until VPU done register is set
HALT                             — Stop execution, assert isDone
```

For TinyTPU, instruction fields are packed into a struct. SXU drives VMEM, VRegFile, MXU, VPU, XLU directly via method calls (not a bus).

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/ScalarUnit.bsv` | Create | SxuInstr struct, SxuOp enum, SXU_IFC, `mkScalarUnit` |
| `test/TbScalarUnit.bsv` | Create | Program: load vreg → VPU ADD → store result |
| `Makefile` | Modify | Add `test-sxu` target |

### Dependencies

`mkScalarUnit` takes interfaces as module parameters (same pattern as `mkController`):

```bsv
module mkScalarUnit#(
   VMEM_IFC#(vmDepth, sublanes, lanes)       vmem,
   VRegFile_IFC#(numRegs, sublanes, lanes)   vrf,
   VPU_IFC#(sublanes, lanes)                 vpu,
   XLU_IFC#(sublanes, lanes)                 xlu
)(SXU_IFC#(progDepth, vmDepth, numRegs, sublanes, lanes))
```

MXU integration is deferred to the TensorCore plan; the SXU here controls VMEM, VRegFile, VPU, and XLU.

---

## Task 1: Define types and write failing test

**Files:**
- Create: `test/TbScalarUnit.bsv`
- Modify: `Makefile`

- [ ] **Step 1: Add Makefile entries**

After existing dependency lines, add:
```makefile
$(BUILDDIR)/ScalarUnit.bo: $(BUILDDIR)/VMEM.bo $(BUILDDIR)/VRegFile.bo $(BUILDDIR)/VPU.bo $(BUILDDIR)/XLU.bo
$(BUILDDIR)/TbScalarUnit.bo: $(BUILDDIR)/ScalarUnit.bo
```

Link rule:
```makefile
$(BUILDDIR)/mkTbScalarUnit.bexe: $(BUILDDIR)/TbScalarUnit.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbScalarUnit $(BUILDDIR)/mkTbScalarUnit.ba
```

Test target:
```makefile
test-sxu: $(BUILDDIR)/mkTbScalarUnit.bexe
	$<
```

Add `test-sxu` to `.PHONY` and to the `test` target.

- [ ] **Step 2: Write failing test**

The testbench loads data into VMEM at addr 0 and addr 1 before starting the SXU. The SXU program:
1. LOAD_VREG 0 → vreg 0
2. LOAD_VREG 1 → vreg 1
3. DISPATCH_VPU ADD vreg0 vreg1 → vreg 2
4. WAIT_VPU
5. STORE_VREG addr 2 ← vreg 2
6. HALT

After halt, testbench reads VMEM[2] and verifies the sum.

Create `test/TbScalarUnit.bsv`:

```bsv
package TbScalarUnit;

import Vector :: *;
import VMEM :: *;
import VRegFile :: *;
import VPU :: *;
import XLU :: *;
import ScalarUnit :: *;

(* synthesize *)
module mkTbScalarUnit();

   VMEM_IFC#(16, 4, 4)     vmem <- mkVMEM;
   VRegFile_IFC#(8, 4, 4)  vrf  <- mkVRegFile;
   VPU_IFC#(4, 4)          vpu  <- mkVPU;
   XLU_IFC#(4, 4)          xlu  <- mkXLU;

   // progDepth=8: up to 8 instructions
   SXU_IFC#(8, 16, 8, 4, 4) sxu <- mkScalarUnit(vmem, vrf, vpu, xlu);

   Reg#(UInt#(8)) cycle  <- mkReg(0);
   Reg#(UInt#(8)) passed <- mkReg(0);
   Reg#(UInt#(8)) failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 100) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   // Preload VMEM before SXU runs
   rule preload_vmem (cycle == 0);
      // VMEM[0]: row0 = [1,2,3,4]
      Vector#(4, Vector#(4, Int#(32))) tA = replicate(replicate(0));
      tA[0][0]=1; tA[0][1]=2; tA[0][2]=3; tA[0][3]=4;
      vmem.write(0, tA);
      // VMEM[1]: row0 = [10,20,30,40]
      Vector#(4, Vector#(4, Int#(32))) tB = replicate(replicate(0));
      tB[0][0]=10; tB[0][1]=20; tB[0][2]=30; tB[0][3]=40;
      vmem.write(1, tB);
      $display("Cycle %0d: VMEM preloaded", cycle);
   endrule

   // Load program: LOAD 0→v0, LOAD 1→v1, DISPATCH_VPU ADD v0 v1 v2, WAIT_VPU, STORE v2→2, HALT
   rule load_program (cycle == 1);
      sxu.loadInstr(0, SxuInstr { op: SXU_LOAD_VREG,    vmemAddr:0, vregDst:0, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0 });
      sxu.loadInstr(1, SxuInstr { op: SXU_LOAD_VREG,    vmemAddr:1, vregDst:1, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0 });
      sxu.loadInstr(2, SxuInstr { op: SXU_DISPATCH_VPU, vmemAddr:0, vregDst:2, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:1 });
      sxu.loadInstr(3, SxuInstr { op: SXU_WAIT_VPU,     vmemAddr:0, vregDst:0, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0 });
      sxu.loadInstr(4, SxuInstr { op: SXU_STORE_VREG,   vmemAddr:2, vregDst:0, vregSrc:2, vpuOp:VPU_ADD, vregSrc2:0 });
      sxu.loadInstr(5, SxuInstr { op: SXU_HALT,         vmemAddr:0, vregDst:0, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0 });
      $display("Cycle %0d: program loaded", cycle);
   endrule

   rule start_sxu (cycle == 2);
      sxu.start(6);  // program length = 6 instructions
      $display("Cycle %0d: SXU started", cycle);
   endrule

   rule wait_sxu (cycle > 2 && !sxu.isDone);
      $display("Cycle %0d: SXU running...", cycle);
   endrule

   rule check_result (cycle > 2 && sxu.isDone);
      // Read VMEM[2] — should contain [11, 22, 33, 44] in row0
      $display("Cycle %0d: SXU done, reading VMEM[2]", cycle);
   endrule

   // Read VMEM after SXU done — need a separate read cycle
   Reg#(Bool) readIssued <- mkReg(False);

   rule issue_read (cycle > 2 && sxu.isDone && !readIssued);
      vmem.readReq(2);
      readIssued <= True;
   endrule

   rule check_vmem2 (readIssued && cycle > 4);
      let t = vmem.readResp;
      Bool ok = (t[0][0] == 11 && t[0][1] == 22 && t[0][2] == 33 && t[0][3] == 44);
      if (ok) begin
         $display("Cycle %0d: PASS SXU program result", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VMEM[2] row0=[%0d,%0d,%0d,%0d]",
            cycle, t[0][0], t[0][1], t[0][2], t[0][3]);
         failed <= failed + 1;
      end
      $display("Results: %0d passed, %0d failed", passed + (ok ? 1 : 0), failed + (ok ? 0 : 1));
      if (ok) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
```

- [ ] **Step 3: Run — expect compile error**

```bash
cd /home/hanwang/p/tinytpu && make test-sxu
```

---

## Task 2: Implement `src/ScalarUnit.bsv`

**Files:**
- Create: `src/ScalarUnit.bsv`

- [ ] **Step 1: Write ScalarUnit.bsv**

```bsv
package ScalarUnit;

import Vector :: *;
import RegFile :: *;
import VMEM :: *;
import VRegFile :: *;
import VPU :: *;
import XLU :: *;

typedef enum {
   SXU_LOAD_VREG,
   SXU_STORE_VREG,
   SXU_DISPATCH_VPU,
   SXU_WAIT_VPU,
   SXU_HALT
} SxuOpCode deriving (Bits, Eq, FShow);

typedef struct {
   SxuOpCode op;
   UInt#(8)  vmemAddr;   // VMEM address (for LOAD/STORE)
   UInt#(4)  vregDst;    // Destination vreg index
   UInt#(4)  vregSrc;    // Source vreg index (for STORE / VPU src1)
   VpuOp     vpuOp;      // VPU operation (for DISPATCH_VPU)
   UInt#(4)  vregSrc2;   // Second source vreg (for VPU)
} SxuInstr deriving (Bits, Eq);

interface SXU_IFC#(numeric type progDepth,
                   numeric type vmDepth,
                   numeric type numRegs,
                   numeric type sublanes,
                   numeric type lanes);
   method Action loadInstr(UInt#(TLog#(progDepth)) pc, SxuInstr instr);
   method Action start(UInt#(TLog#(progDepth)) len);
   method Bool isDone;
endinterface

typedef enum { SXU_IDLE, SXU_FETCH, SXU_EXEC_LOAD_REQ, SXU_EXEC_LOAD_RESP,
               SXU_EXEC_STORE, SXU_EXEC_VPU, SXU_EXEC_WAIT_VPU, SXU_HALTED }
   SxuState deriving (Bits, Eq, FShow);

module mkScalarUnit#(
   VMEM_IFC#(vmDepth, sublanes, lanes)      vmem,
   VRegFile_IFC#(numRegs, sublanes, lanes)  vrf,
   VPU_IFC#(sublanes, lanes)                vpu,
   XLU_IFC#(sublanes, lanes)                xlu
)(SXU_IFC#(progDepth, vmDepth, numRegs, sublanes, lanes))
   provisos(
      Add#(1, p_, progDepth),
      Add#(1, v_, vmDepth),
      Add#(1, r_, numRegs),
      Add#(1, s_, sublanes),
      Add#(1, l_, lanes),
      Log#(progDepth, logProg),
      Bits#(Vector#(sublanes, Vector#(lanes, Int#(32))), vsz),
      Bits#(SxuInstr, isz),
      Add#(0, sublanes, lanes)  // square vregs (XLU requirement)
   );

   RegFile#(UInt#(TLog#(progDepth)), SxuInstr) prog <- mkRegFileFull;

   Reg#(SxuState)                pc_state <- mkReg(SXU_IDLE);
   Reg#(UInt#(TLog#(progDepth))) pc       <- mkReg(0);
   Reg#(UInt#(TLog#(progDepth))) progLen  <- mkReg(0);
   Reg#(SxuInstr)                curInstr <- mkRegU;

   // FETCH: read instruction at pc, advance to EXEC
   rule do_fetch (pc_state == SXU_FETCH);
      let instr = prog.sub(pc);
      curInstr <= instr;
      case (instr.op)
         SXU_LOAD_VREG:    pc_state <= SXU_EXEC_LOAD_REQ;
         SXU_STORE_VREG:   pc_state <= SXU_EXEC_STORE;
         SXU_DISPATCH_VPU: pc_state <= SXU_EXEC_VPU;
         SXU_WAIT_VPU:     pc_state <= SXU_EXEC_WAIT_VPU;
         SXU_HALT:         pc_state <= SXU_HALTED;
      endcase
   endrule

   // LOAD step 1: issue VMEM readReq
   rule do_load_req (pc_state == SXU_EXEC_LOAD_REQ);
      vmem.readReq(extend(curInstr.vmemAddr));
      pc_state <= SXU_EXEC_LOAD_RESP;
   endrule

   // LOAD step 2: read response, write to VRegFile
   rule do_load_resp (pc_state == SXU_EXEC_LOAD_RESP);
      vrf.write(extend(curInstr.vregDst), vmem.readResp);
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // STORE: read VRegFile, write to VMEM
   rule do_store (pc_state == SXU_EXEC_STORE);
      let data = vrf.read(extend(curInstr.vregSrc));
      vmem.write(extend(curInstr.vmemAddr), data);
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // DISPATCH_VPU: read two source vregs, dispatch VPU
   rule do_vpu (pc_state == SXU_EXEC_VPU);
      let s1 = vrf.read(extend(curInstr.vregSrc));
      let s2 = vrf.read(extend(curInstr.vregSrc2));
      vpu.execute(curInstr.vpuOp, s1, s2);
      // Write result to vregDst next cycle (VPU is 1-cycle latency)
      // We'll handle the write in a follow-up rule
      pc_state <= SXU_EXEC_WAIT_VPU;
   endrule

   // WAIT_VPU: collect VPU result into VRegFile, advance
   rule do_wait_vpu (pc_state == SXU_EXEC_WAIT_VPU);
      vrf.write(extend(curInstr.vregDst), vpu.result);
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   method Action loadInstr(UInt#(TLog#(progDepth)) addr, SxuInstr instr);
      prog.upd(addr, instr);
   endmethod

   method Action start(UInt#(TLog#(progDepth)) len) if (pc_state == SXU_IDLE);
      progLen  <= len;
      pc       <= 0;
      pc_state <= SXU_FETCH;
   endmethod

   method Bool isDone;
      return pc_state == SXU_HALTED;
   endmethod

endmodule

export SxuOpCode(..);
export SxuInstr(..);
export SXU_IFC(..);
export mkScalarUnit;

endpackage
```

**Note on DISPATCH_VPU + WAIT_VPU:** When the SXU encounters DISPATCH_VPU it dispatches and immediately transitions to WAIT_VPU state (which collects the 1-cycle-latency result). The explicit `SXU_WAIT_VPU` instruction in the program is redundant if the SXU handles the wait implicitly, but is kept for clarity. The program in the testbench includes the explicit WAIT_VPU instruction; the SXU DISPATCH_VPU handler itself transitions to `SXU_EXEC_WAIT_VPU`, so the explicit WAIT_VPU instruction in the program should advance pc normally. To avoid double-waiting: the `DISPATCH_VPU` rule transitions state to `SXU_EXEC_WAIT_VPU` and the result is collected there, then `pc` is incremented to skip past the (already executed) WAIT_VPU instruction.

**Correction:** The testbench program has DISPATCH_VPU at pc=2 and WAIT_VPU at pc=3. When DISPATCH_VPU fires, it goes to `SXU_EXEC_WAIT_VPU` (collecting result, writing vreg, incrementing pc to 3). Then pc=3 is WAIT_VPU. In `SXU_EXEC_WAIT_VPU` state, the `do_wait_vpu` rule already ran (advancing to pc=3). Now the fetch gets WAIT_VPU, transitions to `SXU_EXEC_WAIT_VPU` again — but the VPU result was written in the previous `do_wait_vpu` firing. To avoid this issue, simplify: **remove the explicit WAIT_VPU from the program** and have the SXU implicitly wait one cycle after DISPATCH_VPU. Change the program to:

```
pc0: LOAD_VREG  vmemAddr=0 vregDst=0
pc1: LOAD_VREG  vmemAddr=1 vregDst=1
pc2: DISPATCH_VPU ADD vsrc=0 vsrc2=1 vdst=2   (SXU waits internally for 1 cycle)
pc3: STORE_VREG vmemAddr=2 vregSrc=2
pc4: HALT
```

And update the testbench `loadInstr` calls accordingly (5 instructions, len=5).

- [ ] **Step 2: Update testbench for 5-instruction program (no explicit WAIT_VPU)**

In `test/TbScalarUnit.bsv`, replace the `load_program` rule with:

```bsv
   rule load_program (cycle == 1);
      sxu.loadInstr(0, SxuInstr { op: SXU_LOAD_VREG,    vmemAddr:0, vregDst:0, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0 });
      sxu.loadInstr(1, SxuInstr { op: SXU_LOAD_VREG,    vmemAddr:1, vregDst:1, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0 });
      sxu.loadInstr(2, SxuInstr { op: SXU_DISPATCH_VPU, vmemAddr:0, vregDst:2, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:1 });
      sxu.loadInstr(3, SxuInstr { op: SXU_STORE_VREG,   vmemAddr:2, vregDst:0, vregSrc:2, vpuOp:VPU_ADD, vregSrc2:0 });
      sxu.loadInstr(4, SxuInstr { op: SXU_HALT,         vmemAddr:0, vregDst:0, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0 });
      $display("Cycle %0d: program loaded (5 instrs)", cycle);
   endrule
```

And change `sxu.start(6)` to `sxu.start(5)`.

- [ ] **Step 3: Run — expect PASS**

```bash
cd /home/hanwang/p/tinytpu && make test-sxu
```
Expected:
```
Cycle 0: VMEM preloaded
Cycle 1: program loaded (5 instrs)
Cycle 2: SXU started
Cycle N: SXU done, reading VMEM[2]
Cycle N+1: PASS SXU program result
Results: 1 passed, 0 failed
```

- [ ] **Step 4: Run regression + commit**

```bash
cd /home/hanwang/p/tinytpu && make test
git add src/ScalarUnit.bsv test/TbScalarUnit.bsv Makefile
git commit --author="Han Wang <h1337h4x0r@gmail.com>" -m "feat: add ScalarUnit microprogram sequencer controlling VMEM/VRegFile/VPU/XLU"
```

---

*Plan created: 2026-04-08*
