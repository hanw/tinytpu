# VMEM (Vector Memory / On-Chip Scratchpad) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a unified on-chip SRAM scratchpad (VMEM) that replaces the separate WeightSRAM and ActivationSRAM with a single address-space tile store serving all TensorCore sub-units.

**Architecture:** VMEM stores vreg-sized tiles (Vector#(sublanes, Vector#(lanes, Int#(32)))) at addressed locations. It exposes one write port and one pipelined read port (readReq + readResp). Reads have 1-cycle latency (request in cycle N, response readable in cycle N+1), matching existing SRAM patterns in the codebase.

**Tech Stack:** Bluespec SystemVerilog (BSV), BSC compiler, Bluesim simulator, GNU Make. Follows patterns in `src/WeightSRAM.bsv` and `src/ActivationSRAM.bsv`.

---

## Why This Replaces WeightSRAM + ActivationSRAM

The existing `WeightSRAM` and `ActivationSRAM` are separate modules with different element types. VMEM unifies them into a single address space where the Scalar Unit controls allocation. This matches the TPU spec (Section 3.5): a single 16–64 MB scratchpad with compiler-managed regions for weights, activations, and accumulators.

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/VMEM.bsv` | Create | VMEM types, interface, `mkVMEM` module |
| `test/TbVMEM.bsv` | Create | Write/read testbench: basic, multi-addr, ordering |
| `Makefile` | Modify | Add `test-vmem` target |

---

## Task 1: Makefile target + failing write/read test

**Files:**
- Create: `test/TbVMEM.bsv`
- Modify: `Makefile`

- [ ] **Step 1: Add Makefile entries**

Add after the `$(BUILDDIR)/TbXLU.bo` dependency line:
```makefile
$(BUILDDIR)/TbVMEM.bo: $(BUILDDIR)/VMEM.bo
```

Add link rule after the mkTbXLU rule:
```makefile
$(BUILDDIR)/mkTbVMEM.bexe: $(BUILDDIR)/TbVMEM.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbVMEM $(BUILDDIR)/mkTbVMEM.ba
```

Add test target:
```makefile
test-vmem: $(BUILDDIR)/mkTbVMEM.bexe
	$<
```

Add `test-vmem` to the `.PHONY` line and to the `test` target.

- [ ] **Step 2: Write failing test**

Create `test/TbVMEM.bsv`:

```bsv
package TbVMEM;

import Vector :: *;
import VMEM :: *;

(* synthesize *)
module mkTbVMEM();

   // depth=16, sublanes=4, lanes=4
   VMEM_IFC#(16, 4, 4) mem <- mkVMEM;

   Reg#(UInt#(8)) cycle  <- mkReg(0);
   Reg#(UInt#(8)) passed <- mkReg(0);
   Reg#(UInt#(8)) failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 50) begin
         $display("FAIL: timeout"); $finish(1);
      end
   endrule

   // Test 1: write addr 0, read it back
   rule write_0 (cycle == 0);
      Vector#(4, Vector#(4, Int#(32))) tile = replicate(replicate(0));
      tile[0][0] = 1; tile[0][1] = 2; tile[1][0] = 5; tile[3][3] = 99;
      mem.write(0, tile);
      $display("Cycle %0d: wrote tile to addr 0", cycle);
   endrule

   rule read_req_0 (cycle == 1);
      mem.readReq(0);
      $display("Cycle %0d: issued readReq addr 0", cycle);
   endrule

   rule check_0 (cycle == 2);
      let t = mem.readResp;
      Bool ok = (t[0][0] == 1 && t[0][1] == 2 && t[1][0] == 5 && t[3][3] == 99);
      if (ok) begin
         $display("Cycle %0d: PASS write/read addr 0", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL write/read addr 0 [0][0]=%0d [3][3]=%0d",
            cycle, t[0][0], t[3][3]);
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 3);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
```

- [ ] **Step 3: Run — expect compile error (mkVMEM not defined)**

```bash
cd /home/hanwang/p/tinytpu && make test-vmem
```
Expected: `No rule to make target 'build/VMEM.bo'` or similar. Confirms wiring is correct.

---

## Task 2: Implement `src/VMEM.bsv`

**Files:**
- Create: `src/VMEM.bsv`

- [ ] **Step 1: Write VMEM.bsv**

```bsv
package VMEM;

import Vector :: *;
import RegFile :: *;

interface VMEM_IFC#(numeric type depth, numeric type sublanes, numeric type lanes);
   method Action write(UInt#(TLog#(depth)) addr,
                       Vector#(sublanes, Vector#(lanes, Int#(32))) data);
   method Action readReq(UInt#(TLog#(depth)) addr);
   method Vector#(sublanes, Vector#(lanes, Int#(32))) readResp;
endinterface

module mkVMEM(VMEM_IFC#(depth, sublanes, lanes))
   provisos(
      Add#(1, d_, depth),
      Add#(1, s_, sublanes),
      Add#(1, l_, lanes),
      Log#(depth, logDepth),
      Bits#(Vector#(sublanes, Vector#(lanes, Int#(32))), vsz)
   );

   RegFile#(UInt#(TLog#(depth)), Vector#(sublanes, Vector#(lanes, Int#(32))))
      mem <- mkRegFileFull;

   Reg#(Vector#(sublanes, Vector#(lanes, Int#(32)))) resp <- mkRegU;

   method Action write(UInt#(TLog#(depth)) addr,
                       Vector#(sublanes, Vector#(lanes, Int#(32))) data);
      mem.upd(addr, data);
   endmethod

   method Action readReq(UInt#(TLog#(depth)) addr);
      resp <= mem.sub(addr);
   endmethod

   method Vector#(sublanes, Vector#(lanes, Int#(32))) readResp;
      return resp;
   endmethod

endmodule

export VMEM_IFC(..);
export mkVMEM;

endpackage
```

- [ ] **Step 2: Run — expect PASS**

```bash
cd /home/hanwang/p/tinytpu && make test-vmem
```
Expected:
```
Cycle 0: wrote tile to addr 0
Cycle 1: issued readReq addr 0
Cycle 2: PASS write/read addr 0
Results: 1 passed, 0 failed
```

- [ ] **Step 3: Commit**

```bash
git add src/VMEM.bsv test/TbVMEM.bsv Makefile
git commit --author="Han Wang <h1337h4x0r@gmail.com>" -m "feat: add VMEM unified scratchpad SRAM"
```

---

## Task 3: Multi-address isolation test

**Files:**
- Modify: `test/TbVMEM.bsv`

- [ ] **Step 1: Replace `finish` at cycle==3 with multi-addr test**

Replace `rule finish (cycle == 3)` and everything after `check_0` with:

```bsv
   // Test 2: write two different addresses, verify no aliasing
   rule write_multi (cycle == 3);
      Vector#(4, Vector#(4, Int#(32))) tileA = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) tileB = replicate(replicate(0));
      tileA[0][0] = 42;
      tileB[0][0] = 99;
      mem.write(3, tileA);
      mem.write(7, tileB);
      $display("Cycle %0d: wrote addr 3 (42) and addr 7 (99)", cycle);
   endrule

   rule read_req_3 (cycle == 4);
      mem.readReq(3);
   endrule

   rule check_addr3 (cycle == 5);
      let t = mem.readResp;
      Bool ok = (t[0][0] == 42);
      if (ok) begin
         $display("Cycle %0d: PASS addr 3 isolation", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL addr 3 got %0d", cycle, t[0][0]);
         failed <= failed + 1;
      end
   endrule

   rule read_req_7 (cycle == 6);
      mem.readReq(7);
   endrule

   rule check_addr7 (cycle == 7);
      let t = mem.readResp;
      Bool ok = (t[0][0] == 99);
      if (ok) begin
         $display("Cycle %0d: PASS addr 7 isolation", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL addr 7 got %0d", cycle, t[0][0]);
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 8);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0) $finish(0); else $finish(1);
   endrule
```

- [ ] **Step 2: Run — expect 3 tests PASS**

```bash
cd /home/hanwang/p/tinytpu && make test-vmem
```
Expected:
```
Cycle 2: PASS write/read addr 0
Cycle 5: PASS addr 3 isolation
Cycle 7: PASS addr 7 isolation
Results: 3 passed, 0 failed
```

- [ ] **Step 3: Run regression + commit**

```bash
cd /home/hanwang/p/tinytpu && make test
git add test/TbVMEM.bsv
git commit --author="Han Wang <h1337h4x0r@gmail.com>" -m "test: add VMEM multi-address isolation tests"
```

---

*Plan created: 2026-04-08*
