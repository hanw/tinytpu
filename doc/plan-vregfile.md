# VRegFile (Vector Register File) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the vector register file (VRegFile) — the working memory shared between the VPU, XLU, and Scalar Unit, storing named vreg tiles that execution units read and write.

**Architecture:** A RegFile of vreg-sized tiles (Vector#(sublanes, Vector#(lanes, Int#(32)))) indexed by register number. Writes are registered (write in cycle N, visible at start of cycle N+1). Reads are combinatorial (read port returns current value immediately), consistent with how a register file works in a VLIW pipeline.

**Tech Stack:** Bluespec SystemVerilog (BSV), BSC compiler, Bluesim simulator, GNU Make. Follows patterns in `src/WeightSRAM.bsv`.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/VRegFile.bsv` | Create | VRegFile interface and `mkVRegFile` module |
| `test/TbVRegFile.bsv` | Create | Write/read, multi-register, write-then-read ordering tests |
| `Makefile` | Modify | Add `test-vregfile` target |

---

## Task 1: Makefile target + failing test

**Files:**
- Create: `test/TbVRegFile.bsv`
- Modify: `Makefile`

- [ ] **Step 1: Add Makefile entries**

After the `$(BUILDDIR)/TbVMEM.bo` dependency line, add:
```makefile
$(BUILDDIR)/TbVRegFile.bo: $(BUILDDIR)/VRegFile.bo
```

Add link rule:
```makefile
$(BUILDDIR)/mkTbVRegFile.bexe: $(BUILDDIR)/TbVRegFile.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbVRegFile $(BUILDDIR)/mkTbVRegFile.ba
```

Add test target:
```makefile
test-vregfile: $(BUILDDIR)/mkTbVRegFile.bexe
	$<
```

Add `test-vregfile` to `.PHONY` and to the `test` target.

- [ ] **Step 2: Write failing test**

Create `test/TbVRegFile.bsv`:

```bsv
package TbVRegFile;

import Vector :: *;
import VRegFile :: *;

(* synthesize *)
module mkTbVRegFile();

   // 16 vregs, sublanes=4, lanes=4
   VRegFile_IFC#(16, 4, 4) vrf <- mkVRegFile;

   Reg#(UInt#(8)) cycle  <- mkReg(0);
   Reg#(UInt#(8)) passed <- mkReg(0);
   Reg#(UInt#(8)) failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 50) begin
         $display("FAIL: timeout"); $finish(1);
      end
   endrule

   // Test 1: write vreg 0, read back next cycle
   rule write_vreg0 (cycle == 0);
      Vector#(4, Vector#(4, Int#(32))) v = replicate(replicate(0));
      v[0][0] = 7; v[1][2] = 13; v[3][3] = 55;
      vrf.write(0, v);
      $display("Cycle %0d: wrote vreg 0", cycle);
   endrule

   rule check_vreg0 (cycle == 1);
      let v = vrf.read(0);
      Bool ok = (v[0][0] == 7 && v[1][2] == 13 && v[3][3] == 55);
      if (ok) begin
         $display("Cycle %0d: PASS write/read vreg 0", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL vreg 0: [0][0]=%0d [3][3]=%0d",
            cycle, v[0][0], v[3][3]);
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 2);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
```

- [ ] **Step 3: Run — expect compile error**

```bash
cd /home/hanwang/p/tinytpu && make test-vregfile
```
Expected: `No rule to make target 'build/VRegFile.bo'`.

---

## Task 2: Implement `src/VRegFile.bsv`

**Files:**
- Create: `src/VRegFile.bsv`

- [ ] **Step 1: Write VRegFile.bsv**

```bsv
package VRegFile;

import Vector :: *;
import RegFile :: *;

interface VRegFile_IFC#(numeric type numRegs, numeric type sublanes, numeric type lanes);
   method Action write(UInt#(TLog#(numRegs)) idx,
                       Vector#(sublanes, Vector#(lanes, Int#(32))) data);
   method Vector#(sublanes, Vector#(lanes, Int#(32))) read(UInt#(TLog#(numRegs)) idx);
endinterface

module mkVRegFile(VRegFile_IFC#(numRegs, sublanes, lanes))
   provisos(
      Add#(1, r_, numRegs),
      Add#(1, s_, sublanes),
      Add#(1, l_, lanes),
      Bits#(Vector#(sublanes, Vector#(lanes, Int#(32))), vsz)
   );

   RegFile#(UInt#(TLog#(numRegs)), Vector#(sublanes, Vector#(lanes, Int#(32))))
      rf <- mkRegFileFull;

   method Action write(UInt#(TLog#(numRegs)) idx,
                       Vector#(sublanes, Vector#(lanes, Int#(32))) data);
      rf.upd(idx, data);
   endmethod

   method Vector#(sublanes, Vector#(lanes, Int#(32))) read(UInt#(TLog#(numRegs)) idx);
      return rf.sub(idx);
   endmethod

endmodule

export VRegFile_IFC(..);
export mkVRegFile;

endpackage
```

- [ ] **Step 2: Run — expect PASS**

```bash
cd /home/hanwang/p/tinytpu && make test-vregfile
```
Expected:
```
Cycle 0: wrote vreg 0
Cycle 1: PASS write/read vreg 0
Results: 1 passed, 0 failed
```

- [ ] **Step 3: Commit**

```bash
git add src/VRegFile.bsv test/TbVRegFile.bsv Makefile
git commit --author="Han Wang <h1337h4x0r@gmail.com>" -m "feat: add VRegFile vector register file"
```

---

## Task 3: Multi-register isolation test

**Files:**
- Modify: `test/TbVRegFile.bsv`

- [ ] **Step 1: Extend testbench — replace `finish` at cycle==2**

Replace `rule finish (cycle == 2)` with:

```bsv
   // Test 2: write vregs 3 and 12, verify no cross-contamination
   rule write_multi (cycle == 2);
      Vector#(4, Vector#(4, Int#(32))) vA = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) vB = replicate(replicate(0));
      vA[0][0] = 100;
      vB[0][0] = 200;
      vrf.write(3, vA);
      vrf.write(12, vB);
      $display("Cycle %0d: wrote vreg 3=100, vreg 12=200", cycle);
   endrule

   rule check_multi (cycle == 3);
      let vA = vrf.read(3);
      let vB = vrf.read(12);
      Bool ok = (vA[0][0] == 100 && vB[0][0] == 200);
      if (ok) begin
         $display("Cycle %0d: PASS multi-register isolation", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL vreg3=%0d vreg12=%0d", cycle, vA[0][0], vB[0][0]);
         failed <= failed + 1;
      end
   endrule

   // Test 3: overwrite vreg 3, verify vreg 12 unchanged
   rule overwrite (cycle == 4);
      Vector#(4, Vector#(4, Int#(32))) vNew = replicate(replicate(0));
      vNew[0][0] = 999;
      vrf.write(3, vNew);
      $display("Cycle %0d: overwrote vreg 3 with 999", cycle);
   endrule

   rule check_overwrite (cycle == 5);
      let vA = vrf.read(3);
      let vB = vrf.read(12);
      Bool ok = (vA[0][0] == 999 && vB[0][0] == 200);
      if (ok) begin
         $display("Cycle %0d: PASS overwrite isolation", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL overwrite: vreg3=%0d vreg12=%0d",
            cycle, vA[0][0], vB[0][0]);
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 6);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0) $finish(0); else $finish(1);
   endrule
```

- [ ] **Step 2: Run — expect 3 tests PASS**

```bash
cd /home/hanwang/p/tinytpu && make test-vregfile
```
Expected:
```
Cycle 1: PASS write/read vreg 0
Cycle 3: PASS multi-register isolation
Cycle 5: PASS overwrite isolation
Results: 3 passed, 0 failed
```

- [ ] **Step 3: Run regression + commit**

```bash
cd /home/hanwang/p/tinytpu && make test
git add test/TbVRegFile.bsv
git commit --author="Han Wang <h1337h4x0r@gmail.com>" -m "test: add VRegFile isolation tests"
```

---

*Plan created: 2026-04-08*
