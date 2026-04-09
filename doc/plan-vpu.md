# VPU (Vector Processing Unit) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Vector Processing Unit (VPU) — a 128-lane SIMD functional unit that performs element-wise arithmetic, activation functions, and lane-wide reductions on vreg tiles.

**Architecture:** Combinatorial ALU applied element-wise across all (sublanes × lanes) positions, with a registered output (1-cycle latency). SUM_REDUCE uses a compile-time unrolled adder tree across the lane dimension, broadcasting the scalar sum to all lanes. Parameterized by `sublanes` and `lanes`. Depends on no other new modules.

**Tech Stack:** Bluespec SystemVerilog (BSV), BSC compiler, Bluesim simulator, GNU Make. Follows patterns in `src/XLU.bsv`.

---

## Operations

| Op | Description | src2 used? |
|---|---|---|
| `VPU_ADD` | element-wise: `dst[s][l] = src1[s][l] + src2[s][l]` | Yes |
| `VPU_MUL` | element-wise: `dst[s][l] = src1[s][l] * src2[s][l]` | Yes |
| `VPU_RELU` | element-wise: `dst[s][l] = max(src1[s][l], 0)` | No |
| `VPU_MAX` | element-wise: `dst[s][l] = max(src1[s][l], src2[s][l])` | Yes |
| `VPU_SUM_REDUCE` | sum all lanes in each sublane row, broadcast to all lanes: `dst[s][l] = Σ src1[s][0..lanes-1]` | No |

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/VPU.bsv` | Create | VpuOp enum, VPU_IFC interface, `mkVPU` module |
| `test/TbVPU.bsv` | Create | Sequential tests for all 5 operations |
| `Makefile` | Modify | Add `test-vpu` target |

---

## Task 1: Makefile + failing VPU_ADD test

**Files:**
- Create: `test/TbVPU.bsv`
- Modify: `Makefile`

- [ ] **Step 1: Add Makefile entries**

After existing dependency lines, add:
```makefile
$(BUILDDIR)/TbVPU.bo: $(BUILDDIR)/VPU.bo
```

Link rule:
```makefile
$(BUILDDIR)/mkTbVPU.bexe: $(BUILDDIR)/TbVPU.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbVPU $(BUILDDIR)/mkTbVPU.ba
```

Test target:
```makefile
test-vpu: $(BUILDDIR)/mkTbVPU.bexe
	$<
```

Add `test-vpu` to `.PHONY` and to the `test` target.

- [ ] **Step 2: Write failing test (VPU_ADD only)**

Create `test/TbVPU.bsv`:

```bsv
package TbVPU;

import Vector :: *;
import VPU :: *;

(* synthesize *)
module mkTbVPU();

   VPU_IFC#(4, 4) vpu <- mkVPU;

   Reg#(UInt#(8)) cycle  <- mkReg(0);
   Reg#(UInt#(8)) passed <- mkReg(0);
   Reg#(UInt#(8)) failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 50) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   // Test 1: VPU_ADD
   // src1 row0: [1, 2, 3, 4], src2 row0: [10, 20, 30, 40]
   // expected row0: [11, 22, 33, 44]
   rule dispatch_add (cycle == 0);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 1; s1[0][1] = 2; s1[0][2] = 3;  s1[0][3] = 4;
      s2[0][0] = 10; s2[0][1] = 20; s2[0][2] = 30; s2[0][3] = 40;
      vpu.execute(VPU_ADD, s1, s2);
      $display("Cycle %0d: dispatched VPU_ADD", cycle);
   endrule

   rule check_add (cycle == 1);
      let res = vpu.result;
      Bool ok = (res[0][0] == 11 && res[0][1] == 22 && res[0][2] == 33 && res[0][3] == 44);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_ADD", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_ADD got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
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
cd /home/hanwang/p/tinytpu && make test-vpu
```
Expected: `No rule to make target 'build/VPU.bo'`.

---

## Task 2: Implement `src/VPU.bsv` (ADD + MUL + RELU + MAX)

**Files:**
- Create: `src/VPU.bsv`

- [ ] **Step 1: Write VPU.bsv**

```bsv
package VPU;

import Vector :: *;

typedef enum { VPU_ADD, VPU_MUL, VPU_RELU, VPU_MAX, VPU_SUM_REDUCE }
   VpuOp deriving (Bits, Eq, FShow);

interface VPU_IFC#(numeric type sublanes, numeric type lanes);
   method Action execute(
      VpuOp op,
      Vector#(sublanes, Vector#(lanes, Int#(32))) src1,
      Vector#(sublanes, Vector#(lanes, Int#(32))) src2
   );
   method Vector#(sublanes, Vector#(lanes, Int#(32))) result;
endinterface

// Sum all lanes in one row: unrolled adder tree
function Int#(32) lane_sum(Vector#(lanes, Int#(32)) row)
   provisos(Add#(1, l_, lanes));
   Int#(32) acc = 0;
   for (Integer i = 0; i < valueOf(lanes); i = i + 1)
      acc = acc + row[i];
   return acc;
endfunction

module mkVPU(VPU_IFC#(sublanes, lanes))
   provisos(
      Add#(1, s_, sublanes),
      Add#(1, l_, lanes),
      Bits#(Vector#(sublanes, Vector#(lanes, Int#(32))), vsz)
   );

   Reg#(Vector#(sublanes, Vector#(lanes, Int#(32)))) resultReg <- mkRegU;

   method Action execute(
      VpuOp op,
      Vector#(sublanes, Vector#(lanes, Int#(32))) src1,
      Vector#(sublanes, Vector#(lanes, Int#(32))) src2
   );
      Vector#(sublanes, Vector#(lanes, Int#(32))) res = newVector;
      for (Integer s = 0; s < valueOf(sublanes); s = s + 1) begin
         Vector#(lanes, Int#(32)) row = newVector;
         case (op)
            VPU_ADD: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = src1[s][l] + src2[s][l];
            end
            VPU_MUL: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = src1[s][l] * src2[s][l];
            end
            VPU_RELU: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = (src1[s][l] > 0) ? src1[s][l] : 0;
            end
            VPU_MAX: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = (src1[s][l] > src2[s][l]) ? src1[s][l] : src2[s][l];
            end
            VPU_SUM_REDUCE: begin
               Int#(32) s_val = lane_sum(src1[s]);
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = s_val;
            end
         endcase
         res[s] = row;
      end
      resultReg <= res;
   endmethod

   method Vector#(sublanes, Vector#(lanes, Int#(32))) result;
      return resultReg;
   endmethod

endmodule

export VpuOp(..);
export VPU_IFC(..);
export mkVPU;

endpackage
```

- [ ] **Step 2: Run — expect PASS (1 test)**

```bash
cd /home/hanwang/p/tinytpu && make test-vpu
```
Expected:
```
Cycle 0: dispatched VPU_ADD
Cycle 1: PASS VPU_ADD
Results: 1 passed, 0 failed
```

- [ ] **Step 3: Commit**

```bash
git add src/VPU.bsv test/TbVPU.bsv Makefile
git commit --author="Han Wang <h1337h4x0r@gmail.com>" -m "feat: add VPU with ADD/MUL/RELU/MAX/SUM_REDUCE"
```

---

## Task 3: Add remaining operation tests

**Files:**
- Modify: `test/TbVPU.bsv`

- [ ] **Step 1: Replace `finish` at cycle==2 with remaining tests**

Replace `rule finish (cycle == 2)` with:

```bsv
   // Test 2: VPU_MUL
   // [2,3,4,5] * [3,4,5,6] = [6,12,20,30]
   rule dispatch_mul (cycle == 2);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 2; s1[0][1] = 3; s1[0][2] = 4; s1[0][3] = 5;
      s2[0][0] = 3; s2[0][1] = 4; s2[0][2] = 5; s2[0][3] = 6;
      vpu.execute(VPU_MUL, s1, s2);
      $display("Cycle %0d: dispatched VPU_MUL", cycle);
   endrule

   rule check_mul (cycle == 3);
      let res = vpu.result;
      Bool ok = (res[0][0] == 6 && res[0][1] == 12 && res[0][2] == 20 && res[0][3] == 30);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_MUL", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_MUL got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 3: VPU_RELU
   // src1 row0: [-5, -1, 0, 7] -> [0, 0, 0, 7]
   rule dispatch_relu (cycle == 4);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = -5; s1[0][1] = -1; s1[0][2] = 0; s1[0][3] = 7;
      vpu.execute(VPU_RELU, s1, s2);
      $display("Cycle %0d: dispatched VPU_RELU", cycle);
   endrule

   rule check_relu (cycle == 5);
      let res = vpu.result;
      Bool ok = (res[0][0] == 0 && res[0][1] == 0 && res[0][2] == 0 && res[0][3] == 7);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_RELU", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_RELU got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 4: VPU_MAX
   // max([1,5,2,6], [3,4,3,4]) = [3,5,3,6]
   rule dispatch_max (cycle == 6);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 1; s1[0][1] = 5; s1[0][2] = 2; s1[0][3] = 6;
      s2[0][0] = 3; s2[0][1] = 4; s2[0][2] = 3; s2[0][3] = 4;
      vpu.execute(VPU_MAX, s1, s2);
      $display("Cycle %0d: dispatched VPU_MAX", cycle);
   endrule

   rule check_max (cycle == 7);
      let res = vpu.result;
      Bool ok = (res[0][0] == 3 && res[0][1] == 5 && res[0][2] == 3 && res[0][3] == 6);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_MAX", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_MAX got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 5: VPU_SUM_REDUCE
   // sum([10, 20, 30, 40]) = 100, broadcast -> [100, 100, 100, 100]
   rule dispatch_sum (cycle == 8);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 10; s1[0][1] = 20; s1[0][2] = 30; s1[0][3] = 40;
      vpu.execute(VPU_SUM_REDUCE, s1, s2);
      $display("Cycle %0d: dispatched VPU_SUM_REDUCE", cycle);
   endrule

   rule check_sum (cycle == 9);
      let res = vpu.result;
      Bool ok = (res[0][0] == 100 && res[0][1] == 100 && res[0][2] == 100 && res[0][3] == 100);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_SUM_REDUCE", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_SUM_REDUCE got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 10);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0) $finish(0); else $finish(1);
   endrule
```

- [ ] **Step 2: Run — expect 5 tests PASS**

```bash
cd /home/hanwang/p/tinytpu && make test-vpu
```
Expected:
```
Cycle 1: PASS VPU_ADD
Cycle 3: PASS VPU_MUL
Cycle 5: PASS VPU_RELU
Cycle 7: PASS VPU_MAX
Cycle 9: PASS VPU_SUM_REDUCE
Results: 5 passed, 0 failed
```

- [ ] **Step 3: Run regression + commit**

```bash
cd /home/hanwang/p/tinytpu && make test
git add test/TbVPU.bsv
git commit --author="Han Wang <h1337h4x0r@gmail.com>" -m "test: add VPU tests for all 5 operations"
```

---

*Plan created: 2026-04-08*
