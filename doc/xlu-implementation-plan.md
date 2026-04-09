# XLU (Cross-Lane Unit) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Cross-Lane Unit (XLU) as a parameterized BSV module supporting lane permutation, rotation, broadcast, and transpose of vector register tiles.

**Architecture:** A standalone BSV module `mkXLU` with a combinatorial permutation network (barrel shifter for ROTATE, XOR-butterfly for PERMUTE, direct for BROADCAST/TRANSPOSE) and a single-cycle registered output. The XLU operates on square vector registers parameterized by `n` (sublanes = lanes = n), matching the existing project's pattern of parameterized modules.

**Tech Stack:** Bluespec SystemVerilog (BSV), BSC compiler, Bluesim simulator, GNU Make. Matches patterns in `src/Controller.bsv` and `test/TbAccelerator4x4.bsv`.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/XLU.bsv` | Create | XLU types, helper functions, `mkXLU` module |
| `test/TbXLU.bsv` | Create | Sequential testbench: ROTATE, BROADCAST, PERMUTE, TRANSPOSE |
| `Makefile` | Modify | Add `test-xlu` target + dependency chain |

---

## Background: Key Design Decisions

**Square vregs only:** The XLU interface uses `n` for both sublanes and lanes (`XLU_IFC#(n, n)`). The BSV proviso `Add#(0, sublanes, lanes)` (i.e., `sublanes == lanes`) enforces this at compile time, enabling TRANSPOSE to swap the two axes cleanly.

**Combinatorial permutation, registered output:** The helper functions (`lane_rotate`, `lane_butterfly`, `lane_broadcast`) are purely combinatorial. `mkXLU` registers the result on dispatch — result is readable the cycle after `execute*` is called. This is correct-but-simple; a pipelined version would add registers between butterfly stages.

**Barrel shifter for ROTATE:** `output[i] = input[(i + K) mod n]`. Decomposed into `TLog#(n)` stages: stage `k` shifts by `2^k` positions if bit `k` of `K` is set. This is NOT the same as the XOR-butterfly used for PERMUTE.

**XOR-butterfly for PERMUTE:** Stage `k` pairs lane `i` with lane `i XOR 2^k`. Per-lane Bool control word decides swap-or-pass. With `TLog#(n)` stages, this represents a large subset of permutations (not all — a Beneš network would need `2*TLog#(n) - 1` stages for full generality, deferred as noted in spec Section 11.1).

**Test parameters:** `n=4` (4 sublanes, 4 lanes). Fast to compile and easy to verify by hand.

---

## Task 1: Add Makefile targets and write failing ROTATE test

**Files:**
- Create: `test/TbXLU.bsv`
- Modify: `Makefile`

- [ ] **Step 1: Add Makefile targets** (edit lines after line 54 in `Makefile`)

Add to `Makefile` after the `$(BUILDDIR)/TbAccelerator4x4.bo` dependency line:

```makefile
$(BUILDDIR)/TbXLU.bexe: $(BUILDDIR)/TbXLU.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbXLU $(BUILDDIR)/mkTbXLU.ba

test-xlu: $(BUILDDIR)/TbXLU.bexe
	$<
```

And add `$(BUILDDIR)/TbXLU.bo: $(BUILDDIR)/XLU.bo` in the dependencies section.

Update the `test` phony target to include `test-xlu`:
```makefile
test: test-pe test-array test-accel test-4x4 test-xlu
```

And add `test-xlu` to `.PHONY`.

- [ ] **Step 2: Write TbXLU.bsv with ROTATE test only**

Create `test/TbXLU.bsv`:

```bsv
package TbXLU;

import Vector :: *;
import XLU :: *;

(* synthesize *)
module mkTbXLU();

   XLU_IFC#(4, 4) xlu <- mkXLU;

   Reg#(UInt#(8)) cycle   <- mkReg(0);
   Reg#(UInt#(8)) passed  <- mkReg(0);
   Reg#(UInt#(8)) failed  <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 50) begin
         $display("FAIL: test timed out at cycle %0d", cycle);
         $finish(1);
      end
   endrule

   // ---- Test 1: ROTATE ----
   // Input row0: [0, 1, 2, 3], rotate by 1 -> [1, 2, 3, 0]
   // Input row1: [10, 20, 30, 40], rotate by 1 -> [20, 30, 40, 10]
   rule dispatch_rotate (cycle == 0);
      Vector#(4, Vector#(4, Int#(32))) src = replicate(replicate(0));
      src[0][0] = 0;  src[0][1] = 1;  src[0][2] = 2;  src[0][3] = 3;
      src[1][0] = 10; src[1][1] = 20; src[1][2] = 30; src[1][3] = 40;
      xlu.executeRotate(src, 1);
      $display("Cycle %0d: dispatched ROTATE by 1", cycle);
   endrule

   rule check_rotate (cycle == 1);
      let res = xlu.result;
      Bool ok = (res[0][0] == 1  && res[0][1] == 2  && res[0][2] == 3  && res[0][3] == 0 &&
                 res[1][0] == 20 && res[1][1] == 30 && res[1][2] == 40 && res[1][3] == 10);
      if (ok) begin
         $display("Cycle %0d: PASS ROTATE", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL ROTATE row0=[%0d,%0d,%0d,%0d] row1=[%0d,%0d,%0d,%0d]",
            cycle,
            res[0][0], res[0][1], res[0][2], res[0][3],
            res[1][0], res[1][1], res[1][2], res[1][3]);
         failed <= failed + 1;
      end
   endrule

   // ---- (placeholder for future tests) ----

   rule finish (cycle == 2);
      $display("Results: %0d passed, %0d failed", passed + 1, failed);
      if (failed == 0)
         $finish(0);
      else
         $finish(1);
   endrule

endmodule

endpackage
```

- [ ] **Step 3: Run — expect compile error (mkXLU not yet defined)**

```bash
cd /home/hanwang/p/tinytpu && make test-xlu
```

Expected: BSC compile error mentioning unknown module `mkXLU` or package `XLU`. This confirms the test is wired up correctly.

---

## Task 2: Implement `src/XLU.bsv` with ROTATE support

**Files:**
- Create: `src/XLU.bsv`

- [ ] **Step 1: Create XLU.bsv with types, lane_rotate, and minimal mkXLU**

Create `src/XLU.bsv`:

```bsv
package XLU;

import Vector :: *;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

typedef enum { PERMUTE, ROTATE, BROADCAST, TRANSPOSE }
   XluOp deriving (Bits, Eq, FShow);

// VReg: sublanes x lanes matrix of 32-bit elements
// (sublanes == lanes for this module — enforced by proviso)

// ---------------------------------------------------------------------------
// Interface
// ---------------------------------------------------------------------------

interface XLU_IFC#(numeric type sublanes, numeric type lanes);
   // Cyclic rotate: output[s][i] = input[s][(i + amount) mod lanes]
   method Action executeRotate(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src,
      UInt#(TLog#(lanes)) amount
   );

   // Broadcast: output[s][i] = input[s][srcLane] for all i
   method Action executeBroadcast(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src,
      UInt#(TLog#(lanes)) srcLane
   );

   // Butterfly permutation: applies TLog#(lanes) XOR-swap stages
   // ctrl[k][i] = True means lane i swaps with lane (i XOR 2^k) at stage k
   method Action executePermute(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src,
      Vector#(TLog#(lanes), Vector#(lanes, Bool)) ctrl
   );

   // Transpose: output[r][c] = input[c][r]  (requires sublanes == lanes)
   method Action executeTranspose(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src
   );

   // Result is valid the cycle after any execute* call
   method Vector#(sublanes, Vector#(lanes, Int#(32))) result;

endinterface

// ---------------------------------------------------------------------------
// Helper: barrel-shifter rotation for one lane row
// output[i] = input[(i + amount) mod lanes]
// Decomposed into TLog#(lanes) stages: stage k shifts by 2^k if bit k of
// 'amount' is set.
// ---------------------------------------------------------------------------

function Vector#(lanes, t) lane_rotate(
   Vector#(lanes, t) v,
   UInt#(TLog#(lanes)) amount
) provisos(Log#(lanes, logLanes));
   Vector#(lanes, t) cur = v;
   for (Integer k = 0; k < valueOf(logLanes); k = k + 1) begin
      Integer stride = 1 << k;
      Bool do_shift = unpack((pack(amount) >> fromInteger(k))[0]);
      Vector#(lanes, t) nxt = newVector;
      for (Integer i = 0; i < valueOf(lanes); i = i + 1)
         nxt[i] = do_shift ? cur[(i + stride) % valueOf(lanes)] : cur[i];
      cur = nxt;
   end
   return cur;
endfunction

// ---------------------------------------------------------------------------
// Helper: broadcast one lane value to all lanes
// ---------------------------------------------------------------------------

function Vector#(lanes, t) lane_broadcast(
   Vector#(lanes, t) v,
   UInt#(TLog#(lanes)) srcLane
) provisos(Log#(lanes, logLanes));
   t val = v[srcLane];
   return replicate(val);
endfunction

// ---------------------------------------------------------------------------
// Helper: XOR-swap butterfly permutation for one lane row
// ctrl[k][i] = True: swap lane i with lane (i XOR 2^k) at stage k
// ---------------------------------------------------------------------------

function Vector#(lanes, t) lane_butterfly(
   Vector#(TLog#(lanes), Vector#(lanes, Bool)) ctrl,
   Vector#(lanes, t) v
) provisos(Log#(lanes, logLanes));
   Vector#(lanes, t) cur = v;
   for (Integer k = 0; k < valueOf(logLanes); k = k + 1) begin
      Integer stride = 1 << k;
      Vector#(lanes, t) nxt = newVector;
      for (Integer i = 0; i < valueOf(lanes); i = i + 1) begin
         Integer partner = xor(i, stride);
         nxt[i] = ctrl[k][i] ? cur[partner] : cur[i];
      end
      cur = nxt;
   end
   return cur;
endfunction

// ---------------------------------------------------------------------------
// Helper: 2D transpose for square vreg (sublanes == lanes == n)
// output[r][c] = input[c][r]
// ---------------------------------------------------------------------------

function Vector#(n, Vector#(n, t)) vreg_transpose(
   Vector#(n, Vector#(n, t)) v
);
   Vector#(n, Vector#(n, t)) res = newVector;
   for (Integer r = 0; r < valueOf(n); r = r + 1) begin
      res[r] = newVector;
      for (Integer c = 0; c < valueOf(n); c = c + 1)
         res[r][c] = v[c][r];
   end
   return res;
endfunction

// ---------------------------------------------------------------------------
// Module: mkXLU
// Proviso Add#(0, sublanes, lanes) enforces sublanes == lanes (for TRANSPOSE)
// ---------------------------------------------------------------------------

module mkXLU(XLU_IFC#(sublanes, lanes))
   provisos(
      Log#(lanes, logLanes),
      Add#(1, l_, lanes),        // lanes >= 2
      Add#(1, s_, sublanes),     // sublanes >= 1
      Add#(0, sublanes, lanes),  // sublanes == lanes (needed for TRANSPOSE)
      Bits#(Vector#(sublanes, Vector#(lanes, Int#(32))), vsz)
   );

   Reg#(Vector#(sublanes, Vector#(lanes, Int#(32)))) resultReg <- mkRegU;

   method Action executeRotate(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src,
      UInt#(TLog#(lanes)) amount
   );
      Vector#(sublanes, Vector#(lanes, Int#(32))) res = newVector;
      for (Integer s = 0; s < valueOf(sublanes); s = s + 1)
         res[s] = lane_rotate(src[s], amount);
      resultReg <= res;
   endmethod

   method Action executeBroadcast(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src,
      UInt#(TLog#(lanes)) srcLane
   );
      Vector#(sublanes, Vector#(lanes, Int#(32))) res = newVector;
      for (Integer s = 0; s < valueOf(sublanes); s = s + 1)
         res[s] = lane_broadcast(src[s], srcLane);
      resultReg <= res;
   endmethod

   method Action executePermute(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src,
      Vector#(TLog#(lanes), Vector#(lanes, Bool)) ctrl
   );
      Vector#(sublanes, Vector#(lanes, Int#(32))) res = newVector;
      for (Integer s = 0; s < valueOf(sublanes); s = s + 1)
         res[s] = lane_butterfly(ctrl, src[s]);
      resultReg <= res;
   endmethod

   method Action executeTranspose(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src
   );
      // sublanes == lanes enforced by proviso; safe to call vreg_transpose
      resultReg <= vreg_transpose(src);
   endmethod

   method Vector#(sublanes, Vector#(lanes, Int#(32))) result;
      return resultReg;
   endmethod

endmodule

export XluOp(..);
export XLU_IFC(..);
export mkXLU;

endpackage
```

- [ ] **Step 2: Run ROTATE test — expect PASS**

```bash
cd /home/hanwang/p/tinytpu && make test-xlu
```

Expected output:
```
Cycle 0: dispatched ROTATE by 1
Cycle 1: PASS ROTATE
Results: 1 passed, 0 failed
```

- [ ] **Step 3: Commit**

```bash
cd /home/hanwang/p/tinytpu
git add src/XLU.bsv test/TbXLU.bsv Makefile
git commit --author="Han Wang <h1337h4x0r@gmail.com>" -m "feat: add XLU module with ROTATE support and testbench"
```

---

## Task 3: Add and implement BROADCAST test

**Files:**
- Modify: `test/TbXLU.bsv`
- (No change to `src/XLU.bsv` needed — executeBroadcast already implemented)

- [ ] **Step 1: Replace `finish` rule and add BROADCAST test in TbXLU.bsv**

Replace the `finish` rule (currently at `cycle == 2`) with:

```bsv
   // ---- Test 2: BROADCAST ----
   // Input row0: [10, 20, 30, 40], broadcast lane 2 (value 30)
   // Expected row0: [30, 30, 30, 30]
   rule dispatch_broadcast (cycle == 2);
      Vector#(4, Vector#(4, Int#(32))) src = replicate(replicate(0));
      src[0][0] = 10; src[0][1] = 20; src[0][2] = 30; src[0][3] = 40;
      src[1][0] = 1;  src[1][1] = 2;  src[1][2] = 3;  src[1][3] = 4;
      xlu.executeBroadcast(src, 2);
      $display("Cycle %0d: dispatched BROADCAST lane 2", cycle);
   endrule

   rule check_broadcast (cycle == 3);
      let res = xlu.result;
      Bool ok = (res[0][0] == 30 && res[0][1] == 30 && res[0][2] == 30 && res[0][3] == 30 &&
                 res[1][0] == 3  && res[1][1] == 3  && res[1][2] == 3  && res[1][3] == 3);
      if (ok) begin
         $display("Cycle %0d: PASS BROADCAST", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL BROADCAST row0=[%0d,%0d,%0d,%0d] row1=[%0d,%0d,%0d,%0d]",
            cycle,
            res[0][0], res[0][1], res[0][2], res[0][3],
            res[1][0], res[1][1], res[1][2], res[1][3]);
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 4);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0)
         $finish(0);
      else
         $finish(1);
   endrule
```

- [ ] **Step 2: Run — expect 2 tests PASS**

```bash
cd /home/hanwang/p/tinytpu && make test-xlu
```

Expected output:
```
Cycle 0: dispatched ROTATE by 1
Cycle 1: PASS ROTATE
Cycle 2: dispatched BROADCAST lane 2
Cycle 3: PASS BROADCAST
Results: 2 passed, 0 failed
```

- [ ] **Step 3: Commit**

```bash
cd /home/hanwang/p/tinytpu
git add test/TbXLU.bsv
git commit --author="Han Wang <h1337h4x0r@gmail.com>" -m "test: add BROADCAST test to TbXLU"
```

---

## Task 4: Add and implement PERMUTE tests

**Files:**
- Modify: `test/TbXLU.bsv`
- (No change to `src/XLU.bsv` needed — executePermute already implemented)

Two sub-tests:
- **Identity permutation:** all butterfly ctrl = False → output == input
- **Reversal permutation:** all butterfly ctrl = True → output is lane-reversed

Mathematical verification for reversal on lanes=4, input=[5,10,15,20]:
- Stage 0 (stride=1, all swap): [10,5,20,15]
- Stage 1 (stride=2, all swap): [20,15,10,5] ✓

- [ ] **Step 1: Replace `finish` rule and add PERMUTE tests in TbXLU.bsv**

Replace the `finish` rule at `cycle == 4` with:

```bsv
   // ---- Test 3a: PERMUTE identity (all ctrl False -> no change) ----
   // Input row0: [5, 10, 15, 20]
   // Expected row0: [5, 10, 15, 20]
   rule dispatch_permute_id (cycle == 4);
      Vector#(4, Vector#(4, Int#(32))) src = replicate(replicate(0));
      src[0][0] = 5; src[0][1] = 10; src[0][2] = 15; src[0][3] = 20;
      // ctrl: TLog#(4)=2 stages, all False
      Vector#(2, Vector#(4, Bool)) ctrl = replicate(replicate(False));
      xlu.executePermute(src, ctrl);
      $display("Cycle %0d: dispatched PERMUTE identity", cycle);
   endrule

   rule check_permute_id (cycle == 5);
      let res = xlu.result;
      Bool ok = (res[0][0] == 5 && res[0][1] == 10 && res[0][2] == 15 && res[0][3] == 20);
      if (ok) begin
         $display("Cycle %0d: PASS PERMUTE identity", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL PERMUTE identity got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // ---- Test 3b: PERMUTE reversal (all ctrl True -> lane-reversed) ----
   // Input row0: [5, 10, 15, 20]
   // After stage0 (stride=1, all swap): [10, 5, 20, 15]
   // After stage1 (stride=2, all swap): [20, 15, 10, 5]
   // Expected row0: [20, 15, 10, 5]
   rule dispatch_permute_rev (cycle == 6);
      Vector#(4, Vector#(4, Int#(32))) src = replicate(replicate(0));
      src[0][0] = 5; src[0][1] = 10; src[0][2] = 15; src[0][3] = 20;
      // ctrl: all True -> full reversal
      Vector#(2, Vector#(4, Bool)) ctrl = replicate(replicate(True));
      xlu.executePermute(src, ctrl);
      $display("Cycle %0d: dispatched PERMUTE reversal", cycle);
   endrule

   rule check_permute_rev (cycle == 7);
      let res = xlu.result;
      Bool ok = (res[0][0] == 20 && res[0][1] == 15 && res[0][2] == 10 && res[0][3] == 5);
      if (ok) begin
         $display("Cycle %0d: PASS PERMUTE reversal", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL PERMUTE reversal got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 8);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0)
         $finish(0);
      else
         $finish(1);
   endrule
```

- [ ] **Step 2: Run — expect 4 tests PASS**

```bash
cd /home/hanwang/p/tinytpu && make test-xlu
```

Expected output:
```
Cycle 0: dispatched ROTATE by 1
Cycle 1: PASS ROTATE
Cycle 2: dispatched BROADCAST lane 2
Cycle 3: PASS BROADCAST
Cycle 4: dispatched PERMUTE identity
Cycle 5: PASS PERMUTE identity
Cycle 6: dispatched PERMUTE reversal
Cycle 7: PASS PERMUTE reversal
Results: 4 passed, 0 failed
```

- [ ] **Step 3: Commit**

```bash
cd /home/hanwang/p/tinytpu
git add test/TbXLU.bsv
git commit --author="Han Wang <h1337h4x0r@gmail.com>" -m "test: add PERMUTE identity and reversal tests to TbXLU"
```

---

## Task 5: Add and implement TRANSPOSE test

**Files:**
- Modify: `test/TbXLU.bsv`
- (No change to `src/XLU.bsv` needed — executeTranspose already implemented)

Input (4×4 vreg):
```
row0: [1, 2, 3, 4]
row1: [5, 6, 7, 8]
row2: [9, 10, 11, 12]
row3: [13, 14, 15, 16]
```

Expected output (transposed, `result[r][c] = input[c][r]`):
```
row0: [1, 5, 9, 13]
row1: [2, 6, 10, 14]
row2: [3, 7, 11, 15]
row3: [4, 8, 12, 16]
```

- [ ] **Step 1: Replace `finish` rule and add TRANSPOSE test in TbXLU.bsv**

Replace `finish` rule at `cycle == 8` with:

```bsv
   // ---- Test 4: TRANSPOSE ----
   // result[r][c] = input[c][r]
   rule dispatch_transpose (cycle == 8);
      Vector#(4, Vector#(4, Int#(32))) src = newVector;
      src[0] = cons(1, cons(2, cons(3, cons(4, nil))));
      src[1] = cons(5, cons(6, cons(7, cons(8, nil))));
      src[2] = cons(9, cons(10, cons(11, cons(12, nil))));
      src[3] = cons(13, cons(14, cons(15, cons(16, nil))));
      xlu.executeTranspose(src);
      $display("Cycle %0d: dispatched TRANSPOSE", cycle);
   endrule

   rule check_transpose (cycle == 9);
      let res = xlu.result;
      // row0: [1,5,9,13], row1: [2,6,10,14], row2: [3,7,11,15], row3: [4,8,12,16]
      Bool ok = (res[0][0] == 1  && res[0][1] == 5  && res[0][2] == 9  && res[0][3] == 13 &&
                 res[1][0] == 2  && res[1][1] == 6  && res[1][2] == 10 && res[1][3] == 14 &&
                 res[2][0] == 3  && res[2][1] == 7  && res[2][2] == 11 && res[2][3] == 15 &&
                 res[3][0] == 4  && res[3][1] == 8  && res[3][2] == 12 && res[3][3] == 16);
      if (ok) begin
         $display("Cycle %0d: PASS TRANSPOSE", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL TRANSPOSE", cycle);
         $display("  row0: [%0d,%0d,%0d,%0d]", res[0][0], res[0][1], res[0][2], res[0][3]);
         $display("  row1: [%0d,%0d,%0d,%0d]", res[1][0], res[1][1], res[1][2], res[1][3]);
         $display("  row2: [%0d,%0d,%0d,%0d]", res[2][0], res[2][1], res[2][2], res[2][3]);
         $display("  row3: [%0d,%0d,%0d,%0d]", res[3][0], res[3][1], res[3][2], res[3][3]);
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 10);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0)
         $finish(0);
      else
         $finish(1);
   endrule
```

- [ ] **Step 2: Run — expect all 5 tests PASS**

```bash
cd /home/hanwang/p/tinytpu && make test-xlu
```

Expected output:
```
Cycle 0: dispatched ROTATE by 1
Cycle 1: PASS ROTATE
Cycle 2: dispatched BROADCAST lane 2
Cycle 3: PASS BROADCAST
Cycle 4: dispatched PERMUTE identity
Cycle 5: PASS PERMUTE identity
Cycle 6: dispatched PERMUTE reversal
Cycle 7: PASS PERMUTE reversal
Cycle 8: dispatched TRANSPOSE
Cycle 9: PASS TRANSPOSE
Results: 5 passed, 0 failed
```

- [ ] **Step 3: Run full test suite to confirm no regressions**

```bash
cd /home/hanwang/p/tinytpu && make test
```

Expected: all existing tests (PE, SystolicArray, Accelerator 2x2, Accelerator 4x4) still pass, plus XLU passes.

- [ ] **Step 4: Commit**

```bash
cd /home/hanwang/p/tinytpu
git add test/TbXLU.bsv
git commit --author="Han Wang <h1337h4x0r@gmail.com>" -m "test: add TRANSPOSE test to TbXLU; all XLU tests pass"
```

---

## Notes on Deferred Items

Per spec Section 11 (Open Questions):

- **GATHER (dynamic index):** Not implemented. Would require passing an index vreg as a second Vector argument to `executePermute`. Lower priority — omitted per YAGNI.
- **SUBLANE_ROTATE:** Spec notes this may live in the VPU, not XLU. Not implemented here.
- **Pipelined butterfly (4–5 cycle latency):** Current implementation is combinatorial (1-cycle). To pipeline, add `Reg` arrays between `lane_butterfly` stages and an FSM similar to `mkController`. Not needed for functional correctness.
- **Beneš network (full arbitrary permutation):** The XOR-butterfly with TLog#(n) stages does NOT reach all permutations. A Beneš network (2*TLog#(n) - 1 stages) would. Not needed for the tested operations (rotate, broadcast, reversal, transpose).
- **Non-square vregs (sublanes ≠ lanes):** Remove the `Add#(0, sublanes, lanes)` proviso and implement TRANSPOSE as a multi-op sequence per spec Section 6.6. Not needed for current use cases.

---

*Plan created: 2026-04-08*
