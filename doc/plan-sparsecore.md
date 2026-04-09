# SparseCore (Embedding Lookup Accelerator) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the SparseCore — a dedicated accelerator for embedding table lookups that accepts a bag of sparse indices, retrieves the corresponding embedding vectors, and returns a pooled (summed) result vector.

**Architecture:** SparseCore holds an embedding table in a RegFile (`tableDepth` entries, each `embWidth` Int#(32) elements). It accepts a batch of up to `bagSize` indices via a simple Action method, performs sequential lookups (one per cycle), accumulates (sums) the retrieved embeddings, and signals done. Parameterized by `tableDepth`, `embWidth`, `bagSize`. No dependencies on other new modules.

**Tech Stack:** BSV, BSC, Bluesim, GNU Make. Follows patterns in `src/XLU.bsv` and `src/Controller.bsv`.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/SparseCore.bsv` | Create | SparseCore_IFC, `mkSparseCore` module |
| `test/TbSparseCore.bsv` | Create | Single-index lookup, multi-index sum-pooling tests |
| `Makefile` | Modify | Add `test-sc` target |

---

## Task 1: Makefile + failing lookup test

**Files:**
- Create: `test/TbSparseCore.bsv`
- Modify: `Makefile`

- [ ] **Step 1: Add Makefile entries**

```makefile
$(BUILDDIR)/TbSparseCore.bo: $(BUILDDIR)/SparseCore.bo
```
```makefile
$(BUILDDIR)/mkTbSparseCore.bexe: $(BUILDDIR)/TbSparseCore.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbSparseCore $(BUILDDIR)/mkTbSparseCore.ba
test-sc: $(BUILDDIR)/mkTbSparseCore.bexe
	$<
```

Add `test-sc` to `.PHONY` and `test` target.

- [ ] **Step 2: Write failing test**

Create `test/TbSparseCore.bsv`:

```bsv
package TbSparseCore;

import Vector :: *;
import SparseCore :: *;

(* synthesize *)
module mkTbSparseCore();

   // tableDepth=8, embWidth=4, bagSize=3
   SparseCore_IFC#(8, 4, 3) sc <- mkSparseCore;

   Reg#(UInt#(8)) cycle  <- mkReg(0);
   Reg#(UInt#(8)) passed <- mkReg(0);
   Reg#(UInt#(8)) failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 100) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   // Preload embedding table
   // entry 0: [1, 0, 0, 0]
   // entry 2: [0, 5, 0, 0]
   // entry 5: [0, 0, 3, 7]
   rule load_table (cycle == 0);
      Vector#(4, Int#(32)) e0 = replicate(0); e0[0] = 1;
      Vector#(4, Int#(32)) e2 = replicate(0); e2[1] = 5;
      Vector#(4, Int#(32)) e5 = replicate(0); e5[2] = 3; e5[3] = 7;
      sc.loadEmbedding(0, e0);
      sc.loadEmbedding(2, e2);
      sc.loadEmbedding(5, e5);
      $display("Cycle %0d: embedding table loaded", cycle);
   endrule

   // Test 1: single-index lookup — look up index 2, expect [0, 5, 0, 0]
   rule submit_single (cycle == 1);
      Vector#(3, UInt#(3)) idx = replicate(0);
      idx[0] = 2;
      sc.submitBag(idx, 1);  // count=1, only idx[0] is valid
      $display("Cycle %0d: submitted bag [2] count=1", cycle);
   endrule

   rule wait_single (cycle > 1 && !sc.isDone);
      $display("Cycle %0d: SC computing...", cycle);
   endrule

   rule check_single (cycle > 1 && sc.isDone);
      Vector#(4, Int#(32)) res = sc.result;
      Bool ok = (res[0] == 0 && res[1] == 5 && res[2] == 0 && res[3] == 0);
      if (ok) begin
         $display("Cycle %0d: PASS single lookup", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL single lookup [%0d,%0d,%0d,%0d]",
            cycle, res[0], res[1], res[2], res[3]);
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 20);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
```

- [ ] **Step 3: Run — expect compile error**

```bash
cd /home/hanwang/p/tinytpu && make test-sc
```

---

## Task 2: Implement `src/SparseCore.bsv`

**Files:**
- Create: `src/SparseCore.bsv`

- [ ] **Step 1: Write SparseCore.bsv**

```bsv
package SparseCore;

import Vector :: *;
import RegFile :: *;

interface SparseCore_IFC#(numeric type tableDepth, numeric type embWidth, numeric type bagSize);
   // Load one embedding vector into the table
   method Action loadEmbedding(UInt#(TLog#(tableDepth)) idx,
                                Vector#(embWidth, Int#(32)) emb);
   // Submit a bag of up to bagSize indices; count = number of valid entries
   method Action submitBag(Vector#(bagSize, UInt#(TLog#(tableDepth))) indices,
                           UInt#(8) count);
   method Vector#(embWidth, Int#(32)) result;
   method Bool isDone;
endinterface

typedef enum { SC_IDLE, SC_LOOKUP, SC_DONE } SCState deriving (Bits, Eq, FShow);

module mkSparseCore(SparseCore_IFC#(tableDepth, embWidth, bagSize))
   provisos(
      Add#(1, t_, tableDepth),
      Add#(1, e_, embWidth),
      Add#(1, b_, bagSize),
      Log#(tableDepth, logTable),
      Log#(bagSize, logBag),
      Bits#(Vector#(embWidth, Int#(32)), esz)
   );

   RegFile#(UInt#(TLog#(tableDepth)), Vector#(embWidth, Int#(32)))
      table <- mkRegFileFull;

   Reg#(SCState)                                   state    <- mkReg(SC_IDLE);
   Reg#(Vector#(bagSize, UInt#(TLog#(tableDepth)))) bagReg  <- mkRegU;
   Reg#(UInt#(8))                                  countReg <- mkReg(0);
   Reg#(UInt#(8))                                  cursor   <- mkReg(0);
   Reg#(Vector#(embWidth, Int#(32)))               accum    <- mkReg(replicate(0));

   rule do_lookup (state == SC_LOOKUP);
      if (cursor < countReg) begin
         let idx = bagReg[cursor];
         let emb = table.sub(idx);
         Vector#(embWidth, Int#(32)) newAccum = newVector;
         for (Integer i = 0; i < valueOf(embWidth); i = i + 1)
            newAccum[i] = accum[i] + emb[i];
         accum  <= newAccum;
         cursor <= cursor + 1;
      end else begin
         state <= SC_DONE;
      end
   endrule

   method Action loadEmbedding(UInt#(TLog#(tableDepth)) idx,
                                Vector#(embWidth, Int#(32)) emb);
      table.upd(idx, emb);
   endmethod

   method Action submitBag(Vector#(bagSize, UInt#(TLog#(tableDepth))) indices,
                           UInt#(8) count) if (state == SC_IDLE);
      bagReg   <= indices;
      countReg <= count;
      cursor   <= 0;
      accum    <= replicate(0);
      state    <= SC_LOOKUP;
   endmethod

   method Vector#(embWidth, Int#(32)) result if (state == SC_DONE);
      return accum;
   endmethod

   method Bool isDone;
      return state == SC_DONE;
   endmethod

endmodule

export SparseCore_IFC(..);
export mkSparseCore;

endpackage
```

- [ ] **Step 2: Run — expect single-lookup test to pass**

```bash
cd /home/hanwang/p/tinytpu && make test-sc
```
Expected:
```
Cycle 0: embedding table loaded
Cycle 1: submitted bag [2] count=1
Cycle N: PASS single lookup
Results: 1 passed, 0 failed
```

- [ ] **Step 3: Commit**

```bash
git add src/SparseCore.bsv test/TbSparseCore.bsv Makefile
git commit --author="Han Wang <h1337h4x0r@gmail.com>" -m "feat: add SparseCore embedding lookup accelerator"
```

---

## Task 3: Multi-index sum-pooling test

**Files:**
- Modify: `test/TbSparseCore.bsv`

- [ ] **Step 1: Replace `rule finish (cycle == 20)` with multi-index test**

The SparseCore returns to SC_IDLE after done... wait, it doesn't reset automatically. We need a way to re-submit. Two options: (a) add a `reset` method, or (b) instantiate a second SparseCore. For simplicity, add a `reset` method.

Add to `SparseCore_IFC`:
```bsv
method Action reset;
```

Add to `mkSparseCore`:
```bsv
method Action reset if (state == SC_DONE);
   state <= SC_IDLE;
endmethod
```

Then in `TbSparseCore.bsv`, replace `rule finish (cycle == 20)` with:

```bsv
   // After single-index test passes, reset and run multi-index test
   Reg#(Bool) phase2Started <- mkReg(False);

   rule reset_sc (cycle > 1 && sc.isDone && !phase2Started);
      sc.reset;
      phase2Started <= True;
      $display("Cycle %0d: SC reset", cycle);
   endrule

   // Test 2: bag of 3 indices [0, 2, 5]
   // Sum: e0+e2+e5 = [1,0,0,0]+[0,5,0,0]+[0,0,3,7] = [1,5,3,7]
   Reg#(Bool) bag2Submitted <- mkReg(False);

   rule submit_multi (phase2Started && !bag2Submitted && !sc.isDone);
      Vector#(3, UInt#(3)) idx = newVector;
      idx[0] = 0; idx[1] = 2; idx[2] = 5;
      sc.submitBag(idx, 3);
      bag2Submitted <= True;
      $display("Cycle %0d: submitted bag [0,2,5] count=3", cycle);
   endrule

   Reg#(Bool) checked2 <- mkReg(False);

   rule check_multi (bag2Submitted && sc.isDone && !checked2);
      checked2 <= True;
      Vector#(4, Int#(32)) res = sc.result;
      Bool ok = (res[0] == 1 && res[1] == 5 && res[2] == 3 && res[3] == 7);
      if (ok) begin
         $display("Cycle %0d: PASS multi-index sum-pooling", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL multi-index [%0d,%0d,%0d,%0d]",
            cycle, res[0], res[1], res[2], res[3]);
         failed <= failed + 1;
      end
      $display("Results: %0d passed, %0d failed", passed + (ok?1:0), failed + (ok?0:1));
      if (ok) $finish(0); else $finish(1);
   endrule
```

- [ ] **Step 2: Run — expect 2 tests PASS**

```bash
cd /home/hanwang/p/tinytpu && make test-sc
```
Expected:
```
Cycle N: PASS single lookup
Cycle M: PASS multi-index sum-pooling
Results: 2 passed, 0 failed
```

- [ ] **Step 3: Run regression + commit**

```bash
cd /home/hanwang/p/tinytpu && make test
git add src/SparseCore.bsv test/TbSparseCore.bsv
git commit --author="Han Wang <h1337h4x0r@gmail.com>" -m "test: add SparseCore multi-index sum-pooling test"
```

---

*Plan created: 2026-04-08*
