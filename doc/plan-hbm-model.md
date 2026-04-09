# HBM Behavioral Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a behavioral HBM (High Bandwidth Memory) model — a large parameterized SRAM with configurable read latency that represents off-chip DRAM for simulation purposes, enabling TensorCores to read/write data with realistic (but modeled) memory latency.

**Architecture:** HBM is modeled as a deep RegFile of vreg-sized tiles with a pipelined read path: `readReq` issues the request, and `readResp` is valid `latency` cycles later. A FIFO pipeline of depth `latency` holds in-flight requests. For TinyTPU, `latency=3` cycles models the difference between VMEM (~3 cycles) and true HBM (~200 cycles, simplified for simulation). No dependencies on other new modules.

**Tech Stack:** BSV, BSC, Bluesim, GNU Make. Follows VMEM pattern in `src/VMEM.bsv`.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/HBMModel.bsv` | Create | HBMModel_IFC, `mkHBMModel` with latency pipeline |
| `test/TbHBMModel.bsv` | Create | Write/read with latency, multiple in-flight tests |
| `Makefile` | Modify | Add `test-hbm` target |

---

## Task 1: Makefile + failing write/read test

**Files:**
- Create: `test/TbHBMModel.bsv`
- Modify: `Makefile`

- [ ] **Step 1: Add Makefile entries**

```makefile
$(BUILDDIR)/TbHBMModel.bo: $(BUILDDIR)/HBMModel.bo
$(BUILDDIR)/mkTbHBMModel.bexe: $(BUILDDIR)/TbHBMModel.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbHBMModel $(BUILDDIR)/mkTbHBMModel.ba
test-hbm: $(BUILDDIR)/mkTbHBMModel.bexe
	$<
```

Add `test-hbm` to `.PHONY` and `test` target.

- [ ] **Step 2: Write failing test**

Create `test/TbHBMModel.bsv`:

```bsv
package TbHBMModel;

import Vector :: *;
import HBMModel :: *;

(* synthesize *)
module mkTbHBMModel();

   // depth=64, sublanes=4, lanes=4, latency=3
   HBMModel_IFC#(64, 4, 4, 3) hbm <- mkHBMModel;

   Reg#(UInt#(8)) cycle  <- mkReg(0);
   Reg#(UInt#(8)) passed <- mkReg(0);
   Reg#(UInt#(8)) failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 50) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   // Write to addr 10
   rule write_data (cycle == 0);
      Vector#(4, Vector#(4, Int#(32))) t = replicate(replicate(0));
      t[0][0] = 42; t[2][3] = 77;
      hbm.write(10, t);
      $display("Cycle %0d: wrote addr 10 (42, 77)", cycle);
   endrule

   // Issue readReq at cycle 1 — response valid at cycle 1+latency = cycle 4
   rule read_req (cycle == 1);
      hbm.readReq(10);
      $display("Cycle %0d: issued readReq addr 10", cycle);
   endrule

   // Check at cycle 4 (latency=3 after req at cycle 1)
   rule check_resp (cycle == 4);
      let t = hbm.readResp;
      Bool ok = (t[0][0] == 42 && t[2][3] == 77);
      if (ok) begin
         $display("Cycle %0d: PASS HBM read with latency-3", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL HBM got [0][0]=%0d [2][3]=%0d",
            cycle, t[0][0], t[2][3]);
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 5);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
```

- [ ] **Step 3: Run — expect compile error**

```bash
cd /home/hanwang/p/tinytpu && make test-hbm
```

---

## Task 2: Implement `src/HBMModel.bsv`

**Files:**
- Create: `src/HBMModel.bsv`

- [ ] **Step 1: Write HBMModel.bsv**

The latency pipeline uses a `Vector#(latency, ...)` of pipeline registers. Each cycle, data shifts one stage toward the output. A valid bit tracks whether each stage holds valid data.

```bsv
package HBMModel;

import Vector :: *;
import RegFile :: *;

interface HBMModel_IFC#(numeric type depth,
                         numeric type sublanes,
                         numeric type lanes,
                         numeric type latency);
   method Action write(UInt#(TLog#(depth)) addr,
                       Vector#(sublanes, Vector#(lanes, Int#(32))) data);
   method Action readReq(UInt#(TLog#(depth)) addr);
   method Vector#(sublanes, Vector#(lanes, Int#(32))) readResp;
endinterface

module mkHBMModel(HBMModel_IFC#(depth, sublanes, lanes, latency))
   provisos(
      Add#(1, d_, depth),
      Add#(1, s_, sublanes),
      Add#(1, l_, lanes),
      Add#(1, lat_, latency),
      Bits#(Vector#(sublanes, Vector#(lanes, Int#(32))), vsz)
   );

   RegFile#(UInt#(TLog#(depth)), Vector#(sublanes, Vector#(lanes, Int#(32))))
      mem <- mkRegFileFull;

   // Shift register pipeline: pipeline[0] is filled on readReq,
   // pipeline[latency-1] is the output (readResp).
   // Each cycle, values shift from index 0 toward index latency-1.
   // valid[i] indicates whether stage i holds valid data.
   Reg#(Vector#(latency, Vector#(sublanes, Vector#(lanes, Int#(32))))) pipeline
      <- mkReg(replicate(replicate(replicate(0))));
   Reg#(Vector#(latency, Bool)) valid <- mkReg(replicate(False));

   // Shift pipeline each cycle
   rule shift_pipeline;
      Vector#(latency, Vector#(sublanes, Vector#(lanes, Int#(32)))) newPipe = pipeline;
      Vector#(latency, Bool) newValid = valid;
      // Shift from high to low index (index 0 = newest, index latency-1 = oldest/output)
      for (Integer i = valueOf(latency) - 1; i > 0; i = i - 1) begin
         newPipe[i]  = pipeline[i-1];
         newValid[i] = valid[i-1];
      end
      newValid[0] = False;  // slot 0 is cleared until a new readReq fills it
      pipeline <= newPipe;
      valid    <= newValid;
   endrule

   method Action write(UInt#(TLog#(depth)) addr,
                       Vector#(sublanes, Vector#(lanes, Int#(32))) data);
      mem.upd(addr, data);
   endmethod

   method Action readReq(UInt#(TLog#(depth)) addr);
      // Load data into pipeline stage 0; shift_pipeline will propagate it
      Vector#(latency, Vector#(sublanes, Vector#(lanes, Int#(32)))) newPipe = pipeline;
      Vector#(latency, Bool) newValid = valid;
      newPipe[0]  = mem.sub(addr);
      newValid[0] = True;
      pipeline <= newPipe;
      valid    <= newValid;
   endrule

   method Vector#(sublanes, Vector#(lanes, Int#(32))) readResp;
      return pipeline[valueOf(latency) - 1];
   endmethod

endmodule

export HBMModel_IFC(..);
export mkHBMModel;

endpackage
```

**Note on scheduling:** Both `readReq` and `shift_pipeline` write to `pipeline` and `valid`. BSV will flag a conflict. Resolve by folding the pipeline shift into the `readReq` and a separate always-running rule that only shifts when no readReq fires in the same cycle. Use `(*descending_urgency*)` or restructure so only one rule writes per cycle. The simplest fix: remove the `shift_pipeline` rule and instead shift inside `readReq` explicitly, and add a separate rule that shifts when no readReq fires:

```bsv
// Replace shift_pipeline rule with a conditional shift rule
// that fires only when readReq does not fire this cycle.
// BSV handles this via rule priorities — mark shift as lower urgency:
(* descending_urgency = "do_readReq, shift_pipeline" *)
rule shift_pipeline;
   ...
```

If BSV conflict resolution is problematic, use a `Wire#(Maybe#(...))` to communicate between `readReq` and the shift rule.

- [ ] **Step 2: Run — expect PASS**

```bash
cd /home/hanwang/p/tinytpu && make test-hbm
```
Expected:
```
Cycle 0: wrote addr 10 (42, 77)
Cycle 1: issued readReq addr 10
Cycle 4: PASS HBM read with latency-3
Results: 1 passed, 0 failed
```

- [ ] **Step 3: Run regression + commit**

```bash
cd /home/hanwang/p/tinytpu && make test
git add src/HBMModel.bsv test/TbHBMModel.bsv Makefile
git commit --author="Han Wang <h1337h4x0r@gmail.com>" -m "feat: add HBM behavioral model with configurable read latency"
```

---

*Plan created: 2026-04-08*
