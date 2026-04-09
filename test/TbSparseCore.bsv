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
   rule load_e0 (cycle == 0);
      Vector#(4, Int#(32)) e0 = replicate(0); e0[0] = 1;
      sc.loadEmbedding(0, e0);
   endrule

   rule load_e2 (cycle == 1);
      Vector#(4, Int#(32)) e2 = replicate(0); e2[1] = 5;
      sc.loadEmbedding(2, e2);
   endrule

   rule load_e5 (cycle == 2);
      Vector#(4, Int#(32)) e5 = replicate(0); e5[2] = 3; e5[3] = 7;
      sc.loadEmbedding(5, e5);
      $display("Cycle %0d: embedding table loaded", cycle);
   endrule

   // Test 1: single-index lookup — look up index 2, expect [0, 5, 0, 0]
   rule submit_single (cycle == 3);
      Vector#(3, UInt#(3)) idx = replicate(0);
      idx[0] = 2;
      sc.submitBag(idx, 1);  // count=1, only idx[0] valid
      $display("Cycle %0d: submitted bag [2] count=1", cycle);
   endrule

   rule wait_single (cycle > 3 && !sc.isDone);
      $display("Cycle %0d: SC computing...", cycle);
   endrule

   Reg#(Bool) checked1 <- mkReg(False);

   rule check_single (cycle > 3 && sc.isDone && !checked1);
      checked1 <= True;
      Vector#(4, Int#(32)) res = sc.result;
      Bool ok = (res[0] == 0 && res[1] == 5 && res[2] == 0 && res[3] == 0);
      if (ok) begin
         $display("Cycle %0d: PASS single lookup [0,5,0,0]", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL single lookup [%0d,%0d,%0d,%0d]",
            cycle, res[0], res[1], res[2], res[3]);
         failed <= failed + 1;
      end
   endrule

   // Test 2: reset then multi-index bag [0, 2, 5] → sum = [1,5,3,7]
   Reg#(Bool) phase2Started <- mkReg(False);

   rule reset_sc (sc.isDone && !phase2Started);
      sc.reset;
      phase2Started <= True;
      $display("Cycle %0d: SC reset", cycle);
   endrule

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
         $display("Cycle %0d: PASS multi-index sum-pooling [1,5,3,7]", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL multi-index [%0d,%0d,%0d,%0d]",
            cycle, res[0], res[1], res[2], res[3]);
         failed <= failed + 1;
      end
      $display("Results: %0d passed, %0d failed", passed + (ok ? 1 : 0), failed + (ok ? 0 : 1));
      if (ok) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
