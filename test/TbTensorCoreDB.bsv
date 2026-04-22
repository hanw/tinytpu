package TbTensorCoreDB;

// Exercise the TensorCore double-buffered SRAM API end-to-end:
//   1. Fill ACTIVE bank with tile A (identity, acts=[1,2,3,4]).
//   2. Fill INACTIVE bank with tile B (doubled identity, acts=[10,20,30,40])
//      via preloadWeightTile / preloadActivationTile — must not be visible
//      to the dispatch yet.
//   3. Dispatch against tile A via the stored program. Expect [1,2,3,4].
//   4. Swap both SRAM banks. Now tile B is active.
//   5. Dispatch against tile B with the same program. Expect [20,40,60,80]
//      (2*identity applied to [10,20,30,40]).

import Vector :: *;
import TensorCore :: *;
import ScalarUnit :: *;
import VPU :: *;

(* synthesize *)
module mkTbTensorCoreDB();

   TensorCore_IFC#(4, 4, 16) tc <- mkTensorCore;

   Reg#(UInt#(16)) cycle  <- mkReg(0);
   Reg#(UInt#(4))  phase  <- mkReg(0);
   Reg#(UInt#(8))  passed <- mkReg(0);
   Reg#(UInt#(8))  failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 500) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   // Cycle 0: ACTIVE-bank weights = identity.
   rule load_w_active (cycle == 0);
      Vector#(4, Vector#(4, Int#(8))) w = replicate(replicate(0));
      w[0][0] = 1; w[1][1] = 1; w[2][2] = 1; w[3][3] = 1;
      tc.loadWeightTile(0, w);
   endrule

   // Cycle 1: ACTIVE-bank activations = [1,2,3,4] at addr 1.
   rule load_a_active (cycle == 1);
      Vector#(4, Int#(8)) a = newVector;
      a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4;
      tc.loadActivationTile(1, a);
   endrule

   // Cycle 2: INACTIVE-bank weights = 2*identity at same addr 0.
   rule preload_w_inactive (cycle == 2);
      Vector#(4, Vector#(4, Int#(8))) w = replicate(replicate(0));
      w[0][0] = 2; w[1][1] = 2; w[2][2] = 2; w[3][3] = 2;
      tc.preloadWeightTile(0, w);
   endrule

   // Cycle 3: INACTIVE-bank activations = [10,20,30,40] at addr 1.
   rule preload_a_inactive (cycle == 3);
      Vector#(4, Int#(8)) a = newVector;
      a[0] = 10; a[1] = 20; a[2] = 30; a[3] = 40;
      tc.preloadActivationTile(1, a);
   endrule

   // Cycle 4+: program — MXU wBase=0 aBase=1 tLen=1, WAIT, HALT.
   rule load_prog0 (cycle == 4);
      tc.loadProgram(0, SxuInstr { op: SXU_DISPATCH_MXU, vmemAddr:0, vregDst:0, vregSrc:0,
                                   vpuOp: VPU_ADD, vregSrc2:0,
                                   mxuWBase:0, mxuABase:1, mxuTLen:1 });
   endrule
   rule load_prog1 (cycle == 5);
      tc.loadProgram(1, SxuInstr { op: SXU_WAIT_MXU, vmemAddr:0, vregDst:0, vregSrc:0,
                                   vpuOp: VPU_ADD, vregSrc2:0,
                                   mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule
   rule load_prog2 (cycle == 6);
      tc.loadProgram(2, SxuInstr { op: SXU_HALT, vmemAddr:0, vregDst:0, vregSrc:0,
                                   vpuOp: VPU_ADD, vregSrc2:0,
                                   mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   // Phase 0 → 1: start dispatch #1 (tile A).
   rule start1 (cycle == 7 && phase == 0);
      tc.start(3);
      phase <= 1;
      $display("Cycle %0d: dispatched against ACTIVE bank (tile A)", cycle);
   endrule

   // Phase 1 → 2: check result. Identity * [1,2,3,4] = [1,2,3,4].
   rule finish1 (phase == 1 && tc.isDone);
      let r = tc.getMxuResult;
      Bool ok = (r[0] == 1 && r[1] == 2 && r[2] == 3 && r[3] == 4);
      if (!ok) begin
         $display("FAIL dispatch 1: got [%0d,%0d,%0d,%0d] want [1,2,3,4]",
                  r[0], r[1], r[2], r[3]);
         $display("Results: 0 passed, 1 failed"); $finish(1);
      end
      $display("Cycle %0d: dispatch 1 = [%0d,%0d,%0d,%0d]",
               cycle, r[0], r[1], r[2], r[3]);
      passed <= passed + 1;
      phase <= 2;
   endrule

   // Phase 2 → 3: swap both banks, then re-arm the program by reloading
   // pc. SXU is at HALT; calling start() again resets pc to 0.
   rule swap (phase == 2 && tc.isDone);
      tc.swapWeightBanks;
      tc.swapActivationBanks;
      phase <= 3;
      $display("Cycle %0d: swapped both banks", cycle);
   endrule

   // Phase 3 → 4: dispatch #2 against the newly-active bank (tile B).
   rule start2 (phase == 3);
      tc.start(3);
      phase <= 4;
      $display("Cycle %0d: dispatched against swapped bank (tile B)", cycle);
   endrule

   rule finish2 (phase == 4 && tc.isDone);
      let r = tc.getMxuResult;
      // 2*identity @ [10,20,30,40] = [20,40,60,80]
      Bool ok = (r[0] == 20 && r[1] == 40 && r[2] == 60 && r[3] == 80);
      if (!ok) begin
         $display("FAIL dispatch 2: got [%0d,%0d,%0d,%0d] want [20,40,60,80]",
                  r[0], r[1], r[2], r[3]);
         $display("Results: %0d passed, 1 failed", passed); $finish(1);
      end
      $display("Cycle %0d: dispatch 2 = [%0d,%0d,%0d,%0d]",
               cycle, r[0], r[1], r[2], r[3]);
      $display("PASS TensorCore preload + swap + ping-pong dispatch");
      $display("Results: 2 passed, 0 failed");
      $finish(0);
   endrule

endmodule

endpackage
