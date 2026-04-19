package TbCtrlPSUM;

// Drive mkController directly with startPsum(...) so the per-dispatch
// result is deposited into a PSUM bucket row. Two back-to-back
// dispatches (WRITE then ACCUMULATE) should land 2x the single-
// dispatch result in the same bucket row.

import Vector :: *;
import SystolicArray :: *;
import WeightSRAM :: *;
import ActivationSRAM :: *;
import Controller :: *;
import PSUMBank :: *;

(* synthesize *)
module mkTbCtrlPSUM();

   SystolicArray_IFC#(4, 4)   array  <- mkSystolicArray;
   WeightSRAM_IFC#(16, 4, 4)  wsram  <- mkWeightSRAM;
   ActivationSRAM_IFC#(16, 4) asram  <- mkActivationSRAM;
   PSUMBank_IFC#(8, 4, 4)     psum   <- mkPSUMBank;
   Controller_IFC#(4, 4, 16)  ctrl   <- mkController(array, wsram, asram, psum);

   Reg#(UInt#(16)) cycle <- mkReg(0);
   Reg#(UInt#(2))  phase <- mkReg(0);

   rule tick;
      cycle <= cycle + 1;
      if (cycle > 200) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   // Weights = identity int8 4x4.
   rule load_w (cycle == 0);
      Vector#(4, Vector#(4, Int#(8))) w = replicate(replicate(0));
      w[0][0] = 1; w[1][1] = 1; w[2][2] = 1; w[3][3] = 1;
      wsram.write(0, w);
   endrule

   // Activations [1,2,3,4].
   rule load_a (cycle == 1);
      Vector#(4, Int#(8)) a = newVector;
      a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4;
      asram.write(0, a);
   endrule

   // First dispatch: PSUM_WRITE into psum[0] row 2.
   rule dispatch1 (cycle == 2 && phase == 0);
      ctrl.startPsum(0, 0, 1, 0, 2, PSUM_WRITE);
      phase <= 1;
      $display("Cycle %0d: dispatch 1 (WRITE)", cycle);
   endrule

   // Second dispatch: PSUM_ACCUMULATE into same row once first is done.
   rule dispatch2 (phase == 1 && ctrl.isDone);
      ctrl.startPsum(0, 0, 1, 0, 2, PSUM_ACCUMULATE);
      phase <= 2;
      $display("Cycle %0d: dispatch 2 (ACCUMULATE)", cycle);
   endrule

   // Check psum[0] row 2 after second dispatch completes. MXU output
   // for weight=identity, act=[1..4], tileLen=1 is [1,2,3,4]; accumulating
   // twice leaves [2,4,6,8] in the targeted row.
   rule check (phase == 2 && ctrl.isDone);
      let r = psum.peekRow(0, 2);
      Bool ok = (r[0] == 2 && r[1] == 4 && r[2] == 6 && r[3] == 8);
      if (ok)
         $display("Cycle %0d: PASS MXU->PSUM accumulate row=[%0d,%0d,%0d,%0d]",
                  cycle, r[0], r[1], r[2], r[3]);
      else
         $display("Cycle %0d: FAIL MXU->PSUM row=[%0d,%0d,%0d,%0d]",
                  cycle, r[0], r[1], r[2], r[3]);
      $display("Results: %0d passed, %0d failed", ok ? 1 : 0, ok ? 0 : 1);
      $finish(ok ? 0 : 1);
   endrule

endmodule

endpackage
