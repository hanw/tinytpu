package TbCtrlOS;

// Exercise the Controller startOS() output-stationary path end-to-end:
// preload the WeightSRAM and ActivationSRAM with a small 4x4 @ 4x4 OS
// dispatch, fire startOS(kLen=4), wait for Done, and check the full
// psum matrix via resultsMatrix().
//
// Test case: A = identity (so column k of A is e_k), W = arange.
//   A @ W = W, so resultsMatrix must equal W cast to int32.

import Vector :: *;
import SystolicArray :: *;
import WeightSRAM :: *;
import ActivationSRAM :: *;
import Controller :: *;
import PSUMBank :: *;

(* synthesize *)
module mkTbCtrlOS();

   SystolicArray_IFC#(4, 4)   array  <- mkSystolicArray;
   WeightSRAM_IFC#(16, 4, 4)  wsram  <- mkWeightSRAM;
   ActivationSRAM_IFC#(16, 4) asram  <- mkActivationSRAM;
   PSUMBank_IFC#(8, 4, 4)     psum   <- mkPSUMBank;
   Controller_IFC#(4, 4, 16)  ctrl   <- mkController(array, wsram, asram, psum);

   Reg#(UInt#(16)) cycle <- mkReg(0);
   Reg#(UInt#(4))  phase <- mkReg(0);

   rule tick;
      cycle <= cycle + 1;
      if (cycle > 400) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   // Cycle 0-3: preload W[k][c] = k*4 + c + 1 (values 1..16).
   rule load_w (cycle == 0);
      Vector#(4, Vector#(4, Int#(8))) w = replicate(replicate(0));
      for (Integer k = 0; k < 4; k = k + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            w[k][c] = fromInteger(k * 4 + c + 1);
      wsram.write(0, w);
   endrule

   // Activation[k] = column k of identity = unit vector with 1 at row k.
   rule load_a0 (cycle == 1);
      Vector#(4, Int#(8)) a = replicate(0); a[0] = 1;
      asram.write(0, a);
   endrule
   rule load_a1 (cycle == 2);
      Vector#(4, Int#(8)) a = replicate(0); a[1] = 1;
      asram.write(1, a);
   endrule
   rule load_a2 (cycle == 3);
      Vector#(4, Int#(8)) a = replicate(0); a[2] = 1;
      asram.write(2, a);
   endrule
   rule load_a3 (cycle == 4);
      Vector#(4, Int#(8)) a = replicate(0); a[3] = 1;
      asram.write(3, a);
   endrule

   rule dispatch_os (cycle == 5 && phase == 0);
      ctrl.startOS(0, 0, 4);
      phase <= 1;
      $display("Cycle %0d: dispatch startOS(kLen=4)", cycle);
   endrule

   rule finish (phase == 1 && ctrl.isDone);
      let m = ctrl.resultsMatrix;
      let mode_ok = (ctrl.getDataflowMode == DF_OUTPUT_STATIONARY);
      Bool all_ok = True;
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1) begin
            Int#(32) want = fromInteger(r * 4 + c + 1);
            if (m[r][c] != want) all_ok = False;
         end
      $display("Cycle %0d: OS resultsMatrix =", cycle);
      for (Integer r = 0; r < 4; r = r + 1)
         $display("  [%0d, %0d, %0d, %0d]", m[r][0], m[r][1], m[r][2], m[r][3]);
      if (all_ok && mode_ok) begin
         $display("PASS OS 4x4 matmul + mode latch");
         $display("Results: 1 passed, 0 failed");
         $finish(0);
      end else begin
         $display("FAIL: all_ok=%0d mode_ok=%0d", pack(all_ok), pack(mode_ok));
         $display("Results: 0 passed, 1 failed");
         $finish(1);
      end
   endrule

endmodule

endpackage
