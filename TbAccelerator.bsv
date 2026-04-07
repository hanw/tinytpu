package TbAccelerator;

import Vector :: *;
import TensorAccelerator :: *;

(* synthesize *)
module mkTbAccelerator();

   TensorAccelerator_IFC#(2, 2, 4) accel <- mkTensorAccelerator;

   Reg#(UInt#(8)) cycle <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 50) begin
         $display("FAIL: test timed out at cycle %0d", cycle);
         $finish(1);
      end
   endrule

   // Cycle 0: load weights at address 0
   // W = [[1, 2], [3, 4]]
   rule load_weights (cycle == 0);
      Vector#(2, Vector#(2, Int#(8))) w = newVector;
      w[0][0] = 1; w[0][1] = 2;
      w[1][0] = 3; w[1][1] = 4;
      accel.loadWeightTile(0, w);
      $display("Cycle %0d: weights loaded to SRAM", cycle);
   endrule

   // Cycle 1: load activations at address 0
   // a = [5, 6]
   rule load_acts (cycle == 1);
      Vector#(2, Int#(8)) a = newVector;
      a[0] = 5;
      a[1] = 6;
      accel.loadActivationTile(0, a);
      $display("Cycle %0d: activations loaded to SRAM", cycle);
   endrule

   // Cycle 2: start compute (weightBase=0, actBase=0, tileLen=1)
   rule start (cycle == 2);
      accel.startCompute(0, 0, 1);
      $display("Cycle %0d: compute started", cycle);
   endrule

   // Poll for completion
   rule wait_done (cycle > 2 && !accel.computeDone);
      $display("Cycle %0d: computing...", cycle);
   endrule

   rule check_done (cycle > 2 && accel.computeDone);
      Vector#(2, Int#(32)) out = accel.getOutput;
      $display("Cycle %0d: compute done", cycle);
      $display("  output[0] = %0d (expect 23)", out[0]);
      $display("  output[1] = %0d (expect 34)", out[1]);
      if (out[0] == 23 && out[1] == 34) begin
         $display("PASS: full accelerator test");
      end else begin
         $display("FAIL: unexpected output");
      end
      $finish(0);
   endrule

endmodule

endpackage
